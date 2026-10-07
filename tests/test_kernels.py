"""Unit tests of src/kernels.py (squared exponential kernel, Eq. 2 of the reference article)."""
import numpy as np
import pytest
from src.kernels import (Kernel, SquaredExponentialKernel, base_covariance_matrix,
                         cov_fct, cross_covariance_matrix, k_l_vector)

THETA = np.array([0.3, 0.7, 1.4, 0.05])  # [l_1, l_2, t1, t2]


@pytest.fixture
def points():
    return np.random.default_rng(0).random((12, 2))


def test_vectorized_matrix_equals_reference_loop(points):
    """[FIX-N1] the vectorized matrix equals the previous double loop over cov_fct."""
    reference = np.array([[cov_fct(a, b, THETA) for b in points] for a in points])
    np.testing.assert_allclose(base_covariance_matrix(points, THETA), reference, atol=1e-14)


def test_matrix_is_symmetric_positive_semidefinite(points):
    k_mat = base_covariance_matrix(points, THETA)
    np.testing.assert_allclose(k_mat, k_mat.T)
    assert np.min(np.linalg.eigvalsh(k_mat)) > -1e-10


def test_diagonal_is_t1_plus_t2(points):
    np.testing.assert_allclose(np.diag(base_covariance_matrix(points, THETA)),
                               THETA[2] + THETA[3])


def test_cross_vector_and_matrix_are_consistent(points):
    k_mat = base_covariance_matrix(points, THETA)
    for i in (0, 5, 11):
        np.testing.assert_allclose(k_l_vector(points[i], points, THETA)[:, 0], k_mat[:, i])
    batch = cross_covariance_matrix(points[:4], points, THETA)
    np.testing.assert_allclose(batch, k_mat[:4])


def test_kernel_class_wraps_functions(points):
    kernel = SquaredExponentialKernel()
    kernel.set_params(THETA)
    np.testing.assert_allclose(kernel.get_params(), THETA)
    np.testing.assert_allclose(kernel.get_covariance_matrix(points),
                               base_covariance_matrix(points, THETA))
    np.testing.assert_allclose(kernel.get_cross_covariance_matrix(points[:3], points),
                               base_covariance_matrix(points, THETA)[:3])


def test_default_batch_cross_covariance_of_base_class(points):
    """A user kernel implementing only get_cross_variance_vector still works in batch."""
    class VectorOnlyKernel(Kernel):
        def get_cross_variance_vector(self, x_new, x_train):
            return k_l_vector(x_new, x_train, THETA)

    batch = VectorOnlyKernel().get_cross_covariance_matrix(points[:3], points)
    np.testing.assert_allclose(batch, cross_covariance_matrix(points[:3], points, THETA))


def test_log_gradients_match_finite_differences(points):
    """[FIX-N3] dK/dlog(theta_j) of the SE kernel vs central finite differences."""
    kernel = SquaredExponentialKernel()
    kernel.set_params(THETA)
    gradients = kernel.get_log_params_gradients(points)
    assert len(gradients) == len(THETA)
    eps = 1e-6
    for j, grad in enumerate(gradients):
        log_theta = np.log(THETA)
        plus, minus = log_theta.copy(), log_theta.copy()
        plus[j] += eps
        minus[j] -= eps
        fd = (base_covariance_matrix(points, np.exp(plus))
              - base_covariance_matrix(points, np.exp(minus))) / (2 * eps)
        np.testing.assert_allclose(grad, fd, atol=1e-7)
