"""Unit tests of src/surrogate_models.GaussianProcess (single GP)."""
import numpy as np
import pytest
from conftest import forrester_lf
from scipy.optimize import check_grad
from scipy.special import gammaln
from scipy.stats import invgamma, multivariate_normal
from src.kernels import SquaredExponentialKernel
from src.surrogate_models import (BIAS_VARIANCE_BOUNDS, LENGTHSCALE_BOUNDS, LENGTHSCALE_PRIOR,
                                  NOISE_BOUNDS, SIGNAL_VARIANCE_BOUNDS, GaussianProcess)


@pytest.fixture
def data_3d():
    rng = np.random.default_rng(1)
    x = rng.random((18, 3))
    y = np.sin(3 * x).sum(axis=1)
    f_prev = np.cos(2 * x).sum(axis=1)
    return x, y, f_prev


@pytest.fixture
def forrester_1d():
    x = np.linspace(0, 1, 9).reshape(-1, 1)
    return x, forrester_lf(x[:, 0])


def test_nll_matches_multivariate_normal(data_3d):
    """The NLL (Eqs. 15-16) equals -log N(y_n; 0, K + noise I)."""
    x, y, _ = data_3d
    gp = GaussianProcess(SquaredExponentialKernel())
    gp.x_train = x
    y_n, _ = gp._normalize(y, None, 1.0)
    params = np.array([0.4, 0.6, 0.9, 1.3, 0.01, 1e-4])
    nll = gp.negative_log_likelihood(np.log(params), y_n)
    cov = gp.kernel.get_covariance_matrix(x) + params[-1] * np.eye(len(x))
    assert nll == pytest.approx(-multivariate_normal.logpdf(y_n, cov=cov), rel=1e-10)


def test_map_adds_invgamma_prior_on_lengthscales_only(data_3d):
    """[MAP] use_map adds -log InvGamma(l_m) (up to a constant) on l_1..l_d, not on t1, t2, noise.
    """
    x, y, _ = data_3d
    gp = GaussianProcess(SquaredExponentialKernel())
    gp.x_train = x
    y_n, _ = gp._normalize(y, None, 1.0)
    alpha, beta = LENGTHSCALE_PRIOR
    constant = 3 * (alpha * np.log(beta) - gammaln(alpha))
    for params in ([0.4, 0.6, 0.9, 1.3, 0.01, 1e-4], [0.4, 0.6, 0.9, 50.0, 1e-6, 1e-8]):
        log_params = np.log(params)
        prior = (gp.negative_log_likelihood(log_params, y_n, use_map=True)
                 - gp.negative_log_likelihood(log_params, y_n))
        expected = -np.sum(invgamma.logpdf(params[:3], a=alpha, scale=beta)) + constant
        assert prior == pytest.approx(expected, rel=1e-9)


@pytest.mark.parametrize("use_map", [False, True])
@pytest.mark.parametrize("estimate_rho", [False, True])
def test_analytical_gradient_matches_finite_differences(data_3d, estimate_rho, use_map):
    """[FIX-N3] analytical gradient (log-space, profiled rho, [MAP] with or without the prior)
    vs scipy check_grad."""
    x, y, f_prev = data_3d
    gp = GaussianProcess(SquaredExponentialKernel())
    gp.x_train = x
    y_n, f_n = gp._normalize(y, f_prev, 1.0)
    args = (y_n, f_n, 1.0, estimate_rho, (-5.0, 5.0))

    def func(p):
        return gp.negative_log_likelihood(p, *args, with_grad=False, use_map=use_map)

    def grad(p):
        return gp.negative_log_likelihood(p, *args, with_grad=True, use_map=use_map)[1]

    for params in ([0.4, 0.6, 0.9, 1.3, 0.01, 1e-3], [1.5, 0.2, 2.0, 0.4, 1e-5, 1e-6]):
        log_params = np.log(params)
        error = check_grad(func, grad, log_params, epsilon=1e-6)
        assert error / np.linalg.norm(grad(log_params)) < 1e-4


def test_interpolates_training_points(forrester_1d):
    x, y = forrester_1d
    gp = GaussianProcess(SquaredExponentialKernel(), seed=0)
    gp.fit(x, y)
    mean, var = gp.predict_batch(x)
    np.testing.assert_allclose(mean, y, atol=1e-3 * np.std(y))
    assert np.all(var < 1e-3 * np.var(y))


def test_variance_grows_away_from_data(forrester_1d):
    x, y = forrester_1d
    gp = GaussianProcess(SquaredExponentialKernel(), seed=0)
    gp.fit(x, y)
    _, var_near = gp.predict(x[3] + 0.01)
    _, var_far = gp.predict(np.array([5.0]))
    prior = (gp.kernel.signal_variance + gp.kernel.bias_variance + gp.noise) * gp.y_std ** 2
    assert var_near < var_far <= prior * (1 + 1e-12)


def test_predictions_are_scale_equivariant(forrester_1d):
    """[FIX-N7] with output normalization, y -> a*y + b gives mean -> a*mean + b, var -> a^2 var.
    (Before the fix, absolute bounds made the model depend on the units of y.)"""
    x, y = forrester_1d
    x_test = np.linspace(0, 1, 37).reshape(-1, 1)
    gp_ref = GaussianProcess(SquaredExponentialKernel(), seed=0)
    gp_ref.fit(x, y)
    mean_ref, var_ref = gp_ref.predict_batch(x_test)
    for scale, shift in ((1e-3, 0.02), (1e3, -50.0)):
        gp = GaussianProcess(SquaredExponentialKernel(), seed=0)
        gp.fit(x, scale * y + shift)
        mean, var = gp.predict_batch(x_test)
        np.testing.assert_allclose(mean, scale * mean_ref + shift,
                                   atol=1e-4 * scale * np.std(y))
        np.testing.assert_allclose(var, scale ** 2 * var_ref,
                                   atol=1e-4 * scale ** 2 * np.var(y), rtol=1e-3)


def test_no_active_bound_on_lengthscale_and_signal_variance(forrester_1d):
    """[FIX-N2/N7] lengthscale and signal variance are interior on Forrester (the bias and
    the noise may legitimately reach their lower bound on deterministic data)."""
    x, y = forrester_1d
    gp = GaussianProcess(SquaredExponentialKernel(), seed=0)
    gp.fit(x, y)
    length, signal = gp.kernel.lengthscale[0], gp.kernel.signal_variance
    assert LENGTHSCALE_BOUNDS[0] * 1.01 < length < LENGTHSCALE_BOUNDS[1] / 1.01
    assert SIGNAL_VARIANCE_BOUNDS[0] * 1.01 < signal < SIGNAL_VARIANCE_BOUNDS[1] / 1.01
    tol = 1 + 1e-9  # exp(log(bound)) round-off
    assert BIAS_VARIANCE_BOUNDS[0] / tol <= gp.kernel.bias_variance <= BIAS_VARIANCE_BOUNDS[1] * tol
    assert NOISE_BOUNDS[0] / tol <= gp.noise <= NOISE_BOUNDS[1] * tol


def test_fit_is_reproducible_with_a_seed(data_3d):
    """[FIX-R1] seeded restarts give identical hyperparameters."""
    x, y, _ = data_3d
    params = []
    for _ in range(2):
        gp = GaussianProcess(SquaredExponentialKernel(), seed=123)
        gp.fit(x, y)
        params.append(np.append(gp.kernel.get_params(), gp.noise))
    np.testing.assert_array_equal(params[0], params[1])


def test_explicit_error_when_every_restart_fails(data_3d, monkeypatch):
    """[FIX-R2] an explicit ValueError (the previous ValueError was unreachable)."""
    x, y, _ = data_3d
    gp = GaussianProcess(SquaredExponentialKernel(), seed=0)
    monkeypatch.setattr(gp.kernel, "get_covariance_matrix",
                        lambda x_train: np.full((len(x_train), len(x_train)), np.nan))
    with pytest.raises(ValueError, match="GP fit failed"):
        gp.fit(x, y)


def test_condition_reproduces_the_fitted_gp(data_3d):
    """[FIX-N4] condition() with the fitted hyperparameters gives the same predictions."""
    x, y, _ = data_3d
    gp = GaussianProcess(SquaredExponentialKernel(), seed=0)
    gp.fit(x, y)
    clone = GaussianProcess(SquaredExponentialKernel())
    clone.kernel.set_params(gp.kernel.get_params())
    clone.noise, clone.y_mean, clone.y_std = gp.noise, gp.y_mean, gp.y_std
    clone.condition(x, y)
    x_test = np.random.default_rng(2).random((20, 3))
    for a, b in zip(gp.predict_batch(x_test), clone.predict_batch(x_test)):
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-14)


def test_predict_single_point_matches_batch(data_3d):
    x, y, _ = data_3d
    gp = GaussianProcess(SquaredExponentialKernel(), seed=0)
    gp.fit(x, y)
    mean, var = gp.predict_batch(x[:2] + 0.05)
    assert gp.predict(x[1] + 0.05) == pytest.approx((mean[1], var[1]), rel=1e-12)
