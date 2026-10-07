"""Unit tests of src/acquisition.py (merit function, Eqs. 19-24 of the reference article)."""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.stats import norm
from src.acquisition import AcquisitionFunction
from src.data_management import ExperimentData
from src.kernels import SquaredExponentialKernel
from src.surrogate_models import GaussianProcess, MultifidelityModel


class FakeModel:
    """Model returning prescribed predictions (to test the merit formulas alone)."""
    def __init__(self, mean, var, gp_vars, noises, rhos):
        self.mean, self.var, self.gp_vars = mean, var, np.atleast_2d(gp_vars)
        self.num_levels = len(noises)
        self.rhos = rhos
        self.gps = [SimpleNamespace(get_noise_variance=lambda n=n: n) for n in noises]

    def predict_batch(self, x, level=None):
        m = len(np.atleast_2d(x))
        return (np.full(m, self.mean), np.full(m, self.var),
                np.repeat(self.gp_vars, m, axis=0))


def acquisition_with(model, costs, f_best=0.0):
    data = ExperimentData(bounds=[(0.0, 1.0)], costs=costs)
    acq = AcquisitionFunction(model=model, data=data)
    acq.f_best = f_best
    return acq


def test_ei_matches_monte_carlo():
    """Noise-free single level: the merit reduces to the EI, checked by Monte Carlo."""
    mean, var = 0.3, 0.25
    acq = acquisition_with(FakeModel(mean, var, [var], [0.0], []), [1.0], f_best=0.1)
    samples = np.random.default_rng(0).normal(mean, np.sqrt(var), 2_000_000)
    ei_mc = np.mean(np.maximum(0.1 - samples, 0.0))
    assert acq.evaluate_merit(np.array([0.5]), 1) == pytest.approx(ei_mc, rel=5e-3)


def test_aei_factor_and_latent_variance_reduction():
    """[FIX-T2/T4] single level with noise: AEI factor of Eq. 20 and the variance reduction
    s^4 / (s^2 + noise) computed with the latent variance s^2 = var - noise."""
    mean, var, noise = 0.0, 0.5, 0.1
    acq = acquisition_with(FakeModel(mean, var, [var], [noise], []), [1.0], f_best=0.2)
    sigma = np.sqrt(var)
    u = (0.2 - mean) / sigma
    ei = sigma * (u * norm.cdf(u) + norm.pdf(u))
    aei = ei * (1 - np.sqrt(noise) / np.sqrt(var + noise))
    s2 = var - noise
    expected = aei * (s2 ** 2 / (s2 + noise)) / var
    assert acq.evaluate_merit(np.array([0.5]), 1) == pytest.approx(expected, rel=1e-12)


def test_variance_reduction_matches_block_inverse_update():
    """[FIX-T4] Eqs. 21-22 / 28-29: adding an observation at x reduces the predictive
    variance at x by s^4 / (s^2 + noise) (the previous formula used the noisy variance)."""
    rng = np.random.default_rng(3)
    x_train = rng.random((8, 1))
    gp = GaussianProcess(SquaredExponentialKernel())
    gp.kernel.set_params(np.array([0.3, 1.0, 0.01]))
    gp.noise = 0.05
    gp.condition(x_train, np.sin(6 * x_train[:, 0]))
    x_new = np.array([[0.37]])
    var_before = gp.predict_batch(x_new)[1][0]
    gp.condition(np.vstack([x_train, x_new]), np.append(np.sin(6 * x_train[:, 0]), 0.0))
    var_after = gp.predict_batch(x_new)[1][0]
    s2 = var_before - gp.noise
    assert var_before - var_after == pytest.approx(s2 ** 2 / (s2 + gp.noise), rel=1e-9)
    assert var_before - var_after != pytest.approx(var_before ** 2 / (var_before + gp.noise),
                                                   rel=1e-3)


def test_r2_product_and_cost_ratio_three_levels():
    """Eq. 24: R2_l = prod_{i >= l} rho_i^2 and cost ratio W_L / W_l."""
    gp_vars = [0.2, 0.3, 0.4]
    model = FakeModel(0.0, 1.0, gp_vars, [0.0, 0.0, 0.0], rhos=[2.0, 3.0])
    acq = acquisition_with(model, costs=[1.0, 10.0, 100.0], f_best=0.5)
    merits = acq.evaluate_merits(np.array([0.5]))
    sigma = 1.0
    u = 0.5
    ei = sigma * (u * norm.cdf(u) + norm.pdf(u))
    # noise-free: variance reduction = gp variance, information ratio = R2 * var_l / var_L
    expected = [ei * 100.0 * (2.0 ** 2 * 3.0 ** 2) * 0.2,
                ei * 10.0 * 3.0 ** 2 * 0.3,
                ei * 1.0 * 0.4]
    assert merits == pytest.approx(expected, rel=1e-12)


def test_merit_vanishes_without_improvement():
    acq = acquisition_with(FakeModel(10.0, 1e-8, [1e-8], [0.0], []), [1.0], f_best=0.0)
    assert acq.evaluate_merit(np.array([0.5]), 1) == 0.0


def test_batch_and_single_point_merits_agree(forrester_data):
    model = MultifidelityModel(2, SquaredExponentialKernel, seed=0)
    model.fit(forrester_data)
    acq = AcquisitionFunction(model=model, data=forrester_data)
    acq.update()
    x = np.linspace(0, 1, 7).reshape(-1, 1)
    batch = acq.evaluate_merits_batch(x)
    for i in (0, 3, 6):
        assert acq.evaluate_merits(x[i]) == pytest.approx(batch[i].tolist(), rel=1e-12)
        assert acq.evaluate_merit(x[i], 2) == pytest.approx(batch[i, 1], rel=1e-12)


def test_effective_best_solution(forrester_data):
    """[FIX-T3] x_best = argmin over all the training points of f_hat_L + sigma_L, f_best =
    f_hat_L(x_best) (Eq. 19, Huang et al.), low-fidelity points included."""
    model = MultifidelityModel(2, SquaredExponentialKernel, seed=0)
    model.fit(forrester_data)
    acq = AcquisitionFunction(model=model, data=forrester_data)
    acq.update()
    x_all = np.vstack([forrester_data.get_training_data(l)[0] for l in (1, 2)])
    mean, var, _ = model.predict_batch(x_all)
    i_best = np.argmin(mean + np.sqrt(var))
    np.testing.assert_allclose(acq.x_best, x_all[i_best])
    assert acq.f_best == pytest.approx(mean[i_best])
