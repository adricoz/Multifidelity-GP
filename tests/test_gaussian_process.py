"""Unit tests of src/surrogate_models.GaussianProcess (single GP)."""
import numpy as np
import pytest
from conftest import forrester_lf
from scipy.optimize import check_grad, minimize, minimize_scalar
from scipy.special import gammaln
from scipy.stats import gamma, invgamma, lognorm, multivariate_normal
from src.kernels import Matern52Kernel, SquaredExponentialKernel
from src.surrogate_models import (BIAS_VARIANCE_BOUNDS, DEFAULT_PRIORS, LENGTHSCALE_BOUNDS,
                                  LENGTHSCALE_ONLY_PRIORS, LENGTHSCALE_PRIOR, NOISE_BOUNDS,
                                  RHO_LEARNED_MIN, SIGNAL_VARIANCE_BOUNDS, GaussianProcess,
                                  check_priors, neg_log_prior)


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


def scipy_neg_log_prior(prior, values):
    """-log p(values) with scipy.stats, minus the constant terms dropped by neg_log_prior."""
    family, a, b = prior
    values = np.asarray(values, dtype=float)
    if family == "invgamma":
        return -np.sum(invgamma.logpdf(values, a=a, scale=b)) + len(values) * (
            a * np.log(b) - gammaln(a))
    if family == "gamma":
        return -np.sum(gamma.logpdf(values, a=a, scale=1.0 / b)) - len(values) * (
            gammaln(a) - a * np.log(b))
    return -np.sum(lognorm.logpdf(values, s=b, scale=np.exp(a))) - len(values) * np.log(
        b * np.sqrt(2.0 * np.pi))


@pytest.mark.parametrize("prior", [("invgamma", 3.0, 2.0), ("gamma", 3.0, 6.0),
                                   ("lognormal", -0.5, 0.8)])
def test_neg_log_prior_matches_scipy(prior):
    """[PRIORS] value of each family vs scipy.stats, derivative w.r.t. log(theta) vs finite
    differences."""
    values = np.array([0.05, 0.3, 1.7])
    value, grad = neg_log_prior(prior, np.log(values))
    assert value == pytest.approx(scipy_neg_log_prior(prior, values), rel=1e-10)
    step = 1e-6
    for i in range(len(values)):
        shifted = np.log(values)
        shifted[i] += step
        assert grad[i] == pytest.approx((neg_log_prior(prior, shifted)[0] - value) / step,
                                        rel=1e-4)


def test_map_adds_the_prior_of_every_hyperparameter(data_3d):
    """[PRIORS] use_map adds -log p (up to a constant) of the lengthscales, t1, t2 and the noise
    (DEFAULT_PRIORS)."""
    x, y, _ = data_3d
    gp = GaussianProcess(SquaredExponentialKernel())
    gp.x_train = x
    y_n, _ = gp._normalize(y, None, 1.0)
    for params in ([0.4, 0.6, 0.9, 1.3, 0.01, 1e-4], [0.4, 0.6, 0.9, 50.0, 1e-6, 1e-8]):
        log_params = np.log(params)
        prior = (gp.negative_log_likelihood(log_params, y_n, use_map=True)
                 - gp.negative_log_likelihood(log_params, y_n))
        expected = (scipy_neg_log_prior(DEFAULT_PRIORS["lengthscale"], params[:3])
                    + scipy_neg_log_prior(DEFAULT_PRIORS["signal_variance"], params[3:4])
                    + scipy_neg_log_prior(DEFAULT_PRIORS["bias_variance"], params[4:5])
                    + scipy_neg_log_prior(DEFAULT_PRIORS["noise"], params[5:6]))
        assert prior == pytest.approx(expected, rel=1e-9)


def test_lengthscale_only_priors_reproduce_previous_behaviour(data_3d):
    """[MAP] LENGTHSCALE_ONLY_PRIORS: -log InvGamma(l_m) on l_1..l_d only (use_map before the
    priors on t1, t2 and the noise)."""
    x, y, _ = data_3d
    gp = GaussianProcess(SquaredExponentialKernel(), priors=LENGTHSCALE_ONLY_PRIORS)
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


def test_priors_are_resolved_and_checked():
    """[PRIORS] defaults, InvGamma(lengthscale_prior), partial overrides, None and errors."""
    gp = GaussianProcess(SquaredExponentialKernel(), lengthscale_prior=(3.0, 1.0),
                         priors={"noise": None, "signal_variance": ("gamma", 2, 1)})
    assert gp.priors["lengthscale"] == ("invgamma", 3.0, 1.0)
    assert gp.priors["signal_variance"] == ("gamma", 2.0, 1.0)
    assert gp.priors["bias_variance"] == DEFAULT_PRIORS["bias_variance"]
    assert gp.priors["noise"] is None
    for wrong in ({"rho": ("gamma", 2, 1)}, {"noise": ("normal", 0, 1)},
                  {"noise": ("lognormal", 0, -1)}, {"noise": ("gamma", 0, 1)}):
        with pytest.raises(ValueError):
            check_priors(wrong)


def test_rho_prior_closed_form_is_the_minimum(data_3d):
    """[RHO] with a normal prior (MAP), the profiled rho minimizes
    0.5 (y - rho f)^T K^-1 (y - rho f) + (rho - mean)^2 / (2 std^2) for fixed theta."""
    x, y, f_prev = data_3d
    gp = GaussianProcess(SquaredExponentialKernel(), use_map=True,
                         priors={"rho": ("normal", 1.0, 0.7)})
    gp.x_train = x
    y_n, f_n = gp._normalize(y, f_prev, 1.0)
    gp.kernel.set_params(np.array([0.4, 0.6, 0.9, 1.3, 0.01]))
    k_mat = gp.kernel.get_covariance_matrix(x) + 1e-4 * np.eye(len(x))
    l_chol = np.linalg.cholesky(k_mat)
    rho = gp._profile_rho(l_chol, y_n, f_n, 1.0, True, (-50.0, 50.0), gp._rho_prior(True))
    k_inv = np.linalg.inv(k_mat)

    def objective(r):
        res = y_n - r * f_n
        return 0.5 * res @ k_inv @ res + (r - 1.0) ** 2 / (2 * 0.7 ** 2)

    assert rho == pytest.approx(minimize_scalar(objective).x, abs=1e-6)
    assert gp._rho_prior(False) is None  # MLE: no prior on rho


def test_centred_rho_is_not_pulled_towards_rho_init():
    """[RHO] level below with a mean far from 0 (+50): the v1 normalization (constant of the
    residual fixed for rho = 1) pulls rho towards 1; the centred one recovers rho = 2."""
    x = np.linspace(0, 1, 15).reshape(-1, 1)
    f_low = np.sin(8 * x[:, 0]) + 50.0
    y = 2.0 * f_low + 0.3 * np.cos(3 * x[:, 0])
    rhos = {}
    for centering in (True, False):
        gp = GaussianProcess(SquaredExponentialKernel(), seed=0, rho_centering=centering)
        rhos[centering] = gp.fit(x, y, f_prev=f_low, estimate_rho=True)
    assert rhos[True] == pytest.approx(2.0, abs=0.05)
    assert abs(rhos[False] - 2.0) > 0.3


def test_centering_changes_nothing_with_a_fixed_rho(data_3d):
    """[RHO] estimate_rho=False: same model with and without rho_centering (mfego v1)."""
    x, y, f_prev = data_3d
    predictions = []
    for centering in (True, False):
        gp = GaussianProcess(SquaredExponentialKernel(), seed=0, rho_centering=centering)
        gp.fit(x, y, f_prev=f_prev, rho_init=1.5, estimate_rho=False)
        predictions.append(gp.predict_batch(x[:5] + 0.03))
    for a, b in zip(*predictions):
        np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-12)


def test_rho_prior_family_is_checked():
    """[RHO] [V3] 'rho' takes ('normal', mean, std) or ('lognormal', mu, s); 'normal' is only for
    'rho'."""
    assert check_priors({"rho": ("normal", 1, 0.5)}) == {"rho": ("normal", 1.0, 0.5)}
    assert check_priors({"rho": ("lognormal", 1, 1)}) == {"rho": ("lognormal", 1.0, 1.0)}
    for wrong in ({"rho": ("gamma", 2, 1)}, {"rho": ("invgamma", 3, 2)},
                  {"noise": ("normal", 0, 1)}, {"rho": ("normal", 1, 0)},
                  {"rho": ("lognormal", 0, -1)}):
        with pytest.raises(ValueError):
            check_priors(wrong)


V3_PRIOR = {"rho": ("lognormal", 1.0, 1.0)}  # [V3] LogNormal(1, 1): mode 1


def test_learning_rho_with_a_normal_prior_gives_the_closed_form(data_3d):
    """[V3] with the normal prior of v2, learning rho WITH the kernel hyperparameters (joint
    L-BFGS-B over (theta, rho), as in Sacher et al.) reaches the same optimum as the closed-form
    profile of v2: for a fixed theta, the closed form is the optimum in rho."""
    x, y, f_prev = data_3d
    gp = GaussianProcess(SquaredExponentialKernel(), use_map=True, rho_centering=True)
    gp.x_train = x
    y_n, f_n = gp._normalize(y, f_prev, 1.0)
    mean, std = DEFAULT_PRIORS["rho"][1:]
    bounds = [(np.log(lo), np.log(hi)) for lo, hi in [LENGTHSCALE_BOUNDS] * 3 + [
        SIGNAL_VARIANCE_BOUNDS, BIAS_VARIANCE_BOUNDS, NOISE_BOUNDS]]

    def profiled(theta):  # v2: rho in closed form for every theta
        return gp.negative_log_likelihood(theta, y_n, f_n, 1.0, True, (-5.0, 5.0),
                                          with_grad=True, use_map=True)

    def joint(q):  # rho as a hyperparameter: NLL at a fixed rho + normal prior on rho
        theta, rho = q[:-1], q[-1]
        nll, grad = gp.negative_log_likelihood(theta, y_n, f_n, rho, False, with_grad=True,
                                               use_map=True)
        k_mat = gp.kernel.get_covariance_matrix(x) + gp.noise * np.eye(len(x))
        alpha = np.linalg.solve(k_mat, y_n - rho * f_n)
        return (nll + (rho - mean) ** 2 / (2.0 * std ** 2),
                np.append(grad, -f_n @ alpha + (rho - mean) / std ** 2))

    rng = np.random.default_rng(0)
    starts = [np.log(np.r_[rng.uniform(0.2, 1.5, 3), 1.0, 1e-4, 1e-4]) for _ in range(5)]
    options = {"ftol": 1e-14, "gtol": 1e-9, "maxiter": 2000}
    best_profiled = min((minimize(profiled, s, jac=True, bounds=bounds, method="L-BFGS-B",
                                  options=options) for s in starts), key=lambda r: r.fun)
    best_joint = min((minimize(joint, np.r_[s, r0], jac=True, bounds=bounds + [(-5.0, 5.0)],
                               method="L-BFGS-B", options=options)
                      for s in starts for r0 in (0.5, 2.0)), key=lambda r: r.fun)
    profiled(best_profiled.x)  # sets theta, then the closed-form rho at this theta
    l_chol = np.linalg.cholesky(gp.kernel.get_covariance_matrix(x) + gp.noise * np.eye(len(x)))
    rho_profiled = gp._profile_rho(l_chol, y_n, f_n, 1.0, True, (-5.0, 5.0), (mean, std))
    assert best_joint.fun == pytest.approx(best_profiled.fun, rel=1e-6)
    assert best_joint.x[-1] == pytest.approx(rho_profiled, abs=1e-3)


@pytest.mark.parametrize("rho_centering", [False, True])
def test_learned_rho_objective_and_gradient(data_3d, rho_centering):
    """[V3] learned rho (lognormal prior): the objective is the NLL at that rho plus -log p(rho),
    and its analytical gradient (log rho included) matches finite differences."""
    x, y, f_prev = data_3d
    gp = GaussianProcess(Matern52Kernel(), use_map=True, priors=V3_PRIOR,
                         rho_centering=rho_centering)
    gp.x_train = x
    y_n, f_n = gp._normalize(y, f_prev, 1.0)
    args = (y_n, f_n, 1.0, True, (1e-3, 50.0))
    for params in ([0.4, 0.6, 0.9, 1.3, 0.01, 1e-3, 1.7], [1.5, 0.2, 2.0, 0.4, 1e-5, 1e-6, 0.3]):
        log_params = np.log(params)
        fixed = gp.negative_log_likelihood(log_params[:-1], y_n, f_n, params[-1], False,
                                           use_map=True)
        assert gp.negative_log_likelihood(log_params, *args, use_map=True) == pytest.approx(
            fixed + neg_log_prior(V3_PRIOR["rho"], log_params[-1:])[0], rel=1e-12)
        gradient = gp.negative_log_likelihood(log_params, *args, with_grad=True, use_map=True)[1]
        error = check_grad(lambda p: gp.negative_log_likelihood(p, *args, use_map=True),
                           lambda p: gp.negative_log_likelihood(p, *args, with_grad=True,
                                                                use_map=True)[1],
                           log_params, epsilon=1e-6)
        assert len(gradient) == 7 and error / np.linalg.norm(gradient) < 1e-4


def test_learned_rho_recovers_rho_and_stays_positive():
    """[V3] centred rho learned with LogNormal(1, 1): rho = 2 recovered on the shifted case
    (+50) of test_centred_rho_is_not_pulled_towards_rho_init; with a negative correlation, the
    learned rho stays positive even for rho_bounds (-5, 5): the likelihood pulls it towards 0,
    the prior (-log p -> infinity when rho -> 0) stops it inside (RHO_LEARNED_MIN, 1)."""
    x = np.linspace(0, 1, 15).reshape(-1, 1)
    f_low = np.sin(8 * x[:, 0]) + 50.0
    rhos = {}
    for sign in (1.0, -1.0):
        y = sign * 2.0 * f_low + 0.3 * np.cos(3 * x[:, 0])
        gp = GaussianProcess(SquaredExponentialKernel(), seed=0, use_map=True,
                             rho_centering=True, priors=V3_PRIOR)
        rhos[sign] = gp.fit(x, y, f_prev=f_low, estimate_rho=True, rho_bounds=(-5.0, 5.0))
    assert rhos[1.0] == pytest.approx(2.0, abs=0.05)
    assert RHO_LEARNED_MIN < rhos[-1.0] < 1.0
    with pytest.raises(ValueError, match="upper bound"):  # no positive rho in the bounds
        GaussianProcess(SquaredExponentialKernel(), use_map=True, rho_centering=True,
                        priors=V3_PRIOR).fit(x, y, f_prev=f_low, estimate_rho=True,
                                             rho_bounds=(-5.0, 1e-4))
    assert GaussianProcess(SquaredExponentialKernel(), use_map=False,
                           priors=V3_PRIOR)._learns_rho(f_low, True) is False  # MLE: profiled


@pytest.mark.parametrize("rho_centering", [False, True])
@pytest.mark.parametrize("priors", [None, {"signal_variance": ("gamma", 2.0, 1.0)}])
@pytest.mark.parametrize("use_map", [False, True])
@pytest.mark.parametrize("estimate_rho", [False, True])
def test_analytical_gradient_matches_finite_differences(data_3d, estimate_rho, use_map, priors,
                                                        rho_centering):
    """[FIX-N3] analytical gradient (log-space, profiled rho, [MAP] with or without the priors,
    [RHO] prior on rho and centring included) vs scipy check_grad."""
    x, y, f_prev = data_3d
    gp = GaussianProcess(SquaredExponentialKernel(), priors=priors, rho_centering=rho_centering)
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


def test_matern_nll_gradient_matches_finite_differences(data_3d):
    """[MAT] analytical gradient of the NLL with the Matern 5/2 kernel (MAP, profiled rho)."""
    x, y, f_prev = data_3d
    gp = GaussianProcess(Matern52Kernel(), use_map=True)
    gp.x_train = x
    y_n, f_n = gp._normalize(y, f_prev, 1.0)
    args = (y_n, f_n, 1.0, True, (-5.0, 5.0))
    log_params = np.log([0.4, 0.6, 0.9, 1.3, 0.01, 1e-3])
    error = check_grad(lambda p: gp.negative_log_likelihood(p, *args, use_map=True),
                       lambda p: gp.negative_log_likelihood(p, *args, with_grad=True,
                                                            use_map=True)[1],
                       log_params, epsilon=1e-6)
    gradient = gp.negative_log_likelihood(log_params, *args, with_grad=True, use_map=True)[1]
    assert error / np.linalg.norm(gradient) < 1e-4


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
