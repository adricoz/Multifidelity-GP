"""
Core Module, implementing the Gaussian Process and Multifidelity Model classes.
"""
import json
import logging

import numpy as np
from scipy.linalg import cho_solve, solve_triangular
from scipy.optimize import minimize
from src.data_management import ExperimentData
from src.kernels import Kernel, Matern52Kernel, SquaredExponentialKernel

# [FIX-R3] logging.basicConfig() removed from this module: a library must not configure the
# root logger at import time (it silently disabled the basicConfig() of the calling scripts).
logger = logging.getLogger(__name__)

# [FIX-N2] Search bounds of the hyperparameters. They are expressed for NORMALIZED outputs
# (zero mean / unit variance, see FIX-N7) and inputs scaled in [0, 1], and they are
# explored in log-space. Previous (absolute) bounds: l in [0.01, 5], t1 in [1e-3, 50],
# t2 in [1e-6, 1], noise in [1e-8, 1e-5] (active on all the saved runs).
LENGTHSCALE_BOUNDS = (1e-2, 1e1)
SIGNAL_VARIANCE_BOUNDS = (1e-4, 1e2)
BIAS_VARIANCE_BOUNDS = (1e-8, 1e2)
NOISE_BOUNDS = (1e-8, 1e-2)
# [FIX-N5] relative diagonal jitters tried when a Cholesky factorization fails
JITTERS = (0.0, 1e-10, 1e-8, 1e-6)
# value returned by the NLL when the covariance matrix cannot be factorized
NLL_FAILURE = 1e10
# [MAP] (alpha, beta) of the InvGamma prior on each lengthscale (MAP estimation, use_map=True).
# Like LENGTHSCALE_BOUNDS, it assumes inputs in [0, 1]: mode beta / (alpha + 1) = 0.5, mean 1.
# The light left tail strongly penalizes small lengthscales (overfitting, ill-conditioned K),
# the heavy right tail lets an irrelevant dimension go to large lengthscales.
# Default value only: every GaussianProcess / MultifidelityModel can use its own prior
# (argument lengthscale_prior, see benchmarks/map_hartmann/RAPPORT_MAP.md for the choice).
LENGTHSCALE_PRIOR = (3.0, 2.0)
# [PRIORS] Priors of the MAP estimation (use_map=True) on EVERY hyperparameter group. Like the
# bounds, they assume NORMALIZED outputs (FIX-N7) and inputs in [0, 1]. A prior is a triple
# (family, a, b) on the hyperparameter itself (MAP in parameter space, no Jacobian of the log):
#   ("invgamma", alpha, beta)  mode beta / (alpha + 1)
#   ("gamma", shape, rate)     mode (shape - 1) / rate
#   ("lognormal", mu, sigma)   median exp(mu), mode exp(mu - sigma^2)
# None = no prior on this group. "lengthscale" is set from LENGTHSCALE_PRIOR / lengthscale_prior.
# [RHO] "rho" (correlation between levels) takes
#   ("normal", mean, std): rho profiled in closed form (any sign),
#       rho_hat = (F^T K^-1 y + mean / std^2) / (F^T K^-1 F + 1 / std^2),
#   [V3] ("lognormal", mu, s): rho > 0 learned with the other hyperparameters (log rho is one
#       more parameter of L-BFGS-B); mode exp(mu - s^2), so mu = s^2 puts the mode at 1, and
#       the penalty -log p is then (log rho)^2 / (2 s^2) + constant: a N(0, s^2) on log rho,
#       symmetric in factor around 1 (rho = 2 and rho = 0.5 cost the same).
# Profiling and learning give the same rho for the same prior: for a fixed theta the closed form
# IS the optimum in rho. The lognormal prior has no closed form, hence the learned rho.
PRIOR_GROUPS = ("lengthscale", "signal_variance", "bias_variance", "noise", "rho")
PRIOR_FAMILIES = ("invgamma", "gamma", "lognormal", "normal")
DEFAULT_PRIORS = {
    "lengthscale": ("invgamma",) + LENGTHSCALE_PRIOR,
    # t1: unit-variance outputs -> mode 1 (median e, 95 % in [0.38, 19])
    "signal_variance": ("lognormal", 1.0, 1.0),
    # t2: badly identified constant term (the mean is already removed): small, wide prior
    "bias_variance": ("lognormal", float(np.log(1e-2)), 2.0),
    # noise: nugget of a deterministic code: median 1e-4, mode 2e-6
    "noise": ("lognormal", float(np.log(1e-4)), 2.0),
    # [RHO] rho: levels describing the same physics, rho close to 1; std 2 chosen by
    # benchmarks/surrogate_v2/run_selection.py (weak prior: on Hartmann 6D it brings the share of
    # rho at its bound 5 from 68 % to 27 %; std 1: 8 %, see RAPPORT_V2.md Section 3).
    # Default only with rho_centering=True (see GaussianProcess): no prior on rho otherwise.
    "rho": ("normal", 1.0, 2.0),
}
# [PRIORS] behaviour of use_map=True before the priors on t1, t2, the noise and rho
# (lengthscales only)
LENGTHSCALE_ONLY_PRIORS = {"signal_variance": None, "bias_variance": None, "noise": None,
                           "rho": None}
# [RHO] priors of mfego v1 (16/10/2026: priors on l, t1, t2 and the noise, none on rho)
V1_PRIORS = {"rho": None}
# [MAP] relative margin used to flag a hyperparameter sitting at a search bound (diagnostics)
BOUND_FLAG_MARGIN = 1.05
# [V3] lower bound of a learned rho (lognormal prior: rho > 0)
RHO_LEARNED_MIN = 1e-3
# [V3] rho of mfego v3 (default of CoKriging, chosen by benchmarks/surrogate_v3/run_benchmark.py):
# learned with LogNormal(1, 1) (mode 1; penalty (log rho)^2 / 2, i.e. 2 nats at rho = 0.14 and
# 7.1) within wide bounds, so that the prior and not the bound limits rho
V3_RHO_PRIOR = ("lognormal", 1.0, 1.0)
V3_RHO_BOUNDS = (RHO_LEARNED_MIN, 50.0)


def neg_log_prior(prior: tuple, log_value: np.ndarray) -> tuple[float, np.ndarray]:
    """
    [PRIORS] -log p(theta) summed over the values (constant terms dropped) and its derivative
    with respect to log(theta), for theta = exp(log_value):
    - ("invgamma", alpha, beta): (alpha + 1) log t + beta / t
    - ("gamma", k, rate):        -(k - 1) log t + rate t
    - ("lognormal", mu, sigma):  log t + (log t - mu)^2 / (2 sigma^2)

    Args:
    - prior: (family, a, b) triple (see DEFAULT_PRIORS).
    - log_value: numpy array of log-hyperparameters.
    Returns:
    - value: float, -log p summed over the values.
    - grad: numpy array (same shape as log_value), d(-log p) / dlog(theta).
    """
    family, a, b = prior
    log_value = np.asarray(log_value, dtype=float)
    value = np.exp(log_value)
    if family == "invgamma":
        return float(np.sum((a + 1.0) * log_value + b / value)), (a + 1.0) - b / value
    if family == "gamma":
        return float(np.sum(-(a - 1.0) * log_value + b * value)), -(a - 1.0) + b * value
    if family == "lognormal":
        return (float(np.sum(log_value + (log_value - a) ** 2 / (2.0 * b ** 2))),
                1.0 + (log_value - a) / b ** 2)
    raise ValueError(f"Unknown prior family: {family!r} (allowed: {PRIOR_FAMILIES})")


def check_priors(priors) -> dict:
    """
    [PRIORS] Validates a (possibly partial) dict of priors {group: (family, a, b) or None}.

    Returns:
    - dict with the same groups, each value a (family, float, float) tuple or None.
    Raises:
    - ValueError for an unknown group, an unknown family or invalid parameters.
    """
    if priors is None:
        return {}
    checked = {}
    for group, prior in dict(priors).items():
        if group not in PRIOR_GROUPS:
            raise ValueError(f"Unknown prior group {group!r} (allowed: {PRIOR_GROUPS})")
        if prior is None:
            checked[group] = None
            continue
        if len(prior) != 3 or prior[0] not in PRIOR_FAMILIES:
            raise ValueError(f"Prior of {group!r} must be (family, a, b) with family in "
                             f"{PRIOR_FAMILIES} or None, got {prior!r}")
        if (group == "rho" and prior[0] not in ("normal", "lognormal")) \
           or (group != "rho" and prior[0] == "normal"):
            raise ValueError(f"The prior of 'rho' must be ('normal', mean, std) or ('lognormal', "
                             f"mu, s), and 'normal' is only for 'rho': got {prior!r} for {group!r}")
        family, a, b = prior[0], float(prior[1]), float(prior[2])
        if not (np.isfinite(a) and np.isfinite(b) and b > 0.0
                and (family in ("lognormal", "normal") or a > 0.0)):
            raise ValueError(f"Invalid parameters for the prior of {group!r}: {prior!r}")
        checked[group] = (family, a, b)
    return checked


def describe_priors(priors: dict) -> str:
    """[PRIORS] Readable description of a priors dict, e.g. 'lengthscale invgamma(3, 2); ...'."""
    return "; ".join(f"{group} " + ("none" if priors.get(group) is None else
                                    f"{priors[group][0]}({priors[group][1]:.3g}, "
                                    f"{priors[group][2]:.3g})")
                     for group in PRIOR_GROUPS)


def saved_priors(resolved: dict, overrides: dict) -> dict:
    """
    [PRIORS] JSON form of the priors actually used by a model: every group of `resolved` (the
    priors of a GP), except the lengthscales when they come from lengthscale_prior (saved
    separately, possibly one pair per level) rather than from the `overrides` dict.
    """
    return {group: None if prior is None else list(prior) for group, prior in resolved.items()
            if group != "lengthscale" or group in overrides}


def scale_to_unit(x: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    """
    [NORM] Inputs scaled to [0, 1]^d with the bounds of the design space,
    x_unit = (x - lower) / (upper - lower); bounds None: x unchanged (identity).

    Args:
    - x: numpy array (m, d) of raw inputs.
    - bounds: numpy array (d, 2) of [lower, upper] per input, or None.
    """
    if bounds is None:
        return x
    lower, upper = bounds[:, 0], bounds[:, 1]
    span = np.where(upper > lower, upper - lower, 1.0)
    return (np.asarray(x, dtype=float) - lower) / span


def check_lengthscale_prior(prior) -> tuple[float, float]:
    """
    [MAP] Validates an InvGamma(alpha, beta) lengthscale prior.

    Args:
    - prior: (alpha, beta) pair with alpha > 0 and beta > 0, or None (module default).
    Returns:
    - (alpha, beta) as floats.
    Raises:
    - ValueError for an invalid prior.
    """
    if prior is None:
        return tuple(float(v) for v in LENGTHSCALE_PRIOR)
    values = tuple(float(v) for v in np.asarray(prior, dtype=float).reshape(-1))
    if len(values) != 2 or not all(np.isfinite(v) and v > 0.0 for v in values):
        raise ValueError(f"lengthscale_prior must be a pair (alpha > 0, beta > 0), got {prior!r}")
    return values


def lengthscale_priors_per_level(lengthscale_prior, num_levels: int) -> list[tuple[float, float]]:
    """
    [MAP] One validated (alpha, beta) pair per level from None, a single pair or a list of
    num_levels pairs.
    """
    if lengthscale_prior is None:
        return [check_lengthscale_prior(None)] * num_levels
    array = np.asarray(lengthscale_prior, dtype=float)
    if array.shape == (2,):
        return [check_lengthscale_prior(array)] * num_levels
    if array.shape == (num_levels, 2):
        return [check_lengthscale_prior(pair) for pair in array]
    raise ValueError(f"lengthscale_prior must be None, one (alpha, beta) pair or "
                     f"{num_levels} pairs, got {lengthscale_prior!r}")


def safe_cholesky(matrix: np.ndarray) -> tuple[np.ndarray, float]:
    """
    [FIX-N5] Cholesky factorization with an increasing diagonal jitter if needed.

    Args:
    - matrix: symmetric (n, n) numpy array.
    Returns:
    - l_chol: lower triangular Cholesky factor.
    - jitter: the absolute jitter added to the diagonal (0.0 if none).
    """
    if not np.all(np.isfinite(matrix)):
        raise np.linalg.LinAlgError("Covariance matrix contains non finite values.")
    scale = max(float(np.mean(np.diag(matrix))), 1e-300)
    for rel_jitter in JITTERS:
        try:
            jitter = rel_jitter * scale
            return np.linalg.cholesky(matrix + jitter * np.eye(len(matrix))), jitter
        except np.linalg.LinAlgError:
            continue
    raise np.linalg.LinAlgError("Covariance matrix is not positive definite, even with jitter.")


class GaussianProcess:
    """
    Core of the Gaussian Process. This class handles
      the fitting and prediction of the GP model.
    """
    def __init__(self, kernel: Kernel, seed: int = None, use_map: bool = False,
                 lengthscale_prior: tuple[float, float] = None, priors: dict = None,
                 rho_centering: bool = False):
        self.x_train = None
        self.y_train = None
        self.kernel = kernel
        self.noise = 1e-6
        self.l_chol = None
        # [FIX-N4] alpha = K^-1 y_normalized replaces the explicit inverse matrix k_inv
        self.alpha = None
        self.jitter = 0.0
        # [FIX-N7] output normalization: raw value = y_mean + y_std * normalized value
        self.y_mean = 0.0
        self.y_std = 1.0
        # [FIX-T1] correlation coefficient with the previous level (None for the first level)
        self.rho = None
        # [FIX-R1] local seeded random generator for the restarts (was the global np.random)
        self.rng = np.random.default_rng(seed)
        # [MAP] True: MAP estimation of the hyperparameters (priors self.priors) instead of the
        # maximum likelihood
        self.use_map = bool(use_map)
        # [MAP] (alpha, beta) of the InvGamma prior of this GP (None -> LENGTHSCALE_PRIOR)
        self.lengthscale_prior = check_lengthscale_prior(lengthscale_prior)
        # [PRIORS] prior of each hyperparameter group: DEFAULT_PRIORS, InvGamma(lengthscale_prior)
        # on the lengthscales, then the entries of `priors` (None in `priors` = no prior; a
        # "lengthscale" entry of `priors` wins over lengthscale_prior)
        overrides = check_priors(priors)
        self.priors = {**DEFAULT_PRIORS, "lengthscale": ("invgamma",) + self.lengthscale_prior,
                       **overrides}
        # [RHO] True: the level below is centred too, so that the constant of the residual
        # y - rho f follows rho (False: constant fixed for rho = rho_init, which pulls the
        # profiled rho towards rho_init when the level below has a mean far from 0; mfego v1)
        self.rho_centering = bool(rho_centering)
        # [RHO] the default prior on rho goes with the centred rho (benchmarked together):
        # without centring, no prior on rho unless `priors` sets one (mfego v1 behaviour)
        if not self.rho_centering and "rho" not in overrides:
            self.priors["rho"] = None

    def fit(self, x_train, y_train, n_restarts=3, f_prev: np.ndarray = None,
            rho_init: float = 1.0, estimate_rho: bool = False,
            rho_bounds: tuple[float, float] = (-5.0, 5.0),
            fixed_lengthscale: np.ndarray = None) -> float:
        """
        Fit the Gaussian Process model to the training data and optimize hyperparameters.

        For a level l >= 2 of the multifidelity model, f_prev contains the predictions of the
        previous level at x_train and the GP models the residual y - rho * f_prev (Eq. 18).
        [FIX-T1] If estimate_rho is True, rho is profiled out of the likelihood with its
        closed form (Le Gratiet thesis Eq. 4.10, non-nested case B.1.2):
            rho_hat = (F^T K^-1 y) / (F^T K^-1 F),
        [RHO] or, with use_map and a ("normal", mean, std) prior on rho,
            rho_hat = (F^T K^-1 y + mean / std^2) / (F^T K^-1 F + 1 / std^2),
        [V3] or, with use_map and a ("lognormal", mu, s) prior on rho, rho > 0 is learned with
        the kernel hyperparameters (log rho is the last parameter of L-BFGS-B, bounds
        [max(rho_bounds[0], RHO_LEARNED_MIN), rho_bounds[1]]),
        otherwise rho is fixed to rho_init.
        [SHARE] fixed_lengthscale (d,): the lengthscales are FIXED to these values (lower bound =
        upper bound for L-BFGS-B) and only t1, t2, the noise (and rho) are estimated. Used by
        CoKriging(share_lengthscales=True): the correction of a level takes the lengthscales of
        the level below, learned on more points (vpp_acceleration/examples/joint_study.py).

        Returns:
        - rho: the correlation coefficient used (None if f_prev is None).
        """
        self.x_train = np.atleast_2d(np.asarray(x_train, dtype=float))
        y_obs = np.asarray(y_train, dtype=float).reshape(-1)
        n, d = self.x_train.shape
        if f_prev is not None:
            f_prev = np.asarray(f_prev, dtype=float).reshape(-1)
        learn_rho = self._learns_rho(f_prev, estimate_rho)

        # Define bounds for the hyperparameters
        # [FIX-N2] log-space bounds (see the module constants)
        param_bounds = [LENGTHSCALE_BOUNDS] * d \
                       + [SIGNAL_VARIANCE_BOUNDS, BIAS_VARIANCE_BOUNDS, NOISE_BOUNDS]
        if fixed_lengthscale is not None:
            # [SHARE] lower bound = upper bound: L-BFGS-B keeps the lengthscales at these values
            fixed = np.asarray(fixed_lengthscale, dtype=float).reshape(-1)
            if fixed.shape != (d,) or np.any(fixed <= 0.0):
                raise ValueError(f"fixed_lengthscale must be {d} positive values, got {fixed}")
            param_bounds[:d] = [(float(value), float(value)) for value in fixed]
        if learn_rho:
            # [V3] learned rho: one more (log) parameter, positive
            if rho_bounds[1] <= RHO_LEARNED_MIN:
                raise ValueError(f"A learned rho (lognormal prior) is positive: its upper bound "
                                 f"must exceed {RHO_LEARNED_MIN}, got rho_bounds={rho_bounds}")
            param_bounds.append((max(rho_bounds[0], RHO_LEARNED_MIN), rho_bounds[1]))
        log_bounds = [(np.log(lo), np.log(hi)) for lo, hi in param_bounds]

        # [FIX-N7] normalization at rho_init, then (if rho is estimated and rho_hat is far from
        # rho_init) a second, warm-started pass normalized at rho_hat so that the bounds of the
        # residual GP stay relevant whatever the value of rho.
        rho_ref = rho_init
        restarts = n_restarts
        for _ in range(2):
            y_n, f_n = self._normalize(y_obs, f_prev, rho_ref)
            rho_args = (y_n, f_n, rho_init, estimate_rho, rho_bounds)
            best_nll = np.inf
            best_params = None

            for init_guess in self._initial_guesses(d, restarts, log_bounds, learn_rho):

                res = minimize(self.negative_log_likelihood, init_guess,
                               args=rho_args + (self._kernel_has_gradients(), self.use_map),
                               jac=self._kernel_has_gradients(),
                               bounds=log_bounds, method='L-BFGS-B')
                if res.fun < best_nll:
                    best_nll = res.fun
                    best_params = res.x

            # [FIX-R2] explicit failure: a failed factorization returns NLL_FAILURE (finite),
            # so the previous `best_params is None` test could never be True.
            if best_params is None or not np.isfinite(best_nll) or best_nll >= NLL_FAILURE:
                raise ValueError(
                    f"GP fit failed: no positive definite covariance found (n={n}, d={d}, "
                    f"bounds={param_bounds}). Check for duplicated points or constant outputs.")

            params = np.exp(best_params)
            self.kernel.set_params(params[:d + 2])
            self.noise = params[d + 2]

            if learn_rho:
                self.rho = float(params[d + 3])
            else:
                covariance_matrix = self.kernel.get_covariance_matrix(self.x_train) \
                                           + self.noise * np.eye(n)
                l_chol, _ = safe_cholesky(covariance_matrix)
                self.rho = self._profile_rho(l_chol, *rho_args, self._rho_prior(self.use_map))
            if f_prev is None or not estimate_rho \
               or abs(self.rho - rho_ref) <= 1e-3 * max(1.0, abs(rho_ref)):
                break
            rho_ref, restarts = self.rho, 1

        # [FIX-N4] training target in raw units, then factorization shared with the reloading
        target_raw = y_obs if f_prev is None else y_obs - self.rho * f_prev
        if f_prev is not None and self.rho_centering:
            # [RHO] constant of the residual for the final rho
            self.y_mean = float(np.mean(target_raw))
        self.condition(self.x_train, target_raw)
        return self.rho

    def negative_log_likelihood(self, log_params: np.ndarray, y_n: np.ndarray,
                                f_n: np.ndarray = None, rho_init: float = 1.0,
                                estimate_rho: bool = False,
                                rho_bounds: tuple[float, float] = (-5.0, 5.0),
                                with_grad: bool = False, use_map: bool = False):
        """
        Negative log-likelihood (Eqs. 15-16 of the reference article) of the normalized
        residual y_n - rho * f_n, as a function of the LOG of [l_1..l_d, t1, t2, noise]
        ([V3] and rho as a last parameter when it is learned, see fit()).
        [FIX-N2] log-space parameters, [FIX-N3] returns (nll, gradient) if with_grad.
        [MAP] [PRIORS] if use_map, adds -log p of the prior of every hyperparameter group
        (self.priors, constant terms dropped): negative log-posterior, MAP in parameter space.
        (Was a closure inside fit(); it is a method now so that it can be unit-tested.)
        Side effect: sets the kernel parameters and the noise.
        """
        n, d = len(y_n), self.x_train.shape[1]
        params = np.exp(log_params)
        self.kernel.set_params(params[:d + 2])
        self.noise = params[d + 2]
        learned_rho = len(log_params) == d + 4

        try:
            # Compute the covariance matrix K and its Cholesky decomposition
            covariance_matrix = self.kernel.get_covariance_matrix(self.x_train) \
                                       + self.noise * np.eye(n)
            # [FIX-N5] jitter instead of an immediate failure
            l_chol, _ = safe_cholesky(covariance_matrix)
        except np.linalg.LinAlgError:
            return (NLL_FAILURE, np.zeros_like(log_params)) if with_grad else NLL_FAILURE

        # [FIX-T1] profiled (or fixed) rho, [V3] or learned rho, then residual target of Eq. 18
        if learned_rho:
            rho, rho_prior = float(params[d + 3]), None
        else:
            rho_prior = self._rho_prior(use_map)
            rho = self._profile_rho(l_chol, y_n, f_n, rho_init, estimate_rho, rho_bounds,
                                    rho_prior)
        target = y_n if f_n is None else y_n - rho * f_n

        # [FIX-N4] cho_solve exploits the triangular factor (was a generic solve)
        alpha = cho_solve((l_chol, True), target)
        log_det = 2.0 * np.sum(np.log(np.diag(l_chol)))
        data_fit = 0.5 * np.dot(target, alpha)

        # Eqs. (15), (16) from the reference article
        nll = data_fit + 0.5 * log_det + 0.5 * n * np.log(2 * np.pi)

        # [MAP] [PRIORS] -log p of the priors of l_1..l_d, t1, t2, the noise ([V3] and rho)
        if use_map:
            prior_value, prior_grad = self.neg_log_priors(log_params)
            nll += prior_value
        # [RHO] -log p(rho) of the normal prior, when rho is estimated
        if rho_prior is not None and f_n is not None and estimate_rho:
            nll += (rho - rho_prior[0]) ** 2 / (2.0 * rho_prior[1] ** 2)
        if not with_grad:
            #must be float for scipy
            return float(nll)

        # [FIX-N3] analytical gradient (GPML Eq. 5.9): 0.5 * tr((K^-1 - a a^T) dK/dlog(t)).
        # By the envelope theorem, d(rho_hat)/dtheta is not needed (d(NLL + prior)/drho = 0 at
        # rho_hat, which minimizes NLL + the prior on rho for the current theta).
        w_mat = cho_solve((l_chol, True), np.eye(n)) - np.outer(alpha, alpha)
        grad = [0.5 * np.sum(w_mat * d_k)
                for d_k in self.kernel.get_log_params_gradients(self.x_train)]
        grad.append(0.5 * self.noise * np.trace(w_mat))
        if learned_rho:
            # [V3] d(0.5 t^T K^-1 t) / dlog(rho) with t = y_n - rho f_n
            grad.append(-rho * float(np.dot(f_n, alpha)))
        grad = np.array(grad)
        if use_map:
            grad += prior_grad
        return float(nll), grad

    def neg_log_priors(self, log_params: np.ndarray) -> tuple[float, np.ndarray]:
        """
        [PRIORS] Sum of -log p over the hyperparameter groups that have a prior (self.priors),
        and its gradient w.r.t. the log-parameters [log l_1..log l_d, log t1, log t2, log noise]
        ([V3] and log rho when rho is learned: lognormal prior of the "rho" group).
        """
        d = self.x_train.shape[1]
        value, grad = 0.0, np.zeros(len(log_params))
        groups = [("lengthscale", slice(0, d)), ("signal_variance", slice(d, d + 1)),
                  ("bias_variance", slice(d + 1, d + 2)), ("noise", slice(d + 2, d + 3))]
        if len(log_params) == d + 4:
            groups.append(("rho", slice(d + 3, d + 4)))
        for group, index in groups:
            if self.priors[group] is not None:
                group_value, group_grad = neg_log_prior(self.priors[group], log_params[index])
                value += group_value
                grad[index] += group_grad
        return value, grad

    def _normalize(self, y_obs: np.ndarray, f_prev: np.ndarray,
                   rho_ref: float) -> tuple[np.ndarray, np.ndarray]:
        """
        [FIX-N7] Sets y_std (and y_mean) from the residual y - rho_ref * f_prev and returns the
        normalized observations and previous-level predictions.
        [RHO] With rho_centering, y and f_prev are both centred: the normalized residual
        y_n - rho f_n = (y - mean(y) - rho (f - mean(f))) / y_std has a zero mean for ANY rho,
        and y_mean is set at the end of fit(), for the final rho.
        """
        residual = y_obs if f_prev is None else y_obs - rho_ref * f_prev
        mean, std = float(np.mean(residual)), float(np.std(residual))
        self.y_std = std if std > 1e-12 * max(1.0, abs(mean)) else 1.0
        if f_prev is not None and self.rho_centering:
            return ((y_obs - np.mean(y_obs)) / self.y_std,
                    (f_prev - np.mean(f_prev)) / self.y_std)
        self.y_mean = mean
        y_n = (y_obs - self.y_mean) / self.y_std
        f_n = None if f_prev is None else f_prev / self.y_std
        return y_n, f_n

    def condition(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        [FIX-N4] Conditions the GP on (x_train, y_train) with the CURRENT hyperparameters and
        normalization constants (no optimization): Cholesky factor and alpha = K^-1 y.
        Used by fit() and to rebuild a model from a JSON file (single source of truth).

        Args:
        - x_train: numpy array of shape (n, d).
        - y_train: numpy array of shape (n,), the raw target modelled by this GP.
        """
        self.x_train = np.atleast_2d(np.asarray(x_train, dtype=float))
        self.y_train = np.asarray(y_train, dtype=float).reshape(-1)

        covariance_matrix = self.kernel.get_covariance_matrix(self.x_train) \
                                   + self.noise * np.eye(len(self.x_train))
        self.l_chol, self.jitter = safe_cholesky(covariance_matrix)
        y_n = (self.y_train - self.y_mean) / self.y_std
        self.alpha = cho_solve((self.l_chol, True), y_n)

    def predict(self, x_new: np.ndarray) -> tuple[float, float]:
        """
        Predicts the mean and variance of the Gaussian Process at new input point.
        (Eqs. 4-5: the variance includes the observation noise.)
        """
        f_hat, sigma2_hat = self.predict_batch(np.atleast_2d(x_new))
        return float(f_hat[0]), float(sigma2_hat[0])

    def predict_batch(self, x_new: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        [FIX-N4/N6] Vectorized prediction at m points: mean k^T alpha in O(n) per point and
        variance with a triangular solve (no explicit inverse).

        Args:
        - x_new: numpy array of shape (m, d).
        Returns:
        - f_hat: numpy array of shape (m,), predicted means (raw units).
        - sigma2_hat: numpy array of shape (m,), predicted variances incl. noise (raw units).
        """
        k_mat = self.kernel.get_cross_covariance_matrix(np.atleast_2d(x_new), self.x_train)
        kappa = self.kernel.signal_variance + self.kernel.bias_variance

        f_hat_n = k_mat @ self.alpha
        v_mat = solve_triangular(self.l_chol, k_mat.T, lower=True)
        # [FIX-T4] clipped at 0 (round-off could give tiny negative variances)
        sigma2_hat_n = np.maximum(kappa + self.noise - np.sum(v_mat ** 2, axis=0), 0.0)

        return self.y_mean + self.y_std * f_hat_n, (self.y_std ** 2) * sigma2_hat_n

    def get_noise_variance(self) -> float:
        """[FIX-N7] Observation noise variance in raw output units."""
        return float(self.noise * self.y_std ** 2)

    def _kernel_has_gradients(self) -> bool:
        """[FIX-N3] True if the kernel overrides get_log_params_gradients (analytical gradients),
        otherwise L-BFGS-B falls back to finite differences."""
        return type(self.kernel).get_log_params_gradients is not \
               Kernel.get_log_params_gradients

    def _initial_guesses(self, d: int, n_restarts: int, log_bounds: list[tuple[float, float]],
                         learn_rho: bool = False) -> list[np.ndarray]:
        """
        Initial guesses (log-space) for the L-BFGS-B restarts.
        [FIX-N8] the first restart is warm-started from the previous hyperparameters (if any),
        the others keep the previous random initialization.
        [V3] learned rho: the previous rho (if positive) for the first restart, 1 for the others.
        """
        lows = np.array([b[0] for b in log_bounds])
        highs = np.array([b[1] for b in log_bounds])
        rho_guess = [self.rho if self.rho is not None and self.rho > 0 else 1.0] if learn_rho \
            else []
        guesses = []
        if self.kernel.lengthscale is not None and len(self.kernel.lengthscale) == d:
            previous = np.concatenate([self.kernel.get_params(), [self.noise], rho_guess])
            guesses.append(np.clip(np.log(previous), lows, highs))
        while len(guesses) < max(n_restarts, 1):
            init_guess = np.concatenate((self.rng.uniform(0.2, 1.5, d), [1.0, 1e-4, 1e-6],
                                         [1.0] if learn_rho else []))
            guesses.append(np.clip(np.log(init_guess), lows, highs))
        return guesses

    def _learns_rho(self, f_prev: np.ndarray, estimate_rho: bool) -> bool:
        """[V3] True if rho is learned with the other hyperparameters (estimated rho, MAP with a
        lognormal prior on rho, which has no closed form); False: profiled or fixed rho."""
        prior = self.priors.get("rho")
        return (f_prev is not None and estimate_rho and self.use_map and prior is not None
                and prior[0] == "lognormal")

    def _rho_prior(self, use_map: bool):
        """[RHO] (mean, std) of the normal prior on rho if it is active (use_map), else None."""
        prior = self.priors.get("rho")
        return None if not use_map or prior is None or prior[0] != "normal" \
            else (prior[1], prior[2])

    @staticmethod
    def _profile_rho(l_chol: np.ndarray, y_n: np.ndarray, f_n: np.ndarray, rho_init: float,
                     estimate_rho: bool, rho_bounds: tuple[float, float],
                     rho_prior: tuple[float, float] = None) -> float:
        """
        [FIX-T1] Closed-form (GLS) estimate of rho for fixed kernel hyperparameters
        (Le Gratiet thesis Eq. 4.10 with H = F): rho = F^T K^-1 y / F^T K^-1 F, clipped.
        [RHO] With a normal prior (mean, std) on rho (MAP):
            rho = (F^T K^-1 y + mean / std^2) / (F^T K^-1 F + 1 / std^2).
        Returns None for the first level and rho_init if rho is not estimated.
        """
        if f_n is None:
            return None
        if not estimate_rho:
            return float(rho_init)
        k_inv_f = cho_solve((l_chol, True), f_n)
        numerator, denominator = float(np.dot(k_inv_f, y_n)), float(np.dot(f_n, k_inv_f))
        if rho_prior is not None:
            numerator += rho_prior[0] / rho_prior[1] ** 2
            denominator += 1.0 / rho_prior[1] ** 2
        if denominator <= 1e-14:
            return float(rho_init)
        return float(np.clip(numerator / denominator, *rho_bounds))

    def to_dict(self) -> dict:
        """[FIX-X1] Hyperparameters and normalization constants of the GP (JSON friendly)."""
        return {
            "kernel_params": self.kernel.get_params().tolist(),
            "noise": float(self.noise),
            "y_mean": float(self.y_mean),
            "y_std": float(self.y_std),
            "rho": None if self.rho is None else float(self.rho),
        }

    def diagnostics(self) -> dict:
        """
        [MAP] Fitted hyperparameters of the GP and the ones sitting at a search bound (a
        lengthscale at its upper bound usually means an inactive input, at its lower bound an
        overfitted / white-noise model; a noise at its upper bound means the bound is too tight).

        Returns:
        - dict with the estimator, the lengthscales, t1, t2, the noise (normalized and raw),
          rho, n and the lists of lengthscale indices at the lower / upper bound.
        """
        estimator = f"MAP ({describe_priors(self.priors)})" if self.use_map else "MLE"
        if self.kernel.lengthscale is None:
            return {"estimator": estimator, "fitted": False}
        lengthscales = np.asarray(self.kernel.lengthscale, dtype=float)
        lo, hi = LENGTHSCALE_BOUNDS
        return {
            "fitted": True,
            "estimator": estimator,
            "n_train": 0 if self.x_train is None else int(len(self.x_train)),
            "lengthscales": lengthscales.tolist(),
            "signal_variance": float(self.kernel.signal_variance),
            "bias_variance": float(self.kernel.bias_variance),
            "noise_normalized": float(self.noise),
            "noise_variance": self.get_noise_variance(),
            "y_std": float(self.y_std),
            "rho": None if self.rho is None else float(self.rho),
            "lengthscales_at_lower_bound": np.flatnonzero(
                lengthscales <= lo * BOUND_FLAG_MARGIN).tolist(),
            "lengthscales_at_upper_bound": np.flatnonzero(
                lengthscales >= hi / BOUND_FLAG_MARGIN).tolist(),
            "noise_at_upper_bound": bool(self.noise >= NOISE_BOUNDS[1] / BOUND_FLAG_MARGIN),
        }

# [FIX-X1] registry used to rebuild a model from a JSON file
KERNELS = {"SquaredExponentialKernel": SquaredExponentialKernel,
           "Matern52Kernel": Matern52Kernel}


class MultifidelityModel:
    """
    Multifidelity Gaussian Process model that combines multiple fidelity levels.
    """
    def __init__(self, l, kernel_class: Kernel, estimate_rho: bool = True,
                 rho_init: float = 1.0, rho_bounds: tuple[float, float] = (-5.0, 5.0),
                 min_points_rho: int = None, n_restarts: int = 3, seed: int = None,
                 use_map: bool = False, lengthscale_prior=None, priors: dict = None,
                 rho_centering: bool = False):
        self.num_levels = l
        self.kernel_class = kernel_class
        # [MAP] MAP estimation (priors on the hyperparameters) of every level
        self.use_map = bool(use_map)
        # [MAP] InvGamma(alpha, beta) prior of each level: None (LENGTHSCALE_PRIOR), one
        # (alpha, beta) pair for every level, or a list of l pairs (one per level)
        self.lengthscale_priors = lengthscale_priors_per_level(lengthscale_prior, l)
        # [PRIORS] overrides of DEFAULT_PRIORS shared by every level (None: default priors;
        # LENGTHSCALE_ONLY_PRIORS: prior on the lengthscales only, as before)
        self.priors = check_priors(priors)
        # [RHO] centred level below in the profiled rho, with the N(1, 2^2) prior if use_map
        # (see GaussianProcess). Default False (mfego v1): the v2 rho improves the surrogate on
        # nested designs (CoKriging) but not NN-MF-EGO (benchmarks/surrogate_v2/RAPPORT_V2.md)
        self.rho_centering = bool(rho_centering)
        # List to hold GaussianProcess instances for each fidelity level
        # [FIX-R1] each GP gets its own seed (reproducible restarts)
        self.gps = [GaussianProcess(kernel_class(), seed=None if seed is None else seed + i,
                                    use_map=use_map,
                                    lengthscale_prior=self.lengthscale_priors[i],
                                    priors=self.priors, rho_centering=rho_centering)
                    for i in range(l)]
        # [NORM] bounds of the design space, set by fit(): the GPs work in [0, 1]^d (the domain
        # of the priors and of the search bounds); None = inputs used as they are
        self.input_bounds = None
        # Initialize correlation coefficients between levels
        # (level 1 has no rho since Y(0) = 0, Sacher Eq. 9: rhos[l-2] links level l-1 to l)
        self.rhos = [rho_init for _ in range(l - 1)]
        # [FIX-T1c] default: rho is ALWAYS computed, for every level l >= 2 (level 2 included):
        # rho_(l-1) is a parameter of the likelihood of level l (Sacher Eq. 15, Algorithm 1),
        # profiled with its closed form (Le Gratiet Eq. 4.10).
        # [FIX-T1b] options: min_points_rho=k keeps rho = rho_init until a level has k points
        # (hybrid mode, no fallback by default); estimate_rho=False -> rho = rho_init.
        self.estimate_rho = estimate_rho
        self.rho_init = rho_init
        self.rho_bounds = rho_bounds
        self.min_points_rho = min_points_rho
        self.n_restarts = n_restarts
        # [FIX-X1] data actually used by the last fit (exact export / reloading)
        self.train_data = {}

    def estimator_label(self) -> str:
        """[MAP] Readable description of the hyperparameter estimation (for the logs)."""
        if not self.use_map:
            return "MLE"
        labels = sorted({describe_priors(gp.priors) for gp in self.gps})
        if len(labels) == 1:
            return f"MAP, priors: {labels[0]}"
        return "MAP, priors per level: " + " | ".join(describe_priors(gp.priors)
                                                     for gp in self.gps)

    def fit(self, experiment_data: ExperimentData) -> None:
        """
        Fit one Gaussian process to each fidelity level in the data.
        """
        logger.debug("Fitting the multifidelity model (%s).", self.estimator_label())
        # [NORM] the GPs see the inputs scaled to [0, 1] with the bounds of the design space
        self.input_bounds = np.asarray(experiment_data.bounds, dtype=float)
        for l in range(1, self.num_levels + 1):
            # [FIX-R4] failed evaluations (NaN) are excluded from the training data
            x_l, y_l = experiment_data.get_training_data(l)
            if len(y_l) == 0:
                raise ValueError(f"No valid observation available at fidelity level {l}.")
            self.train_data[l] = (x_l, y_l)
            x_l = scale_to_unit(x_l, self.input_bounds)

            if l == 1:
                self.gps[0].fit(x_l, y_l, n_restarts = self.n_restarts)
            else:
                #NoN nested approach (Eq. 18): residual w.r.t. the previous level prediction
                f_prev = self._predict_batch_up_to(x_l, l - 1)[0]
                # [FIX-T1c] rho estimated (profiled) at every fit; [FIX-T1b] optional
                # fallback to rho_init while the level has fewer than min_points_rho points
                min_points = self.min_points_rho if self.min_points_rho is not None else 0
                estimate = self.estimate_rho and len(y_l) >= min_points
                self.rhos[l - 2] = self.gps[l - 1].fit(
                    x_l, y_l, n_restarts = self.n_restarts, f_prev = f_prev,
                    rho_init = self.rho_init, estimate_rho = estimate,
                    rho_bounds = self.rho_bounds)

            logger.info("GP level %s trained (n=%d). Noise: %.3e | rho: %s", l, len(y_l),
                        self.gps[l - 1].get_noise_variance(),
                        "-" if l == 1 else f"{self.rhos[l - 2]:.4f}")
            # [MAP] hyperparameters at a search bound (see GaussianProcess.diagnostics)
            diag = self.gps[l - 1].diagnostics()
            if diag["lengthscales_at_lower_bound"] or diag["noise_at_upper_bound"]:
                logger.warning("GP level %s (%s): lengthscales %s at the lower bound, noise at "
                               "the upper bound: %s (possible degenerate fit).", l,
                               diag["estimator"], diag["lengthscales_at_lower_bound"],
                               diag["noise_at_upper_bound"])

    def diagnostics(self) -> list[dict]:
        """
        [MAP] GaussianProcess.diagnostics() of every level (fitted hyperparameters, rho and
        the values at a search bound), e.g. to store them in results.json.
        """
        return [{"level": l + 1, **gp.diagnostics()} for l, gp in enumerate(self.gps)]

    def is_fitted(self) -> bool:
        """[FIX-X1] True if every GP of the model has been fitted (or rebuilt)."""
        return all(gp.alpha is not None for gp in self.gps)

    def _predict_up_to(self, x_new: np.ndarray, level: int) -> tuple[float, float]:
        """
        Predict the mean and variance of the multifidelity model up to a specified fidelity level.
        """
        f_hat, sigma_2_hat, _ = self.predict_batch(x_new, level)
        return float(f_hat[0]), float(sigma_2_hat[0])

    def _predict_batch_up_to(self, x_new: np.ndarray, level: int
                             ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        [FIX-N6] Vectorized recursive prediction (Eqs. 11-12) up to a given level.
        [NORM] x_new is in the (scaled) input space of the GPs: use predict_batch for raw inputs.

        Returns:
        - f_hat: (m,) means, sigma_2_hat: (m,) variances, gp_variances: (m, level) variances
          of each discrepancy GP.
        """
        x_new = np.atleast_2d(x_new)
        f_hat = np.zeros(len(x_new))
        sigma_2_hat = np.zeros(len(x_new))
        gp_variances = np.zeros((len(x_new), level))

        for l in range(1, level + 1):
            delta_f, delta_sigma2 = self.gps[l-1].predict_batch(x_new)
            gp_variances[:, l - 1] = delta_sigma2

            if l == 1:
                f_hat = delta_f
                sigma_2_hat = delta_sigma2
            else:
                rho = self.rhos[l - 2]
                f_hat = rho * f_hat + delta_f
                sigma_2_hat = (rho**2) * sigma_2_hat + delta_sigma2
        return f_hat, sigma_2_hat, gp_variances

    def predict(self, x_new):
        """
        Return the final prediction AND individual variances for the merit function
        """
        f_hat, sigma_2_hat, gp_variances = self.predict_batch(x_new)
        return float(f_hat[0]), float(sigma_2_hat[0]), gp_variances[0].tolist()

    def predict_batch(self, x_new: np.ndarray, level: int = None
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        [FIX-X1] Vectorized prediction at m points, at the highest level (default) or at an
        intermediate level (the recursive formulation provides a surrogate for every level).

        Args:
        - x_new: numpy array of shape (m, d) (or (d,) for a single point).
        - level: fidelity level of the surrogate (default: num_levels).
        Returns:
        - mean (m,), variance (m,) and variances of each discrepancy GP (m, level).
        """
        level = self.num_levels if level is None else level
        return self._predict_batch_up_to(scale_to_unit(np.atleast_2d(x_new), self.input_bounds),
                                         level)

    def to_dict(self) -> dict:
        """
        [FIX-X1] Complete JSON-friendly state of the trained model: data used by the last fit,
        hyperparameters, normalization constants and rhos. from_dict() rebuilds exactly the
        same model (no re-training).
        """
        levels = []
        for l in range(1, self.num_levels + 1):
            x_l, y_l = self.train_data[l]
            level_state = self.gps[l - 1].to_dict()
            level_state.update({"x_train": x_l.tolist(), "y_train": y_l.tolist()})
            levels.append(level_state)
        return {
            "kernel": self.kernel_class.__name__,
            "num_levels": self.num_levels,
            "rhos": [float(r) for r in self.rhos],
            "estimate_rho": self.estimate_rho,
            "rho_init": self.rho_init,
            "rho_bounds": list(self.rho_bounds),
            "min_points_rho": self.min_points_rho,
            "use_map": self.use_map,
            "lengthscale_prior": [list(p) for p in self.lengthscale_priors],
            # [PRIORS] priors actually used (self-describing file: a later change of
            # DEFAULT_PRIORS does not change a reloaded model; the lengthscale prior of each level
            # is in "lengthscale_prior" unless "priors" sets it)
            "priors": saved_priors(self.gps[0].priors, self.priors),
            "rho_centering": self.rho_centering,
            "input_bounds": None if self.input_bounds is None else self.input_bounds.tolist(),
            "fit_sizes":[len(self.train_data[l][1]) for l in range(1, self.num_levels + 1)],
            "levels": levels,
        }

    @classmethod
    def from_dict(cls, state: dict, kernel_class: Kernel = None) -> "MultifidelityModel":
        """
        [FIX-X1] Rebuilds a trained model from to_dict() output (same predictions, no fit).
        The residual targets of the levels l >= 2 are recomputed with Eq. 18.
        """
        kernel_class = kernel_class or KERNELS[state.get("kernel", "SquaredExponentialKernel")]
        # [PRIORS] files written before the priors on t1, t2 and the noise: lengthscale prior
        # only; [RHO] files written before rho_centering (mfego v1 and older): no prior on rho,
        # constant of the residual at rho_init (same model if it is fitted again)
        legacy = "rho_centering" not in state
        priors = dict(state.get("priors", LENGTHSCALE_ONLY_PRIORS))
        if legacy:
            priors["rho"] = None
        model = cls(state["num_levels"], kernel_class,
                    estimate_rho=state.get("estimate_rho", False),
                    rho_init=state.get("rho_init", 1.0),
                    rho_bounds=tuple(state.get("rho_bounds", (-5.0, 5.0))),
                    min_points_rho=state.get("min_points_rho"),
                    use_map=state.get("use_map", False),
                    # [MAP] files written before the configurable prior: module default
                    lengthscale_prior=state.get("lengthscale_prior"),
                    priors=priors, rho_centering=state.get("rho_centering", False))
        model.rhos = [float(r) for r in state.get("rhos", [1.0] * (model.num_levels - 1))]
        # [NORM] files written before the input normalization: inputs used as they are
        bounds = state.get("input_bounds")
        model.input_bounds = None if bounds is None else np.asarray(bounds, dtype=float)

        for l in range(1, model.num_levels + 1):
            level_state = state["levels"][l - 1]
            gp = model.gps[l - 1]
            gp.kernel.set_params(np.array(level_state["kernel_params"], dtype=float))
            gp.noise = float(level_state["noise"])
            gp.y_mean = float(level_state.get("y_mean", 0.0))
            gp.y_std = float(level_state.get("y_std", 1.0))
            x_l = np.atleast_2d(np.array(level_state["x_train"], dtype=float))
            y_l = np.array(level_state["y_train"], dtype=float).reshape(-1)
            model.train_data[l] = (x_l, y_l)
            x_l = scale_to_unit(x_l, model.input_bounds)

            if l == 1:
                target_y = y_l
            else: # this is essentially Eq. 13/18 of the reference article
                gp.rho = model.rhos[l - 2]
                target_y = y_l - gp.rho * model._predict_batch_up_to(x_l, l - 1)[0]
            gp.condition(x_l, target_y)
        return model


def load_surrogate(json_filepath: str) -> MultifidelityModel:
    """
    [FIX-X1] Loads a ready-to-use surrogate from a file written by
    EGOOptimizer.export_surrogate() or EGOOptimizer.save_state().

    Example:
        model = load_surrogate("surrogate.json")
        mean, variance, _ = model.predict_batch(x)
    """
    with open(json_filepath, 'r', encoding='utf-8') as f:
        state = json.load(f)
    if "surrogate" not in state:
        raise KeyError(f"No 'surrogate' entry in {json_filepath} (file written by an older "
                       "version: use ModelVisualizer, which rebuilds it from the data).")
    return MultifidelityModel.from_dict(state["surrogate"])
