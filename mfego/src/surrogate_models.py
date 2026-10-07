"""
Core Module, implementing the Gaussian Process and Multifidelity Model classes.
"""
import json
import logging

import numpy as np
from scipy.linalg import cho_solve, solve_triangular
from scipy.optimize import minimize
from src.data_management import ExperimentData
from src.kernels import Kernel, SquaredExponentialKernel

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
    def __init__(self, kernel: Kernel, seed: int = None):
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

    def fit(self, x_train, y_train, n_restarts=3, f_prev: np.ndarray = None,
            rho_init: float = 1.0, estimate_rho: bool = False,
            rho_bounds: tuple[float, float] = (-5.0, 5.0)) -> float:
        """
        Fit the Gaussian Process model to the training data and optimize hyperparameters.

        For a level l >= 2 of the multifidelity model, f_prev contains the predictions of the
        previous level at x_train and the GP models the residual y - rho * f_prev (Eq. 18).
        [FIX-T1] If estimate_rho is True, rho is profiled out of the likelihood with its
        closed form (Le Gratiet thesis Eq. 4.10, non-nested case B.1.2):
            rho_hat = (F^T K^-1 y) / (F^T K^-1 F),
        otherwise rho is fixed to rho_init.

        Returns:
        - rho: the correlation coefficient used (None if f_prev is None).
        """
        self.x_train = np.atleast_2d(np.asarray(x_train, dtype=float))
        y_obs = np.asarray(y_train, dtype=float).reshape(-1)
        n, d = self.x_train.shape
        if f_prev is not None:
            f_prev = np.asarray(f_prev, dtype=float).reshape(-1)

        # Define bounds for the hyperparameters
        # [FIX-N2] log-space bounds (see the module constants)
        param_bounds = [LENGTHSCALE_BOUNDS] * d                        + [SIGNAL_VARIANCE_BOUNDS, BIAS_VARIANCE_BOUNDS, NOISE_BOUNDS]
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

            for init_guess in self._initial_guesses(d, restarts, log_bounds):

                res = minimize(self.negative_log_likelihood, init_guess,
                               args=rho_args + (self._kernel_has_gradients(),),
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
            self.kernel.set_params(params[:-1])
            self.noise = params[-1]

            covariance_matrix = self.kernel.get_covariance_matrix(self.x_train)                                        + self.noise * np.eye(n)
            l_chol, _ = safe_cholesky(covariance_matrix)
            self.rho = self._profile_rho(l_chol, *rho_args)
            if f_prev is None or not estimate_rho                or abs(self.rho - rho_ref) <= 1e-3 * max(1.0, abs(rho_ref)):
                break
            rho_ref, restarts = self.rho, 1

        # [FIX-N4] training target in raw units, then factorization shared with the reloading
        target_raw = y_obs if f_prev is None else y_obs - self.rho * f_prev
        self.condition(self.x_train, target_raw)
        return self.rho

    def negative_log_likelihood(self, log_params: np.ndarray, y_n: np.ndarray,
                                f_n: np.ndarray = None, rho_init: float = 1.0,
                                estimate_rho: bool = False,
                                rho_bounds: tuple[float, float] = (-5.0, 5.0),
                                with_grad: bool = False):
        """
        Negative log-likelihood (Eqs. 15-16 of the reference article) of the normalized
        residual y_n - rho * f_n, as a function of the LOG of [l_1..l_d, t1, t2, noise].
        [FIX-N2] log-space parameters, [FIX-N3] returns (nll, gradient) if with_grad.
        (Was a closure inside fit(); it is a method now so that it can be unit-tested.)
        Side effect: sets the kernel parameters and the noise.
        """
        n = len(y_n)
        params = np.exp(log_params)
        self.kernel.set_params(params[:-1])
        self.noise = params[-1]

        try:
            # Compute the covariance matrix K and its Cholesky decomposition
            covariance_matrix = self.kernel.get_covariance_matrix(self.x_train)                                        + self.noise * np.eye(n)
            # [FIX-N5] jitter instead of an immediate failure
            l_chol, _ = safe_cholesky(covariance_matrix)
        except np.linalg.LinAlgError:
            return (NLL_FAILURE, np.zeros_like(log_params)) if with_grad else NLL_FAILURE

        # [FIX-T1] profiled (or fixed) rho, then residual target of Eq. 18
        rho = self._profile_rho(l_chol, y_n, f_n, rho_init, estimate_rho, rho_bounds)
        target = y_n if f_n is None else y_n - rho * f_n

        # [FIX-N4] cho_solve exploits the triangular factor (was a generic solve)
        alpha = cho_solve((l_chol, True), target)
        log_det = 2.0 * np.sum(np.log(np.diag(l_chol)))
        data_fit = 0.5 * np.dot(target, alpha)

        # Eqs. (15), (16) from the reference article
        nll = data_fit + 0.5 * log_det + 0.5 * n * np.log(2 * np.pi)
        if not with_grad:
            #must be float for scipy
            return float(nll)

        # [FIX-N3] analytical gradient (GPML Eq. 5.9): 0.5 * tr((K^-1 - a a^T) dK/dlog(t)).
        # By the envelope theorem, d(rho_hat)/dtheta is not needed (dNLL/drho = 0 at rho_hat).
        w_mat = cho_solve((l_chol, True), np.eye(n)) - np.outer(alpha, alpha)
        grad = [0.5 * np.sum(w_mat * d_k)
                for d_k in self.kernel.get_log_params_gradients(self.x_train)]
        grad.append(0.5 * self.noise * np.trace(w_mat))
        return float(nll), np.array(grad)

    def _normalize(self, y_obs: np.ndarray, f_prev: np.ndarray,
                   rho_ref: float) -> tuple[np.ndarray, np.ndarray]:
        """
        [FIX-N7] Sets y_mean / y_std from the residual y - rho_ref * f_prev and returns the
        normalized observations and previous-level predictions.
        """
        residual = y_obs if f_prev is None else y_obs - rho_ref * f_prev
        self.y_mean = float(np.mean(residual))
        std = float(np.std(residual))
        self.y_std = std if std > 1e-12 * max(1.0, abs(self.y_mean)) else 1.0
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

    def _initial_guesses(self, d: int, n_restarts: int,
                         log_bounds: list[tuple[float, float]]) -> list[np.ndarray]:
        """
        Initial guesses (log-space) for the L-BFGS-B restarts.
        [FIX-N8] the first restart is warm-started from the previous hyperparameters (if any),
        the others keep the previous random initialization.
        """
        lows = np.array([b[0] for b in log_bounds])
        highs = np.array([b[1] for b in log_bounds])
        guesses = []
        if self.kernel.lengthscale is not None and len(self.kernel.lengthscale) == d:
            previous = np.concatenate([self.kernel.get_params(), [self.noise]])
            guesses.append(np.clip(np.log(previous), lows, highs))
        while len(guesses) < max(n_restarts, 1):
            init_guess = np.concatenate((self.rng.uniform(0.2, 1.5, d), [1.0, 1e-4, 1e-6]))
            guesses.append(np.clip(np.log(init_guess), lows, highs))
        return guesses

    @staticmethod
    def _profile_rho(l_chol: np.ndarray, y_n: np.ndarray, f_n: np.ndarray, rho_init: float,
                     estimate_rho: bool, rho_bounds: tuple[float, float]) -> float:
        """
        [FIX-T1] Closed-form (GLS) estimate of rho for fixed kernel hyperparameters
        (Le Gratiet thesis Eq. 4.10 with H = F): rho = F^T K^-1 y / F^T K^-1 F, clipped.
        Returns None for the first level and rho_init if rho is not estimated.
        """
        if f_n is None:
            return None
        if not estimate_rho:
            return float(rho_init)
        k_inv_f = cho_solve((l_chol, True), f_n)
        denominator = float(np.dot(f_n, k_inv_f))
        if denominator <= 1e-14:
            return float(rho_init)
        return float(np.clip(np.dot(k_inv_f, y_n) / denominator, *rho_bounds))

    def to_dict(self) -> dict:
        """[FIX-X1] Hyperparameters and normalization constants of the GP (JSON friendly)."""
        return {
            "kernel_params": self.kernel.get_params().tolist(),
            "noise": float(self.noise),
            "y_mean": float(self.y_mean),
            "y_std": float(self.y_std),
            "rho": None if self.rho is None else float(self.rho),
        }

# [FIX-X1] registry used to rebuild a model from a JSON file
KERNELS = {"SquaredExponentialKernel": SquaredExponentialKernel}


class MultifidelityModel:
    """
    Multifidelity Gaussian Process model that combines multiple fidelity levels.
    """
    def __init__(self, l, kernel_class: Kernel, estimate_rho: bool = True,
                 rho_init: float = 1.0, rho_bounds: tuple[float, float] = (-5.0, 5.0),
                 min_points_rho: int = None, n_restarts: int = 3, seed: int = None):
        self.num_levels = l
        self.kernel_class = kernel_class
        # List to hold GaussianProcess instances for each fidelity level
        # [FIX-R1] each GP gets its own seed (reproducible restarts)
        self.gps = [GaussianProcess(kernel_class(), seed=None if seed is None else seed + i)
                    for i in range(l)]
        # Initialize correlation coefficients between levels
        # (level 1 has no rho since Y(0) = 0, Sacher Eq. 9: rhos[l-2] links level l-1 to l)
        self.rhos = [rho_init for _ in range(l - 1)]
        # [FIX-T1/T1b] rho options: profiled estimate (Sacher Eq. 15), or fixed to rho_init,
        # or fixed until a level has at least min_points_rho points (default: d + 4)
        self.estimate_rho = estimate_rho
        self.rho_init = rho_init
        self.rho_bounds = rho_bounds
        self.min_points_rho = min_points_rho
        self.n_restarts = n_restarts
        # [FIX-X1] data actually used by the last fit (exact export / reloading)
        self.train_data = {}

    def fit(self, experiment_data: ExperimentData) -> None:
        """
        Fit one Gaussian process to each fidelity level in the data.
        """
        for l in range(1, self.num_levels + 1):
            # [FIX-R4] failed evaluations (NaN) are excluded from the training data
            x_l, y_l = experiment_data.get_training_data(l)
            if len(y_l) == 0:
                raise ValueError(f"No valid observation available at fidelity level {l}.")
            self.train_data[l] = (x_l, y_l)

            if l == 1:
                self.gps[0].fit(x_l, y_l, n_restarts = self.n_restarts)
            else:
                #NoN nested approach (Eq. 18): residual w.r.t. the previous level prediction
                f_prev = self._predict_batch_up_to(x_l, l - 1)[0]
                # [FIX-T1/T1b] rho is estimated (profiled) only with enough points
                min_points = self.min_points_rho if self.min_points_rho is not None \
                             else x_l.shape[1] + 4
                estimate = self.estimate_rho and len(y_l) >= min_points
                self.rhos[l - 2] = self.gps[l - 1].fit(
                    x_l, y_l, n_restarts = self.n_restarts, f_prev = f_prev,
                    rho_init = self.rho_init, estimate_rho = estimate,
                    rho_bounds = self.rho_bounds)

            logger.info("GP level %s trained (n=%d). Noise: %.3e | rho: %s", l, len(y_l),
                        self.gps[l - 1].get_noise_variance(),
                        "-" if l == 1 else f"{self.rhos[l - 2]:.4f}")

    def is_fitted(self) -> bool:
        """[FIX-X1] True if every GP of the model has been fitted (or rebuilt)."""
        return all(gp.alpha is not None for gp in self.gps)

    def _predict_up_to(self, x_new: np.ndarray, level: int) -> tuple[float, float]:
        """
        Predict the mean and variance of the multifidelity model up to a specified fidelity level.
        """
        f_hat, sigma_2_hat, _ = self._predict_batch_up_to(np.atleast_2d(x_new), level)
        return float(f_hat[0]), float(sigma_2_hat[0])

    def _predict_batch_up_to(self, x_new: np.ndarray, level: int
                             ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        [FIX-N6] Vectorized recursive prediction (Eqs. 11-12) up to a given level.

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
        f_hat, sigma_2_hat, gp_variances = self._predict_batch_up_to(
            np.atleast_2d(x_new), self.num_levels)
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
        return self._predict_batch_up_to(np.atleast_2d(x_new), level)

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
            "fit_sizes": [len(self.train_data[l][1]) for l in range(1, self.num_levels + 1)],
            "levels": levels,
        }

    @classmethod
    def from_dict(cls, state: dict, kernel_class: Kernel = None) -> "MultifidelityModel":
        """
        [FIX-X1] Rebuilds a trained model from to_dict() output (same predictions, no fit).
        The residual targets of the levels l >= 2 are recomputed with Eq. 18.
        """
        kernel_class = kernel_class or KERNELS[state.get("kernel", "SquaredExponentialKernel")]
        model = cls(state["num_levels"], kernel_class,
                    estimate_rho=state.get("estimate_rho", False),
                    rho_init=state.get("rho_init", 1.0),
                    rho_bounds=tuple(state.get("rho_bounds", (-5.0, 5.0))),
                    min_points_rho=state.get("min_points_rho"))
        model.rhos = [float(r) for r in state.get("rhos", [1.0] * (model.num_levels - 1))]

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
