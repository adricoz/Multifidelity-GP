"""
[CK] Co-kriging surrogate conditioned on ALL the data (Le Gratiet thesis [LG], Eqs. 3.8-3.11),
for NESTED designs only (X_L included in ... included in X_1). Surrogate only: no infilling.

Auto-regressive model (Kennedy & O'Hagan, [LG] Eq. 3.1, Sacher Eq. 9):
    Z_1 = delta_1,   Z_l = rho_(l-1) Z_(l-1) + delta_l,   delta_1, ..., delta_L independent GPs.

1. Hyperparameters, level by level. With nested designs the likelihood of all the data
   factorizes ([LG] p. 91 and Prop. 3.1): level 1 is a GP fitted on z_1, level l a GP fitted on
   the residual z_l - rho_(l-1) z_(l-1)(X_l), where z_(l-1)(X_l) are the OBSERVED values of the
   level below at the points of level l (GaussianProcess.fit with f_prev; rho learned under a
   lognormal prior by default, or profiled in closed form with a normal prior or none).
2. Prediction: Gaussian conditioning of Z_L(x) on the vector z of ALL the observations, with the
   joint covariance matrix V of size n_1 + ... + n_L ([LG] Eq. 3.11 for two levels):
       mean(x) = m_L + k(x)^T V^-1 (z - m),   var(x) = c_LL(x, x) - k(x)^T V^-1 k(x).
   V is factorized once (Cholesky): fine for a few hundred points.

For GIVEN hyperparameters, step 2 is the exact posterior of the AR(1) model (any noise, any
design). Step 1 assumes noise-free (or nearly noise-free) lower levels: with noisy low-fidelity
data, the residual of level l also contains rho^2 times the noise of the level below, so the noise
of level l is overestimated and rho is biased towards 0 (noisy regressor).
With nested noise-free data the predictions equal the recursive formulation of
MultifidelityModel ([LG] Prop. 3.1).
"""
import numpy as np
from scipy.linalg import cho_solve, solve_triangular
from src.data_management import ExperimentData
from src.kernels import Matern52Kernel
from src.surrogate_models import (KERNELS, V3_RHO_BOUNDS, V3_RHO_PRIOR, GaussianProcess,
                                  check_priors, lengthscale_priors_per_level, safe_cholesky,
                                  saved_priors, scale_to_unit)


def nested_index(x_high: np.ndarray, x_low: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    """
    [CK] For every point of x_high, index of the same point in x_low (-1 if it is missing).

    Args:
    - x_high: numpy array (n_high, d), points of a level.
    - x_low: numpy array (n_low, d), points of the level below.
    - tol: distance below which two points are the same.
    Returns:
    - numpy integer array (n_high,).
    """
    if len(x_low) == 0:
        return np.full(len(x_high), -1)
    distances = np.linalg.norm(x_high[:, None, :] - x_low[None, :, :], axis=2)
    index = np.argmin(distances, axis=1)
    index[distances[np.arange(len(x_high)), index] > tol] = -1
    return index


def missing_points_for_nesting(experiment_data: ExperimentData, num_levels: int,
                               tol: float = 1e-8) -> dict:
    """
    [CK] Points to evaluate at the lower levels to make the design nested (e.g. data produced
    by the infilling of EGOOptimizer, whose levels are not nested). Failed (NaN) evaluations
    count as missing.

    Returns:
    - {level: numpy array (n_missing, d)} for the levels 1..num_levels-1 (empty arrays if the
      design is already nested).
    """
    points = {l: experiment_data.get_training_data(l)[0] for l in range(1, num_levels + 1)}
    missing = {}
    for l in range(num_levels, 1, -1):
        absent = points[l][nested_index(points[l], points[l - 1], tol) < 0]
        missing[l - 1] = absent
        # the points added at level l-1 must also exist at the level below
        points[l - 1] = np.vstack([points[l - 1], absent])
    return dict(sorted(missing.items()))


class CoKriging:
    """
    [CK] Co-kriging surrogate of L nested levels conditioned on all the data (see the module
    docstring). Same options as MultifidelityModel for the hyperparameters (use_map, priors,
    lengthscale_prior), but the mfego v3 defaults (benchmarks/surrogate_v3/RAPPORT_V3.md): Matern
    5/2 kernel, MAP, centred rho learned with the kernel hyperparameters under the lognormal prior
    V3_RHO_PRIOR, bounds V3_RHO_BOUNDS. mfego v2 (closed-form rho, N(1, 2^2) prior, bounds +-5):
    priors={"rho": ("normal", 1.0, 2.0)}.

    Example:
        data.generate_nested_design([20, 5])          # X_2 included in X_1
        ...                                           # evaluate the two levels
        model = CoKriging(2, use_map=True)
        model.fit(data)
        mean, var = model.predict_batch(x)
    """
    def __init__(self, num_levels: int, kernel_class=Matern52Kernel,
                 use_map: bool = True, priors: dict = None, lengthscale_prior=None,
                 rho_bounds: tuple[float, float] = None, n_restarts: int = 3,
                 seed: int = None, tol: float = 1e-8, rho_centering: bool = True,
                 share_lengthscales: bool = False):
        self.num_levels = num_levels
        self.kernel_class = kernel_class
        self.use_map = bool(use_map)
        self.rho_centering = bool(rho_centering)
        # [SHARE] True: the GP of the correction delta_l takes the lengthscales of the GP of
        # level l-1 (learned on more points) instead of learning its own on the points of level
        # l only; its variances, noise and rho are still estimated. -7 % of error on the target
        # region of the airfoil test, 8 seeds out of 10 (vpp_acceleration/examples/joint_study.py).
        self.share_lengthscales = bool(share_lengthscales)
        self.priors = check_priors(priors)
        # [V3] default prior of the centred rho with MAP: rho learned under V3_RHO_PRIOR
        if self.rho_centering and self.use_map and "rho" not in self.priors:
            self.priors["rho"] = V3_RHO_PRIOR
        # [V3] bounds of rho (None): wide for a learned rho (MAP with a lognormal prior, which
        # limits rho), +-5 for a profiled one (normal prior, no prior or maximum likelihood)
        if rho_bounds is None:
            rho_prior = self.priors.get("rho")
            learned = self.use_map and rho_prior is not None and rho_prior[0] == "lognormal"
            rho_bounds = V3_RHO_BOUNDS if learned else (-5.0, 5.0)
        self.rho_bounds = tuple(rho_bounds)
        # InvGamma(alpha, beta) lengthscale prior: None, one pair, or one pair per level
        self.lengthscale_priors = lengthscale_priors_per_level(lengthscale_prior, num_levels)
        self.n_restarts = n_restarts
        self.tol = tol
        # one GP per level: delta_1, ..., delta_L (hyperparameters and normalization)
        self.gps = [GaussianProcess(kernel_class(), seed=None if seed is None else seed + i,
                                    use_map=use_map, lengthscale_prior=self.lengthscale_priors[i],
                                    priors=self.priors, rho_centering=rho_centering)
                    for i in range(num_levels)]
        # rhos[l-2] links level l-1 to level l (same convention as MultifidelityModel)
        self.rhos = [1.0] * (num_levels - 1)
        # [NORM] bounds of the design space (set by fit): the GPs work in [0, 1]^d
        self.input_bounds = None
        # data of the joint system: all the points (raw units), their level, the observations
        self.x_all = None
        self.levels_all = None
        self.z_all = None
        self.l_chol = None
        self.alpha = None
        self.jitter = 0.0

    # 1/3 -----------------------------------------------------------------------------------------
    # Fit: hyperparameters level by level, then conditioning on all the data
    # ---------------------------------------------------------------------------------------------
    def fit(self, experiment_data: ExperimentData) -> None:
        """
        Fits the GP of each level (nested design: residual with the OBSERVED values of the level
        below), then conditions the co-kriging on all the observations.

        Raises:
        - ValueError if the design is not nested (a point of level l is missing, or failed, at
          level l-1).
        """
        xs, ys = [], []
        for l in range(1, self.num_levels + 1):
            x_l, y_l = experiment_data.get_training_data(l)
            if len(y_l) == 0:
                raise ValueError(f"No valid observation available at fidelity level {l}.")
            xs.append(x_l)
            ys.append(y_l)
        # [NORM] the GPs see the inputs scaled to [0, 1] with the bounds of the design space
        self.input_bounds = np.asarray(experiment_data.bounds, dtype=float)

        self.gps[0].fit(self._unit(xs[0]), ys[0], n_restarts=self.n_restarts)
        for l in range(2, self.num_levels + 1):
            index = nested_index(xs[l - 1], xs[l - 2], self.tol)
            if np.any(index < 0):
                missing = xs[l - 1][index < 0]
                raise ValueError(
                    f"CoKriging needs nested designs: {len(missing)} point(s) of level {l} are not "
                    f"(validly) evaluated at level {l - 1}, e.g. {missing[0]}. Use "
                    "ExperimentData.generate_nested_design, or evaluate the points given by "
                    "missing_points_for_nesting() at the lower levels.")
            # observed values of the level below at the points of level l ([LG] Prop. 3.1)
            f_prev = ys[l - 2][index]
            # [SHARE] lengthscales of the level below (already fitted), or None (learned here)
            shared = self.gps[l - 2].kernel.lengthscale if self.share_lengthscales else None
            self.rhos[l - 2] = self.gps[l - 1].fit(
                self._unit(xs[l - 1]), ys[l - 1], n_restarts=self.n_restarts, f_prev=f_prev,
                estimate_rho=True, rho_bounds=self.rho_bounds, fixed_lengthscale=shared)
        self.condition(xs, ys)

    def _unit(self, x: np.ndarray) -> np.ndarray:
        """[NORM] Raw inputs -> inputs of the GPs (scaled to [0, 1]^d with input_bounds)."""
        return scale_to_unit(np.atleast_2d(x), self.input_bounds)

    def condition(self, xs: list, ys: list) -> None:
        """
        Builds the joint covariance matrix V of all the observations with the CURRENT
        hyperparameters and rhos, factorizes it and computes alpha = V^-1 (z - m).
        Used by fit() and by from_dict() (no optimization).

        Args:
        - xs: list of L numpy arrays (n_l, d), points of each level (raw units).
        - ys: list of L numpy arrays (n_l,), raw observations of each level.
        """
        self.x_all = np.vstack([np.atleast_2d(x) for x in xs])
        self.levels_all = np.concatenate([np.full(len(x), l) for l, x in enumerate(xs, start=1)])
        self.z_all = np.concatenate([np.asarray(y, dtype=float).reshape(-1) for y in ys])

        noise = np.array([self.gps[l - 1].get_noise_variance() for l in self.levels_all])
        u_all = self._unit(self.x_all)
        cov_matrix = self._covariance(u_all, self.levels_all, u_all, self.levels_all)
        self.l_chol, self.jitter = safe_cholesky(cov_matrix + np.diag(noise))
        self.alpha = cho_solve((self.l_chol, True), self.z_all - self._mean(self.levels_all))

    # 2/3 -----------------------------------------------------------------------------------------
    # Covariance structure of the auto-regressive model
    # ---------------------------------------------------------------------------------------------
    def _factors(self, levels: np.ndarray) -> np.ndarray:
        """
        Weight of each delta_m in Z_l: Z_l = sum_m P(m, l) delta_m with
        P(m, l) = rho_m * ... * rho_(l-1) (1 if m = l, 0 if m > l).

        Returns:
        - numpy array (n, L), factors[i, m-1] = P(m, levels[i]).
        """
        factors = np.zeros((len(levels), self.num_levels))
        for level in np.unique(levels):
            for m in range(1, level + 1):
                factors[levels == level, m - 1] = np.prod(self.rhos[m - 1:level - 1])
        return factors

    def _covariance(self, x_a: np.ndarray, levels_a: np.ndarray, x_b: np.ndarray,
                    levels_b: np.ndarray) -> np.ndarray:
        """
        Cov(Z_i(x), Z_j(x')) = sum_(m <= min(i, j)) P(m, i) P(m, j) c_m(x, x'), with c_m the
        covariance of delta_m in raw units (y_std_m^2 times the kernel of level m).
        For two levels this is the block matrix of [LG] Eq. 3.11.

        Returns:
        - numpy array (n_a, n_b).
        """
        p_a, p_b = self._factors(levels_a), self._factors(levels_b)
        cov = np.zeros((len(x_a), len(x_b)))
        for m, gp in enumerate(self.gps):
            c_m = gp.y_std ** 2 * gp.kernel.get_cross_covariance_matrix(x_a, x_b)
            cov += np.outer(p_a[:, m], p_b[:, m]) * c_m
        return cov

    def _mean(self, levels: np.ndarray) -> np.ndarray:
        """Prior mean of Z_l: sum_m P(m, l) y_mean_m (constant mean of each delta_m)."""
        return self._factors(levels) @ np.array([gp.y_mean for gp in self.gps])

    # 3/3 -----------------------------------------------------------------------------------------
    # Prediction, diagnostics, JSON
    # ---------------------------------------------------------------------------------------------
    def predict_batch(self, x_new: np.ndarray, level: int = None,
                      include_noise: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """
        Mean and variance of Z_level(x) conditioned on all the data ([LG] Eqs. 3.8-3.10).

        Args:
        - x_new: numpy array (m, d) (or (d,) for a single point).
        - level: fidelity level of the prediction (default: the highest level).
        - include_noise: True adds the observation noise of the level (same convention as
          GaussianProcess.predict_batch), False gives the variance of the latent function.
        Returns:
        - mean (m,), variance (m,), raw units.
        """
        level = self.num_levels if level is None else level
        x_new = self._unit(x_new)
        levels_new = np.full(len(x_new), level)

        k_mat = self._covariance(x_new, levels_new, self._unit(self.x_all), self.levels_all)
        mean = self._mean(levels_new) + k_mat @ self.alpha

        # prior variance c_ll(x, x) = sum_m P(m, l)^2 y_std_m^2 (t1_m + t2_m)
        p_new = self._factors(levels_new[:1])[0]
        prior_var = sum(p_new[m] ** 2 * gp.y_std ** 2
                        * (gp.kernel.signal_variance + gp.kernel.bias_variance)
                        for m, gp in enumerate(self.gps))
        v_mat = solve_triangular(self.l_chol, k_mat.T, lower=True)
        var = np.maximum(prior_var - np.sum(v_mat ** 2, axis=0), 0.0)
        if include_noise:
            var = var + self.gps[level - 1].get_noise_variance()
        return mean, var

    def leave_one_out(self, level: int = None, by_location: bool = True
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        [QUAL] Closed-form leave-one-out (Dubrule 1983; Rasmussen & Williams GPML Eq. 5.12):
        prediction of each observation of `level` (default: the highest) from the OTHER
        observations, with the fitted hyperparameters (no refit).
        - by_location=True (default): ALL the observations at the same point (every level) are
          left out together. On a nested design the lower-level value at the point would
          otherwise stay known, which makes the plain leave-one-out optimistic.
        - by_location=False: only the observation itself is left out.
        With C the matrix of condition() (noise included), alpha = C^-1 (z - m) and S the
        indices left out together (block formula; S = {i} gives the classical formula):
            z_S | rest ~ N(z_S - [(C^-1)_SS]^-1 alpha_S, [(C^-1)_SS]^-1).
        Use validation.prediction_scores(y, mean, var) for Q^2, coverage and NLPD.

        Returns:
        - y (observed), mean, var (noise included) at the observations of the level.
        """
        level = self.num_levels if level is None else level
        c_inv = cho_solve((self.l_chol, True), np.eye(len(self.z_all)))
        rows = np.flatnonzero(self.levels_all == level)
        means, variances = np.zeros(len(rows)), np.zeros(len(rows))
        for j, i in enumerate(rows):
            block = [i]
            if by_location:
                block = list(np.flatnonzero(
                    np.linalg.norm(self.x_all - self.x_all[i], axis=1) <= self.tol))
            inv_block = np.linalg.inv(c_inv[np.ix_(block, block)])
            k = block.index(i)
            means[j] = self.z_all[i] - (inv_block @ self.alpha[block])[k]
            variances[j] = inv_block[k, k]
        return self.z_all[rows], means, variances

    @classmethod
    def from_multifidelity(cls, model, experiment_data: ExperimentData) -> "CoKriging":
        """
        [QUAL] Exact conditioning, on ALL the data of `experiment_data` (any design, nested or
        not), of the AR(1) model of a fitted MultifidelityModel: same options (priors, rho
        centring), hyperparameters, normalization and rhos, no fit. E.g. to qualify with
        leave_one_out() the surrogate of an EGO run, whose levels are not nested.
        """
        cokriging = cls(model.num_levels, model.kernel_class, use_map=model.use_map,
                        priors={**model.priors, "rho": model.gps[0].priors["rho"]},
                        lengthscale_prior=model.lengthscale_priors,
                        rho_bounds=model.rho_bounds, rho_centering=model.rho_centering)
        cokriging.rhos = list(model.rhos)
        cokriging.input_bounds = model.input_bounds
        for gp_ck, gp in zip(cokriging.gps, model.gps):
            gp_ck.kernel.set_params(gp.kernel.get_params())
            gp_ck.noise, gp_ck.y_mean, gp_ck.y_std = gp.noise, gp.y_mean, gp.y_std
            gp_ck.rho = gp.rho
        levels = range(1, model.num_levels + 1)
        cokriging.condition([experiment_data.get_training_data(l)[0] for l in levels],
                            [experiment_data.get_training_data(l)[1] for l in levels])
        return cokriging

    def is_fitted(self) -> bool:
        return self.alpha is not None

    def diagnostics(self) -> dict:
        """Hyperparameters of each level (GaussianProcess.diagnostics), rhos, size of V."""
        return {"levels": [{"level": l + 1, **gp.diagnostics()} for l, gp in enumerate(self.gps)],
                "rhos": [float(r) for r in self.rhos],
                "n_total": 0 if self.x_all is None else int(len(self.x_all)),
                "jitter": float(self.jitter)}

    def to_dict(self) -> dict:
        """JSON-friendly state (data, hyperparameters, normalization, rhos): from_dict()
        rebuilds the same surrogate without fitting."""
        levels = []
        for l, gp in enumerate(self.gps, start=1):
            mask = self.levels_all == l
            levels.append({**gp.to_dict(), "x_train": self.x_all[mask].tolist(),
                           "y_train": self.z_all[mask].tolist()})
        return {"model": "CoKriging", "kernel": self.kernel_class.__name__,
                "num_levels": self.num_levels, "rhos": [float(r) for r in self.rhos],
                "use_map": self.use_map,
                "priors": saved_priors(self.gps[0].priors, self.priors),
                "lengthscale_prior": [list(p) for p in self.lengthscale_priors],
                "rho_centering": self.rho_centering, "rho_bounds": list(self.rho_bounds),
                "share_lengthscales": self.share_lengthscales,
                "input_bounds": None if self.input_bounds is None else self.input_bounds.tolist(),
                "levels": levels}

    @classmethod
    def from_dict(cls, state: dict) -> "CoKriging":
        """Rebuilds a fitted CoKriging from to_dict() output (same predictions, no fit).
        Files written before rho_centering / input_bounds (mfego v1) keep their behaviour."""
        priors = dict(state.get("priors") or {})
        if "rho_centering" not in state:
            priors["rho"] = None
        model = cls(state["num_levels"], KERNELS[state.get("kernel", "SquaredExponentialKernel")],
                    use_map=state.get("use_map", True), priors=priors,
                    lengthscale_prior=state.get("lengthscale_prior"),
                    rho_bounds=state.get("rho_bounds"),
                    rho_centering=state.get("rho_centering", False),
                    share_lengthscales=state.get("share_lengthscales", False))
        model.rhos = [float(r) for r in state["rhos"]]
        bounds = state.get("input_bounds")
        model.input_bounds = None if bounds is None else np.asarray(bounds, dtype=float)
        xs, ys = [], []
        for gp, level_state in zip(model.gps, state["levels"]):
            gp.kernel.set_params(np.array(level_state["kernel_params"], dtype=float))
            gp.noise = float(level_state["noise"])
            gp.y_mean = float(level_state["y_mean"])
            gp.y_std = float(level_state["y_std"])
            gp.rho = level_state.get("rho")
            xs.append(np.atleast_2d(np.array(level_state["x_train"], dtype=float)))
            ys.append(np.array(level_state["y_train"], dtype=float))
        model.condition(xs, ys)
        return model
