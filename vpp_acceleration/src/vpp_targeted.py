"""
VPP-targeted sequential co-kriging of the foil torsor: refine the surrogate WHERE the VPP operates,
and on WHAT matters for the boat speed.

Inputs of the surrogate (16): the 12 PCA scores of the Kulfan section, Cant_Eff, Rake_Eff, Yaw_Eff
and Hvol (meta_foil.INPUT_NAMES). Outputs: the 6 force / moment coefficients of the foil (one
CoKriging of mfego per coefficient, rho bounded "v2", as S1 of the region test).

The criterion. At an equilibrium state s of the VPP, an error dC_k on the coefficient k changes the
speed by about (dV/dC_k) dC_k. With independent models per coefficient, the variance of the
predicted speed is, to first order,
    Var V(s) = sum_k (dV/dC_k)^2 Var C_k(s).
Adding one observation at (x, level l) lowers it, averaged over the integration points x_int, by
    G(x, l) = sum_k w_k G_k(x, l),     w_k = mean over the states of (dV/dC_k)^2,
where G_k is the IMSE gain of the co-kriging of coefficient k (sequential_design.
batched_imse_gains, exact with the joint covariance) and the integration points are clouds around
the VPP states. The pair (x, l) with the best G / cumulative cost is run (same rule as the ALC
choice of sequential_design.find_next_point_alc: the design stays nested, adding level 2 where
level 1 is missing costs both). Units: kt^2 per unit of cost.

One iteration:
  1. refit the hyperparameters of the 6 co-krigings every `refit_every` iterations, otherwise only
     condition them on the new data (fast);
  2. every `vpp_every` iterations: put the co-kriging means into the MDS VPP (vpp_coupling),
     solve the equilibrium of every (design section, navigation condition): the states; the
     integration points are a Sobol cloud around each state (attitude +- `spread`, section fixed);
     every `weights_every` state updates, the speed sensitivities dV/dC_k by finite differences
     at the reference section;
  3. candidates: points around the states, points anywhere in the domain (sections of the
     family, attitude in its box), and the LF points without HF (HF for the cost of the HF only);
  4. choose (x, l) by the criterion, run the meta-surrogate (or the codes) at the missing levels.
Strategies compared in the studies (same code, `strategy`): "vpp" (above), "global" (integration
over the whole domain, outputs weighted by the inverse of their variance: the classic ALC).

The data of all coefficients are the same points: one run of a code gives the whole torsor.
"""
import copy
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.stats import qmc

from src.cokriging import CoKriging
from src.data_management import ExperimentData, greedy_maximin
from src.meta_foil import ATTITUDE_BOUNDS, N_SCORES
from src.sequential_design import append_observations, batched_imse_gains

N_FORCES = 6                                     # Fx, Fy, Fz, Mx, My, Mz (Cpmin is not modelled)
RHO_PRIOR = {"rho": ("normal", 1.0, 2.0)}        # "v2": rho bounded +-5, N(1, 2^2) prior
# half-width of the cloud of integration points around a state: Cant, Rake, Yaw [deg], Hvol [m]
SPREAD = np.array([1.0, 0.3, 1.0, 0.05])


# 1/4 ---------------------------------------------------------------------------------------------
# The surrogate of the torsor: one co-kriging per coefficient, on the same points
# -------------------------------------------------------------------------------------------------
class TorsorCoKriging:
    """
    Six (or seven) CoKriging (mfego) sharing one nested design: x_lists[l] (n_l, d), y_lists[l]
    (n_l, 7). n_models = 6: the force / moment coefficients (Cpmin stored, not modelled, predicted
    as 0: the meta studies); n_models = 7: Cpmin modelled too (the real run of 2026-10-22).
    """

    def __init__(self, bounds, costs=(1.0, 10.0), share_lengthscales=False, seed=0,
                 n_models: int = N_FORCES, n_threads: int = 1):
        """n_threads: the models are fitted, conditioned and evaluated in that many threads
        (numpy releases the GIL in its linear algebra); same results as 1 thread."""
        self.bounds = list(bounds)
        self.costs = np.asarray(costs, dtype=float)
        self.share_lengthscales = share_lengthscales
        self.seed = seed
        self.n_models = int(n_models)
        self.n_threads = int(n_threads)
        self.x_lists = [np.zeros((0, len(bounds))), np.zeros((0, len(bounds)))]
        self.y_lists = [np.zeros((0, 7)), np.zeros((0, 7))]
        self.models = None

    def map_models(self, function, items) -> list:
        """[function(item) for item in items], in n_threads threads."""
        if self.n_threads <= 1:
            return [function(item) for item in items]
        with ThreadPoolExecutor(max_workers=min(self.n_threads, len(items))) as pool:
            return list(pool.map(function, items))

    def add(self, level: int, x: np.ndarray, y: np.ndarray) -> None:
        """Adds observations (m, 16) -> (m, 7) at a level (1 or 2)."""
        self.x_lists[level - 1] = np.vstack([self.x_lists[level - 1], np.atleast_2d(x)])
        self.y_lists[level - 1] = np.vstack([self.y_lists[level - 1], np.atleast_2d(y)])

    def believe(self, x: np.ndarray, levels) -> None:
        """Kriging believer (vpp_optimize): conditions every model on observations at x (m, d),
        one per level of `levels`, equal to their predicted means, by extending the Cholesky
        factor (sequential_design.append_observations): the same posterior as add() of the
        predicted means + fit(refit=False), without rebuilding the covariance matrix. The data
        lists x_lists / y_lists are unchanged: the next fit forgets these observations."""
        x = np.atleast_2d(x)
        levels = np.asarray(levels, dtype=int)
        points = np.repeat(x, len(levels), axis=0)
        point_levels = np.tile(levels, len(x))
        self.map_models(lambda model: append_observations(model, points, point_levels),
                        self.models)

    def truncate(self, sizes) -> None:
        """Keeps the first sizes[l] observations of each level (removes the placeholders of a
        batch choice, vpp_optimize). The models stay as they are until the next fit."""
        self.x_lists = [x[:n] for x, n in zip(self.x_lists, sizes)]
        self.y_lists = [y[:n] for y, n in zip(self.y_lists, sizes)]

    def _data(self, k: int) -> ExperimentData:
        data = ExperimentData(bounds=self.bounds, costs=list(self.costs))
        data.x_dict = {1: self.x_lists[0], 2: self.x_lists[1]}
        data.y_dict = {1: self.y_lists[0][:, k], 2: self.y_lists[1][:, k]}
        data.metrics_dict = {1: [{}] * len(self.x_lists[0]), 2: [{}] * len(self.x_lists[1])}
        return data

    def fit(self, refit: bool = True) -> None:
        """refit: new hyperparameters (MAP); otherwise the current ones, conditioned on the data."""
        if refit or self.models is None:
            def fit_one(k):
                model = CoKriging(2, priors=RHO_PRIOR, seed=self.seed + k,
                                  share_lengthscales=self.share_lengthscales)
                model.fit(self._data(k))
                return model
            self.models = self.map_models(fit_one, range(self.n_models))
        else:
            def condition_one(k):
                self.models[k].condition(list(self.x_lists), [y[:, k] for y in self.y_lists])
            self.map_models(condition_one, range(self.n_models))

    def predict(self, x: np.ndarray, level: int = None) -> np.ndarray:
        """Means (m, 7) of a level (default the top one): the 6 coefficients, and Cpmin (0 when
        it is not modelled)."""
        x = np.atleast_2d(x)
        means = np.zeros((len(x), 7))
        for k, model in enumerate(self.models):
            means[:, k] = latent_mean(model, x, level)
        return means

    def mean_only_copy(self) -> "TorsorCoKriging":
        """A copy that can only predict the means (no Cholesky factor: ~100 times lighter to
        pickle for the VPP worker processes, vpp_optimize.VPPPool)."""
        light = copy.copy(self)
        light.models = []
        for model in self.models:
            model = copy.copy(model)
            model.l_chol = None
            light.models.append(model)
        return light

    def variances(self, x: np.ndarray) -> np.ndarray:
        """Latent variances (m, n_models) of the modelled outputs."""
        x = np.atleast_2d(x)
        return np.column_stack(self.map_models(
            lambda model: model.predict_batch(x, include_noise=False)[1], self.models))

    @property
    def cost(self) -> float:
        return float(len(self.x_lists[0]) * self.costs[0] + len(self.x_lists[1]) * self.costs[1])

    @property
    def sizes(self) -> tuple:
        return len(self.x_lists[0]), len(self.x_lists[1])


def latent_mean(model: CoKriging, x: np.ndarray, level: int = None) -> np.ndarray:
    """
    Mean of a level (default the top one) of a CoKriging at x (m, d): the mean of
    model.predict_batch, without its variance. The VPP evaluates the foil hundreds of times per
    solve and only reads the mean; the variance costs O(n^2) per point (a triangular solve with
    the n observations), the mean O(n).
    """
    levels = np.full(len(x), model.num_levels if level is None else level)
    # pylint: disable=protected-access
    k_mat = model._covariance(model._unit(x), levels, model._unit(model.x_all), model.levels_all)
    return model._mean(levels) + k_mat @ model.alpha


def nested_design(meta, n_low: int, n_high: int, seed: int) -> TorsorCoKriging:
    """
    Nested initial design, evaluated with the meta-surrogate: n_low points (sections drawn in the
    family, attitude by a Latin hypercube in its box), then n_high of them chosen by greedy
    maximin (in the unit cube of the bounds) for the high fidelity.
    """
    bounds = meta.bounds()
    scores = meta.sample_sections(n_low, seed)
    lo = np.array([b[0] for b in ATTITUDE_BOUNDS])
    hi = np.array([b[1] for b in ATTITUDE_BOUNDS])
    attitude = lo + qmc.LatinHypercube(d=4, seed=seed).random(n_low) * (hi - lo)
    x_low = np.column_stack([scores, attitude])
    x_high = greedy_maximin(x_low, n_high, bounds) if n_high < n_low else x_low.copy()
    surrogate = TorsorCoKriging(bounds, seed=seed)
    surrogate.add(1, x_low, meta(x_low, 1))
    surrogate.add(2, x_high, meta(x_high, 2))
    return surrogate


# 2/4 ---------------------------------------------------------------------------------------------
# The VPP: states, integration points, speed sensitivities
# -------------------------------------------------------------------------------------------------
def vpp_states(vpp, predict, sections, conditions, guesses=None) -> list:
    """Equilibrium of every (section, condition) with the foil model `predict`: a list of state
    dicts (vpp_coupling.IntensSYVPP.solve) with "section" (index), "point" (the 16 inputs)."""
    states = []
    for i, scores in enumerate(sections):
        vpp.set_foil(predict, scores)
        for j, (twa, tws) in enumerate(conditions):
            guess = None if guesses is None else guesses[i * len(conditions) + j]
            state = vpp.solve(twa, tws, guess=guess)
            state["section"] = i
            state["point"] = np.concatenate([scores, [state["Cant_Eff"], state["Rake_Eff"],
                                                      state["Yaw_Eff"], state["Hvol"]]])
            states.append(state)
    return states


def integration_cloud(points: np.ndarray, n_per_state: int, seed: int,
                      spread=SPREAD) -> np.ndarray:
    """
    Integration points around each state: section fixed, attitude = state + spread x (Sobol point
    of [-1, 1]^4), clipped to the attitude box. n_per_state should be a power of 2 (Sobol).

    Returns:
    - numpy array (n_states n_per_state, 16).
    """
    sobol = 2.0 * qmc.Sobol(d=4, scramble=True, seed=seed).random(n_per_state) - 1.0
    lo = np.array([b[0] for b in ATTITUDE_BOUNDS])
    hi = np.array([b[1] for b in ATTITUDE_BOUNDS])
    cloud = []
    for point in np.atleast_2d(points):
        attitude = np.clip(point[N_SCORES:] + sobol * spread, lo, hi)
        cloud.append(np.column_stack([np.repeat(point[None, :N_SCORES], n_per_state, axis=0),
                                      attitude]))
    return np.vstack(cloud)


def global_points(meta, n_points: int, seed: int) -> np.ndarray:
    """Points of the whole domain: sections of the family, attitude uniform (Sobol) in its box."""
    lo = np.array([b[0] for b in ATTITUDE_BOUNDS])
    hi = np.array([b[1] for b in ATTITUDE_BOUNDS])
    attitude = lo + qmc.Sobol(d=4, scramble=True, seed=seed).random(n_points) * (hi - lo)
    return np.column_stack([meta.sample_sections(n_points, seed), attitude])


def speed_weights(vpp, predict, scores, conditions, states) -> np.ndarray:
    """w_k = mean over the conditions of (dV/dC_k)^2 [kt^2 per coefficient^2], finite differences
    at the section `scores` (vpp_coupling.IntensSYVPP.speed_sensitivities)."""
    derivatives = []
    for (twa, tws), state in zip(conditions, states):
        derivatives.append(vpp.speed_sensitivities(predict, scores, twa, tws, state))
    return np.nanmean(np.array(derivatives) ** 2, axis=0)


# 3/4 ---------------------------------------------------------------------------------------------
# The choice of the point and of the level
# -------------------------------------------------------------------------------------------------
def candidates(surrogate: TorsorCoKriging, meta, state_points, n_local: int, n_global: int,
               seed: int) -> tuple:
    """
    Candidates and their missing levels (M, 2): Sobol clouds around the states (n_local in all),
    points of the whole domain (n_global), and the LF points without HF.
    """
    parts = []
    if state_points is not None and len(state_points) and n_local:
        per_state = max(1, n_local // len(state_points))
        parts.append(integration_cloud(state_points, per_state, seed + 1))
    if n_global:
        parts.append(global_points(meta, n_global, seed + 2))
    x_new = np.vstack(parts)
    missing_new = np.ones((len(x_new), 2), dtype=bool)
    x_low, x_high = surrogate.x_lists
    has_high = np.array([np.any(np.all(np.isclose(x, x_high), axis=1)) for x in x_low])
    x_lf_only = x_low[~has_high]
    missing_lf = np.column_stack([np.zeros(len(x_lf_only), bool), np.ones(len(x_lf_only), bool)])
    return np.vstack([x_new, x_lf_only]), np.vstack([missing_new, missing_lf])


def choose(surrogate: TorsorCoKriging, weights: np.ndarray, x_cand: np.ndarray,
           missing: np.ndarray, x_int: np.ndarray, remaining_budget: float) -> dict:
    """
    The pair (candidate, level) with the largest yield sum_k w_k G_k(x, l) / C(x, l), C = cost
    of the missing levels <= l (nested design). Levels that do not fit in the budget are excluded.

    weights: one weight per output (K,), the gain G_k being the plain mean over the integration
    points; or one weight per integration point and output (N, K), the gain being the weighted
    sum over the points (vpp_optimize: probability of the condition of the point x (dV/dC_k)^2).
    The first form is the second with weights w_k / N.

    Returns:
    - dict: index, level (None if nothing fits), x, ratio, gains (M, 2) of the weighted sum.
    """
    weights = np.asarray(weights, dtype=float)
    if weights.ndim == 1:
        weights = np.tile(weights / len(x_int), (len(x_int), 1))           # (N, K)
    used = [k for k in range(len(surrogate.models))
            if k < weights.shape[1] and weights[:, k].sum() > 0.0]
    gains = np.zeros((len(x_cand), 2))
    for gain in surrogate.map_models(
            lambda k: batched_imse_gains(surrogate.models[k], x_cand, x_int, missing,
                                         int_weights=weights[:, k]), used):
        gains += gain
    cumulative = np.cumsum(missing * surrogate.costs, axis=1)            # (M, 2)
    ratios = np.where(missing & (cumulative <= remaining_budget + 1e-12),
                      gains / np.maximum(cumulative, 1e-12), -np.inf)
    if not np.isfinite(ratios).any():
        return {"index": None, "level": None, "x": None, "ratio": None, "gains": gains}
    index, level = np.unravel_index(np.argmax(ratios), ratios.shape)
    return {"index": int(index), "level": int(level) + 1, "x": x_cand[index].copy(),
            "ratio": float(ratios[index, level]), "gains": gains,
            "missing": missing[index].copy(), "cost": float(cumulative[index, level])}


# 4/4 ---------------------------------------------------------------------------------------------
# The loop
# -------------------------------------------------------------------------------------------------
class VPPTargetedDesign:
    """
    The sequential design (module docstring). Usage:
        design = VPPTargetedDesign(surrogate, meta, vpp, sections, conditions, strategy="vpp")
        history = design.run(budget=200.0)
    """

    def __init__(self, surrogate: TorsorCoKriging, meta, vpp, sections, conditions,
                 strategy: str = "vpp", n_per_state: int = 64, n_global_int: int = 1024,
                 n_local: int = 768, n_global: int = 256, refit_every: int = 5,
                 vpp_every: int = 5, weights_every: int = 2, seed: int = 0, verbose=False):
        if strategy not in ("vpp", "global"):
            raise ValueError(f"strategy must be 'vpp' or 'global', got {strategy!r}")
        self.surrogate, self.meta, self.vpp = surrogate, meta, vpp
        self.sections, self.conditions = np.atleast_2d(sections), list(conditions)
        self.strategy = strategy
        self.n_per_state, self.n_global_int = n_per_state, n_global_int
        self.n_local, self.n_global = n_local, n_global
        self.refit_every, self.vpp_every, self.weights_every = refit_every, vpp_every, weights_every
        self.seed, self.verbose = seed, verbose
        self.cost_spent, self.iteration = 0.0, 0
        self.states, self.weights, self.x_int = None, None, None
        self.history = []

    def update_states(self) -> None:
        """States of the VPP with the current surrogate; integration points; weights."""
        guesses = None if self.states is None else [s["unknowns"] for s in self.states]
        self.states = vpp_states(self.vpp, self.surrogate.predict, self.sections,
                                 self.conditions, guesses)
        points = np.array([s["point"] for s in self.states])
        n_updates = self.iteration // self.vpp_every
        if self.strategy == "vpp":
            self.x_int = integration_cloud(points, self.n_per_state, self.seed + self.iteration)
            if self.weights is None or n_updates % self.weights_every == 0:
                reference = [s for s in self.states if s["section"] == 0]
                self.weights = speed_weights(self.vpp, self.surrogate.predict, self.sections[0],
                                             self.conditions, reference)
        else:
            self.x_int = global_points(self.meta, self.n_global_int, self.seed + self.iteration)
            # classic multi-output ALC: every coefficient counts by its variance (scale-free)
            self.weights = 1.0 / np.var(self.surrogate.y_lists[1][:, :self.surrogate.n_models],
                                        axis=0)

    def step(self, remaining_budget: float) -> dict:
        start = time.perf_counter()
        self.surrogate.fit(refit=(self.iteration % self.refit_every == 0))
        if self.iteration % self.vpp_every == 0 or self.states is None:
            self.update_states()
        points = np.array([s["point"] for s in self.states]) if self.strategy == "vpp" else None
        x_cand, missing = candidates(self.surrogate, self.meta, points,
                                     self.n_local if self.strategy == "vpp" else 0,
                                     self.n_global if self.strategy == "vpp"
                                     else self.n_local + self.n_global,
                                     self.seed + 7919 * (self.iteration + 1))
        choice = choose(self.surrogate, self.weights, x_cand, missing, self.x_int,
                        remaining_budget)
        record = {"iteration": self.iteration, "cost_before": self.cost_spent,
                  "level": choice["level"], "ratio": choice["ratio"],
                  "speeds": [s["speed"] for s in self.states],
                  "states_ok": int(sum(s["success"] for s in self.states)),
                  "weights": None if self.weights is None else self.weights.tolist()}
        if choice["level"] is not None:
            x = choice["x"]
            for level in (1, 2):
                if level <= choice["level"] and choice["missing"][level - 1]:
                    self.surrogate.add(level, x, self.meta(x[None, :], level))
            self.cost_spent += choice["cost"]
            record["x"] = x.tolist()
            record["levels_run"] = [l for l in (1, 2) if l <= choice["level"]
                                    and choice["missing"][l - 1]]
            # distance (attitude, scaled by SPREAD) to the nearest state of the same section
            state_points = np.array([s["point"] for s in self.states])
            same = np.all(np.isclose(state_points[:, :N_SCORES], x[:N_SCORES]), axis=1)
            record["near_state"] = bool(same.any())
            dist = np.min(np.linalg.norm((state_points[:, N_SCORES:] - x[N_SCORES:]) / SPREAD,
                                         axis=1))
            record["attitude_distance"] = float(dist)
        record["time_s"] = time.perf_counter() - start
        self.history.append(record)
        self.iteration += 1
        if self.verbose:
            print(f"it {record['iteration']:3d} cost {self.cost_spent:6.1f} level "
                  f"{record['level']} ratio {record['ratio']} {record['time_s']:.1f} s",
                  flush=True)
        return record

    def run(self, budget: float) -> list:
        while budget - self.cost_spent >= self.surrogate.costs[0] - 1e-12:
            record = self.step(budget - self.cost_spent)
            if record["level"] is None:
                break
        self.surrogate.fit(refit=True)
        return self.history
