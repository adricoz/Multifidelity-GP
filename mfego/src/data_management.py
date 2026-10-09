"""
This module is the only module that can modify the data.
"""
import numpy as np
from scipy.stats import qmc

# [FIX-R6] removed `from traitlets import List, Tuple`: traitlets is not a typing module
# (TypeError at import on Python <= 3.13, ModuleNotFoundError if not installed).
# Built-in generics (list[...], tuple[...]) are used instead, as in the rest of the code.


class ExperimentData:
    """
    Class to manage the experimental data for multifidelity optimization.
    """
    def __init__(self, bounds: list[tuple[float, float]], costs: list[float]) -> None:

        self.bounds = bounds
        self.costs = costs
        self.dim = len(bounds)
        self.x_dict = {} #Format: {1: array(...), 2: array(...)}
        self.y_dict = {} #Format: {1: array(...), 2: array(...)}
        self.metrics_dict = {} #Format: {1: array(...), 2: array(...)}

    def generate_initial_design(self, points_per_level: list[int], seed: int = 42,
                                feasibility=None, oversampling: int = 20) -> None:
        """Generates an initial design of experiments based
        on the provided number of points for each fidelity level

        [KC] With a known-constraint function feasibility(x) -> bool array (x of shape (m, d)),
        a Latin hypercube of oversampling * n points is drawn, its feasible points are kept and
        n of them are selected by greedy maximin distance (well spread feasible design).
        Without it (default), the design is the plain Latin hypercube of n points.
        """
        lower_bounds = [b[0] for b in self.bounds]
        upper_bounds = [b[1] for b in self.bounds]

        for l, n_points in enumerate(points_per_level, start = 1):
            # we make sure to have different seeds for the different levels
            # to avoid redundancy
            # [FIX-R1] base seed is now a parameter (default 42 = previous behaviour)
            sampler = qmc.LatinHypercube(d = self.dim, seed = seed + l)
            if feasibility is None:
                sample_unit = sampler.random(n = n_points)
                self.x_dict[l] = qmc.scale(sample_unit, lower_bounds, upper_bounds)
            else:
                candidates = qmc.scale(sampler.random(n = max(1, oversampling) * n_points),
                                       lower_bounds, upper_bounds)
                feasible = candidates[np.asarray(feasibility(candidates), dtype=bool)]
                if len(feasible) < n_points:
                    raise ValueError(
                        f"Initial design of level {l}: only {len(feasible)} feasible points out "
                        f"of {len(candidates)} candidates for {n_points} requested (constraints "
                        "too tight for the bounds, or increase oversampling).")
                self.x_dict[l] = greedy_maximin(feasible, n_points, self.bounds)
            self.y_dict[l] = np.array([])  # Initialize y_dict for this level
            self.metrics_dict[l] = []  # Initialize metrics_dict for this level

    def is_already_evaluated(self, level: int, x: np.ndarray, tol: float = 1e-6) -> bool:
        """
        Check if a point has already been evaluated at a given fidelity level.
        (Failed evaluations, stored with a NaN value, also count as evaluated.)
        """
        if level not in self.x_dict or self.x_dict[level].size == 0:
            return False
        distances = np.linalg.norm(self.x_dict[level] - x, axis=1)
        return np.min(distances) < tol

    def get_training_data(self, level: int) -> tuple[np.ndarray, np.ndarray]:
        """
        [FIX-R4] Returns the (x, y) observations of a level that can be used to train a GP,
        i.e. without the failed evaluations (NaN or inf outputs of the simulator).

        Args:
        - level: fidelity level (1 to L).
        Returns:
        - x: numpy array of shape (n_valid, d).
        - y: numpy array of shape (n_valid,).
        """
        x = np.atleast_2d(np.asarray(self.x_dict[level], dtype=float))
        y = np.asarray(self.y_dict[level], dtype=float).reshape(-1)
        valid = np.isfinite(y)
        return x[valid], y[valid]

    def n_failed(self, level: int) -> int:
        """[FIX-R4] Number of failed (non finite) evaluations stored at a given level."""
        y = np.asarray(self.y_dict.get(level, []), dtype=float).reshape(-1)
        return int(np.sum(~np.isfinite(y)))

    def best_observation(self, level: int) -> tuple[np.ndarray, float]:
        """
        [FIX-R4] Best (minimum) finite observation of a level, ignoring failed evaluations.

        Returns:
        - x_best: numpy array of shape (d,) (None if no valid observation).
        - y_best: float (np.nan if no valid observation).
        """
        x, y = self.get_training_data(level)
        if y.size == 0:
            return None, np.nan
        i_best = int(np.argmin(y))
        return x[i_best], float(y[i_best])

    def add_observation(self, level: int, x_new: np.ndarray,
                        y_new: float, metrics: dict) -> None:
        """
        Add a new observation to the experimental data.
        A failed evaluation (y_new NaN or inf) is stored as NaN: it is kept to avoid
        evaluating the same point again but it is excluded from the GP training.
        """
        if metrics is None:
            metrics = {}

        # [FIX-R4] non finite outputs (failed simulations) are stored as NaN
        if y_new is None or not np.isfinite(y_new):
            y_new = np.nan

        if level not in self.x_dict or self.x_dict[level].size == 0:
            self.x_dict[level] = np.array([x_new])
            self.y_dict[level] = np.array([y_new])
            self.metrics_dict[level] = [metrics]
        else:
            self.x_dict[level] = np.vstack([self.x_dict[level], x_new])
            self.y_dict[level] = np.append(self.y_dict[level], y_new)
            self.metrics_dict[level].append(metrics)


def greedy_maximin(candidates: np.ndarray, n_points: int,
                   bounds: list[tuple[float, float]]) -> np.ndarray:
    """
    [KC] Selects n_points well spread points among candidates: starts with the candidate
    closest to the centre of the domain, then repeatedly adds the candidate whose distance to
    the already selected points is the largest (distances in coordinates scaled to [0, 1]).

    Args:
    - candidates: numpy array of shape (m, d), m >= n_points.
    - n_points: number of points to select.
    - bounds: list of (lower, upper) bounds used to scale the distances.
    Returns:
    - numpy array of shape (n_points, d).
    """
    candidates = np.atleast_2d(np.asarray(candidates, dtype=float))
    if n_points <= 0:
        return np.empty((0, candidates.shape[1]))
    if len(candidates) < n_points:
        raise ValueError(f"greedy_maximin: {len(candidates)} candidates for {n_points} points")
    lows = np.array([b[0] for b in bounds], dtype=float)
    spans = np.array([b[1] - b[0] for b in bounds], dtype=float)
    unit = (candidates - lows) / np.where(spans > 0, spans, 1.0)
    selected = [int(np.argmin(np.linalg.norm(unit - 0.5, axis=1)))]
    min_dist = np.linalg.norm(unit - unit[selected[0]], axis=1)
    while len(selected) < n_points:
        i_next = int(np.argmax(min_dist))
        selected.append(i_next)
        min_dist = np.minimum(min_dist, np.linalg.norm(unit - unit[i_next], axis=1))
    return candidates[selected]
