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

    def generate_initial_design(self, points_per_level: list[int], seed: int = 42) -> None:
        """Generates an initial design of experiments based
        on the provided number of points for each fidelity level
        """
        lower_bounds = [b[0] for b in self.bounds]
        upper_bounds = [b[1] for b in self.bounds]

        for l, n_points in enumerate(points_per_level, start = 1):
            # we make sure to have different seeds for the different levels
            # to avoid redondency
            # [FIX-R1] base seed is now a parameter (default 42 = previous behaviour)
            sampler = qmc.LatinHypercube(d = self.dim, seed = seed + l)
            sample_unit = sampler.random(n = n_points)
            self.x_dict[l] = qmc.scale(sample_unit, lower_bounds, upper_bounds)
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
