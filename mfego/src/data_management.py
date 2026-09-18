"""
This module is the only module that can modify the data. 
"""
import numpy as np
from scipy.stats import qmc
from traitlets import List, Tuple


class ExperimentData:
    """
    Class to manage the experimental data for multifidelity optimization.
    """
    def __init__(self, bounds: List[Tuple[float, float]], costs: List[float]) -> None:

        self.bounds = bounds
        self.costs = costs
        self.dim = len(bounds)
        self.x_dict = {} #Format: {1: array(...), 2: array(...)}
        self.y_dict = {} #Format: {1: array(...), 2: array(...)}
        self.metrics_dict = {} #Format: {1: array(...), 2: array(...)}

    def generate_initial_design(self, points_per_level: List[int]) -> None:
        """Generates an initial design of experiments based 
        on the provided number of points for each fidelity level
        """
        lower_bounds = [b[0] for b in self.bounds]
        upper_bounds = [b[1] for b in self.bounds]

        for l, n_points in enumerate(points_per_level, start = 1):
            # we make sure to have different seeds for the different levels 
            # to avoid redondency
            sampler = qmc.LatinHypercube(d = self.dim, seed = 42 + l)
            sample_unit = sampler.random(n = n_points)
            self.x_dict[l] = qmc.scale(sample_unit, lower_bounds, upper_bounds)
            self.y_dict[l] = np.array([])  # Initialize y_dict for this level
            self.metrics_dict[l] = []  # Initialize metrics_dict for this level

    def is_already_evaluated(self, level: int, x: np.ndarray, tol: float = 1e-6) -> bool:
        """
        Check if a point has already been evaluated at a given fidelity level.
        """
        if level not in self.x_dict or self.x_dict[level].size == 0:
            return False
        distances = np.linalg.norm(self.x_dict[level] - x, axis=1)
        return np.min(distances) < tol



    def add_observation(self, level: int, x_new: np.ndarray, 
                        y_new: float, metrics: dict) -> None:
        """
        Add a new observation to the experimental data.
        """
        if metrics is None:
            metrics = {}

        if level not in self.x_dict or self.x_dict[level].size == 0:
            self.x_dict[level] = np.array([x_new])
            self.y_dict[level] = np.array([y_new])
            self.metrics_dict[level] = [metrics]
        else:
            self.x_dict[level] = np.vstack([self.x_dict[level], x_new])
            self.y_dict[level] = np.append(self.y_dict[level], y_new)
            self.metrics_dict[level].append(metrics)
