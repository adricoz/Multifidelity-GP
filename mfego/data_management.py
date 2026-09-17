import numpy as np
from scipy.stats import qmc
from traitlets import List, Tuple


class ExperimentData:
    def __init__(self, bounds: List[Tuple[float, float]], costs: List[float]) -> None:

        self.bounds = bounds
        self.costs = costs
        self.dim = len(bounds)
        self.X_dict = {} #Format: {1: array(...), 2: array(...)}
        self.Y_dict = {} #Format: {1: array(...), 2: array(...)}

    def generate_initial_design(self, points_per_level):
        """Generates an initial design of experiments based on the provided number of points for each fidelity level
        """
        lower_bounds = [b[0] for b in self.bounds]
        upper_bounds = [b[1] for b in self.bounds]

        for l, n_points in enumerate(points_per_level, start = 1):
            sampler = qmc.LatinHypercube(d = self.dim, seed = 42 + l)
            sample_unit = sampler.random(n = n_points)
            self.X_dict[l] = qmc.scale(sample_unit, lower_bounds, upper_bounds)
            self.Y_dict[l] = np.array([])  # Initialize Y_dict for this level

    def is_already_evaluated(self, level, x, tol=1e-6):
        if level not in self.X_dict or self.X_dict[level].size == 0:
            return False
        distances = np.linalg.norm(self.X_dict[level] - x, axis=1)
        return np.min(distances) < tol
        

    def add_observation(self, level, x_new, y_new):

        if level not in self.X_dict or self.X_dict[level].size == 0:
            self.X_dict[level] = np.array([x_new])
            self.Y_dict[level] = np.array([y_new])
        else:
            self.X_dict[level] = np.vstack([self.X_dict[level], x_new])
            self.Y_dict[level] = np.append(self.Y_dict[level], y_new)
