import numpy as np
from scipy.stats import qmc
from traitlets import List, Tuple


def generate_non_nested_lhs(d, n_levels_points):
    """
    Generates non-nested Latin Hypercube Sampling (LHS) designs for N levels of fidelity.
    Each higher fidelity level's points are independent of the previous lower fidelity level.
    
    Args:
    - d: int, dimension of the input space (e.g., 6 for Hartmann 6D)
    - n_levels_points: list of int, number of points for each level from lowest to highest 
                       
    Returns:
    - X_levels: list of numpy arrays, containing the design points for each fidelity level.
    """
    # Use scipy's LatinHypercube
    sampler = qmc.LatinHypercube(d=d, seed=42)

    X_levels = []

    for n_pts in n_levels_points:
       
        X_sub = sampler.random(n=n_pts)
        X_levels.append(X_sub)
        
    return X_levels

def is_already_evaluated(x, X_train_level, tol=1e-6):
    """
    Checks if a point x is already present in the dataset X_train_level.
    
    Arguments:
    - x: numpy array of shape (d,), the point to check
    - X_train_level: numpy array of shape (n, d), the existing dataset for a specific level
    - tol: float, distance tolerance
    
    Returns:
    - True if the point is already in the dataset, False otherwise.
    """
    if X_train_level.shape[0] == 0:
        return False
        
    distances = np.linalg.norm(X_train_level - x, axis=1)
    return np.min(distances) < tol

# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------

class ExperimentData:
    def __init__(self, bounds: List[Tuple[float, float]], costs: List[float]) -> None:
        """Initializes the ExperimentData class with bounds and costs.
        
        Args:
        - bounds: list of tuples, where each tuple defines the lower and upper bounds for a dimension.
        - costs: list of floats, representing the cost associated with each fidelity level.

        Returns:
        - None
        """
        self.bounds = bounds
        self.costs = costs
        self.dim = len(bounds)
        self.X_dict = None
        self.Y_dict = None

    def generate_initial_design(self, n_points_list):
        """Generates an initial design of experiments based on the provided number of points for each fidelity level

        Args:
        - n_points_list: list of integers
        
        Returns:
        - None
        """
        self.X_dict = generate_non_nested_lhs(self.dim, n_points_list)

    def add_observation(self, level, x_new, y_new):

        if self.Y_dict is None:
            self.Y_dict = [np.array([]) for _ in range(len(self.X_dict))]
        self.Y_dict[level] = np.append(self.Y_dict[level], y_new)
        self.X_dict[level] = np.vstack([self.X_dict[level], x_new]) if self.X_dict[level].size else np.array([x_new])

    def check_already_evaluated(self, level, x):
        return is_already_evaluated(x, self.X_dict[level])