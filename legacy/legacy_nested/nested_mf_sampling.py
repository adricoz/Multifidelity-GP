import numpy as np
from scipy.stats import qmc

# Generating the Latin Hyper Square 
def generate_nested_lhs(d, n_levels_points):
    """
    Generates nested Latin Hypercube Sampling (LHS) designs for N levels of fidelity.
    Each higher fidelity level's points are a subset of the previous lower fidelity level.
    
    Parameters:
    - d: int, dimension of the input space (e.g., 6 for Hartmann 6D)
    - n_levels_points: list of int, number of points for each level from lowest to highest 
                       [n_1, n_2, ..., n_N], where n_1 > n_2 > ... > n_N
                       
    Returns:
    - X_levels: list of numpy arrays, containing the design points for each fidelity level.
    """
    # Use scipy's LatinHypercube to generate the largest set first (Level 1)
    sampler = qmc.LatinHypercube(d=d, seed=42)
    max_points = n_levels_points[0]
    base_sample = sampler.random(n=max_points)
    
    X_levels = []
    # Sort or select nested subsets from the base sample
    # For a strictly nested design, Level k is a prefix subset of Level k-1
    for n_pts in n_levels_points:
        # Take the first n_pts from the base Latin Hypercube sample to preserve space-filling properties
        X_sub = base_sample[:n_pts, :]
        X_levels.append(X_sub)
        
    return X_levels


def Delta_Y_l(Y_l, Y_l_minus_1, rho_l_minus_1):
    """
    Computes the residuals for multi-fidelity Gaussian Process modeling.
    
    Parameters:
    - Y_l: numpy array, observations at fidelity level l
    - Y_l_minus_1: numpy array, observations at fidelity level l-1
    - rho_l_minus_1: float, correlation coefficient between levels l and l-1
    
    Returns:
    - Delta_Y: numpy array, residuals for level l
    """
    if Y_l_minus_1 is None or rho_l_minus_1 is None:
        return Y_l  
    # ----------------------------------------
    
    if len(Y_l) != len(Y_l_minus_1):
        raise ValueError("Y_l and Y_l_minus_1 must have the same length.")
        
    return Y_l - rho_l_minus_1 * Y_l_minus_1

# -----------------------------------------------------------------------------------------
# Here I'm not entirely sure 
# -----------------------------------------------------------------------------------------
def extract_subpart_vector(X_higher, X_lower, Y_lower, tol=1e-6):
    """
    Extracts the observation values from the lower fidelity dataset (Y_lower)
    that correspond exactly to the spatial coordinates in X_higher.
    
    Arguments:
    - X_higher: numpy array of shape (n_higher, d), points at level l
    - X_lower: numpy array of shape (n_lower, d), points at level l-1
    - Y_lower: numpy array of shape (n_lower,), observations at level l-1
    - tol: float, distance tolerance for floating-point comparisons
    
    Returns:
    - subpart_Y: numpy array of shape (n_higher,) containing the matching Y values.
    """
    subpart_Y = np.zeros(X_higher.shape[0])
    
    for i, x in enumerate(X_higher):
        # Calculate the Euclidean distance between x and all points in X_lower
        distances = np.linalg.norm(X_lower - x, axis=1)
        
        # Find the index of the closest point
        closest_idx = np.argmin(distances)
        
        # Ensure the point is actually the same (distance is close to 0)
        if distances[closest_idx] < tol:
            subpart_Y[i] = Y_lower[closest_idx]
        else:
            raise ValueError(f"Point {x} from higher fidelity not found in lower fidelity dataset. Nested property violated.")
            
    return subpart_Y
# -----------------------------------------------------------------------------------------
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