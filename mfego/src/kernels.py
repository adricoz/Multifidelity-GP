"""
Kernel (Covariance) functions and Classes for the multifidelity Gaussian process.
"""
import numpy as np


# 1/3 ---------------------------------------------------------------------------------------------
def cov_fct(x: np.ndarray, y: np.ndarray, theta: np.ndarray) -> float:
    """
    Squared Exponential covariance kernel between two individual points.
    
    Args:
    - x, y: 1D numpy arrays of shape (d,), the two points to compare.
    - theta: 1D numpy array of length d+2. 
             Contains [l_1, ..., l_d, t1, t2]
             where l_i are the length scales, t1 is the signal variance, 
             and t2 is the local variance/constant term.
    Returns:
    - covariance: float, the computed covariance value between x and y.
    """
    d = len(x)
    l = theta[:d]
    t1 = theta[d]
    t2 = theta[d+1]

    sum_dist = np.sum(((x - y) ** 2) / (2.0 * (l ** 2)))
    return t1 * np.exp(-sum_dist) + t2
# 2/3 ---------------------------------------------------------------------------------------------
def base_covariance_matrix(x: np.ndarray, theta_l: np.ndarray) -> np.ndarray:
    """
    Builds the full n x n covariance matrix for a training dataset X.
    
    Args:
    - x: numpy array of shape (n, d), the training dataset.
    - theta_l: numpy array containing the hyperparameters [l_1..l_d, t1, t2].
    
    Returns:
    - covariance_matrix: numpy array of shape (n, n), the covariance matrix.
    """
    n = x.shape[0]
    covariance_matrix = np.zeros((n, n))

    # symmetric matrix
    for i in range(n):
        for j in range(i, n):
            val = cov_fct(x[i], x[j], theta_l)
            covariance_matrix[i, j] = val
            covariance_matrix[j, i] = val

            return covariance_matrix
# 3/3 ---------------------------------------------------------------------------------------------
def k_l_vector(x: np.ndarray, x_train: np.ndarray, theta_l: np.ndarray) -> np.ndarray:
    """
    Builds the cross-covariance vector between a new candidate point x 
    and the existing training points X_train. (Highly optimized/vectorized).
    
    Args:
    - x: numpy array of shape (d,), the target point.
    - x_train: numpy array of shape (n, d), the training dataset.
    - theta_l: numpy array containing the hyperparameters [l_1..l_d, t1, t2].
    
    Returns:
    - k_vec: numpy array of shape (n, 1), the covariance vector.
    """
    d = x_train.shape[1]

    # hyperparameters
    l = theta_l[:d]
    t1 = theta_l[d]
    t2 = theta_l[d+1]

    # vectorial computation
    diff = x_train - x
    scaled_diff_sq = (diff ** 2) / (2.0 * (l ** 2))
    sum_dist = np.sum(scaled_diff_sq, axis=1)

    k_vec = t1 * np.exp(-sum_dist) + t2

    return k_vec.reshape(-1, 1)

# ----------------------------------------------------------------------
# -----------######-#------######-######-######-######-######-----------
# -----------#------#------#----#-#------#------#------#----------------
# -----------#------#------######-######-######-######-######-----------
# -----------#------#------#----#------#------#-#-----------#-----------
# -----------######-######-#----#-######-######-######-######-----------
# ----------------------------------------------------------------------

class Kernel:
    """
    Base class for kernel functions.
    """
    def __init__(self):

        self.lengthscale = None
        self.signal_variance = None
        self.bias_variance = None

    def get_params(self) -> np.ndarray:
        """
        Getter for the params of the kernel. 
        Returns:
        - 1D numpy array containing the hyperparameters.
        """
        return np.concatenate([self.lengthscale, [self.signal_variance, self.bias_variance]])

    def set_params(self, theta: np.ndarray) -> None:
        """
        Setter for the params of the kernel.
        Args:
        - theta: 1D numpy array containing the hyperparameters.
        """
        d = len(theta) - 2
        self.lengthscale = theta[:d]
        self.signal_variance = theta[d]
        self.bias_variance = theta[d + 1]

    def get_covariance_matrix(self, x_train: np.ndarray) -> np.ndarray:
        """
        Compute the covariance matrix for the given input points.
        (Not implemented in the mother class, MUST be implemented in subclasses.)
        """
        del x_train
        raise NotImplementedError("This method should be implemented in subclasses.")

    def get_cross_variance_vector(self, x_new: np.ndarray, x_train: np.ndarray) -> np.ndarray:
        """
        Compute the cross-variance vector between a new point and existing points.
        (Not implemented in the mother class, MUST be implemented in subclasses.)
        """
        del x_new, x_train
        raise NotImplementedError("This method should be implemented in subclasses.")

class SquaredExponentialKernel(Kernel):
    """
    A kernel that uses the squared exponential (RBF) covariance function.
    """
    def get_covariance_matrix(self, x_train: np.ndarray) -> np.ndarray:
        """
        Compute the covariance matrix for the given 
        input points using the squared exponential kernel.
        """
        return base_covariance_matrix(x_train, np.concatenate(
            [self.lengthscale, [self.signal_variance, self.bias_variance]]))

    def get_cross_variance_vector(self, x_new: np.ndarray, x_train: np.ndarray) -> np.ndarray:
        """
        Compute the cross-variance vector between a new point 
        and existing points using the squared exponential kernel.
        """
        return k_l_vector(x_new, x_train, np.concatenate(
            [self.lengthscale, [self.signal_variance, self.bias_variance]]))
