"""
Kernel (Covariance) functions and Classes for the multifidelity Gaussian process.
"""
import numpy as np


# 1/5 ---------------------------------------------------------------------------------------------
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
# 2/5 ---------------------------------------------------------------------------------------------
def base_covariance_matrix(x: np.ndarray, theta_l: np.ndarray) -> np.ndarray:
    """
    Builds the full n x n covariance matrix for a training dataset X.

    Args:
    - x: numpy array of shape (n, d), the training dataset.
    - theta_l: numpy array containing the hyperparameters [l_1..l_d, t1, t2].

    Returns:
    - covariance_matrix: numpy array of shape (n, n), the covariance matrix.
    """
    d = x.shape[1]
    l = theta_l[:d]
    t1 = theta_l[d]
    t2 = theta_l[d+1]

    # [FIX-N1] vectorized computation (broadcasting) instead of the double Python loop
    # over cov_fct: same values (up to round-off), O(n^2 d) numpy operations.
    sum_dist = np.sum(pairwise_sq_diff(x) / (2.0 * (l ** 2)), axis=2)
    covariance_matrix = t1 * np.exp(-sum_dist) + t2

    return covariance_matrix
# 3/5 ---------------------------------------------------------------------------------------------
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
# 4/5 ---------------------------------------------------------------------------------------------
def cross_covariance_matrix(x_new: np.ndarray, x_train: np.ndarray,
                            theta_l: np.ndarray) -> np.ndarray:
    """
    [FIX-N6] Batch version of k_l_vector: cross-covariance between m new points
    and the n training points (used for vectorized predictions on grids / DE populations).

    Args:
    - x_new: numpy array of shape (m, d), the target points.
    - x_train: numpy array of shape (n, d), the training dataset.
    - theta_l: numpy array containing the hyperparameters [l_1..l_d, t1, t2].

    Returns:
    - k_mat: numpy array of shape (m, n).
    """
    d = x_train.shape[1]
    l = theta_l[:d]
    t1 = theta_l[d]
    t2 = theta_l[d+1]

    diff = x_new[:, None, :] - x_train[None, :, :]
    sum_dist = np.sum((diff ** 2) / (2.0 * (l ** 2)), axis=2)
    return t1 * np.exp(-sum_dist) + t2
# 5/5 ---------------------------------------------------------------------------------------------
def pairwise_sq_diff(x: np.ndarray) -> np.ndarray:
    """
    [FIX-N1] Squared coordinate differences between all pairs of points.

    Args:
    - x: numpy array of shape (n, d).

    Returns:
    - sq_diff: numpy array of shape (n, n, d), sq_diff[i, j, m] = (x[i, m] - x[j, m])**2.
    """
    diff = x[:, None, :] - x[None, :, :]
    return diff ** 2

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

    def get_cross_covariance_matrix(self, x_new: np.ndarray, x_train: np.ndarray) -> np.ndarray:
        """
        [FIX-N6] Batch cross-covariance (m new points x n training points).
        Default implementation: loop over get_cross_variance_vector (subclasses may vectorize).
        """
        return np.hstack([self.get_cross_variance_vector(x, x_train) for x in x_new]).T

    def get_log_params_gradients(self, x_train: np.ndarray) -> list[np.ndarray]:
        """
        [FIX-N3] Derivatives of the covariance matrix w.r.t. the LOG of each kernel
        hyperparameter (same order as get_params). Optional: if a subclass does not
        implement it, the GP falls back to finite-difference gradients.
        """
        del x_train
        raise NotImplementedError("Analytical gradients are not implemented for this kernel.")

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

    def get_cross_covariance_matrix(self, x_new: np.ndarray, x_train: np.ndarray) -> np.ndarray:
        """
        [FIX-N6] Vectorized cross-covariance matrix (m, n) for the squared exponential kernel.
        """
        return cross_covariance_matrix(np.atleast_2d(x_new), x_train, np.concatenate(
            [self.lengthscale, [self.signal_variance, self.bias_variance]]))

    def get_log_params_gradients(self, x_train: np.ndarray) -> list[np.ndarray]:
        """
        [FIX-N3] dK/dlog(theta_j) for theta = [l_1..l_d, t1, t2] (Rasmussen & Williams,
        GPML Eq. 5.9 needs dK/dtheta_j; the log-derivative is theta_j * dK/dtheta_j):
        - dK/dlog(l_m) = t1 * exp(-sum_dist) * (x_m - x'_m)^2 / l_m^2
        - dK/dlog(t1)  = t1 * exp(-sum_dist)
        - dK/dlog(t2)  = t2 * ones
        """
        l = self.lengthscale
        sq_diff = pairwise_sq_diff(x_train)
        exp_part = self.signal_variance * np.exp(-np.sum(sq_diff / (2.0 * l ** 2), axis=2))

        grads = [exp_part * sq_diff[:, :, m] / l[m] ** 2 for m in range(len(l))]
        grads.append(exp_part)
        grads.append(self.bias_variance * np.ones_like(exp_part))
        return grads
