import numpy as np


# 1/3 ---------------------------------------------------------------------------------------------
def Cov_fct(x, y, Theta):
    """
    Squared Exponential covariance kernel between two individual points.
    
    Args:
    - x, y: 1D numpy arrays of shape (d,), the two points to compare.
    - Theta: 1D numpy array of length d+2. 
             Contains [l_1, ..., l_d, t1, t2]
             where l_i are the length scales, t1 is the signal variance, 
             and t2 is the local variance/constant term.
    Returns:
    - covariance: float, the computed covariance value between x and y.
    """
    d = len(x)
    l = Theta[:d]
    t1 = Theta[d]
    t2 = Theta[d+1]
    
    sum_dist = np.sum(((x - y) ** 2) / (2.0 * (l ** 2)))
    return t1 * np.exp(-sum_dist) + t2
# 2/3 ---------------------------------------------------------------------------------------------
def base_covariance_matrix(X, Theta_l):
    """
    Builds the full n x n covariance matrix for a training dataset X.
    
    Args:
    - X: numpy array of shape (n, d), the training dataset.
    - Theta_l: numpy array containing the hyperparameters [l_1..l_d, t1, t2].
    
    Returns:
    - K: numpy array of shape (n, n), the covariance matrix.
    """
    n = X.shape[0]
    K = np.zeros((n, n))
    
    # symmetric matrix
    for i in range(n):
        for j in range(i, n):
            val = Cov_fct(X[i], X[j], Theta_l)
            K[i, j] = val
            K[j, i] = val
            
    return K
# 3/3 ---------------------------------------------------------------------------------------------
def k_l_vector(x, X_train, Theta_l):
    """
    Builds the cross-covariance vector between a new candidate point x 
    and the existing training points X_train. (Highly optimized/vectorized).
    
    Args:
    - x: numpy array of shape (d,), the target point.
    - X_train: numpy array of shape (n, d), the training dataset.
    - Theta_l: numpy array containing the hyperparameters [l_1..l_d, t1, t2].
    
    Returns:
    - k_vec: numpy array of shape (n, 1), the covariance vector.
    """
    d = X_train.shape[1]
    
    # hyperparameters
    l = Theta_l[:d]
    t1 = Theta_l[d]
    t2 = Theta_l[d+1]
    
    # vectorial computation
    diff = X_train - x
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
    def __init__(self):
        """
        Initializes the Kernel with the given hyperparameters.
        
        Args:
        - Theta: list or numpy array, hyperparameters of the kernel.
        
        Returns:
        - None
        """
        self.Theta = None

    def __call__(self, x, y):
        """
        Computes the covariance between two input points x and y.
        
        Args:
        - x: numpy array of shape (n_features,), first input point.
        - y: numpy array of shape (n_features,), second input point.
        
        Returns:
        - covariance: float, the computed covariance value between x and y.
        """
        return Cov_fct(x, y, self.Theta)
    
    def get_covariance_matrix(self, X):
        """
        Computes the covariance matrix for a set of input points X.
        
        Args:
        - X: numpy array of shape (n, d), the input dataset.
        
        Returns:
        - K: numpy array of shape (n, n), the covariance matrix.
        
        """
        return base_covariance_matrix(X, self.Theta)
    
    def get_cross_variance_vector(self, x_new, X):
        """
        Computes the cross-variance vector between a new input point and a set of training points.
        
        Args:
        - x_new: numpy array of shape (n_features,), the new input point.
        - X: numpy array of shape (n, d), the training dataset.
        
        Returns:
        - k_star: numpy array of shape (n,), the cross-variance vector.
        """
        return k_l_vector(x_new, X, self.Theta)

class SquaredExponentialKernel(Kernel):
    def __init__(self, Theta):
        """
        Initializes the Squared Exponential Kernel with the given hyperparameters.
        
        Args:
        - Theta: list or numpy array, hyperparameters of the kernel (lengthscale and variance).
        
        Returns:
        - None
        """
        self.lengthscale = Theta[:-1]
        self.variance = Theta[-1]

    def __call__(self, x, y):
        """
        Computes the covariance between two input points x and y using the squared exponential formula.
        
        Args:
        - x: numpy array of shape (n_features,), first input point.
        - y: numpy array of shape (n_features,), second input point.
        
        Returns:
        - covariance: float, the computed covariance value between x and y.
        """
        pass