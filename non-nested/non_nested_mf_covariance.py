import numpy as np
#OK

def Cov_fct(x, y, Theta):
    """
    Squared Exponential covariance kernel between two individual points.
    
    Arguments:
    - x, y: 1D numpy arrays of shape (d,), the two points to compare.
    - Theta: 1D numpy array of length d+2. 
             Contains [l_1, ..., l_d, t1, t2]
             where l_i are the length scales, t1 is the signal variance, 
             and t2 is the local variance/constant term.
    """
    d = len(x)
    l = Theta[:d]
    t1 = Theta[d]
    t2 = Theta[d+1]
    
    sum_dist = np.sum(((x - y) ** 2) / (2.0 * (l ** 2)))
    return t1 * np.exp(-sum_dist) + t2
# -----------------------------------------------------------------------------------------
def base_covariance_matrix(X, Theta_l):
    """
    Builds the full n x n covariance matrix for a training dataset X.
    
    Arguments:
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

# -----------------------------------------------------------------------------------------
# def nested_mf_covariance(x, y, Theta, rho_l , fidelity_level):
#     """
#     Computes the covariance between two points x and y for a nested multi-fidelity Gaussian Process.
#     Essentially Equation 14 in the referenced paper, which sums the contributions from each fidelity level.
    
#     Parameters:
#     - x: numpy array, point in the input space
#     - y: numpy array, point in the input space
#     - length: numpy array, length scales for each dimension
#     - t1: float, signal variance
#     - t2: float, noise variance
#     - rho_l: float, correlation coefficient for level l
#     - level: int, fidelity level (1 for lowest fidelity)
    
#     Returns:
#     - cov: float, covariance value between x and y at the specified fidelity level
#     """
#     n_params = len(x)

#     Sum_corr = 0.0
#     for j in range(fidelity_level):

#         rho_2_i = 1
#         for i in range(j,fidelity_level-1):
#             rho_2_i *= rho_l[i]**2
            
#         Sum_corr += rho_2_i * Cov_fct(x, y, Theta[j][0:n_params], Theta[j][n_params], Theta[j][n_params+1])

#     return Sum_corr
# -----------------------------------------------------------------------------------------

# def nested_mf_covariance_matrix(X, Theta, rho_l, fidelity_level):
#     """
#     Computes the covariance matrix for a set of points X at a given fidelity level.
    
#     Parameters:
#     - X: numpy array, shape (n_samples, n_features), input points
#     - Theta: list of tuples, each containing (length scales, t1, t2) for each fidelity level
#     - rho_l: list of floats, correlation coefficients for each fidelity level
#     - fidelity_level: int, the fidelity level for which to compute the covariance matrix
    
#     Returns:
#     - C: numpy array, shape (n_samples, n_samples), covariance matrix at the specified fidelity level
#     """
#     n_samples = X.shape[0]
#     C = np.zeros((n_samples, n_samples))
    
#     for i in range(n_samples):
#         for j in range(n_samples):
#             C[i, j] = nested_mf_covariance(X[i], X[j], Theta, rho_l, fidelity_level)
    
#     return C
# -----------------------------------------------------------------------------------------

def k_l_vector(x, X_train, Theta_l):
    """
    Builds the cross-covariance vector between a new candidate point x 
    and the existing training points X_train. (Highly optimized/vectorized).
    
    Arguments:
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
