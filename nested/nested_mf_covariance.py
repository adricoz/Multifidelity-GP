import numpy as np
#OK

def Cov_fct(x, y, l, t1, t2):
    """
    Covariance function for Gaussian Process.

    Parameters
    ----------
    x : array-like, shape (n_samples_x, n_features)
        Input samples.
    y : array-like, shape (n_samples_y, n_features)
        Input samples.
    l : float
        Length scale parameter.
    t1 : float
        Signal variance parameter.
    t2 : float
        Noise variance parameter.


    Returns
    -------
    cov : float 
        Covariance between x and y.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    l = np.asarray(l)

    if x.ndim == 1:
        x = x.reshape(1, -1)
    if y.ndim == 1:
        y = y.reshape(1, -1)
    if l.ndim == 0:
        l = np.full(x.shape[1], l)
    if x.shape[1] != y.shape[1] != l.shape[0]:
        raise ValueError("Input samples must have the same number of features.")

    cov = t1
    for i in range(x.shape[0]):
        dist = x[0][i] - y[0][i]
        cov *= np.exp(-(dist**2)/(2 * l[i]**2))

    return cov + t2
# -----------------------------------------------------------------------------------------
def base_covariance_matrix(X, Theta_l):
    """
    Computes the base covariance matrix.

    parameters
    ----------  
    X : array-like, shape (n_samples, n_features)
        Input samples.
    Theta_l : array-like, shape (n_features + 2,)
        Hyperparameters for the covariance function (length scales, t1, t2).

    Returns
    -------
    C : array-like, shape (n_samples, n_samples)
        Covariance matrix.
    """
    n = len(X)
    C = np.zeros((n, n))
    d = X.shape[1]
    
    # On gère l'extraction selon que vous optimisiez t1/t2 ou non
    lengthscales = Theta_l[0:d]
    t1 = Theta_l[d] if len(Theta_l) > d else 1.0
    t2 = Theta_l[d+1] if len(Theta_l) > d+1 else 1.0
    
    for i in range(n):
        for j in range(n):
            C[i, j] = Cov_fct(X[i], X[j], lengthscales, t1, t2)
    return C

# -----------------------------------------------------------------------------------------
def nested_mf_covariance(x, y, Theta, rho_l , fidelity_level):
    """
    Computes the covariance between two points x and y for a nested multi-fidelity Gaussian Process.
    Essentially Equation 14 in the referenced paper, which sums the contributions from each fidelity level.
    
    Parameters:
    - x: numpy array, point in the input space
    - y: numpy array, point in the input space
    - length: numpy array, length scales for each dimension
    - t1: float, signal variance
    - t2: float, noise variance
    - rho_l: float, correlation coefficient for level l
    - level: int, fidelity level (1 for lowest fidelity)
    
    Returns:
    - cov: float, covariance value between x and y at the specified fidelity level
    """
    n_params = len(x)

    Sum_corr = 0.0
    for j in range(fidelity_level):

        rho_2_i = 1
        for i in range(j,fidelity_level-1):
            rho_2_i *= rho_l[i]**2
            
        Sum_corr += rho_2_i * Cov_fct(x, y, Theta[j][0:n_params], Theta[j][n_params], Theta[j][n_params+1])

    return Sum_corr
# -----------------------------------------------------------------------------------------

def nested_mf_covariance_matrix(X, Theta, rho_l, fidelity_level):
    """
    Computes the covariance matrix for a set of points X at a given fidelity level.
    
    Parameters:
    - X: numpy array, shape (n_samples, n_features), input points
    - Theta: list of tuples, each containing (length scales, t1, t2) for each fidelity level
    - rho_l: list of floats, correlation coefficients for each fidelity level
    - fidelity_level: int, the fidelity level for which to compute the covariance matrix
    
    Returns:
    - C: numpy array, shape (n_samples, n_samples), covariance matrix at the specified fidelity level
    """
    n_samples = X.shape[0]
    C = np.zeros((n_samples, n_samples))
    
    for i in range(n_samples):
        for j in range(n_samples):
            C[i, j] = nested_mf_covariance(X[i], X[j], Theta, rho_l, fidelity_level)
    
    return C
# -----------------------------------------------------------------------------------------

def k_l_vector(new_x, X_vector, Theta, rho_l, fidelity_level):
    """
    Computes the covariance vector between a new point and a set of existing points for a given fidelity level.

    Parameters
    ----------
    new_x : array-like, shape (n_features,)
        New input point.
    X_vector : array-like, shape (n_samples, n_features)
        Existing input points.
    Theta : list of tuples
        Each tuple contains (length scales, t1, t2) for each fidelity level.
    rho_l : list of floats
        Correlation coefficients for each fidelity level.
    fidelity_level : int
        The fidelity level for which to compute the covariance vector.

    Returns
    -------
    k_vector : array-like, shape (n_samples,)
        Covariance vector between new_x and each point in X_vector at the specified fidelity level.
    """
    n_samples = X_vector.shape[0]
    k_vector = np.zeros(n_samples)

    for i in range(n_samples):
        k_vector[i] = nested_mf_covariance(new_x, X_vector[i], Theta, rho_l, fidelity_level)

    return k_vector