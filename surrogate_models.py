from venv import logger

import numpy as np
from scipy.linalg import cholesky, solve
from scipy.optimize import minimize

from kernels import Cov_fct, base_covariance_matrix, k_l_vector, Kernel


def predict_base_gp(x_new, X_train, Y_train, Theta_l, sigma_epsilon_l):
    """
    Simply predicts the average and variance given by Eqs. 11 and 12 in the paper.

    Args
    x_new : array-like, shape (d,)
        New point where the prediction is desired.  
    X_train : array-like, shape (n_samples, d)
        Training input points.  
    Y_train : array-like, shape (n_samples,)
        Training output values.
    Theta_l : array-like, shape (d+2,)
        Hyperparameters for the covariance function at level l.
    sigma_epsilon_l : float
        Noise variance at level l.

    Returns
    delta_hat_scalar : float
        Predicted mean at the new point.
    sigma2_delta_scalar : float
        Predicted variance at the new point.
    """
    C = base_covariance_matrix(X_train, Theta_l)
    K = C + sigma_epsilon_l * np.eye(len(X_train))
    K_inv = np.linalg.inv(K)

    k_vec = np.zeros(len(X_train))
    for i in range(len(X_train)):
        k_vec[i] = Cov_fct(x_new, X_train[i], Theta_l)

    # after some debugging we found that diemnsions were not right when 1 single point was found 
    Y_train_1D = Y_train.reshape(-1)  # Ensure Y_train is 1D
    k_vec = k_vec.reshape(-1)  # Ensure k_vec is 1D

    kappa = Cov_fct(x_new, x_new, Theta_l)
    delta_hat = k_vec.T @ K_inv @ Y_train_1D
    sigma2_delta = kappa - (k_vec.T @ K_inv @ k_vec)
    
    delta_hat_scalar = delta_hat.item() if hasattr(delta_hat, 'item') else delta_hat
    sigma2_delta_scalar = max(sigma2_delta.item() if hasattr(sigma2_delta, 'item') else sigma2_delta, 1e-12)

    return delta_hat_scalar, sigma2_delta_scalar
# ----------------------------------------------------------------------
def predict_mf_mean_up_to_level(X_target, target_level, thetas, rhos, noises, X_train, Y_train):
    """
    Predicts the MF surrogate mean up to a specific level (target_level) 
    for a set of target points X_target. (NON-NESTED approach).
    """
    n_points = X_target.shape[0]
    f_hat_prev = np.zeros(n_points)
    
    for l in range(1, target_level + 1):

        Y_l_1d = np.squeeze(Y_train[l])

        # residual
        if l == 1:
            # 1d correction
            Delta_Y_train_l = Y_l_1d
        else:
            # internal recursion
            f_hat_train_prev = predict_mf_mean_up_to_level(X_train[l], l - 1, thetas, rhos, noises, X_train, Y_train)
            f_hat_train_prev_1d = np.squeeze(f_hat_train_prev)

            # C1d correction
            Delta_Y_train_l = Y_l_1d - rhos[l-2] * f_hat_train_prev_1d
            
        # Covariance matrix for level l
        K_l = base_covariance_matrix(X_train[l], thetas[l-1]) + noises[l-1] * np.eye(X_train[l].shape[0])
        K_inv_l = np.linalg.inv(K_l)
        
        # prediction of the residual at level l for our target points (X_target)
        delta_hat_l = np.zeros(n_points)
        for i, x in enumerate(X_target):
            k_vec = k_l_vector(x, X_train[l], thetas[l-1])
            # cross-covariance between X_target[i] and X_train[l]
            weights = np.squeeze(k_vec.T @ K_inv_l)
            delta_hat_l[i] = np.dot(weights, Delta_Y_train_l)
            
        # recursive update of the mean prediction
        if l == 1:
            f_hat_prev = delta_hat_l
        else:
            f_hat_prev = rhos[l-2] * f_hat_prev + delta_hat_l
            
    return f_hat_prev
# ----------------------------------------------------------------------
def log_likelihood_mf(rho_l_minus1, Theta_l, sigma_epsilon_l, X_l, Y_l, Y_l_minus_1, fidelity_level):
    """
    Eq.15 Log likelyhood implementation
    """
    # Residuals 
    n = X_l.shape[0]

    # avoid issues of type (5,1) vs (5,) when computing the log-likelihood
    Y_l_1d = np.squeeze(Y_l)
    if fidelity_level == 1:
        Delta_Y_l_val = Y_l_1d
    else:
        Y_l_minus_1_1d = np.squeeze(Y_l_minus_1)
        Delta_Y_l_val = Y_l_1d - rho_l_minus1 * Y_l_minus_1_1d
    
    # basic covariance matrix
    C_l = base_covariance_matrix(X_l, Theta_l)
    # noise
    K_l = C_l + sigma_epsilon_l * np.eye(len(X_l))
    
    # compute the standard likelyhood using colesky decomposition for numerical stability 
    try:
       
        L_chol = np.linalg.cholesky(K_l)
        log_det = 2.0 * np.sum(np.log(np.diag(L_chol)))
        
        # Solve for K_l * alpha = Delta_Y_l_val
        import scipy.linalg
        alpha = scipy.linalg.solve(L_chol.T, scipy.linalg.solve(L_chol, Delta_Y_l_val))
        
        Delta_Y_1d = np.squeeze(Delta_Y_l_val)
        alpha_1d = np.squeeze(alpha)

        data_fit = -0.5 * np.dot(Delta_Y_1d, alpha_1d)

        log_lik = data_fit - 0.5 * log_det - 0.5 * n * np.log(2 * np.pi)
        # -1/2 * Y^T * K^-1 * Y - 1/2 * log|K| - n/2 * log(2pi)
        # #log_lik = -0.5 * np.dot(Delta_Y_l_val, alpha) - 0.5 * log_det - 0.5 * n * np.log(2 * np.pi)
        # data_fit = -0.5 * (Delta_Y_l_val.T @ alpha)
        # log_lik = data_fit.item() - 0.5 * log_det - 0.5 * n * np.log(2 * np.pi)

        return log_lik
    except np.linalg.LinAlgError:
        logger.error("Matrix is ill-conditioned, returning penalty.")
        # penality if the matric is conditionned
        return -1e10
# ----------------------------------------------------------------------
def predict_non_nested_mf(x_new, thetas, rhos, noises, X_train, Y_train):
    """
    Recursively computes the mean and variance of the Multi-Fidelity model (NON-NESTED).
    """
    L = len(thetas)
    f_hat_prev = 0.0
    sigma2_hat_prev = 0.0
    gp_variances_at_x = []
    
    for l in range(1, L + 1):
        X_l = X_train[l]
        
        # 1. SÉCURITÉ ANTI-BROADCASTING : Forcer Y en 1D
        Y_l_1d = np.squeeze(Y_train[l])
        
        theta_l = thetas[l-1]
        noise_l = noises[l-1]
        
        if l == 1:
            target_Y = Y_l_1d
        else:
            rho_prev = rhos[l-2]
            Y_l_minus_1 = predict_mf_mean_up_to_level(
                X_target=X_l, 
                target_level=l-1, 
                thetas=thetas, 
                rhos=rhos, 
                noises=noises, 
                X_train=X_train, 
                Y_train=Y_train
            )
            # security: forces the preediction to be 1D
            Y_l_minus_1_1d = np.squeeze(Y_l_minus_1)
            target_Y = Y_l_1d - rho_prev * Y_l_minus_1_1d
            
        # Calling the real function to get the prediction at the new point
        delta_hat, sigma2_delta = predict_base_gp(x_new, X_l, target_Y, theta_l, noise_l)
        gp_variances_at_x.append(sigma2_delta)
        
        if l == 1:
            f_hat_current = delta_hat
            sigma2_hat_current = sigma2_delta
        else:
            rho_prev = rhos[l-2]
            f_hat_current = rho_prev * f_hat_prev + delta_hat
            sigma2_hat_current = (rho_prev ** 2) * sigma2_hat_prev + sigma2_delta
            
        f_hat_prev = f_hat_current
        sigma2_hat_prev = sigma2_hat_current
        
    return f_hat_prev, sigma2_hat_prev, gp_variances_at_x
# ----------------------------------------------------------------------


class GaussianProcess:
    def __init__(self, X_train, Y_train, kernel, noise_variance, L_chol):
        self.X_train = X_train
        self.Y_train = Y_train
        self.kernel = kernel
        self.noise_variance = noise_variance
        self.L_chol = L_chol #cholesky decomposition 

    def fit(self, X, Y):
        rhos = []
        thetas = []
        noises = []
        L = len(X) #not certain if this is correct but this should be right

        for l in range(1, L + 1):
            X_l = X[l]
            Y_l = Y[l]
            d = X_l.shape[1] 
            
            # -----------------------------------------------------------
            # NON-NESTED:
            # Instead of extracting exact spatial points from the lower level,
            # we use the lower-level GP to predict the mean at the current X_l.
            # -----------------------------------------------------------
            Y_l_minus_1_target = None 
            if l > 1:
                Y_l_minus_1_target = predict_mf_mean_up_to_level(
                    X_target=X_l, 
                    target_level=l-1, 
                    thetas=thetas, rhos=rhos, noises=noises, 
                    X_train=self.X_train, Y_train=self.Y_train
                )
            
            # Wrapper function to parameterize the log-likelihood for the scipy minimizer.
            # It returns the negative log-likelihood because scipy only minimizes.
            def objective_nll(params):
                if l == 1:
                    rho_l_minus1 = None
                    Theta_l = params[:-1] 
                else:
                    rho_l_minus1 = params[0]
                    Theta_l = params[1:-1]
                    
                sigma_epsilon_l = params[-1]
                ll = log_likelihood_mf(rho_l_minus1, Theta_l, sigma_epsilon_l, X_l, Y_l, Y_l_minus_1_target, l)
                return -ll
            
            best_ll = np.inf
            best_res = None
            
            # Multi-start training to avoid getting stuck in local minima (flat models)
            for attempt in range(3):
                if l == 1:
                    init_guess = np.concatenate((np.random.uniform(0.2, 1.5, d), [1.0, 1.0], [1e-4]))
                    param_bounds = [(0.01, 5.0)] * d + [(1e-3, 50.0)] * 2 + [(1e-8, 1e-5)] 
                else:
                    init_guess = np.concatenate(([np.random.uniform(0.5, 1.5)], np.random.uniform(0.2, 1.5, d), [1.0, 1.0], [1e-4]))
                    param_bounds = [(-5.0, 5.0)] + [(0.01, 5.0)] * d + [(1e-3, 50.0)] * 2 + [(1e-8, 1e-5)]
                
                # Minimize the negative log-likelihood to find optimal hyperparameters
                res = minimize(objective_nll, init_guess, bounds=param_bounds, method="L-BFGS-B")
                
                if res.fun < best_ll:
                    best_ll = res.fun
                    best_res = res
            
            # Store the best optimized hyperparamet3ers for the current level
            if l == 1:
                thetas.append(best_res.x[:-1]) 
                noises.append(best_res.x[-1])  
                logger.info(f"   GP Level {l} | Log-Likelihood: {-best_ll:.2f}")
            else:
                rhos.append(best_res.x[0])     
                thetas.append(best_res.x[1:-1])
                noises.append(best_res.x[-1])  
                logger.info(f"   GP Level {l} | Log-Likelihood: {-best_ll:.2f} | rho = {best_res.x[0]:.3f}")

    def predict(self, X_new):
        return predict_base_gp(X_new, self.X_train, self.Y_train, self.kernel.Theta, self.noise_variance)
        
    def _compute_log_likelyhood(self, theta_flat):
        try:
            return log_likelihood_mf(self.rho, theta_flat[:-1], theta_flat[-1], self.X_train, self.Y_train, self.Y_train_minus_1, self.fidelity_level)
        except np.linalg.LinAlgError:
            logger.error("Problem with the call to log_likelihood_mf.")
            return None

class MultifidelityModel:
    def __init__(self, L):
        self.L = L
        self.gps = [None] * L  # List to hold GaussianProcess instances for each fidelity level

    def fit(self, experiment_data):
        pass
    def predict(self, x_new):
        return predict_non_nested_mf(x_new, self.thetas, self.rhos, self.noises, self.X_train, self.Y_train)