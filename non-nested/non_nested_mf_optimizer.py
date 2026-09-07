import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize, differential_evolution

# Local imports
from Hartmann6d import f_l
from non_nested_mf_sampling import Delta_Y_l, extract_subpart_vector, is_already_evaluated
from non_nested_mf_covariance import base_covariance_matrix, Cov_fct

def log_likelihood_mf(rho_l_minus1, Theta_l, sigma_epsilon_l, X_l, Y_l, Y_l_minus_1, fidelity_level):
    """
    Eq.15 Log likelyhood implementation
    """
    # Residuals 
    Delta_Y_l_val = Delta_Y_l(Y_l, Y_l_minus_1, rho_l_minus1)
    
    # basic covariance matrix
    C_l = base_covariance_matrix(X_l, Theta_l)
    
    # noise
    K_l = C_l + sigma_epsilon_l * np.eye(len(X_l))
    
    # compute the standard likelyhood using colesky decomposition for numerical stability 
    try:
       
        L_chol = np.linalg.cholesky(K_l)
        log_det = 2.0 * np.sum(np.log(np.diag(L_chol)))
        
        # Résolution de K_l * alpha = Delta_Y_l_val
        alpha = np.linalg.solve(L_chol.T, np.linalg.solve(L_chol, Delta_Y_l_val))
        
        # -1/2 * Y^T * K^-1 * Y - 1/2 * log|K| - n/2 * log(2pi)
        n = len(X_l)
        log_lik = -0.5 * np.dot(Delta_Y_l_val, alpha) - 0.5 * log_det - 0.5 * n * np.log(2 * np.pi)
        return log_lik
    except np.linalg.LinAlgError:
        # prnslity if the matric is conditionned
        return -1e10

# -----------------------------------------------------------------------------------------

# Note : using np.linalg.solve() is mathematically equivalent to 
# np.linalg.inv() but more stable and efficient
def predict_base_gp(x_new, X_train, Y_train, Theta_l, sigma_epsilon_l):
    """
    Simply predicts the average and variance given by Eqs. 11 and 12 in the paper.
    Parameters
    ----------
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
    -------
    delta_hat_scalar : float
        Predicted mean at the new point.
    sigma2_delta_scalar : float
        Predicted variance at the new point.
    """
    C = base_covariance_matrix(X_train, Theta_l)
    K = C + sigma_epsilon_l * np.eye(len(X_train))
    K_inv = np.linalg.inv(K)

    d = X_train.shape[1]
    lengthscales = Theta_l[0:d]
    t1 = Theta_l[d] if len(Theta_l) > d else 1.0
    t2 = Theta_l[d+1] if len(Theta_l) > d+1 else 1.0

    k_vec = np.zeros(len(X_train))
    for i in range(len(X_train)):
        k_vec[i] = Cov_fct(x_new, X_train[i], lengthscales, t1, t2)

    kappa = Cov_fct(x_new, x_new, lengthscales, t1, t2)
    delta_hat = k_vec.T @ K_inv @ Y_train
    sigma2_delta = kappa - (k_vec.T @ K_inv @ k_vec)
    
    delta_hat_scalar = delta_hat.item() if hasattr(delta_hat, 'item') else delta_hat
    sigma2_delta_scalar = max(sigma2_delta.item() if hasattr(sigma2_delta, 'item') else sigma2_delta, 1e-12)

    return delta_hat_scalar, sigma2_delta_scalar

def predict_non_nested_mf(x_new, thetas, rhos, noises, X_train, Y_train):
    """
    Recursively computes the mean and variance of the Multi-Fidelity model.

    Parameters
    ----------
    x_new : array-like, shape (d,)
        New point where the prediction is desired.
    thetas : list of array-like
        List of hyperparameters for each fidelity level.
    rhos : list of floats
        List of correlation coefficients between fidelity levels.   
    noises : list of floats
        List of noise variances for each fidelity level.
    X_train : dict
        Dictionary containing training input points for each fidelity level.
    Y_train : dict
        Dictionary containing training output values for each fidelity level.   

    Returns
    -------
    f_hat_prev : float
        Predicted mean at the new point for the highest fidelity level.
    sigma2_hat_prev : float
        Predicted variance at the new point for the highest fidelity level.
    gp_variances_at_x : list of floats
        List of predicted variances at the new point for each fidelity level.
    
    """
    L = len(thetas)
    f_hat_prev = 0.0
    sigma2_hat_prev = 0.0
    gp_variances_at_x = []
    
    for l in range(1, L + 1):
        X_l = X_train[l]
        Y_l = Y_train[l]
        theta_l = thetas[l-1]
        noise_l = noises[l-1]
        
        if l == 1:
            target_Y = Y_l
        else:
            rho_prev = rhos[l-2]
            Y_l_minus_1 = extract_subpart_vector(X_l, X_train[l-1], Y_train[l-1])
            target_Y = Y_l - rho_prev * Y_l_minus_1
            
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

def expected_improvement(f_hat, sigma2_hat, f_best):
    """
    Compute classical EI

    Parameters
    ----------
    f_hat : float   
        Predicted mean at the new point.
    sigma2_hat : float  
        Predicted variance at the new point.
    f_best : float
        Best observed value.    

    Returns
    -------
    float   
        Expected Improvement (EI) value.    
    """
    sigma_hat = np.sqrt(np.maximum(sigma2_hat, 1e-10))
    
    u = (f_best - f_hat) / sigma_hat 
    ei = sigma_hat * (u * norm.cdf(u) + norm.pdf(u))
    return ei
# -----------------------------------------------------------------------------------------

def aei_multi_fidelity(f_hat_L, sigma2_hat_L, f_best_L, sigma2_e_L):
    """
    Compute the Augmented Expected Improvement (AEI) for multi-fidelity Gaussian Process.

    Parameters
    ----------
    f_hat_L : float 
        Predicted mean at fidelity level L.
    sigma2_hat_L : float    
        Predicted variance at fidelity level L.
    f_best_L : float
        Best observed value at fidelity level L.
    sigma2_e_L : float
        Noise variance at fidelity level L. 

    Returns
    -------
    float   
        Augmented Expected Improvement (AEI) value.
    """
    sigma_e_L = np.sqrt(max(sigma2_e_L, 1e-12))
    sigma2_hat_L = np.maximum(sigma2_hat_L, 1e-10)
    
    # EI
    ei = expected_improvement(f_hat_L, sigma2_hat_L, f_best_L)
    
    # AEI : EI penalized by the uncertainty of the model
    denominator = np.sqrt(sigma2_hat_L + sigma2_e_L)
    penalty = 0.0 if denominator == 0 else 1.0 - (sigma_e_L / denominator)
    penalty = np.clip(penalty, 0.0, 1.0)
    
    return (ei * penalty).item() if hasattr(ei * penalty, 'item') else (ei * penalty)
# ------------------------------------------------------------------------------------------

def merit_non_nested(x, l_candidate, L, costs, f_best_L, sigma2_e_L, rhos, noise_lp, gp_variances_at_x, f_hat_L, sigma2_hat_L):

    """
    Compute the merit function for a candidate point in a multi-fidelity Gaussian Process.

    Parameters
    ----------
    x : array-like  
        Candidate point in the input space.
    l_candidate : int
        Fidelity level of the candidate point.
    L : int
        Highest fidelity level.
    costs : list of floats
        Costs associated with each fidelity level.
    f_best_L : float
        Best observed value at fidelity level L.
    sigma2_e_L : float  
        Noise variance at fidelity level L.
    rhos : list of floats   
        Correlation coefficients between fidelity levels.  
    noise_lp : float
        Noise variance at fidelity level lp.
    gp_variances_at_x : list of floats  
        Predicted variances at the candidate point for each fidelity level.
    f_hat_L : float
        Predicted mean at fidelity level L. 
    sigma2_hat_L : float
        Predicted variance at fidelity level L. 

    Returns
    ------- 
    float
        Merit value for the candidate point.    
    """
    #aei_L = aei_multi_fidelity(f_hat_L, sigma2_hat_L, f_best_L, sigma2_e_L)
    # quick test for hartmann wuith the expected improvement
    aei_L = expected_improvement(f_hat_L, sigma2_hat_L, f_best_L)

    if aei_L <= 0:
        return 0.0 
    
  
    cost_total_L = np.sum(costs) 
    cost_total_l_candidate = np.sum(costs[:l_candidate])
    cost_ratio = cost_total_L / cost_total_l_candidate
    
    
    variance_reduction_sum = 0.0
    
    for lp in range(1, l_candidate + 1):
        idx = lp - 1 
        
        var_gp_lp = gp_variances_at_x[idx] 
        

        delta_sigma2_lp = (var_gp_lp ** 2) / (var_gp_lp + noise_lp)
        
    
        R2_lp = 1.0
        if lp < L:
            for i in range(lp, L):
                R2_lp *= (rhos[i-1] ** 2)
                
        
        variance_reduction_sum += R2_lp * delta_sigma2_lp
        
    
    information_ratio = max(0.0, variance_reduction_sum / max(sigma2_hat_L, 1e-12))
    
    
    merit = aei_L * cost_ratio * information_ratio
    return merit


def run_non_nested_mf_ego(X_train, Y_train, L, costs, bounds, n_iterations, true_function):
    """
    Main loop for the Nested Multi-Fidelity Efficient Global Optimization.
    """
    for iteration in range(n_iterations):
        print(f"\n--- EGO Iteration {iteration + 1}/{n_iterations} ---")
        
        # ==========================================
        # STEP 1: MODEL TRAINING (From 1 to L)
        # ==========================================
        rhos = []
        thetas = []
        noises = []
        
        for l in range(1, L + 1):
            X_l = X_train[l]
            Y_l = Y_train[l]
            d = X_l.shape[1] 
            
            Y_l_minus_1 = None 
            if l > 1:
                Y_l_minus_1 = extract_subpart_vector(X_l, X_train[l-1], Y_train[l-1])
            
            # a quick function to properly parametrize the log_likelyhood and return the negative for minimization
            def objective_nll(params):
                if l == 1:
                    rho_l_minus1 = None
                    Theta_l = params[:-1] 
                else:
                    rho_l_minus1 = params[0]
                    Theta_l = params[1:-1]
                    
                sigma_epsilon_l = params[-1]
                ll = log_likelihood_mf(rho_l_minus1, Theta_l, sigma_epsilon_l, X_l, Y_l, Y_l_minus_1, l)
                return -ll
            
            best_ll = np.inf
            best_res = None
            
            # Multi-start training to avoid flat models
            for attempt in range(3):
                if l == 1:
                    init_guess = np.concatenate((np.random.uniform(0.2, 1.5, d), [1.0, 1.0], [1e-4]))
                    param_bounds = [(0.01, 5.0)] * d + [(1e-3, 50.0)] * 2 + [(1e-8, 1e-5)] 
                else:
                    init_guess = np.concatenate(([np.random.uniform(0.5, 1.5)], np.random.uniform(0.2, 1.5, d), [1.0, 1.0], [1e-4]))
                    param_bounds = [(-5.0, 5.0)] + [(0.01, 5.0)] * d + [(1e-3, 50.0)] * 2 + [(1e-8, 1e-5)]
                
                # here we minimize the log likelyhood
                res = minimize(objective_nll, init_guess, bounds=param_bounds, method="L-BFGS-B")
                
                if res.fun < best_ll:
                    best_ll = res.fun
                    best_res = res
            
            if l == 1:
                thetas.append(best_res.x[:-1]) 
                noises.append(best_res.x[-1])  
                print(f"   GP Level {l} | Log-Likelihood: {-best_ll:.2f}")
            else:
                rhos.append(best_res.x[0])     
                thetas.append(best_res.x[1:-1])
                noises.append(best_res.x[-1])  
                print(f"   GP Level {l} | Log-Likelihood: {-best_ll:.2f} | rho = {best_res.x[0]:.3f}")

        # ==========================================
        # STEP 2: SEARCH FOR THE NEXT POINT
        # ==========================================
        f_best_L = np.min(Y_train[L]) 
        sigma2_e_L = noises[-1]       
        
        best_merit_overall = -np.inf
        next_x = None
        next_l = None
        
        for l_candidate in range(1, L + 1):
            
            # once again a function to properly parametrize the merit function for optimization and return the negative for minimization
            def objective_merit(x):
                f_hat_L, sigma2_hat_L, gp_variances_at_x = predict_non_nested_mf(x, thetas, rhos, noises, X_train, Y_train)
                noise_lp = noises[l_candidate - 1] 
                
                merit_val = merit_non_nested(
                    x, l_candidate, L, costs, f_best_L, sigma2_e_L, rhos, 
                    noise_lp, gp_variances_at_x, f_hat_L, sigma2_hat_L
                )
                
                # Ensure the value is a scalar and not an error
                if hasattr(merit_val, 'item'):
                    merit_val = merit_val.item()
                    
                return -merit_val 
            
            # Global Optimization using Differential Evolution
            res_merit = differential_evolution(
                objective_merit, 
                bounds=bounds, 
                popsize=10, 
                maxiter=50,
                tol=1e-3,
                updating='deferred'
            )
            
            merit_value = -res_merit.fun
            
            if merit_value > best_merit_overall:
                best_merit_overall = merit_value
                next_x = res_merit.x
                next_l = l_candidate

        # Failsafe in case of extremely flat merit landscape.        be aware that 0.0 is a special case 
        if next_x is None or next_l is None or best_merit_overall <= 0.0:
            print("   Warning: Global merit is null/too low. Random selection triggered (exploration).")
            next_x = np.array([np.random.uniform(b[0], b[1]) for b in bounds])
            next_l = L 

        print(f"-> Next point: x = {np.round(next_x, 4)} | Fidelity = {next_l} | Merit = {best_merit_overall:.5f}")

        # ==========================================
        # STEP 3: EVALUATION AND NESTED UPDATE
        # ==========================================
        for l_eval in range(1, next_l + 1):
            if not is_already_evaluated(next_x, X_train[l_eval]):
                new_y = true_function(next_x, l_eval)
                X_train[l_eval] = np.vstack((X_train[l_eval], next_x))
                Y_train[l_eval] = np.append(Y_train[l_eval], new_y)
                
    return X_train, Y_train, thetas, rhos, noises