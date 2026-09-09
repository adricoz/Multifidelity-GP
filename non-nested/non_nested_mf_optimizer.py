import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize, differential_evolution

# Local imports
from Hartmann6d import f_l
from non_nested_mf_sampling import Delta_Y_l, extract_subpart_vector, is_already_evaluated
from non_nested_mf_covariance import base_covariance_matrix, Cov_fct, k_l_vector

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
        
        # Résolution de K_l * alpha = Delta_Y_l_val
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
        # penality if the matric is conditionned
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
            # 2. SÉCURITÉ : Forcer la prédiction en 1D
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
    
  
    cost_ratio = costs[L-1] / costs[l_candidate-1]
    

    idx = l_candidate - 1 
    var_gp_lp = gp_variances_at_x[idx] 
    
    delta_sigma2_lp = (var_gp_lp ** 2) / (var_gp_lp + noise_lp)
    
    R2_lp = 1.0
    if l_candidate < L:
        for i in range(l_candidate, L):
            R2_lp *= (rhos[i-1] ** 2)
            
    variance_reduction = R2_lp * delta_sigma2_lp
    information_ratio = max(0.0, variance_reduction / max(sigma2_hat_L, 1e-12))
    
    # Equation (24) in the reference paper, combining AEI, cost ratio, and information ratio
    merit = aei_L * cost_ratio * information_ratio
    return merit

#----------------------------------------------------
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
#----------------------------------------------------

def run_non_nested_mf_ego(X_train, Y_train, L, costs, bounds, n_iterations, true_function):
    """
    Main loop for the Non-Nested Multi-Fidelity Efficient Global Optimization (MF-EGO).
    """
    for iteration in range(n_iterations):
        print(f"\n--- EGO Iteration {iteration + 1}/{n_iterations} ---")
        
        # ==========================================
        # STEP 1: MODEL TRAINING (From level 1 to L)
        # ==========================================
        rhos = []
        thetas = []
        noises = []
        
        for l in range(1, L + 1):
            X_l = X_train[l]
            Y_l = Y_train[l]
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
                    X_train=X_train, Y_train=Y_train
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
            
            # Store the best optimized hyperparameters for the current level
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
        
        # Evaluate the merit landscape for each potential fidelity level candidate
        for l_candidate in range(1, L + 1):
            
            # Wrapper function for the acquisition optimization (returns negative merit)
            def objective_merit(x):
                try: # we try to predict the error
                    # Predict final high-fidelity statistics at candidate point x
                    f_hat_L, sigma2_hat_L, gp_variances_at_x = predict_non_nested_mf(x, thetas, rhos, noises, X_train, Y_train)
                    noise_lp = noises[l_candidate - 1] 
                    
                    f_hat_L = float(np.squeeze(f_hat_L))
                    sigma2_hat_L = float(np.squeeze(sigma2_hat_L))
                    gp_variances_at_x = [float(np.squeeze(v)) for v in gp_variances_at_x]
                    noise_lp = float(noises[l_candidate - 1])
                    
                    # Compute the non-nested merit value
                    merit_val = merit_non_nested(
                        x, l_candidate, L, costs, f_best_L, sigma2_e_L, rhos, 
                        noise_lp, gp_variances_at_x, f_hat_L, sigma2_hat_L
                    )
                    
                    # Ensure the returned value is a standard scalar
                    if hasattr(merit_val, 'item'):
                        merit_val = merit_val.item()
                        
                    return -float(np.squeeze(merit_val))
            
                except Exception as e:
                    # showing the real error 
                    import traceback
                    print("\nHidden scipy error!")
                    traceback.print_exc()
                    raise e
            
            # Global Optimization of the merit function using Differential Evolution
            res_merit = differential_evolution(
                objective_merit, 
                bounds=bounds, 
                popsize=10, 
                maxiter=50,
                tol=1e-3,
                updating='deferred'
            )
            
            merit_value = -res_merit.fun
            
            # Keep track of the absolute best merit across all fidelity levels
            if merit_value > best_merit_overall:
                best_merit_overall = merit_value
                next_x = res_merit.x
                next_l = l_candidate

        # Failsafe: Trigger random exploration if the merit landscape is completely flat.
        # A merit of 0.0 indicates that no significant variance reduction or expected improvement was found.
        if next_x is None or next_l is None or best_merit_overall <= 0.0:
            print("   Warning: Global merit is null/too low. Random selection triggered (exploration).")
            next_x = np.array([np.random.uniform(b[0], b[1]) for b in bounds])
            next_l = L 

        print(f"-> Next point: x = {np.round(next_x, 4)} | Fidelity = {next_l} | Merit = {best_merit_overall:.5f}")

        # ==========================================
        # STEP 3: EVALUATION AND DATASET UPDATE
        # ==========================================
        # -----------------------------------------------------------
        # CRITICAL NON-NESTED CHANGE:
        # We ONLY evaluate the selected level (next_l). 
        # -----------------------------------------------------------
        l_eval = next_l 
        
        if not is_already_evaluated(next_x, X_train[l_eval]):
            # Evaluate the true black-box function at the chosen fidelity
            new_y = true_function(next_x, l_eval)
            
            # Append the new observation to the specific fidelity dataset
            X_train[l_eval] = np.vstack((X_train[l_eval], next_x))
            Y_train[l_eval] = np.append(Y_train[l_eval], new_y)
        else:
            print(f"   Warning: Point already evaluated at level {l_eval}. Skipping to avoid duplicate.")
                
    return X_train, Y_train, thetas, rhos, noises