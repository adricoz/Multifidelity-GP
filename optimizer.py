
from venv import logger

import numpy as np
from scipy.optimize import differential_evolution
from scipy.stats import norm


# ----------------------------------------------------------------------
def expected_improvement(f_hat, sigma2_hat, f_best):
    """
    Compute classical EI

    Args
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
# ----------------------------------------------------------------------

def aei_multi_fidelity(f_hat_L, sigma2_hat_L, f_best_L, sigma2_e_L):
    """
    Compute the Augmented Expected Improvement (AEI) for multi-fidelity Gaussian Process.

    Args
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

# ----------------------------------------------------------------------
def merit_non_nested(x, l_candidate, L, costs, f_best_L, sigma2_e_L, rhos, noise_lp, gp_variances_at_x, f_hat_L, sigma2_hat_L):

    """Compute the merit function for a candidate point in a multi-fidelity Gaussian Process.

    Args
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
    aei_L = aei_multi_fidelity(f_hat_L, sigma2_hat_L, f_best_L, sigma2_e_L)
    # quick test for hartmann wuith the expected improvement
    #aei_L = expected_improvement(f_hat_L, sigma2_hat_L, f_best_L)

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

# ----------------------------------------------------------------------
# -----------######-#------######-######-######-######-######-----------
# -----------#------#------#----#-#------#------#------#----------------
# -----------#------#------######-######-######-######-######-----------
# -----------#------#------#----#------#------#-#-----------#-----------
# -----------######-######-#----#-######-######-######-######-----------
# ----------------------------------------------------------------------

class AcquisitionFunction:
    def __init__(self, model, costs):
        self.model = model
        self.costs = costs

    def _expected_improvement(self, f_hat, sigma2_hat, f_best):
        return expected_improvement(f_hat, sigma2_hat, f_best)
    
    def _augmented_expected_improvement(self, f_hat_L, sigma2_hat_L, f_best_L, sigma2_e_L):
        return aei_multi_fidelity(f_hat_L, sigma2_hat_L, f_best_L, sigma2_e_L)
    
    def evaluate_merit(self, x, candidate_level):
        return merit_non_nested(
            x=x,
            l_candidate=candidate_level,
            L=self.model.L,
            costs=self.costs,
            f_best_L=self.model.f_best_L,
            sigma2_e_L=self.model.sigma2_e_L,
            rhos=self.model.rhos,
            noise_lp=self.model.noise_lp,
            gp_variances_at_x=self.model.gp_variances_at_x,
            f_hat_L=self.model.f_hat_L,
            sigma2_hat_L=self.model.sigma2_hat_L
        )

class EGOOptimizer:
    def __init__(self, data, model, simulator, AcquisitionFunction):
        self.data = data
        self.model = model
        self.simulator = simulator
        self.AcquisitionFunction = AcquisitionFunction

    def step(self):
        pass
    def run(self, n_iterations):
        pass
    def _find_next_point(self):
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
                    logger.error("\nHidden scipy error!")
                    traceback.print_exc()
                    raise e
            
            # Global Optimization of the merit function using Differential Evolution
            res_merit = differential_evolution(
                objective_merit, 
                bounds=bounds, 
                popsize=10, 
                maxiter=50, #hard coded but could be changed
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
            logger.warning("   Warning: Global merit is null/too low. Random selection triggered (exploration).")
            next_x = np.array([np.random.uniform(b[0], b[1]) for b in bounds])
            next_l = L 

        logger.info(f"-> Next point: x = {np.round(next_x, 4)} | Fidelity = {next_l} | Merit = {best_merit_overall:.5f}")