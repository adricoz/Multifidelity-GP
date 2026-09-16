
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
    def __init__(self, data, model, simulator, acquisition):
        self.data = data
        self.model = model
        self.simulator = simulator
        self.acquisition = acquisition

    def step(self):
        self.model.fit(self.data)
    def run(self, n_iterations):
        for iteration in range(n_iterations):
            logger.info(f"\n--- EGO Iteration {iteration + 1}/{n_iterations} ---")
            self.step()

    def _find_next_point(self):
        def objective_wrapper(x):
            merits = [self.acquisition.evaluate_merit(x, l) for l in range(1, self.model.L + 1)]
            best_merit = max(merits)
            return -best_merit  # We minimize the negative merit
        # differential evolution to find the next point
        result = differential_evolution(objective_wrapper, self.data.bounds)

        x_optimal = result.x
        l_optimal = np.argmax([self.acquisition.evaluate_merit(x_optimal, l) for l in range(1, self.model.L + 1)]) + 1
        return x_optimal, l_optimal