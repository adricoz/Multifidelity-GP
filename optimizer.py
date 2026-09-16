
import logging
logger = logging.getLogger(__name__)

import numpy as np
from scipy.optimize import differential_evolution
from scipy.stats import norm

# ----------------------------------------------------------------------
# -----------######-#------######-######-######-######-######-----------
# -----------#------#------#----#-#------#------#------#----------------
# -----------#------#------######-######-######-######-######-----------
# -----------#------#------#----#------#------#-#-----------#-----------
# -----------######-######-#----#-######-######-######-######-----------
# ----------------------------------------------------------------------

class AcquisitionFunction:
    def __init__(self, model, data):
        self.model = model
        self.data = data

    def evaluate_merit(self, x, candidate_level):
        f_hat_L, sigma2_hat_L, gp_variances = self.model.predict(x)

        f_best_L = np.min(self.data.Y_dict[self.model.L])  # Best observed value at the highest fidelity level

        sigma_L = np.sqrt(max(sigma2_hat_L, 1e-12))
        if sigma_L > 0:
            u = (f_best_L - f_hat_L) / sigma_L
            ei = sigma_L * (u * norm.cdf(u) + norm.pdf(u))
        else:
            ei = 0.0
        if ei <= 0:
            return 0.0

        cost_ratio = self.data.costs[self.model.L - 1] / self.data.costs[candidate_level - 1]
        var_gp_lp = gp_variances[candidate_level - 1]
        noise_lp = self.model.gps[candidate_level - 1].noise
        delta_sigma2_lp = (var_gp_lp ** 2) / (var_gp_lp + noise_lp)

        R2_lp = 1.0
        if candidate_level < self.model.L:
            for i in range(candidate_level, self.model.L):
                R2_lp *= (self.model.rhos[i-1]**2)

        information_ratio = max(0.0, (R2_lp * delta_sigma2_lp) / max(sigma2_hat_L, 1e-12))
        return float(ei * cost_ratio * information_ratio)

#-----------------------------------------------------------------------
class EGOOptimizer:
    def __init__(self, data, model, simulator, acquisition):
        self.data = data
        self.model = model
        self.simulator = simulator
        self.acquisition = acquisition

    def _find_next_point(self):
            def objective_wrapper(x):
                merits = [self.acquisition.evaluate_merit(x, l) for l in range(1, self.model.L + 1)]
                best_merit = max(merits)
                return -best_merit  # We minimize the negative merit
            # differential evolution to find the next point
            result = differential_evolution(objective_wrapper, self.data.bounds, popsize=10, maxiter=50, updating="deferred")
    
            x_optimal = result.x
            l_optimal = np.argmax([self.acquisition.evaluate_merit(x_optimal, l) for l in range(1, self.model.L + 1)]) + 1
            return x_optimal, l_optimal, -result.fun  # Return the merit value as well
    
    def run(self, n_iterations):
        for iteration in range(n_iterations):
            logger.info(f"\n--- EGO Iteration {iteration + 1}/{n_iterations} ---")
            #train the model
            self.model.fit(self.data)

            # search point
            x_next, l_next, merit = self._find_next_point()

            #Failsafe
            if merit <= 0.0:
                logger.warning("Warning: Merit is 0. Random selection triggered.")
                x_next = np.array([np.random.uniform(b[0], b[1]) for b in self.data.bounds])
                l_next = self.model.L

            logger.info(f"Next selected point to evaluate: {np.round(x_next, 4)} | Level: {l_next} | Merit: {merit:.6f}")

            if not self.data.is_already_evaluated(l_next, x_next):
                y_new = self.simulator.evaluate(x_next, l_next)
                self.data.add_observation(l_next, x_next, y_new)
                logger.info(f"    -> Evaluated value: {y_new:.6f} at level {l_next}")

            else:
                logger.warning(f"Point {np.round(x_next, 4)} at level {l_next} has already been evaluated. Skipping evaluation.")
