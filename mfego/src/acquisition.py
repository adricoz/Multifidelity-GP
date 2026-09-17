"""
This module essnetialy implements the merit function for the EGO algo.
"""
import numpy as np  # noqa: I001
from scipy.stats import norm
from data_management import ExperimentData
from surrogate_models import MultifidelityModel


class AcquisitionFunction:
    """
    Class for the acquisition/merit function Eq.20 and 24 of the reference article.
    """
    def __init__(self, model: type[MultifidelityModel], data: type[ExperimentData]):
        self.model = model
        self.data = data

    def evaluate_merit(self, x: np.ndarray, candidate_level: int) -> float:
        """
        Evaluate the merit function at a given point and fidelity level.
        """
        f_hat_l, sigma2_hat_l, gp_variances = self.model.predict(x)

        # Best observed value at the highest fidelity level
        f_best_l = np.min(self.data.Y_dict[self.model.L])

        sigma_l = np.sqrt(max(sigma2_hat_l, 1e-12))
        if sigma_l > 0:
            u = (f_best_l - f_hat_l) / sigma_l
            ei = sigma_l * (u * norm.cdf(u) + norm.pdf(u))
        else:
            ei = 0.0
        if ei <= 0:
            return 0.0

        # Essentially compute cost and infromation ratios to return AEI
        cost_ratio = self.data.costs[self.model.L - 1] / self.data.costs[candidate_level - 1]
        var_gp_lp = gp_variances[candidate_level - 1]
        noise_lp = self.model.gps[candidate_level - 1].noise
        delta_sigma2_lp = (var_gp_lp ** 2) / (var_gp_lp + noise_lp)

        r2_lp = 1.0
        if candidate_level < self.model.L:
            for i in range(candidate_level, self.model.L):
                r2_lp *= (self.model.rhos[i-1]**2)

        information_ratio = max(0.0, (r2_lp * delta_sigma2_lp) / max(sigma2_hat_l, 1e-12))
        #AEI = EI * cost_ratio * information_ratio
        return float(ei * cost_ratio * information_ratio)
