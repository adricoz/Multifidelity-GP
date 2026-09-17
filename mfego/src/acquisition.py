import numpy as np
from scipy.stats import norm


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