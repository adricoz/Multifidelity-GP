"""
This module essentially implements the merit function for the EGO algorithm.
"""
import numpy as np  # noqa: I001
from scipy.stats import norm
from src.data_management import ExperimentData
from src.surrogate_models import MultifidelityModel


class AcquisitionFunction:
    """
    Class for the acquisition/merit function Eq.20 and 24 of the reference article.
    """
    def __init__(self, model: MultifidelityModel, data: ExperimentData, feasibility=None):
        self.model = model
        self.data = data
        # [KC] optional known-constraint function: feasibility(x) with x of shape (m, d) returns
        # a boolean array (m,), True where the point satisfies the cheap a priori constraints
        # (e.g. geometric constraints of a section). Infeasible points get a zero merit, so the
        # search of the next point never proposes them (None: every point is feasible).
        self.feasibility = feasibility
        # [FIX-T3] effective best solution (Eq. 19), updated after each fit by update()
        self.x_best = None
        self.f_best = None

    def update(self) -> None:
        """
        [FIX-T3] Computes the "effective best solution" of Eq. 19 (Huang et al. 2006, ref. [25]
        of the article): x_best = argmin over the training points of ALL the levels of
        f_hat_L(x) + sigma_hat_L(x), and f_best = f_hat_L(x_best).
        Must be called after each fit of the model (done by EGOOptimizer.ask()).
        """
        x_all = np.vstack([self.data.get_training_data(l)[0]
                           for l in range(1, self.model.num_levels + 1)])
        f_hat, sigma2_hat, _ = self.model.predict_batch(x_all)
        i_best = int(np.argmin(f_hat + np.sqrt(np.maximum(sigma2_hat, 0.0))))
        self.x_best = x_all[i_best]
        self.f_best = float(f_hat[i_best])

    def evaluate_merits_batch(self, x: np.ndarray) -> np.ndarray:
        """
        [FIX-N6] Merit function (Eq. 24) of m points for ALL the candidate levels with a
        single (vectorized) prediction of the model (was one prediction per level and point).

        Args:
        - x: numpy array of shape (m, d).
        Returns:
        - merits: numpy array of shape (m, L), merits[i, l-1] = M_NN-MF(x_i, l).
        """
        if self.f_best is None:
            self.update()
        num_levels = self.model.num_levels
        f_hat_l, sigma2_hat_l, gp_variances = self.model.predict_batch(np.atleast_2d(x))

        # Best observed value at the highest fidelity level
        # [FIX-T3] replaced by the effective best solution f_hat_L(x_best) of Eq. 19
        f_best_l = self.f_best

        sigma_l = np.sqrt(np.maximum(sigma2_hat_l, 1e-12))
        u = (f_best_l - f_hat_l) / sigma_l
        ei = np.maximum(sigma_l * (u * norm.cdf(u) + norm.pdf(u)), 0.0)

        # [FIX-T2] Augmented EI (Eq. 20): penalization 1 - sigma_eps / sqrt(sigma^2 + sigma_eps^2)
        noise_top = self.model.gps[num_levels - 1].get_noise_variance()
        aei = ei * np.clip(1.0 - np.sqrt(noise_top) / np.sqrt(sigma2_hat_l + noise_top),
                           0.0, 1.0)

        merits = np.zeros((len(f_hat_l), num_levels))
        for candidate_level in range(1, num_levels + 1):
            # Essentially compute cost and infromation ratios to return AEI
            cost_ratio = self.data.costs[num_levels - 1] / self.data.costs[candidate_level - 1]
            noise_lp = self.model.gps[candidate_level - 1].get_noise_variance()
            # [FIX-T4] variance reduction at x of an observation at x (Eqs. 22, 28-29):
            # s^4 / (s^2 + noise) with the LATENT variance s^2 (the GP variance includes noise)
            s2_lp = np.maximum(gp_variances[:, candidate_level - 1] - noise_lp, 0.0)
            denominator = s2_lp + noise_lp
            delta_sigma2_lp = np.divide(s2_lp ** 2, denominator,
                                        out=np.zeros_like(s2_lp), where=denominator > 0)

            r2_lp = 1.0
            if candidate_level < num_levels:
                for i in range(candidate_level, num_levels):
                    r2_lp *= (self.model.rhos[i-1]**2)

            information_ratio = np.maximum(
                0.0, (r2_lp * delta_sigma2_lp) / np.maximum(sigma2_hat_l, 1e-12))
            #AEI = EI * cost_ratio * information_ratio
            merits[:, candidate_level - 1] = aei * cost_ratio * information_ratio
        if self.feasibility is not None:
            # [KC] known constraints: zero merit outside the feasible domain
            feasible = np.asarray(self.feasibility(np.atleast_2d(x)), dtype=bool).reshape(-1)
            merits[~feasible, :] = 0.0
        return merits

    def evaluate_merits(self, x: np.ndarray) -> list[float]:
        """
        [FIX-N6] Merits of a single point for all the candidate levels (one prediction).
        """
        return self.evaluate_merits_batch(np.atleast_2d(x))[0].tolist()

    def evaluate_merit(self, x: np.ndarray, candidate_level: int) -> float:
        """
        Evaluate the merit function at a given point and fidelity level.
        (Kept for compatibility: wrapper of evaluate_merits_batch.)
        """
        return float(self.evaluate_merits_batch(np.atleast_2d(x))[0, candidate_level - 1])
