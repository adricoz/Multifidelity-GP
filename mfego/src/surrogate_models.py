"""
Core Module, implementing the Gaussian Process and Multifidelity Model classes.
"""
import logging

import numpy as np
import scipy
from scipy.optimize import minimize
from src.data_management import ExperimentData
from src.kernels import Kernel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GaussianProcess:
    """
    Core of the Gaussian Process. This class handels
      the fitting and prediction of the GP model.
    """
    def __init__(self, kernel: Kernel):
        self.x_train = None
        self.y_train = None
        self.kernel = kernel
        self.noise = 1e-6
        self.l_chol = None
        self.k_inv = None

    def fit(self, x_train, y_train, n_restarts=3):
        """
        Fit the Gaussian Process model to the training data and optimize hyperparameters.
        """

        self.x_train = x_train
        self.y_train = np.squeeze(y_train)
        d = x_train.shape[1]

        def objective_nll(params):
            """
            Negative log-likelihood function to be minimized.
            """

            self.kernel.set_params(params[:-1])
            self.noise = params[-1]

            try:
                # Compute the covariance matrix K and its Cholesky decomposition
                covariance_matrix = self.kernel.get_covariance_matrix(self.x_train) \
                                           + self.noise * np.eye(len(self.x_train))
                l_chol = np.linalg.cholesky(covariance_matrix)
                alpha = scipy.linalg.solve(l_chol.T, scipy.linalg.solve(l_chol, self.y_train))
                log_det = 2.0 * np.sum(np.log(np.diag(l_chol)))
                data_fit = 0.5 * np.dot(self.y_train, alpha)

                # Eqs. (15), (16) from the reference article
                nll = data_fit + 0.5 * log_det + 0.5 * len(self.x_train) * np.log(2 * np.pi)
                #must be float for scipy
                return float(np.squeeze(nll))

            except np.linalg.LinAlgError:
                return 1e10

        best_nll = np.inf
        best_params = None

        # Define bounds for the hyperparameters
        param_bounds = [(0.01, 5.0)] * d + [(1e-3, 50.0), (1e-6, 1.0), (1e-8, 1e-5)]

        for _ in range(n_restarts):

            init_guess = np.concatenate((np.random.uniform(0.2, 1.5, d), [1.0, 1e-4, 1e-6]))
            res = minimize(objective_nll, init_guess, bounds=param_bounds, method='L-BFGS-B')
            if res.fun < best_nll:
                best_nll = res.fun
                best_params = res.x

        # Need to take care of the else case too...
        if best_params is not None:
            self.kernel.set_params(best_params[:-1])
            self.noise = best_params[-1]

        else:
            raise ValueError("No valid parameters found.")

        covariance_matrix = self.kernel.get_covariance_matrix(self.x_train) \
                                   + self.noise * np.eye(len(self.x_train))

        self.l_chol = np.linalg.cholesky(covariance_matrix)
        self.k_inv  = np.linalg.solve(self.l_chol.T,
                  np.linalg.solve(self.l_chol, np.eye(len(self.x_train)))
                                        )

    def predict(self, x_new: np.ndarray) -> tuple[float, float]:
        """
        Predicts the mean and variance of the Gaussian Process at new input points.
        """

        k_vec = self.kernel.get_cross_variance_vector(x_new, self.x_train)
        kappa = self.kernel.signal_variance + self.kernel.bias_variance

        f_hat = float(np.squeeze(k_vec.T @ self.k_inv @ self.y_train))
        sigma2_hat = float(np.squeeze(kappa +self.noise - (k_vec.T @ self.k_inv @ k_vec)))

        return f_hat, sigma2_hat

class MultifidelityModel:
    """
    Multifidelity Gaussian Process model that combines multiple fidelity levels.
    """
    def __init__(self, l, kernel_class: Kernel):
        self.num_levels = l
        # List to hold GaussianProcess instances for each fidelity level
        self.gps = [GaussianProcess(kernel_class()) for _ in range(l)]
        # Initialize correlation coefficients between levels
        self.rhos = [1.0 for _ in range(l - 1)]

    def fit(self, experiment_data: ExperimentData) -> None:
        """
        Fit one Gaussian process to each fidelity level in the data.
        """
        for l in range(1, self.num_levels + 1):
            x_l = experiment_data.x_dict[l]
            y_l = experiment_data.y_dict[l]

            if l == 1:
                target_y = y_l
            else:
                #NoN nested approach
                f_prev = np.array([self._predict_up_to(x, l-1)[0] for x in x_l])
                rho = 1.0
                self.rhos[l - 2] = rho
                target_y = y_l - rho * f_prev

            self.gps[l - 1].fit(x_l, target_y, n_restarts = 3)
            logger.info("GP level %s trained. Noise: %.6f", l, self.gps[l - 1].noise)

    def _predict_up_to(self, x_new: np.ndarray, level: int) -> tuple[float, float]:
        """
        Predict the mean and variance of the multifidelity model up to a specified fidelity level.
        """
        f_hat = 0.0
        sigma_2_hat = 0.0

        for l in range(1, level + 1):
            delta_f, delta_sigma2 = self.gps[l-1].predict(x_new)

            if l == 1:
                f_hat = delta_f
                sigma_2_hat = delta_sigma2
            else:
                rho = self.rhos[l - 2]
                f_hat = rho * f_hat + delta_f
                sigma_2_hat = (rho**2) * sigma_2_hat + delta_sigma2
        return f_hat, sigma_2_hat

    def predict(self, x_new):
        """
        Return the final prediction AND individual variances for the merit function
        """
        f_hat = 0.0
        sigma_2_hat = 0.0
        gp_variances = []

        for l in range(1, self.num_levels + 1):
            #GaussianProcess prediction for each level
            delta_f, delta_sigma2 = self.gps[l-1].predict(x_new)
            gp_variances.append(delta_sigma2)

            if l == 1:
                f_hat = delta_f
                sigma_2_hat = delta_sigma2
            else:
                rho = self.rhos[l - 2]
                f_hat = rho * f_hat + delta_f
                sigma_2_hat = (rho**2) * sigma_2_hat + delta_sigma2

        return f_hat, sigma_2_hat, gp_variances
