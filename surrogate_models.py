import logging

import scipy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import numpy as np
from scipy.optimize import minimize


class GaussianProcess:
    def __init__(self, kernel):
        self.X_train = None
        self.Y_train = None
        self.kernel = kernel
        self.noise = 1e-6
        self.L_chol = None
        self.K_inv = None

    def fit(self, X, Y, n_restarts = 3):
        self.X_train = X
        self.Y_train = Y.reshape(-1, 1) 
        d = X.shape[1]
        def objective_nll(params):
            self.kernel.set_params(params[:-1])
            self.noise = params[-1]

            try:
                K = self.kernel.get_covariance_matrix(self.X_train) + self.noise * np.eye(len(self.X_train))
                L = np.linalg.cholesky(K)
                alpha = scipy.linalg.solve(L.T, scipy.linalg.solve(L, self.Y_train))
                print("We got here")
                log_det = 2.0 * np.sum(np.log(np.diag(L)))
                data_fit = 0.5 * np.dot(self.Y_train, alpha)
                nll = data_fit + 0.5 * log_det + 0.5 * len(self.X_train) * np.log(2 * np.pi)
                return float(nll) #must be float for scipy
            except np.linalg.LinAlgError:
                return 1e10
        best_nll = np.inf
        best_params = None

        param_bounds = [(0.01, 5.0)] * d + [(1e-3, 50.0), (1e-6, 1.0), (1e-8, 1e-5)]

        for _ in range(n_restarts):

            init_guess = np.concatenate((np.random.uniform(0.2, 1.5, d), [1.0, 1e-4, 1e-6]))
            res = minimize(objective_nll, init_guess, bounds=param_bounds, method='L-BFGS-B')
            if res.fun < best_nll:
                best_nll = res.fun
                best_params = res.x

        if best_params is not None:
            self.kernel.set_params(best_params[:-1])
            self.noise = best_params[-1]
        K = self.kernel.get_covariance_matrix(self.X_train) + self.noise * np.eye(len(self.X_train))
        self.L_chol = np.linalg.cholesky(K)
        self.K_inv = np.linalg.solve(self.L_chol.T, np.linalg.solve(self.L_chol, np.eye(len(self.X_train)))
                                        )
    def predict(self, X_new):
        k_vec = self.kernel.get_cross_covariance_vector(X_new, self.X_train)
        kappa = self.kernel.signal_variance + self.kernel.bias_variance

        f_hat = float(k_vec.T @ self.K_inv @ self.Y_train)
        sigma2_hat = float(kappa +self.noise - (k_vec.T @ self.K_inv @ k_vec))

        return f_hat, sigma2_hat

class MultifidelityModel:
    def __init__(self, L, kernel_class):
        self.L = L
        self.gps = [GaussianProcess(kernel_class()) for _ in range(L)]  # List to hold GaussianProcess instances for each fidelity level
        self.rhos = [1.0 for _ in range(L - 1)]  # Initialize correlation coefficients between levels

    def fit(self, experiment_data):
        for l in range(1, self.L + 1):
            X_l = experiment_data.X_dict[l]
            Y_l = experiment_data.Y_dict[l]

            if l == 1:
                target_Y = Y_l
            else:
                #NoN nested approach
                f_prev = np.array([self._predict_up_to(x, l-1)[0] for x in X_l])
                rho = 1.0
                self.rhos[l - 2] = rho
                target_Y = Y_l - rho * f_prev
            #self.gps[l - 1].fit(X_l, target_Y, bounds = experiment_data.bounds, n_restarts = 3)
            self.gps[l - 1].fit(X_l, target_Y, n_restarts = 3)
            logger.info(f"GP level {l} trained. Noise: {self.gps[l - 1].noise:.6f}")

    def _predict_up_to(self, x_new, level):
        fhat = 0.0
        sigma_2_hat = 0.0

        for l in range(1, level + 1):
            delta_f, delta_sigma2 = self.gps[l-1].predict(x_new)

            if l == 1:
                fhat = delta_f
                sigma_2_hat = delta_sigma2
            else:
                rho = self.rhos[l - 2]
                f_hat = rho * fhat + delta_f
                sigma_2_hat = (rho**2) * sigma_2_hat + delta_sigma2
        return f_hat, sigma_2_hat
    
    def predict(self, x_new):
        "Return the final prediction AND individual variances for the merit function"
        f_hat = 0.0
        sigma_2_hat = 0.0
        gp_variances = []

        for l in range(1, self.L + 1):
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