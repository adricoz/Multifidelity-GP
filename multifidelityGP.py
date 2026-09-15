# This file is all about defining the classes about the multifidelity Gaussian process. The main class is the MultifidelityGP, which is a wrapper around the GPy library. It allows for the creation of a multifidelity Gaussian process regressor
import numpy as np
#local files and clasees
from non_nested.non_nested_mf_optimizer import *

class gaussian_process_regression:

    def __init__(self, X_train, Y_train, kernel, noise_variance):
        """
        Initializes the Gaussian Process Regression model.
        
        Parameters:
        - X_train: numpy array of shape (n_samples, n_features), training input data.
        - Y_train: numpy array of shape (n_samples,), training output data.
        - kernel: instance of the Kernel class, defining the covariance function.
        - noise_variance: float, variance of the observation noise.
        """
        self.X_train = X_train
        self.Y_train = Y_train
        self.kernel = kernel
        self.noise_variance = noise_variance

    def predict_gp(self, x_new):
        """
        Predicts the mean and variance at a new point using the Gaussian Process model.
        
        Parameters:
        - x_new: numpy array of shape (n_features,), new input point for prediction.
        
        Returns:
        - delta_hat_scalar: float, predicted mean at the new point.
        - sigma2_delta_scalar: float, predicted variance at the new point.
        """
        Theta_l = np.concatenate((self.kernel.lengthscale, [self.kernel.variance]))
        return predict_base_gp(x_new, self.X_train, self.Y_train, Theta_l, self.noise_variance)


class MultifidelityGP:
    def __init__(self, fidelity_levels, kernel, noise_variances):
        """
        Initializes the Multifidelity Gaussian Process model.
        
        Parameters:
        - fidelity_levels: integer number of fidelity levels
        - kernel: instance of the Kernel class, defining the covariance function.
        - noise_variances: list of floats, variances of the observation noise for each fidelity level.
        """
        self.fidelity_levels = fidelity_levels
        self.kernel = kernel
        self.noise_variances = noise_variances

    def Log_likelyhood_mf(rho_l_minus1, Theta_l, sigma_epsilon_l, X_l, Y_l, Y_l_minus_1, fidelity_level):
        """
        Computes the log-likelihood of the multifidelity Gaussian Process model.
        
        Returns:
        - log_likelihood: float, the computed log-likelihood value.
        """
        return log_likelihood_mf(rho_l_minus1, Theta_l, sigma_epsilon_l, X_l, Y_l, Y_l_minus_1, fidelity_level)
        