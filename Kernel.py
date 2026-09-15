# this file is all about defining the kernel (covariance function) used for gaussian processes
import numpy as np
from non_nested.non_nested_mf_covariance import base_covariance_matrix

class Kernel:
    def __init__(self, lengthscale, variance):
        self.lengthscale = lengthscale
        self.variance = variance

    def compute_covariance_matrix(self, X):
        """
        Computes the covariance matrix for the dataset X using the specified kernel.
        
        Arguments:
        - X: numpy array of shape (n, d), the input dataset.
        
        Returns:
        - K: numpy array of shape (n, n), the covariance matrix.
        """
        Theta_l = np.concatenate((self.lengthscale, [self.variance]))
        return base_covariance_matrix(X, Theta_l)