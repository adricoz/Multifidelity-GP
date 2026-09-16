# this file is all about defining the kernel (covariance function) used for gaussian processes

class Kernel:
    def __call__(self, x, y):
        """
        Computes the covariance between two input points x and y.
        
        Args:
        - x: numpy array of shape (n_features,), first input point.
        - y: numpy array of shape (n_features,), second input point.
        
        Returns:
        - covariance: float, the computed covariance value between x and y.
        """
        pass
    def get_covariance_matrix(self, X):
        """
        Computes the covariance matrix for a set of input points X.
        
        Args:
        - X: numpy array of shape (n, d), the input dataset.
        
        Returns:
        - K: numpy array of shape (n, n), the covariance matrix.
        
        """
        pass
    def get_cross_variance_vector(self, x_new, X):
        """
        Computes the cross-variance vector between a new input point and a set of training points.
        
        Args:
        - x_new: numpy array of shape (n_features,), the new input point.
        - X: numpy array of shape (n, d), the training dataset.
        
        Returns:
        - k_star: numpy array of shape (n,), the cross-variance vector.
        """
        pass

class SquaredExponentialKernel(Kernel):
    def __init__(self, Theta):
        """
        Initializes the Squared Exponential Kernel with the given hyperparameters.
        
        Args:
        - Theta: list or numpy array, hyperparameters of the kernel (lengthscale and variance).
        
        Returns:
        - None
        """
        self.lengthscale = Theta[:-1]
        self.variance = Theta[-1]

    def __call__(self, x, y):
        """
        Computes the covariance between two input points x and y using the squared exponential formula.
        
        Args:
        - x: numpy array of shape (n_features,), first input point.
        - y: numpy array of shape (n_features,), second input point.
        
        Returns:
        - covariance: float, the computed covariance value between x and y.
        """
        pass