from traitlets import List, Tuple


class ExperimentData:
    def __init__(self, bounds: List[Tuple[float, float]], costs: List[float]) -> None:
        """Initializes the ExperimentData class with bounds and costs.
        
        Args:
        - bounds: list of tuples, where each tuple defines the lower and upper bounds for a dimension.
        - costs: list of floats, representing the cost associated with each fidelity level.

        Returns:
        - None
        """
        self.bounds = bounds
        self.costs = costs
        self.dim = len(bounds)
        self.X_train = None
        self.Y_train = None

    def generate_initial_design(self, n_points_list):
        """Generates an initial design of experiments based on the provided number of points for each fidelity level

        Args:
        - n_points_list: list of integers
        
        Returns:
        - None
        """
        pass

    def add_observation(self, level, x_new, y_new):

        pass
    def check_already_evaluated(self, level, x):

        pass