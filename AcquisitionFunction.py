
class AcquisitionFunction:
    def __init__(self, model, costs):
        self.model = model
        self.costs = costs
    def _expected_improvement(self, f_hat, sigma2_hat, f_best):
        pass
    def _augmented_expected_improvement(self):
        pass
    def evaluate_merit(self, x, candidate_level):
        pass