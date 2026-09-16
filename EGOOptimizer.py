class EGOOptimizer:
    def __init__(self, data, model, simulator, AcquisitionFunction):
        self.data = data
        self.model = model
        self.simulator = simulator
        self.AcquisitionFunction = AcquisitionFunction

    def step(self):
        pass
    def run(self, n_iterations):
        pass
    def _find_next_point(self):
        pass
    