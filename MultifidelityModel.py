class MultifidelityModel:
    def __init__(self, L):
        self.L = L
        gps = [None] * L  # List to hold GaussianProcess instances for each fidelity level

    def fit(self, experiment_data):
        pass
    def predict(self, x_new):
        pass