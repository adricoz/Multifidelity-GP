import logging

import numpy as np
from src.acquisition import AcquisitionFunction
from src.data_management import ExperimentData
from src.kernels import SquaredExponentialKernel
from src.optimizer import EGOOptimizer
from src.simulator import BaseSimulator
from src.surrogate_models import MultifidelityModel

logging.basicConfig(level=logging.INFO)



if __name__ == "__main__":
    class FoilSimulator(BaseSimulator):
        def __init__(self, L, target_cl_value):
            super().__init__(L)
            self.target_cl = target_cl_value

        def evaluate(self, x, level):
            """
            Evaluate the simulator at a given point and fidelity level.
            This is a placeholder implementation. Replace with actual simulation code.
            """
            # Example: simple quadratic function with noise
            noise = np.random.normal(0, 0.01)*1.0/level  # Noise increases with lower fidelity
            return (x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2 + noise
    
    # Define bounds for the design variables
    L = 2  # Number of fidelity levels
    bounds = [(0.0, 1.0), (0.0, 1.0)] # 2D: Camber, Thickness
    costs = [1.0, 1.0]  # Example costs for three fidelity levels
    initial_points = [10, 10]  # Number of points for each fidelity level
    target_cl = 1.0  # Target lift coefficient

    data = ExperimentData(bounds=bounds, costs=costs)
    simu = FoilSimulator(L=L, target_cl_value=target_cl)
    model = MultifidelityModel(L=L, kernel_class = SquaredExponentialKernel)
    acq = AcquisitionFunction(model=model, data=data)
    ego = EGOOptimizer(data=data, model=model, simulator=simu, acquisition=acq)

    print("Generating initial design...")
    data.generate_initial_design(points_per_level=initial_points)

    for l in range(1, L + 1):
        y_values = []
        for x in data.X_dict[l]:
            y = simu.evaluate(x, level=l)
            y_values.append(y)
        data.Y_dict[l] = np.array(y_values)
    print(f"Initial best HF observation: { np.min(data.Y_dict[L]):.4f}")

    # Launch
    ego.run(n_iterations = 10)
    print(f"Final best HF observation: { np.min(data.Y_dict[L]):.4f}")
