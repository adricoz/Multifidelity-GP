"""
Main script to run a Multi Fidelity Efficient Global Optimization (EGO) process.
"""
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
    # Define a simple simulator for demonstration purposes
    class FunctionSimulator(BaseSimulator):
        """
        A simple simulator that evaluates a quadratic function with noise.
        Subclass of BaseSimulator
        """
        def __init__(self, fidelity_levels: int, target_cl_value: float):
            super().__init__(fidelity_levels)
            self.target_cl = target_cl_value

        def evaluate(self, design_point: list, level: int) -> float:
            """
            Evaluate the simulator at a given point and fidelity level.
            This is a placeholder implementation. Replace with actual simulation code.
            """
            # Example: simple quadratic function with noise
            noise = np.random.normal(0, 0.01)*1.0/level  # Noise increases with lower fidelity
            return (design_point[0] - 0.5) ** 2 + (design_point[1] - 0.5) ** 2 + noise

    # Define bounds for the design variables
    L = 2  # Number of fidelity levels
    bounds = [(0.0, 1.0), (0.0, 1.0)] # 2D: Camber, Thickness
    costs = [1.0, 1.0]  # Example costs for three fidelity levels
    initial_points = [10, 10]  # Number of points for each fidelity level
    TARGET_CL = 1.0  # Target lift coefficient

    data = ExperimentData(bounds=bounds, costs=costs)
    simu = FunctionSimulator(fidelity_levels=L, target_cl_value=TARGET_CL)
    model = MultifidelityModel(l=L, kernel_class = SquaredExponentialKernel)
    acq = AcquisitionFunction(model=model, data=data)
    ego = EGOOptimizer(data=data, model=model, simulator=simu, acquisition=acq)

    print("Generating initial design...")
    data.generate_initial_design(points_per_level=initial_points)

    for l in range(1, L + 1):
        y_values = []
        for x in data.x_dict[l]:
            y = simu.evaluate(x, level=l)
            y_values.append(y)
        data.y_dict[l] = np.array(y_values)
    print(f"Initial best HF observation: { np.min(data.y_dict[L]):.4f}")

    # Launch
    ego.run(n_iterations = 10)
    print(f"Final best HF observation: { np.min(data.y_dict[L]):.4f}")
