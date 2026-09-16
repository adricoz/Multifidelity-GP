import logging

import numpy as np

from data_management import ExperimentData
from kernels import SqaredExponentialKernel
from optimizer import AcquisitionFunction, EGOOptimizer
from simulator import Simulator
from surrogate_models import MultifidelityModel

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    # Define bounds for the design variables
    L = 2  # Number of fidelity levels
    bounds = [(0.0, 1.0), (0.0, 1.0)] # 2D: Camber, Thickness
    costs = [1.0, 1.0]  # Example costs for three fidelity levels
    initial_points = [10, 10]  # Number of points for each fidelity level

    data = ExperimentData(bounds=bounds, costs=costs)
    simu = Simulator(L=L)
    model = MultifidelityModel(L=L, kernel_class = SqaredExponentialKernel)
    acq = AcquisitionFunction(model=model, data=data)
    ego = EGOOptimizer(data=data, model=model, simulator=simu, acquisition=acq)

    print("Generating initial design...")
    data.generate_initial_design(points_per_level=initial_points)

    for l in range(1, L + 1):
        for x in data.X_dict[l]:
            y = simu.evaluate(x, level=l)
            data.add_observation(l, x, y)
    print(f"Initial best HF observation: { np.min(data.Y_dict[L]):.4f}")

    # Launch
    ego.run(n_iterations = 20)
