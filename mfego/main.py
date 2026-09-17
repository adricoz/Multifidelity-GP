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
        def evaluate(self, design_point: list, level: int) -> float:
            """
            Evaluate the simulator at a given point and fidelity level.
            This is a placeholder implementation. Replace with actual simulation code.
            """
            # Example: Eqs: (17) of the reference article.
            # It should always deal with exections...
            try:
                x = design_point[0]
                f_1 = 0.5 *(6 * x - 2)**2 * np.sin(12 * x - 4) + 10 * (x - 1)
                if level == 1:
                    return f_1
                if level == 2:
                    return 2 * f_1 - 20* (x -1)
                if level > 2:
                    raise ValueError(f"Invalid fidelity level: {level}. Must be 1 or 2.")
                if not isinstance(level, int):
                    raise TypeError(f"Fidelity level must be an integer, got {type(level)}.")

            except (IndexError, TypeError, ValueError) as e:
                logging.error( \
                    "Error evaluating simulator at point %s and level %s: %s", \
                     design_point, level, e)  # noqa: LOG015
                return np.nan  # Return NaN to indicate an error in evaluation

    # Define bounds for the design variables
    L = 2  # Number of fidelity levels
    bounds = [(0.0, 1.0)] # 1D: Normalized
    costs = [1.0, 1.0]  # Example costs
    initial_points = [11, 4]  # Number of points for each fidelity level

    data = ExperimentData(bounds=bounds, costs=costs)
    simu = FunctionSimulator(num_levels=L)
    model = MultifidelityModel(l=L, kernel_class = SquaredExponentialKernel)
    acq = AcquisitionFunction(model=model, data=data)
    ego = EGOOptimizer(data=data, model=model, simulator=simu, acquisition=acq)

    logging.info("Generating initial design...")  # noqa: LOG015

    data.generate_initial_design(points_per_level=initial_points)
    for l in range(1, L + 1):
        y_values = []
        logging.info(f"Level {l} design points: {data.x_dict[l]}")
        for x in data.x_dict[l]:
            y = simu.evaluate(x, level=l)
            y_values.append(y)
        data.y_dict[l] = np.array(y_values)
    logging.info(f"Initial best HF observation: { np.min(data.y_dict[L]):.4f}")  # noqa: LOG015

    # Launch
    ego.run(n_iterations = 1)
    logging.info(f"Final best HF observation: { np.min(data.y_dict[L]):.4f}")  # noqa: LOG015
