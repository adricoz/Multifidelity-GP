"""
Main script to run a Multi Fidelity Efficient Global Optimization (EGO) process.
"""
import logging
import os
import sys

import numpy as np

# Resolve imports relative to this file rather than the process working
# directory (which may be different when the example is launched externally).
mfego_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "mfego"))
sys.path.append(mfego_path)

# pylint: disable=import-error,wrong-import-position
from .Hartmann6d import evaluate_fidelity
from src.acquisition import AcquisitionFunction
from src.data_management import ExperimentData
from src.kernels import SquaredExponentialKernel
from src.optimizer import EGOOptimizer
from src.simulator import BaseSimulator
from src.surrogate_models import MultifidelityModel
from src.visualization import ModelVisualizer

logging.basicConfig(
    filename='example/hartmann_6d/logfile.log',
    filemode='w',  # [FIX-E3] one clean log per run
    level=logging.INFO,
    format=' %(levelname)s - %(message)s',
    force = True,
    )
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    # Define a simple simulator for demonstration purposes
    class FunctionSimulator(BaseSimulator):
        """
        A simple simulator for the Hartmann 6d function with multi-fidelity approximation.
        Subclass of BaseSimulator
        """
        def evaluate(self, design_point: list, level: int) -> tuple[float, dict]:
            """
            Evaluate the simulator at a given point and fidelity level.
            This is a placeholder implementation. Replace with actual simulation code.
            """
            # Example: Eqs: (30)-(32) of the reference article.
            # It should always deal with exections...
            try:
               # [FIX-E1] a single evaluation (the function was evaluated twice) and the
               # number of levels of the simulator (was hard-coded to 2 here)
               y_value = evaluate_fidelity(design_point, level, self.num_levels)
               return y_value, {"y": y_value}

            except (IndexError, TypeError, ValueError) as e:
                logger.error( \
                    "Error evaluating simulator at point %s and level %s: %s", \
                     design_point, level, e)
                return np.nan, {}  # Return NaN to indicate an error in evaluation

    # Define bounds for the design variables
    L = 2  # Number of fidelity levels
    bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), \
               (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)] # 6D: Normalized
    costs = [1.0, 10.0]  # Example costs
    initial_points = [20, 10]  # Number of points for each fidelity level
    SEED = 0  # [FIX-R1] reproducible run

    data = ExperimentData(bounds=bounds, costs=costs)
    simu = FunctionSimulator(num_levels=L)
    model = MultifidelityModel(l=L, kernel_class = SquaredExponentialKernel, seed = SEED)
    acq = AcquisitionFunction(model=model, data=data)
    ego = EGOOptimizer(data=data, model=model, simulator=simu, acquisition=acq, 
                       save_state_path = "example/hartmann_6d/ego_backup.json", seed = SEED)

    logger.info("Generating initial design...")

    data.generate_initial_design(points_per_level=initial_points)
    for l in range(1, L + 1):
        y_values = []
        metrics_list = []

        logger.info("Level %s design points: %s", l, data.x_dict[l])

        for x in data.x_dict[l]:
            y_opt, metrics = simu.evaluate(x, level=l)

            y_values.append(y_opt)
            metrics_list.append(metrics)

        data.y_dict[l] = np.array(y_values)
        data.metrics_dict[l] = metrics_list

    logger.info("Initial best HF observation: %.4f", np.nanmin(data.y_dict[L]))

    # Launch
    _, _ = ego.run(n_iterations = 30)
    logger.info("Final best HF observation: %.4f", np.nanmin(data.y_dict[L]))
    # [FIX-X1] self-contained surrogate file (reload with load_surrogate)
    ego.export_surrogate("example/hartmann_6d/surrogate.json")

    # Visualize the results
    vizualizer = ModelVisualizer(num_levels=L, 
                                 json_filepath = "example/hartmann_6d/ego_backup.json")
    vizualizer.plot_convergence(target = -3.32236801141551385541, \
                                save_path = "example/hartmann_6d/convergence_plot.png")
    vizualizer.plot_response_surface_2d(save_path = \
                                        "example/hartmann_6d/response_surface_2d.png")
    # [FIX-X3] interactive (plotly) versions of the plots
    vizualizer.plot_convergence_interactive(target = -3.32236801141551385541, \
                                save_path = "example/hartmann_6d/convergence_plot.html")
    vizualizer.plot_response_surface_2d_interactive(save_path = \
                                        "example/hartmann_6d/response_surface_2d.html")
