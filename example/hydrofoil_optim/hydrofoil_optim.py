"""
Main script to run a Multi Fidelity Efficient Global Optimization (EGO) process.
"""
import logging
import os
import sys

import aerosandbox as asb
import numpy as np

# Resolve imports relative to this file rather than the process working
# directory (which may be different when the example is launched externally).
mfego_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "mfego"))
sys.path.append(mfego_path)

# pylint: disable=import-error,wrong-import-position
from src.acquisition import AcquisitionFunction
from src.data_management import ExperimentData
from src.kernels import SquaredExponentialKernel
from src.optimizer import EGOOptimizer
from src.simulator import BaseSimulator
from src.surrogate_models import MultifidelityModel
from src.visualization import ModelVisualizer

from .optim_neuralfoil import generate_continuous_naca4, objective_function

logging.basicConfig(
    filename='example/hydrofoil_optim/logfile.log',
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
        A simple simulator for the hydrofoil drag (NeuralFoil) at a target lift coefficient.
        Subclass of BaseSimulator
        """
        def evaluate(self, design_point: list, level: int) -> tuple[float, dict]:
            """
            Evaluate the simulator at a given point and fidelity level.
            This is a placeholder implementation. Replace with actual simulation code.
            """
            # Example: Eqs: (17) of the reference article.
            # It should always deal with exections...
            # [FIX-E2] number of levels of the simulator (was hard-coded to 2 here)
            n_levels = self.num_levels
            try:
               m_camber = 0.02 + design_point[0] * (0.09 - 0.02)    # Exact camber (between 2% and 9%)
               p_position = 0.3
               pos_int = round(p_position * 10)                # Fixed position of maximum camber (30%)
               t_thickness = 0.08 + design_point[1] * (0.17 - 0.08)
               naca_foil = generate_continuous_naca4(m_camber, p_position, t_thickness, n_points=100)

               exact_name = f"naca_{m_camber*100:.2f}_{pos_int}_{t_thickness*100:.2f}"
               
               custom_naca = asb.Airfoil(name=exact_name, coordinates=naca_foil)
               cd, cl, alpha_perfect = objective_function(custom_naca, target_cl=1.0, level=level, L=n_levels)
               return cd, \
                   {"cd": cd, "cl": cl, "alpha": alpha_perfect}

            except (IndexError, TypeError, ValueError) as e:
                logger.error( \
                    "Error evaluating simulator at point %s and level %s: %s", \
                     design_point, level, e)
                return np.nan, {}  # Return NaN to indicate an error in evaluation

    # Define bounds for the design variables
    # [FIX-E2] L = 2: "xxsmall" (level 1) and "xxxlarge" (level 2) drag models. Both levels
    # find alpha(Cl = 1) with "xxxlarge", so their costs are comparable (costs = [1, 1]).
    # L = 1: single-fidelity test of the framework ("xxxlarge" only).
    L = 2  # Number of fidelity levels
    #2D design space: camber, thickness
    bounds = [(0.0, 1.0), (0.0, 1.0)]
    costs = [1.0, 1.0][:L]  # Example costs
    initial_points = [6, 2] if L == 2 else [8]  # Number of points for each fidelity level
    SEED = 0  # [FIX-R1] reproducible run

    data = ExperimentData(bounds=bounds, costs=costs)
    simu = FunctionSimulator(num_levels=L)
    model = MultifidelityModel(l=L, kernel_class = SquaredExponentialKernel, seed = SEED)
    acq = AcquisitionFunction(model=model, data=data)
    ego = EGOOptimizer(data=data, model=model, simulator=simu, acquisition=acq, 
                       save_state_path = "example/hydrofoil_optim/ego_backup.json", seed = SEED)

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
    _, _ = ego.run(n_iterations = 40)
    logger.info("Final best HF observation: %.4f", np.nanmin(data.y_dict[L]))

    # Loggimg the final best design point
    # [FIX-X1] summary (best point, cost, points per level) + self-contained surrogate file
    logger.info("Final summary: %s", ego.summary(log=False))
    ego.export_surrogate("example/hydrofoil_optim/surrogate.json")

    # Visualize the results
    # [FIX-E2] reference minimum of the HF objective on the design bounds, computed by
    # analysis/scripts/check_simulators.py (grid 21x21 + Nelder-Mead): Cd = 0.0138978
    # (camber 4.51 %, thickness 8 %). The previous value 0.0139 was a rounded estimate.
    HF_TARGET = 0.0138978
    vizualizer = ModelVisualizer(num_levels=L,
                                 json_filepath = "example/hydrofoil_optim/ego_backup.json")
    vizualizer.plot_convergence(target = HF_TARGET,
                                save_path = "example/hydrofoil_optim/convergence_plot.png")
    vizualizer.plot_response_surface_2d(
        save_path = "example/hydrofoil_optim/response_surface_2d.png")
    # [FIX-X3] interactive (plotly) versions of the plots
    vizualizer.plot_convergence_interactive(
        target = HF_TARGET, save_path = "example/hydrofoil_optim/convergence_plot.html")
    vizualizer.plot_response_surface_2d_interactive(
        save_path = "example/hydrofoil_optim/response_surface_2d.html")
