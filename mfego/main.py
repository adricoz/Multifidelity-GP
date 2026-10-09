"""
Main script to run a Multi Fidelity Efficient Global Optimization (EGO) process.
"""
import logging
import os

import numpy as np
from src.acquisition import AcquisitionFunction
from src.data_management import ExperimentData
from src.kernels import SquaredExponentialKernel
from src.optimizer import EGOOptimizer
from src.run_utils import RunTimer, create_run, setup_logging
from src.simulator import BaseSimulator
from src.surrogate_models import MultifidelityModel
from src.visualization import ModelVisualizer

# [FIX-L1] the logging is configured in the __main__ block, in a timestamped run directory
# (bdFoil convention: runs/<MMDD_HHMMSS>/mfego_<MMDD_HHMMSS>.log, Paris time)
HERE = os.path.dirname(os.path.abspath(__file__))
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    # [FIX-L1] one directory per run (log, ego_backup.json, surrogate.json, figures)
    run = create_run(HERE, prefix="mfego")
    setup_logging(run.log_path)

    # Define a simple simulator for demonstration purposes
    class FunctionSimulator(BaseSimulator):
        """
        A simple simulator that evaluates a quadratic function with noise.
        Subclass of BaseSimulator
        """
        def evaluate(self, design_point: list, level: int) -> tuple[float, dict]:
            """
            Evaluate the simulator at a given point and fidelity level.
            This is a placeholder implementation. Replace with actual simulation code.
            """
            # Example: Eqs: (17) of the reference article.
            # It should always deal with exections...
            try:
                # [FIX-E3] the level is checked first (a level <= 0 used to return None)
                if not isinstance(level, (int, np.integer)):
                    raise TypeError(f"Fidelity level must be an integer, got {type(level)}.")
                if level < 1 or level > 2:
                    raise ValueError(f"Invalid fidelity level: {level}. Must be 1 or 2.")
                x = design_point[0]
                f_1 = 0.5 *(6 * x - 2)**2 * np.sin(12 * x - 4) + 10 * (x - 1)
                if level == 1:
                    return f_1, {"y": f_1}
                f2 = 2 * f_1 - 20* (x -1)
                return f2, {"y": f2}

            except (IndexError, TypeError, ValueError) as e:
                logger.error( \
                    "Error evaluating simulator at point %s and level %s: %s", \
                     design_point, level, e)
                return np.nan, {}  # Return NaN to indicate an error in evaluation

    # Define bounds for the design variables
    L = 2  # Number of fidelity levels
    bounds = [(0.0, 1.0)] # 1D: Normalized
    costs = [1.0, 10.0]  # Example costs
    initial_points = [10, 4]  # Number of points for each fidelity level
    SEED = 0  # [FIX-R1] reproducible run
    # [MAP] MAP estimation of the GP hyperparameters (InvGamma prior on the lengthscales,
    # see README.md and benchmarks/map_hartmann/RAPPORT_MAP.md); False = maximum likelihood
    USE_MAP = True

    # [FIX-L1] start banner and total computation time written at the end of the log
    with RunTimer("mfego - Forrester example (Eq. 17)"):
        logger.info("Run directory: %s", run.run_dir)
        data = ExperimentData(bounds=bounds, costs=costs)
        simu = FunctionSimulator(num_levels=L)
        # [FIX-T1c] rho is computed at every fit (default, Sacher Eq. 15): the levels of Eq. 17
        # differ by a factor rho = 2
        model = MultifidelityModel(l=L, kernel_class = SquaredExponentialKernel, seed = SEED,
                                   use_map = USE_MAP)
        acq = AcquisitionFunction(model=model, data=data)
        ego = EGOOptimizer(data=data, model=model, simulator=simu, acquisition=acq,
                           save_state_path = run.path("ego_backup.json"), seed = SEED)

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
        _, _ = ego.run(n_iterations = 10)
        logger.info("Final best HF observation: %.4f", np.nanmin(data.y_dict[L]))
        # [FIX-X1] self-contained surrogate file, reload with src.surrogate_models.load_surrogate
        ego.export_surrogate(run.path("surrogate.json"))

        # Visualize the results
        vizualizer = ModelVisualizer(run.path("ego_backup.json"), num_levels=L)
        vizualizer.plot_convergence(target = -6.020740055767082786553,
                                    save_path = run.path("convergence_plot.png"))
        vizualizer.plot_response_1d(save_path = run.path("response_1d.png"))
        # [FIX-X3] interactive (plotly) versions of the plots
        vizualizer.plot_convergence_interactive(target = -6.020740055767082786553,
                                                save_path = run.path("convergence_plot.html"))
        vizualizer.plot_response_1d_interactive(save_path = run.path("response_1d.html"))
