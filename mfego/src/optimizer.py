"""
This module implements the Efficient Global Optimization
(EGO) algorithm for multifidelity optimization.
"""
import json
import logging

import numpy as np
import scipy.optimize

from mfego.src.acquisition import AcquisitionFunction
from mfego.src.simulator import BaseSimulator
from mfego.src.surrogate_models import MultifidelityModel

logger = logging.getLogger(__name__)


class EGOOptimizer:
    """Class for the Efficient Global Optimization (EGO) algorithm."""

    def __init__(self, data, model: MultifidelityModel,
                 simulator: BaseSimulator, acquisition: AcquisitionFunction):
        self.data = data
        self.model = model
        self.simulator = simulator
        self.acquisition = acquisition

    def save_state(self, filename: str) -> None:
        """Save the current state of the optimizer to a JSON file."""
        state = {
            "X_dict": {str(k): v.tolist() for k, v in self.data.X_dict.items()},
            "Y_dict": {str(k): v.tolist() for k, v in self.data.Y_dict.items()},
            "rhos": self.model.rhos,
            "gp_params": [gp.kernel.get_params().tolist() for gp in self.model.gps],
        }
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=4)

    def _find_next_point(self) -> tuple[np.ndarray, int, float]:
        """Find the next point to evaluate by maximizing the acquisition function."""
        def objective_wrapper(x: np.ndarray) -> float:
            """Wrapper function for the objective function to be minimized."""
            merits = [self.acquisition.evaluate_merit(x, l) for l in range(1, self.model.L + 1)]
            best_merit = max(merits)

            return -best_merit  # We minimize the negative merit
        # differential evolution to find the next point
        result = scipy.optimize.differential_evolution(objective_wrapper, self.data.bounds,
                                                        popsize=10, maxiter=50, updating="deferred")
        x_optimal = result.x
        l_optimal = np.argmax([self.acquisition.evaluate_merit(x_optimal, l)
                               for l in range(1, self.model.L + 1)]) + 1
        return x_optimal, l_optimal, -result.fun  # Return the merit value as well

    def ask(self) -> tuple[np.ndarray, int, float]:
        """
        Phase 1: Asking for the next point to evaluate.
        Ideal for Human in the loop type of process
        """
        #train the model
        self.model.fit(self.data)
        x_next, l_next, merit = self._find_next_point()
        # Security: saves the optimizer state in json file for later analysis
        self.save_state("ego_backup.json")
        return x_next, l_next, merit

    def tell(self, x_evaluated: np.ndarray, level: int, y_result: float) -> None:
        """
        Phase 2: Telling the optimizer the result of the
        evaluation. Ideal for Human in the loop type of process
        """
        self.data.add_observation(level, x_evaluated, y_result)
        #Security: saves the optimizer state in json file for later analysis
        self.save_state("ego_backup.json")

    def run(self, n_iterations) -> None:
        """
        Auto Pilot mode: runs the EGO optimization loop for a specified number of iterations.
        (iterations with ask/tell scheme)
        """
        for iteration in range(n_iterations):
            logger.info(
                "--- EGO Iteration %d/%d ---",
                iteration + 1,
                n_iterations,
            )
            # we use also the ask/tell scheme
            x_next, l_next, merit = self.ask()

            #Failsafe
            if merit <= 0.0:
                logger.warning("Warning: Merit is 0. Random selection triggered.")
                x_next = np.array([np.random.uniform(b[0], b[1]) for b in self.data.bounds])
                l_next = self.model.L

            logger.info(
                "\nNext selected point to evaluate: %s | Level: %d | Merit: %f",
                np.round(x_next, 4),
                l_next,
                merit,
            )

            if not self.data.is_already_evaluated(l_next, x_next):
                y_new = self.simulator.evaluate(x_next, l_next)
                self.tell(x_next, l_next, y_new)
                logger.info("    -> Evaluated value: %.6f at level %d", y_new, l_next)

            else:
                logger.warning(
                    "Point %s at level %d has already been evaluated. Skipping evaluation.",
                    np.round(x_next, 4),
                    l_next,
                )
