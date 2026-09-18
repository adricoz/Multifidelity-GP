"""
This module implements the Efficient Global Optimization
(EGO) algorithm for multifidelity optimization.
"""
import json
import logging

import numpy as np
import scipy.optimize
from src.acquisition import AcquisitionFunction
from src.simulator import BaseSimulator
from src.surrogate_models import MultifidelityModel

logger = logging.getLogger(__name__)

class NumpyEncoder(json.JSONEncoder):
    """ Custom JSON encoder for numpy data types.
    This encoder converts numpy arrays to lists and numpy 
    scalar types to native Python types.
    """
    def default(self, obj: object) -> object:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.void, np.number)):
            return obj.item()
        return super().default(obj)


class EGOOptimizer:
    """Class for the Efficient Global Optimization (EGO) algorithm."""

    def __init__(self, data, model: MultifidelityModel,
                 simulator: BaseSimulator, acquisition: AcquisitionFunction):
        self.data = data
        self.model = model
        self.simulator = simulator
        self.acquisition = acquisition

        # follow the convergence and the costs
        self.current_total_cost = 0.0
        self.cost_history = []
        self.best_y_history = []
        self._init_doe_cost()

    def _init_doe_cost(self) -> None:
        """Initialize the total cost based on the initial DOE."""
        for l in range(1, self.model.num_levels + 1):
            n_evals = len(self.data.y_dict.get(l, []))
            self.current_total_cost += n_evals * self.data.costs[l - 1]

    def save_state(self, filename: str) -> None:
        """Save the current state of the optimizer to a JSON file."""
        state = {
            "X_dict": {str(k): v.tolist() for k, v in self.data.x_dict.items()},
            "Y_dict": {str(k): v.tolist() for k, v in self.data.y_dict.items()},
            "Metrics_dict": {str(k): v for k, v in self.data.metrics_dict.items()},
            "rhos": self.model.rhos,
            "gp_params": [gp.kernel.get_params().tolist() for gp in self.model.gps],
            "noises": [float(gp.noise) for gp in self.model.gps],
            "cost_history": self.cost_history,
            "best_y_history": self.best_y_history,
        }
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=4, cls=NumpyEncoder)

    def load_state(self, filename: str) -> None:
        """Load the state of the optimizer from a JSON file."""
        with open(filename, 'r', encoding='utf-8') as f:
            state = json.load(f)

        # Metrics
        self.data.x_dict = {int(k): np.array(v) for k, v in state["X_dict"].items()}
        self.data.y_dict = {int(k): np.array(v) for k, v in state["Y_dict"].items()}
        self.data.metrics_dict = {int(k): v for k, v in state.get("Metrics_dict", {}).items()}

        # Hyperparameters of the model
        self.model.rhos = state.get("rhos", [1.0] * (self.model.num_levels - 1))

        if "gp_params" in state:
            for l in range(self.model.num_levels):
                # give the thetas directly to each GP kernel
                self.model.gps[l].kernel.set_params(np.array(state["gp_params"][l]))

        if "noises" in state:
            for l in range(self.model.num_levels):
                self.model.gps[l].noise = state["noises"][l]
      
        # Budget history
        self.cost_history = state.get("cost_history", [])
        self.best_y_history = state.get("best_y_history", [])

        if self.cost_history:
            self.current_total_cost = self.cost_history[-1]

    def _find_next_point(self) -> tuple[np.ndarray, int, float]:
        """Find the next point to evaluate by maximizing the acquisition function."""
        def objective_wrapper(x: np.ndarray) -> float:
            """Wrapper function for the objective function to be minimized."""
            merits = [self.acquisition.evaluate_merit(x, l) \
                      for l in range(1, self.model.num_levels + 1)]
            best_merit = max(merits)

            return -best_merit  # We minimize the negative merit
        # differential evolution to find the next point
        result = scipy.optimize.differential_evolution(objective_wrapper, self.data.bounds,
                                                        popsize=10, maxiter=50, updating="deferred")
        x_optimal = result.x
        l_optimal = np.argmax([self.acquisition.evaluate_merit(x_optimal, l)
                               for l in range(1, self.model.num_levels + 1)]) + 1
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

    def tell(self, x_evaluated: np.ndarray, level: int, 
             y_result: float, metrics: dict = None) -> None:
        """
        Phase 2: Telling the optimizer the result of the
        evaluation. Ideal for Human in the loop type of process
        """
        self.data.add_observation(level, x_evaluated, y_result, metrics)
        #Security: saves the optimizer state in json file for later analysis
        self.current_total_cost += self.data.costs[level - 1]
        best_hf = np.min(self.data.y_dict[self.model.num_levels])

        self.cost_history.append(self.current_total_cost)
        self.best_y_history.append(best_hf)

        self.save_state("ego_backup.json")

    def run(self, n_iterations) -> tuple[list[float], list[float]]:
        """
        Auto Pilot mode: runs the EGO optimization loop for a specified number of iterations.
        (iterations with ask/tell scheme)
        """
        if not self.cost_history:
            best_hf_initial = np.min(self.data.y_dict[self.model.num_levels])
            self.cost_history.append(self.current_total_cost)
            self.best_y_history.append(best_hf_initial)

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
                l_next = self.model.num_levels

            logger.info(
                "\nNext selected point to evaluate: %s | Level: %d | Merit: %f",
                np.round(x_next, 4),
                l_next,
                merit,
            )

            if not self.data.is_already_evaluated(l_next, x_next):
                y_new, metrics = self.simulator.evaluate(x_next, l_next)
                self.tell(x_next, l_next, y_new, metrics)
                logger.info("    -> Evaluated value: %.6f at level %d", y_new, l_next)

            else:
                logger.warning(
                    "Point %s at level %d has already been evaluated. Skipping evaluation.",
                    np.round(x_next, 4),
                    l_next,
                )
        return self.cost_history, self.best_y_history

