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
                 simulator: BaseSimulator, acquisition: AcquisitionFunction,
                 save_state_path: str = "ego_backup.json", seed: int = None):
        self.data = data
        self.model = model
        self.simulator = simulator
        self.acquisition = acquisition
        self.save_state_path = save_state_path
        # [FIX-R1] seeded generator for the differential evolution and the random failsafe
        self.rng = np.random.default_rng(seed)
        self.seed = seed

        # follow the convergence and the costs
        self.current_total_cost = 0.0
        self.cost_history = []
        self.best_y_history = []
        # [FIX-R5] the DOE cost is counted lazily (first run()/tell()): in the scripts the
        # optimizer is created BEFORE the DOE is generated and evaluated, so counting it here
        # always gave 0.
        self._doe_cost_counted = False

    def _init_doe_cost(self) -> None:
        """Initialize the total cost based on the initial DOE."""
        if self._doe_cost_counted:
            return
        for l in range(1, self.model.num_levels + 1):
            # [FIX-R4] failed evaluations are counted too (the simulation cost was spent)
            n_evals = len(self.data.y_dict.get(l, []))
            self.current_total_cost += n_evals * self.data.costs[l - 1]
        self._doe_cost_counted = True

    def _init_history(self) -> None:
        """[FIX-R5] Counts the DOE cost and stores the first point of the convergence history."""
        if self.cost_history:
            return
        self._init_doe_cost()
        self.cost_history.append(self.current_total_cost)
        self.best_y_history.append(self.data.best_observation(self.model.num_levels)[1])

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
            # [FIX-X2] problem definition + exact snapshot of the trained model (data used by
            # the last fit, hyperparameters, normalization) for an exact reloading
            "num_levels": self.model.num_levels,
            "bounds": [list(b) for b in self.data.bounds],
            "costs": list(self.data.costs),
            "surrogate": self.model.to_dict() if self.model.is_fitted() else None,
        }
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=4, cls=NumpyEncoder)

    def load_state(self, filename: str) -> None:
        """Load the state of the optimizer from a JSON file."""
        with open(filename, 'r', encoding='utf-8') as f:
            state = json.load(f)

        # Metrics
        self.data.x_dict = {int(k): np.array(v) for k, v in state["X_dict"].items()}
        self.data.y_dict = {int(k): np.array(v, dtype=float) for k, v in state["Y_dict"].items()}
        self.data.metrics_dict = {int(k): v for k, v in state.get("Metrics_dict", {}).items()}

        # Hyperparameters of the model
        self.model.rhos = state.get("rhos", [1.0] * (self.model.num_levels - 1))

        if "gp_params" in state:
            for l in range(self.model.num_levels):
                # give the thetas directly to each GP kernel
                # [FIX-N8] they are now used as warm start by the next fit
                self.model.gps[l].kernel.set_params(np.array(state["gp_params"][l]))

        if "noises" in state:
            for l in range(self.model.num_levels):
                self.model.gps[l].noise = state["noises"][l]

        # Budget history
        self.cost_history = state.get("cost_history", [])
        self.best_y_history = state.get("best_y_history", [])

        if self.cost_history:
            self.current_total_cost = self.cost_history[-1]
            self._doe_cost_counted = True

    def _find_next_point(self) -> tuple[np.ndarray, int, float]:
        """Find the next point to evaluate by maximizing the acquisition function."""
        def objective_wrapper(x: np.ndarray) -> float:
            """Wrapper function for the objective function to be minimized.
            [FIX-N6] vectorized: x has shape (d,) or (d, S) (whole DE population at once)."""
            points = np.atleast_2d(x.T) if x.ndim == 2 else x.reshape(1, -1)
            best_merit = np.max(self.acquisition.evaluate_merits_batch(points), axis=1)

            # We minimize the negative merit
            return -best_merit if x.ndim == 2 else -float(best_merit[0])
        # differential evolution to find the next point
        # [FIX-R1] seeded, [FIX-N6] vectorized population evaluation
        result = scipy.optimize.differential_evolution(
            objective_wrapper, self.data.bounds, popsize=10, maxiter=50,
            updating="deferred", vectorized=True,
            seed=None if self.seed is None else int(self.rng.integers(2**31 - 1)))
        x_optimal = result.x
        l_optimal = int(np.argmax(self.acquisition.evaluate_merits(x_optimal))) + 1
        return x_optimal, l_optimal, -result.fun  # Return the merit value as well

    def ask(self) -> tuple[np.ndarray, int, float]:
        """
        Phase 1: Asking for the next point to evaluate.
        Ideal for Human in the loop type of process
        """
        #train the model
        self.model.fit(self.data)
        # [FIX-T3] effective best solution of Eq. 19 for the new model
        self.acquisition.update()
        x_next, l_next, merit = self._find_next_point()
        # Security: saves the optimizer state in json file for later analysis
        self.save_state(self.save_state_path)
        return x_next, l_next, merit

    def tell(self, x_evaluated: np.ndarray, level: int,
             y_result: float, metrics: dict = None) -> None:
        """
        Phase 2: Telling the optimizer the result of the
        evaluation. Ideal for Human in the loop type of process
        """
        # [FIX-R5] DOE cost and first history point (if not done yet)
        self._init_history()
        # [FIX-R4] a failed evaluation (NaN/inf) is stored as NaN and excluded from the GP
        self.data.add_observation(level, x_evaluated, y_result, metrics)
        #Security: saves the optimizer state in json file for later analysis
        self.current_total_cost += self.data.costs[level - 1]
        best_hf = self.data.best_observation(self.model.num_levels)[1]

        self.cost_history.append(self.current_total_cost)
        self.best_y_history.append(best_hf)

        self.save_state(self.save_state_path)

    def run(self, n_iterations, stop_on_convergence: bool = False
            ) -> tuple[list[float], list[float]]:
        """
        Auto Pilot mode: runs the EGO optimization loop for a specified number of iterations.
        (iterations with ask/tell scheme)
        [FIX-T3b] stop_on_convergence=True stops the loop when the merit vanishes and the
        surrogate optimum has already been evaluated at the highest level (EI-based stopping
        criterion, Sacher Sec. 3.4) instead of spending random high-fidelity evaluations.
        """
        # [FIX-R5] DOE cost counted here (was always 0)
        self._init_history()

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
                l_next = self.model.num_levels
                x_best = self.acquisition.x_best
                # [FIX-T3b] the merit vanishes when the surrogate is confident about its
                # optimum (effective best solution of Eq. 19, possibly a low-fidelity point):
                # the highest level is first evaluated AT that optimum (confirmation), the
                # random selection is only used if this point is already known.
                if x_best is not None and not self.data.is_already_evaluated(l_next, x_best):
                    logger.warning("Warning: Merit is 0. Evaluating the surrogate optimum.")
                    x_next = np.array(x_best, dtype=float)
                elif stop_on_convergence:
                    logger.info("Merit is 0 and the surrogate optimum is known: stopping.")
                    break
                else:
                    logger.warning("Warning: Merit is 0. Random selection triggered.")
                    # [FIX-R1] seeded generator (was the global np.random)
                    x_next = np.array([self.rng.uniform(b[0], b[1])
                                       for b in self.data.bounds])

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

        # [FIX-X2] final fit on ALL the data: the saved state (and the in-memory model) now
        # correspond to a trained model (previously the last point was saved with the
        # hyperparameters of the previous fit).
        self.model.fit(self.data)
        self.acquisition.update()
        self.save_state(self.save_state_path)
        self.summary()
        return self.cost_history, self.best_y_history

    def summary(self, log: bool = True) -> dict:
        """
        [FIX-X1] Summary of the optimization (also written in the log): best observed
        high-fidelity point, effective best of the surrogate, cost and data per level.
        """
        x_obs, y_obs = self.data.best_observation(self.model.num_levels)
        info = {
            "best_observed_x": None if x_obs is None else np.asarray(x_obs).tolist(),
            "best_observed_y": y_obs,
            "surrogate_best_x": None if self.acquisition.x_best is None
                                else self.acquisition.x_best.tolist(),
            "surrogate_best_f": self.acquisition.f_best,
            "total_cost": self.current_total_cost,
            "n_points_per_level": {l: len(self.data.y_dict.get(l, []))
                                   for l in range(1, self.model.num_levels + 1)},
            "n_failed_per_level": {l: self.data.n_failed(l)
                                   for l in range(1, self.model.num_levels + 1)},
            "rhos": [float(r) for r in self.model.rhos],
        }
        if log:
            logger.info("Optimization summary: %s", json.dumps(info, cls=NumpyEncoder))
        return info

    def export_surrogate(self, filename: str = "surrogate.json") -> None:
        """
        [FIX-X1] Exports a self-contained surrogate file: trained model (data used by the
        last fit, hyperparameters, normalization, rhos), problem bounds/costs and summary.
        Reload it with `src.surrogate_models.load_surrogate(filename)`.
        """
        if not self.model.is_fitted():
            self.model.fit(self.data)
            self.acquisition.update()
        export = {
            "surrogate": self.model.to_dict(),
            "num_levels": self.model.num_levels,
            "bounds": [list(b) for b in self.data.bounds],
            "costs": list(self.data.costs),
            "summary": self.summary(log=False),
        }
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(export, f, indent=4, cls=NumpyEncoder)
        logger.info("Surrogate exported to %s", filename)
