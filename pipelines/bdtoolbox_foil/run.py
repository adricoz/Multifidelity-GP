"""
Foil optimization with mfego and the bdToolbox / bdFoil solvers.

    python -m pipelines.bdtoolbox_foil.run --config pipelines/configs/section2d_naca_3levels.json
    python -m pipelines.bdtoolbox_foil.run --config ... --calibrate-costs 5

Every run writes pipelines/runs/<MMDD_HHMMSS>/: <name>_<MMDD_HHMMSS>.log (with the total
computation time and the time per level), config.json, ego_backup.json, surrogate.json,
results.json (best design in physical units) and the figures (PNG + interactive HTML).
Recommended interpreter: conda env 'bdToolbox' (Python 3.10, NeuralFoil) or any environment with
the mfego requirements + neuralfoil >= 0.2.0.
"""
import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "mfego") not in sys.path:
    sys.path.insert(0, str(ROOT / "mfego"))

# pylint: disable=wrong-import-position,import-error
from src.acquisition import AcquisitionFunction  # noqa: E402
from src.data_management import ExperimentData  # noqa: E402
from src.kernels import SquaredExponentialKernel  # noqa: E402
from src.optimizer import EGOOptimizer, NumpyEncoder  # noqa: E402
from src.run_utils import RunTimer, create_run, setup_logging  # noqa: E402
from src.surrogate_models import MultifidelityModel  # noqa: E402
from src.visualization import ModelVisualizer  # noqa: E402

from . import bridge  # noqa: E402
from .config import load_config  # noqa: E402
from .constraints import feasibility_function  # noqa: E402
from .geometry import to_physical  # noqa: E402
from .simulator import BdToolboxFoilSimulator  # noqa: E402

logger = logging.getLogger(__name__)


def calibrate_costs(simulator, problem, n_points: int, seed: int = 0) -> list:
    """Mean wall time of each level on n_points random (feasible) designs (to set the costs)."""
    rng = np.random.default_rng(seed)
    points = rng.random((20 * n_points, problem.dim))
    feasibility = feasibility_function(problem) \
        if problem.optimization.get("known_constraints", True) else None
    if feasibility is not None:
        points = points[feasibility(points)]
    points = points[:n_points]
    if not len(points):
        raise ValueError("calibrate_costs: no feasible random design (check the constraints)")
    n_points = len(points)
    times = []
    for level in range(1, len(problem.levels) + 1):
        start = time.perf_counter()
        for point in points:
            simulator.evaluate(point, level)
        times.append((time.perf_counter() - start) / n_points)
        logger.info("Level %d (%s): %.4f s per evaluation", level,
                    problem.levels[level - 1].solver, times[-1])
    relative = [t / times[0] for t in times]
    logger.info("Measured relative costs: %s (configured: %s)", np.round(relative, 2).tolist(),
                problem.costs)
    return times


def verify_optimum(problem, simulator, data, model, acq, ego, run) -> bool:
    """
    Evaluates the optimum of the surrogate (effective best solution, Eq. 19) at the highest
    level if it was not (the merit may favour the cheap levels), adds it to the data and
    re-fits the model. Returns True if an evaluation was made. The verification points carry
    the metric "verification": True.
    """
    top = len(problem.levels)
    if acq.x_best is None or data.is_already_evaluated(top, acq.x_best):
        return False
    logger.info("Verification of the surrogate optimum at level %d: %s", top,
                to_physical(acq.x_best, problem.variables))
    x_best, prediction = np.array(acq.x_best, dtype=float), acq.f_best
    y_verify, metrics_verify = simulator.evaluate(x_best, top)
    metrics_verify["verification"] = True
    metrics_verify["surrogate_prediction"] = prediction
    ego.tell(x_best, top, y_verify, metrics_verify)
    logger.info("    -> verified objective: %s (surrogate prediction %.6g)", y_verify, prediction)
    model.fit(data)
    acq.update()
    ego.save_state(run.path("ego_backup.json"))
    return True


def run_optimization(problem, run, simulator=None) -> dict:
    """DOE, NN-MF-EGO loop, export and figures; returns the results dict."""
    opt = problem.optimization
    simulator = simulator or BdToolboxFoilSimulator(problem)
    num_levels = len(problem.levels)
    data = ExperimentData(bounds=[(0.0, 1.0)] * problem.dim, costs=problem.costs)
    # [MAP] MAP estimation of the hyperparameters (InvGamma lengthscale prior) unless
    # "use_map": false; the inputs are in [0, 1]^d, the domain the prior is defined for
    model = MultifidelityModel(num_levels, SquaredExponentialKernel, seed=opt["seed"],
                               estimate_rho=opt.get("estimate_rho", True),
                               min_points_rho=opt.get("min_points_rho"),
                               n_restarts=int(opt.get("n_restarts", 3)),
                               use_map=bool(opt.get("use_map", True)),
                               lengthscale_prior=opt.get("lengthscale_prior"))
    # [KC] known geometric constraints: feasible DOE, zero merit outside the feasible domain
    feasibility = feasibility_function(problem) if opt.get("known_constraints", True) else None
    acq = AcquisitionFunction(model=model, data=data, feasibility=feasibility)
    ego = EGOOptimizer(data=data, model=model, simulator=simulator, acquisition=acq,
                       save_state_path=run.path("ego_backup.json"), seed=opt["seed"])

    logger.info("Problem %s: %d variables %s, levels %s, costs %s", problem.name, problem.dim,
                [v.name for v in problem.variables], [lv.solver for lv in problem.levels],
                problem.costs)
    logger.info("GP hyperparameters: %s | known constraints: %s", model.estimator_label(),
                "on" if feasibility is not None else "off")
    data.generate_initial_design(points_per_level=opt["doe"], seed=opt["seed"],
                                 feasibility=feasibility)
    for level in range(1, num_levels + 1):
        results = [simulator.evaluate(x, level) for x in data.x_dict[level]]
        data.y_dict[level] = np.array([r[0] for r in results], dtype=float)
        data.metrics_dict[level] = [r[1] for r in results]
        logger.info("DOE level %d: %d points, %d failed", level, len(results),
                    data.n_failed(level))

    # optimization loop, in chunks of "verify_every" iterations when set: after each chunk the
    # optimum of the surrogate is evaluated at the highest level (with a large cost ratio the
    # merit function seldom chooses the expensive level by itself)
    iterations, every = int(opt["iterations"]), opt.get("verify_every")
    chunks = [iterations] if not every else \
        [int(every)] * (iterations // int(every)) + ([iterations % int(every)]
                                                    if iterations % int(every) else [])
    for k, n_iterations in enumerate(chunks, start=1):
        if len(chunks) > 1:
            logger.info("=== Optimization chunk %d/%d (%d iterations) ===", k, len(chunks),
                        n_iterations)
        n_before = sum(len(data.y_dict.get(level, [])) for level in data.y_dict)
        ego.run(n_iterations=n_iterations,
                stop_on_convergence=bool(opt.get("stop_on_convergence", False)))
        stopped = sum(len(data.y_dict.get(level, [])) for level in data.y_dict) \
            - n_before < n_iterations
        # final verification (verify_best) or periodic one (verify_every)
        if (k < len(chunks) and not stopped) or opt.get("verify_best", True):
            verify_optimum(problem, simulator, data, model, acq, ego, run)
        if stopped:
            logger.info("Stopping criterion met in chunk %d: end of the loop", k)
            break
    ego.export_surrogate(run.path("surrogate.json"))

    x_best, y_best = data.best_observation(num_levels)
    level_metrics = data.metrics_dict[num_levels]
    i_best = int(np.nanargmin(np.asarray(data.y_dict[num_levels], dtype=float)))
    results = {"problem": problem.name, "best_objective": y_best,
               "best_design_normalized": None if x_best is None else x_best.tolist(),
               "best_design_physical": None if x_best is None
                                       else to_physical(x_best, problem.variables),
               "best_metrics": level_metrics[i_best], "summary": ego.summary(log=False),
               "gp": {"estimator": model.estimator_label(), "diagnostics": model.diagnostics()},
               "environment": bridge.environment_report(simulator.paths)}
    with open(run.path("results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, cls=NumpyEncoder)
    logger.info("Best design (level %d): objective %.6g, %s", num_levels, y_best,
                results["best_design_physical"])

    visualizer = ModelVisualizer(run.path("ego_backup.json"))
    visualizer.plot_convergence(save_path=run.path("convergence_plot.png"))
    visualizer.plot_convergence_interactive(save_path=run.path("convergence_plot.html"))
    if problem.dim >= 2:
        visualizer.plot_response_surface_2d(save_path=run.path("response_surface_2d.png"))
        visualizer.plot_response_surface_2d_interactive(
            save_path=run.path("response_surface_2d.html"))
    else:
        visualizer.plot_response_1d(save_path=run.path("response_1d.png"))
        visualizer.plot_response_1d_interactive(save_path=run.path("response_1d.html"))
    return results


def main(argv: list = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="JSON configuration file")
    parser.add_argument("--iterations", type=int, help="override optimization.iterations")
    parser.add_argument("--calibrate-costs", type=int, metavar="N",
                        help="only measure the time per level on N random designs")
    parser.add_argument("--output-dir", default=str(ROOT / "pipelines"),
                        help="directory receiving runs/<timestamp>/ (default: pipelines/)")
    args = parser.parse_args(argv)

    problem = load_config(args.config)
    if args.iterations is not None:
        problem.optimization["iterations"] = args.iterations
    run = create_run(args.output_dir, prefix=problem.name)
    setup_logging(run.log_path)
    shutil.copy(args.config, run.path("config.json"))
    with RunTimer(f"mfego x bdToolbox - {problem.name}"):
        logger.info("Configuration: %s | run directory: %s", problem.source, run.run_dir)
        simulator = BdToolboxFoilSimulator(problem)
        if args.calibrate_costs:
            calibrate_costs(simulator, problem, args.calibrate_costs)
        else:
            run_optimization(problem, run, simulator)
    print(f"Run directory: {run.run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
