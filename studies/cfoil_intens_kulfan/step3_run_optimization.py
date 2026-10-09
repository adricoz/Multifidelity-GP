"""
STEP 3 - Multi-fidelity Bayesian optimization of the section (NN-MF-EGO, mfego).

The optimization itself is the generic pipeline (pipelines/bdtoolbox_foil/run.py,
run_optimization); this script only prepares it with what steps 1 and 2 measured:
* the GP estimator chosen in step 2 (results/step2_choice.json: MLE or MAP + prior);
* the relative costs of the levels measured in step 1 (results/step1_summary.json), which the
  merit function uses to decide whether a cheap or an expensive evaluation is worth it.

What happens in run_optimization (see GUIDE_PIPELINE.md for the details):
1. initial design: a feasible, space-filling set of sections per level (DOE sizes of the
   configuration), every section projected on CL2d(0) = 0.45 and evaluated;
2. loop (iterations of the configuration): fit the 3-level GP (MAP), find the section and the
   level maximizing the merit (augmented expected improvement x cost ratio x information
   ratio, zero outside the thickness constraint), evaluate it, add it to the data;
3. the optimum of the surrogate is verified at L3 (AVL x XFOIL), the trained surrogate is saved
   (surrogate.json: the response surface, reload with src.surrogate_models.load_surrogate).

After the run, the final surrogate predicts the step 1 L3 sections, which it never saw: an
independent check of the response surface (results/step3_summary.json).

Outputs: runs/<MMDD_HHMMSS>/ (log, config, ego_backup.json, surrogate.json, results.json,
convergence and response-surface figures of mfego) and results/step3_summary.json.
Run (about one hour, in the background):
    C:\\Users\\SIM\\.conda\\envs\\bdToolbox\\python.exe step3_run_optimization.py
"""
import argparse
import json
import logging
import shutil

import numpy as np

import study_lib as lib
from pipelines.bdtoolbox_foil.run import run_optimization  # pylint: disable=wrong-import-order
from src.run_utils import RunTimer, create_run
from src.surrogate_models import load_surrogate
from step2_validate_gp import load_step1, scores

logger = logging.getLogger("step3")


def apply_calibration(problem, keep_config: bool, keep_costs: bool) -> dict:
    """GP estimator of step 2 and level costs of step 1 written into the problem."""
    applied = {}
    choice_path = lib.RESULTS / "step2_choice.json"
    if not keep_config and choice_path.is_file():
        choice = lib.load_json(choice_path)
        problem.optimization["use_map"] = bool(choice["use_map"])
        if choice.get("lengthscale_prior"):
            problem.optimization["lengthscale_prior"] = list(choice["lengthscale_prior"])
        applied["gp"] = choice
    summary_path = lib.RESULTS / "step1_summary.json"
    if not keep_costs and summary_path.is_file():
        costs = lib.load_json(summary_path)["costs"]
        for level, lv in enumerate(problem.levels, start=1):
            lv.cost = round(float(costs[str(level)]["relative"]), 3)
        applied["costs"] = problem.costs
        if not all(a < b for a, b in zip(problem.costs, problem.costs[1:])):
            logger.warning("Measured costs %s are not increasing with the level: the merit "
                           "function will prefer the most accurate of the equally cheap levels",
                           problem.costs)
    return applied


def holdout(run_dir) -> dict:
    """Final surrogate (L3 prediction) on the step 1 L3 sections (never seen by the run);
    None when step 1 has not been run."""
    if not (lib.RESULTS / "step1_calibration.csv").is_file():
        logger.warning("No step 1 data: the independent check of the surrogate is skipped")
        return None
    x, values = load_step1()
    valid = np.isfinite(values[3])
    model = load_surrogate(str(run_dir / "surrogate.json"))
    mean, var, _ = model.predict_batch(x[valid])
    y = values[3][valid]
    return {"n": int(valid.sum()), "y": y.tolist(), "mean": mean.tolist(), "var": var.tolist(),
            **scores(y, mean, var, float(np.std(y)))}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--iterations", type=int, help="override optimization.iterations")
    parser.add_argument("--doe", type=int, nargs=3, help="override the DOE sizes (3 levels)")
    parser.add_argument("--keep-config", action="store_true",
                        help="keep the GP estimator of the configuration (ignore step 2)")
    parser.add_argument("--keep-costs", action="store_true",
                        help="keep the costs of the configuration (ignore step 1)")
    parser.add_argument("--costs", type=float, nargs=3,
                        help="relative costs of the 3 levels (e.g. measured without the XFOIL "
                             "polar cache), instead of the step 1 costs")
    args = parser.parse_args(argv)

    problem = lib.load_problem()
    if args.iterations is not None:
        problem.optimization["iterations"] = args.iterations
    if args.doe:
        problem.optimization["doe"] = list(args.doe)
    run = create_run(lib.STUDY, prefix=problem.name)
    lib.setup_logging("step3", path=run.log_path)
    applied = apply_calibration(problem, args.keep_config, args.keep_costs or bool(args.costs))
    if args.costs:
        for lv, cost in zip(problem.levels, args.costs):
            lv.cost = float(cost)
        applied["costs"] = problem.costs
        applied["costs_source"] = "command line (--costs)"
    logger.info("Calibration applied: %s", applied)
    shutil.copy(lib.CONFIG, run.path("config.json"))
    effective = json.loads(lib.CONFIG.read_text(encoding="utf-8"))
    effective["optimization"] = problem.optimization
    for raw, lv in zip(effective["levels"], problem.levels):
        raw["cost"] = lv.cost
    lib.save_json(run.run_dir / "config_effective.json", effective)

    with RunTimer(f"C-foil section optimization - {problem.name}"):
        simulator = lib.make_simulator(problem)
        results = run_optimization(problem, run, simulator)
    summary = {"run_dir": str(run.run_dir), "applied_calibration": applied, "results": results}
    lib.save_json(lib.RESULTS / "step3_summary.json", summary)
    check = holdout(run.run_dir)
    if check is not None:
        logger.info("Final surrogate on the %d step 1 L3 sections: RMSE/spread %.3f, NLPD %.3f, "
                    "95 %% coverage %.2f", check["n"], check["rmse_relative"], check["nlpd"],
                    check["coverage95"])
        lib.save_json(run.run_dir / "holdout_step1.json", check)
        summary["holdout_step1"] = {k: v for k, v in check.items()
                                    if k not in ("y", "mean", "var")}
        lib.save_json(lib.RESULTS / "step3_summary.json", summary)
    print(f"Run directory: {run.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
