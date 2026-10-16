"""
STEP 1 - Multi-fidelity Bayesian optimization of the section AT THE LIFT OF THE REFERENCE FOIL.

Problem (config_cfoil_trim.json): the Intens SY C-foil, one constant Kulfan section with 8
variables (thickness t_0..t_3, camber shape s_1..s_3, yaw_target; the camber offset delta is
solved by the level-1 yaw model of step 0; no 2D lift constraint), minimum 3D drag. The
reference is the section of the first study: thickness of MC2_60_6101_1 + the frozen camber
delta C(x), delta = 0.108139. The lift is matched EXACTLY: for every section and every level the
yaw is trimmed (secant iteration, pipelines/bdtoolbox_foil/simulator.py _trim) until the total
lift |(Cy, Cz)| equals the lift of the reference section at the nominal attitude (yaw 4 deg) at
the same level; the objective is the drag CD = -Cx at the trimmed yaw. The trimmed yaw must stay
within +-5 % of 4 deg: outside, the objective is penalized (study_trim.TrimBandSimulator).

Known constraints of the section, relative to the reference (study_trim.feasibility_function,
from the aerosandbox airfoil optimization of a colleague): t/c >= 0.1225, Ixx, thickness at x/c
0.90 and 0.95, leading-edge weights, wiggliness < 2 x reference, NeuralFoil xxxlarge confidence
> 0.9. They define the feasible initial design and give a zero merit outside the feasible domain;
their feasible share of the design space is logged.

Three levels, same flow (Re 5e6, n_crit 1, forced transition at 10 % of the chord):
L1 NPLLT x NeuralFoil xxxlarge, L2 NPLLT x XFOIL, L3 AVL x XFOIL. L2 and L3 use the same XFOIL
PolarSpec, so a polar computed by one of them is read from the cache by the other. GP
hyperparameters: MAP, InvGamma(3, 2) lengthscale prior (calibration of the first study); the
optimum of the surrogate is evaluated at L3 every 10 iterations ("verify_every").

The relative cost of the levels is measured first, each level on its OWN feasible sections and
with an empty temporary XFOIL cache (every XFOIL level pays its polar, as in the initial design).
The one-off solve of the reference section (reference lift) is not included in these timings.

Outputs: runs/<MMDD_HHMMSS>/ (log, config, yaw model used, ego_backup.json, surrogate.json = the
response surface, results.json) and results/run_summary.json. Then run step2_analysis.py.

Run (conda environment bdToolbox; first step0_check_setup.py):
    C:\\Users\\SIM\\.conda\\envs\\bdToolbox\\python.exe step1_run_optimization.py
        [--doe 150 40 25] [--iterations 200] [--cost-sections 2]
"""
import argparse
import json
import logging
import shutil
import tempfile
from pathlib import Path

import numpy as np

import study_trim as st
from pipelines.bdtoolbox_foil.run import run_optimization  # pylint: disable=wrong-import-order
from src.run_utils import RunTimer, create_run

logger = logging.getLogger("step1")


def measure_costs(problem, simulator, n_sections: int, seed: int = 11) -> dict:
    """
    Wall time of one trimmed evaluation per level, each level on n feasible sections of its own
    (no polar shared between levels through the cache), and the relative costs.
    """
    points = st.feasible_design(problem, n_sections * len(st.LEVELS), seed=seed)
    times = {level: [] for level in st.LEVELS}
    failed = {level: 0 for level in st.LEVELS}
    # empty temporary XFOIL cache: a polar already in the cache of the study would hide its cost
    saved = {}
    with tempfile.TemporaryDirectory(prefix="mfcost_") as tmp:
        for backend in simulator.backends:
            if hasattr(backend, "xfoil_settings"):
                saved[id(backend)] = backend.xfoil_settings["cache_dir"]
                backend.xfoil_settings["cache_dir"] = Path(tmp)
            elif hasattr(backend, "cache_dir"):
                saved[id(backend)] = backend.cache_dir
                backend.cache_dir = Path(tmp)
        try:
            for level in st.LEVELS:
                for x in points[(level - 1) * n_sections: level * n_sections]:
                    value, metrics = simulator.evaluate(x, level)
                    if np.isfinite(value):
                        times[level].append(metrics["time_s"])
                    else:
                        failed[level] += 1
                    logger.info("cost measurement, level %d: %.2f s (CD %s, yaw %s, %s solves)",
                                level, metrics["time_s"], metrics.get("CD"),
                                metrics.get("trim_yaw"), metrics.get("trim_solves"))
        finally:
            for backend in simulator.backends:
                if hasattr(backend, "xfoil_settings"):
                    backend.xfoil_settings["cache_dir"] = saved[id(backend)]
                elif hasattr(backend, "cache_dir"):
                    backend.cache_dir = saved[id(backend)]
    per_level = {level: float(np.mean(t)) for level, t in times.items() if t}
    relative = {level: per_level[level] / per_level[1] for level in per_level} \
        if 1 in per_level else {}
    return {"seconds": per_level, "relative": relative, "n_sections": n_sections,
            "failed": failed, "seconds_all": {level: [float(v) for v in t]
                                              for level, t in times.items()}}


def run_failures(run_dir) -> dict:
    """Failed evaluations of a run per level and their causes (XFOIL, trim, other)."""
    state = st.load_json(run_dir / "ego_backup.json")
    out = {}
    for key, metrics in state.get("Metrics_dict", {}).items():
        errors = [m.get("error") for m in metrics if m and m.get("error")]
        out[key] = {"n": len(metrics), "failed": len(errors),
                    "xfoil": sum("XFOIL" in e for e in errors),
                    "trim": sum("trim" in e for e in errors),
                    "other": sum("XFOIL" not in e and "trim" not in e for e in errors),
                    "out_of_band": sum(1 for m in metrics if m and m.get("trim_in_band") is False)}
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--iterations", type=int, help="override optimization.iterations")
    parser.add_argument("--doe", type=int, nargs=3, help="override the DOE sizes (3 levels)")
    parser.add_argument("--cost-sections", type=int, default=2,
                        help="feasible sections per level used to measure the costs")
    parser.add_argument("--keep-costs", action="store_true",
                        help="keep the costs of the configuration (no measurement)")
    args = parser.parse_args(argv)

    problem = st.load_problem()
    if args.iterations is not None:
        problem.optimization["iterations"] = args.iterations
    if args.doe:
        problem.optimization["doe"] = list(args.doe)
    run = create_run(st.STUDY, prefix=problem.name)
    st.setup_logging("step1", path=run.log_path)
    simulator = st.make_simulator(problem)
    settings = st.level_settings(problem, simulator)
    feasibility = st.feasibility_function(problem)
    shares = st.feasibility_report(problem)
    # enough Latin hypercube candidates for about 4 feasible ones per initial point
    oversampling = max(20, int(np.ceil(4.0 / max(shares["all"], 1e-3))))
    logger.info("Reference %s: %s", st.BASELINE_NAME, st.baseline_parameters(problem)["physical"])
    logger.info("Flow settings per level (identical): %s", settings)
    logger.info("Yaw band %s deg, penalty %.3g CD per deg", st.yaw_band(problem), st.YAW_PENALTY)
    logger.info("Known constraints, feasible share of the design space: %s -> DOE oversampling "
                "%d", {k: round(v, 3) for k, v in shares.items()}, oversampling)
    costs = None
    if not args.keep_costs:
        costs = measure_costs(problem, simulator, args.cost_sections)
        for level, lv in enumerate(problem.levels, start=1):
            if level in costs["relative"]:
                lv.cost = round(costs["relative"][level], 2)
        logger.info("Measured costs: %s s per evaluation -> relative costs %s",
                    {k: round(v, 2) for k, v in costs["seconds"].items()}, problem.costs)
    shutil.copy(st.CONFIG, run.path("config.json"))
    # the yaw model of this run (step 2 reads it from here: delta of every design point)
    shutil.copy(st.YAW_MODEL_FILE, run.path(st.YAW_MODEL_FILE.name))
    effective = json.loads(st.CONFIG.read_text(encoding="utf-8"))
    effective["optimization"] = problem.optimization
    for raw, lv in zip(effective["levels"], problem.levels):
        raw["cost"] = lv.cost
    st.save_json(run.run_dir / "config_effective.json", effective)

    with RunTimer(f"C-foil section optimization at the reference lift - {problem.name}"):
        results = run_optimization(problem, run, simulator, feasibility=feasibility,
                                   doe_oversampling=oversampling)
    failures = run_failures(run.run_dir)
    logger.info("Failed evaluations per level: %s", failures)
    st.save_json(st.RESULTS / "run_summary.json",
                 {"run_dir": str(run.run_dir), "cost_measurement": costs,
                  "costs": problem.costs, "reference_lift": simulator.reference_lift,
                  "baseline": st.baseline_parameters(problem),
                  "reference_properties": st.reference_properties(problem),
                  "flow_settings": settings, "yaw_band": st.yaw_band(problem),
                  "yaw_penalty_per_deg": st.YAW_PENALTY, "feasible_share": shares,
                  "failures": failures, "results": results})
    print(f"Run directory: {run.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
