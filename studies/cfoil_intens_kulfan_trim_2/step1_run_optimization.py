"""
STEP 1 - Multi-fidelity Bayesian optimization of the section AT THE LIFT OF THE BASELINE FOIL.

Problem (config_cfoil_trim.json): the Intens SY C-foil of the first study, one constant Kulfan
section (7 variables, CL2d(0 deg) = 0.45 by projection), minimum 3D drag. The baseline is the
NACA 4412 (initial section of the Intens foil, Kulfan fit, t/c 12 %). The lift is matched
EXACTLY: for every section and every level the yaw is trimmed (secant iteration,
pipelines/bdtoolbox_foil/simulator.py _trim) until the total lift |(Cy, Cz)| equals the lift of
the baseline section at the nominal attitude (yaw 4 deg) at the same level; the objective is the
drag CD = -Cx at the trimmed yaw.

Known constraints of the section, relative to the baseline (study_trim.feasibility_function,
transposed from the aerosandbox airfoil optimization): t/c >= 0.12, Ixx, Ixx + Iyy, thickness at
x/c 0.90 and 0.95, wiggliness < 2 x baseline. They define the feasible initial design and give a
zero merit outside the feasible domain; their feasible share of the design space is logged.

Settings taken from the calibration of the first study (no calibration step here):
* 2 levels: NPLLT x NeuralFoil xxlarge (L1 here) and AVL x XFOIL (L2 here); NeuralFoil xxsmall
  cost as much as xxlarge and ranked the sections worse;
* GP hyperparameters: MAP, InvGamma(3, 2) lengthscale prior (well calibrated with few AVL
  points, where the maximum likelihood was overconfident);
* the optimum of the surrogate is evaluated with AVL every 10 iterations ("verify_every"): with
  a cost ratio of ~50, the merit function seldom chooses the expensive level by itself.

The relative cost of the levels is measured first on a few feasible sections (the trim changes
it: usually 1-2 NPLLT solves and 1-2 AVL runs + the XFOIL polar per evaluation). The one-off
solve of the baseline section (reference lift) is not included in these timings.

Outputs: runs/<MMDD_HHMMSS>/ (log, config, ego_backup.json, surrogate.json = the response
surface, results.json) and results/run_summary.json. Then run step2_analysis.py.

Run (conda environment bdToolbox; ~0.3 s per NPLLT and 15-20 s per AVL evaluation, ~40 s for a
failed XFOIL polar, so the AVL initial design dominates; run 1012_173702: 80 + 12 points, 50
iterations and 2 cost sections in 6 min; the configuration sizes 80 + 30 and 100 iterations take
about 13 min):
    C:\\Users\\SIM\\.conda\\envs\\bdToolbox\\python.exe step1_run_optimization.py --doe 80 12
        --iterations 50 --cost-sections 2
"""
import argparse
import json
import logging
import shutil

import numpy as np

import study_trim as st
from pipelines.bdtoolbox_foil.run import run_optimization  # pylint: disable=wrong-import-order
from src.run_utils import RunTimer, create_run

logger = logging.getLogger("step1")


def measure_costs(problem, simulator, n_sections: int, seed: int = 11) -> dict:
    """
    Wall time of one trimmed evaluation per level on n feasible sections (the CL2d
    projection, paid once per section, is added to every level) and the relative costs.
    """
    points = st.feasible_design(problem, n_sections, seed=seed)
    times = {level: [] for level in st.LEVELS}
    projection = []
    for x in points:
        for level in st.LEVELS:
            value, metrics = simulator.evaluate(x, level)
            solver = metrics["time_s"] - metrics.get("time_geometry_s", 0.0)
            if level == 1:
                projection.append(metrics.get("time_geometry_s", 0.0))
            if np.isfinite(value):
                times[level].append(solver)
            logger.info("cost measurement, level %d: %.2f s (CD %s, %s solves)", level,
                        metrics["time_s"], value, metrics.get("trim_solves"))
    per_level = {level: float(np.mean(t)) + float(np.mean(projection))
                 for level, t in times.items() if t}
    relative = {level: per_level[level] / per_level[1] for level in per_level}
    return {"seconds": per_level, "relative": relative, "n_sections": n_sections,
            "solver_seconds": {level: [float(v) for v in t] for level, t in times.items()},
            "projection_seconds": [float(v) for v in projection]}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--iterations", type=int, help="override optimization.iterations")
    parser.add_argument("--doe", type=int, nargs=2, help="override the DOE sizes (2 levels)")
    parser.add_argument("--cost-sections", type=int, default=3,
                        help="random sections used to measure the cost of the levels")
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
    feasibility = st.feasibility_function(problem)
    shares = st.feasibility_report(problem)
    # enough Latin hypercube candidates for about 4 feasible ones per initial point
    oversampling = max(20, int(np.ceil(4.0 / max(shares["all"], 1e-3))))
    logger.info("Baseline %s: %s", st.BASELINE_NAME, st.baseline_parameters(problem)["physical"])
    logger.info("Known constraints, feasible share of the design space: %s -> DOE oversampling "
                "%d", {k: round(v, 3) for k, v in shares.items()}, oversampling)
    costs = None
    if not args.keep_costs:
        costs = measure_costs(problem, simulator, args.cost_sections)
        for level, lv in enumerate(problem.levels, start=1):
            lv.cost = round(costs["relative"][level], 2)
        logger.info("Measured costs: %s s per evaluation -> relative costs %s",
                    {k: round(v, 2) for k, v in costs["seconds"].items()}, problem.costs)
    shutil.copy(st.CONFIG, run.path("config.json"))
    effective = json.loads(st.CONFIG.read_text(encoding="utf-8"))
    effective["optimization"] = problem.optimization
    for raw, lv in zip(effective["levels"], problem.levels):
        raw["cost"] = lv.cost
    st.save_json(run.run_dir / "config_effective.json", effective)

    with RunTimer(f"C-foil section optimization at the baseline lift - {problem.name}"):
        results = run_optimization(problem, run, simulator, feasibility=feasibility,
                                   doe_oversampling=oversampling)
    st.save_json(st.RESULTS / "run_summary.json",
                 {"run_dir": str(run.run_dir), "cost_measurement": costs,
                  "costs": problem.costs, "reference_lift": simulator.reference_lift,
                  "baseline": st.baseline_parameters(problem), "feasible_share": shares,
                  "results": results})
    print(f"Run directory: {run.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
