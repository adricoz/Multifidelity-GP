"""
Validation of the sequential co-kriging (src/sequential_design.py) on two harder problems, at
EQUAL COST, with the problems and the test sets of benchmarks/cokriging:
  - hartmann: Hartmann 6D (benchmarks/bench_lib.HartmannMF, Sacher Eqs. 31-32). The low fidelity
    is a nonlinear transform of the high fidelity (one Newton step of a square root, shifted
    inputs): the auto-regressive model is misspecified, rho is about 2.4.
  - airfoil: NACA 4-digit drag, 4D (benchmarks/cokriging/airfoil_problem.AirfoilMF), NeuralFoil
    "xxsmall" (low fidelity) against "xxxlarge" (high fidelity): strongly correlated levels.
Costs 1 (low fidelity) and 10 (high fidelity) for both.

Three strategies of the same total cost, 10 seeds:
  1. sequential, adjusted variance (adjust=True): nested start (default 40 LF + 10 HF, cost 140,
     the smallest design of benchmarks/cokriging), then a budget of new runs chosen by the
     algorithm (default 140);
  2. sequential, plain variance (adjust=False): same start, same budget;
  3. one-shot nested Latin hypercube of the same total cost (default 80 LF + 20 HF = 280).
Options: --point variance (maximum of the variance, then the level: Le Gratiet & Cannamela) or
alc (point and level chosen together by the integrated gain per unit of cost); --rho v3 (default
of CoKriging, rho learned under a lognormal prior) or v2 (rho bounded to +-5, N(1, 2^2) prior);
--start, --budget, --lhs: sizes of the designs. Accuracy of the high-fidelity prediction on the
fixed test sets of benchmarks/cokriging/results (2000 points for Hartmann, 1000 for the airfoil).

Run from the repository root (Multifidelity-GP-Clean), a few minutes on 32 cores:
    .venv-benchmark\\Scripts\\python vpp_acceleration\\examples\\benchmark_study.py --point alc --rho v2
    .venv-benchmark\\Scripts\\python vpp_acceleration\\examples\\benchmark_study.py --point alc --rho v2 ^
        --start 80,20 --budget 280 --lhs 160,40
Outputs (the name carries the options): examples/results/benchmark_study_<tag>.csv,
benchmark_curves_<tag>.csv, benchmark_summary_<tag>.txt, examples/figures/benchmark_study_<tag>.html
"""
import os

# one BLAS thread per process: the runs are parallelized over the processes instead
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

# pylint: disable=wrong-import-position
import argparse  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent             # .../vpp_acceleration/examples
VPP_DIR = HERE.parent                               # .../vpp_acceleration
REPO_DIR = VPP_DIR.parent                           # .../Multifidelity-GP-Clean
for path in (REPO_DIR / "mfego", VPP_DIR, REPO_DIR / "benchmarks",
             REPO_DIR / "benchmarks" / "cokriging"):
    if str(path) not in sys.path:
        sys.path.append(str(path))

# pylint: disable=import-error
import bench_lib  # noqa: E402
from airfoil_problem import AirfoilMF  # noqa: E402
from src.cokriging import CoKriging  # noqa: E402
from src.data_management import ExperimentData  # noqa: E402
from src.sequential_design import SequentialCoKriging, evaluate_initial_design  # noqa: E402
from src.simulator import BaseSimulator  # noqa: E402
from src.validation import prediction_scores  # noqa: E402

COSTS = [1.0, 10.0]
PROBLEMS = ("hartmann", "airfoil")
VARIANTS = ("adjusted", "plain", "lhs")
LABELS = {"adjusted": "séquentiel, variance ajustée", "plain": "séquentiel, variance seule",
          "lhs": "LHS imbriqué (coût égal)"}
# outputs of the co-kriging benchmarks (2026-10-21): one run folder, runs/README.md
RUN_DIR = VPP_DIR / "runs" / "2026-10-21_cokriging_benchmarks"
RESULTS = RUN_DIR / "results"
FIGURES = RUN_DIR / "figures"
# estimation of rho (option --rho): v3 = default of CoKriging (rho learned with the kernel under a
# lognormal prior, bounds [1e-3, 50]); v2 = rho in closed form under a N(1, 2^2) prior, bounds +-5
RHO_PRIORS = {"v3": None, "v2": {"rho": ("normal", 1.0, 2.0)}}
RHO_TEXT = {"v3": "ρ appris, prior lognormale (v3)", "v2": "ρ borné à ±5 (v2)"}
POINT_TEXT = {"variance": "point au maximum de variance (LG)", "alc": "point et niveau par l'ALC"}


def make_model(rho_setting, share_lengthscales: bool = False):
    """New CoKriging (mfego defaults: Matern 5/2, MAP) with the chosen estimation of rho;
    share_lengthscales: the correction takes the lengthscales of the LF level (joint_study.py)."""
    return CoKriging(2, priors=RHO_PRIORS[rho_setting], seed=0,
                     share_lengthscales=share_lengthscales)


def make_problem(name):
    """The multi-fidelity problem (inputs in the unit cube, vectorized: problem(x, level))."""
    return bench_lib.HartmannMF() if name == "hartmann" else AirfoilMF()


class ProblemSimulator(BaseSimulator):
    """A benchmark problem in the simulator format of mfego: evaluate(x, level) for one point."""

    def __init__(self, problem):
        super().__init__(num_levels=problem.num_levels)
        self.problem = problem

    def evaluate(self, design_point, level):
        """Value of the problem at one point and one level (no metrics)."""
        value = self.problem(np.atleast_2d(design_point), level)[0]
        return float(value), {}


def test_set(name):
    """Fixed test set of benchmarks/cokriging (same points as the previous reports)."""
    data = np.load(REPO_DIR / "benchmarks" / "cokriging" / "results" / f"test_set_{name}.npz")
    return data["x"], data["y"]


def nested_design(problem, points_per_level, seed):
    """Nested Latin hypercube X_2 in X_1 (as bench_lib.nested_doe), evaluated."""
    data = ExperimentData(bounds=[(0.0, 1.0)] * problem.dim, costs=COSTS)
    data.generate_nested_design(list(points_per_level), seed=seed * 100)
    evaluate_initial_design(data, ProblemSimulator(problem))
    return data


def design_cost(points_per_level):
    """Cost of a design: number of points of each level times its cost."""
    return float(sum(n * c for n, c in zip(points_per_level, COSTS)))


def scores(model, x_test, y_test):
    """Accuracy and calibration of the high-fidelity prediction on the test set."""
    mean, variance = model.predict_batch(x_test, include_noise=False)
    return prediction_scores(y_test, mean, variance)


def run_task(task):
    """
    One run. task = (problem name, seed, variant, settings), settings = dict with rho, point,
    start, budget, lhs (passed with the task: the worker processes do not see the arguments).

    Returns:
    - final scores (dict), error after each iteration (list of dicts).
    """
    name, seed, variant, settings = task
    problem = make_problem(name)
    x_test, y_test = test_set(name)
    start = time.perf_counter()
    curve, ask_times = [], []
    if variant == "lhs":
        data = nested_design(problem, settings["lhs"], seed)
        model = make_model(settings["rho"])
        model.fit(data)
    else:
        data = nested_design(problem, settings["start"], seed)
        initial_cost = design_cost(settings["start"])
        design = SequentialCoKriging(data, ProblemSimulator(problem),
                                     model_factory=lambda: make_model(settings["rho"]),
                                     adjust=(variant == "adjusted"),
                                     point_criterion=settings["point"],
                                     n_candidates=settings["n_candidates"], seed=seed)
        while True:
            remaining = settings["budget"] - design.cost_spent
            if remaining < min(COSTS):
                break
            ask_start = time.perf_counter()
            proposal = design.ask(remaining)           # fits the model on the current data
            ask_times.append(time.perf_counter() - ask_start)
            curve.append((initial_cost + design.cost_spent,
                          scores(design.model, x_test, y_test)["rmse_rel"]))
            if proposal["level"] is None:
                break
            design.evaluate(proposal)
        model = design.fit()
        curve.append((initial_cost + design.cost_spent, scores(model, x_test, y_test)["rmse_rel"]))
    n_points = [len(data.get_training_data(l)[1]) for l in (1, 2)]
    row = {"problem": name, "seed": seed, "variant": variant, "rho_setting": settings["rho"],
           "point_criterion": settings["point"], "total_cost": design_cost(n_points),
           "n_lf": n_points[0], "n_hf": n_points[1], "rho": float(model.rhos[0]),
           "iterations": len(ask_times),
           "ask_time_median_s": float(np.median(ask_times)) if ask_times else np.nan,
           "wall_time_s": time.perf_counter() - start, **scores(model, x_test, y_test)}
    curve_rows = [{"problem": name, "seed": seed, "variant": variant, "cost": c, "rmse_rel": e}
                  for c, e in curve]
    return row, curve_rows


def summarize(results, total_cost):
    """Medians and 10-90 % quantiles over the seeds, and the count of wins against the LHS."""
    lines = []
    for name in results["problem"].unique():
        part = results[results["problem"] == name]
        lines.append(f"=== {name} (total cost {total_cost:.0f}, {part['seed'].nunique()} seeds) ===")
        for variant in VARIANTS:
            v = part[part["variant"] == variant]
            if v.empty:
                continue
            lines.append(
                f"{LABELS[variant]:30s} RMSE rel {v['rmse_rel'].median():.4f} "
                f"[{v['rmse_rel'].quantile(0.1):.4f} ; {v['rmse_rel'].quantile(0.9):.4f}] | "
                f"Q2 {v['q2'].median():.3f} | cov95 {v['coverage_95'].median():.2f} | "
                f"NLPD {v['nlpd'].median():.2f} | LF/HF {v['n_lf'].median():.0f}/"
                f"{v['n_hf'].median():.0f} | rho {v['rho'].median():.2f} | "
                f"ask {v['ask_time_median_s'].median():.2f} s")
        table = part.pivot(index="seed", columns="variant", values="rmse_rel")
        for variant in ("adjusted", "plain"):
            if variant in table and "lhs" in table:
                wins = int((table[variant] < table["lhs"]).sum())
                lines.append(f"  {LABELS[variant]}: better than the LHS on {wins} / "
                             f"{len(table)} seeds")
        lines.append("")
    return "\n".join(lines)


def parse_sizes(text):
    """'40,10' -> [40, 10] (points of the low and high fidelity)."""
    return [int(n) for n in text.split(",")]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="2 seeds only")
    parser.add_argument("--workers", type=int, default=min(30, os.cpu_count() or 1))
    parser.add_argument("--rho", choices=sorted(RHO_PRIORS), default="v3")
    parser.add_argument("--point", choices=("variance", "alc"), default="variance")
    parser.add_argument("--start", type=parse_sizes, default=[40, 10])
    parser.add_argument("--budget", type=float, default=140.0)
    parser.add_argument("--lhs", type=parse_sizes, default=[80, 20])
    parser.add_argument("--problems", default=",".join(PROBLEMS),
                        help="comma-separated list among " + ", ".join(PROBLEMS))
    parser.add_argument("--seeds", type=int, default=10, help="number of seeds (0, 1, ...)")
    parser.add_argument("--n-candidates", type=int, default=2048,
                        help="Sobol candidates of the ALC search at each iteration")
    parser.add_argument("--figure-only", action="store_true",
                        help="redraw the figure from the CSV files of a previous run")
    args = parser.parse_args()
    problems = tuple(args.problems.split(","))
    settings = {"rho": args.rho, "point": args.point, "start": args.start,
                "budget": args.budget, "lhs": args.lhs, "n_candidates": args.n_candidates}
    tag = (f"_rho_{args.rho}" + ("" if args.point == "variance" else f"_{args.point}")
           + ("" if args.start == [40, 10] else f"_start_{args.start[0]}_{args.start[1]}")
           + ("" if problems == PROBLEMS else "_" + "_".join(problems))
           + ("" if args.seeds == 10 else f"_{args.seeds}seeds")
           + ("" if args.n_candidates == 2048 else f"_{args.n_candidates}cand")
           + ("_quick" if args.quick else ""))
    total_cost = design_cost(args.start) + args.budget
    if abs(design_cost(args.lhs) - total_cost) > 1e-9:
        raise ValueError(f"The LHS {args.lhs} costs {design_cost(args.lhs)}, the sequential "
                         f"strategies {total_cost}: the comparison would not be at equal cost.")
    if args.figure_only:
        make_figure(pd.read_csv(RESULTS / f"benchmark_study{tag}.csv"),
                    pd.read_csv(RESULTS / f"benchmark_curves{tag}.csv"), tag, settings)
        return
    seeds = range(2) if args.quick else range(args.seeds)
    tasks = [(name, seed, variant, settings)
             for name in problems for seed in seeds for variant in VARIANTS]
    rows, curves = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for row, curve_rows in pool.map(run_task, tasks):
            rows.append(row)
            curves += curve_rows
            print(f"{row['problem']:8s} seed {row['seed']} {row['variant']:8s} RMSE rel "
                  f"{row['rmse_rel']:.4f} | LF/HF {row['n_lf']}/{row['n_hf']} | "
                  f"{row['wall_time_s']:.0f} s", flush=True)
    results, curves = pd.DataFrame(rows), pd.DataFrame(curves)
    RESULTS.mkdir(exist_ok=True)
    results.to_csv(RESULTS / f"benchmark_study{tag}.csv", index=False)
    curves.to_csv(RESULTS / f"benchmark_curves{tag}.csv", index=False)
    summary = summarize(results, total_cost)
    (RESULTS / f"benchmark_summary{tag}.txt").write_text(summary, encoding="utf-8")
    print("\n" + summary)
    make_figure(results, curves, tag, settings)


def make_figure(results, curves, tag, settings):
    """One panel per problem: relative RMSE against the total cost (median of the seeds) for the
    two sequential strategies, and the one-shot LHS at the same total cost (bar: 10 % - 90 %)."""
    # imported here: the worker processes do not need plotly
    import plotly.graph_objects as go  # noqa: PLC0415
    from figure_style import SERIES, save, style  # noqa: PLC0415
    from plotly.subplots import make_subplots  # noqa: PLC0415

    initial_cost = design_cost(settings["start"])
    total_cost = initial_cost + settings["budget"]
    colors = {"adjusted": SERIES[0], "plain": SERIES[1], "lhs": SERIES[2]}
    titles = {"hartmann": "Hartmann 6D (BF non linéaire, ρ ≈ 2,4)",
              "airfoil": "Profil NACA 4D (NeuralFoil xxsmall → xxxlarge)"}
    problems = [p for p in PROBLEMS if p in set(results["problem"])]
    fig = make_subplots(rows=1, cols=len(problems), horizontal_spacing=0.1,
                        subplot_titles=[titles[p] for p in problems])
    cost_grid = np.arange(initial_cost, total_cost + 1.0, 1.0)
    for col, name in enumerate(problems, start=1):
        for variant in ("adjusted", "plain"):
            subset = curves[(curves["problem"] == name) & (curves["variant"] == variant)]
            per_seed = []
            for _, group in subset.groupby("seed"):
                group = group.sort_values("cost")
                # error of the last model fitted at a cost <= c (step function of the cost)
                index = np.searchsorted(group["cost"].to_numpy(), cost_grid, side="right") - 1
                per_seed.append(group["rmse_rel"].to_numpy()[np.maximum(index, 0)])
            if not per_seed:
                continue
            median = np.median(np.array(per_seed), axis=0)
            fig.add_trace(go.Scatter(x=cost_grid, y=median, name=LABELS[variant],
                                     legendgroup=variant, showlegend=(col == 1),
                                     line={"color": colors[variant], "width": 2, "shape": "hv"},
                                     hovertemplate="coût %{x:.0f} : RMSE relative %{y:.4f}"
                                                   "<extra>" + LABELS[variant] + "</extra>"),
                          row=1, col=col)
        lhs = results[(results["problem"] == name) & (results["variant"] == "lhs")]["rmse_rel"]
        if not lhs.empty:
            fig.add_trace(go.Scatter(
                x=[total_cost], y=[lhs.median()], mode="markers", name=LABELS["lhs"],
                legendgroup="lhs", showlegend=(col == 1),
                marker={"size": 11, "color": colors["lhs"], "symbol": "square"},
                error_y={"type": "data", "symmetric": False, "color": colors["lhs"],
                         "array": [lhs.quantile(0.9) - lhs.median()],
                         "arrayminus": [lhs.median() - lhs.quantile(0.1)]},
                hovertemplate="RMSE relative médiane %{y:.4f} (barre : 10 % - 90 %)"
                              "<extra>LHS imbriqué</extra>"), row=1, col=col)
        fig.update_xaxes(title_text="coût total (BF = 1, HF = 10)", row=1, col=col)
        fig.update_yaxes(title_text="RMSE / écart-type de la HF", rangemode="tozero",
                         row=1, col=col)
    start, lhs_design = settings["start"], settings["lhs"]
    style(fig, f"Séquentiel contre LHS imbriqué, à coût égal — {POINT_TEXT[settings['point']]}, "
               f"{RHO_TEXT[settings['rho']]}",
          f"Départ {start[0]} BF + {start[1]} HF (coût {initial_cost:.0f}), budget de "
          f"{settings['budget']:.0f} choisi par l'algorithme ; LHS {lhs_design[0]} BF + "
          f"{lhs_design[1]} HF (coût {total_cost:.0f}) ; médiane de {results['seed'].nunique()} "
          "graines, jeux de test de benchmarks/cokriging", height=560, legend_rows=1)
    print(f"figure: {save(fig, FIGURES, 'benchmark_study' + tag)}")


if __name__ == "__main__":
    main()
