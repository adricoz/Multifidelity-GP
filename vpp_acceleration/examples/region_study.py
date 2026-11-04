"""
Goal-oriented test of the sequential co-kriging ("VPP-like"), protocol validated by the user on
2026-10-21 (20 seeds).

Question: when we know where the VPP will query the surrogate, does the sequential design put
its runs there, and does it then beat a Latin hypercube of the same cost?

Problem: NACA 4-digit drag, 4D (benchmarks/cokriging/airfoil_problem.AirfoilMF), NeuralFoil
"xxsmall" (low fidelity, cost 1) against "xxxlarge" (high fidelity, cost 10).

Target region = "trim states": the points (m, p, t, alpha) where the LOW-fidelity lift coefficient
is between 0.4 and 0.6, the equivalent of the equilibrium incidences of the VPP over a range of
conditions. It exists for every section, alpha between -1.6 and 5.7 deg (about 1/5 of the domain).
For each section, the alpha interval of the region is found on a grid of 41 incidences.
  - integration points of S1: 1024 points of the region;
  - test set: 1000 other points of the region, evaluated at the high fidelity (main metric);
    the global test set of benchmarks/cokriging (secondary metric).

Strategies, same total cost 560:
  S1  sequential, ALC, integration points = the region (goal-oriented), rho bounded (v2), plain
      variance: nested start 80 LF + 20 HF over the whole domain (cost 280), then 280 of budget;
  S2  same as S1 with the integration points over the whole cube (isolates the effect of the target);
  B1  nested Latin hypercube over the whole domain, 160 LF + 40 HF (without knowing the region);
  B2  same with another low / high fidelity ratio, 240 LF + 32 HF;
  B3  space-filling nested design INSIDE the region, 160 LF + 40 HF ("oracle" that knows the
      region in advance: reference bound).
Paired seeds: S1 and S2 share their start, the designs of B1..B3 depend on the seed only.

Success criterion, fixed before the computation: S1 reduces the median error on the region by at
least 20 % against B1, wins at least 80 % of the seeds (16 / 20; one-sided paired Wilcoxon test
p < 0.05) and its 95 % coverage on the region is at least 0.90. S1 against B3 is reported without
criterion.

Run from the repository root (Multifidelity-GP-Clean), about 30 to 60 min:
    .venv-benchmark\\Scripts\\python vpp_acceleration\\examples\\region_study.py
    .venv-benchmark\\Scripts\\python vpp_acceleration\\examples\\region_study.py --seeds 2   # check
Outputs: examples/results/region_study.csv, region_curves.csv, region_summary.txt,
region_points.npz (region data, computed once) and examples/figures/region_study.html
"""
import os

# one BLAS thread per process: the runs are parallelized over the processes instead
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

# pylint: disable=wrong-import-position,import-error
import argparse  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import qmc, wilcoxon  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import benchmark_study as bs  # noqa: E402  (paths, problem, simulator, model, scores)
from airfoil_problem import FLOW, LOWER, UPPER, kulfan_parameters  # noqa: E402
from src.data_management import ExperimentData, greedy_maximin  # noqa: E402
from src.sequential_design import SequentialCoKriging, evaluate_initial_design  # noqa: E402

CL_RANGE = (0.4, 0.6)                    # lift coefficients of the target region
ALPHA_GRID = np.linspace(0.0, 1.0, 41)   # incidences (unit cube) used to find the region
N_INTEGRATION, N_TEST = 1024, 1000
START, BUDGET = [80, 20], 280.0          # sequential strategies: cost 280 + 280 = 560
LHS_DESIGNS = {"B1": [160, 40], "B2": [240, 32], "B3": [160, 40]}
STRATEGIES = ("S1", "S2", "B1", "B2", "B3")
LABELS = {"S1": "S1 séquentiel ALC orienté région", "S2": "S2 séquentiel ALC global",
          "B1": "B1 LHS imbriqué, tout le domaine (160/40)",
          "B2": "B2 LHS imbriqué, tout le domaine (240/32)",
          "B3": "B3 plan dans la région (oracle, 160/40)",
          "S3": "S3 = S1 avec les longueurs de la BF pour la correction"}
REGION_FILE = bs.RESULTS / "region_points.npz"
UNIT_BOUNDS = [(0.0, 1.0)] * 4


# 1/3 ---------------------------------------------------------------------------------------------
# The target region
# -------------------------------------------------------------------------------------------------
def lift_coefficient(x_unit, model_size="xxsmall"):
    """Lift coefficient by NeuralFoil of the points x (unit cube, shape (m, 4)), as
    airfoil_problem.neuralfoil_cd does for the drag (same sections, same flow)."""
    import neuralfoil as nf  # noqa: PLC0415
    physical = LOWER + np.atleast_2d(x_unit) * (UPPER - LOWER)
    params = [kulfan_parameters(*(round(float(v), 12) for v in row[:3])) for row in physical]
    kulfan = {key: np.array([p[key] for p in params]).T for key in params[0]}
    aero = nf.get_aero_from_kulfan_parameters(kulfan, alpha=physical[:, 3],
                                              model_size=model_size, **FLOW)
    return np.asarray(aero["CL"], dtype=float).reshape(-1)


def alpha_interval(sections):
    """
    For each section (m, p, t in the unit cube), the interval of incidence (unit cube) where the
    low-fidelity lift is in CL_RANGE, read on a grid of 41 incidences.

    Returns:
    - low, high: numpy arrays (n,), in [0, 1].
    """
    n = len(sections)
    grid = np.column_stack([np.repeat(sections, len(ALPHA_GRID), axis=0),
                            np.tile(ALPHA_GRID, n)])
    lift = lift_coefficient(grid).reshape(n, len(ALPHA_GRID))
    lift = np.maximum.accumulate(lift, axis=1)     # increasing envelope (np.interp needs it)
    low = np.array([np.interp(CL_RANGE[0], row, ALPHA_GRID) for row in lift])
    high = np.array([np.interp(CL_RANGE[1], row, ALPHA_GRID) for row in lift])
    return low, high


def region_points(n, seed):
    """n points of the region: Sobol sections, and for each of them an incidence drawn uniformly
    in its interval (every section counts the same, as designs explored by an optimizer)."""
    sobol = qmc.Sobol(d=4, scramble=True, seed=seed).random(n)
    low, high = alpha_interval(sobol[:, :3])
    return np.column_stack([sobol[:, :3], low + sobol[:, 3] * (high - low)])


def in_region(x_unit):
    """True for the points whose low-fidelity lift is in CL_RANGE."""
    lift = lift_coefficient(x_unit)
    return (lift >= CL_RANGE[0]) & (lift <= CL_RANGE[1])


def region_data():
    """Integration points, test set (high fidelity) and volume of the region, computed once and
    stored in REGION_FILE (the worker processes read the file)."""
    if REGION_FILE.exists():
        stored = np.load(REGION_FILE)
        return {key: stored[key] for key in stored.files}
    start = time.perf_counter()
    problem = bs.make_problem("airfoil")
    x_int = region_points(N_INTEGRATION, seed=11)
    x_test = region_points(N_TEST, seed=12)
    y_test = problem(x_test, 2)
    sections = qmc.Sobol(d=3, scramble=True, seed=13).random(1024)
    low, high = alpha_interval(sections)
    data = {"x_int": x_int, "x_test": x_test, "y_test": y_test,
            "volume_fraction": np.array(np.mean(high - low))}
    bs.RESULTS.mkdir(exist_ok=True)
    np.savez(REGION_FILE, **data)
    print(f"region data computed in {time.perf_counter() - start:.0f} s, region = "
          f"{100 * data['volume_fraction']:.0f} % of the domain")
    return data


# 2/3 ---------------------------------------------------------------------------------------------
# One run
# -------------------------------------------------------------------------------------------------
def design_in_region(points_per_level, seed):
    """Space-filling nested design inside the region (B3): 2048 region points, a greedy maximin
    subset for the low fidelity, then a greedy maximin subset of it for the high fidelity."""
    pool = region_points(2048, seed=1000 + seed)
    x_low = greedy_maximin(pool, points_per_level[0], UNIT_BOUNDS)
    x_high = greedy_maximin(x_low, points_per_level[1], UNIT_BOUNDS)
    data = ExperimentData(bounds=UNIT_BOUNDS, costs=bs.COSTS)
    data.x_dict = {1: x_low, 2: x_high}
    evaluate_initial_design(data, bs.ProblemSimulator(bs.make_problem("airfoil")))
    return data


def all_scores(model, region, global_test):
    """Scores on the region (main) and on the global test set (secondary)."""
    s_region = bs.scores(model, region["x_test"], region["y_test"])
    s_global = bs.scores(model, *global_test)
    return {"rmse_region": s_region["rmse_rel"], "q2_region": s_region["q2"],
            "coverage_region": s_region["coverage_95"], "nlpd_region": s_region["nlpd"],
            "rmse_global": s_global["rmse_rel"], "coverage_global": s_global["coverage_95"]}


def run_task(task):
    """One run (strategy, seed): final scores, error after each iteration."""
    strategy, seed = task
    start = time.perf_counter()
    region = region_data()
    global_test = bs.test_set("airfoil")
    problem = bs.make_problem("airfoil")
    curve = []
    if strategy in ("S1", "S2", "S3"):
        # S3 (share_study.py): S1 whose correction takes the lengthscales of the LF level
        share = strategy == "S3"
        data = bs.nested_design(problem, START, seed)
        design = SequentialCoKriging(
            data, bs.ProblemSimulator(problem),
            model_factory=lambda: bs.make_model("v2", share_lengthscales=share),
            adjust=False, point_criterion="alc", n_candidates=1024,
            integration_points=None if strategy == "S2" else region["x_int"], seed=seed)
        while True:
            remaining = BUDGET - design.cost_spent
            if remaining < min(bs.COSTS):
                break
            proposal = design.ask(remaining)          # fits the model on the current data
            curve.append({"cost": bs.design_cost(START) + design.cost_spent,
                          **all_scores(design.model, region, global_test)})
            if proposal["level"] is None:
                break
            design.evaluate(proposal)
        model = design.fit()
        curve.append({"cost": bs.design_cost(START) + design.cost_spent,
                      **all_scores(model, region, global_test)})
        new_points = np.atleast_2d(data.x_dict[1])[START[0]:]
    else:
        if strategy == "B3":
            data = design_in_region(LHS_DESIGNS["B3"], seed)
        else:
            data = bs.nested_design(problem, LHS_DESIGNS[strategy], seed)
        model = bs.make_model("v2")
        model.fit(data)
        new_points = np.atleast_2d(data.x_dict[1])
    n_points = [len(data.get_training_data(l)[1]) for l in (1, 2)]
    row = {"strategy": strategy, "seed": seed, "total_cost": bs.design_cost(n_points),
           "n_lf": n_points[0], "n_hf": n_points[1], "rho": float(model.rhos[0]),
           "fraction_in_region": float(np.mean(in_region(new_points))) if len(new_points) else
           np.nan, "wall_time_s": time.perf_counter() - start,
           **all_scores(model, region, global_test)}
    return row, [{"strategy": strategy, "seed": seed, **point} for point in curve]


# 3/3 ---------------------------------------------------------------------------------------------
# Summary, criterion, figure
# -------------------------------------------------------------------------------------------------
def summarize(results):
    """Medians, paired comparisons and the success criterion fixed before the computation."""
    n_seeds = results["seed"].nunique()
    lines = [f"=== Region test (CL_LF in {CL_RANGE}), total cost 560, {n_seeds} seeds ==="]
    for strategy in STRATEGIES:
        r = results[results["strategy"] == strategy]
        if r.empty:
            continue
        lines.append(
            f"{LABELS[strategy]:44s} region RMSE {r['rmse_region'].median():.4f} "
            f"[{r['rmse_region'].quantile(0.1):.4f} ; {r['rmse_region'].quantile(0.9):.4f}] | "
            f"cov95 {r['coverage_region'].median():.2f} | NLPD {r['nlpd_region'].median():.2f} | "
            f"global RMSE {r['rmse_global'].median():.4f} | LF/HF {r['n_lf'].median():.0f}/"
            f"{r['n_hf'].median():.0f} | in region {100 * r['fraction_in_region'].median():.0f} %")
    table = results.pivot(index="seed", columns="strategy", values="rmse_region")
    lines.append("")
    for other in ("B1", "B2", "B3", "S2"):
        if "S1" in table and other in table:
            wins = int((table["S1"] < table[other]).sum())
            p_value = wilcoxon(table["S1"], table[other], alternative="less").pvalue
            change = 1.0 - table["S1"].median() / table[other].median()
            lines.append(f"S1 against {other}: median error {100 * change:+.0f} % lower | wins "
                         f"{wins} / {len(table)} | one-sided paired Wilcoxon p = {p_value:.2g}")
    if "S1" in table and "B1" in table:
        s1 = results[results["strategy"] == "S1"]
        reduction = 1.0 - table["S1"].median() / table["B1"].median()
        wins = int((table["S1"] < table["B1"]).sum())
        p_value = wilcoxon(table["S1"], table["B1"], alternative="less").pvalue
        coverage = s1["coverage_region"].median()
        checks = {"median error at least 20 % lower than B1": reduction >= 0.20,
                  f"wins at least 80 % of the seeds ({wins} / {len(table)})":
                      wins >= 0.8 * len(table),
                  f"Wilcoxon p < 0.05 (p = {p_value:.2g})": p_value < 0.05,
                  f"coverage on the region at least 0.90 ({coverage:.2f})": coverage >= 0.90}
        lines.append("\nSuccess criterion fixed before the computation:")
        lines += [f"  [{'OK' if ok else 'NO'}] {text}" for text, ok in checks.items()]
        lines.append(f"  => {'SUCCESS' if all(checks.values()) else 'FAILURE'}")
    return "\n".join(lines)


def make_figure(results, curves):
    """Error on the region (left) and on the whole domain (right) against the cost: median of the
    seeds for S1 and S2 (lines), the one-shot designs at cost 560 (markers, bar 10 % - 90 %)."""
    import plotly.graph_objects as go  # noqa: PLC0415
    from figure_style import SERIES, save, style  # noqa: PLC0415
    from plotly.subplots import make_subplots  # noqa: PLC0415

    colors = dict(zip(STRATEGIES, SERIES))
    symbols = {"B1": "square", "B2": "diamond", "B3": "triangle-up"}
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.1, subplot_titles=(
        "Erreur dans la région cible (états de trim)", "Erreur sur tout le domaine"))
    start_cost = bs.design_cost(START)
    cost_grid = np.arange(start_cost, start_cost + BUDGET + 1.0, 1.0)
    for col, metric in ((1, "rmse_region"), (2, "rmse_global")):
        for strategy in ("S1", "S2"):
            per_seed = []
            for _, group in curves[curves["strategy"] == strategy].groupby("seed"):
                group = group.sort_values("cost")
                index = np.searchsorted(group["cost"].to_numpy(), cost_grid, side="right") - 1
                per_seed.append(group[metric].to_numpy()[np.maximum(index, 0)])
            if not per_seed:
                continue
            fig.add_trace(go.Scatter(
                x=cost_grid, y=np.median(np.array(per_seed), axis=0), name=LABELS[strategy],
                legendgroup=strategy, showlegend=(col == 1),
                line={"color": colors[strategy], "width": 2, "shape": "hv"},
                hovertemplate="coût %{x:.0f} : RMSE relative %{y:.4f}<extra>"
                              + LABELS[strategy] + "</extra>"), row=1, col=col)
        for strategy in ("B1", "B2", "B3"):
            values = results[results["strategy"] == strategy][metric]
            if values.empty:
                continue
            fig.add_trace(go.Scatter(
                x=[start_cost + BUDGET], y=[values.median()], mode="markers",
                name=LABELS[strategy], legendgroup=strategy, showlegend=(col == 1),
                marker={"size": 11, "color": colors[strategy], "symbol": symbols[strategy]},
                error_y={"type": "data", "symmetric": False, "color": colors[strategy],
                         "array": [values.quantile(0.9) - values.median()],
                         "arrayminus": [values.median() - values.quantile(0.1)]},
                hovertemplate="RMSE relative médiane %{y:.4f} (barre : 10 % - 90 %)<extra>"
                              + LABELS[strategy] + "</extra>"), row=1, col=col)
        fig.update_xaxes(title_text="coût total (BF = 1, HF = 10)", row=1, col=col)
        fig.update_yaxes(title_text="RMSE / écart-type de la HF", rangemode="tozero",
                         row=1, col=col)
    style(fig, "Test orienté région : séquentiel contre plans en un coup, à coût égal (560)",
          f"Profil NACA 4D, région = portance BF entre {CL_RANGE[0]} et {CL_RANGE[1]} ; "
          f"médiane de {results['seed'].nunique()} graines appariées", height=600, legend_rows=3)
    print(f"figure: {save(fig, bs.FIGURES, 'region_study')}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--figure-only", action="store_true")
    args = parser.parse_args()
    if args.figure_only:
        results = pd.read_csv(bs.RESULTS / "region_study.csv")
        make_figure(results, pd.read_csv(bs.RESULTS / "region_curves.csv"))
        print(summarize(results))
        return
    region_data()                                   # computed once, before the workers start
    # the heavy (sequential) runs first, so that the light ones fill the gaps at the end
    tasks = [(s, seed) for s in ("S1", "S2") for seed in range(args.seeds)]
    tasks += [(s, seed) for s in ("B1", "B2", "B3") for seed in range(args.seeds)]
    rows, curves = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for row, curve_rows in pool.map(run_task, tasks):
            rows.append(row)
            curves += curve_rows
            print(f"{row['strategy']} seed {row['seed']:2d}: region RMSE {row['rmse_region']:.4f}"
                  f" | global {row['rmse_global']:.4f} | LF/HF {row['n_lf']}/{row['n_hf']} | "
                  f"in region {100 * row['fraction_in_region']:.0f} % | "
                  f"{row['wall_time_s']:.0f} s", flush=True)
    results, curves = pd.DataFrame(rows), pd.DataFrame(curves)
    results.to_csv(bs.RESULTS / "region_study.csv", index=False)
    curves.to_csv(bs.RESULTS / "region_curves.csv", index=False)
    summary = summarize(results)
    (bs.RESULTS / "region_summary.txt").write_text(summary, encoding="utf-8")
    print("\n" + summary)
    make_figure(results, curves)


if __name__ == "__main__":
    main()
