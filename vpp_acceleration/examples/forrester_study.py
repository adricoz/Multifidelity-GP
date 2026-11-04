"""
Validation of the sequential co-kriging (src/sequential_design.py) on Forrester, at EQUAL COST.

Problem: Forrester 1D of Sacher et al. 2021, Eq. 17 (same as the tests):
    f1(x) = 0.5 (6x - 2)^2 sin(12x - 4) + 10 (x - 1)      low fidelity,  cost 1
    f2(x) = 2 f1(x) - 20 (x - 1)                          high fidelity, cost 10

Three strategies, same total cost 96, 10 seeds:
  1. sequential, adjusted variance (Le Gratiet & Cannamela corrected, adjust=True):
     initial nested design 6 LF + 3 HF (cost 36), then 60 of new runs chosen by the algorithm;
  2. sequential, plain variance (adjust=False): same start, same budget;
  3. one-shot nested Latin hypercube 16 LF + 8 HF (cost 16 + 80 = 96).
Accuracy of the high-fidelity prediction on a grid of 501 points: relative RMSE (RMSE / std of
f2), Q^2, coverage of the 95 % interval and NLPD (src/validation.prediction_scores).

Run from the repository root (Multifidelity-GP-Clean):
    .venv-benchmark\\Scripts\\python vpp_acceleration\\examples\\forrester_study.py
Outputs:
    vpp_acceleration/examples/results/forrester_study.csv    final scores, one row per run
    vpp_acceleration/examples/results/forrester_curves.csv   error after each iteration
    vpp_acceleration/examples/figures/forrester_study.html   interactive figure
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

HERE = Path(__file__).resolve().parent             # .../vpp_acceleration/examples
VPP_DIR = HERE.parent                               # .../vpp_acceleration
REPO_DIR = VPP_DIR.parent                           # .../Multifidelity-GP-Clean
sys.path.insert(0, str(REPO_DIR / "mfego"))
sys.path.insert(1, str(VPP_DIR))
sys.path.insert(2, str(REPO_DIR / "benchmarks"))

# pylint: disable=wrong-import-position,import-error
from figure_style import MUTED, PRIMARY, SECONDARY, SERIES, rgba, save, style  # noqa: E402
from src.cokriging import CoKriging  # noqa: E402
from src.data_management import ExperimentData  # noqa: E402
from src.kernels import SquaredExponentialKernel  # noqa: E402
from src.sequential_design import SequentialCoKriging, evaluate_initial_design  # noqa: E402
from src.simulator import BaseSimulator  # noqa: E402
from src.validation import prediction_scores  # noqa: E402

COSTS = [1.0, 10.0]
INITIAL_DESIGN = [6, 3]            # LF, HF points of the start of the sequential runs
BUDGET = 60.0                      # cost of the new runs of the sequential strategies
LHS_DESIGN = [16, 8]               # one-shot design of the same total cost: 16 + 8 * 10 = 96
SEEDS = range(10)
X_GRID = np.linspace(0.0, 1.0, 501).reshape(-1, 1)
VARIANTS = {"adjusted": "séquentiel, variance ajustée",
            "plain": "séquentiel, variance seule",
            "lhs": "LHS imbriqué (coût égal)"}


def bs_run_dir():
    """Run folder of the co-kriging benchmarks (2026-10-21), runs/README.md."""
    return Path(__file__).resolve().parents[1] / "runs" / "2026-10-21_cokriging_benchmarks"


def forrester_lf(x):
    """Low fidelity f1 (Sacher Eq. 17)."""
    x = np.asarray(x, dtype=float)
    return 0.5 * (6 * x - 2) ** 2 * np.sin(12 * x - 4) + 10 * (x - 1)


def forrester_hf(x):
    """High fidelity f2 = 2 f1 - 20 (x - 1) (true rho = 2)."""
    x = np.asarray(x, dtype=float)
    return 2 * forrester_lf(x) - 20 * (x - 1)


class ForresterSimulator(BaseSimulator):
    """The two Forrester levels, in the simulator format of mfego: evaluate(x, level)."""

    def __init__(self):
        super().__init__(num_levels=2)

    def evaluate(self, design_point, level):
        x = float(np.asarray(design_point, dtype=float).reshape(-1)[0])
        value = forrester_lf(x) if level == 1 else forrester_hf(x)
        return float(value), {}


def forrester_model():
    """CoKriging with the squared exponential kernel (Forrester is analytic), mfego defaults
    otherwise (MAP estimation of the hyperparameters, rho learned under a lognormal prior)."""
    return CoKriging(2, kernel_class=SquaredExponentialKernel, seed=0)


def nested_design(points_per_level, seed):
    """Nested Latin hypercube design X_2 in X_1, evaluated."""
    data = ExperimentData(bounds=[(0.0, 1.0)], costs=COSTS)
    data.generate_nested_design(list(points_per_level), seed=seed)
    evaluate_initial_design(data, ForresterSimulator())
    return data


def scores(model):
    """Accuracy and calibration of the high-fidelity prediction on X_GRID."""
    mean, variance = model.predict_batch(X_GRID, include_noise=False)
    return prediction_scores(forrester_hf(X_GRID[:, 0]), mean, variance)


def run_sequential(seed, adjust):
    """
    One sequential run. The loop is written out (ask, then evaluate) instead of calling
    SequentialCoKriging.run, to record the error of the model after each iteration.

    Returns:
    - design (SequentialCoKriging, final model fitted), curve: list of (total cost, rmse_rel).
    """
    data = nested_design(INITIAL_DESIGN, seed)
    initial_cost = sum(n * c for n, c in zip(INITIAL_DESIGN, COSTS))
    design = SequentialCoKriging(data, ForresterSimulator(), model_factory=forrester_model,
                                 adjust=adjust, seed=seed)
    curve = []
    while True:
        remaining = BUDGET - design.cost_spent
        if remaining < min(COSTS):
            break
        proposal = design.ask(remaining)          # fits the model on the current data
        curve.append((initial_cost + design.cost_spent, scores(design.model)["rmse_rel"]))
        if proposal["level"] is None:
            break
        design.evaluate(proposal)
    design.fit()
    curve.append((initial_cost + design.cost_spent, scores(design.model)["rmse_rel"]))
    return design, curve


def main():
    rows, curves, examples = [], [], {}
    for seed in SEEDS:
        for variant in ("adjusted", "plain"):
            design, curve = run_sequential(seed, adjust=(variant == "adjusted"))
            n_points = [len(design.data.get_training_data(l)[1]) for l in (1, 2)]
            rows.append({"seed": seed, "variant": variant,
                         "total_cost": n_points[0] * COSTS[0] + n_points[1] * COSTS[1],
                         "n_lf": n_points[0], "n_hf": n_points[1],
                         "rho": design.model.rhos[0], **scores(design.model)})
            curves += [{"seed": seed, "variant": variant, "cost": c, "rmse_rel": e}
                       for c, e in curve]
            if seed == 0:
                examples[variant] = design
        lhs_data = nested_design(LHS_DESIGN, seed)
        lhs_model = forrester_model()
        lhs_model.fit(lhs_data)
        rows.append({"seed": seed, "variant": "lhs",
                     "total_cost": LHS_DESIGN[0] * COSTS[0] + LHS_DESIGN[1] * COSTS[1],
                     "n_lf": LHS_DESIGN[0], "n_hf": LHS_DESIGN[1], "rho": lhs_model.rhos[0],
                     **scores(lhs_model)})
        print(f"seed {seed}: " + " | ".join(f"{r['variant']} RMSE {r['rmse_rel']:.2e} "
                                             f"({r['n_lf']} LF / {r['n_hf']} HF)"
                                             for r in rows[-3:]))
    results = pd.DataFrame(rows)
    results_dir = bs_run_dir() / "results"
    results_dir.mkdir(exist_ok=True)
    results.to_csv(results_dir / "forrester_study.csv", index=False)
    pd.DataFrame(curves).to_csv(results_dir / "forrester_curves.csv", index=False)

    summary = results.groupby("variant").agg(
        rmse_median=("rmse_rel", "median"), rmse_p10=("rmse_rel", lambda s: s.quantile(0.1)),
        rmse_p90=("rmse_rel", lambda s: s.quantile(0.9)), q2_median=("q2", "median"),
        coverage_median=("coverage_95", "median"), nlpd_median=("nlpd", "median"),
        lf_median=("n_lf", "median"), hf_median=("n_hf", "median"),
        cost_max=("total_cost", "max"))
    print("\nMedians over the seeds:\n" + summary.to_string(float_format=lambda v: f"{v:.3g}"))
    wins = (results.pivot(index="seed", columns="variant", values="rmse_rel")
            .pipe(lambda t: (t["adjusted"] < t["lhs"]).sum()))
    print(f"\nSeeds where the adjusted sequential design beats the LHS: {wins} / {len(SEEDS)}")
    make_figure(results, pd.DataFrame(curves), examples)


def make_figure(results, curves, examples):
    """Left: final fit of seed 0 (adjusted). Right: error against the cost (median of the seeds)."""
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.1, subplot_titles=(
        "Graine 0 : modèle final du séquentiel (variance ajustée)",
        "Erreur HF selon le coût (médiane de 10 graines)"))
    colors = {"adjusted": SERIES[0], "plain": SERIES[1], "lhs": SERIES[2]}

    # left panel: truth, mean, 95 % band, the points chosen
    design = examples["adjusted"]
    mean, variance = design.model.predict_batch(X_GRID, include_noise=False)
    band = 1.96 * np.sqrt(variance)
    x = X_GRID[:, 0]
    fig.add_trace(go.Scatter(x=np.r_[x, x[::-1]], y=np.r_[mean + band, (mean - band)[::-1]],
                             fill="toself", fillcolor=rgba(colors["adjusted"], 0.15),
                             line={"width": 0}, hoverinfo="skip", name="intervalle à 95 %"),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=forrester_hf(x), name="f2 vraie (haute fidélité)",
                             line={"color": SECONDARY, "width": 2, "dash": "dot"}), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=mean, name="moyenne du co-krigeage",
                             line={"color": colors["adjusted"], "width": 2}), row=1, col=1)
    for level, symbol, ink, label in ((1, "circle-open", MUTED, "points BF"),
                                      (2, "diamond", PRIMARY, "points HF")):
        x_level, y_level = design.data.get_training_data(level)
        y_shown = forrester_hf(x_level[:, 0])        # shown on the f2 curve to see where they are
        fig.add_trace(go.Scatter(x=x_level[:, 0], y=y_shown, mode="markers", name=label,
                                 marker={"symbol": symbol, "size": 9, "color": ink,
                                         "line": {"width": 1.5, "color": ink}},
                                 hovertemplate="x = %{x:.3f}<extra>" + label + "</extra>"),
                      row=1, col=1)

    # right panel: relative RMSE against the total cost, median over the seeds
    cost_grid = np.arange(36.0, 97.0, 1.0)
    for variant in ("adjusted", "plain"):
        per_seed = []
        for _, group in curves[curves["variant"] == variant].groupby("seed"):
            group = group.sort_values("cost")
            # error of the last model fitted at a cost <= c (step function of the cost)
            index = np.searchsorted(group["cost"].to_numpy(), cost_grid, side="right") - 1
            per_seed.append(group["rmse_rel"].to_numpy()[np.maximum(index, 0)])
        median = np.median(np.array(per_seed), axis=0)
        fig.add_trace(go.Scatter(x=cost_grid, y=median, name=VARIANTS[variant],
                                 line={"color": colors[variant], "width": 2, "shape": "hv"},
                                 hovertemplate="coût %{x:.0f} : RMSE relative %{y:.2e}"
                                               "<extra>" + VARIANTS[variant] + "</extra>"),
                      row=1, col=2)
    lhs = results[results["variant"] == "lhs"]["rmse_rel"]
    fig.add_trace(go.Scatter(
        x=[96.0], y=[lhs.median()], mode="markers", name=VARIANTS["lhs"],
        marker={"size": 11, "color": colors["lhs"], "symbol": "square"},
        error_y={"type": "data", "symmetric": False, "color": colors["lhs"],
                 "array": [lhs.quantile(0.9) - lhs.median()],
                 "arrayminus": [lhs.median() - lhs.quantile(0.1)]},
        hovertemplate="coût 96 : RMSE relative médiane %{y:.2e} (barre : 10 % - 90 %)"
                      "<extra>LHS imbriqué</extra>"), row=1, col=2)
    fig.update_xaxes(title_text="x", row=1, col=1)
    fig.update_yaxes(title_text="f2(x)", row=1, col=1)
    fig.update_xaxes(title_text="coût total (BF = 1, HF = 10)", row=1, col=2)
    fig.update_yaxes(title_text="RMSE / écart-type de f2", type="log", row=1, col=2)
    style(fig, "Co-krigeage séquentiel sur Forrester, à coût égal",
          "Départ 6 BF + 3 HF (coût 36), budget de 60 choisi par l'algorithme, contre un LHS "
          "imbriqué 16 BF + 8 HF (coût 96)", height=560, legend_rows=2)
    path = save(fig, bs_run_dir() / "figures", "forrester_study")
    print(f"figure: {path}")


if __name__ == "__main__":
    main()
