"""
[V3] Interactive figures (plotly HTML) of the mfego v1 / v2 / v3 benchmark, built from the results
of run_benchmark.py. Output: benchmarks/surrogate_v3/figures/.

Run from the repository root (benchmark environment):
    .venv-benchmark\\Scripts\\python benchmarks\\surrogate_v3\\make_figures.py
"""
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import lognorm, norm

import v3_lib as v3  # noqa: E402  (sets the paths)
from figure_style import MUTED, SERIES, rgba, save, style  # noqa: E402

FIGURES = v3.HERE / "figures"
PROBLEMS = {"hartmann": "Hartmann 6D", "airfoil": "airfoil NF / NF",
            "airfoil_xfoil": "airfoil NF / XFOIL"}
# fixed color, dash and short label of each mfego variant in every figure
VARIANT_STYLE = {
    "v1": (SERIES[1], "dash", "v1"),
    "v1 Matern": (SERIES[4], "dot", "v1<br>Matérn"),
    "v2 N(1, 2²)": (SERIES[0], "solid", "v2<br>N(1, 2²)"),
    "v2 N(1, 1²)": (SERIES[6], "solid", "v2<br>N(1, 1²)"),
    "v3 LogN s=0.5": (SERIES[2], "solid", "v3<br>s = 0.5"),
    "v3 LogN s=1": (SERIES[3], "solid", "v3<br>s = 1"),
}
COMPETITORS = {"GPy/emukit AR1 (MAP)": (SERIES[5], "dashdot"),
               "BoTorch MF-GP": (SERIES[7], "dash")}


def band_line(fig, frame, x_col, metric, name, color, dash, row, col, offset, legend):
    """Median over the seeds with the interquartile range as error bars."""
    grouped = frame.groupby(x_col)[metric]
    med, q1, q3 = grouped.median(), grouped.quantile(0.25), grouped.quantile(0.75)
    fig.add_trace(go.Scatter(
        x=med.index.to_numpy(dtype=float) * offset, y=med, mode="lines+markers", name=name,
        legendgroup=name, showlegend=legend, line={"color": color, "width": 2, "dash": dash},
        marker={"size": 8},
        error_y={"type": "data", "symmetric": False, "array": q3 - med, "arrayminus": med - q1,
                 "color": rgba(color, 0.6), "thickness": 1.5, "width": 4},
        customdata=np.column_stack([med.index, q1, q3]),
        hovertemplate=f"<b>{name}</b><br>{x_col} = %{{customdata[0]}}<br>median %{{y:.3f}}"
                      "<br>quartiles %{customdata[1]:.3f} – %{customdata[2]:.3f}<extra></extra>"),
        row=row, col=col)


def box(fig, values, label, name, color, row, col, hover=""):
    """Box with every point (one per seed or fit)."""
    fig.add_trace(go.Box(y=values, x=[label] * len(values), name=name, showlegend=False,
                         boxpoints="all", jitter=0.4, pointpos=0,
                         marker={"color": color, "size": 6}, line={"color": color, "width": 1.5},
                         fillcolor=rgba(color, 0.12),
                         hovertemplate=f"<b>{name}</b>{hover}<br>%{{y:.3f}}<extra></extra>"),
                  row=row, col=col)


# 1/5 ---------------------------------------------------------------------------------------------
def figure_priors():
    """Densities of the priors on rho (scaled to a maximum of 1), with the bound 5 of v2."""
    rho = np.linspace(-3.0, 12.0, 1501)
    densities = {"v2 N(1, 2²)": norm.pdf(rho, 1.0, 2.0), "v2 N(1, 1²)": norm.pdf(rho, 1.0, 1.0)}
    for name, s in (("v3 LogN s=0.5", 0.5), ("v3 LogN s=1", 1.0)):
        _, mu, sigma = v3.lognormal_mode_1(s)
        densities[name] = np.where(rho > 0, lognorm.pdf(np.maximum(rho, 1e-12), s=sigma,
                                                        scale=np.exp(mu)), 0.0)
    fig = go.Figure()
    for name, density in densities.items():
        color, dash, _ = VARIANT_STYLE[name]
        fig.add_trace(go.Scatter(x=rho, y=density / density.max(), name=name,
                                 line={"color": color, "width": 2, "dash": dash},
                                 hovertemplate=f"<b>{name}</b><br>rho %{{x:.2f}}<br>"
                                               "p / max %{y:.3f}<extra></extra>"))
    fig.add_vline(x=5.0, line={"color": MUTED, "width": 1, "dash": "dot"})
    fig.add_annotation(x=5.0, y=1.02, text="bound 5 of v1 / v2", showarrow=False,
                       xanchor="left", font={"color": MUTED, "size": 11})
    fig.update_xaxes(title_text="rho")
    fig.update_yaxes(title_text="prior density / its maximum", range=[0, 1.08])
    style(fig, "Priors on rho: normal (v2, closed form) and lognormal (v3, learned)",
          "The MAP is computed in parameter space: rho is pulled towards the MODE of its prior "
          "(1 for all four) · v3 bounds [0.001, 50]", 520)
    return save(fig, FIGURES, "priors")


# 2/5 ---------------------------------------------------------------------------------------------
def figure_rho(nested: pd.DataFrame):
    """Fitted rho of every CoKriging variant on the nested designs, share at a bound."""
    fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.06,
                        subplot_titles=[PROBLEMS[p] for p in PROBLEMS])
    for c, problem in enumerate(PROBLEMS, start=1):
        sub = nested[nested["problem"] == problem]
        for name in v3.VARIANTS:
            color, _, label = VARIANT_STYLE[name]
            values = sub[sub["model"] == f"mfego {name}"]["rho"].dropna()
            share = values.apply(lambda r, n=name: v3.at_bound(r, v3.BOUNDS[n])).mean()
            box(fig, values, f"{label}<br><b>{100 * share:.0f} %</b>", name, color, 1, c,
                hover=f" · {PROBLEMS[problem]} · {100 * share:.0f} % of the fits at a bound")
        fig.update_yaxes(type="log", tickvals=[0.5, 1, 2, 5, 10, 20], row=1, col=c)
    fig.update_yaxes(title_text="fitted rho (log)", row=1, col=1)
    fig.update_xaxes(tickangle=0, tickfont={"size": 11})
    style(fig, "Fitted rho on the nested designs",
          "CoKriging, 3 sizes × 10 seeds per box · in bold: share of the fits with rho at a "
          "bound (±5 for v1 and v2, [0.001, 50] for v3)", 600)
    return save(fig, FIGURES, "rho")


# 3/5 ---------------------------------------------------------------------------------------------
def figure_nested(nested: pd.DataFrame):
    metrics = [("rmse_rel", "relative RMSE"), ("nlpd", "NLPD"),
               ("coverage_95", "coverage of the 95 % interval")]
    series = {f"mfego {name}": (VARIANT_STYLE[name][0], VARIANT_STYLE[name][1])
              for name in v3.VARIANTS}
    series.update(COMPETITORS)
    fig = make_subplots(rows=3, cols=3, horizontal_spacing=0.07, vertical_spacing=0.1,
                        subplot_titles=[f"{t} · {PROBLEMS[p]}" for p in PROBLEMS
                                        for _, t in metrics])
    offsets = np.linspace(0.93, 1.07, len(series))
    for r, problem in enumerate(PROBLEMS, start=1):
        sub = nested[nested["problem"] == problem]
        for c, (metric, _) in enumerate(metrics, start=1):
            for i, (model, (color, dash)) in enumerate(series.items()):
                band_line(fig, sub[sub["model"] == model], "n_hf", metric, model, color, dash,
                          r, c, offsets[i], legend=(r == 1 and c == 1))
            fig.update_xaxes(type="log", tickvals=sorted(sub["n_hf"].unique()),
                             title_text="n_HF (nested, n_LF = 4 n_HF)", row=r, col=c)
        fig.add_hline(y=0.95, line={"color": MUTED, "width": 1}, row=r, col=3)
    style(fig, "Nested designs: mfego v1 / v2 / v3 (CoKriging) and the best other libraries",
          "Median over 10 seeds (bars: interquartile range) · SMT and HF-only models in "
          "results/summary.txt · NF = NeuralFoil", 1150, legend_rows=2)
    return save(fig, FIGURES, "nested_accuracy")


# 4/5 ---------------------------------------------------------------------------------------------
def figure_optimization(histories: list, runs: pd.DataFrame):
    methods = {f"NN-MF-EGO {name}": VARIANT_STYLE[name][0] for name in v3.EGO_VARIANTS}
    methods.update({"SF-EGO (Matern)": SERIES[5], "BoTorch qLogEI (SF)": SERIES[7]})
    labels = {"SF-EGO (Matern)": "SF-EGO<br>Matérn", "BoTorch qLogEI (SF)": "BoTorch<br>qLogEI"}
    fig = make_subplots(rows=1, cols=2, column_widths=[0.55, 0.45], horizontal_spacing=0.1,
                        subplot_titles=["best high-fidelity observation f − f* vs cost",
                                        "final recommendation f − f* (one point per seed)"])
    # up to the largest final cost: the last evaluation of a run may end after the budget 250
    grid = np.arange(120, max(h["cost"][-1] for h in histories if "cost" in h) + 1.0, 1.0)
    for method, color in methods.items():
        curves = []
        for h in histories:
            if h.get("method") != method or "error" in h:
                continue
            cost, best = np.array(h["cost"]), np.array(h["best_observed"])
            index = np.searchsorted(cost, grid, side="right") - 1
            curves.append(np.where(index >= 0, best[np.maximum(index, 0)], np.nan))
        fig.add_trace(go.Scatter(x=grid, y=np.nanmedian(np.array(curves), axis=0), name=method,
                                 legendgroup=method, line={"color": color, "width": 2,
                                                           "shape": "hv"},
                                 hovertemplate=f"<b>{method}</b><br>cost %{{x}}<br>median "
                                               "%{y:.3f}<extra></extra>"), row=1, col=1)
        variant = method.removeprefix("NN-MF-EGO ")
        label = VARIANT_STYLE[variant][2] if variant in VARIANT_STYLE else labels[method]
        box(fig, runs[runs["method"] == method]["recommendation error"], label, method, color,
            1, 2)
    fig.update_yaxes(type="log", row=1, col=1)
    fig.update_yaxes(type="log", row=1, col=2)
    fig.update_xaxes(title_text="cumulative cost (LF = 1, HF = 10)", row=1, col=1)
    fig.update_xaxes(tickangle=0, row=1, col=2)
    style(fig, "Optimization of Hartmann 6D at equal budget (250)",
          "10 seeds · median of the best observation · recommendation = minimizer of the "
          "predicted mean", 640, legend_rows=2)
    return save(fig, FIGURES, "optimization")


# 5/5 ---------------------------------------------------------------------------------------------
def figure_ego(runs: pd.DataFrame, infill: pd.DataFrame):
    """NN-MF-EGO per variant: Hartmann (B1) and airfoils (B2, paired with v1 seed by seed)."""
    titles = ["Hartmann: recommendation f − f* (log)", "Hartmann: share of LF evaluations",
              "Hartmann: final rho (v1 can be negative)", "NF / NF: recommended Cd − v1 (counts)",
              "NF / XFOIL: recommended Cd − v1 (counts)", "infill model Q², Hartmann n_HF = 40"]
    fig = make_subplots(rows=2, cols=3, horizontal_spacing=0.07, vertical_spacing=0.2,
                        subplot_titles=titles)
    key = ["problem", "n_hf", "seed"]
    reference = infill[infill["variant"] == v3.REFERENCE].set_index(key)["recommendation"]
    for name in v3.EGO_VARIANTS:
        color, _, label = VARIANT_STYLE[name]
        sub = runs[runs["method"] == f"NN-MF-EGO {name}"]
        for c, metric in enumerate(("recommendation error", "LF share", "final rho"), start=1):
            box(fig, sub[metric], label, name, color, 1, c)
        own = infill[infill["variant"] == name].set_index(key)["recommendation"]
        excess = (own - reference).dropna()
        for c, problem in ((1, "airfoil"), (2, "airfoil_xfoil")):
            values = excess.xs(problem, level="problem")
            box(fig, values, label, name, color, 2, c, hover=" · both sizes")
        q2 = infill[(infill["variant"] == name) & (infill["problem"] == "hartmann")
                    & (infill["n_hf"] == 40)]["q2"]
        box(fig, q2, label, name, color, 2, 3)
    fig.update_yaxes(type="log", row=1, col=1)
    fig.update_xaxes(tickangle=0)
    style(fig, "NN-MF-EGO with each variant",
          "Hartmann: budget 250, 10 seeds · airfoils: EGO until the cost of the nested design "
          "(n_HF = 10 and 20), difference with v1 for the same seed and size", 900)
    return save(fig, FIGURES, "ego")


def main() -> None:
    nested = pd.read_csv(v3.RESULTS / "nested.csv")
    infill = pd.read_csv(v3.RESULTS / "infill.csv")
    runs = pd.read_csv(v3.RESULTS / "optimization.csv")
    histories = json.loads((v3.RESULTS / "optimization_histories.json").read_text(
        encoding="utf-8"))
    nested = nested[nested["error"].isna()] if "error" in nested else nested
    infill = infill[infill["error"].isna()] if "error" in infill else infill
    for path in (figure_priors(), figure_rho(nested), figure_nested(nested),
                 figure_optimization(histories, runs), figure_ego(runs, infill)):
        print("written:", path)


if __name__ == "__main__":
    main()
