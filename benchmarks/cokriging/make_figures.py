"""
Interactive figures (plotly HTML) of the co-kriging benchmark, built from the results of
run_cokriging_benchmark.py and explore_rho_centering.py. Output: benchmarks/cokriging/figures/.

Run from the repository root (benchmark environment):
    .venv-benchmark\\Scripts\\python benchmarks\\cokriging\\make_figures.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
# pylint: disable=wrong-import-position,import-error
from figure_style import MUTED, SECONDARY, SERIES, hline, rgba, save, style  # noqa: E402

RESULTS, FIGURES = HERE / "results", HERE / "figures"
PROBLEMS = {"hartmann": "Hartmann 6D", "airfoil": "NACA 4-digit airfoil (NeuralFoil)"}
NOMINAL = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
Z_EDGES = np.linspace(-5.0, 5.0, 41)
# fixed color (and dash) of each model in every figure
MODELS = {
    "mfego CoKriging (MAP)": (SERIES[0], "solid", "mfego CoKriging, MAP (= recursive model)"),
    "mfego CoKriging (MLE)": (SERIES[1], "dash", "mfego CoKriging, MLE"),
    "GPy/emukit AR1 (MAP)": (SERIES[2], "solid", "GPy/emukit AR(1), joint MAP"),
    "GPy/emukit AR1 (MLE)": (SERIES[3], "dash", "GPy/emukit AR(1), joint MLE"),
    "mfego HF only (MAP)": (SERIES[4], "dot", "mfego high fidelity only, MAP"),
    "SMT MFK": (SERIES[5], "dashdot", "SMT MFK (recursive co-kriging)"),
}
INFILL = {
    "nested": (SERIES[0], "nested LHS + mfego CoKriging (MAP)"),
    "EGO data + mfego recursive (MAP)": (SERIES[1], "infill (NN-MF-EGO) + recursive model"),
    "EGO data + exact conditioning (same hyperparameters)":
        (SERIES[6], "infill + exact conditioning, same hyperparameters"),
    "EGO data + LF completion + mfego CoKriging (MAP)":
        (SERIES[2], "infill + LF at the HF points + CoKriging"),
}


def medians(frame: pd.DataFrame, metric: str) -> tuple:
    grouped = frame.groupby("n_hf")[metric]
    return grouped.median(), grouped.quantile(0.25), grouped.quantile(0.75)


# 1/5 ---------------------------------------------------------------------------------------------
def figure_nested(nested: pd.DataFrame) -> Path:
    metrics = [("rmse_rel", "relative RMSE", "linear"), ("nlpd", "NLPD", "linear"),
               ("coverage_95", "coverage of the 95 % interval", "linear")]
    titles = [f"{title} · {PROBLEMS[p]}" for p in PROBLEMS for _, title, _ in metrics]
    fig = make_subplots(rows=2, cols=3, subplot_titles=titles, horizontal_spacing=0.07,
                        vertical_spacing=0.16)
    offsets = np.linspace(0.93, 1.07, len(MODELS))
    for r, problem in enumerate(PROBLEMS, start=1):
        sub = nested[nested["problem"] == problem]
        for c, (metric, title, _) in enumerate(metrics, start=1):
            for i, (model, (color, dash, label)) in enumerate(MODELS.items()):
                med, q1, q3 = medians(sub[sub["model"] == model], metric)
                fig.add_trace(go.Scatter(
                    x=med.index * offsets[i], y=med, mode="lines+markers", name=label,
                    legendgroup=model, showlegend=(r == 1 and c == 1),
                    line={"color": color, "width": 2, "dash": dash}, marker={"size": 8},
                    error_y={"type": "data", "symmetric": False, "array": q3 - med,
                             "arrayminus": med - q1, "color": rgba(color, 0.6),
                             "thickness": 1.5, "width": 4},
                    customdata=np.column_stack([med.index, q1, q3]),
                    hovertemplate=f"<b>{label}</b><br>n_HF = %{{customdata[0]}}<br>{title}: "
                                  "%{y:.3f}<br>quartiles %{customdata[1]:.3f} – "
                                  "%{customdata[2]:.3f}<extra></extra>"), row=r, col=c)
            fig.update_xaxes(type="log", tickvals=sorted(sub["n_hf"].unique()),
                             title_text="n_HF (n_LF = 4 n_HF)", row=r, col=c)
        fig.add_hline(y=0.95, line={"color": MUTED, "width": 1}, row=r, col=3)
    style(fig, "Nested designs: co-kriging conditioned on all the data",
          "Median over 10 seeds (bars: interquartile range) · same nested LHS for every model "
          "of a seed · the mfego recursive model gives the same values as CoKriging", 840,
          legend_rows=2)
    return save(fig, FIGURES, "nested_accuracy")


# 2/5 ---------------------------------------------------------------------------------------------
def figure_calibration(nested: pd.DataFrame, infill: pd.DataFrame) -> Path:
    """Reliability diagram (empirical vs nominal coverage) and histogram of the standardized
    errors (y - mean) / std, at one design size per problem."""
    sizes = {"hartmann": 20, "airfoil": 10}
    curves = [("mfego CoKriging (MAP)", nested), ("GPy/emukit AR1 (MAP)", nested),
              ("mfego HF only (MAP)", nested), ("EGO data + mfego recursive (MAP)", infill)]
    fig = make_subplots(rows=2, cols=2, horizontal_spacing=0.1, vertical_spacing=0.16,
                        subplot_titles=[f"{t} · {PROBLEMS[p]} (cost of n_HF = {sizes[p]})"
                                        for p in PROBLEMS for t in
                                        ("coverage of the central intervals",
                                         "standardized errors (y − mean) / std")])
    for r, problem in enumerate(PROBLEMS, start=1):
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", showlegend=(r == 1),
                                 name="perfect calibration", legendgroup="ideal",
                                 line={"color": MUTED, "width": 1}), row=r, col=1)
        z_mid = 0.5 * (Z_EDGES[1:] + Z_EDGES[:-1])
        fig.add_trace(go.Scatter(x=z_mid, y=norm.pdf(z_mid), mode="lines", showlegend=False,
                                 legendgroup="ideal", line={"color": MUTED, "width": 1}),
                      row=r, col=2)
        for model, frame in curves:
            sub = frame[(frame["problem"] == problem) & (frame["model"] == model)
                        & (frame["n_hf"] == sizes[problem])]
            if model in MODELS:
                color, dash, label = MODELS[model]
            else:
                (color, label), dash = INFILL[model], "dash"
            coverage = np.median(np.array([json.loads(c) for c in sub["coverage"]]), axis=0)
            fig.add_trace(go.Scatter(x=NOMINAL, y=coverage, mode="lines+markers", name=label,
                                     legendgroup=model, showlegend=(r == 1),
                                     line={"color": color, "width": 2, "dash": dash},
                                     marker={"size": 7},
                                     hovertemplate=f"<b>{label}</b><br>nominal %{{x:.2f}}<br>"
                                                   "empirical %{y:.3f}<extra></extra>"),
                          row=r, col=1)
            counts = np.sum(np.array([json.loads(h) for h in sub["z_hist"]]), axis=0)
            density = counts / counts.sum() / np.diff(Z_EDGES)
            fig.add_trace(go.Scatter(x=z_mid, y=density, mode="lines", showlegend=False,
                                     legendgroup=model, line={"color": color, "width": 2,
                                                              "dash": dash, "shape": "hvh"},
                                     hovertemplate=f"<b>{label}</b><br>z = %{{x:.2f}}<br>"
                                                   "density %{y:.3f}<extra></extra>"),
                          row=r, col=2)
        fig.update_xaxes(title_text="nominal probability of the interval", row=r, col=1)
        fig.update_yaxes(title_text="share of test points inside", row=r, col=1)
        fig.update_xaxes(title_text="z (clipped at ±5)", row=r, col=2)
    style(fig, "Is the predicted variance a true variance?",
          "Below the diagonal: intervals too narrow (overconfident). Grey curve on the right: "
          "N(0, 1), the ideal distribution of the standardized errors · 10 seeds pooled", 880,
          legend_rows=2)
    return save(fig, FIGURES, "calibration")


# 3/5 ---------------------------------------------------------------------------------------------
def figure_verification(nested: pd.DataFrame) -> Path:
    """[LG] Prop. 3.1 (CoKriging = recursive model on nested data) and exactness vs emukit."""
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.12, subplot_titles=[
        "relative RMSE per fit: recursive model vs CoKriging (nested designs)",
        "same hyperparameters as emukit: max difference over the test points"])
    for problem, symbol in (("hartmann", "circle"), ("airfoil", "diamond")):
        wide = nested[nested["problem"] == problem].pivot_table(
            index=["n_hf", "seed"], columns="model", values="rmse_rel")
        fig.add_trace(go.Scatter(
            x=wide["mfego CoKriging (MAP)"], y=wide["mfego recursive (MAP)"], mode="markers",
            name=PROBLEMS[problem], marker={"color": SERIES[0], "symbol": symbol, "size": 8},
            hovertemplate="CoKriging %{x:.4f}<br>recursive %{y:.4f}<extra></extra>"),
            row=1, col=1)
    fig.add_trace(go.Scatter(x=[0, 1.6], y=[0, 1.6], mode="lines", showlegend=False,
                             line={"color": MUTED, "width": 1}), row=1, col=1)
    exact = nested.dropna(subset=["exact_mean"])
    for column, label in (("exact_mean", "mean / std(y)"),
                          ("exact_var", "latent variance, relative")):
        for problem, symbol in (("hartmann", "circle"), ("airfoil", "diamond")):
            sub = exact[exact["problem"] == problem]
            fig.add_trace(go.Box(
                y=sub[column], x=[f"{label}<br>{PROBLEMS[problem].split(' (')[0]}"] * len(sub),
                showlegend=False, boxpoints="all", jitter=0.4, pointpos=0,
                marker={"color": SERIES[2], "size": 6, "symbol": symbol},
                line={"color": SERIES[2], "width": 1.5}, fillcolor=rgba(SERIES[2], 0.12),
                customdata=np.column_stack([sub["model"], sub["n_hf"], sub["seed"]]),
                hovertemplate="%{customdata[0]}, n_HF %{customdata[1]}, seed %{customdata[2]}"
                              "<br>difference %{y:.2e}<extra></extra>"), row=1, col=2)
    fig.update_yaxes(type="log", row=1, col=2)
    hline(fig, 1e-8, "1e-8", 1, 2, position="top right")
    fig.update_xaxes(title_text="mfego CoKriging (MAP)", row=1, col=1)
    fig.update_yaxes(title_text="mfego recursive (MAP)", row=1, col=1)
    style(fig, "Verification of the co-kriging",
          "Left: every point on the diagonal ([LG] Prop. 3.1). Right: emukit hyperparameters "
          "put in mfego CoKriging, every emukit fit (MLE and MAP)", 560)
    return save(fig, FIGURES, "verification")


# 4/5 ---------------------------------------------------------------------------------------------
def figure_infill(nested: pd.DataFrame, infill: pd.DataFrame) -> Path:
    metrics = [("rmse_rel", "relative RMSE (whole domain)"),
               ("rmse_rel_top10", "relative RMSE (best 10 % of the domain)"),
               ("coverage_95", "coverage of the 95 % interval"),
               ("std_at_hf_points", "predicted std at the HF data points / std(y)")]
    fig = make_subplots(rows=2, cols=4, subplot_titles=[f"{m[1]}" for m in metrics] * 2,
                        horizontal_spacing=0.06, vertical_spacing=0.2)
    for r, problem in enumerate(PROBLEMS, start=1):
        sizes = sorted(infill.loc[infill["problem"] == problem, "n_hf"].unique())
        for c, (metric, _) in enumerate(metrics, start=1):
            for key, (color, label) in INFILL.items():
                if key == "nested":
                    sub = nested[(nested["problem"] == problem) & nested["n_hf"].isin(sizes)
                                 & (nested["model"] == "mfego CoKriging (MAP)")]
                else:
                    sub = infill[(infill["problem"] == problem) & (infill["model"] == key)]
                if metric not in sub or sub[metric].isna().all():
                    continue
                fig.add_trace(go.Box(
                    x=[f"cost of n_HF = {n}" for n in sub["n_hf"]], y=sub[metric], name=label,
                    legendgroup=key, showlegend=(r == 1 and c == 1), offsetgroup=key,
                    boxpoints="all", jitter=0.3, pointpos=0,
                    marker={"color": color, "size": 5}, line={"color": color, "width": 1.5},
                    fillcolor=rgba(color, 0.12),
                    hovertemplate=f"<b>{label}</b><br>%{{y:.3g}}<extra></extra>"),
                    row=r, col=c)
        fig.update_yaxes(type="log", row=r, col=4)
        fig.add_annotation(text=f"<b>{PROBLEMS[problem]}</b>", x=0.0, xref="paper",
                           xanchor="left", y=1.0 if r == 1 else 0.42, yref="paper",
                           yanchor="bottom", showarrow=False, yshift=28,
                           font={"color": SECONDARY, "size": 13})
    fig.update_layout(boxmode="group")
    style(fig, "Surrogate built from the points of the current infilling (NN-MF-EGO)",
          "Same cost as the nested design · 10 seeds · last panel: the value is known at these "
          "points, the exact posterior std is about the noise level", 900, legend_rows=2)
    return save(fig, FIGURES, "infill")


# 5/5 ---------------------------------------------------------------------------------------------
def figure_rho(rho: pd.DataFrame) -> Path:
    models = {"mfego CoKriging (MAP)": (SERIES[0], "current mfego (residual constant at rho = 1)"),
              "mfego CoKriging (MAP, centred rho)": (SERIES[1], "level below centred "
                                                                  "(constant follows rho), rho in [-5, 5]"),
              "mfego CoKriging (MAP, centred rho, bounds 50)": (SERIES[3], "level below centred, "
                                                                             "rho in [-50, 50]"),
              "GPy/emukit AR1 (MAP)": (SERIES[2], "GPy/emukit AR(1), joint MAP")}
    metrics = [("rho", "estimated rho"), ("rmse_rel", "relative RMSE"), ("nlpd", "NLPD")]
    fig = make_subplots(rows=2, cols=3, horizontal_spacing=0.07, vertical_spacing=0.16,
                        subplot_titles=[f"{m[1]} · {PROBLEMS[p]}" for p in PROBLEMS
                                        for m in metrics])
    for r, problem in enumerate(PROBLEMS, start=1):
        for c, (metric, _) in enumerate(metrics, start=1):
            for model, (color, label) in models.items():
                sub = rho[(rho["problem"] == problem) & (rho["model"] == model)]
                fig.add_trace(go.Box(
                    x=sub["n_hf"].astype(str), y=sub[metric], name=label, legendgroup=model,
                    showlegend=(r == 1 and c == 1), offsetgroup=model, boxpoints="all",
                    jitter=0.3, pointpos=0, marker={"color": color, "size": 5},
                    line={"color": color, "width": 1.5}, fillcolor=rgba(color, 0.12),
                    hovertemplate=f"<b>{label}</b><br>n_HF %{{x}}<br>%{{y:.3f}}<extra></extra>"),
                    row=r, col=c)
            fig.update_xaxes(type="category", title_text="n_HF (nested, n_LF = 4 n_HF)", row=r,
                             col=c)
    fig.update_layout(boxmode="group")
    style(fig, "Bias of the profiled rho of mfego (exploration, core not modified)",
          "mfego fixes the constant of the residual for rho = 1: rho is pulled towards 1 · on "
          "Hartmann the centred rho sits on the bound 5; free, it goes to 6-23 and the RMSE gets "
          "worse", 860, legend_rows=2)
    return save(fig, FIGURES, "rho_bias")


def main() -> None:
    nested = pd.read_csv(RESULTS / "nested.csv")
    infill = pd.read_csv(RESULTS / "infill.csv")
    paths = [figure_nested(nested), figure_calibration(nested, infill),
             figure_verification(nested), figure_infill(nested, infill)]
    if (RESULTS / "rho_centering.csv").exists():
        paths.append(figure_rho(pd.read_csv(RESULTS / "rho_centering.csv")))
    for path in paths:
        print("written:", path)


if __name__ == "__main__":
    main()
