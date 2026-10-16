"""
[V2] Interactive figures (plotly HTML) of the mfego v2 benchmarks, built from the results of
run_selection.py and run_benchmark.py. Output: benchmarks/surrogate_v2/figures/.

Run from the repository root (benchmark environment):
    .venv-benchmark\\Scripts\\python benchmarks\\surrogate_v2\\make_figures.py
"""
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm

import v2_lib as v2  # noqa: E402  (sets the paths)
from figure_style import MUTED, SERIES, hline, rgba, save, style  # noqa: E402
from run_selection import REFERENCE, decide  # noqa: E402

FIGURES = v2.HERE / "figures"
# NF = NeuralFoil (low fidelity of both airfoil problems)
PROBLEMS = {"hartmann": "Hartmann 6D", "airfoil": "airfoil NF / NF",
            "airfoil_xfoil": "airfoil NF / XFOIL"}
# fixed color (and dash) of each model in every figure
MODELS = {
    "mfego v2 CoKriging": (SERIES[0], "solid"),
    "mfego v1 CoKriging": (SERIES[1], "dash"),
    "GPy/emukit AR1 (MAP)": (SERIES[2], "solid"),
    "SMT MFK": (SERIES[5], "dashdot"),
    "BoTorch MF-GP": (SERIES[3], "solid"),
    "mfego v2 HF only": (SERIES[4], "dot"),
}
CALIBRATION_SIZE = {"hartmann": 20, "airfoil": 10, "airfoil_xfoil": 10}
# default applied in mfego v2 (the literal rule picks "no prior", whose rho sits on its bound 5
# on Hartmann: the prior chosen by the user is kept, see RAPPORT_V2.md)
APPLIED = "centred rho, N(1, 2.0^2) (Matern)"


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


# 1/6 ---------------------------------------------------------------------------------------------
def figure_selection(selection: pd.DataFrame):
    table, chosen = decide(selection)
    variants = list(table.index)
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.04,
                        subplot_titles=["NLPD difference with mfego v1 SE (lower is better)",
                                        "RMSE ratio to mfego v1 SE (eligible if <= 1.05)"])
    for i, problem in enumerate(PROBLEMS):
        for c, column in enumerate(("dNLPD", "RMSE ratio"), start=1):
            fig.add_trace(go.Scatter(
                x=table[(column, problem)], y=variants, mode="markers", name=PROBLEMS[problem],
                legendgroup=problem, showlegend=(c == 1),
                marker={"color": SERIES[i], "size": 10},
                hovertemplate=f"<b>%{{y}}</b><br>{PROBLEMS[problem]}<br>{column} "
                              "%{x:.3f}<extra></extra>"), row=1, col=c)
    fig.add_trace(go.Scatter(x=table[("mean dNLPD", "")], y=variants, mode="markers",
                             name="mean over the 3 problems (decision)",
                             marker={"color": SERIES[7], "size": 12, "symbol": "diamond"},
                             hovertemplate="<b>%{y}</b><br>mean dNLPD %{x:.3f}<extra></extra>"),
                  row=1, col=1)
    fig.add_vline(x=0.0, line={"color": MUTED, "width": 1}, row=1, col=1)
    fig.add_vline(x=1.05, line={"color": MUTED, "width": 1, "dash": "dot"}, row=1, col=2)
    labels = {chosen: f"{chosen} [rule]", APPLIED: f"<b>{APPLIED} [applied]</b>"}
    fig.update_yaxes(autorange="reversed", ticktext=[labels.get(v, v) for v in variants],
                     tickvals=variants, row=1, col=1)
    style(fig, f"Selection of the mfego v2 defaults · applied: {APPLIED}",
          f"Literal rule: {chosen} (rho on its bound 5 in 68 % of the Hartmann fits; "
          f"N(1, 2²): 27 %, N(1, 1²): 8 %) · mean over the designs and sizes · 10 seeds · "
          f"reference {REFERENCE}", 640)
    fig.update_layout(margin={"l": 300})
    return save(fig, FIGURES, "selection")


# 2/6 ---------------------------------------------------------------------------------------------
def figure_nested(nested: pd.DataFrame):
    metrics = [("rmse_rel", "relative RMSE"), ("nlpd", "NLPD"),
               ("coverage_95", "coverage of the 95 % interval")]
    fig = make_subplots(rows=3, cols=3, horizontal_spacing=0.07, vertical_spacing=0.1,
                        subplot_titles=[f"{t} · {PROBLEMS[p]}" for p in PROBLEMS
                                        for _, t in metrics])
    offsets = np.linspace(0.93, 1.07, len(MODELS))
    for r, problem in enumerate(PROBLEMS, start=1):
        sub = nested[nested["problem"] == problem]
        for c, (metric, _) in enumerate(metrics, start=1):
            for i, (model, (color, dash)) in enumerate(MODELS.items()):
                band_line(fig, sub[sub["model"] == model], "n_hf", metric, model, color, dash,
                          r, c, offsets[i], legend=(r == 1 and c == 1))
            fig.update_xaxes(type="log", tickvals=sorted(sub["n_hf"].unique()),
                             title_text="n_HF (nested, n_LF = 4 n_HF)", row=r, col=c)
        fig.add_hline(y=0.95, line={"color": MUTED, "width": 1}, row=r, col=3)
    style(fig, "Nested designs: mfego v2 against v1 and the other libraries",
          "Median over 10 seeds (bars: interquartile range) · same nested design for every "
          "model of a seed · NF = NeuralFoil", 1150, legend_rows=2)
    return save(fig, FIGURES, "nested_accuracy")


# 3/6 ---------------------------------------------------------------------------------------------
def figure_calibration(nested: pd.DataFrame):
    shown = ["mfego v2 CoKriging", "mfego v1 CoKriging", "GPy/emukit AR1 (MAP)", "BoTorch MF-GP"]
    fig = make_subplots(rows=3, cols=2, horizontal_spacing=0.1, vertical_spacing=0.1,
                        subplot_titles=[f"{t} · {PROBLEMS[p]} (n_HF = {CALIBRATION_SIZE[p]})"
                                        for p in PROBLEMS for t in
                                        ("coverage of the central intervals",
                                         "standardized errors (y − mean) / std")])
    z_mid = 0.5 * (v2.Z_EDGES[1:] + v2.Z_EDGES[:-1])
    for r, problem in enumerate(PROBLEMS, start=1):
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="ideal",
                                 legendgroup="ideal", showlegend=(r == 1),
                                 line={"color": MUTED, "width": 1}), row=r, col=1)
        fig.add_trace(go.Scatter(x=z_mid, y=norm.pdf(z_mid), mode="lines", showlegend=False,
                                 legendgroup="ideal", line={"color": MUTED, "width": 1}),
                      row=r, col=2)
        for model in shown:
            color, dash = MODELS[model]
            sub = nested[(nested["problem"] == problem) & (nested["model"] == model)
                         & (nested["n_hf"] == CALIBRATION_SIZE[problem])]
            coverage = np.median(np.array([json.loads(c) for c in sub["coverage"]]), axis=0)
            fig.add_trace(go.Scatter(x=v2.NOMINAL, y=coverage, mode="lines+markers", name=model,
                                     legendgroup=model, showlegend=(r == 1),
                                     line={"color": color, "width": 2, "dash": dash},
                                     marker={"size": 7}), row=r, col=1)
            counts = np.sum(np.array([json.loads(h) for h in sub["z_hist"]]), axis=0)
            fig.add_trace(go.Scatter(x=z_mid, y=counts / counts.sum() / np.diff(v2.Z_EDGES),
                                     mode="lines", showlegend=False, legendgroup=model,
                                     line={"color": color, "width": 2, "dash": dash,
                                           "shape": "hvh"}), row=r, col=2)
        fig.update_xaxes(title_text="nominal probability", row=r, col=1)
        fig.update_xaxes(title_text="z (clipped at ±5)", row=r, col=2)
    style(fig, "Calibration of the predicted variance",
          "Below the diagonal: intervals too narrow · grey curve: N(0, 1) · 10 seeds pooled",
          1100)
    return save(fig, FIGURES, "calibration")


# 4/6 ---------------------------------------------------------------------------------------------
def figure_loo(nested: pd.DataFrame, infill: pd.DataFrame):
    """Leave-one-out at the HF points vs the same score on the test set, fit by fit."""
    fig = make_subplots(rows=2, cols=3, horizontal_spacing=0.07, vertical_spacing=0.16,
                        subplot_titles=[f"{t} · {PROBLEMS[p]}" for t in ("Q²", "95 % coverage")
                                        for p in PROBLEMS])
    v2_nested = nested[nested["model"] == "mfego v2 CoKriging"]
    v2_infill = infill[(infill["model"] == "EGO exact conditioning") & (infill["version"] == "v2")]
    # (label, frame, column prefix of the LOO, color, symbol)
    sources = [("nested design, LOO by location (default)", v2_nested, "loo", SERIES[0],
                "circle"),
               ("nested design, LOO by observation", v2_nested, "loo_obs", SERIES[1],
                "circle-open"),
               ("infill data (EGO v2), LOO by location", v2_infill, "loo", SERIES[6], "diamond")]
    for c, problem in enumerate(PROBLEMS, start=1):
        for r, metric in enumerate(("q2", "coverage_95"), start=1):
            test, values = metric, []
            for label, frame, prefix, color, symbol in sources:
                loo = f"{prefix}_{metric}"
                sub = frame[frame["problem"] == problem].dropna(subset=[loo])
                values += [sub[test], sub[loo]]
                fig.add_trace(go.Scatter(
                    x=sub[test], y=sub[loo], mode="markers", name=label, legendgroup=label,
                    showlegend=(r == 1 and c == 1), marker={"color": color, "size": 7,
                                                            "symbol": symbol},
                    customdata=np.column_stack([sub["n_hf"], sub["seed"]]),
                    hovertemplate=f"<b>{label}</b><br>n_HF %{{customdata[0]}}, seed "
                                  "%{customdata[1]}<br>test %{x:.3f}<br>LOO %{y:.3f}"
                                  "<extra></extra>"), row=r, col=c)
            low = float(np.nanmin(np.concatenate(values))) if values else 0.0
            fig.add_trace(go.Scatter(x=[low, 1.0], y=[low, 1.0], mode="lines",
                                     showlegend=False, line={"color": MUTED, "width": 1}),
                          row=r, col=c)
            fig.update_xaxes(title_text="on the test set", row=r, col=c)
            fig.update_yaxes(title_text="leave-one-out (HF points)", row=r, col=c)
    style(fig, "Can the leave-one-out replace a test set?",
          "One point per fit · on the diagonal, the LOO gives the score measured on the test set",
          880, legend_rows=2)
    return save(fig, FIGURES, "loo_vs_test")


# 5/6 ---------------------------------------------------------------------------------------------
def figure_infill(infill: pd.DataFrame):
    groups = {("v1", "EGO recursive"): (SERIES[1], "EGO v1, recursive model"),
              ("v2", "EGO recursive"): (SERIES[0], "EGO v2, recursive model"),
              ("v2", "EGO exact conditioning"): (SERIES[6], "EGO v2, exact conditioning")}
    metrics = [("rmse_rel", "relative RMSE"), ("nlpd", "NLPD"),
               ("coverage_95", "95 % coverage"), ("recommendation", "f at recommendation")]
    fig = make_subplots(rows=3, cols=4, horizontal_spacing=0.06, vertical_spacing=0.12,
                        subplot_titles=[f"{t} · {PROBLEMS[p]}" for p in PROBLEMS
                                        for _, t in metrics])
    for r, problem in enumerate(PROBLEMS, start=1):
        for c, (metric, _) in enumerate(metrics, start=1):
            for (version, model), (color, label) in groups.items():
                sub = infill[(infill["problem"] == problem) & (infill["version"] == version)
                             & (infill["model"] == model)]
                fig.add_trace(go.Box(
                    x=[f"cost {b:.0f}" for b in sub["budget"]], y=sub[metric], name=label,
                    legendgroup=label, showlegend=(r == 1 and c == 1), offsetgroup=label,
                    boxpoints="all", jitter=0.3, pointpos=0, marker={"color": color, "size": 5},
                    line={"color": color, "width": 1.5}, fillcolor=rgba(color, 0.12),
                    hovertemplate=f"<b>{label}</b><br>%{{y:.3f}}<extra></extra>"), row=r, col=c)
    fig.update_layout(boxmode="group")
    style(fig, "Surrogate built from the points of the infilling (NN-MF-EGO), v1 against v2",
          "10 seeds · last column: f − f* (Hartmann) or Cd in drag counts (airfoils) at the "
          "recommended design (lower is better)", 1100, legend_rows=1)
    return save(fig, FIGURES, "infill")


# 6/6 ---------------------------------------------------------------------------------------------
def figure_optimization(histories: list, runs: pd.DataFrame):
    methods = {"NN-MF-EGO v1": SERIES[1], "NN-MF-EGO v2": SERIES[0], "SF-EGO v2": SERIES[4],
               "BoTorch qLogEI (SF)": SERIES[3]}
    fig = make_subplots(rows=1, cols=2, column_widths=[0.6, 0.4], horizontal_spacing=0.1,
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
        med = np.nanmedian(np.array(curves), axis=0)
        fig.add_trace(go.Scatter(x=grid, y=med, name=method, legendgroup=method,
                                 line={"color": color, "width": 2, "shape": "hv"},
                                 hovertemplate=f"<b>{method}</b><br>cost %{{x}}<br>median "
                                               "%{y:.3f}<extra></extra>"), row=1, col=1)
        sub = runs[runs["method"] == method]
        fig.add_trace(go.Box(y=sub["recommendation error"], x=[method] * len(sub), name=method,
                             legendgroup=method, showlegend=False, boxpoints="all", jitter=0.4,
                             pointpos=0, marker={"color": color, "size": 7},
                             line={"color": color, "width": 1.5}, fillcolor=rgba(color, 0.12)),
                      row=1, col=2)
    fig.update_yaxes(type="log", row=1, col=1)
    fig.update_yaxes(type="log", row=1, col=2)
    fig.update_xaxes(title_text="cumulative cost (LF = 1, HF = 10)", row=1, col=1)
    style(fig, "Optimization of Hartmann 6D at equal budget (250)",
          "10 seeds · median of the best observation · recommendation = minimizer of the "
          "predicted mean", 600)
    return save(fig, FIGURES, "optimization")


SHORT_VARIANT = {"v1": "v1", "Matern only (v1 rho)": "Matérn<br>only",
                 "centred rho + prior only (SE)": "corrected<br>rho only", "v2": "v2"}


def figure_ego_rho(frame: pd.DataFrame):
    """Which v2 change modifies NN-MF-EGO (explore_ego_rho.py): recommendation, LF share, rho."""
    variants = {"v1": SERIES[1], "Matern only (v1 rho)": SERIES[4],
                "centred rho + prior only (SE)": SERIES[6], "v2": SERIES[0]}
    metrics = [("recommendation error", "recommendation f − f* (log)"),
               ("LF share", "share of low-fidelity evaluations"), ("final rho", "final rho")]
    fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.08,
                        subplot_titles=[m[1] for m in metrics])
    for c, (metric, _) in enumerate(metrics, start=1):
        for variant, color in variants.items():
            sub = frame[frame["variant"] == variant]
            fig.add_trace(go.Box(y=sub[metric], x=[SHORT_VARIANT[variant]] * len(sub),
                                 name=variant, showlegend=False, boxpoints="all", jitter=0.4,
                                 pointpos=0, marker={"color": color, "size": 6},
                                 line={"color": color, "width": 1.5},
                                 fillcolor=rgba(color, 0.12)), row=1, col=c)
    fig.update_yaxes(type="log", row=1, col=1)
    fig.update_xaxes(tickangle=0)
    style(fig, "NN-MF-EGO on Hartmann 6D: the corrected rho makes the merit favour the low fidelity",
          "Budget 250, 10 seeds · the merit (Sacher Eq. 24) weights the information of a lower "
          "level by rho^2", 560)
    return save(fig, FIGURES, "ego_rho")


def main() -> None:
    paths = [figure_selection(pd.read_csv(v2.RESULTS / "selection.csv"))]
    if (v2.RESULTS / "nested.csv").exists():
        nested = pd.read_csv(v2.RESULTS / "nested.csv")
        infill = pd.read_csv(v2.RESULTS / "infill.csv")
        histories = json.loads((v2.RESULTS / "optimization_histories.json")
                               .read_text(encoding="utf-8"))
        runs = pd.read_csv(v2.RESULTS / "optimization.csv")
        paths += [figure_nested(nested), figure_calibration(nested), figure_loo(nested, infill),
                  figure_infill(infill), figure_optimization(histories, runs)]
    if (v2.RESULTS / "ego_rho.csv").exists():
        paths.append(figure_ego_rho(pd.read_csv(v2.RESULTS / "ego_rho.csv")))
    for path in paths:
        print("written:", path)


if __name__ == "__main__":
    main()
