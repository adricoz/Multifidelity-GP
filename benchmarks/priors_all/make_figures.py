"""
Interactive figures (plotly HTML) of the benchmark of the priors on all the hyperparameters,
built from the results of run_priors_benchmark.py. Output: benchmarks/priors_all/figures/.

Run from the repository root (benchmark environment):
    .venv-benchmark\\Scripts\\python benchmarks\\priors_all\\make_figures.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
# pylint: disable=wrong-import-position,import-error
import bench_lib  # noqa: E402,F401  (puts mfego/ on sys.path)
from figure_style import MUTED, SERIES, hline, rgba, save, style  # noqa: E402
from src.surrogate_models import (BIAS_VARIANCE_BOUNDS, DEFAULT_PRIORS,  # noqa: E402
                                  LENGTHSCALE_BOUNDS, NOISE_BOUNDS, SIGNAL_VARIANCE_BOUNDS)

RESULTS, FIGURES = HERE / "results", HERE / "figures"
# fixed color of each estimator / library in every figure
COLOR = {"MAP-all": SERIES[0], "MLE": SERIES[1], "MAP-l": SERIES[2], "BoTorch": SERIES[3],
         "other": SERIES[4]}
LABEL = {"MAP-all": "mfego MAP, priors on all the hyperparameters (new)",
         "MAP-l": "mfego MAP, lengthscale prior only (previous)",
         "MLE": "mfego maximum likelihood (no prior)"}
GROUPS = [("lengthscales", "lengthscale", "median lengthscale l", LENGTHSCALE_BOUNDS),
          ("signal_variance", "signal_variance", "signal variance t1", SIGNAL_VARIANCE_BOUNDS),
          ("bias_variance", "bias_variance", "bias variance t2", BIAS_VARIANCE_BOUNDS),
          ("noise", "noise", "noise (normalized)", NOISE_BOUNDS)]


def median_band(frame: pd.DataFrame, by: str, metric: str) -> tuple:
    grouped = frame.groupby(by)[metric]
    return (grouped.median(), grouped.quantile(0.25), grouped.quantile(0.75))


def line_with_band(fig, frame, by, metric, name, color, row, col, offset=1.0, legend=True,
                   group=None):
    """Median over the seeds (line + markers) with the interquartile range as error bars."""
    med, q1, q3 = median_band(frame, by, metric)
    x = med.index.to_numpy(dtype=float) * offset
    fig.add_trace(go.Scatter(
        x=x, y=med, mode="lines+markers", name=name, legendgroup=group or name,
        showlegend=legend, line={"color": color, "width": 2}, marker={"size": 8},
        error_y={"type": "data", "symmetric": False, "array": q3 - med, "arrayminus": med - q1,
                 "color": rgba(color, 0.6), "thickness": 1.5, "width": 4},
        customdata=np.column_stack([med.index, q1, q3]),
        hovertemplate=f"<b>{name}</b><br>{by} = %{{customdata[0]}}<br>median %{{y:.3f}}"
                      "<br>quartiles %{customdata[1]:.3f} – %{customdata[2]:.3f}<extra></extra>"),
        row=row, col=col)


# 1/6 ---------------------------------------------------------------------------------------------
def figure_accuracy(acc: pd.DataFrame) -> Path:
    rows = {"MF": [("MLE", "mfego MF (MLE)"), ("MAP-l", "mfego MF (MAP-l)"),
                   ("MAP-all", "mfego MF (MAP-all)"), ("BoTorch", "BoTorch MF-GP (MF)"),
                   ("other", "SMT MFK (MF)")],
            "SF": [("MLE", "mfego SF (MLE)"), ("MAP-l", "mfego SF (MAP-l)"),
                   ("MAP-all", "mfego SF (MAP-all)"), ("BoTorch", "BoTorch SingleTaskGP (SF)"),
                   ("other", "scikit-learn GP (SF)")]}
    metrics = [("rmse_rel", "relative RMSE (1 = mean predictor)", "linear"),
               ("nlpd", "NLPD (log scale, lower is better)", "log"),
               ("coverage_95", "coverage of the 95 % interval", "linear")]
    titles = [f"{title} · {'multi-fidelity' if v == 'MF' else 'high fidelity only'}"
              for v in rows for _, title, _ in metrics]
    fig = make_subplots(rows=2, cols=3, subplot_titles=titles, horizontal_spacing=0.08,
                        vertical_spacing=0.16)
    offsets = [0.94, 0.97, 1.0, 1.03, 1.06]
    names = {**LABEL, "BoTorch": "BoTorch (MF-GP / SingleTaskGP)",
             "other": "SMT MFK (multi-fidelity) / scikit-learn (HF only)"}
    for r, models in enumerate(rows.values(), start=1):
        for c, (metric, _, axis) in enumerate(metrics, start=1):
            for i, (key, model) in enumerate(models):
                line_with_band(fig, acc[acc["model"] == model], "n_hf", metric, names[key],
                               COLOR[key], r, c, offsets[i], legend=(r == 1 and c == 1),
                               group=key)
            fig.update_xaxes(type="log", tickvals=[5, 10, 20, 40], title_text="n_HF", row=r,
                             col=c)
            fig.update_yaxes(type=axis, row=r, col=c)
        fig.add_hline(y=0.95, line={"color": MUTED, "width": 1}, row=r, col=3)
    style(fig, "Surrogate accuracy on Hartmann 6D: priors on all the hyperparameters",
          "Median over 10 seeds (bars: interquartile range) · 2000 test points · n_LF = 2 n_HF "
          "· same design for every model of a seed", 820, legend_rows=2)
    return save(fig, FIGURES, "accuracy_vs_n")


# 2/6 ---------------------------------------------------------------------------------------------
def fitted_values(acc: pd.DataFrame, model: str, group: str) -> pd.DataFrame:
    """Fitted HF-level value of a hyperparameter group (median over the d lengthscales)."""
    rows = []
    for _, row in acc[(acc["model"] == model) & acc["hyper"].notna()].iterrows():
        value = json.loads(row["hyper"])[-1][group]
        rows.append({"n_hf": row["n_hf"], "seed": row["seed"],
                     "value": float(np.median(value))})
    return pd.DataFrame(rows)


def figure_hyperparameters(acc: pd.DataFrame) -> Path:
    fig = make_subplots(rows=2, cols=2, subplot_titles=[g[2] for g in GROUPS],
                        horizontal_spacing=0.1, vertical_spacing=0.16)
    for i, (group, prior_key, title, (lo, hi)) in enumerate(GROUPS):
        r, c = divmod(i, 2)
        fig.update_yaxes(type="log", row=r + 1, col=c + 1)
        fig.update_xaxes(type="category", title_text="n_HF", row=r + 1, col=c + 1)
        for key in ("MLE", "MAP-l", "MAP-all"):
            values = fitted_values(acc, f"mfego MF ({key})", group)
            fig.add_trace(go.Box(
                x=values["n_hf"].astype(str), y=values["value"], name=LABEL[key],
                legendgroup=key, showlegend=(i == 0), offsetgroup=key, boxpoints="all",
                jitter=0.4, pointpos=0, marker={"color": COLOR[key], "size": 5},
                line={"color": COLOR[key], "width": 1.5}, fillcolor=rgba(COLOR[key], 0.12),
                hovertemplate=f"<b>{LABEL[key]}</b><br>n_HF = %{{x}}<br>{title} = "
                              "%{y:.3g}<extra></extra>"), row=r + 1, col=c + 1)
        hline(fig, lo, "lower search bound", r + 1, c + 1, position="top left")
        hline(fig, hi, "upper search bound", r + 1, c + 1, position="bottom left")
        family, a, b = DEFAULT_PRIORS[prior_key]
        mode = b / (a + 1.0) if family == "invgamma" else float(np.exp(a - b ** 2))
        hline(fig, mode, f"mode of the default prior ({mode:.2g})", r + 1, c + 1,
              color=COLOR["MAP-all"], dash="solid", position="top right")
    fig.update_layout(boxmode="group")
    style(fig, "Fitted hyperparameters of the high-fidelity level (multi-fidelity model)",
          "One point per seed · without prior, t2 and the noise are not identified by "
          "deterministic data: the MAP puts them at the mode of their prior", 860)
    return save(fig, FIGURES, "hyperparameters")


# 3/6 ---------------------------------------------------------------------------------------------
def figure_many_points(large: pd.DataFrame) -> Path:
    models = [("MLE", "mfego SF (MLE)"), ("MAP-l", "mfego SF (MAP-l)"),
              ("MAP-all", "mfego SF (MAP-all)"), ("BoTorch", "BoTorch SingleTaskGP (SF)"),
              ("other", "scikit-learn GP (SF)")]
    metrics = [("rmse_rel", "relative RMSE"), ("nlpd", "NLPD"), ("fit_time_s", "fit time (s)")]
    fig = make_subplots(rows=1, cols=3, subplot_titles=[m[1] for m in metrics],
                        horizontal_spacing=0.08)
    offsets = [0.94, 0.97, 1.0, 1.03, 1.06]
    for c, (metric, _) in enumerate(metrics, start=1):
        for i, (key, model) in enumerate(models):
            line_with_band(fig, large[large["model"] == model], "n_hf", metric,
                           LABEL.get(key, model), COLOR[key], 1, c, offsets[i],
                           legend=(c == 1), group=model)
        fig.update_xaxes(type="log", tickvals=[80, 160, 300], title_text="n (high fidelity)",
                         row=1, col=c)
    fig.update_yaxes(type="log", row=1, col=3)
    style(fig, "Many high-fidelity points: the maximum likelihood of mfego degenerates",
          "Single fidelity, median over 10 seeds · from n = 160 the MLE stops on lengthscales "
          "≈ 0.01 (white-noise model, RMSE ≈ 1); any MAP avoids it", 560, legend_rows=2)
    return save(fig, FIGURES, "many_points")


# 4/6 ---------------------------------------------------------------------------------------------
OPT_METHODS = [("MLE", "mfego NN-MF-EGO (MLE)"), ("MAP-l", "mfego NN-MF-EGO (MAP-l)"),
               ("MAP-all", "mfego NN-MF-EGO (MAP-all)"), ("BoTorch", "BoTorch qLogEI (SF)"),
               ("other", "mfego SF-EGO (MAP-all)")]
SHORT = {"mfego NN-MF-EGO (MLE)": "NN-MF-EGO<br>MLE", "mfego NN-MF-EGO (MAP-l)":
         "NN-MF-EGO<br>MAP-l", "mfego NN-MF-EGO (MAP-all)": "NN-MF-EGO<br>MAP-all",
         "BoTorch qLogEI (SF)": "BoTorch<br>qLogEI (SF)", "mfego SF-EGO (MAP-all)":
         "SF-EGO<br>MAP-all"}


def figure_optimization(histories: list, runs: pd.DataFrame) -> Path:
    fig = make_subplots(rows=1, cols=2, column_widths=[0.6, 0.4], horizontal_spacing=0.1,
                        subplot_titles=["best high-fidelity observation f - f* vs cost "
                                        "(median, interquartile band)",
                                        "final recommendation f - f* (one point per seed)"])
    grid = np.arange(120, 251, 2.0)
    for key, method in OPT_METHODS:
        name = LABEL.get(key, method).replace("mfego ", "NN-MF-EGO, ") \
            if key in LABEL else method
        curves = []
        for h in histories:
            if h.get("method") != method or "error" in h:
                continue
            cost, best = np.array(h["cost"]), np.array(h["best_observed"])
            index = np.searchsorted(cost, grid, side="right") - 1
            curves.append(np.where(index >= 0, best[np.maximum(index, 0)], np.nan))
        curves = np.array(curves)
        med = np.nanmedian(curves, axis=0)
        q1, q3 = np.nanquantile(curves, 0.25, axis=0), np.nanquantile(curves, 0.75, axis=0)
        fig.add_trace(go.Scatter(x=np.r_[grid, grid[::-1]], y=np.r_[q3, q1[::-1]],
                                 fill="toself", fillcolor=rgba(COLOR[key], 0.06),
                                 line={"width": 0}, hoverinfo="skip", showlegend=False,
                                 legendgroup=method), row=1, col=1)
        fig.add_trace(go.Scatter(x=grid, y=med, name=name, legendgroup=method,
                                 line={"color": COLOR[key], "width": 2, "shape": "hv"},
                                 hovertemplate=f"<b>{name}</b><br>cost %{{x}}<br>median "
                                               "%{y:.3f}<extra></extra>"), row=1, col=1)
        sub = runs[runs["method"] == method]
        fig.add_trace(go.Box(y=sub["recommendation error"], x=[SHORT[method]] * len(sub),
                             name=name, legendgroup=method, showlegend=False, boxpoints="all",
                             jitter=0.4, pointpos=0, marker={"color": COLOR[key], "size": 7},
                             line={"color": COLOR[key], "width": 1.5},
                             fillcolor=rgba(COLOR[key], 0.12),
                             customdata=sub["seed"],
                             hovertemplate=f"<b>{name}</b><br>seed %{{customdata}}<br>"
                                           "f - f* = %{y:.3f}<extra></extra>"), row=1, col=2)
    fig.update_xaxes(title_text="cumulative cost (LF = 1, HF = 10)", row=1, col=1)
    fig.update_yaxes(type="log", row=1, col=1)
    fig.update_yaxes(type="log", row=1, col=2)
    fig.update_xaxes(tickangle=-20, row=1, col=2)
    style(fig, "Optimization of Hartmann 6D at equal budget (250)",
          "10 seeds · initial design of cost 120 (MF: 20 LF + 10 HF, SF: 12 HF) · "
          "recommendation = minimizer of the predicted mean", 600, legend_rows=2)
    return save(fig, FIGURES, "optimization")


# 5/6 ---------------------------------------------------------------------------------------------
def figure_sensitivity(acc: pd.DataFrame, runs: pd.DataFrame) -> Path:
    variants = ["MLE", "MAP-l", "MAP-all", "MAP-all, l IG(3, 1)", "MAP-all, t1 sigma 0.5",
                "MAP-all, t1 sigma 2", "MAP-all, noise 1e-6", "MAP-all, noise 1e-3",
                "MAP-all, no t1 prior", "MAP-all, no t2 prior", "MAP-all, no noise prior"]
    panels = [("NLPD, n_HF = 10", lambda v: acc[(acc["model"] == f"mfego MF ({v})")
                                               & (acc["n_hf"] == 10)]["nlpd"]),
              ("NLPD, n_HF = 40", lambda v: acc[(acc["model"] == f"mfego MF ({v})")
                                               & (acc["n_hf"] == 40)]["nlpd"]),
              ("NN-MF-EGO recommendation f - f*",
               lambda v: runs[runs["method"] == f"mfego NN-MF-EGO ({v})"]
               ["recommendation error"])]
    fig = make_subplots(rows=1, cols=3, subplot_titles=[p[0] for p in panels],
                        shared_yaxes=True, horizontal_spacing=0.04)
    for c, (title, select) in enumerate(panels, start=1):
        for v in variants:
            values = select(v)
            key = v if v in COLOR else "MAP-all"
            med, q1, q3 = values.median(), values.quantile(0.25), values.quantile(0.75)
            fig.add_trace(go.Scatter(
                x=[med], y=[v], mode="markers", showlegend=False,
                marker={"color": COLOR[key], "size": 10},
                error_x={"type": "data", "symmetric": False, "array": [q3 - med],
                         "arrayminus": [med - q1], "color": rgba(COLOR[key], 0.6),
                         "thickness": 1.5, "width": 4},
                hovertemplate=f"<b>{v}</b><br>{title}<br>median {med:.3f}<br>quartiles "
                              f"{q1:.3f} – {q3:.3f}<extra></extra>"), row=1, col=c)
        fig.update_xaxes(type="log", row=1, col=c)
    fig.update_yaxes(autorange="reversed", row=1, col=1)
    style(fig, "Sensitivity of the new priors (one factor at a time around MAP-all)",
          "Median and interquartile range over 10 seeds · t1 priors keep their mode at 1 · "
          "noise variants: median of the noise prior", 620)
    fig.update_layout(margin={"l": 190})
    return save(fig, FIGURES, "sensitivity")


# 6/6 ---------------------------------------------------------------------------------------------
def figure_prior_shapes(acc: pd.DataFrame) -> Path:
    fig = make_subplots(rows=2, cols=2, subplot_titles=[g[2] for g in GROUPS],
                        horizontal_spacing=0.1, vertical_spacing=0.18)
    for i, (group, prior_key, title, (lo, hi)) in enumerate(GROUPS):
        r, c = divmod(i, 2)
        family, a, b = DEFAULT_PRIORS[prior_key]
        dist = stats.invgamma(a, scale=b) if family == "invgamma" else \
            stats.lognorm(b, scale=np.exp(a))
        theta = np.logspace(np.log10(lo) - 1, np.log10(hi) + 1, 600)
        density = dist.pdf(theta) / np.max(dist.pdf(theta))  # peak = mode = target of the MAP
        fig.add_trace(go.Scatter(x=theta, y=density, name="default prior p(θ) / max",
                                 showlegend=(i == 0), legendgroup="prior",
                                 line={"color": COLOR["MAP-all"], "width": 2},
                                 hovertemplate=f"{title} = %{{x:.3g}}<br>p / max = "
                                               "%{y:.3f}<extra></extra>"), row=r + 1, col=c + 1)
        fig.add_vrect(x0=theta[0], x1=lo, fillcolor=rgba(MUTED, 0.12), line_width=0,
                      row=r + 1, col=c + 1)
        fig.add_vrect(x0=hi, x1=theta[-1], fillcolor=rgba(MUTED, 0.12), line_width=0,
                      row=r + 1, col=c + 1)
        values = fitted_values(acc, "mfego MF (MAP-all)", group)["value"]
        fig.add_trace(go.Scatter(x=values, y=np.zeros(len(values)), mode="markers",
                                 name="fitted (MAP-all, HF level, every n_HF and seed)",
                                 showlegend=(i == 0), legendgroup="fitted",
                                 marker={"color": SERIES[2], "size": 7, "symbol": "line-ns-open",
                                         "line": {"width": 2, "color": SERIES[2]}},
                                 hovertemplate=f"fitted {title} = %{{x:.3g}}<extra></extra>"),
                      row=r + 1, col=c + 1)
        fig.update_xaxes(type="log", title_text=f"{title} (grey: outside the search bounds)",
                         row=r + 1, col=c + 1)
    style(fig, "Default priors of mfego (DEFAULT_PRIORS) and fitted values",
          "Relative density p(θ) / max on a log axis: the MAP (taken in θ, no Jacobian) is pulled "
          "towards its peak, the mode β/(α+1) (InvGamma) or exp(μ − σ²) (LogNormal)", 760)
    return save(fig, FIGURES, "prior_shapes")


def main() -> None:
    acc = pd.read_csv(RESULTS / "accuracy.csv")
    large = pd.read_csv(RESULTS / "large_n.csv")
    runs = pd.read_csv(RESULTS / "optimization.csv")
    histories = json.loads((RESULTS / "optimization_histories.json").read_text(encoding="utf-8"))
    for path in (figure_accuracy(acc), figure_hyperparameters(acc), figure_many_points(large),
                 figure_optimization(histories, runs), figure_sensitivity(acc, runs),
                 figure_prior_shapes(acc)):
        print("written:", path)


if __name__ == "__main__":
    main()
