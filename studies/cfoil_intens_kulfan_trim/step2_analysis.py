"""
STEP 2 - Analysis of the optimization at the lift of the baseline foil.

Reads the last run of step 1 (results/run_summary.json, or --run <folder>) and:
1. evaluates the baseline section (MC2 thickness + camber offset) at both levels;
2. VERIFIES with AVL x XFOIL the promising sections the run did not evaluate there: the
   effective best of the surrogate, the minimum of the AVL surrogate mean over a feasible Sobol
   sample, the best NPLLT section, the best AVL sections of the other runs of this study
   (run-to-run variability) and the two optima of the first study (drag at a fixed attitude
   corrected to equal lift at first order, CD*); every candidate is evaluated at both levels
   with the trim and the current code. The optimum reported is the best MEASURED AVL section;
3. checks the first-order correction of the first study: for the same sections, CD* (fixed
   attitude, corrected) against the drag at the exactly trimmed lift;
4. measures the accuracy of the final surrogate: leave-one-out on the AVL points (the NPLLT
   points are kept), the out-of-sample predictions (periodic verifications of the run, step 2
   candidates not evaluated by the run), and how the NPLLT level ranks the AVL sections;
   it also summarizes the run itself: trim statistics (solver calls, interpolations, lift
   residuals, yaw range), wall time per phase, and the size of the root-zone correction of the
   AVL profile drag;
5. draws the figures (figures/*.html, dashboard figures/index.html) and exports the response
   surface (results/response_surface_sobol.csv, results/response_surface_slices.npz);
6. writes every number of the report in results/analysis.json.

Run (a few minutes: about 10 AVL evaluations):
    C:\\Users\\SIM\\.conda\\envs\\bdToolbox\\python.exe step2_analysis.py [--run runs/<id>]
"""
import argparse
import html
import itertools
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from scipy.stats import qmc

import study_trim as st
from study_trim import pk
from pipelines.bdtoolbox_foil import geometry  # pylint: disable=wrong-import-order
from pipelines.bdtoolbox_foil.constraints import feasibility_function
from src.data_management import ExperimentData
from src.kernels import SquaredExponentialKernel
from src.surrogate_models import MultifidelityModel

logger = logging.getLogger("step2")
TOP = 2
KEYS = ("CD", "CL", "Cy", "Cz", "Cdi", "Cdprofile", "trim_yaw", "trim_solves", "trim_residual",
        "trim_interpolated", "delta", "thickness", "cl2d_check_min")


def physical(problem, x) -> dict:
    return {v.name: float(v.lower + u * (v.upper - v.lower)) for v, u in zip(problem.variables, x)}


def evaluate_both(simulator, x, with_strips=False) -> dict:
    """Trimmed evaluation of a section at both levels (value = drag at the baseline lift)."""
    out = {}
    for level in st.LEVELS:
        value, metrics, strips = simulator.evaluate_details(x, level, with_strips=with_strips)
        out[level] = {"value": value, **{k: metrics.get(k) for k in KEYS},
                      "error": metrics.get("error"), "strips": strips}
    return out


# 1/4 ---------------------------------------------------------------------------------------------
def collect_candidates(run, problem, feasibility) -> dict:
    """Promising sections: surrogate optima, best measured sections (this run and the other
    runs of the study), first-study optima."""
    t, model = run["table"], run["surrogate"]
    candidates = {}
    best_x = run["results"]["summary"].get("surrogate_best_x")
    if best_x is not None:
        candidates["effective best of the surrogate"] = np.asarray(best_x)
    points = qmc.Sobol(problem.dim, seed=1).random_base2(14)
    points = points[feasibility(points)]
    mean = model.predict_batch(points, level=TOP)[0]
    candidates["minimum of the AVL surrogate mean"] = points[int(np.argmin(mean))]
    for level, name in ((1, "best NPLLT section of the run"), (TOP, "best AVL section of the run")):
        sub = t[t["level"] == level].dropna(subset=["value"])
        candidates[name] = np.asarray(sub.loc[sub["value"].idxmin(), "x"])
    for other in sorted((st.STUDY / "runs").glob("*/ego_backup.json")):
        if other.parent.resolve() == Path(run["run_dir"]).resolve():
            continue
        state = st.load_json(other)
        ys = np.array([np.nan if y is None else y for y in state["Y_dict"][str(TOP)]], float)
        if np.isfinite(ys).any():
            candidates[f"previous trim run {other.parent.name}, best AVL section"] =                 np.asarray(state["X_dict"][str(TOP)][int(np.nanargmin(ys))])
    first = st.FIRST_STUDY / "results"
    for file, name in (("step4_summary.json", "first study optimum (MAP run, CD*)"),
                       ("step3_summary_mle_run.json", "first study optimum (MLE run, CD*)")):
        path = first / file
        if path.is_file():
            data = st.load_json(path)
            x = data.get("best_design_normalized") or data["results"]["best_design_normalized"]
            candidates[name] = np.asarray(x)
    unique = {}
    for name, x in candidates.items():
        same = [n for n, y in unique.items() if np.linalg.norm(y - x) < 1e-9]
        if same:
            logger.info("candidate '%s' = '%s'", name, same[0])
            continue
        unique[name] = x
    return unique


def first_order_check(sections: dict) -> dict:
    """
    CD* of the first study (drag at the fixed attitude yaw 4 deg, induced drag scaled to the
    reference lift) for the given sections, with the first-study simulator (levels 2 and 3 of
    the first study = the two levels here).
    """
    problem = st.base.load_problem(st.FIRST_STUDY / "config_cfoil.json")
    simulator = st.make_simulator(problem)
    out = {}
    for name, x in sections.items():
        out[name] = {}
        for level_here, level_first in ((1, 2), (2, 3)):
            value, metrics = simulator.evaluate(x, level_first)
            out[name][level_here] = {"cd_star": value, "CD_fixed_attitude": metrics.get("CD"),
                                     "CL_fixed_attitude": metrics.get("CL")}
    return out


# 2/4 ---------------------------------------------------------------------------------------------
def scores(y, mean, var, spread) -> dict:
    var = np.maximum(var, 1e-30)
    error = mean - y
    return {"n": int(len(y)), "rmse": float(np.sqrt(np.mean(error ** 2))),
            "rmse_relative": float(np.sqrt(np.mean(error ** 2)) / spread),
            "nlpd": float(np.mean(0.5 * np.log(2 * np.pi * var) + error ** 2 / (2 * var))),
            "coverage95": float(np.mean(np.abs(error) <= 1.96 * np.sqrt(var)))}


def leave_one_out(run) -> dict:
    """Leave-one-out on the AVL points with the estimator of the run (NPLLT points kept)."""
    t = run["table"]
    opt = run["config"]["optimization"]
    low = t[(t["level"] == 1)].dropna(subset=["value"])
    top = t[(t["level"] == TOP)].dropna(subset=["value"])
    x1, y1 = np.vstack(low["x"]), low["value"].to_numpy()
    x2, y2 = np.vstack(top["x"]), top["value"].to_numpy()
    mean, var = np.zeros(len(y2)), np.zeros(len(y2))
    for i in range(len(y2)):
        keep = np.arange(len(y2)) != i
        data = ExperimentData(bounds=[(0.0, 1.0)] * x1.shape[1], costs=[1.0, 1.0])
        data.x_dict, data.y_dict = {1: x1, 2: x2[keep]}, {1: y1, 2: y2[keep]}
        data.metrics_dict = {1: [{}] * len(y1), 2: [{}] * int(keep.sum())}
        model = MultifidelityModel(2, SquaredExponentialKernel, seed=0,
                                   use_map=bool(opt.get("use_map", True)),
                                   lengthscale_prior=opt.get("lengthscale_prior"),
                                   n_restarts=int(opt.get("n_restarts", 3)))
        model.fit(data)
        m, v, _ = model.predict_batch(x2[i:i + 1])
        mean[i], var[i] = m[0], v[0]
    # how the NPLLT level (level-1 surrogate of the final model) ranks the AVL sections
    npllt = run["surrogate"].predict_batch(x2, level=1)[0]
    return {"y": y2.tolist(), "mean": mean.tolist(), "var": var.tolist(),
            "verification": top["verification"].tolist(),
            **scores(y2, mean, var, float(np.std(y2))),
            "spearman_npllt_surrogate_vs_avl": float(stats.spearmanr(npllt, y2)[0]),
            "npllt_surrogate": npllt.tolist()}


def run_statistics(run) -> dict:
    """
    Trim, timing and root-correction statistics of the run. Evaluation indices are counted in
    the run (1 = first evaluation of the initial design), not with the simulator counter, which
    also counts the cost measurement of step 1.
    """
    t = run["table"].copy()
    t["run_index"] = np.arange(1, len(t) + 1)
    ok = t.dropna(subset=["value"])
    trim = {}
    for level in st.LEVELS:
        sub = ok[ok["level"] == level]
        interpolated = sub["trim_interpolated"].fillna(False).astype(bool) \
            if "trim_interpolated" in sub else pd.Series(False, index=sub.index)
        trim[st.LEVEL_SHORT[level]] = {
            "n": int(len(sub)),
            "solves": {str(int(k)): int(v)
                       for k, v in sub["trim_solves"].value_counts().sort_index().items()},
            "n_interpolated": int(interpolated.sum()),
            "max_abs_residual": float(sub["trim_residual"].abs().max()),
            "max_abs_residual_last_solve": float(sub["trim_residual_last_solve"].abs().max())
            if "trim_residual_last_solve" in sub else None,
            "yaw_min": float(sub["trim_yaw"].min()), "yaw_max": float(sub["trim_yaw"].max())}
    timing = {}
    for level in st.LEVELS:
        sub = t[t["level"] == level]
        loop = sub[~sub["doe"] & ~sub["verification"]]
        timing[st.LEVEL_SHORT[level]] = {
            "doe_s": float(sub.loc[sub["doe"], "time_s"].sum()),
            "loop_s": float(loop["time_s"].sum()),
            "verification_s": float(sub.loc[sub["verification"], "time_s"].sum()),
            "mean_s": float(sub["time_s"].mean())}
    top = t[t["level"] == TOP]
    best = top.loc[top["value"].idxmin()]
    verifications = [{"run_index": int(r["run_index"]), "measured": float(r["value"]),
                      "predicted": r.get("surrogate_prediction"),
                      "error_percent": 100 * (float(r["value"]) - r["surrogate_prediction"])
                      / r["surrogate_prediction"]
                      if np.isfinite(r.get("surrogate_prediction", np.nan)) else None}
                     for _, r in top[top["verification"]].iterrows()]
    root = None
    if {"Cdprofile_core", "Cdprofile"} <= set(top.columns):
        share = 100 * (top["Cdprofile_core"] - top["Cdprofile"]) / top["value"]
        root = {"mean_percent_of_cd": float(share.mean()), "min_percent": float(share.min()),
                "max_percent": float(share.max())}
    merit = top[~top["doe"] & ~top["verification"]]
    return {"n_evaluations": int(len(t)), "trim": trim, "timing": timing,
            "verifications": verifications,
            "best_avl_run_index": int(best["run_index"]),
            "last_avl_run_index": int(top["run_index"].max()),
            "avl_chosen_by_merit": [int(i) for i in merit["run_index"]],
            "root_correction_avl": root}


# 3/4 ---------------------------------------------------------------------------------------------
def figure_convergence(run, baseline_cd: float, best_cd: float) -> None:
    t = run["table"]
    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.14,
                        subplot_titles=("Best AVL drag at the baseline lift vs cumulative cost",
                                        "Every evaluation in its order (ringed: verification "
                                        "of the surrogate optimum with AVL)"))
    fig.add_trace(go.Scatter(x=t["cumulative_cost"], y=t["best_top"], mode="lines",
                             line={"color": st.LEVEL_COLOR[TOP], "width": 2, "shape": "hv"},
                             name="best AVL drag so far",
                             hovertemplate="cost %{x:.0f}<br>best CD %{y:.5f}<extra></extra>"),
                  row=1, col=1)
    for y, text in ((baseline_cd, f"baseline section, AVL CD {baseline_cd:.5f}"),
                    (best_cd, f"best measured (incl. step 2 checks) {best_cd:.5f}")):
        fig.add_hline(y=y, line={"color": pk.MUTED, "width": 1}, row=1, col=1)
        fig.add_annotation(text=text, xref="x domain", x=1.0, xanchor="right", y=y, yref="y",
                           yanchor="bottom", showarrow=False,
                           font={"color": pk.SECONDARY, "size": 11})
    for level in st.LEVELS:
        sub = t[t["level"] == level]
        ring = sub["verification"].to_numpy()
        fig.add_trace(go.Scatter(
            x=sub.index + 1, y=sub["value"], mode="markers", name=st.LEVEL_NAME[level],
            marker={"color": st.LEVEL_COLOR[level], "size": np.where(ring, 13, 8).tolist(),
                    "symbol": st.LEVEL_SYMBOL[level],
                    "line": {"color": np.where(ring, pk.PRIMARY, pk.SURFACE).tolist(),
                             "width": 2}},
            customdata=np.column_stack((sub["cumulative_cost"], sub.get("trim_yaw",
                                                                         np.nan))),
            hovertemplate="evaluation %{x}<br>CD %{y:.5f}<br>cost %{customdata[0]:.0f}<br>"
                          "trimmed yaw %{customdata[1]:.3f} deg<extra></extra>"),
            row=2, col=1)
    doe_end = int(t["doe"].sum())
    fig.add_vline(x=doe_end + 0.5, line={"color": pk.MUTED, "width": 1}, row=2, col=1)
    fig.add_annotation(text="initial design | optimization loop", x=doe_end + 0.5, xref="x2",
                       y=0.98, yref="y2 domain", showarrow=False, yanchor="top",
                       bgcolor=pk.SURFACE, font={"color": pk.SECONDARY, "size": 11})
    fig.update_xaxes(title_text="cumulative cost (NPLLT evaluation = 1)", row=1, col=1)
    fig.update_xaxes(title_text="evaluation number", row=2, col=1)
    fig.update_yaxes(title_text="CD at the baseline lift", row=1, col=1)
    fig.update_yaxes(title_text="CD at its level", row=2, col=1)
    counts = t.groupby("level").size().to_dict()
    pk.style(fig, "Convergence of the optimization at the baseline lift",
             f"Evaluations per level {counts}, costs {run['costs']}; each level has its own drag "
             "offset (bottom): only AVL values are comparable to the baseline lines.", 820)
    pk.save(fig, "step2_convergence")


def figure_trim(run) -> None:
    t = run["table"].dropna(subset=["value"])
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.1,
                        subplot_titles=("Trimmed yaw of every section vs its drag",
                                        "Solver calls per trimmed evaluation"))
    for level in st.LEVELS:
        sub = t[t["level"] == level]
        fig.add_trace(go.Scatter(
            x=sub["trim_yaw"], y=sub["value"], mode="markers", name=st.LEVEL_NAME[level],
            marker={"color": st.LEVEL_COLOR[level], "size": 8, "symbol": st.LEVEL_SYMBOL[level],
                    "line": {"color": pk.SURFACE, "width": 2}},
            hovertemplate="yaw %{x:.3f} deg<br>CD %{y:.5f}<extra></extra>"), row=1, col=1)
        interpolated = sub["trim_interpolated"].fillna(False).astype(bool) \
            if "trim_interpolated" in sub else pd.Series(False, index=sub.index)
        kind = [f"{int(n)} call{'s' if n > 1 else ''}" + (" + interpolation" if i else "")
                for n, i in zip(sub["trim_solves"], interpolated)]
        counts = pd.Series(kind).value_counts().sort_index()
        fig.add_trace(go.Bar(x=counts.index.tolist(), y=counts.values,
                             name=st.LEVEL_NAME[level], showlegend=False,
                             marker={"color": st.LEVEL_COLOR[level]}, width=0.35,
                             hovertemplate="%{x}: %{y} evaluations<extra></extra>"),
                      row=1, col=2)
    fig.add_vline(x=4.0, line={"color": pk.MUTED, "width": 1}, row=1, col=1)
    fig.update_xaxes(title_text="trimmed yaw (deg); nominal 4 deg", row=1, col=1)
    fig.update_yaxes(title_text="CD at the baseline lift", row=1, col=1)
    fig.update_yaxes(title_text="evaluations", row=1, col=2)
    fig.update_layout(barmode="group")
    pk.style(fig, "The trim: how much yaw each section needs to carry the baseline lift",
             "Secant on the yaw from 4 deg until the lift is within 1e-4 of the target; the "
             "coefficients are interpolated when the last call lands within 0.5 % and the target "
             "is bracketed. The drag is then brought to the exact target lift.",
             500)
    pk.save(fig, "step2_trim")


def figure_sections(sections: list) -> None:
    """sections: list of (Section, label, colour, dash)."""
    fig = make_subplots(rows=2, cols=2, vertical_spacing=0.14, horizontal_spacing=0.08,
                        specs=[[{"colspan": 2}, None], [{}, {}]],
                        subplot_titles=("Section shapes", "Thickness t / c", "Camber / c"))
    xs = geometry.THICKNESS_STATIONS
    lines = []
    for section, label, color, dash in sections:
        c = section.coordinates
        fig.add_trace(go.Scatter(x=c[:, 0], y=c[:, 1], mode="lines", name=label,
                                 legendgroup=label, line={"color": color, "width": 2,
                                                          "dash": dash},
                                 hovertemplate=f"{label}<br>x/c %{{x:.4f}}<br>y/c %{{y:.4f}}"
                                               "<extra></extra>"), row=1, col=1)
        up, lo = geometry.surfaces_at(c, xs)
        for col, y in ((1, up - lo), (2, 0.5 * (up + lo))):
            fig.add_trace(go.Scatter(x=xs, y=y, mode="lines", name=label, legendgroup=label,
                                     showlegend=False, line={"color": color, "width": 2,
                                                             "dash": dash},
                                     hovertemplate=f"{label}<br>x/c %{{x:.3f}}<br>%{{y:.4f}}"
                                                   "<extra></extra>"), row=2, col=col)
        p = section.parameters
        thick = up - lo
        lines.append(f"{label}: t/c {section.thickness:.4f} at x/c {xs[np.argmax(thick)]:.2f}, "
                     f"t/c at 40 % {np.interp(0.4, xs, thick):.4f}, delta {p.get('delta', 0):.4f}")
    fig.update_yaxes(scaleanchor="x", scaleratio=1, row=1, col=1)
    fig.update_xaxes(title_text="x / c", row=2, col=1)
    fig.update_xaxes(title_text="x / c", row=2, col=2)
    pk.style(fig, "Baseline and optimized sections", "<br>".join(lines), 900, top=200)
    pk.save(fig, "step2_sections")


def figure_comparison(table: pd.DataFrame) -> None:
    """Drag, induced and profile drag, trimmed yaw of the compared sections at both levels."""
    keys = (("value", "CD at the baseline lift"), ("Cdi", "induced drag Cdi"),
            ("Cdprofile", "profile drag"), ("trim_yaw", "trimmed yaw (deg)"))
    fig = make_subplots(rows=1, cols=4, horizontal_spacing=0.06,
                        subplot_titles=[k[1] for k in keys])
    for col, (key, _) in enumerate(keys, start=1):
        for level in st.LEVELS:
            sub = table[table["level"] == level]
            fig.add_trace(go.Scatter(
                x=sub["label"], y=sub[key], mode="markers", name=st.LEVEL_NAME[level],
                legendgroup=str(level), showlegend=col == 1,
                marker={"color": st.LEVEL_COLOR[level], "size": 11,
                        "symbol": st.LEVEL_SYMBOL[level],
                        "line": {"color": pk.SURFACE, "width": 2}},
                hovertemplate="%{x}<br>%{y:.5f}<extra></extra>"), row=1, col=col)
        fig.update_xaxes(tickangle=-35, row=1, col=col)
    pk.style(fig, "Baseline, optimum and the other candidate sections at the baseline lift",
             "Every section trimmed to the lift of the baseline at each level. Values of the two "
             "levels differ by their own offset; compare the sections within a level.", 620)
    fig.update_layout(margin={"b": 180})
    pk.save(fig, "step2_comparison")


def figure_cdstar(check: dict, measured: dict) -> None:
    """First-order CD* (first study) vs drag at the exactly trimmed lift."""
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.1,
                        subplot_titles=[st.LEVEL_NAME[lv] for lv in st.LEVELS])
    for col, level in enumerate(st.LEVELS, start=1):
        names = list(check)
        fig.add_trace(go.Scatter(x=names, y=[check[n][level]["cd_star"] for n in names],
                                 mode="markers", name="CD* (fixed attitude, first-order lift "
                                 "correction)", showlegend=col == 1,
                                 marker={"color": pk.SECONDARY, "size": 12,
                                         "symbol": "circle-open", "line": {"width": 2}},
                                 hovertemplate="%{x}<br>CD* %{y:.5f}<extra></extra>"),
                      row=1, col=col)
        fig.add_trace(go.Scatter(x=names, y=[measured[n][level]["value"] for n in names],
                                 mode="markers", name="CD at the exactly trimmed lift",
                                 showlegend=col == 1,
                                 marker={"color": st.LEVEL_COLOR[level], "size": 10,
                                         "symbol": st.LEVEL_SYMBOL[level]},
                                 hovertemplate="%{x}<br>trimmed CD %{y:.5f}<extra></extra>"),
                      row=1, col=col)
        fig.update_xaxes(tickangle=-25, row=1, col=col)
    pk.style(fig, "Was the first-order correction of the first study right?",
             "Same sections: CD* (drag at yaw 4 deg, induced drag scaled by (CL_ref/CL)^2) "
             "against the drag with the yaw trimmed to the baseline lift.", 560)
    fig.update_layout(margin={"b": 150})
    pk.save(fig, "step2_cdstar_check")


def figure_spanwise(strips: dict) -> None:
    """strips: {(label, level): strips dict} (trimmed attitudes)."""
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.1,
                        subplot_titles=("Local lift coefficient cl", "Local profile drag cd"))
    for (label, level), s in strips.items():
        if not s:
            continue
        dash = "dot" if label == "baseline" else "solid"
        for col, key in enumerate(("cl", "cd"), start=1):
            fig.add_trace(go.Scatter(
                x=s["s"], y=s[key], mode="lines", name=f"{label}, {st.LEVEL_NAME[level]}",
                legendgroup=f"{label}{level}", showlegend=col == 1,
                line={"color": st.LEVEL_COLOR[level], "width": 2, "dash": dash},
                hovertemplate=f"s %{{x:.3f}} m<br>{key} %{{y:.4f}}<extra></extra>"),
                row=1, col=col)
        fig.update_xaxes(title_text="arc length from the root s (m)")
    fig.update_yaxes(range=[0.0, 1.2], row=1, col=1)
    pk.style(fig, "Spanwise loading at the baseline lift: baseline (dotted) vs optimum (solid)",
             "Each section at its trimmed yaw. The first point at the wall is the root "
             "singularity of the wall image.", 520)
    pk.save(fig, "step2_spanwise")


def figure_loo(loo: dict) -> None:
    y, m, s = np.array(loo["y"]), np.array(loo["mean"]), np.sqrt(np.array(loo["var"]))
    ring = np.array(loo["verification"], dtype=bool)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y, y=m, mode="markers", name="AVL point predicted without itself",
        error_y={"type": "data", "array": 1.96 * s, "thickness": 1, "width": 0,
                 "color": pk.rgba(st.LEVEL_COLOR[TOP], 0.5)},
        marker={"color": st.LEVEL_COLOR[TOP], "size": np.where(ring, 13, 8).tolist(),
                "symbol": "diamond", "line": {"color": np.where(ring, pk.PRIMARY,
                                                                pk.SURFACE).tolist(),
                                              "width": 2}},
        hovertemplate="AVL CD %{x:.5f}<br>predicted %{y:.5f}<extra></extra>"))
    lo, hi = float(min(y.min(), m.min())), float(max(y.max(), m.max()))
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="perfect prediction",
                             line={"color": pk.MUTED, "width": 1}, hoverinfo="skip"))
    fig.update_xaxes(title_text="AVL drag at the baseline lift")
    fig.update_yaxes(title_text="leave-one-out prediction +- 1.96 sigma")
    pk.style(fig, "Accuracy of the final surrogate on the AVL points (leave-one-out)",
             f"{loo['n']} AVL points (ringed: verifications of the optimum): RMSE "
             f"{100 * loo['rmse_relative']:.0f} % of their spread, 95 % coverage "
             f"{100 * loo['coverage95']:.0f} %; Spearman NPLLT surrogate vs AVL "
             f"{loo['spearman_npllt_surrogate_vs_avl']:.2f}.", 520)
    pk.save(fig, "step2_leave_one_out")


def figure_slices_1d(model, problem, x_opt, feasibility) -> dict:
    names = [v.name for v in problem.variables]
    ncols, nrows = 4, int(np.ceil(len(names) / 4))
    fig = make_subplots(rows=nrows, cols=ncols, horizontal_spacing=0.06, vertical_spacing=0.16,
                        subplot_titles=names)
    export = {}
    grid = np.linspace(0.0, 1.0, 61)
    for d, var in enumerate(problem.variables):
        points = np.repeat(x_opt[None, :], len(grid), axis=0)
        points[:, d] = grid
        feasible = feasibility(points)
        phys = var.lower + grid * (var.upper - var.lower)
        row, col = d // ncols + 1, d % ncols + 1
        for level in st.LEVELS:
            mean, var_, _ = model.predict_batch(points, level=level)
            sd = np.sqrt(np.maximum(var_, 0.0))
            mean_f = np.where(feasible, mean, np.nan)
            export[f"{var.name}_level{level}"] = np.column_stack((phys, mean, sd, feasible))
            color = st.LEVEL_COLOR[level]
            fig.add_trace(go.Scatter(x=np.concatenate((phys, phys[::-1])),
                                     y=np.concatenate((mean_f + 2 * sd, (mean_f - 2 * sd)[::-1])),
                                     fill="toself", fillcolor=pk.rgba(color, 0.12),
                                     line={"width": 0}, hoverinfo="skip", showlegend=False),
                          row=row, col=col)
            fig.add_trace(go.Scatter(x=phys, y=mean_f, mode="lines",
                                     line={"color": color, "width": 2},
                                     name=f"{st.LEVEL_NAME[level]} surrogate (mean +- 2 sigma)",
                                     legendgroup=str(level), showlegend=d == 0,
                                     hovertemplate=f"{var.name} %{{x:.4f}}<br>CD %{{y:.5f}}"
                                                   "<extra></extra>"), row=row, col=col)
        fig.add_vline(x=var.lower + x_opt[d] * (var.upper - var.lower),
                      line={"color": pk.MUTED, "width": 1}, row=row, col=col)
    pk.style(fig, "Response surface along each variable, through the optimum",
             "Other variables at the optimum (vertical line); gaps: thickness constraint "
             "violated. CD at the baseline lift, each level with its own surrogate.",
             300 * nrows + 220)
    pk.save(fig, "step2_gp_slices_1d")
    return export


def figure_slices_2d(model, problem, x_opt, feasibility, n: int = 41) -> dict:
    names = [v.name for v in problem.variables]
    pairs = list(itertools.combinations(range(len(names)), 2))
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.12,
                        subplot_titles=("AVL surrogate: mean CD at the baseline lift",
                                        "AVL surrogate: standard deviation"))
    grid = np.linspace(0.0, 1.0, n)
    export = {}
    for k, (i, j) in enumerate(pairs):
        gi, gj = np.meshgrid(grid, grid)
        points = np.repeat(x_opt[None, :], n * n, axis=0)
        points[:, i], points[:, j] = gi.ravel(), gj.ravel()
        mean, var_, _ = model.predict_batch(points, level=TOP)
        feasible = feasibility(points)
        mean = np.where(feasible, mean, np.nan).reshape(n, n)
        sd = np.where(feasible, np.sqrt(np.maximum(var_, 0.0)), np.nan).reshape(n, n)
        vi, vj = problem.variables[i], problem.variables[j]
        xi, xj = vi.lower + grid * (vi.upper - vi.lower), vj.lower + grid * (vj.upper - vj.lower)
        export[f"{vi.name}__{vj.name}"] = np.stack((mean, sd))
        for col, (z, title) in enumerate(((mean, "CD"), (sd, "sigma")), start=1):
            fig.add_trace(go.Heatmap(x=xi, y=xj, z=z, colorscale=pk.SEQUENTIAL_SCALE,
                                     visible=k == 0,
                                     colorbar={"x": 0.44 if col == 1 else 1.0, "len": 0.8,
                                               "tickfont": {"color": pk.MUTED}},
                                     hovertemplate=f"{vi.name} %{{x:.4f}}<br>{vj.name} "
                                                   f"%{{y:.4f}}<br>{title} %{{z:.5f}}"
                                                   "<extra></extra>"), row=1, col=col)
            fig.add_trace(go.Scatter(x=[vi.lower + x_opt[i] * (vi.upper - vi.lower)],
                                     y=[vj.lower + x_opt[j] * (vj.upper - vj.lower)],
                                     mode="markers", visible=k == 0, showlegend=False,
                                     marker={"color": pk.PRIMARY, "size": 11, "symbol": "x"},
                                     hovertemplate="optimum<extra></extra>"), row=1, col=col)
    buttons = []
    for k, (i, j) in enumerate(pairs):
        visible = [False] * (4 * len(pairs))
        visible[4 * k:4 * k + 4] = [True] * 4
        buttons.append({"label": f"{names[i]} x {names[j]}", "method": "update",
                        "args": [{"visible": visible},
                                 {"xaxis.title.text": names[i], "yaxis.title.text": names[j],
                                  "xaxis2.title.text": names[i], "yaxis2.title.text": names[j]}]})
    fig.update_layout(updatemenus=[{"buttons": buttons, "x": 0.0, "xanchor": "left", "y": 1.12,
                                    "yanchor": "bottom", "bgcolor": pk.SURFACE,
                                    "font": {"color": pk.PRIMARY}}])
    fig.update_xaxes(title_text=names[pairs[0][0]])
    fig.update_yaxes(title_text=names[pairs[0][1]])
    pk.style(fig, "Response surface over a pair of variables (AVL surrogate)",
             "Choose the pair in the menu; other variables at the optimum (x); blank: thickness "
             "constraint violated.", 620, top=190)
    pk.save(fig, "step2_gp_slices_2d")
    return export


def figure_parallel(run, problem) -> None:
    t = run["table"].dropna(subset=["value"])
    dims = [{"label": "level", "values": t["level"].tolist(), "tickvals": list(st.LEVELS),
             "ticktext": [st.LEVEL_SHORT[lv] for lv in st.LEVELS]}]
    for name in [v.name for v in problem.variables] + ["delta", "trim_yaw"]:
        if name in t:
            dims.append({"label": name, "values": t[name].tolist()})
    dims.append({"label": "CD", "values": t["value"].tolist()})
    fig = go.Figure(go.Parcoords(
        line={"color": t["value"].tolist(),
              "colorscale": [[1 - p, c] for p, c in pk.SEQUENTIAL_SCALE][::-1],
              "showscale": True, "colorbar": {"title": {"text": "CD"},
                                              "tickfont": {"color": pk.MUTED}}},
        dimensions=dims, labelfont={"color": pk.PRIMARY}, tickfont={"color": pk.MUTED}))
    pk.style(fig, "Every evaluated section (parallel coordinates)",
             "Drag an axis range to filter (e.g. level = AVL and the lowest CD). Dark = low drag.",
             560, top=140)
    pk.save(fig, "step2_designs_parallel")


INDEX = [("step2_convergence", "Convergence and level of every evaluation"),
         ("step2_comparison", "Baseline, optimum and other candidate sections"),
         ("step2_cdstar_check", "First-order CD* of the first study vs exact trim"),
         ("step2_sections", "Section shapes, thickness and camber"),
         ("step2_trim", "The trim: yaw needed by every section"),
         ("step2_spanwise", "Spanwise loading, baseline vs optimum"),
         ("step2_leave_one_out", "Accuracy of the surrogate (leave-one-out on AVL)"),
         ("step2_gp_slices_1d", "Response surface along each variable"),
         ("step2_gp_slices_2d", "Response surface over pairs of variables"),
         ("step2_designs_parallel", "Every evaluated section (parallel coordinates)")]


def write_index(numbers: list) -> Path:
    links = "".join(f'<li><a href="{n}.html">{html.escape(t)}</a></li>' for n, t in INDEX
                    if (st.FIGURES / f"{n}.html").is_file())
    tiles = "".join(f'<div class="tile"><div class="label">{html.escape(k)}</div>'
                    f'<div class="value">{html.escape(v)}</div></div>' for k, v in numbers)
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>C-foil trim study</title>
<style>
:root {{ --surface:#fcfcfb; --page:#f9f9f7; --ink:#0b0b0b; --ink2:#52514e;
        --line:rgba(11,11,11,0.10); --accent:#2a78d6; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --surface:#1a1a19;
  --page:#0d0d0d; --ink:#ffffff; --ink2:#c3c2b7; --line:rgba(255,255,255,0.10);
  --accent:#3987e5; }} }}
:root[data-theme="dark"] {{ --surface:#1a1a19; --page:#0d0d0d; --ink:#ffffff; --ink2:#c3c2b7;
  --line:rgba(255,255,255,0.10); --accent:#3987e5; }}
body {{ margin:0; background:var(--page); color:var(--ink);
       font-family:system-ui,-apple-system,"Segoe UI",sans-serif; }}
main {{ max-width:1100px; margin:0 auto; padding:24px 16px 48px; }}
h1 {{ font-size:22px; margin:0 0 4px; }} p.sub {{ color:var(--ink2); margin:0 0 20px; }}
.tiles {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:12px;
         margin-bottom:24px; }}
.tile {{ background:var(--surface); border:1px solid var(--line); border-radius:8px;
        padding:12px 14px; }}
.label {{ color:var(--ink2); font-size:13px; }}
.value {{ font-size:20px; font-weight:600; margin-top:4px; }}
section {{ background:var(--surface); border:1px solid var(--line); border-radius:8px;
          padding:4px 18px 10px; }}
li {{ margin:6px 0; }} a {{ color:var(--accent); }}
</style></head><body><main>
<h1>Intens SY C-foil: section optimization at the baseline lift</h1>
<p class="sub">2 levels (NPLLT x NeuralFoil xxlarge, AVL x XFOIL), yaw trimmed to the lift of the
baseline foil, CL2d(0 deg) = 0.45. See RAPPORT_TRIM.md.</p>
<div class="tiles">{tiles}</div><section><ul>{links}</ul></section></main></body></html>"""
    path = st.FIGURES / "index.html"
    path.write_text(page, encoding="utf-8")
    return path


# 4/4 ---------------------------------------------------------------------------------------------
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", help="run folder (default: the one of run_summary.json)")
    args = parser.parse_args(argv)
    st.setup_logging("step2_analysis")
    logging.getLogger("src.surrogate_models").setLevel(logging.ERROR)
    problem = st.load_problem()
    run_dir = Path(args.run) if args.run else Path(
        st.load_json(st.RESULTS / "run_summary.json")["run_dir"])
    run = st.load_run(run_dir if run_dir.is_absolute() else st.STUDY / run_dir)
    feasibility = feasibility_function(problem)
    simulator = st.make_simulator(problem)
    x_base = np.asarray(st.baseline_parameters(problem)["x"])

    # 1. baseline and candidates, trimmed, both levels
    measured = {"baseline": evaluate_both(simulator, x_base, with_strips=True)}
    candidates = collect_candidates(run, problem, feasibility)
    model = run["surrogate"]
    for name, x in candidates.items():
        measured[name] = evaluate_both(simulator, x)
        m, v, _ = model.predict_batch(x[None, :], level=TOP)
        measured[name]["prediction"] = {"mean": float(m[0]), "sd": float(np.sqrt(max(v[0], 0)))}
        logger.info("%s: AVL %s (predicted %.5f +- %.5f), NPLLT %s", name,
                    measured[name][TOP]["value"], m[0], np.sqrt(max(v[0], 0)),
                    measured[name][1]["value"])
    t = run["table"]
    run_best = t[t["level"] == TOP].dropna(subset=["value"])
    best_name, x_best = "best AVL section of the run", np.asarray(
        run_best.loc[run_best["value"].idxmin(), "x"])
    best_cd = float(run_best["value"].min())
    x_run = x_best.copy()
    run_name = next((n for n, x in candidates.items() if np.linalg.norm(x - x_run) < 1e-9), None)
    for name, x in candidates.items():
        value = measured[name][TOP]["value"]
        if value is not None and np.isfinite(value) and value < best_cd:
            best_name, x_best, best_cd = name, x, float(value)
    measured["optimum"] = evaluate_both(simulator, x_best, with_strips=True)
    baseline_cd = float(measured["baseline"][TOP]["value"])
    logger.info("Optimum (%s): AVL CD %.5f vs baseline %.5f (%.2f %%)", best_name, best_cd,
                baseline_cd, 100 * (best_cd - baseline_cd) / baseline_cd)

    # 2. first-order correction of the first study vs exact trim
    first_sections = {"baseline": x_base,
                      **{n: x for n, x in candidates.items() if n.startswith("first study")}}
    cdstar = first_order_check(first_sections)

    # 3. surrogate accuracy
    loo = leave_one_out(run)
    logger.info("Leave-one-out on %d AVL points: RMSE/spread %.3f, NLPD %.3f, coverage %.2f; "
                "Spearman NPLLT surrogate vs AVL %.2f", loo["n"], loo["rmse_relative"],
                loo["nlpd"], loo["coverage95"], loo["spearman_npllt_surrogate_vs_avl"])
    # ranking of the distinct candidates by the two levels (the "optimum" entry repeats one of
    # them; with and without the baseline). Few sections: indicative only.
    def finite(n):
        return all(measured[n][lv]["value"] is not None and np.isfinite(measured[n][lv]["value"])
                   for lv in st.LEVELS)
    distinct = [n for n in candidates if finite(n)]

    def spearman(names):
        if len(names) < 3:
            return None
        return float(stats.spearmanr([measured[n][1]["value"] for n in names],
                                     [measured[n][TOP]["value"] for n in names])[0])
    spearman_candidates = {"candidates": spearman(distinct),
                           "candidates_and_baseline": spearman(distinct + ["baseline"]),
                           "n_candidates": len(distinct)}
    # out-of-sample check: candidates the run had not evaluated with AVL
    x_top = np.vstack(t[t["level"] == TOP]["x"])
    out_of_sample = {}
    for name in distinct:
        x, p = candidates[name], measured[name]["prediction"]
        seen = bool(np.min(np.linalg.norm(x_top - x, axis=1)) < 1e-9)
        y = float(measured[name][TOP]["value"])
        out_of_sample[name] = {"in_run_data": seen, "measured": y, "predicted": p["mean"],
                               "sd": p["sd"], "z": (y - p["mean"]) / p["sd"] if p["sd"] > 0
                               else None,
                               "error_percent": 100 * (y - p["mean"]) / p["mean"]}
    run_stats = run_statistics(run)
    logger.info("Run statistics: %s", run_stats)
    cdstar_error = {n: {str(lv): 100 * (c[lv]["cd_star"] - measured[n][lv]["value"])
                        / measured[n][lv]["value"] for lv in st.LEVELS}
                    for n, c in cdstar.items()}

    # 4. figures and exports
    rows = []
    others = [n for n in candidates if n.startswith(("first study", "previous trim run"))
              and n != best_name]
    if run_name is not None and np.linalg.norm(x_best - x_run) > 1e-9:
        others.insert(0, run_name)
    display = {run_name: "best AVL section of this run"} if run_name else {}
    for name in ["baseline", "optimum"] + others:
        for level in st.LEVELS:
            rows.append({"label": display.get(name, name), "level": level,
                         **{k: measured[name][level].get(k) for k in ("value",) + KEYS}})
    comparison = pd.DataFrame(rows)
    figure_convergence(run, baseline_cd, best_cd)
    figure_trim(run)
    sections = [(simulator.section_of(x_base), "baseline", pk.SECONDARY, "dot"),
                (simulator.section_of(x_best), "optimum (best measured section)", pk.PRIMARY,
                 "solid")]
    if np.linalg.norm(x_best - x_run) > 1e-9:
        sections.append((simulator.section_of(x_run), "best AVL section of this run",
                         st.LEVEL_COLOR[TOP], "dashdot"))
    if "first study optimum (MAP run, CD*)" in candidates:
        sections.append((simulator.section_of(candidates["first study optimum (MAP run, CD*)"]),
                         "first study optimum (CD*)", pk.LEVEL_COLOR[1], "dash"))
    figure_sections(sections)
    figure_comparison(comparison)
    figure_cdstar(cdstar, measured)
    figure_spanwise({(label, level): measured[label][level]["strips"]
                     for label in ("baseline", "optimum") for level in st.LEVELS})
    figure_loo(loo)
    slices_1d = figure_slices_1d(model, problem, x_best, feasibility)
    slices_2d = figure_slices_2d(model, problem, x_best, feasibility)
    figure_parallel(run, problem)
    points = qmc.Sobol(problem.dim, seed=0).random_base2(12)
    sobol = pd.DataFrame(points, columns=[f"x_{i}" for i in range(problem.dim)])
    for d, var in enumerate(problem.variables):
        sobol[var.name] = var.lower + points[:, d] * (var.upper - var.lower)
    sobol["feasible"] = feasibility(points)
    for level in st.LEVELS:
        mean, var_, _ = model.predict_batch(points, level=level)
        sobol[f"mean_level{level}"], sobol[f"sd_level{level}"] = mean, np.sqrt(np.maximum(var_, 0))
    sobol.to_csv(st.RESULTS / "response_surface_sobol.csv", index=False)
    np.savez_compressed(st.RESULTS / "response_surface_slices.npz",
                        **{f"slice1d_{k}": v for k, v in slices_1d.items()},
                        **{f"slice2d_{k}": v for k, v in slices_2d.items()})

    gain = 100 * (best_cd - baseline_cd) / baseline_cd
    analysis = {"run_dir": str(run_dir), "baseline_cd_avl": baseline_cd, "best_cd_avl": best_cd,
                "gain_percent": gain, "best_source": best_name,
                "best_design_normalized": x_best.tolist(),
                "best_design_physical": physical(problem, x_best),
                "measured": {n: {str(lv): {k: v for k, v in m[lv].items() if k != "strips"}
                                 for lv in st.LEVELS} | ({"prediction": m["prediction"]}
                                                         if "prediction" in m else {})
                             for n, m in measured.items()},
                "first_order_check": {n: {str(lv): v for lv, v in c.items()}
                                      for n, c in cdstar.items()},
                "leave_one_out": {k: v for k, v in loo.items()
                                  if k not in ("y", "mean", "var", "npllt_surrogate",
                                               "verification")},
                "spearman_candidates_npllt_vs_avl": spearman_candidates,
                "predictions_of_candidates": out_of_sample,
                "cdstar_error_percent": cdstar_error,
                "run_statistics": run_stats,
                "lift_residual_optimum": {str(lv): (measured["optimum"][lv]["CL"]
                                                    - simulator.reference_lift[lv])
                                          / simulator.reference_lift[lv] for lv in st.LEVELS},
                "evaluations": t.groupby("level").size().to_dict(),
                "verifications": int(t["verification"].sum()),
                "gp": run["results"].get("gp"), "costs": run["costs"]}
    st.save_json(st.RESULTS / "analysis.json", analysis)
    numbers = [("Baseline CD (AVL)", f"{baseline_cd:.5f}"), ("Optimum CD (AVL)", f"{best_cd:.5f}"),
               ("Drag change", f"{gain:+.2f} %"),
               ("Evaluations NPLLT / AVL", " / ".join(str(int((t['level'] == lv).sum()))
                                                      for lv in st.LEVELS)),
               ("LOO RMSE / spread (AVL)", f"{100 * loo['rmse_relative']:.0f} %")]
    path = write_index(numbers)
    logger.info("Analysis written: %s, %s", st.RESULTS / "analysis.json", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
