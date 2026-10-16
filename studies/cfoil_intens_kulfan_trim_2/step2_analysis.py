"""
STEP 2 - Analysis of the optimization at the lift of the reference foil (three levels).

Reads the last run of step 1 (results/run_summary.json, or --run <folder>) and:
1. evaluates the reference section (MC2_60_6101_1 thickness + frozen camber of the first study)
   at the three levels: NPLLT x NeuralFoil xxxlarge, NPLLT x XFOIL, AVL x XFOIL;
2. VERIFIES at the three levels the promising sections: the effective best of the surrogate, the
   minimum of the level-3 surrogate mean over a feasible Sobol sample, the best sections of the
   run inside the yaw band at each level, and the best sections of the other runs of this study
   with the same design space; every candidate satisfies the known constraints of the study
   (study_trim.feasibility_function). The optimum reported is the best MEASURED AVL x XFOIL
   section whose trimmed yaw lies inside the +-5 % band (never a penalized one);
3. measures the accuracy of the final surrogate: leave-one-out on the level-3 points (the other
   levels kept), the out-of-sample predictions of the candidates, and how the two lower levels
   rank the level-3 sections; it also summarizes the run: trim statistics (solver calls,
   interpolations, yaw range, sections outside the band), failed evaluations and their causes
   (XFOIL, trim), XFOIL points filled by interpolation, wall time per phase;
4. draws the figures (figures/*.html, dashboard figures/index.html) and exports the response
   surface (results/response_surface_sobol.csv, results/response_surface_slices.npz);
5. writes every number of the report in results/analysis.json.

Run (a few minutes: about 10 sections at the three levels):
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
from src.data_management import ExperimentData
from src.kernels import SquaredExponentialKernel
from src.surrogate_models import MultifidelityModel

logger = logging.getLogger("step2")
TOP = st.LEVELS[-1]
KEYS = ("CD", "CD_unpenalized", "CL", "Cy", "Cz", "Cdi", "Cdprofile", "trim_yaw", "trim_solves",
        "trim_residual", "trim_interpolated", "trim_in_band", "yaw_excess", "thickness",
        "cl2d_check_min", "nf_confidence_min", "xfoil_filled_points", "time_s")


def physical(problem, x) -> dict:
    return {v.name: float(v.lower + u * (v.upper - v.lower)) for v, u in zip(problem.variables, x)}


def evaluate_levels(simulator, x, with_strips=False) -> dict:
    """Trimmed evaluation of a section at the three levels (value = penalized drag)."""
    out = {}
    for level in st.LEVELS:
        value, metrics, strips = simulator.evaluate_details(x, level, with_strips=with_strips)
        out[level] = {"value": value, **{k: metrics.get(k) for k in KEYS},
                      "error": metrics.get("error"), "strips": strips}
    return out


def short_label(name: str) -> str:
    """Short axis label of a candidate section."""
    replacements = (("effective best of the surrogate", "surrogate best"),
                    ("minimum of the AVL surrogate mean", "min. AVL mean"),
                    ("AVL section of the run, rank ", "AVL rank "), (" section of the run", ""),
                    (", best AVL section", " (other run)"))
    for old, new in replacements:
        name = name.replace(old, new)
    return name


def admissible(levels: dict) -> bool:
    """
    A section measured at the three levels counts as a result: finite drag at the top level and
    trimmed yaw inside the band at every level where the band is enforced (st.BAND_LEVELS).
    """
    value = levels[TOP].get("value")
    return value is not None and np.isfinite(value) and all(
        bool(levels[lv].get("trim_in_band")) for lv in st.BAND_LEVELS)


# 1/4 ---------------------------------------------------------------------------------------------
def same_design_space(config: dict, problem) -> bool:
    """True if a run configuration has the variables, the reference and the levels of this one."""
    variables = [(v["name"], v["lower"], v["upper"]) for v in config.get("variables", [])]
    return variables == [(v.name, v.lower, v.upper) for v in problem.variables] and \
        config.get("objective", {}).get("equal_lift_reference") == \
        problem.objective.get("equal_lift_reference") and \
        [lv["solver"] for lv in config.get("levels", [])] == [lv.solver for lv in problem.levels]


def collect_candidates(run, problem, feasibility) -> dict:
    """Promising FEASIBLE sections: surrogate optima, best measured sections inside the yaw band
    (this run, each level, and the other runs of the study with the same design space)."""
    t, model = run["table"], run["surrogate"]
    candidates = {}
    best_x = run["results"]["summary"].get("surrogate_best_x")
    if best_x is not None:
        candidates["effective best of the surrogate"] = np.asarray(best_x)
    points = qmc.Sobol(problem.dim, seed=1).random_base2(14)
    points = points[feasibility(points)]
    mean = model.predict_batch(points, level=TOP)[0]
    candidates["minimum of the AVL surrogate mean"] = points[int(np.argmin(mean))]
    for level in st.LEVELS:
        sub = t[(t["level"] == level) & t["admissible"]]
        if len(sub):
            candidates[f"best {st.LEVEL_SHORT[level]} section of the run"] = np.asarray(
                sub.loc[sub["value"].idxmin(), "x"])
    sub = t[(t["level"] == TOP) & t["admissible"]].nsmallest(3, "value")
    for rank, (_, row) in enumerate(sub.iloc[1:].iterrows(), start=2):
        candidates[f"AVL section of the run, rank {rank}"] = np.asarray(row["x"])
    for other in sorted((st.STUDY / "runs").glob("*/ego_backup.json")):
        if other.parent.resolve() == Path(run["run_dir"]).resolve():
            continue
        config = other.parent / "config_effective.json"
        if not config.is_file() or not same_design_space(st.load_json(config), problem):
            continue
        other_run = st.load_run(other.parent)["table"]
        sub = other_run[(other_run["level"] == TOP) & other_run["admissible"]]
        if len(sub):
            candidates[f"run {other.parent.name}, best AVL section"] = np.asarray(
                sub.loc[sub["value"].idxmin(), "x"])
    unique = {}
    for name, x in candidates.items():
        if not feasibility(x)[0]:
            logger.info("candidate '%s' skipped: known constraints violated", name)
            continue
        same = [n for n, y in unique.items() if np.linalg.norm(y - x) < 1e-9]
        if same:
            logger.info("candidate '%s' = '%s'", name, same[0])
            continue
        unique[name] = x
    return unique


# 2/4 ---------------------------------------------------------------------------------------------
def scores(y, mean, var, spread) -> dict:
    var = np.maximum(var, 1e-30)
    error = mean - y
    return {"n": int(len(y)), "rmse": float(np.sqrt(np.mean(error ** 2))),
            "rmse_relative": float(np.sqrt(np.mean(error ** 2)) / spread),
            "nlpd": float(np.mean(0.5 * np.log(2 * np.pi * var) + error ** 2 / (2 * var))),
            "coverage95": float(np.mean(np.abs(error) <= 1.96 * np.sqrt(var)))}


def leave_one_out(run) -> dict:
    """Leave-one-out on the level-3 points with the estimator of the run (other levels kept).
    Values are those the surrogate was trained on (penalized outside the yaw band)."""
    t = run["table"]
    opt = run["config"]["optimization"]
    data_levels = {}
    for level in st.LEVELS:
        sub = t[t["level"] == level].dropna(subset=["value"])
        data_levels[level] = (np.vstack(sub["x"]), sub["value"].to_numpy(), sub)
    x_top, y_top, top = data_levels[TOP]
    mean, var = np.zeros(len(y_top)), np.zeros(len(y_top))
    for i in range(len(y_top)):
        keep = np.arange(len(y_top)) != i
        data = ExperimentData(bounds=[(0.0, 1.0)] * x_top.shape[1], costs=[1.0] * len(st.LEVELS))
        data.x_dict = {lv: data_levels[lv][0] for lv in st.LEVELS[:-1]} | {TOP: x_top[keep]}
        data.y_dict = {lv: data_levels[lv][1] for lv in st.LEVELS[:-1]} | {TOP: y_top[keep]}
        data.metrics_dict = {lv: [{}] * len(data.y_dict[lv]) for lv in st.LEVELS}
        model = MultifidelityModel(len(st.LEVELS), SquaredExponentialKernel, seed=0,
                                   use_map=bool(opt.get("use_map", True)),
                                   lengthscale_prior=opt.get("lengthscale_prior"),
                                   n_restarts=int(opt.get("n_restarts", 3)))
        model.fit(data)
        m, v, _ = model.predict_batch(x_top[i:i + 1])
        mean[i], var[i] = m[0], v[0]
    # how the lower levels (their surrogates in the final model) rank the level-3 sections
    lower = {level: run["surrogate"].predict_batch(x_top, level=level)[0]
             for level in st.LEVELS[:-1]}
    return {"y": y_top.tolist(), "mean": mean.tolist(), "var": var.tolist(),
            "verification": top["verification"].tolist(),
            "in_band": top["trim_in_band"].tolist(),
            **scores(y_top, mean, var, float(np.std(y_top))),
            **{f"spearman_level{lv}_surrogate_vs_avl": float(stats.spearmanr(p, y_top)[0])
               for lv, p in lower.items()}}


def run_statistics(run) -> dict:
    """
    Trim, band, failure, XFOIL and timing statistics of the run. Evaluation indices are counted
    in the run (1 = first evaluation of the initial design), not with the simulator counter,
    which also counts the cost measurement of step 1.
    """
    t = run["table"].copy()
    t["run_index"] = np.arange(1, len(t) + 1)
    ok = t.dropna(subset=["value"])
    trim, failures, xfoil = {}, {}, {}
    for level in st.LEVELS:
        name = st.LEVEL_SHORT[level]
        sub = ok[ok["level"] == level]
        interpolated = sub["trim_interpolated"].fillna(False).astype(bool) \
            if "trim_interpolated" in sub else pd.Series(False, index=sub.index)
        trim[name] = {
            "n": int(len(sub)),
            "solves": {str(int(k)): int(v)
                       for k, v in sub["trim_solves"].value_counts().sort_index().items()},
            "n_interpolated": int(interpolated.sum()),
            "max_abs_residual": float(sub["trim_residual"].abs().max()),
            "yaw_min": float(sub["trim_yaw"].min()), "yaw_max": float(sub["trim_yaw"].max()),
            "n_in_band": int(sub["trim_in_band"].sum()),
            "n_out_of_band": int((~sub["trim_in_band"]).sum())}
        errors = t[(t["level"] == level) & t["value"].isna()]["error"].fillna("").astype(str)
        failures[name] = {"n": int((t["level"] == level).sum()), "failed": int(len(errors)),
                          "xfoil": int(errors.str.contains("XFOIL").sum()),
                          "trim": int(errors.str.contains("trim").sum()),
                          "other": int((~errors.str.contains("XFOIL|trim")).sum())}
        if "xfoil_filled_points" in sub and sub["xfoil_filled_points"].notna().any():
            filled = sub["xfoil_filled_points"].dropna()
            xfoil[name] = {"evaluations": int(len(filled)),
                           "with_filled_points": int((filled > 0).sum()),
                           "mean_filled_points": float(filled.mean())}
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
    admissible_top = top[top["admissible"]]
    verifications = [{"run_index": int(r["run_index"]), "measured": float(r["value"]),
                      "in_band": bool(r["trim_in_band"]),
                      "predicted": r.get("surrogate_prediction"),
                      "error_percent": 100 * (float(r["value"]) - r["surrogate_prediction"])
                      / r["surrogate_prediction"]
                      if np.isfinite(r.get("surrogate_prediction", np.nan)) else None}
                     for _, r in top[top["verification"]].dropna(subset=["value"]).iterrows()]
    merit = {st.LEVEL_SHORT[lv]: int(len(t[(t["level"] == lv) & ~t["doe"] & ~t["verification"]]))
             for lv in st.LEVELS}
    return {"n_evaluations": int(len(t)), "trim": trim, "failures": failures, "xfoil": xfoil,
            "timing": timing, "verifications": verifications,
            "best_avl_run_index": int(admissible_top.loc[admissible_top["value"].idxmin(),
                                                         "run_index"])
            if len(admissible_top) else None,
            "last_avl_run_index": int(top["run_index"].max()),
            "chosen_by_merit_per_level": merit,
            "avl_chosen_by_merit": [int(i) for i in
                                    top[~top["doe"] & ~top["verification"]]["run_index"]]}


# 3/4 ---------------------------------------------------------------------------------------------
def figure_convergence(run, baseline_cd: float, best_cd: float) -> None:
    t = run["table"]
    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.14,
                        subplot_titles=("Best admissible AVL drag vs cumulative cost (yaw band "
                                        "enforced on the viscous levels)",
                                        "Every evaluation in its order (open: trimmed yaw outside "
                                        "the band, penalized; ringed: verification at AVL)"))
    fig.add_trace(go.Scatter(x=t["cumulative_cost"], y=t["best_top"], mode="lines",
                             line={"color": st.LEVEL_COLOR[TOP], "width": 2, "shape": "hv"},
                             name="best admissible AVL drag so far",
                             hovertemplate="cost %{x:.0f}<br>best CD %{y:.5f}<extra></extra>"),
                  row=1, col=1)
    for y, text in ((baseline_cd, f"reference, AVL CD {baseline_cd:.5f}"),
                    (best_cd, f"best measured (incl. step 2 checks) {best_cd:.5f}")):
        fig.add_hline(y=y, line={"color": pk.MUTED, "width": 1}, row=1, col=1)
        fig.add_annotation(text=text, xref="x domain", x=1.0, xanchor="right", y=y, yref="y",
                           yanchor="bottom" if y >= max(baseline_cd, best_cd) else "top",
                           showarrow=False, font={"color": pk.SECONDARY, "size": 11})
    for level in st.LEVELS:
        sub = t[t["level"] == level]
        ring = sub["verification"].to_numpy()
        band = sub["trim_in_band"].to_numpy()
        symbol = [st.LEVEL_SYMBOL[level] + ("" if b else "-open") for b in band]
        fig.add_trace(go.Scatter(
            x=sub.index + 1, y=sub.get("CD_unpenalized", sub["value"]), mode="markers",
            name=st.LEVEL_NAME[level],
            marker={"color": st.LEVEL_COLOR[level], "size": np.where(ring, 13, 8).tolist(),
                    "symbol": symbol,
                    "line": {"color": np.where(ring, pk.PRIMARY, st.LEVEL_COLOR[level]).tolist(),
                             "width": np.where(ring, 2, 1).tolist()}},
            customdata=np.column_stack((sub["cumulative_cost"], sub.get("trim_yaw", np.nan))),
            hovertemplate="evaluation %{x}<br>CD %{y:.5f}<br>cost %{customdata[0]:.0f}<br>"
                          "trimmed yaw %{customdata[1]:.3f} deg<extra></extra>"),
            row=2, col=1)
    doe_end = int(t["doe"].sum())
    fig.add_vline(x=doe_end + 0.5, line={"color": pk.MUTED, "width": 1}, row=2, col=1)
    fig.add_annotation(text="initial design | optimization loop", x=doe_end + 0.5, xref="x2",
                       y=0.98, yref="y2 domain", showarrow=False, yanchor="top",
                       bgcolor=pk.SURFACE, font={"color": pk.SECONDARY, "size": 11})
    fig.update_xaxes(title_text="cumulative cost (NPLLT x NeuralFoil evaluation = 1)", row=1,
                     col=1)
    fig.update_xaxes(title_text="evaluation number", row=2, col=1)
    fig.update_yaxes(title_text="CD at the reference lift", row=1, col=1)
    fig.update_yaxes(title_text="CD at its level (unpenalized)", row=2, col=1)
    counts = t.groupby("level").size().to_dict()
    pk.style(fig, "Convergence of the optimization at the reference lift",
             f"Evaluations per level {counts}, costs {run['costs']}; each level has its own drag "
             "offset (bottom): only AVL values are comparable to the reference lines.", 820)
    pk.save(fig, "step2_convergence")


def figure_trim(run, band: tuple) -> None:
    t = run["table"].dropna(subset=["value"])
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.1,
                        subplot_titles=("Trimmed yaw of every section vs its drag",
                                        "Solver calls per trimmed evaluation"))
    # exclude_empty_subplots=False: the shape is added before the traces of the subplot
    fig.add_vrect(x0=band[0], x1=band[1], fillcolor=pk.rgba(pk.MUTED, 0.28), line_width=0,
                  layer="below", row=1, col=1, exclude_empty_subplots=False)
    for level in st.LEVELS:
        sub = t[t["level"] == level]
        fig.add_trace(go.Scatter(
            x=sub["trim_yaw"], y=sub.get("CD_unpenalized", sub["value"]), mode="markers",
            name=st.LEVEL_NAME[level],
            marker={"color": st.LEVEL_COLOR[level], "size": 8, "symbol": st.LEVEL_SYMBOL[level],
                    "line": {"color": pk.SURFACE, "width": 1}},
            hovertemplate="yaw %{x:.3f} deg<br>CD %{y:.5f}<extra></extra>"), row=1, col=1)
        interpolated = sub["trim_interpolated"].fillna(False).astype(bool) \
            if "trim_interpolated" in sub else pd.Series(False, index=sub.index)
        kind = [f"{int(n)} call{'s' if n > 1 else ''}" + (" + interp." if i else "")
                for n, i in zip(sub["trim_solves"], interpolated)]
        counts = pd.Series(kind).value_counts().sort_index()
        fig.add_trace(go.Bar(x=counts.index.tolist(), y=counts.values,
                             name=st.LEVEL_NAME[level], showlegend=False,
                             marker={"color": st.LEVEL_COLOR[level]}, width=0.25,
                             hovertemplate="%{x}: %{y} evaluations<extra></extra>"),
                      row=1, col=2)
    fig.update_xaxes(title_text="trimmed yaw (deg); shaded: admissible band", row=1, col=1)
    fig.update_yaxes(title_text="CD at the reference lift (unpenalized)", row=1, col=1)
    fig.update_yaxes(title_text="evaluations", row=1, col=2)
    fig.update_layout(barmode="group")
    shares = {st.LEVEL_SHORT[lv]: f"{100 * t[t['level'] == lv]['trim_in_band'].mean():.0f} %"
              for lv in st.LEVELS}
    pk.style(fig, "The trim: how much yaw each section needs to carry the reference lift",
             f"Band {band[0]:.2f}-{band[1]:.2f} deg (+-5 % of 4 deg); share of the evaluations "
             f"inside: {shares}. Outside, CD + {st.YAW_PENALTY} per deg is given to the "
             "optimizer (drawn here unpenalized).", 520)
    pk.save(fig, "step2_trim")


def figure_sections(sections: list) -> None:
    """sections: list of (Section, label, colour, dash)."""
    fig = make_subplots(rows=2, cols=3, vertical_spacing=0.14, horizontal_spacing=0.07,
                        specs=[[{"colspan": 2}, None, {}], [{}, {}, {}]],
                        subplot_titles=("Section shapes", "Leading edge", "Thickness t / c",
                                        "Camber / c", "Trailing edge"))
    xs = geometry.THICKNESS_STATIONS
    lines = []
    for section, label, color, dash in sections:
        c = section.coordinates
        for row, col in ((1, 1), (1, 3), (2, 3)):
            fig.add_trace(go.Scatter(x=c[:, 0], y=c[:, 1], mode="lines", name=label,
                                     legendgroup=label, showlegend=(row, col) == (1, 1),
                                     line={"color": color, "width": 2, "dash": dash},
                                     hovertemplate=f"{label}<br>x/c %{{x:.4f}}<br>y/c %{{y:.4f}}"
                                                   "<extra></extra>"), row=row, col=col)
        up, lo = geometry.surfaces_at(c, xs)
        for col, y in ((1, up - lo), (2, 0.5 * (up + lo))):
            fig.add_trace(go.Scatter(x=xs, y=y, mode="lines", name=label, legendgroup=label,
                                     showlegend=False, line={"color": color, "width": 2,
                                                             "dash": dash},
                                     hovertemplate=f"{label}<br>x/c %{{x:.3f}}<br>%{{y:.4f}}"
                                                   "<extra></extra>"), row=2, col=col)
        thick, camber = up - lo, 0.5 * (up + lo)
        lines.append(f"{label}: t/c {section.thickness:.4f} at x/c {xs[np.argmax(thick)]:.2f}, "
                     f"t/c at 40 / 70 % {np.interp(0.4, xs, thick):.4f} / "
                     f"{np.interp(0.7, xs, thick):.4f}, camber {camber.max():.4f} at x/c "
                     f"{xs[np.argmax(camber)]:.2f}")
    fig.update_yaxes(scaleanchor="x", scaleratio=1, row=1, col=1)
    fig.update_xaxes(range=[-0.005, 0.06], row=1, col=3)
    fig.update_yaxes(range=[-0.05, 0.07], scaleanchor="x2", scaleratio=1, row=1, col=3)
    fig.update_xaxes(range=[0.85, 1.005], row=2, col=3)
    fig.update_yaxes(range=[-0.03, 0.05], row=2, col=3)
    for col in (1, 2, 3):
        fig.update_xaxes(title_text="x / c", row=2, col=col)
    pk.style(fig, f"Reference ({st.BASELINE_NAME}) and optimized sections", "<br>".join(lines),
             940, top=210)
    pk.save(fig, "step2_sections")


def figure_comparison(table: pd.DataFrame) -> None:
    """Drag, induced and profile drag, trimmed yaw of the compared sections at the three levels."""
    keys = (("CD_unpenalized", "CD at the reference lift"), ("Cdi", "induced drag Cdi"),
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
    pk.style(fig, "Reference, optimum and the other candidate sections at the reference lift",
             "Every section trimmed to the lift of the reference at each level. Values of the "
             "levels differ by their own offset; compare the sections within a level.", 640)
    fig.update_layout(margin={"b": 200, "l": 150})
    pk.save(fig, "step2_comparison")


def figure_levels(measured: dict, names: list, display: dict, band: tuple) -> dict:
    """
    The three models on the same sections: drag change relative to the reference at each level
    and trimmed yaw at each level. Returns the Spearman rank correlations between the levels.
    """
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.1,
                        subplot_titles=("Drag change vs the reference (%), each level",
                                        "Trimmed yaw (deg); shaded: admissible band"))
    fig.add_hrect(y0=band[0], y1=band[1], fillcolor=pk.rgba(pk.MUTED, 0.28), line_width=0,
                  layer="below", row=1, col=2, exclude_empty_subplots=False)
    labels = [display.get(n, n) for n in names]
    for level in st.LEVELS:
        reference = measured["baseline"][level]["CD_unpenalized"]
        change = [100 * (measured[n][level]["CD_unpenalized"] - reference) / reference
                  if measured[n][level]["CD_unpenalized"] is not None else None for n in names]
        yaw = [measured[n][level]["trim_yaw"] for n in names]
        for col, y in ((1, change), (2, yaw)):
            fig.add_trace(go.Scatter(
                x=labels, y=y, mode="markers", name=st.LEVEL_NAME[level], legendgroup=str(level),
                showlegend=col == 1,
                marker={"color": st.LEVEL_COLOR[level], "size": 11,
                        "symbol": st.LEVEL_SYMBOL[level],
                        "line": {"color": pk.SURFACE, "width": 2}},
                hovertemplate="%{x}<br>%{y:.3f}<extra>" + st.LEVEL_NAME[level] + "</extra>"),
                row=1, col=col)
    fig.add_hline(y=0.0, line={"color": pk.MUTED, "width": 1}, row=1, col=1)
    fig.update_xaxes(tickangle=-35)
    finite = [n for n in names if all(measured[n][lv]["CD_unpenalized"] is not None
                                      for lv in st.LEVELS)]
    spearman = {}
    for a, b in itertools.combinations(st.LEVELS, 2):
        if len(finite) >= 3:
            spearman[f"{st.LEVEL_SHORT[a]} vs {st.LEVEL_SHORT[b]}"] = float(stats.spearmanr(
                [measured[n][a]["CD_unpenalized"] for n in finite],
                [measured[n][b]["CD_unpenalized"] for n in finite])[0])
    pk.style(fig, "NPLLT x NeuralFoil, NPLLT x XFOIL and AVL x XFOIL on the same sections",
             f"{len(finite)} sections at the three levels; Spearman rank correlation of the "
             f"drags: " + ", ".join(f"{k} {v:.2f}" for k, v in spearman.items()), 640)
    fig.update_layout(margin={"b": 200, "l": 150})
    pk.save(fig, "step2_levels")
    return spearman


def constraint_margins(problem, points: dict, model) -> dict:
    """
    Ratio of every constrained property to its limit (> 1: satisfied, = 1: on the limit) for
    named normalized points (camber offset solved by the yaw model); J (not enforced) relative
    to the reference for information.
    """
    reference = st.reference_properties(problem)
    min_thickness = float(problem.constraints.get("min_thickness", 0.0))
    out = {}
    for name, x in points.items():
        delta = st.solve_delta(problem, x, model)
        p = {k: float(v[0]) for k, v in st.section_properties(problem, x, delta).items()}
        out[name] = {
            "t/c / min": p["max_thickness"] / min_thickness,
            "Ixx / Ixx_ref": p["Ixx"] / reference["Ixx"],
            "J / J_ref (not enforced)": p["J"] / reference["J"],
            **{f"t(x/c {s:.2f}) / ref": p[f"thickness_{s:.2f}"] / reference[f"thickness_{s:.2f}"]
               for s in st.TE_STATIONS},
            "upper_0 / 0.05": p["upper_0"] / st.LE_WEIGHT_MIN,
            "-lower_0 / 0.05": -p["lower_0"] / st.LE_WEIGHT_MIN,
            "wiggliness limit / wiggliness": st.WIGGLINESS_RATIO * reference["wiggliness"]
            / p["wiggliness"],
            "NeuralFoil confidence / 0.9": float(st.neuralfoil_confidence(problem, x, delta)[0])
            / st.CONFIDENCE_MIN}
    return out


def figure_constraints(margins: dict) -> None:
    fig = go.Figure()
    for k, (name, values) in enumerate(margins.items()):
        color = pk.SECONDARY if name.startswith("reference") else (
            pk.PRIMARY if name.startswith("optimum") else pk.CATEGORICAL[3 + k % 2][0])
        fig.add_trace(go.Bar(y=list(values), x=list(values.values()), orientation="h", name=name,
                             marker={"color": color},
                             hovertemplate="%{y}: %{x:.3f}<extra>" + name + "</extra>"))
    fig.add_vline(x=1.0, line={"color": pk.MUTED, "width": 1})
    fig.update_xaxes(title_text="property / limit (> 1: satisfied, 1: on the limit)")
    fig.update_layout(barmode="group", yaxis={"autorange": "reversed"})
    pk.style(fig, "Known constraints of the section: margin to each limit",
             "Limits relative to the reference except the leading-edge weights (0.05) and the "
             "NeuralFoil confidence (0.9); J = Ixx + Iyy is shown but not enforced.", 640)
    fig.update_layout(margin={"l": 230})
    pk.save(fig, "step2_constraints")


def figure_xfoil(run_stats: dict) -> None:
    """Failed evaluations per level and cause; XFOIL polars with points filled by interpolation."""
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.12,
                        subplot_titles=("Failed evaluations per level and cause",
                                        "Evaluations with filled XFOIL points"))
    causes = ("xfoil", "trim", "other")
    for k, cause in enumerate(causes):
        fig.add_trace(go.Bar(x=[st.LEVEL_SHORT[lv] for lv in st.LEVELS],
                             y=[run_stats["failures"][st.LEVEL_SHORT[lv]][cause]
                                for lv in st.LEVELS],
                             name=cause, marker={"color": pk.CATEGORICAL[k + 2][0]},
                             hovertemplate="%{x}: %{y}<extra>" + cause + "</extra>"), 1, 1)
    xfoil = run_stats["xfoil"]
    fig.add_trace(go.Bar(x=list(xfoil), y=[v["with_filled_points"] for v in xfoil.values()],
                         name="with filled points", showlegend=False,
                         marker={"color": [st.LEVEL_COLOR[lv] for lv in st.LEVELS
                                           if st.LEVEL_SHORT[lv] in xfoil]},
                         customdata=[[v["evaluations"], v["mean_filled_points"]]
                                     for v in xfoil.values()],
                         hovertemplate="%{x}: %{y} of %{customdata[0]} evaluations, "
                                       "%{customdata[1]:.2f} filled points per polar"
                                       "<extra></extra>"), 1, 2)
    fig.update_layout(barmode="group")
    totals = {st.LEVEL_SHORT[lv]: f"{run_stats['failures'][st.LEVEL_SHORT[lv]]['failed']}/"
              f"{run_stats['failures'][st.LEVEL_SHORT[lv]]['n']}" for lv in st.LEVELS}
    pk.style(fig, "Failures and XFOIL convergence during the run",
             f"Failed / evaluated per level: {totals}. A filled point is an angle XFOIL did not "
             "converge, interpolated by the core.", 500)
    pk.save(fig, "step2_xfoil")


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
                x=s["s"], y=s[key], mode="lines",
                name=f"{'reference' if label == 'baseline' else label}, {st.LEVEL_NAME[level]}",
                legendgroup=f"{label}{level}", showlegend=col == 1,
                line={"color": st.LEVEL_COLOR[level], "width": 2, "dash": dash},
                hovertemplate=f"s %{{x:.3f}} m<br>{key} %{{y:.4f}}<extra></extra>"),
                row=1, col=col)
        fig.update_xaxes(title_text="arc length from the root s (m)")
    fig.update_yaxes(range=[0.0, 1.2], row=1, col=1)
    pk.style(fig, "Spanwise loading at the reference lift: reference (dotted) vs optimum (solid)",
             "Each section at its trimmed yaw. The first point at the wall is the root "
             "singularity of the wall image.", 540)
    pk.save(fig, "step2_spanwise")


def figure_loo(loo: dict) -> None:
    y, m, s = np.array(loo["y"]), np.array(loo["mean"]), np.sqrt(np.array(loo["var"]))
    ring = np.array(loo["verification"], dtype=bool)
    band = np.array(loo["in_band"], dtype=bool)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y, y=m, mode="markers", name="AVL point predicted without itself",
        error_y={"type": "data", "array": 1.96 * s, "thickness": 1, "width": 0,
                 "color": pk.rgba(st.LEVEL_COLOR[TOP], 0.5)},
        marker={"color": st.LEVEL_COLOR[TOP], "size": np.where(ring, 13, 8).tolist(),
                "symbol": ["diamond" if b else "diamond-open" for b in band],
                "line": {"color": np.where(ring, pk.PRIMARY, st.LEVEL_COLOR[TOP]).tolist(),
                         "width": 2}},
        hovertemplate="AVL value %{x:.5f}<br>predicted %{y:.5f}<extra></extra>"))
    lo, hi = float(min(y.min(), m.min())), float(max(y.max(), m.max()))
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="perfect prediction",
                             line={"color": pk.MUTED, "width": 1}, hoverinfo="skip"))
    fig.update_xaxes(title_text="AVL value given to the optimizer (penalized outside the band)")
    fig.update_yaxes(title_text="leave-one-out prediction +- 1.96 sigma")
    lower = ", ".join(f"level {k[len('spearman_level')]} {v:.2f}" for k, v in loo.items()
                      if k.startswith("spearman_level"))
    pk.style(fig, "Accuracy of the final surrogate on the AVL points (leave-one-out)",
             f"{loo['n']} AVL points (open: outside the yaw band; ringed: verifications): RMSE "
             f"{100 * loo['rmse_relative']:.0f} % of their spread, 95 % coverage "
             f"{100 * loo['coverage95']:.0f} %; Spearman of the lower-level surrogates vs AVL: "
             f"{lower}.", 540)
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
             "Other variables at the optimum (vertical line); gaps: known constraints "
             "violated. Penalized CD at the reference lift, each level with its own surrogate.",
             300 * nrows + 220)
    pk.save(fig, "step2_gp_slices_1d")
    return export


def figure_slices_2d(model, problem, x_opt, feasibility, n: int = 41) -> dict:
    names = [v.name for v in problem.variables]
    pairs = list(itertools.combinations(range(len(names)), 2))
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.12,
                        subplot_titles=("AVL surrogate: mean CD at the reference lift",
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
             "Choose the pair in the menu; other variables at the optimum (x); blank: known "
             "constraints violated.", 620, top=190)
    pk.save(fig, "step2_gp_slices_2d")
    return export


def figure_parallel(run, problem) -> None:
    t = run["table"].dropna(subset=["value"])
    dims = [{"label": "level", "values": t["level"].tolist(), "tickvals": list(st.LEVELS),
             "ticktext": [st.LEVEL_SHORT[lv] for lv in st.LEVELS]}]
    for name in [v.name for v in problem.variables] + ["trim_yaw"]:
        if name in t:
            dims.append({"label": name, "values": t[name].tolist()})
    dims.append({"label": "in band", "values": t["trim_in_band"].astype(int).tolist(),
                 "tickvals": [0, 1], "ticktext": ["no", "yes"]})
    cd = t.get("CD_unpenalized", t["value"])
    dims.append({"label": "CD", "values": cd.tolist()})
    fig = go.Figure(go.Parcoords(
        line={"color": cd.tolist(),
              "colorscale": [[1 - p, c] for p, c in pk.SEQUENTIAL_SCALE][::-1],
              "showscale": True, "colorbar": {"title": {"text": "CD"},
                                              "tickfont": {"color": pk.MUTED}}},
        dimensions=dims, labelfont={"color": pk.PRIMARY}, tickfont={"color": pk.MUTED}))
    pk.style(fig, "Every evaluated section (parallel coordinates)",
             "Drag an axis range to filter (e.g. level = AVL, in band = yes and the lowest CD). "
             "Dark = low drag (unpenalized).", 560, top=140)
    pk.save(fig, "step2_designs_parallel")


INDEX = [("step2_convergence", "Convergence and level of every evaluation"),
         ("step2_comparison", "Reference, optimum and other candidate sections"),
         ("step2_levels", "The three levels on the same sections"),
         ("step2_sections", "Section shapes, leading and trailing edges, thickness and camber"),
         ("step2_constraints", "Margin of the optimum to each known constraint"),
         ("step2_trim", "The trim: yaw needed by every section, yaw band"),
         ("step2_xfoil", "Failures and XFOIL convergence during the run"),
         ("step2_spanwise", "Spanwise loading, reference vs optimum"),
         ("step2_leave_one_out", "Accuracy of the surrogate (leave-one-out on AVL)"),
         ("step2_gp_slices_1d", "Response surface along each variable"),
         ("step2_gp_slices_2d", "Response surface over pairs of variables"),
         ("step2_designs_parallel", "Every evaluated section (parallel coordinates)"),
         ("step0_constraints", "Set-up: known constraints and feasible sections"),
         ("step0_xfoil", "Set-up: XFOIL robustness of the settings"),
         ("step0_band", "Set-up: yaw band at level 1"),
         ("step0_trim", "Set-up: yaw needed by a few sections at the three levels")]


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
<h1>Intens SY C-foil: section optimization at the reference lift</h1>
<p class="sub">3 levels (NPLLT x NeuralFoil xxxlarge, NPLLT x XFOIL, AVL x XFOIL), yaw trimmed to
the lift of the reference foil ({st.BASELINE_NAME}) within +-5 % of 4 deg, no 2D lift constraint,
section constraints (t/c, Ixx, trailing-edge thickness, leading-edge weights, wiggliness,
NeuralFoil confidence).</p>
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
    # the yaw model the run used (copied by step 1), else the current one of step 0
    run_model = run["run_dir"] / st.YAW_MODEL_FILE.name
    if not run_model.is_file():
        logger.warning("%s not found: the current yaw model of step 0 is used (it must be the "
                       "one of the run, otherwise delta differs for the same design point)",
                       run_model)
    yaw_model = st.load_yaw_model(run_model if run_model.is_file() else st.YAW_MODEL_FILE)
    feasibility = st.feasibility_function(problem, yaw_model)
    simulator = st.make_simulator(problem, yaw_model)
    band = st.yaw_band(problem)
    x_base = np.asarray(st.baseline_parameters(problem)["x"])

    # 1. reference and candidates, trimmed, three levels
    measured = {"baseline": evaluate_levels(simulator, x_base, with_strips=True)}
    candidates = collect_candidates(run, problem, feasibility)
    model = run["surrogate"]
    for name, x in candidates.items():
        measured[name] = evaluate_levels(simulator, x)
        m, v, _ = model.predict_batch(x[None, :], level=TOP)
        measured[name]["prediction"] = {"mean": float(m[0]), "sd": float(np.sqrt(max(v[0], 0)))}
        logger.info("%s: AVL %s (in band %s, predicted %.5f +- %.5f), NPLLT-XF %s, NPLLT-NF %s",
                    name, measured[name][TOP]["value"], measured[name][TOP]["trim_in_band"],
                    m[0], np.sqrt(max(v[0], 0)), measured[name][2]["value"],
                    measured[name][1]["value"])
    t = run["table"]
    run_best = t[(t["level"] == TOP) & t["admissible"]]
    if len(run_best):
        best_name = "best AVL section of the run"
        x_best = np.asarray(run_best.loc[run_best["value"].idxmin(), "x"])
        best_cd = float(run_best["value"].min())
    else:
        best_name, x_best, best_cd = "reference", x_base, np.inf
    x_run = x_best.copy()
    run_name = next((n for n, x in candidates.items() if np.linalg.norm(x - x_run) < 1e-9), None)
    # the best AVL section of the run is re-measured at the three levels with the candidates:
    # it is kept only if it is also inside the band of the viscous levels
    if run_name is None or not admissible(measured[run_name]):
        best_name, x_best, best_cd = "reference", x_base, np.inf
    for name, x in candidates.items():
        if admissible(measured[name]) and measured[name][TOP]["value"] < best_cd:
            best_name, x_best, best_cd = name, x, float(measured[name][TOP]["value"])
    measured["optimum"] = evaluate_levels(simulator, x_best, with_strips=True)
    baseline_cd = float(measured["baseline"][TOP]["CD_unpenalized"])
    best_cd = float(measured["optimum"][TOP]["CD_unpenalized"])
    logger.info("Optimum (%s): AVL CD %.5f vs reference %.5f (%.2f %%), yaw %s", best_name,
                best_cd, baseline_cd, 100 * (best_cd - baseline_cd) / baseline_cd,
                {lv: measured["optimum"][lv]["trim_yaw"] for lv in st.LEVELS})

    # 2. surrogate accuracy
    loo = leave_one_out(run)
    logger.info("Leave-one-out on %d AVL points: RMSE/spread %.3f, NLPD %.3f, coverage %.2f; %s",
                loo["n"], loo["rmse_relative"], loo["nlpd"], loo["coverage95"],
                {k: round(v, 2) for k, v in loo.items() if k.startswith("spearman")})
    x_top = np.vstack(t[t["level"] == TOP]["x"])
    out_of_sample = {}
    for name in candidates:
        if measured[name][TOP]["value"] is None or not np.isfinite(measured[name][TOP]["value"]):
            continue
        x, p = candidates[name], measured[name]["prediction"]
        seen = bool(np.min(np.linalg.norm(x_top - x, axis=1)) < 1e-9)
        y = float(measured[name][TOP]["value"])
        out_of_sample[name] = {"in_run_data": seen, "measured": y, "predicted": p["mean"],
                               "sd": p["sd"], "z": (y - p["mean"]) / p["sd"] if p["sd"] > 0
                               else None,
                               "error_percent": 100 * (y - p["mean"]) / p["mean"],
                               "in_band": bool(measured[name][TOP]["trim_in_band"])}
    run_stats = run_statistics(run)
    logger.info("Run statistics: %s", run_stats)

    # 3. figures and exports
    # the optimum is drawn once (under "optimum"), whatever candidate name it came from
    others = [n for n in candidates if np.linalg.norm(candidates[n] - x_best) > 1e-9]
    display = {"baseline": "reference", "optimum": "optimum",
               **{n: short_label(n) for n in others}}
    names = ["baseline", "optimum"] + others
    rows = []
    for name in names:
        for level in st.LEVELS:
            rows.append({"label": display.get(name, name), "level": level,
                         **{k: measured[name][level].get(k) for k in ("value",) + KEYS}})
    comparison = pd.DataFrame(rows)
    figure_convergence(run, baseline_cd, best_cd)
    figure_trim(run, band)
    sections = [(simulator.section_of(x_base), f"reference ({st.BASELINE_NAME})", pk.SECONDARY,
                 "dot"),
                (simulator.section_of(x_best), "optimum (best measured section)", pk.PRIMARY,
                 "solid")]
    if np.linalg.norm(x_best - x_run) > 1e-9 and len(run_best):
        sections.append((simulator.section_of(x_run), "best AVL section of this run",
                         st.LEVEL_COLOR[TOP], "dashdot"))
    figure_sections(sections)
    figure_comparison(comparison)
    spearman_levels = figure_levels(measured, names, display, band)
    margins = constraint_margins(problem, {"reference": x_base, "optimum": x_best,
                                           **({"best AVL section of this run": x_run}
                                              if np.linalg.norm(x_best - x_run) > 1e-9
                                              and len(run_best) else {})}, yaw_model)
    figure_constraints(margins)
    figure_xfoil(run_stats)
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

    gain = {lv: 100 * (measured["optimum"][lv]["CD_unpenalized"]
                       - measured["baseline"][lv]["CD_unpenalized"])
            / measured["baseline"][lv]["CD_unpenalized"] for lv in st.LEVELS
            if measured["optimum"][lv]["CD_unpenalized"] is not None}
    analysis = {"run_dir": str(run_dir), "baseline": st.baseline_parameters(problem),
                "yaw_band": band, "baseline_cd_avl": baseline_cd, "best_cd_avl": best_cd,
                "gain_percent": gain.get(TOP), "gain_percent_per_level": gain,
                "best_source": best_name, "best_design_normalized": x_best.tolist(),
                "best_design_physical": physical(problem, x_best),
                "constraint_margins": margins,
                "best_section_properties": {k: float(v[0]) for k, v in st.section_properties(
                    problem, x_best, st.solve_delta(problem, x_best, yaw_model)).items()},
                "best_delta": float(st.solve_delta(problem, x_best, yaw_model)[0]),
                "reference_properties": st.reference_properties(problem),
                "measured": {n: {str(lv): {k: v for k, v in m[lv].items() if k != "strips"}
                                 for lv in st.LEVELS} | ({"prediction": m["prediction"]}
                                                         if "prediction" in m else {})
                             for n, m in measured.items()},
                "leave_one_out": {k: v for k, v in loo.items()
                                  if k not in ("y", "mean", "var", "verification", "in_band")},
                "spearman_candidates_between_levels": spearman_levels,
                "predictions_of_candidates": out_of_sample,
                "run_statistics": run_stats,
                "lift_residual_optimum": {str(lv): (measured["optimum"][lv]["CL"]
                                                    - simulator.reference_lift[lv])
                                          / simulator.reference_lift[lv] for lv in st.LEVELS
                                          if measured["optimum"][lv]["CL"] is not None},
                "reference_lift": {str(k): v for k, v in simulator.reference_lift.items()},
                "evaluations": t.groupby("level").size().to_dict(),
                "verifications": int(t["verification"].sum()),
                "gp": run["results"].get("gp"), "costs": run["costs"]}
    st.save_json(st.RESULTS / "analysis.json", analysis)
    numbers = [("Reference CD (AVL)", f"{baseline_cd:.5f}"), ("Optimum CD (AVL)", f"{best_cd:.5f}"),
               ("Drag change AVL / NPLLT-XF / NPLLT-NF",
                " / ".join(f"{gain[lv]:+.2f} %" for lv in reversed(st.LEVELS) if lv in gain)),
               ("Optimum yaw AVL", f"{measured['optimum'][TOP]['trim_yaw']:.3f} deg"),
               ("Evaluations L1 / L2 / L3", " / ".join(str(int((t['level'] == lv).sum()))
                                                       for lv in st.LEVELS)),
               ("LOO RMSE / spread (AVL)", f"{100 * loo['rmse_relative']:.0f} %")]
    path = write_index(numbers)
    logger.info("Analysis written: %s, %s", st.RESULTS / "analysis.json", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
