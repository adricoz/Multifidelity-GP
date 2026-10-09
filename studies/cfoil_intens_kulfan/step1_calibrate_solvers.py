"""
STEP 1 - Calibrate the three solvers before trusting them in a multi-fidelity optimization.

A multi-fidelity GP pays off when the cheap levels are (a) much cheaper and (b) well
correlated with the expensive one: the recursive model y_l(x) = rho * y_(l-1)(x) + delta_l(x)
only needs a SMOOTH discrepancy delta_l, not equal values. This script measures it on a common
design: the SAME n feasible sections (plus the baseline) evaluated at the three levels.

The compared quantity is the objective of the optimization, the 3D drag AT EQUAL LIFT
CD* = Cdprofile + Cdi (CL_ref / CL)^2 (column "value"); the raw drag at the fixed attitude
(column "CD") is analysed too: the first calibration showed that, at a fixed attitude, the AVL
lift changes from one section to another (AVL only sees the camber line), which hides the
section effect in the raw drag (see RAPPORT.md).

Measured (results/step1_calibration.csv and results/step1_summary.json):
1. cost of each level (wall time per evaluation, and per stage: CL2d projection, NPLLT set-up
   and solve, XFOIL polar, AVL);
2. agreement between levels: Pearson correlation (linear), Spearman and Kendall (ranking: does
   the cheap level sort the sections like the expensive one?), overlap of the best 10 %,
   linear fit y_high = a y_low + b (a plays the role of rho);
3. drag breakdown: induced (Cdi) and profile (Cdprofile) drag of every level;
4. the 2D constraint seen by each level: CL2d(0 deg) of the section at every station
   (NeuralFoil xxsmall / xxlarge inside NPLLT, XFOIL for AVL) against the 0.45 target;
5. numerical noise: small perturbations (1e-3 of the box) of a few designs; the GP noise bound
   (1e-2 of the output variance) must not be exceeded;
6. mesh convergence: NPLLT nodes_count and vortex cut-off (core_radius), AVL spanwise
   vortices, on the baseline;
7. spanwise loading of the baseline, NPLLT against AVL.
A recommendation on the 3 levels closes the summary (keep a level if it is at least 3 times
cheaper than the next one and its Spearman correlation with L3 is at least 0.8).

Figures: figures/step1_*.html.

Run (about 15 min for n = 48):
    C:\\Users\\SIM\\.conda\\envs\\bdToolbox\\python.exe step1_calibrate_solvers.py --n 48
    ... --n 8 for a quick smoke run
"""
import argparse
import copy
import logging
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

import plotkit as pk
import study_lib as lib
from pipelines.bdtoolbox_foil.constraints import feasibility_function  # pylint: disable=C0411
from pipelines.bdtoolbox_foil.geometry import to_physical  # pylint: disable=wrong-import-order
from pipelines.bdtoolbox_foil.solvers import make_backend

logger = logging.getLogger("step1")
LEVELS = (1, 2, 3)
PAIRS = ((1, 3), (2, 3), (1, 2))
KEEP = ("CD", "CL", "CL_ref", "Cdi", "Cdi_nearfield", "Cdprofile", "Cdprofile_core", "Cy", "Cz",
        "cl2d_check_min",
        "cl2d_check_max", "delta", "thickness", "n_clamped", "n_clamped_total", "time_s", "time_geometry_s",
        "time_setup_s", "time_solve_s", "time_polar_s", "time_avl_s", "nf_confidence_min",
        "xfoil_filled_points", "error")


# 1/4 ---------------------------------------------------------------------------------------------
def evaluate_design(simulator, problem, points: np.ndarray, kind: str) -> pd.DataFrame:
    """Every point at every level: one row per (point, level) with the kept metrics."""
    rows = []
    for i, x in enumerate(points):
        for level in LEVELS:
            value, metrics = simulator.evaluate(x, level)
            row = {"design": i, "kind": kind, "level": level, "value": value,
                   **{f"x_{j}": float(u) for j, u in enumerate(x)},
                   **to_physical(x, problem.variables),
                   **{k: metrics.get(k) for k in KEEP}}
            rows.append(row)
        logger.info("%s design %d/%d: CD %s", kind, i + 1, len(points),
                    [None if not np.isfinite(r["value"]) else round(r["value"], 6)
                     for r in rows[-3:]])
    return pd.DataFrame(rows)


def agreement(table: pd.DataFrame, column: str = "value") -> dict:
    """Correlations, ranking agreement and linear fit between every pair of levels."""
    wide = table.pivot(index="design", columns="level", values=column).dropna()
    out = {"n_common": int(len(wide))}
    for lo, hi in PAIRS:
        a, b = wide[lo].to_numpy(), wide[hi].to_numpy()
        slope, intercept = np.polyfit(a, b, 1)
        residual = b - (slope * a + intercept)
        n_top = max(1, int(round(0.1 * len(a))))
        top_a, top_b = set(np.argsort(a)[:n_top]), set(np.argsort(b)[:n_top])
        out[f"L{lo}-L{hi}"] = {
            "pearson": float(stats.pearsonr(a, b)[0]),
            "spearman": float(stats.spearmanr(a, b)[0]),
            "kendall": float(stats.kendalltau(a, b)[0]),
            "top10_overlap": len(top_a & top_b) / n_top,
            "slope": float(slope), "intercept": float(intercept),
            "residual_std": float(np.std(residual)),
            "mean_offset": float(np.mean(b - a)), "std_low": float(np.std(a)),
            "std_high": float(np.std(b))}
    return out


def costs(table: pd.DataFrame) -> dict:
    """Mean wall time per level and per stage (s), and relative costs (L1 = 1)."""
    out = {}
    projection = table.loc[table["level"] == 1, "time_geometry_s"].mean()
    for level in LEVELS:
        sub = table[table["level"] == level]
        solver = (sub["time_s"] - sub["time_geometry_s"]).mean()
        out[level] = {"solver_s": float(solver), "geometry_and_projection_s": float(projection),
                      "per_evaluation_s": float(solver + projection),
                      **{k: float(sub[k].mean()) for k in ("time_setup_s", "time_solve_s",
                                                           "time_polar_s", "time_avl_s")
                         if sub[k].notna().any()}}
    reference = out[1]["per_evaluation_s"]
    for level in LEVELS:
        out[level]["relative"] = out[level]["per_evaluation_s"] / reference
    return out


def recommendation(stats_: dict, cost: dict) -> list:
    """Is each cheap level worth keeping? (cost ratio >= 3 and Spearman with L3 >= 0.8)."""
    lines = []
    for level, nxt in ((1, 2), (2, 3)):
        ratio = cost[nxt]["per_evaluation_s"] / cost[level]["per_evaluation_s"]
        spearman = stats_[f"L{level}-L3"]["spearman"]
        keep = ratio >= 3.0 and spearman >= 0.8
        lines.append({"level": level, "cost_ratio_to_next": ratio, "spearman_with_L3": spearman,
                      "keep": keep,
                      "text": f"L{level}: {ratio:.1f}x cheaper than L{nxt}, Spearman with L3 "
                              f"{spearman:.3f} -> {'keep' if keep else 'questionable'}"})
    return lines


# 2/4 ---------------------------------------------------------------------------------------------
def noise_floor(simulator, problem, points: np.ndarray, n_perturbations: int, scale: float,
                seed: int = 1) -> dict:
    """
    Numerical noise of each level: standard deviation of the output under tiny perturbations
    of the design (scale in normalized units), relative to the standard deviation of the
    output over the whole design (the GP noise bound is a VARIANCE ratio of 1e-2).
    """
    rng = np.random.default_rng(seed)
    out = {}
    feasibility = feasibility_function(problem)
    for level in LEVELS:
        spreads = []
        for x in points:
            values = []
            for _ in range(n_perturbations):
                for _ in range(100):
                    xp = np.clip(x + scale * rng.standard_normal(len(x)), 0.0, 1.0)
                    if feasibility is None or feasibility(xp[None, :])[0]:
                        break
                values.append(simulator.evaluate(xp, level)[0])
            values = np.asarray(values, dtype=float)
            if np.sum(np.isfinite(values)) >= 2:
                spreads.append(float(np.nanstd(values, ddof=1)))
        out[level] = {"perturbation_std": float(np.mean(spreads)) if spreads else np.nan,
                      "n_designs": len(spreads)}
    return out


def mesh_convergence(problem, simulator, x0: np.ndarray) -> dict:
    """NPLLT nodes_count (L1, L2 models) and AVL spanwise vortices on the baseline."""
    out = {"npllt": [], "avl": []}
    saved = list(simulator.backends)
    for nodes in (25, 50, 100, 200, 400):
        for level in (1, 2):
            lv = copy.deepcopy(problem.levels[level - 1])
            lv.options["nodes_count"] = nodes
            simulator.backends[level - 1] = make_backend(lv, problem, simulator.paths)
            start = time.perf_counter()
            value, metrics = simulator.evaluate(x0, level)
            out["npllt"].append({"level": level, "nodes": nodes, "CD": metrics.get("CD"),
                                 "objective": value,
                                 "Cdi": metrics.get("Cdi"), "Cdprofile": metrics.get("Cdprofile"),
                                 "time_s": time.perf_counter() - start})
    for radius in (0.0, 0.002, 0.005, 0.01, 0.02):
        lv = copy.deepcopy(problem.levels[1])
        lv.options["core_radius"] = radius
        simulator.backends[1] = make_backend(lv, problem, simulator.paths)
        value, metrics = simulator.evaluate(x0, 2)
        out.setdefault("core_radius", []).append(
            {"core_radius": radius, "CD": metrics.get("CD"), "objective": value,
             "Cdi": metrics.get("Cdi"), "Cdprofile": metrics.get("Cdprofile"),
             "error": metrics.get("error")})
    simulator.backends[1] = saved[1]
    for nspan in (100, 205, 300):
        lv = copy.deepcopy(problem.levels[2])
        lv.options["settings"] = {**lv.options.get("settings", {}), "nspan": nspan}
        simulator.backends[2] = make_backend(lv, problem, simulator.paths)
        start = time.perf_counter()
        value, metrics = simulator.evaluate(x0, 3)
        out["avl"].append({"nspan": nspan, "CD": metrics.get("CD"), "objective": value,
                           "Cdi": metrics.get("Cdi"),
                           "Cdprofile": metrics.get("Cdprofile"),
                           "time_s": time.perf_counter() - start})
    simulator.backends[:] = saved
    for row in out["npllt"] + out["core_radius"] + out["avl"]:
        logger.info("mesh %s", {k: (round(v, 6) if isinstance(v, float) else v)
                                for k, v in row.items()})
    return out


# 3/4 ---------------------------------------------------------------------------------------------
def figure_scatter(table: pd.DataFrame, stats_: dict, problem, column: str = "value",
                   name: str = "step1_level_agreement", quantity: str = "CD* (equal lift)",
                   title: str = "Do the levels agree? Drag at equal lift of the same sections"
                   ) -> None:
    wide = table.pivot(index="design", columns="level", values=column).dropna()
    kinds = table.groupby("design")["kind"].first()
    fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.08,
                        subplot_titles=[f"L{lo} (x) vs L{hi} (y)" for lo, hi in PAIRS])
    for col, (lo, hi) in enumerate(PAIRS, start=1):
        s = stats_[f"L{lo}-L{hi}"]
        a, b = wide[lo], wide[hi]
        is_base = (kinds.loc[wide.index] == "baseline").to_numpy()
        fig.add_trace(go.Scatter(
            x=a[~is_base], y=b[~is_base], mode="markers",
            name="sections (colour of the level on the x axis)",
            marker={"color": pk.LEVEL_COLOR[lo], "size": 8,
                    "line": {"color": pk.SURFACE, "width": 2}},
            customdata=wide.index[~is_base],
            hovertemplate=f"design %{{customdata}}<br>L{lo} %{{x:.5f}}<br>"
                          f"L{hi} %{{y:.5f}}<extra></extra>", showlegend=col == 1),
            row=1, col=col)
        fig.add_trace(go.Scatter(x=a[is_base], y=b[is_base], mode="markers", name="baseline",
                                 marker={"color": pk.PRIMARY, "size": 11, "symbol": "star"},
                                 showlegend=col == 1,
                                 hovertemplate="baseline<br>%{x:.5f} / %{y:.5f}<extra></extra>"),
                      row=1, col=col)
        grid = np.linspace(a.min(), a.max(), 10)
        fig.add_trace(go.Scatter(x=grid, y=s["slope"] * grid + s["intercept"], mode="lines",
                                 line={"color": pk.SECONDARY, "width": 1}, showlegend=col == 1,
                                 name="linear fit y = a x + b", hoverinfo="skip"), row=1, col=col)
        fig.add_annotation(text=f"Pearson {s['pearson']:.3f}<br>Spearman {s['spearman']:.3f}"
                                f"<br>Kendall {s['kendall']:.3f}<br>top 10 % overlap "
                                f"{100 * s['top10_overlap']:.0f} %<br>a = {s['slope']:.3f}",
                           xref=f"x{col if col > 1 else ''} domain",
                           yref=f"y{col if col > 1 else ''} domain", x=0.02, y=0.98,
                           xanchor="left", yanchor="top", showarrow=False, align="left",
                           font={"color": pk.SECONDARY, "size": 11})
        fig.update_xaxes(title_text=f"{quantity} {lib.level_label(problem, lo)}", row=1,
                         col=col)
        fig.update_yaxes(title_text=f"{quantity} {lib.level_label(problem, hi)}", row=1,
                         col=col)
    pk.style(fig, title,
             f"{stats_['n_common']} sections evaluated at the 3 levels. A multi-fidelity GP needs "
             "a high correlation (ranking), not equal values: the offset and the slope a are "
             "learnt (rho, discrepancy GP).", 560)
    pk.save(fig, name)


def figure_lift(table: pd.DataFrame, problem) -> None:
    """Total 3D lift |(Cy, Cz)| of every section at every level (fixed attitude)."""
    fig = go.Figure()
    for level in LEVELS:
        sub = table[(table["level"] == level) & table["CL"].notna()]
        jitter = np.random.default_rng(level).uniform(-0.15, 0.15, len(sub))
        fig.add_trace(go.Scatter(
            x=level + jitter, y=sub["CL"], mode="markers", name=lib.level_label(problem, level),
            marker={"color": pk.LEVEL_COLOR[level], "size": 8, "symbol": pk.LEVEL_SYMBOL[level],
                    "line": {"color": pk.SURFACE, "width": 2}}, customdata=sub["design"],
            hovertemplate="design %{customdata}<br>CL %{y:.4f}<extra></extra>"))
        spread = 100 * sub["CL"].std() / sub["CL"].mean()
        fig.add_annotation(x=level, y=sub["CL"].max(), yanchor="bottom", showarrow=False,
                           text=f"std {spread:.1f} %", font={"color": pk.SECONDARY, "size": 11})
    fig.update_xaxes(tickvals=list(LEVELS), ticktext=[lib.level_label(problem, lv)
                                                      for lv in LEVELS], range=[0.4, 3.6])
    fig.update_yaxes(title_text="total 3D lift coefficient CL = |(Cy, Cz)|")
    pk.style(fig, "3D lift of the same sections at a fixed attitude",
             "With CL2d(0) fixed by NeuralFoil, the NPLLT lift (viscous polars) barely changes; "
             "the AVL lift comes from the camber line only and changes from one section to "
             "another: the raw drag then compares sections at different lifts.", 520)
    pk.save(fig, "step1_lift_per_level")


def figure_breakdown(table: pd.DataFrame, problem) -> None:
    """Induced and profile drag of every section at every level (sorted by the L3 drag)."""
    order = table[table["level"] == 3].dropna(subset=["value"]).sort_values("value")["design"]
    rank = {d: i for i, d in enumerate(order)}
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.1,
                        subplot_titles=("Induced drag Cdi", "Profile drag Cdprofile"))
    for col, key in enumerate(("Cdi", "Cdprofile"), start=1):
        for level in LEVELS:
            sub = table[(table["level"] == level) & table["design"].isin(rank)]
            fig.add_trace(go.Scatter(
                x=[rank[d] for d in sub["design"]], y=sub[key], mode="markers",
                name=lib.level_label(problem, level), legendgroup=str(level),
                showlegend=col == 1, marker={"color": pk.LEVEL_COLOR[level], "size": 8,
                                             "symbol": pk.LEVEL_SYMBOL[level],
                                             "line": {"color": pk.SURFACE, "width": 2}},
                customdata=sub["design"],
                hovertemplate=f"design %{{customdata}}<br>{key} %{{y:.5f}}<extra></extra>"),
                row=1, col=col)
        fig.update_xaxes(title_text="sections sorted by their L3 drag (best on the left)",
                         row=1, col=col)
    pk.style(fig, "Where does the drag come from?",
             "Induced drag: NPLLT near-field (L1, L2) vs AVL Trefftz plane (L3); profile drag: "
             "polar CD at the strip cl (NeuralFoil vs XFOIL). Same sections at every level.", 540)
    pk.save(fig, "step1_drag_breakdown")


def figure_cl2d(table: pd.DataFrame, problem) -> None:
    target = problem.section_constraint["cl2d_target"]
    fig = go.Figure()
    for level in LEVELS:
        sub = table[(table["level"] == level) & table["cl2d_check_min"].notna()]
        jitter = np.random.default_rng(level).uniform(-0.15, 0.15, len(sub))
        fig.add_trace(go.Scatter(
            x=level + jitter, y=sub["cl2d_check_min"], mode="markers",
            name=lib.level_label(problem, level),
            marker={"color": pk.LEVEL_COLOR[level], "size": 8, "symbol": pk.LEVEL_SYMBOL[level],
                    "line": {"color": pk.SURFACE, "width": 2}}, customdata=sub["design"],
            hovertemplate="design %{customdata}<br>CL2d(0) %{y:.4f}<extra></extra>"))
    fig.add_hline(y=target, line={"color": pk.MUTED, "width": 1})
    fig.add_annotation(text=f"target {target} (projection with NeuralFoil "
                            f"{problem.section_constraint.get('reference_model', 'xxlarge')})",
                       x=0.5, xref="x", y=target, yanchor="bottom", showarrow=False,
                       font={"color": pk.SECONDARY, "size": 11})
    fig.update_xaxes(tickvals=list(LEVELS), ticktext=[lib.level_label(problem, lv)
                                                      for lv in LEVELS], range=[0.4, 3.6])
    fig.update_yaxes(title_text="CL2d at alpha = 0 deg seen by the level (every station)")
    pk.style(fig, "The 2D constraint CL2d(0) = 0.45 as seen by each level",
             "Every section is projected once (same geometry at every level); each solver then "
             "reports the CL2d(0 deg) of its own section polars at every station.", 520)
    pk.save(fig, "step1_cl2d_per_level")


def figure_costs(cost: dict, problem) -> None:
    levels = list(LEVELS)
    values = [cost[lv]["per_evaluation_s"] for lv in levels]
    fig = go.Figure(go.Bar(
        y=[lib.level_label(problem, lv) for lv in levels], x=values, orientation="h",
        marker={"color": [pk.LEVEL_COLOR[lv] for lv in levels]}, width=0.5,
        text=[f"{v:.2f} s ({cost[lv]['relative']:.1f}x)" for lv, v in zip(levels, values)],
        textposition="outside", textfont={"color": pk.PRIMARY},
        hovertemplate="%{y}<br>%{x:.3f} s per evaluation<extra></extra>"))
    fig.update_xaxes(type="log", title_text="wall time per evaluation (s, log scale)")
    fig.update_layout(showlegend=False)
    pk.style(fig, "Cost of one evaluation per level",
             "Mean wall time (CL2d projection + section + solver); the relative costs feed the "
             "merit function of the multi-fidelity EGO.", 380, top=110)
    pk.save(fig, "step1_costs")


def figure_spanwise(strips: dict, problem) -> None:
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.1,
                        subplot_titles=("Local lift coefficient cl", "Local profile drag cd"))
    for level in LEVELS:
        s = strips.get(level)
        if not s:
            continue
        for col, key in enumerate(("cl", "cd"), start=1):
            fig.add_trace(go.Scatter(
                x=s["s"], y=s[key], mode="lines", name=lib.level_label(problem, level),
                legendgroup=str(level), showlegend=col == 1,
                line={"color": pk.LEVEL_COLOR[level], "width": 2},
                hovertemplate=f"s %{{x:.3f}} m<br>{key} %{{y:.4f}}<extra></extra>"),
                row=1, col=col)
        fig.update_xaxes(title_text="arc length from the root s (m)")
    fig.update_yaxes(range=[0.0, 1.2], row=1, col=1)
    pk.style(fig, "Spanwise loading of the baseline section",
             "NPLLT control points (L1, L2) vs AVL stations (L3). The first point at the wall "
             "is a root singularity (wall image met at an angle), excluded from the stall check.",
             520)
    pk.save(fig, "step1_spanwise_loading")


def figure_mesh(mesh: dict, problem) -> None:
    fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.08,
                        subplot_titles=("NPLLT: nodes along the lifting line",
                                        "AVL: spanwise vortices",
                                        "NPLLT (L2): vortex cut-off at the wall"))
    df = pd.DataFrame(mesh["npllt"])
    for level in (1, 2):
        sub = df[df["level"] == level]
        fig.add_trace(go.Scatter(x=sub["nodes"], y=sub["CD"], mode="lines+markers",
                                 name=lib.level_label(problem, level),
                                 line={"color": pk.LEVEL_COLOR[level], "width": 2},
                                 marker={"size": 8, "line": {"color": pk.SURFACE, "width": 2}},
                                 hovertemplate="%{x} nodes<br>CD %{y:.6f}<extra></extra>"),
                      row=1, col=1)
    df = pd.DataFrame(mesh["avl"])
    fig.add_trace(go.Scatter(x=df["nspan"], y=df["CD"], mode="lines+markers",
                             name=lib.level_label(problem, 3),
                             line={"color": pk.LEVEL_COLOR[3], "width": 2},
                             marker={"size": 8, "line": {"color": pk.SURFACE, "width": 2}},
                             hovertemplate="nspan %{x}<br>CD %{y:.6f}<extra></extra>"),
                  row=1, col=2)
    df = pd.DataFrame(mesh.get("core_radius", []))
    if len(df):
        fig.add_trace(go.Scatter(x=df["core_radius"], y=df["CD"], mode="lines+markers",
                                 name="L2, core_radius", showlegend=False,
                                 line={"color": pk.LEVEL_COLOR[2], "width": 2},
                                 marker={"size": 8, "line": {"color": pk.SURFACE, "width": 2}},
                                 hovertemplate="core_radius %{x}<br>CD %{y:.6f}<extra></extra>"),
                      row=1, col=3)
        fig.update_xaxes(title_text="core_radius (fraction of the local chord)", row=1, col=3)
    fig.update_xaxes(type="log", title_text="nodes_count", row=1, col=1)
    fig.update_xaxes(title_text="nspan", row=1, col=2)
    fig.update_yaxes(title_text="CD of the baseline")
    pk.style(fig, "Mesh convergence on the baseline",
             f"Used in the study: nodes_count {problem.levels[0].options.get('nodes_count', 100)}"
             " and core_radius 0.005 (NPLLT, wall image), bdFoil default nspan = max(100, "
             "2.5 n_mesh) = 205 (AVL). Raw CD of the baseline (a missing point = failed).", 480)
    pk.save(fig, "step1_mesh_convergence")


# 4/4 ---------------------------------------------------------------------------------------------
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=48, help="number of designs (plus baseline)")
    parser.add_argument("--noise-designs", type=int, default=6)
    parser.add_argument("--noise-perturbations", type=int, default=4)
    parser.add_argument("--noise-scale", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--skip-mesh", action="store_true")
    parser.add_argument("--figures-only", action="store_true",
                        help="redraw the figures from the saved results (no solver run)")
    args = parser.parse_args(argv)

    lib.setup_logging("step1_figures" if args.figures_only else "step1_calibrate_solvers")
    problem = lib.load_problem()
    if args.figures_only:
        draw_figures(pd.read_csv(lib.RESULTS / "step1_calibration.csv"),
                     lib.load_json(lib.RESULTS / "step1_summary.json"), problem)
        return 0
    simulator = lib.make_simulator(problem)
    baseline = lib.baseline_parameters(problem)
    x0 = np.asarray(baseline["x"])
    start = time.perf_counter()

    points = lib.feasible_design(problem, args.n, seed=args.seed)
    logger.info("Common design: baseline + %d feasible sections, 3 levels each", len(points))
    table = pd.concat([evaluate_design(simulator, problem, x0[None, :], "baseline"),
                       evaluate_design(simulator, problem, points, "lhs")], ignore_index=True)
    table.loc[table["kind"] == "lhs", "design"] += 1
    table.to_csv(lib.RESULTS / "step1_calibration.csv", index=False)

    stats_ = agreement(table)
    stats_raw = agreement(table, "CD")
    cost = costs(table)
    advice = recommendation(stats_, cost)
    failures = {level: int(table[(table["level"] == level)]["value"].isna().sum())
                for level in LEVELS}
    for line in advice:
        logger.info("Recommendation: %s", line["text"])
    for pair in PAIRS:
        logger.info("L%d-L%d (equal lift): %s", *pair, {k: round(v, 4) for k, v in
                                                        stats_[f"L{pair[0]}-L{pair[1]}"].items()})
        logger.info("L%d-L%d (raw CD):     %s", *pair, {k: round(v, 4) for k, v in
                                                        stats_raw[f"L{pair[0]}-L{pair[1]}"].items()})
    logger.info("Costs: %s", {lv: round(c["per_evaluation_s"], 3) for lv, c in cost.items()})
    logger.info("Failures per level: %s", failures)

    noise_points = points[:args.noise_designs]
    noise = noise_floor(simulator, problem, noise_points, args.noise_perturbations,
                        args.noise_scale)
    for level in LEVELS:
        std_all = float(table[table["level"] == level]["value"].std())
        noise[level]["output_std"] = std_all
        noise[level]["noise_variance_ratio"] = (noise[level]["perturbation_std"] / std_all) ** 2
        logger.info("Noise L%d: %s", level, noise[level])

    strips = {}
    for level in LEVELS:
        strips[level] = simulator.evaluate_details(x0, level, with_strips=True)[2]
    mesh = None if args.skip_mesh else mesh_convergence(problem, simulator, x0)

    lift = {level: {"mean": float(table[table["level"] == level]["CL"].mean()),
                    "std": float(table[table["level"] == level]["CL"].std())} for level in LEVELS}
    summary = {"n_designs": int(table["design"].nunique()), "agreement": stats_,
               "agreement_raw_cd": stats_raw, "lift": lift, "costs": cost,
               "recommendation": advice, "failures": failures, "noise": noise, "mesh": mesh,
               "baseline_strips": strips, "baseline": {"x": x0, "physical": baseline["physical"]},
               "wall_time_s": time.perf_counter() - start}
    lib.save_json(lib.RESULTS / "step1_summary.json", summary)

    draw_figures(table, lib.load_json(lib.RESULTS / "step1_summary.json"), problem)
    logger.info("Step 1 done in %.1f min", (time.perf_counter() - start) / 60)
    return 0


def draw_figures(table: pd.DataFrame, summary: dict, problem) -> None:
    """Every figure of step 1 from the saved table and summary (JSON keys are strings)."""
    figure_scatter(table, summary["agreement"], problem)
    figure_scatter(table, summary["agreement_raw_cd"], problem, "CD",
                   "step1_level_agreement_raw_cd", "raw CD",
                   "Raw 3D drag at the fixed attitude: the levels disagree")
    figure_lift(table, problem)
    figure_breakdown(table, problem)
    figure_cl2d(table, problem)
    figure_costs({int(k): v for k, v in summary["costs"].items()}, problem)
    figure_spanwise({int(k): v for k, v in summary["baseline_strips"].items()}, problem)
    if summary.get("mesh"):
        figure_mesh(summary["mesh"], problem)


if __name__ == "__main__":
    raise SystemExit(main())
