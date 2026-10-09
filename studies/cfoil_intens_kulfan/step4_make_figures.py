"""
STEP 4 - Figures of the optimization, export of the response surface, index of the study.

Reads the last optimization run (results/step3_summary.json, or --run <folder>), first VERIFIES
at L3 the promising sections the run did not evaluate there (the effective best of the
surrogate, the minimum of the L3 surrogate mean over a feasible Sobol sample, the best L1 and
L2 sections: results/step4_l3_checks.json; the optimum reported is the best MEASURED L3
section), then writes:
* figures/step4_convergence.html: best L3 drag against the cumulative cost, and every
  evaluation in its order, coloured by level (which level the merit function chose, when);
* figures/step4_designs_parallel.html: every evaluated section (variables, delta, thickness,
  drag) in parallel coordinates, to brush ranges and see where the good sections are;
* figures/step4_sections.html: baseline vs best sections (shape, thickness, camber, weights);
* figures/step4_gp_slices_1d.html: the response surface along each variable through the
  optimum, mean +- 2 sigma of the L1, L2 and L3 surrogates;
* figures/step4_gp_slices_2d.html: L3 mean and standard deviation over any pair of variables
  (dropdown), infeasible sections (thickness constraint) left blank;
* figures/step4_lift_check.html: 3D side force / lift of every evaluated section: the CL2d(0)
  constraint keeps the zero-lift angle, hence the 3D lift at the fixed attitude, nearly fixed;
* figures/step4_spanwise_optimum.html and step4_geometry_3d.html: loading and 3D view of the
  baseline and of the optimum (the solvers are run again for these two sections);
* figures/step4_holdout.html: the final surrogate on the step 1 L3 sections;
* results/response_surface_sobol.csv and results/response_surface_slices.npz: the response
  surface (mean and standard deviation of every level) on a Sobol sample and on the slices;
* figures/index.html: the dashboard of the study (every figure of steps 0 to 4, key numbers).

Run:
    C:\\Users\\SIM\\.conda\\envs\\bdToolbox\\python.exe step4_make_figures.py [--run runs/<id>]
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
from scipy.stats import qmc

import plotkit as pk
import study_lib as lib
from pipelines.bdtoolbox_foil import geometry  # pylint: disable=wrong-import-order
from pipelines.bdtoolbox_foil.constraints import feasibility_function
from pipelines.bdtoolbox_foil.planform import CFoilArc, write_planform_csv
from src.surrogate_models import load_surrogate
from step0_check_setup import figure_geometry

logger = logging.getLogger("step4")
LEVELS = (1, 2, 3)


# 1/5 ---------------------------------------------------------------------------------------------
def load_run(run_dir: Path) -> dict:
    """Evaluations of a run in their order, with the level, the cost and the metrics."""
    state = lib.load_json(run_dir / "ego_backup.json")
    config = lib.load_json(run_dir / "config_effective.json") \
        if (run_dir / "config_effective.json").is_file() else lib.load_json(run_dir / "config.json")
    costs = [float(c) for c in state["costs"]]
    doe = config["optimization"]["doe"]
    rows = []
    for key, xs in state["X_dict"].items():
        level = int(key)
        ys = state["Y_dict"][key]
        metrics = state.get("Metrics_dict", {}).get(key, [{}] * len(xs))
        for i, (x, y, m) in enumerate(zip(xs, ys, metrics)):
            rows.append({"level": level, "index_in_level": i, "doe": i < doe[level - 1],
                         "value": np.nan if y is None else float(y),
                         "evaluation": (m or {}).get("evaluation"), "x": np.asarray(x),
                         **{k: v for k, v in (m or {}).items()
                            if isinstance(v, (int, float)) and k != "evaluation"}})
    table = pd.DataFrame(rows)
    # order: the evaluation counter of the simulator (DOE level by level, then the loop)
    if table["evaluation"].notna().all():
        table = table.sort_values("evaluation")
    else:
        table = table.sort_values(["doe", "level", "index_in_level"], ascending=[False, True,
                                                                                   True])
    table = table.reset_index(drop=True)
    table["cost"] = [costs[lv - 1] for lv in table["level"]]
    table["cumulative_cost"] = table["cost"].cumsum()
    best, history = np.inf, []
    for lv, v in zip(table["level"], table["value"]):
        if lv == 3 and np.isfinite(v):
            best = min(best, v)
        history.append(best if np.isfinite(best) else np.nan)
    table["best_l3"] = history
    return {"state": state, "table": table, "costs": costs, "config": config,
            "results": lib.load_json(run_dir / "results.json"),
            "surrogate": load_surrogate(str(run_dir / "surrogate.json"))}


# 2/5 ---------------------------------------------------------------------------------------------
def figure_convergence(run: dict, problem, baseline_cd: float) -> None:
    t = run["table"]
    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.14, row_heights=[0.5, 0.5],
                        subplot_titles=("Best L3 drag at equal lift found vs cumulative cost",
                                        "Every evaluation in its order (level chosen by the merit"
                                        " function)"))
    fig.add_trace(go.Scatter(x=t["cumulative_cost"], y=t["best_l3"], mode="lines",
                             line={"color": pk.LEVEL_COLOR[3], "width": 2, "shape": "hv"},
                             name="best L3 CD* so far",
                             hovertemplate="cost %{x:.0f}<br>best CD* %{y:.5f}<extra></extra>"),
                  row=1, col=1)
    fig.add_hline(y=baseline_cd, line={"color": pk.MUTED, "width": 1}, row=1, col=1)
    fig.add_annotation(text=f"baseline section, L3 CD* {baseline_cd:.5f}", xref="x domain",
                       x=1.0, xanchor="right", y=baseline_cd, yref="y", yanchor="bottom",
                       showarrow=False, font={"color": pk.SECONDARY, "size": 11})
    doe_end = int(t["doe"].sum())
    for level in LEVELS:
        sub = t[t["level"] == level]
        fig.add_trace(go.Scatter(
            x=sub.index + 1, y=sub["value"], mode="markers", name=lib.level_label(problem, level),
            marker={"color": pk.LEVEL_COLOR[level], "size": 8, "symbol": pk.LEVEL_SYMBOL[level],
                    "line": {"color": pk.SURFACE, "width": 2}},
            customdata=np.column_stack((sub["cumulative_cost"], sub["doe"])),
            hovertemplate="evaluation %{x}<br>CD* %{y:.5f}<br>cumulative cost %{customdata[0]:.0f}"
                          "<br>DOE %{customdata[1]}<extra></extra>"), row=2, col=1)
    fig.add_vline(x=doe_end + 0.5, line={"color": pk.MUTED, "width": 1}, row=2, col=1)
    fig.add_annotation(text="initial design | optimization loop", x=doe_end + 0.5, xref="x2",
                       y=0.98, yref="y2 domain", showarrow=False, yanchor="top",
                       xanchor="center", bgcolor=pk.SURFACE,
                       font={"color": pk.SECONDARY, "size": 11})
    fig.update_xaxes(title_text="cumulative cost (L1 evaluation = 1)", row=1, col=1)
    fig.update_xaxes(title_text="evaluation number", row=2, col=1)
    fig.update_yaxes(title_text="CD* (3D, equal lift)", row=1, col=1)
    fig.update_yaxes(title_text="CD* at its level", row=2, col=1)
    counts = t.groupby("level").size().to_dict()
    pk.style(fig, "Convergence of the multi-fidelity optimization",
             f"Evaluations per level {counts}; costs {run['costs']}. Each level has its own "
             "drag offset (bottom): only L3 values are comparable to the baseline line.", 820)
    pk.save(fig, "step4_convergence")


def figure_parallel(run: dict, problem) -> None:
    t = run["table"].dropna(subset=["value"])
    names = [v.name for v in problem.variables]
    dims = [{"label": "level", "values": t["level"].tolist(), "tickvals": list(LEVELS),
             "ticktext": ["L1", "L2", "L3"]}]
    for name in names + ["delta", "thickness"]:
        if name in t:
            dims.append({"label": name, "values": t[name].tolist()})
    dims.append({"label": "CD* (equal lift)", "values": t["value"].tolist()})
    fig = go.Figure(go.Parcoords(
        # reversed sequential ramp: dark = low drag (the good sections stand out)
        line={"color": t["value"].tolist(),
              "colorscale": [[1 - p, c] for p, c in pk.SEQUENTIAL_SCALE][::-1], "showscale": True,
              "colorbar": {"title": {"text": "CD*", "font": {"color": pk.SECONDARY}},
                           "tickfont": {"color": pk.MUTED}}},
        dimensions=dims, labelfont={"color": pk.PRIMARY}, tickfont={"color": pk.MUTED}))
    pk.style(fig, "Every evaluated section (parallel coordinates)",
             "Drag an axis range to filter (e.g. level = L3 and the lowest CD) and see which "
             "variables the good sections share. Dark = low drag.", 560, top=140)
    pk.save(fig, "step4_designs_parallel")


def figure_sections(problem, sections: list) -> None:
    """sections: list of (Section, label, color, dash)."""
    fig = make_subplots(rows=2, cols=2, vertical_spacing=0.14, horizontal_spacing=0.08,
                        specs=[[{"colspan": 2}, None], [{}, {}]],
                        subplot_titles=("Section shapes", "Thickness t / c", "Camber / c"))
    xs = geometry.THICKNESS_STATIONS
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
    fig.update_yaxes(scaleanchor="x", scaleratio=1, row=1, col=1)
    fig.update_xaxes(title_text="x / c", row=2, col=1)
    fig.update_xaxes(title_text="x / c", row=2, col=2)
    lines = []
    for section, label, _, _ in sections:
        p = section.parameters
        lines.append(f"{label}: t/c {section.thickness:.4f}, delta {p.get('delta', 0):.4f}, t = "
                     + ", ".join(f"{p[f't_{i}']:.3f}" for i in range(4)) + "; s = "
                     + ", ".join(f"{p.get(f's_{i}', 0.0):.3f}" for i in range(1, 4)))
    pk.style(fig, "Baseline and optimized sections", "<br>".join(lines), 900, top=200)
    pk.save(fig, "step4_sections")


def slice_1d(x_opt: np.ndarray, dim: int, n: int = 61):
    grid = np.linspace(0.0, 1.0, n)
    points = np.repeat(x_opt[None, :], n, axis=0)
    points[:, dim] = grid
    return grid, points


def figure_slices_1d(run: dict, problem, x_opt: np.ndarray, feasibility) -> dict:
    model = run["surrogate"]
    names = [v.name for v in problem.variables]
    ncols = 4
    nrows = int(np.ceil(len(names) / ncols))
    fig = make_subplots(rows=nrows, cols=ncols, horizontal_spacing=0.06,
                        vertical_spacing=0.16, subplot_titles=names)
    export = {}
    for d, var in enumerate(problem.variables):
        grid, points = slice_1d(x_opt, d)
        feasible = feasibility(points) if feasibility else np.ones(len(grid), dtype=bool)
        physical = var.lower + grid * (var.upper - var.lower)
        row, col = d // ncols + 1, d % ncols + 1
        for level in LEVELS:
            mean, var_, _ = model.predict_batch(points, level=level)
            sd = np.sqrt(np.maximum(var_, 0.0))
            mean_f = np.where(feasible, mean, np.nan)
            export[f"{var.name}_L{level}"] = np.column_stack((physical, mean, sd, feasible))
            color = pk.LEVEL_COLOR[level]
            fig.add_trace(go.Scatter(
                x=np.concatenate((physical, physical[::-1])),
                y=np.concatenate((mean_f + 2 * sd, (mean_f - 2 * sd)[::-1])), fill="toself",
                fillcolor=pk.rgba(color, 0.12), line={"width": 0}, hoverinfo="skip",
                showlegend=False, legendgroup=str(level)), row=row, col=col)
            fig.add_trace(go.Scatter(
                x=physical, y=mean_f, mode="lines", line={"color": color, "width": 2},
                name=lib.level_label(problem, level) + " surrogate (mean +- 2 sigma)",
                legendgroup=str(level), showlegend=d == 0,
                hovertemplate=f"{var.name} %{{x:.4f}}<br>CD* %{{y:.5f}}<extra>L{level}</extra>"),
                row=row, col=col)
        x_phys = var.lower + x_opt[d] * (var.upper - var.lower)
        fig.add_vline(x=x_phys, line={"color": pk.MUTED, "width": 1}, row=row, col=col)
    pk.style(fig, "Response surface along each variable, through the optimum",
             "Other variables fixed at the optimum (vertical line). Gaps: sections violating the "
             "thickness constraint (active at the optimum: lowering t_0 there makes the section "
             "too thin). Each level has its own surrogate (recursive model).",
             300 * nrows + 220)
    pk.save(fig, "step4_gp_slices_1d")
    return export


def figure_slices_2d(run: dict, problem, x_opt: np.ndarray, feasibility, n: int = 41) -> dict:
    model = run["surrogate"]
    names = [v.name for v in problem.variables]
    pairs = list(itertools.combinations(range(len(names)), 2))
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.12,
                        subplot_titles=("L3 mean of CD* (equal lift)", "L3 standard deviation"))
    grid = np.linspace(0.0, 1.0, n)
    export = {}
    for k, (i, j) in enumerate(pairs):
        gi, gj = np.meshgrid(grid, grid)
        points = np.repeat(x_opt[None, :], n * n, axis=0)
        points[:, i], points[:, j] = gi.ravel(), gj.ravel()
        mean, var_, _ = model.predict_batch(points, level=3)
        feasible = feasibility(points) if feasibility else np.ones(len(points), dtype=bool)
        mean = np.where(feasible, mean, np.nan).reshape(n, n)
        sd = np.where(feasible, np.sqrt(np.maximum(var_, 0.0)), np.nan).reshape(n, n)
        vi, vj = problem.variables[i], problem.variables[j]
        xi, xj = vi.lower + grid * (vi.upper - vi.lower), vj.lower + grid * (vj.upper - vj.lower)
        export[f"{vi.name}__{vj.name}"] = np.stack((mean, sd))
        visible = k == 0
        for col, (z, title) in enumerate(((mean, "CD*"), (sd, "sigma")), start=1):
            fig.add_trace(go.Heatmap(
                x=xi, y=xj, z=z, colorscale=pk.SEQUENTIAL_SCALE, visible=visible,
                colorbar={"x": 0.44 if col == 1 else 1.0, "len": 0.8,
                          "tickfont": {"color": pk.MUTED}},
                hovertemplate=f"{vi.name} %{{x:.4f}}<br>{vj.name} %{{y:.4f}}<br>{title} "
                              "%{z:.5f}<extra></extra>"), row=1, col=col)
            fig.add_trace(go.Scatter(
                x=[vi.lower + x_opt[i] * (vi.upper - vi.lower)],
                y=[vj.lower + x_opt[j] * (vj.upper - vj.lower)], mode="markers",
                marker={"color": pk.PRIMARY, "size": 11, "symbol": "x"}, visible=visible,
                showlegend=False, hovertemplate="optimum<extra></extra>"), row=1, col=col)
    buttons = []
    for k, (i, j) in enumerate(pairs):
        visible = [False] * (4 * len(pairs))
        visible[4 * k:4 * k + 4] = [True] * 4
        buttons.append({"label": f"{names[i]} x {names[j]}", "method": "update",
                        "args": [{"visible": visible},
                                 {"xaxis.title.text": names[i], "yaxis.title.text": names[j],
                                  "xaxis2.title.text": names[i],
                                  "yaxis2.title.text": names[j]}]})
    fig.update_layout(updatemenus=[{"buttons": buttons, "x": 0.0, "xanchor": "left",
                                    "y": 1.12, "yanchor": "bottom", "bgcolor": pk.SURFACE,
                                    "font": {"color": pk.PRIMARY}}])
    fig.update_xaxes(title_text=names[pairs[0][0]])
    fig.update_yaxes(title_text=names[pairs[0][1]])
    pk.style(fig, "Response surface over a pair of variables (L3 surrogate)",
             "Choose the pair in the menu; the other variables are fixed at the optimum (x). "
             "Blank: thickness constraint violated. Right: where the model is still uncertain.",
             620, top=190)
    pk.save(fig, "step4_gp_slices_2d")
    return export


def figure_lift(run: dict, problem) -> None:
    t = run["table"].dropna(subset=["value"])
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.1,
                        subplot_titles=("Side force coefficient Cy", "Vertical lift Cz"))
    for col, key in enumerate(("Cy", "Cz"), start=1):
        for level in LEVELS:
            sub = t[t["level"] == level]
            if key not in sub:
                continue
            fig.add_trace(go.Scatter(
                x=sub["value"], y=sub[key], mode="markers", name=lib.level_label(problem, level),
                legendgroup=str(level), showlegend=col == 1,
                marker={"color": pk.LEVEL_COLOR[level], "size": 7,
                        "symbol": pk.LEVEL_SYMBOL[level],
                        "line": {"color": pk.SURFACE, "width": 1.5}},
                hovertemplate=f"CD* %{{x:.5f}}<br>{key} %{{y:.4f}}<extra></extra>"),
                row=1, col=col)
        fig.update_xaxes(title_text="CD* (equal lift)", row=1, col=col)
    pk.style(fig, "3D forces of every evaluated section (fixed attitude)",
             "NPLLT (viscous polars, CL2d(0) fixed): the lift barely changes; AVL (camber line "
             "only): it changes from one section to another. The objective CD* compares the "
             "sections at the lift of the baseline at each level.", 500)
    pk.save(fig, "step4_lift_check")


def figure_spanwise(problem, strips: dict) -> None:
    """strips: {(label, level): strips dict}."""
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.1,
                        subplot_titles=("Local lift coefficient cl", "Local profile drag cd"))
    for (label, level), s in strips.items():
        if not s:
            continue
        dash = "dot" if label == "baseline" else "solid"
        for col, key in enumerate(("cl", "cd"), start=1):
            fig.add_trace(go.Scatter(
                x=s["s"], y=s[key], mode="lines",
                name=f"{label}, {lib.level_label(problem, level)}", legendgroup=f"{label}{level}",
                showlegend=col == 1, line={"color": pk.LEVEL_COLOR[level], "width": 2,
                                           "dash": dash},
                hovertemplate=f"s %{{x:.3f}} m<br>{key} %{{y:.4f}}<extra></extra>"),
                row=1, col=col)
        fig.update_xaxes(title_text="arc length from the root s (m)")
    fig.update_yaxes(range=[0.0, 1.2], row=1, col=1)
    pk.style(fig, "Spanwise loading: baseline (dotted) vs optimum (solid)",
             "Same attitude; NPLLT (L2) and AVL (L3). The root point at the wall is a "
             "singularity of the wall image.", 520)
    pk.save(fig, "step4_spanwise_optimum")


def figure_holdout(check: dict) -> None:
    y, m, s = np.array(check["y"]), np.array(check["mean"]), np.sqrt(np.array(check["var"]))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y, y=m, mode="markers", name="step 1 L3 section (CD*)",
                             error_y={"type": "data", "array": 1.96 * s, "thickness": 1,
                                      "width": 0, "color": pk.rgba(pk.LEVEL_COLOR[3], 0.5)},
                             marker={"color": pk.LEVEL_COLOR[3], "size": 8, "symbol": "diamond",
                                     "line": {"color": pk.SURFACE, "width": 2}},
                             hovertemplate="L3 CD* %{x:.5f}<br>predicted %{y:.5f}<extra></extra>"))
    lo, hi = float(min(y.min(), m.min())), float(max(y.max(), m.max()))
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="perfect prediction",
                             line={"color": pk.MUTED, "width": 1}, hoverinfo="skip"))
    fig.update_xaxes(title_text="L3 CD* (AVL x XFOIL) of the step 1 sections")
    fig.update_yaxes(title_text="final surrogate +- 1.96 sigma")
    pk.style(fig, "Independent check of the response surface",
             f"{check['n']} step 1 sections never seen by the optimization: RMSE "
             f"{100 * check['rmse_relative']:.0f} % of their spread, 95 % coverage "
             f"{100 * check['coverage95']:.0f} %. The surrogate is accurate where the "
             "optimization sampled, less elsewhere.", 520)
    pk.save(fig, "step4_holdout")


def verify_candidates(run: dict, problem, simulator, feasibility) -> list:
    """
    L3 evaluation of the promising sections the run did not evaluate at L3: the cheap levels
    and the surrogate can point to a section that was never checked with AVL x XFOIL.
    """
    t = run["table"]
    model = run["surrogate"]
    candidates = {}
    best_x = run["results"]["summary"].get("surrogate_best_x")
    if best_x is not None:
        candidates["effective best of the surrogate (Eq. 19)"] = np.asarray(best_x)
    points = qmc.Sobol(problem.dim, seed=1).random_base2(14)
    if feasibility is not None:
        points = points[feasibility(points)]
    mean, _, _ = model.predict_batch(points, level=3)
    candidates["minimum of the L3 surrogate mean (Sobol sample)"] = points[int(np.argmin(mean))]
    for level in (1, 2):
        sub = t[t["level"] == level].dropna(subset=["value"])
        if len(sub):
            candidates[f"best L{level} section"] = np.asarray(sub.loc[sub["value"].idxmin(), "x"])
    l3 = t[t["level"] == 3]
    checks = []
    for name, x in candidates.items():
        m, v, _ = model.predict_batch(x[None, :], level=3)
        known = [i for i, xi in zip(l3.index, l3["x"]) if np.linalg.norm(np.asarray(xi) - x) < 1e-6]
        if known:
            value, metrics = float(l3.loc[known[0], "value"]), dict(l3.loc[known[0]])
            source = "already evaluated by the run"
        else:
            value, metrics = simulator.evaluate(x, 3)
            source = "evaluated by step 4"
        checks.append({"candidate": name, "source": source, "x": x.tolist(),
                       "physical": {v_.name: float(v_.lower + u * (v_.upper - v_.lower))
                                    for v_, u in zip(problem.variables, x)},
                       "predicted": float(m[0]), "predicted_sd": float(np.sqrt(max(v[0], 0.0))),
                       "l3_value": value, **{k: metrics.get(k) for k in
                                             ("CD", "CL", "Cdi", "Cdprofile", "thickness",
                                              "delta")}})
        logger.info("L3 check, %s: measured %s, predicted %.5f +- %.5f (%s)", name, value,
                    m[0], np.sqrt(max(v[0], 0.0)), source)
    lib.save_json(lib.RESULTS / "step4_l3_checks.json", checks)
    return checks


# 3/5 ---------------------------------------------------------------------------------------------
def export_response_surface(run: dict, problem, feasibility, slices_1d: dict,
                            slices_2d: dict, n_log2: int = 12) -> None:
    """Mean / sigma of every level on a Sobol sample (CSV) and on the slices (NPZ)."""
    model = run["surrogate"]
    points = qmc.Sobol(problem.dim, seed=0).random_base2(n_log2)
    table = pd.DataFrame(points, columns=[f"x_{i}" for i in range(problem.dim)])
    for d, var in enumerate(problem.variables):
        table[var.name] = var.lower + points[:, d] * (var.upper - var.lower)
    table["feasible"] = feasibility(points) if feasibility else True
    for level in LEVELS:
        mean, var_, _ = model.predict_batch(points, level=level)
        table[f"mean_L{level}"] = mean
        table[f"sd_L{level}"] = np.sqrt(np.maximum(var_, 0.0))
    table.to_csv(lib.RESULTS / "response_surface_sobol.csv", index=False)
    np.savez_compressed(lib.RESULTS / "response_surface_slices.npz",
                        **{f"slice1d_{k}": v for k, v in slices_1d.items()},
                        **{f"slice2d_{k}": v for k, v in slices_2d.items()})


# 4/5 ---------------------------------------------------------------------------------------------
FIGURE_INDEX = [
    ("Step 0 - set-up", [
        ("step0_geometry_3d", "Positioned C-foil, bdFoil frame, wall image"),
        ("step0_baseline_section", "Current section, Kulfan fit, projected baseline")]),
    ("Step 1 - solver calibration", [
        ("step1_level_agreement", "Same sections at the 3 levels: correlations, ranking (CD*)"),
        ("step1_level_agreement_raw_cd", "The same with the raw CD: the levels disagree"),
        ("step1_lift_per_level", "Why: the 3D lift of the same sections at each level"),
        ("step1_drag_breakdown", "Induced vs profile drag per level"),
        ("step1_cl2d_per_level", "The CL2d(0) = 0.45 constraint seen by each level"),
        ("step1_costs", "Cost of one evaluation per level"),
        ("step1_spanwise_loading", "Spanwise loading of the baseline, NPLLT vs AVL"),
        ("step1_mesh_convergence", "Mesh convergence (NPLLT nodes, AVL vortices)")]),
    ("Step 2 - GP calibration", [
        ("step2_gp_scores", "MLE vs MAP, single vs multi-fidelity: leave-one-out scores"),
        ("step2_leave_one_out", "Leave-one-out predictions of the chosen GP"),
        ("step2_lengthscales", "Fitted lengthscales per variable and level"),
        ("step2_learning_curve", "Accuracy vs number of expensive points")]),
    ("Step 4 - optimization", [
        ("step4_convergence", "Convergence and level chosen at each evaluation"),
        ("step4_sections", "Baseline vs optimized sections"),
        ("step4_designs_parallel", "Every evaluated section (parallel coordinates)"),
        ("step4_gp_slices_1d", "Response surface along each variable"),
        ("step4_gp_slices_2d", "Response surface over pairs of variables"),
        ("step4_lift_check", "3D lift of every section (fixed attitude)"),
        ("step4_spanwise_optimum", "Spanwise loading, baseline vs optimum"),
        ("step4_geometry_3d", "3D view with the optimized section"),
        ("step4_holdout", "Independent check of the response surface")]),
]


def write_index(numbers: list) -> Path:
    """figures/index.html: key numbers and links to every figure that exists."""
    cards = []
    for title, items in FIGURE_INDEX:
        links = "".join(
            f'<li><a href="{name}.html">{html.escape(text)}</a></li>' for name, text in items
            if (lib.FIGURES / f"{name}.html").is_file())
        if links:
            cards.append(f"<section><h2>{html.escape(title)}</h2><ul>{links}</ul></section>")
    tiles = "".join(f'<div class="tile"><div class="label">{html.escape(k)}</div>'
                    f'<div class="value">{html.escape(v)}</div></div>' for k, v in numbers)
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>C-foil section study</title>
<style>
:root {{ --surface:#fcfcfb; --page:#f9f9f7; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781;
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
.label {{ color:var(--ink2); font-size:13px; }} .value {{ font-size:20px; font-weight:600;
        margin-top:4px; }}
section {{ background:var(--surface); border:1px solid var(--line); border-radius:8px;
          padding:4px 18px 10px; margin-bottom:14px; }}
h2 {{ font-size:16px; }} li {{ margin:6px 0; }} a {{ color:var(--accent); }}
</style></head><body><main>
<h1>Intens SY C-foil: multi-fidelity optimization of the section</h1>
<p class="sub">3 levels: NPLLT x NeuralFoil xxsmall, NPLLT x NeuralFoil xxlarge, AVL x XFOIL.
Minimum 3D drag at equal lift (CD*), fixed attitude, CL2d(0 deg) = 0.45. See GUIDE_PIPELINE.md and RAPPORT.md.</p>
<div class="tiles">{tiles}</div>{''.join(cards)}</main></body></html>"""
    path = lib.FIGURES / "index.html"
    path.write_text(page, encoding="utf-8")
    return path


# 5/5 ---------------------------------------------------------------------------------------------
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", help="run folder (default: the one of step3_summary.json)")
    parser.add_argument("--no-solver", action="store_true",
                        help="skip the figures that run the solvers again (spanwise, 3D)")
    args = parser.parse_args(argv)
    lib.setup_logging("step4_make_figures")
    problem = lib.load_problem()
    run_dir = Path(args.run) if args.run else Path(
        lib.load_json(lib.RESULTS / "step3_summary.json")["run_dir"])
    if not run_dir.is_absolute():
        run_dir = lib.STUDY / run_dir
    run = load_run(run_dir)
    feasibility = feasibility_function(problem)
    simulator = lib.make_simulator(problem)
    baseline = lib.baseline_parameters(problem)
    x_base = np.asarray(baseline["x"])
    t = run["table"]
    baseline_cd = simulator.evaluate(x_base, 3)[0]
    checks = verify_candidates(run, problem, simulator, feasibility)
    # optimum = best MEASURED L3 section (run or step 4 checks)
    x_best = np.asarray(run["results"]["best_design_normalized"])
    best_cd = float(run["results"]["best_objective"])
    for check in checks:
        if check["l3_value"] is not None and np.isfinite(check["l3_value"]) \
           and check["l3_value"] < best_cd:
            x_best, best_cd = np.asarray(check["x"]), float(check["l3_value"])

    figure_convergence(run, problem, baseline_cd)
    figure_parallel(run, problem)
    sections = [(simulator.section_of(x_base), "baseline (MC2 thickness + camber offset)",
                 pk.SECONDARY, "dot")]
    for level in (1, 2):
        sub = t[(t["level"] == level)].dropna(subset=["value"])
        if len(sub):
            x_l = np.asarray(sub.loc[sub["value"].idxmin(), "x"])
            sections.append((simulator.section_of(x_l), f"best at L{level}",
                             pk.LEVEL_COLOR[level], "dash"))
    sections.append((simulator.section_of(x_best), "best at L3 (optimum)", pk.PRIMARY, "solid"))
    figure_sections(problem, sections)
    slices_1d = figure_slices_1d(run, problem, x_best, feasibility)
    slices_2d = figure_slices_2d(run, problem, x_best, feasibility)
    figure_lift(run, problem)
    export_response_surface(run, problem, feasibility, slices_1d, slices_2d)
    holdout_path = run_dir / "holdout_step1.json"
    if holdout_path.is_file():
        figure_holdout(lib.load_json(holdout_path))

    if not args.no_solver:
        strips = {}
        for label, x in (("baseline", x_base), ("optimum", x_best)):
            for level in (2, 3):
                strips[(label, level)] = simulator.evaluate_details(x, level, with_strips=True)[2]
        figure_spanwise(problem, strips)
        arc = CFoilArc.from_config(problem.planform)
        avl = simulator.backends[2]
        workdir = lib.RESULTS / "step4_files"
        opt_section = simulator.section_of(x_best)
        xf = geometry.write_xf(opt_section, workdir)
        write_planform_csv(arc.stations(), xf.name, workdir / "planform.csv")
        geometry.write_xf(simulator.section_of(x_base), workdir)
        planform = avl.core.model.load_planform(workdir / "planform.csv", name="cfoil",
                                                kind=arc.kind, section_dirs=[workdir])
        stations = avl.core.geometry.position(planform, avl.attitude, avl.settings)
        figure_geometry(problem, {"stations": {k: np.asarray(getattr(stations, k)).tolist()
                                               for k in ("x", "y", "z", "c", "ainc")}},
                        [(simulator.section_of(x_base), "baseline section", pk.SECONDARY),
                         (opt_section, "optimized section", pk.PRIMARY)],
                        name="step4_geometry_3d", title="C-foil with the optimized section")
    gain = 100 * (baseline_cd - best_cd) / baseline_cd
    summary = lib.load_json(lib.RESULTS / "step3_summary.json") \
        if (lib.RESULTS / "step3_summary.json").is_file() else {}
    numbers = [("Baseline CD* (L3)", f"{baseline_cd:.5f}"), ("Optimum CD* (L3)", f"{best_cd:.5f}"),
               ("Drag reduction", f"{gain:.1f} %"),
               ("Evaluations L1 / L2 / L3", " / ".join(str(int((t['level'] == lv).sum()))
                                                       for lv in LEVELS)),
               ("GP estimator", str(run["results"].get("gp", {}).get("estimator", "?")))]
    if summary.get("holdout_step1"):
        numbers.append(("Hold-out RMSE / spread",
                        f"{100 * summary['holdout_step1']['rmse_relative']:.0f} %"))
    lib.save_json(lib.RESULTS / "step4_summary.json",
                  {"run_dir": str(run_dir), "baseline_cd_l3": baseline_cd, "best_cd_l3": best_cd,
                   "gain_percent": gain, "best_design_normalized": x_best.tolist(),
                   "best_design_physical": {v.name: float(v.lower + u * (v.upper - v.lower))
                                            for v, u in zip(problem.variables, x_best)},
                   "l3_checks": checks,
                   "n_evaluations": {lv: int((t["level"] == lv).sum()) for lv in LEVELS}})
    path = write_index(numbers)
    logger.info("Baseline L3 CD %.5f, optimum %.5f (%.1f %%). Index: %s", baseline_cd, best_cd,
                gain, path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
