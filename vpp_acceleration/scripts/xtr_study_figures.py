"""
Figures of the forced-transition study of 2026-11-09 (user: "voir les sections optimisées et les
différences entre elles, comprendre l'importance de ce paramètre et aussi par rapport à la CFD").

Inputs: finished run folders of scripts/run_real_vpp.py made with --xtr (same search space), the
cross evaluations of scripts/robustness_checks.py cross found next to them
(runs/*_cross_<run folder>), optionally a CSV of CFD forces (--cfd, see cfd_figure).
Output: runs/<date>_xtr_study/ with figures/index.html (+ one html per figure, plotly from the CDN)
and results/sections.csv, results/cross.csv.

  1. the optimized sections and the current one: shape, surface change vs current [mm], thickness
     and camber distributions, geometric table (t/c, camber, positions, LE radius, Ixx, Ixx/v);
  2. the cross matrix: speed of every best section in the physics (surrogate + AVL x XFOIL) of
     every run, i.e. what one loses by optimizing with the wrong transition;
  3. speed gain vs the current section per operating condition (profile analysis of each run);
  4. 2D NeuralFoil polars (Re 5e6, n_crit 1) of the sections at each xtr: the mechanism.

Run (MDS environment):
    ..\\.venv-mds\\Scripts\\python scripts\\xtr_study_figures.py --runs runs/<run 2 %> runs/<run 5 %> ...
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

VPP_DIR = Path(__file__).resolve().parents[1]
if str(VPP_DIR) not in sys.path:
    sys.path.insert(0, str(VPP_DIR))
# pylint: disable=wrong-import-position
from src.run_dir import RunDir  # noqa: E402
from src.section_pca import (CFoilFamily, section_coordinates, section_ixx,  # noqa: E402
                             section_modulus)
from src.foil_codes import STUDY_CONFIG  # noqa: E402

CHORD_MM = 700.0
TE = 0.00197
X = (1.0 - np.cos(np.linspace(0.0, np.pi, 301))) / 2.0        # cosine stations
COLORS = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4", "#9467bd", "#8c564b"]
CURRENT_COLOR = "#444444"


def run_label(config: dict) -> str:
    xtr = config.get("xtr") or [0.1, 0.1]
    label = f"xtr {100 * xtr[0]:g} %"
    if config.get("start_section"):
        label += f" (départ {Path(config['start_section']).parents[1].name[16:22]})"
    return label


def load(runs) -> tuple:
    """{label: weights} of the best sections (+ the current one), {label: run folder}, configs."""
    sections, folders, configs = {}, {}, {}
    for run in runs:
        root = (VPP_DIR / run).resolve()
        config = json.loads((root / "config.json").read_text(encoding="utf-8"))
        best = json.loads((root / "results" / "best_section.json").read_text(encoding="utf-8"))
        label = run_label(config)
        sections[label] = np.array(best["kulfan_upper"] + best["kulfan_lower"])
        folders[label], configs[label] = root, config
    current = CFoilFamily.from_study_config(STUDY_CONFIG).reference_weights()
    return {"actuelle (MC2)": np.asarray(current), **sections}, folders, configs


def geometry(weights: np.ndarray) -> dict:
    y = section_coordinates(weights, X, TE)[0]
    upper, lower = y[:len(X)], y[len(X):]
    tau, camber = upper - lower, (upper + lower) / 2.0
    return {"upper": upper, "lower": lower, "thickness": tau, "camber": camber,
            "t_max": tau.max(), "x_t_max": X[tau.argmax()], "camber_max": camber.max(),
            "x_camber_max": X[camber.argmax()],
            "r_le_upper": weights[0] ** 2 / 2.0, "r_le_lower": weights[8] ** 2 / 2.0,
            "ixx": float(section_ixx(weights[None, :], TE)[0]),
            "ixx_v": float(section_modulus(weights[None, :], TE)[0])}


def sections_figure(sections: dict):
    import plotly.graph_objects as go  # noqa: PLC0415
    from plotly.subplots import make_subplots  # noqa: PLC0415
    geo = {name: geometry(w) for name, w in sections.items()}
    ref = geo["actuelle (MC2)"]
    fig = make_subplots(rows=4, cols=1, vertical_spacing=0.06, subplot_titles=(
        "Sections (échelle verticale x3)",
        "Écart vertical à la section actuelle [mm, corde 700 mm] (trait plein extrados, pointillé intrados)",
        "Épaisseur t(x)/c [%]", "Ligne de cambrure y_c(x)/c [%]"))
    for i, (name, g) in enumerate(geo.items()):
        color = CURRENT_COLOR if i == 0 else COLORS[(i - 1) % len(COLORS)]
        width = 3 if i == 0 else 2
        xs = np.concatenate([X[::-1], X[1:]])
        ys = np.concatenate([g["upper"][::-1], g["lower"][1:]])
        fig.add_trace(go.Scatter(x=xs, y=3 * ys, name=name, legendgroup=name,
                                 line={"color": color, "width": width}), 1, 1)
        if i:
            for key, dash in (("upper", "solid"), ("lower", "dot")):
                fig.add_trace(go.Scatter(x=X, y=(g[key] - ref[key]) * CHORD_MM, name=name,
                                         legendgroup=name, showlegend=False,
                                         line={"color": color, "dash": dash}), 2, 1)
        fig.add_trace(go.Scatter(x=X, y=100 * g["thickness"], legendgroup=name, showlegend=False,
                                 name=name, line={"color": color, "width": width}), 3, 1)
        fig.add_trace(go.Scatter(x=X, y=100 * g["camber"], legendgroup=name, showlegend=False,
                                 name=name, line={"color": color, "width": width}), 4, 1)
    fig.update_xaxes(title_text="x/c", row=4, col=1)
    fig.update_yaxes(scaleanchor="x", scaleratio=1, row=1, col=1)
    fig.update_layout(height=1500, title="Sections optimisées selon la transition forcée",
                      hovermode="x unified")
    rows = []
    for name, g in geo.items():
        rows.append({"section": name, "t/c max [%]": 100 * g["t_max"], "x t max": g["x_t_max"],
                     "cambrure max [%]": 100 * g["camber_max"], "x cambrure max": g["x_camber_max"],
                     "r BA extrados [%c]": 100 * g["r_le_upper"],
                     "r BA intrados [%c]": 100 * g["r_le_lower"],
                     "Ixx / actuelle": g["ixx"] / ref["ixx"], "Ixx/v / actuelle": g["ixx_v"] / ref["ixx_v"]})
    return fig, pd.DataFrame(rows)


def cross_table(folders: dict) -> pd.DataFrame:
    """Rows: physics of a run (its surrogate + AVL at its xtr); columns: sections; speed [kt]."""
    by_folder = {root.name: label for label, root in folders.items()}
    table = {}
    for label, root in folders.items():
        found = sorted(root.parent.glob(f"*_cross_{root.name}"))
        summaries = [f / "results" / "summary.json" for f in found if (f / "results" / "summary.json").exists()]
        if not summaries:
            continue
        summary = json.loads(summaries[-1].read_text(encoding="utf-8"))
        row = {}
        for name, speed in summary["speed_kt"].items():
            if name == "own_best":
                row[label] = speed
            elif name.startswith("best_") and name[5:] in by_folder:
                row[by_folder[name[5:]]] = speed
            elif name == "current_section":
                row["actuelle (MC2)"] = speed
        table[label] = row
    return pd.DataFrame(table).T


def cross_figure(table: pd.DataFrame):
    import plotly.graph_objects as go  # noqa: PLC0415
    columns = [c for c in table.columns if c != "actuelle (MC2)"]
    delta = table[columns].sub(table[columns].max(axis=1), axis=0)
    text = [[f"{table.loc[r, c]:.3f} kt<br>{delta.loc[r, c]:+.3f}" if np.isfinite(table.loc[r, c])
             else "échec" for c in columns] for r in table.index]
    fig = go.Figure(go.Heatmap(z=delta.values, x=columns, y=list(table.index), text=text,
                               texttemplate="%{text}", colorscale="RdYlGn", zmax=0,
                               colorbar={"title": "écart au meilleur<br>de la ligne [kt]"}))
    fig.update_layout(title="Vitesse moyenne profil de chaque section optimisée, évaluée dans la "
                            "physique de chaque run (surrogate + AVL x XFOIL à son xtr)",
                      xaxis_title="section optimisée avec", yaxis_title="évaluée avec la physique",
                      height=520)
    return fig


def gains_figure(folders: dict):
    import plotly.graph_objects as go  # noqa: PLC0415
    from plotly.subplots import make_subplots  # noqa: PLC0415
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Gain vs section actuelle par condition "
                                                        "[kt]", "Gain [% de la vitesse]"))
    for i, (label, root) in enumerate(folders.items()):
        path = root / "results" / "profile_profile_v2.csv"
        if not path.exists():
            continue
        d = pd.read_csv(path)
        best = d[d.section == "best"].set_index(["twa", "tws"])
        cur = d[d.section == "current_section"].set_index(["twa", "tws"])
        common = best.index.intersection(cur.index)
        gain = (best.loc[common, "speed_kt"] - cur.loc[common, "speed_kt"]).sort_index()
        pct = 100 * gain / cur.loc[gain.index, "speed_kt"]
        names = [f"{a:.0f}°/{s:.0f} kn" for a, s in gain.index]
        color = COLORS[i % len(COLORS)]
        fig.add_trace(go.Scatter(x=names, y=gain.values, name=label, mode="markers+lines",
                                 line={"color": color}), 1, 1)
        fig.add_trace(go.Scatter(x=names, y=pct.values, name=label, showlegend=False,
                                 mode="markers+lines", line={"color": color}), 1, 2)
    fig.update_layout(height=550, title="Gain des sections optimisées par condition (profile_v2, "
                                        "TWA/TWS ; chacune dans la physique de son run)")
    return fig


def polars_figure(sections: dict, xtrs):
    import neuralfoil as nf  # noqa: PLC0415
    import plotly.graph_objects as go  # noqa: PLC0415
    from plotly.subplots import make_subplots  # noqa: PLC0415
    alpha = np.arange(-4.0, 10.01, 0.5)
    dashes = ["dot", "dash", "solid", "dashdot"]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("CD(CL) x 10⁴", "CL / CD (alpha)"))
    for i, (name, w) in enumerate(sections.items()):
        color = CURRENT_COLOR if i == 0 else COLORS[(i - 1) % len(COLORS)]
        kulfan = {"upper_weights": w[:8], "lower_weights": w[8:], "leading_edge_weight": 0.0,
                  "TE_thickness": TE}
        for j, xtr in enumerate(xtrs):
            a = nf.get_aero_from_kulfan_parameters(kulfan, alpha=alpha, Re=5e6, n_crit=1.0,
                                                   xtr_upper=xtr, xtr_lower=xtr,
                                                   model_size="xxxlarge")
            label = f"{name} | xtr {100 * xtr:g} %"
            style = {"color": color, "dash": dashes[j % len(dashes)]}
            fig.add_trace(go.Scatter(x=a["CD"] * 1e4, y=a["CL"], name=label, legendgroup=label,
                                     line=style), 1, 1)
            fig.add_trace(go.Scatter(x=alpha, y=a["CL"] / a["CD"], name=label, legendgroup=label,
                                     showlegend=False, line=style), 1, 2)
    fig.update_xaxes(title_text="CD x 10⁴", row=1, col=1)
    fig.update_yaxes(title_text="CL", row=1, col=1)
    fig.update_xaxes(title_text="alpha [deg]", row=1, col=2)
    fig.update_layout(height=650, title="Polaires 2D NeuralFoil (Re 5e6, n_crit 1) : couleur = "
                                        "section, trait = xtr")
    return fig


def cfd_figure(c6_path: Path, c9_path: Path, extra_avl: Path, xtrs):
    """
    The transition against Antoine's RANS (kOmegaSST, fully turbulent, wall functions), current
    section, 23 kn (Re 6.96e6): 2D (c6_cfoil_section.json: CFD points, XFOIL xtr 0.1 / 0.05) with
    NeuralFoil at each xtr; 3D C-foil at neutral attitude (c9/comparison.json: CFD meshes L1 / L2,
    AVL x XFOIL xtr 0.1 / 0.05, wave drag removed) with the other xtr values of `extra_avl`
    (same case, computed 2026-11-09 by the scratch script c9_xtr.py: {xtr: {"totals": ...}}).
    """
    import neuralfoil as nf  # noqa: PLC0415
    import plotly.graph_objects as go  # noqa: PLC0415
    from plotly.subplots import make_subplots  # noqa: PLC0415
    c6 = json.loads(Path(c6_path).read_text(encoding="utf-8"))
    c9 = json.loads(Path(c9_path).read_text(encoding="utf-8"))
    fig = make_subplots(rows=2, cols=2, vertical_spacing=0.14, subplot_titles=(
        "2D section actuelle, Re 6.96e6 : CD(alpha)", "2D : CL(alpha)",
        "3D C-foil attitude neutre, 23 kn : traînée Fx [N]", "3D : écart à la CFD L2 [%]"))
    points = pd.DataFrame(c6["points"])
    for ok, symbol in ((True, "circle"), (False, "circle-open")):
        p = points[points.converged == ok]
        name = "CFD kOmegaSST" + ("" if ok else " (non convergé)")
        fig.add_trace(go.Scatter(x=p.alpha_deg, y=p.CD * 1e4, mode="markers", name=name,
                                 marker={"color": "black", "symbol": symbol, "size": 9}), 1, 1)
        fig.add_trace(go.Scatter(x=p.alpha_deg, y=p.CL, mode="markers", showlegend=False,
                                 marker={"color": "black", "symbol": symbol, "size": 9}), 1, 2)
    for i, (name, polar) in enumerate(c6["xfoil"].items()):
        fig.add_trace(go.Scatter(x=polar["alpha"], y=np.array(polar["cd"]) * 1e4,
                                 name=f"XFOIL {name}", line={"color": COLORS[i + 2]}), 1, 1)
        fig.add_trace(go.Scatter(x=polar["alpha"], y=polar["cl"], showlegend=False,
                                 line={"color": COLORS[i + 2]}), 1, 2)
    # NeuralFoil on the Kulfan weights of the current section (the search space's reference)
    current =CFoilFamily.from_study_config(STUDY_CONFIG).reference_weights()
    kulfan = {"upper_weights": current[:8], "lower_weights": current[8:],
              "leading_edge_weight": 0.0, "TE_thickness": TE}
    alpha = np.arange(-6.0, 6.01, 0.5)
    for j, xtr in enumerate(xtrs):
        a = nf.get_aero_from_kulfan_parameters(kulfan, alpha=alpha, Re=float(c6["re"]), n_crit=1.0,
                                               xtr_upper=xtr, xtr_lower=xtr, model_size="xxxlarge")
        style = {"color": "#7f7f7f", "dash": ["dot", "dash", "solid", "dashdot"][j % 4]}
        fig.add_trace(go.Scatter(x=alpha, y=a["CD"] * 1e4, name=f"NeuralFoil xtr {100 * xtr:g} %",
                                 line=style), 1, 1)
        fig.add_trace(go.Scatter(x=alpha, y=a["CL"], showlegend=False, line=style), 1, 2)
    # 3D
    bars = {f"CFD {lv}": c9["cfd"][lv]["totals"] for lv in c9["cfd"]}
    avl = {f"AVL x XFOIL xtr {100 * float(n.split()[1]):g} %": v["totals"]
           for n, v in c9["avl"].items()}
    if extra_avl and Path(extra_avl).exists():
        for xtr, v in json.loads(Path(extra_avl).read_text(encoding="utf-8")).items():
            name = f"AVL x XFOIL xtr {100 * float(xtr):g} %"
            if "totals" in v:       # recomputed with today's bdfoil_v1: replaces the 20/10 value
                avl[name] = v["totals"]
    avl = dict(sorted(avl.items(), key=lambda kv: -float(kv[0].split()[-2])))
    bars.update(avl)
    names = list(bars)
    fig.add_trace(go.Bar(x=names, y=[-bars[n]["Fx"] for n in names], showlegend=False,
                         marker={"color": ["black"] * len(c9["cfd"]) + [COLORS[2]] * len(avl)},
                         text=[f"{-bars[n]['Fx']:.0f}" for n in names], textposition="outside"),
                  2, 1)
    ref = c9["cfd"]["L2"]["totals"]
    for k, comp in enumerate(("Fx", "Fy", "Fz", "Mx")):
        fig.add_trace(go.Bar(x=list(avl), y=[100 * (avl[n][comp] / ref[comp] - 1) for n in avl],
                             name=comp, marker={"color": COLORS[k]}), 2, 2)
    fig.update_xaxes(title_text="alpha [deg]", row=1, col=1)
    fig.update_xaxes(title_text="alpha [deg]", row=1, col=2)
    fig.update_yaxes(title_text="CD x 10⁴", row=1, col=1)
    fig.update_layout(height=1000, barmode="group",
                      title="Transition forcée et CFD (RANS kOmegaSST pleinement turbulent, "
                            "lois de paroi ; cas C6 / C9 d'Antoine)")
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--polar-xtr", type=float, nargs="+", default=[0.02, 0.05, 0.10])
    parser.add_argument("--name", default="xtr_study")
    cfd_dir = VPP_DIR.parents[1] / "bdfoil_v1" / "validation" / "openfoam"
    parser.add_argument("--cfd-2d", type=Path, default=cfd_dir / "c6_cfoil_section.json")
    parser.add_argument("--cfd-3d", type=Path, default=cfd_dir / "c9" / "comparison.json")
    parser.add_argument("--avl-xtr", type=Path, default=None,
                        help="AVL x XFOIL of the C9 case at other xtr ({xtr: {totals}})")
    args = parser.parse_args()
    sections, folders, _ = load(args.runs)
    run = RunDir.create(args.name, {"runs": args.runs, "polar_xtr": args.polar_xtr})
    figures = {}
    fig, table = sections_figure(sections)
    figures["sections"] = fig
    table.to_csv(run.path("results", "sections.csv"), index=False)
    cross = cross_table(folders)
    if not cross.empty:
        cross.to_csv(run.path("results", "cross.csv"))
        figures["cross"] = cross_figure(cross)
    figures["gains"] = gains_figure(folders)
    figures["polars"] = polars_figure(sections, args.polar_xtr)
    if args.cfd_2d.exists() and args.cfd_3d.exists():
        figures["cfd"] = cfd_figure(args.cfd_2d, args.cfd_3d, args.avl_xtr, args.polar_xtr)
    for name, figure in figures.items():
        figure.write_html(run.path("figures", f"{name}.html"), include_plotlyjs="cdn")
    links = "".join(f'<li><a href="{n}.html">{n}</a></li>' for n in figures)
    html_table = table.to_html(index=False, float_format=lambda v: f"{v:.3f}")
    cross_html = cross.to_html(float_format=lambda v: f"{v:.4f}") if not cross.empty else ""
    page = (f"<html><head><meta charset='utf-8'><title>Etude xtr</title></head><body "
            f"style='font-family:sans-serif;max-width:1100px;margin:auto'><h1>Transition forcée : "
            f"sections optimisées</h1><ul>{links}</ul><h2>Géométrie</h2>{html_table}"
            f"<h2>Matrice croisée [kt]</h2>{cross_html}</body></html>")
    run.path("figures", "index.html").write_text(page, encoding="utf-8")
    run.log(f"figures: {run.path('figures', 'index.html')}")


if __name__ == "__main__":
    main()
