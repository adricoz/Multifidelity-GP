"""
Figures of the robustness checks of scripts/robustness_checks.py, written to the figures/ folder
of each check (from its results, nothing is solved again):
  1. posterior.html: mean speed of each section in every draw of the GP (lines join the sections
     of one draw), and the gap best - other per draw, at each scale of the variance;
  2. tolerance.html: speed change of the perturbed best sections against Ixx(current) /
     Ixx(section), with the Ixx constraint, and the surface displacements of the perturbations.
Run: ..\\.venv-mds\\Scripts\\python scripts\\robustness_figures.py runs\\<posterior folder> runs\\<tolerance folder>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

VPP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VPP_DIR.parent / "benchmarks"))
sys.path.insert(0, str(VPP_DIR))
sys.path.insert(0, str(VPP_DIR / "scripts"))

LABELS = {"best": "best", "region_1": "région 1", "closest_to_current": "départ"}


def label(name: str) -> str:
    return LABELS.get(name, f"best du run {name[5:20]}" if name.startswith("best_") else name)


def draw_means(table: pd.DataFrame) -> pd.DataFrame:
    """Weighted mean speed per (scale, draw, section) over the conditions solved for all the
    sections of the draw (rows: (scale, draw), columns: sections)."""
    table = table.assign(key=table.twa.astype(str) + "/" + table.tws.astype(str))
    rows = {}
    for (scale, draw), g in table.groupby(["scale", "draw"]):
        speeds = g.pivot(index="section", columns="key", values="speed_kt")
        solved = g.pivot(index="section", columns="key", values="success").astype(bool)
        weights = g.drop_duplicates("key").set_index("key")["weight"].loc[speeds.columns]
        common = solved.all(axis=0)
        w = weights[common]
        rows[(scale, draw)] = speeds.loc[:, common] @ w / w.sum()
    return pd.DataFrame(rows).T


def posterior_figure(root: Path) -> Path:
    import plotly.graph_objects as go  # noqa: PLC0415
    from figure_style import SERIES, hline, rgba, save, style  # noqa: PLC0415
    from plotly.subplots import make_subplots  # noqa: PLC0415

    summary = json.loads((root / "results" / "summary.json").read_text(encoding="utf-8"))
    means = draw_means(pd.read_csv(root / "results" / "draws.csv"))
    names = [n for n in summary["nominal"] if n != "closest_to_current"]   # the start is 0.26 kt below
    others = [n for n in names if n != "best"]
    scales = sorted(means.index.get_level_values(0).unique())
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.1, column_widths=[0.55, 0.45],
                        subplot_titles=("Vitesse moyenne de chaque section, un trait par tirage",
                                        "Écart best − autre section dans le même tirage"))
    x = [label(n) for n in names]
    for k, scale in enumerate(scales):
        color = SERIES[k]
        part = means.loc[scale]
        for i, (draw, row) in enumerate(part.iterrows()):
            fig.add_trace(go.Scatter(
                x=x, y=row[names], mode="lines+markers", line={"color": rgba(color, 0.35), "width": 1},
                marker={"color": color, "size": 5}, legendgroup=str(scale),
                name=f"tirages à {scale:g} σ", showlegend=(i == 0),
                hovertemplate=f"tirage {draw}, {scale:g} σ<br>%{{x}} : %{{y:.4f}} nd<extra></extra>"),
                row=1, col=1)
        for other in others:
            gap = part["best"] - part[other]
            fig.add_trace(go.Box(
                x=[label(other)] * len(gap), y=gap, boxpoints="all", jitter=0.4, pointpos=0,
                marker={"color": color, "size": 5}, line={"color": color}, fillcolor=rgba(color, 0.1),
                offsetgroup=str(scale), legendgroup=str(scale), showlegend=False,
                name=f"{scale:g} σ", hovertemplate="%{y:.4f} nd<extra></extra>"), row=1, col=2)
    fig.add_trace(go.Scatter(x=x, y=[summary["nominal"][n] for n in names], mode="markers",
                             marker={"color": SERIES[3], "size": 11, "symbol": "diamond"},
                             name="nominal (moyenne du GP)"), row=1, col=1)
    fig.update_layout(boxmode="group")
    hline(fig, 0.0, "best = autre section", row=1, col=2)
    fig.update_yaxes(title_text="vitesse moyenne [nd]", row=1, col=1)
    fig.update_yaxes(title_text="best − section [nd]", rangemode="tozero", row=1, col=2)
    lines = [f"{float(s):g} σ : P(best) = {v['p_best']['best']:.2f}, regret {v['regret_mean_kt']:.4f} nd"
             for s, v in summary["scales"].items()]
    style(fig, "Tirages a posteriori du surrogate (§6.3)",
          f"{root.name} ; {len(means.loc[scales[0]])} fonctions GP cohérentes par échelle ; "
          f"{' ; '.join(lines)} ; départ hors figure ("
          f"{summary['nominal']['closest_to_current']:.3f} nd)", height=560)
    return save(fig, root / "figures", "posterior")


def tolerance_figure(root: Path) -> Path:
    import plotly.graph_objects as go  # noqa: PLC0415
    from figure_style import SERIES, rgba, save, style  # noqa: PLC0415
    from plotly.subplots import make_subplots  # noqa: PLC0415
    from robustness_checks import CHORD_M, load_run, perturbations  # noqa: PLC0415
    from profile_analysis import stiffness_ratios  # noqa: PLC0415
    from src.section_pca import surface_matrix  # noqa: PLC0415

    setup = json.loads((root / "config.json").read_text(encoding="utf-8"))
    summary = json.loads((root / "results" / "summary.json").read_text(encoding="utf-8"))
    config, pca, _, _, sections = load_run(VPP_DIR / setup["of_run"])
    best = pca.decode(sections["best"][None, :])[0]
    cases = [(0.0, best)] + perturbations(best, setup["amplitudes_chord"],
                                          setup["n_per_amplitude"], setup["seed"], pca.te_thickness)
    ratios = stiffness_ratios(config, pca, np.array([pca.encode(w[None, :])[0] for _, w in cases]))
    stations = np.linspace(0.0, 1.0, 201)
    matrix, _ = surface_matrix(stations, len(best) // 2, pca.te_thickness)

    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.1, subplot_titles=(
        "Écart de vitesse contre la souplesse de la section",
        "Défauts de surface appliqués (extrados en trait plein, intrados en tirets)"))
    changes = np.concatenate([summary["amplitudes"][str(a)] for a in setup["amplitudes_chord"]])
    x = ratios[1:] - ratios[0]
    slope, intercept = np.polyfit(x, changes, 1)
    line = np.linspace(min(ratios), max(ratios), 2)
    fig.add_trace(go.Scatter(x=line, y=slope * (line - ratios[0]) + intercept, mode="lines",
                             line={"color": SERIES[3], "width": 1, "dash": "dash"},
                             name=f"droite : {slope:.3f} nd par unité "
                                  f"(corrélation {np.corrcoef(x, changes)[0, 1]:.2f})"),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=[ratios[0]], y=[0.0], mode="markers", name="best (référence)",
                             marker={"color": SERIES[3], "size": 12, "symbol": "diamond"}),
                  row=1, col=1)
    start = 1
    for k, amplitude in enumerate(setup["amplitudes_chord"]):
        n = setup["n_per_amplitude"]
        color = SERIES[k]
        name = f"défaut max {amplitude * CHORD_M * 1e3:.2f} mm"
        idx = list(range(start, start + n))
        fig.add_trace(go.Scatter(
            x=ratios[idx], y=changes[[i - 1 for i in idx]], mode="markers+text",
            text=[str(i) for i in idx], textposition="top center", legendgroup=name, name=name,
            marker={"color": color, "size": 10},
            hovertemplate="cas %{text}<br>Ixx(actuelle)/Ixx = %{x:.3f}<br>Δv = %{y:.4f} nd"
                          "<extra></extra>"), row=1, col=1)
        for i in idx:
            dy = (matrix @ (cases[i][1] - best)) * CHORD_M * 1e3
            for side, dash in ((slice(0, len(stations)), "solid"),
                               (slice(len(stations), None), "dash")):
                fig.add_trace(go.Scatter(
                    x=stations, y=dy[side], mode="lines", legendgroup=name, showlegend=False,
                    line={"color": color, "width": 1, "dash": dash},
                    hovertemplate=f"cas {i}<br>x/c = %{{x:.2f}}<br>Δy = %{{y:.2f}} mm"
                                  "<extra></extra>"), row=1, col=2)
        start += n
    # shapes after the traces: plotly drops shapes of subplots that are still empty
    fig.add_vrect(x0=1.0, x1=max(ratios) + 0.005, fillcolor=rgba(SERIES[7], 0.10), line_width=0,
                  row=1, col=1)
    fig.add_vline(x=1.0, line={"color": SERIES[7], "width": 1, "dash": "dot"}, row=1, col=1)
    fig.add_annotation(x=1.0, xref="x", xanchor="left", y=0.0, yref="y domain", yanchor="bottom",
                       showarrow=False, text=" Ixx < Ixx(actuelle) : hors contrainte",
                       font={"size": 11})
    fig.update_xaxes(title_text="Ixx(actuelle) / Ixx(section)   (> 1 : plus souple)", row=1, col=1)
    fig.update_yaxes(title_text="Δ vitesse moyenne, AVL × XFOIL [nd]", row=1, col=1)
    fig.update_xaxes(title_text="x / c", row=1, col=2)
    fig.update_yaxes(title_text="Δy [mm]", row=1, col=2)
    style(fig, "Tolérances de fabrication de la meilleure section",
          f"{root.name} ; référence {summary['reference_kt']:.4f} nd ; vitesse sur le surrogate "
          f"corrigée au premier ordre par AVL × XFOIL (poids non projetés) ; corde "
          f"{CHORD_M:.2f} m", height=560)
    return save(fig, root / "figures", "tolerance")


def main() -> None:
    from real_run_figures import write_index, write_runs_index  # noqa: PLC0415
    for folder in sys.argv[1:]:
        root = (VPP_DIR / folder).resolve()
        check = json.loads((root / "config.json").read_text(encoding="utf-8"))["check"]
        path = {"posterior": posterior_figure, "tolerance": tolerance_figure}[check](root)
        write_index(root)
        print(path)
    write_runs_index()


if __name__ == "__main__":
    main()
