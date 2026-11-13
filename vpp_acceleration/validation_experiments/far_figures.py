"""
Figures (plotly, HTML, French) of far_experiments.py: buffered cross-validation, certificate, summary
along the direction that matters, operating profile, cross judgement. Called by figures.py
(far_figures(results, figures) -> [(file, title)]); a missing result gives no figure.
"""
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#7f7f7f"]
LEVEL_NAMES = {1: "niveau 1 (NPLLT × NeuralFoil)", 2: "niveau 2 (AVL × XFOIL)", 3: "niveau 3 (CFD 3D)"}


def _load(results: Path, name: str):
    path = results / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _write(figure: go.Figure, figures: Path, name: str, title: str, height: int = 520) -> tuple:
    figure.update_layout(title=title, template="plotly_white", height=height,
                         margin={"l": 70, "r": 30, "t": 80, "b": 60})
    figures.mkdir(parents=True, exist_ok=True)
    figure.write_html(figures / f"{name}.html", include_plotlyjs="cdn")
    return f"{name}.html", title


def buffered_cv(data, figures: Path) -> list:
    levels = sorted({e["level"] for e in data["entries"]})
    figure = make_subplots(rows=1, cols=len(levels), shared_yaxes=True,
                           subplot_titles=[LEVEL_NAMES.get(l, f"niveau {l}") for l in levels])
    for col, level in enumerate(levels, start=1):
        for k, output in enumerate(dict.fromkeys(e["output"] for e in data["entries"])):
            part = [e for e in data["entries"] if e["level"] == level and e["output"] == output]
            figure.add_scatter(x=[e["radius"] for e in part], y=[e["z_rms"] for e in part],
                               mode="lines+markers", name=output, legendgroup=output,
                               showlegend=col == 1, line={"color": COLORS[k % len(COLORS)], "width": 1},
                               customdata=[[e["median_left_out"], e["relative_rmse"]] for e in part],
                               hovertemplate=output + "<br>rayon %{x}<br>écart réduit %{y:.2f}<br>"
                               "points retirés (médiane) %{customdata[0]:.0f}<br>erreur relative "
                               "%{customdata[1]:.3f}<extra></extra>", row=1, col=col)
        kappa = data["kappa_far"].get(str(level), {})
        figure.add_scatter(x=[float(r) for r in kappa], y=list(kappa.values()), mode="lines+markers",
                           name="6 efforts ensemble", legendgroup="all", showlegend=col == 1,
                           line={"color": "black", "width": 3}, row=1, col=col)
        figure.add_hline(y=1.0, line_dash="dash", line_color="gray", row=1, col=col)
        figure.update_xaxes(title_text="rayon de la zone d'exclusion (scores)", row=1, col=col)
    figure.update_yaxes(title_text="écart réduit (1 = incertitude annoncée juste)", row=1, col=1)
    return [_write(figure, figures, "validation_croisee_exclusion",
                   "Le modèle est-il honnête loin des données ? Validation croisée avec zone d'exclusion")]


def certificate(data, figures: Path, label: str) -> list:
    pool = data["pool"]
    gain, sigma, distance = (np.array(pool[key]) for key in ("gain", "sigma", "distance"))
    figure = go.Figure()
    figure.add_scatter(x=distance, y=gain, mode="markers", name="sections candidates",
                       marker={"size": 5, "color": "#7f7f7f"},
                       error_y={"type": "data", "array": 2.0 * sigma, "width": 0, "thickness": 0.6},
                       hovertemplate="distance %{x:.2f}<br>gain %{y:.4f} nd<extra></extra>")
    for k, kappa in enumerate(data["kappas"]):
        part = [m for m in data["maximisers"] if m["kappa"] == kappa]
        figure.add_scatter(x=[m["distance"] for m in part], y=[m["bound_kt"] for m in part],
                           mode="markers", name=f"maximum de la borne haute, κ = {kappa:g}",
                           marker={"size": 11, "symbol": "star", "color": COLORS[k]},
                           customdata=[[m["gain_kt"], m["sigma_kt"]] for m in part],
                           hovertemplate="borne %{y:.4f} nd<br>gain %{customdata[0]:.4f}, écart-type "
                           "%{customdata[1]:.4f}<extra></extra>")
    figure.add_hline(y=0.0, line_dash="dash", line_color="black")
    figure.add_hline(y=0.02, line_dash="dot", line_color="red", annotation_text="critère : +0,02 nd")
    figure.update_layout(xaxis_title="distance à la meilleure section (scores)",
                         yaxis_title="gain sur la meilleure section (nd) ; barres : ± 2 écarts-types")
    bounds = " · ".join(f"κ = {float(k):g} : {v:+.3f} nd".replace(".", ",")
                        for k, v in data["bound_by_kappa"].items())
    return [_write(figure, figures, f"certificat_{label}",
                   f"Certificat ({LEVEL_NAMES.get(data['level'], data['level'])}) : borne haute "
                   f"maximale sur le domaine<br><sup>{bounds}</sup>", 560)]


def summary(data, figures: Path) -> list:
    key = "active_projection" if "active_projection" in data else "projection"
    x, gain = np.array(data[key]), np.array(data["gain"])
    level = np.array(data["highest_level"])
    figure = go.Figure()
    for value in sorted(set(level)):
        rows = level == value
        figure.add_scatter(x=x[rows], y=gain[rows], mode="markers", marker={"size": 5},
                           name=f"sections calculées jusqu'au {LEVEL_NAMES.get(int(value), value)}",
                           hovertemplate="position %{x:.2f}<br>gain %{y:.4f} nd<extra></extra>")
    if key == "projection":
        figure.add_scatter(x=[v["projection"] for v in data["named"].values()],
                           y=[v["gain"] for v in data["named"].values()], mode="markers+text",
                           text=list(data["named"]), textposition="top center", name="sections du run",
                           marker={"size": 11, "symbol": "star", "color": "black"})
    figure.update_layout(
        xaxis_title="position le long de la direction " + ("active" if key != "projection"
                                                           else "du gradient") + " (scores)",
        yaxis_title="vitesse prédite − meilleure section (nd, premier ordre)")
    return [_write(figure, figures, "direction_active",
                   "Toutes les sections calculées, le long de la direction qui porte la vitesse")]


def profile(data, figures: Path) -> list:
    out = []
    modes = data["modes"]
    figure = go.Figure()
    for k, (mode, entry) in enumerate(modes.items()):
        figure.add_scatter(x=data["factors"], y=entry["gain_when_weight_scaled"], mode="lines",
                           name=f"{mode} ({100 * entry['weight']:.0f} %)",
                           line={"color": COLORS[k % len(COLORS)]})
    figure.add_vline(x=1.0, line_dash="dash", line_color="gray", annotation_text="profil actuel")
    figure.update_layout(xaxis_title="poids du mode multiplié par (les autres inchangés, total renormalisé)",
                         yaxis_title="gain moyen de la meilleure section (nd)")
    out.append(_write(figure, figures, "profil_sensibilite",
                      "Le gain tient-il si le profil d'usage change ? Gain en fonction du poids de chaque mode"))
    gain = np.array([np.nan if v is None else v for v in data["gain_by_condition"]], dtype=float)
    figure = go.Figure(go.Scatter(
        x=data["twa"], y=data["tws"], mode="markers",
        marker={"size": 8 + 500 * np.array(data["weights"]), "color": gain, "colorscale": "RdBu_r",
                "cmid": 0.0, "colorbar": {"title": "gain (nd)"}, "line": {"width": 0.5}},
        customdata=np.column_stack([gain, 100 * np.array(data["weights"])]),
        hovertemplate="TWA %{x:.0f}°, TWS %{y:.1f} nd<br>gain %{customdata[0]:.3f} nd<br>poids "
                      "%{customdata[1]:.1f} %<extra></extra>"))
    figure.update_layout(xaxis_title="TWA (degrés)", yaxis_title="TWS (nd)")
    out.append(_write(figure, figures, "profil_gain_par_condition",
                      f"Gain par condition (section de départ : {data['start']}) ; taille = poids"))
    return out


def cross(data, figures: Path) -> list:
    if not data.get("judgements"):
        return []
    figure = make_subplots(rows=1, cols=len(data["judgements"]), shared_yaxes=True, subplot_titles=[
        f"juge : {j['judge'][11:]}<br>{LEVEL_NAMES.get(j['level'], j['level'])}"
        for j in data["judgements"]])
    for col, judgement in enumerate(data["judgements"], start=1):
        colors = ["#2166ac" if n.startswith("own") else "#b2182b" for n in judgement["names"]]
        figure.add_bar(x=judgement["names"], y=judgement["gain_kt"], marker_color=colors,
                       error_y={"type": "data", "array": [2 * s for s in judgement["sigma_kt"]]},
                       showlegend=False, row=1, col=col)
    figure.update_yaxes(title_text="vitesse − sa propre meilleure section (nd), ± 2 écarts-types",
                        row=1, col=1)
    return [_write(figure, figures, "jugement_croise",
                   "Jugement croisé : chaque modèle évalue les sections de l'autre run (premier ordre)",
                   560)]


def far_figures(results: Path, figures: Path) -> list:
    entries = []
    for name, builder in (("buffered_cv", buffered_cv), ("summary", summary), ("profile", profile),
                          ("cross", cross)):
        data = _load(results, name)
        if data is not None:
            try:
                entries += builder(data, figures)
            except Exception as error:  # noqa: BLE001  (a figure must not stop the report)
                print(f"figure {name}: {error}", flush=True)
    for path in sorted(results.glob("certificate_level*.json")):
        try:
            entries += certificate(json.loads(path.read_text(encoding="utf-8")), figures,
                                   path.stem.replace("certificate_", ""))
        except Exception as error:  # noqa: BLE001
            print(f"figure {path.name}: {error}", flush=True)
    return entries
