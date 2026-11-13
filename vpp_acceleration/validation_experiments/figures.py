"""
Figures (plotly, HTML) of the validation experiments: reads results/*.json written by
experiments.py, writes figures/*.html and figures/index.html. Texts in French (reports).

Run:
    ..\\.venv-mds\\Scripts\\python validation_experiments\\figures.py
"""
import argparse
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

HERE = Path(__file__).resolve().parent
RESULTS, FIGURES = HERE / "results", HERE / "figures"
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#7f7f7f"]
LEVEL_NAMES = {1: "niveau 1 (NPLLT × NeuralFoil)", 2: "niveau 2 (AVL × XFOIL)",
               3: "niveau 3 (CFD 3D)"}


def load(name: str):
    path = RESULTS / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def write(figure: go.Figure, name: str, title: str, height: int = 520) -> str:
    figure.update_layout(title=title, template="plotly_white", height=height,
                         margin={"l": 70, "r": 30, "t": 80, "b": 60})
    FIGURES.mkdir(exist_ok=True)
    figure.write_html(FIGURES / f"{name}.html", include_plotlyjs="cdn")
    return f"{name}.html"


# -------------------------------------------------------------------------------------------------
def calibration_figures(data) -> list:
    out = []
    figure = go.Figure()
    outputs = list(dict.fromkeys(e["output"] for e in data["loo"]))
    for level in sorted({e["level"] for e in data["loo"]}):
        rows = {e["output"]: e for e in data["loo"] if e["level"] == level}
        figure.add_bar(x=outputs, y=[rows[o]["z_rms"] for o in outputs],
                       name=f"{LEVEL_NAMES.get(level, level)}, n = {rows[outputs[0]]['n']}",
                       marker_color=COLORS[level - 1],
                       customdata=[[rows[o]["q2"], rows[o]["coverage_95"]] for o in outputs],
                       hovertemplate="%{x}<br>écart réduit RMS %{y:.2f}<br>Q² %{customdata[0]:.3f}"
                                     "<br>couverture 95 %% : %{customdata[1]:.2f}<extra></extra>")
    figure.add_hline(y=1.0, line_dash="dash", line_color="black",
                     annotation_text="calibré (1)", annotation_position="top left")
    figure.update_layout(barmode="group", yaxis_title="écart réduit RMS (validation croisée)",
                         xaxis_title="sortie du modèle")
    out.append((write(figure, "calibration_loo",
                      "Calibration entre les données : validation croisée par sortie et par niveau"
                      "<br><sup>au-dessus de 1 : le modèle est trop sûr de lui ; en dessous : "
                      "trop prudent</sup>"),
                "Calibration entre les données (validation croisée)"))

    figure = make_subplots(rows=1, cols=2, subplot_titles=(
        "Écart réduit de la vitesse, par état VPP", "Erreur de vitesse et écart-type annoncé"))
    names, errors, sigmas, texts = [], [], [], []
    for index, entry in enumerate(data["external"]):
        label = f"{entry['section']} · {LEVEL_NAMES.get(entry['level'], entry['level'])}"
        figure.add_trace(go.Box(y=entry["speed_z"], name=label, boxpoints="all", jitter=0.4,
                                marker_color=COLORS[index % len(COLORS)], showlegend=False),
                         row=1, col=1)
        names.append(label)
        errors.append(entry["speed_error_rms_kt"])
        sigmas.append(entry["speed_sigma_rms_kt"])
        texts.append(f"×{entry['speed']['z_rms']:.1f}")
    for y in (-2, 2):
        figure.add_hline(y=y, line_dash="dash", line_color="black", row=1, col=1)
    figure.add_trace(go.Bar(x=names, y=errors, name="erreur RMS contre le code [nd]",
                            marker_color="#d62728", text=texts, textposition="outside"),
                     row=1, col=2)
    figure.add_trace(go.Bar(x=names, y=sigmas, name="écart-type annoncé RMS [nd]",
                            marker_color="#1f77b4"), row=1, col=2)
    figure.update_yaxes(title_text="erreur / écart-type annoncé", row=1, col=1)
    figure.update_yaxes(title_text="nœuds", row=1, col=2)
    figure.update_xaxes(tickangle=30)
    out.append((write(figure, "calibration_externe",
                      "Calibration près de l'optimum : le modèle contre les codes aux états VPP "
                      "(points jamais vus)<br><sup>à gauche, un modèle calibré reste entre −2 et 2 ; "
                      "à droite, le facteur κ = erreur / écart-type annoncé</sup>", 620),
                "Calibration près de l'optimum (codes aux états VPP)"))
    return out


def gradient_figures(data) -> list:
    k = len(data["gradient"])
    modes = [f"mode {j + 1}" for j in range(k)]
    figure = make_subplots(rows=1, cols=3, subplot_titles=(
        "Gradient de la vitesse à la meilleure section",
        "Gain d'une montée sous contraintes sur le modèle", "Courbure (valeurs propres)"))
    figure.add_trace(go.Bar(x=modes, y=data["gradient"], name="gradient complet",
                            error_y={"type": "data",
                                     "array": [2 * s for s in data["gradient_sigma"]]},
                            marker_color="#7f7f7f"), row=1, col=1)
    figure.add_trace(go.Bar(x=modes, y=data["tangent_gradient"],
                            name="partie le long des contraintes actives",
                            marker_color="#d62728"), row=1, col=1)
    ascent = data["ascent"]
    figure.add_trace(go.Scatter(
        x=[a["radius"] for a in ascent], y=[a["gain_kt"] for a in ascent], mode="lines+markers",
        error_y={"type": "data", "array": [2 * a["gain_sigma_kt"] for a in ascent]},
        name="gain prédit ± 2σ", marker_color="#1f77b4"), row=1, col=2)
    figure.add_trace(go.Bar(x=[f"λ{j + 1}" for j in range(k)],
                            y=sorted(data["hessian_eigenvalues"]), name="courbure",
                            marker_color="#2ca02c"), row=1, col=3)
    figure.update_yaxes(title_text="nd par unité de score", row=1, col=1)
    figure.update_xaxes(title_text="rayon autorisé autour de la meilleure section (scores)",
                        row=1, col=2)
    figure.update_yaxes(title_text="gain de vitesse [nd]", row=1, col=2)
    figure.update_yaxes(title_text="nd par unité de score²", row=1, col=3)
    test = data["kkt"]["1"]
    return [(write(figure, "gradient_kkt",
                   "Reste-t-il de la pente ? Gradient, contraintes actives et montée résiduelle"
                   f"<br><sup>test KKT : χ² = {test['statistic']:.1f} pour {test['dof']} degrés de "
                   f"liberté, p = {test['p_value']:.3f} (p = {data['kkt']['2']['p_value']:.3f} avec "
                   "un écart-type doublé) · barres d'erreur : 2 σ du modèle</sup>", 560),
             "Gradient, conditions KKT et montée résiduelle")]


def slices_figures(data) -> list:
    t = np.array(data["t"])
    n = len(data["modes"])
    columns = 5
    figure = make_subplots(rows=int(np.ceil(n / columns)), cols=columns, shared_yaxes=True,
                           subplot_titles=[f"mode {m['mode']}" for m in data["modes"]])
    for index, mode in enumerate(data["modes"]):
        row, col = index // columns + 1, index % columns + 1
        gain, sigma = np.array(mode["gain"]), np.array(mode["sigma"])
        ok = np.array(mode["admissible"])
        figure.add_trace(go.Scatter(x=np.r_[t, t[::-1]],
                                    y=np.r_[gain + 2 * sigma, (gain - 2 * sigma)[::-1]],
                                    fill="toself", fillcolor="rgba(31,119,180,0.18)",
                                    line={"width": 0}, hoverinfo="skip", showlegend=False),
                         row=row, col=col)
        figure.add_trace(go.Scatter(x=t, y=gain, mode="lines", line={"color": "#7f7f7f"},
                                    name="modèle, hors domaine", showlegend=index == 0),
                         row=row, col=col)
        figure.add_trace(go.Scatter(x=t, y=np.where(ok, gain, np.nan), mode="lines",
                                    line={"color": "#1f77b4", "width": 3},
                                    name="modèle, section admissible", showlegend=index == 0),
                         row=row, col=col)
        figure.add_trace(go.Scatter(x=t, y=mode["gain_level2"], mode="lines",
                                    line={"color": "#ff7f0e", "dash": "dot"},
                                    name="niveau 2 (AVL × XFOIL)", showlegend=index == 0),
                         row=row, col=col)
        figure.add_vline(x=0, line_color="black", line_width=1, row=row, col=col)
    figure.update_yaxes(title_text="écart de vitesse [nd]", col=1)
    figure.update_xaxes(title_text="déplacement (scores)", row=int(np.ceil(n / columns)))
    return [(write(figure, "coupes",
                   "Coupes du modèle à travers la meilleure section, le long de chaque mode PCA"
                   "<br><sup>0 = meilleure section ; bande : ± 2 σ de l'écart ; trait épais : "
                   "sections admissibles ; états VPP figés (premier ordre)</sup>", 700),
             "Coupes 1D à travers la meilleure section")]


def regret_figures(data) -> list:
    figure = make_subplots(rows=1, cols=2, column_widths=[0.6, 0.4], subplot_titles=(
        "Sections candidates : écart de vitesse prédit contre la meilleure",
        "Probabilité qu'une candidate batte la meilleure de plus de ε"))
    groups = np.array(data["groups"])
    gain, sigma = np.array(data["gain"]), np.array(data["sigma_diff"])
    distance = np.array(data["distance"])
    labels = {"global": "partout dans le domaine", "local_0.1": "autour, rayon 0,1",
              "local_0.3": "autour, rayon 0,3", "local_1.0": "autour, rayon 1"}
    for index, name in enumerate(dict.fromkeys(groups)):
        mask = groups == name
        named = name not in labels
        figure.add_trace(go.Scatter(
            x=distance[mask], y=gain[mask], mode="markers", name=labels.get(name, name),
            error_y={"type": "data", "array": 2 * sigma[mask], "thickness": 0.6, "width": 0},
            marker={"size": 11 if named else 5, "symbol": "star" if named else "circle",
                    "color": COLORS[index % len(COLORS)]}), row=1, col=1)
    figure.add_hline(y=0, line_color="black", line_width=1, row=1, col=1)
    figure.update_xaxes(title_text="distance à la meilleure section (scores)", row=1, col=1)
    figure.update_yaxes(title_text="écart de vitesse prédit ± 2σ [nd]", row=1, col=1)
    for index, (kappa, scale) in enumerate(data["scales"].items()):
        eps = list(scale["p_better_by"])
        figure.add_trace(go.Bar(x=[f"{1000 * float(e):.0f}" for e in eps],
                                y=[scale["p_better_by"][e] for e in eps],
                                name=f"écart-type × {float(kappa):.0f} "
                                     f"(regret {1000 * scale['expected_regret_kt']:.1f} mnd)",
                                marker_color=["#1f77b4", "#ff7f0e", "#d62728"][index]),
                         row=1, col=2)
    figure.update_xaxes(title_text="ε [millièmes de nœud]", row=1, col=2)
    figure.update_yaxes(title_text="probabilité", range=[0, 1], row=1, col=2)
    return [(write(figure, "regret",
                   f"Une meilleure section existe-t-elle ? {len(gain)} candidates, "
                   f"{data['n_draws']} tirages de la loi jointe du modèle"
                   "<br><sup>au premier ordre autour des états VPP de la meilleure section ; "
                   "mnd = millième de nœud</sup>", 600),
             "Probabilité qu'une meilleure section existe, regret")]


def dimension_figures(data, gradient) -> list:
    figure = make_subplots(rows=1, cols=2, subplot_titles=(
        "Part cumulée de la variation de vitesse par direction active",
        "Directions actives (domaine entier) dans la base des modes PCA"))
    labels = {"global": "sections de tout le domaine", "local_1.0": "sections autour de l'optimum"}
    for index, (name, entry) in enumerate(data.items()):
        values = np.array(entry["eigenvalues"])
        figure.add_trace(go.Scatter(x=np.arange(1, len(values) + 1),
                                    y=np.cumsum(values) / values.sum(), mode="lines+markers",
                                    name=labels.get(name, name),
                                    marker_color=COLORS[index]), row=1, col=1)
    vectors = np.array(data["global"]["eigenvectors"])
    figure.add_trace(go.Heatmap(z=vectors[:5], x=[f"mode {j + 1}" for j in range(vectors.shape[1])],
                                y=[f"direction {i + 1}" for i in range(5)], colorscale="RdBu",
                                zmid=0, colorbar={"title": "poids"}), row=1, col=2)
    figure.update_xaxes(title_text="nombre de directions gardées", row=1, col=1)
    figure.update_yaxes(title_text="part cumulée", range=[0, 1.02], row=1, col=1)
    return [(write(figure, "dimension",
                   "Le problème est-il plus simple que ses 10 dimensions ? Sous-espaces actifs"
                   "<br><sup>valeurs propres de E[∇J ∇Jᵀ] sur des sections admissibles ; "
                   "états VPP figés à ceux de la meilleure section</sup>", 540),
             "Dimension effective (sous-espaces actifs)")]


def levels_figures(data) -> list:
    rows = data["levels"]
    names = [LEVEL_NAMES.get(r["level"], str(r["level"])) for r in rows]
    figure = make_subplots(rows=1, cols=3, subplot_titles=(
        "Accord des pentes avec le niveau le plus haut (cosinus)",
        "Déplacement de l'optimum (scores)", "Vitesse perdue en optimisant ce niveau [nd]"))
    figure.add_trace(go.Bar(x=names, y=[r["cosine_with_top"] for r in rows],
                            marker_color="#1f77b4", showlegend=False), row=1, col=1)
    figure.add_trace(go.Bar(x=names, y=[r["optimum_shift"] for r in rows],
                            marker_color="#ff7f0e", showlegend=False), row=1, col=2)
    figure.add_trace(go.Bar(x=names, y=[r["loss_at_top_level_kt"] for r in rows],
                            error_y={"type": "data",
                                     "array": [2 * r["loss_sigma_kt"] for r in rows]},
                            marker_color="#d62728", showlegend=False), row=1, col=3)
    return [(write(figure, "niveaux",
                   "Un niveau moins cher trouve-t-il le même optimum ? "
                   "(jugé par le niveau le plus haut du modèle)"
                   f"<br><sup>montée sous contraintes sur chaque niveau, dans un rayon de "
                   f"{data['radius']} autour de la meilleure section ; barres : 2 σ</sup>", 520),
             "Déplacement de l'optimum entre niveaux")]


def main():
    global RESULTS, FIGURES  # noqa: PLW0603  (one results / figures folder per analysed run)
    parser = argparse.ArgumentParser(description="Figures of the validation experiments")
    parser.add_argument("--results", default="results", help="relative to validation_experiments")
    parser.add_argument("--figures", default="figures")
    args = parser.parse_args()
    RESULTS, FIGURES = HERE / args.results, HERE / args.figures
    FIGURES.mkdir(parents=True, exist_ok=True)
    run = load("run") or {}
    entries = []
    builders = [("calibration", calibration_figures), ("gradient", gradient_figures),
                ("slices", slices_figures), ("regret", regret_figures),
                ("levels", levels_figures)]
    for name, builder in builders:
        data = load(name)
        if data is not None:
            entries += builder(data)
    if load("dimension") is not None:
        entries += dimension_figures(load("dimension"), load("gradient"))
    from far_figures import far_figures  # noqa: PLC0415  (far_experiments.py)
    entries += far_figures(RESULTS, FIGURES)
    items = "\n".join(f'<li><a href="{file}">{title}</a></li>' for file, title in entries)
    (FIGURES / "index.html").write_text(
        "<!doctype html><html lang='fr'><meta charset='utf-8'>"
        "<title>Validation théorique de l'optimum</title>"
        "<body style='font-family:sans-serif;max-width:820px;margin:2em auto'>"
        "<h1>Validation théorique de l'optimum</h1>"
        f"<p>Run analysé : <code>{run.get('run', '?')}</code> · plan par niveau "
        f"{run.get('sizes', '?')} · {run.get('n_solved', '?')}/{run.get('n_conditions', '?')} "
        "conditions résolues. Aucun nouveau calcul des codes ni de la VPP : tout vient du modèle "
        "final du run. Le détail et les conclusions sont dans <code>../README.md</code>.</p>"
        f"<ol>{items}</ol></body></html>", encoding="utf-8")
    print(f"{len(entries)} figures in {FIGURES}")


if __name__ == "__main__":
    main()
