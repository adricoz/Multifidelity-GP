"""
Report of the digital twin (French, for Adrian and Antoine): README.md and plotly figures from
results/reference.json (direct search), results/compare.json (true speeds) and the twin run folder.
Never fails on a missing part: it reports what exists.

    ..\\.venv-mds\\Scripts\\python twin_experiments\\report.py
"""
import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

HERE = Path(__file__).resolve().parent
COLORS = {"reference": "#1b7837", "method": "#b2182b", "start": "#f1a340", "truth_run": "#2166ac",
          "nocfd": "#762a83", "certificate": "#7f7f7f"}
LABELS = {"reference": "recherche directe (référence)", "method": "méthode sur le jumeau",
          "start": "section de départ", "truth_run": "run réel du 11/11",
          "nocfd": "run réel sans CFD", "certificate": "sections de borne haute maximale"}


def read(name: str):
    path = HERE / "results" / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def fr(value, digits: int = 4) -> str:
    return "–" if value is None or not np.isfinite(value) else f"{value:+.{digits}f}".replace(".", ",")


def write(figure: go.Figure, folder: Path, name: str, title: str, height: int = 520) -> str:
    figure.update_layout(title=title, height=height, template="plotly_white",
                         margin={"l": 70, "r": 30, "t": 60, "b": 60})
    figure.write_html(folder / f"{name}.html", include_plotlyjs="cdn")
    return f"{name}.html"


def verdict_figure(compare, folder: Path) -> str:
    rows = sorted((r for r in compare["sections"] if r["gap_to_anchor_kt"] is not None),
                  key=lambda r: -r["gap_to_anchor_kt"])
    figure = go.Figure()
    for group in dict.fromkeys(r["group"] for r in rows):
        part = [r for r in rows if r["group"] == group]
        figure.add_bar(x=[r["name"] for r in part], y=[r["gap_to_anchor_kt"] for r in part],
                       name=LABELS.get(group, group), marker_color=COLORS.get(group, "#444"),
                       customdata=[r["distance_to_anchor"] for r in part],
                       hovertemplate="%{x}<br>écart %{y:.4f} nd<br>distance %{customdata:.2f} "
                                     "score<extra></extra>")
    figure.add_hline(y=-0.02, line_dash="dash", line_color="black",
                     annotation_text="critère : −0,02 nd")
    figure.update_layout(xaxis={"categoryorder": "array", "categoryarray": [r["name"] for r in rows],
                                "tickangle": -45},
                         yaxis_title="vitesse vraie − meilleur optimum de référence (nd)")
    return write(figure, folder, "verdict", "Jumeau numérique : vitesse vraie de chaque section", 620)


def map_figure(compare, folder: Path) -> str:
    figure = go.Figure()
    rows = compare["sections"]
    for group in dict.fromkeys(r["group"] for r in rows):
        part = [r for r in rows if r["group"] == group]
        figure.add_scatter(
            x=[r["scores"][0] for r in part], y=[r["scores"][1] for r in part], mode="markers+text",
            text=[r["name"].replace("reference_from_", "réf. ").replace("truth_run_", "11/11 ")
                  for r in part], textposition="top center", textfont={"size": 9},
            name=LABELS.get(group, group),
            marker={"size": 12, "color": COLORS.get(group, "#444"), "line": {"width": 1}},
            customdata=[r["gap_to_anchor_kt"] for r in part],
            hovertemplate="%{text}<br>écart %{customdata:.4f} nd<extra></extra>")
    figure.update_layout(xaxis_title="score du mode 1", yaxis_title="score du mode 2")
    return write(figure, folder, "carte_sections",
                 "Où sont les sections (modes 1 et 2, les deux qui portent la vitesse)", 600)


def reference_figure(reference, folder: Path) -> str:
    figure = go.Figure()
    for start in reference["starts"]:
        gain, x, y = 0.0, [0], [0.0]
        for record in start["history"]:
            if record["accepted"]:
                gain += record["gain"]
            x.append(record["iteration"] + 1)
            y.append(gain)
        figure.add_scatter(x=x, y=y, mode="lines+markers", name=f"départ : {start['name']}",
                           hovertemplate="itération %{x}<br>gain cumulé %{y:.4f} nd<extra></extra>")
    figure.update_layout(xaxis_title="itération de la recherche directe",
                         yaxis_title="gain cumulé de vitesse vraie depuis le départ (nd)")
    return write(figure, folder, "recherche_directe",
                 "Recherche directe sur la vérité : gain depuis chaque départ")


def method_figure(twin_run: Path, folder: Path):
    path = twin_run / "results" / "updates.json"
    if not path.exists():
        return None
    updates = json.loads(path.read_text(encoding="utf-8"))
    figure = go.Figure()
    for region in dict.fromkeys(u["region"] for u in updates):
        part = [u for u in updates if u["region"] == region]
        figure.add_scatter(x=[u["elapsed_min"] / 60 for u in part],
                           y=[u["incumbent_speed"] for u in part], mode="lines+markers",
                           name=f"région {region}")
    figure.update_layout(xaxis_title="temps (h)",
                         yaxis_title="vitesse de la section courante, selon son propre modèle (nd)")
    return write(figure, folder, "methode_convergence",
                 "La méthode sur le jumeau : convergence, région par région")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", maxsplit=1)[0])
    parser.add_argument("--smoke", action="store_true", help="the files of the chain test")
    args = parser.parse_args()
    suffix = "_smoke" if args.smoke else ""
    reference, compare = read("reference" + suffix), read("compare" + suffix)
    folder = HERE / ("figures" + suffix)
    folder.mkdir(exist_ok=True)
    figures, lines = [], []
    lines += ["# Jumeau numérique de l'optimisation de section", "",
              f"Compte rendu écrit le {dt.datetime.now():%d/%m/%Y à %H:%M} par `report.py`.", "",
              "## Le principe", "",
              "On remplace les codes aéro par une fonction **connue** : la moyenne du modèle final du run "
              "du 11/11 (niveau 1 = NPLLT × NeuralFoil, niveau 2 = AVL × XFOIL). Sur cette « vérité », "
              "qui ne coûte rien, on peut chercher l'optimum **directement** (montée sous contraintes "
              "avec la vraie VPP, chaque pas vérifié par la VPP). On relance ensuite notre méthode "
              "complète depuis zéro, comme sur un run réel, et l'on regarde si elle retrouve cet optimum.",
              "", "Critère fixé avant de lancer : la section trouvée par la méthode doit être à moins de "
              "**0,02 nd** du meilleur optimum de la recherche directe, en vitesse vraie.", "",
              "Comment lire le résultat :", "",
              "- **Perte ≤ 0,02 nd** : l'algorithme (plan initial, ciblage, criblage, redémarrages, "
              "convergence) retrouve l'optimum d'une fonction dont on connaît la réponse.",
              "- **Perte plus grande, section éloignée** : la méthode s'est arrêtée dans un autre bassin ; "
              "le criblage ou les redémarrages ne suffisent pas.",
              "- **Section éloignée mais même vitesse** : la vérité a une crête plate ; l'optimum n'est pas "
              "unique, ce qui n'est pas un défaut.",
              "- **Départs de la recherche directe à des vitesses différentes** : la vérité a plusieurs "
              "bassins ; le tableau dit lesquels comptent.",
              "- Les lignes « sections de borne haute maximale » sont les sections du certificat (test 2) "
              "résolues par la VPP sur la vérité : si aucune ne bat la référence, l'endroit le plus "
              "prometteur selon le modèle ne cache rien.",
              "- La ligne « run réel sans CFD » est la section de l'autre run jugée exactement par le "
              "modèle du 11/11 (jugement croisé, test 3).", ""]
    if compare is None:
        lines += ["## Verdict", "", "La comparaison finale (`compare.py`) n'a pas tourné : pas de verdict.", ""]
    else:
        verdict = compare.get("verdict")
        lines += ["## Verdict", ""]
        if verdict is None:
            lines += ["Verdict impossible : il manque la section de la méthode ou la référence.", ""]
        else:
            lines += [f"- Vitesse perdue par la méthode sur le meilleur optimum de référence : "
                      f"**{fr(verdict['regret_kt'])} nd** (critère : 0,02 nd) → "
                      f"**{'réussi' if verdict['passed'] else 'échoué'}**.",
                      f"- Distance entre les deux sections : {verdict['distance_scores']:.2f} score."
                      .replace(".", ","), ""]
        lines += [f"Référence de comparaison : `{compare['anchor']}`. Run de la méthode : "
                  f"`{compare['twin_run']}`.", "",
                  "| Section | Famille | Écart de vitesse vraie (nd) | Distance (scores) | Conditions résolues |",
                  "|---|---|---|---|---|"]
        for row in sorted(compare["sections"], key=lambda r: -(r["gap_to_anchor_kt"]
                                                               if r["gap_to_anchor_kt"] is not None else -9)):
            lines.append(f"| `{row['name']}` | {LABELS.get(row['group'], row['group'])} | "
                         f"{fr(row['gap_to_anchor_kt'])} | "
                         f"{row['distance_to_anchor']:.2f} | {row['n_solved']} |".replace(".", ","))
        lines.append("")
        figures += [verdict_figure(compare, folder), map_figure(compare, folder)]
        if compare.get("twin_run"):
            figure = method_figure(HERE / "twin_runs" / compare["twin_run"], folder)
            if figure:
                figures.append(figure)
    if reference is not None:
        lines += ["## La recherche directe", "",
                  "| Départ | Arrêt | Itérations | Déplacement (scores) | Gain de vitesse vraie (nd) | Durée (min) |",
                  "|---|---|---|---|---|---|"]
        for start in reference["starts"]:
            gain = sum(r["gain"] for r in start["history"] if r["accepted"])
            lines.append(f"| `{start['name']}` | {start['stop']} | {len(start['history'])} | "
                         f"{start['distance_from_start']:.2f} | {fr(gain)} | {start['time_min']:.0f} |"
                         .replace(".", ","))
        lines += ["", "Lecture : si des départs différents arrivent à la même vitesse vraie, la vérité n'a "
                  "qu'un sommet (ou une crête) ; s'ils diffèrent, elle a plusieurs bassins et la colonne "
                  "« écart » du tableau du verdict dit lesquels comptent.", ""]
        figures.append(reference_figure(reference, folder))
    lines += ["## Limites", "",
              "- La vérité est elle-même un krigeage : plus lisse que les vrais codes, et de la même "
              "famille que le modèle qui l'apprend. Le test juge **l'algorithme** (plan initial, ciblage "
              "des points, criblage, redémarrages, convergence), pas l'hypothèse de régularité.",
              "- La recherche directe est une montée locale depuis quelques départs : elle peut manquer "
              "un bassin qu'aucun départ ne voit.",
              "- Les vitesses sont comparées sur les conditions résolues pour les deux sections.", "",
              "## Figures", ""] + [f"- [{name}](figures{suffix}/{name})" for name in figures] + [
              "", "## Refaire", "", "```",
              r"..\.venv-mds\Scripts\python twin_experiments\reference.py --workers 12",
              r"..\.venv-mds\Scripts\python twin_experiments\run_twin.py --time-limit-min 330",
              r"..\.venv-mds\Scripts\python twin_experiments\compare.py --workers 24",
              r"..\.venv-mds\Scripts\python twin_experiments\report.py", "```", "",
              "Fichiers : `twin.py` (vérité, remplaçant des codes), `reference.py` (recherche directe), "
              "`run_twin.py` (méthode complète, `scripts/run_real_vpp.py` inchangé), `compare.py` "
              "(vitesses vraies), `twin_runs/` (dossiers des runs sur le jumeau).", ""]
    (HERE / f"README{suffix}.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"README{suffix}.md, {len(figures)} figures", flush=True)


if __name__ == "__main__":
    main()
