"""
Figures of a real run folder (runs/<date>_<name>/): written to its figures/ folder.
  1. convergence.html: predicted mean speed of the incumbent and of the candidates at each update,
     sizes of the design, levels chosen along the steps;
  2. points.html: points added by the loop in the attitude planes, with the VPP states;
  3. sections.html: the best section against the mean section of the PCA (start of the search),
     and its scores;
  4. validation.html: surrogate (mean +- 2 sigma) against the HF code (AVL x XFOIL or NPLLT x
     XFOIL), the LF code and AVL when it is a validation level, at the VPP states;
  5. speeds.html: speed per condition of every validated section, corrected to first order
     against the HF code (and AVL).
Run: ..\\.venv-mds\\Scripts\\python scripts\\real_run_figures.py runs\\<folder>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

VPP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VPP_DIR.parent / "benchmarks"))
sys.path.insert(0, str(VPP_DIR))


def make_figures(root) -> None:
    import plotly.graph_objects as go  # noqa: PLC0415
    from figure_style import SERIES, save, style  # noqa: PLC0415
    from plotly.subplots import make_subplots  # noqa: PLC0415
    from src.section_pca import SectionPCA, section_coordinates  # noqa: PLC0415

    root = Path(root)
    figures = root / "figures"
    updates = json.loads((root / "results" / "updates.json").read_text(encoding="utf-8"))
    history = pd.read_csv(root / "results" / "history.csv")
    validation = pd.read_csv(root / "results" / "validation.csv")
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    k = int(config["k"])

    # 1. convergence
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.1, subplot_titles=(
        "Vitesse moyenne prédite (VPP avec le surrogate)", "Taille du plan et niveaux choisis"))
    t = [u["elapsed_min"] for u in updates]
    fig.add_trace(go.Scatter(x=t, y=[u["incumbent_speed"] for u in updates], mode="lines+markers",
                             name="meilleure section du moment", line={"color": SERIES[0]}),
                  row=1, col=1)
    for i, u in enumerate(updates):
        fig.add_trace(go.Scatter(x=[u["elapsed_min"]] * (len(u["candidate_speeds"]) - 1),
                                 y=u["candidate_speeds"][1:], mode="markers",
                                 marker={"color": SERIES[1], "size": 6, "symbol": "circle-open"},
                                 name="candidats du tri", showlegend=(i == 0)), row=1, col=1)
    fig.add_trace(go.Scatter(x=history["elapsed_min"], y=history["n_lf"], name="points BF",
                             line={"color": SERIES[2]}), row=1, col=2)
    fig.add_trace(go.Scatter(x=history["elapsed_min"], y=history["n_hf"], name="points HF",
                             line={"color": SERIES[3]}), row=1, col=2)
    fig.update_xaxes(title_text="temps du run [min]")
    fig.update_yaxes(title_text="vitesse moyenne [nd]", row=1, col=1)
    fig.update_yaxes(title_text="nombre de points", row=1, col=2)
    style(fig, "Convergence du run", f"{root.name} ; {len(updates)} mises à jour de l'optimiseur",
          height=500)
    save(fig, figures, "convergence")

    # 2. points added
    xs = np.array([json.loads(x) if isinstance(x, str) else x for x in history["x"]])
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Cant / Yaw", "Cant / Rake"))
    for col, (j, name) in enumerate(((k + 2, "Yaw_Eff"), (k + 1, "Rake_Eff")), start=1):
        for level, symbol, color in ((1, "circle-open", SERIES[0]), (2, "diamond", SERIES[3])):
            sel = history["level"] == level
            fig.add_trace(go.Scatter(x=xs[sel, k], y=xs[sel, j], mode="markers",
                                     name=f"point ajouté, niveau {level}", showlegend=(col == 1),
                                     marker={"symbol": symbol, "color": color, "size": 7}),
                          row=1, col=col)
        fig.add_trace(go.Scatter(x=validation["att0"], y=validation[f"att{j - k}"], mode="markers",
                                 name="états VPP finaux", showlegend=(col == 1),
                                 marker={"symbol": "x", "color": "black", "size": 9}), row=1, col=col)
        fig.update_xaxes(title_text="Cant_Eff [°]", row=1, col=col)
        fig.update_yaxes(title_text=f"{name} [°]", row=1, col=col)
    style(fig, "Où le raffinement a placé ses points", root.name, height=500)
    save(fig, figures, "points")

    # 3. sections
    pca = SectionPCA.load(root / "data" / "pca.json")
    best = json.loads((root / "results" / "best_section.json").read_text(encoding="utf-8"))
    x = pca.stations
    fig = make_subplots(rows=1, cols=2, column_widths=[0.65, 0.35],
                        subplot_titles=("Profils", "Scores PCA de la meilleure section"))
    start = np.array(best.get("start_scores", np.zeros(k)))
    start_label = {"current_section": "profil actuel (départ)",
                   "closest_to_current": "profil admissible le plus proche de l'actuel (départ)"
                   }.get(best.get("start"), "section moyenne (départ)")
    for name, scores, color in ((start_label, start, SERIES[1]),
                                ("meilleure section", np.array(best["scores"]), SERIES[0])):
        y = section_coordinates(pca.decode(scores[None]), x, pca.te_thickness)[0]
        n = len(x)
        fig.add_trace(go.Scatter(x=np.concatenate([x, x[::-1]]),
                                 y=np.concatenate([y[:n], y[n:][::-1]]), mode="lines",
                                 name=name, line={"color": color}), row=1, col=1)
    fig.add_trace(go.Bar(x=[f"z{j + 1}" for j in range(k)], y=best["scores"],
                         marker_color=SERIES[0], showlegend=False), row=1, col=2)
    fig.update_yaxes(scaleanchor="x", scaleratio=1, row=1, col=1)
    style(fig, "La meilleure section trouvée",
          f"vitesse moyenne prédite {best['mean_speed']:.3f} nd", height=500)
    save(fig, figures, "sections")

    # 4. validation
    hf_name = "NPLLT × XFOIL" if config.get("hf") == "npllt_xfoil" else "AVL × XFOIL"
    has_avl = "Fx_avl" in validation
    names = ["Fx", "Fy", "Fz", "Mx", "My", "Mz", "Cpmin"]
    symbols = ("circle", "square", "diamond", "triangle-up")
    fig = make_subplots(rows=2, cols=4, subplot_titles=names + ["Erreur de vitesse (1er ordre)"])
    for idx, name in enumerate(names):
        row, col = idx // 4 + 1, idx % 4 + 1
        for section, symbol in zip(validation["section"].unique(), symbols):
            g = validation[validation["section"] == section]
            fig.add_trace(go.Scatter(
                x=g[f"{name}_hf"], y=g[f"{name}_sur"], mode="markers",
                error_y={"type": "data", "array": 2 * g[f"{name}_sigma"]},
                marker={"symbol": symbol, "color": SERIES[0]}, name=f"surrogate, {section}",
                showlegend=(idx == 0)), row=row, col=col)
            fig.add_trace(go.Scatter(x=g[f"{name}_hf"], y=g[f"{name}_lf"], mode="markers",
                                     marker={"symbol": symbol, "color": SERIES[2], "size": 6},
                                     name=f"NPLLT × NeuralFoil, {section}", showlegend=(idx == 0)),
                          row=row, col=col)
            if has_avl:
                fig.add_trace(go.Scatter(x=g[f"{name}_hf"], y=g[f"{name}_avl"], mode="markers",
                                         marker={"symbol": symbol, "color": SERIES[1], "size": 6},
                                         name=f"AVL × XFOIL, {section}", showlegend=(idx == 0)),
                              row=row, col=col)
        columns = [f"{name}_hf", f"{name}_sur"] + ([f"{name}_avl"] if has_avl else [])
        lo = float(np.nanmin(validation[columns].to_numpy()))
        hi = float(np.nanmax(validation[columns].to_numpy()))
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", showlegend=False,
                                 line={"color": "grey", "dash": "dot"}), row=row, col=col)
    fig.add_trace(go.Bar(x=validation["section"] + " " + validation["condition"],
                         y=validation["speed_error_kt"],
                         error_y={"type": "data", "array": 2 * validation["speed_sigma_kt"]},
                         marker_color=SERIES[0], showlegend=False), row=2, col=4)
    style(fig, f"Validation : surrogate contre {hf_name} aux états VPP",
          f"abscisse : {hf_name} ; barres : ± 2σ du surrogate", height=760, legend_rows=2)
    save(fig, figures, "validation")

    # 5. speeds of the validated sections, corrected to first order against the codes:
    #    V_code = V_surrogate - sum_k dV/dC_k (C_k surrogate - C_k code)
    fig = make_subplots(rows=1, cols=2 if has_avl else 1, subplot_titles=(
        [f"Vitesse corrigée {hf_name}"] + (["Vitesse corrigée AVL × XFOIL"] if has_avl else [])))
    corrections = [("speed_error_kt", 1)] + ([("speed_error_avl_kt", 2)] if has_avl else [])
    for i, section in enumerate(validation["section"].unique()):
        g = validation[validation["section"] == section]
        for column, col in corrections:
            fig.add_trace(go.Bar(x=g["condition"], y=g["speed_kt"] - g[column],
                                 name=section, marker_color=SERIES[i % len(SERIES)],
                                 showlegend=(col == 1)), row=1, col=col)
    fig.update_yaxes(title_text="vitesse [nd]")
    lows = [float((validation["speed_kt"] - validation[c]).min()) for c, _ in corrections]
    fig.update_yaxes(range=[min(lows) - 1.0, None])
    style(fig, "Vitesse du bateau par condition, sections validées",
          "VPP avec le surrogate final, corrigée au 1er ordre de l'écart mesuré aux états",
          height=480)
    save(fig, figures, "speeds")
    print(f"figures written to {figures}")


if __name__ == "__main__":
    make_figures(sys.argv[1])
