"""
Overview figure of the meta-problem (what the studies see), from the meta-surrogate and the cached
truth (vpp_study_common), one process, a few seconds:
  1. Fx (drag) and Fy (side force) against Yaw_Eff at a typical state, levels 1 and 2, for the
     reference section and two design sections;
  2. the true VPP speeds of the 4 design sections at the 5 navigation conditions;
  3. the true equilibrium states in the (Cant_Eff, Yaw_Eff) and (Cant_Eff, Hvol) planes;
  4. the speed sensitivities dV/dC_k (median over the sections) at each condition.
Run: ..\\.venv-mds\\Scripts\\python examples\\vpp_meta_overview.py -> figures/vpp_meta_overview.html
"""
import numpy as np

import vpp_study_common as common


def main():
    import plotly.graph_objects as go  # noqa: PLC0415
    from figure_style import SERIES, save, style  # noqa: PLC0415
    from plotly.subplots import make_subplots  # noqa: PLC0415

    reference = common.truth()
    meta = common.MetaFoil()
    sections = reference["sections"]
    fig = make_subplots(rows=2, cols=3, horizontal_spacing=0.08, vertical_spacing=0.2,
                        subplot_titles=("Traînée Fx selon Yaw_Eff (cant 6°, rake 1°, Hvol 0,15 m)",
                                        "Effort latéral Fy selon Yaw_Eff",
                                        "Vitesse VPP vraie (méta niveau 2)",
                                        "États d'équilibre : cant / yaw",
                                        "États d'équilibre : cant / Hvol",
                                        "Sensibilité de la vitesse |dV/dC_k| (médiane)"))
    yaw = np.linspace(-6, 6, 61)
    for i, color in zip(range(3), SERIES):
        x = np.column_stack([np.repeat(sections[i][None], len(yaw), axis=0),
                             np.full(len(yaw), 6.0), np.full(len(yaw), 1.0), yaw,
                             np.full(len(yaw), 0.15)])
        name = "profil de référence" if i == 0 else f"section de conception {i}"
        for level, dash in ((2, "solid"), (1, "dash")):
            y = meta(x, level)
            for col, k in ((1, 0), (2, 1)):
                fig.add_trace(go.Scatter(
                    x=yaw, y=y[:, k], mode="lines", line={"color": color, "dash": dash},
                    name=f"{name}, niveau {level}", legendgroup=f"{i}{level}",
                    showlegend=(col == 1),
                    hovertemplate="yaw %{x:.1f}° : %{y:.4f}<extra></extra>"), row=1, col=col)
    labels = [f"TWA {a:.0f} / TWS {s:.0f}" for a, s in reference["conditions"]]
    n_conditions = len(labels)
    speeds = reference["speeds"].reshape(len(sections), n_conditions)
    for i, color in zip(range(len(sections)), SERIES):
        fig.add_trace(go.Bar(x=labels, y=speeds[i], marker_color=color, showlegend=False,
                             hovertemplate="%{x} : %{y:.3f} nd<extra>section " + str(i)
                                           + "</extra>"), row=1, col=3)
    points = reference["points"]
    symbols = ["circle", "square", "diamond", "triangle-up", "x"]
    for j, label in enumerate(labels):
        rows = np.arange(j, len(points), n_conditions)
        for col, (x_index, y_index) in ((1, (12, 14)), (2, (12, 15))):
            fig.add_trace(go.Scatter(x=points[rows, x_index], y=points[rows, y_index],
                                     mode="markers", name=label, legendgroup=label,
                                     showlegend=(col == 1),
                                     marker={"symbol": symbols[j], "size": 9,
                                             "color": SERIES[j % len(SERIES)]}),
                          row=2, col=col)
    derivatives = np.abs(reference["derivatives"].reshape(len(sections), n_conditions, 6)[0])
    for j, label in enumerate(labels):
        fig.add_trace(go.Bar(x=common.FORCE_NAMES, y=derivatives[j], name=label,
                             legendgroup=label, showlegend=False,
                             marker_color=SERIES[j % len(SERIES)],
                             hovertemplate="%{x} : %{y:.2f} nd / unité<extra>" + label
                                           + "</extra>"), row=2, col=3)
    fig.update_yaxes(type="log", title_text="nd par unité de coefficient", row=2, col=3)
    fig.update_yaxes(title_text="Fx (coefficient)", row=1, col=1)
    fig.update_yaxes(title_text="Fy (coefficient)", row=1, col=2)
    fig.update_yaxes(title_text="vitesse [nd]", row=1, col=3)
    fig.update_xaxes(title_text="Yaw_Eff [°]", row=1, col=1)
    fig.update_xaxes(title_text="Yaw_Eff [°]", row=1, col=2)
    fig.update_xaxes(title_text="Cant_Eff [°]", row=2, col=1)
    fig.update_xaxes(title_text="Cant_Eff [°]", row=2, col=2)
    fig.update_yaxes(title_text="Yaw_Eff [°]", row=2, col=1)
    fig.update_yaxes(title_text="Hvol [m]", row=2, col=2)
    fig.update_layout(barmode="group")
    style(fig, "Le méta-problème : foil C d'Intens SY dans la VPP MDS",
          "Trait plein : niveau 2 (« AVL ») ; tirets : niveau 1 (« NPLLT ») ; états et vitesses : "
          "VPP MDS avec le niveau 2 comme foil", height=820, legend_rows=4)
    print(f"figure: {save(fig, common.FIGURES, 'vpp_meta_overview')}")


if __name__ == "__main__":
    main()
