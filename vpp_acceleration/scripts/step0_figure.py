"""
Figure of a step-0 calibration run (scripts/real_step0_calibration.py): for each output, the RS
of production (CFoilFSI at 7 kt, almost rigid) against our two codes on the 18 attitudes, with
the straight line RS = a code + b. Written to <run>/figures/calibration.html.
Run: ..\\.venv-mds\\Scripts\\python scripts\\step0_figure.py runs\\<step0 folder>
"""
import sys
from pathlib import Path

import numpy as np

VPP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VPP_DIR))
sys.path.insert(0, str(VPP_DIR.parent / "benchmarks"))
# pylint: disable=wrong-import-position
from real_step0_calibration import PRODUCTION_RS  # noqa: E402
from src.meta_foil import ChebyshevRS  # noqa: E402

NAMES = ["Fx", "Fy", "Fz", "Mx", "My", "Mz", "Cpmin"]


def make_figure(root) -> None:
    import plotly.graph_objects as go  # noqa: PLC0415
    from figure_style import SERIES, save, style  # noqa: PLC0415
    from plotly.subplots import make_subplots  # noqa: PLC0415

    root = Path(root)
    data = np.load(root / "results" / "calibration.npz")
    attitudes = data["attitudes"]
    rs = ChebyshevRS(PRODUCTION_RS)(np.column_stack([attitudes, np.full(len(attitudes), 7.0)]))
    fig = make_subplots(rows=2, cols=4, horizontal_spacing=0.06, vertical_spacing=0.16,
                        subplot_titles=NAMES + ["Pentes a de RS = a code + b"])
    slopes = {1: [], 2: []}
    for k, name in enumerate(NAMES):
        row, col = k // 4 + 1, k % 4 + 1
        for level, label, color in ((1, "NPLLT × NeuralFoil", SERIES[2]),
                                    (2, "AVL × XFOIL", SERIES[0])):
            code = data[f"level{level}"][:, k]
            ok = np.isfinite(code)
            slope, intercept = np.polyfit(code[ok], rs[ok, k], 1)
            slopes[level].append(slope)
            fig.add_trace(go.Scatter(
                x=code[ok], y=rs[ok, k], mode="markers", name=label, legendgroup=label,
                showlegend=(k == 0), marker={"color": color, "size": 7},
                customdata=attitudes[ok],
                hovertemplate="cant %{customdata[0]:.0f}°, rake %{customdata[1]:.0f}°, yaw "
                              "%{customdata[2]:.0f}°<br>code %{x:.4f}, RS %{y:.4f}<extra>"
                              + label + "</extra>"), row=row, col=col)
        values = np.concatenate([rs[:, k], data["level1"][:, k], data["level2"][:, k]])
        lo, hi = np.nanmin(values), np.nanmax(values)
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", showlegend=(k == 0),
                                 name="égalité", line={"color": "grey", "dash": "dot"}),
                      row=row, col=col)
        fig.update_xaxes(title_text="code", row=row, col=col)
        fig.update_yaxes(title_text="RS de production", row=row, col=col)
    for level, color, label in ((1, SERIES[2], "NPLLT × NeuralFoil"), (2, SERIES[0], "AVL × XFOIL")):
        fig.add_trace(go.Bar(x=NAMES, y=slopes[level], marker_color=color, name=label,
                             showlegend=False), row=2, col=4)
    fig.add_hline(y=1.0, line={"color": "grey", "dash": "dot"}, row=2, col=4)
    fig.update_layout(barmode="group")
    style(fig, "Étape 0 : nos codes contre la RS de production du C-foil",
          f"{root.name} ; profil cambré de référence, 18 attitudes, RS CFoilFSI à 7 nd (quasi "
          "rigide) ; forces et moments sans conversion", height=760)
    print(f"figure: {save(fig, root / 'figures', 'calibration')}")


if __name__ == "__main__":
    make_figure(sys.argv[1])
