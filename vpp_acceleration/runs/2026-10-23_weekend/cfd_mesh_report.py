"""
Report of the CFD mesh study of the weekend of 2026-10-23 (cfd_mesh_study.sh): reads cfd/results.csv
and writes cfd/report.md, figures/cfd_mesh_study.html and the figure indexes.

1. GCI of the series G1 / G2 / G3 (y+ ~ 50, backgrounds /1.3, x1, x1.3), procedure of Celik et al.
   (2008, ASME J. Fluids Eng. 130, 078001): h_i = N_i^(-1/3), r = h_coarse / h_fine, observed
   order p by fixed-point iteration, Richardson extrapolation phi_ext, GCI_fine = 1.25 e_a / (r^p - 1)
   for the fine mesh and the same for the medium one (G2, the candidate production mesh).
2. Wall resolution: Y30, G2 (y+ ~ 50), Y100 at the same mesh.
3. Far field: D2 (domain x 2) against G1.
4. The speed equivalent of a force difference, to first order with the sensitivities of the third
   run at 45/16 (dV/dC_k, kt per unit coefficient; an order of magnitude: the attitude of the STL
   is not that state): dV = sum_k d_k dF_k / qS, qS = 0.5 rho U^2 S.
Run (MDS environment): ..\\..\\..\\.venv-mds\\Scripts\\python cfd_mesh_report.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
VPP_DIR = next(p for p in HERE.parents if p.name == "vpp_acceleration")
sys.path.insert(0, str(VPP_DIR / "scripts"))
sys.path.insert(0, str(VPP_DIR.parent / "benchmarks"))

QS = 0.5 * 1026.17 * 11.832 ** 2 * 2.27497          # N, the forceCoeffs settings of the cases
SENSITIVITIES = {"Fx": 111.0, "Fy": -0.5, "Fz": 6.9, "Mx": 4.8, "My": 0.6, "Mz": 0.0}
QUANTITIES = ["Fx", "Fx_pressure", "Fx_viscous", "Fy", "Fz", "Mx", "My", "Mz"]


def observed_order(phi, h, iterations=100):
    """Celik et al. (2008) steps 3-4 for phi = (fine, medium, coarse), h = cell sizes."""
    e21, e32 = phi[1] - phi[0], phi[2] - phi[1]
    r21, r32 = h[1] / h[0], h[2] / h[1]
    if e21 == 0 or e32 == 0:
        return np.nan, 0
    s = np.sign(e32 / e21)
    p = 2.0
    for _ in range(iterations):
        q = np.log((r21 ** p - s) / (r32 ** p - s))
        p_new = abs(np.log(abs(e32 / e21)) + q) / np.log(r21)
        if abs(p_new - p) < 1e-10:
            break
        p = p_new
    return p, s


# Celik et al. (2008): the GCI assumes the asymptotic range, i.e. differences between meshes that
# DECREASE: R = e21 / e32 in (0, 1). R >= 1 (the change grows) or R < 0 (oscillation): no GCI. R in
# [ASYMPTOTIC_MAX, 1): converging so slowly (almost a constant change per mesh) that the
# extrapolation is not trusted either (corrected 2026-10-26: the first report gave GCI of 4-17 % for
# the lift with R = 0.93-1.11).
ASYMPTOTIC_MAX = 0.9
RECOMMENDED_RATIO = 1.3                     # minimum refinement ratio of Celik et al. (2008)


def gci_table(rows: pd.DataFrame) -> pd.DataFrame:
    rows = rows.set_index("case")
    fine, medium, coarse = (rows.loc[n] for n in ("G3_fine_yp50", "G2_medium_yp50", "G1_coarse_yp50"))
    h = np.array([fine.cells, medium.cells, coarse.cells], dtype=float) ** (-1.0 / 3.0)
    out = []
    for name in QUANTITIES:
        phi = np.array([fine[name], medium[name], coarse[name]], dtype=float)
        e21, e32 = phi[1] - phi[0], phi[2] - phi[1]
        ratio = e21 / e32 if e32 != 0 else np.nan
        if not np.isfinite(ratio) or ratio < 0:
            verdict = "oscille : pas de GCI"
        elif ratio >= 1:
            verdict = "diverge : pas de GCI"
        elif ratio >= ASYMPTOTIC_MAX:
            verdict = "hors régime asymptotique : pas de GCI"
        else:
            verdict = "converge"
        row = {"quantity": name, "coarse": phi[2], "medium": phi[1], "fine": phi[0],
               "step_coarse_medium_%": 100 * (phi[1] / phi[2] - 1),
               "step_medium_fine_%": 100 * (phi[0] / phi[1] - 1), "R": ratio, "verdict": verdict,
               "observed_order": np.nan, "extrapolated": np.nan, "GCI_fine_%": np.nan,
               "GCI_medium_%": np.nan}
        if verdict == "converge":
            p, _ = observed_order(phi, h)
            r21, r32 = h[1] / h[0], h[2] / h[1]
            row.update({"observed_order": p,
                        "extrapolated": (r21 ** p * phi[0] - phi[1]) / (r21 ** p - 1),
                        "GCI_fine_%": 125 * abs((phi[0] - phi[1]) / phi[0]) / (r21 ** p - 1),
                        "GCI_medium_%": 125 * abs((phi[1] - phi[2]) / phi[1]) / (r32 ** p - 1)})
        out.append(row)
    return pd.DataFrame(out), (h[2] / h[1], h[1] / h[0])


def speed_equivalent(a: pd.Series, b: pd.Series) -> float:
    """Speed change to first order when the forces go from a to b (kt)."""
    return float(sum(d * (b[k] - a[k]) / QS for k, d in SENSITIVITIES.items()))


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")          # the console of Windows is cp1252
    rows = pd.read_csv(HERE / "cfd" / "results.csv")
    ok = rows[rows.rc == 0].copy()
    lines = ["# Étude de maillage CFD du C-foil (week-end du 23/10/2026)", "",
             "Lecture automatique de `cfd/results.csv` par `cfd_mesh_report.py`. Plan de l'étude et "
             "justification : `README.md` du dossier.", "",
             "## Les calculs", "",
             ok.assign(cells_M=ok.cells / 1e6)[["case", "cells_M", "layer_percent", "yplus_avg",
                                                "yplus_max", "Fx", "Fy", "Fz", "Mx",
                                                "Fx_drift_last200", "mesh_s", "solve_s"]]
             .to_markdown(index=False, floatfmt=".4g"), ""]
    failed = rows[rows.rc != 0]
    if len(failed):
        lines += ["Échecs : " + ", ".join(f"{r.case} (code {r.rc})" for r in failed.itertuples()), ""]
    series = {"G1_coarse_yp50", "G2_medium_yp50", "G3_fine_yp50"}
    if series <= set(ok.case):
        table, (r32, r21) = gci_table(ok)
        lines += ["## Convergence en maillage (GCI, Celik et al. 2008)", "",
                  f"Rapports de raffinement réels : {r32:.3f} (G1 → G2) et {r21:.3f} (G2 → G3)"
                  + (f", sous le minimum de {RECOMMENDED_RATIO} recommandé : le nombre de cellules "
                     "croît moins vite que le cube du rapport de fond (surface et couches)."
                     if min(r32, r21) < RECOMMENDED_RATIO else "."), "",
                  "Le GCI n'est donné que si les écarts entre maillages diminuent nettement "
                  f"(R = ε21 / ε32 entre 0 et {ASYMPTOTIC_MAX}).", "",
                  table.to_markdown(index=False, floatfmt=".4g"), ""]
        trends = table.set_index("quantity")
        if np.sign(trends.loc["Fx_pressure", "step_medium_fine_%"]) \
                != np.sign(trends.loc["Fx_viscous", "step_medium_fine_%"]):
            lines += ["**Compensation.** La traînée de pression et la traînée de frottement varient en "
                      "sens contraires : la convergence de la traînée totale est en partie une "
                      "compensation. Leurs propres GCI disent la vraie incertitude.", ""]
        by_case = ok.set_index("case")
        lines += [f"Équivalent vitesse (sensibilités du run 3 à 45/16) : G2 → G3 "
                  f"{speed_equivalent(by_case.loc['G2_medium_yp50'], by_case.loc['G3_fine_yp50']):+.3f} nd, "
                  f"G1 → G2 {speed_equivalent(by_case.loc['G1_coarse_yp50'], by_case.loc['G2_medium_yp50']):+.3f} nd.", ""]
    else:
        table = None
    wall = [c for c in ("Y30_medium", "G2_medium_yp50", "Y100_medium") if c in set(ok.case)]
    if len(wall) >= 2:
        targets = {"Y30_medium": 30, "G2_medium_yp50": 50, "Y100_medium": 100}
        w = ok.set_index("case").loc[wall].assign(yplus_target=[targets[c] for c in wall])
        lines += ["## Résolution de paroi (même maillage, y+ différents)", "",
                  w[["yplus_target", "yplus_avg", "layer_percent", "Fx", "Fx_pressure",
                     "Fx_viscous", "Fz", "Mx"]].to_markdown(floatfmt=".4g"), "",
                  "Le y+ moyen est celui de toute la surface du foil, zones sans couches comprises "
                  "(y+ maximal de 1500 à 2300) : il ne mesure pas le y+ des zones à couches. Les y+ "
                  "obtenus restent loin des y+ visés, et l'écart entre ces cas ne mesure que la "
                  "plage obtenue.", ""]
    if {"D2_coarse_yp50", "G1_coarse_yp50"} <= set(ok.case):
        d = ok.set_index("case")
        rel = {k: 100 * (d.loc["D2_coarse_yp50", k] / d.loc["G1_coarse_yp50", k] - 1)
               for k in ("Fx", "Fy", "Fz", "Mx")}
        lines += ["## Domaine (D2, deux fois plus grand, contre G1)", "",
                  ", ".join(f"{k} {v:+.2f} %" for k, v in rel.items()) + "; équivalent vitesse "
                  f"{speed_equivalent(d.loc['G1_coarse_yp50'], d.loc['D2_coarse_yp50']):+.3f} nd.", ""]
    (HERE / "cfd" / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    make_figure(ok, table)
    from real_run_figures import write_index, write_runs_index  # noqa: PLC0415
    write_index(HERE)
    write_runs_index(HERE.parent)
    print("\n".join(lines))


def make_figure(ok: pd.DataFrame, table) -> None:
    import plotly.graph_objects as go  # noqa: PLC0415
    from figure_style import SERIES, save, style  # noqa: PLC0415
    from plotly.subplots import make_subplots  # noqa: PLC0415
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.12, subplot_titles=(
        "Série G : valeur / valeur du maillage moyen", "Résolution de paroi : traînée selon le y+"))
    series = ok[ok.case.str.startswith("G")].sort_values("cells")
    if table is not None and len(series):
        h = series.cells.to_numpy(float) ** (-1 / 3)
        for i, row in enumerate(table.itertuples()):
            # relative to the medium mesh: the extrapolation only exists for the converging ones
            values = series[row.quantity].to_numpy(float) / row.medium
            fig.add_trace(go.Scatter(x=h, y=values, mode="lines+markers",
                                     name=f"{row.quantity} ({row.verdict.split(':')[0]})",
                                     line={"color": SERIES[i % len(SERIES)]}), row=1, col=1)
    wall = ok[ok.case.isin(["Y30_medium", "G2_medium_yp50", "Y100_medium"])].sort_values("yplus_avg")
    for i, name in enumerate(("Fx", "Fx_pressure", "Fx_viscous")):
        fig.add_trace(go.Scatter(x=wall.yplus_avg, y=wall[name], mode="lines+markers",
                                 name=f"{name} (paroi)", line={"color": SERIES[i % len(SERIES)],
                                                               "dash": "dot"}), row=1, col=2)
    fig.update_xaxes(title_text="h = N^(-1/3)  (plus fin vers la gauche)", row=1, col=1)
    fig.update_xaxes(title_text="y+ moyen du foil", row=1, col=2)
    fig.update_yaxes(title_text="rapport à la valeur du maillage moyen", row=1, col=1)
    fig.update_yaxes(title_text="force [N]", row=1, col=2)
    style(fig, "Étude de maillage CFD du C-foil", "OpenFOAM v2512, kOmegaSST, 16 cœurs ; "
          "week-end du 23/10/2026", height=520, legend_rows=2)
    save(fig, HERE / "figures", "cfd_mesh_study")


if __name__ == "__main__":
    main()
