"""CFD checks of the best sections of the xtr study (2026-11-09) with Antoine's bdfoil_v1 OpenFOAM chain, unchanged:
2D section points (section2d.run_pair, levels L1' + L2' as in his case C6) and the 3D C-foil of his case C9 (R9800,
neutral attitude, 23 kn, hull symmetry plane, kOmegaSST + wall functions) with the section swapped, plus AVL x XFOIL
of the same C9 geometry at several xtr. The 3D runs can solve the pressure on the GPU (PETSc FCG + AmgX, the config
of GPU-CFD/weekend): only the fvSolution / controlDict builders of case3d are wrapped in this process, and the
results go to their own cache so they never mix with Antoine's CPU cache.

  python scripts/cfd_sections.py prepare  --sections NAME=run_dir ...
  python scripts/cfd_sections.py avl      --xtr 0.03 0.05 0.10
  python scripts/cfd_sections.py run2d    --sections NAME ... --alphas -3 -2 -1 0 1 --np 10
  python scripts/cfd_sections.py run3d    --section NAME --level 1 [--gpu] [--np 4] [--hours 2]
  python scripts/cfd_sections.py figures
Everything lives in data/xtr_study/cfd/ (sections/<name>/, avl.json, cfd2d.json, cfd3d.json) and the figures in
runs/<date>_xtr_study/figures/cfd_sections.html.
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

BDFOIL = Path(r"C:\Users\SIM\Desktop\Code-Adri\bdfoil_v1")
os.environ.setdefault("BDTOOLBOX_ROOT", r"C:\Users\SIM\Outils - banulsdesign\bdToolbox")
sys.path.insert(0, str(BDFOIL / "src"))
sys.path.insert(0, str(BDFOIL / "scripts" / "openfoam"))

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "xtr_study" / "cfd"
SEC = OUT / "sections"
REF_SECTION = "MC2"                      # Antoine's section: CFD already done (C6 2D, C9 3D L1/L2)
GPU_BASHRC = "/home/bdcfd/gpu_solvers/env_gpu.sh"   # sources OpenFOAM v2512, then PETSc + AmgX
GPU_CACHE = Path(os.environ["BDTOOLBOX_ROOT"]) / "tmp" / "bdfoil_v1" / "cfd3d" / "points_gpu_adri"


def log(msg: str) -> None:
    print(f"[{dt.datetime.now():%H:%M:%S}] {msg}", flush=True)


def load_json(path: Path, default):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def save_json(path: Path, body) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(body, indent=1, default=float) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def read_xy(path: Path) -> np.ndarray:
    rows = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        try:
            rows.append([float(v) for v in line.split()[:2]])
        except ValueError:
            continue
    return np.array([r for r in rows if len(r) == 2])


def section_xf(name: str) -> Path:
    if name == REF_SECTION:
        return BDFOIL / "tests" / "data" / "cfoil_r9800" / "MC2_60_6101_1_re5M.xf"
    return SEC / name / f"{name}.xf"


def study_for(name: str):
    """Antoine's C9 study (structure dropped) with every planform station on section ``name``."""
    import c9_cfoil_neutral as c9
    from bdfoil_v1.study import load_study
    study = dataclasses.replace(load_study(c9.CASE), structure=None)
    if name == REF_SECTION:
        return study
    pf = dataclasses.replace(study.appendage.planform, csv=str(SEC / name / "planform.csv"), sha256=None)
    return dataclasses.replace(study, appendage=dataclasses.replace(study.appendage, planform=pf))


# ------------------------------------------------------------------------------------------------ prepare
def prepare(args) -> int:
    """.xf (in the order of Antoine's MC2 file: TE, lower side, LE, upper side) + a copy of his planform CSV."""
    src_csv = BDFOIL / "tests" / "data" / "cfoil_r9800" / "CFoil_R9800.csv"
    lines = src_csv.read_text(encoding="utf-8-sig").splitlines()
    head = lines[0].split(",")
    col = head.index("Section")
    for item in args.sections:
        name, run = item.split("=", 1)
        xy = read_xy(Path(run) / "results" / "sections" / "best_section.dat")
        i_le = int(xy[:, 0].argmin())
        if xy[:i_le, 1].mean() > 0:              # upper side first (Selig): reverse to MC2's order
            xy = xy[::-1]
        if args.n_side:                          # AVL reads at most a few hundred points per section file
            i_le = int(xy[:, 0].argmin())
            x = 0.5 * (1 - np.cos(np.linspace(0, np.pi, args.n_side)))
            a, b = xy[:i_le + 1][::-1], xy[i_le:]          # both from the LE to the TE
            ya, yb = np.interp(x, a[:, 0], a[:, 1]), np.interp(x, b[:, 0], b[:, 1])
            xy = np.vstack([np.c_[x, ya][::-1], np.c_[x, yb][1:]])
        d = SEC / name
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{name}.xf").write_text(f"{name}\n" + "".join(f"{x:.6f}    {y:.6f}\n" for x, y in xy),
                                      encoding="utf-8")
        rows = [lines[0]]
        for ln in lines[1:]:
            parts = ln.split(",")
            if len(parts) > col:
                parts[col] = f"{name}.xf"
            rows.append(",".join(parts))
        (d / "planform.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
        (d / "source.txt").write_text(f"{run}\n", encoding="utf-8")
        log(f"{name}: {len(xy)} points from {run}")
    return 0


def section_names() -> list[str]:
    return [REF_SECTION] + sorted(p.name for p in SEC.iterdir() if (p / "planform.csv").exists())


# ------------------------------------------------------------------------------------------------ AVL x XFOIL
def avl(args) -> int:
    import c9_cfoil_neutral as c9
    from bdfoil_v1.session import Session
    path = OUT / "avl.json"
    body = load_json(path, {})
    for name in args.sections or section_names():
        study = study_for(name)
        for xtr in args.xtr:
            key = f"{xtr:g}"
            if key in body.get(name, {}) and "totals" in body[name][key]:
                continue
            st = dataclasses.replace(study, polars=dataclasses.replace(study.polars, xtr=(xtr, xtr)))
            t0 = time.perf_counter()
            try:
                r = Session(st).evaluate(c9.NEUTRAL)
                si = {k: float(v) for k, v in dict(r.si).items() if isinstance(v, (int, float))}
                rec = {"totals": c9.avl_totals(si), "time_s": time.perf_counter() - t0}
            except Exception as exc:              # noqa: BLE001
                rec = {"error": f"{type(exc).__name__}: {exc}"[:400], "time_s": time.perf_counter() - t0}
            body.setdefault(name, {})[key] = rec
            save_json(path, body)
            log(f"AVL x XFOIL {name} xtr {key}: {rec.get('totals', rec.get('error'))}")
    return 0


# ------------------------------------------------------------------------------------------------ 2D
def run2d(args) -> int:
    from bdfoil_v1.session import Session
    from bdfoil_v1.solvers.openfoam import bridge, case2d, section2d
    from bdfoil_v1.xfoil import PolarSpec, run_polar
    path = OUT / "cfd2d.json"
    body = load_json(path, {})
    session = Session(study_for(REF_SECTION))
    re = 11.832 * 0.7 / session.fluid.nu                    # as C6
    env = bridge.detect()
    settings = case2d.Settings(nproc=args.np)
    for name in args.sections:
        rec = body.setdefault(name, {"re": re, "points": {}, "xfoil": {}})
        sref = Session(study_for(name)).planform.sections[0]
        xf = sref.path
        for xtr in args.xfoil_xtr:                          # XFOIL polars at the same Re, for the figures
            if f"{xtr:g}" in rec["xfoil"]:
                continue
            try:
                pol = run_polar(sref, PolarSpec(re=re, xtr=(xtr, xtr), alpha=(-6.0, 6.0, 0.25)),
                                workdir=OUT / "xfoil_work" / f"{name}_{xtr:g}", timeout=600.0)
                rec["xfoil"][f"{xtr:g}"] = {"alpha": pol.alpha.tolist(), "cl": pol.cl.tolist(),
                                            "cd": pol.cd.tolist(), "cm": pol.cm.tolist()}
            except Exception as exc:                        # noqa: BLE001
                rec["xfoil"][f"{xtr:g}"] = {"error": f"{type(exc).__name__}: {exc}"[:300]}
            save_json(path, body)
        for a in args.alphas:
            if f"{a:g}" in rec["points"]:
                continue
            t0 = time.perf_counter()
            try:
                p = section2d.run_pair(xf, alpha_deg=a, re=re, settings=settings, env=env, timeout_s=3 * 3600)
                row = {"CL": p.CL, "CD": p.CD, "CDp": p.CDp, "CDf": p.CDf, "CM": p.CM, "Cpmin": p.Cpmin,
                       "converged": p.converged, "message": p.message, "CD_L1": p.l1.CD, "CD_L2": p.l2.CD,
                       "n_iter_L2": p.l2.n_iter, "wall_s": time.perf_counter() - t0}
            except Exception as exc:                        # noqa: BLE001
                row = {"error": f"{type(exc).__name__}: {exc}"[:400], "wall_s": time.perf_counter() - t0}
            rec["points"][f"{a:g}"] = row
            save_json(path, body)
            log(f"2D {name} alpha {a:g}: {row}")
    return 0


# ------------------------------------------------------------------------------------------------ 3D
def use_gpu() -> None:
    """Pressure (and Phi, which copies p) on PETSc FCG + AmgX classical AMG, strength 0.8 (GPU-CFD weekend ★)."""
    from bdfoil_v1.solvers.openfoam import case3d
    from bdfoil_v1.solvers.openfoam.dicts import Quoted, Raw
    fv0, cd0 = case3d._fv_solution, case3d._control_dict

    def fv(s):
        d = fv0(s)
        p = d["solvers"]["p"]
        d["solvers"]["p"] = {"solver": "petsc", "tolerance": p["tolerance"], "relTol": p["relTol"],
                             "petsc": {"options": {"ksp_type": "fcg", "mat_type": "aijcusparse", "vec_type": "cuda",
                                                   "pc_type": "amgx", "pc_amgx_strength_threshold": Quoted("0.8")},
                                       "caching": {"matrix": {"update": "always"},
                                                   "preconditioner": {"update": "adaptive"}}}}
        return d

    def cd(s):
        d = cd0(s)
        d["libs"] = [Raw("petscFoam")]
        return d

    case3d._fv_solution, case3d._control_dict = fv, cd


def run3d(args) -> int:
    import c9_cfoil_neutral as c9
    from bdfoil_v1.session import Session
    from bdfoil_v1.solvers.openfoam import bridge, case3d, foil3d
    path = OUT / "cfd3d.json"
    tag = f"{args.section} L{args.level}" + (" GPU" if args.gpu else " CPU") + (f" yaw {args.yaw:g}" if args.yaw else "")
    body = load_json(path, {})
    if tag in body and "totals" in body[tag]:
        log(f"{tag}: already done")
        return 0
    if args.gpu:
        use_gpu()
        env = bridge.detect(bashrc=GPU_BASHRC)
        cache = GPU_CACHE
    else:
        env, cache = bridge.detect(), None
    session = Session(study_for(args.section))
    fl = session.fluid
    st = case3d.Settings3D(speed=c9.NEUTRAL.speed, nu=fl.nu, rho=fl.rho, chord=0.7, sref=2.274974, cref=0.7,
                           mesh_level=args.level, nproc=args.np)
    log(f"{tag}: start ({args.hours:g} h at most)")
    t0 = time.perf_counter()
    try:
        r = foil3d.run_foil3d(session, dataclasses.replace(c9.NEUTRAL, yaw=args.yaw), settings=st, env=env, cache_dir=cache,
                              timeout_s=args.hours * 3600, poll_s=60.0, progress=lambda m: log(f"  {m}"))
        rec = {"result": r.to_dict(), "totals": c9.cfd_totals(r), "wall_s": time.perf_counter() - t0}
    except Exception as exc:                                # noqa: BLE001
        rec = {"error": f"{type(exc).__name__}: {exc}"[:600], "wall_s": time.perf_counter() - t0}
    body = load_json(path, {})
    body[tag] = rec
    save_json(path, body)
    log(f"{tag}: {rec.get('totals', rec.get('error'))}")
    return 0


# ------------------------------------------------------------------------------------------------ figures
def figures(args) -> int:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    studies = sorted((ROOT / "runs").glob("*_xtr_study"))
    out_dir = Path(args.out) if args.out else (studies[-1] if studies else ROOT / "runs" / "xtr_study") / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    names = section_names()
    colours = dict(zip(names, ["#444444", "#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]))
    html = []

    # sections
    fig = go.Figure()
    for n in names:
        xy = read_xy(section_xf(n))
        fig.add_scatter(x=xy[:, 0] * 700, y=xy[:, 1] * 700, name=n, line=dict(color=colours[n]))
    fig.update_layout(title="Sections calculées en CFD [mm, corde 700 mm]", yaxis_scaleanchor="x", height=420)
    html.append(fig.to_html(full_html=False, include_plotlyjs="cdn"))

    # 2D: CFD points vs XFOIL polars, CD(CL)
    c2 = load_json(OUT / "cfd2d.json", {})
    c6 = load_json(BDFOIL / "validation" / "openfoam" / "c6_cfoil_section.json", {})
    fig = make_subplots(rows=1, cols=2, subplot_titles=("CD en fonction de CL", "CL en fonction de alpha"))
    dashes = {"0.03": "dot", "0.05": "dash", "0.1": "solid"}
    for n in names:
        if n == REF_SECTION:
            pts = [(p["alpha_deg"], p["CL"], p["CD"], p["converged"]) for p in c6.get("points", [])]
            xfs = {"0.1": c6.get("xfoil", {}).get("xtr 0.1 (production)"), "0.05": c6.get("xfoil", {}).get("xtr 0.05")}
        else:
            r = c2.get(n, {})
            pts = [(float(a), p["CL"], p["CD"], p["converged"]) for a, p in r.get("points", {}).items() if "CL" in p]
            xfs = r.get("xfoil", {})
        pts.sort()
        if pts:
            a, cl, cd, ok = map(np.array, zip(*pts))
            sym = ["circle" if o else "circle-open" for o in ok]
            fig.add_scatter(x=cd * 1e4, y=cl, mode="markers+lines", name=f"{n} CFD", marker=dict(size=9, symbol=sym),
                            line=dict(color=colours[n], width=3), legendgroup=n, row=1, col=1)
            fig.add_scatter(x=a, y=cl, mode="markers+lines", showlegend=False, marker=dict(size=9, symbol=sym),
                            line=dict(color=colours[n], width=3), legendgroup=n, row=1, col=2)
        for k, pol in (xfs or {}).items():
            if not pol or "cl" not in pol:
                continue
            al, cl, cd = (np.array(pol[q]) for q in ("alpha", "cl", "cd"))
            m = (al >= -6) & (al <= 6)
            fig.add_scatter(x=cd[m] * 1e4, y=cl[m], mode="lines", name=f"{n} XFOIL xtr {float(k):.0%}",
                            line=dict(color=colours[n], width=1, dash=dashes.get(k, "dashdot")), legendgroup=n,
                            row=1, col=1)
            fig.add_scatter(x=al[m], y=cl[m], mode="lines", showlegend=False,
                            line=dict(color=colours[n], width=1, dash=dashes.get(k, "dashdot")), legendgroup=n,
                            row=1, col=2)
    fig.update_xaxes(title_text="CD [1e-4]", range=[60, 140], row=1, col=1)
    fig.update_yaxes(title_text="CL", range=[-0.3, 1.0], row=1, col=1)
    fig.update_xaxes(title_text="alpha [deg]", row=1, col=2)
    fig.update_layout(title="2D, Re 6.96e6 : CFD k-ω SST (points, vides = non convergés) contre XFOIL (traits)",
                      height=560)
    html.append(fig.to_html(full_html=False, include_plotlyjs=False))

    # 2D table: CD at matched CL (CFD and XFOIL interpolated)
    rows = []
    for n in names:
        if n == REF_SECTION:
            pts = [(p["CL"], p["CD"]) for p in c6.get("points", []) if p["converged"]]
            xfs = {"0.1": c6.get("xfoil", {}).get("xtr 0.1 (production)"), "0.05": c6.get("xfoil", {}).get("xtr 0.05")}
        else:
            r = c2.get(n, {})
            pts = [(p["CL"], p["CD"]) for p in r.get("points", {}).values() if p.get("converged")]
            xfs = r.get("xfoil", {})
        pts.sort()
        for target in (0.1, 0.2, 0.3):
            cells = [n, f"{target:.1f}"]
            if len(pts) >= 2 and pts[0][0] <= target <= pts[-1][0]:
                cells.append(f"{1e4 * np.interp(target, *zip(*pts)):.1f}")
            else:
                cells.append("—")
            for k in ("0.03", "0.05", "0.1"):
                pol = (xfs or {}).get(k)
                if pol and "cl" in pol:
                    cl, cd = np.array(pol["cl"]), np.array(pol["cd"])
                    o = np.argsort(cl)
                    cells.append(f"{1e4 * np.interp(target, cl[o], cd[o]):.1f}")
                else:
                    cells.append("—")
            rows.append(cells)
    html.append("<h3>2D : CD [1e-4] à CL imposé (interpolé entre points CFD convergés)</h3><table><tr><th>section</th><th>CL</th><th>CFD</th>"
                "<th>XFOIL 3 %</th><th>XFOIL 5 %</th><th>XFOIL 10 %</th></tr>"
                + "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows) + "</table>")

    # 3D at equal lift: MC2 at neutral (Antoine's CPU runs) against opt003 at the yaw of the same lift (AVL x XFOIL
    # xtr 5 %), converged runs only; the unconverged runs are listed below the figure
    av = load_json(OUT / "avl.json", {})
    c3 = load_json(OUT / "cfd3d.json", {})
    c9 = load_json(BDFOIL / "validation" / "openfoam" / "c9" / "comparison.json", {})
    pairs = []                                   # (label, MC2 totals, optimised totals)
    if "totals" in c9.get("cfd", {}).get("L1", {}):
        for tag, rec in c3.items():
            if " yaw " in tag and rec.get("result", {}).get("converged") and "totals" in rec:
                pairs.append((f"CFD L1 ({tag.split()[0]} yaw {tag.split()[-1]}°)", c9["cfd"]["L1"]["totals"],
                              rec["totals"], tag.split()[0]))
    for key, recs in av.items():
        if " yaw " not in key:
            continue
        n = key.split()[0]
        for x, r in sorted(recs.items(), key=lambda kv: float(kv[0])):
            ref = av.get(REF_SECTION, {}).get(x, {})
            if "totals" in r and "totals" in ref:
                pairs.append((f"AVL×XFOIL xtr {float(x):.0%}", ref["totals"], r["totals"], n))
    fig = make_subplots(rows=1, cols=2, subplot_titles=("traînée -Fx [N]", "gain de traînée sur MC2 [N]"))
    labels = [p[0] for p in pairs]
    fig.add_bar(x=labels, y=[-p[1]["Fx"] for p in pairs], name="MC2 (assiette neutre)",
                marker_color=colours[REF_SECTION], row=1, col=1)
    fig.add_bar(x=labels, y=[-p[2]["Fx"] for p in pairs], name="section optimisée (même portance)",
                marker_color=colours.get(pairs[0][3], "#d62728") if pairs else "#d62728", row=1, col=1)
    fig.add_bar(x=labels, y=[p[2]["Fx"] - p[1]["Fx"] for p in pairs], showlegend=False,
                marker_color="#2ca02c", text=[f"{100 * (p[2]['Fx'] / p[1]['Fx'] - 1):+.1f} %" for p in pairs],
                row=1, col=2)
    fig.update_layout(barmode="group", height=500,
                      title="3D C9 (R9800, 23 kn) à portance égale : MC2 au neutre contre section optimisée en yaw")
    html.append(fig.to_html(full_html=False, include_plotlyjs=False))
    rows = [[lab, f"{-a['Fx']:.0f}", f"{-b['Fx']:.0f}", f"{a['Fx'] - b['Fx']:+.0f}",
             f"{100 * (b['Fx'] / a['Fx'] - 1):+.1f} %", f"{a['Fz']:.0f}", f"{b['Fz']:.0f}", f"{-a['Fy']:.0f}",
             f"{-b['Fy']:.0f}"] for lab, a, b, _ in pairs]
    html.append("<h3>3D à portance égale</h3><table><tr><th>modèle</th><th>-Fx MC2</th><th>-Fx optimisée</th>"
                "<th>gain [N]</th><th>gain</th><th>Fz MC2</th><th>Fz optimisée</th><th>-Fy MC2</th>"
                "<th>-Fy optimisée</th></tr>"
                + "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows) + "</table>"
                "<p>Le gain CFD est à comparer à celui d'AVL × XFOIL : même niveau de maillage L1 des deux côtés "
                "(L1 MC2 = run CPU d'Antoine du 20/10), l'erreur de maillage (~2.5 % sur Fx entre L1 et L2) "
                "s'annule en grande partie dans l'écart.</p>")
    runs = {tag: {k: rec.get("result", {}).get(k) for k in ("n_cells", "n_iter", "converged", "stop")}
            for tag, rec in c3.items()}
    html.append("<h3>Tous les runs 3D (les GPU de la nuit n'ont pas convergé : non utilisés)</h3><pre>"
                + json.dumps(runs, indent=1) + "</pre>")
    page = ("<!doctype html><html><head><meta charset='utf-8'><title>CFD des sections optimisées</title>"
            "<style>body{font-family:sans-serif;margin:16px;background:#fff}table{border-collapse:collapse}"
            "td,th{border:1px solid #bbb;padding:3px 8px;text-align:right}</style></head><body>"
            "<h1>CFD des meilleures sections de l'étude xtr</h1>"
            "<p>Chaîne OpenFOAM de bdfoil_v1 (Antoine) inchangée : 2D = paire L1'+L2' extrapolée (cas C6), 3D = cas "
            "C9 avec la section remplacée sur toute l'envergure ; MC2 = runs d'Antoine (CPU). GPU = pression "
            "PETSc/AmgX, mêmes réglages sinon.</p>" + "".join(html) + "</body></html>")
    (out_dir / "cfd_sections.html").write_text(page, encoding="utf-8")
    index = out_dir / "index.html"
    if index.exists() and "cfd_sections.html" not in index.read_text(encoding="utf-8"):
        text = index.read_text(encoding="utf-8")
        link = "<p><a href='cfd_sections.html'><b>CFD des meilleures sections (nuit du 09/11)</b></a></p>"
        index.write_text(text.replace("</body>", link + "</body>") if "</body>" in text else text + link,
                         encoding="utf-8")
    log(f"figures: {out_dir / 'cfd_sections.html'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--sections", nargs="+", required=True, help="NAME=run_dir")
    p.add_argument("--n-side", type=int, default=125, help="points per side after resampling (0 = keep)")
    p = sub.add_parser("avl")
    p.add_argument("--sections", nargs="*")
    p.add_argument("--xtr", type=float, nargs="+", default=[0.03, 0.05, 0.10])
    p = sub.add_parser("run2d")
    p.add_argument("--sections", nargs="+", required=True)
    p.add_argument("--alphas", type=float, nargs="+", default=[-3.0, -2.0, -1.0, 0.0])
    p.add_argument("--xfoil-xtr", type=float, nargs="*", default=[0.03, 0.05, 0.10])
    p.add_argument("--np", type=int, default=10)
    p = sub.add_parser("run3d")
    p.add_argument("--section", required=True)
    p.add_argument("--level", type=int, default=1)
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--np", type=int, default=4)
    p.add_argument("--hours", type=float, default=3.0)
    p.add_argument("--yaw", type=float, default=0.0, help="deg (e.g. the yaw of equal lift to MC2 from AVL x XFOIL)")
    p = sub.add_parser("figures")
    p.add_argument("--out", default=None)
    args = ap.parse_args()
    return {"prepare": prepare, "avl": avl, "run2d": run2d, "run3d": run3d, "figures": figures}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
