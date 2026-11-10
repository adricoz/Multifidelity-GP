"""2D CFD polars at Re 5e6 for AVL (night of 2026-11-10, user: "des polaires CFD 2D pour AVL ... prêtes demain
matin"): Antoine's bdfoil_v1 2D chain unchanged (section2d L1' + L2' C-grid pair, kOmegaSST, Spalding wall treatment
at y+ < 1, runTimeControl on Cl / Cd, CD extrapolated by Richardson), at the Reynolds number of the fidelity levels of
the optimizer (src/foil_codes.FLOW: Re 5e6 on the 0.7 m chord), for the current C-foil section (MC2, Antoine's file)
and the best section of the xtr 3 % run (opt003, data/xtr_study/cfd/sections). XFOIL polars at the same Re
(n_crit 1, xtr 3 / 5 / 10 %) for the comparison.

  python scripts/cfd2d_polars.py run --run-dir runs/<dir> [--sections MC2 opt003] [--alphas ...] [--workers 4]
         [--np 4] [--deadline "2026-11-11 07:30"]
  python scripts/cfd2d_polars.py figures --run-dir runs/<dir>

Several points run at once (--workers cases of --np MPI ranks each: on 65 k / 247 k cells, few ranks per case gives
more points per hour than one case on 16 ranks). The points are taken in the order of --alphas, the two sections
alternating, so that a short night still gives the same incidences for both sections. Every level is cut at the
deadline (bridge timeout, stopAt writeNow); a point whose L2' did not finish keeps its L1' result, flagged.
Outputs in the run folder: data/polars.json (every point, raw L1' / L2'), data/<section>_cfd2d_re5e6.dat (converged
points: alpha CL CD CDp CDf CM), figures/polars.html, run.log. The GPU is not used: at these sizes it is slower than
the CPU (measured 06-10/11), and it stays free for Antoine.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import json
import logging
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cfd_sections as cs  # noqa: E402  (sets the bdfoil_v1 paths and BDTOOLBOX_ROOT)

RE = 5.0e6                      # src/foil_codes.FLOW["re"], the Reynolds number of the optimizer's levels
XTRS = (0.03, 0.05, 0.10)
DEFAULT_ALPHAS = (-1.0, 0.0, -2.0, 1.0, -3.0, 2.0, -4.0, 3.0)   # most used incidences first
LOCK = threading.Lock()


def log(run_dir: Path, msg: str) -> None:
    line = f"[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    with LOCK:
        print(line, flush=True)
        with open(run_dir / "run.log", "a", encoding="utf-8") as f:
            f.write(line + "\n")


def point_row(p) -> dict:
    keys = ("CL", "CD", "CDp", "CDf", "CM", "Cpmin", "converged", "stop", "n_iter", "CL_half_amplitude",
            "CD_half_amplitude", "yplus_max", "yplus_mean", "n_cells", "wall_s", "message")
    return {k: getattr(p, k) for k in keys}


def run(args) -> int:
    from bdfoil_v1.session import Session
    from bdfoil_v1.solvers.openfoam import bridge, case2d, section2d
    from bdfoil_v1.xfoil import PolarSpec, run_polar
    logging.basicConfig(level=logging.WARNING)
    run_dir = Path(args.run_dir)
    (run_dir / "data").mkdir(parents=True, exist_ok=True)
    path = run_dir / "data" / "polars.json"
    body = cs.load_json(path, {"re": RE, "sections": {}})
    deadline = dt.datetime.fromisoformat(args.deadline)
    env = bridge.detect()
    settings = case2d.Settings(nproc=args.np)
    log(run_dir, f"start: sections {args.sections}, alphas {args.alphas}, {args.workers} cases x {args.np} ranks, "
                 f"Re {RE:.3g}, deadline {deadline:%Y-%m-%d %H:%M}, OpenFOAM {env.tag}")
    for name in args.sections:                      # XFOIL first (seconds), for the figures
        rec = body["sections"].setdefault(name, {"xf": str(cs.section_xf(name)), "points": {}, "xfoil": {}})
        sref = Session(cs.study_for(name)).planform.sections[0]
        for xtr in XTRS:
            if f"{xtr:g}" in rec["xfoil"]:
                continue
            try:
                pol = run_polar(sref, PolarSpec(re=RE, xtr=(xtr, xtr), alpha=(-6.0, 6.0, 0.25)),
                                workdir=run_dir / "xfoil_work" / f"{name}_{xtr:g}", timeout=600.0)
                rec["xfoil"][f"{xtr:g}"] = {"alpha": pol.alpha.tolist(), "cl": pol.cl.tolist(),
                                            "cd": pol.cd.tolist(), "cm": pol.cm.tolist()}
            except Exception as exc:                # noqa: BLE001
                rec["xfoil"][f"{xtr:g}"] = {"error": f"{type(exc).__name__}: {exc}"[:300]}
            log(run_dir, f"XFOIL {name} xtr {xtr:g}: {'ok' if 'alpha' in rec['xfoil'][f'{xtr:g}'] else 'failed'}")
    cs.save_json(path, body)

    def left_s() -> float:
        return (deadline - dt.datetime.now()).total_seconds()

    def job(name: str, alpha: float) -> None:
        rec = body["sections"][name]
        if rec["points"].get(f"{alpha:g}", {}).get("level") == "L1'+L2'":
            return
        src1, src2 = section2d.pair_sources(cs.section_xf(name), RE)
        rows = {}
        for level, src in (("L1'", src1), ("L2'", src2)):
            if left_s() < 900:
                log(run_dir, f"{name} alpha {alpha:g} {level}: skipped, {left_s() / 60:.0f} min left")
                break
            t0 = time.perf_counter()
            try:
                p = section2d.run_point(src, alpha_deg=alpha, re=RE, settings=settings, env=env,
                                        timeout_s=max(600.0, left_s() - 300.0))
                rows[level] = p
                log(run_dir, f"{name} alpha {alpha:g} {level}: CL {p.CL:.4f} CD {p.CD:.5f} converged {p.converged} "
                             f"({p.stop}, {p.n_iter} it, {time.perf_counter() - t0:.0f} s)")
            except Exception as exc:                # noqa: BLE001
                log(run_dir, f"{name} alpha {alpha:g} {level}: error {type(exc).__name__}: {exc}"[:500])
                break
            if not p.converged:
                break
        if "L1'" not in rows:
            return
        out = {"l1": point_row(rows["L1'"])}
        if "L2'" in rows:
            pair = section2d.combine_pair(rows["L1'"], rows["L2'"])
            out.update(level="L1'+L2'", l2=point_row(rows["L2'"]), CL=pair.CL, CD=pair.CD, CDp=pair.CDp,
                       CDf=pair.CDf, CM=pair.CM, Cpmin=pair.Cpmin, converged=pair.converged, message=pair.message)
        else:
            l1 = rows["L1'"]
            out.update(level="L1' only", CL=l1.CL, CD=l1.CD, CDp=l1.CDp, CDf=l1.CDf, CM=l1.CM, Cpmin=l1.Cpmin,
                       converged=False, message="L2' not finished: L1' values, not extrapolated")
        with LOCK:
            rec["points"][f"{alpha:g}"] = out
            cs.save_json(path, body)

    jobs = [(n, a) for a in args.alphas for n in args.sections]
    with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for fut in [pool.submit(job, n, a) for n, a in jobs]:
            fut.result()
    write_tables(run_dir, body)
    log(run_dir, "all points done")
    return 0


def write_tables(run_dir: Path, body: dict) -> None:
    for name, rec in body["sections"].items():
        pts = sorted((float(a), p) for a, p in rec["points"].items() if p.get("level") == "L1'+L2'")
        lines = [f"# {name}: 2D CFD (bdfoil_v1 section2d L1'+L2', kOmegaSST, Richardson CD), Re {body['re']:.4g}",
                 "# converged: 1 = both levels stopped by runTimeControl",
                 "# alpha_deg  CL  CD  CDp  CDf  CM  converged"]
        lines += [f"{a:7.2f} {p['CL']:.5f} {p['CD']:.6f} {p['CDp']:.6f} {p['CDf']:.6f} {p['CM']:.5f} "
                  f"{int(bool(p['converged']))}" for a, p in pts]
        (run_dir / "data" / f"{name}_cfd2d_re5e6.dat").write_text("\n".join(lines) + "\n", encoding="utf-8")


def figures(args) -> int:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    run_dir = Path(args.run_dir)
    body = cs.load_json(run_dir / "data" / "polars.json", None)
    if body is None:
        print("no data/polars.json yet")
        return 1
    write_tables(run_dir, body)
    colours = {"MC2": "#1f77b4", "opt003": "#d62728", "opt005": "#2ca02c"}
    fig = make_subplots(rows=1, cols=3, subplot_titles=("CL(alpha)", "CD(CL)", "CM(alpha)"))
    table = []
    for name, rec in body["sections"].items():
        c = colours.get(name, "#555")
        for xtr, pol in rec["xfoil"].items():
            if "alpha" not in pol:
                continue
            dash = {"0.03": "solid", "0.05": "dash", "0.1": "dot"}.get(xtr, "dashdot")
            kw = dict(mode="lines", line=dict(color=c, dash=dash, width=1), legendgroup=name,
                      name=f"{name} XFOIL xtr {float(xtr):.0%}")
            fig.add_trace(go.Scatter(x=pol["alpha"], y=pol["cl"], **kw), 1, 1)
            fig.add_trace(go.Scatter(x=pol["cl"], y=pol["cd"], showlegend=False, **kw), 1, 2)
            fig.add_trace(go.Scatter(x=pol["alpha"], y=pol["cm"], showlegend=False, **kw), 1, 3)
        pts = sorted((float(a), p) for a, p in rec["points"].items())
        for full in (True, False):
            sel = [(a, p) for a, p in pts if (p.get("level") == "L1'+L2'" and p.get("converged")) == full]
            if not sel:
                continue
            a = [s[0] for s in sel]
            kw = dict(mode="markers", marker=dict(color=c, size=10 if full else 8,
                                                  symbol="circle" if full else "x"),
                      legendgroup=name, name=f"{name} CFD 2D" + ("" if full else " (L1' only / not converged)"))
            fig.add_trace(go.Scatter(x=a, y=[s[1]["CL"] for s in sel], **kw), 1, 1)
            fig.add_trace(go.Scatter(x=[s[1]["CL"] for s in sel], y=[s[1]["CD"] for s in sel], showlegend=False,
                                     **kw), 1, 2)
            fig.add_trace(go.Scatter(x=a, y=[s[1]["CM"] for s in sel], showlegend=False, **kw), 1, 3)
        for a, p in pts:
            x3 = rec["xfoil"].get("0.03", {})
            cd_x = float(np.interp(p["CL"], *zip(*sorted(zip(x3["cl"], x3["cd"]))))) if "cl" in x3 else np.nan
            table.append(f"<tr><td>{name}</td><td>{a:g}</td><td>{p['level']}</td><td>{p['converged']}</td>"
                         f"<td>{p['CL']:.4f}</td><td>{1e4 * p['CD']:.1f}</td><td>{1e4 * cd_x:.1f}</td>"
                         f"<td>{100 * (p['CD'] / cd_x - 1):+.1f} %</td><td>{p['CM']:.4f}</td></tr>")
    fig.update_xaxes(title_text="alpha [deg]", row=1, col=1)
    fig.update_xaxes(title_text="CL", row=1, col=2)
    fig.update_xaxes(title_text="alpha [deg]", row=1, col=3)
    fig.update_layout(height=520, template="plotly_white", title=f"Polaires 2D à Re {body['re']:.3g} : CFD (points) "
                      "contre XFOIL n_crit 1 (traits : xtr 3 % plein, 5 % tirets, 10 % pointillés)")
    html = ["<html><head><meta charset='utf-8'><title>Polaires CFD 2D</title></head><body style='font-family:sans-serif'>",
            fig.to_html(full_html=False, include_plotlyjs="cdn"),
            "<h3>Points CFD (CD en 1e-4 ; XFOIL xtr 3 % interpolé au même CL)</h3><table border=1 cellpadding=4 "
            "style='border-collapse:collapse'><tr><th>section</th><th>alpha</th><th>niveaux</th><th>convergé</th>"
            "<th>CL</th><th>CD CFD</th><th>CD XFOIL 3 %</th><th>écart</th><th>CM</th></tr>", *table, "</table>",
            f"<p>Généré le {dt.datetime.now():%Y-%m-%d %H:%M} par scripts/cfd2d_polars.py figures.</p></body></html>"]
    (run_dir / "figures").mkdir(exist_ok=True)
    (run_dir / "figures" / "polars.html").write_text("\n".join(html), encoding="utf-8")
    print(f"written {run_dir / 'figures' / 'polars.html'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--run-dir", required=True)
    r.add_argument("--sections", nargs="+", default=["MC2", "opt003"])
    r.add_argument("--alphas", nargs="+", type=float, default=list(DEFAULT_ALPHAS))
    r.add_argument("--workers", type=int, default=4)
    r.add_argument("--np", type=int, default=4)
    r.add_argument("--deadline", default="2026-11-11 07:30")
    f = sub.add_parser("figures")
    f.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    return run(args) if args.cmd == "run" else figures(args)


if __name__ == "__main__":
    sys.exit(main())
