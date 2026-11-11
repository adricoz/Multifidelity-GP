"""
Fidelity level "cfd3d" of the section optimisation (plan of 2026-11-11, decisions of Antoine): the C-foil in 3D RANS on
Antoine's swept structured mesh (bdfoil_v1 ``sweep3d``, level ``medium``, 2.54 M cells), one point = one section at one
yaw, at the fixed speed of the other levels (8.49 m/s, Re 5e6 on the 0.70 m chord, foil_codes).

bdfoil_v1 is NOT modified: the builders of ``case3d`` are wrapped in this process (as scripts/cfd_sections.use_gpu).
- Attitude: the swept mesh is built at the neutral attitude, so cant = rake = 0 (decision of 2026-11-11). The yaw is
  the direction of the freestream: bdfoil_v1 places the foil by R = Rz(-yaw) (geometry.rotation_matrix: +yaw turns +x
  towards -y) in a flow along -x; the same relative flow on the unrotated foil is U = Rz(+yaw) (-V, 0, 0) =
  (-V cos yaw, -V sin yaw, 0), parallel to the hull symmetry plane z = 0, and the forces and moments of the run are
  brought back to the boat frame by Rz(-yaw).
- Settings of Antoine's case C9_SW_moyen_v2box3_U099: U relaxation 0.99 (k, omega 0.7), 200 upwind iterations then
  second order, pressure to relTol 0.1 with DICGaussSeidel (case3d._sweep_fv_solution). With ``gpu``: the pressure on
  PETSc FCG + AmgX (GPU-CFD weekend), np 4.
- Convergence (Antoine, 2026-11-11): relative variation of Cd, Cl and Cs below 0.2 %, at most 1500 iterations. The
  variation is measured by ``Monitor`` on postProcessing/forceCoeffs1 (OpenFOAM's runTimeControl ``average`` tests
  the change of a running mean, not this): mean of the last WINDOW iterations against the mean of the WINDOW before,
  relative to the last mean, all three below CONV_REL, checked from MIN_SECOND_ORDER second-order iterations on; then
  ``stopAt writeNow`` is set in system/controlDict (what Antoine did by hand on 2026-11-05), the run writes, reconstructs
  and ends normally.
- Outputs: the order of foil_codes.OUTPUT_NAMES; Fx..Mz are bdfoil_v1's foil.py coefficients (force / (q Sref),
  moment / (q Sref) in metres, about the origin of the planform, BD frame) times foil_codes.CONVERSION, with the q and
  Sref of the AVL level (``reference`` from the calibration of scripts/cfd_level.py calibrate). Cpmin is not computed
  here (NaN): the caller takes the one of the level below at the same point.
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

BDFOIL = Path(os.environ.get("BDFOIL_V1_DIR", r"C:\Users\SIM\Desktop\Code-Adri\bdfoil_v1"))
os.environ.setdefault("BDTOOLBOX_ROOT", r"C:\Users\SIM\Outils - banulsdesign\bdToolbox")
for _p in (BDFOIL / "src", BDFOIL / "scripts" / "openfoam"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

SPEED = 8.49                       # m/s, foil_codes (Re 5e6 on the chord, seawater 15 deg C)
MESH_LEVEL = "medium"
N_ITER_MAX = 1500                  # Antoine, 2026-11-11
N_ITER_UPWIND = 200
CONV_REL = 2e-3                    # Antoine, 2026-11-11: 0.2 %
WINDOW = 100
MIN_SECOND_ORDER = 2 * WINDOW      # second-order iterations before the test
U_RELAX = 0.99
NPROC = 4
SREF, CREF = 2.274974, 0.7         # AVL's references of the C9 case (forceCoeffs only)
GPU_BASHRC = "/home/bdcfd/gpu_solvers/env_gpu.sh"
CASE = BDFOIL / "cases" / "cfoil_r9800_fsi.json"
MC2_XF = BDFOIL / "tests" / "data" / "cfoil_r9800" / "MC2_60_6101_1_re5M.xf"
MC2_CSV = BDFOIL / "tests" / "data" / "cfoil_r9800" / "CFoil_R9800.csv"
# fixed noise of the level in the co-kriging, relative to |y| (scripts/cfd_level.py gci, 2026-11-11: medium against
# fine of Antoine's U099 cases 0.07-0.1 %, coarse against medium 0.1-0.5 %, none stopped by its runTimeControl;
# plus the 0.2 % convergence criterion)
NOISE_REL = np.array([5e-3, 2.5e-3, 2.5e-3, 2.5e-3, 2.5e-3, 2.5e-3, 0.0])

_STATE = {"yaw": 0.0, "gpu": False, "wrapped": False}
# finished points by section and yaw (scripts/cfd_level.py run / serve): a point computed once (e.g. the validation
# point of Antoine) is not computed again by the queue
CACHE = Path(__file__).resolve().parents[1] / "data" / "cfd_level" / "points"


def cache_key(weights, yaw: float) -> str:
    """Key of a point: the 16 Kulfan weights (10 decimals), the yaw (0.01 deg) and the settings of this module."""
    import hashlib
    blob = json.dumps({"w": np.round(np.asarray(weights, float), 10).tolist(), "yaw": round(float(yaw), 2),
                       "settings": [SPEED, MESH_LEVEL, N_ITER_MAX, N_ITER_UPWIND, CONV_REL, WINDOW, U_RELAX]})
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


def log(msg: str) -> None:
    print(f"[{dt.datetime.now():%H:%M:%S}] {msg}", flush=True)


# ------------------------------------------------------------------------------------------------ section and study
def write_section(coordinates: np.ndarray, name: str, folder: Path) -> Path:
    """``<folder>/<name>.xf`` in the order of Antoine's MC2 file (TE, lower side, LE, upper side) and a copy of his
    planform CSV with every station on it. ``coordinates``: (n, 2), any closed order."""
    xy = np.asarray(coordinates, dtype=float)
    i_le = int(xy[:, 0].argmin())
    if xy[:i_le, 1].mean() > 0:                 # upper side first (Selig): reverse
        xy = xy[::-1]
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{name}.xf").write_text(f"{name}\n" + "".join(f"{x:.6f}    {y:.6f}\n" for x, y in xy),
                                       encoding="utf-8")
    lines = MC2_CSV.read_text(encoding="utf-8-sig").splitlines()
    col = lines[0].split(",").index("Section")
    rows = [lines[0]]
    for ln in lines[1:]:
        parts = ln.split(",")
        if len(parts) > col:
            parts[col] = f"{name}.xf"
        rows.append(",".join(parts))
    (folder / "planform.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return folder / "planform.csv"


def study_for(planform_csv: Path | None):
    """Antoine's C9 case without its structure block, every station on the section of ``planform_csv`` (None: MC2)."""
    from bdfoil_v1.study import load_study
    study = dataclasses.replace(load_study(CASE), structure=None)
    if planform_csv is None:
        return study
    pf = dataclasses.replace(study.appendage.planform, csv=str(planform_csv), sha256=None)
    return dataclasses.replace(study, appendage=dataclasses.replace(study.appendage, planform=pf))


def kulfan_coordinates(weights: np.ndarray) -> np.ndarray:
    """Coordinates of a section of 16 Kulfan weights, as the levels 1-2 see it (foil_codes.FoilCodes.section)."""
    from src.foil_codes import FoilCodes
    return np.asarray(FoilCodes.section(weights).coordinates, dtype=float)


# ------------------------------------------------------------------------------------------------ case set-up
def settings(nu: float, rho: float, nproc: int = NPROC):
    from bdfoil_v1.solvers.openfoam import case3d
    return case3d.Settings3D(speed=SPEED, nu=nu, rho=rho, chord=0.7, sref=SREF, cref=CREF, nproc=nproc,
                             n_iter_max=N_ITER_MAX, n_iter_upwind=N_ITER_UPWIND)


def wrap_case3d(yaw: float, gpu: bool) -> None:
    """Point the wrapped builders of case3d at this yaw and solver (module docstring). Idempotent."""
    from bdfoil_v1.solvers.openfoam import case3d
    from bdfoil_v1.solvers.openfoam.dicts import Quoted, Raw, uniform
    _STATE.update(yaw=float(yaw), gpu=bool(gpu))
    if _STATE["wrapped"]:
        return
    fields0, fv0, cd0 = case3d._fields, case3d._sweep_fv_solution, case3d._control_dict   # noqa: SLF001

    def fields(s, boundary="box"):
        d = fields0(s, boundary)
        if boundary == "freestream":
            psi = math.radians(_STATE["yaw"])
            u = uniform((-s.speed * math.cos(psi), -s.speed * math.sin(psi), 0.0))
            body = d["U"][1]
            body["internalField"] = u
            for patch in ("farfield", "outlet"):
                body["boundaryField"][patch]["freestreamValue"] = u
        return d

    def fv(s):
        d = fv0(s)
        d["relaxationFactors"]["equations"]["U"] = U_RELAX
        if _STATE["gpu"]:
            p = d["solvers"]["p"]
            d["solvers"]["p"] = {"solver": "petsc", "tolerance": p["tolerance"], "relTol": p["relTol"],
                                 "petsc": {"options": {"ksp_type": "fcg", "mat_type": "aijcusparse",
                                                       "vec_type": "cuda", "pc_type": "amgx",
                                                       "pc_amgx_strength_threshold": Quoted("0.8")},
                                           "caching": {"matrix": {"update": "always"},
                                                       "preconditioner": {"update": "adaptive"}}}}
        return d

    def cd(s):
        d = cd0(s)
        if _STATE["gpu"]:
            d["libs"] = [Raw("petscFoam")]
        return d

    case3d._fields, case3d._sweep_fv_solution, case3d._control_dict = fields, fv, cd   # noqa: SLF001
    _STATE["wrapped"] = True


def foam_env(gpu: bool):
    from bdfoil_v1.solvers.openfoam import bridge
    return bridge.detect(host="local", bashrc=GPU_BASHRC) if gpu else bridge.detect(host="local")


def prepare(folder: Path, planform_csv: Path | None, yaw: float, *, gpu: bool, purpose: str,
            nproc: int = NPROC, views: bool = True) -> dict:
    """Launch folder of one point (Antoine's c9_sweep_launch at SPEED and ``yaw``): swept mesh, case written and
    ``check.sh`` run in WSL (``<runs>/lancement/<folder name>``), PARAMETRES.md, y+ map and, with ``views``, the
    ParaView views; the folder is copied to ``folder``. Nothing is solved. Returns the record of launch.json."""
    import c9_sweep_launch as c9s
    from bdfoil_v1.model import Attitude
    from bdfoil_v1.session import Session
    from bdfoil_v1.solvers.openfoam import geom3d, launch3d, sweep3d
    import subprocess
    wrap_case3d(yaw, gpu)
    study = study_for(planform_csv)
    session = Session(study)
    st = settings(session.fluid.nu, session.fluid.rho, nproc)
    pl = session.planform
    meta = {"title": folder.name, "purpose": purpose, "heading": "Dossier de lancement (maillage balayé, niveau 3 "
                                                                   "de l'optimisation de section)",
            "case_file": str(CASE), "case_sha256": c9s.sha256(CASE),
            "case_note": "bloc structure retiré (foil rigide) ; section remplacée par celle du point",
            "planform": str(planform_csv or MC2_CSV), "planform_sha256": pl.source_sha256,
            "sections": [(str(r.path), r.sha256) for r in pl.sections],
            "attitude": dataclasses.asdict(Attitude(cant=0.0, rake=0.0, yaw=0.0, sink=0.0, speed=SPEED)),
            "attitude_note": f"maillage neutre ; lacet {yaw:g}° par la direction de l'écoulement amont "
                             "U = Rz(+lacet)(-V, 0, 0), efforts ramenés au repère bateau par Rz(-lacet)",
            "fluid": {"name": f"eau de mer {study.fluid.temperature:g} °C, salinité {study.fluid.salinity:g} g/kg",
                      "note": "`bdfoil_v1.fluid` (ITTC)"},
            "git": "copie bdfoil_v1 du 27/10 (non modifiée), enveloppes dans vpp_acceleration/src/cfd_level.py",
            "run_host": "ce PC (WSL)" + (", GPU" if gpu else "")}
    res = launch3d.prepare_sweep_launch(folder, session, settings=st, options=sweep3d.SweepOptions(level=MESH_LEVEL),
                                        meta=meta, env=foam_env(gpu))   # run.sh sources the GPU environment
    info, mesh = res["info"], res["mesh"]
    vues = folder / "vues"
    vues.mkdir(exist_ok=True)
    shown = [(f"vues/{p}", c) for p, c in c9s.yplus_map(mesh["yplus"], vues, folder.name)]
    if views and c9s.PVPYTHON.exists():
        v, t, skin = geom3d.foil_cfd_surface(session, Attitude(cant=0.0, rake=0.0, yaw=0.0, sink=0.0, speed=SPEED))
        spec = launch3d.mesh_shots(skin, v, t, label=folder.name, n_layers=1)
        spec["shots"] = [s for s in spec["shots"] if not s["name"].startswith("couches_")]
        (vues / "shots.json").write_text(json.dumps(spec, indent=1), encoding="utf-8")
        out = subprocess.run([str(c9s.PVPYTHON), "--force-offscreen-rendering", str(c9s.HERE / "pv_mesh3d.py"),
                              str(folder / "case.foam"), str(vues / "shots.json"), str(vues)],
                             capture_output=True, text=True)
        if out.returncode:
            log(f"pvpython rc={out.returncode}: {out.stderr[-500:]}")
        shown += [(f"vues/{s['name']}.png", s["title"].split(" : ", 1)[-1]) for s in spec["shots"]
                  if (vues / f"{s['name']}.png").exists()]
    rec = json.loads((folder / "launch.json").read_text(encoding="utf-8"))
    rec["meta"].update(views=shown, yaw=yaw, gpu=gpu, convergence=f"variation relative de Cd, Cl, Cs < {CONV_REL:.1%} "
                       f"(moyenne des {WINDOW} dernières itérations contre les {WINDOW} d'avant), plafond {N_ITER_MAX}")
    launch3d.write_sweep_sheet(folder, rec["meta"], st, info, mesh)
    (folder / "launch.json").write_text(json.dumps(rec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
                                        newline="\n")
    return rec


# ------------------------------------------------------------------------------------------------ run
def relative_variation(coef, window: int = WINDOW) -> dict:
    """|mean(last window) - mean(window before)| / |mean(last window)| for Cd, Cl, Cs of a forceCoeffs history."""
    out = {}
    for c in ("Cd", "Cl", "Cs"):
        v = coef[c].to_numpy(dtype=float)
        if v.size < 2 * window:
            out[c] = float("inf")
            continue
        a, b = v[-window:].mean(), v[-2 * window:-window].mean()
        out[c] = float(abs(a - b) / max(abs(a), 1e-12))
    return out


# sed, not foamDictionary: bridge.run_script does not source OpenFOAM, foamDictionary was silently missing (2026-11-11)
STOP_CMD = "cd {case} && sed -i 's/^stopAt .*/stopAt          writeNow;/' system/controlDict && grep -q '^stopAt *writeNow' system/controlDict\n"


def converged(coef, n_upwind: int = N_ITER_UPWIND) -> tuple[bool, dict]:
    """The convergence test of the module docstring on a forceCoeffs history (column Time = iteration)."""
    second = coef[coef["Time"] > n_upwind]
    var = relative_variation(second)
    ok = len(second) >= MIN_SECOND_ORDER and all(v < CONV_REL for v in var.values())
    return ok, var


def run(case_lx: str, env, *, timeout_s: float, poll_s: float = 60.0, progress=log) -> dict:
    """Start ``run.sh`` of the WSL case ``case_lx`` (prepared by ``prepare``), stop it at convergence, wait for
    ``run.done``. Returns {"done", "stop", "n_iter", "variation", "wall_s"}."""
    from bdfoil_v1.solvers.openfoam import bridge, results
    work = env.to_unc(case_lx)
    for f in ("run.done", "run.pid", "run.status"):
        (work / f).unlink(missing_ok=True)
    # bdfoil_v1's launcher: wsl.exe stays open for the whole run (a script started in the background of a short wsl
    # call is killed when that call ends, 2026-11-11)
    proc = bridge.launch(env, f"{case_lx}/run.sh")
    t0, stop, var, n_iter = time.time(), "", {}, 0
    while not (work / "run.done").exists():
        if proc.poll() is not None:                 # wsl.exe ended without run.done: the run died
            time.sleep(5)
            if not (work / "run.done").exists():
                break
        time.sleep(poll_s)
        try:
            coef = results.read_history(work, "forceCoeffs1", "coefficient.dat")
        except (FileNotFoundError, ValueError):
            coef = None
        if coef is not None and len(coef):
            n_iter = int(coef["Time"].iloc[-1])
            ok, var = converged(coef)
            if ok and not stop:
                stop = "converged"
                bridge.run_script(env.distro, STOP_CMD.format(case=case_lx), timeout=60)
                progress(f"{case_lx}: converged at {n_iter} it ({', '.join(f'{k} {v:.2%}' for k, v in var.items())})")
        if time.time() - t0 > timeout_s and not stop:
            stop = "timeout"
            bridge.run_script(env.distro, STOP_CMD.format(case=case_lx), timeout=60)
            progress(f"{case_lx}: time limit at {n_iter} it, stopped")
    done = (work / "run.done").read_text().strip() if (work / "run.done").exists() else "missing (wsl.exe ended)"
    return {"done": done, "stop": stop or ("n_iter_max" if n_iter >= N_ITER_MAX else "ended"), "n_iter": n_iter,
            "variation": var, "wall_s": time.time() - t0}


def reduce(case_lx: str, env, yaw: float, reference: dict) -> dict:
    """Forces and moments of a finished case, in the boat frame (Rz(-yaw)), and the outputs of the level."""
    from bdfoil_v1.solvers.openfoam import foil3d, results
    from src.foil_codes import CONVERSION, OUTPUT_NAMES
    work = env.to_unc(case_lx)
    force = results.read_history(work, "forces1", "force.dat").iloc[-1]
    moment = results.read_history(work, "forces1", "moment.dat").iloc[-1]
    psi = math.radians(yaw)
    rot = np.array([[math.cos(psi), math.sin(psi), 0.0], [-math.sin(psi), math.cos(psi), 0.0], [0.0, 0.0, 1.0]])
    f = rot @ np.array([float(force[f"total_{c}"]) for c in "xyz"])
    m = rot @ np.array([float(moment[f"total_{c}"]) for c in "xyz"])
    q_sref = 0.5 * reference["rho"] * SPEED ** 2 * reference["sref"]
    raw = np.concatenate([f, m]) / q_sref
    y = np.full(len(OUTPUT_NAMES), np.nan)
    y[:6] = np.asarray(CONVERSION["sign"]) * np.asarray(CONVERSION["factor"]) * raw
    coef = results.read_history(work, "forceCoeffs1", "coefficient.dat")
    ok, var = converged(coef)
    yp = results.read_history(work, "yPlus1", "yPlus.dat")
    yp = yp[yp["patch"] == "foil"]
    return {"y": y.tolist(), "F_boat": f.tolist(), "M_boat": m.tolist(), "converged": bool(ok), "variation": var,
            "n_iter": int(coef["Time"].iloc[-1]), "yplus_mean": float(yp["average"].iloc[-1]) if len(yp) else None,
            "n_cells": foil3d._n_cells(work)}   # noqa: SLF001
