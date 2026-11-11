"""Commands of the 3D CFD level (src/cfd_level.py, plan of 2026-11-11).

  python scripts/cfd_level.py prepare --name L3_MC2_moyen_yawm1 --yaw -1 [--section-json best_section.json] [--gpu]
      launch folder of one point for Antoine's validation (mesh, check.sh, PARAMETRES.md, y+, views); nothing solved
  python scripts/cfd_level.py zip --name L3_MC2_moyen_yawm1
      the launch folder as a zip next to it (to copy to another computer)
Launch folders: <bdToolbox>/tmp/bdfoil_v1/cfd3d/lancement/<name> (as Antoine's c9_sweep_launch).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import cfd_level  # noqa: E402

LAUNCH = Path(cfd_level.os.environ["BDTOOLBOX_ROOT"]) / "tmp" / "bdfoil_v1" / "cfd3d" / "lancement"


def prepare(args) -> int:
    folder = LAUNCH / args.name
    planform = None
    if args.section_json:
        best = json.loads(Path(args.section_json).read_text(encoding="utf-8"))
        weights = np.concatenate([best["kulfan_upper"], best["kulfan_lower"]])
        planform = cfd_level.write_section(cfd_level.kulfan_coordinates(weights), args.name,
                                           LAUNCH.parent / "sections" / args.name)
    cfd_level.log(f"{args.name}: swept mesh {cfd_level.MESH_LEVEL} at {cfd_level.SPEED} m/s, yaw {args.yaw:g} deg, "
                  f"check.sh in WSL")
    rec = cfd_level.prepare(folder, planform, args.yaw, gpu=args.gpu, purpose=args.purpose, nproc=args.np)
    mesh = rec.get("mesh", {})
    cfd_level.log(f"{folder}: done ({json.dumps(mesh.get('checkMesh', {}), default=str)[:300]})")
    return 0


def serve(args) -> int:
    """The server of src/cfd_queue.py: one request at a time, oldest first, until the deadline or a STOP file in the
    queue folder. A point = launch folder (mesh, check.sh, PARAMETRES.md, y+ map) in <queue>/cases/<id>, run.sh in WSL
    with the convergence monitor of src/cfd_level.run, then done/<id>.json."""
    import datetime as dt
    import time
    queue = Path(args.queue)
    deadline = dt.datetime.fromisoformat(args.deadline)
    reference = json.loads(Path(args.reference).read_text(encoding="utf-8")) if args.reference else \
        {"rho": None, "sref": None}
    env = cfd_level.foam_env(args.gpu)
    cfd_level.log(f"serve {queue}: GPU {args.gpu}, np {args.np}, deadline {deadline}, reference {reference}")
    while dt.datetime.now() < deadline and not (queue / "STOP").exists():
        done = {p.stem for p in (queue / "done").glob("*.json")}
        todo = [p for p in sorted((queue / "requests").glob("*.json")) if p.stem not in done]
        if not todo:
            time.sleep(30)
            continue
        req = json.loads(todo[0].read_text(encoding="utf-8"))
        rid, yaw = req["id"], float(req["yaw"])
        out = {"id": rid, "x": req["x"], "yaw": yaw, "ok": False, "error": ""}
        t0 = time.time()
        key = cfd_level.cache_key(req["weights"], yaw)
        cached = cfd_level.CACHE / f"{key}.json"
        if cached.exists():                                 # computed before (e.g. Antoine's validation point)
            out.update({k: v for k, v in json.loads(cached.read_text(encoding="utf-8")).items()
                        if k not in ("id", "x")}, cache=key)
            (queue / "done" / f"{rid}.json").write_text(json.dumps(out, indent=1, default=float) + "\n",
                                                        encoding="utf-8")
            cfd_level.log(f"{rid}: from the cache {cached.name}")
            continue
        try:
            name = f"L3_{queue.parent.name[:15]}_{rid}"
            folder = queue / "cases" / rid
            planform = cfd_level.write_section(cfd_level.kulfan_coordinates(np.asarray(req["weights"])), name,
                                               queue / "sections" / rid) if req.get("weights") else None
            cfd_level.log(f"{rid}: prepare ({req.get('reason', '')})")
            rec = cfd_level.prepare(folder, planform, yaw, gpu=args.gpu, purpose=f"point {rid} de la file CFD : "
                                    f"{req.get('reason', '')}", nproc=args.np, views=False)
            case_lx = rec["meta"]["mesh_linux"]
            left = (deadline - dt.datetime.now()).total_seconds()
            r = cfd_level.run(case_lx, env, timeout_s=min(args.hours * 3600, max(left - 600, 600)))
            out.update(r)
            if r["done"] != "rc=0":
                raise RuntimeError(f"run.done {r['done']}")
            red = cfd_level.reduce(case_lx, env, yaw, reference)
            out.update(red, ok=True)
        except Exception as exc:                            # noqa: BLE001
            out["error"] = f"{type(exc).__name__}: {exc}"[:600]
        out["wall_s"] = time.time() - t0
        if out["ok"]:
            cfd_level.CACHE.mkdir(parents=True, exist_ok=True)
            cached.write_text(json.dumps(out, indent=1, default=float) + "\n", encoding="utf-8")
        tmp = queue / "done" / f"{rid}.tmp"
        tmp.write_text(json.dumps(out, indent=1, default=float) + "\n", encoding="utf-8")
        tmp.replace(queue / "done" / f"{rid}.json")
        cfd_level.log(f"{rid}: {'ok' if out['ok'] else out['error']} in {out['wall_s'] / 3600:.2f} h, "
                      f"variation {out.get('variation')}, y {out.get('y')}")
    cfd_level.log("serve: end")
    return 0


def run_folder(args) -> int:
    """Solve a validated launch folder (prepare) in WSL with the convergence monitor, then put the point in the cache
    (src/cfd_level.CACHE) so that the queue does not compute it again."""
    import time
    folder = LAUNCH / args.name
    rec = json.loads((folder / "launch.json").read_text(encoding="utf-8"))
    case_lx = rec["meta"]["mesh_linux"]
    reference = json.loads(Path(args.reference).read_text(encoding="utf-8"))
    best = json.loads(Path(args.section_json).read_text(encoding="utf-8"))
    weights = np.concatenate([best["kulfan_upper"], best["kulfan_lower"]])
    env = cfd_level.foam_env(args.gpu)
    cfd_level.log(f"{args.name}: run.sh in {case_lx} (GPU {args.gpu})")
    t0 = time.time()
    out = {"name": args.name, "yaw": args.yaw, "weights": weights.tolist(), "ok": False, "error": ""}
    try:
        r = cfd_level.run(case_lx, env, timeout_s=args.hours * 3600)
        out.update(r)
        if r["done"] != "rc=0":
            raise RuntimeError(f"run.done {r['done']}")
        out.update(cfd_level.reduce(case_lx, env, args.yaw, reference), ok=True)
    except Exception as exc:                                # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {exc}"[:600]
    out["wall_s"] = time.time() - t0
    (folder / "result.json").write_text(json.dumps(out, indent=1, default=float) + "\n", encoding="utf-8")
    if out["ok"]:
        cfd_level.CACHE.mkdir(parents=True, exist_ok=True)
        (cfd_level.CACHE / f"{cfd_level.cache_key(weights, args.yaw)}.json").write_text(
            json.dumps(out, indent=1, default=float) + "\n", encoding="utf-8")
    cfd_level.log(f"{args.name}: {'ok' if out['ok'] else out['error']} in {out['wall_s'] / 3600:.2f} h, "
                  f"stop {out.get('stop')}, {out.get('n_iter')} it, variation {out.get('variation')}, y {out.get('y')}")
    return 0


def zip_folder(args) -> int:
    folder = LAUNCH / args.name
    out = shutil.make_archive(str(folder), "zip", root_dir=folder.parent, base_dir=folder.name)
    cfd_level.log(f"{out}: {Path(out).stat().st_size / 1e6:.0f} MB")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--name", required=True)
    p.add_argument("--yaw", type=float, default=0.0)
    p.add_argument("--section-json", default=None, help="best_section.json of a run (default: MC2)")
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--np", type=int, default=cfd_level.NPROC)
    p.add_argument("--purpose", default="niveau 3 (CFD 3D) de l'optimisation de section Ixx/v >= 0.95, point de "
                                        "validation : MC2 au lacet médian de ses états VPP, 8.49 m/s")
    p.set_defaults(func=prepare)
    s = sub.add_parser("serve")
    s.add_argument("--queue", required=True, help="<run>/cfd of run_real_vpp --cfd-level")
    s.add_argument("--deadline", required=True, help="ISO time, e.g. 2026-11-12T08:15")
    s.add_argument("--reference", required=True, help="JSON {rho, sref} of the AVL level (calibration)")
    s.add_argument("--gpu", action="store_true")
    s.add_argument("--np", type=int, default=cfd_level.NPROC)
    s.add_argument("--hours", type=float, default=4.0, help="time limit of one point")
    s.set_defaults(func=serve)
    r = sub.add_parser("run")
    r.add_argument("--name", required=True)
    r.add_argument("--yaw", type=float, required=True)
    r.add_argument("--section-json", required=True, help="the section of the folder (cache key)")
    r.add_argument("--reference", required=True)
    r.add_argument("--gpu", action="store_true")
    r.add_argument("--hours", type=float, default=4.0)
    r.set_defaults(func=run_folder)
    z = sub.add_parser("zip")
    z.add_argument("--name", required=True)
    z.set_defaults(func=zip_folder)
    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
