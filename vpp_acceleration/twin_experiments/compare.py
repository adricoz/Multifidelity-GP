"""
Verdict of the digital twin: the TRUE speed (VPP on the truth of twin.py, same conditions) of
  - the optima of the direct search (results/reference.json): what there is to find;
  - the section found by the method on the twin (last run of twin_runs/) and its local optima;
  - the sections of the truth run and, when given, of other real runs (--runs) and extra sections
    (--sections file.json: {name: scores}), e.g. the upper-confidence maximisers of the certificate.
Every section is solved twice (fixed and generic guesses, then from the two fastest states found by
any section at the same condition) and the fastest equilibrium is kept (the VPP solve is path
dependent, scripts/profile_analysis.robust_solve). Speeds are compared with the anchor (the best
reference optimum) on the conditions solved for both.

Writes results/compare.json, figures/*.html and README.md (French).
    ..\\.venv-mds\\Scripts\\python twin_experiments\\compare.py --workers 12
"""
import argparse
import datetime as dt
import json
import os
from pathlib import Path

for _variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_variable, "2")

import numpy as np  # noqa: E402

from twin import (HERE, TRUTH_RUN, VPP_DIR, SearchDomain, TwinTruth, make_pool,  # noqa: E402
                  run_sections)

TOLERANCE_KT = 0.02                    # criterion fixed on 2026-11-12, before the runs


def log(message: str) -> None:
    print(f"[{dt.datetime.now():%H:%M:%S}] {message}", flush=True)


def last_twin_run(name: str = "twin_method"):
    runs = sorted(p for p in (HERE / "twin_runs").glob(f"*_{name}") if p.is_dir())
    return runs[-1] if runs else None


def collect(args, domain) -> dict:
    """name -> {"scores", "group"} of every section to judge."""
    sections = {}

    def add(name, scores, group):
        scores = np.asarray(scores, dtype=float)
        if scores.shape == (domain.k,) and np.all(np.isfinite(scores)):
            sections[name] = {"scores": scores, "group": group}

    reference = HERE / "results" / f"{args.reference}.json"
    if reference.exists():
        for start in json.loads(reference.read_text(encoding="utf-8"))["starts"]:
            add(f"reference_from_{start['name']}", start["scores"], "reference")
    twin_run = Path(args.twin_run) if args.twin_run else last_twin_run()
    if twin_run is not None and (twin_run / "results" / "best_section.json").exists():
        for name, scores in run_sections(twin_run).items():
            add(f"method_{name}", scores, "method" if name != "current_section" else "start")
    elif twin_run is not None and (twin_run / "results").exists():
        # run stopped before its end: the last incumbent of results/updates.json, if written
        updates = twin_run / "results" / "updates.json"
        if updates.exists():
            add("method_best", json.loads(updates.read_text(encoding="utf-8"))[-1]["incumbent"],
                "method")
    for name, scores in run_sections(Path(args.truth_run)).items():
        add(f"truth_run_{name}", scores, "truth_run")
    for path in args.runs or []:
        path = Path(path) if Path(path).is_absolute() else VPP_DIR / path
        if (path / "results" / "best_section.json").exists():
            tag = "nocfd" if "nocfd" in path.name else path.name[:16]
            for name, scores in run_sections(path).items():
                if name != "current_section":
                    add(f"{tag}_{name}", scores, tag)
    for path in args.sections or []:
        path = Path(path)
        if path.exists():
            for name, scores in json.loads(path.read_text(encoding="utf-8")).items():
                add(name, scores, "certificate")
    return sections, twin_run


def solve_all(pool, scores: list, conditions) -> list:
    """Fastest converged state [section][condition] (module docstring)."""
    from src.vpp_optimize import GENERIC_GUESSES  # noqa: PLC0415
    n, ratios = len(conditions), [1.0] * len(scores)
    first = pool.solve(scores, ratios, conditions, extra_guesses=[[GENERIC_GUESSES] * n] * len(scores))
    found = []
    for j in range(n):
        states = sorted((row[j] for row in first if row[j]["success"]), key=lambda s: -s["speed"])
        found.append([s["unknowns"] for s in states[:2]])
    second = pool.solve(scores, ratios, conditions, extra_guesses=[found] * len(scores))
    best = []
    for row_1, row_2 in zip(first, second):
        row = []
        for a, b in zip(row_1, row_2):
            candidates = [s for s in (a, b) if s["success"]]
            row.append(max(candidates, key=lambda s: s["speed"]) if candidates else a)
        best.append(row)
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--truth-run", default=str(TRUTH_RUN))
    parser.add_argument("--twin-run", default=None, help="default: the last twin_runs/*_twin_method")
    parser.add_argument("--reference", default="reference")
    parser.add_argument("--runs", nargs="*", help="other finished runs judged on the truth")
    parser.add_argument("--sections", nargs="*", help="JSON files {name: scores}")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--n-conditions", type=int, default=None, help="first conditions only (tests)")
    parser.add_argument("--out", default="compare")
    args = parser.parse_args()

    truth_run = Path(args.truth_run)
    truth, domain = TwinTruth.load(truth_run), SearchDomain(truth_run)
    sections, twin_run = collect(args, domain)
    names = list(sections)
    scores = [sections[name]["scores"] for name in names]
    conditions, weights = domain.conditions, domain.weights
    if args.n_conditions:
        conditions, weights = conditions[:args.n_conditions], weights[:args.n_conditions]
    log(f"{len(names)} sections, {len(conditions)} conditions, {args.workers} workers")
    pool = make_pool(args.workers, HERE / "work" / args.out)
    pool.publish(truth)
    states = solve_all(pool, scores, conditions)
    pool.close()

    solved = np.array([[s["success"] for s in row] for row in states])
    speeds = np.array([[s["speed"] for s in row] for row in states])
    own = np.array([speeds[i, solved[i]] @ weights[solved[i]] / weights[solved[i]].sum()
                    if solved[i].any() else np.nan for i in range(len(names))])
    reference = [i for i, n in enumerate(names) if sections[n]["group"] == "reference"]
    pool_of_anchor = reference or list(range(len(names)))
    anchor = max(pool_of_anchor, key=lambda i: own[i] if np.isfinite(own[i]) else -np.inf)
    rows = []
    for i, name in enumerate(names):
        common = solved[i] & solved[anchor]
        w = weights * common
        gap = float((speeds[i, common] - speeds[anchor, common]) @ weights[common] / w.sum()) \
            if w.sum() > 0 else None
        rows.append({"name": name, "group": sections[name]["group"], "scores": scores[i].tolist(),
                     "speed_own_conditions": float(own[i]), "n_solved": int(solved[i].sum()),
                     "gap_to_anchor_kt": gap, "probability_compared": float(w.sum() / weights.sum()),
                     "distance_to_anchor": float(np.linalg.norm(scores[i] - scores[anchor])),
                     "min_margin": float(domain.constraints(scores[i]).min()),
                     "admissible": bool(domain.admissible(scores[i])[0]),
                     "speeds": [float(v) if ok else None for v, ok in zip(speeds[i], solved[i])]})
    method = [r for r in rows if r["name"] == "method_best"]
    verdict = None
    if method and reference and method[0]["gap_to_anchor_kt"] is not None:
        regret = -method[0]["gap_to_anchor_kt"]
        verdict = {"regret_kt": regret, "tolerance_kt": TOLERANCE_KT, "passed": bool(regret <= TOLERANCE_KT),
                   "distance_scores": method[0]["distance_to_anchor"]}
    out = {"truth_run": truth_run.name, "twin_run": twin_run.name if twin_run else None,
           "anchor": names[anchor], "conditions": [list(c) for c in conditions],
           "weights": weights.tolist(), "verdict": verdict, "sections": rows}
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / f"{args.out}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    log(f"anchor {names[anchor]}; verdict {verdict}")
    for row in sorted(rows, key=lambda r: -(r["gap_to_anchor_kt"] if r["gap_to_anchor_kt"] is not None
                                            else -9)):
        log(f"  {row['name']:34s} {row['gap_to_anchor_kt'] if row['gap_to_anchor_kt'] is None else round(row['gap_to_anchor_kt'], 4)} kt, "
            f"distance {row['distance_to_anchor']:.2f}, {row['n_solved']} solved")


if __name__ == "__main__":
    main()
