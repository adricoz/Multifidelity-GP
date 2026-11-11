"""
Asynchronous top fidelity level of the section optimisation (plan of 2026-11-11): 3D CFD points computed one at a time
by a separate process (scripts/cfd_level.py serve, on the GPU) while the loop goes on with the synchronous levels.

Files, in <run>/cfd/:
  requests/<id>.json  written by the loop: {"id", "x" (scores + Cant, Rake, Yaw; Cant = Rake = 0), "yaw", "weights"
                      (16 Kulfan weights), "reason", "created"}
  done/<id>.json      written by the server: {"id", "x", "y" (7 outputs, Cpmin NaN), "ok", "converged", "variation",
                      "n_iter", "wall_s", "error", ...}
  ingested.json       ids already added to the surrogate (by the loop)
  points.csv          one line per ingested point (x, y, level-2 values at x)

The loop side (CFDQueue, a hook of VPPSectionOptimizer, called before every batch):
- ingestion: a finished point enters if it converged (src/cfd_level.converged) or ended with every relative
  variation below ACCEPT_REL; the levels 1..L-1 are run at its x first if they are missing (nested design), its
  Cpmin is the one of level L-1 at x (the CFD gives none). Before N_ACTIVATE points the surrogate keeps its L-1 levels
  and the points wait; at N_ACTIVATE the surrogate is rebuilt with L levels (the same data plus the CFD points,
  hyperparameters refitted), afterwards each point is added to level L;
- request: one point at a time (a new request when the previous one is done). First the given start points (MC2 at
  its median VPP yaw); then, while the level is not active, the incumbent section at the median yaw of its VPP states;
  once it is active, the candidate of the loop (cant = rake = 0) with the largest weighted IMSE gain of level L
  (vpp_targeted.choose, the criterion of the synchronous levels).
- noise: the level-L GP of every output estimates its noise as the others (mfego CoKriging); NOISE_REL of
  src/cfd_level is reported against it in the log.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import uuid
from pathlib import Path

import numpy as np

N_ACTIVATE = 3
ACCEPT_REL = 1e-2          # a point stopped by the iteration cap or the time limit enters below 1 % of variation


def _save(path: Path, body) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(body, indent=1, default=float) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _load(path: Path, default):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


class CFDQueue:
    """The loop side of the module docstring. ``start_points``: list of (x, reason) requested first."""

    def __init__(self, folder, problem, cost: float, start_points=(), n_activate: int = N_ACTIVATE,
                 yaw_step: float = 0.5):
        self.folder = Path(folder)
        for sub in ("requests", "done"):
            (self.folder / sub).mkdir(parents=True, exist_ok=True)
        self.problem, self.cost = problem, float(cost)
        self.start_points = list(start_points)
        self.n_activate, self.yaw_step = int(n_activate), float(yaw_step)
        self.ingested = _load(self.folder / "ingested.json", [])
        self.waiting = []                       # (x, y) of the points before activation
        self.active = False

    # --- files ---
    def pending(self) -> list:
        done = {p.stem for p in (self.folder / "done").glob("*.json")}
        return [p for p in sorted((self.folder / "requests").glob("*.json")) if p.stem not in done]

    def request(self, x: np.ndarray, reason: str, log) -> str:
        k = self.problem.n_scores
        x = np.asarray(x, dtype=float).copy()
        x[k], x[k + 1] = 0.0, 0.0                               # swept mesh: neutral cant and rake
        x[k + 2] = round(x[k + 2] / self.yaw_step) * self.yaw_step
        rid = f"{dt.datetime.now():%m%d_%H%M%S}_{uuid.uuid4().hex[:4]}"
        weights = self.problem.weights(x[:k][None])[0]
        _save(self.folder / "requests" / f"{rid}.json",
              {"id": rid, "x": x.tolist(), "yaw": float(x[k + 2]), "weights": weights.tolist(), "reason": reason,
               "created": dt.datetime.now().isoformat(timespec="seconds")})
        log(f"CFD request {rid}: {reason}, yaw {x[k + 2]:g} deg")
        return rid

    # --- the hook ---
    def __call__(self, opt) -> None:
        self.ingest(opt)
        if not self.pending():
            self.next_request(opt)

    def ingest(self, opt) -> None:
        log = opt.run.log
        for path in sorted((self.folder / "done").glob("*.json")):
            if path.stem in self.ingested:
                continue
            rec = _load(path, {})
            self.ingested.append(path.stem)
            _save(self.folder / "ingested.json", self.ingested)
            var = rec.get("variation") or {}
            ok = bool(rec.get("ok")) and (rec.get("converged") or (var and max(var.values()) < ACCEPT_REL))
            y = np.asarray(rec.get("y", []), dtype=float)
            if not ok or y.size < 6 or not np.all(np.isfinite(y[:6])):
                log(f"CFD point {path.stem} not used: {rec.get('error') or rec.get('stop')} variation {var}")
                continue
            x = np.asarray(rec["x"], dtype=float)
            below = self.lower_levels(opt, x)
            if below is None:
                log(f"CFD point {path.stem} not used: a lower level failed at its x")
                continue
            y[6] = below[6]                                     # Cpmin: the level below (module docstring)
            self.write_point(path.stem, x, y, below)
            if self.active:
                opt.surrogate.add(opt.surrogate.n_levels, x, y)
                log(f"CFD point {path.stem} added to level {opt.surrogate.n_levels} "
                    f"({rec.get('n_iter')} it, {rec.get('wall_s', 0) / 3600:.1f} h)")
            else:
                self.waiting.append((x, y))
                log(f"CFD point {path.stem} kept ({len(self.waiting)}/{self.n_activate} before the level is used)")
                if len(self.waiting) >= self.n_activate:
                    self.activate(opt)

    def lower_levels(self, opt, x: np.ndarray):
        """Values of the top synchronous level at x, the missing synchronous levels run first (nested design)."""
        s = opt.surrogate
        top = opt.max_sync_level or s.n_levels
        values = None
        for level in range(1, top + 1):
            have = np.all(np.isclose(s.x_lists[level - 1], x), axis=1)
            if have.any():
                values = s.y_lists[level - 1][int(np.argmax(have))]
                continue
            y, infos = opt.runner.run_tasks(x[None], [level], tag="cfd_nested")
            if not np.all(np.isfinite(y[0])):
                opt.failed.append(x.copy())
                return None
            s.add(level, x, y[0])
            values = y[0]
        return np.asarray(values, dtype=float)

    def write_point(self, rid: str, x, y, below) -> None:
        path = self.folder / "points.csv"
        names = ["Fx", "Fy", "Fz", "Mx", "My", "Mz", "Cpmin"]
        if not path.exists():
            path.write_text("id," + ",".join(f"x{i}" for i in range(len(x))) + ","
                            + ",".join(f"{n}_cfd" for n in names) + "," + ",".join(f"{n}_below" for n in names)
                            + "\n", encoding="utf-8")
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(rid + "," + ",".join(f"{v:.10g}" for v in (*x, *y, *below)) + "\n")

    def activate(self, opt) -> None:
        """Rebuild the surrogate with one more level: the same data plus the waiting CFD points (module docstring)."""
        from src.vpp_targeted import TorsorCoKriging
        old = opt.surrogate
        new = TorsorCoKriging(old.bounds, costs=list(old.costs) + [self.cost], share_lengthscales=old.share_lengthscales,
                              seed=old.seed, n_models=old.n_models, n_threads=old.n_threads, n_outputs=old.n_outputs)
        for level in range(1, old.n_levels + 1):
            new.x_lists[level - 1] = old.x_lists[level - 1].copy()
            new.y_lists[level - 1] = old.y_lists[level - 1].copy()
        for x, y in self.waiting:
            new.add(new.n_levels, x, y)
        new.fit(refit=True)
        opt.surrogate, opt.fitted_size = new, sum(new.sizes)
        opt.states = None                                     # a new update with the new model at the next batch
        if opt.vpp_pool is not None:
            opt.vpp_pool.publish(new)
        self.active, self.waiting = True, []
        opt.run.log(f"CFD level active: surrogate rebuilt with {new.n_levels} levels, sizes {new.sizes}")

    def next_request(self, opt) -> None:
        k = self.problem.n_scores
        n_asked = len(list((self.folder / "requests").glob("*.json")))
        if n_asked < len(self.start_points):
            x, reason = self.start_points[n_asked]
            self.request(x, reason, opt.run.log)
            return
        if opt.incumbent_states is None or opt.states is None:     # no request before the next VPP update
            return
        asked = [np.asarray(json.loads(p.read_text(encoding="utf-8"))["x"])
                 for p in (self.folder / "requests").glob("*.json")]
        if not self.active:
            # the incumbent at the median yaw of its VPP states, else at the quartiles (never twice the same point)
            points = opt.state_points(opt.incumbent, opt.incumbent_states)
            for q in (50, 25, 75):
                yaw = round(float(np.nanpercentile(points[:, k + 2], q)) / self.yaw_step) * self.yaw_step
                x = np.concatenate([opt.incumbent, [0.0, 0.0, yaw]])
                if not any(np.allclose(x, a) for a in asked):
                    self.request(x, f"incumbent at the {q}th percentile of the yaw of its VPP states (level not "
                                    "active yet)", opt.run.log)
                    return
            return
        from src.vpp_targeted import choose
        x_cand, _ = opt.candidates()
        x_cand = x_cand.copy()
        x_cand[:, k:k + 2] = 0.0
        missing = np.ones((len(x_cand), opt.surrogate.n_levels), dtype=bool)
        choice = choose(opt.surrogate, opt.point_weights(), x_cand, missing, opt.x_int, np.inf)
        gains = choice["gains"][:, -1].copy()
        rounded = x_cand.copy()
        rounded[:, k + 2] = np.round(rounded[:, k + 2] / self.yaw_step) * self.yaw_step
        for j, x in enumerate(rounded):                    # never twice the same point
            if any(np.allclose(x, a) for a in asked):
                gains[j] = -np.inf
        i = int(np.argmax(gains))
        self.request(x_cand[i], f"largest level-{opt.surrogate.n_levels} gain of the loop's candidates "
                                f"({gains[i]:.3g})", opt.run.log)
