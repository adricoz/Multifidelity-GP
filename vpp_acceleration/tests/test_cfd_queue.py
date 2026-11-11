"""The asynchronous CFD level (src/cfd_queue.py, src/cfd_level.py; 2026-11-11): convergence test, requests,
ingestion with the nested design, activation of the top level."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import cfd_level  # noqa: E402
from src.cfd_queue import CFDQueue  # noqa: E402
from src.vpp_targeted import TorsorCoKriging  # noqa: E402

BOUNDS = [(-1.0, 1.0), (0.0, 8.0), (-1.5, 1.5), (-6.0, 6.0)]     # 1 score, cant, rake, yaw


def truth(x, level):
    x = np.atleast_2d(x)
    f = np.sin(2 * x[:, 0]) + 0.1 * x[:, 3] + 0.05 * level * x[:, 0] ** 2
    return np.column_stack([f, 2 * f] + [np.full(len(x), 0.1 * level)] * 4 + [np.full(len(x), -1.0 * level)])


class FakeRunner:
    def __init__(self):
        self.calls = []

    def run_tasks(self, points, levels, tag=""):
        self.calls.append((np.array(points), list(levels)))
        return np.vstack([truth(p, l) for p, l in zip(points, levels)]), [{"error": ""}] * len(points)


def fake_optimizer(tmp_path):
    rng = np.random.default_rng(0)
    x1 = np.column_stack([rng.uniform(lo, hi, 30) for lo, hi in BOUNDS])
    s = TorsorCoKriging(BOUNDS, costs=(1.0, 6.0), n_models=2, n_outputs=7)
    s.add(1, x1, truth(x1, 1))
    s.add(2, x1[:12], truth(x1[:12], 2))
    s.fit()
    logs = []
    run = SimpleNamespace(log=logs.append)
    problem = SimpleNamespace(n_scores=1, weights=lambda z: np.tile(np.r_[np.atleast_2d(z)[0], np.zeros(15)], (1, 1)))
    opt = SimpleNamespace(surrogate=s, runner=FakeRunner(), run=run, max_sync_level=2, vpp_pool=None, failed=[],
                          incumbent=np.zeros(1), incumbent_states=None, fitted_size=None, states="old")
    return opt, problem, logs


def write_done(queue, rid, x, y, converged=True, variation=None):
    path = Path(queue.folder) / "done" / f"{rid}.json"
    path.write_text(json.dumps({"id": rid, "x": list(map(float, x)), "y": list(map(float, y)), "ok": True,
                                "converged": converged, "variation": variation or {"Cd": 1e-3}}))


def test_convergence_criterion():
    """0.2 % of relative variation of the window means, after MIN_SECOND_ORDER second-order iterations."""
    it = np.arange(1, 801)
    flat = pd.DataFrame({"Time": it, "Cd": 0.014 + 1e-3 * np.exp(-it / 30), "Cl": 0.18 + 0 * it, "Cs": 0.3 + 0 * it})
    ok, var = cfd_level.converged(flat)
    assert ok and max(var.values()) < cfd_level.CONV_REL
    drift = flat.assign(Cd=0.014 * (1 + 0.0001 * it))                    # 1 % per 100 iterations
    ok, var = cfd_level.converged(drift)
    assert not ok and var["Cd"] > cfd_level.CONV_REL
    short = flat[flat.Time <= cfd_level.N_ITER_UPWIND + cfd_level.MIN_SECOND_ORDER - 1]
    assert not cfd_level.converged(short)[0]


def test_request_forces_neutral_cant_rake_and_rounds_the_yaw(tmp_path):
    opt, problem, logs = fake_optimizer(tmp_path)
    q = CFDQueue(tmp_path / "cfd", problem, cost=240.0)
    rid = q.request(np.array([0.3, 2.0, 0.7, -0.8]), "test", logs.append)
    req = json.loads((tmp_path / "cfd" / "requests" / f"{rid}.json").read_text())
    assert req["x"] == [0.3, 0.0, 0.0, -1.0] and req["yaw"] == -1.0
    assert q.pending() and q.pending()[0].stem == rid


def test_ingest_nested_then_activation(tmp_path):
    """Points wait until N_ACTIVATE, the missing lower levels are run at their x, Cpmin comes from the level below,
    then the surrogate gets a third level with every point; later points go straight to level 3."""
    opt, problem, logs = fake_optimizer(tmp_path)
    q = CFDQueue(tmp_path / "cfd", problem, cost=240.0, n_activate=3)
    xs = [np.array([v, 0.0, 0.0, -1.0]) for v in (-0.5, 0.1, 0.6, 0.9)]
    for i, x in enumerate(xs[:2]):
        y = truth(x, 3)[0]
        y[6] = np.nan
        write_done(q, f"a{i}", x, y)
    q.ingest(opt)
    assert opt.surrogate.n_levels == 2 and len(q.waiting) == 2
    assert [lv for _, lvs in opt.runner.calls for lv in lvs] == [1, 2, 1, 2]       # nested design
    assert np.isclose(q.waiting[0][1][6], truth(xs[0], 2)[0, 6])                 # Cpmin of level 2
    write_done(q, "a2", xs[2], truth(xs[2], 3)[0])
    q.ingest(opt)
    assert opt.surrogate.n_levels == 3 and opt.surrogate.sizes[2] == 3 and q.active and opt.states is None
    assert opt.surrogate.sizes[:2] == (33, 15)
    write_done(q, "a3", xs[3], truth(xs[3], 3)[0])
    q.ingest(opt)
    assert opt.surrogate.sizes[2] == 4
    lines = (tmp_path / "cfd" / "points.csv").read_text().splitlines()
    assert len(lines) == 5 and json.loads((tmp_path / "cfd" / "ingested.json").read_text()) == ["a0", "a1", "a2",
                                                                                              "a3"]


def test_unconverged_point_is_dropped(tmp_path):
    opt, problem, logs = fake_optimizer(tmp_path)
    q = CFDQueue(tmp_path / "cfd", problem, cost=240.0)
    write_done(q, "b0", [0.2, 0, 0, 0], truth([0.2, 0, 0, 0], 3)[0], converged=False, variation={"Cd": 0.05})
    q.ingest(opt)
    assert not q.waiting and not opt.runner.calls and any("not used" in m for m in logs)
