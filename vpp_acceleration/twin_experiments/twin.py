"""
Digital twin of the section optimisation (plan of 2026-11-12): the aero codes are replaced by a KNOWN
function, so that the optimum the method should find can be computed directly.

The truth is the mean of the saved surrogate of a finished run (default: the run of 2026-11-11):
    level 1 of the twin = its level-1 mean (NPLLT x NeuralFoil),
    level 2 of the twin = its level-2 mean (AVL x XFOIL), the level the VPP of the twin uses.
The codes are never called. Nothing of src/ or scripts/ is modified: the twin plugs in from outside.

- TwinTruth: what the VPP workers need of a surrogate (predict, mean_only_copy), answering with the
  truth at a fixed level. Lives in this importable module so that the workers can unpickle it.
- TwinRunner: stands for vpp_optimize.CodeRunner (run_tasks, evaluate, n_runs, time, close) and
  answers with the truth (same principle as FakeRunner of tests/test_cfd_queue.py).
- SearchDomain: the search space of the truth run and its constraints as functions of the scores.
- mean_speed: probability-weighted speed over the conditions solved for all the compared sections.

Limit: the truth is itself a co-kriging mean, smoother than the real codes and of the same family as
the model that learns it. The twin tests the ALGORITHM (start design, targeting, screening, restarts,
convergence), not the regularity hypothesis.
"""
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
VPP_DIR = HERE.parent
for _path in (HERE, VPP_DIR / "scripts", VPP_DIR, VPP_DIR.parent / "mfego"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

TRUTH_RUN = VPP_DIR / "runs" / "2026-11-11_1245_ixxv095_rigid_xtr003_k10_cfd3d"
TRUTH_LEVEL = 2                                   # the level the VPP of the twin sees (AVL x XFOIL)
OUTPUT_NAMES = ["Fx", "Fy", "Fz", "Mx", "My", "Mz", "Cpmin"]
VPP_OPTIONS = {"foil": "CFoilFSI", "with_hvol": False, "fsi": False}


class TwinTruth:
    """The truth: means of a saved surrogate; predict(x) answers at `level` (default TRUTH_LEVEL)."""

    def __init__(self, surrogate, level: int = TRUTH_LEVEL):
        self.surrogate, self.level = surrogate, int(level)

    @classmethod
    def load(cls, run: Path = TRUTH_RUN, level: int = TRUTH_LEVEL) -> "TwinTruth":
        with open(Path(run) / "results" / "surrogate.pkl", "rb") as file:
            return cls(pickle.load(file), level)        # a mean-only copy: predictions only

    def predict(self, x: np.ndarray, level: int = None) -> np.ndarray:
        """(m, 7) outputs at points (m, k + 3) = (scores, cant, rake, yaw)."""
        return self.surrogate.predict(np.atleast_2d(x), level=self.level if level is None else level)

    def mean_only_copy(self) -> "TwinTruth":         # VPPPool.publish
        return self


class TwinRunner:
    """CodeRunner of the twin: level l of the run is level l of the truth, no failure, no cost."""

    def __init__(self, problem, run, truth: TwinTruth, n_levels: int):
        self.problem, self.run, self.truth = problem, run, truth
        self.n_runs = {level: 0 for level in range(1, n_levels + 1)}
        self.time = {level: 0.0 for level in range(1, n_levels + 1)}

    def run_tasks(self, points: np.ndarray, levels, tag: str = "") -> tuple:
        points = np.atleast_2d(points)
        levels = np.broadcast_to(np.asarray(levels, dtype=int), (len(points),))
        start = time.perf_counter()
        y = np.empty((len(points), len(OUTPUT_NAMES)))
        for level in np.unique(levels):
            rows = levels == level
            y[rows] = self.truth.predict(points[rows], level=min(int(level), self.truth.level))
        each = (time.perf_counter() - start) / len(points)
        for point, level, values in zip(points, levels, y):
            self.n_runs[int(level)] += 1
            self.time[int(level)] += each
            self.run.add_evaluation({
                "tag": tag, "level": int(level), "ok": True,
                **{f"x{i}": float(v) for i, v in enumerate(point)},
                **dict(zip(OUTPUT_NAMES, (float(v) for v in values))),
                "time_s": round(each, 4), "error": "", "xtr": float("nan"),
                "elapsed_min": round(self.run.elapsed() / 60, 3)})
        return y, [{"time_s": each, "error": "", "level": int(level)} for level in levels]

    def evaluate(self, points: np.ndarray, level: int, tag: str = "") -> tuple:
        return self.run_tasks(points, level, tag)

    def close(self) -> None:
        pass


class SearchDomain:
    """Search space of a run (ScoreBoxFamily, PCA) and its constraints as functions of the scores
    (all >= 0 inside): linear constraints of the domain, Ixx / v relative to its limit, score box."""

    def __init__(self, run: Path = TRUTH_RUN):
        from src.section_domain import load_search_space  # noqa: PLC0415
        self.config = json.loads((Path(run) / "config.json").read_text(encoding="utf-8"))
        self.k = int(self.config["k"])
        self.family, self.pca, self.current = load_search_space(
            VPP_DIR / self.config["search_box"], self.k, self.config["search_box_name"])
        self.domain = self.family.domain
        self.conditions = [tuple(c) for c in self.config["conditions"]]
        self.weights = np.array(self.config["condition_weights"], dtype=float)

    def constraints(self, scores: np.ndarray) -> np.ndarray:
        from src.section_pca import section_stiffness  # noqa: PLC0415
        weights = self.pca.decode(np.atleast_2d(scores))
        ixx_v = section_stiffness(weights, self.domain.te_thickness)[1][0]
        return np.concatenate([self.domain.margins(weights)[0],
                               [ixx_v / self.domain.ixx_v_min - 1.0],
                               scores - self.family.lower, self.family.upper - scores])

    def constraint_jacobian(self, scores: np.ndarray, step: float = 1e-5) -> np.ndarray:
        return np.column_stack([(self.constraints(scores + step * e)
                                 - self.constraints(scores - step * e)) / (2.0 * step)
                                for e in np.eye(self.k)])

    def admissible(self, scores: np.ndarray) -> np.ndarray:
        return self.family.contains(self.pca.decode(np.atleast_2d(scores)))

    def sample(self, n: int, seed: int) -> np.ndarray:
        scores = self.pca.encode(self.family.sample(4 * n, seed))
        return scores[self.admissible(scores)][:n]


def run_sections(run: Path) -> dict:
    """name -> scores of a finished run: best, local optima (region_i), start section."""
    run = Path(run)
    best = json.loads((run / "results" / "best_section.json").read_text(encoding="utf-8"))
    sections = {"best": np.array(best["scores"])}
    optima = run / "results" / "optima.json"
    if optima.exists():
        for entry in json.loads(optima.read_text(encoding="utf-8")):
            if not entry["best"]:
                sections[f"region_{entry['region']}"] = np.array(entry["scores"])
    sections[best["start"]] = np.array(best["start_scores"])
    return sections


def mean_speed(states_by_section: list, weights: np.ndarray) -> tuple:
    """Weighted mean speed of each section over the conditions solved for ALL of them, and the
    share of the probability these conditions carry."""
    solved = np.array([[s["success"] for s in states] for states in states_by_section])
    speeds = np.array([[s["speed"] for s in states] for states in states_by_section])
    w = weights * solved.all(axis=0)
    if w.sum() <= 0.0:
        return np.full(len(states_by_section), np.nan), 0.0
    return np.where(solved.all(axis=0), speeds, 0.0) @ w / w.sum(), float(w.sum() / weights.sum())


def make_pool(n_workers: int, work: Path):
    """VPPPool (one MDS VPP per process) with its scratch folder under `work`."""
    from src.vpp_optimize import VPPPool  # noqa: PLC0415  (imports MDS)

    class Scratch:
        def __init__(self, root):
            self.root = Path(root)

        def path(self, *parts) -> Path:
            path = self.root.joinpath(*parts)
            path.parent.mkdir(parents=True, exist_ok=True)
            return path

    return VPPPool(Scratch(work), n_workers, VPP_OPTIONS)
