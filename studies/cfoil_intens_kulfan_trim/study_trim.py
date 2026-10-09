"""
Shared tools of the trim study (second C-foil test): paths, configuration, names and colours of
the levels, loading of an optimization run.

The baseline section, the feasible design, the JSON helpers and the figure style come from the
first study (../cfoil_intens_kulfan: study_lib.py, plotkit.py), whose calibration fixed the
settings of this one (levels, estimator, budget): see ../cfoil_intens_kulfan/RAPPORT.md.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

STUDY = Path(__file__).resolve().parent
FIRST_STUDY = STUDY.parent / "cfoil_intens_kulfan"
if str(FIRST_STUDY) not in sys.path:
    sys.path.insert(0, str(FIRST_STUDY))

# pylint: disable=wrong-import-position,import-error
import plotkit as pk  # noqa: E402
import study_lib as base  # noqa: E402
from src.surrogate_models import load_surrogate  # noqa: E402

CONFIG = STUDY / "config_cfoil_trim.json"
RESULTS = STUDY / "results"
FIGURES = STUDY / "figures"
pk.FIGURES = FIGURES          # every figure of this study is saved here

# levels of this study (2 levels): each solver keeps the colour it had in the first study
LEVELS = (1, 2)
LEVEL_NAME = {1: "NPLLT x NeuralFoil xxlarge", 2: "AVL x XFOIL"}
LEVEL_SHORT = {1: "NPLLT", 2: "AVL"}
LEVEL_COLOR = {1: pk.LEVEL_COLOR[2], 2: pk.LEVEL_COLOR[3]}
LEVEL_SYMBOL = {1: "square", 2: "diamond"}

save_json, load_json = base.save_json, base.load_json
baseline_parameters, feasible_design = base.baseline_parameters, base.feasible_design
make_simulator = base.make_simulator


def load_problem():
    """FoilProblem of the trim study."""
    return base.load_problem(CONFIG)


def setup_logging(step: str, path: Path = None) -> Path:
    """Console + results/<step>.log of this study (or path)."""
    return base.setup_logging(step, path or RESULTS / f"{step}.log")


def load_run(run_dir: Path) -> dict:
    """
    Evaluations of a run in their order (simulator counter "evaluation"), with the level, the
    cost, the cumulative cost, the best level-2 value so far and the metrics; plus the state,
    the results and the trained surrogate.
    """
    run_dir = Path(run_dir)
    state = load_json(run_dir / "ego_backup.json")
    costs = [float(c) for c in state["costs"]]
    config = load_json(run_dir / "config_effective.json")
    doe = config["optimization"]["doe"]
    rows = []
    for key, xs in state["X_dict"].items():
        level = int(key)
        metrics = state.get("Metrics_dict", {}).get(key, [{}] * len(xs))
        for i, (x, y, m) in enumerate(zip(xs, state["Y_dict"][key], metrics)):
            m = m or {}
            rows.append({"level": level, "index_in_level": i, "doe": i < doe[level - 1],
                         "verification": bool(m.get("verification", False)),
                         "value": np.nan if y is None else float(y),
                         "evaluation": m.get("evaluation"), "x": np.asarray(x),
                         **{k: v for k, v in m.items()
                            if isinstance(v, (int, float)) and not isinstance(v, bool)
                            and k != "evaluation"},
                         # flags of the trim (e.g. trim_interpolated)
                         **{k: v for k, v in m.items()
                            if isinstance(v, bool) and k.startswith("trim_")}})
    table = pd.DataFrame(rows).sort_values("evaluation").reset_index(drop=True)
    table["cost"] = [costs[lv - 1] for lv in table["level"]]
    table["cumulative_cost"] = table["cost"].cumsum()
    top = len(costs)
    best, history = np.inf, []
    for level, value in zip(table["level"], table["value"]):
        if level == top and np.isfinite(value):
            best = min(best, value)
        history.append(best if np.isfinite(best) else np.nan)
    table["best_top"] = history
    return {"run_dir": run_dir, "state": state, "table": table, "costs": costs,
            "config": config, "results": load_json(run_dir / "results.json"),
            "surrogate": load_surrogate(str(run_dir / "surrogate.json"))}
