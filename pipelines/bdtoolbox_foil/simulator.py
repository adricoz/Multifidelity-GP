"""
mfego simulator of a foil optimization: normalized design point -> section -> polar of the
fidelity level -> objective (mfego BaseSimulator contract: (value, metrics), NaN on failure).
"""
import logging
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "mfego") not in sys.path:
    sys.path.insert(0, str(ROOT / "mfego"))

# pylint: disable=wrong-import-position,import-error
from src.simulator import BaseSimulator  # noqa: E402

from . import bridge  # noqa: E402
from .geometry import build_section, to_physical  # noqa: E402
from .objectives import evaluate_objective  # noqa: E402
from .solvers import SolverFailure, make_backend  # noqa: E402

logger = logging.getLogger(__name__)


class BdToolboxFoilSimulator(BaseSimulator):
    """
    Foil simulator built from a configuration (config.FoilProblem).

    Args:
    - problem: FoilProblem.
    - paths: bridge.BridgePaths (default: resolved from the configuration).
    - backends: optional list of backends (one per level), e.g. fakes for the tests.
    """
    def __init__(self, problem, paths=None, backends: list = None):
        super().__init__(num_levels=len(problem.levels))
        self.problem = problem
        self.paths = paths or bridge.resolve_paths(problem.paths)
        if backends is None:
            bridge.setup(self.paths)
            backends = [make_backend(level, problem.flow, self.paths) for level in problem.levels]
        self.backends = backends
        self.n_evaluations = 0

    def evaluate(self, design_point: list, level: int) -> tuple[float, dict]:
        """Objective (to minimize) of a normalized design point at a fidelity level."""
        start = time.perf_counter()
        self.n_evaluations += 1
        metrics = {"level": int(level), "solver": self.problem.levels[level - 1].solver}
        try:
            if self.problem.kind == "foil3d":
                return self._evaluate_3d(design_point, level, metrics, start)
            section = build_section(self.problem, design_point)
            metrics.update({"section": section.name, "thickness": section.thickness,
                            **{k: float(v) for k, v in section.parameters.items()}})
            violation = self._constraint_violation(section)
            if violation:
                raise SolverFailure(violation)
            polar = self.backends[level - 1].polar(section)
            value, objective_metrics = evaluate_objective(polar, self.problem.objective)
            metrics.update(objective_metrics)
        except (SolverFailure, ValueError, FloatingPointError, KeyError) as error:
            logger.warning("Evaluation failed (level %d, point %s): %s", level,
                           np.round(design_point, 4), error)
            value = np.nan
            metrics["error"] = str(error)
        metrics["time_s"] = time.perf_counter() - start
        return value, _json_friendly(metrics)

    def _evaluate_3d(self, design_point, level, metrics, start) -> tuple[float, dict]:
        """TEMPLATE (3D): physical parameters -> backend coefficients; objective
        {"type": "coefficient", "name": "Cx", "sign": -1} (drag = -Cx in bdFoil frames)."""
        parameters = to_physical(design_point, self.problem.variables)
        workdir = Path(tempfile.mkdtemp(prefix="mfego_3d_"))
        coefficients = self.backends[level - 1].coefficients(parameters, workdir)
        name = self.problem.objective.get("name", "Cx")
        value = float(self.problem.objective.get("sign", 1.0)) * float(coefficients[name])
        metrics.update({**parameters, **coefficients, "time_s": time.perf_counter() - start})
        return value, _json_friendly(metrics)

    def _constraint_violation(self, section) -> str:
        """Geometric constraints of the configuration ("constraints" entry)."""
        min_thickness = self.problem.constraints.get("min_thickness")
        if min_thickness is not None and section.thickness < float(min_thickness):
            return f"thickness {section.thickness:.4f} < {min_thickness}"
        return ""


def _json_friendly(metrics: dict) -> dict:
    out = {}
    for key, value in metrics.items():
        if isinstance(value, (np.floating, np.integer)):
            value = value.item()
        if isinstance(value, float) and not np.isfinite(value):
            value = None
        out[key] = value
    return out
