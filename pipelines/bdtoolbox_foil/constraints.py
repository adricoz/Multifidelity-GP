"""
Known (a priori) geometric constraints of a section.

These constraints only need the geometry, not a solver, so they are cheap and KNOWN before any
evaluation. They are therefore handled where they belong (mfego [KC] hooks):
* the initial design only contains feasible points (ExperimentData.generate_initial_design),
* the search of the next point gives a zero merit to infeasible points
  (AcquisitionFunction(feasibility=...)), and the random fallbacks draw feasible points,
* the simulator still checks them (an infeasible point is a failed evaluation, NaN).
Without these hooks, infeasible points were proposed, returned NaN and were excluded from the
GP, so the model never learnt where they were and kept proposing them.

Configuration entry: "constraints": {"min_thickness": 0.1225, "max_thickness": 0.16}
(maximum thickness / chord of the section; both optional).

The vectorized check exists for the Kulfan parametrization, whose thickness is an affine
function of the weights (geometry.cst_thickness). For the other parametrizations the check is
done point by point on the section coordinates (slower but rarely used inside a search).
"""
import numpy as np

from .bridge import BridgeError
from .geometry import (THICKNESS_STATIONS, build_section, cst_thickness, max_thickness,
                       section_parameters)


def section_violation(problem, thickness: float) -> str:
    """Message describing the violated thickness constraint ("" when feasible)."""
    constraints = problem.constraints or {}
    lo, hi = constraints.get("min_thickness"), constraints.get("max_thickness")
    if lo is not None and thickness < float(lo):
        return f"thickness {thickness:.4f} < min_thickness {lo}"
    if hi is not None and thickness > float(hi):
        return f"thickness {thickness:.4f} > max_thickness {hi}"
    return ""


def _kulfan_thickness_weights(problem, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Thickness weights (m, n) and TE thicknesses (m,) of a batch of normalized points."""
    form = problem.parametrization.get("form", "weights")
    rows, te = [], []
    for point in points:
        params = section_parameters(problem, point)
        te.append(float(params.get("TE_thickness", 0.0)))
        if form == "thickness_camber":
            keys = sorted((k for k in params if k.startswith("t_") and k[2:].isdigit()),
                          key=lambda k: int(k[2:]))
            rows.append([params[k] for k in keys])
        else:
            n = len([k for k in params if k.startswith("upper_") and k[6:].isdigit()])
            rows.append([params[f"upper_{i}"] - params[f"lower_{i}"] for i in range(n)])
    return np.asarray(rows, dtype=float), np.asarray(te, dtype=float)


def max_thickness_batch(problem, points: np.ndarray) -> np.ndarray:
    """
    Maximum thickness / chord of a batch of normalized design points (m, d) -> (m,).
    Also returns -inf where the thickness becomes negative somewhere (crossing surfaces).
    """
    points = np.atleast_2d(np.asarray(points, dtype=float))
    if problem.parametrization.get("type") == "kulfan":
        weights, te = _kulfan_thickness_weights(problem, points)
        thickness = cst_thickness(THICKNESS_STATIONS, weights, te)
        out = thickness.max(axis=1)
        out[thickness.min(axis=1) < 0.0] = -np.inf
        return out
    out = np.full(len(points), -np.inf)
    for i, point in enumerate(points):
        try:      # a section that cannot be built is infeasible (it would fail in the simulator)
            out[i] = max_thickness(build_section(problem, point).coordinates)
        except (ValueError, FloatingPointError, np.linalg.LinAlgError, BridgeError):
            pass
    return out


def feasibility_function(problem):
    """
    Known-constraint function of a configuration for the mfego [KC] hooks:
    feasibility(points (m, d)) -> bool array (m,), or None when there is no geometric
    constraint (every point is feasible).
    """
    constraints = problem.constraints or {}
    lo, hi = constraints.get("min_thickness"), constraints.get("max_thickness")
    if problem.kind == "foil3d" and problem.parametrization.get("type") == "planform":
        return None
    if lo is None and hi is None and problem.parametrization.get("type") != "kulfan":
        return None

    def feasibility(points: np.ndarray) -> np.ndarray:
        thickness = max_thickness_batch(problem, points)
        ok = np.isfinite(thickness)
        if lo is not None:
            ok &= thickness >= float(lo)
        if hi is not None:
            ok &= thickness <= float(hi)
        return ok

    return feasibility
