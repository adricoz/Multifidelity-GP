"""
Objectives computed from a section polar.

* cd_at_cl: drag coefficient at a target lift coefficient (to be minimized), interpolated on the
  monotone CL branch containing alpha ~ 0 (same branch definition as bdFoil core
  forces.monotone_branch). Unlike forces.at_cl, which clamps at the branch ends, a target outside
  the branch returns NaN: the point is a failed evaluation (no silent wrong value).
* max_lift_to_drag: -max(CL/CD) on the same branch (minimized).
"""
import numpy as np


def monotone_branch(alpha: np.ndarray, cl: np.ndarray) -> slice:
    """Maximal run of strictly increasing CL containing the alpha closest to 0."""
    i0 = int(np.argmin(np.abs(alpha)))
    lo = i0
    while lo > 0 and cl[lo - 1] < cl[lo]:
        lo -= 1
    hi = i0
    while hi < len(cl) - 1 and cl[hi + 1] > cl[hi]:
        hi += 1
    return slice(lo, hi + 1)


def value_at_cl(polar, cl_target: float, column: str = "cd") -> float:
    """Value of a polar column at a target CL on the monotone branch (NaN if outside)."""
    branch = monotone_branch(polar.alpha, polar.cl)
    cl = np.asarray(polar.cl)[branch]
    values = np.asarray(getattr(polar, column))[branch]
    if len(cl) < 2 or not cl[0] <= cl_target <= cl[-1]:
        return np.nan
    return float(np.interp(cl_target, cl, values))


def evaluate_objective(polar, objective: dict) -> tuple:
    """
    Objective value (minimized) and metrics of a polar.

    Args:
    - polar: solvers.Polar.
    - objective: {"type": "cd_at_cl", "cl_target": ..., "log10": false} or
      {"type": "max_lift_to_drag"}.
    Returns:
    - (value, metrics dict). value is NaN if the objective cannot be computed.
    """
    branch = monotone_branch(polar.alpha, polar.cl)
    metrics = {"cl_max_branch": float(np.max(np.asarray(polar.cl)[branch]))}
    if objective["type"] == "cd_at_cl":
        cl_target = float(objective["cl_target"])
        cd = value_at_cl(polar, cl_target, "cd")
        metrics.update({"cl_target": cl_target, "cd": cd,
                        "alpha": value_at_cl(polar, cl_target, "alpha")})
        if polar.cm is not None and np.all(np.isfinite(polar.cm)):
            metrics["cm"] = value_at_cl(polar, cl_target, "cm")
        if polar.cpmin is not None:
            metrics["cpmin"] = value_at_cl(polar, cl_target, "cpmin")
        value = np.log10(cd) if objective.get("log10") and cd > 0 else cd
    else:
        ratio = np.asarray(polar.cl)[branch] / np.asarray(polar.cd)[branch]
        i_best = int(np.argmax(ratio))
        metrics.update({"max_lift_to_drag": float(ratio[i_best]),
                        "alpha": float(np.asarray(polar.alpha)[branch][i_best])})
        value = -float(ratio[i_best])
    return (float(value) if np.isfinite(value) else np.nan), metrics
