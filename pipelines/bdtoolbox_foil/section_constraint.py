"""
2D lift constraint of the section, imposed by projection.

The section of a 3D foil optimization may have to satisfy an EQUALITY constraint on its 2D lift,
e.g. CL2d(alpha = 0 deg) = 0.45 for the house NeuralFoil model. An equality constraint cannot be
handled by rejection (its feasible set has zero volume), so it is eliminated: in the Kulfan
thickness_camber form (geometry.py) the camber weights are c_i = delta + s_i, and for every
design point (t_i, s_i) the offset delta is SOLVED so that

    CL2d(alpha_deg; t, s, delta) = cl2d_target        (reference NeuralFoil model, flow of the
                                                        configuration: Re, n_crit, xtr)

A uniform delta adds the basic camber mode delta C(x), so CL2d grows almost linearly with
delta: a vectorized NeuralFoil evaluation on a grid of delta values brackets the root, then
Brent's method refines it. The solved delta is a deterministic function of the design point,
so every fidelity level evaluates EXACTLY the same geometry; each level then reports the CL2d
it sees itself (NeuralFoil xxsmall / xxlarge inside NPLLT, XFOIL for AVL).

Configuration entry (all keys but cl2d_target optional):
    "section_constraint": {"cl2d_target": 0.45, "alpha_deg": 0.0, "reference_model": "xxlarge",
                           "delta_bounds": [-0.3, 0.6], "tolerance": 1e-5, "n_grid": 25}
"""
import logging
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from . import bridge
from .geometry import elevation_matrix, kulfan_parameters, section_parameters

logger = logging.getLogger(__name__)

DEFAULTS = {"alpha_deg": 0.0, "reference_model": "xxlarge", "delta_bounds": (-0.3, 0.6),
            "tolerance": 1e-5, "n_grid": 25}


class ConstraintFailure(ValueError):
    """The 2D lift target cannot be reached within the delta bounds."""


@dataclass
class ProjectionResult:
    """Solved camber offset and the reference CL2d (and confidence) of the projected section."""
    delta: float
    cl2d: float
    confidence: float
    n_calls: int


def settings(problem) -> dict:
    """Constraint settings of the configuration merged with the defaults (None: no constraint)."""
    raw = getattr(problem, "section_constraint", None) or {}
    if not raw:
        return None
    return {**DEFAULTS, **raw}


class CamberProjector:
    """
    Solves the camber offset delta of a design point (see the module docstring). Results are
    cached by design point, since the same point is often evaluated at several levels.

    Args:
    - problem: FoilProblem with a kulfan thickness_camber parametrization and a
      "section_constraint" entry.
    """
    def __init__(self, problem):
        self.problem = problem
        self.config = settings(problem)
        if self.config is None:
            raise ValueError("CamberProjector needs a 'section_constraint' entry")
        parametrization = problem.parametrization
        if parametrization.get("type") != "kulfan" or \
           parametrization.get("form") != "thickness_camber":
            raise ValueError("the section constraint needs a kulfan thickness_camber "
                             "parametrization (camber offset delta)")
        if "delta" in parametrization.get("fixed", {}) or \
           any(v.name == "delta" for v in problem.variables):
            raise ValueError("delta is solved by the section constraint: it cannot be a design "
                             "variable or a fixed parameter")
        self.nf = bridge.neuralfoil()
        self.flow = problem.flow
        self.cache = {}

    def _kulfan_batch(self, params: dict, deltas: np.ndarray) -> dict:
        """NeuralFoil Kulfan inputs (8 weights, exact elevation) for a vector of deltas."""
        base = kulfan_parameters({**params, "delta": 0.0}, "thickness_camber")
        if base["leading_edge_weight"] != 0.0 and len(base["upper_weights"]) != 8:
            raise ValueError("the section constraint needs leading_edge_weight = 0 or 8 weights "
                             "per side (exact NeuralFoil Kulfan input)")
        matrix = elevation_matrix(len(base["upper_weights"]), 8)
        # c_i = delta + s_i: a delta shifts every upper AND lower weight by delta, and the
        # elevation preserves constants (its rows sum to 1)
        upper = (matrix @ base["upper_weights"])[:, None] + deltas[None, :]
        lower = (matrix @ base["lower_weights"])[:, None] + deltas[None, :]
        return {"upper_weights": upper, "lower_weights": lower,
                "leading_edge_weight": base["leading_edge_weight"],
                "TE_thickness": base["TE_thickness"]}

    def cl2d(self, params: dict, deltas) -> tuple[np.ndarray, np.ndarray]:
        """Reference CL2d and NeuralFoil confidence at alpha_deg for a vector of deltas."""
        deltas = np.atleast_1d(np.asarray(deltas, dtype=float))
        xtr = self.flow.get("xtr", (0.1, 0.1))
        aero = self.nf.get_aero_from_kulfan_parameters(
            kulfan_parameters=self._kulfan_batch(params, deltas),
            alpha=float(self.config["alpha_deg"]), Re=float(self.flow["re"]),
            n_crit=float(self.flow.get("n_crit", 1.0)), xtr_upper=float(xtr[0]),
            xtr_lower=float(xtr[1]), model_size=self.config["reference_model"])
        cl = np.ravel(aero["CL"]) * np.ones(len(deltas))
        confidence = np.ravel(aero.get("analysis_confidence", np.ones(1))) * np.ones(len(deltas))
        return cl, confidence

    def solve(self, design_point: np.ndarray) -> ProjectionResult:
        """
        Camber offset delta of a normalized design point such that CL2d = cl2d_target.

        Raises:
        - ConstraintFailure if the target is outside the CL2d range of the delta bounds (the
          failure is cached too: the same point at another level fails at once).
        """
        key = tuple(np.round(np.asarray(design_point, dtype=float), 12))
        if key in self.cache:
            cached = self.cache[key]
            if isinstance(cached, ConstraintFailure):
                raise cached
            return ProjectionResult(cached.delta, cached.cl2d, cached.confidence, 0)
        try:
            result = self._solve(design_point)
        except ConstraintFailure as failure:
            self.cache[key] = failure
            raise
        self.cache[key] = result
        return result

    def _solve(self, design_point: np.ndarray) -> ProjectionResult:
        """Root finding of solve() (no cache)."""
        params = section_parameters(self.problem, design_point)
        target = float(self.config["cl2d_target"])
        lo, hi = (float(v) for v in self.config["delta_bounds"])
        grid = np.linspace(lo, hi, int(self.config["n_grid"]))
        cl, _ = self.cl2d(params, grid)
        residual = cl - target
        n_calls = 1
        crossing = np.flatnonzero(np.sign(residual[:-1]) * np.sign(residual[1:]) <= 0)
        if not crossing.size or not np.all(np.isfinite(residual)):
            raise ConstraintFailure(
                f"CL2d target {target} not reachable for delta in [{lo}, {hi}] (CL2d from "
                f"{np.nanmin(cl):.3f} to {np.nanmax(cl):.3f})")
        i = int(crossing[0])

        def f(delta):
            nonlocal n_calls
            n_calls += 1
            return float(self.cl2d(params, [delta])[0][0]) - target

        if residual[i] == 0.0:
            delta = float(grid[i])
        else:
            delta = brentq(f, grid[i], grid[i + 1], xtol=float(self.config["tolerance"]) * 1e-2)
        cl_final, confidence = self.cl2d(params, [delta])
        result = ProjectionResult(delta=float(delta), cl2d=float(cl_final[0]),
                                  confidence=float(confidence[0]), n_calls=n_calls + 1)
        if abs(result.cl2d - target) > float(self.config["tolerance"]):
            raise ConstraintFailure(f"CL2d projection did not converge: {result.cl2d:.6f} for "
                                    f"the target {target}")
        return result
