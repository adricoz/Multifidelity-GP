"""
mfego simulator of a foil optimization (mfego BaseSimulator contract: evaluate(x, level) ->
(value, metrics), value = NaN on a failed evaluation).

* section2d: normalized design point -> section -> polar of the level -> objective
  (objectives.py).
* foil3d, SECTION on a fixed planform (planform.py), objective "cd3d":
    1. design point -> physical parameters; with a section constraint the camber offset delta is
       solved so that CL2d(alpha) = target (section_constraint.py) -> Kulfan section;
    2. known geometric constraints (constraints.py): a violation is a failed evaluation;
    3. the section .xf file and the planform CSV (metres) are written in a temporary folder;
    4. the 3D solver of the level (npllt / avl) returns the bdFoil coefficients;
    5. objective CD = -Cx (positive drag: induced + profile, no wave drag) or, with an
       "equal_lift_reference" section, the drag AT EQUAL LIFT
           CD* = Cdprofile + Cdi (CL_ref / CL)^2,   CL = |(Cy, Cz)| (total lift),
       CL_ref = lift of the reference section at the same level and attitude (evaluated once
       per level). At a fixed attitude the 3D lift changes from one section to another (in AVL
       it only comes from the camber line), and the induced drag scales with CL^2: CD* compares
       the sections at the lift of the reference (first-order correction, same span loading).
    6. with a "trim" entry the lift is matched EXACTLY instead: the trim angle (yaw or rake) is
       adjusted by a secant iteration until CL = CL_ref (reference section at the configured
       attitude, same level), and the objective is the drag CD = -Cx at the trimmed attitude.
* foil3d template with PLANFORM design variables: kept for the tests with fake backends.
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
from .constraints import section_violation  # noqa: E402
from .geometry import build_section, to_physical, write_xf  # noqa: E402
from .objectives import evaluate_objective  # noqa: E402
from .solvers import SolverFailure, make_backend  # noqa: E402

logger = logging.getLogger(__name__)
# trim: the secant slope is only updated from solves at least this far apart (deg); closer solves
# give a slope dominated by the output resolution of the solvers (5 decimals in AVL)
TRIM_MIN_STEP = 0.02


def coefficient_name(key: str) -> str:
    """Short metric name of a bdFoil column: "Cx [-]" -> "Cx", "Ref Surf [m^2]" -> "Ref_Surf"."""
    return key.split(" [")[0].strip().replace(" ", "_")


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
            backends = [make_backend(level, problem, self.paths) for level in problem.levels]
        self.backends = backends
        self.n_evaluations = 0
        self.stations = None
        self.projector = None
        if problem.section_on_planform:
            from .planform import CFoilArc  # noqa: PLC0415
            self.stations = CFoilArc.from_config(problem.planform).stations()
            if problem.section_constraint:
                from .section_constraint import CamberProjector  # noqa: PLC0415
                self.projector = CamberProjector(problem)
        # equal-lift objective: lift of the reference section at each level (lazy, cached)
        self.reference_lift = {}
        # trim: last secant slope dCL / d(angle) of each level (first guess of the next trim)
        self.trim_slope = {}
        # bdFoil core errors of a 3D level are failed evaluations (NaN), except a missing
        # executable (ExeError), which is a set-up problem and stops the run
        self._core_errors, self._exe_error = (), None
        if problem.section_on_planform:
            try:
                core = bridge.bdfoil_core()
                self._core_errors, self._exe_error = (core.model.CoreError,), core.model.ExeError
            except bridge.BridgeError:
                pass

    def evaluate(self, design_point: list, level: int) -> tuple[float, dict]:
        """Objective (to minimize) of a normalized design point at a fidelity level."""
        value, metrics, _ = self.evaluate_details(design_point, level)
        return value, metrics

    def evaluate_details(self, design_point: list, level: int,
                         with_strips: bool = False) -> tuple[float, dict, dict]:
        """
        evaluate() plus the spanwise distributions of a 3D level (with_strips=True), for the
        diagnostic figures of a study (not stored in the optimizer state).

        Returns:
        - (value, metrics, strips) with strips None when not requested or not available.
        """
        start = time.perf_counter()
        self.n_evaluations += 1
        # "evaluation": rank of the call, to rebuild the order of the evaluations of all the
        # levels afterwards (the optimizer state stores the points level by level)
        metrics = {"level": int(level), "solver": self.problem.levels[level - 1].solver,
                   "evaluation": self.n_evaluations}
        strips = None
        try:
            if self.problem.section_on_planform and (
                    self.problem.trim or self.problem.objective.get("equal_lift_reference")):
                # one-off solve of the reference section, kept out of the timing of this point
                self.lift_reference(level)
                start = time.perf_counter()
            if self.problem.section_on_planform:
                value, strips = self._evaluate_section_on_planform(design_point, level, metrics,
                                                                   with_strips)
            elif self.problem.kind == "foil3d":
                value = self._evaluate_3d_template(design_point, level, metrics)
            else:
                value = self._evaluate_section_2d(design_point, level, metrics)
        except (SolverFailure, ValueError, FloatingPointError, KeyError,
                np.linalg.LinAlgError, *self._core_errors) as error:
            if self._exe_error is not None and isinstance(error, self._exe_error):
                raise
            logger.warning("Evaluation failed (level %d, point %s): %s", level,
                           np.round(design_point, 4), error)
            value = np.nan
            metrics["error"] = str(error)
        metrics["time_s"] = time.perf_counter() - start
        return value, _json_friendly(metrics), strips

    # ------------------------------------------------------------------ 2D
    def _evaluate_section_2d(self, design_point, level, metrics) -> float:
        section = build_section(self.problem, design_point)
        metrics.update({"section": section.name, "thickness": section.thickness,
                        **{k: float(v) for k, v in section.parameters.items()}})
        violation = section_violation(self.problem, section.thickness)
        if violation:
            raise SolverFailure(violation)
        polar = self.backends[level - 1].polar(section)
        value, objective_metrics = evaluate_objective(polar, self.problem.objective)
        metrics.update(objective_metrics)
        return value

    # ------------------------------------------------------------------ 3D section
    def project(self, design_point) -> dict:
        """Solved parameters of a design point ({"delta": ...} with a section constraint)."""
        if self.projector is None:
            return {}
        result = self.projector.solve(design_point)
        return {"delta": result.delta}

    def section_of(self, design_point):
        """Section (geometry.Section) of a design point, section constraint included."""
        return build_section(self.problem, design_point, self.project(design_point))

    def reference_point(self):
        """Normalized design point of the equal-lift reference section (None if not used)."""
        reference = self.problem.objective.get("equal_lift_reference")
        if not reference:
            return None
        return np.array([(float(reference[v.name]) - v.lower) / (v.upper - v.lower)
                         for v in self.problem.variables])

    def lift_reference(self, level: int) -> float:
        """Total lift |(Cy, Cz)| of the reference section at a level (evaluated once)."""
        if level not in self.reference_lift:
            raw = self._solve_section(self.reference_point(), level, {}, False)[0]
            lift = float(np.hypot(raw["Cy [-]"], raw["Cz [-]"]))
            if not np.isfinite(lift) or lift <= 0.0:
                raise SolverFailure(f"lift of the equal-lift reference section at level {level}"
                                    f" is {lift}")
            self.reference_lift[level] = lift
            logger.info("Equal-lift reference: CL_ref = %.5f at level %d", lift, level)
        return self.reference_lift[level]

    def _solve_section(self, design_point, level, metrics, with_strips,
                       trim_target: float = None) -> tuple[dict, dict]:
        """
        Section of a design point (projection, constraints) and raw coefficients of a level;
        with trim_target, at the attitude where the total lift equals trim_target (see _trim).
        """
        start = time.perf_counter()
        # geometric constraints first: they do not depend on the camber offset delta
        thickness = build_section(self.problem, design_point,
                                  {"delta": 0.0} if self.projector is not None else None).thickness
        violation = section_violation(self.problem, thickness)
        if violation:
            raise SolverFailure(violation)
        solved = {}
        if self.projector is not None:
            projection = self.projector.solve(design_point)
            solved = {"delta": projection.delta}
            metrics.update({"cl2d_reference": projection.cl2d,
                            "cl2d_reference_confidence": projection.confidence,
                            "projection_calls": projection.n_calls})
        section = build_section(self.problem, design_point, solved)
        metrics.update({"section": section.name, "thickness": section.thickness,
                        **{k: float(v) for k, v in section.parameters.items()
                           if isinstance(v, (int, float))},
                        "time_geometry_s": time.perf_counter() - start})
        from .planform import write_planform_csv  # noqa: PLC0415
        with tempfile.TemporaryDirectory(prefix="mf3d_", ignore_cleanup_errors=True) as tmp:
            workdir = Path(tmp)
            xf_path = write_xf(section, workdir)
            csv_path = write_planform_csv(self.stations, xf_path.name, workdir / "planform.csv")
            backend = self.backends[level - 1]

            def solve(attitude=None):
                return backend.evaluate_planform(csv_path, workdir, with_strips=with_strips,
                                                 attitude=attitude)

            result = solve() if trim_target is None else \
                self._trim(solve, level, trim_target, metrics)
        strips = result.pop("strips", None)
        return result, strips

    def _trim(self, solve, level: int, target: float, metrics: dict) -> dict:
        """
        Secant iteration on the trim angle (configuration "trim": variable yaw or rake, bounds,
        relative tolerance on the lift, max_iterations = maximum number of secant steps) until
        |(Cy, Cz)| = target. It starts from the configured attitude; the first step uses the
        slope learnt on the previous trims of the level (updated only from solves at least
        TRIM_MIN_STEP apart, clamped to [0.25, 4] x initial_slope) or initial_slope.
        Interpolation ("interpolation_band", default 0.005): when the last solve is within this
        relative distance of the target AND the target lies between the last two solves (10 %
        margin), the force coefficients (the "[-]" values) are linearly interpolated at the
        target instead of solving again; the other entries (strips, timings, diagnostics) are
        those of the last solve. The small lift residual left (tolerance or interpolation) is
        removed from the drag by the caller (_evaluate_section_on_planform).
        Raises SolverFailure for a non-finite lift, a target out of reach within the bounds or
        no convergence.
        """
        from .planform import attitude_angles  # noqa: PLC0415
        config = self.problem.trim
        variable = config.get("variable", "yaw")
        lo, hi = (float(v) for v in config.get("bounds", (-5.0, 15.0)))
        tolerance = float(config.get("tolerance", 1e-4))
        band = float(config.get("interpolation_band", 0.005))
        initial = float(config.get("initial_slope", 0.1))
        slope_lo, slope_hi = sorted((0.25 * initial, 4.0 * initial))

        def lift_of(coefficients: dict) -> float:
            lift = float(np.hypot(coefficients["Cy [-]"], coefficients["Cz [-]"]))
            if not np.isfinite(lift):
                raise SolverFailure(f"non-finite lift during the trim on {variable}")
            return lift

        attitude = attitude_angles(self.problem.attitude)
        angle = attitude[variable]
        result = solve({**attitude, variable: angle})
        lift = lift_of(result)
        slope = self.trim_slope.get(level, initial)
        n_solves, interpolated, last_solve_residual = 1, False, (lift - target) / target
        while abs(lift - target) > tolerance * target:
            if n_solves > int(config.get("max_iterations", 8)):
                raise SolverFailure(f"trim on {variable} did not converge: CL {lift:.5f} for "
                                    f"{target:.5f} after {n_solves} solves")
            new_angle = float(np.clip(angle + (target - lift) / slope, lo, hi))
            if new_angle == angle:
                raise SolverFailure(f"trim on {variable}: target CL {target:.5f} out of reach "
                                    f"within {variable} bounds [{lo}, {hi}]")
            new_result = solve({**attitude, variable: new_angle})
            new_lift = lift_of(new_result)
            n_solves += 1
            last_solve_residual = (new_lift - target) / target
            if abs(new_angle - angle) >= TRIM_MIN_STEP and new_lift != lift:
                slope = float(np.clip((new_lift - lift) / (new_angle - angle), slope_lo,
                                      slope_hi))
            weight = (target - lift) / (new_lift - lift) if new_lift != lift else np.nan
            if band > 0.0 and tolerance < abs(last_solve_residual) <= band \
               and -0.1 <= weight <= 1.1:
                merged = dict(new_result)
                for key, value in new_result.items():
                    old = result.get(key)
                    if key.endswith("[-]") and isinstance(value, float) \
                       and isinstance(old, float):
                        merged[key] = (1.0 - weight) * old + weight * value
                result, angle = merged, (1.0 - weight) * angle + weight * new_angle
                lift, interpolated = lift_of(result), True
                break
            angle, lift, result = new_angle, new_lift, new_result
        self.trim_slope[level] = slope
        metrics.update({f"trim_{variable}": angle, "trim_solves": n_solves,
                        "trim_residual": (lift - target) / target,
                        "trim_residual_last_solve": last_solve_residual, "trim_slope": slope,
                        "trim_interpolated": interpolated})
        return result

    def _evaluate_section_on_planform(self, design_point, level, metrics,
                                      with_strips) -> tuple[float, dict]:
        target = self.lift_reference(level) if self.problem.trim else None
        result, strips = self._solve_section(design_point, level, metrics, with_strips,
                                             trim_target=target)
        metrics.update({coefficient_name(k): v for k, v in result.items()})
        cd = -float(result["Cx [-]"])
        lift = float(np.hypot(result["Cy [-]"], result["Cz [-]"]))
        metrics.update({"CD": cd, "CL": lift})
        value = cd
        if self.problem.trim:
            # drag at exactly the target lift: the lift residual left by the trim (<= its
            # tolerance) is removed from the induced drag, Cdi ~ CL^2 (first order, exact here)
            value = float(result["Cdprofile [-]"]) \
                + float(result["Cdi [-]"]) * (target / lift) ** 2
            metrics.update({"CL_ref": target, "CD_trimmed": value})
        elif self.problem.objective.get("equal_lift_reference"):
            reference = self.lift_reference(level)
            value = float(result["Cdprofile [-]"]) \
                + float(result["Cdi [-]"]) * (reference / lift) ** 2
            metrics.update({"CL_ref": reference, "CD_equal_lift": value})
        return (value if np.isfinite(value) else np.nan), strips

    # ------------------------------------------------------------------ 3D template
    def _evaluate_3d_template(self, design_point, level, metrics) -> float:
        """TEMPLATE (planform design variables, fake backends only): physical parameters ->
        backend coefficients; objective {"type": "coefficient", "name": "Cx", "sign": -1}."""
        parameters = {**self.problem.parametrization.get("fixed", {}),
                      **to_physical(design_point, self.problem.variables)}
        with tempfile.TemporaryDirectory(prefix="mf3d_", ignore_cleanup_errors=True) as tmp:
            coefficients = self.backends[level - 1].coefficients(parameters, Path(tmp))
        name = self.problem.objective.get("name", "Cx")
        value = float(self.problem.objective.get("sign", 1.0)) * float(coefficients[name])
        metrics.update({**{k: v for k, v in parameters.items() if isinstance(v, (int, float))},
                        **coefficients})
        return value


def _json_friendly(metrics: dict) -> dict:
    out = {}
    for key, value in metrics.items():
        if isinstance(value, (np.floating, np.integer)):
            value = value.item()
        if isinstance(value, float) and not np.isfinite(value):
            value = None
        out[key] = value
    return out
