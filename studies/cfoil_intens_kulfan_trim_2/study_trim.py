"""
Shared tools of the trim study (third C-foil test): paths, configuration, names and colours of
the three levels, reference section, YAW MODEL and camber projection, KNOWN CONSTRAINTS of the
section, yaw band of the trim, feasible design, loading of an optimization run.

Reference section: the one of the first study (../cfoil_intens_kulfan, study_lib.py): the
thickness of MC2_60_6101_1 (current section of the Intens SY foil, symmetric, t/c 12.25 %)
fitted by a Kulfan section in the thickness/camber form (4 + 4 weights, TE thickness of the
configuration), plus the camber delta C(x) of the first study (delta = REFERENCE_CAMBER, frozen).
The configuration stores it as objective.equal_lift_reference (t_i, s_i = 0, yaw_target = 4).

Design variables: thickness weights t_0..t_3, camber SHAPE weights s_1..s_3 (camber weights
c_i = delta + s_i, s_0 = 0) and yaw_target. There is no 2D lift constraint: the trimmed yaw must
stay within +-YAW_BAND_FRACTION of the nominal yaw, and, without a lift constraint, only a thin
oblique layer of the camber space satisfies it (step 0, first sample of the yaw-model fit with a
free delta: 8 of 300 feasible sections inside the band, 129 out of reach within 1-8 deg). The
camber offset delta is therefore SOLVED (YawProjector): the level-1 yaw model (YawModel, a
quadratic regression of the trimmed yaw of NPLLT x NeuralFoil fitted by step 0 and exact at the
reference) gives the yaw a section needs; delta is the camber offset for which it equals
yaw_target, a design variable bounded by the band. Every design is then inside the band of the
viscous level by construction (up to the model error); the trim of every level still matches the
reference lift exactly, and its yaw is checked against the band (TrimBandSimulator).

Known constraints of the section (feasibility_function), transposed from the aerosandbox airfoil
optimization of a colleague, relative to the reference unless stated, on the section with its
solved delta:
* delta exists within DELTA_BOUNDS (the yaw target can be reached);
* maximum thickness >= constraints.min_thickness (t/c of MC2), no crossing surfaces;
* Ixx >= Ixx_ref: bending stiffness (second moment of area about the centroid, camber included);
* local thickness at x/c = 0.90 and 0.95 >= ref: the trailing edge is not thinner;
* leading-edge weights upper_0 > LE_WEIGHT_MIN and lower_0 < -LE_WEIGHT_MIN (absolute): a
  rounded leading edge on both surfaces (near the leading edge y ~ w_0 sqrt(x), radius w_0^2 / 2);
* wiggliness (sum of the squared second differences of the upper and lower weights)
  < WIGGLINESS_RATIO x ref;
* NeuralFoil analysis confidence > CONFIDENCE_MIN at CONFIDENCE_ALPHAS (model CONFIDENCE_MODEL,
  flow of the configuration): the section lies in the training domain of NeuralFoil, which also
  discards the shapes on which XFOIL does not converge (step 0: 0 failed polar out of 30).
mfego handles them as known constraints: feasible initial design and zero merit outside the
feasible domain (run_optimization(feasibility=...)).
Not transposed: J = Ixx + Iyy (torsion; computed and reported, enforced only if added to
STRUCTURAL), CL(0) >= CL_ref and the multipoint CL targets (replaced by the 3D trim to the lift of
the reference and the yaw band).

Yaw band (TrimBandSimulator): the trim searches the yaw in the trim bounds of the configuration
(wider than the band) and every level records whether its trimmed yaw lies in the band. The band
is ENFORCED on the levels with a viscous lift (BAND_LEVELS: NPLLT x NeuralFoil, NPLLT x XFOIL):
there, outside the band, the objective is penalized, CD + YAW_PENALTY x excess (deg). AVL x XFOIL
is not penalized: its lift only sees the camber line (no viscous decambering), so it often needs
less yaw than the viscous levels (up to 0.55 deg less on the check sections of step 0); its
trimmed yaw is reported. step2_analysis.py never retains a section outside the band of a
BAND_LEVELS level as the optimum (the optimizer itself only sees the penalized values).

The JSON helpers, the logging and the figure style come from the first study
(../cfoil_intens_kulfan: study_lib.py, plotkit.py).
"""
import copy
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from scipy.stats import qmc

STUDY = Path(__file__).resolve().parent
FIRST_STUDY = STUDY.parent / "cfoil_intens_kulfan"
if str(FIRST_STUDY) not in sys.path:
    sys.path.insert(0, str(FIRST_STUDY))

# pylint: disable=wrong-import-position,import-error
import plotkit as pk  # noqa: E402
import study_lib as base  # noqa: E402
from pipelines.bdtoolbox_foil import bridge, geometry  # noqa: E402
from pipelines.bdtoolbox_foil.config import from_dict  # noqa: E402
from pipelines.bdtoolbox_foil.simulator import BdToolboxFoilSimulator  # noqa: E402
from src.data_management import greedy_maximin  # noqa: E402
from src.surrogate_models import load_surrogate  # noqa: E402

CONFIG = STUDY / "config_cfoil_trim.json"
RESULTS = STUDY / "results"
FIGURES = STUDY / "figures"
YAW_MODEL_FILE = RESULTS / "yaw_model_level1.json"
pk.FIGURES = FIGURES          # every figure of this study is saved here
# XFOIL polars only in the cache of the study (the levels pass it explicitly; the variable
# keeps any other core call from writing in the default cache of the bdFoil clone)
os.environ.setdefault("BDFOIL_CORE_CACHE", str(RESULTS / "polar_cache"))

LEVELS = (1, 2, 3)
LEVEL_NAME = {1: "NPLLT x NeuralFoil xxxlarge", 2: "NPLLT x XFOIL", 3: "AVL x XFOIL"}
LEVEL_SHORT = {1: "NPLLT-NF", 2: "NPLLT-XF", 3: "AVL"}
LEVEL_COLOR = {1: pk.LEVEL_COLOR[1], 2: pk.LEVEL_COLOR[2], 3: pk.LEVEL_COLOR[3]}
LEVEL_SYMBOL = {1: "circle", 2: "square", 3: "diamond"}

BASELINE_NAME = "MC2_60_6101_1 + camber"
REFERENCE_CAMBER = 0.108139             # delta of the first study (CL2d(0 deg) = 0.45, xxlarge)
DELTA_BOUNDS = (-0.10, 0.40)            # camber offset solved by the yaw model
TE_STATIONS = (0.90, 0.95)              # local thickness >= reference at these x/c
WIGGLINESS_RATIO = 2.0                  # wiggliness < 2 x reference
LE_WEIGHT_MIN = 0.05                    # upper_0 > 0.05, lower_0 < -0.05
CONFIDENCE_MIN = 0.90                   # NeuralFoil analysis confidence
CONFIDENCE_ALPHAS = (0.0, 2.0, 4.0, 6.0)
CONFIDENCE_MODEL = "xxxlarge"
STRUCTURAL = ("Ixx",)                   # stiffness constraints >= reference ("J" for torsion)
YAW_BAND_FRACTION = 0.05                # trimmed yaw within +-5 % of the nominal yaw
YAW_PENALTY = 0.01                      # CD per degree outside the band
BAND_LEVELS = (1, 2)                    # levels whose trimmed yaw must lie in the band
# stations of the section integrals (cosine spacing, ends included)
STATIONS = geometry.cosine_spacing(401)
# relative tolerance keeping the reference itself feasible (it is ON the >= boundaries)
SLACK = 1.0 - 1e-9

save_json, load_json = base.save_json, base.load_json


def load_problem():
    """FoilProblem of the study (the reference is checked inside the bounds)."""
    problem = base.load_problem(CONFIG)
    if problem.section_constraint:
        raise ValueError("configuration: no section_constraint in this study (delta is solved "
                         "by the yaw model)")
    baseline_parameters(problem)                           # raises if outside the bounds
    return problem


def setup_logging(step: str, path: Path = None) -> Path:
    """Console + results/<step>.log of this study (or path)."""
    return base.setup_logging(step, path or RESULTS / f"{step}.log")


def yaw_band(problem) -> tuple[float, float]:
    """Admissible trimmed yaw (deg): nominal yaw x (1 -+ YAW_BAND_FRACTION)."""
    nominal = float(problem.attitude["yaw"])
    return (nominal * (1.0 - YAW_BAND_FRACTION), nominal * (1.0 + YAW_BAND_FRACTION))


# ------------------------------------------------------------------ reference section
def baseline_parameters(problem) -> dict:
    """
    Reference design: objective.equal_lift_reference (physical values) and its normalized point
    (raises if outside the bounds).
    """
    reference = problem.objective["equal_lift_reference"]
    physical = {v.name: float(reference[v.name]) for v in problem.variables}
    return {"name": BASELINE_NAME, "physical": physical,
            "x": base.to_normalized(problem, physical).tolist()}


def reference_fit_check(problem) -> dict:
    """
    The thickness of the reference against a fresh Kulfan fit of MC2_60_6101_1 (first study,
    study_lib.baseline_parameters, TE thickness of the configuration).
    """
    fit = base.baseline_parameters(problem)
    reference = baseline_parameters(problem)["physical"]
    thickness = [k for k in reference if k.startswith("t_")]
    return {"file": fit["file"], "max_thickness_real": fit["max_thickness_real"],
            "fit_thickness_error": fit["fit_thickness_error"],
            "thickness_weights_fit": {k: fit["fitted_physical"][k] for k in thickness},
            "thickness_weights_gap": max(abs(reference[k] - fit["fitted_physical"][k])
                                         for k in thickness),
            "reference_camber": REFERENCE_CAMBER}


# ------------------------------------------------------------------ section geometry
def physical_columns(problem, points) -> dict:
    """Physical value of every variable and fixed parameter of a batch of normalized points."""
    points = np.atleast_2d(np.asarray(points, dtype=float))
    lower = np.array([v.lower for v in problem.variables])
    upper = np.array([v.upper for v in problem.variables])
    phys = lower + np.clip(points, 0.0, 1.0) * (upper - lower)
    out = {k: np.full(len(phys), float(v))
           for k, v in problem.parametrization.get("fixed", {}).items()}
    out.update({v.name: phys[:, i] for i, v in enumerate(problem.variables)})
    return out


def kulfan_weights(problem, points, delta=None) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Kulfan weights of a batch of normalized points (m, d): thickness weights t (m, n), camber
    weights c (m, n) with c_i = delta + s_i, and the TE thickness. delta: array (m,) or scalar;
    None: the "delta" variable or fixed parameter of the problem (0 if absent). The upper and
    lower weights are c + t / 2 and c - t / 2 (geometry.kulfan_parameters).
    """
    if float(problem.parametrization.get("fixed", {}).get("leading_edge_weight", 0.0)) != 0.0:
        raise ValueError("section properties: leading_edge_weight must be 0")
    columns = physical_columns(problem, points)
    m = len(next(iter(columns.values())))
    n = len([k for k in columns if k.startswith("t_") and k[2:].isdigit()])
    if delta is None:
        delta = columns.get("delta", np.zeros(m))
    delta = np.broadcast_to(np.asarray(delta, dtype=float), (m,))
    t = np.column_stack([columns[f"t_{i}"] for i in range(n)])
    c = np.column_stack([columns.get(f"s_{i}", np.zeros(m)) for i in range(n)]) + delta[:, None]
    te = float(problem.parametrization.get("fixed", {}).get("TE_thickness", 0.0))
    return t, c, te


def section_properties(problem, points, delta=None) -> dict:
    """
    Geometric properties of a batch of normalized points (m, d), chord 1 (delta: see
    kulfan_weights). Vertical strips: the thickness tau(x) = C(x) sum t_i B_i(x) + x TE is
    centred on the camber line y_c(x) = C(x) sum c_i B_i(x), so that (exact up to the quadrature)
        A = int tau,  x_g = int tau x / A,  y_g = int tau y_c / A,
        Ixx = int [tau^3 / 12 + tau (y_c - y_g)^2],   Iyy = int tau (x - x_g)^2
    (second moments of area about the centroid, same definitions as aerosandbox).

    Returns:
    - dict of arrays (m,): Ixx, Iyy, J (= Ixx + Iyy), max_thickness (-inf if the surfaces
      cross), thickness_<x> at TE_STATIONS, wiggliness, upper_0, lower_0 (leading-edge weights),
      le_radius_upper / le_radius_lower (w_0^2 / 2).
    """
    t, c, te = kulfan_weights(problem, points, delta)
    tau = geometry.cst_thickness(STATIONS, t, te)
    y_c = c @ geometry.cst_basis(STATIONS, c.shape[1]).T
    area = trapezoid(tau, STATIONS, axis=1)
    x_g = trapezoid(tau * STATIONS, STATIONS, axis=1) / area
    y_g = trapezoid(tau * y_c, STATIONS, axis=1) / area
    ixx = trapezoid(tau ** 3 / 12.0 + tau * (y_c - y_g[:, None]) ** 2, STATIONS, axis=1)
    iyy = trapezoid(tau * (STATIONS[None, :] - x_g[:, None]) ** 2, STATIONS, axis=1)
    local = geometry.cst_thickness(np.array(TE_STATIONS), t, te)
    # maximum thickness on the stations of the pipeline check (constraints.py)
    thickness = geometry.cst_thickness(geometry.THICKNESS_STATIONS, t, te)
    max_thickness = np.where(thickness.min(axis=1) < 0.0, -np.inf, thickness.max(axis=1))
    upper_w, lower_w = c + t / 2.0, c - t / 2.0
    wiggliness = sum(np.sum(np.diff(w, n=2, axis=1) ** 2, axis=1) for w in (upper_w, lower_w))
    return {"Ixx": ixx, "Iyy": iyy, "J": ixx + iyy, "max_thickness": max_thickness,
            "wiggliness": wiggliness, "upper_0": upper_w[:, 0], "lower_0": lower_w[:, 0],
            "le_radius_upper": upper_w[:, 0] ** 2 / 2.0,
            "le_radius_lower": lower_w[:, 0] ** 2 / 2.0,
            **{f"thickness_{x:.2f}": local[:, k] for k, x in enumerate(TE_STATIONS)}}


def neuralfoil_confidence(problem, points, delta=None, alphas=CONFIDENCE_ALPHAS,
                          model: str = CONFIDENCE_MODEL) -> np.ndarray:
    """
    Minimum NeuralFoil analysis confidence over the angles `alphas` (deg) of a batch of
    normalized points (m, d) (delta: see kulfan_weights), flow of the configuration (Re, n_crit,
    xtr): ONE vectorized call (Kulfan inputs raised exactly from 4 to 8 weights).
    """
    t, c, te = kulfan_weights(problem, points, delta)
    if len(t) == 0:
        return np.zeros(0)
    elevation = geometry.elevation_matrix(t.shape[1], 8)
    n_alpha = len(alphas)
    upper = np.repeat((c + t / 2.0) @ elevation.T, n_alpha, axis=0).T     # (8, m * n_alpha)
    lower = np.repeat((c - t / 2.0) @ elevation.T, n_alpha, axis=0).T
    flow = problem.flow
    xtr = flow.get("xtr", (0.1, 0.1))
    aero = bridge.neuralfoil().get_aero_from_kulfan_parameters(
        kulfan_parameters={"upper_weights": upper, "lower_weights": lower,
                           "leading_edge_weight": 0.0, "TE_thickness": te},
        alpha=np.tile(np.asarray(alphas, dtype=float), len(t)), Re=float(flow["re"]),
        n_crit=float(flow.get("n_crit", 1.0)), xtr_upper=float(xtr[0]),
        xtr_lower=float(xtr[1]), model_size=model)
    confidence = np.ravel(aero["analysis_confidence"]) * np.ones(len(t) * n_alpha)
    return confidence.reshape(len(t), n_alpha).min(axis=1)


# ------------------------------------------------------------------ yaw model
class YawModel:
    """
    Trimmed yaw of NPLLT x NeuralFoil (level 1) as a quadratic polynomial of the section
    parameters (feature names, e.g. t_0..t_3, s_1..s_3, delta; each scaled to [0, 1] by its
    range), fitted by least squares on level-1 trims (step 0), then shifted so that it is exact
    at the reference (whose trimmed yaw is the nominal yaw by definition of the reference lift).
    """
    def __init__(self, names: list, lower, upper, coefficients=None):
        self.names = list(names)
        self.lower, self.upper = np.asarray(lower, float), np.asarray(upper, float)
        self.coefficients = None if coefficients is None else np.asarray(coefficients, float)

    def _basis(self, features: np.ndarray) -> np.ndarray:
        z = (np.atleast_2d(features) - self.lower) / (self.upper - self.lower)
        columns = [np.ones(len(z))] + [z[:, i] for i in range(z.shape[1])]
        columns += [z[:, i] * z[:, j] for i in range(z.shape[1]) for j in range(i, z.shape[1])]
        return np.column_stack(columns)

    def fit(self, features: np.ndarray, yaw: np.ndarray, ridge: float = 1e-8) -> "YawModel":
        basis = self._basis(features)
        gram = basis.T @ basis + ridge * np.eye(basis.shape[1])
        self.coefficients = np.linalg.solve(gram, basis.T @ np.asarray(yaw, float))
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self._basis(features) @ self.coefficients

    def anchor(self, features: np.ndarray, yaw: float) -> None:
        """Shift the constant term so that the model is exact at one point (the reference)."""
        self.coefficients[0] += float(yaw) - float(self.predict(features)[0])

    def solve_delta(self, shape: np.ndarray, yaw: np.ndarray,
                    bounds: tuple = DELTA_BOUNDS) -> np.ndarray:
        """
        Camber offset delta (last feature) for which the model gives `yaw`, for a batch of the
        other features `shape` (m, n - 1): the model is quadratic in delta, the root inside
        `bounds` where the yaw decreases with delta (more camber, more lift, less yaw) is kept;
        NaN where there is none.
        """
        shape = np.atleast_2d(shape)
        yaw = np.broadcast_to(np.asarray(yaw, float), (len(shape),))
        # never extrapolate the quadratic beyond the delta range it was fitted on
        bounds = (max(float(bounds[0]), float(self.lower[-1])),
                  min(float(bounds[1]), float(self.upper[-1])))
        probes = np.array(bounds[0] + np.array([0.0, 0.5, 1.0]) * (bounds[1] - bounds[0]))
        values = np.column_stack([self.predict(np.column_stack([shape, np.full(len(shape), d)]))
                                  for d in probes])                  # (m, 3), exact quadratic
        # y(u) = a u^2 + b u + c through the three probes u = 0, 0.5, 1
        c = values[:, 0] - yaw
        a = 2.0 * values[:, 2] - 4.0 * values[:, 1] + 2.0 * values[:, 0]
        b = values[:, 2] - values[:, 0] - a
        # roots of a u^2 + b u + c = 0 (linear where a ~ 0), on the decreasing branch
        # 2 a u + b < 0, inside [0, 1]; the smallest one is kept
        with np.errstate(divide="ignore", invalid="ignore"):
            root_d = np.sqrt(np.where(b * b - 4.0 * a * c >= 0.0, b * b - 4.0 * a * c, np.nan))
            quadratic = np.abs(a) > 1e-12
            candidates = np.column_stack([
                np.where(quadratic, (-b - root_d) / (2.0 * a), -c / b),
                np.where(quadratic, (-b + root_d) / (2.0 * a), np.nan)])
            valid = (candidates >= -1e-9) & (candidates <= 1.0 + 1e-9) \
                & (2.0 * a[:, None] * candidates + b[:, None] < 0.0)
        u_root = np.where(valid, candidates, np.inf).min(axis=1)
        u_root[~np.isfinite(u_root)] = np.nan
        return probes[0] + np.clip(u_root, 0.0, 1.0) * (probes[-1] - probes[0])

    def to_dict(self) -> dict:
        return {"names": self.names, "lower": self.lower.tolist(), "upper": self.upper.tolist(),
                "coefficients": self.coefficients.tolist()}

    @classmethod
    def from_dict(cls, data: dict) -> "YawModel":
        return cls(data["names"], data["lower"], data["upper"], data["coefficients"])


def load_yaw_model(path: Path = YAW_MODEL_FILE) -> YawModel:
    """Yaw model fitted by step0_check_setup.py (raises with a hint if it is missing)."""
    if not Path(path).is_file():
        raise FileNotFoundError(f"{path} not found: run step0_check_setup.py first (it fits the "
                                "level-1 yaw model)")
    return YawModel.from_dict(load_json(path)["model"])


def shape_names(problem) -> list:
    """Section variables of the problem (thickness and camber shape), without yaw_target."""
    return [v.name for v in problem.variables if v.name != "yaw_target"]


def solve_delta(problem, points, model: YawModel) -> np.ndarray:
    """Camber offset of a batch of normalized design points (NaN where the target is out of
    reach): the yaw model gives yaw_target."""
    columns = physical_columns(problem, points)
    shape = np.column_stack([columns[name] for name in model.names[:-1]])
    return model.solve_delta(shape, columns["yaw_target"])


class YawProjector:
    """
    Camber offset of a design point for the simulator (same interface as the CL2d projector of
    the pipeline: solve(design_point) -> .delta, .cl2d, .confidence, .n_calls; here .cl2d is the
    yaw predicted by the model, i.e. yaw_target).
    """
    def __init__(self, problem, model: YawModel):
        self.problem, self.model = problem, model

    def solve(self, design_point):
        delta = float(solve_delta(self.problem, design_point, self.model)[0])
        if not np.isfinite(delta):
            raise ValueError(f"yaw model: no camber offset in {DELTA_BOUNDS} gives the yaw target")
        target = float(physical_columns(self.problem, design_point)["yaw_target"][0])
        return SimpleNamespace(delta=delta, cl2d=target, confidence=np.nan, n_calls=0)


def fitting_problem(problem, trim_bounds: tuple = (1.0, 8.0)):
    """
    Problem of the yaw-model fit: same flow, levels and reference, variables t_i, s_i (bounds of
    the study) and delta (DELTA_BOUNDS) instead of yaw_target, wide trim bounds (the trim measures
    the yaw a section needs, inside or outside the band).
    """
    source = Path(getattr(problem, "source", None) or CONFIG)
    raw = json.loads(source.read_text(encoding="utf-8"))
    raw["variables"] = [v for v in raw["variables"] if v["name"] != "yaw_target"] + [
        {"name": "delta", "lower": DELTA_BOUNDS[0], "upper": DELTA_BOUNDS[1]}]
    reference = {k: v for k, v in raw["objective"]["equal_lift_reference"].items()
                 if k != "yaw_target"}
    raw["objective"]["equal_lift_reference"] = {**reference, "delta": REFERENCE_CAMBER}
    raw["trim"] = {**raw["trim"], "bounds": list(trim_bounds)}
    raw["paths"] = {k: str((source.parent / v).resolve()) for k, v in raw["paths"].items()}
    out = from_dict(raw)
    out.paths = copy.deepcopy(raw["paths"])
    return out


# ------------------------------------------------------------------ known constraints
def reference_properties(problem) -> dict:
    """section_properties of the reference section (floats, frozen camber REFERENCE_CAMBER)."""
    x = np.asarray(baseline_parameters(problem)["x"])
    delta = None if "delta" in [v.name for v in problem.variables] else REFERENCE_CAMBER
    return {k: float(v[0]) for k, v in section_properties(problem, x, delta).items()}


def _geometric_checks(problem, reference: dict) -> dict:
    """Geometric constraints: name -> function(properties) -> bool array."""
    min_thickness = float(problem.constraints.get("min_thickness", 0.0))
    checks = {"min_thickness": lambda p: np.isfinite(p["max_thickness"])
              & (p["max_thickness"] >= min_thickness)}
    for key in (*STRUCTURAL, *(f"thickness_{x:.2f}" for x in TE_STATIONS)):
        checks[key] = lambda p, key=key: p[key] >= SLACK * reference[key]
    checks["leading_edge_upper"] = lambda p: p["upper_0"] > LE_WEIGHT_MIN
    checks["leading_edge_lower"] = lambda p: p["lower_0"] < -LE_WEIGHT_MIN
    checks["wiggliness"] = lambda p: p["wiggliness"] < WIGGLINESS_RATIO * reference["wiggliness"]
    return checks


def _deltas(problem, points, model) -> np.ndarray:
    """delta of the points: solved by the yaw model, or the delta variable (fitting problem)."""
    if "yaw_target" in [v.name for v in problem.variables]:
        return solve_delta(problem, points, model or load_yaw_model())
    columns = physical_columns(problem, points)
    return columns.get("delta", np.zeros(len(np.atleast_2d(points))))


def constraint_checks(problem, model: YawModel = None) -> dict:
    """
    Every known constraint of the study as a function points (m, d) -> bool array (m,):
    {"delta_solved", "min_thickness", "Ixx", "thickness_0.90", "thickness_0.95",
    "leading_edge_upper", "leading_edge_lower", "wiggliness", "nf_confidence"} (the section of
    a point whose delta cannot be solved violates every constraint).
    """
    reference = reference_properties(problem)
    checks = {"delta_solved": lambda points: np.isfinite(_deltas(problem, points, model))}

    def on_section(check):
        def wrapped(points):
            points = np.atleast_2d(np.asarray(points, dtype=float))
            delta = _deltas(problem, points, model)
            ok = np.isfinite(delta)
            out = np.zeros(len(points), dtype=bool)
            if np.any(ok):
                out[ok] = check(points[ok], delta[ok])
            return out
        return wrapped

    for key, check in _geometric_checks(problem, reference).items():
        checks[key] = on_section(lambda pts, d, check=check: np.asarray(
            check(section_properties(problem, pts, d)), dtype=bool))
    checks["nf_confidence"] = on_section(
        lambda pts, d: neuralfoil_confidence(problem, pts, d) > CONFIDENCE_MIN)
    return checks


def feasibility_function(problem, model: YawModel = None):
    """
    Known constraints of the study (module docstring) for the mfego [KC] hooks:
    feasibility(points (m, d)) -> bool array (m,). delta first, then the geometric constraints,
    then the NeuralFoil confidence on the remaining points only.
    """
    checks = _geometric_checks(problem, reference_properties(problem))
    if "yaw_target" in [v.name for v in problem.variables] and model is None:
        model = load_yaw_model()

    def feasibility(points: np.ndarray) -> np.ndarray:
        points = np.atleast_2d(np.asarray(points, dtype=float))
        delta = _deltas(problem, points, model)
        ok = np.isfinite(delta)
        if np.any(ok):
            properties = section_properties(problem, points[ok], delta[ok])
            sub = np.ones(int(ok.sum()), dtype=bool)
            for check in checks.values():
                sub &= np.asarray(check(properties), dtype=bool)
            if np.any(sub):
                idx = np.flatnonzero(sub)
                sub[idx] = neuralfoil_confidence(problem, points[ok][idx], delta[ok][idx]) \
                    > CONFIDENCE_MIN
            ok[ok] = sub
        return ok

    return feasibility


def feasibility_report(problem, model: YawModel = None, n: int = 2 ** 14,
                       seed: int = 0) -> dict:
    """Share of the design space satisfying each known constraint and all of them (Sobol)."""
    points = qmc.Sobol(problem.dim, seed=seed).random(n)
    shares = {key: float(np.mean(check(points)))
              for key, check in constraint_checks(problem, model).items()}
    shares["all"] = float(np.mean(feasibility_function(problem, model)(points)))
    return shares


def feasible_design(problem, n_points: int, seed: int = 0, oversampling: int = 50,
                    model: YawModel = None) -> np.ndarray:
    """
    Space-filling FEASIBLE design of n_points NORMALIZED points (the simulator input): Latin
    hypercube of oversampling * n_points candidates, known constraints of the study, greedy
    maximin selection (same procedure as the initial design of the optimization).
    """
    candidates = qmc.LatinHypercube(d=problem.dim, seed=seed).random(oversampling * n_points)
    candidates = candidates[feasibility_function(problem, model)(candidates)]
    if len(candidates) < n_points:
        raise ValueError(f"feasible_design: {len(candidates)} feasible candidates for {n_points}"
                         " points (increase oversampling)")
    return greedy_maximin(candidates, n_points, [(0.0, 1.0)] * problem.dim)


# ------------------------------------------------------------------ simulator
class TrimBandSimulator(BdToolboxFoilSimulator):
    """
    Simulator of the study: the camber offset of every section solved by the yaw model
    (YawProjector, when the problem has the yaw_target variable), the trimmed evaluation of the
    pipeline, then the yaw band. The metrics record CD_unpenalized, yaw_excess, trim_in_band and
    band_enforced; on a BAND_LEVELS level a trimmed yaw outside band = (lo, hi) makes the value
    CD + penalty x excess (deg). Failed evaluations stay NaN.
    """
    def __init__(self, problem, band: tuple, penalty: float = YAW_PENALTY,
                 model: YawModel = None, **kwargs):
        super().__init__(problem, **kwargs)
        self.band = (float(band[0]), float(band[1]))
        self.penalty = float(penalty)
        self.trim_key = f"trim_{(problem.trim or {}).get('variable', 'yaw')}"
        if "yaw_target" in [v.name for v in problem.variables]:
            self.projector = YawProjector(problem, model or load_yaw_model())

    def evaluate_details(self, design_point, level: int, with_strips: bool = False):
        value, metrics, strips = super().evaluate_details(design_point, level, with_strips)
        metrics.pop("cl2d_reference_confidence", None)
        metrics.pop("projection_calls", None)
        if "cl2d_reference" in metrics:
            metrics["yaw_model"] = metrics.pop("cl2d_reference")
        angle = metrics.get(self.trim_key)
        if angle is not None and np.isfinite(value):
            excess = max(self.band[0] - float(angle), float(angle) - self.band[1], 0.0)
            enforced = int(level) in BAND_LEVELS
            metrics.update({"CD_unpenalized": float(value), "yaw_excess": excess,
                            "trim_in_band": bool(excess == 0.0), "band_enforced": enforced})
            if enforced:
                value = float(value) + self.penalty * excess
        return value, metrics, strips


def make_simulator(problem, model: YawModel = None) -> TrimBandSimulator:
    """Simulator with the real backends of the three levels and the yaw band of the study."""
    return TrimBandSimulator(problem, yaw_band(problem), model=model)


def level_settings(problem, simulator) -> dict:
    """
    Flow settings each level actually uses (turbulence: n_crit, xtr), read from the backends:
    NeuralFoil inside NPLLT, the XFOIL PolarSpec of NPLLT x XFOIL and of AVL; plus the settings
    of the NeuralFoil confidence constraint. Raises if they differ.
    """
    out = {"confidence_constraint": {"n_crit": float(problem.flow.get("n_crit", 1.0)),
                                     "xtr": [float(v) for v in problem.flow.get("xtr",
                                                                                (0.1, 0.1))],
                                     "re": float(problem.flow["re"]), "model": CONFIDENCE_MODEL}}
    for level, backend in zip(LEVELS, simulator.backends):
        spec = getattr(backend, "spec", None)
        if spec is not None:
            out[LEVEL_NAME[level]] = {"n_crit": float(spec.ncrit), "xtr": list(spec.xtr),
                                      "re": float(spec.re), "alpha": list(spec.alpha),
                                      "npane": int(spec.npane), "niter": int(spec.niter)}
        else:
            options = backend.options
            out[LEVEL_NAME[level]] = {"n_crit": float(options["neuralfoil_n_crit"]),
                                      "xtr": list(options["neuralfoil_xtr"]),
                                      "re": float(options["operating_re"]),
                                      "model": options["neuralfoil_model"]}
    values = {(v["n_crit"], tuple(v["xtr"]), v["re"]) for v in out.values()}
    if len(values) != 1:
        raise ValueError(f"the levels do not share the same flow settings: {out}")
    # the XFOIL levels must use the SAME PolarSpec to share the polar cache
    specs = {repr(backend.spec) for backend in simulator.backends
             if getattr(backend, "spec", None) is not None}
    if len(specs) > 1:
        raise ValueError(f"the XFOIL levels do not share the same PolarSpec: {specs}")
    return out


# ------------------------------------------------------------------ runs
def load_run(run_dir: Path) -> dict:
    """
    Evaluations of a run in their order (simulator counter "evaluation"), with the level, the
    cost, the cumulative cost, the best admissible level-3 value so far (in the band where it is
    enforced, see BAND_LEVELS) and the metrics; plus the state, the results and the trained
    surrogate.
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
                         "error": m.get("error"),
                         **{k: v for k, v in m.items()
                            if isinstance(v, (int, float)) and not isinstance(v, bool)
                            and k != "evaluation"},
                         # flags of the trim (trim_interpolated, trim_in_band)
                         **{k: v for k, v in m.items()
                            if isinstance(v, bool) and k.startswith("trim_")}})
    table = pd.DataFrame(rows).sort_values("evaluation").reset_index(drop=True)
    if "trim_in_band" not in table:
        table["trim_in_band"] = False
    table["trim_in_band"] = table["trim_in_band"].astype("boolean").fillna(False).astype(bool)
    table["admissible"] = table["value"].notna() & (table["trim_in_band"]
                                                    | ~table["level"].isin(BAND_LEVELS))
    table["cost"] = [costs[lv - 1] for lv in table["level"]]
    table["cumulative_cost"] = table["cost"].cumsum()
    top = len(costs)
    best, history = np.inf, []
    for level, value, admissible in zip(table["level"], table["value"], table["admissible"]):
        if level == top and admissible:
            best = min(best, value)
        history.append(best if np.isfinite(best) else np.nan)
    table["best_top"] = history
    return {"run_dir": run_dir, "state": state, "table": table, "costs": costs,
            "config": config, "results": load_json(run_dir / "results.json"),
            "surrogate": load_surrogate(str(run_dir / "surrogate.json"))}
