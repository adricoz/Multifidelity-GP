"""
Shared tools of the trim study (second C-foil test, NACA 4412 baseline): paths, configuration,
names and colours of the levels, baseline section, KNOWN CONSTRAINTS of the section, feasible
design, loading of an optimization run.

Baseline: the NACA 4412, initial section of the Intens foil, fitted by a Kulfan section in the
thickness/camber form (4 + 4 weights, TE thickness of the configuration) with its thickness
scaled to t/c = 12 % (naca4_reference). The configuration stores it (rounded) as
objective.equal_lift_reference: it gives the lift of the trim AND the references of the section
constraints below. Like every section, it is projected on CL2d(0 deg) = 0.45 (camber offset
delta solved by the simulator).

Known constraints of the section (feasibility_function), transposed from the aerosandbox
airfoil optimization, each one relative to the baseline:
* maximum thickness >= constraints.min_thickness, no crossing surfaces (pipeline check);
* Ixx >= Ixx_ref (bending) and Ixx + Iyy >= ref (polar moment about the centroid, torsion);
* local thickness at x/c = 0.90 and 0.95 >= ref (trailing edge);
* wiggliness (sum of the squared second differences of the upper and lower weights)
  < WIGGLINESS_RATIO x ref.
They only need the geometry (cheap, vectorized), so mfego handles them as known constraints:
feasible initial design and zero merit outside the feasible domain
(run_optimization(feasibility=...)).
Not transposed:
* the leading-edge weights (lower_0 < -0.05, upper_0 > 0.05): with s_0 = 0, lower_0 =
  delta - t_0 / 2 depends on the camber offset delta solved by the CL2d projection (NeuralFoil),
  too expensive inside the search; the CL2d constraint already sets the camber;
* CL(alpha = 0) >= CL_ref: replaced by the projection CL2d(0 deg) = 0.45;
* the multipoint CL targets: replaced by the 3D trim to the lift of the baseline.
The section properties use the camber WITHOUT the offset delta (delta = 0): delta changes neither
the thickness nor the wiggliness (same shift of every weight) and changes Ixx by a few % for the
candidate and the reference alike.

The JSON helpers, the logging and the figure style come from the first study
(../cfoil_intens_kulfan: study_lib.py, plotkit.py), whose calibration fixed the levels and the
estimator: see ../cfoil_intens_kulfan/RAPPORT.md.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from scipy.optimize import brentq
from scipy.stats import qmc

STUDY = Path(__file__).resolve().parent
FIRST_STUDY = STUDY.parent / "cfoil_intens_kulfan"
if str(FIRST_STUDY) not in sys.path:
    sys.path.insert(0, str(FIRST_STUDY))

# pylint: disable=wrong-import-position,import-error
import plotkit as pk  # noqa: E402
import study_lib as base  # noqa: E402
from pipelines.bdtoolbox_foil import geometry  # noqa: E402
from pipelines.bdtoolbox_foil.constraints import feasibility_function as thickness_feasibility  # noqa: E402
from src.data_management import greedy_maximin  # noqa: E402
from src.surrogate_models import load_surrogate  # noqa: E402

CONFIG = STUDY / "config_cfoil_trim.json"
RESULTS = STUDY / "results"
FIGURES = STUDY / "figures"
pk.FIGURES = FIGURES          # every figure of this study is saved here

LEVELS = (1, 2)
LEVEL_NAME = {1: "NPLLT x NeuralFoil xxlarge", 2: "AVL x XFOIL"}
LEVEL_SHORT = {1: "NPLLT", 2: "AVL"}
LEVEL_COLOR = {1: pk.LEVEL_COLOR[2], 2: pk.LEVEL_COLOR[3]}
LEVEL_SYMBOL = {1: "square", 2: "diamond"}

BASELINE_NAME = "NACA 4412"
BASELINE_NACA4 = (0.04, 0.4, 0.12)      # camber m, its position p, thickness t (chord fractions)
TE_STATIONS = (0.90, 0.95)              # local thickness >= baseline at these x/c
WIGGLINESS_RATIO = 2.0                  # wiggliness < 2 x baseline
# stations of the section integrals (cosine spacing, ends included)
STATIONS = geometry.cosine_spacing(401)

save_json, load_json = base.save_json, base.load_json
make_simulator = base.make_simulator


def load_problem():
    """FoilProblem of the trim study (the baseline is checked inside the bounds and feasible)."""
    problem = base.load_problem(CONFIG)
    x = np.asarray(baseline_parameters(problem)["x"])      # raises if outside the bounds
    if not thickness_feasibility(problem)(x[None, :])[0]:
        raise ValueError("configuration: the baseline section violates constraints.min_thickness")
    return problem


def setup_logging(step: str, path: Path = None) -> Path:
    """Console + results/<step>.log of this study (or path)."""
    return base.setup_logging(step, path or RESULTS / f"{step}.log")


# ------------------------------------------------------------------ baseline section
def naca4_reference(problem, naca4: tuple = BASELINE_NACA4) -> dict:
    """
    Kulfan thickness/camber parameters of a NACA 4-digit section (how the configuration
    reference was obtained): least-squares fit with as many weights as the configuration and its
    TE thickness, thickness weights scaled so that t/c equals the NACA thickness, s_i = c_i - c_0
    (the mean camber c_0 is left to the offset delta of the CL2d projection).

    Returns:
    - dict with "physical" ({t_i, s_i}), "delta_fit" (= c_0) and the fit errors.
    """
    m, p, thickness = naca4
    coords = geometry.naca4_coordinates(m, p, thickness, 301)
    n_weights = len([v for v in problem.variables if v.name.startswith("t_")])
    te = float(problem.parametrization.get("fixed", {}).get("TE_thickness", 0.0))
    fit = geometry.fit_kulfan_thickness_camber(coords, n_weights, te_thickness=te)
    scale = brentq(lambda k: geometry.cst_thickness(geometry.THICKNESS_STATIONS, k * fit["t"],
                                                    te).max() - thickness, 0.5, 2.0)
    physical = {f"t_{i}": float(scale * w) for i, w in enumerate(fit["t"])}
    physical.update({f"s_{i}": float(c - fit["c"][0]) for i, c in enumerate(fit["c"]) if i > 0})
    return {"physical": physical, "delta_fit": float(fit["c"][0]),
            "thickness_error": fit["thickness_error"], "camber_error": fit["camber_error"]}


def baseline_parameters(problem) -> dict:
    """
    Baseline design: the NACA 4412 fit stored as objective.equal_lift_reference (physical values)
    and its normalized point (raises if outside the bounds).
    """
    reference = problem.objective["equal_lift_reference"]
    physical = {v.name: float(reference[v.name]) for v in problem.variables}
    return {"name": BASELINE_NAME, "physical": physical,
            "x": base.to_normalized(problem, physical).tolist()}


# ------------------------------------------------------------------ known constraints
def kulfan_weights(problem, points) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Kulfan weights of a batch of normalized points (m, d): thickness weights t (m, n), camber
    weights c (m, n) WITHOUT the offset delta (c_i = s_i, s_0 fixed) and the TE thickness.
    The upper and lower weights are c + t / 2 and c - t / 2 (geometry.kulfan_parameters).
    """
    fixed = problem.parametrization.get("fixed", {})
    if float(fixed.get("leading_edge_weight", 0.0)) != 0.0:
        raise ValueError("section properties: leading_edge_weight must be 0")
    points = np.atleast_2d(np.asarray(points, dtype=float))
    names = [v.name for v in problem.variables]
    lower = np.array([v.lower for v in problem.variables])
    upper = np.array([v.upper for v in problem.variables])
    phys = lower + np.clip(points, 0.0, 1.0) * (upper - lower)

    def column(name):
        if name in names:
            return phys[:, names.index(name)]
        return np.full(len(phys), float(fixed.get(name, 0.0)))

    n = len([k for k in [*names, *fixed] if k.startswith("t_") and k[2:].isdigit()])
    t = np.column_stack([column(f"t_{i}") for i in range(n)])
    c = np.column_stack([column(f"s_{i}") for i in range(n)])
    return t, c, float(fixed.get("TE_thickness", 0.0))


def section_properties(problem, points) -> dict:
    """
    Geometric properties of a batch of normalized points (m, d), chord 1, offset delta = 0.
    Vertical strips: the thickness tau(x) = C(x) sum t_i B_i(x) + x TE is centred on the camber
    line y_c(x) = C(x) sum c_i B_i(x), so that (exact up to the quadrature)
        A = int tau,  x_g = int tau x / A,  y_g = int tau y_c / A,
        Ixx = int [tau^3 / 12 + tau (y_c - y_g)^2],   Iyy = int tau (x - x_g)^2
    (second moments of area about the centroid, same definitions as aerosandbox).

    Returns:
    - dict of arrays (m,): Ixx, Iyy, J (= Ixx + Iyy), thickness_<x> at TE_STATIONS, wiggliness.
    """
    t, c, te = kulfan_weights(problem, points)
    tau = geometry.cst_thickness(STATIONS, t, te)
    y_c = c @ geometry.cst_basis(STATIONS, c.shape[1]).T
    area = trapezoid(tau, STATIONS, axis=1)
    x_g = trapezoid(tau * STATIONS, STATIONS, axis=1) / area
    y_g = trapezoid(tau * y_c, STATIONS, axis=1) / area
    ixx = trapezoid(tau ** 3 / 12.0 + tau * (y_c - y_g[:, None]) ** 2, STATIONS, axis=1)
    iyy = trapezoid(tau * (STATIONS[None, :] - x_g[:, None]) ** 2, STATIONS, axis=1)
    local = geometry.cst_thickness(np.array(TE_STATIONS), t, te)
    wiggliness = sum(np.sum(np.diff(w, n=2, axis=1) ** 2, axis=1) for w in (c + t / 2, c - t / 2))
    return {"Ixx": ixx, "Iyy": iyy, "J": ixx + iyy, "wiggliness": wiggliness,
            **{f"thickness_{x:.2f}": local[:, k] for k, x in enumerate(TE_STATIONS)}}


def constraint_checks(problem) -> dict:
    """
    Every known constraint of the study as a function points (m, d) -> bool array (m,):
    {"min_thickness", "Ixx", "J", "thickness_0.90", "thickness_0.95", "wiggliness"}.
    """
    reference = section_properties(problem, np.asarray(baseline_parameters(problem)["x"]))
    reference = {k: float(v[0]) for k, v in reference.items()}
    # a relative tolerance keeps the baseline itself feasible (it is ON every boundary)
    slack = 1.0 - 1e-9
    checks = {"min_thickness": thickness_feasibility(problem)}
    for key in ("Ixx", "J", *(f"thickness_{x:.2f}" for x in TE_STATIONS)):
        checks[key] = lambda points, key=key: section_properties(problem, points)[key] \
            >= slack * reference[key]
    checks["wiggliness"] = lambda points: section_properties(problem, points)["wiggliness"] \
        < WIGGLINESS_RATIO * reference["wiggliness"]
    return checks


def feasibility_function(problem):
    """
    Known constraints of the study (module docstring) for the mfego [KC] hooks:
    feasibility(points (m, d)) -> bool array (m,).
    """
    reference = section_properties(problem, np.asarray(baseline_parameters(problem)["x"]))
    reference = {k: float(v[0]) for k, v in reference.items()}
    thickness_ok = thickness_feasibility(problem)
    slack = 1.0 - 1e-9

    def feasibility(points: np.ndarray) -> np.ndarray:
        points = np.atleast_2d(np.asarray(points, dtype=float))
        p = section_properties(problem, points)
        ok = np.asarray(thickness_ok(points), dtype=bool)
        for key in ("Ixx", "J", *(f"thickness_{x:.2f}" for x in TE_STATIONS)):
            ok &= p[key] >= slack * reference[key]
        ok &= p["wiggliness"] < WIGGLINESS_RATIO * reference["wiggliness"]
        return ok

    return feasibility


def feasibility_report(problem, n: int = 2 ** 14, seed: int = 0) -> dict:
    """Share of the design space satisfying each known constraint and all of them (Sobol)."""
    points = qmc.Sobol(problem.dim, seed=seed).random(n)
    shares = {key: float(np.mean(check(points)))
              for key, check in constraint_checks(problem).items()}
    shares["all"] = float(np.mean(feasibility_function(problem)(points)))
    return shares


def feasible_design(problem, n_points: int, seed: int = 0, oversampling: int = 50) -> np.ndarray:
    """
    Space-filling FEASIBLE design of n_points NORMALIZED points (the simulator input): Latin
    hypercube of oversampling * n_points candidates, known constraints of the study, greedy
    maximin selection (same procedure as the initial design of the optimization).
    """
    candidates = qmc.LatinHypercube(d=problem.dim, seed=seed).random(oversampling * n_points)
    candidates = candidates[feasibility_function(problem)(candidates)]
    if len(candidates) < n_points:
        raise ValueError(f"feasible_design: {len(candidates)} feasible candidates for {n_points}"
                         " points (increase oversampling)")
    return greedy_maximin(candidates, n_points, [(0.0, 1.0)] * problem.dim)


# ------------------------------------------------------------------ runs
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
