"""
Configuration of a foil optimization (JSON file, see pipelines/configs/).

Example (section2d_naca_3levels.json):
{
  "name": "naca4_cd_at_cl",
  "kind": "section2d",
  "parametrization": {"type": "naca4", "n_points": 161, "fixed": {"p_position": 0.4}},
  "variables": [{"name": "m_camber", "lower": 0.0, "upper": 0.06}, ...],
  "flow": {"re": 5e5, "n_crit": 1.0, "xtr": [0.1, 0.1], "alpha": [-4.0, 12.0, 0.5]},
  "levels": [{"solver": "neuralfoil", "cost": 1.0, "options": {"model_size": "xxsmall"}}, ...],
  "objective": {"type": "cd_at_cl", "cl_target": 0.5},
  "constraints": {"min_thickness": 0.08},
  "optimization": {"doe": [20, 10, 4], "iterations": 25, "seed": 0, "use_map": true},
  "paths": {}
}
A "foil3d" configuration optimizes the SECTION of a fixed 3D planform (planform.py); it adds
"planform", "attitude", "image" and optionally "section_constraint" (section_constraint.py), and
its objective is {"type": "cd3d"} (3D drag coefficient CD = -Cx of the levels npllt / avl) or,
at equal lift, {"type": "cd3d", "equal_lift_reference": {<variable>: <physical value>, ...}}:
CD* = Cdprofile + Cdi (CL_ref / CL)^2, CL = |(Cy, Cz)|, CL_ref = lift of the reference section
at the same level (see simulator.py).
The legacy template with planform design variables ("parametrization": {"type": "planform"})
is kept for the tests with fake backends.

Every key is checked: an unknown key (e.g. a misspelt "use_map") is an error, not ignored.
The design variables are optimized in [0, 1]^d and mapped to [lower, upper] (geometry.py).
"""
import json
from dataclasses import dataclass, field
from pathlib import Path

SOLVERS_2D = ("neuralfoil", "xfoil")
SOLVERS_3D = ("npllt", "avl")
PARAMETRIZATIONS = ("naca4", "kulfan", "parsec")
OBJECTIVES = ("cd_at_cl", "max_lift_to_drag", "coefficient", "cd3d")
IMAGES = ("wall", "free_surface", "none")
TOP_LEVEL_KEYS = ("name", "description", "kind", "parametrization", "variables", "flow",
                  "levels", "objective", "constraints", "optimization", "paths", "planform",
                  "attitude", "image", "section_constraint", "trim")
TRIM_KEYS = ("variable", "bounds", "tolerance", "max_iterations", "initial_slope",
             "interpolation_band")
# optimization entry: defaults ([MAP] MAP estimation of the GP hyperparameters by default, with
# the InvGamma(3, 2) lengthscale prior of mfego; [KC] known geometric constraints in the search)
OPTIMIZATION_DEFAULTS = {"iterations": 20, "seed": 0, "estimate_rho": True,
                         "min_points_rho": None, "use_map": True, "lengthscale_prior": (3.0, 2.0),
                         "n_restarts": 3, "verify_best": True, "stop_on_convergence": False,
                         "known_constraints": True, "verify_every": None}
PATH_KEYS = ("bdfoil_root", "bdtoolbox_root", "soft_dir", "polar_cache")
CONSTRAINT_KEYS = ("min_thickness", "max_thickness")
SECTION_CONSTRAINT_KEYS = ("cl2d_target", "alpha_deg", "reference_model", "delta_bounds",
                           "tolerance", "n_grid")
# AVL 3.40 airfoil buffer (AFILE): 321 points fail with "READBL: Too many airfoil points"
AVL_MAX_AIRFOIL_POINTS = 300


@dataclass
class Variable:
    """Design variable: physical bounds [lower, upper] (optimized in [0, 1])."""
    name: str
    lower: float
    upper: float


@dataclass
class Level:
    """Fidelity level: solver, relative cost and solver options."""
    solver: str
    cost: float
    options: dict = field(default_factory=dict)


@dataclass
class FoilProblem:
    """Validated configuration of a foil optimization."""
    name: str
    kind: str
    parametrization: dict
    variables: list
    levels: list
    flow: dict
    objective: dict
    optimization: dict
    constraints: dict = field(default_factory=dict)
    paths: dict = field(default_factory=dict)
    description: str = ""
    source: str = ""
    planform: dict = field(default_factory=dict)
    attitude: dict = field(default_factory=dict)
    image: str = "wall"
    section_constraint: dict = field(default_factory=dict)
    trim: dict = field(default_factory=dict)

    @property
    def dim(self) -> int:
        return len(self.variables)

    @property
    def costs(self) -> list:
        return [level.cost for level in self.levels]

    @property
    def section_on_planform(self) -> bool:
        """True for a 3D problem whose design variables describe the section."""
        return self.kind == "foil3d" and self.parametrization.get("type") != "planform"


def load_config(path: str) -> FoilProblem:
    """
    Reads and validates a configuration file. Relative "paths" entries are resolved against
    the folder of the file.

    Args:
    - path: JSON configuration file.
    Returns:
    - FoilProblem.
    Raises:
    - ValueError with an explicit message for an invalid configuration.
    """
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    problem = from_dict(raw)
    problem.source = str(Path(path).resolve())
    folder = Path(path).resolve().parent
    for key, value in list(problem.paths.items()):
        if value and not Path(value).is_absolute():
            problem.paths[key] = str((folder / value).resolve())
    return problem


def _check_keys(entry: dict, allowed, where: str) -> None:
    unknown = sorted(set(entry) - set(allowed))
    if unknown:
        raise ValueError(f"configuration: unknown key(s) {unknown} in {where} "
                         f"(allowed: {sorted(allowed)})")


def _check_3d(raw: dict, parametrization: dict, objective: dict) -> None:
    """Checks of a foil3d configuration whose design variables describe the section."""
    from .planform import ATTITUDE_KEYS, CFoilArc  # noqa: PLC0415
    for key in ("planform", "attitude"):
        if key not in raw:
            raise ValueError(f"configuration: a foil3d section optimization needs '{key}'")
    CFoilArc.from_config(raw["planform"])
    _check_keys(raw["attitude"], ATTITUDE_KEYS, "attitude")
    if raw.get("image", "wall") not in IMAGES:
        raise ValueError(f"configuration: image must be in {IMAGES}")
    if parametrization.get("type") != "kulfan":
        raise ValueError("configuration: a foil3d section optimization needs a kulfan "
                         "parametrization")
    if any(level.get("solver") == "avl" for level in raw.get("levels", [])) and \
       2 * int(parametrization.get("n_points", 161)) - 1 > AVL_MAX_AIRFOIL_POINTS:
        raise ValueError(f"configuration: AVL reads at most {AVL_MAX_AIRFOIL_POINTS} airfoil "
                         "points (AFILE): parametrization.n_points must be <= "
                         f"{(AVL_MAX_AIRFOIL_POINTS + 1) // 2} per side")
    for level in raw.get("levels", []):
        shared = sorted({"attitude", "image", "kind", "check_alpha_deg", "flow"}
                        & set(level.get("options", {})))
        if shared:
            raise ValueError(f"configuration: level options {shared} are shared by every level "
                             "and come from the top-level entries (attitude, image, planform, "
                             "section_constraint, flow)")
    if objective.get("type") != "cd3d":
        raise ValueError("configuration: the objective of a foil3d section optimization is "
                         "{'type': 'cd3d'}")
    _check_keys(objective, ("type", "equal_lift_reference"), "objective")
    reference = objective.get("equal_lift_reference")
    if reference is not None:
        names = {v["name"] for v in raw["variables"]}
        if set(reference) != names:
            raise ValueError("configuration: objective.equal_lift_reference must give a value "
                             f"for every design variable {sorted(names)}")
        for v in raw["variables"]:
            if not float(v["lower"]) <= float(reference[v["name"]]) <= float(v["upper"]):
                raise ValueError(f"configuration: equal_lift_reference {v['name']} outside its "
                                 "bounds")
    trim = raw.get("trim") or {}
    if trim:
        _check_keys(trim, TRIM_KEYS, "trim")
        variable = trim.get("variable", "yaw")
        if variable not in ("yaw", "rake"):
            raise ValueError("configuration: trim.variable must be 'yaw' or 'rake'")
        lo, hi = (float(v) for v in trim.get("bounds", (-5.0, 15.0)))
        nominal = float(raw["attitude"].get(variable, 0.0))
        if not lo < nominal < hi:
            raise ValueError(f"configuration: trim.bounds must contain the nominal {variable} "
                             f"({nominal}) with lower < upper")
        if not float(trim.get("tolerance", 1e-4)) > 0.0 \
           or int(trim.get("max_iterations", 8)) < 1 \
           or float(trim.get("interpolation_band", 0.005)) < 0.0 \
           or float(trim.get("initial_slope", 0.1)) == 0.0:
            raise ValueError("configuration: trim needs tolerance > 0, max_iterations >= 1, "
                             "interpolation_band >= 0 and initial_slope != 0")
        if not objective.get("equal_lift_reference"):
            raise ValueError("configuration: a trim needs objective.equal_lift_reference (the "
                             "section whose lift is kept)")
    section_constraint = raw.get("section_constraint") or {}
    if section_constraint:
        _check_keys(section_constraint, SECTION_CONSTRAINT_KEYS, "section_constraint")
        if "cl2d_target" not in section_constraint:
            raise ValueError("configuration: section_constraint needs 'cl2d_target'")
        if parametrization.get("form") != "thickness_camber":
            raise ValueError("configuration: section_constraint needs the kulfan "
                             "'thickness_camber' form (solved camber offset delta)")


def _check_prior(prior, num_levels: int):
    """[MAP] None, one [alpha, beta] pair or one pair per level (alpha > 0, beta > 0)."""
    if prior is None:
        return None
    try:
        values = [[float(v) for v in prior]] if not hasattr(prior[0], "__len__") \
            else [[float(v) for v in pair] for pair in prior]
    except (TypeError, ValueError, IndexError) as error:
        raise ValueError("configuration: optimization.lengthscale_prior must be [alpha, beta] "
                         "or one [alpha, beta] per level") from error
    if len(values) not in (1, num_levels) or \
       not all(len(pair) == 2 and pair[0] > 0 and pair[1] > 0 for pair in values):
        raise ValueError("configuration: optimization.lengthscale_prior must be [alpha, beta] "
                         "(alpha > 0, beta > 0) or one such pair per level")
    return values[0] if len(values) == 1 and not hasattr(prior[0], "__len__") else values


def from_dict(raw: dict) -> FoilProblem:
    """Builds and validates a FoilProblem from a dict (see load_config)."""
    def require(key):
        if key not in raw:
            raise ValueError(f"configuration: missing key '{key}'")
        return raw[key]

    _check_keys(raw, TOP_LEVEL_KEYS, "the configuration")
    kind = require("kind")
    if kind not in ("section2d", "foil3d"):
        raise ValueError(f"configuration: kind must be 'section2d' or 'foil3d', got {kind!r}")
    variables = [Variable(v["name"], float(v["lower"]), float(v["upper"]))
                 for v in require("variables")]
    if not variables:
        raise ValueError("configuration: at least one design variable is required")
    for v in variables:
        if not v.upper > v.lower:
            raise ValueError(f"configuration: variable {v.name!r} needs upper > lower")
    levels = [Level(lv["solver"], float(lv.get("cost", 1.0)), dict(lv.get("options", {})))
              for lv in require("levels")]
    allowed = SOLVERS_2D if kind == "section2d" else SOLVERS_3D
    for level in levels:
        if level.solver not in allowed:
            raise ValueError(f"configuration: solver {level.solver!r} not in {allowed} "
                             f"for kind {kind!r}")
        if level.cost <= 0:
            raise ValueError("configuration: level costs must be > 0")
    parametrization = dict(require("parametrization"))
    if kind == "section2d" and parametrization.get("type") not in PARAMETRIZATIONS:
        raise ValueError(f"configuration: parametrization type must be in {PARAMETRIZATIONS}")
    objective = dict(require("objective"))
    if objective.get("type") not in OBJECTIVES:
        raise ValueError(f"configuration: objective type must be in {OBJECTIVES}")
    if objective["type"] == "cd_at_cl" and "cl_target" not in objective:
        raise ValueError("configuration: objective 'cd_at_cl' needs 'cl_target'")
    if kind == "foil3d" and parametrization.get("type") != "planform":
        _check_3d(raw, parametrization, objective)
    else:
        unused = sorted({"planform", "attitude", "image", "section_constraint", "trim"}
                        & set(raw))
        if unused:
            raise ValueError(f"configuration: {unused} only apply to a foil3d section "
                             "optimization (kind 'foil3d' with a section parametrization)")
    optimization = {"doe": [10] * len(levels), **OPTIMIZATION_DEFAULTS,
                    **raw.get("optimization", {})}
    _check_keys(optimization, ["doe", *OPTIMIZATION_DEFAULTS], "optimization")
    if len(optimization["doe"]) != len(levels):
        raise ValueError("configuration: optimization.doe needs one size per level")
    optimization["lengthscale_prior"] = _check_prior(optimization["lengthscale_prior"],
                                                     len(levels))
    every = optimization.get("verify_every")
    if every is not None and (isinstance(every, bool) or not isinstance(every, int)
                              or every < 1):
        raise ValueError("configuration: optimization.verify_every must be null or an integer "
                         ">= 1")
    constraints = dict(raw.get("constraints", {}))
    _check_keys(constraints, CONSTRAINT_KEYS, "constraints")
    paths = dict(raw.get("paths", {}))
    _check_keys(paths, PATH_KEYS, "paths")
    return FoilProblem(
        name=str(require("name")), kind=kind, parametrization=parametrization,
        variables=variables, levels=levels, flow=dict(raw.get("flow", {})),
        objective=objective, optimization=optimization, constraints=constraints, paths=paths,
        description=str(raw.get("description", "")), planform=dict(raw.get("planform", {})),
        attitude=dict(raw.get("attitude", {})), image=str(raw.get("image", "wall")),
        section_constraint=dict(raw.get("section_constraint") or {}),
        trim=dict(raw.get("trim") or {}))
