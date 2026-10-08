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
  "constraints": {"min_thickness": null, "on_violation": "fail"},
  "optimization": {"doe": [20, 10, 4], "iterations": 25, "seed": 0, "estimate_rho": true},
  "paths": {}
}
The design variables are optimized in [0, 1]^d and mapped to [lower, upper] (geometry.py).
"""
import json
from dataclasses import dataclass, field
from pathlib import Path

SOLVERS_2D = ("neuralfoil", "xfoil")
SOLVERS_3D = ("npllt", "avl")
PARAMETRIZATIONS = ("naca4", "kulfan", "parsec")
OBJECTIVES = ("cd_at_cl", "max_lift_to_drag", "coefficient")


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

    @property
    def dim(self) -> int:
        return len(self.variables)

    @property
    def costs(self) -> list:
        return [level.cost for level in self.levels]


def load_config(path: str) -> FoilProblem:
    """
    Reads and validates a configuration file.

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
    return problem


def from_dict(raw: dict) -> FoilProblem:
    """Builds and validates a FoilProblem from a dict (see load_config)."""
    def require(key):
        if key not in raw:
            raise ValueError(f"configuration: missing key '{key}'")
        return raw[key]

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
    optimization = {"doe": [10] * len(levels), "iterations": 20, "seed": 0,
                    "estimate_rho": True, **raw.get("optimization", {})}
    if len(optimization["doe"]) != len(levels):
        raise ValueError("configuration: optimization.doe needs one size per level")
    return FoilProblem(
        name=str(require("name")), kind=kind, parametrization=parametrization,
        variables=variables, levels=levels, flow=dict(raw.get("flow", {})),
        objective=objective, optimization=optimization,
        constraints=dict(raw.get("constraints", {})), paths=dict(raw.get("paths", {})),
        description=str(raw.get("description", "")))
