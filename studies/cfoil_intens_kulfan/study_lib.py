"""
Shared tools of the C-foil study scripts (step0 ... step4): paths, configuration, baseline
section, simulator, feasible samples, JSON input / output and logging.

Every script of the study starts with:
    import study_lib as lib
    problem = lib.load_problem()
"""
import json
import logging
import sys
from pathlib import Path

import numpy as np

STUDY = Path(__file__).resolve().parent
REPO = STUDY.parents[1]                      # Multifidelity-GP-Clean
for _path in (REPO, REPO / "mfego"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

# pylint: disable=wrong-import-position,import-error
from pipelines.bdtoolbox_foil import geometry  # noqa: E402
from pipelines.bdtoolbox_foil.config import load_config  # noqa: E402
from pipelines.bdtoolbox_foil.constraints import feasibility_function  # noqa: E402
from pipelines.bdtoolbox_foil.simulator import BdToolboxFoilSimulator  # noqa: E402
from scipy.optimize import brentq  # noqa: E402
from scipy.stats import qmc  # noqa: E402
from src.data_management import greedy_maximin  # noqa: E402
from src.optimizer import NumpyEncoder  # noqa: E402

CONFIG = STUDY / "config_cfoil.json"
RESULTS = STUDY / "results"
FIGURES = STUDY / "figures"
# current section of the Intens SY daggerboard / C-foil (symmetric, t/c 12.25 %): the baseline
BASELINE_XF = REPO.parent / "bdFoil" / "Scripts" / "Intens" / "Dagg" / "MC2_60_6101_1.xf"

logger = logging.getLogger("study")


# ------------------------------------------------------------------ configuration
def load_problem(config: Path = CONFIG):
    """FoilProblem of the study configuration (pipelines/bdtoolbox_foil/config.py)."""
    return load_config(config)


def make_simulator(problem) -> BdToolboxFoilSimulator:
    """Simulator with the real backends of the three levels."""
    return BdToolboxFoilSimulator(problem)


def level_label(problem, level: int) -> str:
    """'L2 npllt (xxlarge)' style label of a fidelity level."""
    lv = problem.levels[level - 1]
    model = lv.options.get("neuralfoil_model")
    return f"L{level} {lv.solver}" + (f" ({model})" if model else " (XFOIL)")


# ------------------------------------------------------------------ design space helpers
def to_normalized(problem, physical: dict) -> np.ndarray:
    """Normalized design point [0, 1]^d of physical values {name: value} (checked in bounds)."""
    x = np.array([(physical[v.name] - v.lower) / (v.upper - v.lower) for v in problem.variables])
    if np.any(x < -1e-9) or np.any(x > 1 + 1e-9):
        names = [v.name for v, u in zip(problem.variables, x) if not 0 <= u <= 1]
        raise ValueError(f"design point outside the bounds for {names}")
    return np.clip(x, 0.0, 1.0)


def baseline_parameters(problem, xf_path: Path = BASELINE_XF) -> dict:
    """
    Baseline design: the current section (BASELINE_XF) fitted by a Kulfan section in the
    thickness/camber form with as many thickness weights as the configuration, the weights
    scaled so that its maximum thickness / chord equals the one of the real section, and a zero
    camber shape (s_i = 0): its camber is then the basic mode delta C(x) solved by the CL2d
    constraint.

    Returns:
    - dict with "physical" ({t_i, s_i}), "x" (normalized), and the fit information.
    """
    coords = geometry.read_xf(xf_path)
    n_weights = len([v for v in problem.variables if v.name.startswith("t_")])
    te = float(problem.parametrization.get("fixed", {}).get("TE_thickness", 0.0))
    fit = geometry.fit_kulfan_thickness_camber(coords, n_weights, te_thickness=te)
    target = geometry.max_thickness(coords)

    def thickness_gap(scale):
        return geometry.cst_thickness(geometry.THICKNESS_STATIONS, scale * fit["t"],
                                      te).max() - target

    scale = brentq(thickness_gap, 0.5, 2.0)
    physical = {f"t_{i}": float(scale * w) for i, w in enumerate(fit["t"])}
    physical.update({v.name: 0.0 for v in problem.variables if v.name.startswith("s_")})
    # the configuration stores this section (rounded) as the equal-lift reference: use exactly
    # the same point, so that the baseline drag at equal lift is its raw drag
    fitted = dict(physical)
    reference = problem.objective.get("equal_lift_reference")
    if reference:
        physical = {name: float(value) for name, value in reference.items()}
    return {"physical": physical, "x": to_normalized(problem, physical).tolist(),
            "fitted_physical": fitted, "reference_vs_fit": max(
                abs(physical[k] - fitted[k]) for k in fitted),
            "file": str(xf_path), "max_thickness_real": float(target),
            "fit_thickness_error": fit["thickness_error"], "fit_camber_error":
            fit["camber_error"], "scale": float(scale), "real_camber_weights":
            fit["c"].tolist(), "real_coordinates": coords.tolist()}


def feasible_design(problem, n_points: int, seed: int = 0, oversampling: int = 30) -> np.ndarray:
    """
    Space-filling FEASIBLE design of n_points normalized points: Latin hypercube of
    oversampling * n_points candidates, known geometric constraints, greedy maximin selection
    (same procedure as the initial design of the optimization).
    """
    candidates = qmc.LatinHypercube(d=problem.dim, seed=seed).random(oversampling * n_points)
    feasibility = feasibility_function(problem)
    if feasibility is not None:
        candidates = candidates[feasibility(candidates)]
    return greedy_maximin(candidates, n_points, [(0.0, 1.0)] * problem.dim)


# ------------------------------------------------------------------ input / output
def save_json(path: Path, obj) -> Path:
    """JSON file (numpy arrays and scalars accepted)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, cls=NumpyEncoder), encoding="utf-8")
    return path


def load_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def setup_logging(step: str, path: Path = None) -> Path:
    """Log to the console and to results/<step>.log (overwritten at each run) or to path."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = Path(path) if path is not None else RESULTS / f"{step}.log"
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for handler in list(root.handlers):
        root.removeHandler(handler)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S")
    for handler in (logging.FileHandler(path, mode="w", encoding="utf-8"),
                    logging.StreamHandler(sys.stdout)):
        handler.setFormatter(formatter)
        root.addHandler(handler)
    # quiet libraries (the solvers log every evaluation at INFO level)
    for name in ("np_llt_solver", "bdFoil", "pipelines.bdtoolbox_foil.bridge"):
        logging.getLogger(name).setLevel(logging.WARNING)
    # the root strip next to the wall image is always clamped by the core viscous model (root
    # singularity, counted in the n_clamped_total metric): one warning per AVL case otherwise
    logging.getLogger("bdFoil.Code.core.forces").setLevel(logging.ERROR)
    return path

