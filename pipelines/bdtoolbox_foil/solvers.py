"""
Solver backends of the fidelity levels. Each 2D backend returns a polar (alpha, cl, cd and, when
available, cm / cpmin) of a section; the objective is computed from the polar (objectives.py).

2D (operational):
* NeuralFoilBackend: NeuralFoil (model_size xxsmall ... xxxlarge), vectorized over the alpha grid.
* XfoilCoreBackend: bdFoil core XFOIL engine (run_polar with re-panelling, PolarSpec).
3D (templates, signatures aligned on bdFoil; tested with fakes only):
* NpLltBackend: bdFoil/NonPlanarSolver NonPlanarLiftingLine (planform CSV, attitude).
* AvlCoreBackend: bdFoil core campaign.run_case (planform, polars, attitude, settings).

The flow conditions shared by every level ("flow" entry of the configuration: re, n_crit, xtr,
alpha = [start, stop, step]) are the same for all the levels, so that the discrepancy between
levels models the physics and not convention differences.
"""
import hashlib
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import bridge
from .geometry import Section, write_xf

logger = logging.getLogger(__name__)


@dataclass
class Polar:
    """Section polar (arrays over alpha, degrees)."""
    alpha: np.ndarray
    cl: np.ndarray
    cd: np.ndarray
    cm: np.ndarray = None
    cpmin: np.ndarray = None


class SolverFailure(RuntimeError):
    """The solver did not return a usable result (the point is a failed evaluation)."""


def alpha_grid(flow: dict) -> np.ndarray:
    start, stop, step = flow.get("alpha", (-4.0, 12.0, 0.5))
    return np.arange(start, stop + 0.5 * step, step)


# 1/4 ---------------------------------------------------------------------------------------------
class NeuralFoilBackend:
    """NeuralFoil polar of a section (options: model_size)."""
    name = "neuralfoil"

    def __init__(self, flow: dict, model_size: str = "xxxlarge"):
        self.flow = flow
        self.model_size = model_size
        self.nf = bridge.neuralfoil()

    def polar(self, section: Section, workdir: Path = None) -> Polar:
        del workdir
        alpha = alpha_grid(self.flow)
        xtr = self.flow.get("xtr", (0.1, 0.1))
        kwargs = {"alpha": alpha, "Re": float(self.flow["re"]),
                  "n_crit": float(self.flow.get("n_crit", 1.0)), "xtr_upper": float(xtr[0]),
                  "xtr_lower": float(xtr[1]), "model_size": self.model_size}
        if section.kulfan is not None:
            aero = self.nf.get_aero_from_kulfan_parameters(kulfan_parameters=section.kulfan,
                                                           **kwargs)
        else:
            aero = self.nf.get_aero_from_coordinates(coordinates=section.coordinates, **kwargs)
        cl, cd = np.ravel(aero["CL"]), np.ravel(aero["CD"])
        if not np.all(np.isfinite(cl)) or not np.all(np.isfinite(cd)):
            raise SolverFailure(f"NeuralFoil returned non finite values for {section.name}")
        cm = np.ravel(aero.get("CM", np.full_like(cl, np.nan)))
        return Polar(alpha=alpha, cl=cl, cd=cd, cm=cm)


# 2/4 ---------------------------------------------------------------------------------------------
class XfoilCoreBackend:
    """bdFoil core XFOIL polar (options: npane, niter, timeout, workdir)."""
    name = "xfoil"

    def __init__(self, flow: dict, npane: int = 250, niter: int = 100, timeout: float = 120.0,
                 workdir: str = None):
        core = bridge.bdfoil_core()
        self.xfoil = core.xfoil
        self.model = core.model
        start, stop, step = flow.get("alpha", (-4.0, 12.0, 0.5))
        self.spec = self.xfoil.PolarSpec(
            re=float(flow["re"]), ncrit=float(flow.get("n_crit", 1.0)),
            xtr=tuple(flow.get("xtr", (0.1, 0.1))), alpha=(float(start), float(stop), float(step)),
            npane=int(npane), niter=int(niter))
        self.timeout = float(timeout)
        # short working directory (XFOIL / Windows path-length limits of bdFoil core)
        self.workdir = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="mfego_xf_"))

    def polar(self, section: Section, workdir: Path = None) -> Polar:
        del workdir
        path = write_xf(section, self.workdir / "sections")
        ref = self.model.SectionRef(path=path,
                                    sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        try:
            polar = self.xfoil.run_polar(ref, self.spec, workdir=self.workdir / section.name,
                                         timeout=self.timeout)
        except self.model.CoreError as error:
            raise SolverFailure(f"XFOIL failed for {section.name}: {error}") from error
        return Polar(alpha=np.asarray(polar.alpha), cl=np.asarray(polar.cl),
                     cd=np.asarray(polar.cd), cm=np.asarray(polar.cm),
                     cpmin=np.asarray(polar.cpmin))


# 3/4 ---------------------------------------------------------------------------------------------
class NpLltBackend:
    """
    TEMPLATE (3D): bdFoil non-planar lifting line. The planform CSV is produced from the design
    variables by a user-supplied planform_writer(parameters, path) (bdGeometry CSV format).
    Returns the bdFoil coefficients dict (Cx, Cy, Cz, Cdi, Cdprofile...).
    """
    name = "npllt"

    def __init__(self, paths, profiles_folder: str, nodes_count: int = 100,
                 polar_source: str = "neuralfoil", neuralfoil_model: str = "xxxlarge",
                 attitude: dict = None, planform_writer=None, **solver_options):
        self.module = bridge.np_llt_solver(paths)
        self.options = {"profiles_folder": profiles_folder, "nodes_count": int(nodes_count),
                        "polar_source": polar_source, "neuralfoil_model": neuralfoil_model,
                        # house convention (n_crit 1, forced transition at 10 % chord)
                        "neuralfoil_n_crit": 1.0, "neuralfoil_xtr": (0.1, 0.1),
                        **(attitude or {}), **solver_options}
        self.planform_writer = planform_writer

    def coefficients(self, parameters: dict, workdir: Path) -> dict:
        if self.planform_writer is None:
            raise SolverFailure("NpLltBackend needs a planform_writer(parameters, path)")
        csv_path = Path(workdir) / "planform.csv"
        self.planform_writer(parameters, csv_path)
        solver = self.module.NonPlanarLiftingLine(str(csv_path), **self.options)
        solver.solve_lifting_line()
        if not getattr(solver, "converged", False):
            raise SolverFailure("non-planar lifting line did not converge")
        return dict(zip(["Cx", "Cy", "Cz", "Cmx", "Cmy", "Cmz", "Cdi", "Cdprofile", "Speed"],
                        solver.get_bdfoil_coefficients()))


# 4/4 ---------------------------------------------------------------------------------------------
class AvlCoreBackend:
    """
    TEMPLATE (3D): bdFoil core AVL case (campaign.run_case). planform_builder(parameters)
    must return a bdFoil core Planform (model.Planform or model.load_planform(csv)).
    """
    name = "avl"

    def __init__(self, flow: dict, attitude: dict = None, settings: dict = None,
                 planform_builder=None, timeout: float = 60.0):
        core = bridge.bdfoil_core()
        self.core = core
        self.flow = flow
        self.attitude = attitude or {}
        self.settings = settings or {"n_mesh": 40, "nspan": 100, "zsym": -1,
                                     "viscous_model": "profile"}
        self.planform_builder = planform_builder
        self.timeout = float(timeout)

    def coefficients(self, parameters: dict, workdir: Path) -> dict:
        if self.planform_builder is None:
            raise SolverFailure("AvlCoreBackend needs a planform_builder(parameters)")
        from bdFoil.Code.core import campaign, model, xfoil  # noqa: PLC0415
        planform = self.planform_builder(parameters)
        start, stop, step = self.flow.get("alpha", (-12.0, 12.0, 2.0))
        spec = xfoil.PolarSpec(re=float(self.flow["re"]), alpha=(start, stop, step))
        polars = xfoil.get_polars(planform, spec)
        try:
            result = campaign.run_case(planform, polars, model.Attitude(**self.attitude),
                                       model.Settings(**self.settings), workdir=Path(workdir),
                                       timeout=self.timeout)
        except model.CoreError as error:
            raise SolverFailure(f"AVL case failed: {error}") from error
        if not result.ok:
            raise SolverFailure(f"AVL case failed: {result.error}")
        return dict(result.values)


def make_backend(level, flow: dict, paths):
    """Backend instance of a fidelity level (config.Level)."""
    options = dict(level.options)
    if level.solver == "neuralfoil":
        return NeuralFoilBackend(flow, **options)
    if level.solver == "xfoil":
        return XfoilCoreBackend(flow, **options)
    if level.solver == "npllt":
        return NpLltBackend(paths, **options)
    return AvlCoreBackend(flow, **options)
