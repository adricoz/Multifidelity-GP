"""
Solver backends of the fidelity levels. Each 2D backend returns a polar (alpha, cl, cd and, when
available, cm / cpmin) of a section; the objective is computed from the polar (objectives.py).

2D (operational):
* NeuralFoilBackend: NeuralFoil (model_size xxsmall ... xxxlarge), vectorized over the alpha grid.
* XfoilCoreBackend: bdFoil core XFOIL engine (run_polar with re-panelling, PolarSpec).
3D (a fixed planform CSV + one section file -> bdFoil coefficients, see planform.py):
* NpLltBackend: bdFoil/NonPlanarSolver NonPlanarLiftingLine with NeuralFoil polars.
* AvlCoreBackend: bdFoil core campaign.run_case (AVL) with XFOIL polars (core polar cache).

The flow conditions shared by every level ("flow" entry of the configuration: re, n_crit, xtr,
alpha = [start, stop, step]) are the same for all the levels, so that the discrepancy between
levels models the physics and not convention differences. The 3D levels also share the
attitude, the image of the plane z = 0 and the profile drag definition (CD of the section
polar at the strip cl).
"""
import hashlib
import logging
import tempfile
import time
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
# Image of the plane z = 0 (bdFoil convention): "wall" = symmetry plane of a foil attached to the
# hull (AVL iZsym +1, NPLLT surface_type 0: the mirrored horseshoes continue the bound vortex
# across z = 0, no trailing vortex at the root), "free_surface" = high Froude free surface
# (iZsym -1, surface_type 1, zero loading at z = 0), "none" = no image.
IMAGES = {"wall": {"zsym": 1, "surface_type": 0},
          "free_surface": {"zsym": -1, "surface_type": 1},
          "none": {"zsym": 0, "surface_type": None}}

# A lifting line / vortex lattice meeting a WALL image at an angle has a kink singularity at its
# root: the first strip or control point next to the wall gets a meaningless local cl (cl ~ 8 on
# a single AVL strip, alpha ~ 11 deg on one NPLLT control point) carrying a negligible force. The
# stall check (a clamped polar lookup makes the evaluation fail) ignores this root zone, given as
# a fraction of the immersed span; the clamped points are still counted ("n_clamped_total").
ROOT_ZONE_FRACTION = 0.02


def _check_image(image: str) -> dict:
    if image not in IMAGES:
        raise ValueError(f"image must be in {tuple(IMAGES)}, got {image!r}")
    return IMAGES[image]


class NpLltBackend:
    """
    3D level: bdFoil non-planar lifting line (bdFoil/NonPlanarSolver, Phillips & Snyder 2000)
    with NeuralFoil section polars of the model of the level (options: neuralfoil_model,
    nodes_count, core_radius, spacing and any other NonPlanarLiftingLine keyword).

    The flow is the one shared by every level: a single polar Reynolds number (local_reynolds
    False, operating_re = flow.re, like the AVL level), n_crit and xtr of the configuration.
    The attitude (cant = heel + trunk_cant, rake, yaw, sink) and the image of the plane z = 0
    come from the configuration. A wall image met at an angle has a kink singularity at the
    root of the lifting line: the default vortex cut-off is then core_radius = 0.005 (fraction of
    the local chord, as advised by np_llt_solver).

    evaluate_planform(csv, section_dir) returns the bdFoil coefficients ("Cx [-]"...,
    "Cdprofile [-]") and diagnostics: CL2d / CD2d of the section polars at check_alpha_deg at
    EVERY control point (the 2D constraint seen by this level), NeuralFoil confidence, number
    of angles of attack clamped at the ends of the polar table (a failure if fail_on_clamp,
    outside the root zone of a wall image, see ROOT_ZONE_FRACTION).
    """
    name = "npllt"

    def __init__(self, paths, flow: dict, attitude: dict = None, image: str = "wall",
                 kind: str = "FOIL", neuralfoil_model: str = "xxlarge", nodes_count: int = 100,
                 core_radius: float = None, fail_on_clamp: bool = True,
                 check_alpha_deg: float = 0.0, root_zone: float = None, **solver_options):
        self.module = bridge.np_llt_solver(paths)
        logging.getLogger(self.module.__name__).setLevel(logging.WARNING)
        surface_type = _check_image(image)["surface_type"]
        attitude = attitude or {}
        xtr = flow.get("xtr", (0.1, 0.1))
        if core_radius is None:
            core_radius = 0.005 if image == "wall" else 0.0
        self.options = {
            "nodes_count": int(nodes_count), "cant_deg": float(attitude.get("cant", 0.0)),
            "rake_deg": float(attitude.get("rake", 0.0)),
            "yaw_deg": float(attitude.get("yaw", 0.0)), "sink": float(attitude.get("sink", 0.0)),
            "surface_type": surface_type, "operating_re": float(flow["re"]),
            "local_reynolds": False, "polar_source": "neuralfoil", "foil_kind": kind,
            "length_factor": 1.0, "neuralfoil_model": neuralfoil_model,
            "neuralfoil_n_crit": float(flow.get("n_crit", 1.0)),
            "neuralfoil_xtr": (float(xtr[0]), float(xtr[1])), "core_radius": float(core_radius),
            **solver_options}
        self.fail_on_clamp = bool(fail_on_clamp)
        self.check_alpha = np.radians(float(check_alpha_deg))
        self.root_zone = (ROOT_ZONE_FRACTION if image == "wall" else 0.0) \
            if root_zone is None else float(root_zone)

    def coefficients(self, parameters: dict, workdir: Path) -> dict:
        """Planform design variables (template path) are not supported by the real backend."""
        del parameters, workdir
        raise SolverFailure("NpLltBackend: planform design variables are not supported, use a "
                            "fixed planform with a section parametrization")

    def evaluate_planform(self, csv_path: Path, section_dir: Path, with_strips: bool = False,
                          attitude: dict = None) -> dict:
        """
        Coefficients and diagnostics of the planform CSV (sections in section_dir).
        attitude: optional {cant, rake, yaw, sink} overriding the configured one (trim).
        """
        start = time.perf_counter()
        options = dict(self.options)
        if attitude:
            options.update({f"{k}_deg": float(v) for k, v in attitude.items() if k != "sink"})
            if "sink" in attitude:
                options["sink"] = float(attitude["sink"])
        solver = self.module.NonPlanarLiftingLine(str(csv_path), str(section_dir), **options)
        setup_time = time.perf_counter() - start
        solver.solve_lifting_line()
        if not solver.converged:
            raise SolverFailure(f"NPLLT did not converge (max residual {solver.residual_max:.2e}, "
                                f"{solver.n_iterations} iterations)")
        out = {key: float(value) for key, value in solver.get_bdfoil_coefficients().items()
               if value is not None}
        lo, hi = solver.alpha_table_rad
        alpha = np.asarray(solver.alpha_all, dtype=float)
        out_of_table = (alpha < lo) | (alpha > hi)
        s_cp = np.asarray(solver.s_cp_full, dtype=float)
        checked = s_cp > self.root_zone * float(solver.L_arc)
        clamped = int(np.sum(out_of_table & checked))
        cl2d = np.array([float(spline(self.check_alpha)) for spline in solver.cl_splines])
        cd2d = np.array([float(spline(self.check_alpha)) for spline in solver.cd_splines])
        confidence_check = np.array([float(spline(self.check_alpha))
                                     for spline in solver.unique_conf_splines])
        local_conf = np.array([float(solver.unique_conf_splines[k](a)) for k, a in
                               zip(solver.polar_key_index, np.clip(alpha, lo, hi))])
        out.update({"n_clamped": clamped, "n_clamped_total": int(np.sum(out_of_table)),
                    "cl2d_check_min": float(cl2d.min()),
                    "cl2d_check_max": float(cl2d.max()), "cd2d_check_mean": float(cd2d.mean()),
                    "n_sections_checked": int(len(cl2d)),
                    "nf_confidence_min": float(local_conf.min()),
                    "nf_confidence_check": float(confidence_check.min()),
                    "n_iterations": int(solver.n_iterations),
                    "time_setup_s": setup_time, "time_solve_s": time.perf_counter() - start
                    - setup_time})
        if self.fail_on_clamp and clamped:
            raise SolverFailure(f"NPLLT: {clamped} angles of attack outside the polar table "
                                "(stall), the clamped polar values would be exploited")
        if with_strips:
            cd_local = np.array([float(s(a)) for s, a in zip(solver.cd_splines,
                                                               np.clip(alpha, lo, hi))])
            out["strips"] = {"s": np.asarray(solver.s_cp_full, dtype=float).tolist(),
                             "cl": np.asarray(solver.local_cl, dtype=float).tolist(),
                             "cd": cd_local.tolist(),
                             "alpha_deg": np.degrees(alpha).tolist(),
                             "cl2d_check": cl2d.tolist(),
                             "chord": np.asarray(solver.all_chords_cp, dtype=float).tolist()}
        return out


# 4/4 ---------------------------------------------------------------------------------------------
class AvlCoreBackend:
    """
    3D level: bdFoil core AVL (vortex lattice, Trefftz-plane induced drag) with the profile
    drag of XFOIL polars (bdFoil core XFOIL driver and polar cache), the bdFoil "profile"
    viscous model: cd of each strip = CD of the section polar at the strip cl (the same
    definition as the Cdprofile of the NPLLT levels).

    Options: npane, niter (XFOIL), polar_timeout, timeout (AVL), settings (core Settings:
    n_mesh, nspan, nchord... ; zsym comes from the image, viscous_model is always "profile"),
    cache_dir (XFOIL polar cache, default: bdFoil core default), fail_on_clamp, root_zone
    (fraction of the span excluded from the stall check, default ROOT_ZONE_FRACTION with a wall).
    The XFOIL polar uses the flow of the configuration (re, n_crit, xtr, alpha = [start, stop,
    step], 0 included for the 2D check).
    """
    name = "avl"

    def __init__(self, paths, flow: dict, attitude: dict = None, image: str = "wall",
                 kind: str = "FOIL", settings: dict = None, npane: int = 250, niter: int = 100,
                 polar_timeout: float = 300.0, timeout: float = 120.0, cache_dir: str = None,
                 fail_on_clamp: bool = True, check_alpha_deg: float = 0.0,
                 root_zone: float = None):
        del paths
        core = bridge.bdfoil_core()
        from bdFoil.Code.core import campaign, forces  # noqa: PLC0415
        self.core, self.campaign, self.forces = core, campaign, forces
        settings = dict(settings or {})
        if settings.get("viscous_model", "profile") != "profile":
            raise ValueError("AvlCoreBackend: viscous_model must be 'profile' (same profile drag "
                             "definition as the NPLLT levels)")
        settings.update({"zsym": _check_image(image)["zsym"], "viscous_model": "profile"})
        self.settings = core.model.Settings(**settings)
        # default range: down to -6 deg so that the polar covers the low cl of the tip strips
        start, stop, step = flow.get("alpha", (-6.0, 12.0, 0.5))
        self.spec = core.xfoil.PolarSpec(
            re=float(flow["re"]), ncrit=float(flow.get("n_crit", 1.0)),
            xtr=tuple(float(v) for v in flow.get("xtr", (0.1, 0.1))),
            alpha=(float(start), float(stop), float(step)), npane=int(npane), niter=int(niter))
        attitude = attitude or {}
        self.attitude = core.model.Attitude(cant=float(attitude.get("cant", 0.0)),
                                            rake=float(attitude.get("rake", 0.0)),
                                            yaw=float(attitude.get("yaw", 0.0)),
                                            sink=float(attitude.get("sink", 0.0)))
        self.kind = kind
        self.polar_timeout, self.timeout = float(polar_timeout), float(timeout)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.fail_on_clamp = bool(fail_on_clamp)
        self.check_alpha = float(check_alpha_deg)
        self.root_zone = (ROOT_ZONE_FRACTION if image == "wall" else 0.0) \
            if root_zone is None else float(root_zone)

    def coefficients(self, parameters: dict, workdir: Path) -> dict:
        """Planform design variables (template path) are not supported by the real backend."""
        del parameters, workdir
        raise SolverFailure("AvlCoreBackend: planform design variables are not supported, use "
                            "a fixed planform with a section parametrization")

    def evaluate_planform(self, csv_path: Path, section_dir: Path, with_strips: bool = False,
                          attitude: dict = None) -> dict:
        """
        Coefficients and diagnostics of the planform CSV (sections in section_dir).
        attitude: optional {cant, rake, yaw, sink} overriding the configured one (trim); every
        call runs AVL in its own work folder, the XFOIL polar comes from the cache after the
        first call.
        """
        model = self.core.model
        start = time.perf_counter()
        case_attitude = self.attitude if not attitude else model.Attitude(
            cant=float(attitude.get("cant", self.attitude.cant)),
            rake=float(attitude.get("rake", self.attitude.rake)),
            yaw=float(attitude.get("yaw", self.attitude.yaw)),
            sink=float(attitude.get("sink", self.attitude.sink)))
        self._calls = getattr(self, "_calls", 0) + 1
        try:
            planform = model.load_planform(csv_path, name="cfoil", kind=self.kind,
                                           section_dirs=[section_dir])
            polars = self.core.xfoil.get_polars(planform, self.spec, cache_dir=self.cache_dir,
                                                nproc=1, timeout=self.polar_timeout)
            polar_time = time.perf_counter() - start
            result = self.campaign.run_case(planform, polars, case_attitude, self.settings,
                                            workdir=Path(section_dir) / f"avl{self._calls}",
                                            timeout=self.timeout)
        except model.ExeError:
            raise
        except model.CoreError as error:
            raise SolverFailure(f"AVL x XFOIL failed: {error}") from error
        if not result.ok:
            raise SolverFailure(f"AVL case failed: {result.error}")
        out = {key: float(value) for key, value in result.values.items()
               if isinstance(value, (int, float)) and not isinstance(value, bool)}
        # same names as the NPLLT levels: the induced drag that enters Cx (Trefftz-plane CDff);
        # AVL's near-field induced drag is kept for the comparison
        out["Cdi_nearfield [-]"] = out.get("Cdi [-]", np.nan)
        out["Cdi [-]"] = out.get("Cd trefftz [-]", np.nan)
        strips = result.strips
        s_strips = strips["s [m]"].to_numpy(dtype=float)
        checked = s_strips > self.root_zone * float(s_strips.max())
        # profile drag without the root singularity: the first station takes the cl of the first
        # AVL strip, which is meaningless next to a wall image met at an angle (cl ~ 8), and
        # the core clamps it to the end of the polar (high-alpha cd of the section, i.e. a
        # design-dependent artefact of ~0.25 % of CD). The stations of the root zone take the cd
        # of the first checked station; Cx is rebuilt with this profile drag (core value kept).
        cd_strips = strips["cd [-]"].to_numpy(dtype=float)
        area = strips["dA [m^2]"].to_numpy(dtype=float)
        sref = out["Ref Surf [m^2]"]
        out["Cdprofile_core [-]"] = out.get("Cdvisc [-]", np.nan)
        out["Cx_core [-]"] = out["Cx [-]"]
        if np.any(~checked) and np.any(checked):
            cd_strips = np.where(checked, cd_strips, cd_strips[np.argmax(checked)])
            out["Cdprofile [-]"] = float(np.dot(cd_strips, area) / sref)
            out["Cx [-]"] = out["Cx [-]"] + out["Cdprofile_core [-]"] - out["Cdprofile [-]"]
        else:
            out["Cdprofile [-]"] = out["Cdprofile_core [-]"]
        # 2D check and polar quality, per section (one section on the whole planform here)
        cl2d, filled = [], 0
        clamped, clamped_total = 0, 0
        for section in planform.sections:
            polar = polars.polar(section)
            cl2d.append(float(np.interp(self.check_alpha, polar.alpha, polar.cl)))
            filled += int(np.sum(polar.filled))
            branch_cl = np.asarray(polar.cl)[self.forces.monotone_branch(polar)]
            on_section = strips["section"].to_numpy() == section.name
            cl_strips = strips["cl [-]"].to_numpy(dtype=float)
            outside = on_section & ((cl_strips < branch_cl[0]) | (cl_strips > branch_cl[-1]))
            clamped += int(np.sum(outside & checked))
            clamped_total += int(np.sum(outside))
        out.update({"n_clamped": clamped, "n_clamped_total": clamped_total,
                    "cl2d_check_min": float(min(cl2d)),
                    "cl2d_check_max": float(max(cl2d)), "n_sections_checked": len(cl2d),
                    "xfoil_filled_points": filled, "time_polar_s": polar_time,
                    "time_avl_s": time.perf_counter() - start - polar_time})
        if self.fail_on_clamp and clamped:
            raise SolverFailure(f"AVL: {clamped} strips with a cl outside the monotone branch of "
                                "the XFOIL polar (stall), the clamped drag would be exploited")
        if with_strips:
            out["strips"] = {"s": strips["s [m]"].tolist(), "cl": strips["cl [-]"].tolist(),
                             "cd": strips["cd [-]"].tolist(), "y": strips["y [m]"].tolist(),
                             "z": strips["z [m]"].tolist(), "ainc_deg": strips["ainc [°]"].tolist(),
                             "chord": strips["c [m]"].tolist()}
        return out


def make_backend(level, problem, paths):
    """
    Backend instance of a fidelity level (config.Level) of a configuration: the flow shared by
    every level and, for the 3D levels, the attitude, the image and the 2D check angle.
    """
    options = dict(level.options)
    flow = problem.flow
    if level.solver == "neuralfoil":
        return NeuralFoilBackend(flow, **options)
    if level.solver == "xfoil":
        return XfoilCoreBackend(flow, **options)
    from .planform import attitude_angles  # noqa: PLC0415
    common = {"attitude": attitude_angles(problem.attitude or {}),
              "image": problem.image or "wall",
              "kind": (problem.planform or {}).get("kind", "FOIL"),
              "check_alpha_deg": float((problem.section_constraint or {}).get("alpha_deg", 0.0))}
    if level.solver == "npllt":
        return NpLltBackend(paths, flow, **common, **options)
    if problem.paths.get("polar_cache") and "cache_dir" not in options:
        options["cache_dir"] = problem.paths["polar_cache"]
    return AvlCoreBackend(paths, flow, **common, **options)
