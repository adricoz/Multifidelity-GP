"""
Solver backends of the fidelity levels. Each 2D backend returns a polar (alpha, cl, cd and, when
available, cm / cpmin) of a section; the objective is computed from the polar (objectives.py).

2D (operational):
* NeuralFoilBackend: NeuralFoil (model_size xxsmall ... xxxlarge), vectorized over the alpha grid.
* XfoilBackend: bdfoil_v1 XFOIL engine (run_polar, PolarSpec).
3D (a fixed planform CSV + one section file -> foil.py-format coefficients, see planform.py):
* BdfoilV1Backend: one case of bdfoil_v1 (Session.evaluate) with the AVL backend or the
  non-planar lifting line (NP-LLT), and XFOIL or NeuralFoil xxxlarge polars on the Reynolds
  lattice of bdfoil_v1 (local Re of every strip). Since 2026-10-27, replacing the old bdFoil core
  (AVL) and bdFoil/NonPlanarSolver (NPLLT) backends (git tag pre-bdfoil-v1).

The flow conditions shared by every level ("flow" entry of the configuration: re, n_crit, xtr,
alpha = [start, stop, step]) are the same for all the levels, so that the discrepancy between
levels models the physics and not convention differences. The 3D levels also share the
attitude, the image of the plane z = 0 and the post-processing of bdfoil_v1 (viscous drag and
Cpmin of every strip at its cl and local Re).
"""
import hashlib
import json
import logging
import os
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


# 1/3 ---------------------------------------------------------------------------------------------
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


# 2/3 ---------------------------------------------------------------------------------------------
class XfoilBackend:
    """bdfoil_v1 XFOIL polar of a section (options: npane, niter, timeout, workdir)."""
    name = "xfoil"

    def __init__(self, flow: dict, npane: int = 250, niter: int = 100, timeout: float = 120.0,
                 workdir: str = None):
        package = bridge.bdfoil_v1()
        self.xfoil, self.model = package.xfoil, package.model
        start, stop, step = flow.get("alpha", (-4.0, 12.0, 0.5))
        self.spec = self.xfoil.PolarSpec(
            re=float(flow["re"]), ncrit=float(flow.get("n_crit", 1.0)),
            xtr=tuple(float(v) for v in flow.get("xtr", (0.1, 0.1))),
            alpha=(float(start), float(stop), float(step)), npane=int(npane), niter=int(niter))
        self.timeout = float(timeout)
        # short working directory (XFOIL / Windows path-length limits)
        self.workdir = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="mfego_xf_"))

    def polar(self, section: Section, workdir: Path = None) -> Polar:
        del workdir
        path = write_xf(section, self.workdir / "sections").resolve()
        ref = self.model.SectionRef(path=path,
                                    sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        try:
            polar = self.xfoil.run_polar(ref, self.spec, workdir=self.workdir / section.name,
                                         timeout=self.timeout)
        except self.model.ExeError:
            raise
        except self.model.CoreError as error:
            raise SolverFailure(f"XFOIL failed for {section.name}: {error}") from error
        return Polar(alpha=np.asarray(polar.alpha), cl=np.asarray(polar.cl),
                     cd=np.asarray(polar.cd), cm=np.asarray(polar.cm),
                     cpmin=np.asarray(polar.cpmin))


# 3/3 ---------------------------------------------------------------------------------------------
# Image of the plane z = 0 (bdfoil_v1 boundary.zsym): "wall" = symmetry plane of a foil attached
# to the hull (+1: the mirrored vortices continue the bound vortex across z = 0), "free_surface" =
# high Froude free surface (-1, zero loading at z = 0), "none" = no image (0).
IMAGES = {"wall": 1, "free_surface": -1, "none": 0}

# A lifting line / vortex lattice meeting a WALL image at an angle has a kink singularity at its
# root: the first strip or panel next to the wall gets a meaningless local cl carrying a
# negligible force. The stall check (a clamped polar lookup makes the evaluation fail) ignores
# this root zone, given as a fraction of the immersed span; the clamped strips are still counted
# ("n_clamped_total").
ROOT_ZONE_FRACTION = 0.02
POLAR_PROVIDERS = ("xfoil", "neuralfoil")


def _check_image(image: str) -> int:
    if image not in IMAGES:
        raise ValueError(f"image must be in {tuple(IMAGES)}, got {image!r}")
    return IMAGES[image]


class BdfoilV1Backend:
    """
    3D level: one bdfoil_v1 case per evaluation (Study.from_dict -> Session.evaluate), solver
    "avl" (vortex lattice, Trefftz-plane induced drag) or "npllt" (non-planar lifting line,
    near-field loads), polars "xfoil" or "neuralfoil" (xxxlarge, fitted Kulfan of the section).

    Inputs of the case (the conventions of Antoine's bdfoil_v1 cases where nothing else is said):
    - the planform CSV and its section files (planform.py), lengths in metres;
    - the flow of the configuration: re = Reynolds number on the reference chord, n_crit, xtr,
      alpha grid of the polars; the polars are computed on the Reynolds lattice of bdfoil_v1
      that brackets the local Re of the strips (8 nodes per decade by default);
    - speed: the speed of the case [m/s], by default the one that gives re on the reference
      chord in seawater at `temperature` (V = re nu / chord; decision of 2026-10-27: a fixed
      reference speed, the surrogate has no speed input);
    - wave_drag False by default (decision of 2026-10-27: the VPP scales the coefficients by
      V^2, a Froude-dependent term would be wrong at a fixed speed), free_surface_height (depth
      of the strips for sigma and the wave drag);
    - cp_min_percentile: the Cpmin of the foil = this percentile of the Cpmin of the strips
      (bdfoil_v1 default 1; 20 = the production RS of the C-foil).
    Options of the solvers: nchord, n_mesh, nspan (AVL), n_panels (NP-LLT), npane, niter,
    polar_timeout (XFOIL), timeout (AVL), cache_dir (polar cache, env BDFOIL_V1_CACHE of
    bdfoil_v1, <= 182 characters), fail_on_clamp, root_zone (fraction of the span excluded from
    the stall check, default ROOT_ZONE_FRACTION with a wall), neuralfoil_min_confidence (polars
    "neuralfoil": the confidence threshold of the NeuralFoil polars, default None = bdfoil_v1's
    0.90; see _set_neuralfoil_confidence).

    evaluate_planform(csv, section_dir) returns the foil.py-format coefficients of bdfoil_v1
    ("Cx [-]" ... "Cmz [-]": moments M / (q Sref) in metres about (0, 0, sink), BD frame x
    forward, y port, z up; "Cpmin [-]", "Ref Surf [m^2]"...), plus the names the simulator uses:
    "Cdprofile [-]" (viscous drag, = "Cdvisc [-]") and "Cdi [-]" (the induced drag in Cx:
    Trefftz plane for AVL, near field for the NP-LLT), and diagnostics (clamped strips, filled
    polar points, Reynolds range, NP-LLT convergence).
    """

    def __init__(self, solver: str, flow: dict, attitude: dict = None, image: str = "wall",
                 kind: str = "FOIL", reference_chord: float = None, polars: str = "xfoil",
                 speed: float = None, temperature: float = 15.0,
                 free_surface_height: float = 0.0, wave_drag: bool = False,
                 cp_min_percentile: float = 20.0, nchord: int = 8, n_mesh: int = None,
                 nspan: int = None, n_panels: int = None, npane: int = 250, niter: int = 100,
                 polar_timeout: float = 120.0, timeout: float = 60.0, cache_dir: str = None,
                 fail_on_clamp: bool = True, root_zone: float = None,
                 neuralfoil_min_confidence: float = None):
        if solver not in ("avl", "npllt"):
            raise ValueError(f"BdfoilV1Backend: solver must be 'avl' or 'npllt', got {solver!r}")
        if polars not in POLAR_PROVIDERS:
            raise ValueError(f"BdfoilV1Backend: polars must be in {POLAR_PROVIDERS}, got "
                             f"{polars!r}")
        package = bridge.bdfoil_v1()
        self.package, self.model = package, package.model
        self.name, self.polars = solver, polars
        zsym = _check_image(image)
        attitude = attitude or {}
        self.attitude = {key: float(attitude.get(key, 0.0)) for key in ("cant", "rake", "yaw",
                                                                         "sink")}
        if speed is None:
            if not reference_chord:
                raise ValueError("BdfoilV1Backend: give speed or reference_chord (speed = re nu "
                                 "/ chord)")
            nu = package.fluid.seawater(float(temperature)).nu
            speed = float(flow["re"]) * nu / float(reference_chord)
        self.speed = float(speed)
        start, stop, step = flow.get("alpha", (-12.0, 12.0, 0.5))
        self.case = {
            "schema_version": "1.0", "name": f"mfego_{solver}_{polars}",
            "description": "one evaluation of pipelines/bdtoolbox_foil (Multifidelity-GP-Clean)",
            "fluid": {"temperature": float(temperature)},
            "boundary": {"zsym": zsym, "free_surface_height": float(free_surface_height)},
            "appendage": {"kind": kind, "planform": {"csv": "planform.csv", "length_unit": "m"}},
            "solver": {"backend": solver,
                       "avl": {"nchord": int(nchord), "n_mesh": n_mesh, "nspan": nspan,
                               "timeout": float(timeout)},
                       "npllt": {"n_mesh": n_mesh, "n_panels": n_panels}},
            "polars": {"provider": polars, "ncrit": float(flow.get("n_crit", 1.0)),
                       "xtr": [float(v) for v in flow.get("xtr", (0.1, 0.1))],
                       "alpha": [float(start), float(stop), float(step)], "npane": int(npane),
                       "niter": int(niter), "timeout": float(polar_timeout)},
            "post": {"cp_min_percentile": float(cp_min_percentile), "wave_drag": bool(wave_drag)},
            "attitudes": {"list": [{**self.attitude, "speed": self.speed}]},
            "outputs": {"strips": True}}
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir is not None and len(str(self.cache_dir.resolve())) > 182:
            raise ValueError(f"BdfoilV1Backend: cache_dir longer than 182 characters (Windows "
                             f"MAX_PATH of the polar cache): {self.cache_dir}")
        self.fail_on_clamp = bool(fail_on_clamp)
        self.root_zone = (ROOT_ZONE_FRACTION if image == "wall" else 0.0) \
            if root_zone is None else float(root_zone)
        if neuralfoil_min_confidence is not None and polars != "neuralfoil":
            raise ValueError("BdfoilV1Backend: neuralfoil_min_confidence needs polars 'neuralfoil'")
        self.neuralfoil_min_confidence = None if neuralfoil_min_confidence is None \
            else float(neuralfoil_min_confidence)

    def _set_neuralfoil_confidence(self) -> None:
        """
        bdfoil_v1 cuts its NeuralFoil polars where NeuralFoil's analysis confidence is under
        neuralfoil_polars.MIN_CONFIDENCE (0.90, Antoine's decision J7-2 of 2026-10-21): on the
        C-foil sections at Re 5e6 the confidence falls to 0.4-0.5 at -4 deg, the polar starts at
        -1.5 or -0.5 deg, and the strips at a slightly negative incidence (yaw ~ 0, downwind) are
        clamped: 24 % of the NP-LLT x NeuralFoil runs of the smoke of 2026-10-27 failed. With
        neuralfoil_min_confidence the threshold of this PROCESS is set to that value (0 = no cut,
        the NeuralFoil level of the old bdFoil pipeline), and bdfoil_v1's in-memory NeuralFoil
        polars are dropped when it changes (their cache key does not hold the threshold).
        bdfoil_v1 itself is not modified (decision of the user, 2026-10-27; its effect on the
        results: pipelines/CHANGEMENTS_2026-10-27_bdfoil_v1.md).
        """
        if self.neuralfoil_min_confidence is None:
            return
        from bdfoil_v1 import neuralfoil_polars  # noqa: PLC0415  pylint: disable=import-outside-toplevel
        # pylint: disable=protected-access
        with neuralfoil_polars._LOCK:
            if neuralfoil_polars.MIN_CONFIDENCE != self.neuralfoil_min_confidence:
                neuralfoil_polars.MIN_CONFIDENCE = self.neuralfoil_min_confidence
                neuralfoil_polars._MEMO.clear()

    def coefficients(self, parameters: dict, workdir: Path) -> dict:
        """Planform design variables (template path) are not supported by the real backend."""
        del parameters, workdir
        raise SolverFailure("BdfoilV1Backend: planform design variables are not supported, use "
                            "a fixed planform with a section parametrization")

    def session(self, csv_path: Path, section_dir: Path):
        """The bdfoil_v1 Session of the planform CSV (sections in section_dir and next to the
        CSV); a StudyError (invalid case) stops the run: it is a set-up error."""
        csv_path = Path(csv_path).resolve()
        case = json.loads(json.dumps(self.case))
        case["appendage"]["planform"].update({"csv": str(csv_path),
                                              "section_dirs": [str(Path(section_dir).resolve())]})
        study = self.package.study.from_dict(case, base_dir=csv_path.parent)
        return self.package.session.Session(study)

    def evaluate_planform(self, csv_path: Path, section_dir: Path, with_strips: bool = False,
                          attitude: dict = None) -> dict:
        """
        Coefficients and diagnostics of the planform CSV (sections in section_dir).
        attitude: optional {cant, rake, yaw, sink} overriding the configured one (trim). The
        polars come from the polar cache after the first call with a section.
        """
        start = time.perf_counter()
        angles = {**self.attitude, **{k: float(v) for k, v in (attitude or {}).items()}}
        if self.cache_dir is not None:
            os.environ["BDFOIL_V1_CACHE"] = str(self.cache_dir.resolve())
        self._set_neuralfoil_confidence()
        session = self.session(csv_path, section_dir)
        try:
            result = session.evaluate(self.model.Attitude(**angles, speed=self.speed))
        except self.model.ExeError:
            raise
        except self.model.CoreError as error:
            raise SolverFailure(f"bdfoil_v1 {self.name} x {self.polars} failed: "
                                f"{type(error).__name__}: {error}") from error
        if not result.ok:
            raise SolverFailure(f"bdfoil_v1 {self.name} case failed: {result.error}")
        out = {key: float(value) for key, value in result.values.items()
               if isinstance(value, (int, float)) and not isinstance(value, bool)}
        out["Cdprofile [-]"] = out["Cdvisc [-]"]
        # the induced drag that enters Cx: AVL's Trefftz plane (force_integral "trefftz", the
        # default of bdfoil_v1), the near field of the NP-LLT (no Trefftz plane)
        trefftz = out.get("Cd trefftz [-]", np.nan)
        out["Cdi_nearfield [-]"] = out.get("Cdi [-]", np.nan)
        if self.name == "avl" and np.isfinite(trefftz):
            out["Cdi [-]"] = trefftz
        si = dict(result.si or {})
        for key in ("Re min [-]", "Re max [-]", "n Re clamped", "n cl clamped", "n filled",
                    "near-field gap [-]", "n iterations", "residual max [-]", "n alpha clamped"):
            value = si.get(key)
            if isinstance(value, (int, float, np.integer, np.floating)) \
                    and not isinstance(value, bool):
                out[key] = float(value)
        if self.name == "npllt":
            out["converged"] = bool(si.get("converged", True))
        # stall check outside the root zone of a wall image: strips whose cl leaves the monotone
        # branch of their polar (AVL and NP-LLT post), NP-LLT panels whose incidence leaves it
        strips = result.strips
        width = strips["width [m]"].to_numpy(dtype=float)
        s_mid = np.cumsum(width) - 0.5 * width
        checked = s_mid > self.root_zone * float(width.sum())
        clamped = strips["cl_clamped"].to_numpy(dtype=bool)
        n_clamped, n_total = int(np.sum(clamped & checked)), int(np.sum(clamped))
        if self.name == "npllt" and result.panels is not None and len(result.panels):
            panels = result.panels
            s = panels["s [m]"].to_numpy(dtype=float)
            frozen = panels["alpha_clamped"].to_numpy(dtype=bool)
            n_clamped += int(np.sum(frozen & (s > self.root_zone * float(s.max()))))
            n_total += int(np.sum(frozen))
        out.update({"n_clamped": n_clamped, "n_clamped_total": n_total,
                    "speed [m/s]": self.speed, "time_case_s": time.perf_counter() - start})
        if self.fail_on_clamp and n_clamped:
            raise SolverFailure(f"bdfoil_v1 {self.name}: {n_clamped} strips or panels outside "
                                "the monotone branch of their polar (stall), the clamped values "
                                "would be exploited")
        if with_strips:
            out["strips"] = {"s": s_mid.tolist(), "cl": strips["cl [-]"].tolist(),
                             "cd": strips["cd [-]"].tolist(), "y": strips["y [m]"].tolist(),
                             "z": strips["z [m]"].tolist(),
                             "chord": strips["chord [m]"].tolist(),
                             "Re": strips["Re [-]"].tolist(),
                             "cpmin": strips["Cpmin [-]"].tolist()}
            if self.name == "npllt" and result.panels is not None:
                # effective incidence of every lifting-line panel (e.g. the NeuralFoil Cpmin of
                # vpp_acceleration/src/foil_codes.py: bdfoil_v1 gives none with NeuralFoil)
                out["strips"]["alpha_deg"] = result.panels["alpha [deg]"].tolist()
        return out


def make_backend(level, problem, paths):
    """
    Backend instance of a fidelity level (config.Level) of a configuration: the flow shared by
    every level and, for the 3D levels, the attitude, the image, the kind and the chord of the
    planform (reference chord of the Reynolds number).
    """
    del paths
    options = dict(level.options)
    flow = problem.flow
    if level.solver == "neuralfoil":
        return NeuralFoilBackend(flow, **options)
    if level.solver == "xfoil":
        return XfoilBackend(flow, **options)
    from .planform import attitude_angles  # noqa: PLC0415
    planform = problem.planform or {}
    if problem.paths.get("polar_cache") and "cache_dir" not in options:
        options["cache_dir"] = problem.paths["polar_cache"]
    return BdfoilV1Backend(level.solver, flow, attitude=attitude_angles(problem.attitude or {}),
                           image=problem.image or "wall", kind=planform.get("kind", "FOIL"),
                           reference_chord=planform.get("chord"), **options)
