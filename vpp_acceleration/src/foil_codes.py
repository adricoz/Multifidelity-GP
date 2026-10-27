"""
The real codes of the C-foil torsor, through pipelines/bdtoolbox_foil on bdfoil_v1 (since
2026-10-27; the old bdFoil core / NonPlanarSolver version is the git tag pre-bdfoil-v1), for the
foil of the Intens SY VPP. Levels (LEVELS_XFOIL, the real runs since the third one):
  1 = NPLLT x NeuralFoil xxxlarge, 2 = NPLLT x XFOIL, 3 = AVL x XFOIL (validation only);
LEVELS (first two runs): 1 = NPLLT x NeuralFoil, 2 = AVL x XFOIL.

The foil is the one of the production RS of MDS (bdFoil/Scripts/IntensSY/CFoil_geom/
CFoil_generation.ipynb, read 2026-10-22): single circular arc CFoil_Arc_T1, radius 9.8 m, arc
3.25 m, chord 0.70 m, root tangent cant_geom = 20.168 deg, tip inboard; wall image (zsym +1),
hull wall at z = 0 and free surface 0.654 m above it (Antoine's case cfoil_r9800_fsi.json; it
only sets the depth of the strips, sigma, since the wave drag is off). Flow: Reynolds 5e6 on the
chord, n_crit 1, transition at 10 %, polars -12..12 deg by 0.5 deg (bdfoil_v1 defaults, as
Antoine's cases); bdfoil_v1 needs a speed: the fixed speed that gives Re 5e6 on the chord in
seawater at 15 deg C (8.49 m/s; decision of 2026-10-27, the surrogate has no speed input), local
Reynolds number of every strip, no wave drag (a Froude-dependent term would be wrong at a fixed
speed). bdFoil frame (pipelines/bdtoolbox_foil/planform.py): origin at the leading edge of the
LOWER BEARING, x forward, y to port, z up; the moments are reduced at this origin, as in the
production RS (torsor_ref = 0). Attitude: the inputs of the RS are passed as they are (the
notebook passes its CANTS = heel 0..8 deg, RAKES = trim, YAWS = leeway directly to bdFoil):
cant = Cant_Eff, rake = Rake_Eff, yaw = Yaw_Eff, sink = 0 (no Hvol input, decision of 2026-10-22).

Outputs (7, order of the MDS foil RS): Fx, Fy, Fz, Mx, My, Mz (MDS multiplies them by
0.5 rho S V^2; moments in metres) and Cpmin. The foil.py-format coefficients Cx..Cmz of bdfoil_v1
become the RS outputs with CONVERSION (sign and factor per component), fixed by the step-0
calibration against the production RS (scripts/real_step0_calibration.py). With bdfoil_v1 the
moments include the viscous part (foil.py: induced only).
Cpmin: CP_MIN_PERCENTILE (20, the production RS) percentile of the Cpmin of the strips:
  XFOIL polars: the Cpmin of bdfoil_v1 (XFOIL polar cpmin of every strip at its cl and Re);
  NeuralFoil polars: bdfoil_v1 gives none, so the Cpmin of NeuralFoil (1 - (ue/V)^2 on its
           boundary-layer stations) at the local incidence of every NP-LLT panel.
A failed run (SolverFailure, XFOIL or AVL error, stall) returns NaN outputs and the error.

Run in the MDS environment (Multifidelity-GP-Clean/.venv-mds: conda bdToolbox packages).
"""
import hashlib
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO_DIR = Path(__file__).resolve().parents[2]                  # .../Multifidelity-GP-Clean
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))
STUDY_CONFIG = REPO_DIR / "studies" / "cfoil_intens_kulfan_trim_2" / "config_cfoil_trim.json"
OUTPUT_NAMES = ["Fx", "Fy", "Fz", "Mx", "My", "Mz", "Cpmin"]
BDFOIL_NAMES = ["Cx [-]", "Cy [-]", "Cz [-]", "Cmx [-]", "Cmy [-]", "Cmz [-]"]
# production planform (CFoil_generation.ipynb, cell 4)
PLANFORM = {"type": "cfoil_arc", "radius": 9.8, "span": 3.25, "chord": 0.70, "taper": 1.0,
            "root_twist": 0.0, "tip_twist": 0.0, "cant_geom": 20.168, "tip_side": -1,
            "ori_y": 0.0, "ori_z": 0.0, "n_stations": 41, "kind": "FOIL"}
FLOW = {"re": 5e6, "n_crit": 1.0, "xtr": [0.1, 0.1], "alpha": [-12.0, 12.0, 0.5]}
# Cpmin of the foil = this percentile of the Cpmin of the strips, every strip counted: the value
# of the production RS (CFoil_generation.ipynb, CP_MIN_PERC = 20): a few singular strips (root
# next to the wall, tip) do not decide it (the plain minimum gave -6 against -0.69 for the RS,
# 2026-10-22); bdfoil_v1's default is 1
CP_MIN_PERCENTILE = 20.0
# shared by the levels (module docstring): hull wall at z = 0, free surface 0.654 m above it
CASE_OPTIONS = {"cp_min_percentile": CP_MIN_PERCENTILE, "wave_drag": False,
                "free_surface_height": 0.654, "temperature": 15.0}
# NeuralFoil polars without bdfoil_v1's confidence cut (0.90): with it, 24 % of the NP-LLT x
# NeuralFoil runs failed in the smoke of 2026-10-27 (strips at a slightly negative incidence, where
# NeuralFoil's confidence is 0.4-0.5), the old pipeline had no cut (user decision 2026-10-27; effect
# on the results: pipelines/CHANGEMENTS_2026-10-27_bdfoil_v1.md)
NEURALFOIL_MIN_CONFIDENCE = 0.0
LEVELS = [{"solver": "npllt", "cost": 1.0,
           "options": {"polars": "neuralfoil", "neuralfoil_min_confidence": NEURALFOIL_MIN_CONFIDENCE,
                       **CASE_OPTIONS}},
          {"solver": "avl", "cost": 15.0,
           "options": {"polars": "xfoil", "polar_timeout": 120, "timeout": 120,
                       **CASE_OPTIONS}}]
# third run (2026-10-22): high fidelity = NPLLT x XFOIL, the lifting line of level 1 with XFOIL
# polars (the trim recommendation of the C-foil study of 2026-10-15: its lift sees the viscous
# polar, AVL's only the camber line); AVL x XFOIL stays as level 3, for the validation only
NPLLT_XFOIL = {"solver": "npllt", "cost": 2.0,
               "options": {"polars": "xfoil", "polar_timeout": 120, **CASE_OPTIONS}}
LEVELS_XFOIL = [LEVELS[0], NPLLT_XFOIL, LEVELS[1]]
TE_THICKNESS = 0.00197
NEURALFOIL_MODEL = "xxxlarge"         # the NeuralFoil polars of bdfoil_v1 (its MODEL_SIZE)
N_SECTION_POINTS = 121                # coordinates per side of a Kulfan section (AVL limit)
# bdFoil coefficient -> RS output: output_k = sign_k * factor_k * bdFoil_k (set by step 0)
CONVERSION = {"sign": [1.0] * 6, "factor": [1.0] * 6}


def section_name(weights: np.ndarray) -> str:
    """Short unique name of a section (XFOIL cache and file names)."""
    return "k" + hashlib.sha1(np.round(np.asarray(weights, float), 10).tobytes()).hexdigest()[:12]


class FoilCodes:
    """
    The levels of the module docstring.

    Usage:
        codes = FoilCodes(work_dir)                 # work_dir holds the XFOIL polar cache
        y, info = codes.evaluate(weights, (cant, rake, yaw), level)
    """

    def __init__(self, work_dir, conversion: dict = None, levels=LEVELS, planform=PLANFORM,
                 flow=FLOW):
        # pylint: disable=import-outside-toplevel
        from pipelines.bdtoolbox_foil.config import load_config
        from pipelines.bdtoolbox_foil.simulator import BdToolboxFoilSimulator
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        config = json.loads(STUDY_CONFIG.read_text(encoding="utf-8"))
        config.update({"planform": dict(planform), "flow": dict(flow), "levels": list(levels),
                       "attitude": {"heel": 0.0, "trunk_cant": 0.0, "rake": 0.0, "yaw": 0.0,
                                    "sink": 0.0},
                       "paths": {**config.get("paths", {}),
                                 "polar_cache": str(self.work_dir / "polar_cache")}})
        config.pop("trim", None)
        # the optimizer block of the study is not used here, but must have one size per level
        config["optimization"] = {**config.get("optimization", {}),
                                  "doe": [10] * len(levels)}
        # one file per process: the worker processes of a pool build their FoilCodes at the same
        # time, a shared file would be read while another process writes it
        import os  # noqa: PLC0415
        path = self.work_dir / f"codes_config_{os.getpid()}.json"
        path.write_text(json.dumps(config, indent=1), encoding="utf-8")
        self.problem = load_config(str(path))
        self.problem.trim = None
        self.simulator = BdToolboxFoilSimulator(self.problem)
        self.conversion = conversion or CONVERSION
        self.costs = [float(level["cost"]) for level in levels]

    # --- geometry ---
    @staticmethod
    def section(weights: np.ndarray, te_thickness: float = TE_THICKNESS):
        """bdtoolbox_foil Section of 16 Kulfan weights (8 per side, LE weight 0)."""
        from pipelines.bdtoolbox_foil.geometry import Section, kulfan_coordinates  # noqa
        weights = np.asarray(weights, dtype=float)
        kulfan = {"upper_weights": weights[:8], "lower_weights": weights[8:],
                  "leading_edge_weight": 0.0, "TE_thickness": float(te_thickness)}
        from src.section_pca import THICKNESS_STATIONS, thickness_distribution  # noqa
        thickness = thickness_distribution(weights, THICKNESS_STATIONS, te_thickness).max()
        # 121 points per side, as the C-foil studies (parametrization n_points): with the 161 of
        # the default, AVL fails to read the section (no output file written, 2026-10-22)
        return Section(name=section_name(weights),
                       coordinates=kulfan_coordinates(kulfan, n_points=N_SECTION_POINTS),
                       parameters={}, thickness=float(thickness), kulfan=kulfan)

    # --- one run ---
    def evaluate(self, weights: np.ndarray, attitude, level: int, section=None) -> tuple:
        """
        Torsor coefficients of a section at an attitude (Cant_Eff, Rake_Eff, Yaw_Eff) [deg].

        Args:
        - weights: 16 Kulfan weights (ignored if `section` is given, e.g. a section read from a
          .xf file for the calibration).
        - level: rank of the level in `levels`, from 1 (LEVELS: 1 NPLLT x NeuralFoil,
          2 AVL x XFOIL; LEVELS_XFOIL: 1 NPLLT x NeuralFoil, 2 NPLLT x XFOIL, 3 AVL x XFOIL).
        Returns:
        - y: numpy array (7,), OUTPUT_NAMES (NaN if the run failed);
        - info: dict (time_s, error, raw bdfoil_v1 coefficients, diagnostics).
        """
        # pylint: disable=import-outside-toplevel
        from pipelines.bdtoolbox_foil.geometry import write_xf
        from pipelines.bdtoolbox_foil.planform import write_planform_csv
        from pipelines.bdtoolbox_foil.solvers import SolverFailure
        start = time.perf_counter()
        section = section or self.section(weights)
        cant, rake, yaw = (float(a) for a in attitude)
        info = {"level": level, "section": section.name, "error": ""}
        y = np.full(len(OUTPUT_NAMES), np.nan)
        backend = self.simulator.backends[level - 1]
        try:
            with tempfile.TemporaryDirectory(prefix="vppc_", ignore_cleanup_errors=True) as tmp:
                workdir = Path(tmp)
                xf_path = write_xf(section, workdir)
                csv_path = write_planform_csv(self.simulator.stations, xf_path.name,
                                              workdir / "planform.csv")
                out = backend.evaluate_planform(csv_path, workdir, with_strips=True,
                                                attitude={"cant": cant, "rake": rake,
                                                          "yaw": yaw, "sink": 0.0})
                strips = out.pop("strips")
                if backend.polars == "neuralfoil":
                    cpmin = self._cpmin_neuralfoil(section, np.asarray(strips["alpha_deg"]))
                else:
                    cpmin = float(out["Cpmin [-]"])
            raw = np.array([out[name] for name in BDFOIL_NAMES], dtype=float)
            y[:6] = np.asarray(self.conversion["sign"]) * np.asarray(self.conversion["factor"]) * raw
            y[6] = cpmin
            info.update({name: float(out[name]) for name in BDFOIL_NAMES})
            info.update({key: out.get(key, np.nan) for key in ("Cdi [-]", "Cdprofile [-]",
                                                               "n_clamped_total", "Re min [-]",
                                                               "Re max [-]")})
        except (SolverFailure, ValueError, FloatingPointError, KeyError,
                np.linalg.LinAlgError, *self.simulator._core_errors) as error:  # noqa: SLF001
            info["error"] = f"{type(error).__name__}: {error}"[:300]
        info["time_s"] = time.perf_counter() - start
        return y, info

    # --- Cpmin ---
    @staticmethod
    def _cpmin_neuralfoil(section, alphas: np.ndarray) -> float:
        """CP_MIN_PERCENTILE percentile over the NP-LLT panels of the NeuralFoil Cpmin at their
        local alpha (the Cpmin of a section = min over its boundary-layer stations), at the
        Reynolds number of the flow (the local Re of the strips stays within 0.02 % of it)."""
        import neuralfoil as nf  # noqa: PLC0415
        xtr = FLOW["xtr"]
        kwargs = {"alpha": np.asarray(alphas, dtype=float), "Re": FLOW["re"],
                  "n_crit": FLOW["n_crit"], "xtr_upper": xtr[0], "xtr_lower": xtr[1],
                  "model_size": NEURALFOIL_MODEL}
        if section.kulfan is not None:
            aero = nf.get_aero_from_kulfan_parameters(kulfan_parameters=section.kulfan, **kwargs)
        else:                                       # a section read from a .xf file
            aero = nf.get_aero_from_coordinates(coordinates=section.coordinates, **kwargs)
        velocity = np.column_stack([np.ravel(aero[f"{side}_bl_ue/vinf_{i}"])
                                    for side in ("upper", "lower")
                                    for i in range(len(nf.bl_x_points))])
        return float(np.percentile(np.min(1.0 - velocity ** 2, axis=1), CP_MIN_PERCENTILE))


def read_xf_section(path):
    """A section read from a .xf file (name line + x z coordinates), e.g. the production
    MC2_60_6101_1.xf, as a bdtoolbox_foil Section (no Kulfan parameters: NeuralFoil reads the
    coordinates)."""
    from pipelines.bdtoolbox_foil.geometry import Section  # noqa: PLC0415
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    coordinates = np.array([[float(v) for v in line.split()[:2]] for line in lines[1:]
                            if len(line.split()) >= 2])
    return Section(name=Path(path).stem.replace(".", "_"), coordinates=coordinates,
                   parameters={}, thickness=float(np.ptp(coordinates[:, 1])), kulfan=None)
