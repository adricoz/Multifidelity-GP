"""
Coupling of a foil surrogate with the MDS VPP of the Intens SY (MDS branch new_master).

What MDS does with the foil (Systems/Ships/Sail/IntensSY/intenssy_subsystems.py, IntensSY_foil_dyn,
read 2026-10-21):
  * Cant_Eff = Heel + Foil_Cant, Rake_Eff = Trim + Foil_Rake, Yaw_Eff = Leeway + Foil_Yaw;
  * the input vector of the RS is assembled BY NAME (foil_rs_vector) from its input_names;
  * surrogate.regression(x) fills surrogate.prediction (Fx, Fy, Fz, Mx, My, Mz, Cpmin);
  * the 6 first outputs are multiplied by qS = 0.5 rho_water S V^2 (S = 2.275 m2 for the C-foil),
    Cpmin is read by name (surrogate.out('Cpmin')).
So any object with the same few attributes as MDS' VppChebyRS can replace the RS of the foil:
SurrogateFoilRS below. It adds what the RS does not see: the section (12 PCA scores, fixed
for a VPP run) and the boat sink Hvol, read in the global variables of the system (the hulls,
evaluated before the foil, update it).

The VPP solve (IntensSYVPP.solve) is the one of Systems/Ships/Sail/IntensSY/
IntensSY_flying_NCpre.py: SLSQP maximizes the boat speed subject to the 6 equilibrium equations,
unknowns Speed, Leeway, Heel, Trim, CC_Dspl, Rudder_Angle, Flat (foil controls at 0, as the vpp36
navigation cases). Differences: the C-foil variant (build_IntensSY(foil='CFoil')), a fixed initial
guess for every condition (no path dependence) and one retry from the previous solution.

Environment: MDS needs pymoo, numba, ... : run with Multifidelity-GP-Clean/.venv-mds (venv of
the conda env bdToolbox + future, transforms3d, PyWavelets, pydmd; created 2026-10-21 so as not
to touch the shared environments).
"""
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

MDS_DIR = Path(__file__).resolve().parents[3] / "MDS"          # .../Code-Adri/MDS
# production FSI RS of the C-foil of the boat (bdFoil CFoil_fsi.ipynb: AVL x XFOIL and the beam
# model "supported", 625 points, Speed_kts in [7, 28]; the current section)
FSI_RS_FILE = (MDS_DIR / "Systems" / "Ships" / "Sail" / "IntensSY" / "Data" / "Foil"
               / "CFoil_R9800_S3250_C700_fsi_rs_vpp.json")
FSI_SPEEDS = np.array([7.0, 10.5, 14.0, 17.5, 21.0, 24.5, 28.0])
FOIL_INPUTS = ["FOIL_PRT_Cant_Eff", "FOIL_PRT_Rake_Eff", "FOIL_PRT_Yaw_Eff"]
FOIL_OUTPUTS = ["Fx", "Fy", "Fz", "Mx", "My", "Mz", "Cpmin"]
SOLVE_VARS = ["Speed [kts]", "Leeway [°]", "Heel [°]", "Trim [°]", "CC_Dspl [kg]",
              "Rudder_Angle [°]", "Flat [-]"]
EQUATIONS = ["Eq.Fx", "Eq.Fy", "Eq.Fz", "Eq.Mx", "Eq.My", "Eq.Mz"]
EQ_SCALE = np.array([1e3, 1e3, 1e3, 1e4, 1e4, 1e4])     # forces (N) ~ moments (N.m)
# perturbation of each foil coefficient for the speed sensitivities (about 3 % of its size)
SENSITIVITY_STEPS = np.array([5e-4, 1e-2, 1e-2, 2e-2, 3e-3, 3e-3])


class _RSData:
    """The attributes of an MDS Data object that the sub-system base classes read."""

    def __init__(self, input_range, n_out):
        input_range = np.asarray(input_range, dtype=float)
        self.x_dim, self.y_dim = input_range.shape[0], n_out
        self.x_bounds = np.vstack([input_range[:, 0], input_range[:, 1]])
        self.y_bounds = np.vstack([np.full(n_out, -np.inf), np.full(n_out, np.inf)])


class FSICorrection:
    """
    Effect of the deformation of the flexible C-foil on its coefficients, taken from the
    production FSI RS (FSI_RS_FILE), to be added to a RIGID foil model (the codes and the
    co-kriging are rigid).

    The deformation follows the load, i.e. V^2 (linear structure), and so does its effect: at a
    fixed attitude, C_k(V) = C_k,rigid + b_k V^2 (checked on the RS on 2026-10-22: the change from
    7 kt divided by V^2 - 49 is constant to 3 % up to 24 kt, Fy +4 % at 18 kt). b_k(attitude) is
    the least-squares slope of the RS outputs against V^2 over FSI_SPEEDS, and
        delta_k(attitude, V) = stiffness_ratio * b_k(attitude) * V^2,
    stiffness_ratio = Ixx(current section) / Ixx(section): a stiffer section deforms less
    (bending ~ 1 / EI, twist ~ 1 / GJ, both taken proportional to Ixx for sections of one family,
    same chord and structure).
    Assumptions: at the same attitude the loads of the section are those of the current one (a
    few % apart), the structure is the one of the production campaign. Bending [m] and Twisting
    [deg] follow the same law (rigid value 0).
    """

    def __init__(self, path=FSI_RS_FILE, speeds=FSI_SPEEDS):
        from src.meta_foil import ChebyshevRS  # noqa: PLC0415
        self.rs = ChebyshevRS(path)
        self.output_names = list(self.rs.output_names)       # Fx..Mz, Cpmin, Bending, Twisting
        self.speed_name = self.rs.input_names[3]              # "Speed_kts"
        self.speeds = np.asarray(speeds, dtype=float)
        self.lo, self.hi = self.rs.input_range[:3, 0], self.rs.input_range[:3, 1]

    def slopes(self, attitude) -> np.ndarray:
        """b (m, 9): slope of every RS output against V^2 [per kt^2], attitude (m, 3) clipped
        to the box of the RS."""
        attitude = np.clip(np.atleast_2d(np.asarray(attitude, dtype=float)), self.lo, self.hi)
        m, n = len(attitude), len(self.speeds)
        x = np.column_stack([np.repeat(attitude, n, axis=0), np.tile(self.speeds, m)])
        out = self.rs(x).reshape(m, n, -1)
        v2 = self.speeds ** 2 - np.mean(self.speeds ** 2)
        return np.einsum("j,mjk->mk", v2, out - out.mean(axis=1, keepdims=True)) / np.sum(v2 ** 2)

    def __call__(self, attitude, speed, stiffness_ratio=1.0) -> np.ndarray:
        """delta (m, 9) at attitude (m, 3) [deg] and speed (m,) [kt]."""
        slopes = self.slopes(attitude)
        speed = np.broadcast_to(np.asarray(speed, dtype=float), (len(slopes),))
        ratio = np.broadcast_to(np.asarray(stiffness_ratio, dtype=float), (len(slopes),))
        return ratio[:, None] * slopes * speed[:, None] ** 2


class SurrogateFoilRS:
    """
    Stand-in for MDS' VppChebyRS on the "Foil Dyn" component, backed by any foil model.

    Args:
    - predict: function x (m, n) -> coefficients (m, 7) in the order FOIL_OUTPUTS, x = (section
      scores, Cant_Eff, Rake_Eff, Yaw_Eff[, Hvol]) (for example the means of the co-kriging, or
      the meta-surrogate).
    - scores: the scores of the section of this VPP run.
    - global_variables: the dict of the MDS system (Hvol is read there at each evaluation).
    - input_range: box (3, 2) of Cant_Eff, Rake_Eff, Yaw_Eff (informative: no clamping here).
    - with_hvol: True (meta-surrogate) appends Hvol to the inputs; False (real codes of
      2026-10-22: no Hvol input) does not.
    - extra_outputs: outputs the MDS component reads but the model does not give (e.g. Bending
      and Twisting of the CFoilFSI variant), returned as 0 (rigid foil), or by `fsi`.
    - fsi: None (rigid foil) or an FSICorrection: the RS then declares the input Speed_kts (MDS
      passes the boat speed, foil_rs_vector) and its correction is added to the model, scaled by
      stiffness_ratio (Ixx current / Ixx of the section).
    """

    def __init__(self, predict, scores, global_variables, input_range, with_hvol=True,
                 extra_outputs=(), fsi=None, stiffness_ratio=1.0):
        self.predict = predict
        self.scores = np.asarray(scores, dtype=float).reshape(-1)
        self.global_variables = global_variables
        self.fsi, self.stiffness_ratio = fsi, float(stiffness_ratio)
        self.input_names = list(FOIL_INPUTS) + ([fsi.speed_name] if fsi is not None else [])
        self.output_names = list(FOIL_OUTPUTS) + list(extra_outputs)
        self.with_hvol = bool(with_hvol)
        self.input_range = np.asarray(input_range, dtype=float)
        self.prediction = np.zeros(len(self.output_names))
        self.variance = None                       # deterministic VPP (the mean of the surrogate)
        self.data = _RSData(self.input_range, len(self.output_names))
        self.subsystem_name = ""
        self.x_extrapolation_flag = False
        self.y_extrapolation_flag = False
        self.last_point = None                     # the 16 inputs of the last evaluation
        self.n_calls = 0

    def eval(self, x, **kwargs):
        """x = (Cant_Eff, Rake_Eff, Yaw_Eff[, Speed_kts]) given by MDS; returns the outputs."""
        del kwargs
        x = np.asarray(x, dtype=float).reshape(-1)
        parts = [self.scores, x[:3]]
        if self.with_hvol:
            parts.append([float(self.global_variables["Hvol [m]"])])
        point = np.concatenate(parts)
        self.last_point = point
        values = np.asarray(self.predict(point[None, :]), dtype=float)[0][:len(FOIL_OUTPUTS)]
        extra = np.zeros(len(self.output_names) - len(FOIL_OUTPUTS))
        if self.fsi is not None:
            delta = self.fsi(x[None, :3], x[3], self.stiffness_ratio)[0]
            values = values + delta[:len(FOIL_OUTPUTS)]
            extra = np.array([delta[self.fsi.output_names.index(name)]
                              for name in self.output_names[len(FOIL_OUTPUTS):]])
        self.prediction = np.concatenate([values, extra])
        self.n_calls += 1
        return self.prediction

    def regression(self, x):
        return self.eval(x)

    def out(self, name):
        return float(self.prediction[self.output_names.index(name)])

    def reset_extrapolation_flags(self):
        self.x_extrapolation_flag = self.y_extrapolation_flag = False

    def plot_point(self, *args, **kwargs):
        pass


class IntensSYVPP:
    """
    The Intens SY of MDS with its C-foil, whose "Foil Dyn" RS can be replaced by any foil model.

    Usage:
        vpp = IntensSYVPP()
        vpp.set_foil(predict, scores)          # predict: x (m, 16) -> (m, 7)
        state = vpp.solve(twa=45.0, tws=16.0)  # dict: speed, attitude, foil inputs, success...
    """

    def __init__(self, mds_dir=MDS_DIR, foil="CFoil", with_hvol=True, fsi=None):
        """fsi: None (rigid foil model) or an FSICorrection added to the foil model (FSI run of
        2026-10-22)."""
        self.fsi = fsi
        if str(mds_dir) not in sys.path:
            sys.path.insert(0, str(mds_dir))
        # pylint: disable=import-outside-toplevel,import-error
        from Code.Math.frame import Frame
        from Systems.Ships.Sail.IntensSY.IntensSY_config import build_IntensSY
        # foil variant of the boat: "CFoil" (R 7.8 m, meta-surrogate) or "CFoilFSI" (R 9.8 m, the
        # foil of the real codes; its dyn RS is replaced by a rigid surrogate)
        self.system, self.origin = build_IntensSY(foil=foil)
        self.with_hvol = bool(with_hvol)
        gv = self.system.global_variables
        self.ref = Frame(np.array([[gv["XCG [m]"], 0.0], [0.0, 0.0], [0.0, 0.0]]),
                         self.origin, "CG ref")
        for model in self.system.models:
            if model.name == "Rudder":          # as IntensSY_flying_NCpre.py: gentle extrapolation
                model.surrogate_model.clamp = False
        self.foil = next(m for m in self.system.models if m.name == "Foil Dyn")
        self.original_rs = self.foil.surrogate_model
        dspl_lb, dspl_ub = self.system.build_profile["dspl_box"]
        self.lower = np.array([6.0, -6.0, 0.0, -1.5, dspl_lb, -12.0,
                               self.system.build_profile["flat_lb"]])
        self.upper = np.array([24.0, 6.0, 8.0, 1.5, dspl_ub, 12.0, 1.0])
        self.initial_guess = np.array([gv[v] for v in SOLVE_VARS])

    # --- the foil model ---
    def set_foil(self, predict, scores, stiffness_ratio=1.0) -> SurrogateFoilRS:
        """Puts a foil model (predict, section scores) in place of the C-foil RS; with FSI, the
        deformation correction is scaled by stiffness_ratio (Ixx current / Ixx of the section)."""
        extra = [name for name in getattr(self.original_rs, "output_names", [])
                 if name not in FOIL_OUTPUTS]
        rs = SurrogateFoilRS(predict, scores, self.system.global_variables,
                             self.original_rs.input_range[:3], with_hvol=self.with_hvol,
                             extra_outputs=extra, fsi=self.fsi, stiffness_ratio=stiffness_ratio)
        self.foil.surrogate_model = rs
        return rs

    def reset_foil(self) -> None:
        """Back to the production C-foil RS."""
        self.foil.surrogate_model = self.original_rs

    # --- the solve ---
    def _scaled_residuals(self, x):
        gv = self.system.global_variables
        for name, value in zip(SOLVE_VARS, x):
            gv[name] = value
        self.system.t_eval(gv, self.ref)
        return np.array([gv[k] for k in EQUATIONS]) / EQ_SCALE

    def _slsqp(self, guess):
        return minimize(lambda x: -x[0], guess, method="SLSQP",
                        bounds=list(zip(self.lower, self.upper)),
                        constraints={"type": "eq", "fun": self._scaled_residuals},
                        options={"ftol": 1e-9, "maxiter": 400})

    def solve(self, twa: float, tws: float, guess=None, extra_guesses=()) -> dict:
        """
        Maximum boat speed at (TWA, TWS) subject to the 6 equilibrium equations.

        Args:
        - twa [deg], tws [kt] (TWS at 10 m, MDS convention since August 2026).
        - guess: initial unknowns (SOLVE_VARS order); default the fixed initial guess. If the first
          solve fails, it is retried once from the fixed initial guess (or from guess).
        - extra_guesses: other starting points; the solve is redone from each of them and the
          converged state with the HIGHEST speed is kept. SLSQP can stop, "successfully", on a
          slower local equilibrium, and the MDS state (e.g. the sink closure of the hull) carries
          over from one solve to the next: from 2026-10-21, section 2 at TWA 90 / TWS 14 gave
          14.83 kt once and 17.58 kt from the same starting point after other solves.
        Returns:
        - dict: success, speed [kt], the unknowns, Hvol, the foil inputs Cant_Eff, Rake_Eff,
          Yaw_Eff, the scaled residual norm, the solve time and the number of model evaluations.
        """
        gv = self.system.global_variables
        gv["TWA [°]"], gv["TWS [kts]"] = float(twa), float(tws)
        start = time.perf_counter()
        first = self.initial_guess if guess is None else np.asarray(guess, dtype=float)
        result = self._slsqp(first)
        if not result.success:
            other = self.initial_guess if guess is not None else np.clip(
                first + np.array([2.0, 0.0, -2.0, 0.0, 0.0, 0.0, 0.0]), self.lower, self.upper)
            retry = self._slsqp(other)
            if retry.success or retry.fun < result.fun:
                result = retry
        for other_guess in extra_guesses:
            other = self._slsqp(np.asarray(other_guess, dtype=float))
            if other.success and (not result.success or other.x[0] > result.x[0]):
                result = other
        residual = float(np.linalg.norm(self._scaled_residuals(result.x)))   # leaves the state
        state = {"success": bool(result.success) and residual < 1e-4, "twa": float(twa),
                 "tws": float(tws), "residual": residual, "time_s": time.perf_counter() - start,
                 "unknowns": result.x.copy()}
        for name in SOLVE_VARS:
            state[name.split(" [")[0]] = float(gv[name])
        state["speed"] = state["Speed"]
        state["Hvol"] = float(gv["Hvol [m]"])
        state["Cant_Eff"], state["Rake_Eff"], state["Yaw_Eff"] = (
            float(self.foil.cant_eff), float(self.foil.rake_eff), float(self.foil.yaw_eff))
        return state

    # --- speed sensitivities ---
    def speed_sensitivities(self, predict, scores, twa, tws, base_state,
                            steps=SENSITIVITY_STEPS, stiffness_ratio=1.0, outputs=None) -> np.ndarray:
        """
        dV/dC_k [kt per unit coefficient] for the 6 force / moment coefficients of the foil, by
        forward finite differences: the model plus steps[k] on output k, re-solved from the base
        state. Leaves the foil model set to `predict`. outputs: the indices k to compute (default
        all; the others are NaN), so that a pool of processes can share the work.
        """
        derivatives = np.full(len(steps), np.nan)
        for k in (range(len(steps)) if outputs is None else outputs):
            shift = np.zeros(len(FOIL_OUTPUTS))
            shift[k] = steps[k]

            def shifted(x, shift=shift):
                return predict(x) + shift
            self.set_foil(shifted, scores, stiffness_ratio)
            state = self.solve(twa, tws, guess=base_state["unknowns"])
            derivatives[k] = (state["speed"] - base_state["speed"]) / steps[k] \
                if state["success"] else np.nan
        self.set_foil(predict, scores, stiffness_ratio)
        return derivatives
