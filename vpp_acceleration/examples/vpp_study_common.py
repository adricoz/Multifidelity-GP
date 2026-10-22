"""
Shared pieces of the VPP-targeted studies (vpp_*_study.py): navigation conditions, design sections,
the "truth" (VPP of the Intens SY with the level-2 meta-surrogate as foil), and the error measures.

Truth (cached in results/vpp_truth.npz, computed once, about 4 min):
  for every design section s and navigation condition c: the equilibrium state x_sc (16 inputs) and
  the speed V_sc of the MDS VPP whose foil is the level-2 meta-surrogate ("AVL-like"), solved from
  the fixed initial guess AND from the state of the reference section (best converged speed kept,
  vpp_coupling.IntensSYVPP.solve), and the speed sensitivities dV/dC_k at that state (finite
  differences). The error measures use, for each condition, the MEDIAN of dV/dC_k over the
  sections: the sensitivities are a property of the boat and of the condition (they differ by
  ~10 % between sections), and a finite difference that jumps to another local equilibrium gives
  an outlier (x 20 to 400 seen once) that the median ignores.

Error measures of a surrogate (mean prediction mu_k of the coefficient k):
  * speed error to first order (no VPP solve, cheap, used everywhere):
        e_V = sqrt( mean_sc sum_k (dV/dC_k)_sc^2 (mu_k(x_sc) - C_k(x_sc))^2 )      [kt]
    i.e. the RMS over the true states of the speed change that the coefficient errors would cause;
  * true speed error (VPP solved with the surrogate as foil, used for the final checks):
        |V_sc(surrogate) - V_sc(truth)|                                            [kt]
  * relative RMSE of each coefficient over a global test set (whole domain).

Run with the MDS environment (Multifidelity-GP-Clean/.venv-mds).
"""
import os
import sys
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
VPP_DIR = HERE.parent
REPO_DIR = VPP_DIR.parent
for path in (VPP_DIR, REPO_DIR / "mfego", REPO_DIR / "benchmarks"):
    if str(path) not in sys.path:
        sys.path.append(str(path))
# pylint: disable=wrong-import-position
from src.meta_foil import N_SCORES, MetaFoil  # noqa: E402
from src.vpp_coupling import IntensSYVPP  # noqa: E402

# outputs of the meta-surrogate studies (2026-10-21): one run folder, runs/README.md
RUN_DIR = VPP_DIR / "runs" / "2026-10-21_meta_vpp_cible"
RESULTS = RUN_DIR / "results"
FIGURES = RUN_DIR / "figures"
TRUTH_FILE = RESULTS / "vpp_truth.npz"
# navigation conditions (TWA [deg], TWS at 10 m [kt]): all converge from the fixed initial guess
CONDITIONS = [(45.0, 12.0), (45.0, 16.0), (60.0, 16.0), (75.0, 16.0), (90.0, 14.0)]
N_DESIGN_SECTIONS = 4                 # the reference + 3 sections close to it
CAMBER_WINDOW = 2.5                   # |alpha_0 - alpha_0,ref| [deg] of the design sections
FORCE_NAMES = ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]


def design_sections(meta: MetaFoil, n_sections: int = N_DESIGN_SECTIONS, seed: int = 21):
    """The reference section, then sections of the family whose zero-lift angle is within
    CAMBER_WINDOW of the reference one (sections an optimizer would explore first)."""
    reference_alpha = meta._reference[2]["alpha_0"][0]       # pylint: disable=protected-access
    pool = meta.sample_sections(200, seed)
    alpha_0 = meta._polar_rows(pool, 2)["alpha_0"]           # pylint: disable=protected-access
    close = pool[np.abs(alpha_0 - reference_alpha) < CAMBER_WINDOW]
    return np.vstack([meta.reference_scores, close[:n_sections - 1]])


def truth(meta: MetaFoil = None, vpp: IntensSYVPP = None, verbose: bool = True) -> dict:
    """The truth (module docstring), from the cache or computed."""
    if TRUTH_FILE.exists():
        stored = np.load(TRUTH_FILE)
        return {key: stored[key] for key in stored.files}
    meta = meta or MetaFoil()
    vpp = vpp or IntensSYVPP()
    sections = design_sections(meta)
    predict = lambda x: meta(x, 2)                           # noqa: E731
    points, speeds, success, derivatives, unknowns = [], [], [], [], []
    for i, scores in enumerate(sections):
        vpp.set_foil(predict, scores)
        for j, (twa, tws) in enumerate(CONDITIONS):
            extra = [unknowns[j]] if i > 0 else []            # state of the reference section
            state = vpp.solve(twa, tws, extra_guesses=extra)
            points.append(np.concatenate([scores, [state["Cant_Eff"], state["Rake_Eff"],
                                                   state["Yaw_Eff"], state["Hvol"]]]))
            speeds.append(state["speed"])
            success.append(state["success"])
            unknowns.append(state["unknowns"])
            derivatives.append(vpp.speed_sensitivities(predict, scores, twa, tws, state))
            if verbose:
                print(f"truth section {i} TWA {twa:.0f} TWS {tws:.0f}: {state['speed']:.3f} kt "
                      f"(ok {state['success']}), dV/dC {np.round(derivatives[-1], 3)}", flush=True)
    raw = np.array(derivatives).reshape(len(sections), len(CONDITIONS), -1)
    robust = np.repeat(np.nanmedian(raw, axis=0)[None], len(sections), axis=0).reshape(-1, 6)
    data = {"sections": sections, "conditions": np.array(CONDITIONS), "points": np.array(points),
            "speeds": np.array(speeds), "success": np.array(success),
            "raw_derivatives": np.array(derivatives), "derivatives": robust,
            "unknowns": np.array(unknowns), "coefficients": meta(np.array(points), 2)}
    RESULTS.mkdir(parents=True, exist_ok=True)
    np.savez(TRUTH_FILE, **data)
    return data


def speed_error(surrogate, reference: dict) -> float:
    """e_V [kt]: first-order speed error at the true states (module docstring)."""
    ok = reference["success"]
    errors = surrogate.predict(reference["points"][ok])[:, :6] - reference["coefficients"][ok, :6]
    return float(np.sqrt(np.mean(np.sum((reference["derivatives"][ok] * errors) ** 2, axis=1))))


def speed_error_per_state(surrogate, reference: dict) -> np.ndarray:
    """First-order speed error (signed sum over the coefficients) at each true state [kt]."""
    errors = surrogate.predict(reference["points"])[:, :6] - reference["coefficients"][:, :6]
    return np.sum(reference["derivatives"] * errors, axis=1)


def speed_sigma_per_state(surrogate, reference: dict) -> np.ndarray:
    """Standard deviation of the speed predicted at each true state, to first order [kt]."""
    variances = surrogate.variances(reference["points"])
    return np.sqrt(np.sum(reference["derivatives"] ** 2 * variances, axis=1))


def true_speed_errors(surrogate, reference: dict, vpp: IntensSYVPP) -> np.ndarray:
    """|V(surrogate) - V(truth)| at every (section, condition) [kt], VPP solved with the surrogate
    (started from the true state)."""
    errors = []
    n_conditions = len(reference["conditions"])
    for index, point in enumerate(reference["points"]):
        scores = point[:N_SCORES]
        twa, tws = reference["conditions"][index % n_conditions]
        vpp.set_foil(surrogate.predict, scores)
        state = vpp.solve(twa, tws, guess=reference["unknowns"][index],
                          extra_guesses=[vpp.initial_guess])
        errors.append(abs(state["speed"] - reference["speeds"][index]) if state["success"]
                      else np.nan)
    return np.array(errors)


def global_test_set(meta: MetaFoil, n_points: int = 1000, seed: int = 99):
    """Points of the whole domain and their level-2 coefficients."""
    from src.vpp_targeted import global_points  # noqa: PLC0415
    x = global_points(meta, n_points, seed)
    return x, meta(x, 2)


def global_errors(surrogate, test) -> np.ndarray:
    """Relative RMSE (RMSE / std) of each of the 6 coefficients on the global test set."""
    x, y = test
    errors = surrogate.predict(x)[:, :6] - y[:, :6]
    return np.sqrt(np.mean(errors ** 2, axis=0)) / np.std(y[:, :6], axis=0)


if __name__ == "__main__":
    result = truth()
    print("speeds", np.round(result["speeds"], 3), "all ok", bool(result["success"].all()))
