"""
Theoretical validation of the optimum of a FINISHED run, without any new code or VPP call
(doc/validation_theorique_optimum.tex). Everything is computed from what the run saved: the
final surrogate (results/surrogate.pkl), the VPP states of the best section and the codes at
these states (results/validation.csv), the speed sensitivities dV/dC (run.log).

The objective is linearized at the best section, as the screening of the run does
(VPPSectionOptimizer.first_order_scores): attitudes a_c and sensitivities d_ck frozen at the
equilibria of the best section,
    J_lin(z) = sum_c w_c sum_k d_ck m_k(z, a_c) / sum_c w_c        [kt, up to a constant]
with m_k the mean of co-kriging k at a level (default the top one). By the envelope theorem the
GRADIENT of J_lin at the best section is the gradient of the true surrogate objective; away from
it J_lin is a first-order model (no VPP solve). J_lin is linear in the outputs, so its posterior
is Gaussian and the joint covariance between sections is exact (objective_covariance).

Experiments (each writes results/<name>.json):
  calibration  section 3: leave-one-out per output and level; surrogate against the codes at the
               VPP states of the validated sections (points never seen), factor kappa;
  gradient     section 2.2-2.3: gradient and its covariance, active constraints, multipliers
               (price of Ixx / v), KKT test, gain of a constrained ascent on the model, Hessian;
  kkt          the KKT part of `gradient` again from its saved gradient (no model evaluation);
  slices       1D cuts of J_lin through the best section along each PCA mode, with 2 sigma;
  regret       section 2.1: joint law of candidate sections, P(a better one exists), regret;
  dimension    section 4: active subspaces (spectrum of E[grad grad^T]) and Hessian spectrum;
  levels       section 5.2: gradient of the level difference, shift of the optimum between levels.

Run (MDS environment, 2 threads, a few minutes, shared machine):
    ..\\.venv-mds\\Scripts\\python validation_experiments\\experiments.py --run runs/<run>
"""
import argparse
import json
import os
import pickle
import re
import sys
import time
from pathlib import Path

for _variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_variable, "2")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import minimize, nnls  # noqa: E402
from scipy.stats import chi2  # noqa: E402

VPP_DIR = Path(__file__).resolve().parents[1]
for _path in (VPP_DIR, VPP_DIR.parent / "mfego"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
# pylint: disable=wrong-import-position
from src.section_domain import load_search_space  # noqa: E402
from src.section_pca import section_stiffness  # noqa: E402
from src.sequential_design import to_unit, whitened_data_covariance  # noqa: E402
from src.vpp_targeted import latent_mean  # noqa: E402

OUTPUT_NAMES = ["Fx", "Fy", "Fz", "Mx", "My", "Mz", "Cpmin"]
N_FORCES = 6                      # the outputs the speed depends on
HERE = Path(__file__).resolve().parent
RESULTS = {"dir": HERE / "results"}      # --out: one results folder per analysed run


# -------------------------------------------------------------------------------------------------
# The finished run
# -------------------------------------------------------------------------------------------------
def read_sensitivities(log_file: Path, n_conditions: int) -> np.ndarray:
    """dV/dC (n_conditions, 6) of the last 'speed sensitivities' block of run.log."""
    text = log_file.read_text(encoding="utf-8", errors="replace")
    start = text.rindex("speed sensitivities dV/dC (rows = conditions):")
    block = text[text.index("[[", start):text.index("]]", start)]
    numbers = [float(v) for v in re.findall(r"-?\d+\.?\d*(?:e[-+]?\d+)?", block)]
    return np.array(numbers).reshape(n_conditions, N_FORCES)


class FinishedRun:
    """Surrogate, search space, sections, VPP states and sensitivities of a finished run."""

    def __init__(self, root: Path):
        self.root = root
        config = json.loads((root / "config.json").read_text(encoding="utf-8"))
        self.config = config
        self.k = int(config["k"])
        self.family, self.pca, start_scores = load_search_space(
            VPP_DIR / config["search_box"], self.k, config["search_box_name"])
        self.domain = self.family.domain
        with open(root / "results" / "surrogate.pkl", "rb") as file:
            self.surrogate = pickle.load(file)
        self.surrogate.fit(refit=False)             # conditions again: the variances come back
        self.models = self.surrogate.models
        self.n_levels = self.surrogate.n_levels
        best = json.loads((root / "results" / "best_section.json").read_text(encoding="utf-8"))
        self.best = np.array(best["scores"])
        self.sections = {"best": self.best, best["start"]: np.array(best["start_scores"])}
        optima = root / "results" / "optima.json"         # written when the run restarted
        for entry in json.loads(optima.read_text(encoding="utf-8")) if optima.exists() else []:
            self.sections[f"region_{entry['region']}"] = np.array(entry["scores"])
        self.validation = pd.read_csv(root / "results" / "validation.csv")
        rows = self.validation[self.validation["section"] == "best"].reset_index(drop=True)
        self.condition_names = rows["condition"].tolist()
        self.attitudes = rows[["att0", "att1", "att2"]].to_numpy()
        weights = np.array(config["condition_weights"], dtype=float)
        self.derivatives = read_sensitivities(root / "run.log", len(weights))
        solved = rows["success"].to_numpy(dtype=bool) & np.all(np.isfinite(self.derivatives), 1)
        self.solved = solved
        # coef[c, k] = w_c d_ck / sum of w over the solved conditions (first_order_scores)
        self.coef = np.where(solved[:, None], weights[:, None] * self.derivatives, 0.0) \
            / weights[solved].sum()
        self.n_conditions = len(weights)

    # --- linearized objective ---
    def state_inputs(self, sections: np.ndarray) -> np.ndarray:
        """(n, k) sections -> (n * C, k + 3) inputs of the surrogate, section by section."""
        sections = np.atleast_2d(sections)
        return np.hstack([np.repeat(sections, self.n_conditions, axis=0),
                          np.tile(self.attitudes, (len(sections), 1))])

    def objective(self, sections: np.ndarray, level: int = None, chunk: int = 200) -> np.ndarray:
        """J_lin (n,) [kt, up to a constant] of sections (n, k), means of `level`."""
        sections = np.atleast_2d(sections)
        total = np.zeros(len(sections))
        for start in range(0, len(sections), chunk):
            part = sections[start:start + chunk]
            x = self.state_inputs(part)
            for k in range(N_FORCES):
                mean = latent_mean(self.models[k], x, level).reshape(len(part), -1)
                total[start:start + chunk] += mean @ self.coef[:, k]
        return total

    def objective_covariance(self, sections: np.ndarray, level: int = None,
                             chunk: int = 40) -> np.ndarray:
        """Posterior covariance (n, n) [kt^2] of J_lin between sections: sum over the outputs k
        (independent co-krigings) of b_k^T K_k b_k, K_k the joint posterior covariance of output
        k at all the states of all the sections (correlations between sections AND conditions)."""
        sections = np.atleast_2d(sections)
        n, n_c = len(sections), self.n_conditions
        x = self.state_inputs(sections)
        out = np.zeros((n, n))
        for k in range(N_FORCES):
            model, b = self.models[k], self.coef[:, k]
            levels = np.full(len(x), model.num_levels if level is None else level)
            w = whitened_data_covariance(model, x, levels)                # (n_total, n * C)
            wb = w.reshape(len(w), n, n_c) @ b                            # (n_total, n)
            out -= wb.T @ wb
            u = to_unit(model, x)
            for start in range(0, n, chunk):
                rows = slice(start * n_c, min(start + chunk, n) * n_c)
                # pylint: disable=protected-access
                prior = model._covariance(u[rows], levels[rows], u, levels)
                m = prior.shape[0] // n_c
                out[start:start + m] += np.einsum("c,icjd,d->ij", b,
                                                  prior.reshape(m, n_c, n, n_c), b)
        return 0.5 * (out + out.T)

    def gradient(self, center: np.ndarray, level: int = None, step: float = 0.02,
                 with_covariance: bool = False):
        """Central-difference gradient (k,) of J_lin at `center` [kt per unit score]; with its
        posterior covariance (k, k): the differences are linear in the GP, D S D^T is exact for
        the finite-difference gradient."""
        stencil = np.vstack([center + step * np.eye(self.k), center - step * np.eye(self.k)])
        operator = np.hstack([np.eye(self.k), -np.eye(self.k)]) / (2.0 * step)
        gradient = operator @ self.objective(stencil, level)
        if not with_covariance:
            return gradient
        return gradient, operator @ self.objective_covariance(stencil, level) @ operator.T

    def hessian(self, center: np.ndarray, level: int = None, step: float = 0.1) -> np.ndarray:
        """Central-difference Hessian (k, k) of J_lin at `center` (frozen states: approximate)."""
        eye = step * np.eye(self.k)
        points = [center]
        for i in range(self.k):
            for j in range(i, self.k):
                points += [center + eye[i] + eye[j], center + eye[i] - eye[j],
                           center - eye[i] + eye[j], center - eye[i] - eye[j]]
        values = self.objective(np.array(points), level)
        out, index = np.zeros((self.k, self.k)), 1
        for i in range(self.k):
            for j in range(i, self.k):
                pp, pm, mp, mm = values[index:index + 4]
                out[i, j] = out[j, i] = (pp - pm - mp + mm) / (4.0 * step ** 2)
                index += 4
        return out

    # --- constraints of the search space, as functions of the scores (>= 0 inside) ---
    def constraint_names(self) -> list:
        return list(self.domain.row_names) + ["ixx_v"] \
            + [f"box_low_{j + 1}" for j in range(self.k)] \
            + [f"box_high_{j + 1}" for j in range(self.k)]

    def constraints(self, scores: np.ndarray) -> np.ndarray:
        """Margins (n_constraints,) at one section: the linear constraints of the domain on the
        decoded weights (Kulfan-weight units, rows of unit norm), Ixx / v relative to its limit
        (ixx_v / ixx_v_min - 1), the box of the scores."""
        weights = self.pca.decode(np.atleast_2d(scores))
        ixx_v = section_stiffness(weights, self.domain.te_thickness)[1][0]
        return np.concatenate([self.domain.margins(weights)[0],
                               [ixx_v / self.domain.ixx_v_min - 1.0],
                               scores - self.family.lower, self.family.upper - scores])

    def constraint_jacobian(self, scores: np.ndarray, step: float = 1e-5) -> np.ndarray:
        return np.column_stack([(self.constraints(scores + step * e)
                                 - self.constraints(scores - step * e)) / (2.0 * step)
                                for e in np.eye(self.k)])

    def admissible(self, scores: np.ndarray) -> np.ndarray:
        scores = np.atleast_2d(scores)
        return self.family.contains(self.pca.decode(scores))

    def local_sections(self, center, radius: float, n: int, seed: int) -> np.ndarray:
        """Admissible sections around center, as RealFoilProblem.local_sections."""
        rng, kept = np.random.default_rng(seed), []
        for _ in range(200):
            scores = center + radius * rng.standard_normal((2 * n, self.k)) / np.sqrt(self.k)
            kept.append(scores[self.admissible(scores)])
            if sum(len(part) for part in kept) >= n:
                break
        return np.vstack(kept)[:n]

    def global_sections(self, n: int, seed: int) -> np.ndarray:
        """Admissible sections anywhere in the search space (RealFoilProblem.sample_sections)."""
        scores = self.pca.encode(self.family.sample(2 * n, seed))
        return scores[self.admissible(scores)][:n]

    def ascent(self, start: np.ndarray, radius: float, level: int = None) -> np.ndarray:
        """Constrained ascent (SLSQP) of J_lin on the model from `start`, inside the search
        space and the ball of `radius` around the best section (validity of the frozen states)."""
        result = minimize(
            lambda z: -self.objective(z, level)[0], start, method="SLSQP",
            jac=lambda z: -self.gradient(z, level, step=1e-3),
            constraints=[{"type": "ineq", "fun": self.constraints,
                          "jac": self.constraint_jacobian},
                         {"type": "ineq", "fun": lambda z: radius ** 2 - np.sum((z - self.best) ** 2),
                          "jac": lambda z: -2.0 * (z - self.best)}],
            options={"maxiter": 40, "ftol": 1e-7})
        return result.x


def scores_stats(z: np.ndarray) -> dict:
    z = z[np.isfinite(z)]
    return {"n": int(len(z)), "z_rms": float(np.sqrt(np.mean(z ** 2))) if len(z) else None,
            "z_mean": float(np.mean(z)) if len(z) else None,
            "coverage_95": float(np.mean(np.abs(z) <= 1.96)) if len(z) else None}


def save(name: str, data: dict) -> None:
    RESULTS["dir"].mkdir(parents=True, exist_ok=True)
    (RESULTS["dir"] / f"{name}.json").write_text(json.dumps(data, indent=1), encoding="utf-8")
    print(f"  -> {RESULTS['dir'].name}/{name}.json", flush=True)


# -------------------------------------------------------------------------------------------------
# 1. Calibration (section 3 of the note)
# -------------------------------------------------------------------------------------------------
def calibration(run: FinishedRun) -> None:
    out = {"loo": [], "external": [], "levels": run.config["levels"], "sizes": run.surrogate.sizes}
    for k, name in enumerate(OUTPUT_NAMES):
        for level in range(1, run.n_levels + 1):
            y, mean, var = run.models[k].leave_one_out(level, by_location=True)
            z = (y - mean) / np.sqrt(var)
            q2 = 1.0 - np.sum((y - mean) ** 2) / np.sum((y - y.mean()) ** 2)
            out["loo"].append({"output": name, "level": level, "q2": float(q2),
                               **scores_stats(z), "z": z.tolist()})
    # the codes at the VPP states of the validated sections: points the surrogate never saw
    table = run.validation
    for section, scores in run.sections.items():
        rows = table[table["section"] == section].reset_index(drop=True)
        if rows.empty:
            continue
        x = np.hstack([np.repeat(scores[None], len(rows), 0),
                       rows[["att0", "att1", "att2"]].to_numpy()])
        for key, level in (("hf", 2), ("lf", 1)):
            codes = rows[[f"{n}_{key}" for n in OUTPUT_NAMES]].to_numpy()
            ok = rows["success"].to_numpy(dtype=bool) & np.all(np.isfinite(codes), axis=1)
            if not ok.any():
                continue
            mean, var = np.zeros_like(codes), np.zeros_like(codes)
            for k in range(len(OUTPUT_NAMES)):
                mean[:, k], var[:, k] = run.models[k].predict_batch(x, level, include_noise=True)
            z = (mean - codes) / np.sqrt(var)
            speed_error = np.sum((mean - codes)[:, :N_FORCES] * run.derivatives, axis=1)
            speed_sigma = np.sqrt(np.sum(var[:, :N_FORCES] * run.derivatives ** 2, axis=1))
            entry = {"section": section, "level": level, "n_states": int(ok.sum()),
                     "speed_error_rms_kt": float(np.sqrt(np.mean(speed_error[ok] ** 2))),
                     "speed_error_mean_kt": float(np.mean(speed_error[ok])),
                     "speed_sigma_rms_kt": float(np.sqrt(np.mean(speed_sigma[ok] ** 2))),
                     "speed": scores_stats(speed_error[ok] / speed_sigma[ok]),
                     "speed_z": (speed_error[ok] / speed_sigma[ok]).tolist(),
                     "outputs": {n: scores_stats(z[ok, k]) for k, n in enumerate(OUTPUT_NAMES)}}
            # the weighted mean over the conditions: what the objective sees (errors common to
            # the conditions do not average out)
            w = np.array(run.config["condition_weights"])[ok]
            entry["objective_error_kt"] = float(speed_error[ok] @ w / w.sum())
            out["external"].append(entry)
    # top level (the one the run optimised) minus level 2, at the states of the best section
    x = run.state_inputs(run.best)
    shift = {}
    for k, name in enumerate(OUTPUT_NAMES[:N_FORCES]):
        top, second = latent_mean(run.models[k], x), latent_mean(run.models[k], x, 2)
        shift[name] = float(np.median(np.abs(top - second) / np.abs(second)))
    out["top_minus_level2_median_relative"] = shift
    save("calibration", out)


# -------------------------------------------------------------------------------------------------
# 2. Gradient, KKT, residual ascent (sections 2.2 and 2.3)
# -------------------------------------------------------------------------------------------------
def kkt(gradient, covariance, normals):
    """Multipliers mu >= 0 with gradient = -sum mu_i normal_i as well as possible (non-negative
    least squares), the tangent part of the gradient (projection on the null space of the
    normals with mu > 0) and the chi-square test of 'tangent part = 0'."""
    mu = nnls(-normals.T, gradient)[0] if len(normals) else np.zeros(0)
    pushing = normals[mu > 0]
    projector = np.eye(len(gradient))
    if len(pushing):
        q = np.linalg.qr(pushing.T)[0]
        projector -= q @ q.T
    tangent = projector @ gradient
    cov_t = projector @ covariance @ projector
    values, vectors = np.linalg.eigh(cov_t)
    keep = values > 1e-10 * values.max()
    statistic = float(np.sum((vectors[:, keep].T @ tangent) ** 2 / values[keep]))
    return mu, tangent, statistic, int(keep.sum()), projector


def kkt_summary(run: FinishedRun, g, cov, hessian, reach: float = 0.05) -> dict:
    """KKT analysis at the best section. Active constraints = those whose boundary is within
    `reach` (scores) of the section: margin / |normal| < reach (the search moves by steps, the
    section sits near the boundaries, not on them; a tolerance on the raw margins mixes units)."""
    margins = run.constraints(run.best)
    jacobian = run.constraint_jacobian(run.best)
    names = np.array(run.constraint_names())
    norms = np.linalg.norm(jacobian, axis=1)
    distance = margins / np.maximum(norms, 1e-30)
    active = np.flatnonzero(distance < reach)
    mu, tangent, statistic, dof, projector = kkt(g, cov, jacobian[active])
    h_tangent = projector @ hessian @ projector
    nearest = {}
    for name in dict.fromkeys(n.split("_low_")[0].split("_high_")[0] for n in names):
        rows = np.array([n.startswith(name) for n in names])
        nearest[name] = float(distance[rows].min())
    return {
        "reach": reach, "distance_to_constraints": nearest,
        "active_counts": {n: int(np.sum(names[active] == n)) for n in dict.fromkeys(names[active])},
        # multiplier: kt gained per unit of margin released (Ixx / v: per unit of relative
        # change of its limit, i.e. 0.01 x multiplier per percent)
        "pushing": [{"name": str(names[i]), "margin": float(margins[i]), "multiplier": float(m),
                     "distance": float(distance[i])} for i, m in zip(active, mu) if m > 0],
        "tangent_gradient": tangent.tolist(), "tangent_norm": float(np.linalg.norm(tangent)),
        "kkt": {str(kappa): {"statistic": statistic / kappa ** 2, "dof": dof,
                             "p_value": float(chi2.sf(statistic / kappa ** 2, dof))}
                for kappa in (1, 2, 3)},
        "tangent_hessian_eigenvalues": np.linalg.eigvalsh(h_tangent).tolist()}


def kkt_refresh(run: FinishedRun) -> None:
    """The KKT analysis again from the saved gradient and Hessian (no model evaluation)."""
    path = RESULTS["dir"] / "gradient.json"
    out = json.loads(path.read_text(encoding="utf-8"))
    for key in ("active", "smallest_margins"):
        out.pop(key, None)
    out.update(kkt_summary(run, np.array(out["gradient"]), np.array(out["gradient_covariance"]),
                           np.array(out["hessian"])))
    save("gradient", out)


def gradient_experiment(run: FinishedRun) -> None:
    g, cov = run.gradient(run.best, with_covariance=True)
    hessian = run.hessian(run.best)
    out = {
        "gradient": g.tolist(), "gradient_sigma": np.sqrt(np.diag(cov)).tolist(),
        "gradient_covariance": cov.tolist(), "gradient_norm": float(np.linalg.norm(g)),
        "hessian": hessian.tolist(),
        "hessian_eigenvalues": np.linalg.eigvalsh(hessian).tolist(),
        **kkt_summary(run, g, cov, hessian), "ascent": []}
    # what an ascent on the model would still gain, inside balls around the best section
    j0 = run.objective(run.best)[0]
    for radius in (0.1, 0.3, 1.0):
        z = run.ascent(run.best, radius)
        s = run.objective_covariance(np.vstack([run.best, z]))
        out["ascent"].append({
            "radius": radius, "gain_kt": float(run.objective(z)[0] - j0),
            "gain_sigma_kt": float(np.sqrt(max(s[0, 0] + s[1, 1] - 2.0 * s[0, 1], 0.0))),
            "step_norm": float(np.linalg.norm(z - run.best)), "scores": z.tolist(),
            "admissible": bool(run.admissible(z)[0]),
            "min_margin": float(run.constraints(z).min())})
        print(f"  ascent radius {radius}: gain {out['ascent'][-1]['gain_kt']:+.4f} kt "
              f"+- {out['ascent'][-1]['gain_sigma_kt']:.4f}", flush=True)
    save("gradient", out)


# -------------------------------------------------------------------------------------------------
# 3. Cuts through the best section
# -------------------------------------------------------------------------------------------------
def slices(run: FinishedRun, half_width: float = 1.5, n_points: int = 61) -> None:
    t = np.linspace(-half_width, half_width, n_points)
    out = {"t": t.tolist(), "modes": []}
    for j in range(run.k):
        sections = run.best + t[:, None] * np.eye(run.k)[j]
        both = np.vstack([run.best, sections])
        values = run.objective(both)
        s = run.objective_covariance(both)
        sigma = np.sqrt(np.maximum(s[0, 0] + np.diag(s)[1:] - 2.0 * s[0, 1:], 0.0))
        level2 = run.objective(both, level=2)
        out["modes"].append({"mode": j + 1, "gain": (values[1:] - values[0]).tolist(),
                             "sigma": sigma.tolist(),
                             "gain_level2": (level2[1:] - level2[0]).tolist(),
                             "admissible": run.admissible(sections).tolist()})
    save("slices", out)


# -------------------------------------------------------------------------------------------------
# 4. Probability that a better section exists (section 2.1)
# -------------------------------------------------------------------------------------------------
def regret(run: FinishedRun, n_per_group: int = 60, n_draws: int = 20000, seed: int = 0) -> None:
    groups, parts = [], []
    for name, scores in run.sections.items():
        groups.append(name)
        parts.append(scores[None])
    for radius in (0.1, 0.3, 1.0):
        local = run.local_sections(run.best, radius, n_per_group, seed + int(100 * radius))
        groups += [f"local_{radius}"] * len(local)
        parts.append(local)
    spread = run.global_sections(n_per_group, seed + 7)
    groups += ["global"] * len(spread)
    parts.append(spread)
    sections = np.vstack(parts)
    mean = run.objective(sections)
    cov = run.objective_covariance(sections)
    gain = mean - mean[0]
    sigma_diff = np.sqrt(np.maximum(cov[0, 0] + np.diag(cov) - 2.0 * cov[0], 0.0))
    values, vectors = np.linalg.eigh(cov)
    root = vectors * np.sqrt(np.clip(values, 0.0, None))
    normal = np.random.default_rng(seed).standard_normal((n_draws, len(sections)))
    group_array = np.array(groups)
    out = {"groups": groups, "gain": gain.tolist(), "sigma_diff": sigma_diff.tolist(),
           "distance": np.linalg.norm(sections - run.best, axis=1).tolist(),
           "sigma_best": float(np.sqrt(cov[0, 0])), "n_draws": n_draws, "scales": {}}
    for kappa in (1.0, 2.0, 3.0):
        draws = mean + kappa * normal @ root.T
        lead = draws.max(axis=1) - draws[:, 0]              # 0 when the best section wins
        winners = group_array[np.argmax(draws, axis=1)]
        out["scales"][str(kappa)] = {
            "expected_regret_kt": float(lead.mean()),
            "regret_quantiles_kt": {q: float(np.quantile(lead, float(q)))
                                    for q in ("0.5", "0.9", "0.99")},
            "p_better_by": {str(eps): float(np.mean(lead > eps))
                            for eps in (0.0, 0.002, 0.005, 0.01, 0.02)},
            "p_winner_group": {name: float(np.mean(winners == name))
                               for name in dict.fromkeys(groups)},
            "lead_sample": lead[:4000].tolist()}
        print(f"  kappa {kappa}: regret {lead.mean():.4f} kt, P(better by > 0.01 kt) "
              f"{np.mean(lead > 0.01):.3f}", flush=True)
    save("regret", out)


# -------------------------------------------------------------------------------------------------
# 5. Effective dimension (section 4)
# -------------------------------------------------------------------------------------------------
def dimension(run: FinishedRun, n_sections: int = 100, seed: int = 3) -> None:
    out = {}
    samples = {"global": run.global_sections(n_sections, seed),
               "local_1.0": run.local_sections(run.best, 1.0, n_sections, seed + 1)}
    for name, sections in samples.items():
        gradients = np.array([run.gradient(z) for z in sections])
        matrix = gradients.T @ gradients / len(gradients)
        values, vectors = np.linalg.eigh(matrix)
        out[name] = {"n": int(len(sections)), "eigenvalues": values[::-1].tolist(),
                     "eigenvectors": vectors[:, ::-1].T.tolist(),
                     "gradient_norm_mean": float(np.mean(np.linalg.norm(gradients, axis=1)))}
        share = np.cumsum(values[::-1]) / values.sum()
        print(f"  {name}: cumulative share of the eigenvalues {np.round(share[:5], 3)}", flush=True)
    save("dimension", out)


# -------------------------------------------------------------------------------------------------
# 6. Shift of the optimum between levels (section 5.2)
# -------------------------------------------------------------------------------------------------
def levels(run: FinishedRun, radius: float = 1.0) -> None:
    top = run.n_levels
    out = {"levels": [], "top": top, "radius": radius,
           "rhos": {n: [float(r) for r in np.ravel(run.models[k].rhos)]
                    for k, n in enumerate(OUTPUT_NAMES[:N_FORCES])}}
    gradients, optima = {}, {}
    for level in range(1, top + 1):
        gradients[level] = run.gradient(run.best, level)
        optima[level] = run.ascent(run.best, radius, level)
    for level in range(1, top + 1):
        delta = gradients[top] - gradients[level]
        both = np.vstack([optima[top], optima[level]])
        j_top = run.objective(both)                       # judged at the top level
        s = run.objective_covariance(both)
        cosine = float(gradients[top] @ gradients[level]
                       / (np.linalg.norm(gradients[top]) * np.linalg.norm(gradients[level])))
        out["levels"].append({
            "level": level, "gradient": gradients[level].tolist(),
            "gradient_norm": float(np.linalg.norm(gradients[level])),
            "delta_gradient_norm": float(np.linalg.norm(delta)), "cosine_with_top": cosine,
            "optimum": optima[level].tolist(),
            "optimum_shift": float(np.linalg.norm(optima[level] - optima[top])),
            "loss_at_top_level_kt": float(j_top[0] - j_top[1]),
            "loss_sigma_kt": float(np.sqrt(max(s[0, 0] + s[1, 1] - 2.0 * s[0, 1], 0.0))),
            "gain_own_level_kt": float(run.objective(optima[level], level)[0]
                                       - run.objective(run.best, level)[0])})
        print(f"  level {level}: cos(grad, grad top) {cosine:.3f}, loss at the top level "
              f"{out['levels'][-1]['loss_at_top_level_kt']:.4f} kt", flush=True)
    save("levels", out)


EXPERIMENTS = {"calibration": calibration, "gradient": gradient_experiment,
               "kkt": kkt_refresh, "slices": slices,
               "regret": regret, "dimension": dimension, "levels": levels}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", maxsplit=1)[0])
    parser.add_argument("--run", required=True, help="run folder, relative to vpp_acceleration")
    parser.add_argument("--only", nargs="+", choices=list(EXPERIMENTS),
                        default=[name for name in EXPERIMENTS if name != "kkt"])
    parser.add_argument("--out", default="results",
                        help="results folder, relative to validation_experiments (one per run)")
    args = parser.parse_args()
    RESULTS["dir"] = HERE / args.out
    start = time.time()
    run = FinishedRun(VPP_DIR / args.run)
    print(f"run {run.root.name}: design per level {run.surrogate.sizes}, "
          f"{int(run.solved.sum())}/{run.n_conditions} conditions solved, loaded in "
          f"{time.time() - start:.0f} s", flush=True)
    RESULTS["dir"].mkdir(parents=True, exist_ok=True)
    (RESULTS["dir"] / "run.json").write_text(json.dumps({
        "run": run.root.name, "sizes": run.surrogate.sizes, "levels": run.config["levels"],
        "n_solved": int(run.solved.sum()), "n_conditions": run.n_conditions,
        "best": run.best.tolist(), "k": run.k}, indent=1), encoding="utf-8")
    for name in args.only:
        t0 = time.time()
        print(f"[{name}]", flush=True)
        EXPERIMENTS[name](run)
        print(f"  {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
