"""
Quick test (2026-10-21): is a CONSTANT rho too rigid? It is the hypothesis left by
library_study.py: on the same designs, the multi-fidelity GP of BoTorch is about 15 % more
accurate than our CoKriging, whose AR(1) model has one constant rho.

Prototype (this file only, mfego is not modified): two-level RECURSIVE co-kriging of Le Gratiet &
Garnier (2014), which equals the AR(1) co-kriging for nested designs and noise-free low-fidelity
data ([LG] Prop. 3.1, mfego/src/cokriging.py docstring):
    Z_2(x) = rho(x) Z_1(x) + delta(x),     rho(x) = g(x)^T beta_rho,
    P0: g(x) = 1                (constant rho, as our CoKriging),
    P1: g(x) = (1, x_1, ..., x_d) (rho linear in x: d more parameters).
Level 1: GP on all the LF data. Level 2: GP on the HF data whose trend is H beta, with the columns
H = [y_1(x) g(x), 1] (y_1 observed at the HF points: nested design). For given kernel
hyperparameters, beta (rho(x) and the constant of delta) is estimated in closed form by
generalized least squares, exactly as the scalar rho of mfego, and the hyperparameters maximize
the concentrated likelihood. Prediction (recursive formulation):
    mean(x) = rho(x) m_1(x) + mean_delta(x),   var(x) = rho(x)^2 v_1(x) + v_delta(x).
P0 and P1 share everything except g: their difference isolates the effect of rho(x).

Designs (10 seeds each): B1 nested Latin hypercube 160 LF / 40 HF over the whole domain, and B3
nested space-filling design 160 / 40 inside the target region (region_study.py). Models fitted
on each design: CK (our CoKriging v2, as S1), P0, P1, BT (BoTorch SingleTaskMultiFidelityGP, as
T1 of library_study.py). Scores: relative RMSE and 95 % coverage on the region test set and on
the global test set.

Run from the repository root (Multifidelity-GP-Clean), a few minutes:
    .venv-benchmark\\Scripts\\python vpp_acceleration\\examples\\rho_study.py
Outputs: examples/results/rho_study.csv, rho_summary.txt
"""
import os

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

# pylint: disable=wrong-import-position,import-error
import argparse  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.linalg import cho_solve, cholesky, solve_triangular  # noqa: E402
from scipy.optimize import minimize  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import benchmark_study as bs  # noqa: E402
import library_study as ls  # noqa: E402  (BotorchMF)
import region_study as rs  # noqa: E402
from src.cokriging import nested_index  # noqa: E402

MODELS = ("CK", "P0", "P1", "BT")
LABELS = {"CK": "CK notre CoKriging (ρ constant, v2)", "P0": "P0 prototype, ρ constant",
          "P1": "P1 prototype, ρ(x) linéaire", "BT": "BT BoTorch MF-GP"}
DESIGNS = ("B1", "B3")
LENGTHSCALE_BOUNDS = (1e-2, 1e1)        # inputs in the unit cube
NUGGET_BOUNDS = (1e-8, 1e-2)            # noise / signal variance ratio


# 1/3 ---------------------------------------------------------------------------------------------
# A GP with a linear trend (universal kriging)
# -------------------------------------------------------------------------------------------------
def matern52(x_a: np.ndarray, x_b: np.ndarray, lengthscales: np.ndarray) -> np.ndarray:
    """Matern 5/2 correlation (unit variance) between the rows of x_a and of x_b."""
    diff = (x_a[:, None, :] - x_b[None, :, :]) / lengthscales
    r = np.sqrt(np.sum(diff ** 2, axis=2))
    return (1.0 + np.sqrt(5.0) * r + 5.0 * r ** 2 / 3.0) * np.exp(-np.sqrt(5.0) * r)


class TrendGP:
    """
    y = H beta + delta, delta ~ GP(0, s2 (R_theta + g I)), R_theta Matern 5/2 with ARD
    lengthscales theta, g a nugget ratio. For given (theta, g), beta and s2 are estimated in
    closed form (generalized least squares):
        beta = (H^T C^-1 H)^-1 H^T C^-1 y,    s2 = (y - H beta)^T C^-1 (y - H beta) / n,
    with C = R_theta + g I, and (theta, g) minimize the concentrated -log likelihood
        n/2 log s2 + 1/2 log det C        (maximum likelihood, no prior).
    Prediction at x with the trend row h(x) (universal kriging, the uncertainty of beta included):
        mean = h beta + r^T C^-1 (y - H beta),
        var  = s2 [1 - r^T C^-1 r + u^T (H^T C^-1 H)^-1 u],   u = h - H^T C^-1 r.
    """

    def __init__(self, n_restarts: int = 4, seed: int = 0):
        self.n_restarts = n_restarts
        self.seed = seed

    def _solve(self, log_params):
        """Everything that depends on (theta, g): Cholesky of C, beta, s2, -log likelihood."""
        lengthscales, nugget = np.exp(log_params[:-1]), np.exp(log_params[-1])
        n = len(self.y)
        covariance = matern52(self.x, self.x, lengthscales) + (nugget + 1e-10) * np.eye(n)
        lower = cholesky(covariance, lower=True)
        c_inv_h = cho_solve((lower, True), self.h)
        gram = self.h.T @ c_inv_h                                   # H^T C^-1 H
        beta = np.linalg.solve(gram, c_inv_h.T @ self.y)
        residual = self.y - self.h @ beta
        alpha = cho_solve((lower, True), residual)                   # C^-1 (y - H beta)
        s2 = float(residual @ alpha) / n
        nll = 0.5 * n * np.log(s2) + np.sum(np.log(np.diag(lower)))
        return {"lengthscales": lengthscales, "lower": lower, "gram": gram, "beta": beta,
                "alpha": alpha, "s2": s2, "nll": nll}

    def fit(self, x: np.ndarray, y: np.ndarray, h: np.ndarray,
            lengthscales: np.ndarray = None) -> "TrendGP":
        """
        Args:
        - x (n, d), y (n,), h (n, p): data and trend rows.
        - lengthscales: if given, the lengthscales are FIXED to these values and only the nugget
          is estimated (used by joint_study.py: correction with the lengthscales of level 1).
        """
        self.scale = float(np.std(y))                              # y / scale for the numerics
        self.x, self.y, self.h = x, y / self.scale, h
        d = x.shape[1]
        if lengthscales is not None:
            fixed = np.log(np.asarray(lengthscales, dtype=float))

            def objective_nugget(log_nugget):
                return self._solve(np.append(fixed, log_nugget))["nll"]
            result = minimize(objective_nugget, [np.log(1e-6)], method="L-BFGS-B",
                              bounds=[tuple(np.log(NUGGET_BOUNDS))])
            self.state = self._solve(np.append(fixed, result.x))
            return self
        bounds = [tuple(np.log(LENGTHSCALE_BOUNDS))] * d + [tuple(np.log(NUGGET_BOUNDS))]
        rng = np.random.default_rng(self.seed)
        best = None
        for restart in range(self.n_restarts):
            start = np.full(d + 1, np.log(0.3)) if restart == 0 else \
                np.array([rng.uniform(*b) for b in bounds])
            start[-1] = np.log(1e-6) if restart == 0 else start[-1]

            def objective(log_params):
                try:
                    return self._solve(log_params)["nll"]
                except (np.linalg.LinAlgError, ValueError):
                    return 1e10
            result = minimize(objective, start, method="L-BFGS-B", bounds=bounds)
            if best is None or result.fun < best.fun:
                best = result
        self.state = self._solve(best.x)
        return self

    @property
    def beta(self) -> np.ndarray:
        """Trend coefficients in the units of y (the trend columns are not scaled)."""
        return self.state["beta"] * self.scale

    def predict(self, x_new: np.ndarray, h_new: np.ndarray) -> tuple:
        """Mean and latent variance at x_new, trend rows h_new (m, p)."""
        s = self.state
        cross = matern52(x_new, self.x, s["lengthscales"])           # r^T, (m, n)
        mean = h_new @ s["beta"] + cross @ s["alpha"]
        v = solve_triangular(s["lower"], cross.T, lower=True)        # L^-1 r
        c_inv_r = cho_solve((s["lower"], True), cross.T)             # C^-1 r, (n, m)
        u = h_new.T - self.h.T @ c_inv_r                             # (p, m)
        trend_term = np.sum(u * np.linalg.solve(s["gram"], u), axis=0)
        variance = s["s2"] * np.maximum(1.0 - np.sum(v ** 2, axis=0) + trend_term, 0.0)
        return mean * self.scale, variance * self.scale ** 2


# 2/3 ---------------------------------------------------------------------------------------------
# Recursive co-kriging with rho(x) = g(x)^T beta_rho
# -------------------------------------------------------------------------------------------------
class RecursiveCoKriging:
    """Two levels, nested design. rho_form "constant" (P0) or "linear" (P1)."""

    def __init__(self, rho_form: str, seed: int = 0, share_lengthscales: bool = False):
        self.rho_form = rho_form
        self.seed = seed
        # True: the correction delta takes the lengthscales of level 1 (learned on all the LF
        # points) instead of learning its own on the HF points only (joint_study.py)
        self.share_lengthscales = share_lengthscales

    def basis(self, x: np.ndarray) -> np.ndarray:
        """g(x): (m, 1) for a constant rho, (m, 1 + d) for a rho linear in x."""
        ones = np.ones((len(x), 1))
        return ones if self.rho_form == "constant" else np.column_stack([ones, x])

    def trend(self, x: np.ndarray, f_low: np.ndarray) -> np.ndarray:
        """Trend rows of level 2: [f_low(x) g(x), 1]."""
        return np.column_stack([f_low[:, None] * self.basis(x), np.ones(len(x))])

    def fit(self, x_lists, y_lists) -> "RecursiveCoKriging":
        (x1, y1), (x2, y2) = zip(x_lists, y_lists)
        index = nested_index(x2, x1)
        if np.any(index < 0):
            raise ValueError("RecursiveCoKriging needs a nested design")
        self.gp1 = TrendGP(seed=self.seed).fit(x1, y1, np.ones((len(x1), 1)))
        shared = self.gp1.state["lengthscales"] if self.share_lengthscales else None
        self.gp2 = TrendGP(seed=self.seed).fit(x2, y2, self.trend(x2, y1[index]),
                                               lengthscales=shared)
        self.beta_rho = self.gp2.beta[:-1]                           # rho(x) = g(x)^T beta_rho
        return self

    def rho(self, x: np.ndarray) -> np.ndarray:
        return self.basis(x) @ self.beta_rho

    def predict_batch(self, x: np.ndarray, include_noise: bool = False) -> tuple:
        """HF mean and latent variance (same signature as CoKriging.predict_batch)."""
        del include_noise
        m1, v1 = self.gp1.predict(x, np.ones((len(x), 1)))
        m_delta, v_delta = self.gp2.predict(x, self.trend(x, m1))   # includes rho(x) m1(x)
        return m_delta, self.rho(x) ** 2 * v1 + v_delta

    def rho_range(self) -> tuple:
        """Smallest and largest rho(x) over the unit cube (linear: reached at the corners)."""
        if self.rho_form == "constant":
            return float(self.beta_rho[0]), float(self.beta_rho[0])
        slopes = self.beta_rho[1:]
        return (float(self.beta_rho[0] + np.minimum(slopes, 0.0).sum()),
                float(self.beta_rho[0] + np.maximum(slopes, 0.0).sum()))


# 3/3 ---------------------------------------------------------------------------------------------
# Runs, summary
# -------------------------------------------------------------------------------------------------
def run_task(task):
    design_name, seed = task
    region = rs.region_data()
    global_test = bs.test_set("airfoil")
    problem = bs.make_problem("airfoil")
    data = rs.design_in_region(rs.LHS_DESIGNS["B3"], seed) if design_name == "B3" else \
        bs.nested_design(problem, rs.LHS_DESIGNS["B1"], seed)
    x_lists = [data.get_training_data(l)[0] for l in (1, 2)]
    y_lists = [data.get_training_data(l)[1] for l in (1, 2)]
    rows = []
    for name in MODELS:
        start = time.perf_counter()
        extra = {}
        if name == "CK":
            model = bs.make_model("v2")
            model.fit(data)
            extra["rho_min"] = extra["rho_max"] = float(model.rhos[0])
        elif name == "BT":
            model = ls.BotorchMF(region["x_int"], seed).fit(x_lists, y_lists)
        else:
            model = RecursiveCoKriging("constant" if name == "P0" else "linear", seed)
            model.fit(x_lists, y_lists)
            extra["rho_min"], extra["rho_max"] = model.rho_range()
        rows.append({"design": design_name, "seed": seed, "model": name,
                     "fit_time_s": time.perf_counter() - start, **extra,
                     **rs.all_scores(model, region, global_test)})
    return rows


def summarize(results: pd.DataFrame) -> str:
    lines = ["=== Is a constant rho too rigid? (same designs for every model) ==="]
    for design in DESIGNS:
        r = results[results["design"] == design]
        lines.append(f"\n--- design {design} ({r['seed'].nunique()} seeds), median [10 % ; 90 %]")
        for name in MODELS:
            m = r[r["model"] == name]
            rho = "" if name == "BT" else \
                f" | rho {m['rho_min'].median():.2f} to {m['rho_max'].median():.2f}"
            lines.append(f"{LABELS[name]:38s} region {m['rmse_region'].median():.4f} "
                         f"[{m['rmse_region'].quantile(0.1):.4f} ; "
                         f"{m['rmse_region'].quantile(0.9):.4f}] cov {m['coverage_region'].median():.2f}"
                         f" | global {m['rmse_global'].median():.4f} cov "
                         f"{m['coverage_global'].median():.2f}{rho}")
        table = r.pivot(index="seed", columns="model", values="rmse_region")
        for a, b in (("P1", "P0"), ("P1", "CK"), ("BT", "P1"), ("BT", "CK")):
            wins = int((table[a] < table[b]).sum())
            p_value = wilcoxon(table[a], table[b], alternative="less").pvalue
            change = 1.0 - table[a].median() / table[b].median()
            lines.append(f"{a} against {b} (region): median {100 * change:+.0f} % lower | wins "
                         f"{wins} / {len(table)} | Wilcoxon ({a} < {b}) p = {p_value:.2g}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--workers", type=int, default=20)
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    rs.region_data()
    tasks = [(design, seed) for design in DESIGNS for seed in range(args.seeds)]
    rows = []
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks))) as pool:
        for task_rows in pool.map(run_task, tasks):
            rows += task_rows
    print(f"{len(tasks)} designs in {time.perf_counter() - start:.0f} s")
    results = pd.DataFrame(rows)
    results.to_csv(bs.RESULTS / "rho_study.csv", index=False)
    summary = summarize(results)
    print(summary)
    (bs.RESULTS / "rho_summary.txt").write_text(summary + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
