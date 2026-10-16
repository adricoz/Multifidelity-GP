"""
Benchmark of the priors on ALL the hyperparameters of mfego (lengthscales, signal variance t1,
bias variance t2 and noise: use_map=True with DEFAULT_PRIORS) on the multi-fidelity Hartmann 6D
problem of benchmarks/bench_lib.py (Sacher et al. 2021, Eqs. 30-32), against the maximum
likelihood (MLE), the previous MAP (prior on the lengthscales only) and the other GP libraries.

1. Accuracy: n_HF in {5, 10, 20, 40}, n_LF = 2 n_HF, 10 seeds, 2000 test points (same DOE for
   every model of a seed: paired comparisons). Fitted hyperparameters are stored.
2. Many points, high fidelity only (n in {80, 160, 300}): does MAP-all avoid the degenerate
   maximum likelihood solution of benchmarks/map_hartmann/RAPPORT_MAP.md (Section 6)?
3. Optimization convergence at equal cost budget (250, same protocol as map_hartmann).
4. Sensitivity of the new priors, one factor at a time (accuracy and NN-MF-EGO).

Run from the repository root with the benchmark environment (about 10 min on 32 cores):
    .venv-benchmark\\Scripts\\python benchmarks\\priors_all\\run_priors_benchmark.py [--quick]
Results: benchmarks/priors_all/results/ (figures: make_figures.py, report: RAPPORT_PRIORS.md).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

# one BLAS thread per process (set before numpy is imported, also in the worker processes)
os.environ.setdefault("OMP_NUM_THREADS", "1")

# pylint: disable=wrong-import-position,import-error
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import qmc  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import bench_lib as bl  # noqa: E402
from src.surrogate_models import (BIAS_VARIANCE_BOUNDS, LENGTHSCALE_BOUNDS,  # noqa: E402
                                  LENGTHSCALE_ONLY_PRIORS, NOISE_BOUNDS, SIGNAL_VARIANCE_BOUNDS)

PROBLEM = {"ks": (1, np.inf), "delta": 0.05, "costs": (1.0, 10.0)}
MF_DOE, SF_DOE = [20, 10], [12]      # same initial cost: 20 x 1 + 10 x 10 = 12 x 10 = 120
BUDGET = 250                         # total cost of an optimization run (DOE included)
RESULTS = HERE / "results"
LOG10 = np.log(10.0)

# hyperparameter estimators of mfego: name -> keyword arguments of MultifidelityModel
ESTIMATORS = {
    "MLE": {},
    "MAP-l": {"use_map": True, "priors": LENGTHSCALE_ONLY_PRIORS},  # previous use_map=True
    "MAP-all": {"use_map": True},                                   # DEFAULT_PRIORS
}
# one factor at a time around MAP-all (lognormal t1 priors keep their mode at 1: mu = sigma^2)
SENSITIVITY = {
    "MAP-all, l IG(3, 1)": {"use_map": True, "lengthscale_prior": (3.0, 1.0)},
    "MAP-all, t1 sigma 0.5": {"use_map": True,
                              "priors": {"signal_variance": ("lognormal", 0.25, 0.5)}},
    "MAP-all, t1 sigma 2": {"use_map": True,
                            "priors": {"signal_variance": ("lognormal", 4.0, 2.0)}},
    "MAP-all, noise 1e-6": {"use_map": True,
                            "priors": {"noise": ("lognormal", -6.0 * LOG10, 2.0)}},
    "MAP-all, noise 1e-3": {"use_map": True,
                            "priors": {"noise": ("lognormal", -3.0 * LOG10, 2.0)}},
    "MAP-all, no t1 prior": {"use_map": True, "priors": {"signal_variance": None}},
    "MAP-all, no t2 prior": {"use_map": True, "priors": {"bias_variance": None}},
    "MAP-all, no noise prior": {"use_map": True, "priors": {"noise": None}},
}
LIBRARIES = {
    "BoTorch MF-GP (MF)": lambda seed, costs: bl.BotorchSurrogate(True, costs),
    "BoTorch SingleTaskGP (SF)": lambda seed, costs: bl.BotorchSurrogate(False, costs),
    "scikit-learn GP (SF)": lambda seed, costs: bl.SklearnSurrogate(seed=seed),
    "SMT MFK (MF)": lambda seed, costs: bl.SmtSurrogate(multi_fidelity=True),
}


def surrogate(name: str, seed: int, costs) -> bl.Surrogate:
    """'mfego MF (MAP-all)' / 'mfego SF (MLE)' / ... or a library model."""
    if name in LIBRARIES:
        return LIBRARIES[name](seed, costs)
    variant, estimator = name[len("mfego "):len("mfego ") + 2], name[len("mfego MF ("):-1]
    kwargs = {**ESTIMATORS, **SENSITIVITY}[estimator]
    return bl.MfegoSurrogate(variant == "MF", seed, **kwargs)


def hyperparameters(model) -> list:
    """Fitted hyperparameters of every level of an mfego model (JSON friendly)."""
    return [{"lengthscales": gp.kernel.lengthscale.tolist(),
             "signal_variance": float(gp.kernel.signal_variance),
             "bias_variance": float(gp.kernel.bias_variance),
             "noise": float(gp.noise), "rho": gp.rho} for gp in model.gps]


# 1/3 ---------------------------------------------------------------------------------------------
# Accuracy (multi-fidelity sizes and many points)
# -------------------------------------------------------------------------------------------------
def accuracy_task(args: tuple) -> dict:
    """(model name, n_LF, n_HF, seed) -> accuracy metrics (+ hyperparameters for mfego)."""
    name, n_lf, n_hf, seed = args
    problem = bl.HartmannMF(**PROBLEM)
    datasets = problem.doe([n_lf, n_hf] if n_lf else [n_hf], seed=seed)
    x_test = qmc.LatinHypercube(d=problem.dim, seed=12345).random(2000)
    try:
        model = surrogate(name, seed, problem.costs)
        result = bl.accuracy(model, datasets, x_test, bl.hartmann6(x_test))
        if isinstance(model, bl.MfegoSurrogate):
            result["hyper"] = json.dumps(hyperparameters(model.model))
    except Exception as error:  # noqa: BLE001  (a failing model must not stop the benchmark)
        result = {"error": repr(error)}
    result.update({"model": name, "n_lf": n_lf, "n_hf": n_hf, "seed": seed})
    return result


def bound_shares(acc: pd.DataFrame) -> pd.DataFrame:
    """Share of the HF-level hyperparameters at a search bound (lower / upper), per group."""
    bounds = {"lengthscales": LENGTHSCALE_BOUNDS, "signal_variance": SIGNAL_VARIANCE_BOUNDS,
              "bias_variance": BIAS_VARIANCE_BOUNDS, "noise": NOISE_BOUNDS}
    rows = []
    for _, row in acc.dropna(subset=["hyper"]).iterrows():
        top = json.loads(row["hyper"])[-1]
        out = {"model": row["model"], "n_hf": row["n_hf"]}
        for group, (lo, hi) in bounds.items():
            values = np.atleast_1d(top[group])
            out[f"{group} low"] = float(np.mean(values <= lo * 1.05))
            out[f"{group} high"] = float(np.mean(values >= hi / 1.05))
        rows.append(out)
    return pd.DataFrame(rows).groupby(["n_hf", "model"]).mean().round(2)


def paired_wins(acc: pd.DataFrame, better: str, worse: str, metric: str) -> pd.Series:
    """Number of seeds where model `better` has a lower metric than model `worse`."""
    wide = acc.pivot_table(index=["n_hf", "seed"], columns="model", values=metric)
    wins = wide[better] < wide[worse]
    return wins.groupby(level="n_hf").agg(lambda b: f"{int(b.sum())}/{len(b)}")


def summarize_accuracy(acc: pd.DataFrame, title: str) -> str:
    ok = acc[acc["error"].isna()] if "error" in acc else acc
    table = ok.groupby(["n_hf", "model"]).agg(
        rmse_rel=("rmse_rel", "median"), nlpd=("nlpd", "median"),
        coverage_95=("coverage_95", "median"), fit_time_s=("fit_time_s", "median")).round(3)
    lines = [f"=== {title}: medians over the seeds ===", table.to_string(), ""]
    for variant in ("MF", "SF"):
        names = [f"mfego {variant} ({e})" for e in ESTIMATORS]
        if not all(n in set(ok["model"]) for n in names):
            continue
        wins = pd.DataFrame({
            f"{variant} MAP-all < {other} ({metric})": paired_wins(
                ok, f"mfego {variant} (MAP-all)", f"mfego {variant} ({other})", metric)
            for other in ("MLE", "MAP-l") for metric in ("rmse_rel", "nlpd")})
        lines += [f"=== {variant}: seeds where MAP-all is better (same DOE) ===",
                  wins.to_string(), ""]
    if "hyper" in ok:
        lines += ["=== HF level: share of hyperparameters at a search bound ===",
                  bound_shares(ok).to_string(), ""]
    return "\n".join(lines)


# 2/3 ---------------------------------------------------------------------------------------------
# Optimization at equal cost budget
# -------------------------------------------------------------------------------------------------
def optimization_task(args: tuple) -> dict:
    """(method, seed, budget) -> History as a dict."""
    from dataclasses import asdict  # noqa: PLC0415
    method, seed, budget = args
    problem = bl.HartmannMF(**PROBLEM)
    try:
        if method == "BoTorch qLogEI (SF)":
            history = bl.run_botorch(problem, problem.doe(SF_DOE, seed), budget, seed, False)
        else:
            multi_fidelity = method.startswith("mfego NN-MF-EGO")
            estimator = method[method.index("(") + 1:-1]
            history = bl.run_mfego(problem, problem.doe(MF_DOE if multi_fidelity else SF_DOE,
                                                        seed),
                                   budget, seed, multi_fidelity,
                                   **{**ESTIMATORS, **SENSITIVITY}[estimator])
        result = asdict(history)
    except Exception as error:  # noqa: BLE001
        result = {"seed": seed, "error": repr(error)}
    result["method"] = method
    return result


def summarize_optimization(histories: list) -> tuple[pd.DataFrame, str]:
    rows = []
    for h in histories:
        if "error" in h:
            continue
        levels = np.array(h["levels"][1:])
        rows.append({"method": h["method"], "seed": h["seed"],
                     "best HF error": h["best_observed"][-1],
                     "recommendation error": h["recommendation_error"],
                     "iterations": len(levels),
                     "LF share": float(np.mean(levels == 1)) if len(levels) else np.nan,
                     "time / iteration (s)": float(np.mean(h["iteration_time_s"][1:]))
                     if len(levels) else np.nan})
    runs = pd.DataFrame(rows)
    table = runs.drop(columns="seed").groupby("method").agg(["median", "mean"]).round(3)
    wins = {}
    for variant in ("NN-MF-EGO", "SF-EGO"):
        wide = runs.pivot_table(index="seed", columns="method", values="recommendation error")
        for other in ("MLE", "MAP-l"):
            better = wide[f"mfego {variant} (MAP-all)"] < wide[f"mfego {variant} ({other})"]
            wins[f"{variant}: MAP-all < {other} (recommendation)"] = \
                f"{int(better.sum())}/{len(better)}"
    text = "\n".join([f"=== Optimization of Hartmann 6D at equal budget ({BUDGET}), f - f* ===",
                      table.to_string(), "", "=== Paired wins (same DOE) ===",
                      pd.Series(wins).to_string(), ""])
    return runs, text


# 3/3 ---------------------------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="mfego priors on all the hyperparameters.")
    parser.add_argument("--quick", action="store_true", help="3 seeds, fewer sizes")
    args = parser.parse_args()
    seeds = list(range(3 if args.quick else 10))
    n_hf_list = [5, 10, 20] if args.quick else [5, 10, 20, 40]
    n_large = [80] if args.quick else [80, 160, 300]
    RESULTS.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    texts = []

    # 1. accuracy, multi-fidelity sizes
    start = time.perf_counter()
    names = ([f"mfego MF ({e})" for e in {**ESTIMATORS, **SENSITIVITY}]
             + [f"mfego SF ({e})" for e in ESTIMATORS] + list(LIBRARIES))
    tasks = [(name, 2 * n, n, seed) for name in names for n in n_hf_list for seed in seeds]
    acc = pd.DataFrame(bl.run_parallel(accuracy_task, tasks))
    acc.to_csv(RESULTS / "accuracy.csv", index=False)
    texts.append(summarize_accuracy(acc, "Accuracy on 2000 test points (n_LF = 2 n_HF)"))
    print(texts[-1], f"{len(tasks)} fits in {time.perf_counter() - start:.0f} s\n", flush=True)

    # 2. accuracy with many high-fidelity points (single fidelity)
    start = time.perf_counter()
    names = [f"mfego SF ({e})" for e in ESTIMATORS] + ["scikit-learn GP (SF)",
                                                       "BoTorch SingleTaskGP (SF)"]
    tasks = [(name, 0, n, seed) for name in names for n in n_large for seed in seeds]
    large = pd.DataFrame(bl.run_parallel(accuracy_task, tasks))
    large.to_csv(RESULTS / "large_n.csv", index=False)
    texts.append(summarize_accuracy(large, "Many HF points, single fidelity"))
    print(texts[-1], f"{len(tasks)} fits in {time.perf_counter() - start:.0f} s\n", flush=True)

    # 3. optimization (+ 4. sensitivity of NN-MF-EGO)
    start = time.perf_counter()
    methods = ([f"mfego NN-MF-EGO ({e})" for e in {**ESTIMATORS, **SENSITIVITY}]
               + [f"mfego SF-EGO ({e})" for e in ESTIMATORS] + ["BoTorch qLogEI (SF)"])
    tasks = [(method, seed, BUDGET) for method in methods for seed in seeds]
    histories = bl.run_parallel(optimization_task, tasks)
    (RESULTS / "optimization_histories.json").write_text(json.dumps(histories, indent=1),
                                                         encoding="utf-8")
    for h in histories:
        if "error" in h:
            print("FAILED:", h["method"], h["seed"], h["error"])
    runs, text = summarize_optimization(histories)
    runs.to_csv(RESULTS / "optimization.csv", index=False)
    texts.append(text)
    print(text, f"{len(tasks)} runs in {time.perf_counter() - start:.0f} s", flush=True)

    (RESULTS / "summary.txt").write_text(
        f"seeds: {seeds}, n_HF: {n_hf_list}, many points: {n_large}, problem: {PROBLEM}, "
        f"MF DOE {MF_DOE}, SF DOE {SF_DOE}, budget {BUDGET}\n\n" + "\n".join(texts),
        encoding="utf-8")


if __name__ == "__main__":
    main()
