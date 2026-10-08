"""
Quick benchmark of the MAP estimation of mfego (InvGamma prior on the lengthscales, use_map=True)
against the maximum likelihood (use_map=False) and BoTorch, on the multi-fidelity Hartmann 6D
problem of benchmarks/bench_lib.py (Sacher et al. 2021, Eqs. 30-32).

1. Surrogate accuracy on 2000 test points (relative RMSE, NLPD, 95% coverage), n_LF = 2 n_HF,
   same DOE for every model of a given seed (paired MAP / MLE comparison).
2. Optimization at equal cost budget (same setup as benchmarks/hartmann_benchmark.ipynb).
   BoTorch MF-KG (~20 s per iteration, ~25 min per run) only with --mfkg.

Run from the repository root with the benchmark environment (requirements-benchmark.txt):
    .venv-benchmark\\Scripts\\python benchmarks\\map_hartmann\\run_map_benchmark.py [--quick] [--mfkg]
Results (CSV, JSON and text summary): benchmarks/map_hartmann/results/
"""
import argparse
import json
import os
import sys
import time
from dataclasses import asdict
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
from src.surrogate_models import LENGTHSCALE_BOUNDS, LENGTHSCALE_PRIOR  # noqa: E402

PROBLEM = {"ks": (1, np.inf), "delta": 0.05, "costs": (1.0, 10.0)}
MF_DOE, SF_DOE = [20, 10], [12]      # same initial cost: 20 x 1 + 10 x 10 = 12 x 10 = 120
BUDGET = 250                         # total cost of an optimization run (DOE included)
RESULTS = HERE / "results"

SURROGATES = {
    "mfego MF (MLE)": lambda seed, costs: bl.MfegoSurrogate(True, seed),
    "mfego MF (MAP)": lambda seed, costs: bl.MfegoSurrogate(True, seed, use_map=True),
    "mfego SF (MLE)": lambda seed, costs: bl.MfegoSurrogate(False, seed),
    "mfego SF (MAP)": lambda seed, costs: bl.MfegoSurrogate(False, seed, use_map=True),
    "BoTorch MF-GP (MF)": lambda seed, costs: bl.BotorchSurrogate(True, costs),
    "BoTorch SingleTaskGP (SF)": lambda seed, costs: bl.BotorchSurrogate(False, costs),
}

METHODS = {
    "mfego NN-MF-EGO (MLE)": lambda p, s, b: bl.run_mfego(p, p.doe(MF_DOE, s), b, s, True),
    "mfego NN-MF-EGO (MAP)": lambda p, s, b: bl.run_mfego(p, p.doe(MF_DOE, s), b, s, True,
                                                          use_map=True),
    "mfego SF-EGO (MLE)": lambda p, s, b: bl.run_mfego(p, p.doe(SF_DOE, s), b, s, False),
    "mfego SF-EGO (MAP)": lambda p, s, b: bl.run_mfego(p, p.doe(SF_DOE, s), b, s, False,
                                                       use_map=True),
    "BoTorch qLogEI (SF)": lambda p, s, b: bl.run_botorch(p, p.doe(SF_DOE, s), b, s, False),
    "BoTorch MF-KG": lambda p, s, b: bl.run_botorch(p, p.doe(MF_DOE, s), b, s, True),
}


# 1/2 ---------------------------------------------------------------------------------------------
# Surrogate accuracy
# -------------------------------------------------------------------------------------------------
def accuracy_task(args: tuple) -> dict:
    """(surrogate name, n_HF, seed) -> accuracy metrics (+ HF lengthscales for mfego)."""
    name, n_hf, seed = args
    problem = bl.HartmannMF(**PROBLEM)
    datasets = problem.doe([2 * n_hf, n_hf], seed=seed)
    x_test = qmc.LatinHypercube(d=problem.dim, seed=12345).random(2000)
    try:
        surrogate = SURROGATES[name](seed, problem.costs)
        result = bl.accuracy(surrogate, datasets, x_test, bl.hartmann6(x_test))
        if isinstance(surrogate, bl.MfegoSurrogate):
            result["hf_lengthscales"] = surrogate.model.gps[-1].kernel.lengthscale.tolist()
    except Exception as error:  # noqa: BLE001  (a failing model must not stop the benchmark)
        result = {"error": repr(error)}
    result.update({"model": name, "n_hf": n_hf, "seed": seed})
    return result


def summarize_accuracy(acc: pd.DataFrame) -> str:
    """Mean metrics per (n_HF, model), paired MAP / MLE wins and lengthscale diagnostic."""
    ok = acc[acc["error"].isna()] if "error" in acc else acc
    table = ok.groupby(["n_hf", "model"]).agg(
        rmse_rel=("rmse_rel", "mean"), nlpd_median=("nlpd", "median"),
        nlpd_mean=("nlpd", "mean"), coverage_95=("coverage_95", "mean"),
        fit_time_s=("fit_time_s", "mean")).round(3)

    wins = {}
    for variant in ("MF", "SF"):
        for metric, label in (("rmse_rel", "RMSE"), ("nlpd", "NLPD")):
            wide = ok.pivot_table(index=["n_hf", "seed"], columns="model", values=metric)
            better = wide[f"mfego {variant} (MAP)"] < wide[f"mfego {variant} (MLE)"]
            wins[f"{variant} {label}"] = better.groupby(level="n_hf").agg(
                lambda b: f"{int(b.sum())}/{len(b)}")

    mfego = ok[ok["model"].str.startswith("mfego")]
    lengthscales = mfego.assign(
        ls_median=mfego["hf_lengthscales"].apply(np.median),
        at_lower_bound=mfego["hf_lengthscales"].apply(
            lambda ls: np.mean(np.array(ls) <= 1.05 * LENGTHSCALE_BOUNDS[0])),
        at_upper_bound=mfego["hf_lengthscales"].apply(
            lambda ls: np.mean(np.array(ls) >= 0.95 * LENGTHSCALE_BOUNDS[1])),
    ).groupby(["n_hf", "model"])[["ls_median", "at_lower_bound", "at_upper_bound"]].mean()

    return "\n".join([
        "=== Surrogate accuracy on 2000 test points (mean over seeds; NLPD median and mean) ===",
        table.to_string(), "",
        "=== MAP better than MLE (same DOE, number of seeds) ===",
        pd.DataFrame(wins).to_string(), "",
        f"=== mfego HF-level lengthscales (prior InvGamma{LENGTHSCALE_PRIOR}, mode 0.5; "
        f"bounds {LENGTHSCALE_BOUNDS}): median, share at the bounds ===",
        lengthscales.round(3).to_string(), ""])


# 2/2 ---------------------------------------------------------------------------------------------
# Optimization at equal cost budget
# -------------------------------------------------------------------------------------------------
def optimization_task(args: tuple) -> dict:
    """(method, seed, budget) -> History as a dict."""
    method, seed, budget = args
    try:
        history = asdict(METHODS[method](bl.HartmannMF(**PROBLEM), seed, budget))
    except Exception as error:  # noqa: BLE001
        history = {"seed": seed, "error": repr(error)}
    history["method"] = method
    return history


def summarize_optimization(histories: list) -> tuple[pd.DataFrame, str]:
    """Final errors per run, median / mean per method and paired MAP / MLE wins."""
    rows = []
    for h in histories:
        if "error" in h:
            continue
        levels = np.array(h["levels"][1:])
        rows.append({
            "method": h["method"], "seed": h["seed"], "final cost": h["cost"][-1],
            "best HF error": h["best_observed"][-1],
            "recommendation error": h["recommendation_error"], "iterations": len(levels),
            "LF share": float(np.mean(levels == 1)) if len(levels) else np.nan,
            "time / iteration (s)": float(np.mean(h["iteration_time_s"][1:]))
                                    if len(levels) else np.nan})
    runs = pd.DataFrame(rows)
    table = runs.drop(columns="seed").groupby("method").agg(["median", "mean"]).round(4)

    wins = {}
    for variant in ("NN-MF-EGO", "SF-EGO"):
        for metric in ("best HF error", "recommendation error"):
            wide = runs.pivot_table(index="seed", columns="method", values=metric)
            better = wide[f"mfego {variant} (MAP)"] < wide[f"mfego {variant} (MLE)"]
            wins[f"{variant}: {metric}"] = f"{int(better.sum())}/{len(better)}"

    text = "\n".join([
        f"=== Optimization of Hartmann 6D at equal budget ({BUDGET}), f - f* ===",
        table.to_string(), "",
        "=== MAP better than MLE (same DOE, number of seeds) ===",
        pd.Series(wins).to_string(), ""])
    return runs, text


def main() -> None:
    parser = argparse.ArgumentParser(description="mfego MAP vs MLE vs BoTorch on Hartmann 6D.")
    parser.add_argument("--quick", action="store_true", help="3 seeds, n_HF in {5, 10, 20}")
    parser.add_argument("--seeds", type=int, default=None, help="number of seeds (default 10)")
    parser.add_argument("--mfkg", action="store_true", help="also run BoTorch MF-KG (slow)")
    args = parser.parse_args()
    seeds = list(range(args.seeds or (3 if args.quick else 10)))
    n_hf_list = [5, 10, 20] if args.quick else [5, 10, 20, 40]
    RESULTS.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220)

    start = time.perf_counter()
    tasks = [(name, n_hf, seed) for name in SURROGATES for n_hf in n_hf_list for seed in seeds]
    accuracy = pd.DataFrame(bl.run_parallel(accuracy_task, tasks))
    accuracy.to_csv(RESULTS / "accuracy.csv", index=False)
    accuracy_text = summarize_accuracy(accuracy)
    print(accuracy_text)
    print(f"{len(tasks)} fits in {time.perf_counter() - start:.0f} s\n", flush=True)

    start = time.perf_counter()
    methods = [m for m in METHODS if args.mfkg or m != "BoTorch MF-KG"]
    tasks = [(method, seed, BUDGET) for method in methods for seed in seeds]
    histories = bl.run_parallel(optimization_task, tasks)
    (RESULTS / "optimization_histories.json").write_text(json.dumps(histories, indent=1),
                                                         encoding="utf-8")
    for h in histories:
        if "error" in h:
            print("FAILED:", h["method"], h["seed"], h["error"])
    runs, optimization_text = summarize_optimization(histories)
    runs.to_csv(RESULTS / "optimization.csv", index=False)
    print(optimization_text)
    print(f"{len(tasks)} runs in {time.perf_counter() - start:.0f} s")

    (RESULTS / "summary.txt").write_text(
        f"seeds: {seeds}, n_HF: {n_hf_list}, problem: {PROBLEM}, MF DOE {MF_DOE}, "
        f"SF DOE {SF_DOE}\n\n{accuracy_text}\n{optimization_text}", encoding="utf-8")


if __name__ == "__main__":
    main()
