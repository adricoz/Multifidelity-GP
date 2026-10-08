"""
Sensitivity of the mfego MAP estimation to the lengthscale prior, on the multi-fidelity Hartmann
6D problem (same protocol as run_map_benchmark.py, multi-fidelity model NN-MF-EGO only):

1. InvGamma(alpha, beta): location of the mode (alpha = 3, mode beta / 4 from 0.1 to 2) and
   strength (alpha from 1.5 to 10 at mode 0.5).
2. Prior family at mode 0.5 (InvGamma, Gamma, LogNormal) and the BoTorch default priors.
   (priors.PRIORS lists every configuration.)
3. Reference lengthscales of the problem: scikit-learn maximum likelihood, 300 points.
4. Diagnostic of the mfego MLE optimizer (pure NLL of the MLE fit vs at the MAP point).

Run from the repository root with the benchmark environment (after run_map_benchmark.py, whose
MLE / MAP results are the baselines of the figures):
    .venv-benchmark\\Scripts\\python benchmarks\\map_hartmann\\run_prior_sensitivity.py [--seeds N]
Results: benchmarks/map_hartmann/results/prior_*.csv and prior_reference_lengthscales.json
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
from priors import PRIORS, PriorMultifidelityModel, distribution, mode  # noqa: E402
from run_map_benchmark import BUDGET, MF_DOE, PROBLEM, RESULTS  # noqa: E402
from src.kernels import SquaredExponentialKernel  # noqa: E402
from src.surrogate_models import GaussianProcess  # noqa: E402

N_HF_LIST = [5, 10, 20, 40]
N_REFERENCE = 300
N_DIAGNOSTIC = [5, 10, 20, 40, 80, 160]


def accuracy_task(args: tuple) -> dict:
    """(prior name, n_HF, seed) -> accuracy metrics and fitted lengthscales (LF and HF levels)."""
    name, n_hf, seed = args
    problem = bl.HartmannMF(**PROBLEM)
    datasets = problem.doe([2 * n_hf, n_hf], seed=seed)
    x_test = qmc.LatinHypercube(d=problem.dim, seed=12345).random(2000)
    try:
        surrogate = bl.MfegoSurrogate(True, seed, model_class=PriorMultifidelityModel,
                                      prior=PRIORS[name][1])
        result = bl.accuracy(surrogate, datasets, x_test, bl.hartmann6(x_test))
        result["lf_lengthscales"] = surrogate.model.gps[0].kernel.lengthscale.tolist()
        result["hf_lengthscales"] = surrogate.model.gps[-1].kernel.lengthscale.tolist()
    except Exception as error:  # noqa: BLE001  (a failing fit must not stop the benchmark)
        result = {"error": repr(error)}
    result.update({"prior": name, "group": PRIORS[name][0], "n_hf": n_hf, "seed": seed})
    return result


def optimization_task(args: tuple) -> dict:
    """(prior name, seed) -> NN-MF-EGO History (as a dict) at the budget of the benchmark."""
    name, seed = args
    problem = bl.HartmannMF(**PROBLEM)
    try:
        history = asdict(bl.run_mfego(problem, problem.doe(MF_DOE, seed), BUDGET, seed, True,
                                      model_class=PriorMultifidelityModel,
                                      prior=PRIORS[name][1]))
    except Exception as error:  # noqa: BLE001
        history = {"seed": seed, "error": repr(error)}
    history.update({"method": "mfego NN-MF-EGO", "prior": name, "group": PRIORS[name][0]})
    return history


def reference_task(kind: str) -> dict:
    """Reference lengthscales of the problem with N_REFERENCE points, fitted by scikit-learn (ARD
    RBF + white noise, maximum likelihood with restarts): 'HF' (Hartmann), 'LF' (level 1) or
    'delta' (residual y_HF - rho y_LF, rho by least squares at the same points). The mfego MLE
    is not used: with 300 points it stops on a degenerate white-noise solution (see report)."""
    from sklearn.gaussian_process import GaussianProcessRegressor  # noqa: PLC0415
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel  # noqa
    problem = bl.HartmannMF(**PROBLEM)
    x = qmc.LatinHypercube(d=problem.dim, seed=777).random(N_REFERENCE)
    y_hf, y_lf = problem(x, 2), problem(x, 1)
    if kind == "HF":
        y = y_hf
    elif kind == "LF":
        y = y_lf
    else:
        design = np.column_stack([y_lf, np.ones_like(y_lf)])
        rho, _ = np.linalg.lstsq(design, y_hf, rcond=None)[0]
        y = y_hf - rho * y_lf
    kernel = ConstantKernel(1.0, (1e-3, 1e3))         * RBF(length_scale=np.full(problem.dim, 0.3), length_scale_bounds=(1e-2, 1e2))         + WhiteKernel(1e-6, (1e-10, 1e-1))
    model = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=3,
                                     random_state=0).fit(x, y)
    return {kind: model.kernel_.k1.k2.length_scale.tolist()}


def mle_diagnostic_task(args: tuple) -> dict:
    """(n, seed) -> pure NLL reached by the mfego MLE fit and pure NLL at the MAP point
    (single GP on n Hartmann points). NLL(MAP point) < NLL(MLE fit) means the MLE optimizer
    missed a better likelihood optimum; otherwise the MLE optimum is genuine."""
    n, seed = args
    problem = bl.HartmannMF(**PROBLEM)
    x, y = problem.doe([n], seed=seed)[0]
    x_test = qmc.LatinHypercube(d=problem.dim, seed=12345).random(2000)
    y_test = bl.hartmann6(x_test)
    result = {"n": n, "seed": seed}
    for label, use_map in (("mle", False), ("map", True)):
        gp = GaussianProcess(SquaredExponentialKernel(), seed=seed, use_map=use_map)
        gp.fit(x, y)
        y_n = (y - gp.y_mean) / gp.y_std
        result[f"nll_{label}"] = gp.negative_log_likelihood(
            np.log(np.concatenate([gp.kernel.get_params(), [gp.noise]])), y_n)
        mean, _ = gp.predict_batch(x_test)
        result[f"rmse_rel_{label}"] = float(np.sqrt(np.mean((mean - y_test) ** 2))
                                            / np.std(y_test))
        result[f"ls_median_{label}"] = float(np.median(gp.kernel.lengthscale))
    return result


def prior_table() -> pd.DataFrame:
    """Mode, quantiles and tail masses of every prior (l in [0, 1] input units)."""
    rows = []
    for name, (group, prior) in PRIORS.items():
        dist = distribution(prior)
        rows.append({"prior": name, "group": group, "mode": mode(prior),
                     "q05": dist.ppf(0.05), "median": dist.median(), "q95": dist.ppf(0.95),
                     "P(l < 0.1)": dist.cdf(0.1), "P(l > 10)": dist.sf(10.0)})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prior sensitivity of the mfego MAP.")
    parser.add_argument("--seeds", type=int, default=10, help="number of seeds (default 10)")
    args = parser.parse_args()
    seeds = list(range(args.seeds))
    RESULTS.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220)

    table = prior_table()
    table.to_csv(RESULTS / "prior_table.csv", index=False)
    print(table.round(3).to_string(index=False), "\n")

    start = time.perf_counter()
    reference = {}
    for part in bl.run_parallel(reference_task, ["HF", "LF", "delta"]):
        reference.update(part)
    (RESULTS / "prior_reference_lengthscales.json").write_text(json.dumps(reference, indent=1),
                                                               encoding="utf-8")
    print(f"Reference lengthscales (scikit-learn MLE, {N_REFERENCE} points):")
    for key, value in reference.items():
        print(f"  {key}: {np.round(value, 3).tolist()}")
    print(f"({time.perf_counter() - start:.0f} s)\n", flush=True)

    start = time.perf_counter()
    diagnostic = pd.DataFrame(bl.run_parallel(mle_diagnostic_task,
                                              [(n, seed) for n in N_DIAGNOSTIC for seed in seeds]))
    diagnostic.to_csv(RESULTS / "mle_diagnostic.csv", index=False)
    diagnostic["mle_missed"] = diagnostic["nll_map"] < diagnostic["nll_mle"] - 1e-6
    print("mfego MLE optimizer diagnostic (single GP on Hartmann, mean over seeds):")
    print(diagnostic.groupby("n")[["mle_missed", "rmse_rel_mle", "rmse_rel_map",
                                   "ls_median_mle", "ls_median_map"]].mean().round(3).to_string())
    print(f"({time.perf_counter() - start:.0f} s)\n", flush=True)

    start = time.perf_counter()
    tasks = [(name, n_hf, seed) for name in PRIORS for n_hf in N_HF_LIST for seed in seeds]
    accuracy = pd.DataFrame(bl.run_parallel(accuracy_task, tasks))
    accuracy.to_csv(RESULTS / "prior_accuracy.csv", index=False)
    ok = accuracy[accuracy["error"].isna()] if "error" in accuracy else accuracy
    print(ok.pivot_table(index="prior", columns="n_hf", values=["rmse_rel", "nlpd"],
                         aggfunc="median").round(3).to_string())
    print(f"{len(tasks)} fits in {time.perf_counter() - start:.0f} s\n", flush=True)

    start = time.perf_counter()
    histories = bl.run_parallel(optimization_task, [(name, seed) for name in PRIORS
                                                    for seed in seeds])
    (RESULTS / "prior_optimization_histories.json").write_text(json.dumps(histories, indent=1),
                                                               encoding="utf-8")
    runs = pd.DataFrame([{"prior": h["prior"], "seed": h["seed"],
                          "best HF error": h["best_observed"][-1],
                          "recommendation error": h["recommendation_error"],
                          "LF share": float(np.mean(np.array(h["levels"][1:]) == 1))}
                         for h in histories if "error" not in h])
    runs.to_csv(RESULTS / "prior_optimization.csv", index=False)
    for h in histories:
        if "error" in h:
            print("FAILED:", h["prior"], h["seed"], h["error"])
    print(runs.drop(columns="seed").groupby("prior").median().round(3).to_string())
    print(f"{len(histories)} runs in {time.perf_counter() - start:.0f} s")


if __name__ == "__main__":
    main()
