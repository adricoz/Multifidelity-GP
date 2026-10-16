"""
[CK] Benchmark of the co-kriging conditioned on all the data (src/cokriging.CoKriging, nested
designs) on two problems: multi-fidelity Hartmann 6D (bench_lib.HartmannMF) and a small NACA
4-digit airfoil problem (airfoil_problem.AirfoilMF, NeuralFoil xxsmall / xxxlarge).

1. Nested Latin hypercube designs (n_LF = 4 n_HF, 10 seeds), same data for every model:
   mfego recursive model (current MultifidelityModel), mfego CoKriging (MAP and MLE), GPy/emukit
   AR(1) co-kriging (joint MLE and joint MAP with the same priors), SMT MFK, mfego HF only.
   Accuracy (relative RMSE, NLPD), calibration of the variance (coverage of the 50 to 99 %
   intervals, histogram of the standardized errors (y - mean) / std, ideally N(0, 1)).
2. Exactness: the hyperparameters fitted by emukit are put in an mfego CoKriging (no fit):
   the predictions must be the same (same co-kriging, [LG] Eq. 3.11).
3. Data of the CURRENT mfego infilling (NN-MF-EGO, same cost as the nested design): accuracy of
   the final model of the EGO (recursive formula), of the exact conditioning of the same data
   with the same hyperparameters (joint matrix V, valid for any design), and of a CoKriging
   fitted after evaluating the low fidelity at the high-fidelity points (cost counted). The
   predicted standard deviation at the high-fidelity data points (where the value is known)
   shows the error of the recursive variance on non-nested data.

Run from the repository root with the benchmark environment:
    .venv-benchmark\\Scripts\\python benchmarks\\cokriging\\run_cokriging_benchmark.py [--quick]
Results: benchmarks/cokriging/results/ (figures: make_figures.py, report: RAPPORT_COKRIGING.md).
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
from scipy.stats import norm, qmc  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import bench_lib as bl  # noqa: E402
from airfoil_problem import AirfoilMF  # noqa: E402
from src.cokriging import CoKriging, missing_points_for_nesting  # noqa: E402
from src.kernels import SquaredExponentialKernel  # noqa: E402
from src.surrogate_models import DEFAULT_PRIORS  # noqa: E402

# options of mfego v1, the state described by RAPPORT_COKRIGING.md (CoKriging defaults changed
# in v2 and v3): squared exponential kernel, rho not centred, no prior on rho
V1 = {"kernel_class": SquaredExponentialKernel, "rho_centering": False}

RESULTS = HERE / "results"
LF_PER_HF = 4                                   # n_LF = 4 n_HF in the nested designs
SIZES = {"hartmann": [10, 20, 40], "airfoil": [5, 10, 20]}
EGO_SIZES = {"hartmann": [20, 40], "airfoil": [10, 20]}    # budget = cost of these designs
EGO_DOE = {"hartmann": [20, 10], "airfoil": [20, 5]}       # initial design of the EGO
N_TEST = {"hartmann": 2000, "airfoil": 1000}
NOMINAL = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
Z_EDGES = np.linspace(-5.0, 5.0, 41)

MODELS = {
    "mfego recursive (MAP)": lambda seed: bl.MfegoSurrogate(True, seed, use_map=True),
    "mfego CoKriging (MAP)": lambda seed: bl.CoKrigingSurrogate(seed, **V1),
    "mfego CoKriging (MLE)": lambda seed: bl.CoKrigingSurrogate(seed, use_map=False, **V1),
    "GPy/emukit AR1 (MLE)": lambda seed: bl.EmukitAR1Surrogate(seed),
    "GPy/emukit AR1 (MAP)": lambda seed: bl.EmukitAR1Surrogate(seed, priors=DEFAULT_PRIORS),
    "SMT MFK": lambda seed: bl.SmtSurrogate(multi_fidelity=True),
    "mfego HF only (MAP)": lambda seed: bl.MfegoSurrogate(False, seed, use_map=True),
}


def make_problem(name: str):
    return bl.HartmannMF() if name == "hartmann" else AirfoilMF()


def test_set(name: str) -> tuple[np.ndarray, np.ndarray]:
    """Latin hypercube test points and their high-fidelity values (cached in results/)."""
    path = RESULTS / f"test_set_{name}.npz"
    if path.exists():
        data = np.load(path)
        return data["x"], data["y"]
    problem = make_problem(name)
    x = qmc.LatinHypercube(d=problem.dim, seed=12345).random(N_TEST[name])
    y = problem(x, problem.num_levels)
    np.savez(path, x=x, y=y)
    return x, y


def scores(mean: np.ndarray, var: np.ndarray, y_test: np.ndarray) -> dict:
    """Accuracy and calibration of a prediction on the test set."""
    std = np.sqrt(np.maximum(var, 1e-12))
    z = (y_test - mean) / std
    top = y_test <= np.quantile(y_test, 0.1)             # region of the optimum
    out = {"rmse_rel": float(np.sqrt(np.mean((mean - y_test) ** 2)) / np.std(y_test)),
           "rmse_rel_top10": float(np.sqrt(np.mean((mean - y_test)[top] ** 2)) / np.std(y_test)),
           "nlpd": float(np.mean(0.5 * np.log(2 * np.pi * std ** 2) + 0.5 * z ** 2)),
           "coverage": json.dumps([float(np.mean(np.abs(z) <= norm.ppf(0.5 + p / 2)))
                                   for p in NOMINAL]),
           "z_hist": json.dumps(np.histogram(np.clip(z, -5, 5), Z_EDGES)[0].tolist()),
           "z_std": float(np.std(z))}
    for p in (0.5, 0.9, 0.95, 0.99):
        out[f"coverage_{int(100 * p)}"] = float(np.mean(np.abs(z) <= norm.ppf(0.5 + p / 2)))
    return out


# 1/2 ---------------------------------------------------------------------------------------------
# Nested designs: accuracy, calibration and exactness
# -------------------------------------------------------------------------------------------------
def nested_task(args: tuple) -> dict:
    """(problem, model, n_HF, seed) -> scores on the test set (+ exactness check for emukit)."""
    problem_name, model_name, n_hf, seed = args
    problem = make_problem(problem_name)
    x_test, y_test = test_set(problem_name)
    datasets = bl.nested_doe(problem, [LF_PER_HF * n_hf, n_hf], seed=seed)
    row = {"problem": problem_name, "model": model_name, "n_hf": n_hf, "seed": seed}
    try:
        model = MODELS[model_name](seed)
        start = time.perf_counter()
        model.fit(datasets)
        row["fit_time_s"] = time.perf_counter() - start
        mean, var = model.predict(x_test)
        row.update(scores(mean, var, y_test))
        row["rhos"] = json.dumps(model.rhos)
        if isinstance(model, bl.EmukitAR1Surrogate):
            # exactness: emukit hyperparameters in an mfego CoKriging, latent predictions
            cokriging = model.to_cokriging()
            mean_ck, var_ck = cokriging.predict_batch(x_test, include_noise=False)
            x_aug = np.column_stack([x_test, np.ones(len(x_test))])
            mean_gpy, var_gpy = model.model.gpy_model.predict_noiseless(x_aug)
            mean_gpy = model.y_means[-1] + model.y_std * mean_gpy[:, 0]
            var_gpy = model.y_std ** 2 * var_gpy[:, 0]
            row["exact_mean"] = float(np.max(np.abs(mean_ck - mean_gpy)) / np.std(y_test))
            row["exact_var"] = float(np.max(np.abs(var_ck - var_gpy) / var_gpy))
            row["cond_v"] = float(np.linalg.cond(cokriging.l_chol) ** 2)
    except Exception as error:  # noqa: BLE001  (a failing model must not stop the benchmark)
        row["error"] = repr(error)
    return row


# 2/2 ---------------------------------------------------------------------------------------------
# Data of the current mfego infilling (NN-MF-EGO)
# -------------------------------------------------------------------------------------------------
def exact_conditioning(model, data) -> CoKriging:
    """CoKriging with the hyperparameters, normalization and rhos of a fitted recursive
    MultifidelityModel, conditioned on all the data (no fit, so no nesting needed): exact
    posterior of the same AR(1) model. Now in the library: CoKriging.from_multifidelity (same
    kernel class and options as the recursive model)."""
    return CoKriging.from_multifidelity(model, data)


def std_at_hf_points(var_at_points: np.ndarray, y_test: np.ndarray) -> float:
    """Median predicted standard deviation at the high-fidelity data points / std(y)."""
    return float(np.median(np.sqrt(np.maximum(var_at_points, 0.0))) / np.std(y_test))


def infill_task(args: tuple) -> list:
    """(problem, n_HF of the equivalent nested design, seed) -> scores of the EGO model, of the
    exact conditioning with its hyperparameters and of a CoKriging fitted after the
    low-fidelity completion of the EGO data."""
    problem_name, n_hf, seed = args
    problem = make_problem(problem_name)
    x_test, y_test = test_set(problem_name)
    budget = (LF_PER_HF * problem.costs[0] + problem.costs[1]) * n_hf
    base = {"problem": problem_name, "n_hf": n_hf, "seed": seed, "budget": budget}
    try:
        start = time.perf_counter()
        history, model, data = bl.run_mfego(problem, problem.doe(EGO_DOE[problem_name], seed),
                                            budget, seed, True, return_model=True, use_map=True)
        ego_time = time.perf_counter() - start
        n_lf, n_hf_ego = (len(data.get_training_data(l)[1]) for l in (1, 2))
        x_hf = data.get_training_data(2)[0]
        # points of the EGO run (before the completion), for the figures
        points = json.dumps({l: data.get_training_data(l)[0].tolist() for l in (1, 2)})
        mean, var, _ = model.predict_batch(x_test)
        rows = [{**base, "model": "EGO data + mfego recursive (MAP)", "cost": history.cost[-1],
                 "n_lf_data": n_lf, "n_hf_data": n_hf_ego, "fit_time_s": ego_time,
                 "recommendation_error": history.recommendation_error,
                 "rhos": json.dumps([float(r) for r in model.rhos]),
                 "std_at_hf_points": std_at_hf_points(model.predict_batch(x_hf)[1], y_test),
                 "points": points if seed == 0 else None, **scores(mean, var, y_test)}]

        # same hyperparameters, exact conditioning on all the (non-nested) data
        exact = exact_conditioning(model, data)
        mean, var = exact.predict_batch(x_test)
        rows.append({**base, "model": "EGO data + exact conditioning (same hyperparameters)",
                     "cost": history.cost[-1], "n_lf_data": n_lf, "n_hf_data": n_hf_ego,
                     "rhos": rows[0]["rhos"],
                     "std_at_hf_points": std_at_hf_points(exact.predict_batch(x_hf)[1], y_test),
                     **scores(mean, var, y_test)})

        # low-fidelity completion: evaluate level 1 at the high-fidelity points it misses
        missing = missing_points_for_nesting(data, 2)[1]
        for x in missing:
            data.add_observation(1, x, float(problem(x, 1)[0]), {})
        cokriging = CoKriging(2, seed=seed, **V1)
        start = time.perf_counter()
        cokriging.fit(data)
        mean, var = cokriging.predict_batch(x_test)
        rows.append({**base, "model": "EGO data + LF completion + mfego CoKriging (MAP)",
                     "cost": history.cost[-1] + len(missing) * problem.costs[0],
                     "n_lf_data": n_lf + len(missing), "n_hf_data": n_hf_ego,
                     "fit_time_s": time.perf_counter() - start,
                     "rhos": json.dumps([float(r) for r in cokriging.rhos]),
                     "std_at_hf_points": std_at_hf_points(cokriging.predict_batch(x_hf)[1],
                                                          y_test),
                     **scores(mean, var, y_test)})
        return rows
    except Exception as error:  # noqa: BLE001
        return [{**base, "model": "EGO", "error": repr(error)}]


def summarize(nested: pd.DataFrame, infill: pd.DataFrame) -> str:
    cols = ["rmse_rel", "rmse_rel_top10", "nlpd", "coverage_50", "coverage_90", "coverage_95",
            "coverage_99", "z_std", "fit_time_s"]
    ok = nested[nested["error"].isna()] if "error" in nested else nested
    lines = ["=== Nested designs: medians over the seeds ==="]
    for problem, group in ok.groupby("problem"):
        lines += [f"--- {problem} (n_LF = {LF_PER_HF} n_HF) ---",
                  group.groupby(["n_hf", "model"])[cols].median().round(3).to_string(), ""]
    exact = ok.dropna(subset=["exact_mean"]) if "exact_mean" in ok else ok.iloc[:0]
    if len(exact):
        lines += ["=== Exactness: emukit hyperparameters in mfego CoKriging (max over the test "
                  "points; mean / std(y), relative latent variance) ===",
                  exact.groupby(["problem", "model"])[["exact_mean", "exact_var", "cond_v"]]
                  .max().to_string(), ""]
    ok = infill[infill["error"].isna()] if "error" in infill else infill
    if len(ok):
        cols = ["cost", "n_lf_data", "n_hf_data", "rmse_rel", "rmse_rel_top10", "nlpd",
                "coverage_95", "z_std", "std_at_hf_points"]
        lines += ["=== Data of the current infilling (NN-MF-EGO, budget = cost of the nested "
                  "design with the same n_HF): medians ===",
                  ok.groupby(["problem", "n_hf", "model"])[cols].median().round(3).to_string(),
                  ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Co-kriging on nested designs.")
    parser.add_argument("--quick", action="store_true", help="3 seeds, smaller sizes")
    args = parser.parse_args()
    seeds = list(range(3 if args.quick else 10))
    sizes = {p: s[:2] for p, s in SIZES.items()} if args.quick else SIZES
    ego_sizes = {p: s[:1] for p, s in EGO_SIZES.items()} if args.quick else EGO_SIZES
    RESULTS.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    for problem in SIZES:          # test sets computed once, before the worker processes
        test_set(problem)

    start = time.perf_counter()
    tasks = [(p, m, n, s) for p in sizes for m in MODELS for n in sizes[p] for s in seeds]
    nested = pd.DataFrame(bl.run_parallel(nested_task, tasks))
    nested.to_csv(RESULTS / "nested.csv", index=False)
    print(f"{len(tasks)} nested fits in {time.perf_counter() - start:.0f} s", flush=True)

    start = time.perf_counter()
    tasks = [(p, n, s) for p in ego_sizes for n in ego_sizes[p] for s in seeds]
    infill = pd.DataFrame([row for rows in bl.run_parallel(infill_task, tasks) for row in rows])
    infill.to_csv(RESULTS / "infill.csv", index=False)
    print(f"{len(tasks)} EGO runs in {time.perf_counter() - start:.0f} s", flush=True)

    text = summarize(nested, infill)
    print(text)
    (RESULTS / "summary.txt").write_text(
        f"seeds: {seeds}, sizes: {sizes}, EGO sizes: {ego_sizes}, EGO initial designs: "
        f"{EGO_DOE}, n_LF = {LF_PER_HF} n_HF\n\n{text}", encoding="utf-8")


if __name__ == "__main__":
    main()
