"""
[V2] Final benchmark of mfego v2 (centred rho with a normal prior, input normalization, kernel
chosen by run_selection.py, leave-one-out qualification) against mfego v1 and the other GP
libraries, on 3 problems (Hartmann 6D, airfoil NeuralFoil/NeuralFoil, airfoil NeuralFoil/XFOIL).

1. Nested designs (n_LF = 4 n_HF, 10 seeds): accuracy and calibration on the test set of
   mfego v2 / v1 CoKriging, GPy/emukit AR(1) (joint MLE, joint MAP), SMT MFK, BoTorch
   SingleTaskMultiFidelityGP, and high fidelity only (mfego v2, BoTorch SingleTaskGP).
2. Leave-one-out qualification of mfego: Q^2 and coverage from the LOO at the HF points
   compared with the same scores on the test set, fit by fit.
3. Data of the infilling (NN-MF-EGO v1 / v2, cost of the nested design with the same n_HF):
   final recursive model, exact conditioning of the same model (CoKriging.from_multifidelity),
   LOO qualification of it, recommendation.
4. Optimization of Hartmann 6D at equal budget (250, protocol of benchmarks/priors_all):
   NN-MF-EGO v1 / v2, SF-EGO v2, BoTorch qLogEI.

Run from the repository root with the benchmark environment (after run_selection.py):
    .venv-benchmark\\Scripts\\python benchmarks\\surrogate_v2\\run_benchmark.py [--quick]
Results: benchmarks/surrogate_v2/results/ (figures: make_figures.py, report: RAPPORT_V2.md).
"""
import argparse
import json
import time
from dataclasses import asdict

import numpy as np
import pandas as pd

import v2_lib as v2  # noqa: E402  (sets the paths and OMP_NUM_THREADS)
import bench_lib as bl  # noqa: E402
from src.cokriging import CoKriging  # noqa: E402
from src.surrogate_models import DEFAULT_PRIORS  # noqa: E402
from src.validation import prediction_scores  # noqa: E402

EGO_SIZES = {"hartmann": [20, 40], "airfoil": [10, 20], "airfoil_xfoil": [10, 20]}
EGO_DOE = {"hartmann": [20, 10], "airfoil": [20, 5], "airfoil_xfoil": [20, 5]}
OPT_BUDGET, MF_DOE, SF_DOE = 250, [20, 10], [12]


def models(seed: int, costs) -> dict:
    """Surrogates compared on the nested designs (common interface of bench_lib)."""
    return {
        "mfego v2 CoKriging": bl.CoKrigingSurrogate(seed, **v2.v2_options()),
        "mfego v1 CoKriging": bl.CoKrigingSurrogate(seed, **v2.v1_options()),
        "GPy/emukit AR1 (MLE)": bl.EmukitAR1Surrogate(seed),
        "GPy/emukit AR1 (MAP)": bl.EmukitAR1Surrogate(seed, priors=DEFAULT_PRIORS),
        "SMT MFK": bl.SmtSurrogate(multi_fidelity=True),
        "BoTorch MF-GP": bl.BotorchSurrogate(True, costs),
        "mfego v2 HF only": bl.MfegoSurrogate(False, seed, **v2.v2_options()),
        "BoTorch SingleTaskGP (HF only)": bl.BotorchSurrogate(False, costs),
    }


MODEL_NAMES = list(models(0, (1.0, 10.0)))


# 1/3 ---------------------------------------------------------------------------------------------
def nested_task(args: tuple) -> dict:
    """(problem, n_HF, seed, model name, datasets) -> test scores (+ LOO scores for mfego)."""
    problem, n_hf, seed, name, datasets = args
    row = {"problem": problem, "n_hf": n_hf, "seed": seed, "model": name}
    # failed simulations (XFOIL NaN) removed for every library, as a user would do (mfego
    # does it itself); removing a high-fidelity point keeps the design nested
    datasets = [(x[np.isfinite(y)], y[np.isfinite(y)]) for x, y in datasets]
    row["n_hf_valid"] = len(datasets[-1][1])
    try:
        x_test, y_test = v2.test_set(problem)
        surrogate = models(seed, (1.0, 10.0))[name]
        start = time.perf_counter()
        surrogate.fit(datasets)
        row["fit_time_s"] = time.perf_counter() - start
        mean, var = surrogate.predict(x_test)
        row.update(prediction_scores(y_test, mean, var))
        row.update(v2.calibration_curves(y_test, mean, var))
        row["rhos"] = json.dumps(surrogate.rhos)
        if isinstance(surrogate, bl.CoKrigingSurrogate):
            # [QUAL] leave-one-out at the HF points, without the test set: by location (all the
            # levels at the point left out, default) and by observation (plain LOO)
            for prefix, by_location in (("loo", True), ("loo_obs", False)):
                loo = prediction_scores(*surrogate.model.leave_one_out(by_location=by_location))
                row.update({f"{prefix}_{key}": value for key, value in loo.items()})
    except Exception as error:  # noqa: BLE001  (a failing model must not stop the benchmark)
        row["error"] = repr(error)
    return row


# 2/3 ---------------------------------------------------------------------------------------------
def infill_task(args: tuple) -> list:
    """(problem, n_HF, seed, version) -> scores of the surrogate built from an EGO run."""
    problem_name, n_hf, seed, version = args
    problem = v2.PROBLEMS[problem_name]()
    x_test, y_test = v2.test_set(problem_name)
    budget = (v2.LF_PER_HF * problem.costs[0] + problem.costs[1]) * n_hf
    options = v2.v2_options() if version == "v2" else v2.v1_options()
    base = {"problem": problem_name, "n_hf": n_hf, "seed": seed, "version": version,
            "budget": budget}
    try:
        history, model, data = bl.run_mfego(problem, problem.doe(EGO_DOE[problem_name], seed),
                                            budget, seed, True, return_model=True, **options)
        x_hf = data.get_training_data(2)[0]
        common = {**base, "cost": history.cost[-1],
                  "n_lf_data": len(data.get_training_data(1)[1]),
                  "n_hf_data": len(x_hf), "best_observed": history.best_observed[-1],
                  "recommendation": history.recommendation_error,
                  "rhos": json.dumps([float(r) for r in model.rhos])}
        mean, var, _ = model.predict_batch(x_test)
        rows = [{**common, "model": "EGO recursive", **prediction_scores(y_test, mean, var),
                 "std_at_hf": float(np.median(np.sqrt(model.predict_batch(x_hf)[1])))
                 / float(np.std(y_test))}]
        exact = CoKriging.from_multifidelity(model, data)
        mean, var = exact.predict_batch(x_test)
        loo = prediction_scores(*exact.leave_one_out())
        rows.append({**common, "model": "EGO exact conditioning",
                     **prediction_scores(y_test, mean, var),
                     **{f"loo_{key}": value for key, value in loo.items()},
                     "std_at_hf": float(np.median(np.sqrt(exact.predict_batch(x_hf)[1])))
                     / float(np.std(y_test))})
        return rows
    except Exception as error:  # noqa: BLE001
        return [{**base, "model": "EGO", "error": repr(error)}]


# 3/3 ---------------------------------------------------------------------------------------------
def optimization_task(args: tuple) -> dict:
    """(method, seed) -> History as a dict (Hartmann 6D, budget OPT_BUDGET)."""
    method, seed = args
    problem = bl.HartmannMF()
    try:
        if method == "BoTorch qLogEI (SF)":
            history = bl.run_botorch(problem, problem.doe(SF_DOE, seed), OPT_BUDGET, seed, False)
        else:
            options = v2.v1_options() if "v1" in method else v2.v2_options()
            multi_fidelity = method.startswith("NN-MF-EGO")
            history = bl.run_mfego(problem, problem.doe(MF_DOE if multi_fidelity else SF_DOE,
                                                        seed),
                                   OPT_BUDGET, seed, multi_fidelity, **options)
        result = asdict(history)
    except Exception as error:  # noqa: BLE001
        result = {"seed": seed, "error": repr(error)}
    result["method"] = method
    return result


def summary(nested: pd.DataFrame, infill: pd.DataFrame, runs: pd.DataFrame) -> str:
    cols = ["rmse_rel", "q2", "nlpd", "coverage_95", "z_rms", "fit_time_s"]
    lines = ["=== 1. Nested designs: medians over the seeds ==="]
    for problem, group in nested.groupby("problem"):
        lines += [f"--- {problem} ---",
                  group.groupby(["n_hf", "model"])[cols].median().round(3).to_string(), ""]
    loo = nested.dropna(subset=["loo_q2"])
    lines += ["=== 2. LOO (by location / by observation) vs test set (mfego CoKriging): "
              "medians ===",
              loo.groupby(["problem", "model", "n_hf"])[
                  ["q2", "loo_q2", "loo_obs_q2", "coverage_95", "loo_coverage_95",
                   "loo_obs_coverage_95", "nlpd", "loo_nlpd", "loo_obs_nlpd"]]
              .median().round(3).to_string(), ""]
    cols = ["cost", "n_lf_data", "n_hf_data", "rmse_rel", "q2", "nlpd", "coverage_95",
            "loo_q2", "std_at_hf", "recommendation"]
    lines += ["=== 3. Data of the infilling: medians ===",
              infill.groupby(["problem", "n_hf", "version", "model"])[cols].median().round(3)
              .to_string(), ""]
    lines += [f"=== 4. Optimization of Hartmann 6D (budget {OPT_BUDGET}): f - f* ===",
              runs.groupby("method")[["best HF error", "recommendation error"]]
              .agg(["median", "mean"]).round(3).to_string(), ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="mfego v2 against v1 and the competition.")
    parser.add_argument("--quick", action="store_true", help="3 seeds")
    args = parser.parse_args()
    seeds = v2.SEEDS[:3] if args.quick else v2.SEEDS
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    for problem in v2.PROBLEMS:
        v2.test_set(problem)
    designs = v2.designs(seeds)

    start = time.perf_counter()
    tasks = [(p, n, s, name, datasets) for (p, kind, n, s), datasets in designs.items()
             if kind == "nested" and s in seeds for name in MODEL_NAMES]
    nested = pd.DataFrame(bl.run_parallel(nested_task, tasks))
    nested.to_csv(v2.RESULTS / "nested.csv", index=False)
    print(f"{len(tasks)} nested fits in {time.perf_counter() - start:.0f} s", flush=True)

    start = time.perf_counter()
    tasks = [(p, n, s, version) for p in v2.PROBLEMS for n in EGO_SIZES[p] for s in seeds
             for version in ("v1", "v2")]
    infill = pd.DataFrame([row for rows in bl.run_parallel(infill_task, tasks) for row in rows])
    infill.to_csv(v2.RESULTS / "infill.csv", index=False)
    print(f"{len(tasks)} EGO runs in {time.perf_counter() - start:.0f} s", flush=True)

    start = time.perf_counter()
    methods = ["NN-MF-EGO v1", "NN-MF-EGO v2", "SF-EGO v2", "BoTorch qLogEI (SF)"]
    histories = bl.run_parallel(optimization_task, [(m, s) for m in methods for s in seeds])
    (v2.RESULTS / "optimization_histories.json").write_text(json.dumps(histories, indent=1),
                                                            encoding="utf-8")
    runs = pd.DataFrame([{"method": h["method"], "seed": h["seed"],
                          "best HF error": h["best_observed"][-1],
                          "recommendation error": h["recommendation_error"],
                          "LF share": float(np.mean(np.array(h["levels"][1:]) == 1))}
                         for h in histories if "error" not in h])
    runs.to_csv(v2.RESULTS / "optimization.csv", index=False)
    print(f"{len(histories)} optimization runs in {time.perf_counter() - start:.0f} s", flush=True)
    for frame in (nested, infill):
        if "error" in frame and frame["error"].notna().any():
            print("FAILED:", frame["error"].dropna().unique()[:5])
    failed = [h for h in histories if "error" in h]
    if failed:
        print("FAILED optimization runs:", [(h["method"], h["error"]) for h in failed][:5])

    ok = [f[f["error"].isna()] if "error" in f else f for f in (nested, infill)]
    text = summary(ok[0], ok[1], runs)
    print(text)
    (v2.RESULTS / "summary.txt").write_text(
        f"seeds: {seeds}, sizes: {v2.SIZES}, EGO sizes: {EGO_SIZES}, kernel v2: "
        f"{v2.V2_KERNEL.__name__}, priors v2: {DEFAULT_PRIORS}\n\n{text}", encoding="utf-8")


if __name__ == "__main__":
    main()
