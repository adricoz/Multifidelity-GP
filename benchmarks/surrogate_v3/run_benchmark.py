"""
[V3] Benchmark of mfego v1 / v2 / v3 (v3_lib.VARIANTS: rho learned with a lognormal prior in v3)
against the other GP libraries, on the 3 problems of benchmarks/surrogate_v2 (Hartmann 6D,
airfoil NeuralFoil/NeuralFoil, airfoil NeuralFoil/XFOIL; designs and test sets of v2_lib).

A.  Nested designs (n_LF = 4 n_HF, 3 sizes, 10 seeds): accuracy and calibration on the test set
    of CoKriging in every variant, GPy/emukit AR(1) (joint MLE, joint MAP), SMT MFK, BoTorch
    SingleTaskMultiFidelityGP, high fidelity only (mfego, BoTorch SingleTaskGP).
B1. Optimization of Hartmann 6D at equal budget (250, protocol of benchmarks/priors_all):
    NN-MF-EGO in every variant of v3_lib.EGO_VARIANTS, SF-EGO (Matern), BoTorch qLogEI.
B2. Data of the infilling (NN-MF-EGO until the cost of the nested design with the same n_HF, 3
    problems): recommendation, low-fidelity share, final rho, accuracy of the final model.

Decision rules, written BEFORE the computation (decide_surrogate / decide_ego below):
1. Default of CoKriging (surrogate of nested designs, A). Candidates: every variant but v1.
   - score: mean over the (problem, n_HF) cells of median NLPD(variant) - median NLPD(v1);
   - eligible if, for EVERY problem: (i) mean over the sizes of
     median RMSE(variant) / median RMSE(v1) <= 1.05, and (ii) rho at a bound of its search
     interval in at most 10 % of the fits;
   - choice: the eligible variant with the lowest score; none eligible: v2 N(1, 2²) stays.
2. Default of MultifidelityModel and of the foil pipeline (NN-MF-EGO). v1 stays unless a variant
   - has a lower recommendation error than v1 on Hartmann (B1) on at least 6 seeds out of 10 and
     a lower median, and
   - is not worse than v1 on the airfoils (B2): median recommended Cd at most 0.2 drag count
     above the one of v1, for each airfoil problem and size.
   Several variants pass: the lowest median recommendation error on Hartmann.

Run from the repository root with the benchmark environment:
    .venv-benchmark\\Scripts\\python benchmarks\\surrogate_v3\\run_benchmark.py [--quick]
Results: benchmarks/surrogate_v3/results/ (figures: make_figures.py, report: RAPPORT_V3.md).
"""
import argparse
import json
import time
from dataclasses import asdict

import numpy as np
import pandas as pd

import v3_lib as v3  # noqa: E402  (sets the paths)
import v2_lib as v2  # noqa: E402
import bench_lib as bl  # noqa: E402
from src.surrogate_models import DEFAULT_PRIORS  # noqa: E402
from src.validation import prediction_scores  # noqa: E402

EGO_SIZES = {"hartmann": [20, 40], "airfoil": [10, 20], "airfoil_xfoil": [10, 20]}
EGO_DOE = {"hartmann": [20, 10], "airfoil": [20, 5], "airfoil_xfoil": [20, 5]}
OPT_BUDGET, MF_DOE, SF_DOE = 250, [20, 10], [12]
AIRFOIL_TOLERANCE = 0.2     # drag counts (rule 2)
MAX_AT_BOUND = 0.10         # share of the fits (rule 1)
MAX_RMSE_RATIO = 1.05       # rule 1


def models(seed: int, costs) -> dict:
    """Surrogates compared on the nested designs (common interface of bench_lib)."""
    mfego = {f"mfego {name}": bl.CoKrigingSurrogate(seed, **options)
             for name, options in v3.VARIANTS.items()}
    return {**mfego,
            "GPy/emukit AR1 (MLE)": bl.EmukitAR1Surrogate(seed),
            "GPy/emukit AR1 (MAP)": bl.EmukitAR1Surrogate(seed, priors=DEFAULT_PRIORS),
            "SMT MFK": bl.SmtSurrogate(multi_fidelity=True),
            "BoTorch MF-GP": bl.BotorchSurrogate(True, costs),
            "mfego HF only": bl.MfegoSurrogate(False, seed, **v2.v2_options()),
            "BoTorch SingleTaskGP (HF only)": bl.BotorchSurrogate(False, costs)}


MODEL_NAMES = list(models(0, (1.0, 10.0)))


# A -----------------------------------------------------------------------------------------------
def nested_task(args: tuple) -> dict:
    """(problem, n_HF, seed, model name, datasets) -> test scores and rho."""
    problem, n_hf, seed, name, datasets = args
    row = {"problem": problem, "n_hf": n_hf, "seed": seed, "model": name}
    # failed simulations (XFOIL NaN) removed for every library (the design stays nested)
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
        if surrogate.rhos:
            row["rho"] = surrogate.rhos[0]
    except Exception as error:  # noqa: BLE001  (a failing model must not stop the benchmark)
        row["error"] = repr(error)
    return row


# B2 ----------------------------------------------------------------------------------------------
def infill_task(args: tuple) -> dict:
    """(problem, n_HF, seed, variant) -> recommendation and model of an NN-MF-EGO run."""
    problem_name, n_hf, seed, variant = args
    problem = v2.PROBLEMS[problem_name]()
    x_test, y_test = v2.test_set(problem_name)
    budget = (v2.LF_PER_HF * problem.costs[0] + problem.costs[1]) * n_hf
    row = {"problem": problem_name, "n_hf": n_hf, "seed": seed, "variant": variant,
           "budget": budget}
    try:
        history, model, data = bl.run_mfego(problem, problem.doe(EGO_DOE[problem_name], seed),
                                            budget, seed, True, return_model=True,
                                            **v3.EGO_VARIANTS[variant])
        levels = np.array(history.levels[1:])
        mean, var, _ = model.predict_batch(x_test)
        row.update({"cost": history.cost[-1], "n_lf_data": len(data.get_training_data(1)[1]),
                    "n_hf_data": len(data.get_training_data(2)[1]),
                    "lf_share": float(np.mean(levels == 1)) if len(levels) else 0.0,
                    "best_observed": history.best_observed[-1],
                    "recommendation": history.recommendation_error,
                    "rho": float(model.rhos[0]), **prediction_scores(y_test, mean, var)})
    except Exception as error:  # noqa: BLE001
        row["error"] = repr(error)
    return row


# B1 ----------------------------------------------------------------------------------------------
def optimization_task(args: tuple) -> dict:
    """(method, seed) -> History as a dict (Hartmann 6D, budget OPT_BUDGET)."""
    method, seed = args
    problem = bl.HartmannMF()
    try:
        if method == "BoTorch qLogEI (SF)":
            history = bl.run_botorch(problem, problem.doe(SF_DOE, seed), OPT_BUDGET, seed, False)
        elif method == "SF-EGO (Matern)":
            history = bl.run_mfego(problem, problem.doe(SF_DOE, seed), OPT_BUDGET, seed, False,
                                   **v2.v2_options())
        else:
            variant = method.removeprefix("NN-MF-EGO ")
            history = bl.run_mfego(problem, problem.doe(MF_DOE, seed), OPT_BUDGET, seed, True,
                                   **v3.EGO_VARIANTS[variant])
        result = asdict(history)
    except Exception as error:  # noqa: BLE001
        result = {"seed": seed, "error": repr(error)}
    result["method"] = method
    return result


# Decisions ---------------------------------------------------------------------------------------
def decide_surrogate(nested: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Rule 1: (table of the candidates, chosen variant)."""
    medians = nested.groupby(["problem", "n_hf", "model"])[["nlpd", "rmse_rel"]].median()
    reference = medians.xs(f"mfego {v3.REFERENCE}", level="model")
    rows = []
    for name in v3.VARIANTS:
        if name == v3.REFERENCE:
            continue
        cells = medians.xs(f"mfego {name}", level="model")
        row = {"variant": name, "score (mean dNLPD)": float((cells.nlpd - reference.nlpd).mean())}
        fits = nested[nested.model == f"mfego {name}"]
        for problem in v2.PROBLEMS:
            ratio = (cells.rmse_rel / reference.rmse_rel).xs(problem, level="problem").mean()
            share = fits[fits.problem == problem].rho.apply(
                lambda r, n=name: v3.at_bound(r, v3.BOUNDS[n])).mean()
            row[f"RMSE ratio {problem}"] = float(ratio)
            row[f"at bound {problem}"] = float(share)
        row["eligible"] = all(row[f"RMSE ratio {p}"] <= MAX_RMSE_RATIO
                              and row[f"at bound {p}"] <= MAX_AT_BOUND for p in v2.PROBLEMS)
        rows.append(row)
    table = pd.DataFrame(rows).set_index("variant").sort_values("score (mean dNLPD)")
    eligible = table[table.eligible]
    return table, (eligible.index[0] if len(eligible) else v3.CURRENT_DEFAULT)


def decide_ego(runs: pd.DataFrame, infill: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Rule 2: (table of the candidates, chosen variant)."""
    errors = runs.pivot(index="seed", columns="method", values="recommendation error")
    reference = errors[f"NN-MF-EGO {v3.REFERENCE}"]
    airfoils = infill[infill.problem != "hartmann"].groupby(
        ["problem", "n_hf", "variant"]).recommendation.median()
    rows = []
    for name in v3.EGO_VARIANTS:
        if name == v3.REFERENCE:
            continue
        column = errors[f"NN-MF-EGO {name}"]
        excess = (airfoils.xs(name, level="variant")
                  - airfoils.xs(v3.REFERENCE, level="variant"))
        row = {"variant": name, "wins vs v1 (Hartmann)": int((column < reference).sum()),
               "median (Hartmann)": float(column.median()),
               "median v1 (Hartmann)": float(reference.median()),
               "worst airfoil excess (counts)": float(excess.max())}
        row["passes"] = (row["wins vs v1 (Hartmann)"] >= 6
                         and row["median (Hartmann)"] < row["median v1 (Hartmann)"]
                         and row["worst airfoil excess (counts)"] <= AIRFOIL_TOLERANCE)
        rows.append(row)
    table = pd.DataFrame(rows).set_index("variant").sort_values("median (Hartmann)")
    passing = table[table.passes]
    return table, (passing.index[0] if len(passing) else v3.REFERENCE)


def summary(nested: pd.DataFrame, infill: pd.DataFrame, runs: pd.DataFrame) -> str:
    cols = ["rmse_rel", "q2", "nlpd", "coverage_95", "z_rms", "rho", "fit_time_s"]
    lines = ["=== A. Nested designs: medians over the seeds ==="]
    for problem, group in nested.groupby("problem"):
        lines += [f"--- {problem} ---",
                  group.groupby(["n_hf", "model"])[cols].median().round(3).to_string(), ""]
    table, chosen = decide_surrogate(nested)
    lines += ["=== Rule 1: default of CoKriging ===", table.round(3).to_string(),
              f"CHOSEN (CoKriging): {chosen}", ""]
    lines += [f"=== B1. Optimization of Hartmann 6D (budget {OPT_BUDGET}): f - f* ===",
              runs.groupby("method")[["best HF error", "recommendation error", "LF share",
                                      "final rho"]].agg(["median", "mean"]).round(3).to_string(),
              ""]
    cols = ["cost", "n_lf_data", "n_hf_data", "lf_share", "recommendation", "rho", "q2", "nlpd"]
    lines += ["=== B2. Data of the infilling: medians ===",
              infill.groupby(["problem", "n_hf", "variant"])[cols].median().round(3).to_string(),
              ""]
    table, chosen = decide_ego(runs, infill)
    lines += ["=== Rule 2: default of MultifidelityModel / pipeline (EGO) ===",
              table.round(3).to_string(), f"CHOSEN (EGO): {chosen}", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="mfego v1 / v2 / v3 and the competition.")
    parser.add_argument("--quick", action="store_true", help="3 seeds")
    args = parser.parse_args()
    seeds = v2.SEEDS[:3] if args.quick else v2.SEEDS
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    v3.RESULTS.mkdir(parents=True, exist_ok=True)
    for problem in v2.PROBLEMS:
        v2.test_set(problem)
    designs = v2.designs(v2.SEEDS)

    start = time.perf_counter()
    tasks = [(p, n, s, name, datasets) for (p, kind, n, s), datasets in designs.items()
             if kind == "nested" and s in seeds for name in MODEL_NAMES]
    nested = pd.DataFrame(bl.run_parallel(nested_task, tasks))
    nested.to_csv(v3.RESULTS / "nested.csv", index=False)
    print(f"A. {len(tasks)} nested fits in {time.perf_counter() - start:.0f} s", flush=True)

    start = time.perf_counter()
    methods = ([f"NN-MF-EGO {name}" for name in v3.EGO_VARIANTS]
               + ["SF-EGO (Matern)", "BoTorch qLogEI (SF)"])
    histories = bl.run_parallel(optimization_task, [(m, s) for m in methods for s in seeds])
    (v3.RESULTS / "optimization_histories.json").write_text(json.dumps(histories, indent=1),
                                                            encoding="utf-8")
    runs = pd.DataFrame([{"method": h["method"], "seed": h["seed"],
                          "best HF error": h["best_observed"][-1],
                          "recommendation error": h["recommendation_error"],
                          "LF share": float(np.mean(np.array(h["levels"][1:]) == 1)),
                          "final rho": h["rhos"][0] if h.get("rhos") else np.nan}
                         for h in histories if "error" not in h])
    runs.to_csv(v3.RESULTS / "optimization.csv", index=False)
    print(f"B1. {len(histories)} optimization runs in {time.perf_counter() - start:.0f} s",
          flush=True)

    start = time.perf_counter()
    tasks = [(p, n, s, name) for p in v2.PROBLEMS for n in EGO_SIZES[p] for s in seeds
             for name in v3.EGO_VARIANTS]
    infill = pd.DataFrame(bl.run_parallel(infill_task, tasks))
    infill.to_csv(v3.RESULTS / "infill.csv", index=False)
    print(f"B2. {len(tasks)} EGO runs in {time.perf_counter() - start:.0f} s", flush=True)

    for frame in (nested, infill):
        if "error" in frame and frame["error"].notna().any():
            print("FAILED:", frame["error"].dropna().unique()[:5])
    failed = [h for h in histories if "error" in h]
    if failed:
        print("FAILED optimization runs:", [(h["method"], h["error"]) for h in failed][:5])

    ok = [f[f["error"].isna()] if "error" in f else f for f in (nested, infill)]
    text = summary(ok[0], ok[1], runs)
    print(text)
    variants = {name: {key: str(value) for key, value in options.items()}
                for name, options in v3.EGO_VARIANTS.items()}
    (v3.RESULTS / "summary.txt").write_text(
        f"seeds: {seeds}, sizes: {v2.SIZES}, EGO sizes: {EGO_SIZES}\n"
        f"variants: {json.dumps(variants, indent=1)}\n\n{text}", encoding="utf-8")


if __name__ == "__main__":
    main()
