"""
[V2] Selection of the new defaults of mfego (benchmark before the final comparison):
centred rho, normal prior on rho (std 0.5 / 1 / 2) and kernel (squared exponential / Matern 5/2),
against mfego v1, on 3 problems (Hartmann 6D, airfoil NeuralFoil/NeuralFoil, airfoil
NeuralFoil/XFOIL), nested and independent designs (n_LF = 4 n_HF), 3 sizes, 10 seeds.
Model: MultifidelityModel (recursive, used by the EGO; on nested designs it equals CoKriging).

Decision rule (fixed BEFORE the run, see the plan):
- for each problem, mean over the (design, size) cells of [median NLPD(variant) - median
  NLPD(v1 SE)] = dNLPD, and of median RMSE(variant) / median RMSE(v1 SE) = RMSE ratio;
- the chosen variant minimizes the mean dNLPD over the 3 problems, among the variants whose
  RMSE ratio is <= 1.05 on every problem. If none qualifies, the v1 defaults are kept.

Run from the repository root with the benchmark environment:
    .venv-benchmark\\Scripts\\python benchmarks\\surrogate_v2\\run_selection.py
Results: benchmarks/surrogate_v2/results/selection.csv and selection_summary.txt.
"""
import time

import numpy as np
import pandas as pd

import v2_lib as v2  # noqa: E402  (sets the paths and OMP_NUM_THREADS)
import bench_lib as bl  # noqa: E402
from src.validation import prediction_scores  # noqa: E402

REFERENCE = "v1 (SE)"
VARIANTS = {}
for kernel_name, kernel_class in v2.KERNELS.items():
    VARIANTS[f"v1 ({kernel_name})"] = {**v2.v1_options(), "kernel_class": kernel_class}
    VARIANTS[f"centred rho, no prior ({kernel_name})"] = {
        "use_map": True, "priors": {"rho": None}, "rho_centering": True,
        "kernel_class": kernel_class}
    for sigma in (0.5, 1.0, 2.0):
        VARIANTS[f"centred rho, N(1, {sigma}^2) ({kernel_name})"] = {
            "use_map": True, "priors": {"rho": ("normal", 1.0, sigma)}, "rho_centering": True,
            "kernel_class": kernel_class}


def fit_task(args: tuple) -> dict:
    """(problem, kind, n_HF, seed, variant, datasets) -> scores on the test set."""
    problem, kind, n_hf, seed, variant, datasets = args
    row = {"problem": problem, "design": kind, "n_hf": n_hf, "seed": seed, "variant": variant}
    try:
        x_test, y_test = v2.test_set(problem)
        model = v2.make_model("recursive", 2, seed, VARIANTS[variant])
        start = time.perf_counter()
        model.fit(bl.experiment_data(datasets))
        row["fit_time_s"] = time.perf_counter() - start
        mean, var, _ = model.predict_batch(x_test)
        row.update(prediction_scores(y_test, mean, var))
        row["rho"] = float(model.rhos[0])
    except Exception as error:  # noqa: BLE001  (a failing fit must not stop the benchmark)
        row["error"] = repr(error)
    return row


def decide(frame: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Decision rule of the module docstring: table per (variant, problem) and chosen variant."""
    cells = frame.groupby(["problem", "design", "n_hf", "variant"])[["nlpd", "rmse_rel"]].median()
    reference = cells.xs(REFERENCE, level="variant")
    delta = cells["nlpd"] - reference["nlpd"].reindex(cells.index.droplevel("variant")).values
    ratio = cells["rmse_rel"] / reference["rmse_rel"].reindex(
        cells.index.droplevel("variant")).values
    table = pd.DataFrame({"dNLPD": delta, "RMSE ratio": ratio}).groupby(
        ["variant", "problem"]).mean().unstack("problem")
    table[("mean dNLPD", "")] = table["dNLPD"].mean(axis=1)
    eligible = table[(table["RMSE ratio"] <= 1.05).all(axis=1)]
    chosen = REFERENCE if eligible.empty else eligible[("mean dNLPD", "")].idxmin()
    return table.sort_values(("mean dNLPD", "")), chosen


def main() -> None:
    start = time.perf_counter()
    for problem in v2.PROBLEMS:      # test sets and designs computed once, before the fits
        v2.test_set(problem)
    designs = v2.designs()
    print(f"test sets and designs ready in {time.perf_counter() - start:.0f} s", flush=True)

    start = time.perf_counter()
    tasks = [(*key, variant, datasets) for key, datasets in designs.items()
             for variant in VARIANTS]
    frame = pd.DataFrame(bl.run_parallel(fit_task, tasks))
    frame.to_csv(v2.RESULTS / "selection.csv", index=False)
    print(f"{len(tasks)} fits in {time.perf_counter() - start:.0f} s", flush=True)
    if "error" in frame and frame["error"].notna().any():
        print("FAILED fits:", frame["error"].dropna().unique()[:5])
    ok = frame[frame["error"].isna()] if "error" in frame else frame

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    table, chosen = decide(ok)
    medians = ok.groupby(["problem", "variant"])[["rmse_rel", "nlpd", "coverage_95", "z_rms",
                                                  "rho", "fit_time_s"]].median().round(3)
    text = "\n".join([
        "=== Decision table (mean over the design x size cells, reference v1 SE) ===",
        table.round(3).to_string(), "", f"CHOSEN VARIANT: {chosen}", "",
        "=== Medians over all the fits per problem ===", medians.to_string()])
    print(text)
    (v2.RESULTS / "selection_summary.txt").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
