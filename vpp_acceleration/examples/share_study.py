"""
Validation B (plan validated by the user on 2026-10-21): the sequential region test with the
new option CoKriging(share_lengthscales=True) of mfego.

S3 = S1 of region_study.py (ALC, integration on the region, rho bounded v2, plain variance,
nested start 80 LF + 20 HF, budget 280: total cost 560) whose correction takes the lengthscales
of the LF level. Seeds 0 to 9, compared with S1 (results/region_study.csv, same seeds, same
start) and with T1 BoTorch (results/library_study.csv, same seeds).

Criterion fixed before the computation:
  * S3 lowers the median error on the region by at least 5 % against S1;
  * S3 wins at least 7 seeds out of 10 against S1;
  * the 95 % coverage of S3 on the region stays at least 0.90.
S3 against T1 is reported without criterion.

Run from the repository root (Multifidelity-GP-Clean), about 15 min with 10 processes:
    .venv-benchmark\\Scripts\\python vpp_acceleration\\examples\\share_study.py
Outputs: examples/results/share_study.csv, share_curves.csv, share_summary.txt
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

import pandas as pd  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import benchmark_study as bs  # noqa: E402
import region_study as rs  # noqa: E402

COLUMNS = ["strategy", "seed", "rmse_region", "coverage_region", "rmse_global", "n_lf", "n_hf",
           "wall_time_s"]


def summarize(s3: pd.DataFrame) -> str:
    seeds = s3["seed"].unique()
    reference = pd.read_csv(bs.RESULTS / "region_study.csv")
    library = pd.read_csv(bs.RESULTS / "library_study.csv")
    others = pd.concat([reference[reference["strategy"].isin(["S1", "B1"])],
                        library[library["strategy"] == "T1"]])
    results = pd.concat([s3[COLUMNS], others[others["seed"].isin(seeds)][COLUMNS]])
    lines = [f"=== Sequential region test with shared lengthscales, {len(seeds)} seeds ==="]
    for strategy in ("S3", "S1", "T1", "B1"):
        r = results[results["strategy"] == strategy]
        lines.append(f"{strategy}: region RMSE {r['rmse_region'].median():.4f} "
                     f"[{r['rmse_region'].quantile(0.1):.4f} ; "
                     f"{r['rmse_region'].quantile(0.9):.4f}] | cov95 "
                     f"{r['coverage_region'].median():.2f} | global {r['rmse_global'].median():.4f}"
                     f" | LF/HF {r['n_lf'].median():.0f}/{r['n_hf'].median():.0f} | time "
                     f"{r['wall_time_s'].median():.0f} s")
    table = results.pivot(index="seed", columns="strategy", values="rmse_region")
    lines.append("")
    for other in ("S1", "T1", "B1"):
        wins = int((table["S3"] < table[other]).sum())
        p_value = wilcoxon(table["S3"], table[other]).pvalue                  # two-sided
        change = 1.0 - table["S3"].median() / table[other].median()
        lines.append(f"S3 against {other}: median error {100 * change:+.0f} % lower | wins "
                     f"{wins} / {len(table)} | two-sided Wilcoxon p = {p_value:.2g}")
    reduction = 1.0 - table["S3"].median() / table["S1"].median()
    wins = int((table["S3"] < table["S1"]).sum())
    coverage = results[results["strategy"] == "S3"]["coverage_region"].median()
    checks = {f"median error at least 5 % lower than S1 ({100 * reduction:+.1f} %)":
              reduction >= 0.05,
              f"wins at least 7 / 10 against S1 ({wins} / {len(table)})": wins >= 7,
              f"coverage on the region at least 0.90 ({coverage:.2f})": coverage >= 0.90}
    lines.append("\nCriterion fixed before the computation:")
    lines += [f"  [{'OK' if ok else 'NO'}] {text}" for text, ok in checks.items()]
    lines.append(f"  => {'SUCCESS' if all(checks.values()) else 'FAILURE'}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    if args.summary_only:
        s3 = pd.read_csv(bs.RESULTS / "share_study.csv")
    else:
        rs.region_data()
        rows, curves = [], []
        start = time.perf_counter()
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for row, curve_rows in pool.map(rs.run_task, [("S3", s) for s in range(args.seeds)]):
                rows.append(row)
                curves += curve_rows
                print(f"S3 seed {row['seed']}: region RMSE {row['rmse_region']:.4f}, cov "
                      f"{row['coverage_region']:.2f}, LF/HF {row['n_lf']}/{row['n_hf']}, "
                      f"{row['wall_time_s']:.0f} s ({time.perf_counter() - start:.0f} s elapsed)",
                      flush=True)
        s3 = pd.DataFrame(rows)
        s3.to_csv(bs.RESULTS / "share_study.csv", index=False)
        pd.DataFrame(curves).to_csv(bs.RESULTS / "share_curves.csv", index=False)
    summary = summarize(s3)
    print(summary)
    (bs.RESULTS / "share_summary.txt").write_text(summary + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
