"""
[V2] Which change of mfego v2 modifies the behaviour of NN-MF-EGO on Hartmann 6D?
Optimization at equal budget (250, same protocol as run_benchmark.py, 10 seeds) with:
- v1 (squared exponential, constant of the residual at rho = 1, no prior on rho),
- v2 (Matern 5/2, centred rho, N(1, 2^2) prior),
- Matern only (v1 rho), centred rho with its N(1, 2^2) prior only (squared exponential kernel).
The merit function (Sacher Eq. 24) multiplies the information of a lower level by rho^2: a
larger rho makes the low fidelity look more informative.

Run from the repository root (benchmark environment):
    .venv-benchmark\\Scripts\\python benchmarks\\surrogate_v2\\explore_ego_rho.py
Result: benchmarks/surrogate_v2/results/ego_rho.csv (+ printed medians).
"""
import numpy as np
import pandas as pd

import v2_lib as v2  # noqa: E402  (sets the paths)
import bench_lib as bl  # noqa: E402
from src.kernels import Matern52Kernel, SquaredExponentialKernel  # noqa: E402

VARIANTS = {
    "v1": v2.v1_options(),
    "v2": v2.v2_options(),
    "Matern only (v1 rho)": {**v2.v1_options(), "kernel_class": Matern52Kernel},
    "centred rho + prior only (SE)": {**v2.v2_options(), "kernel_class": SquaredExponentialKernel},
}


def task(args: tuple) -> dict:
    variant, seed = args
    problem = bl.HartmannMF()
    history, model, _ = bl.run_mfego(problem, problem.doe([20, 10], seed), 250, seed, True,
                                     return_model=True, **VARIANTS[variant])
    levels = np.array(history.levels[1:])
    return {"variant": variant, "seed": seed, "best HF error": history.best_observed[-1],
            "recommendation error": history.recommendation_error,
            "LF share": float(np.mean(levels == 1)), "final rho": float(model.rhos[0])}


def main() -> None:
    frame = pd.DataFrame(bl.run_parallel(task, [(v, s) for v in VARIANTS for s in v2.SEEDS]))
    frame.to_csv(v2.RESULTS / "ego_rho.csv", index=False)
    pd.set_option("display.width", 200)
    print(frame.groupby("variant")[["best HF error", "recommendation error", "LF share",
                                    "final rho"]].median().round(3).to_string())
    wide = frame.pivot_table(index="seed", columns="variant", values="recommendation error")
    for variant in VARIANTS:
        if variant != "v1":
            print(f"{variant} better than v1 (recommendation): "
                  f"{int((wide[variant] < wide['v1']).sum())}/{len(wide)}")


if __name__ == "__main__":
    main()
