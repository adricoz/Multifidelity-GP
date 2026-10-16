"""
[CK] Exploration (benchmark only, the mfego core is NOT modified): bias of the profiled rho.

In GaussianProcess.fit, the residual of level l is normalized with the mean of
y - rho_init * f_prev (rho_init = 1): the constant of the residual is fixed for rho = 1, then
rho is profiled with this constant (one correction pass only). When the mean of the level below
is far from 0 (Hartmann: low fidelity around -2.6), any rho != 1 shifts the residual by
(rho - 1) * mean(f_prev), which the small bias variance t2 cannot absorb: rho is pulled towards 1.

CenteredGP centres the level below too: the residual is (y - mean(y)) - rho (f - mean(f)), i.e.
its constant follows rho (ordinary least squares intercept), as in the AR(1) model of emukit
with each level centred. Compared on nested designs (CoKriging, MAP) with the current mfego and
GPy/emukit (joint MAP), 10 seeds.

Run from the repository root with the benchmark environment:
    .venv-benchmark\\Scripts\\python benchmarks\\cokriging\\explore_rho_centering.py
Result: benchmarks/cokriging/results/rho_centering.csv (+ printed medians).
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

# pylint: disable=wrong-import-position,import-error
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import bench_lib as bl  # noqa: E402
import run_cokriging_benchmark as rc  # noqa: E402
from src.cokriging import CoKriging  # noqa: E402
from src.kernels import SquaredExponentialKernel  # noqa: E402
from src.surrogate_models import DEFAULT_PRIORS, GaussianProcess  # noqa: E402


class CenteredGP(GaussianProcess):
    """GaussianProcess whose residual constant follows rho (level below centred too)."""
    def _normalize(self, y_obs, f_prev, rho_ref):
        y_n, f_n = super()._normalize(y_obs, f_prev, rho_ref)
        if f_prev is None:
            return y_n, f_n
        self.y_bar, self.f_bar = float(np.mean(y_obs)), float(np.mean(f_prev))
        self.y_mean = self.y_bar
        return (y_obs - self.y_bar) / self.y_std, (f_prev - self.f_bar) / self.y_std

    def fit(self, x_train, y_train, n_restarts=3, f_prev=None, **kwargs):
        rho = super().fit(x_train, y_train, n_restarts, f_prev=f_prev, **kwargs)
        if f_prev is not None:
            # constant of the residual y - rho f for the fitted rho, then the same conditioning
            self.y_mean = self.y_bar - rho * self.f_bar
            self.condition(self.x_train, np.asarray(y_train) - rho * np.asarray(f_prev))
        return rho


class CenteredCoKrigingSurrogate(bl.CoKrigingSurrogate):
    """mfego CoKriging (MAP) whose GPs are CenteredGP (rho_bounds: search bounds of rho)."""
    name = "mfego CoKriging (MAP, centred rho)"

    def __init__(self, seed: int = 0, rho_bounds: tuple = (-5.0, 5.0)):
        super().__init__(seed)
        self.rho_bounds = rho_bounds

    def fit(self, datasets):
        self.model = CoKriging(len(datasets), seed=self.seed, rho_bounds=self.rho_bounds,
                               **rc.V1)
        self.model.gps = [CenteredGP(SquaredExponentialKernel(), seed=self.seed + i,
                                     use_map=True) for i in range(len(datasets))]
        self.model.fit(bl.experiment_data(datasets))
        return self


MODELS = {
    "mfego CoKriging (MAP)": lambda seed: bl.CoKrigingSurrogate(seed, **rc.V1),
    "mfego CoKriging (MAP, centred rho)": CenteredCoKrigingSurrogate,
    # same with rho free to leave [-5, 5] (on Hartmann the centred rho sits on the bound 5)
    "mfego CoKriging (MAP, centred rho, bounds 50)":
        lambda seed: CenteredCoKrigingSurrogate(seed, rho_bounds=(-50.0, 50.0)),
    "GPy/emukit AR1 (MAP)": lambda seed: bl.EmukitAR1Surrogate(seed, priors=DEFAULT_PRIORS),
}


def task(args: tuple) -> dict:
    problem_name, model_name, n_hf, seed = args
    problem = rc.make_problem(problem_name)
    x_test, y_test = rc.test_set(problem_name)
    datasets = bl.nested_doe(problem, [rc.LF_PER_HF * n_hf, n_hf], seed=seed)
    model = MODELS[model_name](seed)
    model.fit(datasets)
    mean, var = model.predict(x_test)
    scores = rc.scores(mean, var, y_test)
    return {"problem": problem_name, "model": model_name, "n_hf": n_hf, "seed": seed,
            "rho": model.rhos[0], **{k: scores[k] for k in ("rmse_rel", "nlpd", "coverage_95",
                                                              "z_std")}}


def main() -> None:
    tasks = [(p, m, n, s) for p in rc.SIZES for m in MODELS for n in rc.SIZES[p]
             for s in range(10)]
    frame = pd.DataFrame(bl.run_parallel(task, tasks))
    frame.to_csv(rc.RESULTS / "rho_centering.csv", index=False)
    pd.set_option("display.width", 200)
    print(frame.groupby(["problem", "n_hf", "model"])[["rho", "rmse_rel", "nlpd", "coverage_95",
                                                        "z_std"]].median().round(3).to_string())
    wide = frame.pivot_table(index=["problem", "n_hf", "seed"], columns="model",
                             values=["rmse_rel", "nlpd"])
    for variant in ("mfego CoKriging (MAP, centred rho)",
                    "mfego CoKriging (MAP, centred rho, bounds 50)"):
        for metric in ("rmse_rel", "nlpd"):
            better = wide[metric][variant] < wide[metric]["mfego CoKriging (MAP)"]
            wins = better.groupby(level=["problem", "n_hf"]).agg(
                lambda b: f"{int(b.sum())}/{len(b)}")
            print(f"{variant} better than the current mfego ({metric}):",
                  json.dumps({f"{p} n_HF={n}": v for (p, n), v in wins.items()}))
    at_bound = frame[frame["model"] == "mfego CoKriging (MAP, centred rho)"]
    print("share of the centred rho at the bound |rho| = 5:",
          at_bound.assign(b=at_bound["rho"].abs() >= 5.0 - 1e-6)
          .groupby(["problem", "n_hf"])["b"].mean().round(2).to_dict())


if __name__ == "__main__":
    main()
