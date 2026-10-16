"""
[CK] Two small checks of CoKriging on the Forrester 1D problem (Sacher Eq. 17, true rho = 2),
quoted in RAPPORT_COKRIGING.md (Sections 2.1 and 5):

1. Very few high-fidelity points (nested designs 11/4 and 11/5, 5 seeds): estimated rho and
   coverage of the 95 % interval. The variance is exact for GIVEN hyperparameters; with 4 HF
   points the estimated hyperparameters can be wrong and the interval too narrow.
2. Noisy low fidelity (nested 30/12, noise std 0.5 on the LF only, 3 seeds): the level-by-level
   estimation then sees the LF noise in the residual of level 2 (rho^2 x LF noise), so the noise
   of level 2 is overestimated and rho is biased.

Run from the repository root: python benchmarks\\cokriging\\forrester_checks.py
"""
import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "mfego"))
# pylint: disable=wrong-import-position,import-error
from src.cokriging import CoKriging  # noqa: E402
from src.data_management import ExperimentData  # noqa: E402
from src.kernels import SquaredExponentialKernel  # noqa: E402
from src.surrogate_models import V1_PRIORS  # noqa: E402

# options of mfego v1 (state of RAPPORT_COKRIGING.md): SE kernel, no prior on rho, v1 rho
V1 = {"priors": V1_PRIORS, "rho_centering": False}


def f_low(x):
    return 0.5 * (6 * x - 2) ** 2 * np.sin(12 * x - 4) + 10 * (x - 1)


def f_high(x):
    return 2 * f_low(x) - 20 * (x - 1)


def nested_forrester(sizes, seed, lf_noise=0.0):
    data = ExperimentData(bounds=[(0.0, 1.0)], costs=[1.0, 10.0])
    data.generate_nested_design(sizes, seed=seed)
    rng = np.random.default_rng(seed)
    data.y_dict[1] = f_low(data.x_dict[1][:, 0]) + lf_noise * rng.standard_normal(sizes[0])
    data.y_dict[2] = f_high(data.x_dict[2][:, 0])
    return data


x_grid = np.linspace(0.0, 1.0, 401).reshape(-1, 1)
y_grid = f_high(x_grid[:, 0])

print("1. Few high-fidelity points (MAP, priors on all the hyperparameters)")
for sizes in ([11, 4], [11, 5]):
    for seed in range(5):
        model = CoKriging(2, SquaredExponentialKernel, seed=0, **V1)
        model.fit(nested_forrester(sizes, seed))
        mean, var = model.predict_batch(x_grid)
        coverage = np.mean(np.abs(mean - y_grid) <= norm.ppf(0.975) * np.sqrt(var))
        print(f"   design {sizes}, seed {seed}: rho = {model.rhos[0]:.3f}, "
              f"95 % coverage = {coverage:.3f}")

print("2. Noisy low fidelity (std 0.5), noise-free high fidelity, nested 30 / 12")
for seed in range(3):
    data = nested_forrester([30, 12], seed, lf_noise=0.5)
    model = CoKriging(2, SquaredExponentialKernel, seed=0, **V1)
    model.fit(data)
    x_hf, y_hf = data.get_training_data(2)
    mean_hf = model.predict_batch(x_hf, include_noise=False)[0]
    print(f"   seed {seed}: rho = {model.rhos[0]:.3f} (true 2) | fitted noise variance: LF "
          f"{model.gps[0].get_noise_variance():.3f} (true 0.25), level 2 "
          f"{model.gps[1].get_noise_variance():.3f} (true 0, rho^2 x LF noise = "
          f"{model.rhos[0] ** 2 * 0.25:.2f}) | max error at the HF points "
          f"{np.max(np.abs(mean_hf - y_hf)):.3f}")
