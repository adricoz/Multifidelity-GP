"""
Example: build a multi-fidelity SURROGATE (no optimization) and qualify it:
1. priors on ALL the GP hyperparameters (use_map=True: lengthscales, t1, t2, noise, rho),
2. the co-kriging conditioned on all the data (src/cokriging.CoKriging, nested designs only),
3. its qualification WITHOUT a test set: leave-one-out at the high-fidelity points
   (Q^2, coverage of the 95 % interval, src/validation.prediction_scores).

Problem: Forrester 1D of Sacher et al. Eq. 17 (same as mfego/main.py): f2 = 2 f1 - 20 (x - 1).

Run from the repository root:
    python example\\cokriging\\main.py
Output: printed summary and example/cokriging/cokriging_forrester.html (interactive figure).

How to launch the rest (benchmark environment .venv-benchmark, see requirements-benchmark.txt):
    .venv-benchmark\\Scripts\\python -m pytest                                    # unit tests
    .venv-benchmark\\Scripts\\python benchmarks\\priors_all\\run_priors_benchmark.py  # priors
    .venv-benchmark\\Scripts\\python benchmarks\\priors_all\\make_figures.py
    .venv-benchmark\\Scripts\\python benchmarks\\cokriging\\run_cokriging_benchmark.py
    .venv-benchmark\\Scripts\\python benchmarks\\cokriging\\make_figures.py
    .venv-benchmark\\Scripts\\python benchmarks\\surrogate_v2\\run_selection.py      # v2
    .venv-benchmark\\Scripts\\python benchmarks\\surrogate_v2\\run_benchmark.py
    .venv-benchmark\\Scripts\\python benchmarks\\surrogate_v2\\make_figures.py
    .venv-benchmark\\Scripts\\python benchmarks\\surrogate_v3\\run_benchmark.py      # v3
    .venv-benchmark\\Scripts\\python benchmarks\\surrogate_v3\\make_figures.py

Priors in the existing EGO optimizer (mfego/main.py): nothing else to change than
    model = MultifidelityModel(l=2, kernel_class=SquaredExponentialKernel, use_map=True)
and, to change one prior (None removes it):
    model = MultifidelityModel(..., use_map=True, priors={"noise": ("lognormal", -9.2, 2.0)})
Keep the default rho_centering=False (mfego v1 rho, no prior on rho) for NN-MF-EGO: the centred
rho makes its merit function over-weight the low fidelity (RAPPORT_V2.md Section 7).
"""
import os
import sys

import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "mfego"))
# pylint: disable=wrong-import-position,import-error
from src.cokriging import CoKriging  # noqa: E402
from src.data_management import ExperimentData  # noqa: E402
from src.kernels import SquaredExponentialKernel  # noqa: E402
from src.surrogate_models import V3_RHO_BOUNDS, V3_RHO_PRIOR, MultifidelityModel  # noqa: E402
from src.validation import prediction_scores  # noqa: E402


def f_low(x):
    """Low fidelity f1 (Eq. 17)."""
    return 0.5 * (6 * x - 2) ** 2 * np.sin(12 * x - 4) + 10 * (x - 1)


def f_high(x):
    """High fidelity f2 = 2 f1 - 20 (x - 1) (Eq. 17, true rho = 2)."""
    return 2 * f_low(x) - 20 * (x - 1)


# 1. Nested design: 11 low-fidelity points, 5 of them also evaluated at the high fidelity
#    (with only 4 high-fidelity points rho is badly estimated and the 95 % band is too narrow:
#    the variance is exact for GIVEN hyperparameters, their uncertainty is not included)
data = ExperimentData(bounds=[(0.0, 1.0)], costs=[1.0, 10.0])
data.generate_nested_design(points_per_level=[11, 5], seed=0)

# 2. Evaluation of the two levels (replace by your simulators)
data.y_dict[1] = f_low(data.x_dict[1][:, 0])
data.y_dict[2] = f_high(data.x_dict[2][:, 0])

# 3. The two surrogates, both with the priors on all the hyperparameters and the centred rho
#    learned under a lognormal prior (mfego v3: default of CoKriging, options of
#    MultifidelityModel). Forrester is an analytic (infinitely smooth) function: the squared
#    exponential kernel fits it best. On real simulation codes the Matern 5/2 kernel
#    (Matern52Kernel, default of CoKriging) is better (benchmarks/surrogate_v2/RAPPORT_V2.md).
recursive = MultifidelityModel(2, SquaredExponentialKernel, use_map=True, rho_centering=True,
                               priors={"rho": V3_RHO_PRIOR}, rho_bounds=V3_RHO_BOUNDS, seed=0)
recursive.fit(data)
cokriging = CoKriging(2, SquaredExponentialKernel, use_map=True, seed=0)
cokriging.fit(data)

# 4. Accuracy and calibration on a fine grid (95 % interval = mean +- 1.96 std)
x_grid = np.linspace(0.0, 1.0, 401).reshape(-1, 1)
y_true = f_high(x_grid[:, 0])
predictions = {"recursive (MultifidelityModel)": recursive.predict_batch(x_grid)[:2],
               "co-kriging (CoKriging)": cokriging.predict_batch(x_grid)}
print(f"rho: recursive {recursive.rhos[0]:.3f} | co-kriging {cokriging.rhos[0]:.3f} (true 2)")
for name, (mean, var) in predictions.items():
    std = np.sqrt(var)
    rmse = np.sqrt(np.mean((mean - y_true) ** 2))
    coverage = np.mean(np.abs(mean - y_true) <= norm.ppf(0.975) * std)
    print(f"{name:32s} RMSE {rmse:.3f} | 95 % coverage {coverage:.2f}")
for level in cokriging.diagnostics()["levels"]:
    print(f"level {level['level']}: lengthscale {level['lengthscales'][0]:.3f}, "
          f"t1 {level['signal_variance']:.3f}, t2 {level['bias_variance']:.1e}, "
          f"noise {level['noise_normalized']:.1e} ({level['estimator']})")

# 5. Qualification WITHOUT a test set: each HF point predicted from all the other data
loo = prediction_scores(*cokriging.leave_one_out())
print(f"leave-one-out at the {len(data.y_dict[2])} HF points: Q2 {loo['q2']:.3f} | "
      f"95 % coverage {loo['coverage_95']:.2f}")

# 6. Interactive figure: truth, data, co-kriging mean and 95 % band
mean, var = predictions["co-kriging (CoKriging)"]
band = 1.96 * np.sqrt(var)
fig = go.Figure([
    go.Scatter(x=np.r_[x_grid[:, 0], x_grid[::-1, 0]], y=np.r_[mean + band, (mean - band)[::-1]],
               fill="toself", fillcolor="rgba(42,120,214,0.15)", line={"width": 0},
               name="co-kriging 95 % interval", hoverinfo="skip"),
    go.Scatter(x=x_grid[:, 0], y=y_true, name="true f2 (high fidelity)",
               line={"color": "#52514e", "width": 2, "dash": "dot"}),
    go.Scatter(x=x_grid[:, 0], y=mean, name="co-kriging mean", line={"color": "#2a78d6",
                                                                      "width": 2}),
    go.Scatter(x=x_grid[:, 0], y=predictions["recursive (MultifidelityModel)"][0],
               name="recursive mean", line={"color": "#eb6834", "width": 2, "dash": "dash"}),
    go.Scatter(x=data.x_dict[1][:, 0], y=data.y_dict[1], mode="markers",
               name="low-fidelity data f1", marker={"color": "#898781", "size": 8}),
    go.Scatter(x=data.x_dict[2][:, 0], y=data.y_dict[2], mode="markers",
               name="high-fidelity data f2", marker={"color": "#0b0b0b", "size": 10,
                                                     "symbol": "diamond"}),
])
fig.update_layout(title="Forrester 1D: co-kriging conditioned on all the data (nested design)",
                  xaxis_title="x", yaxis_title="f", template="plotly_white", hovermode="x")
fig.write_html(os.path.join(HERE, "cokriging_forrester.html"), include_plotlyjs="cdn")
print("Figure written:", os.path.join(HERE, "cokriging_forrester.html"))
