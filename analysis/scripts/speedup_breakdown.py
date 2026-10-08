"""
Where does the speed-up of the corrected code come from? (report, Sec. 9)

1. Cumulative ablation ("waterfall") on 5 EGO iterations of the Hartmann 6D problem (2 levels,
   20 LF / 15 HF points, same seed): initial code, then the corrected code with every
   optimization switched OFF by monkeypatching (double-loop covariance, finite-difference
   gradient, explicit-inverse point-by-point prediction, merit computed point by point and
   level by level with L^2 GP predictions, non-vectorized differential evolution, no warm
   start), then the optimizations switched back ON one at a time: N1, N3, N4, N6, N8.
2. Micro-benchmarks of each mechanism in isolation.

Usage (from the repository root):  python analysis/scripts/speedup_breakdown.py
"""
import logging
import os
import tempfile
import time

import numpy as np
import plotly.graph_objects as go
from _common import hartmann6, hartmann_level, load_mfego, make_data, save_figure, save_results
from plotly.subplots import make_subplots

logging.disable(logging.CRITICAL)
N_ITERATIONS = 5
FUNCS = [lambda x: hartmann_level(x, 1, 0.05), hartmann6]


# ------------------------------------------------------------------------------------------
# Slow versions of the optimized mechanisms (behaviour of the initial code)
# ------------------------------------------------------------------------------------------
def loop_covariance_matrix(initial_kernels):
    """N1 OFF: the initial double Python loop over cov_fct."""
    return initial_kernels.base_covariance_matrix


def slow_predict_batch(self, x_new):
    """N4 OFF: explicit inverse K^-1 (computed once, as the initial fit did) and point-by-point
    prediction k^T K^-1 y and k^T K^-1 k in O(n^2) per point."""
    if getattr(self, "_k_inv", None) is None or self._k_inv.shape[0] != len(self.x_train):
        eye = np.eye(len(self.x_train))
        self._k_inv = np.linalg.solve(self.l_chol.T, np.linalg.solve(self.l_chol, eye))
    y_n = (self.y_train - self.y_mean) / self.y_std
    kappa = self.kernel.signal_variance + self.kernel.bias_variance
    means, variances = [], []
    for x in np.atleast_2d(x_new):
        k_vec = self.kernel.get_cross_variance_vector(x, self.x_train)
        means.append(float(np.squeeze(k_vec.T @ self._k_inv @ y_n)))
        variances.append(float(np.squeeze(kappa + self.noise - k_vec.T @ self._k_inv @ k_vec)))
    means, variances = np.array(means), np.maximum(np.array(variances), 0.0)
    return self.y_mean + self.y_std * means, self.y_std ** 2 * variances


def make_slow_merits(original):
    """N6 OFF: merit of every point and every level computed separately (the initial
    evaluate_merit predicted all the levels for each (point, level): L^2 GP predictions)."""
    def slow_merits(self, x):
        x = np.atleast_2d(x)
        num_levels = self.model.num_levels
        out = np.zeros((len(x), num_levels))
        for i, point in enumerate(x):
            for level in range(num_levels):
                out[i, level] = original(self, point.reshape(1, -1))[0, level]
        return out
    return slow_merits


def make_non_vectorized_de(original):
    """N6 OFF: differential evolution evaluating one candidate per call."""
    def de(*args, **kwargs):
        kwargs["vectorized"] = False
        return original(*args, **kwargs)
    return de


def cold_initial_guesses(original):
    """N8 OFF: no warm start (every restart is random)."""
    def guesses(self, d, n_restarts, log_bounds):
        saved = self.kernel.lengthscale
        self.kernel.lengthscale = None
        try:
            return original(self, d, n_restarts, log_bounds)
        finally:
            self.kernel.lengthscale = saved
    return guesses


# ------------------------------------------------------------------------------------------
def run_iterations(mod, seed=0) -> float:
    """Wall time of N_ITERATIONS EGO iterations (ask + simulation + tell, final fit)."""
    data = make_data(mod, [(0.0, 1.0)] * 6, [1.0, 10.0], [20, 15], FUNCS)

    class Simulator(mod.simulator.BaseSimulator):
        def evaluate(self, design_point, level):
            return float(FUNCS[level - 1](np.atleast_2d(design_point))[0]), {}

    kwargs = {"seed": seed} if hasattr(mod.optimizer.EGOOptimizer, "summary") else {}
    model = mod.surrogate_models.MultifidelityModel(2, mod.kernels.SquaredExponentialKernel,
                                                    **kwargs)
    acq = mod.acquisition.AcquisitionFunction(model, data)
    path = os.path.join(tempfile.mkdtemp(), "state.json")
    ego = mod.optimizer.EGOOptimizer(data, model, Simulator(num_levels=2), acq,
                                     save_state_path=path, **kwargs)
    np.random.seed(seed)
    start = time.perf_counter()
    ego.run(n_iterations=N_ITERATIONS)
    return time.perf_counter() - start


def ablation() -> list[dict]:
    """Cumulative waterfall (initial code, all optimizations OFF, then ON one by one)."""
    initial = load_mfego("initial")
    steps = [{"step": "Initial code (legacy/legacy_mfego_initial)",
              "time_s": run_iterations(initial)}]
    print(steps[-1])

    cur = load_mfego("current")
    gp_class = cur.surrogate_models.GaussianProcess
    acq_class = cur.acquisition.AcquisitionFunction
    originals = {
        "cov": cur.kernels.base_covariance_matrix,
        "grad": gp_class._kernel_has_gradients,
        "predict": gp_class.predict_batch,
        "merits": acq_class.evaluate_merits_batch,
        "de": cur.optimizer.scipy.optimize.differential_evolution,
        "guesses": gp_class._initial_guesses,
    }
    slow = {
        "cov": loop_covariance_matrix(initial.kernels),
        "grad": lambda self: False,
        "predict": slow_predict_batch,
        "merits": make_slow_merits(originals["merits"]),
        "de": make_non_vectorized_de(originals["de"]),
        "guesses": cold_initial_guesses(originals["guesses"]),
    }

    def apply(enabled: set):
        cur.kernels.base_covariance_matrix = originals["cov"] if "N1" in enabled else slow["cov"]
        gp_class._kernel_has_gradients = originals["grad"] if "N3" in enabled else slow["grad"]
        gp_class.predict_batch = originals["predict"] if "N4" in enabled else slow["predict"]
        acq_class.evaluate_merits_batch = (originals["merits"] if "N6" in enabled
                                           else slow["merits"])
        cur.optimizer.scipy.optimize.differential_evolution = (
            originals["de"] if "N6" in enabled else slow["de"])
        gp_class._initial_guesses = (originals["guesses"] if "N8" in enabled
                                     else slow["guesses"])

    labels = {
        frozenset(): "Corrected code, all optimizations OFF",
        frozenset({"N1"}): "+ N1 vectorized covariance matrix",
        frozenset({"N1", "N3"}): "+ N3 analytical gradient (log-space)",
        frozenset({"N1", "N3", "N4"}): "+ N4 Cholesky solves, batch prediction",
        frozenset({"N1", "N3", "N4", "N6"}): "+ N6 vectorized merit and DE",
        frozenset({"N1", "N3", "N4", "N6", "N8"}): "+ N8 warm start (= corrected code)",
    }
    enabled = set()
    try:
        for new in (None, "N1", "N3", "N4", "N6", "N8"):
            if new:
                enabled.add(new)
            apply(enabled)
            steps.append({"step": labels[frozenset(enabled)], "time_s": run_iterations(cur)})
            print(steps[-1])
    finally:
        apply({"N1", "N3", "N4", "N6", "N8"})
    return steps


# ------------------------------------------------------------------------------------------
def timeit(func, repeat=3):
    best = np.inf
    for _ in range(repeat):
        start = time.perf_counter()
        func()
        best = min(best, time.perf_counter() - start)
    return best


def micro_benchmarks() -> dict:
    """Each mechanism in isolation."""
    initial, cur = load_mfego("initial"), load_mfego("current")
    rng = np.random.default_rng(0)
    results = {}

    # M1 covariance matrix (N1)
    theta = np.array([0.3] * 6 + [1.0, 0.01])
    results["covariance_matrix_s"] = {}
    for n in (25, 50, 100, 200, 400):
        x = rng.random((n, 6))
        results["covariance_matrix_s"][n] = {
            "loop": timeit(lambda x=x: initial.kernels.base_covariance_matrix(x, theta)),
            "vectorized": timeit(lambda x=x: cur.kernels.base_covariance_matrix(x, theta))}

    # M2 fit of one GP: covariance matrices (= Cholesky factorizations) per fit (N2/N3)
    x = rng.random((35, 6))
    y = hartmann6(x)
    gp_class = cur.surrogate_models.GaussianProcess
    results["gp_fit"] = {}
    for name, has_grad in (("finite differences", False), ("analytical gradient", True)):
        gp = gp_class(cur.kernels.SquaredExponentialKernel(), seed=0)
        counter = {"n": 0}
        original = gp.kernel.get_covariance_matrix

        def counting(x_train, original=original, counter=counter):
            counter["n"] += 1
            return original(x_train)
        gp.kernel.get_covariance_matrix = counting
        gp._kernel_has_gradients = lambda has_grad=has_grad: has_grad
        start = time.perf_counter()
        gp.fit(x, y)
        results["gp_fit"][name] = {"time_s": time.perf_counter() - start,
                                   "covariance_matrices": counter["n"]}
    gp = initial.surrogate_models.GaussianProcess(initial.kernels.SquaredExponentialKernel())
    counter = {"n": 0}
    original = gp.kernel.get_covariance_matrix
    gp.kernel.get_covariance_matrix = lambda x_train: (counter.__setitem__("n", counter["n"] + 1)
                                                       or original(x_train))
    np.random.seed(0)
    start = time.perf_counter()
    gp.fit(x, y)
    results["gp_fit"]["initial code"] = {"time_s": time.perf_counter() - start,
                                         "covariance_matrices": counter["n"]}

    # M3 prediction of 2500 points (N4/N6)
    fitted = gp_class(cur.kernels.SquaredExponentialKernel(), seed=0)
    fitted.fit(x, y)
    x_grid = rng.random((2500, 6))
    results["prediction_2500_points_s"] = {
        "explicit inverse, point by point": timeit(lambda: slow_predict_batch(fitted, x_grid), 1),
        "Cholesky, batch": timeit(lambda: fitted.predict_batch(x_grid))}

    # M4 merit of a DE population of 60 points, 2 levels (N6)
    data = make_data(cur, [(0.0, 1.0)] * 6, [1.0, 10.0], [20, 15], FUNCS)
    model = cur.surrogate_models.MultifidelityModel(2, cur.kernels.SquaredExponentialKernel,
                                                    seed=0)
    model.fit(data)
    acq = cur.acquisition.AcquisitionFunction(model, data)
    acq.update()
    population = rng.random((60, 6))
    slow = make_slow_merits(cur.acquisition.AcquisitionFunction.evaluate_merits_batch)
    results["merit_population_60x2_s"] = {
        "point by point, level by level (L^2 predictions)": timeit(lambda: slow(acq, population)),
        "one batch": timeit(lambda: acq.evaluate_merits_batch(population))}
    print(results)
    return results


def figures(steps, micro):
    waterfall = go.Figure(go.Bar(
        x=[s["time_s"] for s in steps], y=[s["step"] for s in steps], orientation="h",
        text=[f"{s['time_s']:.2f} s" for s in steps], textposition="outside",
        marker={"color": ["#d62728"] + ["#9467bd"] + ["#1f77b4"] * (len(steps) - 2)}))
    waterfall.update_layout(
        title=f"Wall time of {N_ITERATIONS} EGO iterations (Hartmann 6D, 2 levels, "
              "20 LF / 15 HF): cumulative effect of each optimization",
        xaxis_title="time (s, log scale)", xaxis_type="log", yaxis={"autorange": "reversed"},
        template="plotly_white", height=500, margin={"l": 330})
    save_figure(waterfall, "speedup_waterfall")

    fig = make_subplots(rows=1, cols=4, subplot_titles=(
        "M1 covariance matrix (d = 6)", "M2 matrices per GP fit (n = 35)",
        "M3 prediction of 2 500 points", "M4 merit of 60 points x 2 levels"))
    cov = micro["covariance_matrix_s"]
    for key, color in (("loop", "#d62728"), ("vectorized", "#1f77b4")):
        fig.add_trace(go.Scatter(x=list(cov), y=[v[key] for v in cov.values()], name=key,
                                 mode="lines+markers", line={"color": color}), row=1, col=1)
    fit = micro["gp_fit"]
    fig.add_trace(go.Bar(x=list(fit), y=[v["covariance_matrices"] for v in fit.values()],
                         showlegend=False, marker={"color": "#7f7f7f"}), row=1, col=2)
    for col, key in ((3, "prediction_2500_points_s"), (4, "merit_population_60x2_s")):
        fig.add_trace(go.Bar(x=list(micro[key]), y=list(micro[key].values()), showlegend=False,
                             marker={"color": ["#d62728", "#1f77b4"]}), row=1, col=col)
    fig.update_yaxes(type="log")
    fig.update_xaxes(type="log", title_text="n", row=1, col=1)
    fig.update_layout(title="Speed-up mechanisms in isolation (initial vs corrected)",
                      template="plotly_white", height=480)
    save_figure(fig, "speedup_mechanisms")


if __name__ == "__main__":
    ablation_steps = ablation()
    micro_results = micro_benchmarks()
    figures(ablation_steps, micro_results)
    save_results({"ablation": ablation_steps, "micro_benchmarks": micro_results},
                 "speedup_breakdown")
