"""
Numerical performance and memory, before/after (axes "optimisation numérique" and "mémoire").

1. Covariance matrix: double Python loop (initial) vs vectorized (corrected).
2. Fit of one GP (d = 6): wall time and number of covariance matrices built.
3. One EGO iteration (ask = fit + search of the next point) on Hartmann 6D.
4. Memory: tracemalloc over the EGO iterations (current and peak traced memory).

Usage (from the repository root):  python analysis/scripts/profile_perf.py
"""
import logging
import os
import tempfile
import time
import tracemalloc

import numpy as np
import plotly.graph_objects as go
from _common import (hartmann6, hartmann_level, load_mfego, make_data, save_figure,
                     save_results)
from plotly.subplots import make_subplots

logging.disable(logging.CRITICAL)


def timeit(func, repeat=3):
    best = np.inf
    for _ in range(repeat):
        start = time.perf_counter()
        func()
        best = min(best, time.perf_counter() - start)
    return best


def covariance_timings() -> dict:
    results = {}
    rng = np.random.default_rng(0)
    theta = np.array([0.3] * 6 + [1.0, 0.01])
    for version in ("initial", "current"):
        kernels = load_mfego(version).kernels
        results[version] = {}
        for n in (25, 50, 100, 200, 400):
            x = rng.random((n, 6))
            results[version][n] = timeit(lambda k=kernels, x=x: k.base_covariance_matrix(x, theta))
    return results


def gp_fit_timings() -> dict:
    results = {}
    for version in ("initial", "current"):
        mod = load_mfego(version)
        results[version] = {}
        for n in (20, 40, 80):
            x = np.random.default_rng(n).random((n, 6))
            y = hartmann6(x)
            kwargs = {"seed": 0} if version == "current" else {}
            gp = mod.surrogate_models.GaussianProcess(mod.kernels.SquaredExponentialKernel(),
                                                      **kwargs)
            counter = {"n": 0}
            original = gp.kernel.get_covariance_matrix

            def counting(x_train, original=original, counter=counter):
                counter["n"] += 1
                return original(x_train)
            gp.kernel.get_covariance_matrix = counting
            np.random.seed(0)
            start = time.perf_counter()
            gp.fit(x, y)
            results[version][n] = {"time_s": time.perf_counter() - start,
                                   "covariance_matrices": counter["n"]}
    return results


def ego_setup(version: str):
    mod = load_mfego(version)
    funcs = [lambda x: hartmann_level(x, 1, 0.05), hartmann6]
    data = make_data(mod, [(0.0, 1.0)] * 6, [1.0, 10.0], [20, 10], funcs)

    class Simulator(mod.simulator.BaseSimulator):
        def evaluate(self, design_point, level):
            return float(funcs[level - 1](np.atleast_2d(design_point))[0]), {}

    kwargs = {"seed": 0} if version == "current" else {}
    model = mod.surrogate_models.MultifidelityModel(2, mod.kernels.SquaredExponentialKernel,
                                                    **kwargs)
    acq = mod.acquisition.AcquisitionFunction(model, data)
    path = os.path.join(tempfile.mkdtemp(), "state.json")
    return mod.optimizer.EGOOptimizer(data, model, Simulator(num_levels=2), acq,
                                      save_state_path=path, **kwargs)


def ego_iteration_and_memory(n_iterations: dict) -> dict:
    """Two separate passes: wall time per iteration (no tracing, tracemalloc slows numpy
    down a lot) and traced memory per iteration."""
    results = {}
    for version, n_iter in n_iterations.items():
        ego = ego_setup(version)
        np.random.seed(0)
        times = []
        for _ in range(n_iter):
            start = time.perf_counter()
            ego.run(n_iterations=1)
            times.append(time.perf_counter() - start)

        ego = ego_setup(version)
        np.random.seed(0)
        tracemalloc.start()
        current, peak = [], []
        for _ in range(n_iter):
            ego.run(n_iterations=1)
            cur, pk = tracemalloc.get_traced_memory()
            current.append(cur / 1e6)
            peak.append(pk / 1e6)
        tracemalloc.stop()
        results[version] = {"iteration_time_s": times, "traced_memory_mb": current,
                            "peak_memory_mb": peak,
                            "n_points": [len(v) for v in ego.data.y_dict.values()]}
        print(version, "mean iteration time", np.mean(times), "memory MB", current[-1])
    return results


if __name__ == "__main__":
    output = {"covariance_matrix_s": covariance_timings()}
    print("covariance", output["covariance_matrix_s"])
    output["gp_fit"] = gp_fit_timings()
    print("gp fit", output["gp_fit"])
    output["ego"] = ego_iteration_and_memory({"initial": 6, "current": 40})
    save_results(output, "profile_perf")

    fig = make_subplots(rows=1, cols=3, subplot_titles=(
        "Covariance matrix (d = 6)", "Fit of one GP (d = 6)", "Traced memory during EGO"))
    for version, color in (("initial", "#d62728"), ("current", "#1f77b4")):
        cov = output["covariance_matrix_s"][version]
        fig.add_trace(go.Scatter(x=list(cov), y=list(cov.values()), mode="lines+markers",
                                 line={"color": color}, name=f"{version} code"), row=1, col=1)
        fit = output["gp_fit"][version]
        fig.add_trace(go.Scatter(x=list(fit), y=[v["time_s"] for v in fit.values()],
                                 mode="lines+markers", line={"color": color},
                                 showlegend=False), row=1, col=2)
        mem = output["ego"][version]["traced_memory_mb"]
        fig.add_trace(go.Scatter(x=list(range(1, len(mem) + 1)), y=mem, mode="lines+markers",
                                 line={"color": color}, showlegend=False), row=1, col=3)
    fig.update_xaxes(type="log", title_text="n", row=1, col=1)
    fig.update_yaxes(type="log", title_text="time (s)", row=1, col=1)
    fig.update_xaxes(title_text="n", row=1, col=2)
    fig.update_yaxes(type="log", title_text="time (s)", row=1, col=2)
    fig.update_xaxes(title_text="EGO iteration", row=1, col=3)
    fig.update_yaxes(title_text="MB", row=1, col=3)
    fig.update_layout(title="Performance and memory, initial vs corrected code",
                      template="plotly_white")
    save_figure(fig, "performance_memory")
