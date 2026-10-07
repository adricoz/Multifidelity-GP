"""
Reconstruction of the response surfaces from the JSON files, before/after (axis 4 of the report).

1. Initial code: a short EGO run on Forrester, then the ModelVisualizer rebuilds the model
   from ego_backup.json. The JSON contains the last evaluated point but the hyperparameters
   of the previous fit: the rebuilt model differs from the model trained in memory.
2. Corrected code: same scenario, the rebuilt model must equal the trained model.
3. Scatter of the 2D response surface: rows (initial) vs columns (corrected).

Usage (from the repository root):  python analysis/scripts/check_reconstruction.py
"""
import json
import logging
import os
import tempfile

import matplotlib
import numpy as np
import plotly.graph_objects as go
from _common import (forrester_hf, forrester_lf, load_mfego, make_data, save_figure,
                     save_results)

matplotlib.use("Agg")
logging.disable(logging.CRITICAL)
X_GRID = np.linspace(0, 1, 201).reshape(-1, 1)


def predict(model, version, x):
    if version == "current":
        return model.predict_batch(x)[:2]
    out = np.array([model.predict(xi)[:2] for xi in x])
    return out[:, 0], out[:, 1]


def forrester_run(version: str, n_iterations: int = 4):
    """Short EGO run in a temporary directory; returns the module, optimizer and JSON path."""
    mod = load_mfego(version)
    funcs = [lambda x: forrester_lf(x[:, 0]), lambda x: forrester_hf(x[:, 0])]
    data = make_data(mod, [(0.0, 1.0)], [1.0, 10.0], [10, 4], funcs)

    class Simulator(mod.simulator.BaseSimulator):
        def evaluate(self, design_point, level):
            y = float(funcs[level - 1](np.atleast_2d(design_point))[0])
            return y, {"y": y}

    kwargs = {"seed": 0} if version == "current" else {}
    model = mod.surrogate_models.MultifidelityModel(2, mod.kernels.SquaredExponentialKernel,
                                                    **kwargs)
    acq = mod.acquisition.AcquisitionFunction(model, data)
    path = os.path.join(tempfile.mkdtemp(), "ego_backup.json")
    ego = mod.optimizer.EGOOptimizer(data, model, Simulator(num_levels=2), acq,
                                     save_state_path=path, **kwargs)
    np.random.seed(0)
    ego.run(n_iterations=n_iterations)
    return mod, ego, path


def compare(version: str) -> tuple[dict, dict]:
    mod, ego, path = forrester_run(version)
    with open(path, encoding="utf-8") as f:
        state = json.load(f)
    rebuilt = mod.visualization.ModelVisualizer(path, num_levels=2).model
    mean_mem, _ = predict(ego.model, version, X_GRID)
    mean_rebuilt, _ = predict(rebuilt, version, X_GRID)
    n_data = [len(state["Y_dict"][str(l)]) for l in (1, 2)]
    n_trained = [len(gp.y_train) for gp in ego.model.gps]
    result = {
        "points_in_json": n_data,
        "points_used_by_the_trained_model": n_trained,
        "max_abs_diff_rebuilt_vs_trained (rel. to std f)":
            float(np.max(np.abs(mean_rebuilt - mean_mem)) / np.std(forrester_hf(X_GRID[:, 0]))),
        "rmse_rebuilt_vs_truth": float(np.sqrt(np.mean((mean_rebuilt
                                                        - forrester_hf(X_GRID[:, 0])) ** 2))),
        "rmse_trained_vs_truth": float(np.sqrt(np.mean((mean_mem
                                                        - forrester_hf(X_GRID[:, 0])) ** 2))),
    }
    print(version, result)
    return result, {"trained": mean_mem, "rebuilt": mean_rebuilt}


def scatter_check(version: str) -> dict:
    """2D surface of a 3D model: which coordinates are plotted for the evaluations."""
    mod = load_mfego(version)
    data = make_data(mod, [(0.0, 1.0)] * 3, [1.0], [9], [lambda x: np.sum(x ** 2, axis=1)])
    np.random.seed(0)
    kwargs = {"seed": 0} if version == "current" else {}
    model = mod.surrogate_models.MultifidelityModel(1, mod.kernels.SquaredExponentialKernel,
                                                    **kwargs)
    model.fit(data)
    path = os.path.join(tempfile.mkdtemp(), "state.json")
    mod.optimizer.EGOOptimizer(data, model, None, None, save_state_path=path).save_state(path)
    captured = []
    plt = mod.visualization.plt
    original = plt.scatter
    plt.scatter = lambda x, y, **kw: captured.append((np.asarray(x), np.asarray(y)))
    try:
        viz = mod.visualization.ModelVisualizer(path, num_levels=1)
        viz.plot_response_surface_2d(param_x_idx=0, param_y_idx=1,
                                     save_path=os.path.join(tempfile.mkdtemp(), "s.png"))
    finally:
        plt.scatter = original
    x_plotted, y_plotted = captured[0]
    return {"n_points_plotted": int(np.size(x_plotted)), "n_points": 9,
            "plotted_x_equals_column_0": bool(np.size(x_plotted) == 9 and np.allclose(
                x_plotted, data.x_dict[1][:, 0])),
            "plotted_y_equals_column_1": bool(np.size(y_plotted) == 9 and np.allclose(
                y_plotted, data.x_dict[1][:, 1]))}


if __name__ == "__main__":
    output, curves = {}, {}
    for version in ("initial", "current"):
        output[version], curves[version] = compare(version)
        output[version]["scatter_2d"] = scatter_check(version)
        print(version, output[version]["scatter_2d"])
    save_results(output, "check_reconstruction")

    figure = go.Figure()
    figure.add_trace(go.Scatter(x=X_GRID[:, 0], y=forrester_hf(X_GRID[:, 0]), name="f2 (true)",
                                line={"color": "black"}))
    styles = {("initial", "trained"): "#d62728", ("initial", "rebuilt"): "#ff9896",
              ("current", "trained"): "#1f77b4", ("current", "rebuilt"): "#aec7e8"}
    for (version, kind), color in styles.items():
        figure.add_trace(go.Scatter(x=X_GRID[:, 0], y=curves[version][kind],
                                    line={"color": color,
                                          "dash": "dash" if kind == "rebuilt" else "solid"},
                                    name=f"{version} code - {kind} model"))
    figure.update_layout(title="Model trained in memory vs model rebuilt from ego_backup.json "
                               "(Forrester, 4 EGO iterations)",
                         xaxis_title="x", yaxis_title="f", template="plotly_white")
    save_figure(figure, "reconstruction_trained_vs_rebuilt")
