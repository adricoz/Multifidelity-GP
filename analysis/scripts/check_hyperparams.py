"""
Hyperparameter search bounds, before/after (axis 2 of the report).

For three data sets (Forrester and hydrofoil data saved by the initial code on `main`, and a
fresh Hartmann 6D data set), the initial and the corrected models are fitted and we report:
* the hyperparameters of every level and which ones sit on a bound of the search box;
* the sensitivity to the units of y (y -> a * y, a = 1e-3 and 1e3): a well-posed model gives
  predictions multiplied by a; the error is reported relative to std(prediction);
* for the analytical cases, the accuracy and the calibration of the surrogate.

Usage (from the repository root):  python analysis/scripts/check_hyperparams.py
"""
import json
import subprocess

import numpy as np
import plotly.graph_objects as go
from _common import (ROOT, forrester_hf, hartmann6, hartmann_level, load_mfego, make_data,
                     save_figure, save_results)
from scipy.stats import qmc

INITIAL_BOUNDS = {"lengthscale": (0.01, 5.0), "signal_variance": (1e-3, 50.0),
                  "bias_variance": (1e-6, 1.0), "noise": (1e-8, 1e-5)}


def current_bounds(mod):
    sm = mod.surrogate_models
    return {"lengthscale": sm.LENGTHSCALE_BOUNDS, "signal_variance": sm.SIGNAL_VARIANCE_BOUNDS,
            "bias_variance": sm.BIAS_VARIANCE_BOUNDS, "noise": sm.NOISE_BOUNDS}


def data_from_main(mod, relative_path):
    """X/Y data of a JSON file as committed on `main` by the initial code."""
    content = subprocess.run(["git", "show", f"main:{relative_path}"], cwd=ROOT,
                             capture_output=True, text=True, check=True).stdout
    state = json.loads(content)
    num_levels = len(state["X_dict"])
    data = mod.data_management.ExperimentData(
        bounds=[(0.0, 1.0)] * len(state["X_dict"]["1"][0]), costs=[1.0] * num_levels)
    for l in range(1, num_levels + 1):
        data.x_dict[l] = np.array(state["X_dict"][str(l)], dtype=float)
        data.y_dict[l] = np.array(state["Y_dict"][str(l)], dtype=float)
        data.metrics_dict[l] = [{} for _ in data.y_dict[l]]
    return data


def build(version, case):
    mod = load_mfego(version)
    if case == "forrester (main JSON, 14 LF / 10 HF)":
        data = data_from_main(mod, "mfego/ego_backup.json")
    elif case == "hydrofoil (main JSON, 25 LF / 23 HF)":
        data = data_from_main(mod, "example/hydrofoil_optim/ego_backup.json")
    else:
        data = make_data(mod, [(0.0, 1.0)] * 6, [1.0, 10.0], [20, 15],
                         [lambda x: hartmann_level(x, 1, 0.05), hartmann6])
    return mod, data


def fit(mod, version, data, scale=1.0):
    scaled = mod.data_management.ExperimentData(bounds=data.bounds, costs=data.costs)
    scaled.x_dict = data.x_dict
    scaled.y_dict = {l: scale * y for l, y in data.y_dict.items()}
    scaled.metrics_dict = data.metrics_dict
    np.random.seed(0)
    kwargs = {"seed": 0} if version == "current" else {}
    model = mod.surrogate_models.MultifidelityModel(len(data.y_dict),
                                                    mod.kernels.SquaredExponentialKernel,
                                                    **kwargs)
    model.fit(scaled)
    return model


def predict(model, version, x):
    if version == "current":
        return model.predict_batch(x)[:2]
    out = np.array([model.predict(xi)[:2] for xi in x])
    return out[:, 0], out[:, 1]


def active_bounds(model, bounds):
    """Hyperparameters of each level and the ones within 1 % of a bound."""
    levels = []
    for gp in model.gps:
        params = gp.kernel.get_params()
        values = {f"l_{i + 1}": float(v) for i, v in enumerate(params[:-2])}
        values.update({"signal_variance": float(params[-2]), "bias_variance": float(params[-1]),
                       "noise": float(gp.noise)})
        active = []
        for name, value in values.items():
            low, high = bounds["lengthscale" if name.startswith("l_") else name]
            if value <= low * 1.01:
                active.append(f"{name} = lower bound")
            elif value >= high / 1.01:
                active.append(f"{name} = upper bound")
        levels.append({"values": values, "active": active})
    return levels


if __name__ == "__main__":
    cases = ["forrester (main JSON, 14 LF / 10 HF)", "hydrofoil (main JSON, 25 LF / 23 HF)",
             "hartmann (20 LF / 15 HF, delta = 0.05)"]
    output = {}
    for case in cases:
        output[case] = {}
        for version in ("initial", "current"):
            mod, data = build(version, case)
            bounds = INITIAL_BOUNDS if version == "initial" else current_bounds(mod)
            model = fit(mod, version, data)
            dim = data.x_dict[1].shape[1]
            x_test = qmc.LatinHypercube(d=dim, seed=7).random(500)
            mean_ref, var_ref = predict(model, version, x_test)
            scale_errors = {}
            for scale in (1e-3, 1e3):
                mean_s, _ = predict(fit(mod, version, data, scale), version, x_test)
                scale_errors[str(scale)] = float(np.max(np.abs(mean_s / scale - mean_ref))
                                                 / np.std(mean_ref))
            result = {"levels": active_bounds(model, bounds), "rhos": list(model.rhos),
                      "scale_equivariance_error": scale_errors}
            if case.startswith("forrester") or case.startswith("hartmann"):
                y_test = forrester_hf(x_test[:, 0]) if case.startswith("forrester") \
                    else hartmann6(x_test)
                std = np.sqrt(np.maximum(var_ref, 1e-300))
                result["rmse_rel"] = float(np.sqrt(np.mean((mean_ref - y_test) ** 2))
                                           / np.std(y_test))
                result["coverage_95"] = float(np.mean(np.abs(mean_ref - y_test) <= 1.96 * std))
            output[case][version] = result
            print(case, version, {k: v for k, v in result.items() if k != "levels"},
                  [lvl["active"] for lvl in result["levels"]])
    save_results(output, "check_hyperparams")

    # interactive summary table
    rows = []
    for case, versions in output.items():
        for version, result in versions.items():
            for level, lvl in enumerate(result["levels"], start=1):
                rows.append([case, version, level, ", ".join(lvl["active"]) or "none",
                             f"{max(result['scale_equivariance_error'].values()):.2e}"])
    figure = go.Figure(go.Table(
        header={"values": ["Data set", "Code", "Level", "Hyperparameters on a bound",
                           "Scale error (y x 1e-3 / 1e3)"]},
        cells={"values": list(map(list, zip(*rows)))}))
    figure.update_layout(title="Active bounds of the hyperparameters, initial vs corrected code",
                         height=520)
    save_figure(figure, "hyperparams_active_bounds")
