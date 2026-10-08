"""
Theory checks, before/after (axis 1 of the report).

1. Surrogate accuracy of the initial and corrected code on the same data (Forrester, Hartmann).
2. Profile likelihood of rho (Sacher Eq. 15) on Forrester: the additive model rho = 1 of the
   initial code vs the maximum-likelihood rho.
3. Variance reduction of the merit function (Eqs. 21-22, 28-29): exact value vs the initial
   formula var^2 / (var + noise) computed with the noisy variance.
4. Weight of the AEI factor (Eq. 20) and reference value f_best (Eq. 19).

Usage (from the repository root):  python analysis/scripts/check_theory.py
"""
import numpy as np
import plotly.graph_objects as go
from _common import (FORRESTER_MIN, forrester_hf, forrester_lf, hartmann6, hartmann_level,
                     load_mfego, make_data, save_figure, save_results)
from plotly.subplots import make_subplots
from scipy.stats import qmc


def predict_initial(model, x):
    """Initial code: point by point prediction."""
    out = np.array([model.predict(xi)[:2] for xi in x])
    return out[:, 0], out[:, 1]


def surrogate_accuracy() -> tuple[dict, go.Figure]:
    """Same DOE, initial vs corrected model."""
    results = {}
    x_1d = np.linspace(0, 1, 401).reshape(-1, 1)
    x_6d = qmc.LatinHypercube(d=6, seed=123).random(2000)
    cases = {
        "forrester_10LF_4HF": ([(0.0, 1.0)], [10, 4], [lambda x: forrester_lf(x[:, 0]),
                                                      lambda x: forrester_hf(x[:, 0])],
                               x_1d, forrester_hf(x_1d[:, 0])),
        "forrester_10LF_6HF": ([(0.0, 1.0)], [10, 6], [lambda x: forrester_lf(x[:, 0]),
                                                      lambda x: forrester_hf(x[:, 0])],
                               x_1d, forrester_hf(x_1d[:, 0])),
        "hartmann_20LF_10HF_delta0.05": ([(0.0, 1.0)] * 6, [20, 10],
                                         [lambda x: hartmann_level(x, 1, 0.05), hartmann6],
                                         x_6d, hartmann6(x_6d)),
    }
    curves = {}
    for name, (bounds, points, funcs, x_test, y_test) in cases.items():
        results[name] = {}
        for version in ("initial", "current", "current_estimate_rho"):
            mod = load_mfego("initial" if version == "initial" else "current")
            data = make_data(mod, bounds, [1.0, 10.0], points, funcs)
            np.random.seed(0)
            kwargs = {} if version == "initial" else {"seed": 0}
            if version == "current_estimate_rho":
                kwargs["estimate_rho"] = True
            model = mod.surrogate_models.MultifidelityModel(
                2, mod.kernels.SquaredExponentialKernel, **kwargs)
            model.fit(data)
            if version != "initial":
                mean, var, _ = model.predict_batch(x_test)
            else:
                mean, var = predict_initial(model, x_test)
            std = np.sqrt(np.maximum(var, 1e-300))
            results[name][version] = {
                "rho": [float(r) for r in model.rhos],
                "rmse": float(np.sqrt(np.mean((mean - y_test) ** 2))),
                "rmse_relative_to_std": float(np.sqrt(np.mean((mean - y_test) ** 2))
                                              / np.std(y_test)),
                # fraction of test points inside the 95% band (calibration of sigma)
                "coverage_95": float(np.mean(np.abs(mean - y_test) <= 1.96 * std)),
            }
            if name == "forrester_10LF_6HF":
                curves[version] = (mean, std, data)
    print("Surrogate accuracy:", results)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_1d[:, 0], y=forrester_lf(x_1d[:, 0]), name="f1 (LF, true)",
                             line={"color": "grey", "dash": "dot"}))
    fig.add_trace(go.Scatter(x=x_1d[:, 0], y=forrester_hf(x_1d[:, 0]), name="f2 (HF, true)",
                             line={"color": "black"}))
    colors = {"initial": "#d62728", "current": "#2ca02c", "current_estimate_rho": "#1f77b4"}
    for version, (mean, std, data) in curves.items():
        rho = results["forrester_10LF_6HF"][version]["rho"][0]
        fig.add_trace(go.Scatter(x=np.concatenate([x_1d[:, 0], x_1d[::-1, 0]]),
                                 y=np.concatenate([mean + 1.96 * std, (mean - 1.96 * std)[::-1]]),
                                 fill="toself", line={"width": 0}, opacity=0.2,
                                 fillcolor=colors[version], hoverinfo="skip",
                                 name=f"95% band ({version})"))
        fig.add_trace(go.Scatter(x=x_1d[:, 0], y=mean, line={"color": colors[version]},
                                 name=f"MF surrogate {version} code (rho = {rho:.3f})"))
    data = curves["current_estimate_rho"][2]
    for level, symbol in ((1, "circle"), (2, "square")):
        fig.add_trace(go.Scatter(x=data.x_dict[level][:, 0], y=data.y_dict[level],
                                 mode="markers", marker={"symbol": symbol, "size": 10},
                                 name=f"observations level {level}"))
    fig.update_layout(title="Forrester (Eq. 17), 10 LF / 6 HF points: initial vs corrected model",
                      xaxis_title="x", yaxis_title="f", template="plotly_white")
    return results, fig


def rho_profile_likelihood() -> tuple[dict, go.Figure]:
    """NLL(rho) with the kernel hyperparameters re-optimized for every fixed rho."""
    mod = load_mfego("current")
    funcs = [lambda x: forrester_lf(x[:, 0]), lambda x: forrester_hf(x[:, 0])]
    data = make_data(mod, [(0.0, 1.0)], [1.0, 10.0], [10, 6], funcs)
    model = mod.surrogate_models.MultifidelityModel(2, mod.kernels.SquaredExponentialKernel,
                                                    estimate_rho=True, seed=0)
    model.fit(data)
    x_2, y_2 = data.get_training_data(2)
    f_prev = model.predict_batch(x_2, level=1)[0]
    rhos = np.linspace(0.0, 3.5, 36)
    nll = []
    for rho in rhos:
        gp = mod.surrogate_models.GaussianProcess(mod.kernels.SquaredExponentialKernel(), seed=0)
        gp.fit(x_2, y_2, f_prev=f_prev, rho_init=rho, estimate_rho=False)
        log_params = np.log(np.append(gp.kernel.get_params(), gp.noise))
        y_n, f_n = gp._normalize(y_2, f_prev, rho)
        # raw-unit NLL = normalized NLL + n log(y_std) (comparable between values of rho)
        nll.append(gp.negative_log_likelihood(log_params, y_n, f_n, rho, False)
                   + len(y_2) * np.log(gp.y_std))
    nll = np.array(nll)
    results = {"rho_grid_argmin": float(rhos[np.argmin(nll)]), "rho_profiled": model.rhos[0],
               "nll_at_rho_1": float(nll[np.argmin(np.abs(rhos - 1.0))]),
               "nll_min": float(nll.min())}
    fig = go.Figure(go.Scatter(x=rhos, y=nll, mode="lines+markers", name="NLL(rho)"))
    fig.add_vline(x=1.0, line_dash="dash", line_color="red",
                  annotation_text="rho = 1 (initial code)")
    fig.add_vline(x=model.rhos[0], line_dash="dash", line_color="blue",
                  annotation_text=f"profiled rho = {model.rhos[0]:.3f}")
    fig.update_layout(title="Forrester level 2: negative log-likelihood vs rho (Sacher Eq. 15)",
                      xaxis_title="rho", yaxis_title="NLL (raw units)", template="plotly_white")
    print("Profile likelihood:", results)
    return results, fig


def variance_reduction_and_aei() -> tuple[dict, go.Figure]:
    """Exact variance reduction vs initial formula; AEI factor vs noise ratio."""
    ratio = np.logspace(-6, 1, 200)          # noise / latent variance
    s2 = 1.0
    noise = ratio * s2
    exact = s2 ** 2 / (s2 + noise)           # Eqs. 22, 28-29 at x_tilde = x
    noisy_var = s2 + noise                   # what the initial code used as "variance"
    initial = noisy_var ** 2 / (noisy_var + noise)
    aei_factor = 1.0 - np.sqrt(noise) / np.sqrt(noisy_var + noise)
    results = {f"relative_error_at_noise_ratio_{r:g}":
               float(np.interp(np.log10(r), np.log10(ratio), (initial - exact) / exact))
               for r in (1e-6, 1e-3, 1e-2, 0.1, 1.0)}
    results.update({f"aei_factor_at_noise_ratio_{r:g}":
                    float(np.interp(np.log10(r), np.log10(ratio), aei_factor))
                    for r in (1e-6, 1e-3, 1e-2, 0.1, 1.0)})
    fig = make_subplots(rows=1, cols=2, subplot_titles=(
        "Variance reduction: relative error of the initial formula", "AEI factor (Eq. 20)"))
    fig.add_trace(go.Scatter(x=ratio, y=(initial - exact) / exact, name="(initial - exact) / exact"),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=ratio, y=aei_factor, name="1 - tau / sqrt(sigma^2 + tau^2)"),
                  row=1, col=2)
    fig.update_xaxes(type="log", title_text="noise variance / latent variance")
    fig.update_layout(template="plotly_white")
    print("Variance reduction / AEI:", results)
    return results, fig


def reference_value_f_best() -> dict:
    """Eq. 19: min of the HF observations (initial) vs effective best solution (corrected)."""
    mod = load_mfego("current")
    funcs = [lambda x: forrester_lf(x[:, 0]), lambda x: forrester_hf(x[:, 0])]
    data = make_data(mod, [(0.0, 1.0)], [1.0, 10.0], [10, 4], funcs)
    model = mod.surrogate_models.MultifidelityModel(2, mod.kernels.SquaredExponentialKernel,
                                                    seed=0)
    model.fit(data)
    acq = mod.acquisition.AcquisitionFunction(model, data)
    acq.update()
    results = {"min_observed_hf (initial f_best)": float(np.min(data.y_dict[2])),
               "effective_best f_hat_L(x_best) (Eq. 19)": acq.f_best,
               "x_best": acq.x_best.tolist(), "true_minimum": FORRESTER_MIN}
    print("f_best:", results)
    return results


if __name__ == "__main__":
    output = {}
    output["surrogate_accuracy"], figure = surrogate_accuracy()
    save_figure(figure, "theory_forrester_before_after")
    output["rho_profile_likelihood"], figure = rho_profile_likelihood()
    save_figure(figure, "theory_rho_profile_likelihood")
    output["variance_reduction_aei"], figure = variance_reduction_and_aei()
    save_figure(figure, "theory_variance_reduction_aei")
    output["f_best"] = reference_value_f_best()
    save_results(output, "check_theory")
