"""
STEP 2 - Calibrate the Gaussian process on the step 1 data, BEFORE the optimization.

The step 1 sections were evaluated at the three levels: they are a free test bench for the
surrogate. Three questions:

1. Which hyperparameter estimator? Maximum likelihood (MLE) or MAP with an InvGamma(alpha,
   beta) prior on the lengthscales (inputs in [0, 1]^d): IG(3, 2) (mfego default, mode 0.5),
   IG(3, 1) (mode 0.25, recommended by benchmarks/map_hartmann/RAPPORT_MAP.md "after a check
   on the hydrofoil case": this is that check) and IG(3, 4) (mode 1, advised by the same report
   for a smooth response).
2. Does the multi-fidelity model help? The same L3 points with: L3 only (single fidelity),
   L1 + L3 and L2 + L3 (2 levels: what each cheap level brings on its own), L1 + L2 + L3
   (3 levels, the optimization model). In this test the cheap levels are known at every
   section, so with L2 present L1 cannot add anything: L1 + L3 measures its own value.
3. How accurate and how honest is it? Leave-one-out on the L3 points (each L3 point is
   removed in turn, the cheap levels keep all their points as in an optimization, the model is
   re-fitted and predicts the removed point): RMSE (relative to the spread of the L3 drag),
   NLPD (negative log predictive density: penalizes both errors and wrong uncertainties),
   coverage of the 95 % interval (should be close to 0.95), and a learning curve (L3 points
   needed for a given accuracy, with and without the cheap levels).

4. Large-n check: the MLE fit of mfego is known to degenerate to a white-noise model beyond
   about 80 points per level (benchmarks/map_hartmann/RAPPORT_MAP.md, section 6), and the L1
   level of the optimization goes beyond. 150 extra L1 sections (cheap, saved in
   results/step2_large_n_l1.csv) are fitted by a single GP with every estimator; the fits are
   scored on the step 1 L1 sections and checked for lengthscales at the lower bound.

5. Small-n regime: the optimization starts with few L3 points (initial design of 14): the
   learning curve compares MLE and the best MAP prior from 6 to 30 L3 points (RMSE, NLPD and
   coverage of the 95 % interval).

Choice of the estimator for step 3 (results/step2_choice.json): MAP with the prior of lowest
NLPD, unless the maximum likelihood is SIGNIFICANTLY better (paired Wilcoxon test on the
per-point NLPD of the 3-level model, p < 0.05) AND does not degenerate in the large-n check
AND is not worse than MAP in the small-n regime (median NLPD of the 3-level model with at most
15 L3 points, the regime of the optimization).

Figures: figures/step2_*.html. Run (a few minutes):
    C:\\Users\\SIM\\.conda\\envs\\bdToolbox\\python.exe step2_validate_gp.py
"""
import argparse
import logging
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import wilcoxon

import plotkit as pk
import study_lib as lib
from src.data_management import ExperimentData  # pylint: disable=wrong-import-order
from src.kernels import SquaredExponentialKernel
from src.surrogate_models import MultifidelityModel

logger = logging.getLogger("step2")
ESTIMATORS = {"MLE": (False, None), "MAP IG(3, 2)": (True, (3.0, 2.0)),
              "MAP IG(3, 1)": (True, (3.0, 1.0)), "MAP IG(3, 4)": (True, (3.0, 4.0))}
STRUCTURES = {"L3 only": (3,), "L1 + L3": (1, 3), "L2 + L3": (2, 3), "L1 + L2 + L3": (1, 2, 3)}
# estimators are drawn in ink shades (the categorical hues are reserved for the levels)
ESTIMATOR_STYLE = {"MLE": (pk.MUTED, "circle-open"), "MAP IG(3, 1)": (pk.SECONDARY, "square"),
                   "MAP IG(3, 2)": (pk.PRIMARY, "diamond"),
                   "MAP IG(3, 4)": (pk.PRIMARY, "triangle-up-open")}


def load_step1() -> tuple[np.ndarray, dict]:
    """Normalized design points (n, d) and the drag of every level {level: (n,)} (NaN: fail)."""
    table = pd.read_csv(lib.RESULTS / "step1_calibration.csv")
    x_cols = sorted((c for c in table.columns if c.startswith("x_")), key=lambda c: int(c[2:]))
    designs = table.groupby("design")[x_cols].first().sort_index()
    values = {level: table[table["level"] == level].set_index("design")["value"]
              .reindex(designs.index).to_numpy(dtype=float) for level in (1, 2, 3)}
    return designs.to_numpy(dtype=float), values


def fit(x: np.ndarray, values: dict, levels: tuple, mask_hf: np.ndarray, estimator: str,
        seed: int = 0) -> MultifidelityModel:
    """Model on the given levels; the last level only uses the points of mask_hf."""
    use_map, prior = ESTIMATORS[estimator]
    data = ExperimentData(bounds=[(0.0, 1.0)] * x.shape[1], costs=[1.0] * len(levels))
    for k, level in enumerate(levels, start=1):
        keep = mask_hf if k == len(levels) else np.ones(len(x), dtype=bool)
        data.x_dict[k], data.y_dict[k] = x[keep], values[level][keep]
        data.metrics_dict[k] = [{}] * int(keep.sum())
    model = MultifidelityModel(len(levels), SquaredExponentialKernel, seed=seed,
                               use_map=use_map, lengthscale_prior=prior)
    model.fit(data)
    return model


def pointwise_nlpd(result: dict) -> np.ndarray:
    """Negative log predictive density of every left-out point."""
    y, mean = np.array(result["y"]), np.array(result["mean"])
    var = np.maximum(np.array(result["var"]), 1e-30)
    return 0.5 * np.log(2 * np.pi * var) + (y - mean) ** 2 / (2 * var)


def best_map_prior(loo: dict, structure: str = "L1 + L2 + L3") -> str:
    """MAP prior of lowest leave-one-out NLPD."""
    return min((e for e in ESTIMATORS if ESTIMATORS[e][0]), key=lambda e: loo[structure][e]["nlpd"])


def small_n_nlpd(curve: list, estimator: str, max_hf: int = 15) -> float:
    """Median NLPD of the 3-level model with at most max_hf L3 training points."""
    df = pd.DataFrame(curve)
    sub = df[(df["estimator"] == estimator) & (df["structure"] == "L1 + L2 + L3")
             & (df["n_hf"] <= max_hf)]
    return float(sub["nlpd"].median()) if len(sub) else np.nan


def choose_estimator(loo: dict, large_n: dict, curve: list,
                     structure: str = "L1 + L2 + L3") -> dict:
    """
    MAP prior of lowest NLPD unless MLE is significantly better (paired Wilcoxon), does not
    degenerate with many points (large-n check: no lengthscale at the lower bound and an RMSE
    not worse than the best MAP prior) and is not worse with few L3 points (learning curve).
    """
    best_map = best_map_prior(loo, structure)
    diff = pointwise_nlpd(loo[structure]["MLE"]) - pointwise_nlpd(loo[structure][best_map])
    p_value = float(wilcoxon(diff, alternative="less").pvalue)
    mle_big, map_big = large_n.get("MLE", {}), large_n.get(best_map, {})
    mle_robust = bool(mle_big) and not mle_big["lengthscales_at_lower_bound"] \
        and mle_big["rmse_relative"] <= 1.1 * map_big.get("rmse_relative", np.inf)
    nlpd_small = {"MLE": small_n_nlpd(curve, "MLE"), best_map: small_n_nlpd(curve, best_map)}
    mle_small_ok = not nlpd_small["MLE"] > nlpd_small[best_map]
    chosen = "MLE" if p_value < 0.05 and mle_robust and mle_small_ok else best_map
    use_map, prior = ESTIMATORS[chosen]
    if chosen == "MLE":
        reason = (f"MLE significantly better than {best_map} (Wilcoxon p = {p_value:.3f}), "
                  "robust in the large-n check and not worse with few L3 points")
    elif p_value < 0.05 and not mle_small_ok:
        reason = (f"MLE better on the full step 1 data (p = {p_value:.3f}) but worse with few "
                  f"L3 points (median NLPD {nlpd_small['MLE']:.2f} vs {nlpd_small[best_map]:.2f}"
                  f" for {best_map}, <= 15 L3 points, the regime of the optimization): "
                  f"{best_map} kept")
    elif p_value < 0.05:
        reason = (f"MLE better on the step 1 data (p = {p_value:.3f}) but not robust with many "
                  f"points (large-n check): {best_map} kept")
    else:
        reason = (f"{best_map}: lowest leave-one-out NLPD among the MAP priors ({structure}); "
                  f"MLE not significantly better (p = {p_value:.2f})")
    return {"estimator": chosen, "use_map": use_map,
            "lengthscale_prior": list(prior) if prior else None, "best_map": best_map,
            "mean_nlpd_mle_minus_map": float(np.mean(diff)), "wilcoxon_p_mle_better": p_value,
            "mle_robust_large_n": mle_robust, "small_n_median_nlpd": nlpd_small,
            "reason": reason}


def large_n_check(problem, x_test: np.ndarray, y_test: np.ndarray, n: int = 150) -> dict:
    """Single-level GP of every estimator on n cheap L1 sections, scored on the step 1 ones."""
    path = lib.RESULTS / "step2_large_n_l1.csv"
    if path.is_file():
        table = pd.read_csv(path)
    else:
        simulator = lib.make_simulator(problem)
        points = lib.feasible_design(problem, n, seed=4242)
        values = [simulator.evaluate(x, 1)[0] for x in points]
        table = pd.DataFrame(points, columns=[f"x_{i}" for i in range(problem.dim)])
        table["value"] = values
        table.to_csv(path, index=False)
    table = table[np.isfinite(table["value"])]
    x = table[[f"x_{i}" for i in range(problem.dim)]].to_numpy()
    y = table["value"].to_numpy()
    valid = np.isfinite(y_test)
    out = {}
    for estimator in ESTIMATORS:
        model = fit(x, {1: y, 2: y, 3: y}, (3,), np.ones(len(x), dtype=bool), estimator)
        mean, var, _ = model.predict_batch(x_test[valid])
        diag = model.diagnostics()[0]
        out[estimator] = {"n_train": int(len(y)), **scores(y_test[valid], mean, var,
                                                           float(np.std(y_test[valid]))),
                          "lengthscales": diag["lengthscales"],
                          "lengthscales_at_lower_bound": diag["lengthscales_at_lower_bound"],
                          "noise_at_upper_bound": diag["noise_at_upper_bound"]}
        logger.info("large-n L1 (%d points) %-13s RMSE/spread %.3f NLPD %.3f coverage %.2f "
                    "lengthscales %s", len(y), estimator, out[estimator]["rmse_relative"],
                    out[estimator]["nlpd"], out[estimator]["coverage95"],
                    np.round(diag["lengthscales"], 3).tolist())
    return out


def scores(y: np.ndarray, mean: np.ndarray, var: np.ndarray, spread: float) -> dict:
    """RMSE (absolute and relative to the spread of y), NLPD and 95 % coverage."""
    var = np.maximum(var, 1e-30)
    error = mean - y
    return {"rmse": float(np.sqrt(np.mean(error ** 2))),
            "rmse_relative": float(np.sqrt(np.mean(error ** 2)) / spread),
            "nlpd": float(np.mean(0.5 * np.log(2 * np.pi * var) + error ** 2 / (2 * var))),
            "coverage95": float(np.mean(np.abs(error) <= 1.96 * np.sqrt(var))),
            "mean_sigma_relative": float(np.mean(np.sqrt(var)) / spread)}


def leave_one_out(x, values, levels, estimator) -> dict:
    """LOO on the valid L3 points (cheap levels keep all their points)."""
    valid = np.isfinite(values[3])   # failed cheap evaluations are dropped by the GP itself
    idx = np.flatnonzero(valid)
    mean, var = np.zeros(len(idx)), np.zeros(len(idx))
    for k, i in enumerate(idx):
        mask = valid.copy()
        mask[i] = False
        model = fit(x, values, levels, mask, estimator)
        m, v, _ = model.predict_batch(x[i:i + 1])
        mean[k], var[k] = m[0], v[0]
    y = values[3][idx]
    return {"index": idx.tolist(), "y": y.tolist(), "mean": mean.tolist(), "var": var.tolist(),
            **scores(y, mean, var, float(np.std(y)))}


def learning_curve(x, values, estimators: list, sizes, repeats: int = 5) -> list:
    """Accuracy on the held-out L3 points vs the number of L3 training points."""
    rng = np.random.default_rng(3)
    valid = np.flatnonzero(np.isfinite(values[3]))
    spread = float(np.std(values[3][valid]))
    rows = []
    for n_hf in sizes:
        if n_hf >= len(valid) - 3:
            continue
        for repeat in range(repeats):
            train = rng.choice(valid, n_hf, replace=False)
            test = np.setdiff1d(valid, train)
            mask = np.zeros(len(x), dtype=bool)
            mask[train] = True
            for structure in ("L3 only", "L1 + L2 + L3"):
                for estimator in estimators:
                    model = fit(x, values, STRUCTURES[structure], mask, estimator, seed=repeat)
                    m, v, _ = model.predict_batch(x[test])
                    rows.append({"n_hf": n_hf, "repeat": repeat, "structure": structure,
                                 "estimator": estimator,
                                 **scores(values[3][test], m, v, spread)})
    return rows


# ------------------------------------------------------------------ figures
def figure_loo(loo: dict, estimator: str, problem) -> None:
    fig = make_subplots(rows=1, cols=len(STRUCTURES), horizontal_spacing=0.06,
                        subplot_titles=[f"{s}: RMSE {100 * loo[s][estimator]['rmse_relative']:.0f}"
                                        f" %, coverage "
                                        f"{100 * loo[s][estimator]['coverage95']:.0f} %"
                                        for s in STRUCTURES])
    for col, structure in enumerate(STRUCTURES, start=1):
        r = loo[structure][estimator]
        y, m, s = np.array(r["y"]), np.array(r["mean"]), np.sqrt(np.array(r["var"]))
        fig.add_trace(go.Scatter(
            x=y, y=m, mode="markers", name="L3 point predicted without itself",
            showlegend=col == 1, error_y={"type": "data", "array": 1.96 * s, "thickness": 1,
                                          "color": pk.rgba(pk.LEVEL_COLOR[3], 0.5), "width": 0},
            marker={"color": pk.LEVEL_COLOR[3], "size": 8, "symbol": pk.LEVEL_SYMBOL[3],
                    "line": {"color": pk.SURFACE, "width": 2}}, customdata=r["index"],
            hovertemplate="design %{customdata}<br>L3 CD* %{x:.5f}<br>predicted %{y:.5f}"
                          "<extra></extra>"), row=1, col=col)
        lo, hi = float(min(y.min(), m.min())), float(max(y.max(), m.max()))
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", showlegend=col == 1,
                                 name="perfect prediction", hoverinfo="skip",
                                 line={"color": pk.MUTED, "width": 1}), row=1, col=col)
        fig.update_xaxes(title_text="L3 CD* at equal lift (AVL x XFOIL)", row=1, col=col)
    fig.update_yaxes(title_text="GP prediction +- 1.96 sigma", row=1, col=1)
    pk.style(fig, f"Leave-one-out on the L3 points ({estimator})",
             "Each L3 point is removed, the GP is re-fitted on the others (cheap levels complete) "
             "and predicts it. Points on the diagonal = exact; bars = 95 % intervals.", 540)
    pk.save(fig, "step2_leave_one_out")


def figure_metrics(loo: dict) -> None:
    metrics = (("rmse_relative", "RMSE / spread of the L3 CD*"),
               ("nlpd", "NLPD (lower is better)"), ("coverage95", "95 % interval coverage"))
    fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.08,
                        subplot_titles=[m[1] for m in metrics])
    for col, (key, _) in enumerate(metrics, start=1):
        for estimator, (color, symbol) in ESTIMATOR_STYLE.items():
            fig.add_trace(go.Scatter(
                x=list(STRUCTURES), y=[loo[s][estimator][key] for s in STRUCTURES],
                mode="lines+markers", name=estimator, legendgroup=estimator,
                showlegend=col == 1, line={"color": color, "width": 2},
                marker={"color": color, "size": 10, "symbol": symbol,
                        "line": {"color": pk.SURFACE, "width": 2}},
                hovertemplate=f"{estimator}<br>%{{x}}<br>{key} %{{y:.4f}}<extra></extra>"),
                row=1, col=col)
    fig.add_hline(y=0.95, line={"color": pk.MUTED, "width": 1}, row=1, col=3)
    pk.style(fig, "Which GP? Leave-one-out scores of the L3 predictions",
             "Hyperparameter estimators (MLE / MAP with an InvGamma lengthscale prior) x model "
             "structure (single fidelity L3, 2 levels, 3 levels).", 500)
    pk.save(fig, "step2_gp_scores")


def figure_lengthscales(diagnostics: list, problem, estimator: str) -> None:
    names = [v.name for v in problem.variables]
    fig = go.Figure()
    for diag in diagnostics:
        level = diag["level"]
        fig.add_trace(go.Bar(x=names, y=diag["lengthscales"], name=lib.level_label(problem, level)
                             + ("" if level == 1 else f" (rho {diag['rho']:.3f})"),
                             marker={"color": pk.LEVEL_COLOR[level]},
                             hovertemplate="%{x}: lengthscale %{y:.3f}<extra></extra>"))
    use_map, prior = ESTIMATORS[estimator]
    if use_map:
        mode = prior[1] / (prior[0] + 1)
        fig.add_hline(y=mode, line={"color": pk.MUTED, "width": 1})
        fig.add_annotation(text=f"prior mode {mode:.2f}", x=names[-1], y=mode, xanchor="right",
                           yanchor="bottom", showarrow=False, font={"color": pk.SECONDARY})
    fig.update_yaxes(type="log", title_text="lengthscale (inputs in [0, 1])")
    fig.update_layout(barmode="group", bargap=0.35)
    pk.style(fig, f"Fitted lengthscales per variable and level ({estimator}, all step 1 data)",
             "Small lengthscale = the drag varies quickly with this variable; large = little "
             "influence. Levels 2 and 3 model the discrepancy with the previous level (rho).", 520)
    pk.save(fig, "step2_lengthscales")


def figure_learning(rows: list) -> None:
    df = pd.DataFrame(rows)
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.1,
                        subplot_titles=("RMSE on the other L3 points / spread",
                                        "Coverage of the 95 % interval"))
    for (structure, estimator), group in df.groupby(["structure", "estimator"]):
        color = pk.LEVEL_COLOR[3] if structure == "L1 + L2 + L3" else pk.SECONDARY
        dash = "dash" if estimator == "MLE" else "solid"
        symbol = ("diamond" if structure == "L1 + L2 + L3" else "circle") \
            + ("-open" if estimator == "MLE" else "")
        for col, key in enumerate(("rmse_relative", "coverage95"), start=1):
            sub = group.groupby("n_hf")[key]
            mid, low, high = sub.median(), sub.quantile(0.25), sub.quantile(0.75)
            fig.add_trace(go.Scatter(
                x=mid.index, y=mid.values, mode="lines+markers",
                name=f"{structure}, {estimator}", legendgroup=f"{structure}{estimator}",
                showlegend=col == 1, line={"color": color, "width": 2, "dash": dash},
                marker={"size": 9, "symbol": symbol, "color": color,
                        "line": {"color": pk.SURFACE, "width": 2}},
                error_y={"type": "data", "symmetric": False, "array": (high - mid).values,
                         "arrayminus": (mid - low).values, "color": color, "thickness": 1,
                         "width": 4},
                hovertemplate=f"{structure}, {estimator}<br>%{{x}} L3 points<br>"
                              f"{key} %{{y:.3f}}<extra></extra>"), row=1, col=col)
        fig.update_xaxes(title_text="number of L3 (AVL x XFOIL) training points")
    fig.add_hline(y=0.95, line={"color": pk.MUTED, "width": 1}, row=1, col=2)
    fig.update_yaxes(rangemode="tozero", row=1, col=1)
    pk.style(fig, "How many expensive points? Learning curve, MLE vs MAP",
             "Random L3 training subsets (median and quartiles of the repeats); the "
             "multi-fidelity model also uses every L1 and L2 point. The optimization starts with "
             "14 L3 points.", 500)
    pk.save(fig, "step2_learning_curve")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args(argv)
    lib.setup_logging("step2_validate_gp")
    logging.getLogger("src.surrogate_models").setLevel(logging.ERROR)
    problem = lib.load_problem()
    x, values = load_step1()
    start = time.perf_counter()
    loo = {s: {} for s in STRUCTURES}
    for structure, levels in STRUCTURES.items():
        for estimator in ESTIMATORS:
            loo[structure][estimator] = leave_one_out(x, values, levels, estimator)
            r = loo[structure][estimator]
            logger.info("%-13s %-13s RMSE/spread %.3f  NLPD %.3f  coverage %.2f", structure,
                        estimator, r["rmse_relative"], r["nlpd"], r["coverage95"])
    large_n = large_n_check(problem, x, values[1])
    sizes = [n for n in (6, 10, 15, 20, 30) if n < np.isfinite(values[3]).sum() - 3]
    curve = learning_curve(x, values, ["MLE", best_map_prior(loo)], sizes, args.repeats)
    choice = choose_estimator(loo, large_n, curve)
    best = choice["estimator"]
    logger.info("Chosen estimator: %s", choice)
    full = fit(x, values, (1, 2, 3), np.isfinite(values[3]), best)
    diagnostics = full.diagnostics()
    lib.save_json(lib.RESULTS / "step2_choice.json", choice)
    lib.save_json(lib.RESULTS / "step2_gp_validation.json",
                  {"loo": loo, "choice": choice, "large_n_check": large_n,
                   "full_fit_diagnostics": diagnostics,
                   "learning_curve": curve, "n_designs": int(len(x)),
                   "wall_time_s": time.perf_counter() - start})
    figure_loo(loo, best, problem)
    figure_metrics(loo)
    figure_lengthscales(diagnostics, problem, best)
    if curve:
        figure_learning(curve)
    logger.info("Step 2 done in %.1f min", (time.perf_counter() - start) / 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
