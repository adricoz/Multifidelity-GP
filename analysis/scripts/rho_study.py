"""
Study of the treatment of rho (step 2 of the plan): fixed rho = 1 (additive model, initial code),
profiled rho (closed form, always estimated) and hybrid mode (fixed until d + 4 points).

Part A - surrogate accuracy vs number of high-fidelity points (no optimization):
    Forrester (L = 2) and Hartmann 6D (L = 2, k = 1 / inf, delta in {0, 0.05, 0.1}).
Part B - NN-MF-EGO convergence:
    Forrester (L = 2, costs 1/10, DOE 10/4) and the protocol of Sacher et al. Sec. 4.1 on
    Hartmann 6D (L = 3, k = 1/3/inf, costs 1/100/1000, nested initial DOE 20/15/10).
Metrics: RMSE, negative log predictive density (NLPD), 95% coverage; for EGO the error
|f(x_hat) - f*| of the surrogate optimum x_hat (metric of the article) and of the best
observed high-fidelity value, as functions of the cumulative cost.

Usage (from the repository root):  python analysis/scripts/rho_study.py [--iters 100 --seeds 5]
"""
import argparse
import json
import logging
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import plotly.graph_objects as go
from _common import (FORRESTER_MIN, H6_MIN, RESULTS_DIR, forrester_hf, forrester_lf,
                     hartmann6, hartmann_level, load_mfego, save_figure, save_results)
from plotly.subplots import make_subplots
from scipy.stats import qmc

MODES = {"fixed rho = 1": {"estimate_rho": False},
         "profiled rho": {"estimate_rho": True, "min_points_rho": 1},
         "hybrid (d + 4)": {"estimate_rho": True}}
DELTAS = (0.0, 0.05, 0.1)


def case_definition(case: str):
    """bounds, level functions, costs, f* and test function of a case."""
    if case == "forrester":
        return ([(0.0, 1.0)], [lambda x: forrester_lf(x[:, 0]), lambda x: forrester_hf(x[:, 0])],
                [1.0, 10.0], FORRESTER_MIN, lambda x: forrester_hf(x[:, 0]))
    delta = float(case.split("=")[1])
    if case.startswith("hartmann2"):
        return ([(0.0, 1.0)] * 6, [lambda x: hartmann_level(x, 1, delta), hartmann6],
                [1.0, 10.0], H6_MIN, hartmann6)
    return ([(0.0, 1.0)] * 6, [lambda x: hartmann_level(x, 1, delta),
                               lambda x: hartmann_level(x, 3, delta), hartmann6],
            [1.0, 100.0, 1000.0], H6_MIN, hartmann6)


def build_data(mod, case, points, seed, nested=False):
    """DOE of a case (independent LHS per level, or nested subsets as in Sacher Sec. 4.1)."""
    bounds, funcs, costs, _, _ = case_definition(case)
    data = mod.data_management.ExperimentData(bounds=bounds, costs=costs)
    data.generate_initial_design(points_per_level=points, seed=seed)
    if nested:
        rng = np.random.default_rng(seed)
        for l in range(2, len(points) + 1):
            subset = rng.choice(len(data.x_dict[l - 1]), points[l - 1], replace=False)
            data.x_dict[l] = data.x_dict[l - 1][subset]
    for l, func in enumerate(funcs, start=1):
        data.y_dict[l] = np.asarray(func(data.x_dict[l]), dtype=float)
        data.metrics_dict[l] = [{} for _ in data.y_dict[l]]
    return data


def accuracy_task(args):
    """Part A: one (case, n_HF, mode, seed) fit."""
    case, n_lf, n_hf, mode, seed = args
    logging.disable(logging.CRITICAL)
    mod = load_mfego("current")
    data = build_data(mod, case, [n_lf, n_hf], seed)
    model = mod.surrogate_models.MultifidelityModel(2, mod.kernels.SquaredExponentialKernel,
                                                    seed=seed, **MODES[mode])
    model.fit(data)
    dim = len(data.bounds)
    x_test = qmc.LatinHypercube(d=dim, seed=999).random(2000 if dim > 1 else 400)
    y_test = case_definition(case)[4](x_test)
    mean, var, _ = model.predict_batch(x_test)
    var = np.maximum(var, 1e-12)
    return {"case": case, "n_hf": n_hf, "mode": mode, "seed": seed, "rho": model.rhos[0],
            "rmse_rel": float(np.sqrt(np.mean((mean - y_test) ** 2)) / np.std(y_test)),
            "nlpd": float(np.mean(0.5 * np.log(2 * np.pi * var)
                                  + (y_test - mean) ** 2 / (2 * var))),
            "coverage": float(np.mean(np.abs(mean - y_test) <= 1.96 * np.sqrt(var)))}


def ego_task(args):
    """Part B: one (case, mode, seed) EGO run, history of errors vs cost."""
    case, mode, seed, n_iters = args
    logging.disable(logging.CRITICAL)
    mod = load_mfego("current")
    bounds, funcs, costs, f_star, f_true = case_definition(case)
    num_levels = len(funcs)
    points = [10, 4] if case == "forrester" else [20, 15, 10][:num_levels]
    data = build_data(mod, case, points, seed, nested=case != "forrester")

    class Simulator(mod.simulator.BaseSimulator):
        def evaluate(self, design_point, level):
            return float(funcs[level - 1](np.atleast_2d(design_point))[0]), {}

    model = mod.surrogate_models.MultifidelityModel(num_levels,
                                                    mod.kernels.SquaredExponentialKernel,
                                                    seed=seed, **MODES[mode])
    acq = mod.acquisition.AcquisitionFunction(model, data)
    workdir = tempfile.mkdtemp()
    ego = mod.optimizer.EGOOptimizer(data, model, Simulator(num_levels), acq,
                                     save_state_path=os.path.join(workdir, "state.json"),
                                     seed=seed)
    history = {"cost": [], "surrogate_error": [], "observed_error": [], "level": [], "rho": []}
    for _ in range(n_iters):
        n_before = [len(data.y_dict[l]) for l in range(1, num_levels + 1)]
        ego.run(n_iterations=1)
        n_after = [len(data.y_dict[l]) for l in range(1, num_levels + 1)]
        new_level = next((l + 1 for l in range(num_levels) if n_after[l] > n_before[l]), 0)
        history["cost"].append(ego.current_total_cost)
        history["surrogate_error"].append(
            float(abs(f_true(np.atleast_2d(acq.x_best))[0] - f_star)))
        history["observed_error"].append(float(abs(ego.best_y_history[-1] - f_star)))
        history["level"].append(new_level)
        history["rho"].append([float(r) for r in model.rhos])
    return {"case": case, "mode": mode, "seed": seed, **history}


def run_parallel(func, tasks):
    with ProcessPoolExecutor(max_workers=min(len(tasks), os.cpu_count() or 1)) as pool:
        return list(pool.map(func, tasks))


def summarize_accuracy(results):
    """Mean of each metric per (case, n_hf, mode)."""
    summary = {}
    for r in results:
        key = (r["case"], r["n_hf"], r["mode"])
        summary.setdefault(key, []).append(r)
    return {f"{c} | n_hf={n} | {m}": {k: float(np.mean([r[k] for r in rs]))
                                      for k in ("rmse_rel", "nlpd", "coverage", "rho")}
            for (c, n, m), rs in summary.items()}


def decision(accuracy_summary, ego_final):
    """Decision rule of the plan: the estimation (hybrid) becomes the default if it improves
    at least half of the cases (> 10 %) without degrading any (> 10 %). Two EGO errors both
    below an absolute tolerance (1e-3 * max(1, |f*|): optimum found) are considered equal."""
    verdicts = {}
    f_star = {"forrester": FORRESTER_MIN}
    cases = sorted({k.split(" | ")[0] for k in ego_final})
    for case in cases:
        final = {m: float(np.median([v["surrogate_error"] for k, v in ego_final.items()
                                     if k.startswith(f"{case} | {m} |")])) for m in MODES}
        tol = 1e-3 * max(1.0, abs(f_star.get(case, H6_MIN)))
        hybrid, fixed = final["hybrid (d + 4)"], final["fixed rho = 1"]
        ratio = (hybrid + 1e-12) / (fixed + 1e-12)
        if hybrid < tol and fixed < tol:
            verdict = "neutral (both converged)"
        else:
            verdict = "improves" if ratio < 0.9 else ("degrades" if ratio > 1.1 else "neutral")
        verdicts[case] = {"median_final_error": final, "hybrid/fixed": float(ratio),
                          "verdict": verdict}
    # surrogate accuracy: mean ratio of the RMSE over the numbers of HF points of a case
    for case in sorted({k.split(" | ")[0] for k in accuracy_summary}):
        keys = sorted({k.rsplit(" | ", 1)[0] for k in accuracy_summary if k.startswith(case)})
        ratio = float(np.mean([accuracy_summary[f"{k} | hybrid (d + 4)"]["rmse_rel"]
                               / accuracy_summary[f"{k} | fixed rho = 1"]["rmse_rel"]
                               for k in keys]))
        verdicts[f"accuracy {case}"] = {"hybrid/fixed rmse (mean over n_hf)": ratio,
                                        "verdict": "improves" if ratio < 0.9 else
                                                   ("degrades" if ratio > 1.1 else "neutral")}
    n_improves = sum(v["verdict"] == "improves" for v in verdicts.values())
    n_degrades = sum(v["verdict"] == "degrades" for v in verdicts.values())
    default = ("estimate_rho=True (hybrid)" if n_improves >= len(verdicts) / 2
               and n_degrades == 0 else "estimate_rho=False (fixed rho = 1)")
    return {"verdicts": verdicts, "n_improves": n_improves, "n_degrades": n_degrades,
            "n_cases": len(verdicts), "default": default}


def ego_final_summary(ego_results):
    """Final values of every EGO run (saved in the JSON results)."""
    return {f"{r['case']} | {r['mode']} | seed {r['seed']}":
            {"cost": r["cost"][-1], "surrogate_error": r["surrogate_error"][-1],
             "observed_error": r["observed_error"][-1], "rho": r["rho"][-1],
             "levels_chosen": np.bincount(r["level"], minlength=4).tolist()}
            for r in ego_results}


def accuracy_figure(accuracy_summary):
    cases = sorted({k.split(" | ")[0] for k in accuracy_summary})
    fig = make_subplots(rows=2, cols=len(cases), subplot_titles=cases, vertical_spacing=0.12)
    for col, case in enumerate(cases, start=1):
        for mode in MODES:
            keys = sorted([k for k in accuracy_summary if k.startswith(case + " |")
                           and k.endswith(mode)], key=lambda k: int(k.split("n_hf=")[1].split()[0]))
            n_hf = [int(k.split("n_hf=")[1].split()[0]) for k in keys]
            for row, metric in ((1, "rmse_rel"), (2, "nlpd")):
                fig.add_trace(go.Scatter(x=n_hf, y=[accuracy_summary[k][metric] for k in keys],
                                         mode="lines+markers", name=mode, legendgroup=mode,
                                         showlegend=row == 1 and col == 1),
                              row=row, col=col)
    fig.update_yaxes(title_text="RMSE / std(f)", row=1, col=1)
    fig.update_yaxes(title_text="NLPD", row=2, col=1)
    fig.update_xaxes(title_text="number of high-fidelity points", row=2)
    fig.update_layout(title="Surrogate accuracy vs n_HF (mean over seeds)",
                      template="plotly_white", height=750)
    return fig


def ego_figure(ego_results):
    cases = sorted({r["case"] for r in ego_results})
    fig = make_subplots(rows=1, cols=len(cases), subplot_titles=cases)
    for col, case in enumerate(cases, start=1):
        for mode in MODES:
            runs = [r for r in ego_results if r["case"] == case and r["mode"] == mode]
            grid = np.linspace(min(r["cost"][0] for r in runs),
                               min(r["cost"][-1] for r in runs), 60)
            curves = np.array([np.interp(grid, r["cost"], r["surrogate_error"]) for r in runs])
            fig.add_trace(go.Scatter(x=grid, y=np.median(curves, axis=0), mode="lines",
                                     name=mode, legendgroup=mode, showlegend=col == 1),
                          row=1, col=col)
    fig.update_yaxes(type="log", title_text="median |f(x_hat) - f*|", row=1, col=1)
    fig.update_yaxes(type="log")
    fig.update_xaxes(title_text="cumulative cost")
    fig.update_layout(title="NN-MF-EGO: error of the surrogate optimum vs cost "
                            "(median over seeds)", template="plotly_white")
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--decision-only", action="store_true",
                        help="recompute the decision from analysis/results/rho_study.json")
    args = parser.parse_args()

    if args.decision_only:
        with open(RESULTS_DIR / "rho_study.json", encoding="utf-8") as f:
            output = json.load(f)
        output["decision"] = decision(output["accuracy_summary"], output["ego_final"])
        print(output["decision"])
        save_results(output, "rho_study")
        raise SystemExit(0)

    seeds = range(args.seeds)
    accuracy_tasks = [("forrester", 10, n, m, s) for n in (3, 4, 5, 6, 8, 10)
                      for m in MODES for s in seeds]
    accuracy_tasks += [(f"hartmann2 delta={d}", 20, n, m, s) for d in DELTAS
                       for n in (6, 8, 10, 15, 20, 30) for m in MODES for s in seeds]
    accuracy = run_parallel(accuracy_task, accuracy_tasks)
    accuracy_summary = summarize_accuracy(accuracy)
    save_figure(accuracy_figure(accuracy_summary), "rho_study_accuracy")

    ego_tasks = [("forrester", m, s, 10) for m in MODES for s in seeds]
    ego_tasks += [(f"hartmann3 delta={d}", m, s, args.iters) for d in DELTAS
                  for m in MODES for s in seeds]
    ego = run_parallel(ego_task, ego_tasks)
    save_figure(ego_figure(ego), "rho_study_ego_convergence")

    output = {"accuracy_summary": accuracy_summary, "ego_final": ego_final_summary(ego)}
    output["decision"] = decision(accuracy_summary, output["ego_final"])
    print(output["decision"])
    save_results(output, "rho_study")
