"""
Study 3 (2026-10-21): does the VPP-targeted sequential co-kriging beat, at equal cost, the classic
global ALC and a one-shot nested Latin hypercube, on the error of the boat speed?

Meta-problem of vpp_study_common (16 inputs, levels "NPLLT-like" cost 1 / "AVL-like" cost 10),
4 design sections x 5 navigation conditions. Start: nested LHS START = (LF, HF); budget BUDGET.
Strategies (3 seeds; same start for "vpp" and "global"):
  vpp     VPPTargetedDesign(strategy="vpp"): integration around the VPP states, outputs weighted by
          (dV/dC_k)^2 from the VPP with the current surrogate;
  global  VPPTargetedDesign(strategy="global"): the same loop, integration over the whole domain,
          outputs weighted by the inverse of their variance (classic multi-output ALC);
  lhs     one-shot nested LHS of the same total cost (same LF / HF ratio as the start).
Measured: e_V (first-order speed error at the 20 true states) along the run, and at the end the
true speed error (VPP solved with the surrogate) at the 20 (section, condition).

Run (MDS environment, 3 processes, about 25 min):
    ..\\.venv-mds\\Scripts\\python examples\\vpp_targeted_study.py [--workers 3] [--figure-only]
Outputs: results/vpp_targeted_study.csv, vpp_targeted_curves.csv, vpp_targeted_points.csv,
figures/vpp_targeted_study.html
"""
import argparse
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import vpp_study_common as common
from src.vpp_targeted import N_SCORES, VPPTargetedDesign, nested_design

START = (128, 32)                 # (LF, HF) of the start, cost 448
BUDGET = 150.0                    # sequential budget (total cost 598)
# one-shot nested LHS of the same cost with the best LF / HF ratio of vpp_lhs_study.py (8 LF per
# HF): 264 + 330 = 594
LHS_EQUAL = (264, 33)
SEEDS = (0, 1, 2)
# n_per_state = 16: vpp_integration_study.py (the choice is the reference one from 4 points per
# state); refits and VPP updates every 10 steps to keep the study light
SETTINGS = {"n_per_state": 16, "n_local": 512, "n_global": 256, "refit_every": 10,
            "vpp_every": 10, "weights_every": 3}


def run_task(task):
    strategy, seed = task
    reference = common.truth()
    meta, vpp = common.MetaFoil(), common.IntensSYVPP()
    start = time.perf_counter()
    curve, points = [], []
    if strategy == "lhs":
        surrogate = nested_design(meta, *LHS_EQUAL, seed=seed)
        surrogate.fit()
    else:
        surrogate = nested_design(meta, *START, seed=seed)
        design = VPPTargetedDesign(surrogate, meta, vpp, reference["sections"], common.CONDITIONS,
                                   strategy=strategy, seed=seed, **SETTINGS)
        while BUDGET - design.cost_spent >= 1.0:
            record = design.step(BUDGET - design.cost_spent)
            curve.append({"strategy": strategy, "seed": seed, "iteration": record["iteration"],
                          "cost": surrogate.cost,
                          "speed_error_kt": common.speed_error(surrogate, reference)})
            if record["level"] is None:
                break
            points.append({"strategy": strategy, "seed": seed, "iteration": record["iteration"],
                           "level": record["level"], "near_state": record["near_state"],
                           "attitude_distance": record["attitude_distance"],
                           **{name: value for name, value in zip(
                               ["Cant_Eff", "Rake_Eff", "Yaw_Eff", "Hvol"],
                               record["x"][N_SCORES:])}})
        surrogate.fit(refit=True)
    e_v = common.speed_error(surrogate, reference)
    true_errors = common.true_speed_errors(surrogate, reference, vpp)
    sigma = common.speed_sigma_per_state(surrogate, reference)
    signed = common.speed_error_per_state(surrogate, reference)
    row = {"strategy": strategy, "seed": seed, "cost": surrogate.cost,
           "n_lf": surrogate.sizes[0], "n_hf": surrogate.sizes[1], "speed_error_kt": e_v,
           "true_error_mean_kt": float(np.nanmean(true_errors)),
           "true_error_max_kt": float(np.nanmax(true_errors)),
           "coverage_95": float(np.mean(np.abs(signed) <= 1.96 * sigma)),
           "time_s": time.perf_counter() - start}
    print(f"{strategy:6s} seed {seed}: e_V {e_v:.4f} kt, true |dV| mean "
          f"{row['true_error_mean_kt']:.4f} max {row['true_error_max_kt']:.4f} kt, LF/HF "
          f"{row['n_lf']}/{row['n_hf']}, {row['time_s']:.0f} s", flush=True)
    return row, curve, points


def make_figure(results, curves, points) -> None:
    import plotly.graph_objects as go  # noqa: PLC0415
    from figure_style import SERIES, save, style  # noqa: PLC0415
    from plotly.subplots import make_subplots  # noqa: PLC0415

    reference = common.truth()
    colors = {"vpp": SERIES[0], "global": SERIES[1], "lhs": SERIES[2]}
    labels = {"vpp": "séquentiel ciblé VPP", "global": "séquentiel ALC global",
              "lhs": "LHS imbriqué en un coup"}
    fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.08, column_widths=[0.4, 0.3, 0.3],
                        subplot_titles=("Erreur de vitesse au fil du budget (médiane)",
                                        "Points ajoutés : cant / yaw (ciblé VPP)",
                                        "Points ajoutés : cant / Hvol (ciblé VPP)"))
    start_cost = START[0] + 10 * START[1]
    grid = np.arange(start_cost, start_cost + BUDGET + 1.0)
    for strategy in ("vpp", "global"):
        per_seed = []
        for _, group in curves[curves["strategy"] == strategy].groupby("seed"):
            group = group.sort_values("cost")
            index = np.searchsorted(group["cost"].to_numpy(), grid, side="right") - 1
            per_seed.append(group["speed_error_kt"].to_numpy()[np.maximum(index, 0)])
        if per_seed:
            fig.add_trace(go.Scatter(x=grid, y=np.median(per_seed, axis=0), name=labels[strategy],
                                     line={"color": colors[strategy], "shape": "hv"},
                                     hovertemplate="coût %{x:.0f} : %{y:.4f} nd<extra></extra>"),
                          row=1, col=1)
    lhs = results[results["strategy"] == "lhs"]
    if not lhs.empty:
        fig.add_trace(go.Scatter(x=[lhs["cost"].median()], y=[lhs["speed_error_kt"].median()],
                                 mode="markers", name=labels["lhs"],
                                 marker={"size": 11, "color": colors["lhs"], "symbol": "square"}),
                      row=1, col=1)
    fig.update_xaxes(title_text="coût total (BF = 1, HF = 10)", row=1, col=1)
    fig.update_yaxes(title_text="e_V [nd]", type="log", row=1, col=1)
    states = reference["points"]
    chosen = points[points["strategy"] == "vpp"]
    for col, (x_name, y_name, x_index, y_index) in ((2, ("Cant_Eff", "Yaw_Eff", 12, 14)),
                                                    (3, ("Cant_Eff", "Hvol", 12, 15))):
        for level, symbol in ((1, "circle-open"), (2, "diamond")):
            sub = chosen[chosen["level"] == level]
            fig.add_trace(go.Scatter(x=sub[x_name], y=sub[y_name], mode="markers",
                                     name=f"point ajouté, niveau {level}",
                                     showlegend=(col == 2), legendgroup=f"level{level}",
                                     marker={"color": SERIES[0] if level == 1 else SERIES[3],
                                             "symbol": symbol, "size": 7}), row=1, col=col)
        fig.add_trace(go.Scatter(x=states[:, x_index], y=states[:, y_index], mode="markers",
                                 name="états VPP (vérité)", showlegend=(col == 2),
                                 legendgroup="states",
                                 marker={"color": "black", "symbol": "x", "size": 9}),
                      row=1, col=col)
        fig.update_xaxes(title_text=f"{x_name} [°]", row=1, col=col)
        fig.update_yaxes(title_text=f"{y_name} [{'m' if y_name == 'Hvol' else '°'}]",
                         row=1, col=col)
    style(fig, "Raffinement ciblé VPP contre ALC global et LHS, à coût égal",
          f"Méta-surrogate 16D ; départ {START[0]} BF / {START[1]} HF, budget {BUDGET:.0f} ; "
          f"{len(SEEDS)} graines ; 4 sections x 5 conditions de navigation", height=560,
          legend_rows=2)
    print(f"figure: {save(fig, common.FIGURES, 'vpp_targeted_study')}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--figure-only", action="store_true")
    args = parser.parse_args()
    files = {name: common.RESULTS / f"vpp_targeted_{name}.csv"
             for name in ("study", "curves", "points")}
    if args.figure_only:
        results, curves, points = (pd.read_csv(files[name]) for name in ("study", "curves",
                                                                         "points"))
    else:
        common.truth()
        tasks = [(s, seed) for s in ("vpp", "global") for seed in SEEDS]
        tasks += [("lhs", seed) for seed in SEEDS]
        rows, curve_rows, point_rows = [], [], []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for row, curve, pts in pool.map(run_task, tasks):
                rows.append(row)
                curve_rows += curve
                point_rows += pts
        results, curves, points = (pd.DataFrame(rows), pd.DataFrame(curve_rows),
                                   pd.DataFrame(point_rows))
        for name, frame in (("study", results), ("curves", curves), ("points", points)):
            frame.to_csv(files[name], index=False)
    summary = results.groupby("strategy").agg(
        cost=("cost", "median"), n_lf=("n_lf", "median"), n_hf=("n_hf", "median"),
        speed_error_kt=("speed_error_kt", "median"),
        true_error_mean_kt=("true_error_mean_kt", "median"),
        true_error_max_kt=("true_error_max_kt", "median"),
        coverage_95=("coverage_95", "median"), time_s=("time_s", "median")).reset_index()
    text = summary.to_string(index=False, float_format=lambda v: f"{v:.4f}")
    if not points.empty:
        near = points.groupby("strategy")["attitude_distance"].median()
        text += "\n\nmedian attitude distance of the added points to the nearest state " \
                "(in units of the cloud half-width):\n" + near.to_string()
    print(text)
    (common.RESULTS / "vpp_targeted_summary.txt").write_text(text + "\n", encoding="utf-8")
    make_figure(results, curves, points)


if __name__ == "__main__":
    main()
