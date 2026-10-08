"""
Checks of the simulation functions (axis 3 of the report):

1. Hartmann 6D: constants of the initial example vs the standard function (Sacher Eq. 30),
   value at x*, minimum of each variant.
2. Multi-fidelity Hartmann sequence (Eqs. 31-32): correlation between levels.
3. Hydrofoil (NeuralFoil): feasibility map of the root-finding bracket [-5, 15] deg for
   Cl = 1, drag of both levels on a grid, cost per level, and the reference minimum of the
   high-fidelity objective on the actual design bounds (target of the convergence plot).

Usage (from the repository root):  python analysis/scripts/check_simulators.py [--grid 21]
"""
import argparse
import importlib.util
import time

import numpy as np
import plotly.graph_objects as go
from _common import (H6_MIN, H6_XOPT, ROOT, hartmann6, hartmann_level, save_figure,
                     save_results)
from plotly.subplots import make_subplots
from scipy.optimize import minimize
from scipy.stats import qmc


def load_example_module(relative_path: str, name: str):
    """Imports an example module from its path."""
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def hartmann_checks() -> dict:
    """Initial vs corrected Hartmann constants."""
    initial = load_example_module("legacy/legacy_mfego_initial/example/hartmann_6d/Hartmann6d.py",
                                  "h6_initial")
    current = load_example_module("example/hartmann_6d/Hartmann6d.py", "h6_current")
    results = {}
    rng = np.random.default_rng(0)
    starts = rng.random((60, 6))
    for name, module in (("initial", initial), ("current", current)):
        best = min((minimize(module.Hartmann6D, x0, bounds=[(0, 1)] * 6) for x0 in starts),
                   key=lambda r: r.fun)
        results[name] = {"value_at_standard_xopt": float(module.Hartmann6D(H6_XOPT)),
                         "minimum": float(best.fun), "argmin": best.x.round(5).tolist()}
    results["standard"] = {"value_at_standard_xopt": float(hartmann6(H6_XOPT)[0]),
                           "minimum": H6_MIN, "argmin": H6_XOPT.tolist()}
    # the example sequence (level 1: k = 1, delta = 0.05) vs the code of the example
    x_test = rng.random((5, 6))
    results["level1_example_vs_eq31"] = float(np.max(np.abs(
        [current.evaluate_fidelity(x, 1, 2) for x in x_test]
        - hartmann_level(x_test, 1, 0.05))))
    print("Hartmann:", results)
    return results


def hartmann_correlations() -> tuple[dict, go.Figure]:
    """Correlation between the levels of the multi-fidelity Hartmann sequence."""
    x = qmc.LatinHypercube(d=6, seed=0).random(3000)
    f_hf = hartmann6(x)
    fig = make_subplots(rows=1, cols=3, subplot_titles=[f"delta = {d}" for d in (0, 0.05, 0.1)])
    results = {}
    for col, delta in enumerate((0.0, 0.05, 0.1), start=1):
        for k in (1, 3):
            f_k = hartmann_level(x, k, delta)
            results[f"k={k}, delta={delta}"] = float(np.corrcoef(f_k, f_hf)[0, 1])
            fig.add_trace(go.Scattergl(x=f_k[:800], y=f_hf[:800], mode="markers",
                                       marker={"size": 3}, name=f"k={k}, delta={delta}"),
                          row=1, col=col)
    fig.update_xaxes(title_text="f_k (low fidelity)")
    fig.update_yaxes(title_text="f (Hartmann)")
    fig.update_layout(title="Hartmann multi-fidelity sequence (Eqs. 31-32): f_k vs f",
                      template="plotly_white")
    print("Hartmann correlations:", results)
    return results, fig


def hydrofoil_checks(grid: int) -> tuple[dict, go.Figure]:
    """Feasibility map, drag of both levels and reference minimum of the hydrofoil case."""
    import aerosandbox as asb  # pylint: disable=import-outside-toplevel
    import neuralfoil as nf  # pylint: disable=import-outside-toplevel
    foil = load_example_module("example/hydrofoil_optim/optim_neuralfoil.py", "foil")

    def airfoil(x):
        # same mapping as hydrofoil_optim.FunctionSimulator.evaluate
        m_camber = 0.02 + x[0] * (0.09 - 0.02)
        t_thickness = 0.08 + x[1] * (0.17 - 0.08)
        coords = foil.generate_continuous_naca4(m_camber, 0.3, t_thickness, n_points=100)
        return asb.Airfoil(name="naca_custom", coordinates=coords)

    def cl_at(af, alpha):
        aero = nf.get_aero_from_airfoil(airfoil=af, alpha=alpha, Re=5e5, model_size="xxxlarge",
                                        n_crit=1, xtr_upper=0.1, xtr_lower=0.1)
        return float(np.squeeze(aero["CL"]))

    u = np.linspace(0, 1, grid)
    cd = {1: np.full((grid, grid), np.nan), 2: np.full((grid, grid), np.nan)}
    bracket_ok = np.zeros((grid, grid), dtype=bool)
    timings = {1: [], 2: []}
    for i, x0 in enumerate(u):
        for j, x1 in enumerate(u):
            af = airfoil((x0, x1))
            bracket_ok[j, i] = (cl_at(af, -5.0) - 1.0) * (cl_at(af, 15.0) - 1.0) < 0
            for level in (1, 2):
                start = time.perf_counter()
                cd[level][j, i] = foil.objective_function(af, target_cl=1.0, level=level, L=2)[0]
                timings[level].append(time.perf_counter() - start)

    # reference minimum of the high-fidelity objective on the design bounds
    def objective(x):
        value = foil.objective_function(airfoil(np.clip(x, 0, 1)), 1.0, 2, 2)[0]
        return value if np.isfinite(value) else 1.0
    j_best, i_best = np.unravel_index(np.nanargmin(cd[2]), cd[2].shape)
    refined = minimize(objective, [u[i_best], u[j_best]], method="Nelder-Mead",
                       options={"xatol": 1e-4, "fatol": 1e-8})

    results = {
        "grid": grid,
        "failed_hf": int(np.sum(~np.isfinite(cd[2]))),
        "failed_lf": int(np.sum(~np.isfinite(cd[1]))),
        "bracket_without_sign_change": int(np.sum(~bracket_ok)),
        "mean_time_per_eval_s": {str(k): float(np.mean(v)) for k, v in timings.items()},
        "grid_min_hf": float(np.nanmin(cd[2])), "grid_argmin_hf": [u[i_best], u[j_best]],
        "refined_min_hf": float(refined.fun), "refined_argmin_hf": refined.x.tolist(),
        "refined_camber_thickness": [0.02 + refined.x[0] * 0.07, 0.08 + refined.x[1] * 0.09],
        "corr_lf_hf": float(np.corrcoef(cd[1][np.isfinite(cd[1] + cd[2])],
                                        cd[2][np.isfinite(cd[1] + cd[2])])[0, 1]),
        "lf_minus_hf_mean": float(np.nanmean(cd[1] - cd[2])),
    }
    fig = make_subplots(rows=1, cols=3, subplot_titles=(
        "Cd level 1 (xxsmall)", "Cd level 2 (xxxlarge)", "Bracket [-5, 15] deg valid"))
    for col, z in ((1, cd[1]), (2, cd[2]), (3, bracket_ok.astype(float))):
        fig.add_trace(go.Heatmap(x=u, y=u, z=z, colorscale="Viridis", showscale=col != 3,
                                 colorbar={"x": 0.3 * col + 0.02} if col != 3 else None),
                      row=1, col=col)
    fig.add_trace(go.Scatter(x=[refined.x[0]], y=[refined.x[1]], mode="markers",
                             marker={"symbol": "star", "size": 14, "color": "red"},
                             name=f"HF minimum Cd = {refined.fun:.5f}"), row=1, col=2)
    fig.update_xaxes(title_text="x0 (camber 2% -> 9%)")
    fig.update_yaxes(title_text="x1 (thickness 8% -> 17%)")
    fig.update_layout(title="Hydrofoil (NeuralFoil, Re = 5e5, Cl = 1)", template="plotly_white")
    print("Hydrofoil:", results)
    return results, fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=int, default=21)
    parser.add_argument("--skip-hydrofoil", action="store_true")
    args = parser.parse_args()

    output = {"hartmann": hartmann_checks()}
    output["hartmann_correlations"], figure = hartmann_correlations()
    save_figure(figure, "hartmann_mf_correlations")
    if not args.skip_hydrofoil:
        output["hydrofoil"], figure = hydrofoil_checks(args.grid)
        save_figure(figure, "hydrofoil_feasibility_map")
    save_results(output, "check_simulators")
