"""
STEP 0 - Set-up of the optimization: the level-1 yaw model, then checks (nothing is optimized).

1. Yaw model (study_trim.YawModel): without a 2D lift constraint, only a thin layer of the camber
   space keeps the trimmed yaw inside the +-5 % band, so the camber offset delta is solved from
   a model of the yaw a section needs. The model is fitted on level-1 trims (NPLLT x NeuralFoil,
   cheap) of feasible sections drawn in (t_i, s_i, delta), with wide trim bounds: a first
   space-filling sample, then a second sample concentrated near the band (out-of-sample error of
   the first model on it, then refit on everything), and a final shift that makes it exact at the
   reference. Saved in results/yaw_model_level1.json (read by every later step).
2. Flow settings of every level (turbulence: n_crit, xtr; Re): NeuralFoil inside NPLLT, the
   XFOIL PolarSpec of NPLLT x XFOIL and of AVL x XFOIL, the NeuralFoil confidence constraint.
   They must be identical (study_trim.level_settings raises otherwise).
3. Reference section (MC2_60_6101_1 thickness + frozen camber of the first study): Kulfan fit
   against the .xf file, section properties against aerosandbox (Ixx, Iyy, local thickness),
   delta solved back by the yaw model, every known constraint satisfied.
4. Feasible share of each known constraint and of all of them (Sobol sample), and the feasible
   box (smallest box holding the feasible points plus a margin): bounds proposal.
5. XFOIL robustness: the polars of N feasible sections computed with several settings (alpha
   step, BL iterations niter), in parallel, without the polar cache: share of failed polars,
   points filled by interpolation, wall time per polar.
6. Reference section at the three levels at the nominal yaw (CL_ref, CD, timings): the first
   XFOIL level computes the reference polar (unless it is already in the cache), the second one
   reads it from the cache (same spec), so its reference solve is only the 3D solve.
7. Trim of a few feasible sections at the three levels: trimmed yaw against the band.
8. Figures figures/step0_yaw_model.html, step0_constraints.html, step0_xfoil.html,
   step0_trim.html and every number in results/step0_checks.json.

Run (conda environment bdToolbox, ~20 min, mostly XFOIL):
    C:\\Users\\SIM\\.conda\\envs\\bdToolbox\\python.exe step0_check_setup.py
        [--fit-sections 300 200] [--xfoil-sections 30] [--trim-sections 6] [--workers 8]
        [--skip-xfoil] [--keep-yaw-model]
"""
import argparse
import hashlib
import logging
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import qmc

import study_trim as st
from study_trim import pk
from pipelines.bdtoolbox_foil import bridge, geometry  # pylint: disable=wrong-import-order
from pipelines.bdtoolbox_foil.solvers import xfoil_spec  # pylint: disable=wrong-import-order
from src.data_management import greedy_maximin  # pylint: disable=wrong-import-order

logger = logging.getLogger("step0")

# XFOIL settings compared in the robustness check: (alpha step, niter); npane comes from the
# configuration (the core re-panels with +10 and +15 panels when points do not converge)
XFOIL_SETTINGS = {"step 0.5 / niter 100 (previous runs)": (0.5, 100),
                  "step 0.5 / niter 300": (0.5, 300),
                  "step 0.25 / niter 300": (0.25, 300)}
BOX_MARGIN = 0.03           # margin of the feasible box, fraction of the current range
NEAR_BAND = 0.4             # second fitting sample: predicted yaw within band -+ this (deg)


def oversampling_for(share: float) -> int:
    """Latin hypercube candidates per feasible point wanted (about 4 feasible ones each)."""
    return max(50, int(np.ceil(4.0 / max(share, 1e-3))))


# ------------------------------------------------------------------ 1. yaw model
def level1_yaw(simulator, points) -> np.ndarray:
    """Trimmed yaw at level 1 of normalized points of the fitting problem (NaN if failed)."""
    yaw = []
    for x in points:
        value, metrics = simulator.evaluate(x, 1)
        yaw.append(float(metrics["trim_yaw"]) if np.isfinite(value) else np.nan)
    return np.asarray(yaw)


def fit_yaw_model(problem, n_first: int, n_second: int, seed: int = 5) -> dict:
    """Fit of the level-1 yaw model (module docstring, step 1) and its error estimates."""
    fitting = st.fitting_problem(problem)
    simulator = st.TrimBandSimulator(fitting, st.yaw_band(fitting))
    names = [v.name for v in fitting.variables]
    lower = np.array([v.lower for v in fitting.variables])
    upper = np.array([v.upper for v in fitting.variables])
    nominal = float(problem.attitude["yaw"])
    reference = lower + np.asarray(st.baseline_parameters(fitting)["x"]) * (upper - lower)
    share = st.feasibility_report(fitting, n=2 ** 13)["all"]
    logger.info("Yaw model: feasible share of the fitting space %.3f", share)

    def features(points):
        return lower + np.asarray(points) * (upper - lower)

    t0 = time.perf_counter()
    first = st.feasible_design(fitting, n_first, seed=seed, oversampling=oversampling_for(share))
    yaw_first = level1_yaw(simulator, first)
    ok = np.isfinite(yaw_first)
    model = st.YawModel(names, lower, upper).fit(features(first[ok]), yaw_first[ok])
    model.anchor(reference[None, :], nominal)
    logger.info("Yaw model, first sample: %d sections, %d trimmed, yaw %.2f to %.2f deg (%.0f s)",
                n_first, ok.sum(), np.nanmin(yaw_first), np.nanmax(yaw_first),
                time.perf_counter() - t0)
    # second sample: feasible sections whose predicted yaw is near the band
    lo, hi = st.yaw_band(problem)
    candidates = qmc.LatinHypercube(d=fitting.dim, seed=seed + 1).random(
        oversampling_for(share) * n_second * 4)
    candidates = candidates[st.feasibility_function(fitting)(candidates)]
    predicted = model.predict(features(candidates))
    candidates = candidates[(predicted >= lo - NEAR_BAND) & (predicted <= hi + NEAR_BAND)]
    second = greedy_maximin(candidates, min(n_second, len(candidates)), [(0.0, 1.0)] * fitting.dim)
    yaw_second = level1_yaw(simulator, second)
    ok2 = np.isfinite(yaw_second)
    error_second = model.predict(features(second[ok2])) - yaw_second[ok2]
    # refit on both samples, exact at the reference
    x_all = np.vstack([first[ok], second[ok2]])
    y_all = np.concatenate([yaw_first[ok], yaw_second[ok2]])
    model = st.YawModel(names, lower, upper).fit(features(x_all), y_all)
    model.anchor(reference[None, :], nominal)
    # 10-fold cross-validation of the final procedure (fit + anchor)
    rng = np.random.default_rng(seed)
    folds = rng.integers(0, 10, len(y_all))
    cv = np.zeros(len(y_all))
    for k in range(10):
        test = folds == k
        sub = st.YawModel(names, lower, upper).fit(features(x_all[~test]), y_all[~test])
        sub.anchor(reference[None, :], nominal)
        cv[test] = sub.predict(features(x_all[test])) - y_all[test]
    near = (y_all >= lo - 0.2) & (y_all <= hi + 0.2)
    residual = model.predict(features(x_all)) - y_all
    out = {"model": model.to_dict(), "nominal_yaw": nominal,
           "fitting_bounds": {n: [float(a), float(b)] for n, a, b in zip(names, lower, upper)},
           "n_first": int(n_first), "n_first_trimmed": int(ok.sum()),
           "n_second": int(len(second)), "n_second_trimmed": int(ok2.sum()),
           "second_sample_out_of_sample_rmse_deg": float(np.sqrt(np.mean(error_second ** 2))),
           "second_sample_out_of_sample_max_deg": float(np.max(np.abs(error_second))),
           "cv_rmse_deg": float(np.sqrt(np.mean(cv ** 2))),
           "cv_rmse_near_band_deg": float(np.sqrt(np.mean(cv[near] ** 2))),
           "cv_p95_abs_near_band_deg": float(np.percentile(np.abs(cv[near]), 95)),
           "fit_rmse_deg": float(np.sqrt(np.mean(residual ** 2))),
           "data": {"x": x_all.tolist(), "yaw": y_all.tolist(), "cv_error": cv.tolist(),
                    "sample": [1] * int(ok.sum()) + [2] * int(ok2.sum())},
           "time_s": time.perf_counter() - t0}
    logger.info("Yaw model: second sample out-of-sample RMSE %.3f deg (max %.3f); final 10-fold "
                "CV RMSE %.3f deg, near the band %.3f deg (95 %% of |error| < %.3f deg)",
                out["second_sample_out_of_sample_rmse_deg"],
                out["second_sample_out_of_sample_max_deg"], out["cv_rmse_deg"],
                out["cv_rmse_near_band_deg"], out["cv_p95_abs_near_band_deg"])
    return out


# ------------------------------------------------------------------ 3. reference section
def check_reference(problem, model) -> dict:
    """Fit of MC2, properties against aerosandbox, delta solved back, constraints."""
    import aerosandbox as asb  # noqa: PLC0415  pylint: disable=import-outside-toplevel
    x = np.asarray(st.baseline_parameters(problem)["x"])
    t, c, te = st.kulfan_weights(problem, x, st.REFERENCE_CAMBER)
    airfoil = asb.KulfanAirfoil(upper_weights=(c + t / 2.0)[0], lower_weights=(c - t / 2.0)[0],
                                leading_edge_weight=0.0, TE_thickness=te)
    properties = st.reference_properties(problem)
    aerosandbox = {"Ixx": float(airfoil.Ixx()), "Iyy": float(airfoil.Iyy()),
                   **{f"thickness_{s:.2f}": float(airfoil.local_thickness(x_over_c=s))
                      for s in st.TE_STATIONS},
                   "max_thickness": float(airfoil.max_thickness())}
    gaps = {k: abs(properties[k] / aerosandbox[k] - 1.0) for k in aerosandbox if k in properties}
    delta = float(st.solve_delta(problem, x, model)[0])
    checks = {k: bool(check(x[None, :])[0])
              for k, check in st.constraint_checks(problem, model).items()}
    out = {"physical": st.baseline_parameters(problem)["physical"],
           "fit": st.reference_fit_check(problem), "properties": properties,
           "aerosandbox": aerosandbox, "relative_gap_to_aerosandbox": gaps,
           "delta_solved": delta, "delta_frozen": st.REFERENCE_CAMBER,
           "nf_confidence": float(st.neuralfoil_confidence(problem, x, st.REFERENCE_CAMBER)[0]),
           "constraints_satisfied": checks}
    logger.info("Reference: fit gap %.2e (thickness weights), delta solved %.6f (frozen %.6f)",
                out["fit"]["thickness_weights_gap"], delta, st.REFERENCE_CAMBER)
    logger.info("Reference: Ixx %.4e (aerosandbox %.4e), upper_0 %.4f, lower_0 %.4f, "
                "NeuralFoil confidence %.3f", properties["Ixx"], aerosandbox["Ixx"],
                properties["upper_0"], properties["lower_0"], out["nf_confidence"])
    if max(gaps.values()) > 2e-3 or not all(checks.values()) \
       or abs(delta - st.REFERENCE_CAMBER) > 1e-6:
        raise ValueError(f"reference check failed: gaps {gaps}, constraints {checks}, delta "
                         f"{delta}")
    return out


# ------------------------------------------------------------------ 4. feasible box
def feasible_box(problem, model, n: int = 2 ** 15, seed: int = 0) -> dict:
    """
    Feasible points of a Sobol sample and the smallest box holding them, widened by BOX_MARGIN
    of the current range, clipped to the current bounds, rounded outwards to 1e-3 and
    containing the reference.
    """
    points = qmc.Sobol(problem.dim, seed=seed).random(n)
    feasible = points[st.feasibility_function(problem, model)(points)]
    reference = np.asarray(st.baseline_parameters(problem)["x"])
    lo = np.clip(np.minimum(feasible.min(axis=0), reference) - BOX_MARGIN, 0.0, 1.0)
    hi = np.clip(np.maximum(feasible.max(axis=0), reference) + BOX_MARGIN, 0.0, 1.0)
    box = {}
    for v, a, b in zip(problem.variables, lo, hi):
        lower = np.floor((v.lower + a * (v.upper - v.lower)) * 1000.0) / 1000.0
        upper = np.ceil((v.lower + b * (v.upper - v.lower)) * 1000.0) / 1000.0
        box[v.name] = [float(max(lower, v.lower)), float(min(upper, v.upper))]
    inside = np.all((points >= lo) & (points <= hi), axis=1)
    return {"n_sample": n, "n_feasible": int(len(feasible)),
            "share_current_bounds": float(len(feasible) / n),
            "share_in_box": float(len(feasible) / max(int(inside.sum()), 1)),
            "box_physical": box}


# ------------------------------------------------------------------ 5. XFOIL robustness
def xfoil_robustness(problem, simulator, points: np.ndarray, workers: int,
                     timeout: float) -> dict:
    """
    Polars of the sections of `points` (their solved delta) with every XFOIL_SETTINGS entry
    (run_polar, no cache, temporary folder), `workers` XFOIL processes at a time.
    """
    core = bridge.bdfoil_core()
    npane = int(problem.levels[-1].options.get("npane", 250))
    root = Path(tempfile.mkdtemp(prefix="mfx0_"))
    refs = []
    for x in points:
        path = geometry.write_xf(simulator.section_of(x), root / "sections")
        refs.append(core.model.SectionRef(path=path,
                                          sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    out = {}
    try:
        for label, (step, niter) in XFOIL_SETTINGS.items():
            start, stop, _ = problem.flow["alpha"]
            spec = xfoil_spec(core, {**problem.flow, "alpha": [start, stop, step]}, npane, niter)
            work = root / f"s{len(out)}"

            def run(i, spec=spec, work=work):
                t0 = time.perf_counter()
                try:
                    polar = core.xfoil.run_polar(refs[i], spec, workdir=work / str(i),
                                                 timeout=timeout)
                    return {"ok": True, "filled": int(np.sum(polar.filled)),
                            "npane_used": int(polar.npane_used),
                            "time_s": time.perf_counter() - t0, "error": None}
                except core.model.CoreError as error:
                    return {"ok": False, "filled": None, "npane_used": None,
                            "time_s": time.perf_counter() - t0, "error": str(error)[:300]}

            t0 = time.perf_counter()
            with ThreadPoolExecutor(max_workers=workers) as pool:
                results = list(pool.map(run, range(len(refs))))
            ok = [r for r in results if r["ok"]]
            out[label] = {"alpha_step": step, "niter": niter, "n_points": len(spec.alphas()),
                          "n_sections": len(results), "n_failed": len(results) - len(ok),
                          "failure_share": 1.0 - len(ok) / len(results),
                          "mean_filled": float(np.mean([r["filled"] for r in ok])) if ok
                          else None,
                          "sections_with_filled": int(sum(r["filled"] > 0 for r in ok)),
                          "repanelled": int(sum(r["npane_used"] != npane for r in ok)),
                          "mean_time_s": float(np.mean([r["time_s"] for r in results])),
                          "wall_time_s": time.perf_counter() - t0, "sections": results}
            logger.info("XFOIL %s: %d/%d failed, %.1f filled points per polar, %.1f s per polar",
                        label, out[label]["n_failed"], len(results),
                        out[label]["mean_filled"] or np.nan, out[label]["mean_time_s"])
    finally:
        shutil.rmtree(root, ignore_errors=True)
    return out


# ------------------------------------------------------------------ 6-7. levels and trim
def evaluate_levels(simulator, x) -> dict:
    """Trimmed evaluation of a section at the three levels (with the yaw band)."""
    out = {}
    for level in st.LEVELS:
        value, metrics = simulator.evaluate(x, level)
        out[level] = {"value": float(value), "error": metrics.get("error"),
                      **{k: metrics.get(k) for k in (
                          "CD", "CD_unpenalized", "CL", "CL_ref", "Cdi", "Cdprofile", "trim_yaw",
                          "trim_solves", "trim_in_band", "yaw_excess", "yaw_model", "delta",
                          "cl2d_check_min", "nf_confidence_min", "xfoil_filled_points", "time_s",
                          "time_polar_s")}}
    return out


def reference_at_levels(problem, simulator) -> dict:
    """Lift and drag of the reference section at the nominal attitude, per level (timings)."""
    x = np.asarray(st.baseline_parameters(problem)["x"])
    out = {}
    for level in st.LEVELS:
        t0 = time.perf_counter()
        out[level] = {"CL_ref": float(simulator.lift_reference(level)),
                      "time_s": time.perf_counter() - t0}
    trimmed = evaluate_levels(simulator, x)
    for level in st.LEVELS:
        out[level].update({k: trimmed[level][k] for k in ("CD", "Cdi", "Cdprofile", "trim_yaw",
                                                          "delta", "time_polar_s")})
        logger.info("Reference, %s: CL_ref %.5f, CD %.6f, delta %.6f (%.1f s for the reference "
                    "solve)", st.LEVEL_NAME[level], out[level]["CL_ref"],
                    out[level]["CD"] or np.nan, out[level]["delta"] or np.nan,
                    out[level]["time_s"])
    return out


# ------------------------------------------------------------------ 8. figures
def figure_yaw_model(problem, fit: dict) -> None:
    """Cross-validated prediction of the yaw model against the level-1 trimmed yaw."""
    data = fit["data"]
    yaw, error = np.asarray(data["yaw"]), np.asarray(data["cv_error"])
    sample = np.asarray(data["sample"])
    lo, hi = st.yaw_band(problem)
    fig = go.Figure()
    fig.add_hrect(y0=lo, y1=hi, fillcolor=pk.rgba(pk.MUTED, 0.28), line_width=0, layer="below")
    for k, label in ((1, "space-filling sample"), (2, "sample near the band")):
        mask = sample == k
        fig.add_trace(go.Scatter(
            x=yaw[mask], y=yaw[mask] + error[mask], mode="markers", name=label,
            marker={"color": pk.CATEGORICAL[k - 1][0], "size": 7,
                    "symbol": "circle" if k == 1 else "square",
                    "line": {"color": pk.SURFACE, "width": 1}},
            hovertemplate="trimmed %{x:.3f} deg<br>predicted %{y:.3f} deg<extra></extra>"))
    span = [float(np.min(yaw)), float(np.max(yaw))]
    fig.add_trace(go.Scatter(x=span, y=span, mode="lines", name="perfect prediction",
                             line={"color": pk.MUTED, "width": 1}, hoverinfo="skip"))
    fig.update_xaxes(title_text="trimmed yaw at level 1, NPLLT x NeuralFoil (deg)")
    fig.update_yaxes(title_text="yaw model, 10-fold cross-validated (deg)")
    pk.style(fig, "Level-1 yaw model used to solve the camber offset",
             f"{len(yaw)} trimmed sections; CV RMSE {fit['cv_rmse_deg']:.3f} deg, near the band "
             f"{fit['cv_rmse_near_band_deg']:.3f} deg; out-of-sample RMSE of the first model on "
             f"the second sample {fit['second_sample_out_of_sample_rmse_deg']:.3f} deg; shaded: "
             "band", height=540)
    pk.save(fig, "step0_yaw_model")


def figure_constraints(problem, simulator, shares: dict, points: np.ndarray) -> None:
    """Feasible share per constraint, feasible sections over the reference, leading-edge zoom."""
    fig = make_subplots(rows=1, cols=3, column_widths=[0.3, 0.45, 0.25], horizontal_spacing=0.08,
                        subplot_titles=("Feasible share of the design space",
                                        "Feasible sections (thin) and reference",
                                        "Leading edge"))
    names = list(shares)
    fig.add_trace(go.Bar(x=[100 * shares[k] for k in names], y=names, orientation="h",
                         marker={"color": pk.LEVEL_COLOR[1]}, showlegend=False,
                         hovertemplate="%{y}: %{x:.1f} %<extra></extra>"), 1, 1)
    reference = simulator.section_of(np.asarray(st.baseline_parameters(problem)["x"]))
    shapes = [simulator.section_of(x).coordinates for x in points]
    for col in (2, 3):
        for i, xy in enumerate(shapes):
            fig.add_trace(go.Scatter(x=xy[:, 0], y=xy[:, 1], mode="lines",
                                     line={"color": pk.rgba(pk.MUTED, 0.45), "width": 1},
                                     name="feasible sections", legendgroup="feasible",
                                     showlegend=(col == 2 and i == 0), hoverinfo="skip"), 1, col)
        xy = reference.coordinates
        fig.add_trace(go.Scatter(x=xy[:, 0], y=xy[:, 1], mode="lines",
                                 line={"color": pk.SECONDARY, "width": 2.5},
                                 name=st.BASELINE_NAME, legendgroup="reference",
                                 showlegend=(col == 2),
                                 hovertemplate="x/c %{x:.3f}, y/c %{y:.4f}<extra>reference</extra>"),
                      1, col)
    fig.update_xaxes(title_text="share (%)", row=1, col=1)
    fig.update_xaxes(title_text="x / c", row=1, col=2)
    fig.update_yaxes(title_text="y / c", scaleanchor="x2", scaleratio=1, row=1, col=2)
    fig.update_xaxes(title_text="x / c", range=[-0.005, 0.06], row=1, col=3)
    fig.update_yaxes(range=[-0.05, 0.07], scaleanchor="x3", scaleratio=1, row=1, col=3)
    pk.style(fig, "Known constraints of the section",
             f"{100 * shares['all']:.1f} % of the design space satisfies all of them; "
             f"{len(points)} feasible sections drawn (camber offset solved by the yaw model)",
             height=560)
    pk.save(fig, "step0_constraints")


def figure_xfoil(robustness: dict) -> None:
    """Failed polars, filled points and time per polar for every XFOIL setting."""
    labels = list(robustness)
    fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.1,
                        subplot_titles=("Failed polars (%)", "Filled points per polar",
                                        "Wall time per polar (s)"))
    columns = [[100 * robustness[k]["failure_share"] for k in labels],
               [robustness[k]["mean_filled"] or 0.0 for k in labels],
               [robustness[k]["mean_time_s"] for k in labels]]
    for col, values in enumerate(columns, start=1):
        fig.add_trace(go.Bar(x=labels, y=values, showlegend=False,
                             marker={"color": [pk.CATEGORICAL[i][0] for i in range(len(labels))]},
                             hovertemplate="%{x}: %{y:.2f}<extra></extra>"), 1, col)
    n = robustness[labels[0]]["n_sections"]
    pk.style(fig, "XFOIL robustness on feasible sections",
             f"{n} feasible sections, Re 5e6, n_crit 1, xtr 0.1, alpha -6 to 12 deg "
             "(one ASEQ sweep), npane 250 then re-panelled +10 / +15", height=480, top=110)
    fig.update_xaxes(tickangle=-20)
    pk.save(fig, "step0_xfoil")


def figure_trim(trims: list, band: tuple) -> None:
    """Trimmed yaw of the check sections at each level, with the band and the yaw targets."""
    fig = go.Figure()
    fig.add_hrect(y0=band[0], y1=band[1], fillcolor=pk.rgba(pk.MUTED, 0.28), line_width=0, layer="below")
    x = list(range(1, len(trims) + 1))
    fig.add_trace(go.Scatter(x=x, y=[t[1]["yaw_model"] for t in trims], mode="markers",
                             name="yaw target (model)",
                             marker={"color": pk.PRIMARY, "size": 14, "symbol": "line-ew",
                                     "line": {"color": pk.PRIMARY, "width": 2}},
                             hovertemplate="section %{x}: target %{y:.3f} deg<extra></extra>"))
    for level in st.LEVELS:
        yaw = [t[level]["trim_yaw"] for t in trims]
        fig.add_trace(go.Scatter(
            x=x, y=[np.nan if v is None else v for v in yaw],
            mode="markers", name=st.LEVEL_NAME[level],
            marker={"color": st.LEVEL_COLOR[level], "symbol": st.LEVEL_SYMBOL[level], "size": 10,
                    "line": {"color": pk.SURFACE, "width": 2}},
            customdata=[[t[level]["CD"] or np.nan, t[level]["error"] or ""] for t in trims],
            hovertemplate="section %{x}: yaw %{y:.3f} deg, CD %{customdata[0]:.6f}"
                          "<br>%{customdata[1]}<extra>" + st.LEVEL_NAME[level] + "</extra>"))
    fig.update_xaxes(title_text="check section", dtick=1)
    fig.update_yaxes(title_text="trimmed yaw (deg)")
    pk.style(fig, "Yaw needed for the reference lift, check sections",
             f"shaded: admissible band {band[0]:.2f} to {band[1]:.2f} deg (+-5 % of 4 deg); "
             "dash: yaw target of the section; missing marker = failed evaluation", height=500)
    pk.save(fig, "step0_trim")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fit-sections", type=int, nargs=2, default=(300, 200))
    parser.add_argument("--xfoil-sections", type=int, default=30)
    parser.add_argument("--trim-sections", type=int, default=6)
    parser.add_argument("--workers", type=int, default=8, help="parallel XFOIL processes")
    parser.add_argument("--xfoil-timeout", type=float, default=90.0)
    parser.add_argument("--skip-xfoil", action="store_true", help="no XFOIL robustness check")
    parser.add_argument("--keep-yaw-model", action="store_true",
                        help="reuse results/yaw_model_level1.json instead of fitting it again")
    args = parser.parse_args(argv)

    st.setup_logging("step0")
    problem = st.load_problem()
    out = {}
    if args.keep_yaw_model:
        fit = st.load_json(st.YAW_MODEL_FILE)
    else:
        fit = fit_yaw_model(problem, *args.fit_sections)
        st.save_json(st.YAW_MODEL_FILE, fit)
    out["yaw_model"] = {k: v for k, v in fit.items() if k not in ("data", "model")}
    model = st.load_yaw_model()
    figure_yaw_model(problem, fit)
    simulator = st.make_simulator(problem, model)
    out.update({"settings": st.level_settings(problem, simulator),
                "yaw_band": st.yaw_band(problem), "yaw_penalty_per_deg": st.YAW_PENALTY})
    logger.info("Flow settings per level (identical): %s", out["settings"])
    out["reference"] = check_reference(problem, model)
    out["feasible_share"] = st.feasibility_report(problem, model)
    logger.info("Feasible share: %s", {k: round(v, 4) for k, v in out["feasible_share"].items()})
    out["feasible_box"] = feasible_box(problem, model)
    logger.info("Feasible box (physical, proposed bounds): %s; feasible share in the box %.3f",
                out["feasible_box"]["box_physical"], out["feasible_box"]["share_in_box"])
    oversampling = oversampling_for(out["feasible_share"]["all"])
    drawn = st.feasible_design(problem, 30, seed=1, oversampling=oversampling, model=model)
    figure_constraints(problem, simulator, out["feasible_share"], drawn)
    if not args.skip_xfoil:
        points = st.feasible_design(problem, args.xfoil_sections, seed=2,
                                    oversampling=oversampling, model=model)
        out["xfoil_robustness"] = xfoil_robustness(problem, simulator, points, args.workers,
                                                   args.xfoil_timeout)
        figure_xfoil(out["xfoil_robustness"])
    out["reference_levels"] = reference_at_levels(problem, simulator)
    out["trim_check"] = []
    if args.trim_sections:
        points = st.feasible_design(problem, args.trim_sections, seed=3,
                                    oversampling=oversampling, model=model)
        trims = []
        for i, x in enumerate(points, start=1):
            trims.append(evaluate_levels(simulator, x))
            logger.info("Check section %d: target %.3f, yaw %s, CD %s", i, trims[-1][1][
                "yaw_model"] or np.nan, {lv: trims[-1][lv]["trim_yaw"] for lv in st.LEVELS},
                {lv: trims[-1][lv]["CD"] for lv in st.LEVELS})
        figure_trim(trims, st.yaw_band(problem))
        out["trim_check"] = [{"x": x.tolist(), **{str(k): v for k, v in t.items()}}
                             for x, t in zip(points, trims)]
    st.save_json(st.RESULTS / "step0_checks.json", out)
    logger.info("Checks written in %s", st.RESULTS / "step0_checks.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
