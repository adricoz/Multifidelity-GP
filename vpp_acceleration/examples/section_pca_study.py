"""
Choice of the number of modes k of the section PCA (src/section_pca.py), C-foil family.

1. Family and PCA: N sections of CFoilFamily (configuration of the last C-foil study,
   studies/cfoil_intens_kulfan_trim_2), centred PCA, saved in results/section_pca.json.
2. Test sections: other sections of the family (other seed). Geometric error of their projection
   on the k first modes, k = 1..r.
3. Aerodynamic error: NeuralFoil xxxlarge, flow of the study (Re 5e6, n_crit 1, xtr 0.1), alpha
   0, 2, 4, 6 deg, on the test sections NeuralFoil is confident about (confidence >= 0.9 at every
   alpha: the known constraint of the study). |dCL|, |dCD| / CD, |dCm|, |dCpmin| between each
   section and its projection. Cpmin = min over the boundary-layer stations of NeuralFoil
   (x/c >= 1.6 %) of 1 - (ue / V)^2: the suction peak closer to the leading edge is not seen.
4. Choice of k: the smallest k whose 95th percentiles meet TOLERANCES and whose largest
   geometric error meets GEOMETRY_TOLERANCE.
5. Design space of the surrogate: feasible share of the box of the k scores (what the GP would
   see) against the feasible share of the box of the 8 parameters of the family.
6. Generation time of N sections: the loop of the notebook (aerosandbox KulfanAirfoil, then an
   interpolation, section after section) against the matrix product Eq. (1), numpy on the CPU
   and torch on the GPU; and the PCA itself (numpy SVD against torch.pca_lowrank on the GPU, as
   in the notebook).
7. Figures figures/section_pca_errors.html, figures/section_pca_modes.html; every number in
   results/section_pca_study.json.

Run (environment with neuralfoil, plotly and torch with CUDA: the Anaconda one of the notebook):
    C:\\ProgramData\\Anaconda3\\python.exe examples\\section_pca_study.py [--family cfoil8]
        [--sections 10000] [--test 2000] [--figure-only]
    --family cfoil8: CFoilFamily, the 8 parameters of the last C-foil study (default);
    --family smooth16: SmoothKulfanFamily, 16 free weights around the same reference, wiggliness
      at most 2 x the reference, level / slope box 25 % wider (outputs suffixed _smooth16).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import qmc

HERE = Path(__file__).resolve().parent
VPP_DIR = HERE.parent                                # .../Multifidelity-GP-Clean/vpp_acceleration
REPO_DIR = VPP_DIR.parent                            # .../Multifidelity-GP-Clean
for path in (VPP_DIR, REPO_DIR / "benchmarks"):
    if str(path) not in sys.path:
        sys.path.append(str(path))
# pylint: disable=wrong-import-position
from src.section_pca import (CFoilFamily, SectionPCA, SmoothKulfanFamily,  # noqa: E402
                             feasibility_function, kulfan_dict, score_bounds,
                             section_coordinates)

STUDY_CONFIG = REPO_DIR / "studies" / "cfoil_intens_kulfan_trim_2" / "config_cfoil_trim.json"
# outputs of the section PCA studies (2026-10-21): one run folder, runs/README.md
RUN_DIR = VPP_DIR / "runs" / "2026-10-21_section_pca"
RESULTS = RUN_DIR / "results"
FIGURES = RUN_DIR / "figures"
# suffix of the output files of each family, and its name in the figures
SUFFIX = {"cfoil8": "", "smooth16": "_smooth16"}
FAMILY_TEXT = {"cfoil8": "famille C-foil à 8 paramètres",
               "smooth16": "famille C-foil à 16 poids lisses"}

# Flow of the last C-foil study (config_cfoil_trim.json, "flow") and NeuralFoil model of the study
FLOW = {"Re": 5e6, "n_crit": 1.0, "xtr_upper": 0.1, "xtr_lower": 0.1}
MODEL_SIZE = "xxxlarge"
ALPHAS = np.array([0.0, 2.0, 4.0, 6.0])
CONFIDENCE_MIN = 0.9
# Tolerances on the 95th percentile of the gaps (sections x angles). CD in % of CD, the others
# absolute. CD: 0.2 % is the order of the gains of the last optimizations (0.2 to 1 %).
TOLERANCES = {"CL": 0.002, "CD": 0.2, "CM": 0.001, "Cpmin": 0.01}
# Largest geometric gap, fraction of the chord (0.14 mm on the 0.70 m chord of the C-foil)
GEOMETRY_TOLERANCE = 2e-4
COEFFICIENT_LABEL = {"CL": "|ΔCL|", "CD": "|ΔCD| / CD (%)", "CM": "|ΔCm|", "Cpmin": "|ΔCpmin|"}


# 1/3 ---------------------------------------------------------------------------------------------
# Aerodynamics with NeuralFoil
# -------------------------------------------------------------------------------------------------
def neuralfoil_coefficients(weights: np.ndarray, te_thickness: float) -> dict:
    """
    CL, CD, CM, Cpmin and the confidence of NeuralFoil for every section at every angle of
    ALPHAS, in one vectorized call.

    Returns:
    - dict name -> numpy array (m, len(ALPHAS)).
    """
    import neuralfoil as nf  # noqa: PLC0415
    n_sections = len(weights)
    # one row per (section, angle): section 0 at every angle, then section 1...
    repeated = np.repeat(weights, len(ALPHAS), axis=0)
    alpha = np.tile(ALPHAS, n_sections)
    aero = nf.get_aero_from_kulfan_parameters(kulfan_dict(repeated, te_thickness), alpha=alpha,
                                              model_size=MODEL_SIZE, **FLOW)
    n_stations = len(nf.bl_x_points)
    velocity = np.column_stack([aero[f"{side}_bl_ue/vinf_{i}"]
                                for side in ("upper", "lower") for i in range(n_stations)])
    out = {"CL": aero["CL"], "CD": aero["CD"], "CM": aero["CM"],
           "Cpmin": np.min(1.0 - velocity ** 2, axis=1),
           "confidence": aero["analysis_confidence"]}
    return {name: np.asarray(value, dtype=float).reshape(n_sections, len(ALPHAS))
            for name, value in out.items()}


def aerodynamic_gaps(reference: dict, projected: dict) -> dict:
    """Gaps between the coefficients of the sections and of their projections (m, n_alpha)."""
    return {"CL": np.abs(projected["CL"] - reference["CL"]),
            "CD": 100.0 * np.abs(projected["CD"] / reference["CD"] - 1.0),
            "CM": np.abs(projected["CM"] - reference["CM"]),
            "Cpmin": np.abs(projected["Cpmin"] - reference["Cpmin"])}


# 2/3 ---------------------------------------------------------------------------------------------
# Generation time: notebook loop against the matrix product
# -------------------------------------------------------------------------------------------------
def notebook_features(weights: np.ndarray, te_thickness: float, grid: np.ndarray) -> np.ndarray:
    """The features of Dim-Reduction/experiments.ipynb (cell 2): aerosandbox coordinates of each
    section, split at the leading edge, interpolated on the grid, section after section."""
    import aerosandbox as asb  # noqa: PLC0415
    features = []
    for w in weights:
        airfoil = asb.KulfanAirfoil(upper_weights=w[:8], lower_weights=w[8:],
                                    leading_edge_weight=0.0, TE_thickness=te_thickness)
        x, y = airfoil.coordinates[:, 0], airfoil.coordinates[:, 1]
        leading_edge = np.argmin(x)
        x_upper, y_upper = x[:leading_edge + 1][::-1], y[:leading_edge + 1][::-1]
        x_lower, y_lower = x[leading_edge:], y[leading_edge:]
        features.append(np.concatenate([np.interp(grid, x_upper, y_upper),
                                        np.interp(grid, x_lower, y_lower)]))
    return np.array(features)


def best_time(function, repeats: int = 5) -> float:
    """Shortest wall time (s) of `repeats` calls of function()."""
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        times.append(time.perf_counter() - start)
    return min(times)


def generation_timings(pca: SectionPCA, weights: np.ndarray, n_loop: int = 300) -> dict:
    """Wall times (s) to generate the coordinates of len(weights) sections, and of the PCA."""
    n_sections = len(weights)
    out = {"n_sections": n_sections}
    # notebook loop, timed on n_loop sections and scaled to n_sections
    start = time.perf_counter()
    loop_features = notebook_features(weights[:n_loop], pca.te_thickness, pca.stations)
    out["notebook_loop_s"] = (time.perf_counter() - start) * n_sections / n_loop
    # the loop interpolates a cosine-spaced aerosandbox polygon: gap to the exact coordinates
    out["notebook_interpolation_gap"] = float(np.max(np.abs(
        loop_features - section_coordinates(weights[:n_loop], pca.stations, pca.te_thickness))))
    out["numpy_matrix_product_s"] = best_time(
        lambda: section_coordinates(weights, pca.stations, pca.te_thickness))
    coordinates = section_coordinates(weights, pca.stations, pca.te_thickness)
    out["numpy_svd_s"] = best_time(
        lambda: np.linalg.svd(coordinates - coordinates.mean(axis=0), full_matrices=False))
    try:
        import torch  # noqa: PLC0415
        if not torch.cuda.is_available():
            raise ImportError("no CUDA device")
    except ImportError as error:
        out["gpu"] = f"not measured ({error})"
        return out
    out["gpu"] = torch.cuda.get_device_name(0)
    matrix = torch.tensor(pca.matrix.T, device="cuda")
    offset = torch.tensor(pca.offset, device="cuda")

    def gpu_product():
        # end to end: weights to the GPU, product, coordinates back to the CPU
        result = torch.tensor(weights, device="cuda") @ matrix + offset
        return result.cpu().numpy()

    def gpu_pca():
        torch.pca_lowrank(torch.tensor(coordinates, device="cuda"), q=50, center=True)
        torch.cuda.synchronize()

    gpu_product()                                    # warm-up (CUDA context, kernels)
    gpu_pca()
    out["torch_gpu_matrix_product_s"] = best_time(gpu_product)
    out["torch_gpu_pca_lowrank_s"] = best_time(gpu_pca)
    return out


# 3/3 ---------------------------------------------------------------------------------------------
# Study, figures
# -------------------------------------------------------------------------------------------------
def mode_composition(pca: SectionPCA, family: CFoilFamily) -> np.ndarray:
    """Change of the 8 parameters of the family for +1 standard deviation along each mode, as a
    fraction of the range of each parameter: array (r, 8)."""
    mean_parameters = family.weights_to_parameters(pca.mean_weights)[0][0]
    rows = []
    for j in range(pca.rank):
        moved = pca.mean_weights + pca.sigmas[j] * pca.weight_modes[:, j]
        rows.append((family.weights_to_parameters(moved)[0][0] - mean_parameters)
                    / (family.upper - family.lower))
    return np.array(rows)


def smooth_mode_composition(pca: SectionPCA, family: SmoothKulfanFamily) -> np.ndarray:
    """Same for the smooth 16-weight family: change of the 4 level / slope numbers (fraction of
    their box) and of the norm of the second differences of t and of c (fraction of the radius
    sqrt(max_wiggliness)) for +1 standard deviation along each mode: array (r, 6). split is
    linear, so the change is the split of the move sigma_j m_j itself."""
    k = family.n_weights - 2
    radius = np.sqrt(family.max_wiggliness)
    rows = []
    for j in range(pca.rank):
        level_slope, differences = family.split(pca.sigmas[j] * pca.weight_modes[:, j])
        rows.append(np.concatenate([level_slope[0] / (family.upper - family.lower),
                                    [np.linalg.norm(differences[0, :k]) / radius,
                                     np.linalg.norm(differences[0, k:]) / radius]]))
    return np.array(rows)


def make_family(name: str):
    """The family of sections of the study: "cfoil8" (CFoilFamily, the 8 parameters of the
    last C-foil study) or "smooth16" (SmoothKulfanFamily around the same reference: 16 free
    weights, wiggliness at most 2 x the reference, level / slope box 25 % wider)."""
    cfoil = CFoilFamily.from_study_config(STUDY_CONFIG)
    return cfoil if name == "cfoil8" else SmoothKulfanFamily.around(cfoil)


def run_study(family_name: str, n_sections: int, n_test: int) -> dict:
    family = make_family(family_name)
    start = time.perf_counter()
    training = family.sample(n_sections, seed=0)
    pca = SectionPCA(family.te_thickness).fit(training)
    print(f"PCA of {n_sections} sections: rank {pca.rank}, {time.perf_counter() - start:.2f} s")
    if family_name == "cfoil8":
        columns, composition = list(family.PARAMETERS), mode_composition(pca, family)
    else:
        columns = ["niveau t", "pente t", "niveau c", "pente c", "diff. sec. t", "diff. sec. c"]
        composition = smooth_mode_composition(pca, family)
    results = {"family": family_name, "n_sections": n_sections, "rank": pca.rank,
               "variance_ratio": pca.variance_ratio.tolist(), "sigmas": pca.sigmas.tolist(),
               "parameters": columns, "mode_composition": composition.tolist(),
               "lower": family.lower.tolist(), "upper": family.upper.tolist(),
               "tolerances": TOLERANCES, "geometry_tolerance": GEOMETRY_TOLERANCE}

    # 2. geometric error of the projection
    test = family.sample(n_test, seed=1)
    results["geometry"] = {"max": [], "p95": []}
    for k in range(1, pca.rank + 1):
        error = pca.geometric_error(test, k)
        results["geometry"]["max"].append(float(error.max()))
        results["geometry"]["p95"].append(float(np.percentile(error, 95)))

    # 3. aerodynamic error, on the sections NeuralFoil is confident about
    reference = neuralfoil_coefficients(test, family.te_thickness)
    confident = np.all(reference["confidence"] >= CONFIDENCE_MIN, axis=1)
    results["n_test"], results["n_confident"] = n_test, int(confident.sum())
    print(f"test sections: {n_test}, NeuralFoil confident (>= {CONFIDENCE_MIN} at every "
          f"alpha): {confident.sum()}")
    reference = {name: value[confident] for name, value in reference.items()}
    results["aero"] = {name: {"p95": [], "max": []} for name in TOLERANCES}
    for k in range(1, pca.rank + 1):
        projected = neuralfoil_coefficients(pca.reconstruct(test[confident], k),
                                            family.te_thickness)
        for name, gap in aerodynamic_gaps(reference, projected).items():
            results["aero"][name]["p95"].append(float(np.percentile(gap, 95)))
            results["aero"][name]["max"].append(float(gap.max()))

    # 4. choice of k
    chosen = None
    for k in range(1, pca.rank + 1):
        aero_ok = all(results["aero"][name]["p95"][k - 1] <= TOLERANCES[name]
                      for name in TOLERANCES)
        if aero_ok and results["geometry"]["max"][k - 1] <= GEOMETRY_TOLERANCE:
            chosen = k
            break
    results["chosen_k"] = chosen
    pca.n_modes = chosen if chosen is not None else pca.rank
    print(f"saved: {pca.save(RESULTS / f'section_pca{SUFFIX[family_name]}.json')} "
          f"(n_modes = {pca.n_modes})")

    # 5. feasible share of the score box against the box of the native parameters
    n_points = 2 ** 14
    feasible = feasibility_function(pca, family)
    bounds = score_bounds(pca, family)
    unit = qmc.Sobol(d=pca.n_modes, seed=0).random(n_points)
    scores = bounds[:, 0] + unit * (bounds[:, 1] - bounds[:, 0])
    if family_name == "cfoil8":
        unit = qmc.Sobol(d=8, seed=0).random(n_points)
        native = family.parameters_to_weights(family.lower + unit * (family.upper - family.lower))
    else:
        # level / slope box x the cube [-R, R]^12 around the ball of the second differences
        k = family.n_weights - 2
        radius = np.sqrt(family.max_wiggliness)
        unit = qmc.Sobol(d=4 + 2 * k, seed=0).random(n_points)
        level_slope = family.lower + unit[:, :4] * (family.upper - family.lower)
        upper_lower = radius * (2.0 * unit[:, 4:] - 1.0)
        native = family.join(level_slope, np.column_stack([
            upper_lower[:, :k] - upper_lower[:, k:], (upper_lower[:, :k] + upper_lower[:, k:]) / 2]))
    results["feasible_share"] = {"score_box": float(feasible(scores).mean()),
                                 "native_box": float(family.contains(native).mean())}

    # 6. generation time
    results["timings"] = generation_timings(pca, training)
    return results


def make_figures(results: dict) -> None:
    import plotly.graph_objects as go  # noqa: PLC0415
    from figure_style import SERIES, hline, save, style  # noqa: PLC0415
    from plotly.subplots import make_subplots  # noqa: PLC0415

    rank = results["rank"]
    truncated = np.arange(1, rank)           # k = rank is exact (gap 0): not on the log axes
    # --- errors against k ---
    fig = make_subplots(rows=2, cols=3, horizontal_spacing=0.09, vertical_spacing=0.17,
                        subplot_titles=("Part de variance de chaque mode",
                                        "Écart géométrique (fraction de corde)",
                                        COEFFICIENT_LABEL["CL"], COEFFICIENT_LABEL["CD"],
                                        COEFFICIENT_LABEL["CM"], COEFFICIENT_LABEL["Cpmin"]))
    fig.add_trace(go.Bar(x=np.arange(1, rank + 1), y=100 * np.array(results["variance_ratio"]),
                         marker_color=SERIES[0], showlegend=False,
                         hovertemplate="mode %{x} : %{y:.3f} %<extra></extra>"), row=1, col=1)
    fig.update_yaxes(type="log", title_text="% de la variance", row=1, col=1)
    for name, dash, color in (("max", "solid", SERIES[1]), ("p95", "dot", SERIES[0])):
        fig.add_trace(go.Scatter(
            x=truncated, y=results["geometry"][name][:rank - 1], mode="lines+markers",
            name="maximum" if name == "max" else "95e centile", legendgroup=name,
            line={"color": color, "dash": dash},
            hovertemplate="k = %{x} : %{y:.2e} c<extra></extra>"), row=1, col=2)
    def log_range(values, tolerance):
        """Log axis range holding the curves AND the tolerance line (plotly ignores the lines)."""
        low = min(min(v for v in values if v > 0.0), tolerance)
        return [np.log10(low) - 0.3, np.log10(max(values)) + 0.3]

    geometry = results["geometry"]["max"][:rank - 1] + results["geometry"]["p95"][:rank - 1]
    fig.update_yaxes(type="log", range=log_range(geometry, results["geometry_tolerance"]),
                     row=1, col=2)
    hline(fig, results["geometry_tolerance"], "tolérance (max)", row=1, col=2)
    # same order as the subplot titles
    for (row, col), name in zip(((1, 3), (2, 1), (2, 2), (2, 3)), ("CL", "CD", "CM", "Cpmin")):
        for statistic, dash, color in (("max", "solid", SERIES[1]), ("p95", "dot", SERIES[0])):
            fig.add_trace(go.Scatter(
                x=truncated, y=results["aero"][name][statistic][:rank - 1],
                mode="lines+markers", legendgroup=statistic, showlegend=False,
                line={"color": color, "dash": dash},
                hovertemplate="k = %{x} : %{y:.2e}<extra>" + name + "</extra>"),
                row=row, col=col)
        values = results["aero"][name]["max"][:rank - 1] + results["aero"][name]["p95"][:rank - 1]
        fig.update_yaxes(type="log", range=log_range(values, results["tolerances"][name]),
                         row=row, col=col)
        hline(fig, results["tolerances"][name], "tolérance (95e centile)", row=row, col=col)
    for row in (1, 2):
        for col in (1, 2, 3):
            fig.update_xaxes(title_text="mode" if (row, col) == (1, 1) else "nombre de modes k",
                             dtick=1, row=row, col=col)
    chosen = results["chosen_k"]
    family = results.get("family", "cfoil8")
    suffix = SUFFIX[family]
    style(fig, "Section du C-foil : combien de modes PCA faut-il ?",
          f"{FAMILY_TEXT[family][0].upper() + FAMILY_TEXT[family][1:]} "
          f"({results['n_sections']} sections, rang {rank}) ; "
          f"NeuralFoil "
          f"{MODEL_SIZE}, Re 5e6, α 0 à 6°, {results['n_confident']} sections de test fiables ; "
          f"k = {rank} est exact (écart nul) ; k retenu : {chosen}", height=760)
    print(f"figure: {save(fig, FIGURES, 'section_pca_errors' + suffix)}")

    # --- shape of the modes ---
    pca = SectionPCA.load(RESULTS / f"section_pca{suffix}.json")
    n_shown = min(rank, 12)                  # the last modes of a large family are invisible
    x = pca.stations
    n_stations = len(x)
    fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.08, column_widths=[0.36, 0.36, 0.28],
                        subplot_titles=("Cambrure : variation pour +1 écart-type",
                                        "Épaisseur : variation pour +1 écart-type",
                                        "Paramètres déplacés (fraction de leur plage)"))
    palette = SERIES * 2
    for j in range(n_shown):
        field = pca.sigmas[j] * pca.modes[:, j]
        upper, lower = field[:n_stations], field[n_stations:]
        for col, values in ((1, (upper + lower) / 2.0), (2, upper - lower)):
            fig.add_trace(go.Scatter(
                x=x, y=values, mode="lines", name=f"mode {j + 1}", legendgroup=f"mode {j + 1}",
                showlegend=(col == 1), line={"color": palette[j], "width": 2,
                                             "dash": "solid" if j < len(SERIES) else "dot"},
                hovertemplate="x/c %{x:.2f} : %{y:.4f}<extra>mode " + str(j + 1) + "</extra>"),
                row=1, col=col)
        fig.update_xaxes(title_text="x / c", row=1, col=1)
    fig.update_xaxes(title_text="x / c", row=1, col=2)
    fig.update_yaxes(title_text="Δ / c", row=1, col=1)
    composition = np.array(results["mode_composition"])[:n_shown]
    fig.add_trace(go.Heatmap(
        z=composition, x=results["parameters"], y=[f"mode {j + 1}" for j in range(n_shown)],
        colorscale="RdBu", zmid=0.0, showscale=False,
        text=np.round(composition, 2), texttemplate="%{text}",
        hovertemplate="%{y}, %{x} : %{z:.3f}<extra></extra>"), row=1, col=3)
    fig.update_yaxes(autorange="reversed", row=1, col=3)
    style(fig, f"Les modes de la {FAMILY_TEXT[family]}",
          "Chaque mode : variation de la ligne de cambrure et de l'épaisseur pour +1 écart-type, "
          f"et ce qu'il change dans les paramètres de la famille ({n_shown} premiers modes)",
          height=560, legend_rows=2)
    print(f"figure: {save(fig, FIGURES, 'section_pca_modes' + suffix)}")


def summarize(results: dict) -> str:
    lines = [f"rank {results['rank']}, variance per mode (%): "
             + ", ".join(f"{100 * v:.3g}" for v in results["variance_ratio"])]
    for k in range(1, results["rank"] + 1):
        aero = "  ".join(f"{name} {results['aero'][name]['p95'][k - 1]:.2e}"
                         for name in TOLERANCES)
        lines.append(f"k = {k}: geometry max {results['geometry']['max'][k - 1]:.2e} c | "
                     f"p95 {aero}")
    lines.append(f"chosen k: {results['chosen_k']}")
    lines.append(f"feasible share: {results['feasible_share']}")
    lines.append(f"timings: {results['timings']}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=tuple(SUFFIX), default="cfoil8")
    parser.add_argument("--sections", type=int, default=10000)
    parser.add_argument("--test", type=int, default=2000)
    parser.add_argument("--figure-only", action="store_true")
    args = parser.parse_args()
    path = RESULTS / f"section_pca_study{SUFFIX[args.family]}.json"
    if args.figure_only:
        results = json.loads(path.read_text(encoding="utf-8"))
    else:
        results = run_study(args.family, args.sections, args.test)
        RESULTS.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(summarize(results))
    make_figures(results)


if __name__ == "__main__":
    main()
