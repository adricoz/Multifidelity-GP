"""
Study 1 (2026-10-21): how many integration points (N_sobol) and how many candidates does the
VPP-targeted criterion need?

Set-up: a nested start design 128 LF / 32 HF of the meta-surrogate (seed 0), its 6 co-krigings,
and the VPP states they give for the 4 design sections x 5 navigation conditions (20 states,
vpp_study_common). Weights w_k = mean over the true states of (dV/dC_k)^2. Candidates: 768 around
the states, 256 in the whole domain, plus the LF points without HF.

1. Integration. The criterion of every (candidate, level) is computed with n_per_state Sobol
   points around each state (N_int = 20 n_per_state), n_per_state = 2 ... 256, 5 scrambles each,
   against a reference with n_per_state = 512 (10 240 points). Measured: relative error of the
   yields G / C, rank correlation (Spearman), whether the chosen pair is the reference one, and
   the efficiency of the choice: reference yield of the chosen pair / best reference yield.
2. Candidates. A pool of 8 192 candidates (6 144 local, 2 048 global); for M = 64 ... 4 096,
   random subsets (same proportions, 20 draws): best yield of the subset / best of the pool
   (integration with n_per_state = 128).

Run (MDS environment, one process, about 6 min):
    ..\\.venv-mds\\Scripts\\python examples\\vpp_integration_study.py [--figure-only]
Outputs: results/vpp_integration_study.json, figures/vpp_integration_study.html
"""
import argparse
import json
import time

import numpy as np
from scipy.stats import spearmanr

import vpp_study_common as common
from src.sequential_design import batched_imse_gains
from src.vpp_targeted import candidates, integration_cloud, nested_design, vpp_states

N_PER_STATE = [2, 4, 8, 16, 32, 64, 128, 256]
N_REFERENCE = 512
N_SCRAMBLES = 5
CANDIDATE_SIZES = [64, 128, 256, 512, 1024, 2048, 4096]
CHUNK = 256                                       # candidates per call (memory)


def weighted_yields(surrogate, weights, x_cand, missing, x_int) -> np.ndarray:
    """Yields sum_k w_k G_k / C (M, 2), -inf where the level is not missing (by chunks)."""
    gains = np.zeros((len(x_cand), 2))
    for start in range(0, len(x_cand), CHUNK):
        rows = slice(start, start + CHUNK)
        for k, model in enumerate(surrogate.models):
            gains[rows] += weights[k] * batched_imse_gains(model, x_cand[rows], x_int,
                                                           missing[rows])
    cumulative = np.cumsum(missing * surrogate.costs, axis=1)
    return np.where(missing, gains / np.maximum(cumulative, 1e-12), -np.inf)


def run_study() -> dict:
    reference = common.truth()
    meta, vpp = common.MetaFoil(), common.IntensSYVPP()
    surrogate = nested_design(meta, 128, 32, seed=0)
    surrogate.fit()
    states = vpp_states(vpp, surrogate.predict, reference["sections"], common.CONDITIONS)
    points = np.array([s["point"] for s in states])
    weights = np.mean(reference["derivatives"] ** 2, axis=0)
    x_cand, missing = candidates(surrogate, meta, points, 768, 256, seed=1)
    valid = np.isfinite(np.where(missing, 0.0, -np.inf))
    out = {"n_states": len(points), "n_candidates": len(x_cand), "weights": weights.tolist()}

    # 1. integration
    start = time.perf_counter()
    x_ref = integration_cloud(points, N_REFERENCE, seed=12345)
    yields_ref = weighted_yields(surrogate, weights, x_cand, missing, x_ref)
    out["reference_time_s"] = time.perf_counter() - start
    best_ref = np.max(yields_ref[valid])
    chosen_ref = np.argmax(np.where(valid, yields_ref, -np.inf))
    rows = []
    for n_per_state in N_PER_STATE:
        for scramble in range(N_SCRAMBLES):
            start = time.perf_counter()
            x_int = integration_cloud(points, n_per_state, seed=scramble)
            yields = weighted_yields(surrogate, weights, x_cand, missing, x_int)
            elapsed = time.perf_counter() - start
            chosen = np.argmax(np.where(valid, yields, -np.inf))
            rows.append({
                "n_per_state": n_per_state, "n_int": n_per_state * len(points),
                "scramble": scramble, "time_s": elapsed,
                "relative_error": float(np.linalg.norm(yields[valid] - yields_ref[valid])
                                        / np.linalg.norm(yields_ref[valid])),
                "spearman": float(spearmanr(yields[valid], yields_ref[valid])[0]),
                "same_choice": bool(chosen == chosen_ref),
                "efficiency": float(yields_ref.flat[chosen] / best_ref)})
            print(f"n_per_state {n_per_state:4d} scramble {scramble}: rel. error "
                  f"{rows[-1]['relative_error']:.3f}, efficiency {rows[-1]['efficiency']:.3f}, "
                  f"{elapsed:.1f} s", flush=True)
    out["integration"] = rows

    # 2. candidates
    x_pool, missing_pool = candidates(surrogate, meta, points, 6144, 2048, seed=2)
    x_int = integration_cloud(points, 128, seed=7)
    yields_pool = weighted_yields(surrogate, weights, x_pool, missing_pool, x_int)
    best_pool = np.max(yields_pool[np.isfinite(yields_pool)])
    n_local = 6144 // (6144 // len(points)) * (6144 // len(points))     # rows of the local part
    local, glob = np.arange(n_local), np.arange(n_local, n_local + 2048)
    rng = np.random.default_rng(0)
    rows = []
    for size in CANDIDATE_SIZES:
        for draw in range(20):
            subset = np.concatenate([rng.choice(local, int(0.75 * size), replace=False),
                                     rng.choice(glob, size - int(0.75 * size), replace=False)])
            best = np.max(yields_pool[subset][np.isfinite(yields_pool[subset])])
            rows.append({"n_candidates": size, "draw": draw, "efficiency": float(best / best_pool)})
    out["candidates"] = rows
    return out


def make_figure(results: dict) -> None:
    import pandas as pd  # noqa: PLC0415
    import plotly.graph_objects as go  # noqa: PLC0415
    from figure_style import SERIES, hline, save, style  # noqa: PLC0415
    from plotly.subplots import make_subplots  # noqa: PLC0415

    integ = pd.DataFrame(results["integration"])
    cand = pd.DataFrame(results["candidates"])
    fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.08, subplot_titles=(
        "Erreur relative du critère (rendements G / C)",
        "Efficacité du point choisi", "Efficacité selon le nombre de candidats"))
    per = integ.groupby("n_int")
    for col, column, name in ((1, "relative_error", "erreur relative"),
                              (2, "efficiency", "efficacité")):
        stats = per[column].agg(["median", "min", "max"]).reset_index()
        fig.add_trace(go.Scatter(x=stats["n_int"], y=stats["median"], mode="lines+markers",
                                 name=name, showlegend=False, line={"color": SERIES[0]},
                                 hovertemplate="N_int %{x} : %{y:.3f}<extra></extra>"),
                      row=1, col=col)
        fig.add_trace(go.Scatter(
            x=list(stats["n_int"]) + list(stats["n_int"])[::-1],
            y=list(stats["max"]) + list(stats["min"])[::-1], fill="toself", mode="lines",
            line={"width": 0}, fillcolor="rgba(42,120,214,0.15)", showlegend=False,
            hoverinfo="skip"), row=1, col=col)
        fig.update_xaxes(type="log", title_text="points d'intégration N_int (20 états)",
                         row=1, col=col)
    fig.update_yaxes(type="log", row=1, col=1)
    fig.update_yaxes(range=[0, 1.05], row=1, col=2)
    hline(fig, 0.95, "95 %", row=1, col=2)
    stats = cand.groupby("n_candidates")["efficiency"].agg(["median", "min"]).reset_index()
    for column, dash, name in (("median", "solid", "médiane"), ("min", "dot", "pire tirage")):
        fig.add_trace(go.Scatter(x=stats["n_candidates"], y=stats[column], mode="lines+markers",
                                 name=name, line={"color": SERIES[1], "dash": dash},
                                 hovertemplate="M %{x} : %{y:.3f}<extra></extra>"), row=1, col=3)
    fig.update_xaxes(type="log", title_text="nombre de candidats M", row=1, col=3)
    fig.update_yaxes(range=[0, 1.05], row=1, col=3)
    style(fig, "Combien de points d'intégration et de candidats pour le critère ciblé VPP ?",
          f"Méta-surrogate 16D, départ 128 BF / 32 HF, {results['n_states']} états VPP ; "
          f"référence : {N_REFERENCE} points par état ; bandes : min-max sur {N_SCRAMBLES} "
          "tirages Sobol", height=520)
    print(f"figure: {save(fig, common.FIGURES, 'vpp_integration_study')}")


def summarize(results: dict) -> str:
    import pandas as pd  # noqa: PLC0415
    integ = pd.DataFrame(results["integration"])
    table = integ.groupby(["n_per_state", "n_int"]).agg(
        relative_error=("relative_error", "median"), spearman=("spearman", "median"),
        same_choice=("same_choice", "mean"), efficiency_median=("efficiency", "median"),
        efficiency_min=("efficiency", "min"), time_s=("time_s", "median")).reset_index()
    cand = pd.DataFrame(results["candidates"]).groupby("n_candidates")["efficiency"].agg(
        ["median", "min"]).reset_index()
    return (f"reference time {results['reference_time_s']:.1f} s\n" + table.to_string(
        index=False, float_format=lambda v: f"{v:.3f}") + "\n\ncandidates\n"
        + cand.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--figure-only", action="store_true")
    args = parser.parse_args()
    path = common.RESULTS / "vpp_integration_study.json"
    if args.figure_only:
        results = json.loads(path.read_text(encoding="utf-8"))
    else:
        results = run_study()
        path.write_text(json.dumps(results), encoding="utf-8")
    summary = summarize(results)
    print(summary)
    (common.RESULTS / "vpp_integration_summary.txt").write_text(summary + "\n", encoding="utf-8")
    make_figure(results)


if __name__ == "__main__":
    main()
