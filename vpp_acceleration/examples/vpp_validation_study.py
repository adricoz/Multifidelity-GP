"""
Study 4 (2026-10-21): how many CFD runs to VALIDATE the surrogate (CFD = validation only)?

A validation run gives, at one operating point, the true torsor; the surrogate error there gives
the speed error it causes (first order, vpp_study_common). With n validation runs we estimate
(a) the RMS speed error of the surrogate on the operating region and (b) its calibration: the
share of points where the error is within the 95 % interval of the surrogate (coverage), and the
RMS of the standardized error z = error / sigma (1 when calibrated).
How precise are these estimates with n = 5 ... 160 runs?

Set-up: a co-kriging fitted on a nested LHS 256 LF / 64 HF of the meta-surrogate (seed 0). The
operating region: 4 000 points around the 20 true VPP states (section fixed, attitude in the
cloud of vpp_targeted.integration_cloud), each with the speed sensitivities of its condition.
On this population: the true RMS speed error and the true coverage. Then, for each n, 4 000
random validation sets of n points: the spread of the estimated RMS error (90 % interval of
estimate / truth) and of the estimated coverage.
Theory (Gaussian errors): relative precision of an RMS estimate ~ 1.645 / sqrt(2 n) at 90 %;
coverage estimate: standard deviation sqrt(p (1 - p) / n).

Run (MDS environment, one process, about 1 min):
    ..\\.venv-mds\\Scripts\\python examples\\vpp_validation_study.py [--figure-only]
Outputs: results/vpp_validation_study.json, figures/vpp_validation_study.html
"""
import argparse
import json

import numpy as np

import vpp_study_common as common
from src.vpp_targeted import integration_cloud, nested_design

SIZES = [5, 10, 20, 40, 80, 160]
N_DRAWS = 4000


def run_study() -> dict:
    reference = common.truth()
    meta = common.MetaFoil()
    surrogate = nested_design(meta, 256, 64, seed=0)
    surrogate.fit()
    per_state = 200
    points = integration_cloud(reference["points"], per_state, seed=3)
    derivatives = np.repeat(reference["derivatives"], per_state, axis=0)
    errors = surrogate.predict(points)[:, :6] - meta(points, 2)[:, :6]
    speed_errors = np.sum(derivatives * errors, axis=1)                    # signed [kt]
    sigma = np.sqrt(np.sum(derivatives ** 2 * surrogate.variances(points), axis=1))
    inside = np.abs(speed_errors) <= 1.96 * sigma
    true_rms, true_coverage = float(np.sqrt(np.mean(speed_errors ** 2))), float(inside.mean())
    # calibration: RMS of the standardized error z = error / sigma (1 if the surrogate is
    # calibrated, < 1 if it overestimates its uncertainty)
    z = speed_errors / sigma
    true_z_rms = float(np.sqrt(np.mean(z ** 2)))
    rng = np.random.default_rng(0)
    rows = []
    for n in SIZES:
        draws = rng.integers(0, len(points), size=(N_DRAWS, n))
        rms = np.sqrt(np.mean(speed_errors[draws] ** 2, axis=1)) / true_rms
        coverage = inside[draws].mean(axis=1)
        z_rms = np.sqrt(np.mean(z[draws] ** 2, axis=1))
        rows.append({"n": n, "rms_q05": float(np.quantile(rms, 0.05)),
                     "rms_q50": float(np.quantile(rms, 0.5)),
                     "rms_q95": float(np.quantile(rms, 0.95)),
                     "rms_theory_half_width": 1.645 / np.sqrt(2 * n),
                     "z_rms_q05": float(np.quantile(z_rms, 0.05)),
                     "z_rms_q95": float(np.quantile(z_rms, 0.95)),
                     "coverage_q05": float(np.quantile(coverage, 0.05)),
                     "coverage_q95": float(np.quantile(coverage, 0.95)),
                     "coverage_theory_std": float(np.sqrt(true_coverage * (1 - true_coverage)
                                                          / n))})
    return {"true_rms_kt": true_rms, "true_coverage": true_coverage, "true_z_rms": true_z_rms,
            "n_population": len(points), "kurtosis": float(np.mean(speed_errors ** 4)
                                                           / np.mean(speed_errors ** 2) ** 2),
            "sizes": rows}


def make_figure(results: dict) -> None:
    import pandas as pd  # noqa: PLC0415
    import plotly.graph_objects as go  # noqa: PLC0415
    from figure_style import SERIES, hline, save, style  # noqa: PLC0415
    from plotly.subplots import make_subplots  # noqa: PLC0415

    table = pd.DataFrame(results["sizes"])
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.1, subplot_titles=(
        "Erreur RMS de vitesse estimée / vraie (intervalle à 90 %)",
        "Calibration : RMS de z = erreur / σ estimé (intervalle à 90 %)"))
    fig.add_trace(go.Scatter(x=list(table["n"]) + list(table["n"])[::-1],
                             y=list(table["rms_q95"]) + list(table["rms_q05"])[::-1],
                             fill="toself", mode="lines", line={"width": 0},
                             fillcolor="rgba(42,120,214,0.2)", name="méta-problème (90 %)"),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=table["n"], y=1 + table["rms_theory_half_width"], mode="lines",
                             name="théorie gaussienne", line={"color": SERIES[1], "dash": "dot"}),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=table["n"], y=1 - table["rms_theory_half_width"], mode="lines",
                             showlegend=False, line={"color": SERIES[1], "dash": "dot"}),
                  row=1, col=1)
    hline(fig, 1.0, "valeur vraie", row=1, col=1)
    fig.add_trace(go.Scatter(x=list(table["n"]) + list(table["n"])[::-1],
                             y=list(table["z_rms_q95"]) + list(table["z_rms_q05"])[::-1],
                             fill="toself", mode="lines", line={"width": 0},
                             fillcolor="rgba(235,104,52,0.2)", showlegend=False), row=1, col=2)
    hline(fig, results["true_z_rms"], f"vraie ({results['true_z_rms']:.2f})", row=1, col=2)
    hline(fig, 1.0, "calibré (1)", row=1, col=2, position="bottom left")
    for col in (1, 2):
        fig.update_xaxes(type="log", title_text="nombre de calculs de validation n",
                         tickvals=SIZES, row=1, col=col)
    style(fig, "Combien de calculs CFD pour valider le surrogate ?",
          f"Méta-problème : co-krigeage 256 BF / 64 HF, région de fonctionnement autour des 20 "
          f"états VPP ; erreur RMS vraie {results['true_rms_kt']:.3f} nd ; couverture à 95 % vraie "
          f"{results['true_coverage']:.2f}", height=520)
    print(f"figure: {save(fig, common.FIGURES, 'vpp_validation_study')}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--figure-only", action="store_true")
    args = parser.parse_args()
    path = common.RESULTS / "vpp_validation_study.json"
    if args.figure_only:
        results = json.loads(path.read_text(encoding="utf-8"))
    else:
        results = run_study()
        path.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(json.dumps(results, indent=1))
    make_figure(results)


if __name__ == "__main__":
    main()
