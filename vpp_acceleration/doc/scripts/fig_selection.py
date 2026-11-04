"""
Figure and numbers of doc/selection_points_vpp.tex (educational note on the VPP-targeted choice of the
new points). Everything is computed with the code of the repository, read-only:
  1. a 1D two-level example (Forrester, as in the mfego theory document Doc-GP-mfego): co-kriging of a
     nested design, ALC yield of every candidate and level, with the integration over the whole
     domain (the target of Le Gratiet & Cannamela) or around a "VPP state" x_s; then a batch of 4
     points chosen by kriging believer;
  2. the real numbers of the third run at its best section, condition 45/16: speed sensitivities
     dV/dC_k, posterior standard deviations s_k of the co-krigings, share of each coefficient in the
     variance of the speed.
Outputs: figures/fig_selection.pdf and figures/valeurs_selection.tex (the numbers quoted in the text).
Run, from vpp_acceleration/doc (MDS environment, for part 2):
    ..\\..\\.venv-mds\\Scripts\\python scripts\\fig_selection.py
"""
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DOC = Path(__file__).resolve().parents[1]
VPP_DIR = DOC.parent
for _path in (VPP_DIR.parent / "mfego", VPP_DIR, VPP_DIR / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
# pylint: disable=wrong-import-position
from src.cokriging import CoKriging  # noqa: E402
from src.data_management import ExperimentData  # noqa: E402
from src.sequential_design import batched_imse_gains  # noqa: E402
from src.vpp_targeted import RHO_PRIOR, latent_mean  # noqa: E402

FIGURES = DOC / "figures"
RUN3 = VPP_DIR / "runs" / "2026-10-22_1519_real_npllt_xfoil_fsi"
COSTS = np.array([1.0, 2.0])                     # the cost ratio of the third run
X_STATE, HALF_WIDTH = 0.88, 0.05                 # the "VPP state" of the 1D example and its cloud
LEVEL_COLORS = ["#1f77b4", "#d62728"]            # as Doc-GP-mfego: BF blue, HF red
TRUTH = "#444444"
VALUES = {}


def fr(value, digits=2) -> str:
    """Number with a French decimal comma, usable in text and math mode."""
    return f"{value:.{digits}f}".replace(".", "{,}")


def forrester_lf(x):
    return 0.5 * (6 * x - 2) ** 2 * np.sin(12 * x - 4) + 10 * (x - 1)


def forrester_hf(x):
    return (6 * x - 2) ** 2 * np.sin(12 * x - 4)


# --- part 1: the 1D example ----------------------------------------------------------------------
def fit(xs, ys) -> CoKriging:
    data = ExperimentData(bounds=[(0.0, 1.0)], costs=list(COSTS))
    data.x_dict = {1: xs[0], 2: xs[1]}
    data.y_dict = {1: ys[0], 2: ys[1]}
    data.metrics_dict = {1: [{}] * len(xs[0]), 2: [{}] * len(xs[1])}
    model = CoKriging(2, priors=RHO_PRIOR, seed=0)
    model.fit(data)
    return model


def yields(model, x_cand, missing, x_int) -> np.ndarray:
    """Gain / cumulative cost of every (candidate, level), -inf where nothing is missing."""
    gains = batched_imse_gains(model, x_cand, x_int, missing)
    cost = np.cumsum(missing * COSTS, axis=1)
    return np.where(missing, gains / np.maximum(cost, 1e-12), -np.inf), gains


def example_1d(axes) -> None:
    x_low = np.linspace(0.0, 1.0, 7)[:, None]
    x_high = x_low[[0, 3, 6]]                                     # nested: 0, 0.5, 1
    xs = [x_low, x_high]
    ys = [forrester_lf(x_low[:, 0]), forrester_hf(x_high[:, 0])]
    model = fit(xs, ys)
    VALUES["rhoExemple"] = fr(model.rhos[0], 2)
    grid = np.linspace(0.0, 1.0, 401)[:, None]
    x_global = np.linspace(0.0, 1.0, 257)[:, None]
    x_local = np.linspace(X_STATE - HALF_WIDTH, X_STATE + HALF_WIDTH, 33)[:, None]
    lf_only = x_low[[1, 2, 4, 5]]                                 # LF points without HF
    cand = np.vstack([grid, lf_only])
    missing = np.vstack([np.ones((len(grid), 2), dtype=bool),
                         np.column_stack([np.zeros(len(lf_only), bool), np.ones(len(lf_only), bool)])])
    n_grid = len(grid)

    # (a) the model
    ax = axes[0]
    mean, var = model.predict_batch(grid, include_noise=False)
    sd = np.sqrt(var)
    ax.axvspan(X_STATE - HALF_WIDTH, X_STATE + HALF_WIDTH, color="#f2c14e", alpha=0.35, lw=0,
               label="voisinage de l'état VPP")
    ax.plot(grid, forrester_hf(grid[:, 0]), "--", color=TRUTH, lw=1, label="HF vraie")
    ax.plot(grid, forrester_lf(grid[:, 0]), ":", color=LEVEL_COLORS[0], lw=1, label="BF vraie")
    ax.plot(grid, mean, color=LEVEL_COLORS[1], lw=1.4, label="co-krigeage (HF)")
    ax.fill_between(grid[:, 0], mean - 2 * sd, mean + 2 * sd, color=LEVEL_COLORS[1], alpha=0.15,
                    lw=0, label=r"$\pm 2\sigma$")
    ax.plot(x_low, ys[0], "o", mfc="white", color=LEVEL_COLORS[0], ms=5, label="points BF")
    ax.plot(x_high, ys[1], "D", color=LEVEL_COLORS[1], ms=5, label="points HF")
    ax.set_title("(a) le modèle", fontsize=9)
    ax.set_xlabel("$x$ (attitude)", fontsize=8)
    ax.set_ylim(-12, 18)
    ax.legend(fontsize=6, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.2), frameon=False)

    # (b) yield of every candidate, global and targeted integration
    ax = axes[1]
    picks = {}
    for name, x_int, style in (("global", x_global, "--"), ("cible", x_local, "-")):
        ratio, _ = yields(model, cand, missing, x_int)
        scale = ratio.max()
        where = "sur tout le domaine" if name == "global" else "autour de l'état"
        for level in (0, 1):
            ax.plot(grid, ratio[:n_grid, level] / scale, style, color=LEVEL_COLORS[level], lw=1.3,
                    label=f"{'BF' if level == 0 else 'HF'}, intégration {where}")
        ax.plot(cand[n_grid:, 0], ratio[n_grid:, 1] / scale, "^", color=LEVEL_COLORS[1],
                mfc="white" if name == "global" else LEVEL_COLORS[1], ms=5,
                label=f"HF sur un point BF existant, {where}")
        index, level = np.unravel_index(np.argmax(ratio), ratio.shape)
        picks[name] = (index, float(cand[index, 0]), int(level) + 1)
        ax.plot(cand[index, 0], 1.0, "*", color=LEVEL_COLORS[level], ms=11,
                mec="black", mew=0.5)
    ax.axvspan(X_STATE - HALF_WIDTH, X_STATE + HALF_WIDTH, color="#f2c14e", alpha=0.35, lw=0)
    ax.set_title("(b) rendement gain / coût", fontsize=9)
    ax.set_xlabel("$x$ du candidat", fontsize=8)
    ax.set_ylabel("rendement / son maximum", fontsize=7)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=6, ncol=1, loc="upper center", bbox_to_anchor=(0.5, -0.2), frameon=False)
    VALUES["xGlobal"], VALUES["niveauGlobal"] = fr(picks["global"][1]), str(picks["global"][2])
    VALUES["xCible"], VALUES["niveauCible"] = fr(picks["cible"][1]), str(picks["cible"][2])
    # variance removed NEAR THE STATE by the point each criterion chooses
    rows = [picks["global"][0], picks["cible"][0]]
    local_gains = batched_imse_gains(model, cand[rows], x_local, missing[rows])
    removed_global = local_gains[0, picks["global"][2] - 1]
    removed_target = local_gains[1, picks["cible"][2] - 1]
    VALUES["partGlobal"] = fr(100 * removed_global / removed_target, 0)
    local_var = np.mean(model.predict_batch(x_local, include_noise=False)[1])
    VALUES["reductionCible"] = fr(100 * removed_target / local_var, 0)

    # (c) a batch of 4, kriging believer, targeted integration
    ax = axes[2]
    xs_f, ys_f = [x.copy() for x in xs], [y.copy() for y in ys]
    missing_b = missing.copy()
    shades = ["#000000", "#555555", "#999999", "#cccccc"]
    batch = []
    for b in range(4):
        ratio, _ = yields(model, cand, missing_b, x_local)
        best_level = np.max(ratio[:n_grid], axis=1)
        ax.plot(grid, np.where(np.isfinite(best_level), best_level, np.nan), color=shades[b],
                lw=1.1, label=f"avant le choix {b + 1}")
        VALUES[f"rendementAvant{'ABCD'[b]}"] = fr(np.nanmax(np.where(np.isfinite(best_level),
                                                                     best_level, np.nan)), 2)
        index, level = np.unravel_index(np.argmax(ratio), ratio.shape)
        x_b, l_b = cand[index], int(level) + 1
        batch.append((float(x_b[0]), l_b))
        ax.plot(x_b[0], ratio[index, level], "o" if l_b == 1 else "D", color=LEVEL_COLORS[l_b - 1],
                ms=5, mec="black", mew=0.4)
        ax.annotate(f"{b + 1}", (x_b[0], ratio[index, level]), textcoords="offset points",
                    xytext=(5, 3) if b != 3 else (-10, 3), fontsize=7)
        for j in range(1, l_b + 1):
            if missing_b[index, j - 1]:
                xs_f[j - 1] = np.vstack([xs_f[j - 1], x_b[None]])
                ys_f[j - 1] = np.append(ys_f[j - 1], latent_mean(model, x_b[None], j))
        missing_b[index, :l_b] = False
        model.condition(xs_f, ys_f)                     # hyperparameters unchanged
    ax.axvspan(X_STATE - HALF_WIDTH, X_STATE + HALF_WIDTH, color="#f2c14e", alpha=0.35, lw=0)
    ax.set_title("(c) un lot de 4, ciblé", fontsize=9)
    ax.set_xlabel("$x$ du candidat", fontsize=8)
    ax.set_ylabel("rendement (meilleur niveau)", fontsize=7)
    ax.set_xlim(0.6, 1.0)
    ax.legend(fontsize=6, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.2), frameon=False)
    VALUES["lot"] = ", ".join(f"{fr(x)}\\,({'BF' if l == 1 else 'HF'})" for x, l in batch)
    for ax in axes:
        ax.tick_params(labelsize=7)


# --- part 2: the real numbers of the third run -----------------------------------------------------
def run3_numbers() -> None:
    from profile_analysis import load_surrogate, stiffness_ratios  # noqa: PLC0415
    from src.section_pca import SectionPCA  # noqa: PLC0415
    from src.vpp_coupling import FSICorrection, IntensSYVPP  # noqa: PLC0415
    config = json.loads((RUN3 / "config.json").read_text(encoding="utf-8"))
    pca = SectionPCA.load(RUN3 / "data" / "pca.json")
    surrogate = load_surrogate(RUN3)
    scores = np.array(json.loads((RUN3 / "results" / "best_section.json")
                                 .read_text(encoding="utf-8"))["scores"])
    ratio = float(stiffness_ratios(config, pca, scores[None])[0])
    vpp = IntensSYVPP(foil="CFoilFSI", with_hvol=False, fsi=FSICorrection())
    vpp.set_foil(surrogate.predict, scores, ratio)
    state = vpp.solve(45.0, 16.0)
    d = vpp.speed_sensitivities(surrogate.predict, scores, 45.0, 16.0, state, stiffness_ratio=ratio)
    point = np.concatenate([scores, [state["Cant_Eff"], state["Rake_Eff"], state["Yaw_Eff"]]])[None]
    mu = surrogate.predict(point)[0, :6]
    s = np.sqrt(surrogate.variances(point)[0, :6])
    contribution = (d * s) ** 2
    VALUES["vitesseEtat"] = fr(state["speed"], 2)
    rows = []
    for k, name in enumerate(["F_x", "F_y", "F_z", "M_x", "M_y", "M_z"]):
        rows.append(f"${name}$ & {fr(mu[k], 4)} & {fr(d[k], 1)} & "
                    f"{fr(1e4 * s[k], 2)} & {fr(100 * s[k] / abs(mu[k]), 3)} & "
                    f"{fr(1e3 * abs(d[k]) * s[k], 2)} & {fr(100 * contribution[k] / contribution.sum(), 1)} \\\\")
    VALUES["tableSensibilites"] = "\n".join(rows)
    VALUES["sigmaVitesse"] = fr(1e3 * np.sqrt(contribution.sum()), 1)


def main():
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 3.25), constrained_layout=True)
    example_1d(axes)
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / "fig_selection.pdf")
    run3_numbers()
    lines = ["% Generated by scripts/fig_selection.py: do not edit by hand"]
    lines += [f"\\newcommand{{\\val{key[0].upper() + key[1:]}}}{{{value}}}"
              for key, value in VALUES.items()]
    (FIGURES / "valeurs_selection.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
