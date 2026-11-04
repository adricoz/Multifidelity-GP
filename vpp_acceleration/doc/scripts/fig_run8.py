"""
Figures of the 8th run for the slides (slides_le_maitre.tex), simple matplotlib PNG:
  1. run8_convergence.png: predicted mean speed of the incumbent at each update (one colour per
     region of the search), the candidates of the screening, and the design size per level;
  2. run8_sections.png: the best section, the optimum of region 1 and the start section;
  3. run8_gains.png: speed gain of the best section over the start one at every condition of the
     operating profile, and per mode.
Run: ..\\.venv-mds\\Scripts\\python doc\\scripts\\fig_run8.py
"""
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

VPP_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(VPP_DIR))
sys.path.insert(0, str(VPP_DIR / "scripts"))
RUN = VPP_DIR / "runs" / "2026-10-29_1002_profile_v2_box3_3h"
OUT = VPP_DIR / "doc" / "figures"
REGION_COLORS = ["tab:blue", "tab:orange", "tab:green"]


def convergence() -> None:
    updates = json.loads((RUN / "results" / "updates.json").read_text(encoding="utf-8"))
    history = pd.read_csv(RUN / "results" / "history.csv")
    profile = json.loads((RUN / "results" / "profile_profile_v2.json").read_text(encoding="utf-8"))
    start = profile["sections"]["closest_to_current"]["common"]["expected_speed_kt"]
    best = profile["sections"]["best"]["common"]["expected_speed_kt"]

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.6, 1]})
    for u in updates:
        left.scatter([u["elapsed_min"]] * (len(u["candidate_speeds"]) - 1), u["candidate_speeds"][1:],
                     s=8, color="0.75", zorder=1)
    for region in sorted({u["region"] for u in updates}):
        part = [u for u in updates if u["region"] == region]
        left.plot([u["elapsed_min"] for u in part], [u["incumbent_speed"] for u in part], "o-",
                  ms=3, color=REGION_COLORS[region], label=f"titulaire, région {region}", zorder=2)
    left.scatter([], [], s=8, color="0.75", label="concurrentes du criblage")
    left.axhline(start, color="k", ls=":", lw=1)
    left.text(2, start + 0.004, f"section de départ : {start:.3f} nd", fontsize=8)
    left.axhline(best, color="tab:red", ls="--", lw=1)
    left.text(2, best + 0.004, f"best, surrogate final : {best:.3f} nd", fontsize=8, color="tab:red")
    restart = next(u for u in updates if u["region"] == 1)["elapsed_min"]
    left.axvline(restart, color="0.5", lw=0.8)
    left.text(restart + 1, start + 0.04, "région 0 convergée :\nredémarrage", fontsize=8, color="0.3")
    left.set_xlabel("temps du run [min]")
    left.set_ylabel("vitesse moyenne prédite [nd]")
    left.set_title("Vitesse moyenne sur le profil d'usage (VPP sur le surrogate)", fontsize=10)
    left.legend(fontsize=8, loc="center right")
    left.spines[["top", "right"]].set_visible(False)

    right.plot(history["elapsed_min"], history["n_l1"], color="tab:gray", label="BF : NPLLT × NeuralFoil")
    right.plot(history["elapsed_min"], history["n_l2"], color="tab:purple", label="HF : AVL × XFOIL")
    right.set_xlabel("temps du run [min]")
    right.set_ylabel("nombre de points du plan")
    right.set_title("Taille du plan par niveau", fontsize=10)
    right.legend(fontsize=8)
    right.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "run8_convergence.png", dpi=200)
    print(f"start {start:.4f}, best {best:.4f}, restart at {restart:.1f} min, "
          f"final design {history['n_l1'].iloc[-1]}/{history['n_l2'].iloc[-1]}")


def sections() -> None:
    from robustness_checks import load_run  # noqa: PLC0415
    from src.section_pca import section_coordinates  # noqa: PLC0415
    _, pca, _, _, found = load_run(RUN)
    x = np.linspace(0.0, 1.0, 201)
    fig, ax = plt.subplots(figsize=(10, 3.0))
    for name, label, style in (("closest_to_current", "départ (section actuelle admissible)", ("k", ":")),
                               ("region_1", "optimum de la région 1", ("tab:orange", "-")),
                               ("best", "best (région 0)", ("tab:blue", "-"))):
        y = section_coordinates(pca.decode(found[name][None, :]), x, pca.te_thickness)[0]
        upper, lower = y[:len(x)], y[len(x):]
        ax.plot(x, upper, color=style[0], ls=style[1], lw=1.4, label=label)
        ax.plot(x, lower, color=style[0], ls=style[1], lw=1.4)
    ax.set_aspect("equal")
    ax.set_xlabel("x / c")
    ax.set_ylabel("y / c")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title("Sections (corde 0,70 m)", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "run8_sections.png", dpi=200)


def gains() -> None:
    table = pd.read_csv(RUN / "results" / "profile_profile_v2.csv")
    best = table[table.section == "best"].set_index(["twa", "tws"])
    start = table[table.section == "closest_to_current"].set_index(["twa", "tws"])
    common = best.index[(best.success & start.loc[best.index, "success"]).to_numpy()]
    best, start = best.loc[common], start.loc[common]
    gain = 100.0 * (best.speed_kt - start.speed_kt) / start.speed_kt

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.0), gridspec_kw={"width_ratios": [1.6, 1]})
    twa = np.array([i[0] for i in common])
    tws = np.array([i[1] for i in common])
    points = left.scatter(twa, tws, c=gain, s=600 * best.weight / best.weight.max(), cmap="viridis",
                          edgecolors="k", linewidths=0.4)
    fig.colorbar(points, ax=left, label="gain de vitesse [%]")
    left.set_xlabel("TWA [deg]")
    left.set_ylabel("TWS [nd]")
    left.set_title("Gain de best sur la section de départ, par condition\n(taille = poids dans le profil)",
                   fontsize=10)
    left.spines[["top", "right"]].set_visible(False)

    # modes of profile_v2 (TWA ranges of their conditions)
    modes = {"upwind": "près\n35-53°", "downwind": "largue\n68-92°", "downwind_tail": "queues\n64-96°",
             "manual": "points\nde bord"}
    values = []
    for mode in modes:
        w = best.weight[best["mode"] == mode]
        vb = (w * best.speed_kt[best["mode"] == mode]).sum()
        vs = (w * start.speed_kt[start["mode"] == mode]).sum()
        values.append(100.0 * (vb - vs) / vs)
    w = best.weight
    total = 100.0 * ((w * best.speed_kt).sum() - (w * start.speed_kt).sum()) / (w * start.speed_kt).sum()
    bars = right.bar(list(modes.values()) + ["profil"], values + [total],
                     color=["0.6"] * len(modes) + ["tab:blue"], width=0.6)
    right.bar_label(bars, fmt="%.1f %%", fontsize=8)
    right.set_ylabel("gain de vitesse moyen [%]")
    right.set_title("Gain par allure", fontsize=10)
    right.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "run8_gains.png", dpi=200)
    print(f"gain per condition {gain.min():.2f} to {gain.max():.2f} %, profile {total:.2f} %, "
          f"modes {dict(zip(modes, np.round(values, 2)))}")


def posterior() -> None:
    """run8_posterior.png: the coherent GP draws of scripts/robustness_checks.py posterior."""
    folder = VPP_DIR / "runs" / "2026-11-03_1041_robustness_posterior_2026-10-29_1002"
    sys.path.insert(0, str(VPP_DIR.parent / "benchmarks"))
    from robustness_figures import draw_means  # noqa: PLC0415
    means = draw_means(pd.read_csv(folder / "results" / "draws.csv"))
    names = {"best": "best", "region_1": "région 1",
             "best_2026-10-28_1747_profile_v1_box075": "best du run 7"}
    colors = {1.0: "tab:blue", 2.0: "tab:orange"}
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.0))
    x = np.arange(len(names))
    for scale, color in colors.items():
        part = means.loc[scale]
        for i, (_, row) in enumerate(part.iterrows()):
            left.plot(x + (0.08 if scale == 2.0 else -0.08), row[list(names)], "-o", ms=2.5, lw=0.6,
                      color=color, alpha=0.6, label=f"tirages à {scale:g} σ" if i == 0 else None)
        for j, other in enumerate(list(names)[1:]):
            gap = (part["best"] - part[other]) * 1e3
            jitter = np.random.default_rng(j).uniform(-0.06, 0.06, len(gap))
            right.scatter(j + (0.15 if scale == 2.0 else -0.15) + jitter, gap, s=12, color=color,
                          label=f"{scale:g} σ" if j == 0 else None)
    left.set_xticks(x, list(names.values()))
    left.set_ylabel("vitesse moyenne [nd]")
    left.set_title("Vitesse de chaque section, un trait par tirage", fontsize=10)
    left.legend(fontsize=8)
    right.axhline(0, color="k", lw=0.8)
    right.set_xticks(range(len(names) - 1), [f"best − {v}" for v in list(names.values())[1:]])
    right.set_ylabel("écart de vitesse [$10^{-3}$ nd]")
    right.set_title("Écart dans le même tirage (> 0 : best gagne)", fontsize=10)
    right.set_ylim(bottom=-2)
    right.legend(fontsize=8)
    for ax in (left, right):
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "run8_posterior.png", dpi=200)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    convergence()
    sections()
    gains()
    posterior()
