"""
Figure of the slides for O. Le Maître (slides_le_maitre.tex): which variance the ALC gain reduces.
From the coherent posterior draws of the 8th run (scripts/robustness_checks.py posterior, scale 1):
  left: correlation of the speed errors of the best section between the navigation conditions
        (one GP for every condition: the cross terms of his Eq. 12.6);
  right: standard deviations of the three quantities, sqrt(E_W Var_N V_c) (our ALC),
        sqrt(Var_N J) (his 12.6) and sqrt(Var_N(J_best - J_region1)) (his 7.6).
Also prints the numbers quoted in the slides.
Run: ..\\.venv-mds\\Scripts\\python doc\\scripts\\fig_alc_variance.py
"""
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

VPP_DIR = Path(__file__).resolve().parents[2]
DRAWS = VPP_DIR / "runs" / "2026-11-03_1041_robustness_posterior_2026-10-29_1002" / "results" / "draws.csv"
OUT = VPP_DIR / "doc" / "figures" / "alc_variance.png"


def speeds(table: pd.DataFrame, section: str, keys) -> pd.DataFrame:
    """Speeds (draws x conditions) of a section."""
    part = table[table.section == section]
    return part.pivot(index="draw", columns="key", values="speed_kt")[keys]


def main(scale: float = 1.0) -> None:
    table = pd.read_csv(DRAWS)
    table = table[table.scale == scale].copy()
    table["key"] = list(zip(table.tws.round(3), table.twa.round(3)))
    solved = table.groupby("key").success.all()
    keys = sorted(solved[solved].index)                       # by TWS, then TWA
    table = table[table.key.isin(keys)]
    weights = table.drop_duplicates("key").set_index("key").weight.loc[keys]
    weights = weights / weights.sum()
    best, other = speeds(table, "best", keys), speeds(table, "region_1", keys)

    ours = float((weights * best.var(ddof=1)).sum())          # E_W Var_N V_c
    j_best, j_other = best @ weights, other @ weights
    his = float(j_best.var(ddof=1))                           # Var_N J      (12.6)
    diff = float((j_best - j_other).var(ddof=1))              # Var_N(J_i - J_j) (7.6)
    corr = best.corr().to_numpy()
    upper = corr[np.triu_indices_from(corr, 1)]
    print(f"{len(best)} draws, {len(keys)} conditions, {np.sum(best.std() == 0)} with no spread "
          f"(speed at the 24 kt bound)")
    print(f"sqrt E_W Var V_c = {np.sqrt(ours) * 1e3:.2f} mkt, sqrt Var J = {np.sqrt(his) * 1e3:.2f} "
          f"mkt, ratio of variances {his / ours:.2f}; sqrt Var(J_best - J_r1) = "
          f"{np.sqrt(diff) * 1e3:.2f} mkt; mean correlation between conditions "
          f"{np.nanmean(upper):+.2f}; independent conditions would give a ratio "
          f"{float((weights ** 2 * best.var(ddof=1)).sum()) / ours:.3f}")

    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4.2),
                                      gridspec_kw={"width_ratios": [1.15, 1]})
    image = left.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    left.set_xticks([])
    left.set_yticks([])
    left.set_xlabel(f"les {len(keys)} conditions du profil, rangées par TWS "
                    f"({keys[0][0]:.0f} → {keys[-1][0]:.0f} nd) puis TWA")
    left.set_title("Corrélation des erreurs de vitesse entre conditions\n"
                   f"(best, {len(best)} tirages GP, moyenne {np.nanmean(upper):+.2f})", fontsize=10)
    fig.colorbar(image, ax=left, fraction=0.046, pad=0.04)

    values = np.sqrt([ours, his, diff]) * 1e3
    labels = ["notre ALC\n$\\sqrt{E_W \\,\\mathrm{Var}_N V_c}$",
              "sa (12.6)\n$\\sqrt{\\mathrm{Var}_N J}$",
              "sa (7.6)\n$\\sqrt{\\mathrm{Var}_N (J_{best}-J_{r1})}$"]
    bars = right.bar(labels, values, color=["0.35", "tab:blue", "tab:orange"], width=0.6)
    right.bar_label(bars, fmt="%.2f", fontsize=9)
    right.set_ylabel("écart type [$10^{-3}$ nd]")
    right.set_title("La variance que réduit chaque critère", fontsize=10)
    right.tick_params(axis="x", labelsize=8)
    right.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200)
    print(OUT)


if __name__ == "__main__":
    main()
