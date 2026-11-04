"""
Figure of the operating profile profile_v2 for the slides (slides_le_maitre.tex): density of the
mixture of truncated normal laws (profiles/profile_v2.json, "editor" block) and the 49 nodes of
the quadrature (size = weight, colour = mode), with the marginal law of TWA.
Run: ..\\.venv-mds\\Scripts\\python doc\\scripts\\fig_profile.py
"""
import json
from pathlib import Path

import matplotlib
import numpy as np
from scipy.stats import truncnorm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

VPP_DIR = Path(__file__).resolve().parents[2]
PROFILE = VPP_DIR / "profiles" / "profile_v2.json"
OUT = VPP_DIR / "doc" / "figures" / "profile_v2.png"
COLORS = {"upwind": "tab:blue", "downwind": "tab:orange", "downwind_tail": "tab:green",
          "manual": "k"}
LABELS = {"upwind": "mode près", "downwind": "mode largue", "downwind_tail": "queue",
          "manual": "points de bord (à la main)"}


def pdf(law: dict, x: np.ndarray) -> np.ndarray:
    a, b = (law["lower"] - law["mean"]) / law["std"], (law["upper"] - law["mean"]) / law["std"]
    return truncnorm.pdf(x, a, b, loc=law["mean"], scale=law["std"])


def main() -> None:
    profile = json.loads(PROFILE.read_text(encoding="utf-8"))
    modes = profile["editor"]["modes"]
    twa, tws = np.linspace(30, 132, 300), np.linspace(4.5, 17.5, 200)
    density = sum(m["weight"] * np.outer(pdf(m["tws"], tws), pdf(m["twa"], twa)) for m in modes)
    marginal = sum(m["weight"] * pdf(m["twa"], twa) for m in modes)

    fig, (top, ax) = plt.subplots(2, 1, figsize=(8, 5.8), sharex=True,
                                  gridspec_kw={"height_ratios": [1, 3]})
    top.plot(twa, marginal, color="0.3")
    top.fill_between(twa, marginal, color="0.85")
    top.set_ylabel("densité\nde TWA", fontsize=8)
    top.set_yticks([])
    top.spines[["top", "right", "left"]].set_visible(False)

    ax.contourf(twa, tws, density, levels=12, cmap="Greys", alpha=0.6)
    nodes = profile["nodes"]
    for mode in COLORS:
        part = [n for n in nodes if n["mode"] == mode]
        ax.scatter([n["twa"] for n in part], [n["tws"] for n in part],
                   s=[2500 * n["weight"] for n in part], color=COLORS[mode], edgecolors="w",
                   linewidths=0.6, label=f"{LABELS[mode]} ({len(part)} conditions, "
                                         f"{100 * sum(n['weight'] for n in part):.0f} %)")
    ax.set_xlabel("TWA [deg]")
    ax.set_ylabel("TWS [nd]")
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, markerscale=0.5,
              frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Profil d'usage {profile['name']} : {len(nodes)} conditions (taille = poids)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT, dpi=200)
    print(OUT)


if __name__ == "__main__":
    main()
