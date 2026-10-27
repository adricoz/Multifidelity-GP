"""
Comparison of a run with a reference run (2026-10-27: the control run of the convergence, the
restarts and the faster batches, against the fourth run). Written to <run>/figures/comparison.html,
with a printed summary:
  1. time of a batch against the size of the design: the whole batch, and the batch without its
     update (refit or conditioning, choice of the 16 points, codes);
  2. cumulative gain of the incumbent against time, one curve per region of the search (the gain
     of an update: new incumbent minus old one, both solved in that update, vpp_optimize).
Run: ..\\.venv-mds\\Scripts\\python scripts\\compare_runs.py runs\\<run> runs\\<reference>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

VPP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VPP_DIR.parent / "benchmarks"))


def batch_table(root: Path) -> pd.DataFrame:
    """One row per batch: end time [min], design size, batch time and update time [s]."""
    history = pd.read_csv(root / "results" / "history.csv")
    updates = json.loads((root / "results" / "updates.json").read_text(encoding="utf-8"))
    update_time = {u["iteration"]: u["time_s"] for u in updates}
    rows = []
    for batch, group in history.groupby("batch"):
        rows.append({"batch": batch, "end_min": group.elapsed_min.max(), "n": len(group),
                     "size": int(group.n_lf.iloc[-1] + group.n_hf.iloc[-1]),
                     "update_s": float(group.update_s.iloc[0]) if "update_s" in group
                     else update_time.get(int(group.iteration.min()), np.nan)})
    table = pd.DataFrame(rows).sort_values("batch")
    table["batch_s"] = 60.0 * table.end_min.diff()        # NaN for the first batch
    table["without_update_s"] = table.batch_s - table.update_s
    return table


def gain_table(root: Path) -> pd.DataFrame:
    updates = json.loads((root / "results" / "updates.json").read_text(encoding="utf-8"))
    return pd.DataFrame([{"min": u["elapsed_min"], "region": u.get("region", 0),
                          "gain": u["incumbent_speed"] - u["candidate_speeds"][0]
                          if u["changed"] else 0.0} for u in updates])


def main(run: Path, reference: Path) -> None:
    import plotly.graph_objects as go  # noqa: PLC0415
    from figure_style import SERIES, save, style  # noqa: PLC0415
    from plotly.subplots import make_subplots  # noqa: PLC0415

    names = {run: run.name, reference: reference.name}
    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.14, subplot_titles=(
        "Durée d'un lot de 16 choix selon la taille du plan",
        "Gain cumulé de la titulaire, par région de la recherche"))
    for i, root in enumerate((reference, run)):
        table = batch_table(root)
        color = SERIES[i]
        fig.add_trace(go.Scatter(x=table["size"], y=table.batch_s, mode="markers",
                                 marker={"color": color, "size": 6},
                                 name=f"{names[root]} : lot complet"), row=1, col=1)
        fig.add_trace(go.Scatter(x=table["size"], y=table.without_update_s, mode="markers",
                                 marker={"color": color, "size": 6, "symbol": "circle-open"},
                                 name=f"{names[root]} : lot sans la mise à jour"), row=1, col=1)
        late = table[table["size"] >= 0.8 * table["size"].max()]
        print(f"{names[root]}: {len(table)} batches in {table.end_min.max():.0f} min, "
              f"{60 * table.n.sum() / table.end_min.max():.0f} choices per hour; last 20 % "
              f"of the design ({late['size'].min()}-{late['size'].max()} points): batch "
              f"{late.batch_s.median():.0f} s, update {late.update_s.median():.0f} s, without the "
              f"update {late.without_update_s.median():.0f} s (medians)")
        gains = gain_table(root)
        for r, part in gains.groupby("region"):
            fig.add_trace(go.Scatter(x=part["min"], y=part.gain.cumsum(), mode="lines+markers",
                                     line={"color": color, "dash": "solid" if r == 0 else "dot"},
                                     marker={"size": 4},
                                     name=f"{names[root]} : région {r}"), row=2, col=1)
            print(f"   region {r}: {len(part)} updates, {part['min'].min():.0f}-"
                  f"{part['min'].max():.0f} min, cumulative gain {part.gain.sum():.3f} kt")
    fig.update_xaxes(title_text="taille du plan (points BF + HF)", row=1, col=1)
    fig.update_yaxes(title_text="durée [s]", row=1, col=1)
    fig.update_xaxes(title_text="temps du run [min]", row=2, col=1)
    fig.update_yaxes(title_text="gain cumulé [nd]", row=2, col=1)
    style(fig, "Comparaison avec un run de référence",
          f"{names[run]} contre {names[reference]}", height=900)
    save(fig, run / "figures", "comparison")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
