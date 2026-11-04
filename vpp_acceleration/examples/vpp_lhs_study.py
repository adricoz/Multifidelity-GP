"""
Study 2 (2026-10-21): which initial nested Latin hypercube for our 16-dimensional problem (12 PCA
scores of the section + Cant_Eff, Rake_Eff, Yaw_Eff, Hvol)?

Designs of the meta-surrogate (vpp_targeted.nested_design: sections drawn in the family, attitude
by Latin hypercube, HF subset by greedy maximin), costs LF = 1, HF = 10, 3 seeds each:
  co-kriging (n_HF, n_LF): (16, 64) (16, 128) (32, 128) (32, 256) (64, 256) (64, 512) (128, 512);
  kriging on the HF only (no LF), n_HF = 32, 64, 128 (does the LF help?).
Measured on each fitted surrogate (vpp_study_common):
  * e_V: first-order speed error at the 20 true VPP states [kt] (the target of the study);
  * relative RMSE of each coefficient on 1 000 points of the whole domain.
Rules of thumb checked: n >= 10 d points for a GP (Loeppky, Sacks & Welch 2009: n = 10 d = 160),
and for the co-kriging LF ~ 4 to 8 x HF.

Run (MDS environment, 4 processes, about 6 min):
    ..\\.venv-mds\\Scripts\\python examples\\vpp_lhs_study.py [--workers 4] [--figure-only]
Outputs: results/vpp_lhs_study.csv, figures/vpp_lhs_study.html
"""
import argparse
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import vpp_study_common as common
from src.cokriging import CoKriging
from src.data_management import ExperimentData
from src.vpp_targeted import N_FORCES, RHO_PRIOR, TorsorCoKriging, nested_design

CO_KRIGING = [(16, 64), (16, 128), (32, 128), (32, 256), (64, 256), (64, 512), (128, 512)]
HF_ONLY = [32, 64, 128]
SEEDS = (0, 1, 2)


class TorsorKriging(TorsorCoKriging):
    """Single fidelity: 6 GPs (CoKriging with one level) on the HF points only."""

    def fit(self, refit=True):
        self.models = []
        for k in range(N_FORCES):
            data = ExperimentData(bounds=self.bounds, costs=[float(self.costs[1])])
            data.x_dict = {1: self.x_lists[1]}
            data.y_dict = {1: self.y_lists[1][:, k]}
            data.metrics_dict = {1: [{}] * len(self.x_lists[1])}
            model = CoKriging(1, priors=RHO_PRIOR, seed=self.seed + k)
            model.fit(data)
            self.models.append(model)


def run_task(task):
    kind, n_high, n_low, seed = task
    meta = common.MetaFoil()
    reference = common.truth()
    test = common.global_test_set(meta)
    start = time.perf_counter()
    if kind == "co-kriging":
        surrogate = nested_design(meta, n_low, n_high, seed=seed)
    else:
        multi = nested_design(meta, n_high, n_high, seed=seed)      # same sampler, HF at every point
        surrogate = TorsorKriging(multi.bounds, seed=seed)
        surrogate.x_lists, surrogate.y_lists = multi.x_lists, multi.y_lists
    surrogate.fit()
    errors = common.global_errors(surrogate, test)
    cost = n_high * 10.0 + (n_low if kind == "co-kriging" else 0.0)
    row = {"kind": kind, "n_high": n_high, "n_low": n_low if kind == "co-kriging" else 0,
           "seed": seed, "cost": cost, "speed_error_kt": common.speed_error(surrogate, reference),
           "time_s": time.perf_counter() - start,
           **{f"rel_rmse_{name}": float(e) for name, e in zip(common.FORCE_NAMES, errors)}}
    print(f"{kind:10s} HF {n_high:3d} LF {row['n_low']:3d} seed {seed}: e_V "
          f"{row['speed_error_kt']:.4f} kt, {row['time_s']:.0f} s", flush=True)
    return row


def make_figure(results: pd.DataFrame) -> None:
    import plotly.graph_objects as go  # noqa: PLC0415
    from figure_style import SERIES, save, style  # noqa: PLC0415
    from plotly.subplots import make_subplots  # noqa: PLC0415

    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.1, subplot_titles=(
        "Erreur de vitesse aux états VPP (1er ordre)", "Erreur relative sur tout le domaine (Fx)"))
    groups = [("co-kriging, BF = 4 HF", results[(results["kind"] == "co-kriging")
                                               & (results["n_low"] == 4 * results["n_high"])]),
              ("co-kriging, BF = 8 HF", results[(results["kind"] == "co-kriging")
                                               & (results["n_low"] == 8 * results["n_high"])]),
              ("krigeage HF seul", results[results["kind"] == "HF only"])]
    for color, (name, group) in zip(SERIES, groups):
        stats = group.groupby("cost").agg(e=("speed_error_kt", "median"),
                                          lo=("speed_error_kt", "min"),
                                          hi=("speed_error_kt", "max"),
                                          fx=("rel_rmse_Fx", "median"),
                                          n_high=("n_high", "first")).reset_index()
        for col, column in ((1, "e"), (2, "fx")):
            fig.add_trace(go.Scatter(
                x=stats["cost"], y=stats[column], mode="lines+markers", name=name,
                legendgroup=name, showlegend=(col == 1), line={"color": color},
                text=[f"HF {n}" for n in stats["n_high"]],
                error_y={"type": "data", "symmetric": False,
                         "array": stats["hi"] - stats["e"], "arrayminus": stats["e"] - stats["lo"],
                         "color": color} if col == 1 else None,
                hovertemplate="coût %{x:.0f} (%{text}) : %{y:.4f}<extra>" + name + "</extra>"),
                row=1, col=col)
    for col in (1, 2):
        fig.update_xaxes(type="log", title_text="coût du plan (BF = 1, HF = 10)",
                         tickvals=[200, 300, 500, 700, 1000, 1500, 2000], row=1, col=col)
        fig.update_yaxes(type="log", row=1, col=col)
    fig.update_yaxes(title_text="e_V [nd] (médiane, barre min-max)", row=1, col=1)
    fig.update_yaxes(title_text="RMSE / écart-type", row=1, col=2)
    style(fig, "Quel plan LHS de départ en dimension 16 ?",
          f"Méta-surrogate du C-foil (12 modes PCA + cant, rake, yaw, Hvol) ; {len(SEEDS)} graines ; "
          "e_V = erreur de vitesse que causeraient les erreurs du torseur aux 20 états VPP",
          height=520)
    print(f"figure: {save(fig, common.FIGURES, 'vpp_lhs_study')}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--figure-only", action="store_true")
    args = parser.parse_args()
    path = common.RESULTS / "vpp_lhs_study.csv"
    if args.figure_only:
        results = pd.read_csv(path)
    else:
        common.truth()                                        # computed once, before the workers
        tasks = [("co-kriging", h, l, s) for h, l in CO_KRIGING for s in SEEDS]
        tasks += [("HF only", h, 0, s) for h in HF_ONLY for s in SEEDS]
        tasks.sort(key=lambda t: -(t[1] * 10 + t[2]))          # heavy ones first
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            results = pd.DataFrame(list(pool.map(run_task, tasks)))
        results.to_csv(path, index=False)
    summary = results.groupby(["kind", "n_high", "n_low"]).agg(
        cost=("cost", "first"), speed_error_kt=("speed_error_kt", "median"),
        rel_rmse_Fx=("rel_rmse_Fx", "median"), rel_rmse_Fy=("rel_rmse_Fy", "median"),
        rel_rmse_Mx=("rel_rmse_Mx", "median"), time_s=("time_s", "median")).reset_index()
    text = summary.sort_values(["kind", "cost"]).to_string(index=False,
                                                           float_format=lambda v: f"{v:.4f}")
    print(text)
    (common.RESULTS / "vpp_lhs_summary.txt").write_text(text + "\n", encoding="utf-8")
    make_figure(results)


if __name__ == "__main__":
    main()
