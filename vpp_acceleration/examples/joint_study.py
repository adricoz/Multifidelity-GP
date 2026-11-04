"""
Quick test (2026-10-21): why is the multi-fidelity GP of BoTorch more accurate than our
CoKriging on the target region? Follow-up of rho_study.py (a constant rho is not the cause there).

Hypothesis tested: "BoTorch estimates all its hyperparameters at once on all the data, we
estimate them level by level, so our high-fidelity correction learns its lengthscales on the 40
HF points only".
Theory first: with a nested design and noise-free LF data, the likelihood of the AR(1) model
factorizes EXACTLY, p(y_1, y_2) = p(y_1) p(y_2 | y_1) ([LG] Prop. 3.1), and the two factors share no
hyperparameter. Joint and level-by-level estimation then have the SAME optimum: the joint
estimation alone cannot help the AR(1). What can differ is the STRUCTURE of the model:
  * AR(1) (ours, emukit):   Z_HF = rho Z_LF + delta, delta independent of Z_LF; the lengthscales
    of delta are only seen through the HF points;
  * BoTorch (SingleTaskMultiFidelityGP, linear truncated fidelity kernel): one "unbiased"
    Matern component common to both fidelities (it IS the HF) plus a "biased" component that
    only the LF carries: Z_LF = Z_HF + bias. The lengthscales of the HF function are learned on
    ALL the points; the bias is seen at the 160 LF points.
Models fitted on the same designs (those of rho_study.py: B1 nested LHS 160 / 40 over the domain,
B3 nested design 160 / 40 inside the region, 10 seeds):
  CK  our CoKriging v2 (level by level, MAP)
  CS  CK with share_lengthscales=True (option added to mfego after this test, 2026-10-21)
  P0  recursive co-kriging prototype of rho_study.py (level by level, maximum likelihood)
  EJ  emukit GPyLinearMultiFidelityModel, Matern 5/2: the SAME AR(1), all hyperparameters
      estimated JOINTLY on all the data (tests the "joint estimation" part)
  PS  P0 whose correction delta takes the lengthscales of level 1, learned on the 160 LF points
      (tests the "lengthscales learned on 40 points" part inside the AR(1))
  BT  BoTorch SingleTaskMultiFidelityGP (its default priors)
  BN  BT without its Gamma priors on the lengthscales (tests the effect of these priors)

Run from the repository root (Multifidelity-GP-Clean), about 1 min:
    .venv-benchmark\\Scripts\\python vpp_acceleration\\examples\\joint_study.py
Outputs: examples/results/joint_study.csv, joint_summary.txt
"""
import os

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

# pylint: disable=wrong-import-position,import-error
import argparse  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import benchmark_study as bs  # noqa: E402
import library_study as ls  # noqa: E402  (EmukitAR1, BotorchMF)
import region_study as rs  # noqa: E402
from rho_study import DESIGNS, RecursiveCoKriging  # noqa: E402

MODELS = ("CK", "CS", "P0", "EJ", "PS", "BT", "BN")
LABELS = {"CK": "CK notre CoKriging (niveau par niveau, MAP)",
          "CS": "CS CK avec share_lengthscales=True (mfego)",
          "P0": "P0 prototype AR(1) (niveau par niveau, MV)",
          "EJ": "EJ emukit AR(1) Matérn (conjoint)",
          "PS": "PS P0, longueurs de delta = celles de la BF",
          "BT": "BT BoTorch MF-GP",
          "BN": "BN BoTorch sans prior sur les longueurs"}
COMPARISONS = (("EJ", "P0"), ("PS", "P0"), ("BT", "P0"), ("BN", "BT"), ("BT", "PS"),
               ("CS", "CK"), ("BT", "CS"))


class BotorchNoLengthscalePrior(ls.BotorchMF):
    """BN: the BoTorch model of library_study.py with its Gamma priors on the lengthscales
    removed (gpytorch keeps the priors of a module in its _priors dictionary)."""

    def fit(self, x_lists, y_lists):
        import torch  # noqa: PLC0415
        from botorch.fit import fit_gpytorch_mll  # noqa: PLC0415
        from botorch.models import SingleTaskMultiFidelityGP  # noqa: PLC0415
        from botorch.models.transforms import Normalize  # noqa: PLC0415
        from gpytorch.mlls import ExactMarginalLogLikelihood  # noqa: PLC0415
        if self.standardizer is None:
            self.standardizer = ls.Standardizer(y_lists)
        torch.manual_seed(self.seed)
        dim = x_lists[0].shape[1]
        train_x = np.vstack([self._with_fidelity(x, l + 1) for l, x in enumerate(x_lists)])
        train_y = np.concatenate([self.standardizer.forward(y, l + 1)
                                  for l, y in enumerate(y_lists)])
        self.model = SingleTaskMultiFidelityGP(
            torch.tensor(train_x, dtype=torch.double),
            torch.tensor(train_y, dtype=torch.double).unsqueeze(-1),
            data_fidelities=[dim], input_transform=Normalize(dim + 1))
        for name, module, _, _, _ in list(self.model.named_priors()):
            if "lengthscale_prior" in name:
                del module._priors["lengthscale_prior"]      # pylint: disable=protected-access
        removed = [name for name, *_ in self.model.named_priors() if "lengthscale" in name]
        if removed:
            raise RuntimeError(f"lengthscale priors still present: {removed}")
        fit_gpytorch_mll(ExactMarginalLogLikelihood(self.model.likelihood, self.model))
        return self


def run_task(task):
    design_name, seed = task
    region = rs.region_data()
    global_test = bs.test_set("airfoil")
    problem = bs.make_problem("airfoil")
    data = rs.design_in_region(rs.LHS_DESIGNS["B3"], seed) if design_name == "B3" else \
        bs.nested_design(problem, rs.LHS_DESIGNS["B1"], seed)
    x_lists = [data.get_training_data(l)[0] for l in (1, 2)]
    y_lists = [data.get_training_data(l)[1] for l in (1, 2)]
    rows = []
    for name in MODELS:
        start = time.perf_counter()
        if name in ("CK", "CS"):
            model = bs.make_model("v2", share_lengthscales=(name == "CS"))
            model.fit(data)
        elif name == "P0":
            model = RecursiveCoKriging("constant", seed).fit(x_lists, y_lists)
        elif name == "PS":
            model = RecursiveCoKriging("constant", seed, share_lengthscales=True)
            model.fit(x_lists, y_lists)
        elif name == "EJ":
            model = ls.EmukitAR1(region["x_int"], seed, kernel_name="Matern52")
            model.fit(x_lists, y_lists)
        elif name == "BT":
            model = ls.BotorchMF(region["x_int"], seed).fit(x_lists, y_lists)
        else:
            model = BotorchNoLengthscalePrior(region["x_int"], seed).fit(x_lists, y_lists)
        rows.append({"design": design_name, "seed": seed, "model": name,
                     "fit_time_s": time.perf_counter() - start,
                     **rs.all_scores(model, region, global_test)})
    return rows


def summarize(results: pd.DataFrame) -> str:
    lines = ["=== Joint estimation or model structure? (same designs for every model) ==="]
    for design in DESIGNS:
        r = results[results["design"] == design]
        lines.append(f"\n--- design {design} ({r['seed'].nunique()} seeds), region RMSE median "
                     "[10 % ; 90 %] | coverage | global RMSE")
        for name in MODELS:
            m = r[r["model"] == name]
            lines.append(f"{LABELS[name]:46s} {m['rmse_region'].median():.4f} "
                         f"[{m['rmse_region'].quantile(0.1):.4f} ; "
                         f"{m['rmse_region'].quantile(0.9):.4f}] | "
                         f"{m['coverage_region'].median():.2f} | {m['rmse_global'].median():.4f}")
        table = r.pivot(index="seed", columns="model", values="rmse_region")
        for a, b in COMPARISONS:
            wins = int((table[a] < table[b]).sum())
            p_value = wilcoxon(table[a], table[b]).pvalue          # two-sided
            change = table[a].median() / table[b].median() - 1.0
            lines.append(f"{a} / {b}: median {100 * change:+.0f} % | {a} better on {wins} / "
                         f"{len(table)} | two-sided Wilcoxon p = {p_value:.2g}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--workers", type=int, default=20)
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    rs.region_data()
    tasks = [(design, seed) for design in DESIGNS for seed in range(args.seeds)]
    rows = []
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks))) as pool:
        for task_rows in pool.map(run_task, tasks):
            rows += task_rows
    print(f"{len(tasks)} designs in {time.perf_counter() - start:.0f} s")
    results = pd.DataFrame(rows)
    results.to_csv(bs.RESULTS / "joint_study.csv", index=False)
    summary = summarize(results)
    print(summary)
    (bs.RESULTS / "joint_summary.txt").write_text(summary + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
