"""
Final benchmark: our sequential co-kriging (S1) against the sequential designs of two known GP
libraries, on the region test of region_study.py. Protocol validated by the user on 2026-10-21
(emukit + BoTorch, double evaluation, at most 30 min).

Problem, start, budget, integration points and test sets: those of region_study.py (NACA drag
4D, NeuralFoil xxsmall cost 1 / xxxlarge cost 10, target region = low-fidelity lift between 0.4
and 0.6, nested start 80 LF + 20 HF, then 280 of budget: total cost 560).

Competitors, each with the sequential criterion of its own library (the analogue of our ALC):
  E1  emukit 0.5. Model: GPyLinearMultiFidelityModel, the AR(1) of Kennedy & O'Hagan (the same
      model as ours), one RBF ARD kernel per level, hyperparameters by maximum likelihood (GPy).
      Criterion: IntegratedVarianceReduction, the mean reduction of the HF variance over the
      region points when one observation is added at (x, level), divided by the cost of the level.
      emukit's usual loop: after each new point, set_data then optimize from the previous
      hyperparameters (1 restart).
  T1  BoTorch 0.18. Model: SingleTaskMultiFidelityGP (the fidelity is an extra input s, 0 for
      the LF and 1 for the HF, linear truncated fidelity kernel). Criterion:
      qNegIntegratedPosteriorVariance with the region points at s = 1; the gain is the current
      integrated variance minus the integrated variance after the point, divided by the cost.
      Refitted from scratch after each point (fit_gpytorch_mll), as in the BoTorch tutorials.
  Both choose among 1024 Sobol candidates (new at each iteration, as S1) at both levels. A
  library adds ONE level at a time: its designs are not nested, its cost is not cumulative.
  The outputs are standardized with the statistics of the start design (fixed during the run).
  S1 (ours) and B1 (nested LHS 160 / 40) are read from results/region_study.csv (same seeds).

Double evaluation of every final design (main metric: relative RMSE on the region test set):
  own   the model of the library on its design (end to end);
  ours  our CoKriging (rho bounded, v2, as S1) fitted on the same design. The library designs
        are not nested, so the missing LF values at the HF points are computed for this fit only
        (not counted in the cost; their number is reported). It separates the quality of the
        design from the quality of the model.

Criterion fixed before the computation (L* = the library whose "own" median region error is the
lowest):
  * S1 "at least as good" as L*: median error of S1 <= 1.05 x that of L*, and a one-sided paired
    Wilcoxon test does not find S1 worse (p >= 0.05);
  * S1 "better" than L*: median error of S1 <= 0.90 x that of L*, and Wilcoxon (S1 < L*) p < 0.05.

Run from the repository root (Multifidelity-GP-Clean), about 25 min with 20 processes:
    .venv-benchmark\\Scripts\\python vpp_acceleration\\examples\\library_study.py
    .venv-benchmark\\Scripts\\python vpp_acceleration\\examples\\library_study.py --seeds 1 --quick
Outputs: examples/results/library_study.csv, library_curves.csv, library_summary.txt and
examples/figures/library_study.html
"""
import os

# one BLAS thread per process: the runs are parallelized over the processes instead
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

# pylint: disable=wrong-import-position,import-error
import argparse  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import qmc, wilcoxon  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import benchmark_study as bs  # noqa: E402  (paths, problem, model, costs)
import region_study as rs  # noqa: E402  (region data, start, budget, scores)
from src.cokriging import missing_points_for_nesting  # noqa: E402
from src.data_management import ExperimentData  # noqa: E402

LIBRARIES = ("E1", "T1")
LABELS = {"S1": "S1 notre séquentiel (ALC, ρ borné)", "B1": "B1 LHS imbriqué 160/40",
          "E1": "E1 emukit (AR(1) + IVR)", "T1": "T1 BoTorch (MF-GP + qNIPV)"}
N_CANDIDATES = 1024
CHUNK = 128                    # candidates per batched BoTorch call (memory)
QUICK_BUDGET = 40.0            # budget of the --quick check


# 1/4 ---------------------------------------------------------------------------------------------
# The two libraries: a model with fit / predict and the gain of one observation at (x, level)
# -------------------------------------------------------------------------------------------------
class Standardizer:
    """Outputs of each level standardized with the mean and standard deviation of the START
    design (fixed for the whole run, so that the warm start of emukit stays consistent)."""

    def __init__(self, y_lists):
        self.means = [float(np.mean(y)) for y in y_lists]
        self.scales = [float(np.std(y)) for y in y_lists]

    def forward(self, y, level):
        return (np.asarray(y, dtype=float) - self.means[level - 1]) / self.scales[level - 1]

    def back_mean(self, mean):
        return self.means[-1] + self.scales[-1] * mean

    def back_variance(self, variance):
        return self.scales[-1] ** 2 * variance


class EmukitAR1:
    """E1: AR(1) linear multi-fidelity model of emukit + IntegratedVarianceReduction."""
    name = "E1"

    def __init__(self, x_int, seed, kernel_name: str = "RBF"):
        self.x_int = x_int
        self.seed = seed
        self.kernel_name = kernel_name               # GPy kernel of each level (RBF, Matern52)
        self.wrapper = None
        self.standardizer = None

    def fit(self, x_lists, y_lists):
        import GPy  # noqa: PLC0415
        from emukit.model_wrappers.gpy_model_wrappers import (  # noqa: PLC0415
            GPyMultiOutputWrapper)
        from emukit.multi_fidelity.convert_lists_to_array import (  # noqa: PLC0415
            convert_xy_lists_to_arrays)
        from emukit.multi_fidelity.kernels import LinearMultiFidelityKernel  # noqa: PLC0415
        from emukit.multi_fidelity.models import GPyLinearMultiFidelityModel  # noqa: PLC0415
        if self.standardizer is None:
            self.standardizer = Standardizer(y_lists)
        x_array, y_array = convert_xy_lists_to_arrays(
            x_lists, [self.standardizer.forward(y, l + 1).reshape(-1, 1)
                      for l, y in enumerate(y_lists)])
        if self.wrapper is None:                      # first fit: build the model
            np.random.seed(self.seed)                 # GPy draws its restarts with numpy
            dim = x_lists[0].shape[1]
            kernels = [getattr(GPy.kern, self.kernel_name)(dim, ARD=True) for _ in x_lists]
            model = GPyLinearMultiFidelityModel(x_array, y_array,
                                                LinearMultiFidelityKernel(kernels),
                                                n_fidelities=len(x_lists))
            for likelihood in model.likelihood.likelihoods_list:
                likelihood.variance.constrain_bounded(1e-8, 1e-2, warning=False)
            self.wrapper = GPyMultiOutputWrapper(model, len(x_lists), n_optimization_restarts=5,
                                                 verbose_optimization=False)
        else:                                         # next fits: new data, warm start
            self.wrapper.set_data(x_array, y_array)
            self.wrapper.n_optimization_restarts = 1
        self.wrapper.optimize()
        return self

    def gains(self, x_cand):
        """
        Reduction of the mean HF variance over x_int per observation at (x, level), (M, 2).
        IntegratedVarianceReduction.evaluate loops over the candidates and averages, for each
        one, emukit's calculate_variance_reduction (cov(x_mc, x)^2 / (var(x) + noise)). That
        function takes every candidate at once: one call per level gives the same numbers
        (checked by check_gains), 16 times faster.
        """
        x_monte_carlo = np.column_stack([self.x_int, np.ones(len(self.x_int))])   # HF index 1
        return np.column_stack([
            self.wrapper.calculate_variance_reduction(
                np.column_stack([x_cand, np.full(len(x_cand), index)]), x_monte_carlo
            ).mean(axis=1) for index in (0, 1)])

    def library_gains(self, x_cand):
        """The same gains with the acquisition of emukit itself (slow loop): for check_gains."""
        from emukit.core import (ContinuousParameter, InformationSourceParameter,  # noqa
                                 ParameterSpace)
        from emukit.experimental_design.acquisitions import (  # noqa: PLC0415
            IntegratedVarianceReduction)
        space = ParameterSpace([ContinuousParameter(f"x{i}", 0.0, 1.0)
                                for i in range(x_cand.shape[1])]
                               + [InformationSourceParameter(2)])
        x_monte_carlo = np.column_stack([self.x_int, np.ones(len(self.x_int))])
        acquisition = IntegratedVarianceReduction(self.wrapper, space, x_monte_carlo)
        return np.column_stack([
            acquisition.evaluate(np.column_stack([x_cand, np.full(len(x_cand), index)]))[:, 0]
            for index in (0, 1)])

    def predict_batch(self, x, include_noise=False):
        """HF mean and variance (raw units), same signature as CoKriging.predict_batch."""
        del include_noise                             # latent variance, as for CoKriging
        mean, variance = self.wrapper.gpy_model.predict(np.column_stack([x, np.ones(len(x))]),
                                                        include_likelihood=False)
        return (self.standardizer.back_mean(mean[:, 0]),
                self.standardizer.back_variance(np.maximum(variance[:, 0], 0.0)))


class BotorchMF:
    """T1: SingleTaskMultiFidelityGP of BoTorch + qNegIntegratedPosteriorVariance."""
    name = "T1"
    FIDELITY = (0.0, 1.0)                             # s of the LF and of the HF (costs 1, 10)

    def __init__(self, x_int, seed):
        self.x_int = x_int
        self.seed = seed
        self.model = None
        self.standardizer = None

    def _with_fidelity(self, x, level):
        return np.column_stack([x, np.full(len(x), self.FIDELITY[level - 1])])

    def fit(self, x_lists, y_lists):
        import torch  # noqa: PLC0415
        from botorch.fit import fit_gpytorch_mll  # noqa: PLC0415
        from botorch.models import SingleTaskMultiFidelityGP  # noqa: PLC0415
        from botorch.models.transforms import Normalize  # noqa: PLC0415
        from gpytorch.mlls import ExactMarginalLogLikelihood  # noqa: PLC0415
        if self.standardizer is None:
            self.standardizer = Standardizer(y_lists)
        torch.manual_seed(self.seed)
        dim = x_lists[0].shape[1]
        train_x = np.vstack([self._with_fidelity(x, l + 1) for l, x in enumerate(x_lists)])
        train_y = np.concatenate([self.standardizer.forward(y, l + 1)
                                  for l, y in enumerate(y_lists)])
        self.model = SingleTaskMultiFidelityGP(
            torch.tensor(train_x, dtype=torch.double),
            torch.tensor(train_y, dtype=torch.double).unsqueeze(-1),
            data_fidelities=[dim], input_transform=Normalize(dim + 1))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit_gpytorch_mll(ExactMarginalLogLikelihood(self.model.likelihood, self.model))
        return self

    def gains(self, x_cand):
        """
        Current integrated HF variance over x_int minus the one after an observation at
        (x, level), (M, 2), standardized units. qNegIntegratedPosteriorVariance builds one
        "fantasy" model per candidate (160 s per iteration here). For a GP the reduction does not
        depend on the observed value: adding y = f(x) + noise lowers the variance at x' by
        cov(x', x)^2 / (var(x) + noise). This closed form, computed from the joint posterior of
        a batch of candidates and the points x_int, gives the same numbers (checked by
        check_gains).
        """
        import torch  # noqa: PLC0415
        mc_points = torch.tensor(self._with_fidelity(self.x_int, 2), dtype=torch.double)
        noise = float(self.model.likelihood.noise.mean())
        gains = np.zeros((len(x_cand), 2))
        with torch.no_grad():
            for level in (1, 2):
                x = torch.tensor(self._with_fidelity(x_cand, level), dtype=torch.double)
                for start in range(0, len(x), CHUNK):
                    batch = x[start:start + CHUNK]
                    joint = self.model.posterior(torch.cat([batch, mc_points]))
                    covariance = joint.distribution.covariance_matrix
                    n_batch = len(batch)
                    cross = covariance[:n_batch, n_batch:]               # cov(x, x_mc)
                    variance = torch.diagonal(covariance)[:n_batch] + noise
                    gains[start:start + n_batch, level - 1] = \
                        (cross ** 2 / variance[:, None]).mean(dim=1).numpy()
        return gains

    def library_gains(self, x_cand):
        """The same gains with qNegIntegratedPosteriorVariance itself (slow): for check_gains."""
        import torch  # noqa: PLC0415
        from botorch.acquisition.active_learning import (  # noqa: PLC0415
            qNegIntegratedPosteriorVariance)
        mc_points = torch.tensor(self._with_fidelity(self.x_int, 2), dtype=torch.double)
        acquisition = qNegIntegratedPosteriorVariance(self.model, mc_points=mc_points)
        with torch.no_grad():
            current = float(self.model.posterior(mc_points).variance.mean())
            return np.column_stack([
                current + acquisition(torch.tensor(self._with_fidelity(x_cand, level),
                                                   dtype=torch.double).unsqueeze(1)).numpy()
                for level in (1, 2)])

    def predict_batch(self, x, include_noise=False):
        """HF mean and latent variance (raw units)."""
        import torch  # noqa: PLC0415
        del include_noise
        with torch.no_grad():
            posterior = self.model.posterior(
                torch.tensor(self._with_fidelity(x, 2), dtype=torch.double))
        return (self.standardizer.back_mean(posterior.mean.squeeze(-1).numpy()),
                self.standardizer.back_variance(
                    posterior.variance.squeeze(-1).clamp_min(0.0).numpy()))


# 2/4 ---------------------------------------------------------------------------------------------
# One run of a library, and the double evaluation
# -------------------------------------------------------------------------------------------------
def check_gains(model, x_cand) -> float:
    """Largest relative gap between our fast computation of the gains and the acquisition of
    the library itself, on a few candidates (done once per run, at the first iteration)."""
    fast, library = model.gains(x_cand), model.library_gains(x_cand)
    return float(np.max(np.abs(fast - library)) / np.max(np.abs(library)))


def ours_on_design(problem, x_lists, y_lists):
    """Our CoKriging (v2) fitted on a design of a library, made nested by computing the missing
    LF values at the HF points. Returns the fitted model and the number of LF values added."""
    data = ExperimentData(bounds=rs.UNIT_BOUNDS, costs=bs.COSTS)
    for level, (x_level, y_level) in enumerate(zip(x_lists, y_lists), start=1):
        for x, y in zip(x_level, y_level):
            data.add_observation(level, x, float(y), {})
    missing = missing_points_for_nesting(data, num_levels=2)[1]
    for x in missing:
        data.add_observation(1, x, float(problem(np.atleast_2d(x), 1)[0]), {})
    model = bs.make_model("v2")
    model.fit(data)
    return model, len(missing)


def run_task(task):
    """One run (library, seed, budget): final scores (own and ours), error after each point."""
    library, seed, budget = task
    start_time = time.perf_counter()
    region = rs.region_data()
    global_test = bs.test_set("airfoil")
    problem = bs.make_problem("airfoil")
    start = bs.nested_design(problem, rs.START, seed)         # same start as S1 for this seed
    x_lists = [start.get_training_data(l)[0] for l in (1, 2)]
    y_lists = [start.get_training_data(l)[1] for l in (1, 2)]
    model = (EmukitAR1 if library == "E1" else BotorchMF)(region["x_int"], seed)
    spent, curve, iteration = 0.0, [], 0
    fit_times, gain_times = [], []
    gain_check_gap = np.nan                          # set at the first iteration
    while budget - spent >= min(bs.COSTS):
        t0 = time.perf_counter()
        model.fit(x_lists, y_lists)
        t1 = time.perf_counter()
        curve.append({"cost": bs.design_cost(rs.START) + spent,
                      **rs.all_scores(model, region, global_test)})
        # 1024 new Sobol candidates at each iteration (as S1), gain per unit of cost
        x_cand = qmc.Sobol(d=4, scramble=True, seed=seed * 1000 + iteration).random(N_CANDIDATES)
        if iteration == 0:
            gain_check_gap = check_gains(model, x_cand[:8])
        ratios = model.gains(x_cand) / np.asarray(bs.COSTS)
        ratios[:, np.asarray(bs.COSTS) > budget - spent] = -np.inf      # unaffordable level
        fit_times.append(t1 - t0)
        gain_times.append(time.perf_counter() - t1)
        best = np.unravel_index(np.argmax(ratios), ratios.shape)
        x_new, level = x_cand[best[0]], best[1] + 1
        x_lists[level - 1] = np.vstack([x_lists[level - 1], x_new])
        y_lists[level - 1] = np.append(y_lists[level - 1], problem(np.atleast_2d(x_new), level))
        spent += bs.COSTS[level - 1]
        iteration += 1
    model.fit(x_lists, y_lists)
    own = rs.all_scores(model, region, global_test)
    curve.append({"cost": bs.design_cost(rs.START) + spent, **own})
    ours_model, n_added = ours_on_design(problem, x_lists, y_lists)
    ours = rs.all_scores(ours_model, region, global_test)
    new_points = np.vstack([x[n:] for x, n in zip(x_lists, rs.START)])
    row = {"strategy": library, "seed": seed, "total_cost": bs.design_cost(rs.START) + spent,
           "n_lf": len(x_lists[0]), "n_hf": len(x_lists[1]), "iterations": iteration,
           "lf_added_for_ours": n_added, "gain_check_gap": gain_check_gap,
           "fraction_in_region": float(np.mean(rs.in_region(new_points))),
           "fit_time_median_s": float(np.median(fit_times)),
           "gain_time_median_s": float(np.median(gain_times)),
           "wall_time_s": time.perf_counter() - start_time,
           **{f"{key}": value for key, value in own.items()},
           **{f"{key}_ours": value for key, value in ours.items()}}
    return row, [{"strategy": library, "seed": seed, **point} for point in curve]


# 3/4 ---------------------------------------------------------------------------------------------
# Summary and criterion
# -------------------------------------------------------------------------------------------------
def collect(libraries: pd.DataFrame) -> pd.DataFrame:
    """One row per (strategy, seed): S1 and B1 of region_study.py (seeds of the library runs),
    then the libraries. For S1 and B1 the "ours" evaluation is the same as "own"."""
    reference = pd.read_csv(bs.RESULTS / "region_study.csv")
    reference = reference[reference["strategy"].isin(["S1", "B1"])
                          & reference["seed"].isin(libraries["seed"].unique())].copy()
    for column in ("rmse_region", "coverage_region", "rmse_global"):
        reference[f"{column}_ours"] = reference[column]
    return pd.concat([reference, libraries], ignore_index=True)


def summarize(results: pd.DataFrame) -> str:
    n_seeds = results["seed"].nunique()
    lines = [f"=== Libraries on the region test (total cost 560), {n_seeds} seeds ===",
             "region RMSE: median [10 % ; 90 %], own model | our CoKriging on the design"]
    for strategy in ("S1", "B1") + LIBRARIES:
        r = results[results["strategy"] == strategy]
        if r.empty:
            continue
        lines.append(
            f"{LABELS[strategy]:36s} own {r['rmse_region'].median():.4f} "
            f"[{r['rmse_region'].quantile(0.1):.4f} ; {r['rmse_region'].quantile(0.9):.4f}] | "
            f"ours {r['rmse_region_ours'].median():.4f} | cov95 "
            f"{r['coverage_region'].median():.2f} | global {r['rmse_global'].median():.4f} | "
            f"LF/HF {r['n_lf'].median():.0f}/{r['n_hf'].median():.0f} | in region "
            f"{100 * r['fraction_in_region'].median():.0f} % | time "
            f"{r['wall_time_s'].median():.0f} s")
    own = results.pivot(index="seed", columns="strategy", values="rmse_region")
    ours = results.pivot(index="seed", columns="strategy", values="rmse_region_ours")
    lines.append("")
    for library in LIBRARIES:
        if library not in own or "S1" not in own:
            continue
        for label, table in (("own", own), ("ours", ours)):
            wins = int((table["S1"] < table[library]).sum())
            p_value = wilcoxon(table["S1"], table[library], alternative="less").pvalue
            change = 1.0 - table["S1"].median() / table[library].median()
            lines.append(f"S1 against {library} ({label} model): median error {100 * change:+.0f} "
                         f"% lower | wins {wins} / {len(table)} | Wilcoxon (S1 < {library}) "
                         f"p = {p_value:.2g}")
    available = [library for library in LIBRARIES if library in own]
    if "S1" in own and available:
        best = min(available, key=lambda library: own[library].median())
        ratio = own["S1"].median() / own[best].median()
        p_worse = wilcoxon(own["S1"], own[best], alternative="greater").pvalue
        p_better = wilcoxon(own["S1"], own[best], alternative="less").pvalue
        lines.append(f"\nCriterion fixed before the computation (L* = {best}, median ratio "
                     f"S1 / L* = {ratio:.3f}):")
        lines.append(f"  [{'OK' if ratio <= 1.05 and p_worse >= 0.05 else 'NO'}] S1 at least as "
                     f"good as L* (ratio <= 1.05 and Wilcoxon S1 > L* p = {p_worse:.2g} >= 0.05)")
        lines.append(f"  [{'OK' if ratio <= 0.90 and p_better < 0.05 else 'NO'}] S1 better than "
                     f"L* (ratio <= 0.90 and Wilcoxon S1 < L* p = {p_better:.2g} < 0.05)")
    return "\n".join(lines)


# 4/4 ---------------------------------------------------------------------------------------------
# Figure, main
# -------------------------------------------------------------------------------------------------
def make_figure(results: pd.DataFrame, curves: pd.DataFrame) -> None:
    """Region error per seed of each strategy, own model and our model (left); median error on
    the region against the cost (right)."""
    import plotly.graph_objects as go  # noqa: PLC0415
    from figure_style import SERIES, save, style  # noqa: PLC0415
    from plotly.subplots import make_subplots  # noqa: PLC0415

    order = ("S1", "B1") + LIBRARIES
    colors = dict(zip(order, SERIES))
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.1, column_widths=[0.45, 0.55],
                        subplot_titles=("Erreur sur la région, par graine (coût 560)",
                                        "Erreur sur la région au fil du budget (médiane)"))
    rng = np.random.default_rng(0)
    for position, strategy in enumerate(order):
        r = results[results["strategy"] == strategy]
        if r.empty:
            continue
        for offset, column, symbol, text in ((-0.15, "rmse_region", "circle", "modèle propre"),
                                             (0.15, "rmse_region_ours", "diamond-open",
                                              "notre CoKriging sur le plan")):
            if strategy in ("S1", "B1") and column.endswith("ours"):
                continue
            x = position + offset + rng.uniform(-0.05, 0.05, len(r))
            fig.add_trace(go.Scatter(
                x=x, y=r[column], mode="markers", showlegend=False,
                marker={"color": colors[strategy], "symbol": symbol, "size": 8},
                hovertemplate="graine %{customdata} : %{y:.4f}<extra>" + LABELS[strategy]
                              + ", " + text + "</extra>", customdata=r["seed"]), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=[position + offset - 0.12, position + offset + 0.12],
                y=[r[column].median()] * 2, mode="lines", showlegend=False,
                line={"color": colors[strategy], "width": 3}, hoverinfo="skip"), row=1, col=1)
    fig.update_xaxes(tickvals=list(range(len(order))), ticktext=[LABELS[s].split(" (")[0]
                                                                  for s in order], row=1, col=1)
    fig.update_yaxes(title_text="RMSE / écart-type de la HF (région)", rangemode="tozero",
                     row=1, col=1)
    start_cost = bs.design_cost(rs.START)
    cost_grid = np.arange(start_cost, start_cost + rs.BUDGET + 1.0, 1.0)
    reference = pd.read_csv(bs.RESULTS / "region_curves.csv")
    reference = reference[(reference["strategy"] == "S1")
                          & reference["seed"].isin(results["seed"].unique())]
    for strategy, table in (("S1", reference),) + tuple(
            (library, curves[curves["strategy"] == library]) for library in LIBRARIES):
        per_seed = []
        for _, group in table.groupby("seed"):
            group = group.sort_values("cost")
            index = np.searchsorted(group["cost"].to_numpy(), cost_grid, side="right") - 1
            per_seed.append(group["rmse_region"].to_numpy()[np.maximum(index, 0)])
        if per_seed:
            fig.add_trace(go.Scatter(
                x=cost_grid, y=np.median(np.array(per_seed), axis=0), name=LABELS[strategy],
                line={"color": colors[strategy], "width": 2, "shape": "hv"},
                hovertemplate="coût %{x:.0f} : %{y:.4f}<extra>" + LABELS[strategy]
                              + "</extra>"), row=1, col=2)
    b1 = results[results["strategy"] == "B1"]["rmse_region"]
    if not b1.empty:
        fig.add_trace(go.Scatter(x=[start_cost + rs.BUDGET], y=[b1.median()], mode="markers",
                                 name=LABELS["B1"], marker={"size": 11, "color": colors["B1"],
                                                            "symbol": "square"}), row=1, col=2)
    fig.update_xaxes(title_text="coût total (BF = 1, HF = 10)", row=1, col=2)
    fig.update_yaxes(title_text="RMSE / écart-type de la HF (région)", rangemode="tozero",
                     row=1, col=2)
    style(fig, "Notre co-krigeage séquentiel contre emukit et BoTorch (test région)",
          f"Profil NACA 4D, région = portance BF entre 0,4 et 0,6 ; {results['seed'].nunique()} "
          "graines appariées ; ronds : modèle de la méthode, losanges : notre CoKriging réajusté "
          "sur le plan", height=600, legend_rows=2)
    print(f"figure: {save(fig, bs.FIGURES, 'library_study')}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--quick", action="store_true",
                        help=f"budget {QUICK_BUDGET:.0f} instead of {rs.BUDGET:.0f} (timing check)")
    parser.add_argument("--figure-only", action="store_true")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")         # the labels contain accents and rho
    suffix = "_quick" if args.quick else ""
    if args.figure_only:
        results = pd.read_csv(bs.RESULTS / f"library_study{suffix}.csv")
        curves = pd.read_csv(bs.RESULTS / f"library_curves{suffix}.csv")
    else:
        rs.region_data()                            # computed once, before the workers start
        budget = QUICK_BUDGET if args.quick else rs.BUDGET
        tasks = [(library, seed, budget) for seed in range(args.seeds) for library in LIBRARIES]
        rows, curve_rows = [], []
        start = time.perf_counter()
        with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks))) as pool:
            for row, curve in pool.map(run_task, tasks):
                rows.append(row)
                curve_rows += curve
                print(f"{row['strategy']} seed {row['seed']}: region RMSE own "
                      f"{row['rmse_region']:.4f}, ours {row['rmse_region_ours']:.4f}, LF/HF "
                      f"{row['n_lf']}/{row['n_hf']}, fit {row['fit_time_median_s']:.1f} s, "
                      f"gain {row['gain_time_median_s']:.1f} s, {row['wall_time_s']:.0f} s "
                      f"({time.perf_counter() - start:.0f} s elapsed)", flush=True)
        results = collect(pd.DataFrame(rows))
        curves = pd.DataFrame(curve_rows)
        results.to_csv(bs.RESULTS / f"library_study{suffix}.csv", index=False)
        curves.to_csv(bs.RESULTS / f"library_curves{suffix}.csv", index=False)
    summary = summarize(results)
    print(summary)
    (bs.RESULTS / f"library_summary{suffix}.txt").write_text(summary + "\n", encoding="utf-8")
    if not args.quick:
        make_figure(results, curves)


if __name__ == "__main__":
    main()
