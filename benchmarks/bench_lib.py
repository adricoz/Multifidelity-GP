"""
Helpers of the Hartmann benchmark notebook (benchmarks/hartmann_benchmark.ipynb): mfego compared
with scikit-learn, SMT and BoTorch on the multi-fidelity Hartmann 6D problem of Sacher et al.
(2021), Eqs. 30-32.

* HartmannMF: the multi-fidelity problem (U_k sequence, shift delta / k) and its costs.
* Surrogate wrappers with a common interface: fit(datasets) and predict(x) -> (mean, var), where
  datasets = [(x_1, y_1), ..., (x_L, y_L)] (level L = high fidelity).
* Optimization loops at equal cost budget: mfego NN-MF-EGO / SF-EGO, BoTorch qLogEI (single
  fidelity) and BoTorch multi-fidelity knowledge gradient (cost-aware).

The optional libraries (smt, botorch) are imported lazily: the mfego and scikit-learn parts work
without them (see requirements-benchmark.txt for the dedicated environment).
"""
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution
from scipy.stats import qmc

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "mfego") not in sys.path:
    sys.path.insert(0, str(ROOT / "mfego"))

# pylint: disable=wrong-import-position,import-error
from src.acquisition import AcquisitionFunction  # noqa: E402
from src.cokriging import CoKriging  # noqa: E402
from src.data_management import ExperimentData  # noqa: E402
from src.kernels import SquaredExponentialKernel  # noqa: E402
from src.optimizer import EGOOptimizer  # noqa: E402
from src.simulator import BaseSimulator  # noqa: E402
from src.surrogate_models import MultifidelityModel  # noqa: E402


# 1/4 ---------------------------------------------------------------------------------------------
# Multi-fidelity Hartmann 6D problem (Sacher et al. Eqs. 30-32)
# -------------------------------------------------------------------------------------------------
H6_ALPHA = np.array([1.0, 1.2, 3.0, 3.2])
H6_A = np.array([[10, 3, 17, 3.5, 1.7, 8], [0.05, 10, 17, 0.1, 8, 14],
                 [3, 3.5, 1.7, 10, 17, 8], [17, 8, 0.05, 10, 0.1, 14]], dtype=float)
H6_P = 1e-4 * np.array([[1312, 1696, 5569, 124, 8283, 5886],
                        [2329, 4135, 8307, 3736, 1004, 9991],
                        [2348, 1451, 3522, 2883, 3047, 6650],
                        [4047, 8828, 8732, 5743, 1091, 381]], dtype=float)
H6_XOPT = np.array([0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573])
H6_MIN = -3.32236801141551


def hartmann6(x: np.ndarray) -> np.ndarray:
    """Standard Hartmann 6D function, vectorized on (m, 6) arrays."""
    x = np.atleast_2d(x)
    inner = np.einsum("ij,mij->mi", H6_A, (x[:, None, :] - H6_P[None]) ** 2)
    return -np.exp(-inner) @ H6_ALPHA


def hartmann_level(x: np.ndarray, k: float, delta: float = 0.0, u0: float = -5.0) -> np.ndarray:
    """U_k(x + delta / k) of Eqs. 31-32 (k = inf: the Hartmann function itself)."""
    if np.isinf(k):
        return hartmann6(x)
    f_true = hartmann6(np.atleast_2d(x) + delta / k)
    u_k = np.full_like(f_true, u0)
    for _ in range(int(k)):
        u_k = 0.5 * (f_true ** 2 / u_k + u_k)
    return u_k


@dataclass
class HartmannMF:
    """Multi-fidelity Hartmann problem: level l uses U_{ks[l-1]} (last level = Hartmann)."""
    ks: tuple = (1, np.inf)
    delta: float = 0.05
    costs: tuple = (1.0, 10.0)
    dim: int = 6
    f_star: float = H6_MIN
    n_evaluations: dict = field(default_factory=dict)

    @property
    def num_levels(self) -> int:
        return len(self.ks)

    def __call__(self, x: np.ndarray, level: int) -> np.ndarray:
        """Objective of a level (1 = lowest fidelity), vectorized."""
        x = np.atleast_2d(x)
        self.n_evaluations[level] = self.n_evaluations.get(level, 0) + len(x)
        return hartmann_level(x, self.ks[level - 1], self.delta)

    def doe(self, points_per_level: list, seed: int = 0) -> list:
        """Independent (non-nested) LHS per level: [(x_1, y_1), ..., (x_L, y_L)]."""
        datasets = []
        for level, n_points in enumerate(points_per_level, start=1 + self.num_levels
                                         - len(points_per_level)):
            x = qmc.LatinHypercube(d=self.dim, seed=seed * 100 + level).random(n_points)
            datasets.append((x, self(x, level)))
        return datasets


def nested_doe(problem, points_per_level: list, seed: int = 0) -> list:
    """[CK] Nested design X_L in ... in X_1 of a problem with all its levels (LHS of level 1,
    greedy maximin subsets above, ExperimentData.generate_nested_design):
    [(x_1, y_1), ..., (x_L, y_L)]."""
    data = ExperimentData(bounds=[(0.0, 1.0)] * problem.dim, costs=[1.0] * len(points_per_level))
    data.generate_nested_design(points_per_level, seed=seed * 100)
    return [(data.x_dict[l], problem(data.x_dict[l], l))
            for l in range(1, len(points_per_level) + 1)]


def experiment_data(datasets: list) -> ExperimentData:
    """ExperimentData of the datasets [(x_1, y_1), ..., (x_L, y_L)] (inputs in [0, 1]^d)."""
    data = ExperimentData(bounds=[(0.0, 1.0)] * datasets[0][0].shape[1],
                          costs=[1.0] * len(datasets))
    for level, (x, y) in enumerate(datasets, start=1):
        data.x_dict[level], data.y_dict[level] = np.asarray(x), np.asarray(y, dtype=float)
        data.metrics_dict[level] = [{} for _ in y]
    return data


# 2/4 ---------------------------------------------------------------------------------------------
# Surrogate wrappers (common interface)
# -------------------------------------------------------------------------------------------------
class Surrogate:
    """fit(datasets) with datasets = [(x_l, y_l)] ordered by fidelity; predict(x) -> (mean, var)
    of the highest level. multi_fidelity=False wrappers only use the last dataset."""
    name = "surrogate"
    multi_fidelity = False

    def fit(self, datasets: list) -> "Surrogate":
        raise NotImplementedError

    def predict(self, x: np.ndarray) -> tuple:
        raise NotImplementedError

    @property
    def rhos(self) -> list:
        return []


class MfegoSurrogate(Surrogate):
    """mfego MultifidelityModel (rho computed at every level by default). model_class may be a
    subclass of MultifidelityModel (e.g. other hyperparameter priors), model_kwargs go to it."""
    def __init__(self, multi_fidelity: bool = True, seed: int = 0,
                 model_class: type = MultifidelityModel, **model_kwargs):
        self.multi_fidelity = multi_fidelity
        self.name = "mfego " + ("MF" if multi_fidelity else "SF")
        self.seed = seed
        self.model_class = model_class
        self.model_kwargs = model_kwargs
        self.model = None

    def fit(self, datasets):
        datasets = datasets if self.multi_fidelity else datasets[-1:]
        kwargs = dict(self.model_kwargs)
        kernel_class = kwargs.pop("kernel_class", SquaredExponentialKernel)
        self.model = self.model_class(len(datasets), kernel_class, seed=self.seed, **kwargs)
        self.model.fit(experiment_data(datasets))
        return self

    def predict(self, x):
        mean, var, _ = self.model.predict_batch(np.atleast_2d(x))
        return mean, var

    @property
    def rhos(self):
        return [float(r) for r in self.model.rhos]


class CoKrigingSurrogate(Surrogate):
    """[CK] mfego CoKriging: co-kriging conditioned on all the data, nested designs only
    (model_kwargs go to src.cokriging.CoKriging, e.g. priors=...)."""
    name = "mfego CoKriging (MF)"
    multi_fidelity = True

    def __init__(self, seed: int = 0, **model_kwargs):
        self.seed = seed
        self.model_kwargs = model_kwargs
        self.model = None

    def fit(self, datasets):
        self.model = CoKriging(len(datasets), seed=self.seed, **self.model_kwargs)
        self.model.fit(experiment_data(datasets))
        return self

    def predict(self, x):
        return self.model.predict_batch(np.atleast_2d(x))

    @property
    def rhos(self):
        return [float(r) for r in self.model.rhos]


class EmukitAR1Surrogate(Surrogate):
    """[CK] GPy + emukit linear multi-fidelity model: AR(1) co-kriging of Kennedy & O'Hagan
    conditioned on all the data ([LG] Eq. 3.11), all the hyperparameters estimated JOINTLY by
    maximum likelihood (no prior). Same kernel as mfego for each level (ARD RBF + constant
    bias), Gaussian noise per level, same bounds as mfego. Each level is centred by its own mean
    (constant trend of each level, as mfego; compatible with the AR(1) structure, it only changes
    the constant of delta_l) and all the levels are divided by a common scale.
    priors (e.g. DEFAULT_PRIORS of mfego): joint MAP with the same prior densities. set_prior
    replaces the bounds by the positive (softplus) transformation of GPy and adds its log-Jacobian
    log(1 - exp(-theta)): close to a log-space MAP for small values (noise, t2), close to the
    parameter-space MAP of mfego for t1 and large lengthscales."""
    multi_fidelity = True

    def __init__(self, seed: int = 0, n_restarts: int = 5, priors: dict = None):
        self.seed = seed
        self.n_restarts = n_restarts
        self.priors = priors
        self.name = "GPy/emukit AR1 (MF, " + ("MLE" if priors is None else "MAP") + ")"
        self.model = None
        self.kernels = None
        self.y_means, self.y_std = [], 1.0
        self.datasets = None

    def fit(self, datasets):
        import GPy  # noqa: PLC0415
        from emukit.model_wrappers.gpy_model_wrappers import (  # noqa: PLC0415
            GPyMultiOutputWrapper)
        from emukit.multi_fidelity.convert_lists_to_array import (  # noqa: PLC0415
            convert_xy_lists_to_arrays)
        from emukit.multi_fidelity.kernels import LinearMultiFidelityKernel  # noqa: PLC0415
        from emukit.multi_fidelity.models import GPyLinearMultiFidelityModel  # noqa: PLC0415
        from src.surrogate_models import (BIAS_VARIANCE_BOUNDS, LENGTHSCALE_BOUNDS,  # noqa
                                          NOISE_BOUNDS, SIGNAL_VARIANCE_BOUNDS)
        np.random.seed(self.seed)  # GPy draws its restarts with the global generator
        self.datasets = [(np.asarray(x), np.asarray(y, dtype=float)) for x, y in datasets]
        self.y_means = [float(np.mean(y)) for _, y in self.datasets]
        self.y_std = float(np.std(np.concatenate([y - m for (_, y), m in
                                                  zip(self.datasets, self.y_means)])))
        x_array, y_array = convert_xy_lists_to_arrays(
            [x for x, _ in self.datasets],
            [((y - m) / self.y_std).reshape(-1, 1) for (_, y), m in zip(self.datasets,
                                                                          self.y_means)])
        dim = self.datasets[0][0].shape[1]
        self.kernels = [GPy.kern.RBF(dim, ARD=True) + GPy.kern.Bias(dim) for _ in datasets]
        for kernel in self.kernels:
            kernel.parts[0].lengthscale.constrain_bounded(*LENGTHSCALE_BOUNDS, warning=False)
            kernel.parts[0].variance.constrain_bounded(*SIGNAL_VARIANCE_BOUNDS, warning=False)
            kernel.parts[1].variance.constrain_bounded(*BIAS_VARIANCE_BOUNDS, warning=False)
        gpy_model = GPyLinearMultiFidelityModel(x_array, y_array,
                                                LinearMultiFidelityKernel(self.kernels),
                                                n_fidelities=len(datasets))
        for likelihood in gpy_model.likelihood.likelihoods_list:
            likelihood.variance.constrain_bounded(*NOISE_BOUNDS, warning=False)
        if self.priors is not None:
            for kernel, likelihood in zip(self.kernels, gpy_model.likelihood.likelihoods_list):
                for param, group in ((kernel.parts[0].lengthscale, "lengthscale"),
                                     (kernel.parts[0].variance, "signal_variance"),
                                     (kernel.parts[1].variance, "bias_variance"),
                                     (likelihood.variance, "noise")):
                    if self.priors.get(group) is not None:
                        param.set_prior(_gpy_prior(self.priors[group]), warning=False)
        self.model = GPyMultiOutputWrapper(gpy_model, len(datasets),
                                           n_optimization_restarts=self.n_restarts,
                                           verbose_optimization=False)
        self.model.optimize()
        return self

    def predict(self, x, level: int = None):
        x = np.atleast_2d(x)
        level = len(self.kernels) if level is None else level
        mean, var = self.model.predict(np.column_stack([x, np.full(len(x), level - 1)]))
        return (self.y_means[level - 1] + self.y_std * mean[:, 0],
                self.y_std ** 2 * np.maximum(var[:, 0], 0.0))

    @property
    def rhos(self):
        return [float(r) for r in self.model.gpy_model.kern.scaling_param.values]

    def to_cokriging(self) -> CoKriging:
        """mfego CoKriging with the hyperparameters fitted by emukit (no fit): the predictions
        must be the same if both implement the same co-kriging (exactness check)."""
        model = CoKriging(len(self.kernels), SquaredExponentialKernel, use_map=False)
        model.rhos = self.rhos
        # GPy adds a constant 1e-8 to the noise variance (ExactGaussianInference: variance + 1e-8)
        noises = [float(lik.variance.values[0]) + 1e-8
                  for lik in self.model.gpy_model.likelihood.likelihoods_list]
        for l, (gp, kernel) in enumerate(zip(model.gps, self.kernels)):
            rbf, bias = kernel.parts
            gp.kernel.set_params(np.concatenate([rbf.lengthscale.values,
                                                 [rbf.variance.values[0], bias.variance.values[0]]]))
            gp.noise, gp.y_std = noises[l], self.y_std
            # zero mean of every centred level -> E[Z_l] = m_l = rho_(l-1) m_(l-1) + y_mean_l
            gp.y_mean = self.y_means[0] if l == 0 else (self.y_means[l]
                                                         - model.rhos[l - 1] * self.y_means[l - 1])
        model.condition([x for x, _ in self.datasets], [y for _, y in self.datasets])
        return model


class SklearnSurrogate(Surrogate):
    """scikit-learn GaussianProcessRegressor (ARD RBF + white noise, normalized y), HF only."""
    name = "scikit-learn GP (SF)"

    def __init__(self, seed: int = 0):
        self.seed = seed
        self.model = None

    def fit(self, datasets):
        from sklearn.gaussian_process import GaussianProcessRegressor  # noqa: PLC0415
        from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel  # noqa
        x, y = datasets[-1]
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) \
            * RBF(length_scale=np.ones(x.shape[1]), length_scale_bounds=(1e-2, 1e2)) \
            + WhiteKernel(1e-6, (1e-10, 1e-1))
        self.model = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                              n_restarts_optimizer=3, random_state=self.seed)
        self.model.fit(x, y)
        return self

    def predict(self, x):
        mean, std = self.model.predict(np.atleast_2d(x), return_std=True)
        return mean, std ** 2


class SmtSurrogate(Surrogate):
    """SMT KRG (single fidelity, HF only) or MFK (recursive co-kriging of Le Gratiet, rho
    estimated by GLS with rho_regr='constant')."""
    def __init__(self, multi_fidelity: bool = True):
        self.multi_fidelity = multi_fidelity
        self.name = "SMT MFK (MF)" if multi_fidelity else "SMT KRG (SF)"
        self.model = None

    def fit(self, datasets):
        from smt.applications.mfk import MFK  # noqa: PLC0415
        from smt.surrogate_models import KRG  # noqa: PLC0415
        dim = datasets[0][0].shape[1]
        if self.multi_fidelity:
            self.model = MFK(theta0=[1e-2] * dim, rho_regr="constant", print_global=False)
            for level, (x, y) in enumerate(datasets[:-1]):
                self.model.set_training_values(x, np.asarray(y).reshape(-1, 1), name=level)
        else:
            self.model = KRG(theta0=[1e-2] * dim, print_global=False)
        x, y = datasets[-1]
        self.model.set_training_values(x, np.asarray(y).reshape(-1, 1))
        self.model.train()
        return self

    def predict(self, x):
        x = np.atleast_2d(x)
        mean = self.model.predict_values(x)[:, 0]
        var = np.maximum(self.model.predict_variances(x)[:, 0], 0.0)
        return mean, var

    @property
    def rhos(self):
        if not self.multi_fidelity:
            return []
        return [float(np.ravel(par["beta"])[0]) for par in self.model.optimal_par[1:]]


def _gpy_prior(prior: tuple):
    """GPy prior with the same density as an mfego prior triple (family, a, b)."""
    import GPy  # noqa: PLC0415
    family, a, b = prior
    if family == "invgamma":
        return GPy.priors.InverseGamma(a, b)
    if family == "gamma":
        return GPy.priors.Gamma(a, b)
    return GPy.priors.LogGaussian(a, b)


def fidelity_values(costs) -> list:
    """Fidelity feature s_l in [0, 1] such that the affine cost c0 + w s_l equals the costs
    (BoTorch AffineFidelityCostModel)."""
    costs = np.asarray(costs, dtype=float)
    return list((costs - costs[0]) / (costs[-1] - costs[0])) if len(costs) > 1 else [1.0]


class BotorchSurrogate(Surrogate):
    """BoTorch SingleTaskGP (HF only) or SingleTaskMultiFidelityGP (fidelity s_l as an extra
    input, linear truncated fidelity kernel); predictions at the target fidelity s = 1."""
    def __init__(self, multi_fidelity: bool = True, costs=(1.0, 10.0)):
        self.multi_fidelity = multi_fidelity
        self.name = "BoTorch MF-GP (MF)" if multi_fidelity else "BoTorch SingleTaskGP (SF)"
        self.costs = costs
        self.model = None

    def fit(self, datasets):
        import torch  # noqa: PLC0415
        from botorch.fit import fit_gpytorch_mll  # noqa: PLC0415
        from botorch.models import SingleTaskGP, SingleTaskMultiFidelityGP  # noqa: PLC0415
        from botorch.models.transforms import Normalize, Standardize  # noqa: PLC0415
        from gpytorch.mlls import ExactMarginalLogLikelihood  # noqa: PLC0415
        dim = datasets[0][0].shape[1]
        if self.multi_fidelity:
            fids = fidelity_values(self.costs)
            x = np.vstack([np.column_stack([xl, np.full(len(xl), s)])
                           for (xl, _), s in zip(datasets, fids)])
            y = np.concatenate([yl for _, yl in datasets])
            train_x = torch.tensor(x, dtype=torch.double)
            self.model = SingleTaskMultiFidelityGP(
                train_x, torch.tensor(y, dtype=torch.double).unsqueeze(-1),
                data_fidelities=[dim], input_transform=Normalize(dim + 1),
                outcome_transform=Standardize(1))
        else:
            x, y = datasets[-1]
            self.model = SingleTaskGP(torch.tensor(x, dtype=torch.double),
                                      torch.tensor(y, dtype=torch.double).unsqueeze(-1),
                                      input_transform=Normalize(dim),
                                      outcome_transform=Standardize(1))
        fit_gpytorch_mll(ExactMarginalLogLikelihood(self.model.likelihood, self.model))
        return self

    def predict(self, x):
        import torch  # noqa: PLC0415
        x = np.atleast_2d(x)
        if self.multi_fidelity:
            x = np.column_stack([x, np.ones(len(x))])
        with torch.no_grad():
            posterior = self.model.posterior(torch.tensor(x, dtype=torch.double),
                                             observation_noise=True)
        return (posterior.mean.squeeze(-1).detach().numpy(),
                posterior.variance.squeeze(-1).clamp_min(0.0).detach().numpy())


# 3/4 ---------------------------------------------------------------------------------------------
# Accuracy metrics
# -------------------------------------------------------------------------------------------------
def accuracy(surrogate: Surrogate, datasets: list, x_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Fit the surrogate and compute RMSE / std, NLPD, 95% coverage and timings."""
    start = time.perf_counter()
    surrogate.fit(datasets)
    fit_time = time.perf_counter() - start
    start = time.perf_counter()
    mean, var = surrogate.predict(x_test)
    predict_time = time.perf_counter() - start
    var = np.maximum(var, 1e-12)
    return {"model": surrogate.name, "multi_fidelity": surrogate.multi_fidelity,
            "rmse_rel": float(np.sqrt(np.mean((mean - y_test) ** 2)) / np.std(y_test)),
            "nlpd": float(np.mean(0.5 * np.log(2 * np.pi * var)
                                  + (y_test - mean) ** 2 / (2 * var))),
            "coverage_95": float(np.mean(np.abs(mean - y_test) <= 1.96 * np.sqrt(var))),
            "fit_time_s": fit_time, "predict_time_s": predict_time,
            "rhos": surrogate.rhos}


def recommend(surrogate: Surrogate, dim: int, seed: int = 0) -> np.ndarray:
    """Final recommendation of a method: minimizer of the predicted mean of the highest level
    (same optimizer for every library: differential evolution on [0, 1]^d)."""
    result = differential_evolution(lambda x: float(surrogate.predict(x.reshape(1, -1))[0][0]),
                                    [(0.0, 1.0)] * dim, seed=seed, maxiter=60, popsize=10,
                                    tol=1e-8, polish=True)
    return result.x


# 4/4 ---------------------------------------------------------------------------------------------
# Optimization loops at equal cost budget
# -------------------------------------------------------------------------------------------------
@dataclass
class History:
    """Convergence history of one optimization run."""
    method: str
    seed: int
    cost: list = field(default_factory=list)
    best_observed: list = field(default_factory=list)
    levels: list = field(default_factory=list)
    iteration_time_s: list = field(default_factory=list)
    recommendation_error: float = np.nan
    rhos: list = field(default_factory=list)

    def record(self, cost, best, level, elapsed):
        self.cost.append(float(cost))
        self.best_observed.append(float(best))
        self.levels.append(int(level))
        self.iteration_time_s.append(float(elapsed))


def run_mfego(problem: HartmannMF, datasets: list, budget: float, seed: int = 0,
              multi_fidelity: bool = True, model_class: type = MultifidelityModel,
              return_model: bool = False, **model_kwargs):
    """mfego NN-MF-EGO (all the levels) or SF-EGO (highest level only) until the budget.
    model_kwargs are passed to model_class (MultifidelityModel or a subclass, e.g. use_map=True).
    The recommendation is evaluated with the highest level of the problem (Hartmann: f itself).
    [CK] return_model=True returns (history, final model, ExperimentData of the run).
    """
    datasets = datasets if multi_fidelity else datasets[-1:]
    costs = list(problem.costs) if multi_fidelity else [problem.costs[-1]]
    levels = list(range(1, problem.num_levels + 1)) if multi_fidelity else [problem.num_levels]
    data = ExperimentData(bounds=[(0.0, 1.0)] * problem.dim, costs=costs)
    for level, (x, y) in enumerate(datasets, start=1):
        data.x_dict[level], data.y_dict[level] = np.asarray(x), np.asarray(y, dtype=float)
        data.metrics_dict[level] = [{} for _ in y]

    class Simulator(BaseSimulator):
        def evaluate(self, design_point, level):
            return float(problem(np.atleast_2d(design_point), levels[level - 1])[0]), {}

    kernel_class = model_kwargs.pop("kernel_class", SquaredExponentialKernel)
    model = model_class(len(datasets), kernel_class, seed=seed, **model_kwargs)
    acq = AcquisitionFunction(model, data)
    ego = EGOOptimizer(data, model, Simulator(len(datasets)), acq, seed=seed,
                       save_state_path=str(Path(_tmp_dir()) / f"mfego_{seed}.json"))
    history = History(method="mfego NN-MF-EGO" if multi_fidelity else "mfego SF-EGO", seed=seed)
    ego._init_history()  # pylint: disable=protected-access
    history.record(ego.current_total_cost, ego.best_y_history[-1] - problem.f_star, 0, 0.0)
    while ego.current_total_cost < budget:
        start = time.perf_counter()
        x_next, l_next, merit = ego.ask()
        if merit <= 0.0 or data.is_already_evaluated(l_next, x_next):
            l_next = len(datasets)
            x_next = acq.x_best if not data.is_already_evaluated(l_next, acq.x_best) \
                else ego.rng.random(problem.dim)
        y, metrics = ego.simulator.evaluate(x_next, l_next)
        ego.tell(x_next, l_next, y, metrics)
        history.record(ego.current_total_cost, ego.best_y_history[-1] - problem.f_star,
                       levels[l_next - 1], time.perf_counter() - start)
    surrogate = MfegoSurrogate(multi_fidelity=multi_fidelity, seed=seed)
    surrogate.model = model
    model.fit(data)
    x_rec = recommend(surrogate, problem.dim, seed)
    history.recommendation_error = float(problem(x_rec, problem.num_levels)[0] - problem.f_star)
    history.rhos = surrogate.rhos
    return (history, model, data) if return_model else history


def run_botorch(problem: HartmannMF, datasets: list, budget: float, seed: int = 0,
                multi_fidelity: bool = True, num_fantasies: int = 32,
                num_restarts: int = 4, raw_samples: int = 128) -> History:
    """BoTorch: single-fidelity qLogExpectedImprovement on the highest level, or multi-fidelity
    knowledge gradient (qMultiFidelityKnowledgeGradient) with an affine cost model reproducing
    the costs of the problem (fidelity feature s_l, target s = 1). BoTorch maximizes: -f."""
    import torch  # noqa: PLC0415
    from botorch.acquisition import (PosteriorMean, qLogExpectedImprovement,  # noqa: PLC0415
                                     qMultiFidelityKnowledgeGradient)
    from botorch.acquisition.cost_aware import InverseCostWeightedUtility  # noqa: PLC0415
    from botorch.acquisition.fixed_feature import FixedFeatureAcquisitionFunction  # noqa
    from botorch.acquisition.utils import project_to_target_fidelity  # noqa: PLC0415
    from botorch.models.cost import AffineFidelityCostModel  # noqa: PLC0415
    from botorch.optim import optimize_acqf, optimize_acqf_mixed  # noqa: PLC0415

    torch.manual_seed(seed)
    dim = problem.dim
    datasets = [(np.asarray(x), np.asarray(y, dtype=float)) for x, y in
                (datasets if multi_fidelity else datasets[-1:])]
    fids = fidelity_values(problem.costs)
    history = History(method="BoTorch MF-KG" if multi_fidelity else "BoTorch qLogEI (SF)",
                      seed=seed)
    total_cost = sum(problem.costs[-len(datasets) + i] * len(y)
                     for i, (_, y) in enumerate(datasets))
    history.record(total_cost, np.min(datasets[-1][1]) - problem.f_star, 0, 0.0)
    surrogate = BotorchSurrogate(multi_fidelity=multi_fidelity, costs=problem.costs)
    bounds = torch.tensor([[0.0] * dim + ([0.0] if multi_fidelity else []),
                           [1.0] * dim + ([1.0] if multi_fidelity else [])], dtype=torch.double)

    while total_cost < budget:
        start = time.perf_counter()
        # BoTorch maximizes: the model is fitted on -f
        surrogate.fit([(x, -y) for x, y in datasets])
        model = surrogate.model
        if multi_fidelity:
            cost_model = AffineFidelityCostModel(
                fidelity_weights={dim: problem.costs[-1] - problem.costs[0]},
                fixed_cost=problem.costs[0])
            target = {dim: 1.0}

            def project(x):
                return project_to_target_fidelity(X=x, target_fidelities=target, d=dim + 1)
            current = FixedFeatureAcquisitionFunction(PosteriorMean(model), d=dim + 1,
                                                      columns=[dim], values=[1.0])
            _, current_value = optimize_acqf(current, bounds=bounds[:, :-1], q=1,
                                             num_restarts=num_restarts,
                                             raw_samples=raw_samples)
            acq = qMultiFidelityKnowledgeGradient(
                model=model, num_fantasies=num_fantasies, current_value=current_value,
                cost_aware_utility=InverseCostWeightedUtility(cost_model=cost_model),
                project=project)
            candidate, _ = optimize_acqf_mixed(
                acq, bounds=bounds, q=1, num_restarts=num_restarts, raw_samples=raw_samples,
                fixed_features_list=[{dim: float(s)} for s in fids],
                options={"batch_limit": 4, "maxiter": 100})
            candidate = candidate.detach().numpy()[0]
            level = int(np.argmin(np.abs(np.array(fids) - candidate[-1]))) + 1
            x_next = candidate[:-1]
        else:
            best_f = torch.tensor(np.max(-datasets[-1][1]), dtype=torch.double)
            acq = qLogExpectedImprovement(model=model, best_f=best_f)
            candidate, _ = optimize_acqf(acq, bounds=bounds, q=1, num_restarts=num_restarts,
                                         raw_samples=raw_samples)
            x_next, level = candidate.detach().numpy()[0], problem.num_levels
        y_next = problem(x_next, level if multi_fidelity else problem.num_levels)
        index = level - 1 if multi_fidelity else 0
        datasets[index] = (np.vstack([datasets[index][0], x_next]),
                           np.append(datasets[index][1], y_next))
        total_cost += problem.costs[level - 1]
        history.record(total_cost, np.min(datasets[-1][1]) - problem.f_star, level,
                       time.perf_counter() - start)
    surrogate.fit([(x, -y) for x, y in datasets])
    negated = _Negated(surrogate)
    history.recommendation_error = float(hartmann6(recommend(negated, dim, seed))[0]
                                         - problem.f_star)
    return history


class _Negated(Surrogate):
    """Wrapper returning the predictions of a model fitted on -f."""
    def __init__(self, surrogate):
        self.surrogate = surrogate

    def fit(self, datasets):
        self.surrogate.fit([(x, -np.asarray(y)) for x, y in datasets])
        return self

    def predict(self, x):
        mean, var = self.surrogate.predict(x)
        return -mean, var


def _tmp_dir() -> str:
    import tempfile  # noqa: PLC0415
    return tempfile.mkdtemp(prefix="mfego_bench_")


# -------------------------------------------------------------------------------------------------
# Parallel execution of the benchmark tasks (one process per task, one thread per process)
# -------------------------------------------------------------------------------------------------
MODELS = {
    "mfego MF": lambda seed, costs: MfegoSurrogate(multi_fidelity=True, seed=seed),
    "mfego SF": lambda seed, costs: MfegoSurrogate(multi_fidelity=False, seed=seed),
    "scikit-learn GP (SF)": lambda seed, costs: SklearnSurrogate(seed=seed),
    "SMT KRG (SF)": lambda seed, costs: SmtSurrogate(multi_fidelity=False),
    "SMT MFK (MF)": lambda seed, costs: SmtSurrogate(multi_fidelity=True),
    "BoTorch SingleTaskGP (SF)": lambda seed, costs: BotorchSurrogate(False, costs),
    "BoTorch MF-GP (MF)": lambda seed, costs: BotorchSurrogate(True, costs),
}


def _init_worker() -> None:
    """One thread per worker (BLAS / torch) to avoid oversubscription."""
    import os  # noqa: PLC0415
    os.environ["OMP_NUM_THREADS"] = "1"
    try:
        import torch  # noqa: PLC0415
        torch.set_num_threads(1)
    except ImportError:
        pass
    import warnings  # noqa: PLC0415
    warnings.filterwarnings("ignore")


def accuracy_task(args: tuple) -> dict:
    """(model name, n_LF, n_HF, seed, problem kwargs) -> accuracy metrics on 2000 test points."""
    model_name, n_lf, n_hf, seed, problem_kwargs = args
    problem = HartmannMF(**problem_kwargs)
    datasets = problem.doe([n_lf, n_hf], seed=seed)
    x_test = qmc.LatinHypercube(d=problem.dim, seed=12345).random(2000)
    try:
        result = accuracy(MODELS[model_name](seed, problem.costs), datasets, x_test,
                          hartmann6(x_test))
    except Exception as error:  # noqa: BLE001  (a failing library must not stop the benchmark)
        result = {"model": model_name, "error": repr(error)}
    result.update({"n_lf": n_lf, "n_hf": n_hf, "seed": seed})
    return result


def optimization_task(args: tuple) -> dict:
    """(method, seed, budget, problem kwargs, doe sizes) -> History as a dict."""
    from dataclasses import asdict  # noqa: PLC0415
    method, seed, budget, problem_kwargs, doe = args
    problem = HartmannMF(**problem_kwargs)
    mf_doe, sf_doe = doe
    try:
        if method == "mfego NN-MF-EGO":
            history = run_mfego(problem, problem.doe(mf_doe, seed), budget, seed, True)
        elif method == "mfego SF-EGO":
            history = run_mfego(problem, problem.doe(sf_doe, seed), budget, seed, False)
        elif method == "BoTorch qLogEI (SF)":
            history = run_botorch(problem, problem.doe(sf_doe, seed), budget, seed, False)
        else:
            history = run_botorch(problem, problem.doe(mf_doe, seed), budget, seed, True)
        return asdict(history)
    except Exception as error:  # noqa: BLE001
        return {"method": method, "seed": seed, "error": repr(error)}


def run_parallel(func, tasks: list, max_workers: int = None) -> list:
    """Runs the tasks in separate processes (results in the order of the tasks)."""
    import os  # noqa: PLC0415
    from concurrent.futures import ProcessPoolExecutor  # noqa: PLC0415
    max_workers = max_workers or max(1, min(len(tasks), (os.cpu_count() or 2) - 2))
    with ProcessPoolExecutor(max_workers=max_workers, initializer=_init_worker) as pool:
        return list(pool.map(func, tasks))
