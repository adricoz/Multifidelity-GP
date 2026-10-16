"""
[V2] Shared helpers of the mfego v2 benchmarks (run_selection.py, run_benchmark.py):
problems, test sets and designs of experiments computed ONCE and cached in results/ (XFOIL is
slow), mfego model variants (v1 / v2 options).

Problems (inputs in [0, 1]^d, 2 levels, costs 1 and 10):
- "hartmann":      multi-fidelity Hartmann 6D (bench_lib.HartmannMF, Sacher Eqs. 30-32);
- "airfoil":       NACA 4-digit drag, NeuralFoil xxsmall / xxxlarge (cokriging/airfoil_problem);
- "airfoil_xfoil": same sections, NeuralFoil xxxlarge / XFOIL (a real low / high pair).
"""
import os
import pickle
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

# pylint: disable=wrong-import-position,import-error
import numpy as np  # noqa: E402
from scipy.stats import qmc  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "cokriging"))
import bench_lib as bl  # noqa: E402
from airfoil_problem import AirfoilMF, AirfoilXfoilMF  # noqa: E402
from src.cokriging import CoKriging  # noqa: E402
from src.kernels import Matern52Kernel, SquaredExponentialKernel  # noqa: E402
from src.surrogate_models import V1_PRIORS, MultifidelityModel  # noqa: E402

RESULTS = HERE / "results"
PROBLEMS = {"hartmann": bl.HartmannMF, "airfoil": AirfoilMF, "airfoil_xfoil": AirfoilXfoilMF}
SIZES = {"hartmann": [10, 20, 40], "airfoil": [5, 10, 20], "airfoil_xfoil": [5, 10, 20]}
N_TEST = {"hartmann": 2000, "airfoil": 1000, "airfoil_xfoil": 600}
LF_PER_HF = 4                     # n_LF = 4 n_HF in every design
SEEDS = list(range(10))
KERNELS = {"SE": SquaredExponentialKernel, "Matern": Matern52Kernel}


def v1_options() -> dict:
    """Options of mfego v1 (16/10/2026): priors without rho, constant of the residual at
    rho = 1, squared exponential kernel."""
    return {"use_map": True, "priors": dict(V1_PRIORS), "rho_centering": False,
            "kernel_class": SquaredExponentialKernel}


# [V2] kernel of mfego v2 (chosen by run_selection.py)
V2_KERNEL = Matern52Kernel


def v2_options() -> dict:
    """Options of mfego v2 (defaults of CoKriging until v3): MAP, centred rho profiled with the
    N(1, 2^2) prior, bounds +-5, kernel chosen by the selection."""
    return {"use_map": True, "rho_centering": True, "priors": {"rho": ("normal", 1.0, 2.0)},
            "rho_bounds": (-5.0, 5.0), "kernel_class": V2_KERNEL}


NOMINAL = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
Z_EDGES = np.linspace(-5.0, 5.0, 41)


def calibration_curves(y: np.ndarray, mean: np.ndarray, var: np.ndarray) -> dict:
    """Coverage of the central intervals for NOMINAL probabilities and histogram of the
    standardized errors (y - mean) / std on Z_EDGES (JSON lists, for the figures)."""
    import json  # noqa: PLC0415
    from scipy.stats import norm  # noqa: PLC0415
    z = (y - mean) / np.sqrt(np.maximum(var, 1e-300))
    return {"coverage": json.dumps([float(np.mean(np.abs(z) <= norm.ppf(0.5 + p / 2)))
                                    for p in NOMINAL]),
            "z_hist": json.dumps(np.histogram(np.clip(z, -5, 5), Z_EDGES)[0].tolist())}


def make_model(model_type: str, num_levels: int, seed: int, options: dict):
    """MultifidelityModel ("recursive") or CoKriging ("cokriging") with the options."""
    options = dict(options)
    kernel_class = options.pop("kernel_class", SquaredExponentialKernel)
    if model_type == "cokriging":
        return CoKriging(num_levels, kernel_class, seed=seed, **options)
    return MultifidelityModel(num_levels, kernel_class, seed=seed, **options)


# -------------------------------------------------------------------------------------------------
# Test sets and designs, computed once (in parallel) and cached
# -------------------------------------------------------------------------------------------------
def _test_set_task(args):
    name, x = args
    return PROBLEMS[name]()(x, 2)


def test_set(name: str) -> tuple[np.ndarray, np.ndarray]:
    """Latin hypercube test points and their high-fidelity values (failed XFOIL runs dropped)."""
    path = RESULTS / f"test_set_{name}.npz"
    if not path.exists():
        RESULTS.mkdir(parents=True, exist_ok=True)
        problem = PROBLEMS[name]()
        x = qmc.LatinHypercube(d=problem.dim, seed=12345).random(N_TEST[name])
        chunks = np.array_split(x, 60)
        y = np.concatenate(bl.run_parallel(_test_set_task, [(name, c) for c in chunks]))
        valid = np.isfinite(y)
        np.savez(path, x=x[valid], y=y[valid])
    data = np.load(path)
    return data["x"], data["y"]


def _design_task(args):
    """(problem, kind, n_HF, seed) -> datasets [(x_LF, y_LF), (x_HF, y_HF)]."""
    name, kind, n_hf, seed = args
    problem = PROBLEMS[name]()
    if kind == "nested":
        return bl.nested_doe(problem, [LF_PER_HF * n_hf, n_hf], seed=seed)
    return problem.doe([LF_PER_HF * n_hf, n_hf], seed=seed)


def designs(seeds=SEEDS) -> dict:
    """{(problem, kind, n_HF, seed): datasets} for kind in ("nested", "independent"), cached
    in results/designs.pkl (computed in parallel the first time)."""
    path = RESULTS / "designs.pkl"
    cache = pickle.loads(path.read_bytes()) if path.exists() else {}
    keys = [(p, k, n, s) for p in PROBLEMS for k in ("nested", "independent")
            for n in SIZES[p] for s in seeds]
    missing = [key for key in keys if key not in cache]
    if missing:
        RESULTS.mkdir(parents=True, exist_ok=True)
        cache.update(zip(missing, bl.run_parallel(_design_task, missing)))
        path.write_bytes(pickle.dumps(cache))
    return cache
