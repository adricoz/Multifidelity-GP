"""
Shared helpers of the analysis scripts.

* load_mfego("current" | "initial"): imports the `src` package of the corrected code (mfego/)
  or of the frozen initial code (legacy/legacy_mfego_initial/mfego/). The initial version
  imports `List, Tuple` from traitlets (bug R6): a small shim is injected so that it can be
  imported on any Python version for the "before/after" measurements.
* Analytical test functions (Forrester, Hartmann 6D standard and initial variant).
* save_figure(): writes an interactive plotly figure in analysis/figures/.
"""
import importlib
import json
import sys
import types
import typing
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "analysis" / "figures"
RESULTS_DIR = ROOT / "analysis" / "results"
VERSION_PATHS = {
    "current": ROOT / "mfego",
    "initial": ROOT / "legacy" / "legacy_mfego_initial" / "mfego",
}
MODULES = ["kernels", "data_management", "surrogate_models", "acquisition",
           "optimizer", "simulator", "visualization"]


def load_mfego(version: str = "current") -> SimpleNamespace:
    """Imports the `src` modules of a given version of the framework."""
    for name in list(sys.modules):
        if name == "src" or name.startswith("src."):
            del sys.modules[name]
    if version == "initial" and "traitlets" not in sys.modules:
        shim = types.ModuleType("traitlets")
        shim.List, shim.Tuple = typing.List, typing.Tuple
        sys.modules["traitlets"] = shim
    path = str(VERSION_PATHS[version])
    sys.path.insert(0, path)
    try:
        modules = {name: importlib.import_module(f"src.{name}") for name in MODULES}
    finally:
        sys.path.remove(path)
    return SimpleNamespace(**modules)


# Forrester (Eq. 17 of Sacher et al.) ------------------------------------------------------
def forrester_lf(x):
    """Low fidelity f1 of Eq. 17."""
    x = np.asarray(x, dtype=float)
    return 0.5 * (6 * x - 2) ** 2 * np.sin(12 * x - 4) + 10 * (x - 1)


def forrester_hf(x):
    """High fidelity f2 = 2 f1 - 20 (x - 1) of Eq. 17."""
    x = np.asarray(x, dtype=float)
    return 2 * forrester_lf(x) - 20 * (x - 1)


FORRESTER_MIN = -6.020740055767083

# Hartmann 6D (Eq. 30-32 of Sacher et al.) ---------------------------------------------------
H6_ALPHA = np.array([1.0, 1.2, 3.0, 3.2])
H6_A = np.array([[10, 3, 17, 3.5, 1.7, 8], [0.05, 10, 17, 0.1, 8, 14],
                 [3, 3.5, 1.7, 10, 17, 8], [17, 8, 0.05, 10, 0.1, 14]], dtype=float)
H6_P = 1e-4 * np.array([[1312, 1696, 5569, 124, 8283, 5886],
                        [2329, 4135, 8307, 3736, 1004, 9991],
                        [2348, 1451, 3522, 2883, 3047, 6650],
                        [4047, 8828, 8732, 5743, 1091, 381]], dtype=float)
H6_XOPT = np.array([0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573])
H6_MIN = -3.32236801141551


def hartmann6(x):
    """Standard Hartmann 6D function, vectorized on (m, 6) arrays."""
    x = np.atleast_2d(x)
    inner = np.einsum("ij,mij->mi", H6_A, (x[:, None, :] - H6_P[None]) ** 2)
    return -np.exp(-inner) @ H6_ALPHA


def hartmann_level(x, k, delta=0.0, u0=-5.0):
    """Multi-fidelity sequence U_k(x + delta/k) of Eqs. 31-32 (k = inf -> Hartmann)."""
    if k is None or np.isinf(k):
        return hartmann6(x)
    f_true = hartmann6(np.atleast_2d(x) + delta / k)
    u_k = np.full_like(f_true, u0)
    for _ in range(int(k)):
        u_k = 0.5 * (f_true ** 2 / u_k + u_k)
    return u_k


# Output helpers -----------------------------------------------------------------------------
def save_figure(fig, name: str) -> Path:
    """Writes an interactive plotly figure (HTML, plotly.js from CDN) in analysis/figures."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / f"{name}.html"
    fig.write_html(path, include_plotlyjs="cdn")
    print(f"figure -> {path.relative_to(ROOT)}")
    return path


def save_results(results: dict, name: str) -> Path:
    """Writes the numerical results of a script (JSON) in analysis/results."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=lambda o: np.asarray(o).tolist())
    print(f"results -> {path.relative_to(ROOT)}")
    return path


def make_data(mod, bounds, costs, points, functions, seed=42):
    """ExperimentData filled with an LHS design evaluated with `functions[l-1](x)`."""
    data = mod.data_management.ExperimentData(bounds=bounds, costs=costs)
    if "seed" in mod.data_management.ExperimentData.generate_initial_design.__code__.co_varnames:
        data.generate_initial_design(points_per_level=points, seed=seed)
    else:
        data.generate_initial_design(points_per_level=points)
    for l, func in enumerate(functions, start=1):
        data.y_dict[l] = np.asarray(func(data.x_dict[l]), dtype=float).reshape(-1)
        data.metrics_dict[l] = [{} for _ in data.y_dict[l]]
    return data
