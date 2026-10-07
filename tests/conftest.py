"""
Shared fixtures of the unit tests.

The framework is imported as `src.*` (same as mfego/main.py), so mfego/ is added to sys.path.
Every test that writes files (JSON, logs, figures) runs in a temporary directory.
"""
import importlib.util
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "mfego"))

# pylint: disable=wrong-import-position
from src.acquisition import AcquisitionFunction  # noqa: E402
from src.data_management import ExperimentData  # noqa: E402
from src.kernels import SquaredExponentialKernel  # noqa: E402
from src.optimizer import EGOOptimizer  # noqa: E402
from src.simulator import BaseSimulator  # noqa: E402
from src.surrogate_models import MultifidelityModel  # noqa: E402

FORRESTER_MIN = -6.020740055767083


def forrester_lf(x):
    """Low fidelity f1 of Eq. 17 of the reference article."""
    x = np.asarray(x, dtype=float)
    return 0.5 * (6 * x - 2) ** 2 * np.sin(12 * x - 4) + 10 * (x - 1)


def forrester_hf(x):
    """High fidelity f2 = 2 f1 - 20 (x - 1) of Eq. 17 (true rho = 2)."""
    x = np.asarray(x, dtype=float)
    return 2 * forrester_lf(x) - 20 * (x - 1)


class ForresterSimulator(BaseSimulator):
    """Forrester simulator (Eq. 17): level L is always the high fidelity function."""
    def evaluate(self, design_point, level):
        x = float(design_point[0])
        if self.num_levels == 1 or level == self.num_levels:
            y = float(forrester_hf(x))
        else:
            y = float(forrester_lf(x))
        return y, {"y": y}


def fill_data(data, functions):
    """Evaluates the DOE of `data` with functions[l-1] (vectorized on x[:, 0])."""
    for l, func in enumerate(functions, start=1):
        data.y_dict[l] = np.asarray(func(data.x_dict[l][:, 0]), dtype=float)
        data.metrics_dict[l] = [{} for _ in data.y_dict[l]]
    return data


def load_module(relative_path, name):
    """Imports a module (example scripts) from its path relative to the repository root."""
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def forrester_data():
    """2-level Forrester data (10 LF / 6 HF points, non-nested LHS)."""
    data = ExperimentData(bounds=[(0.0, 1.0)], costs=[1.0, 10.0])
    data.generate_initial_design(points_per_level=[10, 6])
    return fill_data(data, [forrester_lf, forrester_hf])


@pytest.fixture
def single_fidelity_data():
    """1-level Forrester data (8 HF points)."""
    data = ExperimentData(bounds=[(0.0, 1.0)], costs=[1.0])
    data.generate_initial_design(points_per_level=[8])
    return fill_data(data, [forrester_hf])


@pytest.fixture
def in_tmp(tmp_path, monkeypatch):
    """Runs the test in a temporary working directory."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


def make_optimizer(data, num_levels, seed=0, **model_kwargs):
    """Model + acquisition + optimizer on Forrester, with a seed."""
    model = MultifidelityModel(num_levels, SquaredExponentialKernel, seed=seed, **model_kwargs)
    acquisition = AcquisitionFunction(model=model, data=data)
    simulator = ForresterSimulator(num_levels=num_levels)
    return EGOOptimizer(data=data, model=model, simulator=simulator, acquisition=acquisition,
                        save_state_path="ego_backup.json", seed=seed)
