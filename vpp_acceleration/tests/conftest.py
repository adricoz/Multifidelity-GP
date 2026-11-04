"""
Shared fixtures of the tests of vpp_acceleration.

mfego/src and vpp_acceleration/src are both put on sys.path: Python merges them into one
namespace package "src" (see the docstring of src/sequential_design.py).

Test problem: Forrester 1D of Sacher et al. 2021, Eq. 17 (same as mfego/tests/conftest.py):
    f1(x) = 0.5 (6x - 2)^2 sin(12x - 4) + 10 (x - 1)      (low fidelity)
    f2(x) = 2 f1(x) - 20 (x - 1)                          (high fidelity, true rho = 2)
"""
import sys
from pathlib import Path

import numpy as np
import pytest

VPP_DIR = Path(__file__).resolve().parents[1]       # .../Multifidelity-GP-Clean/vpp_acceleration
REPO_DIR = VPP_DIR.parent                            # .../Multifidelity-GP-Clean
sys.path.insert(0, str(REPO_DIR / "mfego"))
sys.path.insert(1, str(VPP_DIR))

# pylint: disable=wrong-import-position
from src.cokriging import CoKriging  # noqa: E402
from src.data_management import ExperimentData  # noqa: E402
from src.kernels import SquaredExponentialKernel  # noqa: E402
from src.sequential_design import evaluate_initial_design  # noqa: E402
from src.simulator import BaseSimulator  # noqa: E402


def forrester_lf(x):
    """Low fidelity f1 (Sacher Eq. 17)."""
    x = np.asarray(x, dtype=float)
    return 0.5 * (6 * x - 2) ** 2 * np.sin(12 * x - 4) + 10 * (x - 1)


def forrester_hf(x):
    """High fidelity f2 = 2 f1 - 20 (x - 1) (Sacher Eq. 17, true rho = 2)."""
    x = np.asarray(x, dtype=float)
    return 2 * forrester_lf(x) - 20 * (x - 1)


class ForresterSimulator(BaseSimulator):
    """The two Forrester levels as a simulator, with a count of the runs of each level.
    fail_above: optional x above which the LOW fidelity fails (returns NaN), to test failures."""

    def __init__(self, fail_above: float = None):
        super().__init__(num_levels=2)
        self.fail_above = fail_above
        self.calls = {1: 0, 2: 0}

    def evaluate(self, design_point, level):
        x = float(np.asarray(design_point, dtype=float).reshape(-1)[0])
        self.calls[level] += 1
        if level == 1 and self.fail_above is not None and x > self.fail_above:
            return np.nan, {}
        value = forrester_lf(x) if level == 1 else forrester_hf(x)
        return float(value), {}


def forrester_model():
    """CoKriging for Forrester: squared exponential kernel (Forrester is an analytic function,
    as in mfego/example/cokriging/main.py), mfego defaults otherwise (MAP, learned rho)."""
    return CoKriging(2, kernel_class=SquaredExponentialKernel, seed=0)


def forrester_data(points_per_level=(11, 5), seed=0):
    """Nested design X_2 in X_1 of the Forrester problem, evaluated. Costs 1 and 10."""
    data = ExperimentData(bounds=[(0.0, 1.0)], costs=[1.0, 10.0])
    data.generate_nested_design(list(points_per_level), seed=seed)
    evaluate_initial_design(data, ForresterSimulator())
    return data


@pytest.fixture
def fitted():
    """(model, data): CoKriging fitted on the nested Forrester design 11 LF / 5 HF."""
    data = forrester_data()
    model = forrester_model()
    model.fit(data)
    return model, data
