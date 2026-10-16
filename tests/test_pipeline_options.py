"""[V2] Unit tests of the GP options of the foil pipeline configuration (optimization.priors,
optimization.rho_centering, optimization.kernel), with fake solver backends."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# pylint: disable=wrong-import-position
from pipelines.bdtoolbox_foil import bridge  # noqa: E402
from pipelines.bdtoolbox_foil.config import from_dict  # noqa: E402
from pipelines.bdtoolbox_foil.run import run_optimization  # noqa: E402
from pipelines.bdtoolbox_foil.simulator import BdToolboxFoilSimulator  # noqa: E402
from pipelines.bdtoolbox_foil.solvers import Polar  # noqa: E402
from src.run_utils import create_run  # noqa: E402


def config(**optimization) -> dict:
    return {
        "name": "options", "kind": "section2d",
        "parametrization": {"type": "naca4", "n_points": 81, "fixed": {"p_position": 0.4}},
        "variables": [{"name": "m_camber", "lower": 0.0, "upper": 0.06},
                      {"name": "t_thickness", "lower": 0.08, "upper": 0.16}],
        "flow": {"re": 5e5, "alpha": [-4.0, 10.0, 0.5]},
        "levels": [{"solver": "neuralfoil", "cost": 1.0}, {"solver": "xfoil", "cost": 10.0}],
        "objective": {"type": "cd_at_cl", "cl_target": 0.5},
        "optimization": {"doe": [8, 4], "iterations": 2, "seed": 0, **optimization},
    }


class AnalyticalBackend:
    """Analytical polar: cl = 0.11 (alpha + 80 m), cd = cd0(t, offset) + 0.01 cl^2."""
    def __init__(self, offset=0.0):
        self.offset = offset

    def polar(self, section, workdir=None):
        del workdir
        alpha = np.arange(-4.0, 10.25, 0.5)
        cl = 0.11 * (alpha + 80.0 * section.parameters["m_camber"])
        cd = 0.006 + self.offset + 0.05 * (section.thickness - 0.12) ** 2 + 0.01 * cl ** 2
        return Polar(alpha=alpha, cl=cl, cd=cd)


def test_defaults_of_the_new_options():
    """The optimization pipeline keeps the mfego v1 behaviour by default."""
    optimization = from_dict(config()).optimization
    assert optimization["kernel"] == "squared_exponential"
    assert optimization["rho_centering"] is False and optimization["priors"] is None


@pytest.mark.parametrize("override", [{"kernel": "rbf"}, {"rho_centering": "yes"},
                                      {"priors": {"rhoo": None}}, {"priors": [1, 2]}])
def test_invalid_options_are_rejected(override):
    with pytest.raises(ValueError):
        from_dict(config(**override))


NOISE_PRIOR = {"noise": ["lognormal", -9.0, 2.0]}


@pytest.mark.parametrize("options, kernel, centring, expected", [
    # defaults: mfego v1 (squared exponential, rho not centred, no prior on rho)
    ({}, "SquaredExponentialKernel", False, ["rho none"]),
    # a priors override alone does not switch the prior on rho on
    ({"priors": NOISE_PRIOR}, "SquaredExponentialKernel", False,
     ["noise lognormal(-9, 2)", "rho none"]),
    # v2 options: Matern 5/2, centred rho with its default N(1, 2^2) prior
    ({"kernel": "matern52", "rho_centering": True, "priors": NOISE_PRIOR}, "Matern52Kernel", True,
     ["noise lognormal(-9, 2)", "rho normal(1, 2)"]),
])
def test_options_reach_the_model(tmp_path, options, kernel, centring, expected):
    """Kernel, priors overrides and centred rho of the configuration reach the GP of run.py."""
    problem = from_dict(config(**options))
    run = create_run(tmp_path, prefix=problem.name)
    simulator = BdToolboxFoilSimulator(problem, paths=bridge.resolve_paths(),
                                       backends=[AnalyticalBackend(),
                                                 AnalyticalBackend(offset=0.001)])
    results = run_optimization(problem, run, simulator)
    for text in expected:
        assert text in results["gp"]["estimator"]
    state = json.loads((run.run_dir / "surrogate.json").read_text(encoding="utf-8"))["surrogate"]
    assert state["kernel"] == kernel and state["rho_centering"] is centring
