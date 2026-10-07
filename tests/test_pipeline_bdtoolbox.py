"""Unit tests of the mfego <-> bdToolbox foil pipeline (pipelines/bdtoolbox_foil)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# pylint: disable=wrong-import-position
from pipelines.bdtoolbox_foil import bridge, geometry, objectives  # noqa: E402
from pipelines.bdtoolbox_foil.config import from_dict, load_config  # noqa: E402
from pipelines.bdtoolbox_foil.run import run_optimization  # noqa: E402
from pipelines.bdtoolbox_foil.simulator import BdToolboxFoilSimulator  # noqa: E402
from pipelines.bdtoolbox_foil.solvers import Polar, SolverFailure  # noqa: E402
from src.run_utils import create_run  # noqa: E402

CONFIGS = sorted((ROOT / "pipelines" / "configs").glob("*.json"))


def base_config(**overrides) -> dict:
    config = {
        "name": "test", "kind": "section2d",
        "parametrization": {"type": "naca4", "n_points": 81, "fixed": {"p_position": 0.4}},
        "variables": [{"name": "m_camber", "lower": 0.0, "upper": 0.06},
                      {"name": "t_thickness", "lower": 0.08, "upper": 0.16}],
        "flow": {"re": 5e5, "alpha": [-4.0, 10.0, 0.5]},
        "levels": [{"solver": "neuralfoil", "cost": 1.0}, {"solver": "xfoil", "cost": 10.0}],
        "objective": {"type": "cd_at_cl", "cl_target": 0.5},
        "optimization": {"doe": [8, 4], "iterations": 3, "seed": 0},
    }
    config.update(overrides)
    return config


class FakeBackend:
    """Analytical polar: cl = 0.11 (alpha + 80 m), cd = cd0(t, level) + 0.01 cl^2."""
    def __init__(self, offset=0.0, fail_below_camber=None):
        self.offset = offset
        self.fail_below_camber = fail_below_camber
        self.calls = 0

    def polar(self, section, workdir=None):
        del workdir
        self.calls += 1
        camber = section.parameters["m_camber"]
        if self.fail_below_camber is not None and camber < self.fail_below_camber:
            raise SolverFailure("fake solver failure")
        alpha = np.arange(-4.0, 10.25, 0.5)
        cl = 0.11 * (alpha + 80.0 * camber)
        cd = 0.006 + self.offset + 0.05 * (section.thickness - 0.12) ** 2 + 0.01 * cl ** 2
        return Polar(alpha=alpha, cl=cl, cd=cd)


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_shipped_configurations_are_valid(path):
    problem = load_config(path)
    assert problem.dim >= 1 and len(problem.optimization["doe"]) == len(problem.levels)


@pytest.mark.parametrize("override, message", [
    ({"kind": "foil"}, "kind"),
    ({"levels": [{"solver": "cfd", "cost": 1.0}]}, "solver"),
    ({"objective": {"type": "cd_at_cl"}}, "cl_target"),
    ({"variables": [{"name": "m_camber", "lower": 1.0, "upper": 0.0}]}, "upper > lower"),
    ({"optimization": {"doe": [5]}}, "one size per level"),
])
def test_invalid_configurations_are_rejected(override, message):
    with pytest.raises(ValueError, match=message):
        from_dict(base_config(**override))


def test_mapping_to_physical_bounds():
    problem = from_dict(base_config())
    params = geometry.to_physical(np.array([0.0, 1.0]), problem.variables)
    assert params == {"m_camber": 0.0, "t_thickness": 0.16}
    with pytest.raises(ValueError):
        geometry.to_physical(np.array([0.5]), problem.variables)


def test_naca4_section_and_xf_file(tmp_path):
    problem = from_dict(base_config())
    section = geometry.build_section(problem, np.array([0.5, 0.5]))
    coords = section.coordinates
    # trailing edge at x ~ 1 (the thickness is applied normal to the camber line)
    assert coords[0, 0] == pytest.approx(1.0, abs=1e-3)
    assert coords[-1, 0] == pytest.approx(1.0, abs=1e-3)
    assert coords[len(coords) // 2, 0] == pytest.approx(0.0)  # Selig order: TE -> LE -> TE
    assert section.thickness == pytest.approx(0.12, abs=2e-3)
    path = geometry.write_xf(section, tmp_path)
    lines = path.read_text(encoding="ascii").splitlines()
    assert lines[0] == section.name and len(lines) == len(coords) + 1
    assert path.is_absolute()


def test_objectives_on_an_analytical_polar():
    alpha = np.linspace(-4, 14, 37)
    cl = np.where(alpha < 12, 0.1 * alpha, 1.2 - 0.05 * (alpha - 12))  # stall after 12 deg
    polar = Polar(alpha=alpha, cl=cl, cd=0.01 + 0.02 * cl ** 2)
    value, metrics = objectives.evaluate_objective(polar, {"type": "cd_at_cl", "cl_target": 0.5})
    assert value == pytest.approx(0.01 + 0.02 * 0.25, rel=1e-3)
    assert metrics["alpha"] == pytest.approx(5.0, abs=1e-6)
    value, _ = objectives.evaluate_objective(polar, {"type": "cd_at_cl", "cl_target": 1.5})
    assert np.isnan(value)  # outside the monotone branch: failed evaluation, no clamping
    value, metrics = objectives.evaluate_objective(polar, {"type": "max_lift_to_drag"})
    assert value == pytest.approx(-metrics["max_lift_to_drag"])


def test_simulator_with_fake_backends():
    problem = from_dict(base_config())
    simulator = BdToolboxFoilSimulator(problem, paths=bridge.resolve_paths(),
                                       backends=[FakeBackend(), FakeBackend(offset=0.001)])
    value_1, metrics_1 = simulator.evaluate(np.array([0.5, 0.5]), 1)
    value_2, _ = simulator.evaluate(np.array([0.5, 0.5]), 2)
    assert value_2 == pytest.approx(value_1 + 0.001)
    assert metrics_1["cl_target"] == 0.5 and metrics_1["solver"] == "neuralfoil"
    json.dumps(metrics_1)  # JSON friendly


def test_failures_and_constraints_give_nan():
    problem = from_dict(base_config(constraints={"min_thickness": 0.12}))
    simulator = BdToolboxFoilSimulator(problem, paths=bridge.resolve_paths(),
                                       backends=[FakeBackend(fail_below_camber=0.03)] * 2)
    value, metrics = simulator.evaluate(np.array([0.1, 0.9]), 1)
    assert np.isnan(value) and "fake solver failure" in metrics["error"]
    value, metrics = simulator.evaluate(np.array([0.9, 0.1]), 1)
    assert np.isnan(value) and "thickness" in metrics["error"]


def test_end_to_end_run_with_fake_backends(tmp_path):
    """DOE + NN-MF-EGO + export + results.json + figures, in a timestamped run directory."""
    problem = from_dict(base_config())
    run = create_run(tmp_path, prefix=problem.name)
    simulator = BdToolboxFoilSimulator(problem, paths=bridge.resolve_paths(),
                                       backends=[FakeBackend(), FakeBackend(offset=0.001)])
    results = run_optimization(problem, run, simulator)
    for name in ("ego_backup.json", "surrogate.json", "results.json", "convergence_plot.html",
                 "response_surface_2d.html"):
        assert (run.run_dir / name).exists()
    assert set(results["best_design_physical"]) == {"m_camber", "t_thickness"}
    assert np.isfinite(results["best_objective"])


def test_3d_template_with_a_fake_backend():
    config = base_config(kind="foil3d", parametrization={"type": "planform"},
                         levels=[{"solver": "npllt", "cost": 1.0}],
                         objective={"type": "coefficient", "name": "Cx", "sign": -1.0},
                         optimization={"doe": [4]})

    class FakeBackend3D:
        def coefficients(self, parameters, workdir):
            del workdir
            return {"Cx": -0.01 - parameters["t_thickness"] / 10, "Cz": 0.3}

    problem = from_dict(config)
    simulator = BdToolboxFoilSimulator(problem, paths=bridge.resolve_paths(),
                                       backends=[FakeBackend3D()])
    value, metrics = simulator.evaluate(np.array([0.5, 0.5]), 1)
    assert value == pytest.approx(0.01 + 0.012) and metrics["Cz"] == 0.3


@pytest.mark.slow
def test_real_solvers_on_a_naca_section():
    """NeuralFoil (both models) and bdFoil core XFOIL on a NACA 3412-like section."""
    pytest.importorskip("neuralfoil")
    paths = bridge.resolve_paths()
    if not (paths.soft_dir / "xfoil.exe").is_file() or not paths.bdfoil_root.is_dir():
        pytest.skip("bdToolbox soft/xfoil.exe or bdFoil clone not found")
    problem = load_config(ROOT / "pipelines" / "configs" / "section2d_naca_3levels.json")
    simulator = BdToolboxFoilSimulator(problem)
    values = [simulator.evaluate(np.array([0.5, 0.5, 0.5]), level)[0] for level in (1, 2, 3)]
    assert all(0.005 < v < 0.03 for v in values)
    assert abs(values[1] - values[2]) / values[2] < 0.1  # NeuralFoil xxxlarge close to XFOIL
