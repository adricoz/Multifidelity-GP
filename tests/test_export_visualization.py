"""Unit tests of the surrogate export (X1) and of src/visualization.py (X3)."""
import json

import numpy as np
import pytest
from conftest import make_optimizer
from src.data_management import ExperimentData
from src.kernels import SquaredExponentialKernel
from src.optimizer import EGOOptimizer
from src.surrogate_models import MultifidelityModel, load_surrogate
from src.visualization import ModelVisualizer

X_TEST = np.linspace(0, 1, 61).reshape(-1, 1)


@pytest.fixture
def trained(forrester_data, in_tmp):
    ego = make_optimizer(forrester_data, 2)
    ego.run(n_iterations=2)
    return ego


def test_export_and_load_surrogate(trained):
    """[FIX-X1] export_surrogate / load_surrogate give the same model as in memory."""
    trained.export_surrogate("surrogate.json")
    loaded = load_surrogate("surrogate.json")
    for a, b in zip(trained.model.predict_batch(X_TEST), loaded.predict_batch(X_TEST)):
        np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-12)
    with open("surrogate.json", encoding="utf-8") as f:
        export = json.load(f)
    assert set(export) == {"surrogate", "num_levels", "bounds", "costs", "summary"}
    assert export["summary"]["total_cost"] == trained.current_total_cost


def test_load_surrogate_from_the_backup_file(trained):
    loaded = load_surrogate("ego_backup.json")
    np.testing.assert_allclose(loaded.predict_batch(X_TEST)[0],
                               trained.model.predict_batch(X_TEST)[0], rtol=1e-9)


def test_visualizer_reads_old_json_files(trained):
    """Backward compatibility: a JSON without the "surrogate" entry is still plotted."""
    with open("ego_backup.json", encoding="utf-8") as f:
        state = json.load(f)
    for key in ("surrogate", "num_levels", "bounds", "costs"):
        state.pop(key)
    with open("old.json", "w", encoding="utf-8") as f:
        json.dump(state, f)
    viz = ModelVisualizer("old.json", num_levels=2)
    assert np.all(np.isfinite(viz.model.predict_batch(X_TEST)[0]))
    viz.plot_response_1d(save_path="old.png")


def test_grid_follows_the_bounds(in_tmp):
    """[FIX-X3] the plotting grid uses the problem bounds (was hard-coded to [0, 1])."""
    data = ExperimentData(bounds=[(2.0, 5.0)], costs=[1.0])
    data.generate_initial_design(points_per_level=[6])
    data.y_dict[1] = np.sin(data.x_dict[1][:, 0])
    data.metrics_dict[1] = [{} for _ in range(6)]
    model = MultifidelityModel(1, SquaredExponentialKernel, seed=0)
    model.fit(data)
    EGOOptimizer(data, model, None, None).save_state("state.json")
    viz = ModelVisualizer("state.json")
    grid = viz._grid_1d(11)
    assert grid.min() == 2.0 and grid.max() == 5.0


def test_scatter_uses_the_columns_of_the_points(in_tmp, monkeypatch):
    """[FIX-X3] the plotted evaluations are points[:, idx] (points[idx] selected rows)."""
    data = ExperimentData(bounds=[(0.0, 1.0)] * 3, costs=[1.0])
    data.generate_initial_design(points_per_level=[9])
    data.y_dict[1] = np.sum(data.x_dict[1] ** 2, axis=1)
    data.metrics_dict[1] = [{} for _ in range(9)]
    model = MultifidelityModel(1, SquaredExponentialKernel, seed=0)
    model.fit(data)
    EGOOptimizer(data, model, None, None).save_state("state.json")
    viz = ModelVisualizer("state.json")
    captured = []
    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel
    monkeypatch.setattr(plt, "scatter", lambda x, y, **kw: captured.append((x, y)))
    viz.plot_response_surface_2d(param_x_idx=0, param_y_idx=2, save_path="s.png")
    np.testing.assert_allclose(captured[0][0], data.x_dict[1][:, 0])
    np.testing.assert_allclose(captured[0][1], data.x_dict[1][:, 2])
    # default fixed value of the third parameter = best observed point
    i_best = np.argmin(data.y_dict[1])
    np.testing.assert_allclose(viz._best_observed_point(), data.x_dict[1][i_best])


def test_interactive_plots_are_written(trained):
    viz = ModelVisualizer("ego_backup.json")
    viz.plot_convergence_interactive(save_path="c.html", target=-6.02074)
    viz.plot_response_1d_interactive(save_path="r.html")
    html = open("r.html", encoding="utf-8").read()
    assert "plotly" in html and "Level 1" in html and "HF (level 2)" in html


def test_failed_points_are_not_plotted_as_observations(trained):
    trained.data.add_observation(2, np.array([0.123]), np.nan, {})
    trained.save_state("with_failure.json")
    viz = ModelVisualizer("with_failure.json")
    x_obs, y_obs, x_failed = viz._observations(2)
    assert np.all(np.isfinite(y_obs)) and x_failed.shape == (1, 1)
