"""Unit tests of src/data_management.py."""
import ast
from pathlib import Path

import numpy as np
from scipy.stats import qmc
from src.data_management import ExperimentData


def test_module_does_not_depend_on_traitlets():
    """[FIX-R6] the module imports without traitlets (TypeError / ModuleNotFoundError before)."""
    source = (Path(__file__).resolve().parents[1] / "mfego" / "src" /
              "data_management.py").read_text(encoding="utf-8")
    imported = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom):
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
    assert "traitlets" not in imported


def test_initial_design_in_bounds_and_reproducible():
    bounds = [(-2.0, 3.0), (10.0, 11.0)]
    data = ExperimentData(bounds=bounds, costs=[1.0, 2.0])
    data.generate_initial_design(points_per_level=[7, 3])
    assert data.x_dict[1].shape == (7, 2) and data.x_dict[2].shape == (3, 2)
    for x in data.x_dict.values():
        assert np.all(x >= [-2.0, 10.0]) and np.all(x <= [3.0, 11.0])
    # default seed = previous behaviour (42 + level)
    expected = qmc.scale(qmc.LatinHypercube(d=2, seed=43).random(7), [-2, 10], [3, 11])
    np.testing.assert_allclose(data.x_dict[1], expected)


def test_add_observation_and_failed_points():
    """[FIX-R4] NaN/inf outputs are kept as NaN, excluded from the training data."""
    data = ExperimentData(bounds=[(0.0, 1.0)], costs=[1.0])
    data.add_observation(1, np.array([0.1]), 1.0, None)
    data.add_observation(1, np.array([0.2]), np.inf, {"error": True})
    data.add_observation(1, np.array([0.3]), -2.0, {})
    data.add_observation(1, np.array([0.4]), np.nan, {})
    assert data.x_dict[1].shape == (4, 1)
    x_train, y_train = data.get_training_data(1)
    np.testing.assert_allclose(x_train[:, 0], [0.1, 0.3])
    np.testing.assert_allclose(y_train, [1.0, -2.0])
    assert data.n_failed(1) == 2
    x_best, y_best = data.best_observation(1)
    assert y_best == -2.0 and x_best[0] == 0.3
    # a failed point is "already evaluated" (it is not proposed again)
    assert data.is_already_evaluated(1, np.array([0.2]))
    assert not data.is_already_evaluated(1, np.array([0.25]))


def test_best_observation_without_valid_point():
    data = ExperimentData(bounds=[(0.0, 1.0)], costs=[1.0])
    data.add_observation(1, np.array([0.1]), np.nan, {})
    x_best, y_best = data.best_observation(1)
    assert x_best is None and np.isnan(y_best)
