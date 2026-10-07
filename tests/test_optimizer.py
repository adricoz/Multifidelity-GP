"""Unit tests of src/optimizer.py (EGO loop, ask/tell, JSON state)."""
import json

import numpy as np
import pytest
from conftest import FORRESTER_MIN, make_optimizer
from src.data_management import ExperimentData
from src.visualization import ModelVisualizer

X_TEST = np.linspace(0, 1, 51).reshape(-1, 1)


def test_doe_cost_is_counted(forrester_data, in_tmp):
    """[FIX-R5] optimizer created before the DOE is evaluated (as in main.py): the DOE cost
    10 * 1 + 6 * 10 = 70 is counted (it was 0 before the fix)."""
    data = ExperimentData(bounds=[(0.0, 1.0)], costs=[1.0, 10.0])
    ego = make_optimizer(data, 2)
    data.x_dict, data.y_dict = forrester_data.x_dict, forrester_data.y_dict
    data.metrics_dict = forrester_data.metrics_dict
    cost_history, _ = ego.run(n_iterations=1)
    assert cost_history[0] == 70.0
    assert cost_history[1] in (71.0, 80.0)


def test_ask_tell_mode_counts_the_doe(forrester_data, in_tmp):
    ego = make_optimizer(forrester_data, 2)
    x_next, l_next, merit = ego.ask()
    assert x_next.shape == (1,) and l_next in (1, 2) and merit >= 0.0
    y, metrics = ego.simulator.evaluate(x_next, l_next)
    ego.tell(x_next, l_next, y, metrics)
    assert ego.cost_history == [70.0, 70.0 + forrester_data.costs[l_next - 1]]


def test_state_is_consistent_after_run(forrester_data, in_tmp):
    """[FIX-X2] after run(), the JSON contains a snapshot of a model trained on ALL the data
    and the ModelVisualizer rebuilds exactly the in-memory model."""
    ego = make_optimizer(forrester_data, 2)
    ego.run(n_iterations=3)
    with open("ego_backup.json", encoding="utf-8") as f:
        state = json.load(f)
    assert state["surrogate"]["fit_sizes"] == [len(state["Y_dict"]["1"]),
                                                len(state["Y_dict"]["2"])]
    assert state["bounds"] == [[0.0, 1.0]] and state["num_levels"] == 2
    model = ModelVisualizer("ego_backup.json").model
    for a, b in zip(ego.model.predict_batch(X_TEST), model.predict_batch(X_TEST)):
        np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-12)


def test_save_and_load_state(forrester_data, in_tmp):
    ego = make_optimizer(forrester_data, 2)
    ego.run(n_iterations=2)
    other = make_optimizer(ExperimentData(bounds=[(0.0, 1.0)], costs=[1.0, 10.0]), 2)
    other.load_state("ego_backup.json")
    for l in (1, 2):
        np.testing.assert_allclose(other.data.x_dict[l], ego.data.x_dict[l])
        np.testing.assert_allclose(other.data.y_dict[l], ego.data.y_dict[l])
    assert other.cost_history == ego.cost_history
    assert other.current_total_cost == ego.current_total_cost
    np.testing.assert_allclose(other.model.gps[1].kernel.get_params(),
                               ego.model.gps[1].kernel.get_params())


def test_seeded_runs_are_reproducible(forrester_data, in_tmp):
    """[FIX-R1] two runs with the same seed give the same designs and histories."""
    histories = []
    for _ in range(2):
        data = ExperimentData(bounds=[(0.0, 1.0)], costs=[1.0, 10.0])
        data.x_dict = {k: v.copy() for k, v in forrester_data.x_dict.items()}
        data.y_dict = {k: v.copy() for k, v in forrester_data.y_dict.items()}
        data.metrics_dict = {k: list(v) for k, v in forrester_data.metrics_dict.items()}
        ego = make_optimizer(data, 2, seed=7)
        histories.append((ego.run(n_iterations=3), np.vstack(list(data.x_dict.values()))))
    assert histories[0][0] == histories[1][0]
    np.testing.assert_array_equal(histories[0][1], histories[1][1])


def test_run_converges_on_forrester(forrester_data, in_tmp):
    """The NN-MF-EGO finds the Forrester minimum (-6.0207) with a few iterations."""
    ego = make_optimizer(forrester_data, 2)
    _, best_y_history = ego.run(n_iterations=8)
    assert best_y_history[-1] <= best_y_history[0]
    assert best_y_history[-1] == pytest.approx(FORRESTER_MIN, abs=5e-3)


def test_failed_simulations_do_not_break_the_run(forrester_data, in_tmp, monkeypatch):
    """[FIX-R4] a simulator returning NaN: the point is stored as failed, the cost is
    counted and the GP is trained on the valid data only."""
    ego = make_optimizer(forrester_data, 2)
    original = ego.simulator.evaluate
    monkeypatch.setattr(ego.simulator, "evaluate",
                        lambda x, level: (np.nan, {}) if level == 2 else original(x, level))
    cost_history, best = ego.run(n_iterations=3)
    assert np.isfinite(best[-1])
    assert all(np.isfinite(ego.model.gps[1].y_train))
    new_evals = {1: len(ego.data.y_dict[1]) - 10, 2: len(ego.data.y_dict[2]) - 6}
    assert new_evals[1] + new_evals[2] == 3
    assert ego.data.n_failed(2) == new_evals[2]
    assert cost_history[-1] == 70.0 + new_evals[1] * 1.0 + new_evals[2] * 10.0


def test_stop_on_convergence(forrester_data, in_tmp):
    """[FIX-T3b] with stop_on_convergence, no random evaluation once the optimum is known."""
    ego = make_optimizer(forrester_data, 2)
    cost_history, _ = ego.run(n_iterations=30, stop_on_convergence=True)
    assert len(cost_history) < 31
