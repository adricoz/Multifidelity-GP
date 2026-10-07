"""
The framework with a SINGLE fidelity level (L = 1): model, merit, EGO loop, JSON, export and
plots must be consistent (user requirement; also used by the hydrofoil example with L = 1).
"""
import json

import numpy as np
import pytest
from conftest import FORRESTER_MIN, make_optimizer
from scipy.stats import norm
from src.acquisition import AcquisitionFunction
from src.data_management import ExperimentData
from src.kernels import SquaredExponentialKernel
from src.surrogate_models import GaussianProcess, MultifidelityModel, load_surrogate
from src.visualization import ModelVisualizer

X_TEST = np.linspace(0, 1, 41).reshape(-1, 1)


def test_single_level_model_is_a_plain_gp(single_fidelity_data):
    model = MultifidelityModel(1, SquaredExponentialKernel, seed=5)
    model.fit(single_fidelity_data)
    assert model.rhos == []
    gp = GaussianProcess(SquaredExponentialKernel(), seed=5)
    gp.fit(*single_fidelity_data.get_training_data(1))
    mean, var, gp_vars = model.predict_batch(X_TEST)
    np.testing.assert_allclose(mean, gp.predict_batch(X_TEST)[0])
    np.testing.assert_allclose(var, gp.predict_batch(X_TEST)[1])
    np.testing.assert_allclose(gp_vars[:, 0], var)


def test_single_level_merit_is_the_aei(single_fidelity_data):
    """With L = 1: cost ratio 1, R2 = 1, merit = AEI * s^4 / ((s^2 + noise) * var)."""
    model = MultifidelityModel(1, SquaredExponentialKernel, seed=5)
    model.fit(single_fidelity_data)
    acq = AcquisitionFunction(model=model, data=single_fidelity_data)
    acq.update()
    mean, var, _ = model.predict_batch(X_TEST)
    noise = model.gps[0].get_noise_variance()
    sigma = np.sqrt(var)
    u = (acq.f_best - mean) / sigma
    aei = sigma * (u * norm.cdf(u) + norm.pdf(u)) * (1 - np.sqrt(noise) / np.sqrt(var + noise))
    s2 = np.maximum(var - noise, 0)
    expected = aei * (s2 ** 2 / (s2 + noise)) / var
    np.testing.assert_allclose(acq.evaluate_merits_batch(X_TEST)[:, 0], expected,
                               rtol=1e-10, atol=1e-14)


def test_single_fidelity_end_to_end(single_fidelity_data, in_tmp):
    """run -> JSON -> export -> load_surrogate -> ModelVisualizer (PNG + HTML)."""
    ego = make_optimizer(single_fidelity_data, 1)
    cost_history, best = ego.run(n_iterations=6)
    assert cost_history[0] == 8.0 and len(cost_history) == 7
    assert best[-1] == pytest.approx(FORRESTER_MIN, abs=5e-2)

    ego.export_surrogate("surrogate.json")
    loaded = load_surrogate("surrogate.json")
    viz = ModelVisualizer("ego_backup.json")
    assert viz.num_levels == 1
    for model in (loaded, viz.model):
        for a, b in zip(ego.model.predict_batch(X_TEST), model.predict_batch(X_TEST)):
            np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-12)

    viz.plot_convergence(save_path="conv.png", target=FORRESTER_MIN)
    viz.plot_response_1d(save_path="resp.png")
    viz.plot_convergence_interactive(save_path="conv.html", target=FORRESTER_MIN)
    viz.plot_response_1d_interactive(save_path="resp.html")
    for name in ("conv.png", "resp.png", "conv.html", "resp.html"):
        assert (in_tmp / name).stat().st_size > 0
    html = (in_tmp / "resp.html").read_text(encoding="utf-8")
    assert "Single fidelity" in html and "HF (level" not in html
    with open("surrogate.json", encoding="utf-8") as f:
        assert json.load(f)["summary"]["rhos"] == []


def test_single_fidelity_2d_plots(in_tmp):
    """2D single-level problem: static and interactive response surfaces."""
    data = ExperimentData(bounds=[(0.0, 1.0), (-1.0, 1.0)], costs=[1.0])
    data.generate_initial_design(points_per_level=[12])
    data.y_dict[1] = np.sin(3 * data.x_dict[1][:, 0]) + data.x_dict[1][:, 1] ** 2
    data.metrics_dict[1] = [{} for _ in data.y_dict[1]]
    model = MultifidelityModel(1, SquaredExponentialKernel, seed=0)
    model.fit(data)
    acq = AcquisitionFunction(model=model, data=data)
    from src.optimizer import EGOOptimizer  # pylint: disable=import-outside-toplevel
    EGOOptimizer(data, model, None, acq).save_state("state.json")
    viz = ModelVisualizer("state.json")
    viz.plot_response_surface_2d(save_path="surface.png")
    viz.plot_response_surface_2d_interactive(save_path="surface.html")
    assert (in_tmp / "surface.png").exists() and (in_tmp / "surface.html").exists()

