"""Unit tests of src/surrogate_models.MultifidelityModel (recursive non-nested formulation)."""
import numpy as np
import pytest
from conftest import fill_data, forrester_hf, forrester_lf
from src.data_management import ExperimentData
from src.kernels import SquaredExponentialKernel
from src.surrogate_models import MultifidelityModel

X_TEST = np.linspace(0, 1, 101).reshape(-1, 1)


def fitted_model(data, **kwargs):
    model = MultifidelityModel(data.y_dict.__len__(), SquaredExponentialKernel, seed=0, **kwargs)
    model.fit(data)
    return model


def rmse(model):
    return float(np.sqrt(np.mean((model.predict_batch(X_TEST)[0]
                                  - forrester_hf(X_TEST[:, 0])) ** 2)))


def test_rho_is_estimated_on_forrester(forrester_data):
    """[FIX-T1] the profiled rho recovers the true value rho = 2 of Eq. 17."""
    model = fitted_model(forrester_data, estimate_rho=True)
    assert model.rhos[0] == pytest.approx(2.0, abs=0.1)


def test_fixed_rho_option_reproduces_previous_behaviour(forrester_data):
    """estimate_rho=False keeps rho = rho_init (the previous code always used rho = 1)."""
    assert fitted_model(forrester_data, estimate_rho=False).rhos == [1.0]
    assert fitted_model(forrester_data, estimate_rho=False, rho_init=1.5).rhos == [1.5]


def test_hybrid_mode_keeps_rho_fixed_with_few_points(forrester_data):
    """[FIX-T1b] below min_points_rho high-fidelity points, rho stays at rho_init."""
    model = fitted_model(forrester_data, estimate_rho=True, min_points_rho=7)  # 6 HF points
    assert model.rhos == [1.0]
    model = fitted_model(forrester_data, estimate_rho=True, min_points_rho=6)
    assert model.rhos[0] != 1.0


def test_estimated_rho_improves_the_surrogate(forrester_data):
    """[FIX-T1] the additive model (rho = 1) is much less accurate on Forrester."""
    assert rmse(fitted_model(forrester_data, estimate_rho=True)) \
           < 0.5 * rmse(fitted_model(forrester_data, estimate_rho=False))


def test_recursive_prediction_formula(forrester_data):
    """Eqs. 11-12: f_2 = rho f_1 + delta_2, s2_2 = rho^2 s2_1 + s2_delta2."""
    model = fitted_model(forrester_data)
    mean_1, var_1 = model.gps[0].predict_batch(X_TEST)
    mean_d, var_d = model.gps[1].predict_batch(X_TEST)
    mean, var, gp_vars = model.predict_batch(X_TEST)
    rho = model.rhos[0]
    np.testing.assert_allclose(mean, rho * mean_1 + mean_d)
    np.testing.assert_allclose(var, rho ** 2 * var_1 + var_d)
    np.testing.assert_allclose(gp_vars, np.column_stack([var_1, var_d]))
    np.testing.assert_allclose(model.predict_batch(X_TEST, level=1)[0], mean_1)


def test_residual_target_is_eq_18(forrester_data):
    """Eq. 18: the level-2 GP is trained on y2 - rho * f1_hat(X2) (non-nested residual)."""
    model = fitted_model(forrester_data)
    x_2, y_2 = forrester_data.get_training_data(2)
    f_prev = model.predict_batch(x_2, level=1)[0]
    np.testing.assert_allclose(model.gps[1].y_train, y_2 - model.rhos[0] * f_prev)


def test_single_point_api_is_kept(forrester_data):
    model = fitted_model(forrester_data)
    f_hat, sigma2, gp_variances = model.predict(np.array([0.3]))
    mean, var, gp_vars = model.predict_batch(np.array([[0.3]]))
    assert (f_hat, sigma2) == pytest.approx((mean[0], var[0]))
    assert gp_variances == pytest.approx(gp_vars[0].tolist())
    assert model._predict_up_to(np.array([0.3]), 1)[0] == \
           pytest.approx(model.gps[0].predict(np.array([0.3]))[0])


def test_three_levels():
    data = ExperimentData(bounds=[(0.0, 1.0)], costs=[1.0, 5.0, 25.0])
    data.generate_initial_design(points_per_level=[12, 8, 6])
    fill_data(data, [forrester_lf, lambda x: 0.5 * (forrester_lf(x) + forrester_hf(x)),
                     forrester_hf])
    model = fitted_model(data, estimate_rho=True, min_points_rho=5)
    assert len(model.rhos) == 2
    assert np.all(np.isfinite(model.predict_batch(X_TEST)[0]))
    assert rmse(model) < 1.0


def test_failed_points_are_excluded_from_training(forrester_data):
    """[FIX-R4] NaN outputs (failed simulations) do not reach the GP."""
    forrester_data.add_observation(2, np.array([0.5]), np.nan, {})
    model = fitted_model(forrester_data)
    assert len(model.gps[1].y_train) == 6
    assert np.all(np.isfinite(model.predict_batch(X_TEST)[0]))


def test_to_dict_from_dict_round_trip(forrester_data):
    """[FIX-X1] the exported state rebuilds exactly the same model (no re-training)."""
    model = fitted_model(forrester_data)
    clone = MultifidelityModel.from_dict(model.to_dict())
    for a, b in zip(model.predict_batch(X_TEST), clone.predict_batch(X_TEST)):
        np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-12)
    assert clone.rhos == model.rhos
