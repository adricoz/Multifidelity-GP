"""Unit tests of src/cokriging.CoKriging (co-kriging conditioned on all the data, nested designs)
and of ExperimentData.generate_nested_design."""
import json

import numpy as np
import pytest
from conftest import fill_data, forrester_hf, forrester_lf
from src.cokriging import CoKriging, missing_points_for_nesting, nested_index
from src.data_management import ExperimentData
from src.kernels import Matern52Kernel, SquaredExponentialKernel
from src.surrogate_models import (DEFAULT_PRIORS, V3_RHO_BOUNDS, V3_RHO_PRIOR, GaussianProcess,
                                  MultifidelityModel)
from src.validation import prediction_scores

X_TEST = np.linspace(0, 1, 101).reshape(-1, 1)


@pytest.fixture
def nested_data():
    """2-level nested Forrester data (11 LF points, 5 HF points among them)."""
    data = ExperimentData(bounds=[(0.0, 1.0)], costs=[1.0, 10.0])
    data.generate_nested_design([11, 5], seed=0)
    return fill_data(data, [forrester_lf, forrester_hf])


def fitted(data, num_levels=2, **kwargs):
    """CoKriging fitted on Forrester data: squared exponential kernel (analytic function, true
    rho = 2 recovered) unless another kernel_class is given."""
    model = CoKriging(num_levels, kernel_class=kwargs.pop("kernel_class",
                                                          SquaredExponentialKernel),
                      seed=0, **kwargs)
    model.fit(data)
    return model


def test_generate_nested_design():
    """[CK] X_3 in X_2 in X_1, with the requested sizes; increasing sizes are refused."""
    data = ExperimentData(bounds=[(0.0, 2.0), (-1.0, 1.0)], costs=[1.0, 5.0, 25.0])
    data.generate_nested_design([20, 8, 3], seed=1)
    assert [len(data.x_dict[l]) for l in (1, 2, 3)] == [20, 8, 3]
    for l in (2, 3):
        assert np.all(nested_index(data.x_dict[l], data.x_dict[l - 1], tol=0.0) >= 0)
    assert data.x_dict[1][:, 0].min() >= 0.0 and data.x_dict[1][:, 0].max() <= 2.0
    with pytest.raises(ValueError, match="must not increase"):
        data.generate_nested_design([5, 8])


def test_joint_covariance_is_eq_3_11(nested_data):
    """[CK] V is the block matrix of [LG] Eq. 3.11:
    [[c1(X1, X1), rho c1(X1, X2)], [rho c1(X2, X1), rho^2 c1(X2, X2) + c2(X2, X2)]]."""
    model = fitted(nested_data)
    (x1, _), (x2, _) = nested_data.get_training_data(1), nested_data.get_training_data(2)
    gp1, gp2 = model.gps
    rho = model.rhos[0]

    def c1(a, b):
        return gp1.y_std ** 2 * gp1.kernel.get_cross_covariance_matrix(a, b)

    def c2(a, b):
        return gp2.y_std ** 2 * gp2.kernel.get_cross_covariance_matrix(a, b)

    expected = np.block([[c1(x1, x1), rho * c1(x1, x2)],
                         [rho * c1(x2, x1), rho ** 2 * c1(x2, x2) + c2(x2, x2)]])
    computed = model._covariance(model.x_all, model.levels_all, model.x_all, model.levels_all)
    np.testing.assert_allclose(computed, expected, rtol=1e-12, atol=1e-12 * np.abs(expected).max())


def test_equivalence_with_the_recursive_formula(nested_data):
    """[CK] [LG] Prop. 3.1: nested and noise-free data, same hyperparameters -> the co-kriging
    equals the recursive formula with the OBSERVED residual z2 - rho z1(X2):
    mean = rho m1 + m_delta, var = rho^2 v1 + v_delta."""
    model = fitted(nested_data)
    (x1, y1), (x2, y2) = nested_data.get_training_data(1), nested_data.get_training_data(2)
    for gp in model.gps:
        gp.noise = 1e-12
    rho = model.rhos[0]
    model.condition([x1, x2], [y1, y2])
    gp1, gp2 = model.gps
    gp1.condition(x1, y1)
    gp2.condition(x2, y2 - rho * y1[nested_index(x2, x1)])

    mean, var = model.predict_batch(X_TEST, include_noise=False)
    mean_1, var_1 = gp1.predict_batch(X_TEST)
    mean_d, var_d = gp2.predict_batch(X_TEST)
    np.testing.assert_allclose(mean, rho * mean_1 + mean_d, rtol=1e-6,
                               atol=1e-6 * np.std(y2))
    np.testing.assert_allclose(var, rho ** 2 * var_1 + var_d, rtol=1e-4,
                               atol=1e-6 * np.var(y2))


def test_single_level_is_the_gaussian_process(nested_data):
    """[CK] with one level, the co-kriging is the GaussianProcess of mfego (same fit)."""
    model = fitted(nested_data, num_levels=1)
    gp = GaussianProcess(SquaredExponentialKernel(), seed=0, use_map=True)
    gp.fit(*nested_data.get_training_data(1))
    for a, b in zip(model.predict_batch(X_TEST), gp.predict_batch(X_TEST)):
        np.testing.assert_allclose(a, b, rtol=1e-8, atol=1e-10)


def test_interpolation_and_positive_variance(nested_data):
    """[CK] the mean interpolates the HF points, the variance is >= 0 and small there."""
    model = fitted(nested_data)
    x2, y2 = nested_data.get_training_data(2)
    mean, var = model.predict_batch(x2, include_noise=False)
    np.testing.assert_allclose(mean, y2, atol=1e-3 * np.std(y2))
    assert np.all(var < 1e-3 * np.var(y2))
    assert np.all(model.predict_batch(X_TEST)[1] >= 0.0)
    assert model.rhos[0] == pytest.approx(2.0, abs=0.1)  # true rho of Eq. 17
    # every level has its surrogate, conditioned on all the data
    mean_1 = model.predict_batch(nested_data.get_training_data(1)[0], level=1)[0]
    np.testing.assert_allclose(mean_1, nested_data.get_training_data(1)[1], atol=1e-2)


def test_non_nested_design_is_refused_then_completed(forrester_data):
    """[CK] independent designs per level -> explicit error; evaluating the missing points
    given by missing_points_for_nesting makes the design nested."""
    with pytest.raises(ValueError, match="nested"):
        fitted(forrester_data)
    missing = missing_points_for_nesting(forrester_data, 2)
    assert len(missing[1]) == len(forrester_data.x_dict[2])
    for x in missing[1]:
        forrester_data.add_observation(1, x, float(forrester_lf(x[0])), {})
    assert len(missing_points_for_nesting(forrester_data, 2)[1]) == 0
    model = fitted(forrester_data)
    assert np.all(np.isfinite(model.predict_batch(X_TEST)[0]))


def test_three_nested_levels():
    data = ExperimentData(bounds=[(0.0, 1.0)], costs=[1.0, 5.0, 25.0])
    data.generate_nested_design([14, 8, 4], seed=0)
    fill_data(data, [forrester_lf, lambda x: 0.5 * (forrester_lf(x) + forrester_hf(x)),
                     forrester_hf])
    model = fitted(data, num_levels=3)
    assert len(model.rhos) == 2 and model.x_all.shape == (26, 1)
    rmse = np.sqrt(np.mean((model.predict_batch(X_TEST)[0] - forrester_hf(X_TEST[:, 0])) ** 2))
    assert rmse < 1.0


def test_to_dict_from_dict_round_trip(nested_data):
    """[CK] the JSON state rebuilds the same surrogate without fitting."""
    model = fitted(nested_data, priors={"noise": None}, lengthscale_prior=(3.0, 1.0))
    clone = CoKriging.from_dict(json.loads(json.dumps(model.to_dict())))
    for a, b in zip(model.predict_batch(X_TEST), clone.predict_batch(X_TEST)):
        np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-12)
    assert clone.rhos == model.rhos
    assert all(a.priors == b.priors for a, b in zip(model.gps, clone.gps))
    assert clone.gps[0].priors["noise"] is None
    assert clone.gps[0].priors["lengthscale"] == ("invgamma", 3.0, 1.0)


def test_three_level_covariance_products():
    """[CK] Cov(Z_i, Z_j) with the products of the rhos, written by hand for 3 levels:
    Cov(Z1, Z3) = r1 r2 c1, Cov(Z2, Z3) = r1^2 r2 c1 + r2 c2, Var(Z3) = r1^2 r2^2 c1 + r2^2 c2 + c3.
    """
    model = CoKriging(3)
    model.rhos = [1.5, -0.7]
    for i, gp in enumerate(model.gps):
        gp.kernel.set_params(np.array([0.2 + 0.1 * i, 1.0 + i, 0.01]))
        gp.y_std = 1.0 + 0.5 * i
    x = np.linspace(0, 1, 5).reshape(-1, 1)
    c1, c2, c3 = (gp.y_std ** 2 * gp.kernel.get_cross_covariance_matrix(x, x)
                  for gp in model.gps)
    r1, r2 = model.rhos

    def cov(i, j):
        return model._covariance(x, np.full(5, i), x, np.full(5, j))

    np.testing.assert_allclose(cov(1, 3), r1 * r2 * c1, rtol=1e-12)
    np.testing.assert_allclose(cov(2, 3), r1 ** 2 * r2 * c1 + r2 * c2, rtol=1e-12)
    np.testing.assert_allclose(cov(3, 3), r1 ** 2 * r2 ** 2 * c1 + r2 ** 2 * c2 + c3, rtol=1e-12)
    np.testing.assert_allclose(cov(3, 1), cov(1, 3).T, rtol=1e-12)


def test_same_hyperparameters_as_the_multifidelity_model(nested_data):
    """[CK] the hyperparameters of a fitted MultifidelityModel put in a CoKriging give the same
    predictions on nested data ([LG] Prop. 3.1; the recursive model uses the low-fidelity
    PREDICTIONS at the HF points, equal to the observations up to the nugget)."""
    recursive = MultifidelityModel(2, SquaredExponentialKernel, seed=0, use_map=True)
    recursive.fit(nested_data)
    model = CoKriging.from_multifidelity(recursive, nested_data)
    mean, var = model.predict_batch(X_TEST, include_noise=False)
    mean_rec, _, gp_vars = recursive.predict_batch(X_TEST)
    # latent variance of the recursive model: rho^2 s1^2 + s_delta^2 without the two noises
    # (its total variance also adds rho^2 times the LOW-fidelity noise)
    rho = recursive.rhos[0]
    var_rec = rho ** 2 * (gp_vars[:, 0] - recursive.gps[0].get_noise_variance()) \
        + gp_vars[:, 1] - recursive.gps[1].get_noise_variance()
    scale = np.std(nested_data.get_training_data(2)[1])
    np.testing.assert_allclose(mean, mean_rec, atol=1e-3 * scale)
    np.testing.assert_allclose(np.sqrt(var), np.sqrt(np.maximum(var_rec, 0.0)),
                               atol=2e-3 * scale)


def test_v3_defaults():
    """[CK] [V3] CoKriging uses the mfego v3 defaults: Matern 5/2, MAP, centred rho learned under
    V3_RHO_PRIOR within V3_RHO_BOUNDS; a normal prior keeps the v2 bounds +-5; no centring keeps
    v1 (no prior on rho)."""
    model = CoKriging(2)
    assert model.kernel_class is Matern52Kernel and model.use_map and model.rho_centering
    assert all(gp.priors["rho"] == V3_RHO_PRIOR for gp in model.gps)
    assert model.rho_bounds == V3_RHO_BOUNDS
    v2 = CoKriging(2, priors={"rho": DEFAULT_PRIORS["rho"]})
    assert v2.rho_bounds == (-5.0, 5.0) and v2.gps[1].priors["rho"] == ("normal", 1.0, 2.0)
    v1 = CoKriging(2, rho_centering=False)
    assert v1.rho_bounds == (-5.0, 5.0) and all(gp.priors["rho"] is None for gp in v1.gps)
    # maximum likelihood: no prior is used, so rho stays profiled within +-5 (any sign)
    mle = CoKriging(2, use_map=False)
    assert mle.rho_bounds == (-5.0, 5.0)
    assert not any(gp._learns_rho(np.zeros(3), True) for gp in mle.gps)
    # explicit bounds are kept with the default prior
    assert CoKriging(2, rho_bounds=(1e-3, 10.0)).rho_bounds == (1e-3, 10.0)


def test_rho_options_survive_the_json(nested_data):
    """[V3] the bounds of rho are saved; a v2 file (normal prior, no saved bounds) is reloaded
    as v2 (bounds +-5)."""
    model = fitted(nested_data, rho_bounds=(1e-3, 10.0))
    assert CoKriging.from_dict(json.loads(json.dumps(model.to_dict()))).rho_bounds == (1e-3, 10.0)
    state = fitted(nested_data, priors={"rho": ("normal", 1.0, 2.0)}).to_dict()
    del state["rho_bounds"]
    clone = CoKriging.from_dict(state)
    assert clone.rho_bounds == (-5.0, 5.0)
    assert all(gp.priors["rho"] == ("normal", 1.0, 2.0) for gp in clone.gps)


def test_from_multifidelity_keeps_the_options(nested_data):
    """[QUAL] the co-kriging built from a MultifidelityModel keeps its options (priors, rho
    centring, lengthscale prior of each level, rho), also after a JSON round trip."""
    recursive = MultifidelityModel(2, SquaredExponentialKernel, seed=0, use_map=True,
                                   rho_centering=True, priors={"noise": None},
                                   lengthscale_prior=[(3.0, 2.0), (3.0, 1.0)])
    recursive.fit(nested_data)
    model = CoKriging.from_multifidelity(recursive, nested_data)
    clone = CoKriging.from_dict(json.loads(json.dumps(model.to_dict())))
    for ck in (model, clone):
        assert ck.rho_centering and ck.lengthscale_priors == recursive.lengthscale_priors
        for gp_ck, gp in zip(ck.gps, recursive.gps):
            assert gp_ck.priors == gp.priors and gp_ck.rho == gp.rho
    for a, b in zip(model.predict_batch(X_TEST), clone.predict_batch(X_TEST)):
        np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-12)


def test_learned_rho_round_trip(nested_data):
    """[V3] CoKriging with rho learned (lognormal prior): true rho = 2 recovered, JSON round
    trip with the same prior and predictions."""
    model = fitted(nested_data, priors={"rho": ("lognormal", 1.0, 1.0)},
                   rho_bounds=(1e-3, 50.0))
    assert model.rhos[0] == pytest.approx(2.0, abs=0.1)
    clone = CoKriging.from_dict(json.loads(json.dumps(model.to_dict())))
    assert all(gp.priors["rho"] == ("lognormal", 1.0, 1.0) for gp in clone.gps)
    for a, b in zip(model.predict_batch(X_TEST), clone.predict_batch(X_TEST)):
        np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-12)


def test_v1_file_keeps_the_v1_behaviour(nested_data):
    """[RHO] [NORM] a file written by v1 (no rho_centering, no input_bounds, lengthscale_prior
    None) is reloaded with the v1 options and gives the same predictions (bounds [0, 1])."""
    model = fitted(nested_data, priors={"rho": None}, rho_centering=False)
    state = model.to_dict()
    for key in ("rho_centering", "input_bounds"):
        del state[key]
    state["lengthscale_prior"] = None
    clone = CoKriging.from_dict(state)
    assert not clone.rho_centering and clone.input_bounds is None
    assert all(gp.priors["rho"] is None and not gp.rho_centering for gp in clone.gps)
    for a, b in zip(model.predict_batch(X_TEST), clone.predict_batch(X_TEST)):
        np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("by_location", [False, True])
def test_leave_one_out_equals_brute_force(nested_data, by_location):
    """[QUAL] closed-form LOO = conditioning on the data without the point (by_location: and
    without the lower-level observation at the same point), same hyperparameters, prediction
    at that point, noise included."""
    model = fitted(nested_data)
    (x1, y1), (x2, y2) = nested_data.get_training_data(1), nested_data.get_training_data(2)
    y_loo, mean_loo, var_loo = model.leave_one_out(by_location=by_location)
    np.testing.assert_allclose(y_loo, y2)
    for i in range(len(y2)):
        keep_lf = np.linalg.norm(x1 - x2[i], axis=1) > 1e-12 if by_location \
            else np.ones(len(y1), dtype=bool)
        clone = CoKriging.from_dict(model.to_dict())
        clone.condition([x1[keep_lf], np.delete(x2, i, axis=0)],
                        [y1[keep_lf], np.delete(y2, i)])
        mean, var = clone.predict_batch(x2[i])
        assert mean_loo[i] == pytest.approx(mean[0], rel=1e-8, abs=1e-10)
        assert var_loo[i] == pytest.approx(var[0], rel=1e-6)
    y_lf, mean_lf, _ = model.leave_one_out(level=1, by_location=by_location)
    assert len(y_lf) == len(y1) and np.all(np.isfinite(mean_lf))


def test_prediction_scores():
    """[QUAL] Q^2 = 1 - RMSE_rel^2, coverage and z of known predictions."""
    y = np.array([0.0, 1.0, 2.0, 3.0])
    perfect = prediction_scores(y, y, np.full(4, 0.01))
    assert perfect["q2"] == pytest.approx(1.0) and perfect["coverage_95"] == 1.0
    mean = y + np.array([1.0, -1.0, 1.0, -1.0])
    scores = prediction_scores(y, mean, np.full(4, 1.0))
    assert scores["rmse_rel"] == pytest.approx(1.0 / np.std(y))
    assert scores["q2"] == pytest.approx(1.0 - 1.0 / np.var(y))
    assert scores["z_rms"] == pytest.approx(1.0)
    assert scores["coverage_50"] == 0.0 and scores["coverage_90"] == 1.0
    # a biased mean with a well spread z: z_rms > 1 (the standard deviation of z would be 0)
    assert prediction_scores(y, y + 2.0, np.full(4, 1.0))["z_rms"] == pytest.approx(2.0)


@pytest.mark.parametrize("priors, atol", [({"rho": ("normal", 1.0, 2.0)}, 1e-10), (None, 1e-5)])
def test_inputs_are_normalized_with_the_bounds(nested_data, priors, atol):
    """[NORM] the same nested data on [2, 12] give the same co-kriging as on [0, 1]: exactly
    with a profiled rho (v2), to the tolerance of L-BFGS-B with a learned rho (v3 default: the
    round-off of the scaled inputs changes the path of the optimizer, one dimension more)."""
    model = fitted(nested_data, priors=priors)
    scaled = ExperimentData(bounds=[(2.0, 12.0)], costs=[1.0, 10.0])
    for l in (1, 2):
        scaled.x_dict[l] = 2.0 + 10.0 * nested_data.x_dict[l]
        scaled.y_dict[l] = nested_data.y_dict[l]
    model_scaled = fitted(scaled, priors=priors)
    scale = np.std(nested_data.y_dict[2])
    for a, b in zip(model.predict_batch(X_TEST), model_scaled.predict_batch(2.0 + 10 * X_TEST)):
        np.testing.assert_allclose(a, b, rtol=1e-8, atol=atol * scale ** 2)
    clone = CoKriging.from_dict(json.loads(json.dumps(model_scaled.to_dict())))
    np.testing.assert_allclose(clone.predict_batch(2.0 + 10 * X_TEST)[0],
                               model_scaled.predict_batch(2.0 + 10 * X_TEST)[0], rtol=1e-10)


def test_include_noise_adds_the_noise_of_the_level(nested_data):
    model = fitted(nested_data)
    for level in (1, 2):
        _, var = model.predict_batch(X_TEST, level=level)
        _, latent = model.predict_batch(X_TEST, level=level, include_noise=False)
        np.testing.assert_allclose(var - latent, model.gps[level - 1].get_noise_variance(),
                                   rtol=1e-6)


def test_missing_points_propagate_down_three_levels():
    """[CK] a level-3 point missing at levels 2 and 1 must be evaluated at both."""
    data = ExperimentData(bounds=[(0.0, 1.0)], costs=[1.0, 5.0, 25.0])
    data.generate_nested_design([6, 3, 2], seed=0)
    fill_data(data, [forrester_lf, forrester_lf, forrester_hf])
    data.add_observation(3, np.array([0.123]), float(forrester_hf(0.123)), {})
    missing = missing_points_for_nesting(data, 3)
    np.testing.assert_allclose(missing[2], [[0.123]])
    np.testing.assert_allclose(missing[1], [[0.123]])
