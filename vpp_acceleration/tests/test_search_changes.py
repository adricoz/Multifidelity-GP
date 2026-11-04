"""
Tests of the changes of 2026-10-23 in the search: integration weighted per point
(sequential_design.batched_imse_gains, vpp_targeted.choose), sections drawn in a trust region
and clouds that also move the section (vpp_optimize.RealFoilProblem), weights of the integration
points (VPPSectionOptimizer.point_weights). numpy / scipy / mfego only.
"""
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src.cokriging import CoKriging
from src.data_management import ExperimentData
from src.section_pca import CFoilFamily, KulfanBoxFamily, SectionPCA
from src.sequential_design import batched_imse_gains
from src.vpp_optimize import RealFoilProblem, VPPSectionOptimizer
from src.vpp_targeted import RHO_PRIOR, TorsorCoKriging, choose

STUDY_CONFIG = (Path(__file__).resolve().parents[2] / "studies" / "cfoil_intens_kulfan_trim_2"
                / "config_cfoil_trim.json")


def two_level_data(seed=0):
    rng = np.random.default_rng(seed)
    x_low = rng.random((30, 2))
    x_high = x_low[:8]
    y_low = np.sin(3 * x_low[:, 0]) + x_low[:, 1] ** 2
    y_high = 1.2 * (np.sin(3 * x_high[:, 0]) + x_high[:, 1] ** 2) + 0.1 * x_high[:, 0]
    return x_low, x_high, y_low, y_high


def fitted_cokriging() -> CoKriging:
    x_low, x_high, y_low, y_high = two_level_data()
    data = ExperimentData(bounds=[(0.0, 1.0), (0.0, 1.0)], costs=[1.0, 2.0])
    data.x_dict, data.y_dict = {1: x_low, 2: x_high}, {1: y_low, 2: y_high}
    data.metrics_dict = {1: [{}] * len(x_low), 2: [{}] * len(x_high)}
    model = CoKriging(2, priors=RHO_PRIOR, seed=0)
    model.fit(data)
    return model


def test_weighted_integration_reduces_to_the_mean():
    """int_weights = 1/N everywhere gives the plain mean; a weight on one point only gives the
    reduction at that point."""
    model = fitted_cokriging()
    rng = np.random.default_rng(1)
    x_cand, x_int = rng.random((20, 2)), rng.random((15, 2))
    missing = np.ones((20, 2), dtype=bool)
    plain = batched_imse_gains(model, x_cand, x_int, missing)
    uniform = batched_imse_gains(model, x_cand, x_int, missing, int_weights=np.full(15, 1 / 15))
    assert np.allclose(plain, uniform, rtol=1e-10, atol=0.0)
    single = np.zeros(15)
    single[4] = 2.0
    one_point = batched_imse_gains(model, x_cand, x_int[4:5], missing)
    # same quantity computed with 1 or 15 integration points: rounding differs at ~1e-8
    assert np.allclose(batched_imse_gains(model, x_cand, x_int, missing, int_weights=single),
                       2.0 * one_point, rtol=1e-6, atol=0.0)


def test_choose_per_point_weights_equal_per_output_weights():
    """weights (K,) and the same weights spread over the points, (N, K) = w_k / N, choose the
    same pair with the same yield."""
    x_low, x_high, y_low, y_high = two_level_data()
    surrogate = TorsorCoKriging([(0.0, 1.0), (0.0, 1.0)], costs=(1.0, 2.0), n_models=2)
    surrogate.add(1, x_low, np.column_stack([y_low, 2 * y_low] + [np.zeros(len(x_low))] * 5))
    surrogate.add(2, x_high, np.column_stack([y_high, 2 * y_high] + [np.zeros(len(x_high))] * 5))
    surrogate.fit()
    rng = np.random.default_rng(2)
    x_cand, x_int = rng.random((25, 2)), rng.random((12, 2))
    missing = np.ones((25, 2), dtype=bool)
    weights = np.array([3.0, 0.5])
    a = choose(surrogate, weights, x_cand, missing, x_int, np.inf)
    b = choose(surrogate, np.tile(weights / 12, (12, 1)), x_cand, missing, x_int, np.inf)
    assert (a["index"], a["level"]) == (b["index"], b["level"])
    assert np.isclose(a["ratio"], b["ratio"], rtol=1e-10)


def small_problem():
    reference = CFoilFamily.from_study_config(STUDY_CONFIG).reference_weights()
    family = KulfanBoxFamily(reference - 0.02, reference + 0.02, 0.00197)
    pca = SectionPCA(family.te_thickness).fit(family.sample(400, seed=0))
    pca.n_modes = 4
    return RealFoilProblem(pca, family), pca.encode(reference)[0]


def test_local_sections_stay_in_the_family_at_the_radius():
    problem, center = small_problem()
    local = problem.local_sections(center, 0.5, 60, seed=3)
    assert len(local) == 60
    assert np.all(problem.family.contains(problem.weights(local)))
    # typical distance radius: E|g| / sqrt(k) = 0.94 for k = 4, so about 0.47
    assert 0.3 < np.mean(np.linalg.norm(local - center, axis=1)) < 0.65


def test_clouds_keep_or_move_the_section():
    problem, center = small_problem()
    states = np.array([np.concatenate([center, [3.0, 0.5, 0.0]]),
                       np.concatenate([center, [5.0, 1.0, -1.0]])])
    fixed, owner = problem.cloud(states, 8, seed=4)
    assert fixed.shape == (16, 7) and np.array_equal(owner, np.repeat([0, 1], 8))
    assert np.allclose(fixed[:, :4], center)
    moved, owner = problem.cloud(states, 8, seed=4, section_radius=0.5)
    assert np.all(problem.family.contains(problem.weights(moved[:, :4])))
    # every second point keeps the section of its state, the others move it
    still = np.all(np.isclose(moved[:, :4], center), axis=1)
    assert still.sum() == 8 and (~still).sum() == len(moved) - 8


def test_point_weights_give_each_state_its_probability():
    """Summed over the points of a state, the weights are w_c (dV/dC_k)_c^2, whatever the
    number of points the state kept."""
    fake = SimpleNamespace(state_conditions=np.array([0, 1, 1]),
                           int_owner=np.array([0, 0, 0, 1, 2, 2]),
                           states=np.zeros((3, 7)), condition_weights=np.array([0.25, 0.75]),
                           derivatives=np.array([[2.0, 0, 0, 0, 0, 0], [0, 1.0, 0, 0, 0, 0]]))
    weights = VPPSectionOptimizer.point_weights(fake)
    per_state = np.array([weights[fake.int_owner == i].sum(axis=0) for i in range(3)])
    assert np.allclose(per_state[0], [0.25 * 4, 0, 0, 0, 0, 0])
    assert np.allclose(per_state[1], [0, 0.75, 0, 0, 0, 0])
    assert np.allclose(per_state[2], [0, 0.75, 0, 0, 0, 0])
