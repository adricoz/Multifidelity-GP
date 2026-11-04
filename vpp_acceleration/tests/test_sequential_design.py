"""
Tests of src/sequential_design.py on the Forrester problem (see conftest.py).

Part 1 checks the formulas against an independent computation (the identities of the note and of
complements_mathematiques.pdf). Part 2 checks the decisions (point, level). Part 3 checks the loop
(nested design kept, budget respected, failed runs).
"""
import numpy as np
import pytest
from conftest import ForresterSimulator, forrester_data, forrester_model
from src.cokriging import CoKriging, missing_points_for_nesting, nested_index
from src.kernels import SquaredExponentialKernel
from src.sequential_design import (SequentialCoKriging, adjusted_variance, batched_imse_gains,
                                   candidate_points, choose_level, find_next_point,
                                   find_next_point_alc, imse_gains, level_variances,
                                   level_weights, loo_errors, loo_squared_errors, stored_value,
                                   to_unit)

X_GRID = np.linspace(0.0, 1.0, 201).reshape(-1, 1)


def zeros_z2(model):
    """z^2 = 0 at every training point: no adjustment (a_j = 1)."""
    return [np.zeros(len(gp.x_train)) for gp in model.gps]


def make_noise_free(model, data, noise=1e-10):
    """Same hyperparameters, but (almost) no observation noise: the GP of each level and the
    joint system are conditioned again. The propositions of the note are exact in this case."""
    for gp in model.gps:
        gp.noise = noise
        gp.condition(gp.x_train, gp.y_train)
    model.condition([data.get_training_data(l)[0] for l in (1, 2)],
                    [data.get_training_data(l)[1] for l in (1, 2)])
    return model


def decomposition_gap(model):
    """max |sum_j P_j^2 sigma_j^2 - variance of the co-kriging| / max variance, on X_GRID."""
    decomposed = adjusted_variance(model, zeros_z2(model), to_unit(model, X_GRID))
    _, cokriging_variance = model.predict_batch(X_GRID, include_noise=False)
    return np.max(np.abs(decomposed - cokriging_variance)) / np.max(cokriging_variance)


def loo_identity_gap(model):
    """Proposition 2: max |e_2 - rho e_1 - e(delta_2)| / std(e_2), e = leave-one-out errors of
    the co-kriging with the point removed at all the levels (leave_one_out(by_location=True))."""
    y_2, mean_2, _ = model.leave_one_out(level=2, by_location=True)
    y_1, mean_1, _ = model.leave_one_out(level=1, by_location=True)
    index = nested_index(model.x_all[model.levels_all == 2], model.x_all[model.levels_all == 1])
    e_2 = y_2 - mean_2
    e_1 = (y_1 - mean_1)[index]
    e_delta_2, _ = loo_errors(model.gps[1])
    return np.max(np.abs(e_2 - model.rhos[0] * e_1 - e_delta_2)) / np.std(e_2)


# 1/3 ---------------------------------------------------------------------------------------------
# Formulas
# -------------------------------------------------------------------------------------------------
def test_level_weights_are_the_rho_products(fitted):
    """2 levels: Z_2 = rho_1 delta_1 + delta_2, so P = [rho_1, 1] (note, Eq. N2)."""
    model, _ = fitted
    np.testing.assert_allclose(level_weights(model), [model.rhos[0], 1.0])
    assert 1.5 < model.rhos[0] < 2.5          # Forrester: true rho = 2


def test_decomposition_equals_cokriging_variance(fitted):
    """Proposition 1: sum_j P_j^2 sigma_j^2 is the variance of the co-kriging when the lower level
    is noise-free (exact up to round-off). With the noise learned by mfego it is only approximate:
    measured 1.5 % here (squared exponential kernel, maximum at the border x = 1)."""
    model, data = fitted
    assert level_variances(model, to_unit(model, X_GRID)).shape == (2, len(X_GRID))
    assert decomposition_gap(model) < 5e-2                       # learned noise: approximate
    assert decomposition_gap(make_noise_free(model, data)) < 1e-4   # noise-free: exact


def test_loo_of_one_gp_matches_explicit_deletion(fitted):
    """Eq. N3 (Dubrule): same z^2 as predicting each observation from the others with the
    Gaussian conditioning formulas written out explicitly."""
    model, _ = fitted
    gp = model.gps[0]
    n = len(gp.x_train)
    covariance = (gp.kernel.get_covariance_matrix(gp.x_train)
                  + (gp.noise + gp.jitter) * np.eye(n))           # matrix factorized by the GP
    y_normalized = (gp.y_train - gp.y_mean) / gp.y_std
    z2_explicit = np.zeros(n)
    for i in range(n):
        others = [k for k in range(n) if k != i]
        weights = np.linalg.solve(covariance[np.ix_(others, others)], covariance[others, i])
        mean = weights @ y_normalized[others]
        variance = covariance[i, i] - weights @ covariance[others, i]
        z2_explicit[i] = (y_normalized[i] - mean) ** 2 / variance
    np.testing.assert_allclose(loo_squared_errors(gp), z2_explicit, rtol=1e-6)


def test_loo_error_of_level_2_is_rho_times_level_1_plus_correction(fitted):
    """Proposition 2: e_2 = rho e_1 + e(delta_2). Exact for a noise-free lower level; with the
    learned noise it is approximate (measured 20 % of std(e_2) here): z^2 stays a heuristic."""
    model, data = fitted
    assert loo_identity_gap(make_noise_free(model, data)) < 1e-3


def brute_force_gain(model, data, x0, added_levels):
    """Integrated drop of the high-fidelity variance when x0 is added to the co-kriging at
    `added_levels` (same hyperparameters, any value: the variance does not depend on it)."""
    xs = [data.get_training_data(l)[0] for l in (1, 2)]
    ys = [data.get_training_data(l)[1] for l in (1, 2)]
    _, variance_before = model.predict_batch(X_GRID, include_noise=False)
    xs_new = [np.vstack([x, x0]) if l + 1 in added_levels else x for l, x in enumerate(xs)]
    ys_new = [np.append(y, 0.0) if l + 1 in added_levels else y for l, y in enumerate(ys)]
    model.condition(xs_new, ys_new)
    _, variance_after = model.predict_batch(X_GRID, include_noise=False)
    model.condition(xs, ys)                                        # back to the real data
    return np.mean(variance_before - variance_after)


def test_imse_gains_match_a_full_reconditioning(fitted):
    """Eq. N5 computed with the joint co-kriging: equal to brute force, WITH the learned noise."""
    model, data = fitted
    x0 = np.array([[0.37]])
    gains = imse_gains(model, zeros_z2(model), x0, X_GRID)
    assert gains[0] == pytest.approx(brute_force_gain(model, data, x0, [1]), rel=1e-6)
    assert gains[1] == pytest.approx(brute_force_gain(model, data, x0, [1, 2]), rel=1e-6)


def test_imse_gains_when_the_low_fidelity_is_already_known(fitted):
    """At a point already computed at level 1, G_1 = 0 and G_2 = gain of the high fidelity only."""
    model, data = fitted
    x_lf_only = data.x_dict[1][~np.isin(data.x_dict[1][:, 0], data.x_dict[2][:, 0])][:1]
    gains = imse_gains(model, zeros_z2(model), x_lf_only, X_GRID, missing=np.array([False, True]))
    assert gains[0] == 0.0
    assert gains[1] == pytest.approx(brute_force_gain(model, data, x_lf_only, [2]), rel=1e-6)


def test_imse_gains_noise_free_equal_the_per_level_formula(fitted):
    """Noise-free lower level: the exact gain equals the per-level formula N5 of the note,
    G_l = sum_(j <= l) P_j^2 mean_x k_n_j(x, x0)^2 / (sigma_j^2(x0) + noise_j)."""
    model, data = fitted
    make_noise_free(model, data)
    x0 = np.array([[0.37]])
    u0, u_grid = to_unit(model, x0), to_unit(model, X_GRID)
    weights_squared = level_weights(model) ** 2
    per_level = []
    for j, gp in enumerate(model.gps):
        k_grid = gp.kernel.get_cross_covariance_matrix(u_grid, gp.x_train)
        k_0 = gp.kernel.get_cross_covariance_matrix(u0, gp.x_train)
        v_grid = np.linalg.solve(gp.l_chol, k_grid.T)
        v_0 = np.linalg.solve(gp.l_chol, k_0.T)
        k_n = gp.y_std ** 2 * (gp.kernel.get_cross_covariance_matrix(u_grid, u0) - v_grid.T @ v_0)
        sigma2_0 = gp.y_std ** 2 * (gp.kernel.get_cross_covariance_matrix(u0, u0) - v_0.T @ v_0)
        per_level.append(weights_squared[j] * np.mean(k_n[:, 0] ** 2)
                         / (sigma2_0[0, 0] + gp.get_noise_variance()))
    gains = imse_gains(model, zeros_z2(model), x0, X_GRID)
    np.testing.assert_allclose(gains, np.cumsum(per_level), rtol=1e-3)


# 2/3 ---------------------------------------------------------------------------------------------
# Decisions
# -------------------------------------------------------------------------------------------------
def test_choose_level_best_yield_and_cumulative_cost():
    """l* = argmax G_l / C_l with C_l cumulative (note, Eq. N6)."""
    costs = np.array([1.0, 10.0])
    all_missing = np.array([True, True])
    # ratios: 1 / 1 = 1 against 3 / (1 + 10) = 0.27 -> the low fidelity
    assert choose_level(np.array([1.0, 3.0]), costs, all_missing)[0] == 1
    # ratios: 1 against 30 / 11 = 2.7 -> the high fidelity
    assert choose_level(np.array([1.0, 30.0]), costs, all_missing)[0] == 2
    # low fidelity already known at x0: C_2 = 10 only, level 1 is not a candidate
    level, ratios = choose_level(np.array([1.0, 3.0]), costs, np.array([False, True]))
    assert level == 2 and ratios[1] == pytest.approx(0.3) and np.isinf(ratios[0])
    # remaining budget 5: the high fidelity (C_2 = 11) is not affordable
    assert choose_level(np.array([1.0, 30.0]), costs, all_missing, remaining_budget=5.0)[0] == 1
    # nothing affordable
    assert choose_level(np.array([1.0, 30.0]), costs, all_missing, remaining_budget=0.5)[0] is None


def test_find_next_point_is_the_maximum_of_the_adjusted_variance(fitted):
    """The differential evolution finds the maximum of k_adj seen on a fine grid."""
    model, data = fitted
    z2 = [loo_squared_errors(gp) for gp in model.gps]
    x0, value = find_next_point(model, z2, data.bounds, seed=0)
    grid = np.linspace(0.0, 1.0, 2001).reshape(-1, 1)
    grid_maximum = np.max(adjusted_variance(model, z2, to_unit(model, grid)))
    assert value >= 0.99 * grid_maximum
    assert 0.0 <= x0[0] <= 1.0


def test_find_next_point_respects_feasibility_and_exclusions(fitted):
    """Criterion set to 0 outside the feasible domain and near the excluded points."""
    model, data = fitted
    z2 = zeros_z2(model)
    x0, _ = find_next_point(model, z2, data.bounds, feasibility=lambda x: x[:, 0] < 0.5, seed=0)
    assert x0[0] < 0.5
    x_free, _ = find_next_point(model, z2, data.bounds, seed=0)
    x_excluded, _ = find_next_point(model, z2, data.bounds, excluded=x_free[None, :], seed=0)
    assert abs(x_excluded[0] - x_free[0]) >= 1e-3


# 3/3 ---------------------------------------------------------------------------------------------
# The loop
# -------------------------------------------------------------------------------------------------
def test_stored_value():
    """Value of a point already computed at a level, None for an unknown point."""
    data = forrester_data()
    x_hf = data.x_dict[2][0]
    assert stored_value(data, 2, x_hf) == pytest.approx(data.y_dict[2][0])
    assert stored_value(data, 2, np.array([0.123456])) is None


def test_run_keeps_the_design_nested_and_respects_the_budget():
    """Every iteration keeps X_2 in X_1, the new runs cost at most the budget, and each
    iteration runs, at one point, the levels 1..level that were still missing there (the
    algorithm may choose a point already computed at level 1: then only level 2 is run)."""
    data = forrester_data(points_per_level=(6, 3))
    simulator = ForresterSimulator()
    design = SequentialCoKriging(data, simulator, model_factory=forrester_model, seed=0)
    history = design.run(budget=40.0)
    assert len(history) > 0
    assert design.cost_spent <= 40.0
    assert simulator.calls[1] + 10 * simulator.calls[2] == pytest.approx(design.cost_spent)
    missing = missing_points_for_nesting(data, 2)
    assert all(len(points) == 0 for points in missing.values())
    for step in history:
        expected = [j for j in range(1, step["level"] + 1) if step["missing"][j - 1]]
        assert step["levels_run"] == expected
    assert design.model is not None and design.model.is_fitted()


def test_failed_low_fidelity_run_stops_the_higher_levels():
    """A failed low-fidelity run: the high fidelity is not run there, the design stays nested
    and the point is excluded from the next searches."""
    data = forrester_data(points_per_level=(6, 3))
    simulator = ForresterSimulator(fail_above=0.9)
    design = SequentialCoKriging(data, simulator, model_factory=forrester_model, seed=0)
    cost, levels_run = design.evaluate({"x": np.array([0.95]), "level": 2})
    assert levels_run == [1] and cost == 1.0
    assert np.isnan(stored_value(data, 1, np.array([0.95])))
    assert stored_value(data, 2, np.array([0.95])) is None
    assert len(design.failed_points) == 1
    design.fit()                     # the failed point is ignored by the fit (NaN excluded)
    assert all(len(points) == 0 for points in missing_points_for_nesting(data, 2).values())


def test_batched_gains_equal_the_gains_of_each_point(fitted):
    """batched_imse_gains computes, for many candidates at once, the same gains as imse_gains
    point by point, including a candidate where the low fidelity is already known."""
    model, data = fitted
    x_lf_only = data.x_dict[1][~np.isin(data.x_dict[1][:, 0], data.x_dict[2][:, 0])][:1]
    x_cand = np.vstack([np.array([[0.2], [0.5], [0.8]]), x_lf_only])
    missing = np.array([[True, True], [True, True], [True, True], [False, True]])
    batched = batched_imse_gains(model, x_cand, X_GRID, missing)
    for m in range(len(x_cand)):
        single = imse_gains(model, zeros_z2(model), x_cand[m], X_GRID, missing[m])
        np.testing.assert_allclose(batched[m], single, rtol=1e-10, atol=1e-15)


def test_alc_chooses_the_best_yield(fitted):
    """find_next_point_alc returns the (candidate, level) pair of largest G_l / C_l."""
    model, data = fitted
    x_cand, missing = candidate_points(data, 64, seed=0)
    z2 = [loo_squared_errors(gp) for gp in model.gps]
    proposal = find_next_point_alc(model, z2, x_cand, missing, X_GRID, np.array(data.costs))
    best_ratio = -np.inf
    for m in range(len(x_cand)):
        gains = imse_gains(model, z2, x_cand[m], X_GRID, missing[m])
        _, ratios = choose_level(gains, np.array(data.costs), missing[m])
        best_ratio = max(best_ratio, np.max(ratios))
    assert proposal["criterion"] == pytest.approx(best_ratio, rel=1e-9)
    assert proposal["level"] in (1, 2)


def test_candidate_points_include_partial_points_and_respect_exclusions():
    """Candidates = Sobol points + low-fidelity points without high fidelity (missing [False,
    True]); infeasible and excluded candidates are removed."""
    data = forrester_data(points_per_level=(6, 3))
    x_cand, missing = candidate_points(data, 32, seed=0)
    partial = x_cand[~missing[:, 0]]
    assert len(partial) == 3                      # 6 LF points, 3 of them have the HF
    assert np.all(missing[~missing[:, 0], 1])     # their high fidelity is missing
    x_cand, _ = candidate_points(data, 32, seed=0, feasibility=lambda x: x[:, 0] < 0.5)
    assert np.all(x_cand[:, 0] < 0.5)
    x_cand, _ = candidate_points(data, 32, seed=0, excluded=partial[:1])
    assert np.min(np.abs(x_cand[:, 0] - partial[0, 0])) >= 1e-3


def test_run_with_alc_keeps_the_design_nested_and_respects_the_budget():
    """Same checks as for the variance criterion, with point_criterion="alc"."""
    data = forrester_data(points_per_level=(6, 3))
    simulator = ForresterSimulator()
    design = SequentialCoKriging(data, simulator, model_factory=forrester_model,
                                 point_criterion="alc", n_candidates=256, seed=0)
    history = design.run(budget=40.0)
    assert len(history) > 0 and design.cost_spent <= 40.0
    assert simulator.calls[1] + 10 * simulator.calls[2] == pytest.approx(design.cost_spent)
    assert all(len(points) == 0 for points in missing_points_for_nesting(data, 2).values())
    for step in history:
        expected = [j for j in range(1, step["level"] + 1) if step["missing"][j - 1]]
        assert step["levels_run"] == expected


def test_q2_target_stops_the_loop():
    """With a Q^2 target already reached, run() stops at the first iteration."""
    data = forrester_data(points_per_level=(11, 5))
    design = SequentialCoKriging(data, ForresterSimulator(), model_factory=forrester_model,
                                 q2_target=-np.inf, seed=0)
    assert design.run(budget=100.0) == []


def test_shared_lengthscales_model_in_the_sequential_design():
    """CoKriging(share_lengthscales=True) (mfego option, 2026-10-21) in the loop: the exact gains
    still equal a full reconditioning (they only use the joint covariance), and an ALC run keeps
    the design nested, respects the budget and ends with the shared lengthscales."""
    def shared_model():
        return CoKriging(2, kernel_class=SquaredExponentialKernel, seed=0,
                         share_lengthscales=True)
    data = forrester_data()
    model = shared_model()
    model.fit(data)
    np.testing.assert_array_equal(model.gps[1].kernel.lengthscale,
                                  model.gps[0].kernel.lengthscale)
    x0 = np.array([[0.37]])
    gains = imse_gains(model, zeros_z2(model), x0, X_GRID)
    assert gains[1] == pytest.approx(brute_force_gain(model, data, x0, [1, 2]), rel=1e-6)
    run_data = forrester_data(points_per_level=(6, 3))
    design = SequentialCoKriging(run_data, ForresterSimulator(), model_factory=shared_model,
                                 point_criterion="alc", n_candidates=256, seed=0)
    design.run(budget=40.0)
    assert design.cost_spent <= 40.0
    assert all(len(p) == 0 for p in missing_points_for_nesting(run_data, 2).values())
    np.testing.assert_array_equal(design.model.gps[1].kernel.lengthscale,
                                  design.model.gps[0].kernel.lengthscale)
