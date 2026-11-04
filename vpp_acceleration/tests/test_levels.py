"""
Tests of the L-level generalisation of the VPP layer (2026-10-27): TorsorCoKriging, the gains and
the choice (vpp_targeted.choose, levels_to_run), the kriging believer, the nested start design
(vpp_optimize.start_design), at 3 levels. numpy / scipy / mfego only.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from src.sequential_design import batched_imse_gains, joint_posterior_covariance
from src.vpp_optimize import start_design
from src.vpp_targeted import TorsorCoKriging, choose, levels_to_run, nested_subset

BOUNDS = [(0.0, 1.0), (0.0, 1.0)]


def three_level_functions(x):
    """Three correlated levels of a smooth 2D function (level 3 = reference)."""
    f1 = np.sin(3 * x[:, 0]) + x[:, 1] ** 2
    f2 = 1.1 * f1 + 0.2 * x[:, 0]
    f3 = 0.9 * f2 - 0.1 * np.cos(2 * x[:, 1])
    return [f1, f2, f3]


def three_level_surrogate(sizes=(40, 16, 8)) -> TorsorCoKriging:
    rng = np.random.default_rng(0)
    x = rng.random((sizes[0], 2))
    surrogate = TorsorCoKriging(BOUNDS, costs=(1.0, 2.0, 4.0), n_models=2, n_outputs=7)
    for level, n in enumerate(sizes, start=1):
        if level > 1:
            x = nested_subset(x, n, BOUNDS)
        f = three_level_functions(x)[level - 1]
        y = np.column_stack([f, 2 * f ** 2] + [np.zeros(len(x))] * 5)
        surrogate.add(level, x, y)
    surrogate.fit()
    return surrogate


def test_three_level_surrogate():
    surrogate = three_level_surrogate()
    assert surrogate.n_levels == 3 and surrogate.sizes == (40, 16, 8)
    assert surrogate.cost == pytest.approx(40 * 1 + 16 * 2 + 8 * 4)
    assert all(model.num_levels == 3 for model in surrogate.models)
    x = np.array([[0.3, 0.6], [0.8, 0.2]])
    truth = three_level_functions(x)
    assert np.allclose(surrogate.predict(x)[:, 0], truth[2], atol=0.05)          # top level
    assert np.allclose(surrogate.predict(x, level=1)[:, 0], truth[0], atol=0.05)
    with pytest.raises(ValueError, match="one value per level"):
        TorsorCoKriging(BOUNDS, costs=(1.0,))


def test_three_level_gains_are_the_exact_variance_reduction():
    """batched_imse_gains at L = 3: the gain of adding the levels 1..l at a candidate equals the
    reduction of the top-level posterior variance at the integration points, computed from the
    joint posterior covariance (exact Gaussian conditioning)."""
    model = three_level_surrogate().models[0]
    rng = np.random.default_rng(1)
    x_cand, x_int = rng.random((5, 2)), rng.random((7, 2))
    missing = np.ones((5, 3), dtype=bool)
    gains = batched_imse_gains(model, x_cand, x_int, missing)
    top = np.full(len(x_int), 3)
    noise = np.array([gp.get_noise_variance() for gp in model.gps])
    for m in range(5):
        for l in (1, 2, 3):
            levels = np.arange(1, l + 1)
            xs = np.repeat(x_cand[m:m + 1], l, axis=0)
            c = joint_posterior_covariance(model, x_int, top, xs, levels)          # (N, l)
            s = joint_posterior_covariance(model, xs, levels, xs, levels) + np.diag(noise[:l])
            reduction = np.einsum("ni,ij,nj->n", c, np.linalg.inv(s), c)
            assert gains[m, l - 1] == pytest.approx(reduction.mean(), rel=1e-6)


def test_choice_levels_and_costs_at_three_levels():
    surrogate = three_level_surrogate()
    rng = np.random.default_rng(2)
    x_cand, x_int = rng.random((30, 2)), rng.random((12, 2))
    missing = np.ones((30, 3), dtype=bool)
    missing[0, :2] = False                     # candidate 0 already has levels 1 and 2
    choice = choose(surrogate, np.array([1.0, 0.2]), x_cand, missing, x_int, np.inf)
    assert choice["gains"].shape == (30, 3) and 1 <= choice["level"] <= 3
    # cumulative cost of the missing levels: level 3 at candidate 0 costs 4 only
    only_top = choose(surrogate, np.array([1.0, 0.2]), x_cand[:1], missing[:1], x_int, np.inf)
    assert only_top["level"] == 3 and only_top["cost"] == pytest.approx(4.0)
    assert levels_to_run(only_top) == [3]
    assert levels_to_run({"level": 2, "missing": np.array([True, True, True])}) == [1, 2]
    assert levels_to_run({"level": 3, "missing": np.array([False, True, True])}) == [2, 3]
    # a budget that only fits level 1 at the new candidates
    cheap = choose(surrogate, np.array([1.0, 0.2]), x_cand[1:], missing[1:], x_int, 1.5)
    assert cheap["level"] == 1


def test_three_level_kriging_believer_matches_a_full_conditioning():
    """believe() at 3 levels: same posterior variances as adding the predicted means at every
    level and conditioning again."""
    surrogate, reference = three_level_surrogate(), three_level_surrogate()
    x = np.array([[0.35, 0.55]])
    surrogate.believe(x, [1, 2, 3])
    for level in (1, 2, 3):
        reference.add(level, x, reference.predict(x, level=level))
    reference.fit(refit=False)
    x_test = np.random.default_rng(3).random((10, 2))
    assert np.allclose(surrogate.variances(x_test), reference.variances(x_test), rtol=1e-6,
                       atol=1e-12)
    assert np.allclose(surrogate.predict(x_test), reference.predict(x_test), atol=1e-8)


def test_nested_start_design_at_three_levels():
    """start_design with 3 levels: each level picks its points among the successful points of
    the level below; a point failed at level 2 never reaches level 3."""
    rng = np.random.default_rng(4)

    class FakeProblem:  # pylint: disable=too-few-public-methods
        lo, hi = np.array([0.0]), np.array([1.0])

        @staticmethod
        def sample_sections(n, seed):
            return np.random.default_rng(seed).random((n, 1))

        @staticmethod
        def bounds():
            return BOUNDS

    calls = []

    def evaluate(x, level, tag):
        calls.append((level, len(x)))
        y = np.column_stack([x[:, 0] + level] * 7)
        if level == 2:
            y[: 2] = np.nan                              # two failures at level 2
        return y, [{}] * len(x)

    runner = SimpleNamespace(evaluate=evaluate, run=SimpleNamespace(log=lambda message: None))
    surrogate = start_design(FakeProblem(), runner, [20, 8, 4], seed=1, n_models=2,
                             costs=(1.0, 2.0, 3.0))
    assert calls == [(1, 20), (2, 8), (3, 4)]
    assert surrogate.sizes == (20, 6, 4)
    for upper, lower in ((surrogate.x_lists[1], surrogate.x_lists[0]),
                         (surrogate.x_lists[2], surrogate.x_lists[1])):
        assert all(np.any(np.all(np.isclose(p, lower), axis=1)) for p in upper)
    with pytest.raises(ValueError, match="sizes"):
        start_design(FakeProblem(), runner, [20, 8], seed=1, costs=(1.0, 2.0, 3.0))
    del rng
