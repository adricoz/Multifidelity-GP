"""
Tests of the changes of 2026-10-27 in the search (CHANGEMENTS_2026-10-27.md): the kriging believer
by extension of the Cholesky factor (sequential_design.append_observations,
TorsorCoKriging.believe), the median of the sensitivities, the convergence of a region and the
restart (VPPSectionOptimizer). numpy / scipy / mfego only.
"""
import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src.cokriging import CoKriging
from src.data_management import ExperimentData
from src.section_pca import CFoilFamily, KulfanBoxFamily, SectionPCA
from src.sequential_design import append_observations, joint_posterior_covariance
from src.vpp_optimize import RealFoilProblem, VPPSectionOptimizer
from src.vpp_targeted import RHO_PRIOR, TorsorCoKriging, choose, latent_mean

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


def test_append_observations_equals_a_full_conditioning():
    """New observations equal to their predicted means: the extended factor gives the same
    means and posterior covariances as conditioning again on all the data."""
    model = fitted_cokriging()
    x_low, x_high, y_low, y_high = two_level_data()
    rng = np.random.default_rng(3)
    x_new = rng.random((3, 2))
    # points 0 and 1 at both levels (nested), point 2 at level 1 only
    x_obs, levels = x_new[[0, 0, 1, 1, 2]], np.array([1, 2, 1, 2, 1])
    full = copy.deepcopy(model)
    lf_new = x_obs[levels == 1]
    hf_new = x_obs[levels == 2]
    full.condition([np.vstack([x_low, lf_new]), np.vstack([x_high, hf_new])],
                   [np.concatenate([y_low, latent_mean(model, lf_new, 1)]),
                    np.concatenate([y_high, latent_mean(model, hf_new, 2)])])
    append_observations(model, x_obs, levels)
    x_test = rng.random((20, 2))
    for level in (1, 2):
        mean_a, var_a = model.predict_batch(x_test, level=level, include_noise=False)
        mean_b, var_b = full.predict_batch(x_test, level=level, include_noise=False)
        assert np.allclose(mean_a, mean_b, rtol=1e-8, atol=1e-10)
        assert np.allclose(var_a, var_b, rtol=1e-6, atol=1e-12)
    levels_test = np.tile([1, 2], 10)
    assert np.allclose(joint_posterior_covariance(model, x_test, levels_test, x_test, levels_test),
                       joint_posterior_covariance(full, x_test, levels_test, x_test, levels_test),
                       rtol=1e-6, atol=1e-12)


def small_surrogate() -> TorsorCoKriging:
    x_low, x_high, y_low, y_high = two_level_data()
    surrogate = TorsorCoKriging([(0.0, 1.0), (0.0, 1.0)], costs=(1.0, 2.0), n_models=2)
    surrogate.add(1, x_low, np.column_stack([y_low, 2 * y_low ** 2] + [np.zeros(len(x_low))] * 5))
    surrogate.add(2, x_high, np.column_stack([y_high, 2 * y_high ** 2]
                                             + [np.zeros(len(x_high))] * 5))
    surrogate.fit()
    return surrogate


def batch(surrogate, believe: bool, n_choices=6) -> list:
    """The batch choice of VPPSectionOptimizer.step: by believe(), or by the former way (the
    predicted means added to the data, conditioning on all of them, data removed at the end)."""
    rng = np.random.default_rng(4)
    x_cand, x_int = rng.random((40, 2)), rng.random((15, 2))
    missing = np.ones((40, 2), dtype=bool)
    weights = np.column_stack([np.full(15, 1.0), np.full(15, 0.3)])
    sizes, choices = surrogate.sizes, []
    for _ in range(n_choices):
        choice = choose(surrogate, weights, x_cand, missing, x_int, np.inf)
        choices.append(choice)
        missing[choice["index"], :choice["level"]] = False
        levels = VPPSectionOptimizer.levels_to_run(choice)
        if believe:
            surrogate.believe(choice["x"], levels)
        else:
            for level in levels:
                surrogate.add(level, choice["x"], surrogate.predict(choice["x"][None], level=level))
            surrogate.fit(refit=False)
    surrogate.truncate(sizes)
    return choices


def test_believe_gives_the_same_batch():
    a = batch(small_surrogate(), believe=False)
    b = batch(small_surrogate(), believe=True)
    assert [(c["index"], c["level"]) for c in a] == [(c["index"], c["level"]) for c in b]
    assert np.allclose([c["ratio"] for c in a], [c["ratio"] for c in b], rtol=1e-6)
    # and the next fit forgets the believed observations
    surrogate = small_surrogate()
    reference = surrogate.variances(np.array([[0.3, 0.7]]))
    batch(surrogate, believe=True)
    surrogate.fit(refit=False)
    assert np.allclose(surrogate.variances(np.array([[0.3, 0.7]])), reference, rtol=1e-10)


def fake_sensitivities(values, memory):
    """VPPSectionOptimizer.sensitivities on a fake whose VPP pool returns `values` in turn."""
    returned = iter(values)
    fake = SimpleNamespace(
        problem=SimpleNamespace(stiffness_ratios=lambda scores: np.ones(1)),
        incumbent=np.zeros(2), incumbent_states=[{"success": True}, {"success": True}],
        conditions=[(45.0, 12.0), (45.0, 16.0)], derivatives=None,
        vpp_pool=SimpleNamespace(sensitivities=lambda *args: next(returned)),
        run=SimpleNamespace(log=lambda message: None),
        sensitivity_history=[], sensitivity_memory=memory)
    out = []
    for _ in values:
        VPPSectionOptimizer.sensitivities(fake)
        out.append(fake.derivatives.copy())
    return out


def test_sensitivities_median_ignores_an_isolated_jump():
    base = np.array([[117.0, -1.0, 0.5, 0.0, 0.1, 0.0], [50.0, 2.0, 0.9, 0.0, 0.1, 0.4]])
    spike = base.copy()
    spike[0, 0] = 224.0                                   # the jump of the fourth run
    out = fake_sensitivities([base, base, spike, base], memory=3)
    assert all(np.allclose(d, base) for d in out)
    # memory 1: the former behaviour, the jump is taken
    assert np.isclose(fake_sensitivities([base, spike], memory=1)[1][0, 0], 224.0)


def fake_region(gains, optima=(), incumbent=(0.0, 0.0)):
    return SimpleNamespace(convergence=(3, 0.01), region_gains=list(gains),
                           incumbent=np.array(incumbent), merge_distance=1.0,
                           optima=[{"scores": np.array(o)} for o in optima])


def test_convergence_of_a_region():
    converged = VPPSectionOptimizer.converged
    assert not converged(fake_region([0.03, 0.0]))                   # fewer updates than the window
    assert not converged(fake_region([0.0, 0.006, 0.005]))           # 0.011 kt in the window
    assert converged(fake_region([0.05, 0.004, 0.0, 0.005]))         # 0.009 kt in the last 3
    assert converged(fake_region([0.05], optima=[(0.5, 0.5)]))       # basin of a known optimum
    assert not converged(fake_region([0.05], optima=[(1.0, 1.0)]))
    assert not converged(SimpleNamespace(convergence=None))           # no test: runs 1 to 4


def test_restart_starts_far_from_the_optima():
    reference = CFoilFamily.from_study_config(STUDY_CONFIG).reference_weights()
    family = KulfanBoxFamily(reference - 0.02, reference + 0.02, 0.00197)
    pca = SectionPCA(family.te_thickness).fit(family.sample(400, seed=0))
    pca.n_modes = 4
    problem = RealFoilProblem(pca, family)
    incumbent = pca.encode(reference)[0]
    target = incumbent + np.array([3.0, 0.0, 0.0, 0.0])

    def scores(candidates):
        # predicted gain: highest near `target`; sigma 0
        return -np.linalg.norm(candidates - target, axis=1), np.zeros(len(candidates))

    fake = SimpleNamespace(
        updates=[{"incumbent_speed": 13.08, "update": 30, "elapsed_min": 70.0}],
        incumbent=incumbent, incumbent_states=["states"], optima=[], region=0,
        problem=problem, n_screen=64, screen_seed=5, n_updates=31, kappa=1.0,
        first_order_scores=scores, restart_distance=2.0, initial_radius=1.0, radius=0.1,
        region_gains=[0.0] * 10, new_region=False, sensitivity_history=[1],
        run=SimpleNamespace(log=lambda message: None))
    VPPSectionOptimizer.restart(fake)
    assert len(fake.optima) == 1 and np.allclose(fake.optima[0]["scores"], incumbent)
    assert fake.optima[0]["speed"] == 13.08 and fake.optima[0]["states"] == ["states"]
    assert fake.region == 1 and fake.incumbent_states is None and fake.radius == 1.0
    assert fake.region_gains == [] and fake.new_region and fake.sensitivity_history == []
    assert np.linalg.norm(fake.incumbent - incumbent) >= 2.0
    assert np.all(problem.family.contains(problem.weights(fake.incumbent)))
    # the best predicted among the sections far enough
    pool = problem.sample_sections(4 * 64, 5 + 31 * 31 + 17)
    far = pool[np.linalg.norm(pool - incumbent, axis=1) >= 2.0]
    assert np.allclose(fake.incumbent, far[np.argmin(np.linalg.norm(far - target, axis=1))])
