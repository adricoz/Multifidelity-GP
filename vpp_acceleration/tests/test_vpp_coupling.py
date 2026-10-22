"""
Tests of src/meta_foil.py, src/vpp_coupling.py and src/vpp_targeted.py. They need the MDS
environment (numba, pymoo, neuralfoil: Multifidelity-GP-Clean/.venv-mds) and are skipped
elsewhere:
    ..\\.venv-mds\\Scripts\\python -m pytest tests/test_vpp_coupling.py
"""
import numpy as np
import pytest

pytest.importorskip("numba")
pytest.importorskip("pymoo")
pytest.importorskip("neuralfoil")

# pylint: disable=wrong-import-position
from src.meta_foil import ANCHOR_RS, HVOL_REF, N_SCORES, MetaFoil  # noqa: E402
from src.sequential_design import batched_imse_gains  # noqa: E402
from src.vpp_coupling import MDS_DIR, IntensSYVPP  # noqa: E402
from src.vpp_targeted import choose, integration_cloud, nested_design  # noqa: E402


@pytest.fixture(scope="module")
def meta():
    return MetaFoil()


@pytest.fixture(scope="module")
def vpp():
    return IntensSYVPP()


def attitudes(n, seed=0):
    rng = np.random.default_rng(seed)
    return np.column_stack([rng.uniform(0, 8, n), rng.uniform(-1.5, 1.5, n),
                            rng.uniform(-6, 6, n), np.full(n, HVOL_REF)])


def test_chebyshev_rs_matches_mds(meta, vpp):
    """The numpy evaluation of the anchor RS equals MDS' VppChebyRS (unclamped)."""
    del vpp                                            # builds the boat: MDS on sys.path
    import sys  # noqa: PLC0415
    assert str(MDS_DIR) in sys.path
    from Systems.Ships.Sail.IntensSY.vpp_cheby_rs import VppChebyRS  # noqa: PLC0415
    rs = VppChebyRS(str(ANCHOR_RS), clamp=False)
    x = attitudes(30)[:, :3] * 1.1                     # a little outside the box too
    expected = np.array([rs.eval(row).copy() for row in x])
    np.testing.assert_allclose(meta.anchor(x), expected, rtol=1e-10, atol=1e-12)


def test_meta_reference_section_is_the_anchor(meta):
    """Level 2 at the reference section and at the reference sink IS the production foil."""
    att = attitudes(20)
    x = np.column_stack([np.repeat(meta.reference_scores[None], 20, axis=0), att])
    np.testing.assert_allclose(meta(x, 2), meta.anchor(att[:, :3]), atol=1e-12)


def test_meta_moments_follow_the_forces(meta):
    """Away from the reference, the roll and yaw moments change by r x dF (the pitch moment also
    gets the section moment): forces and moments stay consistent."""
    scores = meta.sample_sections(10, seed=4)
    att = attitudes(10, seed=1)
    x = np.column_stack([scores, att])
    out, anchor = meta(x, 2), meta.anchor(att[:, :3])
    transported = anchor[:, 3:6] + np.cross(meta.lever_arm, out[:, 0:3] - anchor[:, 0:3])
    np.testing.assert_allclose(out[:, [3, 5]], transported[:, [0, 2]], atol=1e-12)


def test_meta_levels_are_correlated(meta):
    """The low fidelity is informative: correlation > 0.9 with the high fidelity, every output."""
    x = np.column_stack([meta.sample_sections(100, seed=5), attitudes(100, seed=2)])
    low, high = meta(x, 1), meta(x, 2)
    for k in range(6):
        assert np.corrcoef(low[:, k], high[:, k])[0, 1] > 0.9


def test_adapter_reproduces_the_production_foil(meta, vpp):
    """The VPP with the adapter around the anchor RS gives exactly the production speed."""
    vpp.reset_foil()
    production = vpp.solve(45.0, 16.0)
    vpp.set_foil(lambda x: meta.anchor(x[:, N_SCORES:N_SCORES + 3]), meta.reference_scores)
    adapted = vpp.solve(45.0, 16.0)
    vpp.reset_foil()
    assert production["success"] and adapted["success"]
    assert adapted["speed"] == pytest.approx(production["speed"], abs=1e-9)
    assert vpp.foil.surrogate_model is vpp.original_rs


def test_integration_cloud_keeps_the_section(meta):
    points = np.column_stack([meta.sample_sections(3, seed=6), attitudes(3)])
    cloud = integration_cloud(points, 16, seed=0)
    assert cloud.shape == (48, 16)
    for i in range(3):
        np.testing.assert_array_equal(cloud[16 * i:16 * (i + 1), :N_SCORES],
                                      np.repeat(points[i:i + 1, :N_SCORES], 16, axis=0))
    assert np.all(cloud[:, N_SCORES] >= 0.0) and np.all(cloud[:, N_SCORES] <= 8.0)


def test_choose_returns_the_best_weighted_yield(meta):
    """choose() returns the pair (candidate, level) maximizing sum_k w_k G_k / C."""
    surrogate = nested_design(meta, 24, 6, seed=0)
    surrogate.fit()
    x_cand = np.column_stack([meta.sample_sections(12, seed=7), attitudes(12, seed=3)])
    missing = np.ones((12, 2), dtype=bool)
    x_int = integration_cloud(x_cand[:2], 8, seed=1)
    weights = np.array([100.0, 1.0, 5.0, 2.0, 0.5, 0.1])
    choice = choose(surrogate, weights, x_cand, missing, x_int, remaining_budget=np.inf)
    gains = sum(w * batched_imse_gains(m, x_cand, x_int, missing)
                for w, m in zip(weights, surrogate.models))
    ratios = gains / np.array([1.0, 11.0])
    index, level = np.unravel_index(np.argmax(ratios), ratios.shape)
    assert (choice["index"], choice["level"]) == (index, level + 1)
    # nothing affordable at the HF: only the LF can be chosen
    assert choose(surrogate, weights, x_cand, missing, x_int, remaining_budget=5.0)["level"] == 1


def test_mean_only_prediction_and_light_copy(meta):
    """predict() (mean without the variance, latent_mean) = the mean of predict_batch; the
    light copy for the VPP worker processes predicts the same means without a Cholesky factor."""
    surrogate = nested_design(meta, 24, 6, seed=0)
    surrogate.n_threads = 4
    surrogate.fit()
    x = np.column_stack([meta.sample_sections(10, seed=5), attitudes(10, seed=2)])
    reference = np.column_stack([m.predict_batch(x, include_noise=False)[0]
                                 for m in surrogate.models])
    assert np.allclose(surrogate.predict(x)[:, :6], reference, rtol=0.0, atol=1e-12)
    light = surrogate.mean_only_copy()
    assert all(m.l_chol is None for m in light.models)
    assert all(m.l_chol is not None for m in surrogate.models)
    assert np.allclose(light.predict(x), surrogate.predict(x), rtol=0.0, atol=1e-12)


def test_kriging_believer_placeholder(meta):
    """Conditioning on a point with its predicted mean as value (the batch choice of
    vpp_optimize) leaves the means unchanged and removes the variance at that point."""
    surrogate = nested_design(meta, 24, 6, seed=0)
    surrogate.fit()
    x = np.column_stack([meta.sample_sections(6, seed=9), attitudes(6, seed=4)])
    means, variances = surrogate.predict(x), surrogate.variances(x)
    sizes = surrogate.sizes
    for level in (1, 2):
        surrogate.add(level, x[0], surrogate.predict(x[:1], level=level))
    surrogate.fit(refit=False)
    assert np.allclose(surrogate.predict(x), means, rtol=0.0, atol=1e-6)
    after = surrogate.variances(x)
    assert np.all(after[0] < 1e-3 * variances[0] + 1e-12)
    assert np.all(after <= variances * (1 + 1e-9) + 1e-15)
    surrogate.truncate(sizes)
    assert surrogate.sizes == sizes


def test_fsi_correction_reproduces_the_production_vpp():
    """The rigid part of the production FSI RS (intercept of its V^2 law) plus FSICorrection
    gives the speed of the production VPP (2026-10-22: 0.0005 kt at 45/16)."""
    from src.vpp_coupling import FSICorrection  # noqa: PLC0415
    fsi = FSICorrection()
    flexible = IntensSYVPP(foil="CFoilFSI", with_hvol=False, fsi=fsi)

    def rigid(points):
        attitude = points[:, -3:]
        n = len(fsi.speeds)
        x = np.column_stack([np.repeat(attitude, n, axis=0), np.tile(fsi.speeds, len(attitude))])
        out = fsi.rs(x).reshape(len(attitude), n, -1)
        return (out.mean(axis=1) - fsi.slopes(attitude) * np.mean(fsi.speeds ** 2))[:, :7]

    flexible.reset_foil()
    production = flexible.solve(45.0, 16.0)
    flexible.set_foil(rigid, np.zeros(0))
    ours = flexible.solve(45.0, 16.0, guess=production["unknowns"])
    assert production["success"] and ours["success"]
    assert abs(ours["speed"] - production["speed"]) < 0.01
    # the RS of the adapter declares the speed, and the deformation is not zero at 16 kt TWS
    assert flexible.foil.surrogate_model.input_names[-1] == "Speed_kts"
    assert flexible.foil.surrogate_model.out("Bending") > 0.01
