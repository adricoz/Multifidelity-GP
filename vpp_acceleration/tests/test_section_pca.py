"""
Tests of src/section_pca.py: Kulfan geometry as a matrix product, the C-foil family, the PCA with
its exact decoding, the score space of the surrogate.
"""
import numpy as np
import pytest
from scipy.stats import qmc

from conftest import REPO_DIR
from src.section_pca import (CFoilFamily, SectionPCA, SmoothKulfanFamily, elevation_matrix,
                             feasibility_function, kulfan_dict, score_bounds,
                             second_difference_matrix, section_coordinates, wiggliness)

STUDY_CONFIG = REPO_DIR / "studies" / "cfoil_intens_kulfan_trim_2" / "config_cfoil_trim.json"
TE = 0.00197


@pytest.fixture(scope="module")
def family():
    return CFoilFamily.from_study_config(STUDY_CONFIG)


@pytest.fixture(scope="module")
def pca(family):
    return SectionPCA(family.te_thickness).fit(family.sample(4000, seed=0))


def random_weights(n_sections, seed):
    """Sections with 16 free weights (box of Dim-Reduction/experiments.ipynb)."""
    lower = [0.06, 0.12, 0.12, 0.10, 0.05, 0.02, 0.01, 0.005,
             -0.20, -0.15, -0.12, -0.10, -0.08, -0.05, -0.02, -0.01]
    upper = [0.25, 0.28, 0.28, 0.25, 0.20, 0.15, 0.12, 0.08,
             -0.055, -0.02, 0.02, 0.04, 0.06, 0.08, 0.06, 0.03]
    return qmc.scale(qmc.LatinHypercube(d=16, seed=seed).random(n_sections), lower, upper)


# --- Kulfan geometry ---
def test_coordinates_match_aerosandbox():
    """Eq. (1) gives the same surfaces as aerosandbox (same stations)."""
    airfoil_families = pytest.importorskip("aerosandbox.geometry.airfoil.airfoil_families")
    for weights in random_weights(5, seed=1):
        coordinates = airfoil_families.get_kulfan_coordinates(
            upper_weights=weights[:8], lower_weights=weights[8:], leading_edge_weight=0.0,
            TE_thickness=TE, n_points_per_side=101)
        # Selig order: trailing edge -> upper surface -> leading edge -> lower surface
        upper, lower = coordinates[:101][::-1], coordinates[100:]
        np.testing.assert_allclose(upper[:, 0], lower[:, 0], atol=1e-14)
        ours = section_coordinates(weights, upper[:, 0], TE)[0]
        np.testing.assert_allclose(ours, np.concatenate([upper[:, 1], lower[:, 1]]), atol=1e-12)


def test_coordinates_are_affine():
    """y(a w1 + (1 - a) w2) = a y(w1) + (1 - a) y(w2)."""
    w1, w2 = random_weights(2, seed=2)
    stations = np.linspace(0.0, 1.0, 50)
    mixed = section_coordinates(0.3 * w1 + 0.7 * w2, stations, TE)
    expected = 0.3 * section_coordinates(w1, stations, TE) \
        + 0.7 * section_coordinates(w2, stations, TE)
    np.testing.assert_allclose(mixed, expected, atol=1e-15)


def test_elevation_keeps_the_surface():
    """4 weights and their elevation to 8 weights describe the same surface."""
    weights4 = np.array([0.2, 0.25, 0.15, 0.1, -0.1, -0.05, 0.0, 0.02])
    elevation = elevation_matrix(4, 8)
    weights8 = np.concatenate([elevation @ weights4[:4], elevation @ weights4[4:]])
    stations = np.linspace(0.0, 1.0, 77)
    np.testing.assert_allclose(section_coordinates(weights4, stations, TE),
                               section_coordinates(weights8, stations, TE), atol=1e-15)


def test_kulfan_dict_batch_format():
    """NeuralFoil batch format: weights (8, m), one column per section."""
    weights = random_weights(3, seed=3)
    kulfan = kulfan_dict(weights, TE)
    assert kulfan["upper_weights"].shape == (8, 3)
    np.testing.assert_array_equal(kulfan["lower_weights"][:, 1], weights[1, 8:])


# --- the family ---
def test_family_round_trip(family):
    """parameters -> weights -> parameters is exact, with a zero residual."""
    parameters = family.lower + np.random.default_rng(0).random((20, 8)) \
        * (family.upper - family.lower)
    back, residual = family.weights_to_parameters(family.parameters_to_weights(parameters))
    np.testing.assert_allclose(back, parameters, atol=1e-13)
    assert residual.max() < 1e-13


def test_family_membership(family):
    """The reference belongs to the family; a thinner section, a section outside the box and a
    section of another form do not."""
    reference = family.reference.copy()
    assert family.contains(family.reference_weights())[0]
    thin = reference.copy()
    thin[:4] *= 0.8                                                  # thinner than 12.25 %
    outside = reference.copy()
    outside[5] = family.upper[5] + 0.05                             # s_1 above its bound
    other_form = random_weights(1, seed=4)                          # 16 free weights
    candidates = np.vstack([family.parameters_to_weights(thin),
                            family.parameters_to_weights(outside), other_form])
    assert not family.contains(candidates).any()


def test_family_sample(family):
    """Every sampled section belongs to the family."""
    weights = family.sample(500, seed=5)
    assert weights.shape == (500, 16)
    assert family.contains(weights).all()


# --- the smooth 16-weight family ---
@pytest.fixture(scope="module")
def smooth(family):
    return SmoothKulfanFamily.around(family, n_sections=4000)


def test_level_slope_basis(smooth):
    """Q is orthonormal and has no second difference: D2 Q = 0."""
    np.testing.assert_allclose(smooth.basis.T @ smooth.basis, np.eye(2), atol=1e-14)
    np.testing.assert_allclose(second_difference_matrix(8) @ smooth.basis, 0.0, atol=1e-14)


def test_smooth_split_join(smooth):
    """weights -> (level / slope, second differences) -> weights is exact."""
    weights = random_weights(20, seed=12)
    np.testing.assert_allclose(smooth.join(*smooth.split(weights)), weights, atol=1e-14)


def test_smooth_sample(smooth):
    """The sample is in the family, its wiggliness is at most 2 x the reference, and the
    reference belongs to the family."""
    weights = smooth.sample(500, seed=13)
    assert smooth.contains(weights).all()
    assert wiggliness(weights).max() <= 2.0 * wiggliness(smooth.reference_weights())[0] + 1e-12
    assert smooth.contains(smooth.reference_weights())[0]


def test_smooth_family_holds_the_smooth_cfoil_sections(family, smooth):
    """The sections of the 8-parameter family that are smooth enough belong to the 16-weight
    family (its level / slope box covers the 8-parameter family)."""
    weights = family.sample(500, seed=14)
    smooth_enough = wiggliness(weights) <= smooth.max_wiggliness
    assert smooth_enough.sum() > 50
    assert smooth.contains(weights[smooth_enough]).all()


def test_smooth_family_has_rank_16(smooth):
    """16 free weights: rank 16, exact round trip with every mode."""
    pca16 = SectionPCA(smooth.te_thickness).fit(smooth.sample(3000, seed=15))
    assert pca16.rank == 16
    test = smooth.sample(50, seed=16)
    np.testing.assert_allclose(pca16.reconstruct(test), test, atol=1e-12)


# --- the PCA ---
def test_family_has_rank_8(pca):
    """4 thickness weights + 4 camber weights: 8 modes, whatever the 16 weights."""
    assert pca.rank == 8


def test_weight_modes_are_exact(pca):
    """v_j = A m_j: the decoding Eq. (3) is exact."""
    np.testing.assert_allclose(pca.matrix @ pca.weight_modes, pca.modes, atol=1e-12)


def test_round_trip_with_every_mode(family, pca):
    """With every mode, decode(encode(w)) = w for sections of the family (test sample)."""
    weights = family.sample(200, seed=6)
    np.testing.assert_allclose(pca.reconstruct(weights, pca.rank), weights, atol=1e-12)


def test_round_trip_free_weights():
    """16 free weights: rank 16, the round trip is exact."""
    weights = random_weights(2000, seed=7)
    pca16 = SectionPCA(TE).fit(weights)
    assert pca16.rank == 16
    test = random_weights(50, seed=8)
    np.testing.assert_allclose(pca16.reconstruct(test), test, atol=1e-11)


def test_scores_are_whitened(family, pca):
    """On the training sample the scores have zero mean, unit variance, no correlation."""
    scores = pca.encode(family.sample(4000, seed=0))
    np.testing.assert_allclose(scores.mean(axis=0), 0.0, atol=1e-10)
    np.testing.assert_allclose(np.cov(scores.T), np.eye(pca.rank), atol=1e-10)


def test_error_decreases_with_modes(family, pca):
    """The geometric error of the projection decreases with the number of modes."""
    weights = family.sample(300, seed=9)
    errors = [pca.geometric_error(weights, k).max() for k in range(1, pca.rank + 1)]
    assert np.all(np.diff(errors) < 0)
    assert errors[-1] < 1e-12


def test_decode_truncated_is_a_projection(family, pca):
    """With k < r modes, the decoded section is the orthogonal projection of its coordinates on
    the k first modes."""
    weights = family.sample(10, seed=10)
    k = 5
    centred = pca.coordinates(weights) - pca.mean_coordinates
    projection = pca.mean_coordinates + centred @ pca.modes[:, :k] @ pca.modes[:, :k].T
    np.testing.assert_allclose(pca.coordinates(pca.reconstruct(weights, k)), projection,
                               atol=1e-13)


def test_save_and_load(tmp_path, family, pca):
    """The JSON file gives back the same encoding and decoding."""
    pca.n_modes = 6
    loaded = SectionPCA.load(pca.save(tmp_path / "pca.json"))
    weights = family.sample(20, seed=11)
    assert loaded.n_modes == 6
    np.testing.assert_allclose(loaded.encode(weights), pca.encode(weights), atol=1e-14)
    np.testing.assert_allclose(loaded.reconstruct(weights), pca.reconstruct(weights),
                               atol=1e-14)
    pca.n_modes = pca.rank


# --- score space ---
def test_score_space(family, pca):
    """Score box and feasibility: the reference is feasible, a far point is not."""
    bounds = score_bounds(pca, family, n_sections=2000)
    assert bounds.shape == (8, 2)
    feasible = feasibility_function(pca, family)
    reference_scores = pca.encode(family.reference_weights())
    assert np.all((reference_scores >= bounds[:, 0]) & (reference_scores <= bounds[:, 1]))
    assert feasible(reference_scores)[0]
    assert not feasible(np.full((1, 8), 10.0))[0]


# --- the Dim-Reduction box, the box around the current section, the Ixx constraint ---
def test_notebook_bounds_are_read():
    """The 16 bounds of Dim-Reduction/experiments.ipynb, upper weights first."""
    from src.section_pca import DIM_REDUCTION_NOTEBOOK, notebook_bounds  # noqa: PLC0415
    if not DIM_REDUCTION_NOTEBOOK.exists():
        pytest.skip("Dim-Reduction notebook not available")
    lower, upper = notebook_bounds()
    assert lower.shape == upper.shape == (16,) and np.all(upper > lower)
    np.testing.assert_allclose(lower[:3], [0.06, 0.12, 0.12])


def test_ixx_scales_as_the_cube_of_the_thickness():
    """A symmetric section (upper = -lower, no trailing edge) scaled by a in thickness has an
    Ixx a^3 times larger: Ixx = int tau^3 / 12 when the camber line is the chord."""
    from src.section_pca import section_ixx  # noqa: PLC0415
    upper = np.array([0.17, 0.16, 0.15, 0.14, 0.13, 0.12, 0.11, 0.10])
    weights = np.concatenate([upper, -upper])
    base = section_ixx(weights, 0.0)[0]
    np.testing.assert_allclose(section_ixx(1.5 * weights, 0.0)[0], 1.5 ** 3 * base, rtol=1e-10)
    assert base > 0.0


def test_box_centred_on_a_section(family):
    """KulfanBoxFamily.centered: bounds symmetric around the section, which belongs to it."""
    from src.section_pca import KulfanBoxFamily  # noqa: PLC0415
    reference = family.reference_weights()
    widths = np.full(16, 0.05)
    box = KulfanBoxFamily.centered(reference, widths, TE)
    np.testing.assert_allclose((box.lower + box.upper) / 2.0, reference)
    assert box.contains(reference)[0]
    assert box.contains(box.sample(50, seed=3)).all()
