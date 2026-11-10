"""
Tests of src/section_domain.py: the current section is admissible, the chord of the polytope is
exact, the hit-and-run sections are all admissible, a section with a sharp leading edge is not.
"""
import numpy as np
import pytest

from conftest import REPO_DIR
from src.section_domain import SectionDomain
from src.section_pca import CFoilFamily, section_ixx, section_modulus

STUDY_CONFIG = REPO_DIR / "studies" / "cfoil_intens_kulfan_trim_2" / "config_cfoil_trim.json"
TE = 0.00197


@pytest.fixture(scope="module")
def current():
    return np.asarray(CFoilFamily.from_study_config(STUDY_CONFIG).reference_weights(), float)


@pytest.fixture(scope="module")
def domain(current):
    return SectionDomain(current, TE, stations=np.linspace(0.0, 1.0, 301),
                         inflections=(0.467, 0.938))


def test_current_section_is_admissible(domain, current):
    assert domain.contains(current, 1e-9)[0]
    assert domain.ixx_v_min == pytest.approx(section_modulus(current[None], TE)[0])


def test_chord_ends_on_the_boundary(domain, current):
    """At both ends of the chord a linear constraint is active (margin 0), inside it none is broken."""
    directions = np.random.default_rng(0).standard_normal((5, len(current))) * 0.05
    points = np.tile(current, (5, 1))
    s_min, s_max = domain.chord(points, directions)
    for s in (s_min, s_max):
        margins = domain.margins(points + s[:, None] * directions)
        assert np.allclose(margins.min(axis=1), 0.0, atol=1e-12)
    assert np.all(domain.margins(points + 0.5 * (s_min + s_max)[:, None] * directions) >= -1e-12)


def test_hit_and_run_sections_are_admissible(domain):
    sample = domain.sample(200, seed=1, n_chains=20, burn=100, thin=5)
    assert sample.shape == (200, 16)
    assert domain.contains(sample).all()
    assert np.ptp(sample, axis=0).min() > 0          # the chains moved in every weight


def test_sharp_leading_edge_is_rejected(domain, current):
    sharp = current.copy()
    sharp[0] *= 0.2                                   # upper leading-edge radius / 25
    assert domain.violations(sharp)["leading_edge"][0]
    assert not domain.contains(sharp)[0]


# --- Ixx constraint (2026-11-10: Ixx >= 0.95 x current together with Ixx / v >= 0.9 x current) ---
@pytest.fixture(scope="module")
def stiff_domain(current):
    return SectionDomain(current, TE, stations=np.linspace(0.0, 1.0, 301),
                         inflections=(0.467, 0.938), ixx_v_factor=0.9, ixx_factor=0.95)


def test_ixx_factor_zero_is_the_previous_domain(domain, current):
    """Default ixx_factor 0: no Ixx limit, same membership as before on random sections."""
    assert domain.ixx_min == 0.0
    sections = domain.sample(100, seed=3, n_chains=20, burn=50, thin=2)
    shrunk = sections * np.r_[np.full(8, 0.8), np.full(8, 0.8)]     # thinner: Ixx / v breaks
    assert not domain.violations(shrunk)["ixx"].any()


def test_ixx_limit_and_violation(stiff_domain, current):
    assert stiff_domain.contains(current, 1e-9)[0]
    assert stiff_domain.ixx_min == pytest.approx(0.95 * section_ixx(current[None], TE)[0])
    thin = current * 0.97                          # Ixx ~ t^3: -8.7 % < -5 %, Ixx / v ~ t^2: -5.9 %
    broken = stiff_domain.violations(thin)
    assert broken["ixx"][0] and not broken["ixx_v"][0]
    assert not stiff_domain.contains(thin)[0]


def test_hit_and_run_respects_ixx(stiff_domain):
    sample = stiff_domain.sample(200, seed=2, n_chains=20, burn=100, thin=5)
    assert stiff_domain.contains(sample).all()
    assert np.all(section_ixx(sample, TE) >= stiff_domain.ixx_min * (1 - 1e-9))


def test_to_dict_round_trip(stiff_domain):
    d = stiff_domain.to_dict()
    again = SectionDomain(np.array(d["reference"]), d["te_thickness"], d["box_scale"], d["band"],
                          d["le_factor"], inflections=d["inflections"],
                          ixx_v_factor=d["ixx_v_factor"], ixx_factor=d["ixx_factor"])
    assert again.ixx_min == pytest.approx(stiff_domain.ixx_min)
    assert again.ixx_v_min == pytest.approx(stiff_domain.ixx_v_min)
