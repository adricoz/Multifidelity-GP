"""Unit tests of the example simulators (Hartmann 6D, hydrofoil)."""
import numpy as np
import pytest
from conftest import load_module

X_OPT = np.array([0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573])


@pytest.fixture(scope="module")
def hartmann():
    return load_module("example/hartmann_6d/Hartmann6d.py", "hartmann_example")


def test_hartmann_standard_minimum(hartmann):
    """[FIX-E1] standard constants (Sacher Eq. 30): f(x*) = -3.32237 (was -3.39572)."""
    assert hartmann.Hartmann6D(X_OPT) == pytest.approx(-3.32236801141551, abs=1e-6)
    rng = np.random.default_rng(0)
    assert min(hartmann.Hartmann6D(x) for x in rng.random((2000, 6))) > -3.32237


def test_hartmann_multifidelity_sequence(hartmann):
    """Eqs. 31-32: U_{k+1} = (f^2 / U_k + U_k) / 2, U_0 = -5, shift delta / k."""
    x = np.random.default_rng(1).random(6)
    f_shifted = hartmann.Hartmann6D(x + 0.05)
    u_1 = 0.5 * (f_shifted ** 2 / -5.0 - 5.0)
    assert hartmann.evaluate_fidelity(x, 1, 2) == pytest.approx(u_1)
    assert hartmann.evaluate_fidelity(x, 2, 2) == pytest.approx(hartmann.Hartmann6D(x))
    with pytest.raises(ValueError):
        hartmann.evaluate_fidelity(x, 3, 2)


@pytest.fixture(scope="module")
def foil():
    pytest.importorskip("neuralfoil")
    pytest.importorskip("aerosandbox")
    return load_module("example/hydrofoil_optim/optim_neuralfoil.py", "foil_example")


def test_hydrofoil_level_to_model_mapping(foil, monkeypatch):
    """[FIX-E2] level l < L uses the l-th NeuralFoil model (was always "xxsmall"); the
    root-finding of alpha(Cl = 1) always uses "xxxlarge" (deliberate design choice)."""
    calls = []

    def fake_aero(airfoil, alpha, Re, model_size, n_crit, xtr_upper, xtr_lower):
        calls.append(model_size)
        return {"CL": 0.1 * alpha + 0.5, "CD": 0.01}

    monkeypatch.setattr(foil.nf, "get_aero_from_airfoil", fake_aero)
    for level, expected in ((1, "xxsmall"), (2, "xsmall"), (3, "xxxlarge")):
        calls.clear()
        cd, cl, alpha = foil.objective_function(None, target_cl=1.0, level=level, L=3)
        assert calls[-1] == expected and set(calls[:-1]) == {"xxxlarge"}
        assert cd == 0.01 and alpha == pytest.approx(5.0)


def test_hydrofoil_failure_returns_nan(foil, monkeypatch):
    """[FIX-E2/R4] a failed computation returns NaN (was a 1e6 penalty fed to the GP)."""
    def failing(*args, **kwargs):
        raise RuntimeError("solver failure")

    monkeypatch.setattr(foil.nf, "get_aero_from_airfoil", failing)
    cd, _, _ = foil.objective_function(None, target_cl=1.0, level=1, L=2)
    assert np.isnan(cd)


@pytest.mark.slow
def test_hydrofoil_real_evaluation(foil):
    """Real NeuralFoil evaluation of a nominal foil (camber 5.5 %, thickness 12.5 %)."""
    import aerosandbox as asb  # pylint: disable=import-outside-toplevel
    coords = foil.generate_continuous_naca4(0.055, 0.3, 0.125, n_points=100)
    airfoil = asb.Airfoil(name="nominal", coordinates=coords)
    for level in (1, 2):
        cd, cl, alpha = foil.objective_function(airfoil, target_cl=1.0, level=level, L=2)
        assert 0.005 < cd < 0.05 and -5.0 < alpha < 15.0
    assert cl == pytest.approx(1.0, abs=1e-6)
