"""XFOIL transition fallback of src/foil_codes (rule of 2026-11-12): an XFOIL-polar run that fails at
xtr 3 % is run again at XTR_FALLBACK; other failures and other levels are not."""
import numpy as np

from src import foil_codes
from src.foil_codes import FoilCodes, XTR_FALLBACK


class _Backend:
    def __init__(self, polars):
        self.polars = polars


class _Fallback:
    calls = 0

    def evaluate(self, weights, attitude, level, section=None):
        _Fallback.calls += 1
        return np.arange(7.0), {"error": "", "xtr": list(XTR_FALLBACK), "time_s": 1.0}


def _codes(xtr, polars, error):
    """A FoilCodes whose run fails with `error`, without the real codes."""
    codes = object.__new__(FoilCodes)
    codes.flow = {"xtr": list(xtr)}
    codes._fallback = _Fallback()                   # noqa: SLF001

    class _Simulator:
        backends = [_Backend(polars)] * 2
        _core_errors = ()
    codes.simulator = _Simulator()
    codes.section = lambda weights: type("S", (), {"name": "k0"})()

    def _fail(*_args, **_kwargs):                   # the run fails at its first step (write_xf)
        raise ValueError(error)
    return codes, _fail


def _evaluate(codes, fail, monkeypatch):
    import pipelines.bdtoolbox_foil.geometry as geometry  # noqa: PLC0415
    monkeypatch.setattr(geometry, "write_xf", fail)
    return codes.evaluate(np.zeros(16), (0.0, 0.0, 0.0), 2)


def test_xfoil_failure_at_3_percent_is_run_again_at_5(monkeypatch):
    _Fallback.calls = 0
    codes, fail = _codes([0.03, 0.03], "xfoil", "PolarError: XFOIL polar(s) failed")
    y, info = _evaluate(codes, fail, monkeypatch)
    assert _Fallback.calls == 1 and info["fallback"] and info["xtr"] == XTR_FALLBACK
    assert info["error"] == "" and "XFOIL" in info["error_first"] and np.isfinite(y).all()


def test_no_fallback_for_other_errors_levels_or_transitions(monkeypatch):
    _Fallback.calls = 0
    for xtr, polars, error in (([0.03, 0.03], "xfoil", "AVL output missing"),     # not XFOIL
                               ([0.03, 0.03], "neuralfoil", "XFOIL"),            # NeuralFoil level
                               ([0.05, 0.05], "xfoil", "PolarError: XFOIL")):     # already at 5 %
        codes, fail = _codes(xtr, polars, error)
        y, info = _evaluate(codes, fail, monkeypatch)
        assert np.isnan(y).all() and info["error"] and info["xtr"] == xtr and "fallback" not in info
    assert _Fallback.calls == 0
    assert foil_codes.XTR_FALLBACK in ([0.05, 0.05], None)


def test_fallback_off(monkeypatch):
    """run_real_vpp --no-xtr-fallback (XTR_FALLBACK None): the XFOIL failure stays a failure."""
    _Fallback.calls = 0
    monkeypatch.setattr(foil_codes, "XTR_FALLBACK", None)
    codes, fail = _codes([0.03, 0.03], "xfoil", "PolarError: XFOIL polar(s) failed")
    y, info = _evaluate(codes, fail, monkeypatch)
    assert np.isnan(y).all() and info["error"] and _Fallback.calls == 0
