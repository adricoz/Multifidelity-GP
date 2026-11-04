"""
Tests of the bdfoil_v1 backend of pipelines/bdtoolbox_foil (since 2026-10-27, solvers.
BdfoilV1Backend): the case it builds (bdfoil_v1 validates it, no executable run), the mapping of
the bdfoil_v1 outputs to the names of the pipeline, the stall check, the failures; and one cheap
real evaluation per level on the C-foil (marked slow: NP-LLT x NeuralFoil, NP-LLT x XFOIL,
AVL x XFOIL, about 1.5 min on one process, numba compilation included).
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# pylint: disable=wrong-import-position
from pipelines.bdtoolbox_foil import bridge  # noqa: E402
from pipelines.bdtoolbox_foil.geometry import Section, kulfan_coordinates, write_xf  # noqa: E402
from pipelines.bdtoolbox_foil.planform import CFoilArc, write_planform_csv  # noqa: E402

PATHS = bridge.resolve_paths()
if not (PATHS.bdfoil_v1_root / "src" / "bdfoil_v1").is_dir():
    pytest.skip("bdfoil_v1 clone not found", allow_module_level=True)
bridge.setup(PATHS)
from pipelines.bdtoolbox_foil import solvers  # noqa: E402
from pipelines.bdtoolbox_foil.solvers import BdfoilV1Backend, SolverFailure  # noqa: E402

FLOW = {"re": 5e6, "n_crit": 1.0, "xtr": [0.1, 0.1], "alpha": [-12.0, 12.0, 0.5]}
KULFAN = {"upper_weights": np.array([0.17, 0.2, 0.18, 0.2, 0.17, 0.2, 0.18, 0.2]),
          "lower_weights": -np.array([0.12, 0.1, 0.12, 0.1, 0.12, 0.1, 0.12, 0.1]),
          "leading_edge_weight": 0.0, "TE_thickness": 0.00197}
C_FOIL = CFoilArc(radius=9.8, span=3.25, chord=0.70, cant_geom=20.168, tip_side=-1, n_stations=41)


def planform_files(folder: Path) -> Path:
    """The C-foil planform CSV and its section file in folder."""
    section = Section(name="test_section", coordinates=kulfan_coordinates(KULFAN, n_points=121),
                      parameters={}, thickness=0.12, kulfan=KULFAN)
    xf = write_xf(section, folder)
    return write_planform_csv(C_FOIL.stations(), xf.name, folder / "planform.csv")


def test_the_case_carries_the_flow_and_validates(tmp_path):
    """The bdfoil_v1 case: polars from the flow, the reference speed, the choices of 2026-10-27
    (no wave drag, Cpmin percentile 20); bdfoil_v1 accepts it (Session, no executable)."""
    backend = BdfoilV1Backend("avl", FLOW, attitude={"cant": 3.0, "yaw": 2.0}, reference_chord=0.7,
                              free_surface_height=0.654)
    nu = bridge.bdfoil_v1().fluid.seawater(15.0).nu
    assert backend.speed == pytest.approx(5e6 * nu / 0.7)
    assert 8.0 < backend.speed < 9.0
    case = backend.case
    assert case["polars"]["provider"] == "xfoil"
    assert case["polars"]["alpha"] == [-12.0, 12.0, 0.5]
    assert (case["polars"]["ncrit"], case["polars"]["xtr"]) == (1.0, [0.1, 0.1])
    assert case["post"] == {"cp_min_percentile": 20.0, "wave_drag": False}
    assert case["boundary"] == {"zsym": 1, "free_surface_height": 0.654}
    assert case["attitudes"]["list"][0]["cant"] == 3.0
    session = backend.session(planform_files(tmp_path), tmp_path)
    assert session.backend == "avl"
    assert len(session.planform.x) == 41
    npllt = BdfoilV1Backend("npllt", FLOW, reference_chord=0.7, polars="neuralfoil")
    assert npllt.session(planform_files(tmp_path), tmp_path).backend == "npllt"


def test_invalid_options_are_rejected():
    with pytest.raises(ValueError, match="solver"):
        BdfoilV1Backend("vlm", FLOW, reference_chord=0.7)
    with pytest.raises(ValueError, match="polars"):
        BdfoilV1Backend("avl", FLOW, reference_chord=0.7, polars="xxlarge")
    with pytest.raises(ValueError, match="reference_chord"):
        BdfoilV1Backend("avl", FLOW)
    with pytest.raises(ValueError, match="182"):
        BdfoilV1Backend("avl", FLOW, reference_chord=0.7, cache_dir="C:/" + "x" * 190)


def fake_result(solver: str, clamped=(), frozen=(), ok=True):
    """A bdfoil_v1 CaseResult-like object: 10 strips (and 10 NP-LLT panels) of width 0.3 m."""
    n = 10
    strips = pd.DataFrame({"width [m]": np.full(n, 0.3), "cl [-]": np.linspace(0.5, 0.1, n),
                           "cd [-]": np.full(n, 0.008), "y [m]": np.zeros(n),
                           "z [m]": -np.arange(n) * 0.3, "chord [m]": np.full(n, 0.7),
                           "Re [-]": np.full(n, 5e6), "Cpmin [-]": np.full(n, -1.0),
                           "cl_clamped": np.isin(np.arange(n), clamped)})
    panels = pd.DataFrame({"s [m]": np.arange(n) * 0.3 + 0.15, "alpha [deg]": np.full(n, 2.0),
                           "alpha_clamped": np.isin(np.arange(n), frozen)})
    values = {"Cx [-]": -0.018, "Cy [-]": -0.45, "Cz [-]": 0.28, "Cmx [-]": -0.8, "Cmy [-]": 0.11,
              "Cmz [-]": 0.11, "Cdvisc [-]": 0.0083, "Cdi [-]": 0.0089,
              "Cd trefftz [-]": 0.0088 if solver == "avl" else np.nan, "Cpmin [-]": -1.07,
              "Ref Surf [m^2]": 2.275, "ok": True}
    si = {"Re min [-]": 4.9e6, "Re max [-]": 5.1e6, "n cl clamped": len(clamped),
          "converged": True, "n iterations": 4}
    return SimpleNamespace(ok=ok, error=None if ok else "boom", values=values, si=si,
                           strips=strips, panels=panels if solver == "npllt" else None)


def evaluate_with(monkeypatch, backend, result=None, error=None):
    class FakeSession:  # pylint: disable=too-few-public-methods
        def evaluate(self, attitude):
            assert attitude.speed == pytest.approx(backend.speed)
            if error is not None:
                raise error
            return result
    monkeypatch.setattr(backend, "session", lambda csv, folder: FakeSession())
    return backend.evaluate_planform(Path("planform.csv"), Path("."), with_strips=True,
                                     attitude={"yaw": 4.0})


def test_outputs_get_the_names_of_the_pipeline(monkeypatch):
    avl = BdfoilV1Backend("avl", FLOW, reference_chord=0.7)
    out = evaluate_with(monkeypatch, avl, fake_result("avl"))
    assert out["Cdprofile [-]"] == 0.0083                 # viscous drag of bdfoil_v1
    assert out["Cdi [-]"] == 0.0088                       # AVL: Trefftz plane, as in Cx
    assert out["Cdi_nearfield [-]"] == 0.0089
    assert out["Cpmin [-]"] == -1.07 and out["Re min [-]"] == 4.9e6
    assert "ok" not in out and out["n_clamped"] == 0
    assert len(out["strips"]["cl"]) == 10 and "alpha_deg" not in out["strips"]
    npllt = BdfoilV1Backend("npllt", FLOW, reference_chord=0.7, polars="neuralfoil")
    out = evaluate_with(monkeypatch, npllt, fake_result("npllt"))
    assert out["Cdi [-]"] == 0.0089                       # NP-LLT: near field
    assert out["converged"] is True and out["strips"]["alpha_deg"] == [2.0] * 10


def test_stall_check_skips_the_root_zone(monkeypatch):
    """A clamped strip in the root zone of a wall image is counted but not a failure; one
    outside it (or a frozen NP-LLT panel outside it) makes the evaluation fail."""
    avl = BdfoilV1Backend("avl", FLOW, reference_chord=0.7, root_zone=0.1)
    out = evaluate_with(monkeypatch, avl, fake_result("avl", clamped=[0]))
    assert (out["n_clamped"], out["n_clamped_total"]) == (0, 1)
    with pytest.raises(SolverFailure, match="stall"):
        evaluate_with(monkeypatch, avl, fake_result("avl", clamped=[5]))
    npllt = BdfoilV1Backend("npllt", FLOW, reference_chord=0.7, root_zone=0.1)
    with pytest.raises(SolverFailure, match="stall"):
        evaluate_with(monkeypatch, npllt, fake_result("npllt", frozen=[6]))
    lenient = BdfoilV1Backend("avl", FLOW, reference_chord=0.7, fail_on_clamp=False)
    assert evaluate_with(monkeypatch, lenient, fake_result("avl", clamped=[5]))["n_clamped"] == 1


def test_failures_become_solver_failures(monkeypatch):
    """A bdfoil_v1 CoreError (e.g. a non-converged NP-LLT) or a failed case is a failed
    evaluation; a missing executable stops the run."""
    model = bridge.bdfoil_v1().model
    backend = BdfoilV1Backend("npllt", FLOW, reference_chord=0.7)
    with pytest.raises(SolverFailure, match="NplltConvergenceError"):
        evaluate_with(monkeypatch, backend, error=model.NplltConvergenceError("no Newton", {}))
    with pytest.raises(SolverFailure, match="boom"):
        evaluate_with(monkeypatch, backend, fake_result("npllt", ok=False))
    with pytest.raises(model.ExeError):
        evaluate_with(monkeypatch, backend, error=model.ExeError("avl missing"))
    # an AVL lift coupling loop (J8) that did not converge
    avl = BdfoilV1Backend("avl", FLOW, reference_chord=0.7)
    assert avl.case["solver"]["avl"]["coupling"] == "induced_angle"
    unconverged = fake_result("avl")
    unconverged.si["converged"] = False
    with pytest.raises(SolverFailure, match="not converged"):
        evaluate_with(monkeypatch, avl, unconverged)


def test_neuralfoil_confidence_threshold_of_the_process(monkeypatch):
    """neuralfoil_min_confidence sets bdfoil_v1's NeuralFoil threshold for this process and drops
    its in-memory NeuralFoil polars when it changes; it needs NeuralFoil polars."""
    from bdfoil_v1 import neuralfoil_polars  # pylint: disable=import-outside-toplevel
    monkeypatch.setattr(neuralfoil_polars, "MIN_CONFIDENCE", 0.90)
    monkeypatch.setattr(neuralfoil_polars, "_MEMO", {"old": "polar"})
    backend = BdfoilV1Backend("npllt", FLOW, reference_chord=0.7, polars="neuralfoil",
                              neuralfoil_min_confidence=0.0)
    evaluate_with(monkeypatch, backend, fake_result("npllt"))
    assert neuralfoil_polars.MIN_CONFIDENCE == 0.0 and neuralfoil_polars._MEMO == {}  # pylint: disable=protected-access
    default = BdfoilV1Backend("npllt", FLOW, reference_chord=0.7, polars="neuralfoil")
    assert default.neuralfoil_min_confidence is None
    with pytest.raises(ValueError, match="neuralfoil"):
        BdfoilV1Backend("avl", FLOW, reference_chord=0.7, neuralfoil_min_confidence=0.5)


@pytest.mark.slow
@pytest.mark.parametrize("solver, polars",[("npllt", "neuralfoil"), ("npllt", "xfoil"),
                                            ("avl", "xfoil")])
def test_one_real_evaluation_per_level(tmp_path, solver, polars):
    """One attitude of the C-foil (cant 3, rake 0.5, yaw 2 deg): finite coefficients, lift to
    port and up, local Re around 5e6, no stall."""
    if polars == "neuralfoil":
        pytest.importorskip("neuralfoil")
    if solver == "npllt":
        pytest.importorskip("numba")
    if polars == "xfoil" and not (PATHS.soft_dir / "xfoil.exe").is_file():
        pytest.skip("soft/xfoil.exe not found")
    backend = BdfoilV1Backend(solver, FLOW, attitude={"cant": 3.0, "rake": 0.5, "yaw": 2.0},
                              reference_chord=0.7, polars=polars, cache_dir=str(tmp_path / "pc"))
    out = backend.evaluate_planform(planform_files(tmp_path), tmp_path)
    assert all(np.isfinite(out[k]) for k in ("Cx [-]", "Cy [-]", "Cz [-]", "Cmx [-]", "Cmy [-]",
                                             "Cmz [-]", "Cdprofile [-]", "Cdi [-]"))
    assert out["Cx [-]"] < 0.0 and out["Cy [-]"] < 0.0 and out["Cz [-]"] > 0.0
    assert 4.5e6 < out["Re min [-]"] <= out["Re max [-]"] < 5.5e6
    assert out["n_clamped"] == 0
    if polars == "xfoil":
        assert -3.0 < out["Cpmin [-]"] < 0.0
