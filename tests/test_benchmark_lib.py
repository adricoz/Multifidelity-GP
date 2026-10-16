"""Unit tests of benchmarks/bench_lib.py (the SMT / BoTorch parts are skipped outside the
benchmark environment, see requirements-benchmark.txt)."""
import sys
from pathlib import Path

import numpy as np
import pytest
from conftest import load_module

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
import bench_lib as bl  # noqa: E402  pylint: disable=wrong-import-position


def test_hartmann_problem_matches_the_example_functions():
    """Same functions as example/hartmann_6d/Hartmann6d.py (Sacher Eqs. 30-32)."""
    example = load_module("example/hartmann_6d/Hartmann6d.py", "h6_bench")
    x = np.random.default_rng(0).random((5, 6))
    assert bl.hartmann6(bl.H6_XOPT)[0] == pytest.approx(bl.H6_MIN, abs=1e-6)
    np.testing.assert_allclose(bl.hartmann6(x), [example.Hartmann6D(xi) for xi in x])
    np.testing.assert_allclose(bl.hartmann_level(x, 1, 0.05),
                               [example.evaluate_fidelity(xi, 1, 2) for xi in x])


def test_doe_and_costs():
    problem = bl.HartmannMF()
    datasets = problem.doe([8, 4], seed=1)
    assert [len(y) for _, y in datasets] == [8, 4]
    np.testing.assert_allclose(datasets[1][1], bl.hartmann6(datasets[1][0]))
    hf_only = problem.doe([5], seed=1)
    np.testing.assert_allclose(hf_only[0][1], bl.hartmann6(hf_only[0][0]))
    assert bl.fidelity_values([1.0, 10.0]) == [0.0, 1.0]
    fids = bl.fidelity_values([1.0, 100.0, 1000.0])
    np.testing.assert_allclose(1.0 + 999.0 * np.array(fids), [1.0, 100.0, 1000.0])


@pytest.mark.parametrize("surrogate", [bl.MfegoSurrogate(True), bl.MfegoSurrogate(False),
                                       bl.SklearnSurrogate()], ids=lambda s: s.name)
def test_accuracy_metrics(surrogate):
    problem = bl.HartmannMF()
    datasets = problem.doe([16, 8], seed=0)
    x_test = np.random.default_rng(3).random((200, 6))
    result = bl.accuracy(surrogate, datasets, x_test, bl.hartmann6(x_test))
    assert np.isfinite(result["rmse_rel"]) and 0.0 <= result["coverage_95"] <= 1.0
    assert len(result["rhos"]) == (1 if surrogate.name == "mfego MF" else 0)


def test_mfego_optimization_loop_respects_the_budget():
    problem = bl.HartmannMF()
    history = bl.run_mfego(problem, problem.doe([10, 5], seed=0), budget=80, seed=0)
    assert history.cost[0] == 60.0 and history.cost[-1] >= 80.0
    assert all(c2 >= c1 for c1, c2 in zip(history.cost, history.cost[1:]))
    assert np.isfinite(history.recommendation_error) and history.recommendation_error >= -1e-9
    assert set(history.levels[1:]) <= {1, 2}


def test_nested_doe_and_cokriging_surrogate():
    """[CK] nested design (HF points among the LF points) and the CoKriging wrapper."""
    problem = bl.HartmannMF()
    (x_lf, y_lf), (x_hf, y_hf) = bl.nested_doe(problem, [16, 6], seed=0)
    assert len(y_lf) == 16 and len(y_hf) == 6
    assert all(np.min(np.linalg.norm(x_lf - x, axis=1)) == 0.0 for x in x_hf)
    np.testing.assert_allclose(y_hf, bl.hartmann6(x_hf))
    surrogate = bl.CoKrigingSurrogate(seed=0).fit([(x_lf, y_lf), (x_hf, y_hf)])
    mean, var = surrogate.predict(x_hf)
    np.testing.assert_allclose(mean, y_hf, atol=1e-3 * np.std(y_hf))
    assert len(surrogate.rhos) == 1 and np.all(var >= 0)


def test_run_mfego_can_return_the_model():
    """[CK] return_model=True gives the final model and the (non-nested) data of the run."""
    problem = bl.HartmannMF()
    history, model, data = bl.run_mfego(problem, problem.doe([10, 5], seed=0), budget=70,
                                        seed=0, return_model=True, use_map=True)
    assert model.is_fitted() and len(data.get_training_data(2)[1]) >= 5
    assert history.cost[-1] >= 70.0


def test_emukit_and_cokriging_are_the_same_cokriging():
    """[CK] emukit hyperparameters put in mfego CoKriging -> same predictions ([LG] Eq. 3.11)."""
    pytest.importorskip("emukit")
    datasets = bl.nested_doe(bl.HartmannMF(), [16, 6], seed=0)
    surrogate = bl.EmukitAR1Surrogate(seed=0, n_restarts=1).fit(datasets)
    x_test = np.random.default_rng(1).random((50, 6))
    mean_gpy, var_gpy = surrogate.predict(x_test)
    mean_ck, var_ck = surrogate.to_cokriging().predict_batch(x_test)
    np.testing.assert_allclose(mean_ck, mean_gpy, rtol=1e-8, atol=1e-8 * np.std(mean_gpy))
    # GPy adds 1e-8 (standardized) to the noise in its inference only, not in its prediction
    np.testing.assert_allclose(var_ck, var_gpy, rtol=1e-4, atol=2e-8 * surrogate.y_std ** 2)


def test_smt_mfk_rho():
    pytest.importorskip("smt")
    datasets = bl.HartmannMF().doe([20, 10], seed=0)
    surrogate = bl.SmtSurrogate(multi_fidelity=True).fit(datasets)
    assert len(surrogate.rhos) == 1
    mean, var = surrogate.predict(np.random.default_rng(0).random((10, 6)))
    assert mean.shape == var.shape == (10,)


def test_botorch_surrogates():
    pytest.importorskip("botorch")
    datasets = bl.HartmannMF().doe([12, 6], seed=0)
    for multi_fidelity in (False, True):
        surrogate = bl.BotorchSurrogate(multi_fidelity=multi_fidelity).fit(datasets)
        mean, var = surrogate.predict(np.random.default_rng(0).random((10, 6)))
        assert mean.shape == var.shape == (10,) and np.all(var >= 0)
