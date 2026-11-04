"""Tests of src/operating_profile.py (numpy and scipy only)."""
import json

import numpy as np
import pytest
from scipy.stats import truncnorm

from src.operating_profile import OperatingProfile, strata_nodes

PROFILE = {"name": "test", "nodes_per_dim": 5,
           "modes": [{"name": "upwind", "weight": 3.0,
                      "twa": {"mean": 45.0, "std": 4.0, "lower": 38.0, "upper": 52.0},
                      "tws": {"mean": 13.0, "std": 3.0, "lower": 8.0, "upper": 18.0}},
                     {"name": "downwind", "weight": 1.0,
                      "twa": {"mean": 135.0, "std": 7.0, "lower": 120.0, "upper": 148.0},
                      "tws": {"mean": 12.0, "std": 2.0, "lower": 8.0, "upper": 18.0}}]}


def test_nodes_are_weighted_and_inside_the_bounds():
    nodes = OperatingProfile("test", modes=PROFILE["modes"], nodes_per_dim=5).nodes()
    assert len(nodes) == 2 * 25
    assert sum(n["weight"] for n in nodes) == pytest.approx(1.0)
    upwind = [n for n in nodes if n["mode"] == "upwind"]
    assert sum(n["weight"] for n in upwind) == pytest.approx(0.75)        # weights 3 : 1
    assert all(38.0 <= n["twa"] <= 52.0 and 8.0 <= n["tws"] <= 18.0 for n in upwind)


def test_quadrature_of_the_mean_and_of_a_smooth_function():
    """The mid-probability nodes integrate the truncated normal law: its mean exactly for a
    symmetric law, a smooth function to about 1 %."""
    law = PROFILE["modes"][0]["twa"]
    assert np.mean(strata_nodes(law, 5)) == pytest.approx(45.0, abs=1e-9)
    a, b = (law["lower"] - law["mean"]) / law["std"], (law["upper"] - law["mean"]) / law["std"]
    exact = truncnorm.expect(lambda t: np.cos(np.radians(t)), args=(a, b), loc=law["mean"],
                             scale=law["std"])
    approx = np.mean(np.cos(np.radians(strata_nodes(law, 5))))
    assert approx == pytest.approx(exact, rel=1e-2)


def test_explicit_nodes_and_json(tmp_path):
    path = tmp_path / "logs.json"
    path.write_text(json.dumps({"name": "logs", "nodes": [
        {"twa": 45, "tws": 12, "weight": 2, "mode": "upwind"},
        {"twa": 140, "tws": 14, "weight": 6, "mode": "downwind"}]}), encoding="utf-8")
    nodes = OperatingProfile.from_json(path).nodes()
    assert [n["weight"] for n in nodes] == pytest.approx([0.25, 0.75])
    with pytest.raises(ValueError):
        OperatingProfile("bad")
