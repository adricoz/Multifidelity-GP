"""
Operating profile of the boat: the probability distribution of the navigation conditions (TWA, TWS)
it really meets, given per run in a JSON file. It serves twice:
- to ANALYSE sections: their expected speed over that distribution (scripts/profile_analysis.py, or
  run_real_vpp.py --profile; decision of 2026-10-22);
- in the SEARCH, with run_real_vpp.py --search-profile (2026-10-23): the nodes become the conditions
  of the run and their weights w_c weigh the objective, the screening and the criterion
  (vpp_optimize.VPPSectionOptimizer, CHANGEMENTS_2026-10-23.md).

The distribution is a mixture of modes, typically two (upwind and downwind), each with a weight,
a truncated normal law of TWA and a truncated normal law of TWS (independent within a mode):
    p(TWA, TWS) = sum_m pi_m N_trunc(TWA; mean, std, [lower, upper]) N_trunc(TWS; ...).
The expectation of a quantity Q (e.g. the boat speed) is computed by a product quadrature per mode:
n nodes per variable at the mid-probability points of n equiprobable strata (the quantiles
(i + 1/2) / n of the truncated law), weight pi_m / n^2 each:
    E_p[Q] ~ sum_nodes w_node Q(TWA_node, TWS_node).
The nodes stay inside the bounds (no tail outside the domain of the VPP).

A profile can also list its nodes directly ("nodes": [{"twa", "tws", "weight", "mode"}, ...]),
e.g. a histogram of logged conditions; the weights are then normalized to 1.

JSON format (example: profiles/guess_bimodal.json):
    {"name": "guess_bimodal", "nodes_per_dim": 3,
     "modes": [{"name": "upwind", "weight": 0.5,
                "twa": {"mean": 45, "std": 4, "lower": 38, "upper": 52},
                "tws": {"mean": 13, "std": 3, "lower": 8, "upper": 18}}, ...]}
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import truncnorm


def strata_nodes(law: dict, n: int) -> np.ndarray:
    """The n mid-probability points of a truncated normal law {mean, std, lower, upper}."""
    mean, std = float(law["mean"]), float(law["std"])
    lower, upper = float(law.get("lower", -np.inf)), float(law.get("upper", np.inf))
    a, b = (lower - mean) / std, (upper - mean) / std
    return truncnorm.ppf((np.arange(n) + 0.5) / n, a, b, loc=mean, scale=std)


class OperatingProfile:
    """Nodes and weights of a distribution of (TWA, TWS) (module docstring)."""

    def __init__(self, name: str, modes=None, nodes_per_dim: int = 3, nodes=None):
        if (modes is None) == (nodes is None):
            raise ValueError("an operating profile has either modes or explicit nodes")
        self.name, self.modes, self.nodes_per_dim = str(name), modes, int(nodes_per_dim)
        self.explicit_nodes = nodes

    @classmethod
    def from_json(cls, path) -> "OperatingProfile":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(data.get("name", Path(path).stem), modes=data.get("modes"),
                   nodes_per_dim=data.get("nodes_per_dim", 3), nodes=data.get("nodes"))

    def nodes(self) -> list:
        """[{"twa", "tws", "weight", "mode"}], weights summing to 1."""
        if self.explicit_nodes is not None:
            total = sum(float(node["weight"]) for node in self.explicit_nodes)
            return [{"twa": float(node["twa"]), "tws": float(node["tws"]),
                     "weight": float(node["weight"]) / total, "mode": node.get("mode", "")}
                    for node in self.explicit_nodes]
        total = sum(float(mode["weight"]) for mode in self.modes)
        n = self.nodes_per_dim
        out = []
        for mode in self.modes:
            for twa in strata_nodes(mode["twa"], n):
                for tws in strata_nodes(mode["tws"], n):
                    out.append({"twa": float(twa), "tws": float(tws),
                                "weight": float(mode["weight"]) / total / n ** 2,
                                "mode": mode.get("name", "")})
        return out
