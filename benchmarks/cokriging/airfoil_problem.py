"""
[CK] Small multi-fidelity airfoil problem of the co-kriging benchmark: drag of a continuous
NACA 4-digit section computed by NeuralFoil with a small network (low fidelity) and the largest
network (high fidelity). Self-contained: no bdFoil pipeline, no JSON configuration.

Inputs x in [0, 1]^4, mapped linearly to:
    camber m in [0, 0.06], camber position p in [0.2, 0.6], thickness t in [0.08, 0.18],
    angle of attack alpha in [-2, 8] deg.
Output: drag coefficient in drag counts (1e4 CD) at Re = 5e5, n_crit = 1 and forced transition
at 10 % chord on both sides (house convention of the bdFoil pipeline).
Levels: 1 = NeuralFoil "xxsmall", 2 = NeuralFoil "xxxlarge" (nominal costs 1 and 10).
[V2] AirfoilXfoilMF: same sections and flow, 1 = NeuralFoil "xxxlarge", 2 = XFOIL (real pair).

Usage (same interface as bench_lib.HartmannMF):
    problem = AirfoilMF()
    y = problem(x, level)          # x of shape (m, 4), y of shape (m,)
"""
import os
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "example" / "hydrofoil_optim"))
# pylint: disable=wrong-import-position,import-error
from optim_neuralfoil import generate_continuous_naca4  # noqa: E402

LOWER = np.array([0.0, 0.2, 0.08, -2.0])
UPPER = np.array([0.06, 0.6, 0.18, 8.0])
NAMES = ("camber m", "camber position p", "thickness t", "alpha (deg)")
MODEL_SIZES = ("xxsmall", "xxxlarge")
FLOW = {"Re": 5e5, "n_crit": 1.0, "xtr_upper": 0.1, "xtr_lower": 0.1}
# [V2] XFOIL executable of bdToolbox (default of the bdFoil pipeline)
XFOIL_COMMAND = r"C:\Users\SIM\Outils - banulsdesign\bdToolbox\soft\xfoil.exe"


@lru_cache(maxsize=None)
def kulfan_parameters(m: float, p: float, t: float) -> dict:
    """Kulfan (CST) parameters, 8 weights per side, of the continuous NACA section (m, p, t)."""
    from aerosandbox.geometry.airfoil.airfoil_families import (  # noqa: PLC0415
        get_kulfan_parameters)
    return get_kulfan_parameters(generate_continuous_naca4(m, p, t, n_points=161))


def neuralfoil_cd(physical: np.ndarray, model_size: str) -> np.ndarray:
    """Drag counts by NeuralFoil of the sections (m, p, t, alpha) in physical units (m, 4)."""
    import neuralfoil as nf  # noqa: PLC0415
    params = [kulfan_parameters(*(round(float(v), 12) for v in row[:3])) for row in physical]
    # vectorized NeuralFoil call: weights of shape (8, m), scalars of shape (m,)
    kulfan = {key: np.array([p[key] for p in params]).T for key in params[0]}
    aero = nf.get_aero_from_kulfan_parameters(kulfan, alpha=physical[:, 3],
                                              model_size=model_size, **FLOW)
    return 1e4 * np.asarray(aero["CD"], dtype=float).reshape(-1)


def xfoil_cd(m: float, p: float, t: float, alpha: float) -> float:
    """[V2] Drag counts by XFOIL of one section (same flow as NeuralFoil; NaN if XFOIL fails or
    does not converge). Executable: environment variable MFEGO_XFOIL, else XFOIL_COMMAND."""
    import aerosandbox as asb  # noqa: PLC0415
    airfoil = asb.Airfoil(coordinates=generate_continuous_naca4(m, p, t, n_points=161))
    try:
        result = asb.XFoil(airfoil=airfoil, max_iter=100, timeout=30,
                           xfoil_command=os.environ.get("MFEGO_XFOIL", XFOIL_COMMAND),
                           **FLOW).alpha(alpha)
    except Exception:  # noqa: BLE001  (a failed XFOIL run is a failed evaluation)
        return np.nan
    return 1e4 * float(result["CD"][0]) if len(result["CD"]) else np.nan


@dataclass
class AirfoilMF:
    """Multi-fidelity NACA 4-digit drag problem (level 1 = xxsmall, level 2 = xxxlarge)."""
    dim: int = 4
    costs: tuple = (1.0, 10.0)
    f_star: float = 0.0           # unknown optimum: "errors" of the optimization are raw values
    n_evaluations: dict = field(default_factory=dict)

    @property
    def num_levels(self) -> int:
        return len(MODEL_SIZES)

    def __call__(self, x: np.ndarray, level: int) -> np.ndarray:
        """Drag counts of the sections x (unit cube, shape (m, 4)) at a fidelity level."""
        x = np.atleast_2d(x)
        self.n_evaluations[level] = self.n_evaluations.get(level, 0) + len(x)
        return neuralfoil_cd(LOWER + x * (UPPER - LOWER), MODEL_SIZES[level - 1])

    def doe(self, points_per_level: list, seed: int = 0) -> list:
        """Independent (non-nested) LHS per level, as bench_lib.HartmannMF.doe:
        [(x_1, y_1), ..., (x_L, y_L)] (initial design of the EGO runs)."""
        from scipy.stats import qmc  # noqa: PLC0415
        datasets = []
        for level, n_points in enumerate(points_per_level, start=1 + self.num_levels
                                         - len(points_per_level)):
            x = qmc.LatinHypercube(d=self.dim, seed=seed * 100 + level).random(n_points)
            datasets.append((x, self(x, level)))
        return datasets


@dataclass
class AirfoilXfoilMF(AirfoilMF):
    """[V2] Same sections and flow, level 1 = NeuralFoil "xxxlarge", level 2 = XFOIL (a real
    low / high fidelity pair: NeuralFoil is trained on XFOIL; at the high-fidelity points of the
    benchmark designs the median relative difference is 0.8 % (max 16 %) and
    RMS(LF - HF) / std(HF) = 0.13)."""

    def __call__(self, x: np.ndarray, level: int) -> np.ndarray:
        x = np.atleast_2d(x)
        self.n_evaluations[level] = self.n_evaluations.get(level, 0) + len(x)
        physical = LOWER + x * (UPPER - LOWER)
        if level == 1:
            return neuralfoil_cd(physical, "xxxlarge")
        return np.array([xfoil_cd(*row) for row in physical])
