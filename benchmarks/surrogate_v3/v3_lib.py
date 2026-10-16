"""
[V3] Shared definitions of the mfego v3 benchmark: the mfego variants compared (v1, v2 with two
normal priors on rho, v3 with rho learned under a lognormal prior) and the reuse of the problems,
test sets and designs of benchmarks/surrogate_v2 (v2_lib: XFOIL values cached, not recomputed).

v3: rho > 0 is learned WITH the kernel hyperparameters (log rho is one more parameter of
L-BFGS-B), MAP with a lognormal prior LogNormal(s^2, s) whose mode is 1 (mfego computes the MAP
in parameter space, so rho is pulled towards the mode of the prior), wide bounds [1e-3, 50]: the
prior, not the bound, limits rho. A normal prior learned the same way would give v2 again (for a
fixed theta the closed form of v2 is the optimum in rho; tests/test_gaussian_process.py).
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "surrogate_v2"))

# pylint: disable=wrong-import-position,import-error
import v2_lib as v2  # noqa: E402  (sets the paths of bench_lib, mfego and the problems)
from src.kernels import Matern52Kernel, SquaredExponentialKernel  # noqa: E402

RESULTS = HERE / "results"
RHO_BOUNDS_V3 = (1e-3, 50.0)


def lognormal_mode_1(s: float) -> tuple:
    """[V3] LogNormal(mu, s) prior on rho with its mode exp(mu - s^2) at 1."""
    return ("lognormal", s ** 2, s)


def centred(prior: tuple, bounds=(-5.0, 5.0), kernel=Matern52Kernel) -> dict:
    """Options of a centred rho with the given prior on rho (MAP)."""
    return {"use_map": True, "rho_centering": True, "priors": {"rho": prior},
            "rho_bounds": bounds, "kernel_class": kernel}


# name -> options of MultifidelityModel / CoKriging (bench_lib wrappers, v2_lib.make_model)
VARIANTS = {
    "v1": v2.v1_options(),
    "v2 N(1, 2²)": centred(("normal", 1.0, 2.0)),
    "v2 N(1, 1²)": centred(("normal", 1.0, 1.0)),
    "v3 LogN s=0.5": centred(lognormal_mode_1(0.5), RHO_BOUNDS_V3),
    "v3 LogN s=1": centred(lognormal_mode_1(1.0), RHO_BOUNDS_V3),
}
# kernel-only reference for the EGO (v1 rho, Matern 5/2)
EGO_VARIANTS = {**VARIANTS, "v1 Matern": {**v2.v1_options(), "kernel_class": Matern52Kernel}}
REFERENCE = "v1"
CURRENT_DEFAULT = "v2 N(1, 2²)"     # default of CoKriging before v3
# the variants with a centred rho can sit at a bound of rho (v1: bounds +-5, never reached)
BOUNDS = {name: options.get("rho_bounds", (-5.0, 5.0)) for name, options in EGO_VARIANTS.items()}
KERNEL_OF = {name: options["kernel_class"].__name__ for name, options in EGO_VARIANTS.items()}
assert KERNEL_OF["v1"] == SquaredExponentialKernel.__name__


def at_bound(rho: float, bounds: tuple) -> bool:
    """rho on a bound of its search interval (relative margin 1e-3)."""
    lo, hi = bounds
    return rho >= hi * (1 - 1e-3) or rho <= (lo * (1 + 1e-3) if lo > 0 else lo * (1 - 1e-3))
