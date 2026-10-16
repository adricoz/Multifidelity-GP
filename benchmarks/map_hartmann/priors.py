"""
Lengthscale priors of the MAP study (benchmarks/map_hartmann): InvGamma, Gamma and LogNormal
densities on each lengthscale l of the squared exponential kernel, and the mfego GP /
multi-fidelity model classes that use them.

At the time of this study the mfego core only had the InvGamma lengthscale prior: these classes
put any family on the lengthscales ONLY (no prior on t1, t2, noise), to reproduce the study. The
densities now live in the core (src.surrogate_models.neg_log_prior, same formulas). As in the
core, the MAP is taken in l-space (density on l, no Jacobian of the log-parametrization), which is
also what GPyTorch / BoTorch do (prior evaluated on the constrained lengthscale).
"""
import numpy as np
from scipy import stats

# bench_lib puts mfego/ on sys.path
import bench_lib  # noqa: F401  # pylint: disable=unused-import,import-error
from src.surrogate_models import (GaussianProcess, MultifidelityModel,  # noqa: E402
                                  neg_log_prior)

# name -> (group, (family, a, b)); InvGamma(alpha, beta), Gamma(shape, rate), LogNormal(mu, sigma)
PRIORS = {
    "IG(3, 0.4)": ("mode", ("invgamma", 3.0, 0.4)),
    "IG(3, 1)": ("mode", ("invgamma", 3.0, 1.0)),
    "IG(3, 2)": ("mode", ("invgamma", 3.0, 2.0)),
    "IG(3, 4)": ("mode", ("invgamma", 3.0, 4.0)),
    "IG(3, 8)": ("mode", ("invgamma", 3.0, 8.0)),
    "IG(1.5, 1.25)": ("strength", ("invgamma", 1.5, 1.25)),
    "IG(10, 5.5)": ("strength", ("invgamma", 10.0, 5.5)),
    "Gamma(3, 4)": ("family", ("gamma", 3.0, 4.0)),
    "LogN(-0.13, 0.75)": ("family", ("lognormal", np.log(0.5) + 0.75 ** 2, 0.75)),
    "Gamma(3, 6) BoTorch MF": ("library", ("gamma", 3.0, 6.0)),
    "LogN(2.31, 1.73) BoTorch": ("library", ("lognormal", np.sqrt(2) + 0.5 * np.log(6),
                                             np.sqrt(3))),
}


def distribution(prior: tuple):
    """Frozen scipy distribution of the prior (pdf for the figures, quantiles for the report)."""
    family, a, b = prior
    if family == "invgamma":
        return stats.invgamma(a, scale=b)
    if family == "gamma":
        return stats.gamma(a, scale=1.0 / b)
    if family == "lognormal":
        return stats.lognorm(b, scale=np.exp(a))
    raise ValueError(f"Unknown prior family: {family}")


def mode(prior: tuple) -> float:
    """Mode of p(l), i.e. the value the prior pulls the MAP towards."""
    family, a, b = prior
    if family == "invgamma":
        return b / (a + 1.0)
    if family == "gamma":
        return (a - 1.0) / b
    return float(np.exp(a - b ** 2))


class PriorGP(GaussianProcess):
    """GaussianProcess whose objective is NLL - log p(l) for any prior family (None: MLE)."""
    def __init__(self, kernel, seed: int = None, prior: tuple = None):
        super().__init__(kernel, seed=seed)
        self.prior = prior

    def negative_log_likelihood(self, log_params, y_n, f_n=None, rho_init=1.0,
                                estimate_rho=False, rho_bounds=(-5.0, 5.0), with_grad=False,
                                use_map=False):
        """NLL of the parent class (without its InvGamma prior) - log p(l) of self.prior."""
        del use_map  # the prior is self.prior (use_map is the core InvGamma option)
        out = super().negative_log_likelihood(log_params, y_n, f_n, rho_init, estimate_rho,
                                              rho_bounds, with_grad)
        if self.prior is None:
            return out
        d = self.x_train.shape[1]
        value, grad = neg_log_prior(self.prior, np.asarray(log_params[:d]))
        if not with_grad:
            return out + value
        nll, nll_grad = out
        nll_grad[:d] += grad
        return nll + value, nll_grad


class PriorMultifidelityModel(MultifidelityModel):
    """MultifidelityModel whose GPs (every level) use the lengthscale prior `prior`."""
    def __init__(self, l, kernel_class, prior: tuple = None, seed: int = None, **kwargs):
        super().__init__(l, kernel_class, seed=seed, **kwargs)
        self.prior = prior
        self.gps = [PriorGP(kernel_class(), seed=None if seed is None else seed + i, prior=prior)
                    for i in range(l)]

    def estimator_label(self) -> str:
        """The prior of this benchmark model (the core label would say MLE)."""
        return "MLE" if self.prior is None else f"MAP, lengthscale prior {self.prior}"
