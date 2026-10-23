"""
Sequential design of a multi-fidelity co-kriging surrogate.

Method: Le Gratiet & Cannamela (2015), with the corrections described in
vpp_acceleration/doc/sequential_cokriging.pdf (called "the note" below). The proofs are in
vpp_acceleration/doc/complements_mathematiques.pdf. (N1) ... (N6) are the equations of the note.

The surrogate itself is the CoKriging class of mfego (mfego/src/cokriging.py, used read-only).
This module only decides WHERE to run the codes and WHICH fidelity levels to run there.

One iteration (method SequentialCoKriging.ask, then SequentialCoKriging.evaluate):
  1. fit the CoKriging on the nested design (its hyperparameters are re-estimated);
  2. leave-one-out of the GP of each level: squared standardized errors z^2 (N3);
  3. choose the point: x0 = argmax of the adjusted variance (N4), by differential evolution;
  4. choose the level: l* = argmax of (IMSE gain / cumulative cost) (N5)-(N6), the gain being
     computed exactly with the joint co-kriging (valid whatever the noise);
  5. run the levels 1, ..., l* at x0, cheapest first, and stop at the first failed run.
The method SequentialCoKriging.run repeats this until the budget is spent.
With point_criterion="alc", steps 3 and 4 are done together: the pair (point, level) with the
best integrated gain per unit of cost among Sobol candidates (find_next_point_alc). It avoids
the points stuck at the border of the domain that the maximum of the variance produces in
dimension >= 4 (README, section 0).

Notation used in the code
  L        number of fidelity levels (model.num_levels).
  level j  1 = cheapest code ... L = most accurate code. model.gps[j - 1] is the GP of the
           correction delta_j of the auto-regressive model Z_j = rho_(j-1) Z_(j-1) + delta_j (N1).
  P_j      weight of delta_j in the high-fidelity output Z_L: P_j = rho_j * ... * rho_(L-1).
  x        a point in the raw units of the design space (ExperimentData.bounds).
  u        the same point scaled to the unit cube [0, 1]^d: the GPs of mfego work with u.

How to import this module (both folders must be on sys.path, see tests/conftest.py):
    sys.path.insert(0, ".../Multifidelity-GP-Clean/mfego")
    sys.path.insert(1, ".../Multifidelity-GP-Clean/vpp_acceleration")
    from src.sequential_design import SequentialCoKriging
mfego/src and vpp_acceleration/src have no __init__.py, so Python merges them into a single
"namespace package" called src. Therefore: never add an __init__.py in vpp_acceleration/src, and
never give a module here the name of a module of mfego/src (it would be silently ignored).
"""
import numpy as np
from scipy.linalg import cho_solve, solve_triangular
from scipy.optimize import differential_evolution
from scipy.stats import qmc

from src.cokriging import CoKriging
from src.validation import prediction_scores

# Distance (in the unit cube) below which a point counts as "the same point" as a failed run.
EXCLUSION_RADIUS = 1e-3
# Default number of Sobol integration points (a power of 2, as Sobol sequences require).
N_INTEGRATION = 1024


# 1/3 ---------------------------------------------------------------------------------------------
# Quantities read from a fitted CoKriging (note, sections 2 to 4)
# -------------------------------------------------------------------------------------------------
def to_unit(model: CoKriging, x: np.ndarray) -> np.ndarray:
    """
    Raw points -> points of the unit cube, as seen by the GPs of the model.
    (Only access to the private method CoKriging._unit: if mfego changes, fix it here only.)

    Args:
    - model: fitted CoKriging.
    - x: numpy array (m, d) or (d,), raw units.
    Returns:
    - numpy array (m, d) in [0, 1]^d.
    """
    return model._unit(np.atleast_2d(np.asarray(x, dtype=float)))


def level_weights(model: CoKriging) -> np.ndarray:
    """
    P_j for j = 1..L (note, Eq. N2): weight of the correction delta_j in the high-fidelity
    output, Z_L = sum_j P_j delta_j. Example with 2 levels: P = [rho_1, 1].
    (Only access to the private method CoKriging._factors.)

    Returns:
    - numpy array (L,).
    """
    top_level = np.array([model.num_levels])
    return model._factors(top_level)[0]


def check_level_gps(model: CoKriging) -> None:
    """
    The GP of each level is conditioned on its own data only by CoKriging.fit(). A model rebuilt
    with CoKriging.from_dict() or from_multifidelity() conditions the joint system only, and the
    functions of this module would then fail with an obscure error: stop here with a clear one.
    """
    for j, gp in enumerate(model.gps, start=1):
        if gp.l_chol is None:
            raise ValueError(f"The GP of level {j} is not conditioned: fit the model with "
                             "CoKriging.fit() before using sequential_design.")


def level_variances(model: CoKriging, u: np.ndarray) -> np.ndarray:
    """
    sigma_j^2(u) for every level j: posterior variance of the correction delta_j at the points u,
    given the data of its level (latent function, raw units). Eq. N2:
        Var[Z_L(u)] = sum_j P_j^2 sigma_j^2(u).
    This decomposition is exact for noise-free lower levels (proposition 1 of the note); with
    the noise learned by mfego it is approximate (Forrester, squared exponential kernel: 1.5 %
    of the maximum variance, at the border x = 1). It is used for the choice of the POINT, whose
    adjusted variance is a heuristic anyway; the gains of the LEVEL choice are exact (imse_gains).
    GaussianProcess.predict_batch returns the variance of an OBSERVATION (noise included):
    the noise variance is removed to get the variance of the function itself.

    Args:
    - u: numpy array (m, d), unit cube.
    Returns:
    - numpy array (L, m).
    """
    variances = []
    for gp in model.gps:
        _, variance_with_noise = gp.predict_batch(u)
        latent_variance = variance_with_noise - gp.get_noise_variance()
        variances.append(np.maximum(latent_variance, 0.0))  # round-off can give -1e-17
    return np.array(variances)


def joint_posterior_covariance(model: CoKriging, x_a: np.ndarray, levels_a: np.ndarray,
                               x_b: np.ndarray, levels_b: np.ndarray) -> np.ndarray:
    """
    Posterior covariance of the co-kriging between Z_(level a)(x_a) and Z_(level b)(x_b), given
    ALL the observations. Exact Gaussian conditioning of the joint model, for any noise and any
    design (the per-level decomposition of Eq. N2 needs noise-free lower levels):
        k_n(a, b) = c(a, b) - c(a, D) V^-1 c(D, b),
    c = covariance of the auto-regressive model (CoKriging._covariance, raw units), D = all the
    observations, V = their covariance matrix with the noise, factorized once by CoKriging.fit
    (V = C C^T, model.l_chol). With w = C^-1 c(D, .), the second term is w_a^T w_b.
    (Only access to the private method CoKriging._covariance.)

    Args:
    - x_a: numpy array (m, d), raw units; levels_a: integer numpy array (m,), level of each point.
    - x_b: numpy array (k, d), raw units; levels_b: integer numpy array (k,).
    Returns:
    - numpy array (m, k).
    """
    prior = model._covariance(to_unit(model, x_a), levels_a, to_unit(model, x_b), levels_b)
    w_a = whitened_data_covariance(model, x_a, levels_a)                 # (n_total, m)
    w_b = whitened_data_covariance(model, x_b, levels_b)                 # (n_total, k)
    return prior - w_a.T @ w_b


def whitened_data_covariance(model: CoKriging, x: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """
    w = C^-1 c(D, (x, level)): covariance between the observations D and the points x at their
    levels, "whitened" by the Cholesky factor of V = C C^T. Building block of every posterior
    covariance: k_n(a, b) = c(a, b) - w_a^T w_b (see joint_posterior_covariance).

    Returns:
    - numpy array (n_total, m).
    """
    c = model._covariance(to_unit(model, x), levels, to_unit(model, model.x_all),
                          model.levels_all)                              # (m, n_total)
    return solve_triangular(model.l_chol, c.T, lower=True)


def loo_errors(gp) -> tuple[np.ndarray, np.ndarray]:
    """
    Leave-one-out of ONE GP on its own data, in closed form (note, Eq. N3; Dubrule 1983):
    the observation i is predicted from all the other observations, hyperparameters fixed.
    With K the covariance matrix of the observations (noise included) and alpha = K^-1 (y - mean):
        error_i    = y_i - prediction_i = alpha_i / [K^-1]_ii,
        variance_i = 1 / [K^-1]_ii.
    For the GP of level j, these are the errors of the correction delta_j alone: by the
    proposition 2 of the note, they are exactly the terms of Le Gratiet & Cannamela Eq. 8.

    Args:
    - gp: GaussianProcess of one level (model.gps[j - 1]).
    Returns:
    - errors (n,), variances (n,), raw units.
    """
    n = len(gp.x_train)
    k_inverse = cho_solve((gp.l_chol, True), np.eye(n))   # K^-1 from the Cholesky factor
    diagonal = np.diag(k_inverse)
    errors = gp.y_std * gp.alpha / diagonal               # gp.alpha is in normalized units
    variances = gp.y_std ** 2 / diagonal
    return errors, variances


def loo_squared_errors(gp, z2_max: float = None) -> np.ndarray:
    """
    Squared standardized leave-one-out errors z_i^2 = error_i^2 / variance_i of one GP (N3).
    If the model is right, z_i follows N(0, 1): z_i^2 >> 1 means that the GP is wrong at x_i by
    more than it believes. Optional cap z2_max (e.g. 9 = 3 sigma) against a single outlier.

    Returns:
    - numpy array (n,).
    """
    errors, variances = loo_errors(gp)
    z2 = errors ** 2 / variances
    if z2_max is not None:
        z2 = np.minimum(z2, z2_max)
    return z2


def most_correlated_point(gp, u: np.ndarray) -> np.ndarray:
    """
    For each point u, index of the training point of the GP most correlated with it: the nearest
    one in the metric of the kernel, sum_m ((u_m - x_m) / lengthscale_m)^2 (note, section 3).
    This is the Voronoi cell of Le Gratiet & Cannamela Eq. 8, with a distance that does not
    depend on the units and that follows the anisotropy of the kernel.

    Args:
    - u: numpy array (m, d), unit cube.
    Returns:
    - integer numpy array (m,).
    """
    scaled_difference = (u[:, None, :] - gp.x_train[None, :, :]) / gp.kernel.lengthscale
    squared_distance = np.sum(scaled_difference ** 2, axis=2)    # (m, n)
    return np.argmin(squared_distance, axis=1)


def adjustment_factors(model: CoKriging, z2: list, u: np.ndarray) -> np.ndarray:
    """
    a_j(u) = 1 + z^2 of the training point of level j most correlated with u (note, Eq. N4).

    Args:
    - z2: list of L numpy arrays, z2[j - 1] = loo_squared_errors of the GP of level j
      (arrays of zeros: no adjustment, a_j = 1).
    - u: numpy array (m, d), unit cube.
    Returns:
    - numpy array (L, m).
    """
    factors = []
    for j, gp in enumerate(model.gps):
        nearest = most_correlated_point(gp, u)
        factors.append(1.0 + z2[j][nearest])
    return np.array(factors)


def adjusted_variance(model: CoKriging, z2: list, u: np.ndarray) -> np.ndarray:
    """
    Adjusted variance of the high-fidelity output (note, Eq. N4):
        k_adj(u) = sum_j P_j^2 sigma_j^2(u) a_j(u).
    With z2 = zeros, a_j = 1 and this is the variance of the co-kriging (Eq. N2).

    Returns:
    - numpy array (m,).
    """
    weights_squared = level_weights(model) ** 2        # (L,)
    variances = level_variances(model, u)              # (L, m)
    factors = adjustment_factors(model, z2, u)         # (L, m)
    return np.sum(weights_squared[:, None] * variances * factors, axis=0)


def imse_gains(model: CoKriging, z2: list, x0: np.ndarray, x_int: np.ndarray,
               missing: np.ndarray = None) -> np.ndarray:
    """
    G_l(x0) for l = 1..L (note, Eq. N5): reduction of the integrated variance of the
    HIGH-fidelity output if the point x0 is evaluated at the levels 1..l.

    How it is computed (exact for the joint co-kriging, whatever the noise):
    - The new observations are w_j = Z_j(x0) + noise_j for the levels j <= l still missing at x0.
      Before running anything, their posterior covariance matrix is S = K_00 + diag(noise_j),
      and their covariance with the high-fidelity output at x is the vector c(x).
    - Gaussian conditioning on w: the variance at x drops by c(x)^T S^-1 c(x), whatever the
      values that the codes will return (the gain is known BEFORE running them).
    - The integral over the design space is the mean over the integration points x_int.
    With noise-free lower levels this is exactly the per-level formula N5 of the note (each level
    j adds P_j^2 * k_n_j(x, x0)^2 / (sigma_j^2(x0) + noise_j): the rho^2 factor that the rule of
    Le Gratiet & Cannamela misses). With the noise learned by mfego that formula is only
    approximate (measured on Forrester: 26 to 70 % off), hence the exact joint computation.
    - Adjustment (Le Gratiet & Cannamela Eq. 10): the increment G_j - G_(j-1) brought by level j
      is multiplied by a_j(x0) (no effect if z2 = zeros).

    Args:
    - z2: list of L arrays (see adjustment_factors).
    - x0: numpy array (d,) or (1, d), raw units.
    - x_int: numpy array (N, d), integration points, raw units.
    - missing: optional boolean array (L,): levels not yet evaluated at x0 (default: all). A level
      already evaluated at x0 brings no new observation.
    Returns:
    - numpy array (L,), gains[l - 1] = G_l (cumulative).
    """
    x0 = np.atleast_2d(x0)
    if missing is None:
        missing = np.ones(model.num_levels, dtype=bool)
    gains = batched_imse_gains(model, x0, x_int, np.atleast_2d(missing))     # (1, L)
    return adjust_gains(model, z2, x0, gains)[0]


def batched_imse_gains(model: CoKriging, x_cand: np.ndarray, x_int: np.ndarray,
                       missing: np.ndarray, int_weights: np.ndarray = None) -> np.ndarray:
    """
    The gains G_l of imse_gains (without the adjustment) for MANY candidate points at once: the
    same formulas, written with arrays that carry one extra axis for the candidates (this is
    what makes the ALC search affordable: thousands of candidates in one matrix computation).

    Args:
    - x_cand: numpy array (M, d), candidate points, raw units.
    - x_int: numpy array (N, d), integration points, raw units.
    - missing: boolean numpy array (M, L), True if the level is not yet evaluated at the candidate.
    - int_weights: None (the plain mean over the integration points) or numpy array (N,), the
      weight of every integration point: the gain is then sum_i int_weights[i] x reduction_i
      (e.g. the probability of the condition of the point times (dV/dC)^2, vpp_optimize).
    Returns:
    - numpy array (M, L), gains[m, l - 1] = G_l(x_m) (cumulative).
    """
    n_cand, top = len(x_cand), model.num_levels
    all_levels = np.arange(1, top + 1)
    # every possible new observation: candidate m at level j, in the order (m, j)
    x_repeated = np.repeat(x_cand, top, axis=0)                  # (M L, d)
    levels_repeated = np.tile(all_levels, n_cand)                 # (M L,)
    w_new = whitened_data_covariance(model, x_repeated, levels_repeated)            # (n, M L)
    w_int = whitened_data_covariance(model, x_int, np.full(len(x_int), top))        # (n, N)

    # c[m, j, k]: posterior covariance between Z_L(x_int_k) and the observation (x_m, level j)
    prior_cross = model._covariance(to_unit(model, x_int), np.full(len(x_int), top),
                                    to_unit(model, x_repeated), levels_repeated)    # (N, M L)
    c = (prior_cross - w_int.T @ w_new).reshape(len(x_int), n_cand, top).transpose(1, 2, 0)

    # s[m]: posterior covariance matrix (L, L) of the L observations of candidate m, plus noise.
    # Stationary kernels (squared exponential, Matern): the prior part is the same at every point.
    prior_block = model._covariance(to_unit(model, x_repeated[:top]), all_levels,
                                    to_unit(model, x_repeated[:top]), all_levels)   # (L, L)
    w_blocks = w_new.reshape(-1, n_cand, top)                                       # (n, M, L)
    noise = np.array([gp.get_noise_variance() for gp in model.gps])
    s = prior_block[None] - np.einsum("nmi,nmj->mij", w_blocks, w_blocks) + np.diag(noise)

    gains = np.zeros((n_cand, top))
    for l in range(1, top + 1):
        # candidates with the same missing levels <= l add the same observations: one batch each
        patterns = missing[:, :l]
        for pattern in np.unique(patterns, axis=0):
            if not pattern.any():
                continue                                   # nothing new at these candidates
            rows = np.flatnonzero(np.all(patterns == pattern, axis=1))
            added = np.flatnonzero(pattern)                # indices of the new observations
            s_sub = s[np.ix_(rows, added, added)]          # (R, A, A)
            c_sub = c[rows][:, added, :]                   # (R, A, N)
            solution = np.linalg.solve(s_sub, c_sub)       # S^-1 c, batched, (R, A, N)
            reduction = np.sum(c_sub * solution, axis=1)   # c^T S^-1 c at each x_int, (R, N)
            gains[rows, l - 1] = (np.mean(reduction, axis=1) if int_weights is None
                                  else reduction @ np.asarray(int_weights, dtype=float))
    return gains


def adjust_gains(model: CoKriging, z2: list, x_cand: np.ndarray, gains: np.ndarray) -> np.ndarray:
    """
    Adjustment of Le Gratiet & Cannamela Eq. 10: the increment G_j - G_(j-1) brought by level j
    at the candidate is multiplied by a_j (no effect if z2 = zeros).

    Args:
    - x_cand: numpy array (M, d), raw units; gains: numpy array (M, L) from batched_imse_gains.
    Returns:
    - numpy array (M, L), adjusted cumulative gains.
    """
    factors = adjustment_factors(model, z2, to_unit(model, x_cand)).T           # (M, L)
    increments = np.diff(np.concatenate([np.zeros((len(gains), 1)), gains], axis=1), axis=1)
    return np.cumsum(factors * increments, axis=1)


def choose_level(gains: np.ndarray, costs: np.ndarray, missing: np.ndarray,
                 remaining_budget: float = np.inf) -> tuple[int, np.ndarray]:
    """
    Level with the best yield, l* = argmax_l G_l / C_l (note, Eq. N6).
    C_l = cost of the levels <= l that are still missing at x0: because the design must stay
    nested, running level l at x0 also requires the levels below (cumulative cost).
    A level is a candidate only if it is missing at x0 (otherwise it brings nothing new) and if
    C_l fits in the remaining budget.

    Args:
    - gains: numpy array (L,), output of imse_gains.
    - costs: numpy array (L,), cost of one evaluation of each level (ExperimentData.costs).
    - missing: boolean numpy array (L,), True if the level is not evaluated yet at x0.
    - remaining_budget: maximum cost allowed for this iteration.
    Returns:
    - level (1..L, or None if no level is a candidate), ratios G_l / C_l (-inf if not a candidate).
    """
    costs_to_pay = np.where(missing, costs, 0.0)
    cumulative_costs = np.cumsum(costs_to_pay)
    ratios = np.full(len(gains), -np.inf)
    for l in range(len(gains)):
        if missing[l] and cumulative_costs[l] <= remaining_budget:
            ratios[l] = gains[l] / cumulative_costs[l]
    if np.all(np.isinf(ratios)):
        return None, ratios
    return int(np.argmax(ratios)) + 1, ratios


# 2/3 ---------------------------------------------------------------------------------------------
# Where to compute: maximum of the adjusted variance (note, section 3)
# -------------------------------------------------------------------------------------------------
def find_next_point(model: CoKriging, z2: list, bounds: list, feasibility=None,
                    excluded: np.ndarray = None, seed: int = None, popsize: int = 10,
                    maxiter: int = 50) -> tuple[np.ndarray, float]:
    """
    x0 = argmax of the adjusted variance (note, Eq. N4) over the design space.
    Differential evolution is global and needs no gradient, so it copes with the jumps of the
    adjusted variance at the borders of the cells (same optimizer as EGOOptimizer of mfego).

    Args:
    - bounds: list of (low, high) per dimension, raw units (ExperimentData.bounds).
    - feasibility: optional function x (m, d) -> booleans (m,) (known constraints, same convention
      as mfego): the criterion is set to 0 outside the feasible domain.
    - excluded: optional numpy array (k, d), raw points of failed runs: the criterion is set to 0
      within EXCLUSION_RADIUS of them, so that a failed point is not proposed again.
    - seed: seed of the differential evolution (reproducible runs).
    Returns:
    - x0: numpy array (d,), raw units; value of the adjusted variance at x0.
    """
    excluded_unit = None
    if excluded is not None and len(excluded) > 0:
        excluded_unit = to_unit(model, excluded)

    def negative_criterion(x):
        # scipy calls it with one point (d,), or with the whole population (d, S) when vectorized
        points = np.atleast_2d(x.T) if x.ndim == 2 else x[None, :]
        u = to_unit(model, points)
        criterion = adjusted_variance(model, z2, u)
        if feasibility is not None:
            feasible = np.asarray(feasibility(points), dtype=bool).reshape(-1)
            criterion[~feasible] = 0.0
        if excluded_unit is not None:
            distances = np.linalg.norm(u[:, None, :] - excluded_unit[None, :, :], axis=2)
            criterion[np.min(distances, axis=1) < EXCLUSION_RADIUS] = 0.0
        # differential_evolution minimizes: return minus the criterion
        return -criterion if x.ndim == 2 else -float(criterion[0])

    result = differential_evolution(negative_criterion, bounds, popsize=popsize, maxiter=maxiter,
                                    updating="deferred", vectorized=True, seed=seed)
    return np.asarray(result.x, dtype=float), float(-result.fun)


def candidate_points(data, n_candidates: int, seed: int = None, feasibility=None,
                     excluded: np.ndarray = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Candidates of the ALC search, with the levels still missing at each of them:
    - n_candidates Sobol points of the design space, drawn again at every iteration (another
      seed each time: over the iterations, the whole space is explored);
    - the points already computed at some levels but not at the highest one (e.g. a low-fidelity
      point without high fidelity): a higher level can be added there for the cost of the
      missing levels only. Points with a failed run are not candidates.
    Infeasible candidates and candidates close to a failed run are removed.

    Returns:
    - x_cand: numpy array (M, d), raw units; missing: boolean numpy array (M, L).
    """
    num_levels = len(data.costs)
    lows = np.array([b[0] for b in data.bounds], dtype=float)
    highs = np.array([b[1] for b in data.bounds], dtype=float)
    sobol = qmc.Sobol(d=data.dim, scramble=True, seed=seed)
    new_points = qmc.scale(sobol.random(n_candidates), lows, highs)
    new_missing = np.ones((len(new_points), num_levels), dtype=bool)

    partial_points, partial_missing = [], []
    for x in np.atleast_2d(data.x_dict.get(1, np.empty((0, data.dim)))):
        values = [stored_value(data, j, x) for j in range(1, num_levels + 1)]
        has_failure = any(v is not None and not np.isfinite(v) for v in values)
        missing = [v is None for v in values]
        if not has_failure and any(missing):
            partial_points.append(x)
            partial_missing.append(missing)

    x_cand = np.vstack([new_points] + ([np.array(partial_points)] if partial_points else []))
    missing = np.vstack([new_missing] + ([np.array(partial_missing)] if partial_missing else []))
    keep = np.ones(len(x_cand), dtype=bool)
    if feasibility is not None:
        keep &= np.asarray(feasibility(x_cand), dtype=bool).reshape(-1)
    if excluded is not None and len(excluded) > 0:
        spans = np.where(highs > lows, highs - lows, 1.0)
        distances = np.linalg.norm((x_cand[:, None, :] - excluded[None, :, :]) / spans, axis=2)
        keep &= np.min(distances, axis=1) >= EXCLUSION_RADIUS
    return x_cand[keep], missing[keep]


def find_next_point_alc(model: CoKriging, z2: list, x_cand: np.ndarray, missing: np.ndarray,
                        x_int: np.ndarray, costs: np.ndarray,
                        remaining_budget: float = np.inf) -> dict:
    """
    ALC choice ("Active Learning Cohn", Cohn 1996; MR-SUR of Stroh et al. 2022): the point AND
    the level are chosen together, as the pair (x, l) with the largest yield G_l(x) / C_l(x):
    the reduction of the integrated high-fidelity variance per unit of cost (Eqs. N5-N6).
    Compared with the maximum of the variance (ALM, find_next_point), a point at the border of
    the domain informs only the part of its neighbourhood that is inside the domain (about
    1 / 2^d of it in a corner): its integrated gain is small, so it is no longer preferred. And
    the integration points (e.g. the states of the VPP) now drive the choice of the point too.

    Args:
    - x_cand, missing: candidates and their missing levels (candidate_points).
    - x_int: integration points (N, d), raw units; costs: numpy array (L,).
    Returns:
    - dict with x, level (None if nothing is affordable), gains (G_1..G_L at x), ratios,
      missing (at x), criterion (best yield).
    """
    gains = adjust_gains(model, z2, x_cand, batched_imse_gains(model, x_cand, x_int, missing))
    cumulative_costs = np.cumsum(np.where(missing, costs, 0.0), axis=1)         # C_l, (M, L)
    candidate = missing & (cumulative_costs <= remaining_budget)
    ratios = np.where(candidate, gains / np.where(cumulative_costs > 0, cumulative_costs, 1.0),
                      -np.inf)
    if np.all(np.isinf(ratios)):
        return {"x": None, "level": None, "gains": None, "ratios": None, "missing": None,
                "criterion": -np.inf}
    best = np.unravel_index(np.argmax(ratios), ratios.shape)   # (candidate index, level index)
    return {"x": x_cand[best[0]], "level": int(best[1]) + 1, "gains": gains[best[0]],
            "ratios": ratios[best[0]], "missing": missing[best[0]],
            "criterion": float(ratios[best])}


# 3/3 ---------------------------------------------------------------------------------------------
# The loop: ask (fit, point, level), evaluate (run the codes), run (repeat)
# -------------------------------------------------------------------------------------------------
def stored_value(data, level: int, x: np.ndarray, tol: float = 1e-8):
    """
    Value already stored at a level for the point x.

    Returns:
    - float (NaN if the run failed), or None if x has not been evaluated at this level.
    """
    if level not in data.x_dict or len(data.y_dict.get(level, [])) == 0:
        return None
    points = np.atleast_2d(data.x_dict[level])
    distances = np.linalg.norm(points - np.asarray(x, dtype=float), axis=1)
    i = int(np.argmin(distances))
    if distances[i] > tol:
        return None
    return float(data.y_dict[level][i])


def evaluate_initial_design(data, simulator) -> float:
    """
    Runs the simulator at the points of an initial design created by
    ExperimentData.generate_nested_design (which stores the points without their values).
    Failed runs are stored as NaN, like ExperimentData.add_observation does.

    Returns:
    - total cost of these runs.
    """
    cost = 0.0
    for level in sorted(data.x_dict):
        values, metrics = [], []
        for x in np.atleast_2d(data.x_dict[level]):
            y, run_metrics = simulator.evaluate(x, level)
            values.append(float(y) if y is not None and np.isfinite(y) else np.nan)
            metrics.append(run_metrics or {})
            cost += data.costs[level - 1]
        data.y_dict[level] = np.array(values)
        data.metrics_dict[level] = metrics
    return cost


class SequentialCoKriging:
    """
    Sequential design of a co-kriging surrogate (Le Gratiet & Cannamela 2015, corrected).

    Example (see tests/ and examples/forrester_study.py):
        data = ExperimentData(bounds=[(0.0, 1.0)], costs=[1.0, 10.0])
        data.generate_nested_design([6, 3], seed=0)
        evaluate_initial_design(data, simulator)
        design = SequentialCoKriging(data, simulator, seed=0)
        history = design.run(budget=60.0)        # cost of the NEW runs only
        model = design.model                     # final CoKriging, fitted on all the data
    """

    def __init__(self, data, simulator, model_factory=None, integration_points: np.ndarray = None,
                 n_integration: int = N_INTEGRATION, adjust: bool = True, z2_max: float = None,
                 feasibility=None, q2_target: float = None, seed: int = None,
                 point_criterion: str = "variance", n_candidates: int = 2048,
                 de_popsize: int = 10, de_maxiter: int = 50, max_iterations: int = 1000):
        """
        Args:
        - data: ExperimentData with an evaluated NESTED design (CoKriging needs it).
        - simulator: object with evaluate(x, level) -> (value, metrics dict) (mfego BaseSimulator).
        - model_factory: function without argument returning a new, unfitted CoKriging
          (default: CoKriging(L, seed=seed), mfego defaults: Matern 5/2, MAP, learned rho).
        - integration_points: raw points (N, d) where the variance is integrated (Eq. N5). Default:
          N Sobol points of the whole design space (global accuracy). Later: the equilibrium
          states of the VPP (accuracy where the VPP works).
        - adjust: True uses the variance adjusted by the leave-one-out (N4); False the plain one.
        - z2_max: optional cap of the z^2 (e.g. 9).
        - feasibility: optional known-constraint function x (m, d) -> booleans (m,).
        - q2_target: optional stopping criterion: stop when the leave-one-out Q^2 at the highest
          level reaches it.
        - seed: seed of the Sobol points and of the differential evolution.
        - point_criterion: "variance" = point at the maximum of the (adjusted) variance, then the
          level (Le Gratiet & Cannamela, ALM); "alc" = point and level chosen together by the
          integrated gain per unit of cost (find_next_point_alc).
        - n_candidates: number of Sobol candidates of the ALC search (redrawn at each iteration).
        - de_popsize, de_maxiter: settings of the differential evolution (criterion "variance").
        - max_iterations: safety limit of run().
        """
        self.data = data
        self.simulator = simulator
        self.costs = np.asarray(data.costs, dtype=float)
        self.num_levels = len(self.costs)
        if model_factory is None:
            def model_factory():
                return CoKriging(self.num_levels, seed=seed)
        self.model_factory = model_factory
        self.adjust = bool(adjust)
        if point_criterion not in ("variance", "alc"):
            raise ValueError(f"point_criterion must be 'variance' or 'alc', got {point_criterion}")
        self.point_criterion = point_criterion
        self.n_candidates = n_candidates
        self.z2_max = z2_max
        self.feasibility = feasibility
        self.q2_target = q2_target
        self.rng = np.random.default_rng(seed)
        self.de_popsize = de_popsize
        self.de_maxiter = de_maxiter
        self.max_iterations = max_iterations

        if integration_points is None:
            lows = [b[0] for b in data.bounds]
            highs = [b[1] for b in data.bounds]
            sobol = qmc.Sobol(d=data.dim, scramble=True, seed=seed)
            integration_points = qmc.scale(sobol.random(n_integration), lows, highs)
        self.integration_points = np.atleast_2d(np.asarray(integration_points, dtype=float))

        self.model = None
        self.failed_points = np.empty((0, data.dim))   # raw points of failed runs
        self.cost_spent = 0.0                           # cost of the runs made by evaluate()
        self.history = []

    # --- step 1 and 2 ---------------------------------------------------------------------------
    def fit(self) -> CoKriging:
        """Step 1: a new CoKriging fitted on all the data (hyperparameters re-estimated)."""
        self.model = self.model_factory()
        self.model.fit(self.data)
        check_level_gps(self.model)
        return self.model

    def level_z2(self) -> list:
        """Step 2: z^2 of the leave-one-out of each level (N3); zeros if adjust is False."""
        if not self.adjust:
            return [np.zeros(len(gp.x_train)) for gp in self.model.gps]
        return [loo_squared_errors(gp, self.z2_max) for gp in self.model.gps]

    def loo_q2(self) -> float:
        """Q^2 of the leave-one-out at the highest level (1 = perfect, 0 = the mean)."""
        y, mean, variance = self.model.leave_one_out()
        return prediction_scores(y, mean, variance)["q2"]

    # --- steps 1 to 4: what to compute ----------------------------------------------------------
    def ask(self, remaining_budget: float = np.inf) -> dict:
        """
        Steps 1 to 4 of one iteration: fit, leave-one-out, point, level.

        Returns:
        - dict with: x (point, raw units), level (1..L or None), criterion (adjusted variance at x),
          gains (G_1..G_L), ratios (G_l / C_l), missing (levels not yet evaluated at x).
        """
        model = self.fit()                                       # step 1
        z2 = self.level_z2()                                     # step 2
        seed = int(self.rng.integers(2 ** 31 - 1))
        if self.point_criterion == "alc":                        # steps 3 and 4 together
            x_cand, missing = candidate_points(self.data, self.n_candidates, seed=seed,
                                               feasibility=self.feasibility,
                                               excluded=self.failed_points)
            return find_next_point_alc(model, z2, x_cand, missing, self.integration_points,
                                       self.costs, remaining_budget)
        x0, criterion = find_next_point(model, z2, self.data.bounds,      # step 3
                                        feasibility=self.feasibility,
                                        excluded=self.failed_points, seed=seed,
                                        popsize=self.de_popsize, maxiter=self.de_maxiter)
        missing = np.array([stored_value(self.data, j, x0) is None    # step 4
                            for j in range(1, self.num_levels + 1)])
        gains = imse_gains(model, z2, x0, self.integration_points, missing)
        level, ratios = choose_level(gains, self.costs, missing, remaining_budget)
        return {"x": x0, "level": level, "criterion": criterion, "gains": gains,
                "ratios": ratios, "missing": missing}

    # --- step 5: run the codes ------------------------------------------------------------------
    def evaluate(self, proposal: dict) -> tuple[float, list]:
        """
        Step 5: runs the levels 1, ..., level at the proposed point, cheapest first.
        - A level already evaluated there is not run again.
        - At the first failed run (value NaN or inf), the levels above are NOT run: the design
          stays nested, and the point is excluded from the next searches.

        Returns:
        - cost of the runs, list of the levels actually run.
        """
        x, level = proposal["x"], proposal["level"]
        cost, levels_run = 0.0, []
        for j in range(1, level + 1):
            value = stored_value(self.data, j, x)
            if value is not None:
                if np.isfinite(value):
                    continue          # already known: nothing to run at this level
                break                 # a known failure: the levels above cannot be run
            y, metrics = self.simulator.evaluate(x, j)
            self.data.add_observation(j, x, y, metrics)
            cost += self.costs[j - 1]
            levels_run.append(j)
            if y is None or not np.isfinite(y):
                self.failed_points = np.vstack([self.failed_points, x])
                break
        if not levels_run:
            # nothing could be run here (known failure below): never propose this point again
            self.failed_points = np.vstack([self.failed_points, x])
        self.cost_spent += cost
        return cost, levels_run

    # --- the loop -------------------------------------------------------------------------------
    def run(self, budget: float, verbose: bool = False) -> list:
        """
        Repeats ask / evaluate until the cost of the new runs reaches the budget, or until the
        leave-one-out Q^2 reaches q2_target. Then fits the final model on all the data.

        Args:
        - budget: cost allowed for the NEW runs (the initial design is not counted).
        Returns:
        - history: one dict per iteration (also stored in self.history).
        """
        cost_at_start = self.cost_spent
        for _ in range(self.max_iterations):
            remaining = budget - (self.cost_spent - cost_at_start)
            if remaining < self.costs.min():
                break                                   # not even one cheap run is affordable
            proposal = self.ask(remaining)
            q2 = self.loo_q2()                          # quality of the model of this iteration
            if self.q2_target is not None and q2 >= self.q2_target:
                break
            if proposal["level"] is None:
                break                                   # no affordable level at this point
            cost, levels_run = self.evaluate(proposal)
            self.history.append({
                "iteration": len(self.history) + 1,
                "x": proposal["x"].tolist(),
                "level": proposal["level"],
                "missing": proposal["missing"].tolist(),
                "levels_run": levels_run,
                "cost": cost,
                "cost_spent": self.cost_spent - cost_at_start,
                "criterion": proposal["criterion"],
                "gains": proposal["gains"].tolist(),
                "ratios": proposal["ratios"].tolist(),
                "q2_loo_before": q2,
                "rhos": [float(r) for r in self.model.rhos],
                "n_points": [len(self.data.get_training_data(j)[1])
                             for j in range(1, self.num_levels + 1)]})
            if verbose:
                h = self.history[-1]
                print(f"iteration {h['iteration']:3d} | x = {np.round(h['x'], 4)} | level "
                      f"{h['level']} | cost {h['cost_spent']:.1f} | points per level "
                      f"{h['n_points']} | Q2 LOO {q2:.3f}")
        self.fit()                                       # final model on all the data
        return self.history
