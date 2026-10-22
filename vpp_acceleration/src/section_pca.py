"""
Reduction of the Kulfan (CST) parameters of a section by principal component analysis (PCA).

Context: the torsor surrogate of the foil takes the section as an input. A Kulfan section has
16 weights (8 per side, the NeuralFoil format), but the sections the optimizer explores form a
smaller family (here the one of the last C-foil study). The PCA gives orthogonal "shape modes" of
that family, ordered by geometric importance, and the surrogate works with the k first scores.
Starting point: the PCA of Dim-Reduction/experiments.ipynb, with three changes:
  * the coordinates are only centred (geometric metric), not standardized;
  * the coordinates are computed EXACTLY by one matrix product, Eq. (1), instead of aerosandbox
    followed by an interpolation, section after section;
  * the decoding scores -> Kulfan weights is exact, Eq. (3).

1. Why the coordinates are a matrix product
   With a zero leading-edge weight and a fixed trailing-edge thickness TE, a Kulfan section with
   n weights per side is (aerosandbox convention)
       y_upper(x) = C(x) sum_i u_i B_i(x) + x TE / 2
       y_lower(x) = C(x) sum_i l_i B_i(x) - x TE / 2
   with the class function C(x) = sqrt(x) (1 - x) and the Bernstein polynomials
   B_i(x) = binom(n - 1, i) x^i (1 - x)^(n - 1 - i). At fixed stations x_1..x_p the 2p
   coordinates are therefore an AFFINE function of the 2n weights w = (u_0..u_n-1, l_0..l_n-1):
       y = A w + b                                                                   (1)
   with A = [[C B, 0], [0, C B]] (block diagonal) and b = (x TE / 2, -x TE / 2).

2. Centred PCA of the coordinates
   Sample N sections of the family, Y (N, 2p) their coordinates, y_mean the mean section.
   Singular value decomposition of the centred matrix: Y - y_mean = U S V^T. Column j of V is the
   shape mode v_j (a field of coordinates, unit norm), sigma_j = S_j / sqrt(N - 1) is the standard
   deviation of the family along v_j. Score of a section (whitened: unit variance on the family):
       z_j = (y - y_mean) . v_j / sigma_j                                            (2)
   Reconstruction with k modes: y_k = y_mean + sum_(j <= k) sigma_j z_j v_j. With uniform
   stations, the Euclidean distance between coordinate vectors is the discrete L2 norm of the
   gap along the chord: a geometric distance.

3. Exact decoding scores -> Kulfan weights
   Every mode is a combination of differences of sections, so by (1) v_j = A m_j for a weight
   vector m_j (the "weight mode"), found once by least squares (A has full column rank). Then
       w = w_mean + sum_(j <= k) sigma_j z_j m_j                                     (3)
   gives the weights of the reconstructed section exactly (no fit, no iteration): they go
   straight to NeuralFoil, to bdFoil or to a CAD.

4. Inputs of the surrogate
   The design space of the surrogate is a box of the k scores (score_bounds) plus a feasibility
   function: a score point is feasible when its decoded section belongs to the family
   (CFoilFamily.contains). Same convention as mfego and sequential_design.py: x (m, k) ->
   booleans (m,).

The number of modes k is chosen in examples/section_pca_study.py (aerodynamic tolerance with
NeuralFoil plus a geometric bound) and saved with SectionPCA.save.

A future family (another foil, other sections) only needs a class with the two methods
sample(n, seed) -> weights (n, 2n_w) and contains(weights) -> booleans; SectionPCA does not
depend on the family.

Import: as for sequential_design.py, vpp_acceleration must be on sys.path (from src.section_pca
import SectionPCA). This module does not use mfego.
"""
import json
from math import comb
from pathlib import Path

import numpy as np
from scipy.stats import qmc

# Number of weights per side of the NeuralFoil Kulfan input: the PCA works in this format.
N_WEIGHTS = 8
# Number of chordwise stations per side of the PCA (uniform, both ends included), as in
# Dim-Reduction/experiments.ipynb.
N_STATIONS = 200
# Stations of the maximum-thickness computation (uniform, fine enough for 1e-6 c).
THICKNESS_STATIONS = np.linspace(0.0, 1.0, 2001)
# Relative threshold under which a singular value counts as zero (rank of the family).
RANK_TOLERANCE = 1e-10


# 1/4 ---------------------------------------------------------------------------------------------
# Kulfan geometry as linear algebra (module docstring, section 1)
# -------------------------------------------------------------------------------------------------
def class_function(x: np.ndarray) -> np.ndarray:
    """C(x) = sqrt(x) (1 - x): round leading edge, sharp trailing edge (N1 = 0.5, N2 = 1)."""
    return np.sqrt(x) * (1.0 - x)


def bernstein_matrix(x: np.ndarray, n_weights: int) -> np.ndarray:
    """
    Bernstein polynomials of degree n_weights - 1 at the stations x.

    Returns:
    - numpy array (len(x), n_weights), column i = B_i(x) = binom(n-1, i) x^i (1 - x)^(n-1-i).
    """
    x = np.asarray(x, dtype=float)
    degree = n_weights - 1
    columns = [comb(degree, i) * x ** i * (1.0 - x) ** (degree - i) for i in range(n_weights)]
    return np.column_stack(columns)


def surface_matrix(stations: np.ndarray, n_weights: int, te_thickness: float) -> tuple:
    """
    A and b of Eq. (1): the coordinates of a section are y = A w + b, with y = (upper surface at
    the stations, then lower surface at the same stations) and w = (upper weights, lower weights).

    Args:
    - stations: chordwise stations x (p,), in [0, 1].
    - n_weights: number of weights per side n.
    - te_thickness: trailing-edge thickness (fraction of the chord).
    Returns:
    - A: numpy array (2p, 2n).
    - b: numpy array (2p,).
    """
    x = np.asarray(stations, dtype=float)
    shape = class_function(x)[:, None] * bernstein_matrix(x, n_weights)      # C(x) B_i(x), (p, n)
    zero = np.zeros_like(shape)
    matrix = np.block([[shape, zero], [zero, shape]])
    offset = np.concatenate([x * te_thickness / 2.0, -x * te_thickness / 2.0])
    return matrix, offset


def section_coordinates(weights: np.ndarray, stations: np.ndarray,
                        te_thickness: float) -> np.ndarray:
    """
    Coordinates of a batch of sections: one matrix product, Eq. (1).

    Args:
    - weights: numpy array (m, 2n) (upper weights, then lower weights) or (2n,).
    - stations: chordwise stations (p,).
    - te_thickness: trailing-edge thickness.
    Returns:
    - numpy array (m, 2p): y_upper at the stations, then y_lower at the stations.
    """
    weights = np.atleast_2d(np.asarray(weights, dtype=float))
    matrix, offset = surface_matrix(stations, weights.shape[1] // 2, te_thickness)
    return weights @ matrix.T + offset


def thickness_distribution(weights: np.ndarray, stations: np.ndarray,
                           te_thickness: float) -> np.ndarray:
    """Local thickness t(x) = y_upper(x) - y_lower(x) of a batch of sections, (m, p)."""
    coordinates = section_coordinates(weights, stations, te_thickness)
    n_stations = len(stations)
    return coordinates[:, :n_stations] - coordinates[:, n_stations:]


def elevation_matrix(n_weights: int, n_target: int) -> np.ndarray:
    """
    Bernstein degree elevation (same as pipelines/bdtoolbox_foil/geometry.py, copied here because
    that package imports bdToolbox): matrix E (n_target, n_weights) such that the weights E @ w
    describe EXACTLY the same polynomial sum_i w_i B_i(x) with n_target weights. One step from
    degree n to n + 1 gives v_i = i / (n + 1) w_(i-1) + (1 - i / (n + 1)) w_i, i = 0..n+1.
    """
    if n_target < n_weights:
        raise ValueError(f"cannot lower the degree ({n_weights} -> {n_target} weights)")
    matrix = np.eye(n_weights)
    for n in range(n_weights - 1, n_target - 1):        # degree n -> n + 1
        step = np.zeros((n + 2, n + 1))
        for i in range(n + 2):
            if i > 0:
                step[i, i - 1] = i / (n + 1)
            if i < n + 1:
                step[i, i] = 1.0 - i / (n + 1)
        matrix = step @ matrix
    return matrix


def kulfan_dict(weights: np.ndarray, te_thickness: float) -> dict:
    """
    Kulfan parameters of a batch of sections in the format of a vectorized NeuralFoil call
    (neuralfoil.get_aero_from_kulfan_parameters wants the weights of a batch as (n, m): one
    column per section).

    Args:
    - weights: numpy array (m, 2n) or (2n,).
    - te_thickness: trailing-edge thickness.
    Returns:
    - dict upper_weights (n, m), lower_weights (n, m), leading_edge_weight (m,),
      TE_thickness (m,).
    """
    weights = np.atleast_2d(np.asarray(weights, dtype=float))
    n = weights.shape[1] // 2
    return {"upper_weights": weights[:, :n].T, "lower_weights": weights[:, n:].T,
            "leading_edge_weight": np.zeros(len(weights)),
            "TE_thickness": np.full(len(weights), float(te_thickness))}


# 2/4 ---------------------------------------------------------------------------------------------
# The family of sections explored around the C-foil reference
# -------------------------------------------------------------------------------------------------
# Camber offset delta: in the last study (studies/cfoil_intens_kulfan_trim_2) it was solved by a
# yaw model, the yaw being a design variable. Here the yaw is an input of the torsor surrogate,
# so delta becomes a free parameter of the section. The bounds cover every delta solved during
# the last run (runs/1015_120910: -0.006 to 0.193, median 0.091).
DELTA_BOUNDS = (-0.01, 0.20)
# delta of the reference section (first C-foil study, CL2d(0 deg) = 0.45).
REFERENCE_DELTA = 0.108139


class CFoilFamily:
    """
    Sections explored around the C-foil reference, in the thickness/camber form of the last
    C-foil study (pipelines/bdtoolbox_foil/geometry.py), with 4 weights per side:
        thickness weights t_0..t_3,
        camber weights    c_i = delta + s_i (s_0 = 0),
        upper_i = c_i + t_i / 2,   lower_i = c_i - t_i / 2,
    leading-edge weight 0 and trailing-edge thickness TE fixed. The weights are then given in the
    NeuralFoil format (8 per side) by exact degree elevation.
    A section belongs to the family when its parameters lie in their box and its maximum
    thickness is at least min_thickness. The family is the image of an 8-dimensional box by an
    affine map: its PCA has at most 8 modes.

    Parameter vector (8,): PARAMETERS = (t_0, t_1, t_2, t_3, delta, s_1, s_2, s_3).
    """
    PARAMETERS = ("t_0", "t_1", "t_2", "t_3", "delta", "s_1", "s_2", "s_3")
    N_OWN = 4                      # weights per side of the family before the degree elevation

    def __init__(self, lower, upper, te_thickness: float, min_thickness: float,
                 reference=None, n_weights: int = N_WEIGHTS):
        """
        Args:
        - lower, upper: bounds of the 8 parameters, in the order PARAMETERS.
        - te_thickness: trailing-edge thickness (fraction of the chord).
        - min_thickness: minimum of the maximum thickness (fraction of the chord).
        - reference: parameters (8,) of the reference section (optional).
        - n_weights: number of weights per side of the output (8 for NeuralFoil).
        """
        self.lower = np.asarray(lower, dtype=float)
        self.upper = np.asarray(upper, dtype=float)
        if self.lower.shape != (8,) or self.upper.shape != (8,) or np.any(self.upper <= self.lower):
            raise ValueError("CFoilFamily needs 8 lower < upper bounds, order PARAMETERS")
        self.te_thickness = float(te_thickness)
        self.min_thickness = float(min_thickness)
        self.reference = None if reference is None else np.asarray(reference, dtype=float)
        self.n_weights = int(n_weights)
        # E (n_weights, 4): 4 weights per side -> n_weights weights per side, same surface
        self.elevation = elevation_matrix(self.N_OWN, self.n_weights)

    @classmethod
    def from_study_config(cls, path, delta_bounds: tuple = DELTA_BOUNDS,
                          reference_delta: float = REFERENCE_DELTA) -> "CFoilFamily":
        """
        Family of a C-foil study configuration (studies/cfoil_intens_kulfan_trim_2/
        config_cfoil_trim.json): bounds of t_i and s_i from "variables", TE thickness from
        "parametrization.fixed", min thickness from "constraints", reference from
        "objective.equal_lift_reference". delta gets delta_bounds (see DELTA_BOUNDS).
        """
        config = json.loads(Path(path).read_text(encoding="utf-8"))
        fixed = config["parametrization"]["fixed"]
        if config["parametrization"].get("form") != "thickness_camber" \
                or float(fixed.get("leading_edge_weight", 0.0)) != 0.0 \
                or float(fixed.get("s_0", 0.0)) != 0.0:
            raise ValueError("CFoilFamily expects the thickness_camber form, LE weight 0, s_0 = 0")
        bounds = {v["name"]: (v["lower"], v["upper"]) for v in config["variables"]}
        bounds["delta"] = tuple(delta_bounds)
        reference = dict(config["objective"]["equal_lift_reference"], delta=reference_delta)
        return cls(lower=[bounds[name][0] for name in cls.PARAMETERS],
                   upper=[bounds[name][1] for name in cls.PARAMETERS],
                   te_thickness=fixed["TE_thickness"],
                   min_thickness=config["constraints"]["min_thickness"],
                   reference=[reference[name] for name in cls.PARAMETERS])

    # --- parameters <-> weights ---
    def parameters_to_weights(self, parameters: np.ndarray) -> np.ndarray:
        """
        Kulfan weights (NeuralFoil format) of a batch of parameter vectors.

        Args:
        - parameters: numpy array (m, 8) or (8,), order PARAMETERS.
        Returns:
        - numpy array (m, 2 n_weights): upper weights, then lower weights.
        """
        p = np.atleast_2d(np.asarray(parameters, dtype=float))
        thickness = p[:, 0:4]                                        # t_0..t_3
        delta = p[:, 4:5]
        shape = np.column_stack([np.zeros(len(p)), p[:, 5:8]])       # s_0 = 0, s_1..s_3
        camber = delta + shape                                       # c_i = delta + s_i
        upper = camber + thickness / 2.0
        lower = camber - thickness / 2.0
        # degree elevation 4 -> n_weights weights per side: (m, 4) @ E^T = (m, n_weights)
        return np.column_stack([upper @ self.elevation.T, lower @ self.elevation.T])

    def weights_to_parameters(self, weights: np.ndarray) -> tuple:
        """
        Inverse of parameters_to_weights.

        Args:
        - weights: numpy array (m, 2 n_weights) or (2 n_weights,).
        Returns:
        - parameters: numpy array (m, 8), order PARAMETERS.
        - residual: numpy array (m,), largest gap between the weights and the weights of the
          parameters found: 0 when the section has the form of the family (up to rounding).
        """
        w = np.atleast_2d(np.asarray(weights, dtype=float))
        n = self.n_weights
        # E has full column rank: its pseudo-inverse gives back the 4 weights per side
        lowering = np.linalg.pinv(self.elevation)                    # (4, n_weights)
        upper, lower = w[:, :n] @ lowering.T, w[:, n:] @ lowering.T  # (m, 4) each
        thickness = upper - lower
        camber = (upper + lower) / 2.0
        delta = camber[:, 0]
        parameters = np.column_stack([thickness, delta, camber[:, 1:] - delta[:, None]])
        residual = np.max(np.abs(self.parameters_to_weights(parameters) - w), axis=1)
        return parameters, residual

    # --- membership and sampling ---
    def max_thickness(self, weights: np.ndarray) -> np.ndarray:
        """Maximum thickness (fraction of the chord) of a batch of sections, (m,)."""
        thickness = thickness_distribution(weights, THICKNESS_STATIONS, self.te_thickness)
        return thickness.max(axis=1)

    def contains(self, weights: np.ndarray, tolerance: float = 1e-9) -> np.ndarray:
        """
        Do the sections belong to the family?

        Args:
        - weights: numpy array (m, 2 n_weights) or (2 n_weights,).
        - tolerance: slack on the bounds and on the residual of the form.
        Returns:
        - numpy array of booleans (m,).
        """
        parameters, residual = self.weights_to_parameters(weights)
        in_box = np.all((parameters >= self.lower - tolerance)
                        & (parameters <= self.upper + tolerance), axis=1)
        thick_enough = self.max_thickness(weights) >= self.min_thickness - tolerance
        return in_box & thick_enough & (residual <= tolerance)

    def sample(self, n_sections: int, seed: int = 0) -> np.ndarray:
        """
        Sections of the family: Latin hypercube in the parameter box, the sections thinner than
        min_thickness are discarded, and the hypercube is redrawn bigger until n_sections remain.

        Returns:
        - numpy array (n_sections, 2 n_weights): Kulfan weights.
        """
        n_draw = 2 * n_sections
        while True:
            unit = qmc.LatinHypercube(d=8, seed=seed).random(n_draw)
            parameters = self.lower + unit * (self.upper - self.lower)
            weights = self.parameters_to_weights(parameters)
            weights = weights[self.max_thickness(weights) >= self.min_thickness]
            if len(weights) >= n_sections:
                return weights[:n_sections]
            n_draw *= 2

    def reference_weights(self) -> np.ndarray:
        """Kulfan weights (2 n_weights,) of the reference section."""
        if self.reference is None:
            raise ValueError("this family has no reference section")
        return self.parameters_to_weights(self.reference)[0]


def second_difference_matrix(n_weights: int) -> np.ndarray:
    """D2 (n - 2, n): (D2 v)_i = v_i - 2 v_(i+1) + v_(i+2), the second differences of v."""
    matrix = np.zeros((n_weights - 2, n_weights))
    for i in range(n_weights - 2):
        matrix[i, i:i + 3] = (1.0, -2.0, 1.0)
    return matrix


def wiggliness(weights: np.ndarray) -> np.ndarray:
    """
    Wiggliness of the last C-foil study (study_trim.section_properties): sum of the squared
    second differences of the upper weights and of the lower weights, (m,).
    """
    weights = np.atleast_2d(np.asarray(weights, dtype=float))
    n = weights.shape[1] // 2
    return (np.sum(np.diff(weights[:, :n], n=2, axis=1) ** 2, axis=1)
            + np.sum(np.diff(weights[:, n:], n=2, axis=1) ** 2, axis=1))


class SmoothKulfanFamily:
    """
    Sections with 16 free Kulfan weights (8 per side) around a reference, kept smooth.

    Every section is written in thickness / camber form, t = u - l and c = (u + l) / 2 (8 weights
    each, u and l the upper and lower weights). Each of the two vectors v = t or c is split, EXACTLY,
    into
        v = Q g + D2+ d,
    * D2 (6, 8) the second-difference matrix and d = D2 v its 6 second differences;
    * Q (8, 2) an orthonormal basis of the vectors with no second difference (constant and linear
      sequences), and g = Q^T v (2 numbers): the LEVEL and the SLOPE of the weights;
    * D2+ the pseudo-inverse of D2 (D2+ D2 is the projection orthogonal to Q).
    The level and the slope cost no wiggliness: they are the large moves (thickness level,
    thickness forward / aft, camber level, camber position). They move in a box
    (level_slope_lower, level_slope_upper). The second differences carry the finer shape: the
    wiggliness (sum of the squared second differences of u and of l) is at most max_wiggliness.
    Plus, as for CFoilFamily: maximum thickness >= min_thickness, and the surfaces do not cross.

    Level / slope vector (4,): LEVEL_SLOPE = (thickness level, thickness slope, camber level,
    camber slope).
    """
    LEVEL_SLOPE = ("thickness_level", "thickness_slope", "camber_level", "camber_slope")

    def __init__(self, reference_weights, level_slope_lower, level_slope_upper,
                 max_wiggliness: float, te_thickness: float, min_thickness: float):
        """
        Args:
        - reference_weights: Kulfan weights (16,) of the reference section.
        - level_slope_lower, level_slope_upper: box of the 4 level / slope numbers (LEVEL_SLOPE).
        - max_wiggliness: largest wiggliness of the family.
        - te_thickness, min_thickness: fraction of the chord.
        """
        self.reference = np.asarray(reference_weights, dtype=float)
        self.n_weights = len(self.reference) // 2
        self.lower = np.asarray(level_slope_lower, dtype=float)
        self.upper = np.asarray(level_slope_upper, dtype=float)
        self.max_wiggliness = float(max_wiggliness)
        self.te_thickness = float(te_thickness)
        self.min_thickness = float(min_thickness)
        n = self.n_weights
        self.second_difference = second_difference_matrix(n)                # D2 (n - 2, n)
        self.pseudo_inverse = np.linalg.pinv(self.second_difference)         # D2+ (n, n - 2)
        index = np.linspace(-1.0, 1.0, n)
        self.basis = np.linalg.qr(np.column_stack([np.ones(n), index]))[0]  # Q (n, 2)
        # QR may flip the signs: make "level" and "slope" increase with the mean / the index
        self.basis *= np.sign(self.basis.T @ np.column_stack([np.ones(n), index])).diagonal()

    @classmethod
    def around(cls, family: CFoilFamily, wiggliness_ratio: float = 2.0, margin: float = 0.25,
               n_sections: int = 2 ** 14, seed: int = 0) -> "SmoothKulfanFamily":
        """
        Family around the reference of a CFoilFamily: the box of the level / slope numbers is the
        range they take in the CFoilFamily (sample of n_sections), widened by `margin` x its width
        on each side; wiggliness at most wiggliness_ratio x that of the reference.
        """
        reference = family.reference_weights()
        smooth = cls(reference, np.zeros(4), np.ones(4), 0.0, family.te_thickness,
                     family.min_thickness)
        level_slope = smooth.split(family.sample(n_sections, seed))[0]
        low, high = level_slope.min(axis=0), level_slope.max(axis=0)
        width = high - low
        return cls(reference, low - margin * width, high + margin * width,
                   wiggliness_ratio * float(wiggliness(reference)[0]), family.te_thickness,
                   family.min_thickness)

    # --- weights <-> (level / slope, second differences) ---
    def split(self, weights: np.ndarray) -> tuple:
        """
        Level / slope numbers and second differences of a batch of sections.

        Returns:
        - level_slope: numpy array (m, 4), order LEVEL_SLOPE.
        - differences: numpy array (m, 2 (n - 2)): second differences of t, then of c.
        """
        w = np.atleast_2d(np.asarray(weights, dtype=float))
        n = self.n_weights
        thickness = w[:, :n] - w[:, n:]
        camber = (w[:, :n] + w[:, n:]) / 2.0
        level_slope = np.column_stack([thickness @ self.basis, camber @ self.basis])
        differences = np.column_stack([thickness @ self.second_difference.T,
                                       camber @ self.second_difference.T])
        return level_slope, differences

    def join(self, level_slope: np.ndarray, differences: np.ndarray) -> np.ndarray:
        """Inverse of split: Kulfan weights (m, 16)."""
        level_slope = np.atleast_2d(level_slope)
        differences = np.atleast_2d(differences)
        k = self.n_weights - 2
        thickness = level_slope[:, 0:2] @ self.basis.T + differences[:, :k] @ self.pseudo_inverse.T
        camber = level_slope[:, 2:4] @ self.basis.T + differences[:, k:] @ self.pseudo_inverse.T
        return np.column_stack([camber + thickness / 2.0, camber - thickness / 2.0])

    # --- membership and sampling ---
    def thickness_ok(self, weights: np.ndarray, tolerance: float = 1e-9) -> np.ndarray:
        """Maximum thickness >= min_thickness and no crossing surfaces, (m,)."""
        thickness = thickness_distribution(weights, THICKNESS_STATIONS, self.te_thickness)
        return (thickness.max(axis=1) >= self.min_thickness - tolerance) \
            & (thickness.min(axis=1) >= -tolerance)

    def contains(self, weights: np.ndarray, tolerance: float = 1e-9) -> np.ndarray:
        """Do the sections belong to the family? Booleans (m,)."""
        level_slope = self.split(weights)[0]
        in_box = np.all((level_slope >= self.lower - tolerance)
                        & (level_slope <= self.upper + tolerance), axis=1)
        smooth = wiggliness(weights) <= self.max_wiggliness * (1.0 + tolerance)
        return in_box & smooth & self.thickness_ok(weights, tolerance)

    def sample(self, n_sections: int, seed: int = 0) -> np.ndarray:
        """
        Sections of the family, uniform in it: the level / slope numbers uniform in their box,
        and the second differences of u and l, (D2 u, D2 l) (12 numbers), uniform in the ball
        of radius sqrt(max_wiggliness) (a Gaussian direction, a radius R U^(1/12)); the sections
        that fail thickness_ok are discarded, and the draw is repeated until n_sections remain.

        Returns:
        - numpy array (n_sections, 16): Kulfan weights.
        """
        rng = np.random.default_rng(seed)
        k = self.n_weights - 2
        kept = []
        while sum(len(w) for w in kept) < n_sections:
            n_draw = 2 * n_sections
            level_slope = self.lower + rng.random((n_draw, 4)) * (self.upper - self.lower)
            direction = rng.standard_normal((n_draw, 2 * k))
            direction /= np.linalg.norm(direction, axis=1, keepdims=True)
            radius = np.sqrt(self.max_wiggliness) * rng.random((n_draw, 1)) ** (1.0 / (2 * k))
            upper_lower = direction * radius                      # (D2 u, D2 l) in the ball
            # D2 t = D2 u - D2 l and D2 c = (D2 u + D2 l) / 2
            differences = np.column_stack([upper_lower[:, :k] - upper_lower[:, k:],
                                           (upper_lower[:, :k] + upper_lower[:, k:]) / 2.0])
            weights = self.join(level_slope, differences)
            kept.append(weights[self.thickness_ok(weights)])
        return np.vstack(kept)[:n_sections]

    def reference_weights(self) -> np.ndarray:
        """Kulfan weights (16,) of the reference section."""
        return self.reference.copy()


DIM_REDUCTION_NOTEBOOK = Path(__file__).resolve().parents[3] / "Dim-Reduction" / "experiments.ipynb"


def notebook_bounds(path=DIM_REDUCTION_NOTEBOOK) -> tuple:
    """
    Bounds of the 16 Kulfan weights of Dim-Reduction/experiments.ipynb (cell 1: upper_min,
    upper_max, lower_min, lower_max), read in the notebook itself so that nothing is copied by hand.

    Returns:
    - lower (16,), upper (16,): upper weights then lower weights.
    """
    import ast  # noqa: PLC0415
    cells = json.loads(Path(path).read_text(encoding="utf-8"))["cells"]
    values = {}
    for cell in cells:
        for line in "".join(cell.get("source", "")).splitlines():
            name, _, value = line.partition("=")
            name = name.strip()
            if name in ("upper_min", "upper_max", "lower_min", "lower_max"):
                values[name] = ast.literal_eval(value.split("#")[0].strip())
    missing = {"upper_min", "upper_max", "lower_min", "lower_max"} - set(values)
    if missing:
        raise ValueError(f"{path}: bounds {sorted(missing)} not found")
    return (np.array(values["upper_min"] + values["lower_min"], dtype=float),
            np.array(values["upper_max"] + values["lower_max"], dtype=float))


def section_ixx(weights: np.ndarray, te_thickness: float) -> np.ndarray:
    """
    Second moment of area Ixx of a batch of sections about their centroid (chord 1, camber
    included), the bending-stiffness constraint of studies/cfoil_intens_kulfan_trim_2
    (study_trim.section_properties): with the thickness tau(x) = y_upper - y_lower and the camber
    line y_c(x) = (y_upper + y_lower) / 2, by vertical strips,
        A = int tau,   y_g = int tau y_c / A,   Ixx = int [tau^3 / 12 + tau (y_c - y_g)^2].

    Returns:
    - numpy array (m,).
    """
    x = THICKNESS_STATIONS
    coordinates = section_coordinates(weights, x, te_thickness)
    upper, lower = coordinates[:, :len(x)], coordinates[:, len(x):]
    tau, camber = upper - lower, (upper + lower) / 2.0
    # scipy trapezoid: np.trapz is gone in numpy 2 (environment .venv-benchmark)
    from scipy.integrate import trapezoid  # noqa: PLC0415
    area = trapezoid(tau, x, axis=1)
    y_g = trapezoid(tau * camber, x, axis=1) / area
    return trapezoid(tau ** 3 / 12.0 + tau * (camber - y_g[:, None]) ** 2, x, axis=1)


class KulfanBoxFamily:
    """
    Sections whose 16 Kulfan weights (8 per side) lie in a box, e.g. the box of
    Dim-Reduction/experiments.ipynb (KulfanBoxFamily.from_notebook()), leading-edge weight 0 and a
    fixed trailing-edge thickness. A section belongs to the family when its weights are in the
    box, its surfaces do not cross (thickness >= 0 everywhere) and, optionally, `extra_check`
    accepts it (e.g. the NeuralFoil confidence filter of meta_foil.confident).
    """

    def __init__(self, lower, upper, te_thickness: float, extra_check=None):
        self.lower = np.asarray(lower, dtype=float)
        self.upper = np.asarray(upper, dtype=float)
        self.te_thickness = float(te_thickness)
        self.n_weights = len(self.lower) // 2
        self.extra_check = extra_check

    @classmethod
    def from_notebook(cls, path=DIM_REDUCTION_NOTEBOOK, te_thickness: float = 0.00197,
                      extra_check=None) -> "KulfanBoxFamily":
        lower, upper = notebook_bounds(path)
        return cls(lower, upper, te_thickness, extra_check)

    @classmethod
    def centered(cls, reference: np.ndarray, widths: np.ndarray, te_thickness: float,
                 extra_check=None) -> "KulfanBoxFamily":
        """Box of the given widths (16,) centred on a reference section (16 weights), e.g. the
        widths of the Dim-Reduction box around the current C-foil section."""
        reference, widths = np.asarray(reference, float), np.asarray(widths, float)
        return cls(reference - widths / 2.0, reference + widths / 2.0, te_thickness, extra_check)

    def thickness_ok(self, weights: np.ndarray, tolerance: float = 1e-9) -> np.ndarray:
        """No crossing surfaces (local thickness >= 0 at every station), (m,)."""
        thickness = thickness_distribution(weights, THICKNESS_STATIONS, self.te_thickness)
        return thickness.min(axis=1) >= -tolerance

    def contains(self, weights: np.ndarray, tolerance: float = 1e-9) -> np.ndarray:
        weights = np.atleast_2d(weights)
        ok = np.all((weights >= self.lower - tolerance) & (weights <= self.upper + tolerance),
                    axis=1) & self.thickness_ok(weights, tolerance)
        if self.extra_check is not None and ok.any():
            ok[ok] = self.extra_check(weights[ok])
        return ok

    def sample(self, n_sections: int, seed: int = 0) -> np.ndarray:
        """Latin hypercube in the box, kept if contains(); redrawn bigger until n remain."""
        n_draw, kept = 2 * n_sections, []
        attempt = 0
        while sum(len(k) for k in kept) < n_sections:
            unit = qmc.LatinHypercube(d=len(self.lower), seed=seed + 7919 * attempt).random(n_draw)
            weights = self.lower + unit * (self.upper - self.lower)
            kept.append(weights[self.contains(weights)])
            attempt += 1
        return np.vstack(kept)[:n_sections]


# 3/4 ---------------------------------------------------------------------------------------------
# Centred PCA with an exact decoding (module docstring, sections 2 and 3)
# -------------------------------------------------------------------------------------------------
class SectionPCA:
    """
    Centred PCA of the coordinates of a family of Kulfan sections, with whitened scores and the
    exact decoding scores -> weights.

    Attributes (after fit):
    - stations (p,), matrix A (2p, 2n), offset b (2p,): Eq. (1).
    - mean_weights (2n,), mean_coordinates (2p,): the mean section (y_mean = A w_mean + b).
    - modes (2p, r): shape modes v_j (unit norm), r = rank of the family.
    - weight_modes (2n, r): m_j such that v_j = A m_j.
    - sigmas (r,): standard deviation of the family along each mode.
    - variance_ratio (r,): share of the total variance carried by each mode.
    - n_modes: number k of modes used by encode / decode (default r; set it after the choice of
      k, e.g. pca.n_modes = 6).
    """

    def __init__(self, te_thickness: float, n_weights: int = N_WEIGHTS, stations=None):
        self.te_thickness = float(te_thickness)
        self.n_weights = int(n_weights)
        self.stations = (np.linspace(0.0, 1.0, N_STATIONS) if stations is None
                         else np.asarray(stations, dtype=float))
        self.matrix, self.offset = surface_matrix(self.stations, self.n_weights,
                                                  self.te_thickness)
        self.mean_weights = self.mean_coordinates = None
        self.modes = self.weight_modes = self.sigmas = self.variance_ratio = None
        self.n_modes = None
        self.n_sections = 0

    def coordinates(self, weights: np.ndarray) -> np.ndarray:
        """Coordinates y = A w + b (m, 2p) of a batch of sections (m, 2n), Eq. (1)."""
        return np.atleast_2d(np.asarray(weights, dtype=float)) @ self.matrix.T + self.offset

    def fit(self, weights: np.ndarray) -> "SectionPCA":
        """
        PCA of a sample of the family.

        Args:
        - weights: numpy array (N, 2n), Kulfan weights of the sample (e.g. CFoilFamily.sample).
        Returns:
        - self.
        """
        weights = np.atleast_2d(np.asarray(weights, dtype=float))
        if weights.shape[1] != 2 * self.n_weights:
            raise ValueError(f"expected {2 * self.n_weights} weights per section, "
                             f"got {weights.shape[1]}")
        coordinates = self.coordinates(weights)
        self.n_sections = len(weights)
        self.mean_weights = weights.mean(axis=0)
        self.mean_coordinates = coordinates.mean(axis=0)
        # SVD of the centred coordinates: Y - y_mean = U S V^T (rows of vt = modes)
        _, singular, vt = np.linalg.svd(coordinates - self.mean_coordinates,
                                        full_matrices=False)
        rank = int(np.sum(singular > RANK_TOLERANCE * singular[0]))
        modes = vt[:rank].T                                          # (2p, r)
        # the sign of a mode is arbitrary: make its largest component positive (reproducible)
        largest = np.argmax(np.abs(modes), axis=0)
        modes = modes * np.sign(modes[largest, np.arange(rank)])
        self.modes = modes
        self.sigmas = singular[:rank] / np.sqrt(self.n_sections - 1)
        self.variance_ratio = singular[:rank] ** 2 / np.sum(singular ** 2)
        # weight modes m_j: A m_j = v_j, solved by least squares (exact, see Eq. (3))
        self.weight_modes = np.linalg.lstsq(self.matrix, self.modes, rcond=None)[0]
        self.n_modes = rank
        return self

    @property
    def rank(self) -> int:
        """Number of modes of the family (non-zero singular values)."""
        return 0 if self.modes is None else self.modes.shape[1]

    def _k(self, n_modes) -> int:
        """Number of modes to use: n_modes if given, otherwise self.n_modes."""
        k = self.n_modes if n_modes is None else int(n_modes)
        if self.modes is None:
            raise RuntimeError("SectionPCA: call fit (or load) first")
        if not 1 <= k <= self.rank:
            raise ValueError(f"n_modes must be in 1..{self.rank}, got {k}")
        return k

    def encode(self, weights: np.ndarray, n_modes: int = None) -> np.ndarray:
        """
        Whitened scores of a batch of sections, Eq. (2): z_j = (y - y_mean) . v_j / sigma_j.

        Args:
        - weights: numpy array (m, 2n) or (2n,).
        - n_modes: number of modes k (default self.n_modes).
        Returns:
        - numpy array (m, k).
        """
        k = self._k(n_modes)
        centred = self.coordinates(weights) - self.mean_coordinates
        return centred @ self.modes[:, :k] / self.sigmas[:k]

    def decode(self, scores: np.ndarray) -> np.ndarray:
        """
        Kulfan weights of the sections of given scores, Eq. (3) (k = number of columns):
            w = w_mean + sum_(j <= k) sigma_j z_j m_j.

        Args:
        - scores: numpy array (m, k) or (k,).
        Returns:
        - numpy array (m, 2n).
        """
        scores = np.atleast_2d(np.asarray(scores, dtype=float))
        k = self._k(scores.shape[1])
        return self.mean_weights + (scores * self.sigmas[:k]) @ self.weight_modes[:, :k].T

    def reconstruct(self, weights: np.ndarray, n_modes: int = None) -> np.ndarray:
        """Weights of the sections projected on the k first modes: decode(encode(w))."""
        return self.decode(self.encode(weights, n_modes))

    def geometric_error(self, weights: np.ndarray, n_modes: int = None) -> np.ndarray:
        """
        Largest gap (fraction of the chord) between each section and its projection on the k first
        modes, over the stations of the PCA and both surfaces, (m,).
        """
        gap = self.coordinates(weights) - self.coordinates(self.reconstruct(weights, n_modes))
        return np.max(np.abs(gap), axis=1)

    # --- save / load ---
    def to_dict(self) -> dict:
        """Everything needed to rebuild the PCA (JSON-compatible)."""
        return {"te_thickness": self.te_thickness, "n_weights": self.n_weights,
                "stations": self.stations.tolist(), "n_sections": self.n_sections,
                "n_modes": self.n_modes, "mean_weights": self.mean_weights.tolist(),
                "modes": self.modes.tolist(), "weight_modes": self.weight_modes.tolist(),
                "sigmas": self.sigmas.tolist(), "variance_ratio": self.variance_ratio.tolist()}

    @classmethod
    def from_dict(cls, data: dict) -> "SectionPCA":
        """Inverse of to_dict."""
        pca = cls(te_thickness=data["te_thickness"], n_weights=data["n_weights"],
                  stations=data["stations"])
        pca.n_sections = int(data["n_sections"])
        pca.mean_weights = np.asarray(data["mean_weights"], dtype=float)
        pca.mean_coordinates = pca.coordinates(pca.mean_weights)[0]
        pca.modes = np.asarray(data["modes"], dtype=float)
        pca.weight_modes = np.asarray(data["weight_modes"], dtype=float)
        pca.sigmas = np.asarray(data["sigmas"], dtype=float)
        pca.variance_ratio = np.asarray(data["variance_ratio"], dtype=float)
        pca.n_modes = int(data["n_modes"])
        return pca

    def save(self, path) -> Path:
        """Writes the PCA to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict()), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path) -> "SectionPCA":
        """Reads a PCA written by save."""
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


# 4/4 ---------------------------------------------------------------------------------------------
# The scores as inputs of the surrogate (module docstring, section 4)
# -------------------------------------------------------------------------------------------------
def score_bounds(pca: SectionPCA, family, n_sections: int = 2 ** 14, seed: int = 1,
                 margin: float = 0.0) -> np.ndarray:
    """
    Box of the k first scores of the family: per mode, [min, max] of the scores of a sample,
    widened by margin x width on each side.

    Returns:
    - numpy array (k, 2): one [lower, upper] row per mode (format of ExperimentData.bounds).
    """
    scores = pca.encode(family.sample(n_sections, seed))
    low, high = scores.min(axis=0), scores.max(axis=0)
    width = high - low
    return np.column_stack([low - margin * width, high + margin * width])


def feasibility_function(pca: SectionPCA, family):
    """
    Known constraint of the surrogate in the score space: x (m, k) -> booleans (m,), True when
    the decoded section belongs to the family. To give to SequentialCoKriging(feasibility=...).
    """
    def feasible(scores: np.ndarray) -> np.ndarray:
        return family.contains(pca.decode(np.atleast_2d(scores)))
    return feasible
