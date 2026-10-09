"""
Section geometry: mapping of the normalized design variables to physical parameters and section
coordinates (NACA 4-digit continuous, Kulfan/CST through aerosandbox, PARSEC through bdSec).

Coordinates follow the Selig order expected by XFOIL and NeuralFoil: trailing edge -> upper
surface -> leading edge -> lower surface -> trailing edge, chord 1.

Kulfan (CST) sections, two equivalent forms (parametrization "form"):
* "weights" (default): upper_0..upper_n and lower_0..lower_n, the surface weights.
* "thickness_camber": t_0..t_n (thickness weights) and the camber weights c_i = delta + s_i
  (s_0 = 0 unless given). Then upper_i = c_i + t_i / 2 and lower_i = c_i - t_i / 2, so that
      thickness(x) = C(x) sum_i t_i B_i(x) + x TE_thickness,
      camber(x)    = C(x) sum_i c_i B_i(x)              (+ the leading-edge mode, if any),
  with the class function C(x) = sqrt(x) (1 - x) and the Bernstein polynomials B_i. The
  thickness only depends on the t_i (cheap, affine constraints) and t_i >= 0 rules out crossing
  surfaces. A uniform offset delta adds the basic camber mode delta C(x) (the B_i sum to 1):
  delta can be a design variable, a fixed value, or solved so that the section meets a 2D lift
  constraint (section_constraint.py).
"""
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import bridge


@dataclass
class Section:
    """A 2D section: coordinates, physical parameters and characteristics."""
    name: str
    coordinates: np.ndarray
    parameters: dict
    thickness: float
    # Kulfan parameters given directly to NeuralFoil (8 weights per side), None otherwise
    kulfan: dict = None


# 1/4 ---------------------------------------------------------------------------------------------
def to_physical(design_point: np.ndarray, variables: list) -> dict:
    """
    Maps a point of [0, 1]^d to the physical bounds: p = lower + u (upper - lower).

    Args:
    - design_point: normalized design point (d,).
    - variables: list of config.Variable.
    Returns:
    - dict name -> physical value.
    """
    u = np.clip(np.asarray(design_point, dtype=float).reshape(-1), 0.0, 1.0)
    if len(u) != len(variables):
        raise ValueError(f"design point of dimension {len(u)} for {len(variables)} variables")
    return {v.name: float(v.lower + ui * (v.upper - v.lower)) for v, ui in zip(variables, u)}


def cosine_spacing(n_points: int) -> np.ndarray:
    """Chordwise stations clustered at the leading and trailing edges."""
    return 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, n_points)))


# 2/4 ---------------------------------------------------------------------------------------------
def naca4_coordinates(m_camber: float, p_position: float, t_thickness: float,
                      n_points: int = 161) -> np.ndarray:
    """
    Continuous NACA 4-digit section (same equations as
    example/hydrofoil_optim/optim_neuralfoil.py:generate_continuous_naca4, with a cosine
    spacing for a better leading-edge resolution).
    """
    x = cosine_spacing(n_points)
    p = min(max(p_position, 1e-3), 1.0 - 1e-3)
    y_c = np.where(x < p, m_camber / p ** 2 * (2 * p * x - x ** 2),
                   m_camber / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * x - x ** 2))
    dyc_dx = np.where(x < p, 2 * m_camber / p ** 2 * (p - x),
                      2 * m_camber / (1 - p) ** 2 * (p - x))
    theta = np.arctan(dyc_dx)
    y_t = 5 * t_thickness * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x ** 2
                             + 0.2843 * x ** 3 - 0.1015 * x ** 4)
    x_u, y_u = x - y_t * np.sin(theta), y_c + y_t * np.cos(theta)
    x_l, y_l = x + y_t * np.sin(theta), y_c - y_t * np.cos(theta)
    return np.column_stack((np.concatenate((x_u[::-1], x_l[1:])),
                            np.concatenate((y_u[::-1], y_l[1:]))))


KULFAN_FORMS = ("weights", "thickness_camber")
# number of weights per side of the NeuralFoil Kulfan input
NEURALFOIL_N_WEIGHTS = 8


def _indexed(parameters: dict, prefix: str) -> dict:
    """{i: value} of the parameters named <prefix><i> (e.g. t_0, t_1...)."""
    out = {}
    for key, value in parameters.items():
        if key.startswith(prefix) and key[len(prefix):].isdigit():
            out[int(key[len(prefix):])] = float(value)
    return out


def _as_vector(indexed: dict, name: str) -> np.ndarray:
    """Vector of an indexed family, checking that the indices are 0..n-1 without gaps."""
    if sorted(indexed) != list(range(len(indexed))):
        raise ValueError(f"kulfan parametrization: {name} indices must be 0..n-1, got "
                         f"{sorted(indexed)}")
    return np.array([indexed[i] for i in range(len(indexed))], dtype=float)


def kulfan_parameters(parameters: dict, form: str = "weights") -> dict:
    """
    Kulfan (CST) parameters (aerosandbox convention) from flat physical parameters.

    Args:
    - parameters: dict name -> value. form "weights": upper_0..upper_n, lower_0..lower_n;
      form "thickness_camber": t_0..t_n, s_0..s_n (missing s_i = 0) and delta (default 0).
      Both forms: leading_edge_weight (default 0) and TE_thickness (default 0).
    - form: "weights" or "thickness_camber" (see the module docstring).
    Returns:
    - dict upper_weights, lower_weights, leading_edge_weight, TE_thickness.
    """
    if form == "weights":
        upper = _as_vector(_indexed(parameters, "upper_"), "upper_i")
        lower = _as_vector(_indexed(parameters, "lower_"), "lower_i")
        if not upper.size or upper.size != lower.size:
            raise ValueError("kulfan parametrization needs as many upper_i as lower_i variables")
    elif form == "thickness_camber":
        thickness = _as_vector(_indexed(parameters, "t_"), "t_i")
        if not thickness.size:
            raise ValueError("kulfan thickness_camber form needs t_0..t_n")
        shape = _indexed(parameters, "s_")
        if any(i >= thickness.size for i in shape):
            raise ValueError("kulfan thickness_camber form: s_i index beyond the t_i ones")
        camber = float(parameters.get("delta", 0.0)) + np.array(
            [shape.get(i, 0.0) for i in range(thickness.size)])
        upper, lower = camber + thickness / 2.0, camber - thickness / 2.0
    else:
        raise ValueError(f"kulfan form must be in {KULFAN_FORMS}, got {form!r}")
    return {"upper_weights": upper, "lower_weights": lower,
            "leading_edge_weight": float(parameters.get("leading_edge_weight", 0.0)),
            "TE_thickness": float(parameters.get("TE_thickness", 0.0))}


def elevation_matrix(n_weights: int, n_target: int) -> np.ndarray:
    """
    Bernstein degree elevation: matrix E (n_target, n_weights) such that the weights E @ w of
    degree n_target - 1 describe EXACTLY the same polynomial sum_i w_i B_i(x) as the weights w
    of degree n_weights - 1. One elevation step from degree n to n + 1 gives the new weights
    v_i = i / (n + 1) w_(i-1) + (1 - i / (n + 1)) w_i, i = 0..n+1.
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


def neuralfoil_kulfan(kulfan: dict) -> dict:
    """
    Kulfan parameters in the NeuralFoil input format (8 weights per side), obtained by exact
    degree elevation. Returns None when it is not exact (fewer than 8 weights with a non-zero
    leading-edge weight, whose mode x (1 - x)^(n + 0.5) depends on n, or more than 8 weights):
    the section is then given to NeuralFoil by its coordinates.
    """
    n = len(kulfan["upper_weights"])
    if n == NEURALFOIL_N_WEIGHTS:
        return kulfan
    if n > NEURALFOIL_N_WEIGHTS or kulfan["leading_edge_weight"] != 0.0:
        return None
    matrix = elevation_matrix(n, NEURALFOIL_N_WEIGHTS)
    return {**kulfan, "upper_weights": matrix @ np.asarray(kulfan["upper_weights"]),
            "lower_weights": matrix @ np.asarray(kulfan["lower_weights"])}


def kulfan_coordinates(kulfan: dict, n_points: int = 161) -> np.ndarray:
    """Coordinates of a Kulfan section (aerosandbox KulfanAirfoil)."""
    import aerosandbox as asb  # noqa: PLC0415
    airfoil = asb.KulfanAirfoil(name="kulfan", **kulfan)
    return np.asarray(airfoil.to_airfoil(n_coordinates_per_side=n_points).coordinates)


def cst_basis(x: np.ndarray, n_weights: int) -> np.ndarray:
    """
    CST basis C(x) B_i(x) (N1 = 0.5, N2 = 1, the aerosandbox defaults) at the chordwise
    stations x: array (len(x), n_weights). A CST surface is cst_basis(x, n) @ weights.
    """
    from scipy.special import comb  # noqa: PLC0415
    x = np.asarray(x, dtype=float).reshape(-1, 1)
    degree = n_weights - 1
    i = np.arange(n_weights).reshape(1, -1)
    bernstein = comb(degree, i) * x ** i * (1.0 - x) ** (degree - i)
    return np.sqrt(x) * (1.0 - x) * bernstein


def cst_thickness(x: np.ndarray, thickness_weights: np.ndarray,
                  te_thickness=0.0) -> np.ndarray:
    """
    Vectorized thickness distribution of CST sections (vertical distance between the surfaces,
    the leading-edge mode cancels): thickness_weights (m, n) -> (m, len(x)); te_thickness is
    one value for every section or one value per section (m,).
    """
    weights = np.atleast_2d(np.asarray(thickness_weights, dtype=float))
    basis = cst_basis(x, weights.shape[1])
    te = np.asarray(te_thickness, dtype=float).reshape(-1, 1)
    return weights @ basis.T + te * np.asarray(x, dtype=float)[None, :]


def read_xf(path) -> np.ndarray:
    """
    Reads a section file (XFOIL / bdSec .xf or Selig .dat: one name line then "x z" lines,
    UTF-8 with or without BOM) and returns the coordinates in the Selig order (upper surface
    first). bdSec files listing the lower surface first are reversed.
    """
    lines = Path(path).read_text(encoding="utf-8-sig").splitlines()
    rows = []
    for line in lines[1:]:
        parts = line.split()
        if len(parts) >= 2:
            rows.append((float(parts[0]), float(parts[1])))
    coordinates = np.array(rows, dtype=float)
    if len(coordinates) < 10:
        raise ValueError(f"{path}: not a section file (fewer than 10 points)")
    i_le = int(np.argmin(coordinates[:, 0]))
    if np.mean(coordinates[:i_le + 1, 1]) < np.mean(coordinates[i_le:, 1]):
        coordinates = coordinates[::-1].copy()
    return coordinates


def surfaces_at(coordinates: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Upper and lower surface heights of Selig-ordered coordinates at the stations x."""
    i_le = int(np.argmin(coordinates[:, 0]))
    upper, lower = coordinates[:i_le + 1][::-1], coordinates[i_le:]
    return (np.interp(x, upper[:, 0], upper[:, 1]), np.interp(x, lower[:, 0], lower[:, 1]))


def fit_kulfan_thickness_camber(coordinates: np.ndarray, n_weights: int,
                                te_thickness: float = None) -> dict:
    """
    Least-squares fit of a section by a Kulfan section in the thickness_camber form (no
    leading-edge mode): thickness(x) - x TE = C(x) sum t_i B_i(x) and camber(x) =
    C(x) sum c_i B_i(x) are two LINEAR problems on a cosine grid.

    Args:
    - coordinates: Selig-ordered coordinates (see read_xf).
    - n_weights: number of weights per side.
    - te_thickness: trailing-edge thickness to impose (default: the thickness of the section at
      its trailing edge).
    Returns:
    - dict t (thickness weights), c (camber weights), TE_thickness and the maximum fit errors
      of the thickness and camber distributions (fractions of the chord).
    """
    x = cosine_spacing(301)[1:-1]
    y_up, y_lo = surfaces_at(coordinates, x)
    if te_thickness is None:
        te_up, te_lo = surfaces_at(coordinates, np.array([1.0]))
        te_thickness = float(te_up[0] - te_lo[0])
    basis = cst_basis(x, n_weights)
    thickness = (y_up - y_lo) - x * te_thickness
    camber = 0.5 * (y_up + y_lo)
    t_weights = np.linalg.lstsq(basis, thickness, rcond=None)[0]
    c_weights = np.linalg.lstsq(basis, camber, rcond=None)[0]
    return {"t": t_weights, "c": c_weights, "TE_thickness": float(te_thickness),
            "thickness_error": float(np.max(np.abs(basis @ t_weights - thickness))),
            "camber_error": float(np.max(np.abs(basis @ c_weights - camber)))}


def parsec_coordinates(parameters: list, n_points: int = 161) -> np.ndarray:
    """
    Coordinates of a PARSEC section through bdSec (SectionParsec.from_parsec(array), 11
    parameters: rle, Xup, Zup, Zxxup, Xlo, Zlo, Zxxlo, Zte, DZte, alte, bete).
    """
    section_parsec = bridge.section_parsec_class()
    parsec = section_parsec.from_parsec(np.asarray(parameters, dtype=float))
    x = cosine_spacing(n_points)
    z_up = np.array([parsec.zup(xi) for xi in x])
    z_lo = np.array([parsec.zlo(xi) for xi in x])
    return np.column_stack((np.concatenate((x[::-1], x[1:])),
                            np.concatenate((z_up[::-1], z_lo[1:]))))


PARSEC_NAMES = ["rle", "Xup", "Zup", "Zxxup", "Xlo", "Zlo", "Zxxlo", "Zte", "DZte", "alte",
                "bete"]


# 3/4 ---------------------------------------------------------------------------------------------
def max_thickness(coordinates: np.ndarray) -> float:
    """Maximum thickness / chord (upper minus lower surface on a common grid)."""
    i_le = int(np.argmin(coordinates[:, 0]))
    upper, lower = coordinates[:i_le + 1][::-1], coordinates[i_le:]
    x = np.linspace(0.0, 1.0, 201)
    z_up = np.interp(x, upper[:, 0], upper[:, 1])
    z_lo = np.interp(x, lower[:, 0], lower[:, 1])
    return float(np.max(z_up - z_lo))


def section_parameters(problem, design_point: np.ndarray, solved: dict = None) -> dict:
    """
    Physical parameters of a design point: fixed parameters of the configuration, then the
    design variables, then the solved parameters (e.g. the camber offset delta of the section
    constraint, see section_constraint.py).
    """
    return {**problem.parametrization.get("fixed", {}),
            **to_physical(design_point, problem.variables), **(solved or {})}


def build_section(problem, design_point: np.ndarray, solved: dict = None) -> Section:
    """
    Section of a normalized design point for the parametrization of the configuration
    (problem.parametrization: {"type": "naca4" | "kulfan" | "parsec", "n_points", "fixed",
    and for kulfan "form"}). solved: parameters computed outside the design variables
    (e.g. {"delta": ...}).
    """
    params = section_parameters(problem, design_point, solved)
    kind = problem.parametrization["type"]
    n_points = int(problem.parametrization.get("n_points", 161))
    kulfan = None
    thickness = None
    if kind == "naca4":
        coords = naca4_coordinates(params["m_camber"], params.get("p_position", 0.4),
                                   params["t_thickness"], n_points)
    elif kind == "kulfan":
        geometry_kulfan = kulfan_parameters(params, problem.parametrization.get("form", "weights"))
        coords = kulfan_coordinates(geometry_kulfan, n_points)
        # NeuralFoil takes 8 weights per side: fewer weights are raised to 8 by exact degree
        # elevation (same shape); when that is not exact the section is passed by its
        # coordinates (NeuralFoil refits them)
        kulfan = neuralfoil_kulfan(geometry_kulfan)
        params.update({f"upper_{i}": float(w)
                       for i, w in enumerate(geometry_kulfan["upper_weights"])})
        params.update({f"lower_{i}": float(w)
                       for i, w in enumerate(geometry_kulfan["lower_weights"])})
        # analytical thickness, on the same stations as the known-constraint check
        # (constraints.py), so that the search and the evaluation agree exactly
        thickness = kulfan_max_thickness(geometry_kulfan)
    else:
        coords = parsec_coordinates([params[name] for name in PARSEC_NAMES], n_points)
    name = "sec_" + hashlib.sha1(np.round(coords, 10).tobytes()).hexdigest()[:10]
    return Section(name=name, coordinates=coords, parameters=params,
                   thickness=max_thickness(coords) if thickness is None else thickness,
                   kulfan=kulfan)


# chordwise stations of the analytical thickness of Kulfan sections (cosine, no end points)
THICKNESS_STATIONS = cosine_spacing(401)[1:-1]


def kulfan_max_thickness(kulfan: dict) -> float:
    """Maximum thickness / chord of a Kulfan section (analytical, THICKNESS_STATIONS)."""
    weights = np.asarray(kulfan["upper_weights"]) - np.asarray(kulfan["lower_weights"])
    return float(cst_thickness(THICKNESS_STATIONS, weights, kulfan["TE_thickness"]).max())


# 4/4 ---------------------------------------------------------------------------------------------
def write_xf(section: Section, directory: Path) -> Path:
    """
    Writes the section in the XFOIL / bdSec format (name line + x z coordinates).

    Returns:
    - absolute path of the .xf file.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = (directory / f"{section.name}.xf").resolve()
    lines = [section.name] + [f"{x: .8f} {z: .8f}" for x, z in section.coordinates]
    path.write_text("\n".join(lines) + "\n", encoding="ascii")
    return path
