"""
Section geometry: mapping of the normalized design variables to physical parameters and section
coordinates (NACA 4-digit continuous, Kulfan/CST through aerosandbox, PARSEC through bdSec).

Coordinates follow the Selig order expected by XFOIL and NeuralFoil: trailing edge -> upper
surface -> leading edge -> lower surface -> trailing edge, chord 1.
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


def kulfan_parameters(parameters: dict) -> dict:
    """
    Kulfan (CST) parameters from flat variables: upper_0..upper_n, lower_0..lower_n,
    leading_edge_weight (optional) and TE_thickness (optional).
    """
    upper = [parameters[k] for k in sorted((k for k in parameters if k.startswith("upper_")),
                                           key=lambda k: int(k.split("_")[1]))]
    lower = [parameters[k] for k in sorted((k for k in parameters if k.startswith("lower_")),
                                           key=lambda k: int(k.split("_")[1]))]
    if not upper or not lower:
        raise ValueError("kulfan parametrization needs upper_i and lower_i variables")
    return {"upper_weights": np.array(upper), "lower_weights": np.array(lower),
            "leading_edge_weight": float(parameters.get("leading_edge_weight", 0.0)),
            "TE_thickness": float(parameters.get("TE_thickness", 0.0))}


def kulfan_coordinates(kulfan: dict, n_points: int = 161) -> np.ndarray:
    """Coordinates of a Kulfan section (aerosandbox KulfanAirfoil)."""
    import aerosandbox as asb  # noqa: PLC0415
    airfoil = asb.KulfanAirfoil(name="kulfan", **kulfan)
    return np.asarray(airfoil.to_airfoil(n_coordinates_per_side=n_points).coordinates)


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


def build_section(problem, design_point: np.ndarray) -> Section:
    """
    Section of a normalized design point for the parametrization of the configuration
    (problem.parametrization: {"type": "naca4" | "kulfan" | "parsec", "n_points", "fixed"}).
    """
    params = {**problem.parametrization.get("fixed", {}),
              **to_physical(design_point, problem.variables)}
    kind = problem.parametrization["type"]
    n_points = int(problem.parametrization.get("n_points", 161))
    kulfan = None
    if kind == "naca4":
        coords = naca4_coordinates(params["m_camber"], params.get("p_position", 0.4),
                                   params["t_thickness"], n_points)
    elif kind == "kulfan":
        kulfan = kulfan_parameters(params)
        coords = kulfan_coordinates(kulfan, n_points)
        # NeuralFoil evaluates Kulfan parameters directly only with 8 weights per side;
        # otherwise the section is passed by its coordinates (NeuralFoil refits them)
        if len(kulfan["upper_weights"]) != 8 or len(kulfan["lower_weights"]) != 8:
            kulfan = None
    else:
        coords = parsec_coordinates([params[name] for name in PARSEC_NAMES], n_points)
    name = "sec_" + hashlib.sha1(np.round(coords, 10).tobytes()).hexdigest()[:10]
    return Section(name=name, coordinates=coords, parameters=params,
                   thickness=max_thickness(coords), kulfan=kulfan)


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
