"""
Fixed 3D planform of a foil optimization whose design variables describe the SECTION.

C-foil defined by a single circular arc: a clean re-derivation of
bdFoil/Code/foil_family_intensSY.py CFoil_Arc_T1 (arc_geometry, arc_characteristics,
_build_arrays) that does NOT import the legacy foil.py. bdFoil conventions (bdFoil/CLAUDE.md):

* Frame: leading edge (LE) of the LOWER BEARING at the origin, x forward, y to port, z UP
  (negative downward). The arc lies in the YZ plane, starts at the origin with a tangent
  along -z when cant_geom = 0, and closes toward the tip:
      psi = psi0 + theta,  theta in [0, L / R],  psi0 = radians(cant_geom)
      x = 0
      y = ori_y + tip_side R (cos(psi0) - cos(psi))      (tip_side = -1: tip inboard)
      z = ori_z - R (sin(psi) - sin(psi0))
* One single section at every station, chord linear from root to tip (taper), twist linear
  from root_twist to tip_twist (the "Ainc" column of the planform CSV, degrees).
* Main-section markers (MS) at the root and the tip only.
* The heel and the trunk cant are NOT in the geometry: they enter through the attitude,
  cant = heel + trunk_cant (VPP convention FOIL_PRT_Cant_Eff = Heel + FOIL_PRT_Cant), applied
  by bdFoil core (rotation R = Rz(-yaw) Ry(-rake) Rx(-cant) about the origin, then z += sink).

The planform is written ONCE per evaluation as a bdGeometry CSV in METRES
(Xle, Yle, Zle, Chord, Ainc, Section, MS) next to the section .xf file, and both 3D solvers read
this same file (NPLLT with length_factor = 1, AVL through bdFoil core load_planform), so they
see exactly the same geometry.

Configuration entries:
    "planform": {"type": "cfoil_arc", "radius": 9.8, "span": 3.25, "chord": 0.70,
                 "taper": 1.0, "root_twist": 0.0, "tip_twist": 0.0, "cant_geom": 0.0,
                 "tip_side": -1, "ori_y": 0.0, "ori_z": 0.0, "n_stations": 41, "kind": "FOIL"}
    "attitude": {"heel": 5.0, "trunk_cant": 3.28, "rake": 0.5, "yaw": 4.0, "sink": 0.0}
"""
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

PLANFORM_TYPES = ("cfoil_arc",)
FOIL_KINDS = ("FOIL", "DAGG", "RUDDER")
ATTITUDE_KEYS = ("heel", "trunk_cant", "rake", "yaw", "sink")


@dataclass(frozen=True)
class CFoilArc:
    """Single-arc C-foil (lengths in metres, angles in degrees), see the module docstring."""
    radius: float
    span: float
    chord: float
    taper: float = 1.0
    root_twist: float = 0.0
    tip_twist: float = 0.0
    cant_geom: float = 0.0
    tip_side: int = -1
    ori_y: float = 0.0
    ori_z: float = 0.0
    n_stations: int = 41
    kind: str = "FOIL"

    def __post_init__(self):
        for name in ("radius", "span", "chord", "taper"):
            if not float(getattr(self, name)) > 0.0:
                raise ValueError(f"planform: {name} must be > 0, got {getattr(self, name)!r}")
        if int(self.n_stations) < 3:
            raise ValueError("planform: n_stations must be >= 3")
        if self.kind not in FOIL_KINDS:
            raise ValueError(f"planform: kind must be in {FOIL_KINDS}, got {self.kind!r}")
        if int(self.tip_side) not in (-1, 1):
            raise ValueError("planform: tip_side must be -1 (inboard) or +1 (outboard)")

    @classmethod
    def from_config(cls, config: dict) -> "CFoilArc":
        """Planform from the "planform" entry of the configuration."""
        config = dict(config)
        kind = config.pop("type", "cfoil_arc")
        if kind not in PLANFORM_TYPES:
            raise ValueError(f"planform: type must be in {PLANFORM_TYPES}, got {kind!r}")
        unknown = set(config) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"planform: unknown keys {sorted(unknown)}")
        return cls(**config)

    def to_dict(self) -> dict:
        return {"type": "cfoil_arc", **asdict(self)}

    # ------------------------------------------------------------------ geometry
    def arc_geometry(self, n: int = None) -> tuple[np.ndarray, ...]:
        """
        Leading edge of the arc (same formula as CFoil_Arc_T1.arc_geometry).

        Returns:
        - theta (arc angle from the lower bearing, rad), x, y, z (m), n stations.
        """
        n = int(self.n_stations if n is None else n)
        psi_0 = np.radians(self.cant_geom)
        theta = np.linspace(0.0, self.span / self.radius, n)
        psi = psi_0 + theta
        x = np.zeros(n)
        y = self.ori_y + self.tip_side * self.radius * (np.cos(psi_0) - np.cos(psi))
        z = self.ori_z - self.radius * (np.sin(psi) - np.sin(psi_0))
        return theta, x, y, z

    def stations(self) -> dict:
        """
        Planform arrays of the n_stations stations (CFoil_Arc_T1._build_arrays without the
        encastered base): x, y, z, c, twist, ms (root and tip main sections).
        """
        _, x, y, z = self.arc_geometry()
        n = len(x)
        ms = np.zeros(n, dtype=int)
        ms[0] = ms[-1] = 1
        return {"x": x, "y": y, "z": z,
                "c": np.linspace(self.chord, self.chord * self.taper, n),
                "twist": np.linspace(self.root_twist, self.tip_twist, n), "ms": ms}

    def characteristics(self) -> dict:
        """
        Closed-form characteristics (CFoil_Arc_T1.arc_characteristics): arc angle, depth
        (max |z| below the bearing), tip depth, lateral offset of the tip, tip angle from
        vertical, area (chord x arc length for a linear taper), aspect ratio span^2 / area.
        """
        psi_0 = np.radians(self.cant_geom)
        psi_tip = psi_0 + self.span / self.radius
        psi_deep = min(psi_tip, max(psi_0, np.pi / 2))
        area = self.chord * self.span * (1.0 + self.taper) / 2.0
        return {"arc_angle_deg": float(np.degrees(self.span / self.radius)),
                "depth": float(self.radius * (np.sin(psi_deep) - np.sin(psi_0))),
                "tip_depth": float(self.radius * (np.sin(psi_tip) - np.sin(psi_0))),
                "lateral_offset": float(self.radius * (np.cos(psi_0) - np.cos(psi_tip))),
                "tip_angle_deg": float(np.degrees(psi_tip)),
                "area": float(area), "aspect_ratio": float(self.span ** 2 / area)}


def write_planform_csv(stations: dict, section_file: str, path) -> Path:
    """
    Writes the planform as a bdGeometry CSV in METRES, one row per station, root -> tip:
    Xle, Yle, Zle, Chord, Ainc (twist, deg), Section (file name, resolved by the solvers in the
    folder of the CSV), MS. Returns the absolute path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["Xle,Yle,Zle,Chord,Ainc,Section,MS"]
    for i in range(len(stations["x"])):
        lines.append(f"{stations['x'][i]:.10f},{stations['y'][i]:.10f},{stations['z'][i]:.10f},"
                     f"{stations['c'][i]:.10f},{stations['twist'][i]:.10f},{section_file},"
                     f"{int(stations['ms'][i])}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path.resolve()


def attitude_angles(attitude: dict) -> dict:
    """
    bdFoil attitude of the configuration: cant = heel + trunk_cant, rake, yaw (deg), sink (m).
    """
    unknown = set(attitude) - set(ATTITUDE_KEYS)
    if unknown:
        raise ValueError(f"attitude: unknown keys {sorted(unknown)} (allowed {ATTITUDE_KEYS})")
    return {"cant": float(attitude.get("heel", 0.0)) + float(attitude.get("trunk_cant", 0.0)),
            "rake": float(attitude.get("rake", 0.0)), "yaw": float(attitude.get("yaw", 0.0)),
            "sink": float(attitude.get("sink", 0.0))}
