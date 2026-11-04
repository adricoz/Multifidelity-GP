"""
Meta-surrogate of the C-foil torsor coefficients: a CHEAP stand-in for the codes (NPLLT, AVL) with
the structure of the real problem, to develop and test the VPP-targeted sequential co-kriging
without running them. Nothing here is a validated model of the Intens SY foil.

Inputs (16, one row per point): the 12 whitened PCA scores z of the section (SmoothKulfanFamily,
SectionPCA with k = 12, examples/results/section_pca_smooth16.json), then the foil attitude
Cant_Eff [deg], Rake_Eff [deg], Yaw_Eff [deg] and the boat sink Hvol [m] (INPUT_NAMES).
Outputs (7): the coefficients of the MDS C-foil RS, in its order: Fx, Fy, Fz, Mx, My, Mz (MDS
multiplies them by 0.5 rho S V^2) and Cpmin (OUTPUT_NAMES).

1. Anchor. The C-foil RS of the boat (MDS Systems/Ships/Sail/IntensSY/Data/Foil/
   CFoil_R7800_S3250_C700_rs_vpp.json, 3 inputs Cant_Eff, Rake_Eff, Yaw_Eff) gives c_ref(attitude).
   It is evaluated here with numpy's Chebyshev polynomials (same formula as MDS' VppChebyRS,
   checked in the tests). At the reference section, level 2 (without the immersion term at the
   reference sink) IS the production foil: the VPP of the meta-problem is the Intens SY VPP.

2. Section correction (lifting line). For each section, a NeuralFoil polar (Re 6e6, n_crit 1,
   xtr 0.1, alpha -6..12 deg) gives the lift slope a2d (fit on [-2, 6] deg), the zero-lift angle
   alpha_0 and CD, CM, Cpmin against alpha. The 3D lift slope is a3d = a2d / (1 + a2d / (pi AR e)).
   The lift of the anchor is projected on n(cant), the unit direction of the camber lift (the
   anchor at zero rake and yaw, rotated with the cant): L = (Fy, Fz) . n. It is read as
   L = a3d_ref (alpha_eff - alpha_0,ref), which gives alpha_eff; the section gives
       L_new = a3d (alpha_eff - alpha_0) = (a3d / a3d_ref) L + a3d (alpha_0,ref - alpha_0),
       (Fy, Fz) += (L_new - L) n,
       Fx       += -[CD(alpha_eff) - CD_ref(alpha_eff)] - (L_new^2 - L^2) / (pi AR e)   (Fx < 0: drag),
       M        += r x dF + (0, [CM - CM_ref] chord, 0),   r = lever arm fitted on the anchor,
       Cpmin    += Cpmin(alpha_eff) - Cpmin_ref(alpha_eff).
   Linear in L: smooth everywhere, also where the lift changes sign.

3. Immersion. Every force and moment coefficient is multiplied by 1 + (Hvol - HVOL_REF) / span
   (more wetted span when the boat sinks).

4. The two levels ("NPLLT-like" level 1, "AVL-like" level 2) differ like the real codes:
     level 2: NeuralFoil "xxxlarge", e = 0.85, free-surface lift loss on the lift components
              (1 - 0.15 exp(-Hvol / 0.3)) / (1 - 0.15 exp(-HVOL_REF / 0.3)) (1 at HVOL_REF);
     level 1: NeuralFoil "xxsmall", elliptic e = 1 with an aspect ratio 10 % too large, no
              free-surface loss, anchor lift 5 % too high and drag 10 % too low.
   So level 1 ~ rho level 2 + a smooth discrepancy, as the AR(1) model of the co-kriging assumes.
   Every force change (biases, section, free surface) is transported into the moments with the
   lever arm r, so that forces and moments stay consistent.

5. Domain of the test: sections of the family on which NeuralFoil xxxlarge is confident
   (analysis confidence >= 0.9 at alpha 0, 2, 4, 6 deg), as the known constraint of the last C-foil
   study (60 % of the family). A choice of this first test, not yet a decision on the family.

Cost: one NeuralFoil polar per NEW section (cached), then a few array operations per point.
"""
import json
from pathlib import Path

import numpy as np

from src.section_pca import CFoilFamily, SectionPCA, SmoothKulfanFamily, kulfan_dict

VPP_DIR = Path(__file__).resolve().parents[1]                 # .../vpp_acceleration
REPO_DIR = VPP_DIR.parent                                      # .../Multifidelity-GP-Clean
CODE_ADRI = REPO_DIR.parent                                    # .../Code-Adri
ANCHOR_RS = (CODE_ADRI / "MDS" / "Systems" / "Ships" / "Sail" / "IntensSY" / "Data" / "Foil"
             / "CFoil_R7800_S3250_C700_rs_vpp.json")
PCA_FILE = VPP_DIR / "runs" / "2026-10-21_section_pca" / "results" / "section_pca_smooth16.json"
STUDY_CONFIG = REPO_DIR / "studies" / "cfoil_intens_kulfan_trim_2" / "config_cfoil_trim.json"

N_SCORES = 12
INPUT_NAMES = [f"z_{j + 1}" for j in range(N_SCORES)] + ["Cant_Eff", "Rake_Eff", "Yaw_Eff",
                                                         "Hvol"]
OUTPUT_NAMES = ["Fx", "Fy", "Fz", "Mx", "My", "Mz", "Cpmin"]
# attitude box of the surrogate: the box of the anchor RS, and the sink range of the VPP states
ATTITUDE_BOUNDS = [(0.0, 8.0), (-1.5, 1.5), (-6.0, 6.0), (0.0, 0.5)]
# C-foil planform (MDS FOIL_CONFIGS['CFoil']): surface [m2], immersed span [m], mean chord [m]
SURFACE, SPAN, CHORD = 2.2749596294181016, 3.249985306371202, 0.70
ASPECT_RATIO = SPAN ** 2 / SURFACE
HVOL_REF = 0.20                    # sink at which the immersion factor is 1 [m]
FLOW = {"Re": 6e6, "n_crit": 1.0, "xtr_upper": 0.1, "xtr_lower": 0.1}
ALPHAS = np.arange(-6.0, 12.01, 0.5)                          # polar grid [deg]
LEVELS = {
    1: {"model_size": "xxsmall", "efficiency": 1.0, "aspect_ratio": 1.1 * ASPECT_RATIO,
        "free_surface": False, "lift_bias": 1.05, "drag_bias": 0.90},
    2: {"model_size": "xxxlarge", "efficiency": 0.85, "aspect_ratio": ASPECT_RATIO,
        "free_surface": True, "lift_bias": 1.00, "drag_bias": 1.00},
}


# 1/3 ---------------------------------------------------------------------------------------------
# The anchor: the C-foil RS of the boat
# -------------------------------------------------------------------------------------------------
class ChebyshevRS:
    """A vpp36 / MDS Chebyshev response surface (JSON), evaluated for a batch of points:
        x_n = (2 x - (hi + lo)) / (hi - lo),   out_i = sum_j beta_ij prod_k T_(order_jk)(x_n,k)."""

    def __init__(self, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.input_names = list(data["InputNames"])
        self.output_names = list(data["OutputNames"])
        self.input_range = np.asarray(data["InputRange"], dtype=float)
        self.orders = np.asarray(data["ChebyOrders"], dtype=int)          # (n_terms, n_in)
        self.betas = np.asarray(data["ChebyCoefs"], dtype=float)          # (n_out, n_terms)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Outputs (m, n_out) at the points x (m, n_in), no clamping (as MDS for the foil)."""
        x = np.atleast_2d(np.asarray(x, dtype=float))
        lo, hi = self.input_range[:, 0], self.input_range[:, 1]
        x_n = (2.0 * x - (hi + lo)) / (hi - lo)
        # T_n(x) for n = 0..max order, every input: chebvander gives (m, max_order + 1)
        max_order = int(self.orders.max())
        basis = np.ones((len(x), len(self.orders)))
        for k in range(x.shape[1]):
            t_k = np.polynomial.chebyshev.chebvander(x_n[:, k], max_order)
            basis *= t_k[:, self.orders[:, k]]
        return basis @ self.betas.T


# 2/3 ---------------------------------------------------------------------------------------------
# Section polars
# -------------------------------------------------------------------------------------------------
def section_polars(weights: np.ndarray, model_size: str) -> dict:
    """
    NeuralFoil polars of a batch of sections on ALPHAS, and the quantities of the lifting line.

    Returns:
    - dict of arrays: "CL", "CD", "CM", "Cpmin" (m, n_alpha); "a2d" lift slope [1/deg] and
      "alpha_0" zero-lift angle [deg] (m,).
    """
    import neuralfoil as nf  # noqa: PLC0415
    weights = np.atleast_2d(weights)
    m, n_alpha = len(weights), len(ALPHAS)
    repeated = np.repeat(weights, n_alpha, axis=0)
    aero = nf.get_aero_from_kulfan_parameters(kulfan_dict(repeated, 0.00197),
                                              alpha=np.tile(ALPHAS, m), model_size=model_size,
                                              **FLOW)
    velocity = np.column_stack([aero[f"{side}_bl_ue/vinf_{i}"] for side in ("upper", "lower")
                                for i in range(len(nf.bl_x_points))])
    out = {name: np.asarray(aero[name], dtype=float).reshape(m, n_alpha)
           for name in ("CL", "CD", "CM")}
    out["Cpmin"] = np.min(1.0 - velocity ** 2, axis=1).reshape(m, n_alpha)
    linear = (ALPHAS >= -2.0) & (ALPHAS <= 6.0)
    slope, intercept = np.polyfit(ALPHAS[linear], out["CL"][:, linear].T, 1)
    out["a2d"], out["alpha_0"] = slope, -intercept / slope
    return out


def confident(weights: np.ndarray, threshold: float = 0.9) -> np.ndarray:
    """True for the sections whose NeuralFoil xxxlarge analysis confidence is at least
    threshold at alpha 0, 2, 4 and 6 deg (flow FLOW)."""
    import neuralfoil as nf  # noqa: PLC0415
    alphas = np.array([0.0, 2.0, 4.0, 6.0])
    weights = np.atleast_2d(weights)
    aero = nf.get_aero_from_kulfan_parameters(
        kulfan_dict(np.repeat(weights, len(alphas), axis=0), 0.00197),
        alpha=np.tile(alphas, len(weights)), model_size="xxxlarge", **FLOW)
    confidence = np.asarray(aero["analysis_confidence"]).reshape(len(weights), len(alphas))
    return np.all(confidence >= threshold, axis=1)


def interpolate(table: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Row i of table (m, n_alpha) read at alpha[i] (linear, clipped to the grid)."""
    alpha = np.clip(alpha, ALPHAS[0], ALPHAS[-1])
    position = (alpha - ALPHAS[0]) / (ALPHAS[1] - ALPHAS[0])
    left = np.minimum(np.floor(position).astype(int), len(ALPHAS) - 2)
    fraction = position - left
    rows = np.arange(len(table))
    return (1.0 - fraction) * table[rows, left] + fraction * table[rows, left + 1]


# 3/3 ---------------------------------------------------------------------------------------------
# The meta-problem
# -------------------------------------------------------------------------------------------------
class MetaFoil:
    """
    The two-level meta-surrogate (module docstring). Usage:
        meta = MetaFoil()
        y = meta(x, level)          # x (m, 16) -> coefficients (m, 7), level 1 or 2
    """

    def __init__(self, pca_file=PCA_FILE, anchor_file=ANCHOR_RS):
        self.anchor = ChebyshevRS(anchor_file)
        self.pca = SectionPCA.load(pca_file)
        self.family = SmoothKulfanFamily.around(CFoilFamily.from_study_config(STUDY_CONFIG))
        self.reference_scores = self.pca.encode(self.family.reference_weights())[0]
        self._polars = {1: {}, 2: {}}                       # cache: level -> {section key: polar}
        self._reference = {level: self._polar_rows(self.reference_scores[None], level)
                           for level in (1, 2)}
        self.lever_arm, self.moment_offset = self._fit_lever_arm()

    # --- section polars, cached by section ---
    @staticmethod
    def _key(scores: np.ndarray) -> bytes:
        return np.round(scores, 12).tobytes()

    def _polar_rows(self, scores: np.ndarray, level: int) -> dict:
        """Polar quantities of every row of scores (m, 12), from the cache or NeuralFoil."""
        cache = self._polars[level]
        keys = [self._key(row) for row in scores]
        new = {key: row for key, row in zip(keys, scores) if key not in cache}
        if new:
            polars = section_polars(self.pca.decode(np.array(list(new.values()))),
                                    LEVELS[level]["model_size"])
            for i, key in enumerate(new):
                cache[key] = {name: value[i] for name, value in polars.items()}
        return {name: np.array([cache[key][name] for key in keys])
                for name in ("CL", "CD", "CM", "Cpmin", "a2d", "alpha_0")}

    # --- geometry of the anchor ---
    def _fit_lever_arm(self):
        """Lever arm r and offset m0 such that the anchor moments are M ~ r x F + m0, least squares
        over the attitude box (used to transport the force changes into moments)."""
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in ATTITUDE_BOUNDS[:3]])
        hi = np.array([b[1] for b in ATTITUDE_BOUNDS[:3]])
        c = self.anchor(lo + rng.random((400, 3)) * (hi - lo))
        forces, moments = c[:, 0:3], c[:, 3:6]
        rows, rhs = [], []
        for f, mom in zip(forces, moments):
            # r x F = [[0, Fz, -Fy], [-Fz, 0, Fx], [Fy, -Fx, 0]] r
            cross = np.array([[0.0, f[2], -f[1]], [-f[2], 0.0, f[0]], [f[1], -f[0], 0.0]])
            rows.append(np.hstack([cross, np.eye(3)]))
            rhs.append(mom)
        solution = np.linalg.lstsq(np.vstack(rows), np.concatenate(rhs), rcond=None)[0]
        return solution[:3], solution[3:]

    def lift_direction(self, cant: np.ndarray) -> np.ndarray:
        """n(cant) (m, 2): unit direction, in the (y, z) plane, of the anchor lift at zero rake and
        zero yaw (the camber lift of the reference section) at this cant."""
        c = self.anchor(np.column_stack([cant, np.zeros_like(cant), np.zeros_like(cant)]))
        direction = c[:, 1:3]
        return direction / np.linalg.norm(direction, axis=1, keepdims=True)

    # --- evaluation ---
    def __call__(self, x: np.ndarray, level: int) -> np.ndarray:
        """Coefficients (m, 7) at the points x (m, 16) for the fidelity level (1 or 2)."""
        x = np.atleast_2d(np.asarray(x, dtype=float))
        scores, attitude, hvol = x[:, :N_SCORES], x[:, N_SCORES:N_SCORES + 3], x[:, -1]
        settings = LEVELS[level]
        anchor = self.anchor(attitude)
        c = anchor.copy()
        # level 1: biased anchor (lift components and drag); every force change below is also
        # transported into the moments with the lever arm (r x dF)
        c[:, 1:3] *= settings["lift_bias"]
        c[:, 0] *= settings["drag_bias"]
        section = self._polar_rows(scores, level)
        reference = {name: np.repeat(value, len(x), axis=0)
                     for name, value in self._reference[level].items()}
        # lifting line: 3D lift slope [1/deg]
        factor = 180.0 / np.pi / (np.pi * settings["aspect_ratio"] * settings["efficiency"])
        a3d = section["a2d"] / (1.0 + section["a2d"] * factor)
        a3d_ref = reference["a2d"] / (1.0 + reference["a2d"] * factor)
        n = self.lift_direction(attitude[:, 0])
        lift = np.sum(c[:, 1:3] * n, axis=1)                         # anchor lift along n
        alpha_eff = reference["alpha_0"] + lift / a3d_ref
        lift_new = a3d / a3d_ref * lift + a3d * (reference["alpha_0"] - section["alpha_0"])
        d_lift = lift_new - lift
        d_force = np.zeros((len(x), 3))
        d_force[:, 1:3] = d_lift[:, None] * n
        d_force[:, 0] = -(interpolate(section["CD"], alpha_eff)
                          - interpolate(reference["CD"], alpha_eff)) \
            - (lift_new ** 2 - lift ** 2) / (np.pi * settings["aspect_ratio"]
                                             * settings["efficiency"])
        c[:, 0:3] += d_force
        # free-surface lift loss (level 2), relative to the reference sink: none at HVOL_REF
        if settings["free_surface"]:
            loss = (1.0 - 0.15 * np.exp(-hvol / 0.3)) / (1.0 - 0.15 * np.exp(-HVOL_REF / 0.3))
            c[:, 1:3] *= loss[:, None]
        # moments: the anchor ones, plus the transport of every force change
        c[:, 3:6] = anchor[:, 3:6] + np.cross(self.lever_arm, c[:, 0:3] - anchor[:, 0:3])
        c[:, 4] += (interpolate(section["CM"], alpha_eff)
                    - interpolate(reference["CM"], alpha_eff)) * CHORD
        c[:, 6] += interpolate(section["Cpmin"], alpha_eff) \
            - interpolate(reference["Cpmin"], alpha_eff)
        # immersion: more wetted span when the boat sinks
        c[:, 0:6] *= (1.0 + (hvol - HVOL_REF) / SPAN)[:, None]
        return c

    # --- the input domain ---
    def bounds(self, n_sections: int = 2 ** 14, seed: int = 1) -> list:
        """Box of the 16 inputs: the scores of a family sample, then ATTITUDE_BOUNDS."""
        scores = self.pca.encode(self.family.sample(n_sections, seed))
        return [(float(lo), float(hi)) for lo, hi in zip(scores.min(0), scores.max(0))] \
            + list(ATTITUDE_BOUNDS)

    def sample_sections(self, n_sections: int, seed: int) -> np.ndarray:
        """Scores (n, 12) of sections drawn uniformly in the family (never in the score box: only
        2.7 % of it is in the family, README section 0), kept if NeuralFoil is confident on them
        (module docstring, point 5). The reference section is always confident."""
        kept, draw = [], 0
        while sum(len(k) for k in kept) < n_sections:
            weights = self.pca.reconstruct(self.family.sample(2 * n_sections, seed + 1000 * draw))
            kept.append(self.pca.encode(weights[confident(weights)]))
            draw += 1
        return np.vstack(kept)[:n_sections]
