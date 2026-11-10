"""
Convex domain of the C-foil sections (2026-11-04, after the meeting with O. Le Maitre): the sections
the search may propose, written so that the domain is CONVEX in the Kulfan weights, hence in the
PCA scores (the decoding scores -> weights is affine, section_pca Eq. (3)): the segment between two
admissible sections is admissible, the space has no holes and no separate pieces.

1. The constraints (chosen with the user on 2026-11-04). With y = A w + b the coordinates of the
   section at the stations (section_pca Eq. (1)), t(x) = y_upper - y_lower the local thickness and
   y'' the second difference of a surface on the stations, every constraint is LINEAR in w:
     * Kulfan box: the box of the generation (box_scale x the box of Dim-Reduction/experiments.ipynb,
       centred on the reference section);
     * no crossing surfaces: t(x) >= 0 for x > 0;
     * upper surface convex: y_upper'' <= 0 on [X_LO, X_HI];
     * lower surface with fixed sign zones around the inflections x_1, x_2 of the reference section:
       y_lower'' >= 0 on [X_LO, x_1 - g], <= 0 on [x_1 + g, x_2 - g], >= 0 on [x_2 + g, X_HI]
       (free in the bands of half-width g: the inflections may move inside them). A rule that only
       COUNTS the sign changes would let them move anywhere, and is not convex;
     * round leading edge, a minimum radius that grows with the thickness, r >= kappa t_max^2: a
       surface starts as y = w_0 sqrt(x), its radius is w_0^2 / 2, so the condition is
       |w_0| >= sqrt(2 kappa) t_max, i.e. |w_0| >= sqrt(2 kappa) t(x) at every station (linear);
       kappa = le_factor x (r / t_max^2 of the reference section), per surface.
   Each row of G w >= h is scaled to unit norm, so that its margin G w - h is a distance in the
   weight space.
   One constraint is not linear: the bending strength Ixx / v >= ixx_v_factor x that of the
   reference section, v = t_max / 2 (section_modulus; 2026-11-05, factor 0.9). It is checked on
   every section (contains, the sampler); on the segment tests of 2026-11-04 it left the domain
   nearly convex (1 to 3 % of the segments between admissible sections leave it).
   A second one, optional, is the bending stiffness Ixx >= ixx_factor x that of the reference
   section (section_ixx; 2026-11-10: factor 0.95 together with Ixx / v >= 0.9; 0 = not applied,
   as in the domains built before). Ixx (~ int t^3) is not convex either: treated as Ixx / v.

2. Sampling: hit-and-run (Smith 1984; Lovasz 1999). From an admissible point, draw a direction d,
   compute EXACTLY the chord of the polytope along d (the interval of s such that G (w + s d) >= h)
   and draw s uniformly on it; Ixx / v (and Ixx) is handled by the shrinkage of slice sampling (Neal 2003):
   a rejected s shrinks the chord towards 0 (the current point) and s is drawn again. The chain
   leaves the uniform law on the domain invariant; every recorded section satisfies every
   constraint by construction. The directions are Gaussian with the covariance of the box
   (burn-in), then of the chains themselves (the domain is thin in some directions); many chains
   run together (vectorized).

Import: vpp_acceleration on sys.path (from src.section_domain import SectionDomain).
"""
import numpy as np

from src.section_pca import (THICKNESS_STATIONS, notebook_bounds, section_stiffness,
                             surface_matrix)

X_LO, X_HI = 0.01, 0.99          # curvature rules on [X_LO, X_HI]: y'' diverges at the leading edge


def lower_inflections(weights: np.ndarray, te_thickness: float, stations=THICKNESS_STATIONS) -> list:
    """Stations in (X_LO, X_HI) where y_lower'' of one section changes sign."""
    x = np.asarray(stations, dtype=float)
    matrix, offset = surface_matrix(x, len(weights) // 2, te_thickness)
    lower = (matrix @ weights + offset)[len(x):]
    curvature = np.diff(lower, 2)
    inner = x[1:-1]
    return [float(inner[p]) for p in range(len(inner) - 1)
            if X_LO < inner[p] < X_HI and curvature[p] * curvature[p + 1] < 0]


class SectionDomain:
    """The convex domain of the sections (module docstring)."""

    def __init__(self, reference: np.ndarray, te_thickness: float, box_scale: float = 3.0,
                 band: float = 0.05, le_factor: float = 0.5, stations=THICKNESS_STATIONS,
                 inflections=None, ixx_v_factor: float = 1.0, ixx_factor: float = 0.0):
        """
        Args:
        - reference: Kulfan weights (2n,) of the reference (current) section: centre of the Kulfan
          box, inflections of its lower surface, leading-edge ratios, Ixx / v.
        - te_thickness: trailing-edge thickness (fraction of the chord).
        - box_scale: width of the Kulfan box / width of the Dim-Reduction box.
        - band: half-width g of the free bands around the lower inflections.
        - le_factor: kappa / (r / t_max^2 of the reference), per surface.
        - stations: stations of the constraints (a coarser grid samples faster; check the samples
          with a domain on THICKNESS_STATIONS afterwards).
        - inflections: (x_1, x_2); default those of the reference on THICKNESS_STATIONS (the same
          zones whatever the stations of the constraints).
        - ixx_v_factor: Ixx / v >= ixx_v_factor x (Ixx / v of the reference).
        - ixx_factor: Ixx >= ixx_factor x (Ixx of the reference); 0 = no Ixx constraint.
        """
        self.reference = np.asarray(reference, dtype=float)
        self.te_thickness, self.box_scale = float(te_thickness), float(box_scale)
        self.band, self.le_factor = float(band), float(le_factor)
        self.stations = x = np.asarray(stations, dtype=float)
        n_w, p = len(self.reference) // 2, len(x)
        self.n_weights = n_w
        matrix, offset = surface_matrix(x, n_w, te_thickness)
        a_up, a_lo, b_up, b_lo = matrix[:p], matrix[p:], offset[:p], offset[p:]
        a_t, b_t = a_up - a_lo, b_up - b_lo                               # t = a_t w + b_t
        second = np.diff(np.eye(p), 2, axis=0)                            # (p - 2, p), no 1 / h^2
        inner = x[1:-1]
        rows, rhs, names = [], [], []

        def add(g, h, name):                   # g w >= h, rows scaled to unit norm
            norm = np.linalg.norm(g, axis=1)
            keep = norm > 0
            rows.append(g[keep] / norm[keep, None])
            rhs.append(h[keep] / norm[keep])
            names.extend([name] * int(keep.sum()))

        # Kulfan box
        low, high = notebook_bounds()
        half = self.box_scale * (high - low) / 2.0
        self.box_lower, self.box_upper = self.reference - half, self.reference + half
        add(np.eye(2 * n_w), self.box_lower, "kulfan_box")
        add(-np.eye(2 * n_w), -self.box_upper, "kulfan_box")
        # no crossing surfaces (x > 0: at the leading edge both surfaces are at 0)
        add(a_t[1:], -b_t[1:], "crossing")
        # upper surface convex: -(D2 y_up) >= 0
        zone = (inner >= X_LO) & (inner <= X_HI)
        add(-(second @ a_up)[zone], (second @ b_up)[zone], "upper_curvature")
        # lower surface: fixed sign zones around the inflections of the reference
        found = list(inflections) if inflections is not None \
            else lower_inflections(self.reference, te_thickness)
        if len(found) != 2:
            raise ValueError(f"the reference lower surface has {len(found)} inflections, 2 expected")
        self.inflections = tuple(float(f) for f in found)
        x_1, x_2, g = self.inflections[0], self.inflections[1], self.band
        sign = np.zeros(len(inner))
        sign[(inner >= X_LO) & (inner <= x_1 - g)] = 1.0
        sign[(inner >= x_1 + g) & (inner <= x_2 - g)] = -1.0
        sign[(inner >= x_2 + g) & (inner <= X_HI)] = 1.0
        ruled = sign != 0
        add((sign[:, None] * (second @ a_lo))[ruled], -(sign * (second @ b_lo))[ruled],
            "lower_curvature")
        # round leading edge: w_0,up >= c_up t(x), -w_0,lo >= c_lo t(x), at every station
        a_fine, b_fine = surface_matrix(THICKNESS_STATIONS, n_w, te_thickness)
        n_fine = len(THICKNESS_STATIONS)
        t_max = float(np.max((a_fine[:n_fine] - a_fine[n_fine:]) @ self.reference
                             + b_fine[:n_fine] - b_fine[n_fine:]))      # same kappa on any grid
        r_up, r_lo = self.reference[0] ** 2 / 2.0, self.reference[n_w] ** 2 / 2.0
        self.kappa = (self.le_factor * r_up / t_max ** 2, self.le_factor * r_lo / t_max ** 2)
        c_up, c_lo = np.sqrt(2.0 * self.kappa[0]), np.sqrt(2.0 * self.kappa[1])
        e_up, e_lo = np.zeros(2 * n_w), np.zeros(2 * n_w)
        e_up[0], e_lo[n_w] = 1.0, -1.0
        add(e_up[None, :] - c_up * a_t, c_up * b_t, "leading_edge")
        add(e_lo[None, :] - c_lo * a_t, c_lo * b_t, "leading_edge")
        self.g, self.h = np.vstack(rows), np.concatenate(rhs)
        self.row_names = np.array(names)
        self.ixx_v_factor, self.ixx_factor = float(ixx_v_factor), float(ixx_factor)
        ixx, ixx_v = section_stiffness(self.reference[None], te_thickness)
        self.ixx_reference, self.ixx_v_reference = float(ixx[0]), float(ixx_v[0])
        self.ixx_min = self.ixx_factor * self.ixx_reference
        self.ixx_v_min = self.ixx_v_factor * self.ixx_v_reference

    # --- membership ---
    def margins(self, weights: np.ndarray) -> np.ndarray:
        """G w - h (m, n_rows): distances to the linear constraints (>= 0 inside)."""
        return np.atleast_2d(weights) @ self.g.T - self.h

    def violations(self, weights: np.ndarray, tolerance: float = 1e-10) -> dict:
        """Per constraint name (and "ixx_v", "ixx"): booleans (m,), True where it is broken."""
        weights = np.atleast_2d(weights)
        margins = self.margins(weights)
        out = {name: np.any(margins[:, self.row_names == name] < -tolerance, axis=1)
               for name in dict.fromkeys(self.row_names)}
        ixx, ixx_v = section_stiffness(weights, self.te_thickness)
        out["ixx_v"] = ixx_v < self.ixx_v_min * (1.0 - tolerance)
        out["ixx"] = ixx < self.ixx_min * (1.0 - tolerance)
        return out

    def ixx_v_ok(self, weights: np.ndarray, tolerance: float = 1e-10) -> np.ndarray:
        """Booleans (m,): the Ixx / v constraint alone."""
        return section_stiffness(np.atleast_2d(weights), self.te_thickness)[1] \
            >= self.ixx_v_min * (1.0 - tolerance)

    def nonlinear_ok(self, weights: np.ndarray, tolerance: float = 1e-10) -> np.ndarray:
        """Booleans (m,): the non-linear constraints, Ixx / v and Ixx (if ixx_factor > 0)."""
        ixx, ixx_v = section_stiffness(np.atleast_2d(weights), self.te_thickness)
        return (ixx_v >= self.ixx_v_min * (1.0 - tolerance)) \
            & (ixx >= self.ixx_min * (1.0 - tolerance))

    def contains(self, weights: np.ndarray, tolerance: float = 1e-10) -> np.ndarray:
        """Booleans (m,): every linear constraint, Ixx / v and Ixx (SectionPCA family interface)."""
        weights = np.atleast_2d(weights)
        ok = np.all(self.margins(weights) >= -tolerance, axis=1)
        if ok.any():
            ok[ok] = self.nonlinear_ok(weights[ok], tolerance)
        return ok

    # --- sampling ---
    def chord(self, points: np.ndarray, directions: np.ndarray) -> tuple:
        """For each chain, the interval [s_min, s_max] of s with G (w + s d) >= h (w inside)."""
        margins = np.maximum(self.margins(points), 0.0)           # round-off at the boundary
        slopes = directions @ self.g.T
        with np.errstate(divide="ignore", invalid="ignore"):
            limits = -margins / slopes
        s_max = np.min(np.where(slopes < -1e-14, limits, np.inf), axis=1)
        s_min = np.max(np.where(slopes > 1e-14, limits, -np.inf), axis=1)
        return s_min, s_max

    def sample(self, n_sections: int, seed: int = 0, n_chains: int = 500, burn: int = 2000,
               thin: int = 10, start=None, max_shrinks: int = 30, log=None,
               basis=None) -> np.ndarray:
        """
        Sections uniform in the domain by hit-and-run (module docstring, section 2).

        Args:
        - n_sections: number of sections returned ((n_sections / n_chains) per chain).
        - seed, n_chains, burn (steps before recording), thin (steps between two records).
        - start: admissible weights (2n,) the chains start from; default the reference section.
        - log: function(str) for progress lines (optional).
        - basis: (2n, k) directions spanning an affine slice through `start` (e.g. sigma_j m_j of
          k PCA modes): the chains stay in start + span(basis), uniform on the slice of the
          domain. Default: the whole weight space.
        Returns:
        - numpy array (n_sections, 2n), every row in the domain.
        """
        rng = np.random.default_rng(seed)
        start = self.reference if start is None else np.asarray(start, dtype=float)
        if not self.contains(start[None], 1e-9)[0]:
            raise ValueError("the start of the chains is not in the domain")
        points = np.tile(start, (n_chains, 1))
        if basis is None:
            basis = np.eye(len(start))
            scale = np.diag(((self.box_upper - self.box_lower) / 2.0) ** 2)  # burn-in directions
        else:
            basis = np.asarray(basis, dtype=float)
            scale = np.eye(basis.shape[1])
        steps = np.zeros((n_chains, basis.shape[1]))          # coordinates in the basis
        per_chain = int(np.ceil(n_sections / n_chains))
        kept, n_rejected, n_draws = [], 0, 0
        for step in range(burn + per_chain * thin):
            if step == burn // 2:                     # directions from the chains themselves
                scale = np.cov(steps.T) + 1e-12 * np.eye(steps.shape[1])
            factor = np.linalg.cholesky(scale)
            local = rng.standard_normal(steps.shape) @ factor.T
            directions = local @ basis.T
            s_min, s_max = self.chord(points, directions)
            moving = np.ones(n_chains, dtype=bool)
            for _ in range(max_shrinks):
                s = s_min + (s_max - s_min) * rng.random(n_chains)
                trial = points + s[:, None] * directions
                ok = np.zeros(n_chains, dtype=bool)
                ok[moving] = self.nonlinear_ok(trial[moving])
                n_draws += int(moving.sum())
                accepted = moving & ok
                points[accepted] = trial[accepted]
                steps[accepted] += s[accepted, None] * local[accepted]
                moving &= ~ok
                n_rejected += int(moving.sum())
                if not moving.any():
                    break
                s_max = np.where(moving & (s > 0), s, s_max)        # shrink towards 0
                s_min = np.where(moving & (s < 0), s, s_min)
            if step >= burn and (step - burn) % thin == thin - 1:
                kept.append(points.copy())
            if log is not None and (step + 1) % 500 == 0:
                log(f"hit-and-run: step {step + 1} / {burn + per_chain * thin}, "
                    f"Ixx/v + Ixx rejections {n_rejected / max(n_draws, 1):.1%} of the draws")
        return np.vstack(kept)[:n_sections]

    def to_dict(self) -> dict:
        return {"reference": self.reference.tolist(), "te_thickness": self.te_thickness,
                "box_scale": self.box_scale, "band": self.band, "le_factor": self.le_factor,
                "inflections": list(self.inflections), "kappa": list(self.kappa),
                "ixx_v_factor": self.ixx_v_factor, "ixx_v_reference": self.ixx_v_reference,
                "ixx_v_min": self.ixx_v_min, "ixx_factor": self.ixx_factor,
                "ixx_reference": self.ixx_reference, "ixx_min": self.ixx_min, "box_lower": self.box_lower.tolist(),
                "box_upper": self.box_upper.tolist(), "x_lo": X_LO, "x_hi": X_HI}


def load_search_space(pca_run, k: int, box_name: str = "C_envelope"):
    """
    The search space written by scripts/build_section_pca.py and scripts/build_search_box.py in
    the run folder pca_run: (ScoreBoxFamily, PCA anchored on the current section with k modes,
    scores z0 of the current section).
    """
    import json  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415
    from src.section_pca import SectionPCA  # noqa: PLC0415
    pca_run = Path(pca_run)
    d = json.loads((pca_run / "data" / "domain.json").read_text(encoding="utf-8"))
    domain = SectionDomain(np.array(d["reference"]), d["te_thickness"], d["box_scale"], d["band"],
                           d["le_factor"], inflections=d["inflections"],
                           ixx_v_factor=d.get("ixx_v_factor", 1.0),
                           ixx_factor=d.get("ixx_factor", 0.0))
    pca = SectionPCA.load(pca_run / "data" / "pca.json").anchor(domain.reference, k)
    box = json.loads((pca_run / "data" / f"search_box_k{k}.json").read_text(encoding="utf-8"))
    box = box["boxes"][box_name]
    family = ScoreBoxFamily(domain, pca, box["lower"], box["upper"])
    return family, pca, pca.encode(domain.reference[None])[0]


class ScoreBoxFamily:
    """
    The search space of the optimizer (2026-11-05): the sections whose k PCA scores lie in a box
    and whose DECODED section is in the domain (all the constraints, Ixx / v and Ixx), checked
    when the sections are screened. Interface of RealFoilProblem (sample, contains, te_thickness,
    score_bounds).
    """

    def __init__(self, domain: SectionDomain, pca, lower, upper, batch: int = 5000):
        """domain; pca: SectionPCA with n_modes = k (anchored on the current section); lower,
        upper: the box of the k scores; batch: sections per constraint check (memory)."""
        self.domain, self.pca = domain, pca
        self.lower, self.upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
        self.te_thickness = domain.te_thickness
        self.batch = int(batch)

    @property
    def score_bounds(self) -> list:
        return [(float(a), float(b)) for a, b in zip(self.lower, self.upper)]

    def contains(self, weights: np.ndarray) -> np.ndarray:
        """Booleans (m,): scores in the box and decoded section in the domain."""
        weights = np.atleast_2d(weights)
        z = self.pca.encode(weights)
        ok = np.all((z >= self.lower - 1e-9) & (z <= self.upper + 1e-9), axis=1)
        for start in range(0, len(weights), self.batch):
            part = slice(start, start + self.batch)
            inside = ok[part].copy()
            if inside.any():
                inside[inside] = self.domain.contains(weights[part][inside])
            ok[part] = inside
        return ok

    def sample(self, n_sections: int, seed: int = 0) -> np.ndarray:
        """Weights (n, 2n_w) uniform in the box of the scores, kept when admissible."""
        rng = np.random.default_rng(seed)
        kept, total = [], 0
        while total < n_sections:
            z = self.lower + (self.upper - self.lower) * rng.random((self.batch, len(self.lower)))
            weights = self.pca.decode(z)
            kept.append(weights[self.contains(weights)])
            total += len(kept[-1])
        return np.vstack(kept)[:n_sections]
