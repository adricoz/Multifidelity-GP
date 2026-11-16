"""
Inputs for Hugo's robust 2D section optimisation (2026-11-13, asked through Antoine).

Hugo optimises a 2D section with only 4 geometric variables, under uncertainty on the operating
conditions. He needs:
  1. `profile`: the distribution of the boat speed found by the VPP and of the angle of attack of
     the C-foil at the VPP equilibrium (scalar lift-weighted alpha, and alpha along the span), for
     the current Intens foil (MC2 section = anchor of the PCA), by Monte Carlo on the analytic
     operating law of profiles/profile_v2.json (its "editor" block);
  2. `pca`: our section PCA (all matrices + how the sample was generated), its 4 first modes and
     the drag sensitivities dCD/dz_j (NeuralFoil 2D, at the (Re, alpha) of step 1);
  3. `thickness`: the t/c max threshold equivalent to our structural constraint
     Ixx/v >= 0.95 x reference (thickness of the section scaled, camber line kept).

Everything is written in runs/2026-11-13_hugo_inputs/ (README in French for Hugo).

Run (environment .venv-mds):
    .venv-mds\\Scripts\\python vpp_acceleration\\scripts\\hugo_inputs.py thickness
    .venv-mds\\Scripts\\python vpp_acceleration\\scripts\\hugo_inputs.py pca
    .venv-mds\\Scripts\\python vpp_acceleration\\scripts\\hugo_inputs.py profile --n 500 --workers 4
    .venv-mds\\Scripts\\python vpp_acceleration\\scripts\\hugo_inputs.py sensitivities
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import trapezoid

VPP_DIR = Path(__file__).resolve().parents[1]
for _path in (VPP_DIR, VPP_DIR.parent, VPP_DIR.parent / "mfego", VPP_DIR / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
# pylint: disable=wrong-import-position
from src.section_domain import load_search_space  # noqa: E402
from src.section_pca import (THICKNESS_STATIONS, SectionPCA, section_coordinates,  # noqa: E402
                             section_stiffness)

OUT = VPP_DIR / "runs" / "2026-11-13_hugo_inputs"
PCA_RUN = VPP_DIR / "runs" / "2026-11-11_1054_section_pca_convex_ixxv095"
OPTIM_RUN = VPP_DIR / "runs" / "2026-11-12_1118_ixxv095_rigid_xtr003_k10_nocfd"
BEST_RUN = VPP_DIR / "runs" / "2026-11-11_1245_ixxv095_rigid_xtr003_k10_cfd3d"
MC2_XF = VPP_DIR.parents[1] / "bdfoil_v1" / "tests" / "data" / "cfoil_r9800" \
    / "MC2_60_6101_1_re5M.xf"
K_SEARCH = 10          # modes of the optimisation
K_HUGO = 4             # modes given to Hugo


# -------------------------------------------------------------------------------------------------
# Geometry helpers
# -------------------------------------------------------------------------------------------------
def read_xf(path: Path, x=THICKNESS_STATIONS) -> tuple:
    """
    Upper and lower surfaces of a Selig-like file (TE -> upper -> LE -> lower -> TE), normalised to
    a chord of 1 (x from the leading edge), interpolated on the stations x.

    Returns:
    - (y_upper, y_lower): numpy arrays (len(x),).
    """
    rows = []
    for line in path.read_text(encoding="utf-8-sig").splitlines()[1:]:
        parts = line.split()
        if len(parts) >= 2:
            rows.append((float(parts[0]), float(parts[1])))
    xy = np.array(rows)
    i_le = int(np.argmin(xy[:, 0]))
    x_le, x_te = xy[i_le, 0], max(xy[0, 0], xy[-1, 0])
    xy[:, 0] = (xy[:, 0] - x_le) / (x_te - x_le)
    xy[:, 1] = (xy[:, 1] - xy[i_le, 1]) / (x_te - x_le)
    first, second = xy[:i_le + 1][::-1], xy[i_le:]          # both from the leading edge
    up, lo = (first, second) if np.mean(first[:, 1]) > np.mean(second[:, 1]) else (second, first)
    return np.interp(x, up[:, 0], up[:, 1]), np.interp(x, lo[:, 0], lo[:, 1])


def stiffness_of(y_upper, y_lower, x=THICKNESS_STATIONS) -> dict:
    """Same formulas as section_pca.section_stiffness, on given surfaces (chord 1)."""
    tau, camber = y_upper - y_lower, (y_upper + y_lower) / 2.0
    y_g = trapezoid(tau * camber, x) / trapezoid(tau, x)
    a = trapezoid(tau ** 3 / 12.0, x)                         # ~ f^3 when tau -> f tau
    b = trapezoid(tau * (camber - y_g) ** 2, x)               # ~ f   when tau -> f tau
    i_max = int(np.argmax(tau))
    return {"t_max": float(tau[i_max]), "x_tmax": float(x[i_max]),
            "t_at_0.3": float(np.interp(0.3, x, tau)), "ixx": float(a + b),
            "ixx_v": float((a + b) / (tau[i_max] / 2.0)), "a": float(a), "b": float(b),
            "camber_max": float(np.max(camber)), "x_camber_max": float(x[np.argmax(camber)])}


def scale_for_ixx_v(props: dict, target: float) -> float:
    """
    Factor f on the thickness (camber line kept) that gives Ixx/v = target:
    Ixx(f) = f^3 a + f b and v(f) = f t_max / 2, so Ixx/v(f) = 2 (f^2 a + b) / t_max.
    (y_g does not depend on f: it is a tau-weighted mean of the camber.)
    """
    f2 = (target * props["t_max"] / 2.0 - props["b"]) / props["a"]
    return float(np.sqrt(f2))


def kulfan_surfaces(weights, x=THICKNESS_STATIONS, te=0.00197) -> tuple:
    y = section_coordinates(np.asarray(weights)[None], x, te)[0]
    return y[:len(x)], y[len(x):]


def best_weights(run: Path) -> np.ndarray:
    d = json.loads((run / "results" / "best_section.json").read_text(encoding="utf-8"))
    return np.concatenate([d["kulfan_upper"], d["kulfan_lower"]])


# -------------------------------------------------------------------------------------------------
# 3. Thickness threshold
# -------------------------------------------------------------------------------------------------
def thickness(_args) -> dict:
    """t/c max equivalent to Ixx/v >= 0.95 x reference, two variants (plan, step 3)."""
    domain = json.loads((PCA_RUN / "data" / "domain.json").read_text(encoding="utf-8"))
    te = domain["te_thickness"]
    sections = {"MC2 fichier CFD (MC2_60_6101_1_re5M.xf)": read_xf(MC2_XF),
                "MC2 ancre PCA (Kulfan, section actuelle de l'optim)":
                    kulfan_surfaces(domain["reference"], te=te),
                "best 686f (run 1245)": kulfan_surfaces(best_weights(BEST_RUN), te=te)}
    props = {name: stiffness_of(*s) for name, s in sections.items()}
    # check: our formula = section_stiffness on the Kulfan anchor
    ixx, ixx_v = section_stiffness(np.array(domain["reference"])[None], te)
    anchor = props["MC2 ancre PCA (Kulfan, section actuelle de l'optim)"]
    assert abs(anchor["ixx_v"] / ixx_v[0] - 1) < 1e-9 and abs(anchor["ixx"] / ixx[0] - 1) < 1e-9
    thresholds = {"(a) seuil absolu de l'optim = 0.95 x Ixx/v(ancre)": domain["ixx_v_min"],
                  "(b) 0.95 x Ixx/v(fichier MC2)":
                      0.95 * props["MC2 fichier CFD (MC2_60_6101_1_re5M.xf)"]["ixx_v"]}
    equivalents = {}
    for t_name, target in thresholds.items():
        for s_name in list(sections)[:2]:
            f = scale_for_ixx_v(props[s_name], target)
            equivalents[f"{t_name} | forme {s_name}"] = {
                "ixx_v_target": target, "f": f, "t_max_equivalent": f * props[s_name]["t_max"],
                "x_tmax": props[s_name]["x_tmax"]}
    out = {"sections": props, "thresholds": thresholds, "equivalents": equivalents,
           "status": {name: {t: p["ixx_v"] >= v * (1 - 1e-9) for t, v in thresholds.items()}
                      for name, p in props.items()}}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "thickness.json").write_text(json.dumps(out, indent=1, ensure_ascii=False),
                                        encoding="utf-8")
    print(json.dumps(out, indent=1, ensure_ascii=False))
    return out


# -------------------------------------------------------------------------------------------------
# 2. PCA export
# -------------------------------------------------------------------------------------------------
def pca_export(args) -> dict:
    """Everything Hugo needs to rebuild our sections from 4 scores (plan, step 2.1-2.3, 2.5)."""
    family, pca10, z0 = load_search_space(PCA_RUN, K_SEARCH, "C_envelope")
    domain = family.domain
    raw = SectionPCA.load(PCA_RUN / "data" / "pca.json")                 # mean = sample mean
    pca4 = SectionPCA.load(PCA_RUN / "data" / "pca.json").anchor(domain.reference, K_HUGO)
    # anchoring with 4 modes = decoding with 10 modes and z5..z10 frozen at z0
    z = np.zeros((3, K_SEARCH)) + z0
    z[1, :4] += [1.0, -0.5, 0.3, 0.2]
    z[2, :4] = [-2.0, 1.0, 0.0, -1.0]
    gap = np.max(np.abs(pca10.decode(z) - pca4.decode(z[:, :K_HUGO])))
    assert gap < 1e-12, gap
    assert np.max(np.abs(pca4.decode(z0[None, :K_HUGO])[0] - domain.reference)) < 1e-12
    te, x = domain.te_thickness, pca4.stations
    n_x = len(x)

    # per mode: displacement of the surfaces for one unit of score, d t_max / dz
    reference_t = (lambda y: y[:n_x] - y[n_x:])(pca4.coordinates(domain.reference)[0])
    modes = []
    for j in range(K_SEARCH):
        dy = pca4.sigmas[j] * pca4.modes[:, j]
        dw = pca4.sigmas[j] * pca4.weight_modes[:, j]
        h = 1e-3
        zp, zm = z0[:K_SEARCH].copy(), z0[:K_SEARCH].copy()
        zp[j] += h
        zm[j] -= h
        t_p = section_stiffness(pca10.decode(zp[None]), te)
        t_m = section_stiffness(pca10.decode(zm[None]), te)
        tmax = [np.max((lambda y: y[:n_x] - y[n_x:])(pca10.coordinates(pca10.decode(v[None]))[0]))
                for v in (zp, zm)]
        modes.append({"mode": j + 1, "variance_pct": 100 * float(pca4.variance_ratio[j]),
                      "sigma": float(pca4.sigmas[j]), "max_abs_dy_per_unit_z": float(np.max(np.abs(dy))),
                      "dtmax_dz": float((tmax[0] - tmax[1]) / (2 * h)),
                      "dixxv_dz_rel": float((t_p[1][0] - t_m[1][0]) / (2 * h) / domain.ixx_v_reference),
                      "kulfan_direction_per_unit_z": dw.tolist()})

    # 4D box given to Hugo: C_envelope bounds of modes 1-4 (z5..z10 frozen), admissible fraction
    lower, upper = family.lower[:K_HUGO], family.upper[:K_HUGO]
    rng = np.random.default_rng(args.seed)
    zz = lower + (upper - lower) * rng.random((args.n_box, K_HUGO))
    w4 = pca4.decode(zz)
    ok = domain.contains(w4)
    broken = domain.violations(w4)
    broken = {k: float(np.mean(v)) for k, v in broken.items()}
    # tighter box: 0.5-99.5 % of the admissible points (hint for Hugo)
    if ok.sum() > 50:
        tight = np.percentile(zz[ok], [0.5, 99.5], axis=0)
    else:
        tight = np.vstack([lower, upper])

    # the real MC2 file in our PCA: projection errors with 10 and 4 modes
    y_up, y_lo = read_xf(MC2_XF, x)
    y_mc2 = np.concatenate([y_up, y_lo])
    w_mc2 = np.linalg.lstsq(pca4.matrix, y_mc2 - pca4.offset, rcond=None)[0]   # 16-weight fit
    fit16 = float(np.max(np.abs(pca4.coordinates(w_mc2)[0] - y_mc2)))
    proj = {"kulfan16_fit_gap": fit16}
    for k, p in ((K_SEARCH, pca10), (K_HUGO, pca4)):
        zk = p.encode(w_mc2[None], k)[0]
        proj[f"k{k}"] = {"scores": zk.tolist(),
                         "gap_max_chord": float(np.max(np.abs(p.coordinates(p.decode(zk[None]))[0]
                                                               - y_mc2))),
                         "admissible": bool(domain.contains(p.decode(zk[None]))[0])}
    proj["anchor_vs_mc2_file_gap_max_chord"] = float(np.max(np.abs(
        pca4.coordinates(domain.reference)[0] - y_mc2)))

    domain_json = json.loads((PCA_RUN / "data" / "domain.json").read_text(encoding="utf-8"))
    config = json.loads((PCA_RUN / "config.json").read_text(encoding="utf-8"))
    summary = json.loads((PCA_RUN / "results" / "summary.json").read_text(encoding="utf-8"))
    export = {
        "convention": {
            "kulfan": "y(x) = C(x) sum_i w_i B_i(x) +/- x te/2, C(x) = sqrt(x) (1 - x) (N1 = 0.5,"
                      " N2 = 1), B_i Bernstein degree 7 (8 weights per side), chord 1",
            "weights_order": "w = (8 upper weights, then 8 lower weights; lower weights < 0)",
            "te_thickness": te,
            "stations": "200 uniform stations x in [0, 1]; coordinates = (y_upper at the "
                        "stations, then y_lower at the stations), 400 values",
            "scores": "whitened: z_j = (y - y_mean) . v_j / sigma_j (unit variance on the sample)",
            "decode_10": "w = w_mean_anchor10 + sum_{j<=10} sigma_j z_j m_j",
            "decode_4": "w = w_mean_anchor4 + sum_{j<=4} sigma_j z_j m_j  (= decode_10 with "
                        "z5..z10 frozen at the scores of MC2)",
            "coordinates_from_scores_4": "y = y_mean_anchor4 + sum_{j<=4} sigma_j z_j v_j",
        },
        "stations": x.tolist(),
        "sigmas": pca4.sigmas.tolist(),
        "variance_ratio": pca4.variance_ratio.tolist(),
        "modes_coordinates_v": pca4.modes.tolist(),          # (400, 16), unit norm
        "modes_weights_m": pca4.weight_modes.tolist(),       # (16, 16), A m_j = v_j
        "mean_weights_sample": raw.mean_weights.tolist(),
        "mean_weights_anchor10": pca10.mean_weights.tolist(),
        "mean_weights_anchor4": pca4.mean_weights.tolist(),
        "mean_coordinates_anchor4": pca4.mean_coordinates.tolist(),
        "reference_weights_mc2_anchor": domain.reference.tolist(),
        "reference_scores_z0": z0.tolist(),
        "box_4_modes": {"lower": lower.tolist(), "upper": upper.tolist(),
                        "admissible_fraction": float(ok.mean()),
                        "broken_fraction_per_constraint": broken,
                        "admissible_0.5_99.5_pct": {"lower": tight[0].tolist(),
                                                    "upper": tight[1].tolist()}},
        "box_10_modes_C_envelope": {"lower": family.lower.tolist(),
                                    "upper": family.upper.tolist()},
        "modes": modes,
        "mc2_file_projection": proj,
        "generation": {"sampler": "hit-and-run (Smith 1984), uniform in the convex domain, "
                                  "slice-sampling shrinkage for Ixx/v; NOT a Latin hypercube",
                       "config": config, "n_sections_kept": summary["n_sections"],
                       "dropped_fine_check": summary["dropped_fine_check"],
                       "domain": {k: domain_json[k] for k in
                                  ("te_thickness", "box_scale", "band", "le_factor",
                                   "inflections", "kappa", "ixx_v_factor", "ixx_v_reference",
                                   "ixx_v_min", "box_lower", "box_upper", "x_lo", "x_hi")},
                       "constraints": [
                           "Kulfan box: box_scale x (Dim-Reduction box) centred on MC2 anchor",
                           "no crossing: t(x) >= 0",
                           "upper surface convex: y_u'' <= 0 on [0.01, 0.99]",
                           "lower surface sign zones: y_l'' >= 0 on [0.01, x1-g], <= 0 on "
                           "[x1+g, x2-g], >= 0 on [x2+g, 0.99], x1/x2 = inflections of MC2, g=band",
                           "round leading edge: |w0| >= sqrt(2 kappa) t(x) at every station "
                           "(per surface)",
                           "Ixx/v >= ixx_v_min, v = t_max/2, Ixx by vertical strips with camber"]},
        "source": str(PCA_RUN.relative_to(VPP_DIR)),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "pca_hugo.json").write_text(json.dumps(export), encoding="utf-8")
    # light CSVs for people who do not read JSON
    import csv  # noqa: PLC0415
    with open(OUT / "pca_modes_weights.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["weight"] + [f"sigma_m{j + 1}" for j in range(K_SEARCH)] + ["mc2_anchor",
                                                                                "mean_anchor4"])
        names = [f"upper_{i}" for i in range(8)] + [f"lower_{i}" for i in range(8)]
        for i, name in enumerate(names):
            w.writerow([name] + [modes[j]["kulfan_direction_per_unit_z"][i]
                                 for j in range(K_SEARCH)]
                       + [domain.reference[i], pca4.mean_weights[i]])
    with open(OUT / "pca_modes_dy.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["x", "surface", "y_mc2_anchor"] + [f"dy_mode{j + 1}_per_unit_z"
                                                       for j in range(K_SEARCH)])
        y_ref = pca4.coordinates(domain.reference)[0]
        for side, off in (("upper", 0), ("lower", n_x)):
            for i in range(n_x):
                w.writerow([x[i], side, y_ref[off + i]]
                           + [pca4.sigmas[j] * pca4.modes[off + i, j] for j in range(K_SEARCH)])
    short = {k: v for k, v in export.items() if k in ("box_4_modes", "mc2_file_projection",
                                                      "reference_scores_z0")}
    short["modes"] = [{k: v for k, v in m.items() if k != "kulfan_direction_per_unit_z"}
                      for m in modes]
    print(json.dumps(short, indent=1))
    del reference_t
    return export


def constraints_export(_args) -> dict:
    """
    The linear constraints of the domain written on the 4 scores (z5..z10 frozen at MC2):
    G w >= h with w = w_mean4 + S z, S = (sigma_j m_j)_(j<=4), gives (G S) z >= h - G w_mean4.
    Rows scaled to unit norm in z (the margin is then a distance in score units); rows that do not
    depend on z (always satisfied at MC2) dropped. Ixx/v is not linear: formula given apart.
    """
    import csv  # noqa: PLC0415
    family, _, z0 = load_search_space(PCA_RUN, K_SEARCH, "C_envelope")
    domain = family.domain
    pca4 = SectionPCA.load(PCA_RUN / "data" / "pca.json").anchor(domain.reference, K_HUGO)
    s = pca4.weight_modes[:, :K_HUGO] * pca4.sigmas[:K_HUGO]               # (16, 4)
    a = domain.g @ s
    b = domain.h - domain.g @ pca4.mean_weights
    norm = np.linalg.norm(a, axis=1)
    keep = norm > 1e-12
    assert np.all(b[~keep] <= 1e-10), "a z-independent row is broken at MC2"
    a, b, names = a[keep] / norm[keep, None], b[keep] / norm[keep], domain.row_names[keep]
    # check against the domain on random points of the 4D box
    rng = np.random.default_rng(3)
    zz = family.lower[:K_HUGO] + (family.upper[:K_HUGO] - family.lower[:K_HUGO]) \
        * rng.random((20000, K_HUGO))
    lin_z = np.all(zz @ a.T - b >= -1e-10, axis=1)
    lin_w = np.all(domain.margins(pca4.decode(zz)) >= -1e-10, axis=1)
    assert np.array_equal(lin_z, lin_w)
    margin0 = z0[:K_HUGO] @ a.T - b
    with open(OUT / "constraints_z4.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["constraint", "a_z1", "a_z2", "a_z3", "a_z4", "b"])
        for name, row, rhs in zip(names, a, b):
            w.writerow([name, *row, rhs])
    info = {"rows": int(len(b)), "rows_per_constraint": {n: int(np.sum(names == n))
                                                         for n in dict.fromkeys(names)},
            "check_vs_domain_20000_points": "identical", "min_margin_at_mc2": float(margin0.min()),
            "linear_admissible_fraction_of_box": float(lin_w.mean())}
    (OUT / "constraints_z4.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    print(json.dumps(info, indent=1))
    return info


def sensitivities(args) -> dict:
    """
    dCD/dz_j and dCL/dz_j (2D NeuralFoil xxxlarge, n_crit 1, xtr 3 %), j = 1..10, around MC2
    (anchor, other scores at z0), at the (alpha_eq, Re) of the Monte Carlo draws (step 1):
      * at fixed alpha (Hugo imposes alpha);
      * at fixed CL (the CL of MC2 at that alpha: the VPP imposes the lift, not the angle).
    Central differences with h and 2h (linearity check).
    """
    import neuralfoil as nf  # noqa: PLC0415
    import pandas as pd  # noqa: PLC0415
    from src.section_pca import kulfan_dict  # noqa: PLC0415
    _, pca10, z0 = load_search_space(PCA_RUN, K_SEARCH, "C_envelope")
    te = pca10.te_thickness
    table = pd.read_csv(args.samples or OUT / "profile_samples.csv")
    ok = table[table["strips_ok"] == True]                                   # noqa: E712
    rng = np.random.default_rng(args.seed)
    pick = ok.iloc[rng.permutation(len(ok))[:args.m]]
    alphas, res = pick["alpha_eq_deg"].to_numpy(), pick["re_chord_vpp"].to_numpy()
    grid = np.arange(-3.0, 3.01, 0.25)                                   # around each alpha
    h_list = (args.h, 2 * args.h)
    zs = [z0]
    for j in range(K_SEARCH):
        for h in h_list:
            for sign in (1, -1):
                z = z0.copy()
                z[j] += sign * h
                zs.append(z)
    weights = pca10.decode(np.array(zs))                                 # (n_sec, 16)
    n_sec, n_c, n_g = len(weights), len(alphas), len(grid)
    # every section x condition x local alpha grid, one vectorised call
    w_rep = np.repeat(weights, n_c * n_g, axis=0)
    a_rep = np.tile((alphas[:, None] + grid[None, :]).ravel(), n_sec)
    re_rep = np.tile(np.repeat(res, n_g), n_sec)
    aero = nf.get_aero_from_kulfan_parameters(kulfan_dict(w_rep, te), alpha=a_rep, Re=re_rep,
                                              n_crit=1.0, xtr_upper=0.03, xtr_lower=0.03,
                                              model_size="xxxlarge")
    cl = np.asarray(aero["CL"]).reshape(n_sec, n_c, n_g)
    cd = np.asarray(aero["CD"]).reshape(n_sec, n_c, n_g)
    conf = np.asarray(aero["analysis_confidence"]).reshape(n_sec, n_c, n_g)
    i0 = int(np.argmin(np.abs(grid)))                                     # grid point alpha_eq
    cl_target = cl[0, :, i0]
    cd_fixed_cl = np.full((n_sec, n_c), np.nan)
    for s in range(n_sec):
        for c in range(n_c):
            x = cl[s, c]
            if np.all(np.diff(x) > 0) and x[0] <= cl_target[c] <= x[-1]:
                cd_fixed_cl[s, c] = np.interp(cl_target[c], x, cd[s, c])
    rows = []
    for j in range(K_SEARCH):
        row = {"mode": j + 1}
        for k, h in enumerate(h_list):
            ip = 1 + j * 4 + k * 2
            im = ip + 1
            d_cd_a = (cd[ip, :, i0] - cd[im, :, i0]) / (2 * h)
            d_cl_a = (cl[ip, :, i0] - cl[im, :, i0]) / (2 * h)
            d_cd_cl = (cd_fixed_cl[ip] - cd_fixed_cl[im]) / (2 * h)
            tag = "" if k == 0 else "_2h"
            for name, v in (("dCD_dz_fixed_alpha", d_cd_a), ("dCL_dz_fixed_alpha", d_cl_a),
                            ("dCD_dz_fixed_CL", d_cd_cl)):
                row[f"{name}{tag}_mean"] = float(np.nanmean(v))
                if k == 0:
                    row[f"{name}_p05"] = float(np.nanpercentile(v, 5))
                    row[f"{name}_p95"] = float(np.nanpercentile(v, 95))
                    row[f"{name}_rel_mean_pct"] = float(np.nanmean(
                        v / (cd[0, :, i0] if "CD" in name else np.abs(cl[0, :, i0]))) * 100)
        rows.append(row)
    out = {"n_conditions": int(n_c), "h": args.h, "flow": "NeuralFoil xxxlarge, n_crit 1, "
           "xtr 3 % both sides, Re = V_vpp c / nu, alpha = alpha_eq of the draw",
           "mc2": {"CL_mean": float(np.mean(cl[0, :, i0])), "CD_mean": float(np.mean(cd[0, :, i0])),
                   "alpha_mean": float(alphas.mean()), "Re_mean": float(res.mean())},
           "confidence_min": float(conf[:, :, i0].min()),
           "confidence_p05": float(np.percentile(conf[:, :, i0], 5)),
           "fixed_CL_missing_pct": float(100 * np.mean(np.isnan(cd_fixed_cl))),
           "modes": rows}
    dest = OUT if args.samples is None else Path(args.samples).parent
    (dest / "sensitivities.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    pd.DataFrame(rows).to_csv(dest / "sensitivities.csv", index=False)
    print(json.dumps({k: v for k, v in out.items() if k != "modes"}, indent=1))
    print(pd.DataFrame(rows)[["mode", "dCD_dz_fixed_alpha_mean", "dCD_dz_fixed_alpha_2h_mean",
                              "dCD_dz_fixed_alpha_rel_mean_pct", "dCL_dz_fixed_alpha_mean",
                              "dCD_dz_fixed_CL_mean", "dCD_dz_fixed_CL_2h_mean",
                              "dCD_dz_fixed_CL_rel_mean_pct"]].to_string(index=False))
    return out


def _xfoil_polar(task):
    """One XFOIL polar (bdfoil_v1 through pipelines XfoilBackend), NaN arrays on failure."""
    weights, re, alphas, xtr, timeout = task
    from src.foil_codes import FoilCodes  # noqa: PLC0415  (puts the repo on sys.path)
    from pipelines.bdtoolbox_foil import bridge  # noqa: PLC0415
    from pipelines.bdtoolbox_foil.solvers import XfoilBackend  # noqa: PLC0415
    bridge.setup(bridge.resolve_paths())      # bdfoil_v1/src on sys.path, executables
    backend = XfoilBackend({"re": re, "n_crit": 1.0, "xtr": (xtr, xtr),
                            "alpha": (float(alphas[0]), float(alphas[-1]),
                                      float(alphas[1] - alphas[0]))}, timeout=timeout)
    try:
        polar = backend.polar(FoilCodes.section(np.asarray(weights)))
        return (np.interp(alphas, polar.alpha, polar.cl, left=np.nan, right=np.nan),
                np.interp(alphas, polar.alpha, polar.cd, left=np.nan, right=np.nan), "")
    except Exception as error:  # noqa: BLE001
        return np.full(len(alphas), np.nan), np.full(len(alphas), np.nan), str(error)[:200]


def xfoil_check(args) -> dict:
    """dCD/dz_j, dCL/dz_j of modes 1-4 at a few alphas: XFOIL vs NeuralFoil (same flow)."""
    from concurrent.futures import ProcessPoolExecutor  # noqa: PLC0415
    import neuralfoil as nf  # noqa: PLC0415
    from src.section_pca import kulfan_dict  # noqa: PLC0415
    _, pca10, z0 = load_search_space(PCA_RUN, K_SEARCH, "C_envelope")
    alphas = np.arange(-3.0, 4.01, 0.5)
    check = [float(a) for a in args.alphas]
    zs = [z0]
    for j in range(K_HUGO):
        for sign in (1, -1):
            z = z0.copy()
            z[j] += sign * args.h
            zs.append(z)
    weights = pca10.decode(np.array(zs))
    tasks = [(w.tolist(), args.re, alphas, args.xtr, args.timeout) for w in weights]
    with ProcessPoolExecutor(args.workers) as ex:
        xf = list(ex.map(_xfoil_polar, tasks))
    rep = np.repeat(weights, len(alphas), axis=0)
    aero = nf.get_aero_from_kulfan_parameters(kulfan_dict(rep, pca10.te_thickness),
                                              alpha=np.tile(alphas, len(weights)), Re=args.re,
                                              n_crit=1.0, xtr_upper=args.xtr, xtr_lower=args.xtr,
                                              model_size="xxxlarge")
    nf_cl = np.asarray(aero["CL"]).reshape(len(weights), -1)
    nf_cd = np.asarray(aero["CD"]).reshape(len(weights), -1)
    idx = [int(np.argmin(np.abs(alphas - a))) for a in check]
    rows = []
    for j in range(K_HUGO):
        ip, im = 1 + 2 * j, 2 + 2 * j
        for a, i in zip(check, idx):
            rows.append({"mode": j + 1, "alpha": a,
                         "xfoil_dCD_dz": (xf[ip][1][i] - xf[im][1][i]) / (2 * args.h),
                         "nf_dCD_dz": (nf_cd[ip, i] - nf_cd[im, i]) / (2 * args.h),
                         "xfoil_dCL_dz": (xf[ip][0][i] - xf[im][0][i]) / (2 * args.h),
                         "nf_dCL_dz": (nf_cl[ip, i] - nf_cl[im, i]) / (2 * args.h),
                         "xfoil_CD_mc2": xf[0][1][i], "nf_CD_mc2": nf_cd[0, i]})
    out = {"re": args.re, "xtr": args.xtr, "h": args.h,
           "xfoil_errors": [e for _, _, e in xf if e], "rows": rows}
    (OUT / "xfoil_check.json").write_text(json.dumps(out, indent=1, default=float),
                                          encoding="utf-8")
    import pandas as pd  # noqa: PLC0415
    print(out["xfoil_errors"])
    print(pd.DataFrame(rows).round(6).to_string(index=False))
    return out


# -------------------------------------------------------------------------------------------------
# 1. Speed and alpha at the VPP equilibrium, Monte Carlo on the analytic operating law
# -------------------------------------------------------------------------------------------------
PROFILE = VPP_DIR / "profiles" / "profile_v2.json"
NU_SEAWATER = 8.49 * 0.70 / 5e6      # foil_codes: 8.49 m/s gives Re 5e6 on the 0.70 m chord
KT = 0.514444
SPEED_CAP_KT = 24.0                  # MDS hull data limit (IntensSY_config speed_box)


def sample_profile(editor: dict, n: int, seed: int) -> list:
    """
    n draws of the analytic law of the profile editor (Bimodal-Sailing-Discretizer): with the total
    weight of the hand-placed points, one of them (by its weight); otherwise a mode (by its weight),
    then TWA and TWS independent truncated normals of the mode (slope = TWA shift per kt of TWS
    from the mode mean, 0 in profile_v2). Same law as the nodes of operating_profile.py.
    """
    from scipy.stats import truncnorm  # noqa: PLC0415
    rng = np.random.default_rng(seed)
    manual = editor.get("manual", [])
    p_manual = sum(m["weight"] for m in manual)
    modes = editor["modes"]
    p_modes = np.array([m["weight"] for m in modes], dtype=float)
    p_modes /= p_modes.sum()

    def draw(spec, size):
        a, b = (spec["lower"] - spec["mean"]) / spec["std"], (spec["upper"] - spec["mean"]) / spec["std"]
        return truncnorm.rvs(a, b, loc=spec["mean"], scale=spec["std"], size=size, random_state=rng)

    out = []
    for _ in range(n):
        if manual and rng.random() < p_manual:
            m = manual[rng.choice(len(manual), p=[p["weight"] / p_manual for p in manual])]
            out.append({"twa": float(m["twa"]), "tws": float(m["tws"]), "mode": "manual"})
            continue
        mode = modes[rng.choice(len(modes), p=p_modes)]
        tws = float(draw(mode["tws"], 1)[0])
        twa = float(draw(mode["twa"], 1)[0]) + mode.get("slope", 0.0) * (tws - mode["tws"]["mean"])
        out.append({"twa": twa, "tws": tws, "mode": mode["name"]})
    return out


_STRIP_WORKER = {}


def _init_strip_worker(work_dir):
    from src.foil_codes import LEVELS, FoilCodes  # noqa: PLC0415
    _STRIP_WORKER["codes"] = FoilCodes(work_dir, levels=LEVELS)       # 1 nf_llt, 2 avl (run 1118)


def _strip_task(task):
    """AVL x XFOIL (level 2 of run 1118) at one attitude: integral outputs + strips."""
    # pylint: disable=import-outside-toplevel
    from pipelines.bdtoolbox_foil.geometry import write_xf
    from pipelines.bdtoolbox_foil.planform import write_planform_csv
    import tempfile  # noqa: PLC0415
    weights, attitude, level = task
    codes = _STRIP_WORKER["codes"]
    section = codes.section(np.asarray(weights))
    backend = codes.simulator.backends[level - 1]
    cant, rake, yaw = (float(a) for a in attitude)
    try:
        with tempfile.TemporaryDirectory(prefix="hugo_", ignore_cleanup_errors=True) as tmp:
            work = Path(tmp)
            xf_path = write_xf(section, work)
            csv_path = write_planform_csv(codes.simulator.stations, xf_path.name, work / "planform.csv")
            out = backend.evaluate_planform(csv_path, work, with_strips=True,
                                            attitude={"cant": cant, "rake": rake, "yaw": yaw,
                                                      "sink": 0.0})
        return {"ok": True, "strips": out.pop("strips"),
                "CL": out.get("CL [-]", np.nan), "Cy": out.get("Cy [-]"), "Cz": out.get("Cz [-]"),
                "converged": out.get("converged")}
    except Exception as error:  # noqa: BLE001  (a failed state is reported, not fatal)
        return {"ok": False, "error": f"{type(error).__name__}: {error}"[:300]}


def profile(args) -> None:
    """Step 1 of the plan: MC draws -> VPP equilibria (MC2) -> strips (AVL x XFOIL)."""
    import csv  # noqa: PLC0415
    import os  # noqa: PLC0415
    import shutil  # noqa: PLC0415
    import time  # noqa: PLC0415
    from concurrent.futures import ProcessPoolExecutor  # noqa: PLC0415
    from profile_analysis import _ScratchFolder, load_surrogate, robust_solve  # noqa: PLC0415
    # the flow of run 1118 for the workers (spawned processes import foil_codes again)
    os.environ["VPP_FOIL_XTR"] = "0.03,0.03"
    os.environ["VPP_FOIL_XTR_FALLBACK"] = "off"
    from src.vpp_optimize import GENERIC_GUESSES, VPPPool  # noqa: PLC0415

    start = time.time()
    out = OUT if args.out is None else Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "profile.log", "a", encoding="utf-8")       # noqa: SIM115

    def say(msg):
        line = f"[{time.time() - start:7.0f} s] {msg}"
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    editor = json.loads(PROFILE.read_text(encoding="utf-8"))["editor"]
    samples = sample_profile(editor, args.n, args.seed)
    best = json.loads((OPTIM_RUN / "results" / "best_section.json").read_text(encoding="utf-8"))
    scores = np.array(best["start_scores"])                         # MC2 anchor (checked)
    weights = SectionPCA.load(OPTIM_RUN / "data" / "pca.json").decode(scores[None])[0]
    say(f"{len(samples)} draws, section {best['start']}, {args.workers} workers")

    # VPP: the 49 nodes first (robust_solve), their equilibria seed the draws
    surrogate = load_surrogate(OPTIM_RUN)
    pool = VPPPool(_ScratchFolder(), args.workers, {"foil": "CFoilFSI", "with_hvol": False,
                                                    "fsi": False})
    pool.publish(surrogate)
    nodes = json.loads(PROFILE.read_text(encoding="utf-8"))["nodes"]
    node_cond = [(nd["twa"], nd["tws"]) for nd in nodes]
    node_states = robust_solve(pool, [scores], [1.0], node_cond)[0]
    seeds = [(c, s["unknowns"]) for c, s in zip(node_cond, node_states) if s["success"]]
    say(f"nodes: {len(seeds)}/{len(node_cond)} solved")
    seed_xy = np.array([[c[0] / 10.0, c[1] / 3.0] for c, _ in seeds])
    conditions = [(s["twa"], s["tws"]) for s in samples]
    extra = []
    for twa, tws in conditions:
        d = np.hypot(seed_xy[:, 0] - twa / 10.0, seed_xy[:, 1] - tws / 3.0)
        extra.append(list(GENERIC_GUESSES) + [seeds[i][1] for i in np.argsort(d)[:3]])
    states = pool.solve([scores], [1.0], conditions, extra_guesses=[extra])[0]
    pool.close()
    n_ok = sum(s["success"] for s in states)
    say(f"VPP: {n_ok}/{len(states)} converged")

    # strips at every converged state
    work = out / "codes"
    if not (work / "polar_cache").exists():
        shutil.copytree(OPTIM_RUN / "data" / "codes" / "polar_cache", work / "polar_cache")
    tasks = [(weights.tolist(), (s["Cant_Eff"], s["Rake_Eff"], s["Yaw_Eff"]), 2)
             for s in states if s["success"]]
    with ProcessPoolExecutor(args.workers, initializer=_init_strip_worker,
                             initargs=(str(work),)) as ex:
        strip_results = list(ex.map(_strip_task, tasks))
    say(f"strips: {sum(r['ok'] for r in strip_results)}/{len(tasks)} ok")

    # tables
    it = iter(strip_results)
    sample_rows, strip_rows = [], []
    for i, (smp, st) in enumerate(zip(samples, states)):
        row = {"sample": i, "mode": smp["mode"], "twa_deg": smp["twa"], "tws_kt": smp["tws"],
               "vpp_success": st["success"], "speed_kt": st["speed"],
               "at_speed_cap": st["speed"] >= SPEED_CAP_KT - 0.01,
               "cant_eff_deg": st["Cant_Eff"], "rake_eff_deg": st["Rake_Eff"],
               "yaw_eff_deg": st["Yaw_Eff"],
               "re_chord_vpp": st["speed"] * KT * 0.70 / NU_SEAWATER}
        res = next(it) if st["success"] else {"ok": False, "error": "no VPP equilibrium"}
        row["strips_ok"] = res["ok"]
        row["strips_error"] = res.get("error", "")
        if res["ok"]:
            sp = res["strips"]
            cl, chord = np.array(sp["cl"]), np.array(sp["chord"])
            width = np.array(sp["width"])
            alpha = np.array(sp["alpha_eff_deg"])
            lift = cl * chord * width
            row["alpha_eq_deg"] = float(np.sum(lift * alpha) / np.sum(lift)) \
                if np.sum(lift) > 0 else float("nan")
            row["alpha_mean_area_deg"] = float(np.sum(chord * width * alpha)
                                               / np.sum(chord * width))
            s_rel = np.array(sp["s"]) / np.sum(width)
            for name, x in (("root", 0.1), ("mid", 0.5), ("tip", 0.9)):
                row[f"alpha_eff_{name}_deg"] = float(np.interp(x, s_rel, alpha))
            row["cl_foil_mean"] = float(np.sum(lift) / np.sum(chord * width))
            for j in range(len(cl)):
                strip_rows.append({"sample": i, "strip": j, "s_over_span": s_rel[j],
                                   "s_m": sp["s"][j], "y_m": sp["y"][j], "z_m": sp["z"][j],
                                   "chord_m": chord[j], "width_m": width[j],
                                   "incidence_geo_deg": sp["incidence_deg"][j],
                                   "alpha_eff_deg": alpha[j], "cl": cl[j], "cd": sp["cd"][j],
                                   "re_vpp": st["speed"] * KT * chord[j] / NU_SEAWATER})
        sample_rows.append(row)
    for name, rows in (("profile_samples.csv", sample_rows), ("span_states.csv", strip_rows)):
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with open(out / name, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    node_rows = [{"twa": c[0], "tws": c[1], "weight": nd["weight"], "success": s["success"],
                  "speed_kt": s["speed"], "cant": s["Cant_Eff"], "rake": s["Rake_Eff"],
                  "yaw": s["Yaw_Eff"]} for c, nd, s in zip(node_cond, nodes, node_states)]
    (out / "profile_nodes_check.json").write_text(json.dumps(node_rows, indent=1), encoding="utf-8")
    say(f"done: {out}")
    log.close()


# -------------------------------------------------------------------------------------------------
# Figures and quantile summary
# -------------------------------------------------------------------------------------------------
QUANTILES = [5, 25, 50, 75, 95]


def figures(args) -> None:
    """span_summary.csv + figures/hugo_inputs.html (plotly, one page)."""
    import pandas as pd  # noqa: PLC0415
    import plotly.graph_objects as go  # noqa: PLC0415
    from plotly.subplots import make_subplots  # noqa: PLC0415
    folder = Path(args.folder) if args.folder else OUT
    samples = pd.read_csv(folder / "profile_samples.csv")
    span = pd.read_csv(folder / "span_states.csv")
    ok = samples[samples["strips_ok"] == True]                               # noqa: E712
    # quantile summary
    rows = []
    for name, col in (("speed_kt", "speed_kt"), ("re_chord_vpp", "re_chord_vpp"),
                      ("alpha_eq_deg", "alpha_eq_deg"), ("alpha_eff_root_deg (s/S 0.1)",
                                                         "alpha_eff_root_deg"),
                      ("alpha_eff_mid_deg (s/S 0.5)", "alpha_eff_mid_deg"),
                      ("alpha_eff_tip_deg (s/S 0.9)", "alpha_eff_tip_deg")):
        v = ok[col].to_numpy()
        rows.append({"quantity": name, "mean": v.mean(), "std": v.std(),
                     **{f"p{q:02d}": np.percentile(v, q) for q in QUANTILES}})
    v = span["alpha_eff_deg"].to_numpy()
    rows.append({"quantity": "alpha_eff_deg all strips", "mean": v.mean(), "std": v.std(),
                 **{f"p{q:02d}": np.percentile(v, q) for q in QUANTILES}})
    summary = pd.DataFrame(rows)
    summary.to_csv(folder / "span_summary.csv", index=False)
    # spanwise quantile bands
    # by strip index (every draw has the same strips; rounding s split the wide strips in two
    # biased groups on 2026-11-13), x = mean s/S of the strip
    position = span.groupby("strip")["s_over_span"].mean()
    bands = span.groupby("strip")["alpha_eff_deg"].quantile([q / 100 for q in QUANTILES]).unstack()
    geo = span.groupby("strip")["incidence_geo_deg"].quantile([0.05, 0.5, 0.95]).unstack()
    bands.index, geo.index = position.loc[bands.index].to_numpy(), position.loc[geo.index].to_numpy()

    fig = make_subplots(rows=3, cols=2, vertical_spacing=0.09, horizontal_spacing=0.09,
                        subplot_titles=("Tirages TWA / TWS (couleur : vitesse VPP, croix : sans équilibre)",
                                        "Vitesse du bateau à l'équilibre VPP",
                                        "Loi jointe (vitesse VPP, alpha équivalent)",
                                        "Alpha équivalent (moyenne pondérée par la portance)",
                                        "Alpha effectif le long de l'envergure (quantiles)",
                                        "Écart alpha pied / bout vs alpha équivalent"))
    good, bad = samples[samples["vpp_success"] == True], samples[samples["vpp_success"] != True]  # noqa: E712
    fig.add_trace(go.Scatter(x=good["twa_deg"], y=good["tws_kt"], mode="markers",
                             marker=dict(color=good["speed_kt"], colorscale="Viridis", size=6,
                                         colorbar=dict(title="kt", len=0.3, y=0.86)),
                             name="équilibre", hovertemplate="TWA %{x:.1f}°, TWS %{y:.1f} kt"),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=bad["twa_deg"], y=bad["tws_kt"], mode="markers",
                             marker=dict(symbol="x", color="red", size=7), name="sans équilibre"),
                  row=1, col=1)
    fig.add_trace(go.Histogram(x=good["speed_kt"], nbinsx=40, name="vitesse",
                               marker_color="#4C78A8"), row=1, col=2)
    fig.add_trace(go.Scatter(x=ok["speed_kt"], y=ok["alpha_eq_deg"], mode="markers",
                             marker=dict(color=ok["twa_deg"], colorscale="Plasma", size=6,
                                         colorbar=dict(title="TWA", len=0.3, y=0.5)),
                             name="(V, alpha_eq)",
                             hovertemplate="V %{x:.2f} kt, alpha %{y:.2f}°"), row=2, col=1)
    fig.add_trace(go.Histogram(x=ok["alpha_eq_deg"], nbinsx=40, name="alpha_eq",
                               marker_color="#F58518"), row=2, col=2)
    s = bands.index.to_numpy()
    fig.add_trace(go.Scatter(x=s, y=bands[0.95], line=dict(width=0), showlegend=False),
                  row=3, col=1)
    fig.add_trace(go.Scatter(x=s, y=bands[0.05], fill="tonexty", line=dict(width=0),
                             fillcolor="rgba(76,120,168,0.25)", name="alpha_eff 5-95 %"),
                  row=3, col=1)
    fig.add_trace(go.Scatter(x=s, y=bands[0.75], line=dict(width=0), showlegend=False),
                  row=3, col=1)
    fig.add_trace(go.Scatter(x=s, y=bands[0.25], fill="tonexty", line=dict(width=0),
                             fillcolor="rgba(76,120,168,0.5)", name="alpha_eff 25-75 %"),
                  row=3, col=1)
    fig.add_trace(go.Scatter(x=s, y=bands[0.5], line=dict(color="#4C78A8"),
                             name="alpha_eff médiane"), row=3, col=1)
    fig.add_trace(go.Scatter(x=geo.index, y=geo[0.5], line=dict(color="gray", dash="dash"),
                             name="incidence géométrique médiane"), row=3, col=1)
    fig.add_trace(go.Scatter(x=ok["alpha_eq_deg"], y=ok["alpha_eff_root_deg"] - ok["alpha_eq_deg"],
                             mode="markers", marker=dict(size=5, color="#54A24B"),
                             name="pied (s/S 0,1) − alpha_eq"), row=3, col=2)
    fig.add_trace(go.Scatter(x=ok["alpha_eq_deg"], y=ok["alpha_eff_tip_deg"] - ok["alpha_eq_deg"],
                             mode="markers", marker=dict(size=5, color="#E45756"),
                             name="bout (s/S 0,9) − alpha_eq"), row=3, col=2)
    for r, c, xt, yt in ((1, 1, "TWA [°]", "TWS [kt]"), (1, 2, "V [kt]", "tirages"),
                         (2, 1, "V [kt]", "alpha_eq [°]"), (2, 2, "alpha_eq [°]", "tirages"),
                         (3, 1, "s / envergure (0 = pied)", "alpha [°]"),
                         (3, 2, "alpha_eq [°]", "écart [°]")):
        fig.update_xaxes(title_text=xt, row=r, col=c)
        fig.update_yaxes(title_text=yt, row=r, col=c)
    n_fail = int((samples["vpp_success"] != True).sum())                   # noqa: E712
    fig.update_layout(height=1300, template="plotly_white",
                      legend=dict(orientation="h", y=-0.06, x=0),
                      title=f"Hugo — vitesse et alpha du C-foil MC2 à l'équilibre VPP "
                            f"({len(samples)} tirages, {n_fail} sans équilibre, "
                            f"{int(samples['at_speed_cap'].sum())} au plafond 24 kt)")
    (folder / "figures").mkdir(exist_ok=True)
    parts = [fig.to_html(full_html=False, include_plotlyjs="cdn")]

    # PCA figure: MC2 anchor +- 2 per mode (4 modes) and sensitivities if present
    pca_path = folder / "pca_hugo.json"
    if pca_path.exists():
        d = json.loads(pca_path.read_text(encoding="utf-8"))
        x = np.array(d["stations"])
        n_x = len(x)
        sig, v = np.array(d["sigmas"]), np.array(d["modes_coordinates_v"])
        pca4 = SectionPCA.load(PCA_RUN / "data" / "pca.json")
        y_ref = pca4.coordinates(np.array(d["reference_weights_mc2_anchor"]))[0]
        f2 = make_subplots(rows=2, cols=2, subplot_titles=[
            f"Mode {j + 1} ({100 * d['variance_ratio'][j]:.1f} % de variance) : MC2 ± 2 unités"
            for j in range(K_HUGO)], vertical_spacing=0.12)
        for j in range(K_HUGO):
            r, c = j // 2 + 1, j % 2 + 1
            for zj, color, name in ((0, "black", "MC2 (ancre)"), (2, "#E45756", "+2"),
                                    (-2, "#4C78A8", "−2")):
                y = y_ref + zj * sig[j] * v[:, j]
                for side, off in (("up", 0), ("lo", n_x)):
                    f2.add_trace(go.Scatter(x=x, y=y[off:off + n_x], line=dict(color=color),
                                            name=name, legendgroup=name,
                                            showlegend=(j == 0 and side == "up")), row=r, col=c)
            f2.update_yaxes(scaleanchor=f"x{j + 1 if j else ''}", scaleratio=1, row=r, col=c)
        f2.update_layout(height=700, template="plotly_white",
                         title="Les 4 premiers modes de la PCA (z5..z10 figés à MC2)")
        parts.append(f2.to_html(full_html=False, include_plotlyjs=False))
    sens_path = folder / "sensitivities.csv"
    if sens_path.exists():
        sens = pd.read_csv(sens_path)
        f3 = go.Figure()
        for col, name in (("dCD_dz_fixed_alpha_rel_mean_pct", "à alpha fixé"),
                          ("dCD_dz_fixed_CL_rel_mean_pct", "à CL fixé")):
            f3.add_trace(go.Bar(x=[f"mode {m}" for m in sens["mode"]], y=sens[col], name=name))
        f3.update_layout(template="plotly_white", height=420, barmode="group",
                         yaxis_title="dCD/dz / CD [% par unité de score]",
                         title="Sensibilité de la traînée 2D (NeuralFoil, moyenne sur les tirages)")
        parts.append(f3.to_html(full_html=False, include_plotlyjs=False))
    html = ("<html><head><meta charset='utf-8'><title>Données pour Hugo</title></head><body>"
            + "".join(parts) + "</body></html>")
    (folder / "figures" / "hugo_inputs.html").write_text(html, encoding="utf-8")
    print(summary.round(3).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("thickness")
    p = sub.add_parser("pca")
    p.add_argument("--n-box", type=int, default=200_000)
    p.add_argument("--seed", type=int, default=0)
    sub.add_parser("constraints")
    p = sub.add_parser("xfoil_check")
    p.add_argument("--re", type=float, default=3.5e6)
    p.add_argument("--xtr", type=float, default=0.03)
    p.add_argument("--h", type=float, default=0.2)
    p.add_argument("--alphas", type=float, nargs="+", default=[-1.0, 0.5, 1.5])
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--timeout", type=float, default=300.0)
    p = sub.add_parser("figures")
    p.add_argument("--folder", default=None)
    p = sub.add_parser("sensitivities")
    p.add_argument("--m", type=int, default=200, help="conditions drawn from profile_samples.csv")
    p.add_argument("--h", type=float, default=0.1)
    p.add_argument("--samples", default=None, help="profile_samples.csv (default OUT's)")
    p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("profile")
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", default=None, help="output folder (default OUT; smoke tests elsewhere)")
    args = parser.parse_args()
    {"thickness": thickness, "pca": pca_export, "constraints": constraints_export,
     "sensitivities": sensitivities, "profile": profile, "figures": figures,
     "xfoil_check": xfoil_check}[args.command](args)


if __name__ == "__main__":
    main()
