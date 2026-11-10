"""
Search box of the slider scores (2026-11-05): a box of the k first PCA scores that lies ENTIRELY in
the domain of the sections, so that the optimizer needs no constraint (user: "a convex, continuous
domain, no constraint in the optimizer, k = 10").

  1. the PCA of a build_section_pca.py run, anchored on the current section: the k modes span the
     affine space through the current section (SectionPCA.anchor), z0 = its scores;
  2. the decoded section is affine in z: margins of the linear constraints
        m(z) = a + C (z - z0),  a = margins of the current section (>= 0);
     a box [z0 - d_minus, z0 + d_plus] lies in the linear domain iff, for every row r,
        a_r - sum_j (C_rj^+ d_minus_j + C_rj^- d_plus_j) >= 0      (the worst corner, exact);
  3. two candidate boxes:
     A "scaled": the per-axis bounds (one slider at a time, linear constraints and Ixx / v),
       multiplied by the largest factor alpha that keeps the box in the linear domain;
     B "max volume": the box of largest volume (sum of log widths, whitened scores) in the linear
       domain, bounded by the per-axis bounds;
  4. Ixx / v (not linear) on each box: corners, uniform points, local minimisation; the box is
     shrunk towards z0 until min Ixx / v >= the domain limit;
  5. checks: uniform points of each box all admissible; earlier best sections (projected).
  6. box C "envelope" (chosen with the user on 2026-11-05: larger box, the constraints checked
     when the optimizer screens the sections): per mode, the [q, 100 - q] % percentiles of
     sections uniform on the admissible slice (hit-and-run in the k modes through the current
     section); it is NOT entirely admissible (share reported).
Outputs in the PCA run folder: data/search_box_k<k>.json, results/search_box_k<k>.json.

Run: .venv-mds\\Scripts\\python vpp_acceleration\\scripts\\build_search_box.py --pca-run <run> --k 10
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

VPP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VPP_DIR))
# pylint: disable=wrong-import-position
from src.section_domain import SectionDomain  # noqa: E402
from src.section_pca import SectionPCA, section_modulus, section_stiffness  # noqa: E402

EARLIER = {"run4": "2026-10-23_1702_real_search_profile_local",
           "run5": "2026-10-27_1224_control_restarts",
           "run6": "2026-10-27_1739_long_3levels_avl_iter",
           "run7": "2026-10-28_1747_profile_v1_box075",
           "run8": "2026-10-29_1002_profile_v2_box3_3h"}


def load(pca_run: Path, k: int):
    """Domain, PCA anchored on the current section (k modes), z0."""
    d = json.loads((pca_run / "data" / "domain.json").read_text(encoding="utf-8"))
    domain = SectionDomain(np.array(d["reference"]), d["te_thickness"], d["box_scale"], d["band"],
                           d["le_factor"], inflections=d["inflections"],
                           ixx_v_factor=d.get("ixx_v_factor", 1.0),
                           ixx_factor=d.get("ixx_factor", 0.0))
    pca = SectionPCA.load(pca_run / "data" / "pca.json").anchor(domain.reference, k)
    z0 = pca.encode(domain.reference[None])[0]
    return domain, pca, z0


def axis_bounds(domain, pca, z0, a, c, step=0.005, reach=6.0):
    """Per slider, the distance to the domain on each side (linear exact, Ixx / v by steps)."""
    k = len(z0)
    d_minus, d_plus, limits = np.zeros(k), np.zeros(k), []
    for j in range(k):
        out = {}
        for sign, store in ((-1.0, d_minus), (1.0, d_plus)):
            slope = sign * c[:, j]                       # margin along the axis: a + slope s
            with np.errstate(divide="ignore"):
                s_lin = np.min(np.where(slope < 0, a / -slope, np.inf))
            s_lin = min(s_lin, reach)
            steps = np.arange(step, s_lin + step / 2, step)
            z = np.tile(z0, (len(steps), 1))
            z[:, j] += sign * steps
            ok = domain.nonlinear_ok(pca.decode(z))
            if ok.all():
                s, why = s_lin, domain.row_names[np.argmin(a + slope * s_lin)] if s_lin < reach \
                    else "reach"
            else:
                first = int(np.argmax(~ok))
                s, why = (steps[first - 1] if first > 0 else 0.0), "ixx_v / ixx"
            store[j] = s
            out[sign] = why
        limits.append((out[-1.0], out[1.0]))
    return d_minus, d_plus, limits


def worst_linear(a, c, d_minus, d_plus):
    """Smallest margin of the linear constraints over the box (exact, item 2 of the docstring)."""
    return np.min(a - np.clip(c, 0, None) @ d_minus - np.clip(-c, 0, None) @ d_plus)


def max_volume_box(a, c, cap_minus, cap_plus, start_minus, start_plus):
    """Box B: max sum_j log(d_minus_j + d_plus_j) s.t. the box is in the linear domain."""
    k = len(cap_minus)
    # rows that can bind somewhere in the per-axis box (the others never do)
    reach = np.clip(c, 0, None) @ cap_minus + np.clip(-c, 0, None) @ cap_plus
    rows = reach > a
    a_r, cp, cm = a[rows], np.clip(c[rows], 0, None), np.clip(-c[rows], 0, None)
    matrix = np.hstack([cp, cm])                                   # x = [d_minus, d_plus]

    def objective(x):
        width = x[:k] + x[k:]
        return -np.sum(np.log(width)), -np.concatenate([1 / width, 1 / width])

    result = minimize(objective, np.concatenate([start_minus, start_plus]), jac=True,
                      method="SLSQP", bounds=[(0, float(u)) for u in np.concatenate([cap_minus,
                                                                                    cap_plus])],
                      constraints=[{"type": "ineq", "fun": lambda x: a_r - matrix @ x,
                                    "jac": lambda x: -matrix}],
                      options={"maxiter": 500, "ftol": 1e-10})
    x = np.clip(result.x, 0, None)
    x *= min(1.0, 1.0 / max(1e-12, np.max(matrix @ x / a_r)))       # exactly feasible
    return x[:k], x[k:], result


def stiffness_ratio(domain, weights):
    """min(Ixx / v / its limit, Ixx / its limit) per section (the Ixx part only if ixx_factor > 0)."""
    ixx, ixx_v = section_stiffness(np.atleast_2d(weights), domain.te_thickness)
    ratio = ixx_v / domain.ixx_v_min
    return np.minimum(ratio, ixx / domain.ixx_min) if domain.ixx_min > 0 else ratio


def min_ixx_v(domain, pca, z0, d_minus, d_plus, rng, n_uniform=50_000, n_local=10):
    """Smallest Ixx / v / limit over the box: corners, uniform points, local minimisation."""
    lower, upper = z0 - d_minus, z0 + d_plus
    corners = np.array(list(itertools.product(*zip(lower, upper))))
    uniform = lower + (upper - lower) * rng.random((n_uniform, len(z0)))
    points = np.vstack([corners, uniform])
    ratio = stiffness_ratio(domain, pca.decode(points))
    best = float(ratio.min())
    for z in points[np.argsort(ratio)[:n_local]]:
        res = minimize(lambda q: stiffness_ratio(domain, pca.decode(q))[0], z, method="L-BFGS-B",
                       bounds=list(zip(lower, upper)))
        best = min(best, float(res.fun))
    return best


def shrink_for_ixx(domain, pca, z0, d_minus, d_plus, rng):
    """Largest factor in (0, 1] (bisection) such that min Ixx / v >= the limit on the box."""
    if min_ixx_v(domain, pca, z0, d_minus, d_plus, rng) >= 1.0:
        return 1.0
    low, high = 0.0, 1.0
    for _ in range(12):
        mid = (low + high) / 2
        if min_ixx_v(domain, pca, z0, mid * d_minus, mid * d_plus, rng) >= 1.0:
            low = mid
        else:
            high = mid
    return low


def main() -> None:
    """Writes the search boxes (module docstring)."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--pca-run", required=True, help="run folder of build_section_pca.py")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--envelope-quantile", type=float, default=0.5,
                        help="box C: percentiles [q, 100 - q] per mode of the admissible slice")
    parser.add_argument("--n-envelope", type=int, default=80_000)
    args = parser.parse_args()
    pca_run = Path(args.pca_run).resolve()
    rng = np.random.default_rng(args.seed)
    domain, pca, z0 = load(pca_run, args.k)
    current = domain.reference
    weight_modes = pca.weight_modes[:, :args.k] * pca.sigmas[:args.k]
    a = domain.margins(current[None])[0]
    c = domain.g @ weight_modes                                   # (rows, k)
    print(f"k = {args.k}: current decoded exactly ({np.abs(pca.decode(z0)[0] - current).max():.1e}),"
          f" min linear margin {a.min():.2e}, Ixx/v current / limit "
          f"{section_modulus(current[None], domain.te_thickness)[0] / domain.ixx_v_min:.3f}")

    # per-axis bounds
    ax_minus, ax_plus, limits = axis_bounds(domain, pca, z0, a, c)
    # box A: scaled per-axis box
    alpha = min(1.0, float(np.min(a / np.maximum(np.clip(c, 0, None) @ ax_minus
                                                + np.clip(-c, 0, None) @ ax_plus, 1e-300))))
    boxes = {"A_scaled": (alpha * ax_minus, alpha * ax_plus)}
    # box B: max volume
    b_minus, b_plus, res = max_volume_box(a, c, ax_minus, ax_plus, *boxes["A_scaled"])
    print(f"max volume: {res.message} ({res.nit} iterations)")
    boxes["B_max_volume"] = (b_minus, b_plus)
    # box C: envelope of the admissible slice
    sample = domain.sample(args.n_envelope, seed=args.seed + 1, n_chains=400, burn=1000, thin=10,
                           basis=weight_modes)
    z_sample = pca.encode(sample)
    q = args.envelope_quantile
    c_lower = np.minimum(np.percentile(z_sample, q, axis=0), z0)
    c_upper = np.maximum(np.percentile(z_sample, 100 - q, axis=0), z0)
    boxes["C_envelope"] = (z0 - c_lower, c_upper - z0)
    print(f"envelope: {len(sample)} admissible sections of the slice, all admissible "
          f"{bool(domain.contains(sample).all())}")

    summary = {"pca_run": pca_run.name, "k": args.k, "z0": z0.tolist(),
               "ixx_v_limit": domain.ixx_v_min, "ixx_v_factor": domain.ixx_v_factor,
               "ixx_limit": domain.ixx_min, "ixx_factor": domain.ixx_factor,
               "axis": {"d_minus": ax_minus.tolist(), "d_plus": ax_plus.tolist(),
                        "limits": limits}, "alpha_scaled": alpha, "boxes": {}}
    axis_width = ax_minus + ax_plus
    for name, (dm, dp) in boxes.items():
        envelope = name == "C_envelope"            # constraints at screening: no shrink
        factor = 1.0 if envelope else shrink_for_ixx(domain, pca, z0, dm, dp, rng)
        dm, dp = factor * dm, factor * dp
        lower, upper = z0 - dm, z0 + dp
        uniform = lower + (upper - lower) * rng.random((100_000, args.k))
        inside = domain.contains(pca.decode(uniform))
        width = dm + dp
        earlier = {}
        for label, run in EARLIER.items():
            best = json.loads((VPP_DIR / "runs" / run / "results" / "best_section.json")
                              .read_text(encoding="utf-8"))
            z = pca.encode(np.array(best["kulfan_upper"] + best["kulfan_lower"])[None])[0]
            earlier[label] = {"in_box": bool(np.all((z >= lower) & (z <= upper))),
                              "modes_outside": [j + 1 for j in range(args.k)
                                                if not lower[j] <= z[j] <= upper[j]]}
        summary["boxes"][name] = {
            "lower": lower.tolist(), "upper": upper.tolist(), "ixx_shrink": factor,
            "worst_linear_margin": float(worst_linear(a, c, dm, dp)),
            "min_ixx_v_over_limit": None if envelope else min_ixx_v(domain, pca, z0, dm, dp, rng),
            "uniform_admissible_pct": float(100 * inside.mean()),
            "width_over_axis_width": (width / axis_width).tolist(),
            "log10_volume_vs_axis_box": float(np.sum(np.log10(width / axis_width))),
            "earlier_bests": earlier}
        s = summary["boxes"][name]
        print(f"\n{name}: Ixx shrink {factor:.3f}, worst linear margin {s['worst_linear_margin']:.1e},"
              f" min Ixx/v / limit {s['min_ixx_v_over_limit']}, uniform admissible "
              f"{s['uniform_admissible_pct']:.2f} %, volume 10^{s['log10_volume_vs_axis_box']:.1f}"
              f" x per-axis box")
        print(" mode |  z0    | axis -  axis + | box lower  upper | width / axis width")
        for j in range(args.k):
            print(f" {j + 1:4d} | {z0[j]:+.2f} | {-ax_minus[j]:+.2f}  {ax_plus[j]:+.2f} | "
                  f"{lower[j]:+.3f}  {upper[j]:+.3f} | {width[j] / axis_width[j]:.2f}")
        print(" earlier bests:", {k_: v["modes_outside"] or "in" for k_, v in earlier.items()})

    out = json.dumps(summary, indent=1)
    (pca_run / "data" / f"search_box_k{args.k}.json").write_text(out, encoding="utf-8")
    (pca_run / "results" / f"search_box_k{args.k}.json").write_text(out, encoding="utf-8")


if __name__ == "__main__":
    main()
