"""
PCA of the C-foil sections on the CONVEX domain of src/section_domain.py (2026-11-04, after the
meeting with O. Le Maitre), in a run folder runs/<stamp>_section_pca_convex/:
  1. n sections uniform in the domain (hit-and-run from the current section, constraints on
     --sample-stations stations), then every one checked again on the 2001 stations of
     section_pca (the rare ones that break a constraint between two stations are dropped, counted);
  2. PCA of the coordinates of these sections; its stability: the same PCA on 4 000 and 20 000 of
     them (sigma_j and the angle between the subspaces of the first k modes);
  3. the box of the scores, as chosen with the user: from the current section, one mode at a time,
     steps of 0.01 sigma in both directions until the decoded section (all the modes) breaks a
     constraint; the box gives the limit and the constraint that sets it. Ixx / v is not used for
     the box: the current section lies ON that limit (it would give a zero width on one side of
     every mode); it cuts the domain inside the box, checked on every decoded section;
  4. convexity check: segments between admissible sections of the box (scores), 21 points each.
Outputs: data/pca.json (SectionPCA, n_modes = all), data/domain.json (SectionDomain settings and the
score box), results/summary.json, README.md.

Run: .venv-mds\\Scripts\\python vpp_acceleration\\scripts\\build_section_pca.py [--n 100000] ...
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import subspace_angles

VPP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VPP_DIR))
# pylint: disable=wrong-import-position
from src.run_dir import RunDir  # noqa: E402
from src.section_domain import SectionDomain  # noqa: E402
from src.section_pca import CFoilFamily, SectionPCA, wiggliness  # noqa: E402

STUDY_CONFIG = VPP_DIR.parent / "studies" / "cfoil_intens_kulfan_trim_2" / "config_cfoil_trim.json"
TE = 0.00197


def score_box(pca: SectionPCA, domain: SectionDomain, start: np.ndarray, step: float = 0.01,
              reach: float = 6.0) -> list:
    """One mode at a time from `start` (scores on all the modes): per mode, the last admissible
    value on each side for the LINEAR constraints, and the constraint that stops it."""
    steps = np.arange(step, reach + step / 2, step)
    names = [n for n in dict.fromkeys(domain.row_names)]
    box = []
    for j in range(len(start)):
        side = {}
        for sign in (-1.0, 1.0):
            scores = np.tile(start, (len(steps), 1))
            scores[:, j] += sign * steps
            margins = domain.margins(pca.decode(scores))
            broken = np.stack([np.any(margins[:, domain.row_names == n] < -1e-10, axis=1)
                               for n in names], axis=1)                    # (steps, constraints)
            first = np.argmax(broken.any(axis=1)) if broken.any() else None
            if first is None:
                side[sign] = (float(start[j] + sign * reach), "none")
            else:
                last = start[j] + sign * (steps[first - 1] if first > 0 else 0.0)
                side[sign] = (float(last), "+".join(n for n, b in zip(names, broken[first]) if b))
        box.append({"mode": j + 1, "lower": side[-1.0][0], "upper": side[1.0][0],
                    "lower_limit": side[-1.0][1], "upper_limit": side[1.0][1]})
    return box


def main() -> None:
    """Builds the run folder (module docstring)."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--n", type=int, default=100_000, help="number of sections")
    parser.add_argument("--chains", type=int, default=500)
    parser.add_argument("--burn", type=int, default=1000)
    parser.add_argument("--thin", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--box-scale", type=float, default=3.0)
    parser.add_argument("--band", type=float, default=0.05)
    parser.add_argument("--le-factor", type=float, default=0.5)
    parser.add_argument("--ixx-v-factor", type=float, default=1.0,
                        help="Ixx / v >= this factor x Ixx / v of the current section")
    parser.add_argument("--ixx-factor", type=float, default=0.0,
                        help="Ixx >= this factor x Ixx of the current section (0 = no Ixx limit)")
    parser.add_argument("--sample-stations", type=int, default=501)
    parser.add_argument("--n-segments", type=int, default=2000)
    parser.add_argument("--name", default="section_pca_convex")
    args = parser.parse_args()
    run = RunDir.create(args.name, vars(args))

    # 1. the sections
    current = np.asarray(CFoilFamily.from_study_config(STUDY_CONFIG).reference_weights(), float)
    settings = {"box_scale": args.box_scale, "band": args.band, "le_factor": args.le_factor,
                "ixx_v_factor": args.ixx_v_factor, "ixx_factor": args.ixx_factor}
    domain = SectionDomain(current, TE, **settings)
    coarse = SectionDomain(current, TE, **settings, inflections=domain.inflections,
                           stations=np.linspace(0.0, 1.0, args.sample_stations))
    run.log(f"domain: {domain.g.shape[0]} linear constraints, lower inflections "
            f"{domain.inflections[0]:.3f} / {domain.inflections[1]:.3f} +- {args.band}, "
            f"kappa {domain.kappa[0]:.3f} / {domain.kappa[1]:.4f}, Ixx/v >= {domain.ixx_v_min:.4e}, "
            f"Ixx >= {domain.ixx_min:.4e}")
    start = time.perf_counter()
    sample = coarse.sample(args.n, seed=args.seed, n_chains=args.chains, burn=args.burn,
                           thin=args.thin, log=run.log)
    inside = domain.contains(sample)
    broken = {k: int(v.sum()) for k, v in domain.violations(sample[~inside]).items() if v.any()}
    sample = sample[inside]
    run.log(f"hit-and-run: {len(sample)} sections in {time.perf_counter() - start:.0f} s, "
            f"{int((~inside).sum())} dropped on the fine stations {broken}")
    wig = wiggliness(sample) / wiggliness(current[None])[0]

    # 2. PCA and its stability
    order = np.random.default_rng(args.seed + 1).permutation(len(sample))
    pca = SectionPCA(TE).fit(sample)
    stability = []
    for n_sub in (4000, 20000):
        sub = SectionPCA(TE).fit(sample[order[:n_sub]])
        row = {"n": n_sub, "sigma_rel_change_max": float(np.max(np.abs(sub.sigmas / pca.sigmas - 1)))}
        for k in (6, 10, 12):
            row[f"angle_k{k}_deg"] = float(np.degrees(subspace_angles(sub.modes[:, :k],
                                                                      pca.modes[:, :k]).max()))
        stability.append(row)
    run.log(f"PCA: rank {pca.rank}, variance per mode "
            f"{np.round(100 * pca.variance_ratio, 2).tolist()} %; stability {stability}")
    projection = {}
    for k in (6, 8, 10, 12, 14, 16):
        projected = pca.reconstruct(current[None], k)
        fail = [n for n, v in domain.violations(projected).items() if v[0]]
        projection[k] = {"variance_pct": float(100 * pca.variance_ratio[:k].sum()),
                         "gap_max": float(pca.geometric_error(current[None], k)[0]),
                         "current_projected_breaks": fail}
    pca.save(run.path("data", "pca.json"))

    # 3. the box of the scores from the current section
    z_current = pca.encode(current[None], pca.rank)[0]
    box = score_box(pca, domain, z_current)
    lower, upper = np.array([b["lower"] for b in box]), np.array([b["upper"] for b in box])
    scores = pca.encode(sample, pca.rank)
    in_box = np.all((scores >= lower) & (scores <= upper), axis=1)
    run.log(f"score box: {in_box.mean():.1%} of the domain sample inside")

    # 4. convexity: segments between admissible sections of the box (uniform in the box, kept
    # when the decoded section is in the domain)
    rng = np.random.default_rng(args.seed + 2)
    uniform = lower + (upper - lower) * rng.random((200_000, len(lower)))
    admissible = uniform[domain.contains(pca.decode(uniform))]
    hull = {"lower": scores.min(axis=0).tolist(), "upper": scores.max(axis=0).tolist()}
    run.log(f"uniform in the score box: {len(admissible) / len(uniform):.2%} admissible")
    if len(admissible) < 2:
        raise RuntimeError("no admissible section in the score box")
    pairs = rng.integers(0, len(admissible), (args.n_segments, 2))
    t = np.linspace(0.0, 1.0, 21)[1:-1]
    segment = (admissible[pairs[:, 0], None] * (1 - t[None, :, None])
               + admissible[pairs[:, 1], None] * t[None, :, None]).reshape(-1, pca.rank)
    out = domain.violations(pca.decode(segment))
    per_segment = {k: v.reshape(args.n_segments, -1).any(axis=1) for k, v in out.items()}
    leaving = np.any(np.stack(list(per_segment.values())), axis=0)
    convexity = {"segments": args.n_segments, "leaving_pct": float(100 * leaving.mean()),
                 "by_constraint_pct": {k: float(100 * v.mean()) for k, v in per_segment.items()}}
    run.log(f"convexity: {convexity}")

    run.write_json("data/domain.json", {**domain.to_dict(), "score_box": box,
                                        "current_scores": z_current.tolist()})
    summary = {"n_sections": int(len(sample)), "dropped_fine_check": int((~inside).sum()),
               "dropped_by_constraint": broken, "variance_ratio": pca.variance_ratio.tolist(),
               "stability": stability, "projection_of_current": projection,
               "wiggliness_vs_current": {p: float(np.percentile(wig, p)) for p in (1, 50, 99)},
               "score_box": box, "domain_sample_in_box_pct": float(100 * in_box.mean()),
               "box_uniform_admissible_pct": float(100 * len(admissible) / len(uniform)),
               "sample_score_hull": hull,
               "convexity": convexity}
    run.write_json("results/summary.json", summary)
    lines = ["# PCA sur le domaine convexe des sections (2026-11-04)", "",
             f"{len(sample)} sections tirées uniformément dans le domaine (hit-and-run depuis la "
             "section actuelle), src/section_domain.py. Résumé complet : results/summary.json.", "",
             "| mode | variance % | borne − | borne + | contrainte − | contrainte + |",
             "|---|---|---|---|---|---|"]
    lines += [f"| {b['mode']} | {100 * pca.variance_ratio[b['mode'] - 1]:.2f} | {b['lower']:+.2f} | "
              f"{b['upper']:+.2f} | {b['lower_limit']} | {b['upper_limit']} |" for b in box]
    lines += ["", f"Segments entre sections admissibles de la boîte qui sortent du domaine : "
              f"{convexity['leaving_pct']:.1f} % ({convexity['by_constraint_pct']})."]
    run.path("README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    run.log("done")


if __name__ == "__main__":
    main()
