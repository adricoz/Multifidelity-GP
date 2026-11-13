"""
Is the surrogate of a finished run honest FAR from its data, and does the conclusion "no better section
elsewhere" survive it? (plan of 2026-11-12; complements experiments.py, same FinishedRun, no code and no
VPP call.) The design of a run is targeted: its points pile up around a few sections, so the plain
leave-one-out (experiments.py calibration) measures interpolation inside the piles.

Experiments (each writes <out>/<name>.json):
  buffered_cv  cross-validation with an exclusion zone: to predict an observation, ALL the observations
               whose section lies within r scores of it are left out (block formula of
               CoKriging.leave_one_out). r = 0 is the plain leave-one-out; r > 0 is the situation of the
               screening: a section never computed. Gives the standardised error against r, hence the
               factor kappa_far by which the announced sigma must be multiplied far from the data;
  certificate  maximum over the search space of the upper confidence bound
               gain(z) + kappa sigma(z), gain = J_lin(z) - J_lin(best) and sigma the posterior sigma
               of that difference (frozen states, first order): if it is below epsilon, no section
               beats the best one by more than epsilon, as far as the model knows (--level);
  summary      J_lin of every section of the design against its position along the gradient at the
               best section (one hump = one basin along the direction that matters);
  profile      gain of the best section over the start section when the weights of the modes of the
               operating profile change (the 49 solved speeds of results/validation.csv are enough);
  cross        two runs of the same search space (--other): each model judges the best section of the
               other at the AVL x XFOIL level (first order; the exact check with the VPP is the
               digital twin, twin_experiments/compare.py).

    ..\\.venv-mds\\Scripts\\python validation_experiments\\far_experiments.py --run runs/<run> --only buffered_cv
"""
import argparse
import json
import time

import numpy as np
from scipy.linalg import cho_solve
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

from experiments import (HERE, N_FORCES, OUTPUT_NAMES, RESULTS, VPP_DIR, FinishedRun, save,
                         scores_stats)


# -------------------------------------------------------------------------------------------------
# 1. Cross-validation with an exclusion zone
# -------------------------------------------------------------------------------------------------
def buffered_cv(run: FinishedRun, args, radii=(0.0, 0.25, 0.5, 1.0, 2.0, 3.0), n_test: int = 300,
                keep_at_least: int = 40) -> None:
    if args.quick:
        radii, n_test = (0.0, 0.5, 2.0), 20
    reference = run.models[0]
    x_all, levels_all = reference.x_all, reference.levels_all
    sections = x_all[:, :run.k]
    rng = np.random.default_rng(0)
    tests = {}
    for level in range(1, min(2, run.n_levels) + 1):
        rows = np.flatnonzero(levels_all == level)
        tests[level] = np.sort(rng.choice(rows, min(n_test, len(rows)), replace=False))
    every = np.concatenate(list(tests.values()))
    distance = cdist(sections[every], sections)                 # section distance, in scores
    same_point = cdist(x_all[every], x_all) <= reference.tol
    blocks, nearest = {}, {}
    for t, i in enumerate(every):
        for r in radii:
            block = np.flatnonzero(same_point[t] if r == 0.0 else (distance[t] <= r) | same_point[t])
            if len(x_all) - len(block) >= keep_at_least:
                blocks[(i, r)] = block
                rest = np.ones(len(x_all), dtype=bool)
                rest[block] = False
                nearest[(i, r)] = float(distance[t][rest].min())
    out = {"radii": list(radii), "sizes": run.surrogate.sizes, "entries": [], "kappa_far": {}}
    z_by = {}
    for k, name in enumerate(OUTPUT_NAMES):
        model = run.models[k]
        c_inv = cho_solve((model.l_chol, True), np.eye(len(model.z_all)))
        for level, rows in tests.items():
            spread = float(np.std(model.z_all[levels_all == level]))
            for r in radii:
                errors, scores, sizes, gaps = [], [], [], []
                for i in rows:
                    block = blocks.get((i, r))
                    if block is None:
                        continue
                    inverse = np.linalg.inv(c_inv[np.ix_(block, block)])
                    j = int(np.flatnonzero(block == i)[0])
                    error = float((inverse @ model.alpha[block])[j])   # observed - predicted
                    errors.append(error)
                    scores.append(error / np.sqrt(inverse[j, j]))
                    sizes.append(len(block))
                    gaps.append(nearest[(i, r)])
                if not errors:
                    continue
                errors, scores = np.array(errors), np.array(scores)
                z_by[(name, level, r)] = scores
                out["entries"].append({
                    "output": name, "level": level, "radius": r, **scores_stats(scores),
                    "relative_rmse": float(np.sqrt(np.mean(errors ** 2)) / spread),
                    "median_left_out": float(np.median(sizes)),
                    "median_nearest_kept": float(np.median(gaps))})
        print(f"  {name} done", flush=True)
    for level in tests:
        out["kappa_far"][str(level)] = {}
        for r in radii:
            parts = [z_by[(n, level, r)] for n in OUTPUT_NAMES[:N_FORCES] if (n, level, r) in z_by]
            if parts:
                value = float(np.sqrt(np.mean(np.concatenate(parts) ** 2)))
                out["kappa_far"][str(level)][str(r)] = value
                print(f"  level {level}, radius {r}: standardised error (6 forces) {value:.2f}",
                      flush=True)
    save("buffered_cv", out)


# -------------------------------------------------------------------------------------------------
# 2. Certificate: maximum of the upper confidence bound
# -------------------------------------------------------------------------------------------------
def gain_and_sigma(run: FinishedRun, sections: np.ndarray, level, chunk: int = 30) -> tuple:
    """Gain over the best section and posterior sigma of that gain, for sections (n, k)."""
    sections = np.atleast_2d(sections)
    gain, sigma = np.zeros(len(sections)), np.zeros(len(sections))
    for start in range(0, len(sections), chunk):
        both = np.vstack([run.best, sections[start:start + chunk]])
        mean = run.objective(both, level)
        cov = run.objective_covariance(both, level)
        gain[start:start + chunk] = mean[1:] - mean[0]
        sigma[start:start + chunk] = np.sqrt(np.maximum(
            cov[0, 0] + np.diag(cov)[1:] - 2.0 * cov[0, 1:], 0.0))
    return gain, sigma


def certificate(run: FinishedRun, args, kappas=(1.0, 2.0, 3.0), n_pool: int = 240,
                n_starts: int = 5, step: float = 2e-2) -> None:
    level = args.level
    label = level if level is not None else run.n_levels
    pool = np.vstack([run.global_sections(n_pool, 5), run.local_sections(run.best, 1.0, 50, 6),
                      run.local_sections(run.best, 2.5, 50, 7)]
                     + [scores[None] for name, scores in run.sections.items() if name != "best"])
    if args.quick:
        pool, n_starts, kappas = pool[:20], 1, (2.0,)
    gain, sigma = gain_and_sigma(run, pool, level)
    print(f"  pool of {len(pool)} sections: best gain {gain.max():+.4f} kt", flush=True)
    eye = step * np.eye(run.k)
    found = []
    for kappa in kappas:
        def bound(z):
            g, s = gain_and_sigma(run, z, level)
            return float(g[0] + kappa * s[0])

        def slope(z):
            g, s = gain_and_sigma(run, np.vstack([z + eye, z - eye]), level)
            u = g + kappa * s
            return (u[:run.k] - u[run.k:]) / (2.0 * step)

        order = np.argsort(-(gain + kappa * sigma))
        starts = []
        for i in order:                                       # the highest bounds, apart
            if all(np.linalg.norm(pool[i] - s) > 0.5 for s in starts):
                starts.append(pool[i])
            if len(starts) == n_starts:
                break
        for start in starts:
            result = minimize(lambda z: -bound(z), start, method="SLSQP", jac=lambda z: -slope(z),
                              constraints=[{"type": "ineq", "fun": run.constraints,
                                            "jac": run.constraint_jacobian}],
                              options={"maxiter": 20 if not args.quick else 3, "ftol": 1e-6})
            z = np.array(result.x)
            for _ in range(8):                                # back inside if SLSQP left the domain
                if run.admissible(z)[0]:
                    break
                z = start + 0.7 * (z - start)
            else:
                z = np.array(start)
            g, s = gain_and_sigma(run, z, level)
            found.append({"kappa": kappa, "scores": z.tolist(), "gain_kt": float(g[0]),
                          "sigma_kt": float(s[0]), "bound_kt": float(g[0] + kappa * s[0]),
                          "distance": float(np.linalg.norm(z - run.best)),
                          "min_margin": float(run.constraints(z).min())})
            print(f"  kappa {kappa}: bound {found[-1]['bound_kt']:+.4f} kt (gain "
                  f"{g[0]:+.4f}, sigma {s[0]:.4f}) at distance {found[-1]['distance']:.2f}",
                  flush=True)
    all_scores = np.vstack([pool] + [np.array(f["scores"])[None] for f in found])
    all_gain = np.concatenate([gain, [f["gain_kt"] for f in found]])
    all_sigma = np.concatenate([sigma, [f["sigma_kt"] for f in found]])
    out = {"level": label, "kappas": list(kappas), "maximisers": found,
           "pool": {"gain": gain.tolist(), "sigma": sigma.tolist(),
                    "distance": np.linalg.norm(pool - run.best, axis=1).tolist()},
           # the bound for any kappa, over everything evaluated (a lower bound of its maximum)
           "bound_by_kappa": {str(kappa): float((all_gain + kappa * all_sigma).max())
                              for kappa in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0)},
           "all": {"gain": all_gain.tolist(), "sigma": all_sigma.tolist()}}
    save(f"certificate_level{label}", out)
    picked, names = [], {}
    for entry in sorted(found, key=lambda f: -f["bound_kt"]):
        z = np.array(entry["scores"])
        if all(np.linalg.norm(z - p) > 0.3 for p in picked) and len(picked) < 5:
            picked.append(z)
            names[f"certificate_k{entry['kappa']:g}_{len(picked)}"] = entry["scores"]
    (RESULTS["dir"] / f"certificate_sections_level{label}.json").write_text(
        json.dumps(names, indent=1), encoding="utf-8")


# -------------------------------------------------------------------------------------------------
# 3. The design along the direction that matters
# -------------------------------------------------------------------------------------------------
def summary(run: FinishedRun, args) -> None:
    reference = run.models[0]
    sections, index = np.unique(np.round(reference.x_all[:, :run.k], 6), axis=0, return_inverse=True)
    index = np.ravel(index)
    if args.quick:
        sections = sections[:40]
    top_level = np.array([reference.levels_all[index == i].max() for i in range(len(sections))])
    direction = run.gradient(run.best)
    direction /= np.linalg.norm(direction)
    both = np.vstack([run.best, sections])
    values = run.objective(both)
    out = {"n_sections": int(len(sections)), "direction": direction.tolist(),
           "projection": ((sections - run.best) @ direction).tolist(),
           "distance": np.linalg.norm(sections - run.best, axis=1).tolist(),
           "gain": (values[1:] - values[0]).tolist(), "highest_level": top_level.tolist(),
           "admissible": run.admissible(sections).tolist(),
           "named": {name: {"projection": float((scores - run.best) @ direction),
                            "gain": float(run.objective(scores)[0] - values[0])}
                     for name, scores in run.sections.items()}}
    eigen = RESULTS["dir"] / "dimension.json"
    if eigen.exists():
        first = np.array(json.loads(eigen.read_text(encoding="utf-8"))["global"]["eigenvectors"][0])
        out["active_direction"] = first.tolist()
        out["active_projection"] = ((sections - run.best) @ first).tolist()
    save("summary", out)


# -------------------------------------------------------------------------------------------------
# 4. Sensitivity to the operating profile
# -------------------------------------------------------------------------------------------------
def profile(run: FinishedRun, args) -> None:  # pylint: disable=unused-argument
    nodes = json.loads((VPP_DIR / run.config["search_profile"]).read_text(encoding="utf-8"))["nodes"]
    modes = np.array([node.get("mode", "") for node in nodes])
    weights = np.array(run.config["condition_weights"], dtype=float)
    table = run.validation
    start = next(name for name in run.sections if not name.startswith(("best", "region")))
    speeds = {}
    for name in run.sections:
        rows = table[table["section"] == name].reset_index(drop=True)
        if len(rows) == len(nodes):
            ok = rows["success"].to_numpy(dtype=bool)
            speeds[name] = np.where(ok, rows["speed_kt"].to_numpy(dtype=float), np.nan)
    gain = speeds["best"] - speeds[start]
    ok = np.isfinite(gain)

    def mean_gain(w):
        return float(gain[ok] @ w[ok] / w[ok].sum())

    factors = np.linspace(0.0, 3.0, 31)
    out = {"start": start, "gain_kt": mean_gain(weights), "n_solved": int(ok.sum()),
           "twa": [node["twa"] for node in nodes], "tws": [node["tws"] for node in nodes],
           "weights": weights.tolist(), "gain_by_condition": [float(v) if o else None
                                                             for v, o in zip(gain, ok)],
           "factors": factors.tolist(), "modes": {}, "others": {}}
    for mode in dict.fromkeys(modes):
        rows = modes == mode
        out["modes"][mode] = {
            "weight": float(weights[rows].sum()), "gain_kt": mean_gain(np.where(rows, weights, 0.0))
            if (rows & ok).any() else None,
            "gain_when_weight_scaled": [mean_gain(np.where(rows, f * weights, weights))
                                        for f in factors]}
    for name, values in speeds.items():                      # the local optima against the best
        if name not in ("best", start):
            both = np.isfinite(values) & np.isfinite(speeds["best"])
            out["others"][name] = {
                mode: float((values - speeds["best"])[both & (modes == mode)]
                            @ weights[both & (modes == mode)] / weights[both & (modes == mode)].sum())
                for mode in dict.fromkeys(modes) if (both & (modes == mode)).any()}
            out["others"][name]["all"] = float((values - speeds["best"])[both] @ weights[both]
                                               / weights[both].sum())
    save("profile", out)


# -------------------------------------------------------------------------------------------------
# 5. Cross judgement of two runs
# -------------------------------------------------------------------------------------------------
def cross(run: FinishedRun, args) -> None:
    other = FinishedRun(VPP_DIR / args.other)
    same = (run.config["search_box"], run.config["search_box_name"], run.k) == \
        (other.config["search_box"], other.config["search_box_name"], other.k)
    out = {"runs": [run.root.name, other.root.name], "same_search_space": bool(same), "judgements": [],
           "distance_between_best": float(np.linalg.norm(run.best - other.best))}
    if same:
        for judge, judged in ((run, other), (other, run)):
            names = ["own_best"] + [f"own_{n}" for n in judge.sections if n.startswith("region")] \
                + ["other_best"] + [f"other_{n}" for n in judged.sections if n.startswith("region")]
            sections = np.vstack([judge.best] + [s for n, s in judge.sections.items()
                                                 if n.startswith("region")]
                                 + [judged.best] + [s for n, s in judged.sections.items()
                                                    if n.startswith("region")])
            for level in dict.fromkeys([min(2, judge.n_levels), judge.n_levels]):
                mean = judge.objective(sections, level)
                cov = judge.objective_covariance(sections, level)
                sigma = np.sqrt(np.maximum(cov[0, 0] + np.diag(cov) - 2.0 * cov[0], 0.0))
                out["judgements"].append({
                    "judge": judge.root.name, "level": int(level), "names": names,
                    "gain_kt": (mean - mean[0]).tolist(), "sigma_kt": sigma.tolist(),
                    "distance": np.linalg.norm(sections - judge.best, axis=1).tolist()})
                print(f"  judge {judge.root.name[-12:]} level {level}: best of the other run "
                      f"{mean[names.index('other_best')] - mean[0]:+.4f} kt "
                      f"+- {sigma[names.index('other_best')]:.4f}", flush=True)
    save("cross", out)


EXPERIMENTS = {"buffered_cv": buffered_cv, "certificate": certificate, "summary": summary,
               "profile": profile, "cross": cross}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", maxsplit=1)[0])
    parser.add_argument("--run", required=True, help="run folder, relative to vpp_acceleration")
    parser.add_argument("--only", nargs="+", choices=list(EXPERIMENTS), required=True)
    parser.add_argument("--out", default="results", help="results folder (one per run)")
    parser.add_argument("--level", type=int, default=None, help="certificate: level (default top)")
    parser.add_argument("--other", default=None, help="cross: the other run")
    parser.add_argument("--quick", action="store_true", help="tiny sizes (test of the chain)")
    args = parser.parse_args()
    RESULTS["dir"] = HERE / args.out
    start = time.time()
    run = FinishedRun(VPP_DIR / args.run)
    print(f"run {run.root.name}: design per level {run.surrogate.sizes}, loaded in "
          f"{time.time() - start:.0f} s", flush=True)
    for name in args.only:
        t0 = time.time()
        print(f"[{name}]", flush=True)
        EXPERIMENTS[name](run, args)
        print(f"  {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
