"""
Robustness of the optimum of a finished run, without a new search (doc/PLAN_ROBUSTESSE.md, the
"tests sans run"). Two checks, each in a run folder of its own:

posterior: is the ranking of the sections of the run sure, given the uncertainty of the surrogate?
  (Le Maitre's note, section 6.3: solve the VPP again on functions drawn from the posterior of the
  GP, coherent as functions, instead of the linearized variance.)
  1. the VPP with the posterior mean (robust_solve) gives the nominal states of the compared
     sections (best, other local optima, best of the earlier runs, start section);
  2. anchors = every nominal state plus a small Sobol cloud of attitudes around it (all sections
     and conditions together); for each force / moment co-kriging, the JOINT posterior of its top
     level at the anchors (sequential_design.joint_posterior_covariance): the correlations between
     sections are kept, which is what decides a ranking;
  3. a draw = one joint sample at the anchors, scaled by `scale` around the mean (scale 2: the
     variance underestimated by ~2 far from the data, end-of-run validations of runs 5 to 8); the
     co-kriging is then conditioned on these values: a smooth function equal to the sample at the
     anchors and to the posterior mean far from them (approximate path of the GP, exact at the
     anchors, where the VPP equilibria are);
  4. the VPP solved again with every drawn function (start: the nominal state of the section and
     the fixed guess), weighted mean speed over the conditions solved for all sections;
  5. over the draws: probability that each section is the best, expected regret of keeping the
     run's best E[max_i J_i - J_best] (Le Maitre's 8.3, on the compared sections), gains.

tolerance: is the best section a peak or a plateau for small shape errors (manufacturing)?
  1. random smooth perturbations of the 16 Kulfan weights of the best section, scaled so that the
     largest surface displacement is `amplitude` (fraction of the chord, 0.70 m);
  2. for each perturbed section: the VPP with the surrogate at its PCA projection, then AVL x XFOIL
     (the HF code of the run) at the equilibrium states with the UNPROJECTED weights, and the
     speed corrected to first order with the sensitivities of the best section (as
     vpp_optimize.validate): V = V_surrogate - (mu - F_hf) . dV/dC;
  3. the same for the best section itself (the reference), and the speed change of each draw.

Run (MDS environment), on a finished run, a few workers (shared machine):
    ..\\.venv-mds\\Scripts\\python scripts\\robustness_checks.py posterior --run runs/<run> --draws 20
    ..\\.venv-mds\\Scripts\\python scripts\\robustness_checks.py tolerance --run runs/<run>
"""
import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import cho_solve, solve_triangular
from scipy.stats import qmc

VPP_DIR = Path(__file__).resolve().parents[1]
for _path in (VPP_DIR, VPP_DIR.parent / "mfego", VPP_DIR / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
# pylint: disable=wrong-import-position
from profile_analysis import (_ScratchFolder, load_surrogate, robust_solve,  # noqa: E402
                              stiffness_ratios)
from src.run_dir import RunDir  # noqa: E402
from src.section_pca import SectionPCA, surface_matrix  # noqa: E402
from src.sequential_design import joint_posterior_covariance, to_unit  # noqa: E402
from src.vpp_targeted import N_FORCES, latent_mean  # noqa: E402

ANCHOR_SPREAD = np.array([0.5, 0.15, 0.5])     # half-width of the anchor cloud [deg], Cant/Rake/Yaw
CHORD_M = 0.70                                 # foil_codes.PLANFORM


# -------------------------------------------------------------------------------------------------
# The run: its sections and conditions
# -------------------------------------------------------------------------------------------------
def load_run(root: Path) -> tuple:
    """config, PCA, conditions, condition weights and the compared sections (name -> scores)."""
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    pca = SectionPCA.load(root / "data" / "pca.json")
    best = json.loads((root / "results" / "best_section.json").read_text(encoding="utf-8"))
    sections = {"best": np.array(best["scores"])}
    optima = root / "results" / "optima.json"
    if optima.exists():
        for entry in json.loads(optima.read_text(encoding="utf-8")):
            if not entry["best"]:
                sections[f"region_{entry['region']}"] = np.array(entry["scores"])
    for path in config.get("compare") or []:
        path = Path(path) if Path(path).is_absolute() else VPP_DIR / path
        earlier = json.loads(path.read_text(encoding="utf-8"))
        weights = np.array(earlier["kulfan_upper"] + earlier["kulfan_lower"])
        sections[f"best_{path.parents[1].name}"] = pca.encode(weights)[0]
    sections[best["start"]] = np.array(best["start_scores"])
    conditions = [tuple(c) for c in config["conditions"]]
    weights = np.array(config["condition_weights"], dtype=float)
    return config, pca, conditions, weights, sections


def mean_speeds(speeds: np.ndarray, solved: np.ndarray, weights: np.ndarray) -> tuple:
    """Weighted mean speed of each section (rows) over the conditions (columns) solved for ALL
    the sections, and the share of the probability these conditions carry."""
    common = solved.all(axis=0)
    w = weights * common
    return speeds[:, common] @ w[common] / w.sum(), float(w.sum() / weights.sum())


def state_points(scores: np.ndarray, states: list) -> np.ndarray:
    return np.array([np.concatenate([scores, [s["Cant_Eff"], s["Rake_Eff"], s["Yaw_Eff"]]])
                     for s in states])


def make_pool(config: dict, n_workers: int):
    from src.vpp_optimize import VPPPool  # noqa: PLC0415  (imports MDS)
    return VPPPool(_ScratchFolder(), n_workers, {"foil": "CFoilFSI", "with_hvol": False,
                                                 "fsi": bool(config.get("fsi"))})


# -------------------------------------------------------------------------------------------------
# Posterior draws
# -------------------------------------------------------------------------------------------------
def anchor_points(points: np.ndarray, n_cloud: int, seed: int, bounds) -> np.ndarray:
    """Every state plus n_cloud attitudes around it (Sobol, +- ANCHOR_SPREAD, clipped)."""
    lo = np.array([b[0] for b in bounds])[-3:]
    hi = np.array([b[1] for b in bounds])[-3:]
    cloud = 2.0 * qmc.Sobol(d=3, scramble=True, seed=seed).random(n_cloud) - 1.0
    out = []
    for point in points:
        out.append(point)
        for offset in cloud:
            moved = point.copy()
            moved[-3:] = np.clip(point[-3:] + offset * ANCHOR_SPREAD, lo, hi)
            out.append(moved)
    return np.array(out)


def sample_factor(model, anchors: np.ndarray) -> tuple:
    """Mean (a,) and a square root R (a, a) of the joint posterior covariance of the top level
    at the anchors, R R^T = K (eigen-decomposition: K is close to singular, anchors are close)."""
    top = np.full(len(anchors), model.num_levels)
    cov = joint_posterior_covariance(model, anchors, top, anchors, top)
    values, vectors = np.linalg.eigh(0.5 * (cov + cov.T))
    return latent_mean(model, anchors), vectors * np.sqrt(np.clip(values, 0.0, None))


def condition_on_values(model, x: np.ndarray, values: np.ndarray) -> None:
    """Conditions a co-kriging on observations of its top level at x with the given values
    (sequential_design.append_observations with values that are not the means): block Cholesky,
    then alpha = V^-1 (z - m) again. A small jitter relative to the prior variance keeps the new
    block positive (the anchors are close to each other and the latent values carry no noise)."""
    levels = np.full(len(x), model.num_levels)
    u = to_unit(model, x)
    # pylint: disable=protected-access
    c = model._covariance(u, levels, to_unit(model, model.x_all), model.levels_all)
    w = solve_triangular(model.l_chol, c.T, lower=True)
    prior = model._covariance(u, levels, u, levels)
    noise = model.gps[-1].get_noise_variance() + model.jitter
    block = prior - w.T @ w
    jitter = 1e-8 * float(np.mean(np.diag(prior)))
    while True:
        try:
            e = np.linalg.cholesky(block + np.eye(len(x)) * (noise + jitter))
            break
        except np.linalg.LinAlgError:
            jitter *= 10.0
    n, a = len(model.l_chol), len(x)
    l_new = np.zeros((n + a, n + a))
    l_new[:n, :n] = model.l_chol
    l_new[n:, :n] = w.T
    l_new[n:, n:] = e
    model.x_all = np.vstack([model.x_all, x])
    model.levels_all = np.concatenate([model.levels_all, levels])
    model.z_all = np.concatenate([model.z_all, values])
    model.l_chol = l_new
    model.alpha = cho_solve((l_new, True), model.z_all - model._mean(model.levels_all))


def posterior(args) -> None:
    root = (VPP_DIR / args.run).resolve()
    config, pca, conditions, weights, sections = load_run(root)
    names = [n for n in sections if args.sections is None or n in args.sections]
    scores = np.array([sections[n] for n in names])
    ratios = stiffness_ratios(config, pca, scores)
    run = RunDir.create(f"robustness_posterior_{root.name[:15]}", {
        "check": "posterior", "of_run": str(root.relative_to(VPP_DIR)), "sections": names,
        "draws": args.draws, "scales": args.scales, "n_cloud": args.n_cloud, "seed": args.seed,
        "workers": args.workers})
    surrogate = load_surrogate(root)
    pool = make_pool(config, args.workers)
    pool.publish(surrogate)
    nominal = robust_solve(pool, list(scores), ratios, conditions)
    speeds0 = np.array([[s["speed"] for s in row] for row in nominal])
    solved0 = np.array([[s["success"] for s in row] for row in nominal])
    j0, share = mean_speeds(speeds0, solved0, weights)
    run.log("nominal (posterior mean): " + ", ".join(f"{n} {v:.4f}" for n, v in zip(names, j0))
            + f" kt on {100 * share:.0f} % of the probability")

    # anchors: the solved states of every section, with their cloud
    points = np.vstack([state_points(s, [st for st in row if st["success"]])
                        for s, row in zip(scores, nominal)])
    anchors = anchor_points(points, args.n_cloud, args.seed, surrogate.bounds)
    factors = [sample_factor(model, anchors) for model in surrogate.models[:N_FORCES]]
    run.log(f"{len(anchors)} anchors, joint posterior of the {N_FORCES} forces / moments")

    rng = np.random.default_rng(args.seed)
    rows = []
    for draw in range(args.draws):
        noise = [rng.standard_normal(len(anchors)) for _ in factors]
        for scale in args.scales:
            drawn = copy.deepcopy(surrogate)
            for model, (mean, root_k), z in zip(drawn.models, factors, noise):
                condition_on_values(model, anchors, mean + scale * root_k @ z)
            pool.publish(drawn)
            states = pool.solve(list(scores), ratios, conditions,
                                extra_guesses=[[[s["unknowns"]] for s in row] for row in nominal])
            for name, row in zip(names, states):
                for (twa, tws), w, s in zip(conditions, weights, row):
                    rows.append({"draw": draw, "scale": scale, "section": name, "twa": twa,
                                 "tws": tws, "weight": w, "success": s["success"],
                                 "speed_kt": s["speed"]})
            speeds = np.array([[s["speed"] for s in row] for row in states])
            solved = np.array([[s["success"] for s in row] for row in states]) & solved0
            j, share = mean_speeds(speeds, solved, weights)
            run.log(f"draw {draw} scale {scale}: " + ", ".join(f"{n} {v:.4f}"
                                                                for n, v in zip(names, j))
                    + f" ({100 * share:.0f} %)")
        import pandas as pd  # noqa: PLC0415
        pd.DataFrame(rows).to_csv(run.path("results", "draws.csv"), index=False)
    pool.close()
    summary = summarize_draws(rows, names, weights, conditions, solved0, j0)
    run.write_json("results/summary.json", summary)
    for line in summary["lines"]:
        run.log(line)


def summarize_draws(rows, names, weights, conditions, solved0, j0) -> dict:
    """Mean speeds per draw, P(best), expected regret of keeping 'best', gains (module docstring)."""
    import pandas as pd  # noqa: PLC0415
    table = pd.DataFrame(rows)
    keys = [f"{a}/{s}" for a, s in conditions]
    out = {"nominal": dict(zip(names, map(float, j0))), "scales": {}, "lines": []}
    for scale, group in table.groupby("scale"):
        per_draw = []
        for _, g in group.groupby("draw"):
            g = g.assign(key=[f"{a}/{s}" for a, s in zip(g.twa, g.tws)])
            speeds = g.pivot(index="section", columns="key", values="speed_kt").loc[names, keys]
            solved = g.pivot(index="section", columns="key", values="success").loc[names, keys]
            j, _ = mean_speeds(speeds.to_numpy(float), solved.to_numpy(bool) & solved0, weights)
            per_draw.append(j)
        per_draw = np.array(per_draw)                       # (draws, sections)
        ib = names.index("best")
        winners = np.argmax(per_draw, axis=1)
        regret = per_draw.max(axis=1) - per_draw[:, ib]
        result = {"n_draws": len(per_draw),
                  "mean": dict(zip(names, per_draw.mean(axis=0).tolist())),
                  "std": dict(zip(names, per_draw.std(axis=0, ddof=1).tolist())),
                  "p_best": {n: float(np.mean(winners == i)) for i, n in enumerate(names)},
                  "regret_mean_kt": float(regret.mean()), "regret_max_kt": float(regret.max()),
                  "gain_vs": {n: [float(np.mean(per_draw[:, ib] - per_draw[:, i])),
                                  float(np.std(per_draw[:, ib] - per_draw[:, i], ddof=1))]
                              for i, n in enumerate(names) if i != ib}}
        out["scales"][str(scale)] = result
        out["lines"].append(
            f"scale {scale} ({len(per_draw)} draws): P(best) "
            + ", ".join(f"{n} {p:.2f}" for n, p in result["p_best"].items())
            + f"; regret of keeping best {result['regret_mean_kt']:.4f} kt (max "
            f"{result['regret_max_kt']:.4f}); best minus "
            + ", ".join(f"{n} {m:+.4f} +- {s:.4f}" for n, (m, s) in result["gain_vs"].items()))
    return out


# -------------------------------------------------------------------------------------------------
# Manufacturing tolerance
# -------------------------------------------------------------------------------------------------
_CODES = {}


def _init_codes(work_dir, levels):
    from src.foil_codes import FoilCodes  # noqa: PLC0415
    _CODES["codes"] = FoilCodes(work_dir, levels=levels)


def _run_code(task):
    weights, attitude, level = task
    return _CODES["codes"].evaluate(weights, attitude, level)


def perturbations(weights: np.ndarray, amplitudes, n_per_amplitude: int, seed: int,
                  te_thickness: float) -> list:
    """(amplitude, weights): random Gaussian changes of the 16 weights, scaled so that the largest
    surface displacement over the chord is the amplitude (fraction of the chord)."""
    matrix, _ = surface_matrix(np.linspace(0.0, 1.0, 201), len(weights) // 2, te_thickness)
    rng = np.random.default_rng(seed)
    out = []
    for amplitude in amplitudes:
        for _ in range(n_per_amplitude):
            delta = rng.standard_normal(len(weights))
            delta *= amplitude / np.max(np.abs(matrix @ delta))
            out.append((amplitude, weights + delta))
    return out


def tolerance(args) -> None:
    from concurrent.futures import ProcessPoolExecutor  # noqa: PLC0415
    from src.foil_codes import CODE_LEVELS  # noqa: PLC0415
    from src.vpp_optimize import SINGLE_THREAD  # noqa: PLC0415
    import os  # noqa: PLC0415
    root = (VPP_DIR / args.run).resolve()
    config, pca, conditions, weights, sections = load_run(root)
    best_weights = pca.decode(sections["best"][None, :])[0]       # the section the codes saw
    cases = [(0.0, best_weights)] + perturbations(best_weights, args.amplitudes, args.n,
                                                  args.seed, pca.te_thickness)
    run = RunDir.create(f"robustness_tolerance_{root.name[:15]}", {
        "check": "tolerance", "of_run": str(root.relative_to(VPP_DIR)),
        "amplitudes_chord": args.amplitudes, "amplitudes_mm": [a * CHORD_M * 1e3
                                                               for a in args.amplitudes],
        "n_per_amplitude": args.n, "seed": args.seed, "workers": args.workers})
    surrogate = load_surrogate(root)
    pool = make_pool(config, args.workers)
    pool.publish(surrogate)
    scores = np.array([pca.encode(w[None, :])[0] for _, w in cases])
    ratios = stiffness_ratios(config, pca, scores)
    states = robust_solve(pool, list(scores), ratios, conditions)
    derivatives = pool.sensitivities(scores[0], ratios[0], conditions, states[0])
    pool.close()
    run.log("VPP on the surrogate and sensitivities of the best section done")

    # the HF code of the run at the states, with the unprojected weights
    names = list(config["levels"])
    levels = [CODE_LEVELS[name][0] for name in names]
    hf = len(levels)
    tasks, owners = [], []
    for i, ((_, w), row) in enumerate(zip(cases, states)):
        for j, s in enumerate(row):
            if s["success"]:
                tasks.append((w, (s["Cant_Eff"], s["Rake_Eff"], s["Yaw_Eff"]), hf))
                owners.append((i, j))
    os.environ.update(SINGLE_THREAD)
    with ProcessPoolExecutor(args.workers, initializer=_init_codes,
                             initargs=(str(run.path("data", "codes")), levels)) as codes:
        results = list(codes.map(_run_code, tasks))
    run.log(f"{len(tasks)} runs of {CODE_LEVELS[names[-1]][1]} done")

    speeds = np.full((len(cases), len(conditions)), np.nan)
    solved = np.zeros_like(speeds, dtype=bool)
    rows = []
    for (i, j), task, (y, info) in zip(owners, tasks, results):
        s = states[i][j]
        point = np.concatenate([scores[i], task[1]])
        mean = surrogate.predict(point[None, :])[0, :6]
        ok = bool(np.all(np.isfinite(y[:6])))
        corrected = s["speed"] - float((mean - y[:6]) @ derivatives[j]) if ok else np.nan
        speeds[i, j], solved[i, j] = corrected, ok
        rows.append({"case": i, "amplitude": cases[i][0], "twa": conditions[j][0],
                     "tws": conditions[j][1], "speed_surrogate_kt": s["speed"],
                     "speed_corrected_kt": corrected, "error": info["error"][:200]})
    import pandas as pd  # noqa: PLC0415
    pd.DataFrame(rows).to_csv(run.path("results", "cases.csv"), index=False)
    j, share = mean_speeds(np.nan_to_num(speeds), solved, weights)
    lines = [f"best section (reference): {j[0]:.4f} kt on {100 * share:.0f} % of the probability"]
    summary = {"reference_kt": float(j[0]), "share": share, "amplitudes": {}}
    for amplitude in args.amplitudes:
        delta = np.array([j[i] - j[0] for i, (a, _) in enumerate(cases) if a == amplitude])
        summary["amplitudes"][str(amplitude)] = delta.tolist()
        lines.append(f"amplitude {amplitude * 100:.3f} % c ({amplitude * CHORD_M * 1e3:.2f} mm): "
                     f"speed change mean {delta.mean():+.4f} kt, min {delta.min():+.4f}, "
                     f"max {delta.max():+.4f} ({len(delta)} sections)")
    summary["lines"] = lines
    run.write_json("results/summary.json", summary)
    for line in lines:
        run.log(line)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="check", required=True)
    p = sub.add_parser("posterior", help="VPP on functions drawn from the GP posterior")
    p.add_argument("--run", required=True, help="run folder, relative to vpp_acceleration")
    p.add_argument("--draws", type=int, default=20)
    p.add_argument("--scales", type=float, nargs="+", default=[1.0, 2.0],
                   help="standard deviation of the draws in units of the posterior one")
    p.add_argument("--n-cloud", type=int, default=4, help="anchors around each state")
    p.add_argument("--sections", nargs="+", default=None,
                   help="names of the compared sections to keep (default: all; each anchor slows "
                        "every evaluation of the drawn surrogate)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=12)
    t = sub.add_parser("tolerance", help="small shape errors of the best section, HF code")
    t.add_argument("--run", required=True)
    t.add_argument("--amplitudes", type=float, nargs="+", default=[0.0005, 0.0015],
                   help="largest surface displacement, fraction of the chord")
    t.add_argument("--n", type=int, default=5, help="sections per amplitude")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    posterior(args) if args.check == "posterior" else tolerance(args)


if __name__ == "__main__":
    main()
