"""
First real run (2026-10-22): search of the C-foil section that maximizes the mean boat speed of
the Intens SY (MDS VPP, branch new_master), with the VPP-targeted sequential co-kriging of the
torsor, levels NPLLT x NeuralFoil (1) and AVL x XFOIL (2) (src/vpp_optimize.py).

Stages (everything goes to runs/<date>_<name>/, see src/run_dir.py):
  1. family and PCA: KulfanBoxFamily of Dim-Reduction/experiments.ipynb (no crossing, NeuralFoil
     confident), PCA on 4000 sections, k modes (search space: the decoded sections are exact,
     the truncation only restricts the search, README of the run);
  2. start design: nested, n_low LF and n_high HF, codes in parallel worker processes;
  3. the loop (VPPSectionOptimizer) until the wall-clock limit;
  4. validation: AVL x XFOIL and NPLLT at the VPP states of the final incumbent and of the mean
     section (start point of the search);
  5. results/*.csv|json, README.md, figures (scripts/real_run_figures.py).

Run (MDS environment), check the load of the machine first:
    ..\\.venv-mds\\Scripts\\python scripts\\run_real_vpp.py --name real_1h --time-limit-min 42
    ..\\.venv-mds\\Scripts\\python scripts\\run_real_vpp.py --name smoke --smoke     # ~6 min check
Third run (HF = NPLLT x XFOIL, flexible foil, parallel batches on 32 logical cores):
    ... run_real_vpp.py --name real_npllt_xfoil_fsi --box current --box-scale 0.5
        --hf npllt_xfoil --hf-cost 2 --fsi --batch 16 --workers 28 --vpp-workers 30
        --compare runs/2026-10-22_1131_real_1h30_current_box/results/best_section.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(variable, "2")

import numpy as np  # noqa: E402

VPP_DIR = Path(__file__).resolve().parents[1]
for path in (VPP_DIR, VPP_DIR.parent / "mfego"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
# pylint: disable=wrong-import-position
from src.meta_foil import confident  # noqa: E402
from src.run_dir import RunDir  # noqa: E402
from src.section_pca import (CFoilFamily, KulfanBoxFamily, SectionPCA,  # noqa: E402
                             notebook_bounds, section_ixx)
from src.foil_codes import LEVELS, LEVELS_XFOIL  # noqa: E402
from src.vpp_coupling import FSICorrection, IntensSYVPP  # noqa: E402
from src.vpp_optimize import (OUTPUT_NAMES, CodeRunner, RealFoilProblem,  # noqa: E402
                              VPPPool, VPPSectionOptimizer, start_design, validate)

CONDITIONS = [(45.0, 12.0), (45.0, 16.0), (60.0, 16.0), (75.0, 16.0), (90.0, 14.0)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="real_1h")
    parser.add_argument("--k", type=int, default=10, help="number of PCA modes of the section")
    parser.add_argument("--n-low", type=int, default=120)
    parser.add_argument("--n-high", type=int, default=30)
    parser.add_argument("--time-limit-min", type=float, default=42.0,
                        help="the loop stops at this time since the start of the run")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true", help="tiny run to check the chain")
    parser.add_argument("--box", choices=("dimred", "current"), default="dimred",
                        help="dimred: the box of the notebook; current: a box of the notebook "
                             "widths x box-scale centred on the current C-foil section, with "
                             "Ixx >= Ixx of the current section (second run, 2026-10-22)")
    parser.add_argument("--box-scale", type=float, default=1.0)
    # third run (2026-10-22): HF = NPLLT x XFOIL, flexible foil, parallel batches
    parser.add_argument("--hf", choices=("avl", "npllt_xfoil"), default="avl",
                        help="high fidelity: AVL x XFOIL (runs 1-2) or NPLLT x XFOIL (AVL is then "
                             "a validation level)")
    parser.add_argument("--hf-cost", type=float, default=6.0,
                        help="cost of an HF run for the criterion, LF = 1 (module docstring of "
                             "src/vpp_optimize.py)")
    parser.add_argument("--fsi", action="store_true",
                        help="flexible foil: deformation correction of the production FSI RS "
                             "(vpp_coupling.FSICorrection), scaled by Ixx(current) / Ixx")
    parser.add_argument("--batch", type=int, default=1, help="choices per batch (1 = sequential)")
    parser.add_argument("--vpp-workers", type=int, default=0,
                        help="processes for the VPP solves (0 = in the main process)")
    parser.add_argument("--compare", nargs="*", default=[],
                        help="best_section.json of earlier runs, validated at the end as well")
    args = parser.parse_args()
    if args.smoke:
        args.n_low, args.n_high, args.time_limit_min = 30, 8, 4.0
    conditions = CONDITIONS[:2] if args.smoke else CONDITIONS
    # sensitivities every 3 updates: they move little, and the updates took 42 % of the time
    # of the first run. In batches (third run), refit / update every so many BATCHES.
    settings = {"n_per_state": 16, "n_local": 512, "n_global": 256, "refit_every": 10,
                "update_every": 10, "sensitivities_every": 3, "n_screen": 256, "n_top": 3,
                "kappa": 1.0}
    if args.batch > 1:
        settings.update({"update_every": 1, "refit_growth": 1.25, "n_lf_only": 512})
    levels = LEVELS_XFOIL if args.hf == "npllt_xfoil" else LEVELS
    level_names = ("1 = NPLLT x NeuralFoil xxxlarge, 2 = NPLLT x XFOIL, 3 = AVL x XFOIL "
                   "(validation only)" if args.hf == "npllt_xfoil" else
                   "1 = NPLLT x NeuralFoil xxxlarge, 2 = AVL x XFOIL") + " (src/foil_codes.py)"
    config = {**vars(args), "conditions": conditions, "optimizer": settings,
              "levels": level_names, "criterion_costs": [1.0, args.hf_cost],
              "family": "Dim-Reduction box, no crossing, NF confident" if args.box == "dimred" else
              f"Dim-Reduction widths x {args.box_scale} centred on the current C-foil section, "
              "no crossing, Ixx >= Ixx(current), NF confident (checked on the decoded sections)",
              "inputs": f"{args.k} PCA scores + Cant_Eff, Rake_Eff, Yaw_Eff (no Hvol)",
              "foil_variant_mds": "CFoilFSI (R 9.8 m), dyn RS replaced by the rigid surrogate"
              + (" + the FSI correction (Speed_kts input)" if args.fsi else "")}
    run = RunDir.create(args.name, config)

    # 1. family and PCA
    current = CFoilFamily.from_study_config(
        VPP_DIR.parent / "studies" / "cfoil_intens_kulfan_trim_2" / "config_cfoil_trim.json"
    ).reference_weights()                                # the current C-foil section (cambered)
    ixx_current = float(section_ixx(current, 0.00197)[0])
    if args.box == "dimred":
        family = KulfanBoxFamily.from_notebook(extra_check=confident)
    else:
        lower, upper = notebook_bounds()

        def stiff_and_confident(weights):
            return (section_ixx(weights, 0.00197) >= ixx_current) & confident(weights)
        family = KulfanBoxFamily.centered(current, args.box_scale * (upper - lower), 0.00197,
                                          extra_check=stiff_and_confident)
        run.log(f"family: box centred on the current section, scale {args.box_scale}, "
                f"Ixx >= {ixx_current:.4e}")
    pca = SectionPCA(family.te_thickness).fit(family.sample(4000, seed=args.seed))
    pca.n_modes = args.k
    pca.save(run.path("data", "pca.json"))
    run.log(f"PCA: rank {pca.rank}, k = {args.k} modes carry "
            f"{100 * pca.variance_ratio[:args.k].sum():.2f} % of the geometric variance")
    problem = RealFoilProblem(pca, family, ixx_reference=ixx_current if args.fsi else None)
    # start of the search: the current section (projected on the k modes) when it is admissible
    # once decoded, otherwise the admissible sample closest to it; the mean section for "dimred"
    start_name, start_scores = "mean_section", np.zeros(args.k)
    if args.box == "current":
        projected = pca.encode(current)[0]
        if family.contains(pca.decode(projected))[0]:
            start_name, start_scores = "current_section", projected
        else:
            pool = problem.sample_sections(2000, seed=args.seed + 5)
            start_name = "closest_to_current"
            start_scores = pool[np.argmin(np.linalg.norm(pool - projected, axis=1))]
        run.log(f"start of the search: {start_name}")

    # 2. start design
    runner = CodeRunner(problem, run, n_workers=args.workers, levels=levels)
    surrogate = start_design(problem, runner, args.n_low, args.n_high, args.seed,
                             costs=(1.0, args.hf_cost), n_threads=7)

    # 3. the loop
    vpp = IntensSYVPP(foil="CFoilFSI", with_hvol=False,
                      fsi=FSICorrection() if args.fsi else None)
    vpp_pool = VPPPool(run, args.vpp_workers, {"foil": "CFoilFSI", "with_hvol": False,
                                               "fsi": args.fsi}) if args.vpp_workers > 1 else None
    optimizer = VPPSectionOptimizer(problem, surrogate, runner, vpp, conditions, run,
                                    time_limit_s=60.0 * args.time_limit_min, seed=args.seed,
                                    start_scores=start_scores, batch_size=args.batch,
                                    vpp_pool=vpp_pool, **settings)
    optimizer.run_loop()
    run.log(f"loop done: {optimizer.iteration} choices in {optimizer.n_batches} batches, LF/HF "
            f"{surrogate.sizes}, runs {runner.n_runs}, code time {runner.time}")

    # 4. validation: the best section, the start section and the best sections of earlier runs
    sections = {"best": optimizer.incumbent, start_name: start_scores}
    for path in args.compare:
        earlier = json.loads(Path(path).read_text(encoding="utf-8"))
        weights = np.array(earlier["kulfan_upper"] + earlier["kulfan_lower"])
        sections[f"best_{Path(path).parents[1].name}"] = pca.encode(weights)[0]
    rows = validate(optimizer, sections, extra_levels={"avl": 3} if args.hf == "npllt_xfoil"
                    else None)
    run.log("validation done")
    runner.close()
    if vpp_pool is not None:
        vpp_pool.close()

    # 5. results
    import pandas as pd  # noqa: PLC0415
    pd.DataFrame(optimizer.history).to_csv(run.path("results", "history.csv"), index=False)
    run.write_json("results/updates.json", optimizer.updates)
    validation = pd.DataFrame(rows)
    validation.to_csv(run.path("results", "validation.csv"), index=False)
    best_weights = problem.weights(optimizer.incumbent)[0]
    run.write_json("results/best_section.json", {
        "scores": optimizer.incumbent.tolist(), "start": start_name,
        "start_scores": np.asarray(start_scores).tolist(),
        "kulfan_upper": best_weights[:8].tolist(),
        "kulfan_lower": best_weights[8:].tolist(), "te_thickness": family.te_thickness,
        "speeds_by_condition": dict(zip([f"{a:.0f}/{s:.0f}" for a, s in conditions],
                                        optimizer.updates[-1]["incumbent_speeds"])),
        "mean_speed": optimizer.updates[-1]["incumbent_speed"]})
    summary = summarize(run, optimizer, runner, validation, conditions)
    (run.path("results", "summary.txt")).write_text(summary + "\n", encoding="utf-8")
    print(summary)
    from real_run_figures import make_figures  # noqa: PLC0415
    make_figures(run.root)
    run.log("done")


def summarize(run, optimizer, runner, validation, conditions) -> str:
    first, last = optimizer.updates[0], optimizer.updates[-1]
    lines = [f"run {run.root.name}: {optimizer.iteration} steps in {run.elapsed() / 60:.1f} min",
             f"code runs: LF {runner.n_runs[1]} ({runner.time[1]:.0f} s), HF {runner.n_runs[2]} "
             f"({runner.time[2]:.0f} s); final design LF/HF {optimizer.surrogate.sizes}",
             f"mean speed predicted: start section {first['candidate_speeds'][0]:.4f} kt -> "
             f"incumbent {last['incumbent_speed']:.4f} kt",
             "speeds of the incumbent: " + ", ".join(
                 f"{a:.0f}/{s:.0f}: {v:.3f}" for (a, s), v in zip(conditions,
                                                                   last["incumbent_speeds"]))]
    hf = "AVL" if runner.levels[1]["solver"] == "avl" else "NPLLT x XFOIL"
    for name, group in validation.groupby("section"):
        err = group["speed_error_kt"].to_numpy()
        sig = group["speed_sigma_kt"].to_numpy()
        lines.append(f"validation {name}: first-order speed error of the surrogate against {hf} "
                     f"RMS {np.sqrt(np.mean(err ** 2)):.4f} kt, max {np.max(np.abs(err)):.4f} kt;"
                     f" predicted sigma RMS {np.sqrt(np.mean(sig ** 2)):.4f} kt; |z| <= 2 on "
                     f"{np.mean(np.abs(err) <= 2 * sig) * 100:.0f} %")
        if "speed_error_avl_kt" in group:
            err_avl = group["speed_error_avl_kt"].to_numpy()
            lines.append(f"   against AVL x XFOIL (not a level of the surrogate): speed error "
                         f"mean {np.mean(err_avl):+.4f} kt, RMS {np.sqrt(np.mean(err_avl ** 2)):.4f} kt")
        for name_k in OUTPUT_NAMES:
            rel = np.abs(group[f"{name_k}_sur"] - group[f"{name_k}_hf"]) \
                / np.maximum(np.abs(group[f"{name_k}_hf"]), 1e-6)
            lines.append(f"   {name_k:6s} |surrogate - HF| / |HF| median {np.median(rel) * 100:.2f} %")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
