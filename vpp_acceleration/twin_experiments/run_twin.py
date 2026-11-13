"""
The complete method, from scratch, on the digital twin (twin.py): scripts/run_real_vpp.py unchanged,
its CodeRunner replaced at run time by TwinRunner, its run folder created under
twin_experiments/twin_runs/. Same loop, same outputs and figures as a real run.

Default options = those of the run without CFD of 2026-11-12 (runs/launch_2026-11-12_ixxv095_nocfd.ps1:
2 levels, start design 256 / 64, costs 1 / 6, batches of 16, same search box and profile, same
convergence and restarts), with other seeds and no end-of-run profile analysis.

    ..\\.venv-mds\\Scripts\\python twin_experiments\\run_twin.py --time-limit-min 330 --vpp-workers 10
    ..\\.venv-mds\\Scripts\\python twin_experiments\\run_twin.py --smoke --vpp-workers 4      (a few minutes)
Other options are passed on to run_real_vpp.py.
"""
import argparse
import os
import sys
from pathlib import Path

from twin import HERE, TRUTH_RUN, VPP_DIR, TwinRunner, TwinTruth

TWIN_RUNS = HERE / "twin_runs"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--truth-run", default=str(TRUTH_RUN))
    parser.add_argument("--name", default="twin_method")
    parser.add_argument("--time-limit-min", default="330")
    parser.add_argument("--vpp-workers", default="10")
    parser.add_argument("--seed", default="1")
    parser.add_argument("--screen-seed", default="20261112")
    parser.add_argument("--smoke", action="store_true")
    args, passed_on = parser.parse_known_args()
    os.chdir(VPP_DIR)                      # run_real_vpp.py takes its paths from vpp_acceleration

    import run_real_vpp  # noqa: PLC0415  (scripts/, on sys.path through twin)
    from src.run_dir import RunDir  # noqa: PLC0415

    truth = TwinTruth.load(Path(args.truth_run))

    class TwinRunDir(RunDir):
        @classmethod
        def create(cls, name, config, runs=TWIN_RUNS):
            config = {**config, "twin": {"truth_run": str(args.truth_run), "truth_level": truth.level,
                                         "note": "codes replaced by the means of the truth run "
                                                 "(twin_experiments/twin.py)"}}
            return super().create(name, config, runs)

    run_real_vpp.RunDir = TwinRunDir
    run_real_vpp.CodeRunner = lambda problem, run, n_workers=1, levels=None: TwinRunner(
        problem, run, truth, n_levels=len(levels))

    sys.argv = [
        "run_real_vpp.py",
        "--search-box", "runs/2026-11-11_1054_section_pca_convex_ixxv095", "--k", "10",
        "--levels", "nf_llt", "avl", "--n-design", "256", "64", "--costs", "1", "6", "--batch", "16",
        # smoke: the 2 fixed conditions of run_real_vpp.py instead of the 49 of the profile
        *([] if args.smoke else ["--search-profile", "profiles/profile_v2.json"]),
        "--local-fraction", "0.5", "--radius", "1.0", "--converge", "10", "0.01", "--restarts", "3",
        "--min-gain", "0.002", "--sensitivity-median", "3",
        "--seed", args.seed, "--screen-seed", args.screen_seed,
        "--workers", "1", "--vpp-workers", args.vpp_workers,
        "--time-limit-min", args.time_limit_min, "--name", args.name,
    ] + (["--smoke"] if args.smoke else []) + passed_on
    run_real_vpp.main()


if __name__ == "__main__":
    main()
