"""
Night of 2026-11-12 to 13: validation of the optimisation algorithm, unattended (plan approved by Adrian
on 2026-11-12). Started by the scheduled task "Adri validation nuit"
(runs/launch_2026-11-12_validation_nuit.ps1); nothing depends on an interactive session.

1. waits for the end of the run without CFD (its run.log ends with "done"), at most until --start-by;
2. then, in parallel:
   - model-only tests (far_experiments.py: buffered cross-validation, certificate, summary, profile,
     cross judgement) on the run of 11/11 and on the run without CFD, and the existing experiments
     (experiments.py) on the run without CFD;
   - the digital twin (twin_experiments/): direct reference optimum and the method from scratch;
3. verdict of the twin (compare.py), figures, reports;
4. at --stop (08:50): whatever still runs is killed; the reports are written with what is finished and
   the results are committed and pushed (paths of this work only).
Every step is a separate process with its own log (logs/night/<step>.log); a failed step does not stop
the next ones. The state is in logs/night/status.json and VALIDATION_2026-11-13.md is rewritten after every
step (night_report.py).
"""
import argparse
import datetime as dt
import json
import os
import subprocess
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
VPP_DIR = HERE.parent
REPO = VPP_DIR.parent
TWIN = VPP_DIR / "twin_experiments"
PYTHON = REPO / ".venv-mds" / "Scripts" / "python.exe"
LOGS = HERE / "logs" / "night"
RUN_1111 = "runs/2026-11-11_1245_ixxv095_rigid_xtr003_k10_cfd3d"
RUN_NOCFD = "runs/2026-11-12_1118_ixxv095_rigid_xtr003_k10_nocfd"
OUT_NOCFD = "results/nocfd"

STATUS = {"steps": {}, "notes": []}
LOCK = threading.Lock()
REPORT_LOCK = threading.Lock()
CHILDREN = []
ARGS = None


def now() -> dt.datetime:
    return dt.datetime.now()


def log(message: str) -> None:
    line = f"[{now():%Y-%m-%d %H:%M:%S}] {message}"
    print(line, flush=True)
    with LOCK:
        with open(LOGS / "night.log", "a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def save_status() -> None:
    with LOCK:
        (LOGS / "status.json").write_text(json.dumps(STATUS, indent=1, default=str), encoding="utf-8")


def note(message: str) -> None:
    STATUS["notes"].append(f"{now():%H:%M} {message}")
    log(message)
    save_status()


def kill(process) -> None:
    subprocess.run(["taskkill", "/F", "/T", "/PID", str(process.pid)], capture_output=True, check=False)


def step(name: str, command: list, threads: int = 1, cwd: Path = VPP_DIR, deadline=None,
         report: bool = True) -> bool:
    """Runs one process to its end (or to the stop time / its own deadline, then kills its tree)."""
    entry = {"state": "running", "start": now(), "command": " ".join(str(c) for c in command)}
    limit = min(d for d in (ARGS.stop, deadline) if d is not None)
    if now() >= limit:
        entry.update(state="skipped", reason="after the time limit")
        STATUS["steps"][name] = entry
        save_status()
        log(f"{name}: skipped (time limit)")
        return False
    STATUS["steps"][name] = entry
    save_status()
    log(f"{name}: start")
    env = dict(os.environ)
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env[variable] = str(threads)
    env["PYTHONIOENCODING"] = "utf-8"
    with open(LOGS / f"{name}.log", "w", encoding="utf-8") as handle:
        try:
            process = subprocess.Popen([str(c) for c in command], cwd=str(cwd), env=env,
                                       stdout=handle, stderr=subprocess.STDOUT)
        except OSError as error:
            entry.update(state="failed", reason=str(error), end=now())
            save_status()
            log(f"{name}: could not start ({error})")
            return False
        CHILDREN.append(process)
        while process.poll() is None:
            if now() >= limit:
                kill(process)
                entry.update(state="killed", reason="time limit", end=now())
                save_status()
                log(f"{name}: killed at the time limit")
                return False
            time.sleep(10)
    entry.update(state="done" if process.returncode == 0 else "failed", code=process.returncode,
                 end=now())
    save_status()
    log(f"{name}: {entry['state']} (code {process.returncode}, "
        f"{(entry['end'] - entry['start']).total_seconds() / 60:.0f} min)")
    if report:
        write_report()
    return process.returncode == 0


def together(steps: list) -> None:
    """Several steps at once; returns when all have ended."""
    threads = [threading.Thread(target=step, args=a, kwargs=k) for a, k in steps]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


def write_report() -> None:
    """VALIDATION_2026-11-13.md and the figures from what exists now (never raises)."""
    with REPORT_LOCK:
        try:
            with open(LOGS / "report.log", "a", encoding="utf-8") as handle:
                subprocess.run([str(PYTHON), str(HERE / "night_report.py")], cwd=str(VPP_DIR),
                               stdout=handle, stderr=subprocess.STDOUT, timeout=900, check=False)
        except (OSError, subprocess.SubprocessError) as error:
            print(f"report: {error}", flush=True)


def nocfd_finished() -> bool:
    path = VPP_DIR / RUN_NOCFD / "run.log"
    if not path.exists() or not (VPP_DIR / RUN_NOCFD / "results" / "surrogate.pkl").exists():
        return False
    lines = path.read_text(encoding="utf-8", errors="replace").rstrip().splitlines()
    return bool(lines) and lines[-1].rstrip().endswith("] done")


def far(name: str, run: str, out: str, only: str, extra=(), threads: int = 2, **kwargs):
    return ((name, [PYTHON, "-u", HERE / "far_experiments.py", "--run", run, "--out", out,
                    "--only", only, *extra]), {"threads": threads, **kwargs})


def model_chain(nocfd: bool) -> None:
    """Model-only tests (no VPP, no code)."""
    together([far("cv_1111", RUN_1111, "results", "buffered_cv"),
              far("summary_1111", RUN_1111, "results", "summary"),
              far("profile_1111", RUN_1111, "results", "profile", threads=1)])
    together([far("certificate_1111_level2", RUN_1111, "results", "certificate", ["--level", "2"]),
              far("certificate_1111_top", RUN_1111, "results", "certificate")])
    if not nocfd:
        nocfd = nocfd_finished()
    if not nocfd:
        note("run without CFD not finished: its tests are skipped")
        return
    together([((f"{name}_nocfd", [PYTHON, "-u", HERE / "experiments.py", "--run", RUN_NOCFD,
                                  "--out", OUT_NOCFD, "--only", name]), {"threads": 1})
              for name in ("calibration", "gradient", "slices", "regret", "dimension", "levels")])
    together([far("cv_nocfd", RUN_NOCFD, OUT_NOCFD, "buffered_cv"),
              far("summary_nocfd", RUN_NOCFD, OUT_NOCFD, "summary"),
              far("profile_nocfd", RUN_NOCFD, OUT_NOCFD, "profile", threads=1),
              far("certificate_nocfd", RUN_NOCFD, OUT_NOCFD, "certificate")])
    step(*far("cross", RUN_1111, "results", "cross", ["--other", RUN_NOCFD])[0], threads=2)


def main():
    global ARGS  # noqa: PLW0603
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", maxsplit=1)[0])
    parser.add_argument("--stop", default="2026-11-13 08:50", help="everything is killed then")
    parser.add_argument("--start-by", default="2026-11-13 03:00",
                        help="the tests start then even if the run without CFD still runs")
    parser.add_argument("--twin-end", default="2026-11-13 06:50",
                        help="end of the twin searches (reference and method)")
    parser.add_argument("--dry", action="store_true", help="tiny sizes, no wait, no commit (test)")
    ARGS = parser.parse_args()
    for key in ("stop", "start_by", "twin_end"):
        setattr(ARGS, key, dt.datetime.strptime(getattr(ARGS, key), "%Y-%m-%d %H:%M"))
    LOGS.mkdir(parents=True, exist_ok=True)
    note(f"night started (stop {ARGS.stop:%d/%m %H:%M}, dry {ARGS.dry})")

    # 1. the run without CFD
    while not ARGS.dry and not nocfd_finished() and now() < ARGS.start_by:
        time.sleep(60)
    nocfd = nocfd_finished()
    note("run without CFD finished: full use of the machine" if nocfd
         else "run without CFD NOT finished at the start time: reduced number of processes")
    # full load, finish as fast as possible (Adrian 13/11)
    ref_workers, method_workers, compare_workers = (16, 14, 30) if nocfd else (6, 6, 12)

    # 2. model-only tests and the twin, in parallel
    twin_minutes = max(20.0, (ARGS.twin_end - now()).total_seconds() / 60 - 50.0)
    twin_end_text = f"{ARGS.twin_end:%H:%M}"
    method = [PYTHON, "-u", TWIN / "run_twin.py", "--vpp-workers", method_workers,
              "--time-limit-min", f"{min(330.0, twin_minutes):.0f}"]
    reference = [PYTHON, "-u", TWIN / "reference.py", "--workers", ref_workers,
                 "--deadline", twin_end_text]
    if ARGS.dry:
        method = [PYTHON, "-u", TWIN / "run_twin.py", "--smoke", "--vpp-workers", 3, "--name", "smoke_twin"]
        reference = [PYTHON, "-u", TWIN / "reference.py", "--smoke", "--workers", 3]
    chains = [threading.Thread(target=model_chain, args=(nocfd,)),
              threading.Thread(target=step, args=("twin_method", method),
                               kwargs={"threads": 2, "cwd": TWIN}),
              threading.Thread(target=step, args=("twin_reference", reference),
                               kwargs={"threads": 2, "cwd": TWIN})]
    if ARGS.dry:
        chains = chains[1:]
    for chain in chains:
        chain.start()
    for chain in chains[-2:]:                      # the twin searches
        chain.join()
    # the certificate sections go to the verdict of the twin when they are ready in time
    if not ARGS.dry:
        waited = 0
        while chains[0].is_alive() and waited < 30 * 60 and now() < ARGS.stop:
            if (HERE / "results" / "certificate_sections_level2.json").exists():
                break
            time.sleep(30)
            waited += 30

    # 3. verdict of the twin
    compare = [PYTHON, "-u", TWIN / "compare.py", "--workers", compare_workers,
               "--sections", HERE / "results" / "certificate_sections_level2.json"]
    if nocfd_finished():
        compare += ["--runs", RUN_NOCFD]
    if ARGS.dry:
        compare = [PYTHON, "-u", TWIN / "compare.py", "--workers", 3, "--n-conditions", 4,
                   "--reference", "reference_smoke", "--out", "compare_smoke",
                   "--twin-run", str(sorted((TWIN / "twin_runs").glob("*_smoke_twin"))[-1])]
    step("twin_compare", compare, threads=2, cwd=TWIN)
    step("twin_report", [PYTHON, "-u", TWIN / "report.py"] + (["--smoke"] if ARGS.dry else []),
         threads=2, cwd=TWIN)
    for chain in chains:
        chain.join()

    # 4. optional: the real codes at far sections (only when it can end before the stop time)
    verdict = HERE / "codes_verdict.py"
    if not ARGS.dry and verdict.exists() and now() < ARGS.stop - dt.timedelta(minutes=95):
        step("codes_verdict", [PYTHON, "-u", verdict, "--workers", 16], threads=1)
    elif not ARGS.dry:
        note("codes verdict (AVL x XFOIL at far sections) not run: "
             + ("not enough time left" if verdict.exists() else "not implemented"))

    # 5. reports and commit
    for process in CHILDREN:
        if process.poll() is None:
            kill(process)
    note("night finished")
    write_report()
    if not ARGS.dry:
        commit()


def commit() -> None:
    """Commits and pushes the results of this work only (asked by Adrian, 2026-11-12)."""
    import shutil  # noqa: PLC0415
    for leftover in list((TWIN / "twin_runs").glob("*smoke*")) + [HERE / "results_quick"]:
        shutil.rmtree(leftover, ignore_errors=True)            # tests of the chain, not results
    for leftover in list((TWIN / "results").glob("*smoke*")):
        leftover.unlink()
    paths = ["vpp_acceleration/validation_experiments", "vpp_acceleration/twin_experiments",
             "vpp_acceleration/runs/launch_2026-11-12_validation_nuit.ps1"]
    message = ("Validation series of 13/11: digital twin of the section optimisation, buffered "
               "cross-validation, certificate, cross judgement of the runs")
    with open(LOGS / "git.log", "a", encoding="utf-8") as handle:
        for command in (["git", "add", "--", *paths],
                        ["git", "commit", "-m", message, "--", *paths],
                        ["git", "push", "origin", "HEAD"]):
            result = subprocess.run(command, cwd=str(REPO), stdout=handle, stderr=subprocess.STDOUT,
                                    timeout=600, check=False)
            handle.write(f"{' '.join(command[:3])}: code {result.returncode}\n")
            handle.flush()
            if result.returncode != 0:
                break


if __name__ == "__main__":
    main()
