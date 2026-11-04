"""
One folder per run, so that the tests, the data, the results and the figures never mix.

    vpp_acceleration/runs/<YYYY-MM-DD_HHMM>_<name>/
        config.json      every setting of the run (written at the start)
        run.log          progress and timings (also printed)
        data/            evaluations.csv (one row per code run: level, inputs, outputs, time,
                         status), the PCA, the caches of the codes
        results/         tables and JSON written by the run
        figures/         HTML figures
        README.md        summary written at the end of the run

Usage:
    run = RunDir.create("real_1h", config)       # new folder
    run.log("start")                             # timestamped line, file + console
    run.add_evaluation({...})                    # appended to data/evaluations.csv
    run.path("results", "history.csv")           # path inside the run
"""
import csv
import json
import time
from datetime import datetime
from pathlib import Path

RUNS = Path(__file__).resolve().parents[1] / "runs"


class RunDir:
    """A run folder (module docstring)."""

    def __init__(self, root: Path):
        self.root = Path(root)
        for sub in ("data", "results", "figures"):
            (self.root / sub).mkdir(parents=True, exist_ok=True)
        self.start = time.perf_counter()
        self._fields = None

    @classmethod
    def create(cls, name: str, config: dict, runs: Path = RUNS) -> "RunDir":
        stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
        run = cls(Path(runs) / f"{stamp}_{name}")
        run.write_json("config.json", config)
        run.log(f"run folder {run.root}")
        return run

    def path(self, *parts) -> Path:
        return self.root.joinpath(*parts)

    def elapsed(self) -> float:
        """Seconds since the run folder was opened."""
        return time.perf_counter() - self.start

    def log(self, message: str) -> None:
        line = f"{datetime.now().strftime('%H:%M:%S')} [{self.elapsed() / 60:5.1f} min] {message}"
        print(line, flush=True)
        with open(self.root / "run.log", "a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def write_json(self, relative: str, data) -> Path:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=1, default=float), encoding="utf-8")
        return path

    def add_evaluation(self, row: dict) -> None:
        """Appends one code run to data/evaluations.csv (the columns of the first row)."""
        path = self.root / "data" / "evaluations.csv"
        if self._fields is None:
            self._fields = list(row)
            new = not path.exists()
            with open(path, "a", newline="", encoding="utf-8") as handle:
                if new:
                    csv.DictWriter(handle, fieldnames=self._fields).writeheader()
        with open(path, "a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=self._fields, extrasaction="ignore").writerow(row)
