"""
[FIX-L1] Run management helpers: one timestamped directory per run, log file and total
computation time.

Same convention as bdFoil (NonPlanarSolver/main.py): the log file is named
<prefix>_<MMDD_HHMMSS>.log with the Paris time, the log format is ' %(levelname)s - %(message)s'
and the same timestamp is used for every output of the run. Two additions: the log file is
written in UTF-8 (bdFoil logs are written in cp1252 on Windows) and the total computation time
is written at the end of the log.
"""
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

try:
    from zoneinfo import ZoneInfo
    TIMEZONE = ZoneInfo("Europe/Paris")
except Exception:  # noqa: BLE001  (tzdata missing: local time is used)
    TIMEZONE = None

LOG_FORMAT = ' %(levelname)s - %(message)s'
TIMESTAMP_FORMAT = "%m%d_%H%M%S"

logger = logging.getLogger(__name__)


def now() -> datetime:
    """Current date and time (Paris time zone when available)."""
    return datetime.now(TIMEZONE) if TIMEZONE is not None else datetime.now()


@dataclass(frozen=True)
class RunContext:
    """
    Paths of one run.

    Args:
    - timestamp: MMDD_HHMMSS string shared by every output of the run.
    - run_dir: directory <base_dir>/runs/<timestamp>/ holding all the outputs.
    - log_path: <run_dir>/<prefix>_<timestamp>.log.
    """
    timestamp: str
    run_dir: Path
    log_path: Path

    def path(self, filename: str) -> str:
        """Path of an output file of the run (as a string, for the existing APIs)."""
        return str(self.run_dir / filename)


# 1/3 ---------------------------------------------------------------------------------------------
def create_run(base_dir: str, prefix: str = "mfego") -> RunContext:
    """
    Creates the directory of a new run: <base_dir>/runs/<MMDD_HHMMSS>/ (a numeric suffix is
    added if a run was already started in the same second).

    Args:
    - base_dir: directory of the calling script (e.g. os.path.dirname(__file__)).
    - prefix: prefix of the log file name.
    Returns:
    - RunContext of the run.
    """
    timestamp = now().strftime(TIMESTAMP_FORMAT)
    run_dir = Path(base_dir) / "runs" / timestamp
    suffix = 1
    while run_dir.exists():
        run_dir = Path(base_dir) / "runs" / f"{timestamp}_{suffix}"
        suffix += 1
    os.makedirs(run_dir, exist_ok=True)
    return RunContext(timestamp=timestamp, run_dir=run_dir,
                      log_path=run_dir / f"{prefix}_{timestamp}.log")


# 2/3 ---------------------------------------------------------------------------------------------
def setup_logging(log_path: str, level: int = logging.INFO) -> None:
    """
    Configures the root logger to write in log_path (bdFoil format, UTF-8, one file per run).
    The framework modules never configure logging themselves (FIX-R3): call it from the
    script.
    """
    logging.basicConfig(
        filename=str(log_path),
        filemode='w',
        encoding='utf-8',
        level=level,
        format=LOG_FORMAT,
        force=True,
    )


# 3/3 ---------------------------------------------------------------------------------------------
class RunTimer:
    """
    Context manager writing a start banner and, at the end (also when an exception is
    raised), the total computation time of the run.

    Example:
        with RunTimer("Forrester example"):
            ego.run(n_iterations=10)
    """
    def __init__(self, name: str = "mfego run"):
        self.name = name
        self.start = None
        self.elapsed = None

    def __enter__(self) -> "RunTimer":
        self.start = time.perf_counter()
        logger.info("%s", 50 * "-")
        logger.info("%s started on %s", self.name, now().strftime("%d/%m/%Y at %H:%M:%S"))
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        self.elapsed = time.perf_counter() - self.start
        if exc_type is not None:
            logger.error("%s failed: %s: %s", self.name, exc_type.__name__, exc)
        logger.info("%s finished on %s", self.name, now().strftime("%d/%m/%Y at %H:%M:%S"))
        logger.info("Total computation time: %.2f s (%s)", self.elapsed,
                    format_duration(self.elapsed))
        logger.info("%s", 50 * "-")
        return False


def format_duration(seconds: float) -> str:
    """Duration as h:mm:ss (with tenths of a second below one minute)."""
    if seconds < 60.0:
        return f"0:00:{seconds:04.1f}"
    return str(timedelta(seconds=round(seconds)))
