"""Unit tests of src/run_utils.py (timestamped run directory, log file, total computation time)."""
import logging
import re

import pytest
from src.run_utils import RunTimer, create_run, format_duration, setup_logging


@pytest.fixture(autouse=True)
def reset_logging():
    yield
    logging.basicConfig(force=True, handlers=[logging.NullHandler()])


def test_run_directory_and_log_name(tmp_path):
    """[FIX-L1] bdFoil convention: runs/<MMDD_HHMMSS>/<prefix>_<MMDD_HHMMSS>.log."""
    run = create_run(tmp_path, prefix="mfego")
    assert re.fullmatch(r"\d{4}_\d{6}", run.timestamp)
    assert run.run_dir == tmp_path / "runs" / run.timestamp
    assert run.run_dir.is_dir()
    assert run.log_path.name == f"mfego_{run.timestamp}.log"
    assert run.path("ego_backup.json") == str(run.run_dir / "ego_backup.json")


def test_two_runs_in_the_same_second_do_not_collide(tmp_path):
    first, second = create_run(tmp_path), create_run(tmp_path)
    assert first.run_dir != second.run_dir


def test_log_contains_the_total_computation_time(tmp_path):
    run = create_run(tmp_path, prefix="test")
    setup_logging(run.log_path)
    with RunTimer("unit test run") as timer:
        logging.getLogger("unit").info("degree sign: °")
    text = run.log_path.read_text(encoding="utf-8")
    assert "unit test run started on" in text
    assert "degree sign: °" in text  # UTF-8 (bdFoil logs are written in cp1252)
    assert re.search(r"Total computation time: \d+\.\d{2} s \(", text)
    assert timer.elapsed >= 0.0
    assert text.startswith(" INFO - ")  # same format as bdFoil


def test_total_time_is_written_when_the_run_fails(tmp_path):
    run = create_run(tmp_path, prefix="test")
    setup_logging(run.log_path)
    with pytest.raises(RuntimeError):
        with RunTimer("failing run"):
            raise RuntimeError("simulation crashed")
    text = run.log_path.read_text(encoding="utf-8")
    assert "failing run failed: RuntimeError: simulation crashed" in text
    assert "Total computation time" in text


def test_format_duration():
    assert format_duration(2.66) == "0:00:02.7"
    assert format_duration(3725.2) == "1:02:05"
