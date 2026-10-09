"""
Location of bdFoil / bdToolbox and lazy imports of their modules.

Paths (first defined wins):
1. the "paths" entry of the configuration file ("bdfoil_root", "bdtoolbox_root", "soft_dir"),
2. the environment variables MFEGO_BDFOIL_ROOT, MFEGO_BDTOOLBOX_ROOT and BDFOIL_SOFT,
3. the defaults below: the up-to-date bdFoil clone of Code-Adri (bdFoil core XFOIL/AVL engine
   and NonPlanarSolver) and the executables / bdSec of the installed bdToolbox.

bdFoil core is imported as `bdFoil.Code.core` (relative imports only, it never imports the
legacy foil.py); bdSec as `bdSec.Code.section` (bdToolbox root on sys.path).
"""
import importlib
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_BDFOIL_ROOT = Path(r"C:\Users\SIM\Desktop\Code-Adri\bdFoil")
DEFAULT_BDTOOLBOX_ROOT = Path(r"C:\Users\SIM\Outils - banulsdesign\bdToolbox")


class BridgeError(RuntimeError):
    """A bdToolbox / bdFoil component cannot be found or imported."""


@dataclass(frozen=True)
class BridgePaths:
    """Resolved locations of the external toolboxes."""
    bdfoil_root: Path
    bdtoolbox_root: Path
    soft_dir: Path


def resolve_paths(config_paths: dict = None) -> BridgePaths:
    """
    Resolves the toolbox locations (configuration > environment > defaults).

    Args:
    - config_paths: optional dict with "bdfoil_root", "bdtoolbox_root", "soft_dir".
    Returns:
    - BridgePaths.
    """
    config_paths = config_paths or {}

    def pick(key, env, default):
        value = config_paths.get(key) or os.environ.get(env) or default
        return Path(value).expanduser()

    bdtoolbox_root = pick("bdtoolbox_root", "MFEGO_BDTOOLBOX_ROOT", DEFAULT_BDTOOLBOX_ROOT)
    return BridgePaths(
        bdfoil_root=pick("bdfoil_root", "MFEGO_BDFOIL_ROOT", DEFAULT_BDFOIL_ROOT),
        bdtoolbox_root=bdtoolbox_root,
        soft_dir=pick("soft_dir", "BDFOIL_SOFT", bdtoolbox_root / "soft"))


def setup(paths: BridgePaths) -> None:
    """
    Makes the toolboxes importable and points bdFoil core to the executables (BDFOIL_SOFT).
    The parent of the bdFoil clone comes first on sys.path so that `bdFoil.Code.core` is
    found in the up-to-date clone (namespace package shared with bdToolbox/bdFoil).
    """
    for path in (paths.bdtoolbox_root, paths.bdfoil_root.parent):
        if str(path) in sys.path:
            sys.path.remove(str(path))
        sys.path.insert(0, str(path))
    os.environ["BDFOIL_SOFT"] = str(paths.soft_dir)
    logger.info("bdFoil: %s | bdToolbox: %s | executables: %s", paths.bdfoil_root,
                paths.bdtoolbox_root, paths.soft_dir)


def _import(module: str, hint: str):
    try:
        return importlib.import_module(module)
    except ImportError as error:
        raise BridgeError(f"cannot import {module} ({error}). {hint}") from error


def bdfoil_core():
    """bdFoil core package (model, xfoil, forces, campaign...)."""
    core = _import("bdFoil.Code.core", "Check MFEGO_BDFOIL_ROOT (bdFoil clone with Code/core).")
    for sub in ("model", "xfoil", "forces"):
        _import(f"bdFoil.Code.core.{sub}", "Incomplete bdFoil core.")
    return core


def section_parsec_class():
    """bdSec SectionParsec class (PARSEC parametrization)."""
    module = _import("bdSec.Code.section",
                     "Check MFEGO_BDTOOLBOX_ROOT (bdToolbox root with bdSec) and the package "
                     "'tabulate'.")
    return module.SectionParsec


def neuralfoil():
    """NeuralFoil module (>= 0.2.0 for n_crit / xtr and the xxxlarge model)."""
    return _import("neuralfoil", "Install neuralfoil >= 0.2.0 (conda env 'bdToolbox').")


def np_llt_solver(paths: BridgePaths):
    """bdFoil non-planar lifting-line solver module (bdFoil/NonPlanarSolver)."""
    path = paths.bdfoil_root / "NonPlanarSolver"
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    return _import("np_llt_solver", "Check bdFoil/NonPlanarSolver (numba, neuralfoil).")


def _git_head(folder: Path) -> str:
    """Branch and commit of a git clone ("" if not available)."""
    try:
        out = subprocess.run(["git", "-C", str(folder), "log", "-1", "--format=%h %cs"],
                             capture_output=True, text=True, timeout=10, check=False)
        branch = subprocess.run(["git", "-C", str(folder), "branch", "--show-current"],
                                capture_output=True, text=True, timeout=10, check=False)
        return f"{branch.stdout.strip()} {out.stdout.strip()}".strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def environment_report(paths: BridgePaths) -> dict:
    """
    Versions and locations a result depends on (stored with the results: e.g. NeuralFoil
    0.2.3 and 0.3.x do not give the same polars): Python, numpy, scipy, NeuralFoil,
    aerosandbox, numba, the bdFoil clone (branch, commit) and the executables.
    """
    report = {"python": sys.version.split()[0], "executable": sys.executable}
    for name in ("numpy", "scipy", "neuralfoil", "aerosandbox", "numba"):
        try:
            report[name] = getattr(importlib.import_module(name), "__version__", "?")
        except ImportError:
            report[name] = None
    report.update({
        "bdfoil_root": str(paths.bdfoil_root), "bdfoil_git": _git_head(paths.bdfoil_root),
        "soft_dir": str(paths.soft_dir),
        "xfoil_exe": (paths.soft_dir / "xfoil.exe").is_file(),
        "avl_exe": (paths.soft_dir / "avl_3.40b.exe").is_file()})
    return report
