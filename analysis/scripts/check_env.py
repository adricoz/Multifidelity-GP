"""
Environment checks (axis 5 of the report: README vs Python / library versions).

* import of `src.data_management` for the initial and the corrected code, with every Python
  interpreter found by the Windows launcher `py` (and the current interpreter otherwise);
* versions and `Requires-Python` metadata of the dependencies;
* signatures of the third-party functions used by the code (NeuralFoil, scipy qmc / DE).

Usage (from the repository root):  python analysis/scripts/check_env.py
"""
import inspect
import re
import shutil
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, metadata

import plotly.graph_objects as go
from _common import ROOT, VERSION_PATHS, save_figure, save_results

PACKAGES = ["numpy", "scipy", "matplotlib", "plotly", "pytest", "traitlets",
            "neuralfoil", "aerosandbox"]
IMPORT_SNIPPET = ("import sys; sys.path.insert(0, r'{path}'); "
                  "import src.data_management; print('OK')")


def interpreters() -> dict:
    """Python interpreters available through the `py` launcher (Windows)."""
    found = {}
    if shutil.which("py"):
        listing = subprocess.run(["py", "-0p"], capture_output=True, text=True, check=False)
        for line in listing.stdout.splitlines():
            match = re.search(r"-V:(\d+\.\d+)\S*\s+\*?\s*(.+)$", line.strip())
            if match:
                found[match.group(1)] = match.group(2).strip()
    found.setdefault(f"{sys.version_info.major}.{sys.version_info.minor}", sys.executable)
    return found


def import_checks(found: dict) -> dict:
    """Imports src.data_management of both versions with every interpreter."""
    results = {}
    for version, executable in found.items():
        for code_version, path in VERSION_PATHS.items():
            run = subprocess.run([executable, "-I", "-B", "-c",
                                  IMPORT_SNIPPET.format(path=path)],
                                 capture_output=True, text=True, check=False)
            message = run.stdout.strip() or run.stderr.strip().splitlines()[-1]
            results[f"python {version} / {code_version}"] = message
    return results


def package_metadata() -> dict:
    """Installed version and Requires-Python of the dependencies (current interpreter)."""
    results = {}
    for package in PACKAGES:
        try:
            meta = metadata(package)
            results[package] = {"version": meta["Version"],
                                "requires_python": meta.get("Requires-Python")}
        except PackageNotFoundError:
            results[package] = {"version": None, "requires_python": None}
    return results


def signatures() -> dict:
    """Signatures of the third-party functions used by the framework."""
    from scipy.optimize import differential_evolution  # pylint: disable=import-outside-toplevel
    from scipy.stats import qmc  # pylint: disable=import-outside-toplevel
    results = {"scipy.stats.qmc.LatinHypercube": str(inspect.signature(qmc.LatinHypercube)),
               "scipy.optimize.differential_evolution":
                   str(inspect.signature(differential_evolution))}
    try:
        import neuralfoil as nf  # pylint: disable=import-outside-toplevel
        results["neuralfoil.get_aero_from_airfoil"] = \
            str(inspect.signature(nf.get_aero_from_airfoil))
    except ImportError:
        results["neuralfoil.get_aero_from_airfoil"] = "neuralfoil not installed"
    return results


if __name__ == "__main__":
    output = {"python": sys.version, "interpreters": interpreters()}
    output["imports"] = import_checks(output["interpreters"])
    output["packages"] = package_metadata()
    output["signatures"] = signatures()
    for key, value in output["imports"].items():
        print(f"{key:35s} -> {value}")
    save_results(output, "check_env")

    rows = list(output["imports"].items())
    figure = go.Figure(go.Table(
        header={"values": ["Interpreter / code version", "import src.data_management"]},
        cells={"values": [[r[0] for r in rows], [r[1] for r in rows]]}))
    figure.update_layout(title=f"Import checks ({ROOT.name})")
    save_figure(figure, "env_import_checks")
