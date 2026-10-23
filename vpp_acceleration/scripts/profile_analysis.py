"""
Behaviour of the sections of a run on an operating profile (src/operating_profile.py, the
distribution of TWA / TWS the boat really meets, given per run): the VPP with the FINAL surrogate
of the run, and its FSI correction when the run had one, is solved at every node of the
distribution, for the best section, the start section and the sections compared at the end of
the run. The search is not changed: this is an analysis of its result (decision of 2026-10-22).

For every section:
  * the expected speed E_p[V] = sum_nodes w V / sum_nodes w over the nodes the VPP solved for
    EVERY section (the same conditions for all), and the probability each section solved
    (reported, not hidden: the VPP of the boat has no equilibrium deep downwind in light wind,
    heel at its bound 0, even with the production RS; 2026-10-22);
  * the same per mode, and the time-equivalent speed 1 / E_p[1 / V] (a fixed course);
  * the gain against the start section, on the nodes both solved;
  * at every node, the standard deviation of the predicted speed, sqrt(sum_k (dV/dC_k)^2 s_k^2),
    with the sensitivities at that node (VPPPool.sensitivities): the surrogate was refined around
    the states of the conditions of the run, a node far from them (downwind) is less certain.

Outputs (run folder): results/profile_<name>.csv (one row per section and node),
results/profile_<name>.json (expectations), figures/profile_<name>.html.

Run (MDS environment), on a finished run:
    ..\\.venv-mds\\Scripts\\python scripts\\profile_analysis.py --run runs/<run>
        --profile profiles/guess_bimodal.json
or at the end of a run: run_real_vpp.py ... --profile profiles/guess_bimodal.json
"""
import argparse
import json
import pickle
import sys
import tempfile
from pathlib import Path

import numpy as np

VPP_DIR = Path(__file__).resolve().parents[1]
for _path in (VPP_DIR, VPP_DIR.parent / "mfego"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
# pylint: disable=wrong-import-position
from src.operating_profile import OperatingProfile  # noqa: E402
from src.section_pca import CFoilFamily, SectionPCA, section_ixx  # noqa: E402

STUDY_CONFIG = VPP_DIR.parent / "studies" / "cfoil_intens_kulfan_trim_2" / "config_cfoil_trim.json"


class _ScratchFolder:
    """What VPPPool needs of a run folder: a place for the pickled surrogate."""

    def __init__(self):
        self.root = Path(tempfile.mkdtemp(prefix="profile_"))

    def path(self, *parts) -> Path:
        path = self.root.joinpath(*parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path


def load_surrogate(root: Path):
    """The final surrogate of a run: results/surrogate.pkl, or the last surrogate published to the
    VPP workers (data/surrogate_v<n>.pkl). Both are mean-only copies; conditioning them again
    (hyperparameters unchanged) gives back the variances."""
    path = root / "results" / "surrogate.pkl"
    if not path.exists():
        versions = sorted(root.glob("data/surrogate_v*.pkl"),
                          key=lambda p: int(p.stem.split("_v")[-1]))
        if not versions:
            raise FileNotFoundError(f"no saved surrogate in {root}")
        path = versions[-1]
    with open(path, "rb") as file:
        surrogate = pickle.load(file)
    surrogate.fit(refit=False)
    return surrogate


def run_sections(root: Path, config: dict, pca) -> dict:
    """name -> scores: the best section, the start section, the compared sections."""
    best = json.loads((root / "results" / "best_section.json").read_text(encoding="utf-8"))
    sections = {"best": np.array(best["scores"]), best["start"]: np.array(best["start_scores"])}
    for path in config.get("compare") or []:
        path = Path(path) if Path(path).is_absolute() else VPP_DIR / path
        earlier = json.loads(path.read_text(encoding="utf-8"))
        weights = np.array(earlier["kulfan_upper"] + earlier["kulfan_lower"])
        sections[f"best_{path.parents[1].name}"] = pca.encode(weights)[0]
    return sections


def stiffness_ratios(config: dict, pca, scores: np.ndarray) -> np.ndarray:
    """Ixx(current) / Ixx(section) with the FSI correction, 1 without (RealFoilProblem)."""
    if not config.get("fsi"):
        return np.ones(len(scores))
    current = CFoilFamily.from_study_config(STUDY_CONFIG).reference_weights()
    reference = float(section_ixx(current, pca.te_thickness)[0])
    return reference / section_ixx(pca.decode(scores), pca.te_thickness)


def analyse(root, profile: OperatingProfile, n_workers: int = 16, sigma: bool = True,
            vpp_pool=None, surrogate=None) -> dict:
    """The analysis of the module docstring; returns the summary (also written as JSON)."""
    import pandas as pd  # noqa: PLC0415
    from src.vpp_optimize import VPPPool  # noqa: PLC0415
    root = Path(root)
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    pca = SectionPCA.load(root / "data" / "pca.json")
    surrogate = surrogate if surrogate is not None else load_surrogate(root)
    sections = run_sections(root, config, pca)
    names = list(sections)
    scores = np.array([sections[n] for n in names])
    ratios = stiffness_ratios(config, pca, scores)
    nodes = profile.nodes()
    conditions = [(node["twa"], node["tws"]) for node in nodes]

    own_pool = vpp_pool is None
    if own_pool:
        vpp_pool = VPPPool(_ScratchFolder(), n_workers, {"foil": "CFoilFSI", "with_hvol": False,
                                                         "fsi": bool(config.get("fsi"))})
    vpp_pool.publish(surrogate)
    all_states = vpp_pool.solve(list(scores), ratios, conditions)
    rows = []
    for name, section, ratio, states in zip(names, scores, ratios, all_states):
        points = np.array([np.concatenate([section, [s["Cant_Eff"], s["Rake_Eff"], s["Yaw_Eff"]]])
                           for s in states])
        variances = surrogate.variances(points)[:, :6]
        derivatives = vpp_pool.sensitivities(section, ratio, conditions, states) if sigma \
            else np.full((len(nodes), 6), np.nan)
        for node, state, d, s2 in zip(nodes, states, derivatives, variances):
            rows.append({"section": name, "mode": node["mode"], "twa": node["twa"],
                         "tws": node["tws"], "weight": node["weight"], "success": state["success"],
                         "speed_kt": state["speed"],
                         "speed_sigma_kt": float(np.sqrt(np.nansum(d ** 2 * s2)))
                         if sigma and state["success"] else np.nan,
                         "cant_eff": state["Cant_Eff"], "rake_eff": state["Rake_Eff"],
                         "yaw_eff": state["Yaw_Eff"], "flat": state["Flat"],
                         "residual": state["residual"]})
    if own_pool:
        vpp_pool.close()
    table = pd.DataFrame(rows)
    summary = summarize(table, names, profile.name)
    table.to_csv(root / "results" / f"profile_{profile.name}.csv", index=False)
    (root / "results" / f"profile_{profile.name}.json").write_text(
        json.dumps(summary, indent=1), encoding="utf-8")
    make_figure(root, table, names, profile.name)
    return summary


def expectations(rows) -> dict:
    """Expectations over the given (solved) nodes, their weights renormalized."""
    mass = float(rows["weight"].sum())
    if mass <= 0:
        return {"probability": 0.0}
    w = rows["weight"] / mass
    return {"probability": mass,
            "expected_speed_kt": float(np.sum(w * rows["speed_kt"])),
            "time_equivalent_speed_kt": float(1.0 / np.sum(w / rows["speed_kt"])),
            "speed_sigma_kt": float(np.sqrt(np.nansum(w * rows["speed_sigma_kt"] ** 2)))}


def summarize(table, names: list, profile_name: str) -> dict:
    """
    The expectations are taken on the nodes solved by EVERY section ("common" nodes): the VPP
    does not solve some nodes (e.g. deep downwind in light wind, heel at its bound 0), and
    expectations over different sets of nodes would compare different conditions. The
    probability each section solved is reported apart. Same nodes in the same order for every
    section (analyse): the rows are compared one by one.
    """
    start = names[1]
    by_section = {name: table[table["section"] == name].reset_index(drop=True) for name in names}
    common = np.logical_and.reduce([rows["success"].to_numpy() for rows in by_section.values()])
    reference = by_section[start]
    summary = {"profile": profile_name, "start_section": start,
               "probability_common": float(reference["weight"][common].sum()), "sections": {}}
    for name, rows in by_section.items():
        entry = {"probability_solved": float(rows["weight"][rows["success"]].sum()),
                 "common": expectations(rows[common]),
                 "modes": {mode: expectations(rows[common & (rows["mode"] == mode).to_numpy()])
                           for mode in rows["mode"].unique()}}
        if name != start:
            gain = rows["speed_kt"] - reference["speed_kt"]

            def mean_gain(mask, gain=gain, rows=rows):
                w = rows["weight"][mask]
                return float(np.sum(w * gain[mask]) / w.sum()) if w.sum() > 0 else None
            entry["gain_vs_start_kt"] = mean_gain(common)
            entry["gain_by_mode_kt"] = {mode: mean_gain(common & (rows["mode"] == mode).to_numpy())
                                        for mode in rows["mode"].unique()}
        summary["sections"][name] = entry
    return summary


def make_figure(root: Path, table, names: list, profile_name: str) -> None:
    import plotly.graph_objects as go  # noqa: PLC0415
    from plotly.subplots import make_subplots  # noqa: PLC0415
    sys.path.insert(0, str(VPP_DIR.parent / "benchmarks"))      # as real_run_figures.py
    from figure_style import SERIES, save, style  # noqa: PLC0415
    start = names[1]
    fig = make_subplots(rows=1, cols=2, column_widths=[0.55, 0.45], horizontal_spacing=0.12,
                        subplot_titles=("Gain du meilleur profil à chaque nœud [nd]",
                                        "Vitesse par nœud, triée par TWA"))
    best = table[table["section"] == "best"].reset_index(drop=True)
    ref = table[table["section"] == start].reset_index(drop=True)
    both = best["success"] & ref["success"]
    gain = (best["speed_kt"] - ref["speed_kt"]).where(both)
    size = 10 + 60 * best["weight"] / best["weight"].max()
    fig.add_trace(go.Scatter(x=best["twa"][both], y=best["tws"][both], mode="markers",
                             marker={"size": size[both], "color": gain[both], "colorscale": "RdBu",
                                     "cmid": 0.0, "colorbar": {"title": "gain [nd]", "x": 0.47},
                                     "line": {"width": 0.5, "color": "black"}},
                             text=[f"gain {g:+.3f} nd, poids {w:.3f}" for g, w
                                   in zip(gain[both], best["weight"][both])],
                             name="nœud résolu (taille = probabilité)"), row=1, col=1)
    fig.add_trace(go.Scatter(x=best["twa"][~both], y=best["tws"][~both], mode="markers",
                             marker={"symbol": "x", "size": 10, "color": "black"},
                             name="nœud non résolu par la VPP"), row=1, col=1)
    for i, name in enumerate(names):
        group = table[table["section"] == name].sort_values(["twa", "tws"])
        ok = group["success"]
        fig.add_trace(go.Scatter(x=np.arange(len(group))[ok.to_numpy()], y=group["speed_kt"][ok],
                                 mode="markers", name=name,
                                 error_y={"type": "data", "array": 2 * group["speed_sigma_kt"][ok]},
                                 marker={"color": SERIES[i % len(SERIES)], "size": 7}), row=1, col=2)
    fig.update_xaxes(title_text="TWA [°]", row=1, col=1)
    fig.update_yaxes(title_text="TWS [nd]", row=1, col=1)
    fig.update_xaxes(title_text="nœud (TWA croissant, puis TWS)", row=1, col=2)
    fig.update_yaxes(title_text="vitesse [nd] (barres : ± 2σ du surrogate)", row=1, col=2)
    style(fig, f"Comportement sur la distribution d'usage « {profile_name} »",
          f"{root.name} ; VPP avec le surrogate final du run", height=520, legend_rows=2)
    save(fig, root / "figures", f"profile_{profile_name}")
    from real_run_figures import write_index, write_runs_index  # noqa: PLC0415
    write_index(root)
    write_runs_index(root.parent)


def print_summary(summary: dict) -> None:
    print(f"profile {summary['profile']} (start section: {summary['start_section']}); "
          f"expectations on the nodes every section solved: "
          f"{100 * summary['probability_common']:.0f} % of the probability")
    for name, entry in summary["sections"].items():
        a = entry["common"]
        line = (f"  {name}: E[V] {a.get('expected_speed_kt', float('nan')):.3f} kt, time-equivalent "
                f"{a.get('time_equivalent_speed_kt', float('nan')):.3f} kt, sigma "
                f"{a.get('speed_sigma_kt', float('nan')):.3f} kt; solved "
                f"{100 * entry['probability_solved']:.0f} % of the probability")
        if "gain_vs_start_kt" in entry:
            line += f"; gain {entry['gain_vs_start_kt']:+.3f} kt " + str(
                {m: None if g is None else round(g, 3) for m, g in entry["gain_by_mode_kt"].items()})
        print(line)
        for mode, m in entry["modes"].items():
            print(f"      {mode}: E[V] {m.get('expected_speed_kt', float('nan')):.3f} kt on "
                  f"{100 * m['probability']:.0f} % of the probability")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="folder of a finished run")
    parser.add_argument("--profile", required=True, help="JSON of the operating profile")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--no-sigma", action="store_true", help="skip the sensitivities")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    summary = analyse(args.run, OperatingProfile.from_json(args.profile), args.workers,
                      sigma=not args.no_sigma)
    print_summary(summary)


if __name__ == "__main__":
    main()
