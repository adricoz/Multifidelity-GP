"""
STEP 0 - Check the set-up before spending any computation time.

What this script does, in order (each check is PASS / FAIL in results/step0_checks.json):

1. Environment: Python and package versions (NeuralFoil 0.2.3 and 0.3.x do not give the same
   polars, so the versions are stored with every result), bdFoil clone, XFOIL / AVL executables.
2. bdFoil conventions of the 3D geometry (the foil must be generated exactly as bdFoil does):
   a. our circular arc (pipelines/bdtoolbox_foil/planform.py) against bdFoil's own
      CFoil_Arc_T1.arc_geometry / arc_characteristics (imported when possible);
   b. reference values of bdFoil core (Sref = chord x arc length, Cref = chord, Bref = arc
      length for a FOIL);
   c. positioned geometry (core.position): root at the lower bearing, foil below the wall,
      incidence Ainc of the root close to the yaw (core.incidence sign convention);
   d. the AVL geometry file written by bdFoil core, read back and compared to the stations.
3. Section geometry: analytical CST thickness against aerosandbox, exact degree elevation 4 -> 8
   weights (NeuralFoil input), fit of the current section (baseline), feasible fraction of the
   design box (thickness constraint).
4. 2D constraint: projection of the baseline (camber offset delta such that
   CL2d(0 deg) = 0.45 with NeuralFoil xxlarge) and of a few random designs.
5. Solvers: the baseline section evaluated at the three levels, with the spanwise loading and
   the CL2d(0) seen by each level at each section.

Figures: figures/step0_geometry_3d.html, figures/step0_baseline_section.html.

Run (conda environment bdToolbox, from any folder):
    C:\\Users\\SIM\\.conda\\envs\\bdToolbox\\python.exe studies\\cfoil_intens_kulfan\\step0_check_setup.py
"""
import logging
import time

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import plotkit as pk
import study_lib as lib
from pipelines.bdtoolbox_foil import bridge, geometry  # pylint: disable=wrong-import-order
from pipelines.bdtoolbox_foil.constraints import feasibility_function
from pipelines.bdtoolbox_foil.planform import CFoilArc, attitude_angles, write_planform_csv

logger = logging.getLogger("step0")
CHECKS = {}


def check(name: str, ok: bool, detail) -> None:
    """Records and logs one PASS / FAIL check."""
    CHECKS[name] = {"pass": bool(ok), "detail": detail}
    logger.info("[%s] %s: %s", "PASS" if ok else "FAIL", name, detail)


# 1/5 ---------------------------------------------------------------------------------------------
def check_environment(simulator) -> dict:
    report = bridge.environment_report(simulator.paths)
    check("executables", report["xfoil_exe"] and report["avl_exe"],
          {k: report[k] for k in ("soft_dir", "xfoil_exe", "avl_exe")})
    check("neuralfoil installed", report["neuralfoil"] is not None,
          f"NeuralFoil {report['neuralfoil']}, aerosandbox {report['aerosandbox']}, "
          f"numba {report['numba']}, Python {report['python']}")
    return report


# 2/5 ---------------------------------------------------------------------------------------------
def check_arc_against_bdfoil(arc: CFoilArc) -> None:
    """Our arc vs bdFoil CFoil_Arc_T1 (needs the legacy foil.py imports; skipped otherwise)."""
    _, x, y, z = arc.arc_geometry()
    try:
        import sys  # noqa: PLC0415
        code_dir = lib.REPO.parent / "bdFoil" / "Code"
        if str(code_dir) not in sys.path:
            sys.path.append(str(code_dir))
        from foil_family_intensSY import CFoil_Arc_T1  # noqa: PLC0415
        family = CFoil_Arc_T1(np.array([[5.0, 12.0], [2.5, 4.0], [0.5, 0.9]]), "section.xf", 0.0,
                              ori_foil_y=arc.ori_y, ori_foil_z=arc.ori_z,
                              cant_geom=arc.cant_geom, tip_side=arc.tip_side, taper=arc.taper,
                              root_twist=arc.root_twist, tip_twist=arc.tip_twist,
                              project="IntensSY")
        _, xb, yb, zb = family.arc_geometry(arc.radius, arc.span, n=arc.n_stations)
        error = float(np.max(np.abs(np.column_stack((x - xb, y - yb, z - zb)))))
        ours, theirs = arc.characteristics(), family.arc_characteristics(
            arc.radius, arc.span, arc.chord)
        char_error = max(abs(ours["area"] - theirs["surface"]),
                         abs(ours["depth"] - theirs["depth"]),
                         abs(ours["lateral_offset"] - theirs["lateral_offset"]))
        check("arc vs bdFoil CFoil_Arc_T1", error < 1e-12 and char_error < 1e-12,
              f"max station distance {error:.2e} m, characteristics {char_error:.2e}")
    except Exception as error:  # noqa: BLE001  (legacy foil.py dependencies may be missing)
        theta = np.linspace(0.0, arc.span / arc.radius, arc.n_stations)
        y_ref = arc.tip_side * arc.radius * (1.0 - np.cos(theta))
        z_ref = -arc.radius * np.sin(theta)
        ok = arc.cant_geom == 0.0 and np.allclose(y, y_ref) and np.allclose(z, z_ref)
        check("arc vs closed form (bdFoil import skipped)", ok,
              f"CFoil_Arc_T1 not importable ({type(error).__name__}: {error}); closed form used")


def check_core_geometry(simulator, problem, workdir) -> dict:
    """Reference values, positioned stations and the AVL file written by bdFoil core."""
    avl = simulator.backends[2]
    core = avl.core
    arc = CFoilArc.from_config(problem.planform)
    csv = workdir / "planform.csv"
    planform = core.model.load_planform(csv, name="cfoil", kind=arc.kind, section_dirs=[workdir])
    refs = core.geometry.reference_values(planform)
    area, chord = arc.characteristics()["area"], arc.chord
    check("reference values (Sref, Cref, Bref)",
          abs(refs.sref - area) < 2e-3 * area and abs(refs.cref - chord) < 1e-9
          and abs(refs.bref - arc.span) < 1e-3 * arc.span,
          f"Sref {refs.sref:.5f} m2 (c L = {area:.5f}), Cref {refs.cref:.4f} m, "
          f"Bref {refs.bref:.5f} m (arc length {arc.span})")
    stations = core.geometry.position(planform, avl.attitude, avl.settings)
    att = attitude_angles(problem.attitude)
    check("positioned geometry (lower bearing at the origin, foil below z = 0)",
          abs(stations.x[0]) < 1e-9 and abs(stations.y[0]) < 1e-9 and abs(stations.z[0]) < 1e-9
          and np.all(np.asarray(stations.z) <= 1e-12),
          f"root ({stations.x[0]:.2e}, {stations.y[0]:.2e}, {stations.z[0]:.2e}), "
          f"depth {-np.min(stations.z):.4f} m, {len(stations.x)} resampled stations")
    ainc = np.asarray(stations.ainc)
    check("incidence sign (vertical root: Ainc ~ +yaw)", abs(ainc[0] - att["yaw"]) < 0.5,
          f"Ainc root {ainc[0]:.3f} deg, tip {ainc[-1]:.3f} deg for yaw {att['yaw']} deg, "
          f"cant {att['cant']} deg, rake {att['rake']} deg")
    # AVL file written by the core, read back
    polars = core.xfoil.get_polars(planform, avl.spec, cache_dir=avl.cache_dir, nproc=1)
    out = lib.RESULTS / "step0_avl"
    result = avl.campaign.run_case(planform, polars, avl.attitude, avl.settings, workdir=out,
                                   timeout=120)
    lines = (out / "c00000" / "g.avl").read_text().splitlines()
    rows = [[float(v) for v in lines[i + 1].split()[:5]] for i, line in enumerate(lines)
            if line.strip() == "SECTION"]
    header = lines[lines.index("#IYsym IZsym Zsym") + 1].split()
    rows = np.array(rows)
    expected = np.column_stack((-np.asarray(stations.x), -np.asarray(stations.y),
                                np.asarray(stations.z), np.asarray(stations.c), ainc))
    keep = np.asarray(stations.z) <= 0
    error = float(np.max(np.abs(rows - expected[keep])))
    check("AVL file read back (Xle Yle Zle Chord Ainc, AVL axes)",
          error < 1e-4 and int(header[1]) == avl.settings.zsym,
          f"{len(rows)} sections, max deviation {error:.1e} (4 decimals written), "
          f"IZsym {header[1]} ({problem.image})")
    return {"stations": {k: np.asarray(getattr(stations, k)).tolist()
                         for k in ("x", "y", "z", "c", "ainc")},
            "refs": {"sref": refs.sref, "cref": refs.cref, "bref": refs.bref},
            "avl_case_ok": bool(result.ok)}


# 3/5 ---------------------------------------------------------------------------------------------
def check_section_geometry(problem, baseline) -> dict:
    from aerosandbox.geometry.airfoil.airfoil_families import (  # noqa: PLC0415
        get_kulfan_coordinates)
    rng = np.random.default_rng(0)
    worst_thickness, worst_elevation = 0.0, 0.0
    for _ in range(1000):
        params = geometry.section_parameters(problem, rng.random(problem.dim),
                                             {"delta": rng.uniform(-0.1, 0.2)})
        kulfan = geometry.kulfan_parameters(params, "thickness_camber")
        coords = get_kulfan_coordinates(**kulfan, n_points_per_side=120)
        x = coords[:120, 0][::-1]
        thickness = coords[:120, 1][::-1] - coords[119:, 1]
        analytic = geometry.cst_thickness(x, kulfan["upper_weights"] - kulfan["lower_weights"],
                                          kulfan["TE_thickness"])[0]
        worst_thickness = max(worst_thickness, float(np.max(np.abs(thickness - analytic))))
        elevated = get_kulfan_coordinates(**geometry.neuralfoil_kulfan(kulfan),
                                          n_points_per_side=120)
        worst_elevation = max(worst_elevation, float(np.max(np.abs(elevated - coords))))
    check("CST thickness vs aerosandbox (1000 designs)", worst_thickness < 1e-10,
          f"max deviation {worst_thickness:.1e}")
    check("degree elevation 4 -> 8 weights is exact", worst_elevation < 1e-10,
          f"max coordinate deviation {worst_elevation:.1e}")
    check("baseline fit of the current section", baseline["fit_thickness_error"] < 0.01,
          f"{lib.BASELINE_XF.name}: max t/c {baseline['max_thickness_real']:.4f}, fit error "
          f"{baseline['fit_thickness_error']:.4f} c (4 weights), scale {baseline['scale']:.4f}")
    check("equal-lift reference of the configuration = baseline fit",
          baseline["reference_vs_fit"] < 1e-5,
          f"max difference {baseline['reference_vs_fit']:.1e} (rounded values in the config)")
    points = rng.random((20000, problem.dim))
    fraction = float(np.mean(feasibility_function(problem)(points)))
    check("feasible fraction of the design box", fraction > 0.05,
          f"{100 * fraction:.1f} % of random designs meet {problem.constraints}")
    return {"feasible_fraction": fraction}


# 4/5 ---------------------------------------------------------------------------------------------
def check_projection(simulator, problem, baseline) -> dict:
    x0 = np.asarray(baseline["x"])
    simulator.projector.cache.clear()
    start = time.perf_counter()
    result = simulator.projector.solve(x0)
    target = problem.section_constraint["cl2d_target"]
    check("CL2d projection of the baseline", abs(result.cl2d - target) < 1e-4,
          f"delta {result.delta:.5f}, CL2d {result.cl2d:.6f} (target {target}, "
          f"{problem.section_constraint.get('reference_model', 'xxlarge')}), "
          f"{result.n_calls} NeuralFoil calls, {time.perf_counter() - start:.2f} s")
    deltas, times = [], []
    for x in lib.feasible_design(problem, 20, seed=7):
        start = time.perf_counter()
        deltas.append(simulator.projector.solve(x).delta)
        times.append(time.perf_counter() - start)
    check("CL2d projection of 20 random designs", True,
          f"delta in [{min(deltas):.4f}, {max(deltas):.4f}], {np.mean(times):.3f} s each")
    return {"baseline_delta": result.delta, "baseline_cl2d": result.cl2d,
            "random_deltas": deltas, "projection_time_s": float(np.mean(times))}


# 5/5 ---------------------------------------------------------------------------------------------
def check_solvers(simulator, problem, baseline) -> dict:
    x0 = np.asarray(baseline["x"])
    out = {}

    def fmt(metrics, key, digits):
        value = metrics.get(key)
        return "n/a" if value is None else f"{value:.{digits}f}"

    for level in (1, 2, 3):
        start = time.perf_counter()
        value, metrics, strips = simulator.evaluate_details(x0, level, with_strips=True)
        seconds = time.perf_counter() - start
        out[level] = {"value": value, "metrics": metrics, "strips": strips, "time_s": seconds}
        check(f"baseline at {lib.level_label(problem, level)}", np.isfinite(value),
              f"objective {value:.5f} (CD {fmt(metrics, 'CD', 5)}: Cdi {fmt(metrics, 'Cdi', 5)}"
              f", profile {fmt(metrics, 'Cdprofile', 5)}), CL {fmt(metrics, 'CL', 4)} (Cy "
              f"{fmt(metrics, 'Cy', 4)}, Cz {fmt(metrics, 'Cz', 4)}), CL2d(0) seen "
              f"{fmt(metrics, 'cl2d_check_min', 4)}, {seconds:.2f} s"
              if np.isfinite(value) else metrics.get("error"))
    return out


# ------------------------------------------------------------------ figures
def figure_geometry(problem, core_geometry, sections: list, name: str = "step0_geometry_3d",
                    title: str = "Positioned C-foil (bdFoil frame, lower bearing at the origin)"
                    ) -> None:
    """
    3D view of the positioned C-foil with the wall (symmetry plane z = 0).
    sections: list of (geometry.Section, legend name, color) drawn at 5 stations.
    """
    from bdFoil.Code.core.geometry import rotation_matrix  # noqa: PLC0415
    att = attitude_angles(problem.attitude)
    rotation = np.asarray(rotation_matrix(att["cant"], att["rake"], att["yaw"]), dtype=float)
    st = {k: np.asarray(v) for k, v in core_geometry["stations"].items()}
    le = np.column_stack((st["x"], st["y"], st["z"]))
    chord_dir = rotation @ np.array([-1.0, 0.0, 0.0])
    te = le + st["c"][:, None] * chord_dir
    # section shapes drawn at a few stations (chord along -x, thickness along the local normal)
    fig = go.Figure()
    fig.add_trace(go.Surface(x=np.vstack((le[:, 0], te[:, 0])), y=np.vstack((le[:, 1], te[:, 1])),
                             z=np.vstack((le[:, 2], te[:, 2])), showscale=False, opacity=0.85,
                             colorscale=[[0, pk.LEVEL_COLOR[1]], [1, pk.LEVEL_COLOR[1]]],
                             name="positioned foil", hoverinfo="skip"))
    fig.add_trace(go.Scatter3d(x=le[:, 0], y=le[:, 1], z=le[:, 2], mode="lines",
                               line={"color": pk.PRIMARY, "width": 4}, name="leading edge",
                               hovertemplate="LE x %{x:.3f} y %{y:.3f} z %{z:.3f} m"))
    for section, label, color in sections:
        coords = section.coordinates
        for k, i in enumerate(np.linspace(0, len(le) - 1, 5).astype(int)):
            tangent = le[min(i + 1, len(le) - 1)] - le[max(i - 1, 0)]
            normal = np.cross(chord_dir, tangent)
            normal /= np.linalg.norm(normal)
            pts = le[i] + st["c"][i] * (np.outer(coords[:, 0], chord_dir)
                                        + np.outer(coords[:, 1], normal))
            fig.add_trace(go.Scatter3d(x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode="lines",
                                       line={"color": color, "width": 3},
                                       showlegend=k == 0, legendgroup=label, name=label,
                                       hoverinfo="skip"))
    x_plane = [float(np.min(te[:, 0])) - 0.5, float(np.max(le[:, 0])) + 0.5]
    y_plane = [float(np.min(le[:, 1])) - 0.5, float(np.max(le[:, 1])) + 0.5]
    fig.add_trace(go.Surface(x=np.array([[x_plane[0], x_plane[0]], [x_plane[1], x_plane[1]]]),
                             y=np.array([y_plane, y_plane]),
                             z=np.zeros((2, 2)), opacity=0.25, showscale=False,
                             colorscale=[[0, pk.MUTED], [1, pk.MUTED]], name="wall z = 0",
                             hoverinfo="skip"))
    fig.update_layout(scene={"aspectmode": "data", "xaxis_title": "x forward (m)",
                             "yaxis_title": "y to port (m)", "zaxis_title": "z up (m)"})
    pk.style(fig, title,
             f"R {problem.planform['radius']} m, arc {problem.planform['span']} m, chord "
             f"{problem.planform['chord']} m; cant {att['cant']:.2f} deg (heel + trunk), rake "
             f"{att['rake']} deg, yaw {att['yaw']} deg; gray plane: wall image (hull)", 700)
    pk.save(fig, name)


def figure_baseline_section(problem, baseline, simulator) -> None:
    """Current section, its 4+4 Kulfan fit and the projected baseline (CL2d(0) = 0.45)."""
    x0 = np.asarray(baseline["x"])
    projected = simulator.section_of(x0)
    fitted = geometry.build_section(problem, x0, {"delta": 0.0})
    real = np.asarray(baseline["real_coordinates"])
    fig = make_subplots(rows=2, cols=1, row_heights=[0.55, 0.45], vertical_spacing=0.12,
                        subplot_titles=("Section shapes (y / c vs x / c)",
                                        "Thickness and camber distributions"))
    for coords, name, color, dash in ((real, f"current section {lib.BASELINE_XF.stem}",
                                       pk.MUTED, "solid"),
                                      (fitted.coordinates, "Kulfan 4+4 fit (scaled, no camber)",
                                       pk.SECONDARY, "dot"),
                                      (projected.coordinates,
                                       "baseline = fit + camber offset (CL2d(0) = 0.45)",
                                       pk.PRIMARY, "solid")):
        fig.add_trace(go.Scatter(x=coords[:, 0], y=coords[:, 1], mode="lines", name=name,
                                 line={"color": color, "width": 2, "dash": dash},
                                 hovertemplate="x/c %{x:.4f}<br>y/c %{y:.4f}"), row=1, col=1)
    xs = geometry.THICKNESS_STATIONS
    for coords, name, color in ((real, "current section", pk.MUTED),
                                (projected.coordinates, "baseline", pk.PRIMARY)):
        up, lo = geometry.surfaces_at(coords, xs)
        fig.add_trace(go.Scatter(x=xs, y=up - lo, mode="lines", name=f"thickness, {name}",
                                 line={"color": color, "width": 2}, showlegend=False,
                                 hovertemplate="x/c %{x:.3f}<br>t/c %{y:.4f}"), row=2, col=1)
        fig.add_trace(go.Scatter(x=xs, y=0.5 * (up + lo), mode="lines",
                                 name=f"camber, {name}", line={"color": color, "width": 2,
                                                                "dash": "dash"},
                                 showlegend=False,
                                 hovertemplate="x/c %{x:.3f}<br>camber %{y:.4f}"), row=2, col=1)
    fig.update_yaxes(scaleanchor="x", scaleratio=1, row=1, col=1)
    pk.style(fig, "Baseline section",
             f"Current section {lib.BASELINE_XF.stem} (t/c {baseline['max_thickness_real']:.4f},"
             f" symmetric); its 4+4 Kulfan fit; the baseline adds the camber offset delta = "
             f"{simulator.project(x0)['delta']:.4f} solved for CL2d(0 deg) = "
             f"{problem.section_constraint['cl2d_target']} (NeuralFoil "
             f"{problem.section_constraint.get('reference_model', 'xxlarge')})", 760)
    fig.update_xaxes(title_text="x / c", row=2, col=1)
    fig.update_yaxes(title_text="t / c (solid), camber / c (dashed)", row=2, col=1)
    pk.save(fig, "step0_baseline_section")


def main() -> int:
    lib.setup_logging("step0_check_setup")
    problem = lib.load_problem()
    simulator = lib.make_simulator(problem)
    baseline = lib.baseline_parameters(problem)
    environment = check_environment(simulator)
    arc = CFoilArc.from_config(problem.planform)
    check_arc_against_bdfoil(arc)
    # baseline section files kept for inspection (planform CSV + section .xf)
    workdir = lib.RESULTS / "step0_files"
    section = simulator.section_of(np.asarray(baseline["x"]))
    xf = geometry.write_xf(section, workdir)
    write_planform_csv(arc.stations(), xf.name, workdir / "planform.csv")
    core_geometry = check_core_geometry(simulator, problem, workdir)
    geometry_info = check_section_geometry(problem, baseline)
    projection = check_projection(simulator, problem, baseline)
    solvers = check_solvers(simulator, problem, baseline)
    figure_geometry(problem, core_geometry, [(section, "baseline section", pk.SECONDARY)])
    figure_baseline_section(problem, baseline, simulator)
    n_fail = sum(not c["pass"] for c in CHECKS.values())
    lib.save_json(lib.RESULTS / "step0_checks.json",
                  {"checks": CHECKS, "environment": environment, "baseline": {
                      k: v for k, v in baseline.items() if k != "real_coordinates"},
                   "core_geometry": core_geometry, "section_geometry": geometry_info,
                   "projection": projection, "solvers": solvers})
    logger.info("%d checks, %d failed. Results: %s", len(CHECKS), n_fail,
                lib.RESULTS / "step0_checks.json")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
