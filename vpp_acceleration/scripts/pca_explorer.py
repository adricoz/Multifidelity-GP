"""
Interactive explorer of a section PCA (2026-11-04, after the meeting with O. Le Maitre): one slider
per whitened score z_j, the decoded section drawn live,
    w = w_mean + sum_(j <= k) sigma_j z_j m_j        (SectionPCA.decode),
with its structural quantities: maximum thickness t_max at x_tmax, centroid y_g, Ixx (section_ixx),
v = t_max / 2 (definition of 2026-11-05, section_pca.section_modulus) and Ixx / v, compared with the
current C-foil section. The box -a <= z_j <= a of the modes used (k) is drawn on the sliders.
With --search-box (scripts/build_search_box.py, 2026-11-05): the box of the k scores of the
optimizer on the sliders, and the modes beyond k FROZEN at the scores of the current section (the
k modes through the current section, SectionPCA.anchor), as in the search.

The constraints of the section, all checked live on the decoded section, the stretches that break
them drawn in red (their parameters are inputs of the page, to choose them together):
  * no crossing surfaces, t(x) >= 0;
  * upper surface convex, y_u''(x) <= 0 on [0.01, 0.99];
  * lower surface, one of two rules (a choice of the page):
      - without inflection ("profil propre"), y_l''(x) >= 0 on [0.01, 0.99];
      - fixed sign zones, y_l'' >= 0 before x_1 - g, <= 0 between x_1 + g and x_2 - g, >= 0 after
        x_2 + g (free bands of half-width g around the inflections x_1, x_2 of the current
        section);
  * round leading edge: with the class function sqrt(x) (1 - x), a surface starts as
    y = w_0 sqrt(x), so its leading-edge radius is w_0^2 / 2; the edge is round when
    w_0,upper >= sqrt(2 r_min,upper) and w_0,lower <= -sqrt(2 r_min,lower) (lower weights < 0);
  * wiggliness (section_pca.wiggliness) <= W x that of the current section (convex quadratic);
  * Ixx / v >= ixx_v_factor x that of the current section (not convex; factor of domain.json, 0.9
    since 2026-11-05), "sans objet" when the surfaces cross.
All but the last are linear (or convex quadratic) in the weights, hence in the scores: the domain
they cut in the score box is convex. y'' is the second difference of the surfaces on the stations.

The page also draws the clean section closest to the current one (closest_clean_section: least
squares on the coordinates, upper y'' <= 0, lower y'' >= 0, no crossing, Kulfan box x3 of the
current section, the box of the generation chosen on 2026-11-04).

The page is self-contained (the PCA and the reference sections are embedded; plotly from its CDN).
The geometry is computed again in JavaScript on the same stations as section_pca
(THICKNESS_STATIONS); the page checks it at load against the values computed here in Python.

Run: .venv-mds\\Scripts\\python vpp_acceleration\\scripts\\pca_explorer.py [--pca <pca.json>]
     [--best <best_section.json>] [--out <page.html>]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import trapezoid

VPP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VPP_DIR))
# pylint: disable=wrong-import-position
from src.section_pca import (CFoilFamily, SectionPCA, THICKNESS_STATIONS,  # noqa: E402
                             notebook_bounds, section_coordinates, section_ixx, surface_matrix,
                             wiggliness)

RUN_8 = VPP_DIR / "runs" / "2026-10-29_1002_profile_v2_box3_3h"
STUDY_CONFIG = VPP_DIR.parent / "studies" / "cfoil_intens_kulfan_trim_2" / "config_cfoil_trim.json"


def structure(weights: np.ndarray, te_thickness: float) -> dict:
    """t_max, x_tmax, y_g, Ixx, v = t_max / 2, Ixx / v and wiggliness of a section."""
    x = THICKNESS_STATIONS
    coordinates = section_coordinates(weights, x, te_thickness)[0]
    upper, lower = coordinates[:len(x)], coordinates[len(x):]
    tau, camber = upper - lower, (upper + lower) / 2.0
    y_g = trapezoid(tau * camber, x) / trapezoid(tau, x)
    j = int(np.argmax(tau))
    ixx = float(section_ixx(np.atleast_2d(weights), te_thickness)[0])
    v = float(tau[j] / 2.0)
    return {"t_max": float(tau[j]), "x_tmax": float(x[j]), "y_g": float(y_g), "ixx": ixx,
            "v": v, "ixx_v": ixx / v, "wig": float(wiggliness(np.atleast_2d(weights))[0])}


def closest_clean_section(current: np.ndarray, te_thickness: float, box_scale: float = 3.0,
                          n_stations: int = 401) -> np.ndarray:
    """
    Section without inflection closest to `current`: min int (y - y_current)^2 dx subject to
    upper y'' <= 0 and lower y'' >= 0 on [0.01, 0.99], t >= 0, and the Kulfan box of width
    box_scale x the notebook box centred on `current` (all linear: a convex quadratic program).
    """
    from scipy.optimize import LinearConstraint, minimize  # noqa: PLC0415
    x = np.linspace(0.0, 1.0, n_stations)
    step = x[1] - x[0]
    matrix, offset = surface_matrix(x, len(current) // 2, te_thickness)      # y = A w + b
    second = (np.eye(n_stations, k=-1)[1:-1] - 2.0 * np.eye(n_stations)[1:-1]
              + np.eye(n_stations, k=1)[1:-1]) / step ** 2                  # (p - 2, p)
    inner = (x[1:-1] >= 0.01) & (x[1:-1] <= 0.99)
    upper, lower = slice(0, n_stations), slice(n_stations, 2 * n_stations)
    constraints = [
        LinearConstraint((second @ matrix[upper])[inner], -np.inf, -(second @ offset[upper])[inner]),
        LinearConstraint((second @ matrix[lower])[inner], -(second @ offset[lower])[inner], np.inf),
        LinearConstraint((matrix[upper] - matrix[lower])[1:],
                         -(offset[upper] - offset[lower])[1:], np.inf)]
    low, high = notebook_bounds()
    half = box_scale * (high - low) / 2.0
    result = minimize(lambda w: step * np.sum((matrix @ (w - current)) ** 2), current,
                      jac=lambda w: 2.0 * step * matrix.T @ (matrix @ (w - current)),
                      constraints=constraints, bounds=list(zip(current - half, current + half)),
                      method="trust-constr", options={"maxiter": 3000, "gtol": 1e-12,
                                                      "xtol": 1e-12})
    return result.x


def main() -> None:
    """Writes the page for a PCA, the current section and the best section of a run."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--pca", type=Path, default=RUN_8 / "data" / "pca.json")
    parser.add_argument("--best", type=Path, default=RUN_8 / "results" / "best_section.json",
                        help="best_section.json of a run, drawn as a reference")
    parser.add_argument("--out", type=Path, default=VPP_DIR / "tools" / "pca_explorer.html")
    parser.add_argument("--domain", type=Path, default=None,
                        help="domain.json of build_section_pca.py: its score box and constraints")
    parser.add_argument("--search-box", type=Path, default=None,
                        help="search_box_k<k>.json of build_search_box.py: box of the optimizer, "
                             "k modes, the others frozen at the current section")
    parser.add_argument("--search-box-name", default="C_envelope")
    args = parser.parse_args()

    pca = SectionPCA.load(args.pca)
    te = pca.te_thickness
    current = np.asarray(CFoilFamily.from_study_config(STUDY_CONFIG).reference_weights(), float)
    best = json.loads(args.best.read_text(encoding="utf-8"))
    references = {"actuelle": current,
                  f"meilleure {args.best.parents[1].name}":
                      np.asarray(best["kulfan_upper"] + best["kulfan_lower"], float),
                  "propre la plus proche": closest_clean_section(current, te)}
    search_box = None
    if args.search_box is not None:
        boxes = json.loads(args.search_box.read_text(encoding="utf-8"))
        box = boxes["boxes"][args.search_box_name]
        pca.n_modes = int(boxes["k"])
        search_box = {"name": args.search_box_name, "k": int(boxes["k"]), "lower": box["lower"],
                      "upper": box["upper"], "admissible_pct": box["uniform_admissible_pct"],
                      "frozen": pca.encode(current[None], pca.rank)[0].tolist()}
    data = {
        "pca_file": str(args.pca.relative_to(VPP_DIR)) if args.pca.is_relative_to(VPP_DIR)
        else str(args.pca),
        "te": te, "n_weights": pca.n_weights, "k": int(pca.n_modes),
        "mean_weights": pca.mean_weights.tolist(), "sigmas": pca.sigmas.tolist(),
        "weight_modes": pca.weight_modes.tolist(), "variance_ratio": pca.variance_ratio.tolist(),
        "n_stations": len(THICKNESS_STATIONS),
        "domain": None if args.domain is None
        else json.loads(args.domain.read_text(encoding="utf-8")),
        "search_box": search_box,
        # scores on ALL the modes (rank r): decoding with r modes gives the section back exactly
        "references": [{"name": name, "weights": w.tolist(),
                        "scores": pca.encode(w, pca.rank)[0].tolist(),
                        "structure": structure(w, te)} for name, w in references.items()],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(PAGE.replace("__DATA__", json.dumps(data)), encoding="utf-8")
    print(f"written {args.out}")


PAGE = r"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Explorateur PCA des sections</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  :root { --bg: #ffffff; --fg: #1d1d1f; --muted: #6e6e73; --line: #d2d2d7; --box: #e8f0fe;
          --ok: #1a7f37; --bad: #c62828; --accent: #1f6feb; }
  @media (prefers-color-scheme: dark) {
    :root { --bg: #161618; --fg: #ececec; --muted: #9a9aa0; --line: #3a3a3e; --box: #1d2a40;
            --ok: #4ac26b; --bad: #ff6b6b; --accent: #58a6ff; } }
  body { margin: 0; background: var(--bg); color: var(--fg);
         font: 14px/1.4 system-ui, -apple-system, "Segoe UI", sans-serif; }
  header { padding: 12px 16px; border-bottom: 1px solid var(--line); }
  header h1 { font-size: 18px; margin: 0 0 4px; }
  header p { margin: 0; color: var(--muted); }
  main { display: grid; grid-template-columns: 360px 1fr; gap: 16px; padding: 16px; }
  @media (max-width: 900px) { main { grid-template-columns: 1fr; } }
  .panel { border: 1px solid var(--line); border-radius: 8px; padding: 12px; }
  .row { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; margin-bottom: 8px; }
  button { background: transparent; color: var(--fg); border: 1px solid var(--line);
           border-radius: 6px; padding: 4px 8px; cursor: pointer; }
  button:hover { border-color: var(--accent); }
  input[type=number] { width: 56px; background: transparent; color: var(--fg);
                       border: 1px solid var(--line); border-radius: 4px; padding: 2px 4px; }
  .mode { display: grid; grid-template-columns: 74px 1fr 52px; gap: 6px; align-items: center;
          padding: 2px 4px; border-radius: 4px; }
  .mode.off { opacity: 0.45; }
  .mode.shown { outline: 1px solid var(--accent); }
  .mode label { cursor: pointer; font-variant-numeric: tabular-nums; }
  .mode small { color: var(--muted); display: block; font-size: 11px; }
  .track { position: relative; }
  .track .box { position: absolute; top: 50%; height: 18px; transform: translateY(-50%);
                border-left: 2px solid var(--accent); border-right: 2px solid var(--accent);
                pointer-events: none; z-index: 1; box-sizing: border-box; }
  .track input { position: relative; width: 100%; margin: 0; background: transparent; }
  .val { text-align: right; font-variant-numeric: tabular-nums; }
  .val.out { color: var(--bad); font-weight: 600; }
  table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
  td, th { padding: 3px 6px; border-bottom: 1px solid var(--line); text-align: right; }
  td:first-child, th:first-child { text-align: left; }
  .ok { color: var(--ok); } .bad { color: var(--bad); }
  #check { color: var(--muted); font-size: 12px; margin-top: 8px; }
  #plot_section { height: 380px; } #plot_thickness, #plot_curvature { height: 260px; }
</style>
</head>
<body>
<header>
  <h1>Explorateur PCA des sections du C-foil</h1>
  <p>section = section moyenne + &Sigma;<sub>j&le;k</sub> &sigma;<sub>j</sub> z<sub>j</sub> m<sub>j</sub>
     <span id="frozen_note"></span>
     &middot; v = t<sub>max</sub> / 2 &middot; PCA : <span id="pca_file"></span></p>
</header>
<main>
  <section class="panel">
    <div class="row">
      <label>k <input type="number" id="k" min="1" step="1"></label>
      <label>boîte &plusmn;a <input type="number" id="a" min="0.5" max="4" step="0.5" value="3"></label>
      <label><input type="checkbox" id="stretch" checked> y &times;3</label>
    </div>
    <div class="row" title="paramètres des contraintes, à choisir">
      <label>wiggliness &le; <input type="number" id="wmax" min="1" step="1" value="2"> &times; actuelle</label>
    </div>
    <div class="row">
      <label>intrados <select id="lowerRule">
        <option value="none" selected>sans inflexion (y'' &ge; 0)</option>
        <option value="zones">zones de signe +, &minus;, +</option></select></label>
    </div>
    <div class="row" title="rayon de bord d'attaque de chaque surface, w0² / 2 (fraction de corde)">
      <label>r<sub>BA</sub> min : extrados <input type="number" id="rUp" min="0" step="0.001"></label>
      <label>intrados <input type="number" id="rLo" min="0" step="0.0005"></label>
    </div>
    <div class="row" id="zoneRow" title="zones de signe de y'' de l'intrados : + avant x1 - g, - entre x1 + g et x2 - g, + après x2 + g">
      <label>zones : x1 <input type="number" id="x1" min="0" max="1" step="0.01"></label>
      <label>x2 <input type="number" id="x2" min="0" max="1" step="0.01"></label>
      <label>&plusmn;g <input type="number" id="gap" min="0" max="0.3" step="0.01" value="0.05"></label>
    </div>
    <div class="row" id="buttons">
      <button id="mean">moyenne</button>
      <button id="random">aléatoire dans la boîte</button>
      <button id="corner">coin aléatoire</button>
    </div>
    <div id="modes"></div>
    <p style="color: var(--muted); font-size: 12px">Cliquer sur le nom d'un mode : affiche la section
      moyenne &plusmn; a &sigma;<sub>j</sub> m<sub>j</sub> (ce que fait ce mode seul). Les modes au-delà de k
      sont grisés et ne comptent pas dans la section.</p>
  </section>
  <section>
    <div class="panel"><div id="plot_section"></div><div id="plot_thickness"></div><div id="plot_curvature"></div></div>
    <div class="panel" style="margin-top: 16px">
      <table id="metrics"></table>
      <div id="check"></div>
    </div>
  </section>
</main>
<script>
const DATA = __DATA__;
const R = DATA.sigmas.length, NW = DATA.n_weights, TE = DATA.te;
const N = DATA.n_stations, X = Array.from({length: N}, (_, i) => i / (N - 1));
const STEP = 1 / (N - 1);
let z = new Array(R).fill(0), shownMode = null;

// --- geometry (section_pca: Kulfan, N1 = 0.5, N2 = 1, Bernstein of degree NW - 1) ---
const binom = (n, i) => { let c = 1; for (let j = 1; j <= i; j++) c = c * (n - j + 1) / j; return c; };
const SHAPE = X.map(x => { const c = Math.sqrt(x) * (1 - x);
  return Array.from({length: NW}, (_, i) => c * binom(NW - 1, i) * x ** i * (1 - x) ** (NW - 1 - i)); });
// with the search box: the modes beyond k frozen at the current section (k modes through it)
const SB = DATA.search_box;
function decode(scores, k) {
  const w = DATA.mean_weights.slice();
  for (let j = 0; j < R; j++) { if (j >= k && !SB) break;
    const s = DATA.sigmas[j] * (j < k ? scores[j] : SB.frozen[j]);
    for (let i = 0; i < 2 * NW; i++) w[i] += s * DATA.weight_modes[i][j]; }
  return w;
}
function surfaces(w) {
  const up = new Float64Array(N), lo = new Float64Array(N);
  for (let p = 0; p < N; p++) { let u = 0, l = 0;
    for (let i = 0; i < NW; i++) { u += SHAPE[p][i] * w[i]; l += SHAPE[p][i] * w[NW + i]; }
    up[p] = u + X[p] * TE / 2; lo[p] = l - X[p] * TE / 2; }
  return {up, lo};
}
const trapz = f => { let s = 0; for (let p = 1; p < N; p++) s += (f[p] + f[p - 1]) / 2; return s * STEP; };
function structure(w) {
  const {up, lo} = surfaces(w);
  const tau = Array.from(up, (u, p) => u - lo[p]), cam = Array.from(up, (u, p) => (u + lo[p]) / 2);
  const yg = trapz(tau.map((t, p) => t * cam[p])) / trapz(tau);
  const ixx = trapz(tau.map((t, p) => t ** 3 / 12 + t * (cam[p] - yg) ** 2));
  let j = 0; for (let p = 1; p < N; p++) if (tau[p] > tau[j]) j = p;
  const v = tau[j] / 2;                          // 2026-11-05: v = t_max / 2
  // second differences of the surfaces (null at both ends)
  const d2 = y => Array.from(y, (_, p) => p === 0 || p === N - 1 ? null
    : (y[p + 1] - 2 * y[p] + y[p - 1]) / (STEP * STEP));
  return {up, lo, tau, cam, t_max: tau[j], x_tmax: X[j], cam_tmax: cam[j], y_g: yg, ixx, v, ixx_v: ixx / v,
          t_min: Math.min(...tau.slice(1)), d2u: d2(up), d2l: d2(lo), wig: wiggliness(w)};
}
// section_pca.wiggliness: squared second differences of the upper, then of the lower weights
function wiggliness(w) { let s = 0;
  for (const off of [0, NW]) for (let i = 0; i < NW - 2; i++) s += (w[off + i] - 2 * w[off + i + 1] + w[off + i + 2]) ** 2;
  return s; }
// required sign of y'' at x (0 = free): upper <= 0 on [0.01, 0.99]; lower +, -, + zones around x1, x2
const X_LO = 0.01, X_HI = 0.99;
function signUpper(x) { return x >= X_LO && x <= X_HI ? -1 : 0; }
function signLower(x, x1, x2, g, rule = "zones") {
  if (x < X_LO || x > X_HI) return 0;
  if (rule === "none") return 1;
  if (x <= x1 - g) return 1; if (x >= x1 + g && x <= x2 - g) return -1; if (x >= x2 + g) return 1; return 0; }
// stations where a surface breaks its sign rule (y'' ~ 0.1 to 10: 1e-4 is a numerical zero, e.g.
// a straight stretch of a section on the bound of the rule)
const breaks = (d2y, rule) => Array.from(d2y, (c, p) => c !== null && rule(X[p]) !== 0 && c * rule(X[p]) < -1e-4);
// inflections of the lower surface of the current section: the default x1, x2 of the page
function inflections(d2y) { const out = [];
  for (let p = 1; p < N - 2; p++) if (X[p] > X_LO && X[p] < X_HI && d2y[p] * d2y[p + 1] < 0) out.push(X[p]);
  return out; }

// --- controls ---
const kInput = document.getElementById("k"), aInput = document.getElementById("a");
kInput.max = R; kInput.value = DATA.k;
document.getElementById("pca_file").textContent = `${DATA.pca_file} (rang ${R}, k du run = ${DATA.k})`;
const k = () => Math.max(1, Math.min(R, parseInt(kInput.value) || DATA.k));
const a = () => parseFloat(aInput.value) || 3;
// box of mode j: the box of the optimizer (search box) for its k modes, the score box of the
// domain (domain.json) when given, otherwise +-a
const DOM = DATA.domain;
const limits = j => SB && j < SB.k ? [SB.lower[j], SB.upper[j]]
  : DOM ? [DOM.score_box[j].lower, DOM.score_box[j].upper] : [-a(), a()];
const SLIDER = Math.max(4, ...(DOM ? DOM.score_box.flatMap(b => [Math.abs(b.lower), Math.abs(b.upper)]).map(Math.ceil) : []),
  ...(SB ? [...SB.lower, ...SB.upper].map(v => Math.ceil(Math.abs(v))) : []));
if (SB) document.getElementById("frozen_note").innerHTML =
  `(modes &gt; k figés à la section actuelle : les k modes passent par elle) &middot; boîte de l'optimiseur ${SB.name}, k = ${SB.k}, ` +
  `${SB.admissible_pct.toFixed(1)} % admissible`;
const modesDiv = document.getElementById("modes");
for (let j = 0; j < R; j++) {
  modesDiv.insertAdjacentHTML("beforeend",
    `<div class="mode" id="mode${j}"><label data-j="${j}">mode ${j + 1}<small>${(100 * DATA.variance_ratio[j]).toFixed(2)} %</small></label>
     <div class="track"><div class="box"></div><input type="range" min="${-SLIDER}" max="${SLIDER}" step="0.05" value="0" data-j="${j}"></div>
     <span class="val" id="val${j}">0.00</span></div>`);
}
modesDiv.addEventListener("input", e => { if (e.target.dataset.j !== undefined) {
  z[+e.target.dataset.j] = parseFloat(e.target.value); update(); } });
modesDiv.addEventListener("click", e => { const lab = e.target.closest("label"); if (!lab) return;
  const j = +lab.dataset.j; shownMode = shownMode === j ? null : j; update(); });
function setScores(scores) { z = scores.slice(); z.forEach((s, j) => {
  modesDiv.querySelector(`input[data-j="${j}"]`).value = s; }); update(); }
document.getElementById("mean").onclick = () => setScores(new Array(R).fill(0));
document.getElementById("random").onclick = () => setScores(z.map((_, j) => { const [lo, hi] = limits(j);
  return j < k() ? lo + (hi - lo) * Math.random() : 0; }));
document.getElementById("corner").onclick = () => setScores(z.map((_, j) => j < k() ? limits(j)[Math.random() < 0.5 ? 0 : 1] : 0));
const buttons = document.getElementById("buttons");
DATA.references.forEach(ref => { const b = document.createElement("button");
  b.textContent = ref.name; b.title = "ses scores sur tous les modes (seuls les k premiers comptent)";
  b.onclick = () => setScores(ref.scores.map(s => Math.max(-SLIDER, Math.min(SLIDER, s)))); buttons.appendChild(b); });
const num = id => parseFloat(document.getElementById(id).value);
["k", "a", "stretch", "wmax", "x1", "x2", "gap", "lowerRule", "rUp", "rLo"].forEach(id =>
  document.getElementById(id).addEventListener("input", update));

// --- drawing ---
const REF_STYLE = [{color: "#8e8e93", dash: "dash"}, {color: "#e36209", dash: "dot"},
                   {color: "#2da44e", dash: "longdash"}];
const REF = DATA.references.map(r => ({...r, s: structure(r.weights)}));
{ const f = inflections(REF[0].s.d2l);           // x1, x2 default: the current section's
  document.getElementById("x1").value = (f[0] ?? 0.47).toFixed(2);
  document.getElementById("x2").value = (f[1] ?? 0.94).toFixed(2); }
// leading-edge radius of each surface: y = w0 sqrt(x) near x = 0, radius w0^2 / 2
const leRadius = w => ({up: w[0] > 0 ? w[0] ** 2 / 2 : 0, lo: w[NW] < 0 ? w[NW] ** 2 / 2 : 0});
{ const r = leRadius(REF[0].weights);           // default minima: half those of the current section
  document.getElementById("rUp").value = (r.up / 2).toPrecision(2);
  document.getElementById("rLo").value = (r.lo / 2).toPrecision(2); }
const IXX_V_FACTOR = DOM && DOM.ixx_v_factor !== undefined ? DOM.ixx_v_factor : 1;
if (SB) document.getElementById("mean").textContent = "z = 0";
if (DOM) {                                      // the settings of the domain (src/section_domain.py)
  document.getElementById("lowerRule").value = "zones";
  document.getElementById("x1").value = DOM.inflections[0].toFixed(3);
  document.getElementById("x2").value = DOM.inflections[1].toFixed(3);
  document.getElementById("gap").value = DOM.band;
  ["rUp", "rLo"].forEach(id => { document.getElementById(id).disabled = true; });
  document.getElementById("a").disabled = true;
}
const css = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
function layout(title, ytitle, equal) {
  const fg = css("--fg"), grid = css("--line");
  return {title: {text: title, font: {size: 14}}, paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    font: {color: fg}, margin: {l: 56, r: 16, t: 36, b: 40}, legend: {orientation: "h", y: -0.18},
    xaxis: {title: "x / c", gridcolor: grid, zeroline: false, range: [-0.02, 1.02]},
    yaxis: {title: ytitle, gridcolor: grid, zeroline: false,
            ...(equal ? {scaleanchor: "x", scaleratio: document.getElementById("stretch").checked ? 3 : 1} : {})}};
}
const outline = (s, name, line, extra = {}) => ({x: [...X, ...X.slice().reverse()],
  y: [...s.up, ...Array.from(s.lo).reverse()], name, mode: "lines", line, ...extra});
const fmt = (x, d = 4) => x.toFixed(d), sci = x => x.toExponential(4);

function update() {
  const kk = k(), aa = a();
  for (let j = 0; j < R; j++) {
    const [lo, hi] = limits(j), out = j < kk && (z[j] < lo - 1e-9 || z[j] > hi + 1e-9);
    const row = document.getElementById(`mode${j}`);
    row.classList.toggle("off", j >= kk); row.classList.toggle("shown", shownMode === j);
    const box = row.querySelector(".box"), pos = v => 50 + 50 * v / SLIDER;
    box.style.left = `${pos(lo)}%`; box.style.width = `${pos(hi) - pos(lo)}%`;
    const val = document.getElementById(`val${j}`); val.textContent = z[j].toFixed(2); val.classList.toggle("out", out);
  }
  const s = structure(decode(z, kk)), accent = css("--accent");
  const traces = REF.map((r, i) => outline(r.s, r.name + " (exacte)", {...REF_STYLE[i], width: 1.5}));
  if (shownMode !== null) [-1, 1].forEach(sign => { const zz = new Array(R).fill(0); zz[shownMode] = sign * aa;
    traces.push(outline(structure(decode(zz, R)), `moyenne ${sign > 0 ? "+" : "−"} ${aa} σ mode ${shownMode + 1}`,
      {color: sign > 0 ? "#2da44e" : "#cf222e", width: 1, dash: "dashdot"})); });
  traces.push(outline(s, "section (z, k modes)", {color: accent, width: 2.5}));
  traces.push({x: X, y: s.cam, name: "cambrure", mode: "lines", line: {color: accent, width: 1, dash: "dot"}});
  // crossing surfaces (t < 0): the stretch of the upper surface where it is under the lower one
  const crossed = (y) => Array.from(y, (v, p) => (p > 0 && s.tau[p] < 0 ? v : null));
  if (s.t_min < 0) traces.push({x: X, y: crossed(s.up), name: "croisement (t < 0)", mode: "lines",
                                line: {color: css("--bad"), width: 6}, connectgaps: false});
  // curvature rules: the stretches of each surface whose y'' has the wrong sign
  const x1 = num("x1"), x2 = num("x2"), g = num("gap"), wmax = num("wmax");
  const lowerRule = document.getElementById("lowerRule").value;
  document.getElementById("zoneRow").style.display = lowerRule === "zones" ? "" : "none";
  const ruleLower = x => signLower(x, x1, x2, g, lowerRule);
  const badU = breaks(s.d2u, signUpper), badL = breaks(s.d2l, ruleLower);
  const onBad = (y, bad) => Array.from(y, (v, p) => (bad[p] ? v : null));
  if (badU.some(Boolean)) traces.push({x: X, y: onBad(s.up, badU), name: "extrados : y'' > 0", mode: "lines",
                                       line: {color: "#d97706", width: 5}, connectgaps: false});
  if (badL.some(Boolean)) traces.push({x: X, y: onBad(s.lo, badL), name: "intrados : y'' hors zone", mode: "lines",
                                       line: {color: "#a21caf", width: 5}, connectgaps: false});
  traces.push({x: [0, 1], y: [s.y_g, s.y_g], name: "fibre neutre y_g", mode: "lines", line: {color: css("--muted"), width: 1}});
  traces.push({x: [s.x_tmax, s.x_tmax], y: [s.cam_tmax, s.cam_tmax + s.v], name: "v = t_max / 2", mode: "lines+markers",
               line: {color: "#bf8700", width: 3}, marker: {size: 6}});
  Plotly.react("plot_section", traces, layout("Section", "y / c", true), {responsive: true, displaylogo: false});

  const tt = REF.map((r, i) => ({x: X, y: r.s.tau, name: r.name, mode: "lines", line: {...REF_STYLE[i], width: 1.5}}));
  tt.push({x: X, y: s.tau, name: "section", mode: "lines", line: {color: accent, width: 2.5}});
  tt.push({x: [0, 1], y: [0, 0], showlegend: false, mode: "lines", line: {color: css("--muted"), width: 1}});
  if (s.t_min < 0) tt.push({x: X, y: crossed(s.tau), name: "t < 0", mode: "lines",
                            line: {color: css("--bad"), width: 6}, connectgaps: false});
  tt.push({x: [s.x_tmax], y: [s.t_max], name: "t_max", mode: "markers", marker: {color: "#bf8700", size: 8}});
  Plotly.react("plot_thickness", tt, layout("Épaisseur locale t(x)", "t / c", false), {responsive: true, displaylogo: false});

  // y'' of both surfaces, with the sign zones of the lower surface (shaded: + green, - red)
  const view = Array.from(X, x => x >= 0.03 && x <= X_HI);            // the leading edge diverges
  const cut = y => Array.from(y, (v, p) => (view[p] ? v : null));
  const cc = [{x: X, y: cut(REF[0].s.d2u), name: "extrados actuelle", mode: "lines", line: {...REF_STYLE[0], width: 1}},
              {x: X, y: cut(REF[0].s.d2l), name: "intrados actuelle", mode: "lines", line: {...REF_STYLE[0], width: 1, dash: "dot"}},
              {x: X, y: cut(s.d2u), name: "extrados", mode: "lines", line: {color: accent, width: 2}},
              {x: X, y: cut(s.d2l), name: "intrados", mode: "lines", line: {color: "#a21caf", width: 2}},
              {x: [0, 1], y: [0, 0], showlegend: false, mode: "lines", line: {color: css("--muted"), width: 1}}];
  const zone = (x0, x1_, sign) => ({type: "rect", xref: "x", yref: "paper", x0, x1: x1_, y0: 0, y1: 1, layer: "below",
    line: {width: 0}, fillcolor: sign > 0 ? "rgba(46,160,67,0.10)" : "rgba(207,34,46,0.10)"});
  const lc = layout("Dérivée seconde y''(x) — zones de signe de l'intrados (vert +, rouge −, blanc libre)", "y''", false);
  lc.shapes = lowerRule === "zones" ? [zone(X_LO, x1 - g, 1), zone(x1 + g, x2 - g, -1), zone(x2 + g, X_HI, 1)]
    : [zone(X_LO, X_HI, 1)];
  lc.title.text = "Dérivée seconde y''(x) — signe imposé à l'intrados (vert +, rouge −, blanc libre)";
  // y scale from x >= 0.1: the sign changes happen there, the leading-edge values would hide them
  const span = Math.max(...[s.d2u, s.d2l, REF[0].s.d2u, REF[0].s.d2l].flatMap(y =>
    y.filter((c, p) => c !== null && X[p] >= 0.1 && X[p] <= X_HI).map(Math.abs)));
  lc.yaxis.range = [-1.3 * span, 1.3 * span];
  Plotly.react("plot_curvature", cc, lc, {responsive: true, displaylogo: false});

  const cur = REF[0].s, inBox = z.slice(0, kk).every((v, j) => v >= limits(j)[0] - 1e-9 && v <= limits(j)[1] + 1e-9);
  const norm = Math.sqrt(z.slice(0, kk).reduce((t, v) => t + v * v, 0));
  const rows = [["t_max / c", fmt(s.t_max), fmt(cur.t_max)], ["x_tmax / c", fmt(s.x_tmax, 3), fmt(cur.x_tmax, 3)],
    ["y_g / c (fibre neutre)", fmt(s.y_g), fmt(cur.y_g)], ["v / c", fmt(s.v), fmt(cur.v)],
    ["Ixx / c⁴", sci(s.ixx), sci(cur.ixx)], ["Ixx / v / c³", sci(s.ixx_v), sci(cur.ixx_v)]];
  rows.push(["wiggliness", sci(s.wig), sci(cur.wig)]);
  const ratio = s.ixx_v / cur.ixx_v, crossing = s.t_min < 0, wigRatio = s.wig / cur.wig;
  const le = leRadius(decode(z, kk)), leCur = leRadius(REF[0].weights);
  const where = bad => { const xs = X.filter((_, p) => bad[p]); return xs.length ?
    ` (x ${xs[0].toFixed(2)}–${xs[xs.length - 1].toFixed(2)})` : ""; };
  const checks = [
    [(SB ? `dans la boîte de l'optimiseur (${SB.name})` : DOM ? "dans la boîte du domaine" : "dans la boîte ±" + aa)
     + " (k = " + kk + ")", inBox, ""],
    ["surfaces sans croisement", !crossing, ` (t_min ${fmt(s.t_min)})`],
    ["extrados convexe (y'' ≤ 0)", !badU.some(Boolean), where(badU)],
    [lowerRule === "zones" ? "intrados dans ses zones de signe" : "intrados sans inflexion (y'' ≥ 0)",
     !badL.some(Boolean), where(badL)],
    ["bord d'attaque rond" + (DOM ? " (r ≥ κ t_max²)" : ""),
     DOM ? le.up >= DOM.kappa[0] * s.t_max ** 2 * (1 - 1e-9) && le.lo >= DOM.kappa[1] * s.t_max ** 2 * (1 - 1e-9)
         : le.up >= num("rUp") && le.lo >= num("rLo"),
     ` (r extrados ${fmt(le.up)}, intrados ${fmt(le.lo)} ; actuelle ${fmt(leCur.up)}, ${fmt(leCur.lo)})`],
    // not a constraint of the domain yet (threshold to choose): shown, not counted, with a domain
    [`wiggliness ≤ ${wmax} × actuelle` + (DOM ? " (indicatif)" : ""), DOM ? null : wigRatio <= wmax,
     ` (${wigRatio.toFixed(1)} ×)`],
    [`Ixx/v ≥ ${IXX_V_FACTOR} × actuelle`, crossing ? null : ratio >= IXX_V_FACTOR * (1 - 1e-9),
     crossing ? " — sans objet : surfaces croisées" : ` (${ratio.toFixed(3)} ×)`]];
  const admissible = checks.every(([, ok]) => ok !== false) && !crossing;
  const cell = ok => ok === null ? "" : ok ? "ok" : "bad", word = ok => ok === null ? "—" : ok ? "oui" : "non";
  document.getElementById("metrics").innerHTML =
    `<tr><th></th><th>section</th><th>actuelle</th><th>rapport</th></tr>` +
    rows.map(([n, x, c]) => `<tr><td>${n}</td><td>${x}</td><td>${c}</td><td>${(parseFloat(x) / parseFloat(c)).toFixed(3)}</td></tr>`).join("") +
    checks.map(([n, ok, info]) => `<tr><td>${n}</td><td colspan="3" class="${cell(ok)}">${word(ok)}${info}</td></tr>`).join("") +
    `<tr><td><b>section admissible</b></td><td colspan="3" class="${cell(admissible)}"><b>${word(admissible)}</b></td></tr>` +
    `<tr><td>|z| (k modes)</td><td colspan="3">${norm.toFixed(2)}</td></tr>`;
}

// --- check of the JavaScript geometry against Python (computed by pca_explorer.py) ---
const worst = Math.max(...REF.map(r => Math.max(...["t_max", "y_g", "ixx", "v", "ixx_v", "wig"].map(q =>
  Math.abs(r.s[q] / r.structure[q] - 1)))));
document.getElementById("check").textContent =
  `Contrôle JavaScript / Python sur les sections de référence : écart relatif max ${worst.toExponential(1)}` +
  (worst < 1e-6 ? " (ok)" : " — ÉCART, ne pas se fier aux valeurs");
// with a domain, start from the current section (the centre of the sweep of the box)
if (DOM) setScores(DOM.current_scores); else update();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
