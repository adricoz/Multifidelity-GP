"""
Validation of a section PCA run (scripts/build_section_pca.py + scripts/build_search_box.py) before an
optimisation, and comparison with an earlier PCA run (2026-11-10: the PCA under Ixx >= 0.95 and
Ixx / v >= 0.9 x the current section, against 2026-11-05_1212 under Ixx / v >= 0.9 only).

  python scripts/validate_section_pca.py --pca-run runs/<new> --compare runs/<old> [--k 10]

Checks, each with its threshold (results/validation.json, README.md section "Validation"):
  1. the current section is in the domain and decode(encode(current)) is exact (anchored PCA);
  2. sections drawn in the search box (ScoreBoxFamily.sample: uniform in the box, kept when the
     decoded section is in the domain) all satisfy Ixx >= limit, Ixx / v >= limit and the linear
     constraints;
  3. variance kept by k modes;
  4. admissible fraction of the box (uniform);
  5. against the earlier PCA: principal angles between the two k-mode weight subspaces, and the
     error of reconstructing sections of the NEW search space in the OLD basis (anchored on the
     current section, k modes) against the new basis;
  6. the best sections of the earlier runs: Ixx, Ixx / v, in the new search space or not.
Figures (figures/validation.html): variance per mode, Ixx and Ixx / v of the box sections, box per
mode (new vs old), sections of the box with the current section and the earlier bests.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import subspace_angles

VPP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VPP_DIR))
from src.section_domain import load_search_space  # noqa: E402
from src.section_pca import THICKNESS_STATIONS, section_coordinates, section_stiffness  # noqa: E402

EARLIER = {"convexe 05/11 (FSI)": "2026-11-05_1259_convex_envelope_k10_long",
           "xtr 5 %": "2026-11-09_1057_xtr005_rigid_convex_k10",
           "xtr 3 %": "2026-11-09_1411_xtr003_rigid_convex_k10",
           "xtr 10 % depuis 5 %": "2026-11-09_1325_xtr010_from5_rigid_convex_k10",
           "run 8": "2026-10-29_1002_profile_v2_box3_3h"}
N_BOX = 4000


def best_weights(run: str):
    path = VPP_DIR / "runs" / run / "results" / "best_section.json"
    if not path.exists():
        return None
    best = json.loads(path.read_text(encoding="utf-8"))
    return np.array(best["kulfan_upper"] + best["kulfan_lower"], dtype=float)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--pca-run", required=True)
    ap.add_argument("--compare", required=True)
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()
    new_dir, old_dir, k = Path(args.pca_run), Path(args.compare), args.k
    fam, pca, z0 = load_search_space(new_dir, k)
    fam_old, pca_old, _ = load_search_space(old_dir, k)
    dom, te = fam.domain, fam.te_thickness
    cur = dom.reference
    res, checks = {"pca_run": new_dir.name, "compare": old_dir.name, "k": k}, []

    def check(name, value, ok, threshold):
        checks.append({"check": name, "value": value, "threshold": threshold, "ok": bool(ok)})

    # 1. current section
    gap = float(np.abs(pca.decode(z0[None])[0] - cur).max())
    check("section actuelle dans le domaine", bool(dom.contains(cur[None], 1e-9)[0]),
          dom.contains(cur[None], 1e-9)[0], "vrai")
    check("decode(encode(actuelle)) exact (écart max des poids)", gap, gap < 1e-10, "< 1e-10")
    # 2. sections of the search box
    box = fam.sample(N_BOX, seed=11)
    ixx, ixx_v = section_stiffness(box, te)
    viol = {n: int(v.sum()) for n, v in dom.violations(box, 1e-9).items()}
    check(f"{N_BOX} sections de la box : contraintes violées", viol, sum(viol.values()) == 0, "0 partout")
    r_ixx, r_ixxv = ixx / dom.ixx_reference, ixx_v / dom.ixx_v_reference
    res["box_sections"] = {"ixx_ratio_min": float(r_ixx.min()), "ixx_ratio_p5_p50_p95":
                           np.percentile(r_ixx, [5, 50, 95]).tolist(), "ixx_v_ratio_min": float(r_ixxv.min()),
                           "ixx_v_ratio_p5_p50_p95": np.percentile(r_ixxv, [5, 50, 95]).tolist(),
                           "binding_ixx_pct": float(100 * np.mean(r_ixx < dom.ixx_factor + 0.01)),
                           "binding_ixx_v_pct": float(100 * np.mean(r_ixxv < dom.ixx_v_factor + 0.01))}
    # 3. variance
    var = float(100 * pca.variance_ratio[:k].sum())
    check(f"variance gardée par k = {k}", var, var >= 99.8, ">= 99.8 %")
    # 4. admissible fraction of the box
    rng = np.random.default_rng(12)
    u = fam.lower + (fam.upper - fam.lower) * rng.random((100_000, k))
    frac = float(100 * fam.contains(pca.decode(u)).mean())
    u_old = fam_old.lower + (fam_old.upper - fam_old.lower) * rng.random((100_000, k))
    frac_old = float(100 * fam_old.contains(pca_old.decode(u_old)).mean())
    check("fraction admissible de la box (uniforme)", frac, frac >= 0.3, ">= 0.3 % (ancienne : "
          f"{frac_old:.2f} %)")
    # 5. against the earlier PCA
    angles = np.degrees(subspace_angles(pca.weight_modes[:, :k], pca_old.weight_modes[:, :k]))
    err_new = pca.geometric_error(box, k)
    err_old = pca_old.geometric_error(box, k)
    res["vs_old"] = {"principal_angles_deg": np.sort(angles).tolist(),
                     "recon_error_new_basis_max": float(err_new.max()),
                     "recon_error_old_basis_p50_p95_max": [float(np.median(err_old)),
                                                           float(np.percentile(err_old, 95)),
                                                           float(err_old.max())],
                     "old_box_admissible_pct": frac_old,
                     "old_sections_breaking_new_ixx_pct": float(100 * np.mean(
                         section_stiffness(fam_old.sample(N_BOX, seed=13), te)[0] < dom.ixx_min))}
    # 6. earlier bests
    res["earlier_bests"] = {}
    for label, run in EARLIER.items():
        w = best_weights(run)
        if w is None:
            continue
        a, b = section_stiffness(w[None], te)
        res["earlier_bests"][label] = {"run": run, "ixx_ratio": float(a[0] / dom.ixx_reference),
                                       "ixx_v_ratio": float(b[0] / dom.ixx_v_reference),
                                       "in_new_space": bool(fam.contains(w[None])[0]),
                                       "broken": [n for n, v in dom.violations(w[None], 1e-9).items() if v[0]]}
    res["checks"] = checks
    res["all_ok"] = all(c["ok"] for c in checks)
    (new_dir / "results").mkdir(exist_ok=True)
    (new_dir / "results" / "validation.json").write_text(json.dumps(res, indent=1, default=float),
                                                          encoding="utf-8")
    figures(new_dir, res, pca, pca_old, fam, fam_old, box, r_ixx, r_ixxv, k)
    readme(new_dir, res, dom)
    for c in checks:
        print(f"[{'OK ' if c['ok'] else 'NON'}] {c['check']}: {c['value']} ({c['threshold']})")
    print(json.dumps({key: res[key] for key in ("box_sections", "vs_old")}, indent=1, default=float))
    return 0 if res["all_ok"] else 1


def figures(new_dir, res, pca, pca_old, fam, fam_old, box, r_ixx, r_ixxv, k):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    x = THICKNESS_STATIONS
    fig = make_subplots(rows=2, cols=2, subplot_titles=(
        "Variance par mode (%)", "Ixx et Ixx/v des sections de la box (/ actuelle)",
        "Box de recherche par mode (scores)", "Sections : box (gris), actuelle, meilleures précédentes"))
    modes = np.arange(1, 17)
    fig.add_bar(x=modes, y=100 * pca.variance_ratio[:16], name="PCA nouvelle", row=1, col=1)
    fig.add_bar(x=modes, y=100 * pca_old.variance_ratio[:16], name="PCA 05/11 (1212)", row=1, col=1)
    fig.update_yaxes(type="log", row=1, col=1)
    fig.add_histogram(x=r_ixx, name="Ixx / actuelle", opacity=0.6, nbinsx=60, row=1, col=2)
    fig.add_histogram(x=r_ixxv, name="Ixx/v / actuelle", opacity=0.6, nbinsx=60, row=1, col=2)
    for j, (lo, up, name) in enumerate(((fam.lower, fam.upper, "box nouvelle"),
                                        (fam_old.lower, fam_old.upper, "box 05/11"))):
        fig.add_trace(go.Scatter(x=np.arange(1, k + 1) + 0.15 * (j - 0.5), y=(lo + up) / 2,
                                 error_y=dict(type="data", array=(up - lo) / 2), mode="markers",
                                 name=name), row=2, col=1)
    for w in box[:40]:
        c = section_coordinates(w[None], x, fam.te_thickness)[0]
        fig.add_trace(go.Scatter(x=np.r_[x, x[::-1]], y=np.r_[c[:len(x)], c[len(x):][::-1]], mode="lines",
                                 line=dict(color="rgba(120,120,120,0.25)", width=1), showlegend=False,
                                 hoverinfo="skip"), row=2, col=2)
    curves = {"actuelle": fam.domain.reference}
    curves.update({lab: w for lab, run in EARLIER.items() if (w := best_weights(run)) is not None})
    for lab, w in curves.items():
        c = section_coordinates(w[None], x, fam.te_thickness)[0]
        fig.add_trace(go.Scatter(x=np.r_[x, x[::-1]], y=np.r_[c[:len(x)], c[len(x):][::-1]], mode="lines",
                                 line=dict(width=2.5 if lab == "actuelle" else 1.5), name=lab), row=2, col=2)
    fig.update_layout(height=900, template="plotly_white", barmode="overlay",
                      title=f"Validation de la PCA {res['pca_run']} (k = {k})")
    rows = "".join(f"<tr><td>{'OK' if c['ok'] else '<b>NON</b>'}</td><td>{c['check']}</td><td>{c['value']}</td>"
                   f"<td>{c['threshold']}</td></tr>" for c in res["checks"])
    best = "".join(f"<tr><td>{lab}</td><td>{v['ixx_ratio']:.3f}</td><td>{v['ixx_v_ratio']:.3f}</td>"
                   f"<td>{v['in_new_space']}</td><td>{', '.join(v['broken'])}</td></tr>"
                   for lab, v in res["earlier_bests"].items())
    html = ("<html><head><meta charset='utf-8'><title>Validation PCA</title></head>"
            "<body style='font-family:sans-serif'>" + fig.to_html(full_html=False, include_plotlyjs="cdn")
            + "<h3>Contrôles</h3><table border=1 cellpadding=4 style='border-collapse:collapse'>"
            "<tr><th></th><th>contrôle</th><th>valeur</th><th>seuil</th></tr>" + rows + "</table>"
            "<h3>Meilleures sections précédentes</h3><table border=1 cellpadding=4 "
            "style='border-collapse:collapse'><tr><th>run</th><th>Ixx / actuelle</th><th>Ixx/v / actuelle</th>"
            "<th>dans le nouvel espace</th><th>contraintes violées</th></tr>" + best + "</table>"
            f"<pre>{json.dumps(res['vs_old'], indent=1, default=float)}</pre></body></html>")
    (new_dir / "figures").mkdir(exist_ok=True)
    (new_dir / "figures" / "validation.html").write_text(html, encoding="utf-8")


def readme(new_dir, res, dom):
    path = new_dir / "README.md"
    text = path.read_text(encoding="utf-8") if path.exists() else f"# {new_dir.name}\n"
    text = text.split("\n## Validation")[0].rstrip() + "\n"
    v, b = res["vs_old"], res["box_sections"]
    lines = ["", "## Validation (scripts/validate_section_pca.py)", "",
             f"Domaine : Ixx >= {dom.ixx_factor:g} x actuelle ({dom.ixx_min:.4e}) et Ixx/v >= {dom.ixx_v_factor:g}"
             f" x actuelle ({dom.ixx_v_min:.4e}), plus les contraintes linéaires du 04/11. Résultat global : "
             f"**{'tous les contrôles OK' if res['all_ok'] else 'AU MOINS UN CONTRÔLE NON OK'}**.", "",
             "| | contrôle | valeur | seuil |", "|---|---|---|---|"]
    lines += [f"| {'OK' if c['ok'] else '**NON**'} | {c['check']} | {c['value']} | {c['threshold']} |"
              for c in res["checks"]]
    lines += ["", f"Sections de la box : Ixx / actuelle p5/p50/p95 {np.round(b['ixx_ratio_p5_p50_p95'], 3).tolist()},"
              f" Ixx/v / actuelle {np.round(b['ixx_v_ratio_p5_p50_p95'], 3).tolist()}; à moins de 1 % de la limite : "
              f"Ixx {b['binding_ixx_pct']:.1f} %, Ixx/v {b['binding_ixx_v_pct']:.1f} %.",
              f"Contre la PCA {res['compare']} : angles principaux entre les sous-espaces à {res['k']} modes "
              f"{np.round(v['principal_angles_deg'], 1).tolist()} deg ; erreur de reconstruction des sections de la "
              f"nouvelle box dans l'ancienne base p50/p95/max {np.round(v['recon_error_old_basis_p50_p95_max'], 5).tolist()}"
              f" (corde 1 ; nouvelle base : {v['recon_error_new_basis_max']:.1e}) ; "
              f"{v['old_sections_breaking_new_ixx_pct']:.0f} % des sections de l'ancienne box violent Ixx >= "
              f"{dom.ixx_factor:g}.", "", "| meilleure section | Ixx / actuelle | Ixx/v / actuelle | dans le nouvel "
              "espace |", "|---|---|---|---|"]
    lines += [f"| {lab} ({e['run']}) | {e['ixx_ratio']:.3f} | {e['ixx_v_ratio']:.3f} | {e['in_new_space']} |"
              for lab, e in res["earlier_bests"].items()]
    lines += ["", "Figures : figures/validation.html (variance, Ixx et Ixx/v, box par mode, sections), "
              "figures/pca_explorer.html (explorateur interactif)."]
    path.write_text(text + "\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
