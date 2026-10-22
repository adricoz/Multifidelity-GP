"""
Step 0 of the real test (go / no go): do our codes see the same foil, in the same conventions, as
the production RS of the MDS VPP?

The production section is run through src/foil_codes.py at both levels
(NPLLT x NeuralFoil, AVL x XFOIL, production planform) on 18 attitudes (Cant_Eff 0, 4, 8 deg x
Yaw_Eff -3, 0, 3 deg x Rake_Eff 0, 1 deg), and compared component by component with the C-foil
RS of MDS (CFoil_R9800_S3250_C700_fsi_rs_vpp.json, the R 9.8 m foil of the boat; built with
AVL x XFOIL plus a beam model: at 7 kt its tip deflection is 25 mm, so it is almost rigid there;
also shown at 12 and 20 kt to see the effect of the deformation).
Printed: for each component, the straight line RS = a code + b fitted over the 18 attitudes, its
R2, and the values at the zero attitude. Go: a close to 1 (or to a clear constant factor, e.g.
the chord for the moments), b close to 0, R2 near 1, for the AVL level.
Section (first try, 2026-10-22): MC2_60_6101_1.xf of bdFoil/Scripts/Intens/Dagg is SYMMETRIC (no
lift at zero attitude) while the production RS has Fy = -0.29 at zero attitude (zero lift near
Yaw_Eff = -5 deg): the C-foil section is cambered. Second try: the reference section of the
C-foil studies (MC2 thickness + camber delta 0.108139, CL2d(0) = 0.45), --section reference.

Run (MDS environment, one process, about 2 min):
    ..\\.venv-mds\\Scripts\\python scripts\\real_step0_calibration.py
Outputs: runs/<date>_step0_calibration/ (config, log, data/evaluations.csv, results/)
"""
import itertools
import sys
from pathlib import Path

import numpy as np

VPP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VPP_DIR))
sys.path.insert(1, str(VPP_DIR.parent / "mfego"))
# pylint: disable=wrong-import-position
from src.foil_codes import BDFOIL_NAMES, OUTPUT_NAMES, PLANFORM, FoilCodes, read_xf_section  # noqa
from src.meta_foil import ChebyshevRS  # noqa: E402
from src.run_dir import RunDir  # noqa: E402

MDS_FOIL = VPP_DIR.parents[1] / "MDS" / "Systems" / "Ships" / "Sail" / "IntensSY" / "Data" / "Foil"
PRODUCTION_RS = MDS_FOIL / "CFoil_R9800_S3250_C700_fsi_rs_vpp.json"
PRODUCTION_SECTION = VPP_DIR.parents[1] / "bdFoil" / "Scripts" / "Intens" / "Dagg" / "MC2_60_6101_1.xf"
CANTS, YAWS, RAKES = (0.0, 4.0, 8.0), (-3.0, 0.0, 3.0), (0.0, 1.0)  # (0, 0, 0) included
SPEEDS = (7.0, 12.0, 20.0)


def reference_section():
    """Reference section of the C-foil studies, in the 8-weight Kulfan format."""
    from src.section_pca import CFoilFamily  # noqa: PLC0415
    family = CFoilFamily.from_study_config(VPP_DIR.parent / "studies" / "cfoil_intens_kulfan_trim_2"
                                           / "config_cfoil_trim.json")
    return FoilCodes.section(family.reference_weights(), family.te_thickness)


def main():
    import argparse  # noqa: PLC0415
    parser = argparse.ArgumentParser()
    parser.add_argument("--section", choices=("mc2", "reference"), default="reference")
    args = parser.parse_args()
    attitudes = [(c, r, y) for c, y, r in itertools.product(CANTS, YAWS, RAKES)]
    run = RunDir.create(f"step0_calibration_{args.section}", {
        "purpose": "conventions of the codes against the production RS", "planform": PLANFORM,
        "section": str(PRODUCTION_SECTION) if args.section == "mc2" else
        "reference of the C-foil studies (MC2 thickness + camber delta 0.108139)",
        "production_rs": str(PRODUCTION_RS),
        "attitudes_cant_rake_yaw": attitudes, "speeds_kt": SPEEDS})
    codes = FoilCodes(run.path("data"))
    section = read_xf_section(PRODUCTION_SECTION) if args.section == "mc2"         else reference_section()
    results = {1: [], 2: []}
    for level in (1, 2):
        for attitude in attitudes:
            y, info = codes.evaluate(None, attitude, level, section=section)
            results[level].append(y)
            run.add_evaluation({"level": level, "cant": attitude[0], "rake": attitude[1],
                                "yaw": attitude[2], **dict(zip(OUTPUT_NAMES, y)),
                                **{k: info.get(k, np.nan) for k in BDFOIL_NAMES},
                                "time_s": info["time_s"], "error": info["error"]})
            if info["error"]:
                run.log(f"level {level} {attitude}: FAILED {info['error']}")
        run.log(f"level {level}: {len(attitudes)} runs, "
                f"{sum(np.isfinite(r[0]) for r in results[level])} ok")
    rs = ChebyshevRS(PRODUCTION_RS)
    lines = []
    for speed in SPEEDS:
        reference = rs(np.array([[a[0], a[1], a[2], speed] for a in attitudes]))
        for level in (1, 2):
            ours = np.array(results[level])
            lines.append(f"--- level {level} against the production RS at {speed:.0f} kt")
            for k, name in enumerate(OUTPUT_NAMES):
                ok = np.isfinite(ours[:, k]) & np.isfinite(reference[:, k])
                if ok.sum() < 3:
                    lines.append(f"{name:6s}: not enough points")
                    continue
                slope, intercept = np.polyfit(ours[ok, k], reference[ok, k], 1)
                fitted = slope * ours[ok, k] + intercept
                r2 = 1.0 - np.sum((reference[ok, k] - fitted) ** 2)                     / np.sum((reference[ok, k] - reference[ok, k].mean()) ** 2)
                zero = [i for i, a in enumerate(attitudes) if a == (0.0, 0.0, 0.0)]
                at_zero = (f" | at (0, 0, 0): RS {reference[zero[0], k]:+.4f}, code "
                           f"{ours[zero[0], k]:+.4f}") if zero else ""
                lines.append(f"{name:6s}: RS = {slope:+.4f} code {intercept:+.5f}, R2 {r2:.4f}"
                             f" ({ok.sum()} points){at_zero}")
    text = "\n".join(lines)
    (run.path("results", "calibration.txt")).write_text(text + "\n", encoding="utf-8")
    np.savez(run.path("results", "calibration.npz"), attitudes=np.array(attitudes),
             level1=np.array(results[1]), level2=np.array(results[2]))
    print(text)
    run.log("done")


if __name__ == "__main__":
    main()
