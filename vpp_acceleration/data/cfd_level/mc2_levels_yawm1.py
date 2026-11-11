"""Levels 1 (NPLLT x NeuralFoil) and 2 (AVL x XFOIL, xtr 3 %) at the first CFD point: the current section (Kulfan anchor
of the PCA runs/2026-11-11_1054_section_pca_convex_ixxv095) at cant = rake = 0, yaw -1 deg; for the check of the CFD
point (2026-11-11). Writes mc2_levels_yawm1.json next to this file."""
import json
import os
import sys
from pathlib import Path

os.environ["VPP_FOIL_XTR"] = "0.03,0.03"
HERE = Path(__file__).resolve().parent
VPP = HERE.parents[1]
sys.path.insert(0, str(VPP))
import numpy as np  # noqa: E402
from src.foil_codes import FoilCodes, LEVELS  # noqa: E402

best = json.loads((VPP / "runs" / "2026-11-11_1054_section_pca_convex_ixxv095" / "results" /
                   "current_section.json").read_text(encoding="utf-8"))
w = np.concatenate([best["kulfan_upper"], best["kulfan_lower"]])
codes = FoilCodes(HERE / "work_mc2", levels=LEVELS)
out = {}
for level in (1, 2):
    y, info = codes.evaluate(w, (0.0, 0.0, -1.0), level)
    out[f"level{level}"] = {"y": y.tolist(), "error": info["error"], "time_s": info["time_s"]}
    print(level, y, info["error"], flush=True)
(HERE / "mc2_levels_yawm1.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
