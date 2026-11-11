"""Do the outputs of FoilCodes (level avl) and Session(cfoil_r9800_fsi) (the case of the CFD) describe the same foil in
the same frame? AVL x XFOIL xtr 3 % on MC2 at 8.49 m/s, yaw 0 / 2 / 4 deg: ratio FoilCodes_k / (F_k / (q Sref))."""
import dataclasses
import json
import os
import sys
from pathlib import Path

os.environ["VPP_FOIL_XTR"] = "0.05,0.05"
VPP = Path(r"C:\Users\SIM\Desktop\Code-Adri\Multifidelity-GP-Clean\vpp_acceleration")
sys.path.insert(0, str(VPP))
sys.path.insert(0, str(VPP / "scripts"))
import cfd_sections  # noqa: E402  (sets the bdfoil_v1 paths)
from bdfoil_v1.model import Attitude  # noqa: E402
from bdfoil_v1.session import Session  # noqa: E402
from src.foil_codes import FoilCodes, read_xf_section, LEVELS  # noqa: E402

xf = cfd_sections.section_xf("MC2")
codes = FoilCodes(Path(sys.argv[1]), levels=LEVELS)
study = cfd_sections.study_for("MC2")
study = dataclasses.replace(study, polars=dataclasses.replace(study.polars, xtr=(0.05, 0.05)))
session = Session(study)
print("polars", study.polars)
rows = []
for yaw in (0.0, 2.0, 4.0):
    y, info = codes.evaluate(None, (0.0, 0.0, yaw), 2, section=read_xf_section(xf))
    r = session.evaluate(Attitude(cant=0.0, rake=0.0, yaw=yaw, sink=0.0, speed=8.49))
    si = {k: float(v) for k, v in dict(r.si).items() if isinstance(v, (int, float))}
    tot = cfd_sections.__dict__.get("c9") or None
    rows.append({"yaw": yaw, "foilcodes": y.tolist(), "err": info["error"],
                 "si": {k: v for k, v in si.items() if k.split()[0] in ("Fx", "Fy", "Fz", "Mx", "My", "Mz")
                        or "Sref" in k or "Cref" in k or k.startswith("q") or "Speed" in k or "rho" in k.lower()}})
    print(json.dumps(rows[-1], indent=1, default=float), flush=True)
(Path(sys.argv[1]) / "calib_frames.json").write_text(json.dumps(rows, indent=1, default=float))
