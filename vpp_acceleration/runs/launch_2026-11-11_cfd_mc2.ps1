# 2026-11-11, validated by Antoine (ParaView): first point of the 3D CFD level, the current section (MC2, Kulfan anchor
# of the PCA) at yaw -1 deg, 8.49 m/s, swept medium mesh, GPU np 4, convergence 0.2 % / 1500 it. Solves the launch
# folder <bdToolbox>/tmp/bdfoil_v1/cfd3d/lancement/L3_MC2kulfan_moyen_8p49_yawm1 (its WSL copy), writes result.json
# there and the point into data/cfd_level/points (the queue of the optimisation reuses it).
#   schtasks /Create /TN "Adri CFD MC2 L3" /TR "powershell -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File <this file>" /SC ONCE /ST 23:59 /IT /RL LIMITED /F
#   schtasks /Run /TN "Adri CFD MC2 L3"
Set-Location (Split-Path $PSScriptRoot -Parent)
$python = (Resolve-Path '..\.venv-mds\Scripts\python.exe').Path
& $python scripts\cfd_level.py run --name L3_MC2kulfan_moyen_8p49_yawm1 --yaw -1 --gpu --hours 4 `
    --section-json runs\2026-11-11_1054_section_pca_convex_ixxv095\results\current_section.json `
    --reference data\cfd_level\reference.json *> runs\cfd_mc2_l3.out
