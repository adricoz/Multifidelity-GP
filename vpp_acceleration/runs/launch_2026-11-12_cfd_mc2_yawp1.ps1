# 2026-11-12, asked by Adrian: MC2 (Kulfan anchor of the PCA ixxv095) in 3D CFD at yaw +1 deg, the yaw of the final best
# section of run 2026-11-11_1245_ixxv095_rigid_xtr003_k10_cfd3d (CFD point 1112_031300_686f), for a direct CFD
# comparison. Same settings as the MC2 yaw -1 point: swept medium mesh, 8.49 m/s, GPU np 4, 0.2 % / 1500 it.
#   schtasks /Create /TN "Adri CFD MC2 yaw+1" /TR "powershell -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File <this file>" /SC ONCE /ST 23:59 /IT /RL LIMITED /F
#   schtasks /Run /TN "Adri CFD MC2 yaw+1"
Set-Location (Split-Path $PSScriptRoot -Parent)
$python = (Resolve-Path '..\.venv-mds\Scripts\python.exe').Path
$section = 'runs\2026-11-11_1054_section_pca_convex_ixxv095\results\current_section.json'
$name = 'L3_MC2kulfan_moyen_8p49_yawp1'
& $python scripts\cfd_level.py prepare --name $name --yaw 1 --gpu --section-json $section *> runs\cfd_mc2_yawp1_prepare.out
& $python scripts\cfd_level.py run --name $name --yaw 1 --gpu --hours 4 --section-json $section `
    --reference data\cfd_level\reference.json *> runs\cfd_mc2_yawp1.out
