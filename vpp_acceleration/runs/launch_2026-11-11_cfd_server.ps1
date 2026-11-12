# PREPARED 2026-11-11: the server of the 3D CFD level of runs/launch_2026-11-11_ixxv095_cfd.ps1 (start it right after
# the optimisation). It waits for the run folder, then computes the requests of <run>/cfd one at a time on the GPU
# (np 4, PETSc FCG + AmgX), until 12/11 09:00 or a file STOP in <run>/cfd. Each point: launch folder with PARAMETRES.md
# and y+ map in <run>/cfd/cases/<id>, case in WSL ~/bdcfd/lancement/L3_<run>_<id>, result in <run>/cfd/done/<id>.json.
#
#   schtasks /Create /TN "Adri CFD server ixxv095" /TR "powershell -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File <this file>" /SC ONCE /SD 01/01/2099 /ST 00:00 /IT /RL LIMITED /F
#   schtasks /Run /TN "Adri CFD server ixxv095"
Set-Location (Split-Path $PSScriptRoot -Parent)
$python = (Resolve-Path '..\.venv-mds\Scripts\python.exe').Path
$queue = $null
for ($i = 0; $i -lt 120 -and -not $queue; $i++) {
    $queue = Get-ChildItem runs -Directory -Filter '*_ixxv095_rigid_xtr003_k10_cfd3d' | Sort-Object Name | Select-Object -Last 1
    if (-not $queue) { Start-Sleep -Seconds 30 }
}
$q = Join-Path $queue.FullName 'cfd'
New-Item -ItemType Directory -Force (Join-Path $q 'requests') | Out-Null
New-Item -ItemType Directory -Force (Join-Path $q 'done') | Out-Null
& $python scripts\cfd_level.py serve --queue $q --deadline 2026-11-12T09:00 --reference data\cfd_level\reference.json --gpu --np 4 --hours 4 *> (Join-Path $q 'server.log')
