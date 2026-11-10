# 2026-11-10 morning (user: "relance un truc 1 h max"): 3D C9 L1 on the CPU (16 ranks) of opt003 at the yaw of equal
# lift to MC2 at neutral (AVL x XFOIL xtr 5 %: yaw 2.593 deg -> Fz 29.15 kN, Fy 53.4 kN). The run stops itself at
# 57 min (bridge timeout); hard guard at 62 min: bdfoil's kill_run on any case still running under cfd3d.
# Started detached:  Start-Process powershell -WindowStyle Hidden -ArgumentList '-File', 'runs\launch_2026-11-10_cfd_opt003_yaw.ps1'
Set-Location (Split-Path $PSScriptRoot -Parent)
$python = (Resolve-Path '..\.venv-mds\Scripts\python.exe').Path
$job = Start-Job -ArgumentList $python, (Get-Location).Path -ScriptBlock {
    param($python, $here)
    Set-Location $here
    cmd.exe /c "`"$python`" scripts\cfd_sections.py run3d --section opt003 --level 1 --yaw 2.593 --np 16 --hours 0.95 > runs\cfd_opt003_yaw.out 2>&1"
    cmd.exe /c "`"$python`" scripts\cfd_sections.py figures >> runs\cfd_opt003_yaw.out 2>&1"
}
if (-not (Wait-Job $job -Timeout 3720)) {
    & $python -c @"
import subprocess, sys
sys.path.insert(0, 'scripts'); import cfd_sections
from bdfoil_v1.solvers.openfoam import bridge
env = bridge.detect()
out = subprocess.run(['wsl', '-d', env.distro, '-u', env.user, '--', 'bash', '-c',
                      'for p in `$(pgrep -f \"mpirun -np 16\"); do readlink /proc/`$p/cwd; done'],
                     capture_output=True, text=True).stdout.split()
for case in sorted(set(c for c in out if '/cfd3d/' in c)):
    print(case, bridge.kill_run(env, case))
"@ >> runs\cfd_opt003_yaw.out 2>&1
    Stop-Job $job
}
"end $(Get-Date)" | Out-File -Append runs\cfd_opt003_yaw.out
