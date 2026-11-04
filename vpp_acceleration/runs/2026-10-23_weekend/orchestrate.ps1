# Weekend chain of 2026-10-23 (authorized by the user: long run then CFD test, 2 days at most):
#   1. the long search run (run_real_vpp.py, operating profile, local search), ~8.5 h;
#   2. then, if cfd_mesh_study.sh exists and time remains, the CFD mesh study in WSL Ubuntu-24.04.
# Keeps the PC awake while it runs (ES_CONTINUOUS | ES_SYSTEM_REQUIRED, as the bdcfd launcher);
# the power plan itself is not changed. Started detached: Start-Process powershell -File ...
$root = 'C:\Users\SIM\Desktop\Code-Adri\Multifidelity-GP-Clean\vpp_acceleration'
$here = Join-Path $root 'runs\2026-10-23_weekend'
$log = Join-Path $here 'orchestrate.log'
function Note([string]$message) {
    "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $message" | Out-File -Append -Encoding utf8 $log
}
Add-Type -Namespace Weekend -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
[void][Weekend.Power]::SetThreadExecutionState([uint32]2147483649)
$deadline = (Get-Date).AddHours(47)
try {
    Note "start, deadline $deadline"
    Set-Location $root
    $env:PYTHONPATH = 'scripts'
    $arguments = @('scripts\run_real_vpp.py', '--name', 'real_search_profile_local',
        '--box', 'current', '--box-scale', '0.5', '--hf', 'npllt_xfoil', '--hf-cost', '2', '--fsi',
        '--batch', '16', '--workers', '28', '--vpp-workers', '30', '--n-low', '256', '--n-high', '64',
        '--time-limit-min', '480', '--search-profile', 'profiles\guess_bimodal.json',
        '--local-fraction', '0.5', '--radius', '1.0', '--max-refit-size', '2500',
        '--compare', 'runs\2026-10-22_1519_real_npllt_xfoil_fsi\results\best_section.json',
        '--profile', 'profiles\guess_bimodal.json') -join ' '
    Note "long run: python $arguments"
    cmd.exe /c "..\.venv-mds\Scripts\python.exe $arguments > `"$here\long_run.out`" 2>&1"
    Note "long run finished, exit code $LASTEXITCODE"
    $cfd = Join-Path $here 'cfd_mesh_study.sh'
    if ((Test-Path $cfd) -and ((Get-Date) -lt $deadline)) {
        $hours = [math]::Floor(($deadline - (Get-Date)).TotalHours)
        Note "CFD mesh study, time budget $hours h"
        $env:WSL_UTF8 = '1'
        $wslPath = '/mnt/c/Users/SIM/Desktop/Code-Adri/Multifidelity-GP-Clean/vpp_acceleration/runs/2026-10-23_weekend/cfd_mesh_study.sh'
        cmd.exe /c "wsl.exe -d Ubuntu-24.04 --exec /bin/bash $wslPath $hours > `"$here\cfd.out`" 2>&1"
        Note "CFD finished, exit code $LASTEXITCODE"
    } else {
        Note 'no CFD script or past the deadline: CFD skipped'
    }
} finally {
    [void][Weekend.Power]::SetThreadExecutionState([uint32]2147483648)
    Note 'end'
}
