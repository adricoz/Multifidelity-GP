# Cross evaluation after the three xtr runs of 2026-11-09 (user OK): each run's best section, the
# two other bests and the current section, with the surrogate + AVL x XFOIL of each run, i.e. in
# its transition (scripts/robustness_checks.py cross). Waits until the three run_real_vpp.py
# processes are gone, then runs the three checks one after the other. Started detached:
#   Start-Process powershell -WindowStyle Hidden -ArgumentList '-File', 'runs\launch_2026-11-09_xtr_cross.ps1'
Set-Location (Split-Path $PSScriptRoot -Parent)
Add-Type -Namespace XtrCross -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
[void][XtrCross.Power]::SetThreadExecutionState([uint32]2147483649)
while (Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
       Where-Object { $_.CommandLine -match 'run_real_vpp\.py --name xtr0' }) { Start-Sleep -Seconds 120 }
$runs = @(Get-ChildItem runs -Directory | Where-Object Name -match '^2026-11-09_\d{4}_xtr0\d\d_rigid_convex_k10$' |
          ForEach-Object { "runs/$($_.Name)" })
$python = '..\.venv-mds\Scripts\python.exe'
foreach ($run in $runs) {
    $others = ($runs | Where-Object { $_ -ne $run } |
               Where-Object { Test-Path "$_/results/best_section.json" }) -join ' '
    if (-not (Test-Path "$run/results/best_section.json") -or -not $others) { continue }
    $tag = ($run -split '_')[2]
    cmd.exe /c "$python scripts\robustness_checks.py cross --run $run --sections-from $others --workers 28 > `"runs\xtr_cross_2026-11-09_$tag.out`" 2>&1"
}
[void][XtrCross.Power]::SetThreadExecutionState([uint32]2147483648)
