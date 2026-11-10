# Night of 2026-11-10 -> 11/11 07:30 (user: "des polaires CFD 2D pour AVL ... les seuls CFD que je te laisse faire
# sans ma vérif ... prêts demain matin"). scripts/cfd2d_polars.py, Antoine's bdfoil_v1 2D chain unchanged, CPU only
# (4 cases x 4 ranks in WSL), Re 5e6, sections MC2 (current) and opt003 (best of the xtr 3 % run).
# Figures redrawn every 30 min and at the end: runs/2026-11-10_1900_cfd2d_polars_re5e6/figures/polars.html.
# Started by the scheduled task "Adri CFD2D night 2026-11-10" (Interactive, Limited).
Set-Location (Split-Path $PSScriptRoot -Parent)
Add-Type -Namespace Cfd2dNight -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
[void][Cfd2dNight.Power]::SetThreadExecutionState([uint32]2147483649)    # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
$python = (Resolve-Path '..\.venv-mds\Scripts\python.exe').Path
$dir = 'runs\2026-11-10_1900_cfd2d_polars_re5e6'
$env:OMP_NUM_THREADS = '1'
"launcher start $(Get-Date -Format s)" | Out-File -Append -Encoding utf8 "$dir\launcher.log"
$p = Start-Process -FilePath $python -ArgumentList 'scripts\cfd2d_polars.py', 'run', '--run-dir', $dir, '--workers', '4', '--np', '4', '--deadline', '2026-11-11T07:30' `
     -RedirectStandardOutput "$dir\stdout.log" -RedirectStandardError "$dir\stderr.log" -NoNewWindow -PassThru
while (-not $p.HasExited) {
    Start-Sleep -Seconds 1800
    & $python scripts\cfd2d_polars.py figures --run-dir $dir *>> "$dir\figures.log"
}
& $python scripts\cfd2d_polars.py figures --run-dir $dir *>> "$dir\figures.log"
"launcher end $(Get-Date -Format s), python exit $($p.ExitCode)" | Out-File -Append -Encoding utf8 "$dir\launcher.log"
[void][Cfd2dNight.Power]::SetThreadExecutionState([uint32]2147483648)
