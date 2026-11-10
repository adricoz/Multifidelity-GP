# Second batch of the xtr study of 2026-11-09 (user):
#  - xtr 2 % replaces the 1 % run (invalid: polar timeout 30 s made ~50 % of the AVL x XFOIL runs fail).
#    Default polar timeout 120 s: tested on 4 sections at 2 / 2.5 / 3 %, no failure, XFOIL hangs up to
#    the timeout on most sections (200-360 s per new section) at all three values -> 2 % kept.
#  - xtr 10 % again, started from the best section of the 5 % run: the first 10 % run stopped after
#    11 updates (convergence test) 0.025 kt under the 5 % best in 10 % physics.
# Same settings as runs/launch_2026-11-09_xtr.ps1 otherwise. Then the cross evaluation of the
# 2 %, 5 % and new 10 % runs (+ the first 10 % run as a section). Started detached:
#   Start-Process powershell -WindowStyle Hidden -ArgumentList '-File', 'runs\launch_2026-11-09_xtr_b.ps1'
Set-Location (Split-Path $PSScriptRoot -Parent)
Add-Type -Namespace XtrB -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
[void][XtrB.Power]::SetThreadExecutionState([uint32]2147483649)
$common = '--search-box runs/2026-11-05_1212_section_pca_convex_tmax2_0p9 --k 10 ' +
    '--levels nf_llt avl --n-design 256 64 --costs 1 6 --batch 16 ' +
    '--search-profile profiles/profile_v2.json --profile profiles/profile_v2.json ' +
    '--local-fraction 0.5 --radius 1.0 --converge 10 0.01 --restarts 0 ' +
    '--min-gain 0.002 --sensitivity-median 3 --seed 0 --screen-seed 197583743'
$python = '..\.venv-mds\Scripts\python.exe'
$best5 = 'runs/2026-11-09_1057_xtr005_rigid_convex_k10/results/best_section.json'
$jobs = @()
$jobs += Start-Process cmd.exe -PassThru -WindowStyle Hidden -ArgumentList '/c',
    "$python scripts\run_real_vpp.py --name xtr002_rigid_convex_k10 --xtr 0.02 --workers 20 --vpp-workers 12 --time-limit-min 180 $common > `"runs\xtr_2026-11-09_002.out`" 2>&1"
Start-Sleep -Seconds 20
$jobs += Start-Process cmd.exe -PassThru -WindowStyle Hidden -ArgumentList '/c',
    "$python scripts\run_real_vpp.py --name xtr010_from5_rigid_convex_k10 --xtr 0.10 --start-section $best5 --workers 8 --vpp-workers 8 --time-limit-min 120 $common > `"runs\xtr_2026-11-09_010b.out`" 2>&1"
$jobs | Wait-Process

# cross evaluation: each run's surrogate + AVL x XFOIL at its xtr, on the bests of the others
$runs = @('runs/2026-11-09_1057_xtr005_rigid_convex_k10') +
        @(Get-ChildItem runs -Directory | Where-Object Name -match '^2026-11-09_\d{4}_xtr0(02|10_from5)_rigid_convex_k10$' |
          ForEach-Object { "runs/$($_.Name)" })
$extra = 'runs/2026-11-09_1057_xtr010_rigid_convex_k10'
foreach ($run in $runs) {
    if (-not (Test-Path "$run/results/best_section.json")) { continue }
    $others = (@($runs + $extra) | Where-Object { $_ -ne $run } |
               Where-Object { Test-Path "$_/results/best_section.json" }) -join ' '
    $tag = ($run -split '_')[2]
    cmd.exe /c "$python scripts\robustness_checks.py cross --run $run --sections-from $others --polar-timeout 120 --workers 28 > `"runs\xtr_cross_b_2026-11-09_$tag.out`" 2>&1"
}
[void][XtrB.Power]::SetThreadExecutionState([uint32]2147483648)
