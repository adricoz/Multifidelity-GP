# Third batch of the xtr study of 2026-11-09 (user: "teste 2% et si ça marche pas mets 2.5-3%").
# 2 % failed (42 % of the AVL x XFOIL runs, polar negative branch, even with the 120 s timeout;
# C9 case fails too); 3 %: 7/8 sections that failed at 2 % pass (100-360 s each), C9 case passes.
# -> xtr 3 % run, same settings as the others (start = current section). Then, once the 10 % run
# started from the 5 % best is done too: cross evaluation of 3 %, 5 %, 10 % (from 5 %) (+ the
# first 10 % best as a section) and the study figures. Started detached:
#   Start-Process powershell -WindowStyle Hidden -ArgumentList '-File', 'runs\launch_2026-11-09_xtr_c.ps1'
Set-Location (Split-Path $PSScriptRoot -Parent)
Add-Type -Namespace XtrC -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
[void][XtrC.Power]::SetThreadExecutionState([uint32]2147483649)
$common = '--search-box runs/2026-11-05_1212_section_pca_convex_tmax2_0p9 --k 10 ' +
    '--levels nf_llt avl --n-design 256 64 --costs 1 6 --batch 16 ' +
    '--search-profile profiles/profile_v2.json --profile profiles/profile_v2.json ' +
    '--local-fraction 0.5 --radius 1.0 --converge 10 0.01 --restarts 0 ' +
    '--min-gain 0.002 --sensitivity-median 3 --seed 0 --screen-seed 197583743'
$python = '..\.venv-mds\Scripts\python.exe'
cmd.exe /c "$python scripts\run_real_vpp.py --name xtr003_rigid_convex_k10 --xtr 0.03 --workers 20 --vpp-workers 12 --time-limit-min 180 $common > `"runs\xtr_2026-11-09_003.out`" 2>&1"
while (Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
       Where-Object { $_.CommandLine -match 'run_real_vpp\.py --name xtr010_from5' }) { Start-Sleep -Seconds 60 }

function Find-Run($pattern) {
    Get-ChildItem runs -Directory | Where-Object Name -match $pattern | Select-Object -Last 1 |
        ForEach-Object { "runs/$($_.Name)" }
}
$runs = @('runs/2026-11-09_1057_xtr005_rigid_convex_k10',
          (Find-Run '^2026-11-09_\d{4}_xtr003_rigid_convex_k10$'),
          (Find-Run '^2026-11-09_\d{4}_xtr010_from5_rigid_convex_k10$')) |
        Where-Object { $_ -and (Test-Path "$_/results/best_section.json") }
$extra = 'runs/2026-11-09_1057_xtr010_rigid_convex_k10'
foreach ($run in $runs) {
    $others = (@($runs + $extra) | Where-Object { $_ -ne $run }) -join ' '
    $tag = ($run -split '_')[2]
    cmd.exe /c "$python scripts\robustness_checks.py cross --run $run --sections-from $others --polar-timeout 120 --workers 28 > `"runs\xtr_cross_c_2026-11-09_$tag.out`" 2>&1"
}
$list = ($runs + $extra) -join ' '
cmd.exe /c "$python scripts\xtr_study_figures.py --runs $list --polar-xtr 0.03 0.05 0.10 --avl-xtr data/xtr_study/c9_avl_xtr.json > `"runs\xtr_study_figures_2026-11-09.out`" 2>&1"
[void][XtrC.Power]::SetThreadExecutionState([uint32]2147483648)
