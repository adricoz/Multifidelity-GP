# Long run of 2026-11-05 on the convex section domain (user: "go convergence complete run long"):
# run-8 settings, search space = PCA of runs/2026-11-05_1212_section_pca_convex_tmax2_0p9, 10 modes
# through the current section, box C_envelope, constraints (Ixx/v >= 0.9 current, v = t_max/2)
# checked at screening; until full convergence (3 restarts), 12 h safety limit.
# Keeps the PC awake while it runs (power plan unchanged). Started detached:
#   Start-Process powershell -WindowStyle Hidden -ArgumentList '-File', 'runs\launch_2026-11-05_convex_long.ps1'
Set-Location (Split-Path $PSScriptRoot -Parent)
Add-Type -Namespace Long -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
[void][Long.Power]::SetThreadExecutionState([uint32]2147483649)
$arguments = 'scripts\run_real_vpp.py --name convex_envelope_k10_long ' +
    '--search-box runs/2026-11-05_1212_section_pca_convex_tmax2_0p9 --k 10 ' +
    '--levels nf_llt avl --n-design 256 64 --costs 1 6 --fsi --batch 16 --workers 28 --vpp-workers 28 ' +
    '--search-profile profiles/profile_v2.json --profile profiles/profile_v2.json ' +
    '--local-fraction 0.5 --radius 1.0 --converge 10 0.01 --restarts 3 --restart-distance 2.0 ' +
    '--min-gain 0.002 --sensitivity-median 3 --time-limit-min 720 ' +
    '--compare runs/2026-10-29_1002_profile_v2_box3_3h/results/best_section.json ' +
    'runs/2026-10-28_1747_profile_v1_box075/results/best_section.json ' +
    'runs/2026-10-27_1739_long_3levels_avl_iter/results/best_section.json ' +
    'runs/2026-10-27_1224_control_restarts/results/best_section.json'
cmd.exe /c "..\.venv-mds\Scripts\python.exe $arguments > `"runs\convex_long_2026-11-05.out`" 2>&1"
[void][Long.Power]::SetThreadExecutionState([uint32]2147483648)
