# Three section optimizations of 2026-11-09 (user): effect of the forced transition xtr = 1, 5, 10 %
# chord (both sides, XFOIL and NeuralFoil polars, n_crit 1), everything else identical:
# 2 levels NPLLT x NeuralFoil / AVL x XFOIL (AVL = HF and validation), rigid foil (no --fsi),
# convex PCA runs/2026-11-05_1212_section_pca_convex_tmax2_0p9, k 10, box C_envelope, start = current
# (MC2) section, profile_v2, same seeds; one region each (no restart), until convergence
# (10 updates < 0.01 kt) or the time limit, then validation + profile.
# XFOIL polar timeout 30 s for all three: at xtr 0.01 XFOIL hangs on some sweeps (120 s timeouts,
# 267 s per new section); the rows done are kept, 30 s gives the same polar (test 2026-11-09,
# identical torsor) in 85 s; at 0.05 / 0.10 no timeout occurs (9-21 s), so it changes nothing.
# The xtr 0.01 run is ~4-9x slower at HF: it gets more code workers and 20 min more.
# 32 logical cores: code workers 14 + 8 + 8, VPP workers 10 each (the phases alternate).
# Keeps the PC awake while it runs. Started detached:
#   Start-Process powershell -WindowStyle Hidden -ArgumentList '-File', 'runs\launch_2026-11-09_xtr.ps1'
Set-Location (Split-Path $PSScriptRoot -Parent)
Add-Type -Namespace Xtr -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
[void][Xtr.Power]::SetThreadExecutionState([uint32]2147483649)
$common = '--search-box runs/2026-11-05_1212_section_pca_convex_tmax2_0p9 --k 10 ' +
    '--levels nf_llt avl --n-design 256 64 --costs 1 6 --batch 16 --vpp-workers 10 ' +
    '--search-profile profiles/profile_v2.json --profile profiles/profile_v2.json ' +
    '--local-fraction 0.5 --radius 1.0 --converge 10 0.01 --restarts 0 --polar-timeout 30 ' +
    '--min-gain 0.002 --sensitivity-median 3 --seed 0 --screen-seed 197583743'
$python = '..\.venv-mds\Scripts\python.exe'
$jobs = @()
# tag, xtr, code workers, time limit [min]
foreach ($case in @(@('001', '0.01', 14, 140), @('005', '0.05', 8, 120), @('010', '0.10', 8, 120))) {
    $name = "xtr$($case[0])_rigid_convex_k10"
    $out = "runs\xtr_2026-11-09_$($case[0]).out"
    $jobs += Start-Process cmd.exe -PassThru -WindowStyle Hidden -ArgumentList '/c',
        "$python scripts\run_real_vpp.py --name $name --xtr $($case[1]) --workers $($case[2]) --time-limit-min $($case[3]) $common > `"$out`" 2>&1"
    Start-Sleep -Seconds 20      # staggered starts
}
$jobs | Wait-Process
[void][Xtr.Power]::SetThreadExecutionState([uint32]2147483648)
