# PREPARED 2026-11-11 (plan validated by Antoine): section optimisation under Ixx/v >= 0.95 x the current section only
# (PCA runs/2026-11-11_1054_section_pca_convex_ixxv095, box C_envelope, k = 10), rigid, levels NPLLT x NeuralFoil /
# AVL x XFOIL at xtr 3 % (n_crit 1) + the 3D CFD level (swept medium mesh, 8.49 m/s, yaw only, convergence 0.2 % /
# 1500 it) served asynchronously on the GPU by runs/launch_2026-11-11_cfd_server.ps1 (start it right after this one).
# profile_v2, settings of the xtr 3 % run of 09/11, 3 restarts. The loop stops at 07:30 on 12/11 (final validation
# and figures after it, before 09:00). CPU: 16 code workers + 10 VPP workers, 4 cores left to the GPU CFD.
#
# Start detached (never wait on it over SSH):
#   schtasks /Create /TN "Adri optim ixxv095 cfd" /TR "powershell -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File <this file>" /SC ONCE /ST 23:59 /IT /RL LIMITED /F
#   schtasks /Run /TN "Adri optim ixxv095 cfd"
Set-Location (Split-Path $PSScriptRoot -Parent)
Add-Type -Namespace OptimCfd -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
[void][OptimCfd.Power]::SetThreadExecutionState([uint32]2147483649)
$python = (Resolve-Path '..\.venv-mds\Scripts\python.exe').Path
$minutes = [int]((Get-Date '2026-11-12T07:30') - (Get-Date)).TotalMinutes
$common = @('--search-box', 'runs/2026-11-11_1054_section_pca_convex_ixxv095', '--k', '10',
            '--levels', 'nf_llt', 'avl', '--n-design', '256', '64', '--costs', '1', '6', '--batch', '16',
            '--search-profile', 'profiles/profile_v2.json', '--profile', 'profiles/profile_v2.json',
            '--local-fraction', '0.5', '--radius', '1.0', '--converge', '10', '0.01', '--restarts', '3',
            '--min-gain', '0.002', '--sensitivity-median', '3', '--seed', '0', '--screen-seed', '197583743',
            '--xtr', '0.03', '--workers', '16', '--vpp-workers', '10', '--time-limit-min', "$minutes",
            '--cfd-queue', '--cfd-cost', '240', '--cfd-start-yaw', '-1',
            '--name', 'ixxv095_rigid_xtr003_k10_cfd3d')
& $python scripts\run_real_vpp.py @common *> runs\optim_ixxv095_cfd.out
[void][OptimCfd.Power]::SetThreadExecutionState([uint32]2147483648)
