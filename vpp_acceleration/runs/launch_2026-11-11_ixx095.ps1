# PREPARED 2026-11-10 evening, NOT STARTED: section optimisation on the search space under Ixx >= 0.95 and
# Ixx / v >= 0.9 x the current section (PCA runs/2026-11-10_1832_section_pca_convex_ixx095_ixxv090, box C_envelope,
# k = 10), rigid (no --fsi), 2 levels NPLLT x NeuralFoil / AVL x XFOIL at xtr 3 % (n_crit 1), profile_v2, same
# settings as the xtr 3 % run of 09/11 (runs/2026-11-09_1411_xtr003_rigid_convex_k10) except the time limit (~29 h of
# loop) and 3 restarts. The CFD level is NOT in this command: it is to be designed with Antoine first (see
# C:\Users\SIM\Desktop\Code-Adri\A_LIRE_ANTOINE_2026-11-11.md). Check the machine load before starting (WSL / CFD jobs).
#
# Start detached (the session may freeze; never wait on it over SSH):
#   schtasks /Create /TN "Adri optim ixx095" /TR "powershell -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File <this file>" /SC ONCE /ST 23:59 /IT /RL LIMITED /F
#   schtasks /Run /TN "Adri optim ixx095"
Set-Location (Split-Path $PSScriptRoot -Parent)
Add-Type -Namespace OptimIxx -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
[void][OptimIxx.Power]::SetThreadExecutionState([uint32]2147483649)
$python = (Resolve-Path '..\.venv-mds\Scripts\python.exe').Path
$common = @('--search-box', 'runs/2026-11-10_1832_section_pca_convex_ixx095_ixxv090', '--k', '10',
            '--levels', 'nf_llt', 'avl', '--n-design', '256', '64', '--costs', '1', '6', '--batch', '16',
            '--search-profile', 'profiles/profile_v2.json', '--profile', 'profiles/profile_v2.json',
            '--local-fraction', '0.5', '--radius', '1.0', '--converge', '10', '0.01', '--restarts', '3',
            '--min-gain', '0.002', '--sensitivity-median', '3', '--seed', '0', '--screen-seed', '197583743',
            '--xtr', '0.03', '--workers', '20', '--vpp-workers', '12', '--time-limit-min', '1740',
            '--name', 'ixx095_ixxv090_rigid_xtr003_k10')
& $python scripts\run_real_vpp.py @common *> runs\optim_ixx095.out
[void][OptimIxx.Power]::SetThreadExecutionState([uint32]2147483648)
