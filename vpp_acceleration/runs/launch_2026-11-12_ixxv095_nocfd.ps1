# 2026-11-12, asked by Adrian: the comparison run WITHOUT the CFD level. Same problem and settings as
# runs/launch_2026-11-11_ixxv095_cfd.ps1 (PCA runs/2026-11-11_1054_section_pca_convex_ixxv095, Ixx/v >= 0.95 only, box,
# k 10, nf_llt + avl at xtr 3 %, profile_v2, seeds, design sizes, 3 restarts), without --cfd-queue and without the
# XFOIL fallback to xtr 5 % (--no-xtr-fallback: same codes as the CFD run of 11/11). The time limit (20 h) is not
# meant to bind: the CFD run ended by convergence after 14.5 h. Waits for the end of the MC2 yaw +1 CFD point
# (result.json of its launch folder, at most 4 h) so that XFOIL does not share the CPU with it.
#   schtasks /Create /TN "Adri optim ixxv095 nocfd" /TR "powershell -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File <this file>" /SC ONCE /SD 01/01/2099 /ST 00:00 /IT /RL LIMITED /F
#   schtasks /Run /TN "Adri optim ixxv095 nocfd"
Set-Location (Split-Path $PSScriptRoot -Parent)
$result = 'C:\Users\SIM\Outils - banulsdesign\bdToolbox\tmp\bdfoil_v1\cfd3d\lancement\L3_MC2kulfan_moyen_8p49_yawp1\result.json'
for ($i = 0; $i -lt 240 -and -not (Test-Path $result); $i++) { Start-Sleep -Seconds 60 }
Add-Type -Namespace OptimNoCfd -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
[void][OptimNoCfd.Power]::SetThreadExecutionState([uint32]2147483649)
$python = (Resolve-Path '..\.venv-mds\Scripts\python.exe').Path
$common = @('--search-box', 'runs/2026-11-11_1054_section_pca_convex_ixxv095', '--k', '10',
            '--levels', 'nf_llt', 'avl', '--n-design', '256', '64', '--costs', '1', '6', '--batch', '16',
            '--search-profile', 'profiles/profile_v2.json', '--profile', 'profiles/profile_v2.json',
            '--local-fraction', '0.5', '--radius', '1.0', '--converge', '10', '0.01', '--restarts', '3',
            '--min-gain', '0.002', '--sensitivity-median', '3', '--seed', '0', '--screen-seed', '197583743',
            '--xtr', '0.03', '--no-xtr-fallback', '--workers', '16', '--vpp-workers', '10', '--time-limit-min', '1200',
            '--name', 'ixxv095_rigid_xtr003_k10_nocfd')
& $python scripts\run_real_vpp.py @common *> runs\optim_ixxv095_nocfd.out
[void][OptimNoCfd.Power]::SetThreadExecutionState([uint32]2147483648)
