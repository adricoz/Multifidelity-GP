# Night of 2026-11-09 (user: "fais des petites évaluations CFD pour les meilleures sections trouvées ... utilise le GPU
# ... tout doit être fini demain matin 9h"). scripts/cfd_sections.py, Antoine's OpenFOAM chain unchanged:
#  GPU queue: 3D C9 with the section swapped, pressure on PETSc/AmgX (4 ranks): L1 opt005, L2 opt005, L1 opt003,
#             L1 opt010b, then L1 MC2 on GPU (check against Antoine's CPU L1). Each run capped so it ends by 07:45.
#  CPU queue: once the cross evaluations of launch_2026-11-09_xtr_c.ps1 are done, AVL x XFOIL of C9 for every
#             section at xtr 3/5/10 %, then 2D pairs (L1'+L2', 10 ranks) of the optimised sections.
#  Then the figures (runs/2026-11-09_xtr_study/figures/cfd_sections.html).
# Started detached:  Start-Process powershell -WindowStyle Hidden -ArgumentList '-File', 'runs\launch_2026-11-09_cfd_night.ps1'
Set-Location (Split-Path $PSScriptRoot -Parent)
Add-Type -Namespace CfdNight -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
[void][CfdNight.Power]::SetThreadExecutionState([uint32]2147483649)
$python = (Resolve-Path '..\.venv-mds\Scripts\python.exe').Path
$here = (Get-Location).Path
$end = [datetime]'2026-11-10 07:45'

$gpu = Start-Job -ArgumentList $python, $here, $end -ScriptBlock {
    param($python, $here, $end)
    Set-Location $here
    $queue = @(@('opt005', 1, 2.0), @('opt005', 2, 5.0), @('opt003', 1, 2.0), @('opt010b', 1, 2.0), @('MC2', 1, 2.0))
    foreach ($q in $queue) {
        $left = ($end - (Get-Date)).TotalHours
        if ($left -lt 0.75) { "skip $($q[0]) L$($q[1]): $([math]::Round($left, 2)) h left" | Out-File -Append runs\cfd_night_gpu.out; continue }
        $h = [math]::Min($q[2], $left)
        cmd.exe /c "`"$python`" scripts\cfd_sections.py run3d --section $($q[0]) --level $($q[1]) --gpu --np 4 --hours $h >> runs\cfd_night_gpu.out 2>&1"
    }
}

$cpu = Start-Job -ArgumentList $python, $here, $end -ScriptBlock {
    param($python, $here, $end)
    Set-Location $here
    while (Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" |
           Where-Object { $_.CommandLine -match 'launch_2026-11-09_xtr_c' }) { Start-Sleep -Seconds 60 }
    cmd.exe /c "`"$python`" scripts\cfd_sections.py avl --xtr 0.03 0.05 0.10 >> runs\cfd_night_cpu.out 2>&1"
    foreach ($q in @(@('opt005', '-3 -2 -1 0 1'), @('opt003', '-2 -1 0'), @('opt010b', '-2 -1 0'))) {
        if ((Get-Date) -gt $end.AddHours(-1)) { "skip 2D $($q[0])" | Out-File -Append runs\cfd_night_cpu.out; continue }
        cmd.exe /c "`"$python`" scripts\cfd_sections.py run2d --sections $($q[0]) --alphas $($q[1]) --np 10 >> runs\cfd_night_cpu.out 2>&1"
    }
}

# figures as soon as both queues are done, at the latest at 08:15 (then again when the queues end)
while ((($gpu.State -eq 'Running') -or ($cpu.State -eq 'Running')) -and ((Get-Date) -lt $end.AddMinutes(30))) { Start-Sleep -Seconds 120 }
cmd.exe /c "`"$python`" scripts\cfd_sections.py figures > runs\cfd_night_figures.out 2>&1"
Wait-Job $gpu, $cpu | Out-Null
cmd.exe /c "`"$python`" scripts\cfd_sections.py figures > runs\cfd_night_figures.out 2>&1"
"all done $(Get-Date)" | Out-File -Append runs\cfd_night_figures.out
[void][CfdNight.Power]::SetThreadExecutionState([uint32]2147483648)
