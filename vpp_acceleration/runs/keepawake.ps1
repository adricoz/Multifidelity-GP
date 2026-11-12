# Keep-awake of Terminator while long jobs run (rule of Adrian, 2026-11-12: always keep a small process that keeps the
# session active, so that the jobs do not depend on the remote connection). Prevents sleep and the display / idle
# timeout (ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED), writes a heartbeat every 5 min to
# runs\keepawake.log (a gap in the log dates a shutdown / freeze), costs ~0 CPU. Stops at the first of:
#   - a file STOP_KEEPAWAKE on the Desktop;
#   - no other scheduled task "Adri*" running for 15 min in a row (the jobs are done);
#   - -Until (default: 48 h after the start).
# Keyboard / mouse input does NOT stop it. Started by the scheduled task "Adri keep-awake":
#   schtasks /Create /TN "Adri keep-awake" /TR "powershell -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File <this file>" /SC ONCE /SD 01/01/2099 /ST 00:00 /IT /RL LIMITED /F
#   schtasks /Run /TN "Adri keep-awake"
param([datetime]$Until = (Get-Date).AddHours(48))
Set-Location (Split-Path $PSScriptRoot -Parent)
Add-Type -Namespace KeepAwakeG -Name Native -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
$log = 'runs\keepawake.log'
$stopFile = Join-Path ([Environment]::GetFolderPath('Desktop')) 'STOP_KEEPAWAKE'
Remove-Item $stopFile -ErrorAction SilentlyContinue
"$(Get-Date -Format s) start (pid $PID, until $(Get-Date $Until -Format s))" | Out-File -Append -Encoding utf8 $log
$reason, $tick, $idle = '', 0, 0
while ($true) {
    # renewed every minute: ES_CONTINUOUS (0x80000000) | ES_DISPLAY_REQUIRED (2) | ES_SYSTEM_REQUIRED (1)
    [void][KeepAwakeG.Native]::SetThreadExecutionState([uint32]2147483651)
    if (Test-Path $stopFile) { $reason = 'fichier STOP_KEEPAWAKE'; break }
    if ((Get-Date) -ge $Until) { $reason = "limite $(Get-Date $Until -Format s)"; break }
    $jobs = @(Get-ScheduledTask -TaskName 'Adri*' -ErrorAction SilentlyContinue |
              Where-Object { $_.State -eq 'Running' -and $_.TaskName -ne 'Adri keep-awake' })
    if ($jobs.Count -eq 0) { $idle++ } else { $idle = 0 }
    if ($idle -ge 15) { $reason = 'plus aucune tâche Adri* en cours depuis 15 min'; break }
    if ($tick % 5 -eq 0) {
        $cpu = (Get-CimInstance Win32_Processor | Measure-Object LoadPercentage -Average).Average
        $gpu = (& nvidia-smi --query-gpu=utilization.gpu,power.draw --format=csv,noheader 2>$null) -join ' '
        "$(Get-Date -Format s) alive, tasks: $(($jobs | ForEach-Object TaskName) -join ', '), cpu $cpu %, gpu $gpu" |
            Out-File -Append -Encoding utf8 $log
    }
    $tick++
    Start-Sleep -Seconds 60
}
[void][KeepAwakeG.Native]::SetThreadExecutionState([uint32]2147483648)
"$(Get-Date -Format s) arrêté : $reason" | Out-File -Append -Encoding utf8 $log
