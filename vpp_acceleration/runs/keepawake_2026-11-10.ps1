# Light keep-awake for the night 10 -> 11/11/2026 (user: "un petit processus peu coûteux ... qui s'arrête quand
# Antoine vient"). Prevents sleep (not the screen), writes a heartbeat every 5 min to runs\keepawake.log (a gap in the
# log dates a shutdown / freeze), stops at the first of: keyboard / mouse input after 06:00 on 11/11, a file
# STOP_KEEPAWAKE on the Desktop, 11/11 12:00. Started by the scheduled task "Adri keep-awake 2026-11-10".
Set-Location (Split-Path $PSScriptRoot -Parent)
Add-Type -Namespace KeepAwake -Name Native -MemberDefinition @'
[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);
[StructLayout(LayoutKind.Sequential)] public struct LASTINPUTINFO { public uint cbSize; public uint dwTime; }
[DllImport("user32.dll")] public static extern bool GetLastInputInfo(ref LASTINPUTINFO plii);
public static uint IdleMs() { LASTINPUTINFO l = new LASTINPUTINFO(); l.cbSize = 8; GetLastInputInfo(ref l); return (uint)Environment.TickCount - l.dwTime; }
'@
$log = 'runs\keepawake.log'
$stopFile = Join-Path ([Environment]::GetFolderPath('Desktop')) 'STOP_KEEPAWAKE'
$watchFrom = [datetime]'2026-11-11 06:00'
$end = [datetime]'2026-11-11 12:00'
[void][KeepAwake.Native]::SetThreadExecutionState([uint32]2147483649)
"$(Get-Date -Format s) start (pid $PID)" | Out-File -Append -Encoding utf8 $log
$reason = ''
$tick = 0
while ($true) {
    $now = Get-Date
    if (Test-Path $stopFile) { $reason = 'fichier STOP_KEEPAWAKE'; break }
    if ($now -ge $end) { $reason = 'limite 11/11 12:00'; break }
    if ($now -ge $watchFrom -and [KeepAwake.Native]::IdleMs() -lt 90000) { $reason = 'activité clavier/souris (Antoine)'; break }
    if ($tick % 5 -eq 0) {
        $cpu = (Get-CimInstance Win32_Processor | Measure-Object LoadPercentage -Average).Average
        $gpu = (& nvidia-smi --query-gpu=utilization.gpu,power.draw --format=csv,noheader 2>$null) -join ' '
        "$(Get-Date -Format s) alive cpu $cpu % gpu $gpu" | Out-File -Append -Encoding utf8 $log
    }
    $tick++
    Start-Sleep -Seconds 60
}
[void][KeepAwake.Native]::SetThreadExecutionState([uint32]2147483648)
"$(Get-Date -Format s) arrêté : $reason" | Out-File -Append -Encoding utf8 $log
