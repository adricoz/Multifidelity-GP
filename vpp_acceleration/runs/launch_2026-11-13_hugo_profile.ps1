# 2026-11-13, approved by Adrian: Monte Carlo of the operating law for Hugo (scripts/hugo_inputs.py profile, then
# sensitivities). Waits until the validation experiments (validation_experiments / twin_experiments python processes)
# are finished, at most until 17:00 (night.py stops itself at 16:45), then runs with 8 workers. Starts keep-awake.
#   schtasks /Create /TN "Adri hugo profile" /TR "powershell -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File <this file>" /SC ONCE /SD 01/01/2099 /ST 00:00 /IT /RL LIMITED /F
#   schtasks /Run /TN "Adri hugo profile"
Set-Location (Split-Path $PSScriptRoot -Parent)
schtasks /Run /TN "Adri keep-awake" | Out-Null
$out = 'runs\hugo_profile.out'
"$(Get-Date -Format s) waiting for the validation experiments" | Out-File -Append -Encoding utf8 $out
$deadline = Get-Date '2026-11-13 17:00'
while ((Get-Date) -lt $deadline) {
    $busy = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
              Where-Object { $_.CommandLine -match 'validation_experiments|twin_experiments' })
    if ($busy.Count -eq 0) { break }
    Start-Sleep -Seconds 120
}
"$(Get-Date -Format s) start" | Out-File -Append -Encoding utf8 $out
$python = (Resolve-Path '..\.venv-mds\Scripts\python.exe').Path
& $python -u scripts\hugo_inputs.py profile --n 500 --workers 8 *>> $out
& $python -u scripts\hugo_inputs.py sensitivities --m 300 *>> $out
"$(Get-Date -Format s) end" | Out-File -Append -Encoding utf8 $out
