# 2026-11-12, plan approved by Adrian (run on 13/11 09:30-16:45, the night launch was lost with the session): unattended validation night of the optimisation algorithm (no CFD).
# validation_experiments/night.py waits for the end of the run without CFD, then runs the model-only tests
# (buffered cross-validation, certificate, cross judgement) and the digital twin (twin_experiments/), writes
# validation_experiments/NUIT_2026-11-12.md and the figures, stops everything at 08:50 on 13/11 and commits
# the results. Nothing depends on an interactive session. Also starts the keep-awake task.
#   schtasks /Create /TN "Adri validation nuit" /TR "powershell -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File <this file>" /SC ONCE /SD 01/01/2099 /ST 00:00 /IT /RL LIMITED /F
#   schtasks /Run /TN "Adri validation nuit"
Set-Location (Split-Path $PSScriptRoot -Parent)
schtasks /Run /TN "Adri keep-awake" | Out-Null
Add-Type -Namespace ValidationNuit -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
[void][ValidationNuit.Power]::SetThreadExecutionState([uint32]2147483649)
$python = (Resolve-Path '..\.venv-mds\Scripts\python.exe').Path
& $python -u validation_experiments\night.py --start-by '2026-11-13 09:00' --twin-end '2026-11-13 15:30' --stop '2026-11-13 16:45' *> runs\validation_nuit.out
[void][ValidationNuit.Power]::SetThreadExecutionState([uint32]2147483648)
