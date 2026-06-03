# scripts/smoke-windows.ps1
# Launch the built Windows app briefly, capture the startup log, then kill it.
# A smoke test: confirms the assembled app boots (does not crash, reaches setup
# complete, the webview's on-mount IPC is permitted by the capability ACL).
param([string]$Exe = 'C:\Users\anl\meeting-app\target\release\meeting-app.exe')
$ErrorActionPreference = 'Continue'
$exe = $Exe
if (-not (Test-Path $exe)) { Write-Host ("MISSING: " + $exe); exit 1 }
Write-Host ("Launching " + $exe)
$p = Start-Process $exe -PassThru
Start-Sleep -Seconds 14
Write-Host ("process alive after 14s: " + (-not $p.HasExited))
if ($p.HasExited) { Write-Host ("EXIT CODE: " + $p.ExitCode) }
$logdir = Join-Path $env:APPDATA 'net.alelec.meeting-app\logs'
$log = Get-ChildItem (Join-Path $logdir 'meeting-app.log*') -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime | Select-Object -Last 1
if ($log) {
    Write-Host ("LOG: " + $log.FullName)
    Get-Content $log.FullName -Tail 35
} else {
    Write-Host ("NO LOG FILE in " + $logdir)
}
& taskkill /PID $p.Id /T /F 2>&1 | Out-Null
Write-Host "cleaned up"
