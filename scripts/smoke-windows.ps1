# scripts/smoke-windows.ps1
# Launch the built Windows app briefly, capture the startup log, then kill it.
# A smoke test: confirms the assembled app boots (does not crash, reaches setup
# complete, the webview's on-mount IPC is permitted by the capability ACL).
#
# Usage (from WSL):
#   powershell.exe -NoProfile -ExecutionPolicy Bypass `
#     -File "$(wslpath -w scripts/smoke-windows.ps1)"
#
#   # Override build dir (matches what build-windows-app.ps1 -BuildDir set):
#   powershell.exe ... -BuildDir 'C:\dev\minutist'
param(
    # Windows-side mirror/build directory; must match what build-windows-app.ps1 used.
    [string]$BuildDir = 'C:\Users\anl\meeting-app',
    # Full path to the built exe; derived from BuildDir when not set.
    [string]$Exe      = ''
)
$ErrorActionPreference = 'Continue'
if (-not $Exe) { $Exe = Join-Path $BuildDir 'target\release\minutist.exe' }
if (-not (Test-Path $Exe)) { Write-Host ("MISSING: " + $Exe); exit 1 }
Write-Host ("Launching " + $Exe)
$p = Start-Process $Exe -PassThru
Start-Sleep -Seconds 14
Write-Host ("process alive after 14s: " + (-not $p.HasExited))
if ($p.HasExited) { Write-Host ("EXIT CODE: " + $p.ExitCode) }
$logdir = Join-Path $env:APPDATA 'ai.minutist\logs'
$log = Get-ChildItem (Join-Path $logdir 'minutist.log*') -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime | Select-Object -Last 1
if ($log) {
    Write-Host ("LOG: " + $log.FullName)
    Get-Content $log.FullName -Tail 35
} else {
    Write-Host ("NO LOG FILE in " + $logdir)
}
& taskkill /PID $p.Id /T /F 2>&1 | Out-Null
Write-Host "cleaned up"
