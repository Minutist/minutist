# scripts/verify-windows-zip.ps1
# Extract the portable zip to a clean folder (NOT target\release) and launch it,
# to prove the artifact is self-contained: DLLs load and the Silero resource
# resolves relative to the UNZIPPED exe, then kill it.
#
# Usage (from WSL):
#   powershell.exe -NoProfile -ExecutionPolicy Bypass `
#     -File "$(wslpath -w scripts/verify-windows-zip.ps1)"
#
#   # Override build dir or zip path (when build-windows-app.ps1 -BuildDir was set):
#   powershell.exe ... -BuildDir 'C:\dev\minutist'
param(
    # Windows-side mirror/build directory; must match what build-windows-app.ps1 used.
    [string]$BuildDir = 'C:\Users\anl\meeting-app',
    # Zip file to extract + smoke-test; derived from BuildDir when not set.
    [string]$Zip      = '',
    # Temp extraction directory for the self-contained smoke run.
    [string]$SmokeDir = ''
)
$ErrorActionPreference = 'Continue'
if (-not $Zip)      { $Zip      = Join-Path $BuildDir 'dist-windows\minutist-windows-x64.zip' }
if (-not $SmokeDir) { $SmokeDir = Join-Path (Split-Path $BuildDir -Parent) 'mapp-zip-smoke' }
if (-not (Test-Path $Zip)) { Write-Host ("MISSING: " + $Zip); exit 1 }
if (Test-Path $SmokeDir) { Remove-Item -Recurse -Force $SmokeDir }
Expand-Archive -Path $Zip -DestinationPath $SmokeDir
$exe = Join-Path $SmokeDir 'minutist.exe'
Write-Host ("Running unzipped: " + $exe)
$p = Start-Process $exe -PassThru
Start-Sleep -Seconds 14
Write-Host ("alive after 14s: " + (-not $p.HasExited))
if ($p.HasExited) { Write-Host ("EXIT CODE: " + $p.ExitCode) }
$logdir = Join-Path $env:APPDATA 'ai.minutist\logs'
$log = Get-ChildItem (Join-Path $logdir 'minutist.log*') -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime | Select-Object -Last 1
if ($log) { Write-Host ("LOG: " + $log.Name); Get-Content $log.FullName -Tail 16 }
& taskkill /PID $p.Id /T /F 2>&1 | Out-Null
Remove-Item -Recurse -Force $SmokeDir -ErrorAction SilentlyContinue
Write-Host "cleaned up"
