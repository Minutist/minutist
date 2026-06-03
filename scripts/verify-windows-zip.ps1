# scripts/verify-windows-zip.ps1
# Extract the portable zip to a clean folder (NOT target\release) and launch it,
# to prove the artifact is self-contained: DLLs load and the Silero resource
# resolves relative to the UNZIPPED exe, then kill it.
$ErrorActionPreference = 'Continue'
$zip  = 'C:\Users\anl\meeting-app\dist-windows\meeting-app-windows-x64.zip'
$dest = 'C:\Users\anl\mapp-zip-smoke'
if (-not (Test-Path $zip)) { Write-Host ("MISSING: " + $zip); exit 1 }
if (Test-Path $dest) { Remove-Item -Recurse -Force $dest }
Expand-Archive -Path $zip -DestinationPath $dest
$exe = Join-Path $dest 'meeting-app.exe'
Write-Host ("Running unzipped: " + $exe)
$p = Start-Process $exe -PassThru
Start-Sleep -Seconds 14
Write-Host ("alive after 14s: " + (-not $p.HasExited))
if ($p.HasExited) { Write-Host ("EXIT CODE: " + $p.ExitCode) }
$logdir = Join-Path $env:APPDATA 'net.alelec.meeting-app\logs'
$log = Get-ChildItem (Join-Path $logdir 'meeting-app.log*') -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime | Select-Object -Last 1
if ($log) { Write-Host ("LOG: " + $log.Name); Get-Content $log.FullName -Tail 16 }
& taskkill /PID $p.Id /T /F 2>&1 | Out-Null
Remove-Item -Recurse -Force $dest -ErrorAction SilentlyContinue
Write-Host "cleaned up"
