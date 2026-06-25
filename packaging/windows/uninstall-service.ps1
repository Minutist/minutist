#Requires -RunAsAdministrator
<#
.SYNOPSIS
  Stop and remove the Minutist sync hub Windows service.
.DESCRIPTION
  Stops + unregisters the WinSW service. Leaves the data directory (device key,
  peers, meetings) in place by default; pass -PurgeData to remove it too.
#>
param(
    [string]$InstallDir = "$env:ProgramFiles\minutist-hub",
    [string]$DataDir    = "$env:ProgramData\minutist-hub",
    [switch]$PurgeData
)

$ErrorActionPreference = "Stop"
$winsw = Join-Path $InstallDir "minutist-hub-service.exe"
if (Test-Path $winsw) {
    & $winsw stop
    & $winsw uninstall
} else {
    Write-Warning "WinSW wrapper not found at $winsw; attempting 'sc delete minutist-hub'"
    sc.exe delete minutist-hub | Out-Null
}

Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $InstallDir
if ($PurgeData) {
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $DataDir
    Write-Host "Removed service and data dir $DataDir"
} else {
    Write-Host "Removed service. Data dir kept at $DataDir (pass -PurgeData to delete)."
}
