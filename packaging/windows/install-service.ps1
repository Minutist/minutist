#Requires -RunAsAdministrator
<#
.SYNOPSIS
  Install the Minutist headless sync hub (minutist-hub.exe) as a Windows service.

.DESCRIPTION
  Registers the daemon as a real Windows service via WinSW (the SCM wrapper), so
  it auto-starts at boot and is managed from services.msc / `sc`. The daemon
  itself is platform-neutral; WinSW handles the SCM lifecycle and sends Ctrl+C on
  stop, which the daemon's Ctrl-C handler turns into a graceful drain.

  Pair a desktop after install:
    & "$BinaryPath" --data-dir "$DataDir" print-ticket
    & "$BinaryPath" --data-dir "$DataDir" add-peer <desktop-ticket>
  (the running service re-reads its peers file, so add-peer needs no restart.)

.EXAMPLE
  .\install-service.ps1 -BinaryPath C:\minutist\minutist-hub.exe -RelayToken xxxx
#>
param(
    [Parameter(Mandatory)][string]$BinaryPath,                         # minutist-hub.exe
    [Parameter(Mandatory)][string]$RelayToken,                         # MINUTIST_SYNC_TOKEN
    [string]$DataDir    = "$env:ProgramData\minutist-hub",
    [string]$InstallDir = "$env:ProgramFiles\minutist-hub",
    [string]$WinSwUrl   = "https://github.com/winsw/winsw/releases/download/v3.0.0-alpha.11/WinSW-x64.exe"
)

$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

if (-not (Test-Path $BinaryPath)) { throw "minutist-hub.exe not found at $BinaryPath" }

New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
New-Item -ItemType Directory -Force -Path $DataDir    | Out-Null

# Stage the daemon binary and WinSW (named so WinSW finds its own .xml config).
Copy-Item -Force $BinaryPath (Join-Path $InstallDir "minutist-hub.exe")
$winsw = Join-Path $InstallDir "minutist-hub-service.exe"
if (-not (Test-Path $winsw)) {
    Write-Host "Downloading WinSW -> $winsw"
    Invoke-WebRequest -Uri $WinSwUrl -OutFile $winsw
}

# Render the service config from the template.
$template = Get-Content -Raw (Join-Path $scriptDir "minutist-hub-service.xml")
$config = $template `
    -replace '@BINARY@',     [System.Security.SecurityElement]::Escape((Join-Path $InstallDir "minutist-hub.exe")) `
    -replace '@DATA_DIR@',   [System.Security.SecurityElement]::Escape($DataDir) `
    -replace '@RELAY_TOKEN@',[System.Security.SecurityElement]::Escape($RelayToken)
$configPath = Join-Path $InstallDir "minutist-hub-service.xml"
Set-Content -Path $configPath -Value $config -Encoding UTF8

# The config holds the relay token — restrict it to SYSTEM + Administrators.
icacls $configPath /inheritance:r /grant:r "SYSTEM:(R)" "Administrators:(F)" | Out-Null

& $winsw install
& $winsw start
Write-Host "Installed and started service 'minutist-hub'. Data dir: $DataDir"
Write-Host "Next: pair a desktop with"
Write-Host "  & '$(Join-Path $InstallDir "minutist-hub.exe")' --data-dir '$DataDir' print-ticket"
