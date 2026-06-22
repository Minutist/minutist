# scripts/run-doc-vlm-spike-windows.ps1
#
# Build + RUN the doc-vlm benchmark spike (spikes/doc-vlm-spike) natively on
# Windows with the Vulkan GPU backend, replicating the env from
# build-windows-app.ps1 (MSVC VsDevShell, LLVM/libclang, Vulkan SDK, short
# CARGO_TARGET_DIR, Ninja for the feature build). The spike self-acquires the
# Gemma-4-E4B + PaddleOCR-VL-1.6 GGUFs and PDFium, renders synthetic fixtures,
# runs both models, and prints a CER/latency comparison. This does NOT build the
# app -- it only builds + runs the spike binary.
#
# Usage (from WSL):
#   powershell.exe -NoProfile -ExecutionPolicy Bypass `
#     -File "$(wslpath -w scripts/run-doc-vlm-spike-windows.ps1)"

param(
    [string]$WslSrc      = '\\wsl.localhost\Ubuntu\home\anl\meeting-app',
    [string]$BuildDir    = 'C:\Users\anl\meeting-app',
    [string]$Features    = 'vulkan',
    [string[]]$SpikeArgs = @()
)

$ErrorActionPreference = 'Stop'

$env:LIBCLANG_PATH = 'C:\Program Files\LLVM\bin'
$env:VULKAN_SDK    = 'C:\VulkanSDK\1.4.341.1'

$src   = $WslSrc
$build = $BuildDir

Set-Location (Split-Path $build -Parent)
Write-Host "==> Mirroring $src -> $build (incremental; target/ kept)"
$rcArgs = @($src, $build, '/MIR',
    '/XD', 'target', '.git', '.claude', 'node_modules', 'dist-windows',
    '/NFL', '/NDL', '/NP', '/NJH', '/NJS', '/R:1', '/W:1')
$null = & robocopy.exe @rcArgs
if ($LASTEXITCODE -ge 8) { throw "robocopy failed (exit $LASTEXITCODE)" }

# MSVC environment (same derivation as build-windows-app.ps1).
$vswhere = 'C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe'
if (-not (Test-Path $vswhere)) { throw "vswhere.exe not found at $vswhere" }
$vsPath = & $vswhere -latest -products * -property installationPath
if (-not $vsPath) {
    $fallback = 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools'
    if (Test-Path $fallback) { $vsPath = $fallback } else { throw "no VS install" }
}
Write-Host "==> MSVC env from $vsPath"
$null = & "$vsPath\Common7\Tools\Launch-VsDevShell.ps1" -Arch amd64 -SkipAutomaticLocation

# Dedup PATH (VsDevShell can exceed cmd.exe's 8191-char limit; see build script).
$env:PATH = (($env:PATH -split ';') | Where-Object { $_ -ne '' } | Select-Object -Unique) -join ';'
Write-Host ("==> PATH length after dedup: " + $env:PATH.Length)

# Short target dir (avoids rc.exe MAX_PATH in nested CMake) + share the app's
# Vulkan build cache so llama-cpp-sys-2 is not recompiled if already built.
$env:CARGO_TARGET_DIR = 'C:\mt'
if ($Features) {
    $env:CMAKE_GENERATOR = 'Ninja'
    Write-Host "==> CMAKE_GENERATOR=Ninja (feature build)"
}

Set-Location $build

$cargoArgs = @('run', '-p', 'spike-doc-vlm', '--release')
if ($Features)            { $cargoArgs += @('--features', $Features) }
if ($SpikeArgs.Count -gt 0) { $cargoArgs += '--'; $cargoArgs += $SpikeArgs }
Write-Host ("==> cargo " + ($cargoArgs -join ' '))

$ErrorActionPreference = 'Continue'
& cargo @cargoArgs 2>&1
$code = $LASTEXITCODE
Write-Host "==> spike exit $code"
exit $code
