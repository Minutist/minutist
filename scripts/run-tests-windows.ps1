# scripts/run-tests-windows.ps1
#
# Build and run cargo tests on native Windows (MSVC), with the ASR model
# env vars pointed at the staged Qwen3-ASR Q8_0 GGUFs so the #[ignore]'d
# gated tests run. Companion to run-spike-windows.ps1.
#
# Source is robocopy-mirrored from the WSL repo to C:\Users\anl\meeting-app
# so cargo doesn't fight UNC + target/ lockfile races.
#
# Usage (from WSL, after copying this script to the Windows side):
#   powershell.exe -NoProfile -ExecutionPolicy Bypass `
#     -File 'C:\Users\anl\meeting-app\scripts\run-tests-windows.ps1' `
#     -Package asr-runtime -Ignored
#
#   powershell.exe ... -Package orchestrator -Features test-source -Ignored
#
# Toolchain: Rust on PATH, VS Build Tools 2022 (MSVC), LLVM (libclang for
# bindgen). Vulkan SDK is only needed when the build enables a vulkan feature
# (Phase 7); the default CPU build does not require it.

param(
    [Parameter(Mandatory)]
    [string]$Package,

    [string]$Features = '',

    [switch]$Ignored,

    [switch]$Release,

    [switch]$SyncOnly
)

$ErrorActionPreference = 'Stop'

# ---------------------------------------------------------------------------
# Toolchain + model paths
# ---------------------------------------------------------------------------
$env:LIBCLANG_PATH = 'C:\Program Files\LLVM\bin'
$env:VULKAN_SDK    = 'C:\VulkanSDK\1.4.341.1'

# Staged Qwen3-ASR Q8_0 model files (from prior transcribe-rs work).
$env:MEETING_APP_ASR_MODEL_PATH  = 'C:\Users\anl\qwen3-asr-gguf\Qwen3-ASR-0.6B-Q8_0-ggml-org.gguf'
$env:MEETING_APP_ASR_MMPROJ_PATH = 'C:\Users\anl\qwen3-asr-gguf\Qwen3-ASR-0.6B.mmproj-Q8_0.gguf'

foreach ($p in @($env:MEETING_APP_ASR_MODEL_PATH, $env:MEETING_APP_ASR_MMPROJ_PATH)) {
    if (-not (Test-Path $p)) { throw "Model file not found: $p" }
}

# ---------------------------------------------------------------------------
# Sync WSL source to Windows
# ---------------------------------------------------------------------------
$src   = '\\wsl.localhost\Ubuntu\home\anl\meeting-app'
$build = 'C:\Users\anl\meeting-app'

Set-Location C:\Users\anl
Write-Host "==> Syncing $src -> $build"
$rcArgs = @($src, $build, '/MIR', '/XD', 'target', '.git', '.claude', '/NFL', '/NDL', '/NP', '/NJH', '/NJS')
$null = & robocopy.exe @rcArgs
if ($LASTEXITCODE -ge 8) { throw "robocopy failed (exit $LASTEXITCODE)" }

if ($SyncOnly) { Write-Host "==> SyncOnly; done."; return }

# ---------------------------------------------------------------------------
# MSVC environment
# ---------------------------------------------------------------------------
$vswhere = 'C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe'
if (-not (Test-Path $vswhere)) { throw "vswhere.exe not found at $vswhere" }
$vsPath = & $vswhere -latest -products * -property installationPath
if (-not $vsPath) {
    $fallback = 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools'
    if (Test-Path $fallback) { $vsPath = $fallback } else { throw "no VS install" }
}
Write-Host "==> MSVC env from $vsPath"
$null = & "$vsPath\Common7\Tools\Launch-VsDevShell.ps1" -Arch amd64 -SkipAutomaticLocation

Set-Location $build

# ---------------------------------------------------------------------------
# Run tests
# ---------------------------------------------------------------------------
$cargoArgs = @('test', '-p', $Package)
if ($Release) { $cargoArgs += '--release' }
if ($Features) { $cargoArgs += @('--features', $Features) }
$cargoArgs += '--'
if ($Ignored) { $cargoArgs += '--ignored' }
$cargoArgs += '--nocapture'

Write-Host ("==> cargo " + ($cargoArgs -join ' '))
Write-Host "==> MODEL  = $env:MEETING_APP_ASR_MODEL_PATH"
Write-Host "==> MMPROJ = $env:MEETING_APP_ASR_MMPROJ_PATH"
& cargo @cargoArgs
$code = $LASTEXITCODE
Write-Host "==> cargo test exit $code"
exit $code
