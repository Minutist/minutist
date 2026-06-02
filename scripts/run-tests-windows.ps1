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
# Exclude heavy/transient dirs. `/XD <name>` matches that dir name at ANY
# depth, so nested `target` and `node_modules` are all skipped. node_modules
# is thousands of files and murders the 9P/UNC copy if not excluded.
$rcArgs = @($src, $build, '/MIR',
    '/XD', 'target', '.git', '.claude', 'node_modules', 'dist',
    '/NFL', '/NDL', '/NP', '/NJH', '/NJS', '/R:1', '/W:1')
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
# summariser external-ollama feature build (FR-29)
# ---------------------------------------------------------------------------
# The `external-ollama` dispatcher is off by default, so a plain
# `cargo test -p summariser` never COMPILES it and its unit tests never run.
# When testing the summariser, also run the feature build so the ollama
# URL/serde/status-mapping tests are exercised in the gated suite. These are
# fast, non-#[ignore]'d, no-live-server tests, so they always run (no
# `--ignored` / `--test-threads=1` gating needed). Tracked separately so its
# exit code folds into the final result.
$ollamaCode = 0
if ($Package -eq 'summariser') {
    $ollamaArgs = @('test', '-p', 'summariser', '--features', 'external-ollama')
    if ($Release) { $ollamaArgs += '--release' }
    $ollamaArgs += @('--', '--nocapture')
    Write-Host ("==> cargo " + ($ollamaArgs -join ' '))
    $ErrorActionPreference = 'Continue'
    & cargo @ollamaArgs 2>&1
    $ollamaCode = $LASTEXITCODE
    $ErrorActionPreference = 'Stop'
    Write-Host "==> summariser external-ollama test exit $ollamaCode"
}

# ---------------------------------------------------------------------------
# Run tests
# ---------------------------------------------------------------------------
$cargoArgs = @('test', '-p', $Package)
if ($Release) { $cargoArgs += '--release' }
if ($Features) { $cargoArgs += @('--features', $Features) }
$cargoArgs += '--'
if ($Ignored) { $cargoArgs += '--ignored' }
# Gated model tests each load a full ~805 MB GGUF + ~448 MB KV + ~324 MB
# compute buffer. Cargo's default parallel test execution makes two of them
# co-reside and OOM the CPU backend's buffer allocator. Force serial.
$cargoArgs += @('--nocapture', '--test-threads=1')

Write-Host ("==> cargo " + ($cargoArgs -join ' '))
Write-Host "==> MODEL  = $env:MEETING_APP_ASR_MODEL_PATH"
Write-Host "==> MMPROJ = $env:MEETING_APP_ASR_MMPROJ_PATH"
# cargo writes progress to stderr; with ErrorActionPreference=Stop PowerShell
# treats the first stderr line as a terminating NativeCommandError and aborts
# the build. Drop to Continue for the cargo invocation and gate on the exit
# code instead. Redirect stderr→stdout so everything is captured in order.
$ErrorActionPreference = 'Continue'
& cargo @cargoArgs 2>&1
$code = $LASTEXITCODE
$ErrorActionPreference = 'Stop'
Write-Host "==> cargo test exit $code"
# Fold in the summariser external-ollama feature run: fail if either run failed.
if ($code -eq 0 -and $ollamaCode -ne 0) { $code = $ollamaCode }
exit $code
