# scripts/run-spike-windows.ps1
#
# Build and run a Phase 0 spike on native Windows. Source is
# robocopy-mirrored from the WSL repo to a Windows build directory so cargo
# doesn't fight UNC + target/ lockfile races.
#
# Per architecture/agent-dispatch.md, this is the user-side verification
# path for Phase 0 section 4 exit criteria: spike CLIs must run on Windows AND
# Linux. WSL-CPU is verified in WSL; Windows is verified here.
#
# Toolchain expectations (set near the top -- edit if your install paths
# differ):
#   - Rust on PATH (`cargo`, `rustc`)
#   - Visual Studio Build Tools 2022 (or full VS 2022) for MSVC
#   - LLVM (libclang.dll) for bindgen used by llama-cpp-sys-2
#   - Vulkan SDK (for the `vulkan` feature; CPU-only builds don't need it)
#
# Usage:
#   powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\run-spike-windows.ps1 -Spike asr
#   powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\run-spike-windows.ps1 -Spike asr -Run
#
# From WSL:
#   powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'C:\Users\anl\meeting-app\scripts\run-spike-windows.ps1' -Spike asr -Run
#
# (the script must already be on the Windows side, since the WSL UNC
#  copy isn't executable. Run a Sync first, then re-invoke.)

param(
    [Parameter(Mandatory)]
    [ValidateSet('asr', 'llm', 'vad-loop', 'diarize')]
    [string]$Spike,

    [string]$Features = '',

    [switch]$Run,

    [switch]$SyncOnly
)

$ErrorActionPreference = 'Stop'

# ----------------------------------------------------------------------
# Toolchain paths
# ----------------------------------------------------------------------
$env:LIBCLANG_PATH = 'C:\Program Files\LLVM\bin'
$env:VULKAN_SDK    = 'C:\VulkanSDK\1.4.341.1'

# ----------------------------------------------------------------------
# Sync WSL source to Windows
# ----------------------------------------------------------------------
$src   = '\\wsl.localhost\Ubuntu\home\anl\meeting-app'
$build = 'C:\Users\anl\meeting-app'

Write-Host "==> Syncing $src -> $build"
$rcArgs = @($src, $build, '/MIR', '/XD', 'target', '.git', '.claude', '/NFL', '/NDL', '/NP', '/NJH', '/NJS')
$null = & robocopy.exe @rcArgs
# robocopy exit codes 0-7 are success-ish; 8+ are real errors
if ($LASTEXITCODE -ge 8) { throw "robocopy failed (exit $LASTEXITCODE)" }

if ($SyncOnly) {
    Write-Host "==> SyncOnly requested; exiting."
    return
}

# ----------------------------------------------------------------------
# MSVC environment via VS Build Tools dev shell
# ----------------------------------------------------------------------
$vswhere = 'C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe'
if (-not (Test-Path $vswhere)) { throw "vswhere.exe not found at $vswhere" }

# `-products *` is required to also match Visual Studio Build Tools
# (vswhere's default filter is the IDE products only).
$vsPath = & $vswhere -latest -products * -property installationPath
if (-not $vsPath) {
    $fallback = 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools'
    if (Test-Path $fallback) { $vsPath = $fallback }
    else { throw "no VS install reported by vswhere and fallback path missing: $fallback" }
}

Write-Host "==> MSVC env from $vsPath"
$null = & "$vsPath\Common7\Tools\Launch-VsDevShell.ps1" -Arch amd64 -SkipAutomaticLocation

Set-Location $build

# ----------------------------------------------------------------------
# Build
# ----------------------------------------------------------------------
$featureArg = if ($Features) { @('--features', $Features) } else { @() }
Write-Host ("==> cargo build --release -p spike-{0} {1}" -f $Spike, ($featureArg -join ' '))
& cargo build --release -p "spike-$Spike" @featureArg
if ($LASTEXITCODE -ne 0) { throw "cargo build failed (exit $LASTEXITCODE)" }

if (-not $Run) {
    Write-Host "==> Build OK. -Run not specified; not running."
    return
}

# ----------------------------------------------------------------------
# Per-spike runs (CPU-only by default; pass -Features vulkan to test GPU)
# ----------------------------------------------------------------------
$bin = "$build\target\release\spike-$Spike.exe"

switch ($Spike) {
    'asr' {
        Write-Host "==> Running spike-asr on librispeech_30s.wav (full 30 s window -- no truncation hallucination)"
        & $bin `
            --model  'C:\Users\anl\qwen3-asr-gguf\Qwen3-ASR-0.6B-Q8_0-ggml-org.gguf' `
            --mmproj 'C:\Users\anl\qwen3-asr-gguf\Qwen3-ASR-0.6B.mmproj-Q8_0.gguf' `
            --wav    'C:\Users\anl\transcribe-rs-test\fixtures\librispeech_30s.wav' `
            --max-seconds 30 `
            --threads 8
    }
    'llm' {
        Write-Host "==> Running spike-llm with cached Qwen2.5-3B (if present)"
        $qwen = Get-ChildItem 'C:\Users\anl\.cache\huggingface\hub' -Recurse -Filter '*Qwen2.5-3B-Instruct*Q4_K_M*.gguf' -ErrorAction SilentlyContinue | Select-Object -First 1
        if (-not $qwen) { throw "Qwen2.5-3B-Instruct-Q4_K_M.gguf not found in HF cache. Run spike-llm manually with a different model path." }
        & $bin --model $qwen.FullName --threads 8
    }
    'vad-loop' {
        Write-Host "==> Running spike-vad-loop on librispeech_30s.wav"
        & $bin `
            --vad        '\\wsl.localhost\Ubuntu\home\anl\Handy\src-tauri\resources\models\silero_vad_v4.onnx' `
            --asr-model  'C:\Users\anl\qwen3-asr-gguf\Qwen3-ASR-0.6B-Q8_0-ggml-org.gguf' `
            --mmproj     'C:\Users\anl\qwen3-asr-gguf\Qwen3-ASR-0.6B.mmproj-Q8_0.gguf' `
            --wav        'C:\Users\anl\transcribe-rs-test\fixtures\librispeech_30s.wav'
    }
    'diarize' {
        Write-Host "==> spike-diarize run flow not pre-configured (needs sherpa-onnx model + fixture cache on Windows). Skipping."
    }
}

if ($LASTEXITCODE -ne 0) { throw "spike-$Spike exited with $LASTEXITCODE" }
Write-Host "==> spike-$Spike completed."
