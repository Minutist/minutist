# scripts/build-windows-app.ps1
#
# Build a PORTABLE, UNSIGNED native Windows build of minutist for local
# testing. Produces target\release\minutist.exe with the frontend embedded
# and the sherpa-onnx + onnxruntime DLLs beside it (Windows resolves same-dir
# DLLs at runtime), then stages + zips the runnable folder.
#
# This sidesteps the installer/.dll-bundling gap (an MSI/NSIS would not auto-
# include the sherpa DLLs): a run-from-folder build just keeps them adjacent.
#
# Source is robocopy-mirrored from WSL INCLUDING the prebuilt ui/dist, because
# Node is not installed on the Windows side - build the frontend in WSL first
# (`cd ui && npm run build`). Companion to run-tests-windows.ps1.
#
# Usage (from WSL via make):
#   make windows-build          # derives WslSrc + BuildDir from the Makefile vars
#   make windows-build-vulkan
#
# Usage (direct, from WSL):
#   powershell.exe -NoProfile -ExecutionPolicy Bypass `
#     -File "$(wslpath -w scripts/build-windows-app.ps1)"
#   ... -SyncOnly                              # validate sync only
#   ... -Features vulkan                       # Vulkan GPU build
#   ... -WslSrc '\\wsl.localhost\Arch\repo'   # override WSL source UNC
#   ... -BuildDir 'C:\dev\minutist'            # override Windows build dir
#
# Toolchain: Rust on PATH, VS Build Tools 2022 (MSVC), LLVM (libclang). WebView2
# runtime must be present to RUN (Evergreen ships on Win10/11).

param(
    [switch]$SyncOnly,
    [string]$Features  = '',
    # UNC path to the WSL checkout as seen from Windows.  The Makefile derives
    # this via wslpath; override when running the script directly.
    [string]$WslSrc    = '\\wsl.localhost\Ubuntu\home\anl\meeting-app',
    # Windows-side mirror directory; cargo builds here.
    [string]$BuildDir  = 'C:\Users\anl\meeting-app',
    # Cargo target/cache dir (CARGO_TARGET_DIR). MUST be a short root — rc.exe's
    # 260-char MAX_PATH. Empty => derived from BuildDir (see below): the canonical
    # live-test BuildDir keeps the shared 'C:\mt'; any other worktree gets its own
    # short cache so concurrent multi-worktree builds cannot poison each other's
    # crate rlibs. Override explicitly to pin a specific cache.
    [string]$TargetDir = '',
    # Short git SHA of the source, captured in WSL (the mirror excludes .git, so
    # build.rs cannot derive it here). Embedded into the build for the title bar
    # and diagnostic reports. Empty → build.rs records "unknown".
    [string]$GitSha    = ''
)

$ErrorActionPreference = 'Stop'

$env:LIBCLANG_PATH = 'C:\Program Files\LLVM\bin'
$env:VULKAN_SDK    = 'C:\VulkanSDK\1.4.341.1'

$src   = $WslSrc
$build = $BuildDir

Set-Location (Split-Path $build -Parent)
Write-Host "==> Syncing $src -> $build (including ui\dist; Node is absent on Windows)"
# Keep `target` excluded so the Windows incremental build cache survives; keep
# node_modules/.git/.claude out. NOTE: `dist` is NOT excluded here (unlike the
# test script) so the prebuilt ui\dist is mirrored and embedded by the app.
$rcArgs = @($src, $build, '/MIR',
    '/XD', 'target', '.git', '.claude', 'node_modules', 'dist-windows',
    '/NFL', '/NDL', '/NP', '/NJH', '/NJS', '/R:1', '/W:1')
$null = & robocopy.exe @rcArgs
if ($LASTEXITCODE -ge 8) { throw "robocopy failed (exit $LASTEXITCODE)" }

if (-not (Test-Path "$build\ui\dist\index.html")) {
    throw "ui\dist\index.html missing on the Windows side - run 'cd ui && npm run build' in WSL first, then re-run."
}

if ($SyncOnly) { Write-Host "==> SyncOnly; ui\dist present; done."; return }

# MSVC environment
$vswhere = 'C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe'
if (-not (Test-Path $vswhere)) { throw "vswhere.exe not found at $vswhere" }
$vsPath = & $vswhere -latest -products * -property installationPath
if (-not $vsPath) {
    $fallback = 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools'
    if (Test-Path $fallback) { $vsPath = $fallback } else { throw "no VS install" }
}
Write-Host "==> MSVC env from $vsPath"
$null = & "$vsPath\Common7\Tools\Launch-VsDevShell.ps1" -Arch amd64 -SkipAutomaticLocation

# Dedup PATH. VsDevShell can push PATH past cmd.exe's ~8191-char limit, which
# truncates the vcvars batch CMake generates for nested ExternalProject custom
# rules (the Vulkan vulkan-shaders-gen install step) and produces "cannot find
# batch label VCEnd". Deduping the (often duplicate-heavy) PATH shrinks it.
$env:PATH = (($env:PATH -split ';') | Where-Object { $_ -ne '' } | Select-Object -Unique) -join ';'
Write-Host ("==> PATH length after dedup: " + $env:PATH.Length)

# Build under a short target dir. The Vulkan feature build's llama-cpp-sys-2
# nested ExternalProject (vulkan-shaders-gen) runs a CMake compiler-detection
# try_compile; CMake 3.31's CMakeScratch\TryCompile-<id>\CMakeFiles\cmTC_<id>.dir
# scratch layout, under the already-deep target\release\build\<pkg>-<hash>\out\
# build\ggml\src\ggml-vulkan\... tree, pushes the generated manifest.rc path past
# rc.exe's 260-char MAX_PATH (rc.exe does not honour the LongPathsEnabled opt-in)
# and fails with "RC2136: missing '=' in EXSTYLE". A short root keeps every path
# well under 260.
#
# Per-worktree isolation: the canonical live-test BuildDir keeps 'C:\mt' (its warm
# cache + the run-path the user launches). Any OTHER worktree's BuildDir derives a
# short unique cache ('C:\mt-<4hex>' of the BuildDir) so two worktrees building
# concurrently cannot reuse each other's stale crate rlibs (robocopy preserves
# source mtimes, so cargo would otherwise trust a sibling's newer artifact and
# emit phantom cross-crate errors). An explicit -TargetDir wins over the derivation.
if (-not $TargetDir) {
    if ($BuildDir -eq 'C:\Users\anl\meeting-app') {
        $TargetDir = 'C:\mt'
    } else {
        $md5  = [System.Security.Cryptography.MD5]::Create()
        $hash = [System.BitConverter]::ToString(
                    $md5.ComputeHash([System.Text.Encoding]::UTF8.GetBytes($BuildDir))
                ).Replace('-', '').Substring(0, 4).ToLower()
        $TargetDir = "C:\mt-$hash"
    }
}
$env:CARGO_TARGET_DIR = $TargetDir
Write-Host "==> CARGO_TARGET_DIR=$env:CARGO_TARGET_DIR (short path; avoids rc.exe MAX_PATH in nested CMake)"

# Pass the WSL-captured git SHA to build.rs (the mirror excludes .git).
if ($GitSha) {
    $env:MINUTIST_GIT_SHA = $GitSha
    Write-Host "==> MINUTIST_GIT_SHA=$GitSha"
}

Set-Location $build

# GPU/feature builds: use the Ninja generator. llama.cpp's vulkan-shaders-gen
# nested ExternalProject fails under the MSBuild VS generator on Windows
# ("could not load cache" / "cannot find batch label VCEnd"); Ninja sidesteps
# the .vcxproj custom-build-rule machinery. Requires ninja + cl.exe on PATH
# (the VsDevShell above provides cl.exe; ninja is in ~\.cargo\bin). The
# CPU-default build keeps the VS generator (known-good), so this is gated.
if ($Features) {
    $env:CMAKE_GENERATOR = 'Ninja'
    Write-Host "==> CMAKE_GENERATOR=Ninja (feature build)"
}

# Stop any running instance first. A live process (e.g. a smoke test or a
# left-open window) locks target\release\minutist.exe, so the relink fails
# with "failed to remove ...minutist.exe: Access is denied (os error 5)".
# WebView2 child processes exit with the parent.
Get-Process -Name minutist -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Milliseconds 500

# Build the PRODUCTION app via the Tauri CLI (cargo-tauri), NOT a bare
# `cargo build` -- the latter leaves Tauri's `dev` cfg on, so the webview loads
# devUrl (http://localhost:5173) and shows "localhost refused to connect" with
# no dev server. `tauri build` produces a prod binary that serves the embedded
# frontend via the tauri:// protocol. --no-bundle: just the exe + DLLs in
# target\release (run-from-folder finds adjacent DLLs), no installer -> sidesteps
# the .dll-bundling gap. beforeBuildCommand is overridden to "" because Node is
# absent on Windows and ui\dist is prebuilt in WSL + synced.
# Write the beforeBuildCommand override to a temp config FILE and pass its path
# to -c: inline JSON on the cargo command line gets its double-quotes stripped by
# PowerShell's native-argument handling ("invalid value ... key must be a
# string"). A file path has no quotes to mangle. ASCII, no BOM.
$tauriOverride = Join-Path $env:TEMP 'mapp-tauri-override.json'
[System.IO.File]::WriteAllText($tauriOverride, '{ "build": { "beforeBuildCommand": "" } }')
$cargoArgs = @('tauri', 'build', '--no-bundle', '-c', $tauriOverride)
if ($Features) { $cargoArgs += @('--features', $Features) }
Write-Host ("==> cargo " + ($cargoArgs -join ' '))
$ErrorActionPreference = 'Continue'
& cargo @cargoArgs 2>&1
$code = $LASTEXITCODE
$ErrorActionPreference = 'Stop'
Write-Host "==> cargo build exit $code"
if ($code -ne 0) { exit $code }

$rel = "$env:CARGO_TARGET_DIR\release"
$exe = "$rel\minutist.exe"
if (-not (Test-Path $exe)) { throw "build succeeded but $exe is missing" }

Write-Host "==> Built: $exe"
Get-ChildItem "$rel\*.dll" -ErrorAction SilentlyContinue | ForEach-Object {
    Write-Host ("    dll: {0}  ({1:N1} MB)" -f $_.Name, ($_.Length / 1MB))
}

# Stage the runnable artifact (exe + adjacent DLLs) and zip it. A feature build
# gets a suffix so the CPU and GPU artifacts coexist (e.g. ...-x64-vulkan.zip).
$suffix = ''
if ($Features) { $suffix = '-' + ($Features -replace '[, ]+', '-') }
$stage = "$build\dist-windows\minutist$suffix"
if (Test-Path $stage) { Remove-Item -Recurse -Force $stage }
New-Item -ItemType Directory -Force -Path $stage | Out-Null
Copy-Item $exe $stage
Copy-Item "$rel\*.dll" $stage -ErrorAction SilentlyContinue
# Bundled resources (the Silero VAD model) live under target\release\_up_\ at the
# same relpath the app resolves via BaseDirectory::Resource; include the tree so
# the portable zip is self-contained (otherwise VAD/ASR fails on a fresh unzip).
if (Test-Path "$rel\_up_") {
    Copy-Item "$rel\_up_" $stage -Recurse -Force
    Write-Host "    bundled: _up_\resources (Silero VAD)"
}

$zip = "$build\dist-windows\minutist-windows-x64$suffix.zip"
if (Test-Path $zip) { Remove-Item -Force $zip }
Compress-Archive -Path "$stage\*" -DestinationPath $zip
$zipMb = [math]::Round((Get-Item $zip).Length / 1MB, 1)
Write-Host ("==> Zipped: {0} ({1} MB)" -f $zip, $zipMb)
Write-Host "==> Done. Unzip and run minutist.exe (first run downloads the ASR/LLM models)."
