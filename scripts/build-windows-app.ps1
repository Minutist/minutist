# scripts/build-windows-app.ps1
#
# Build a PORTABLE, UNSIGNED native Windows build of meeting-app for local
# testing. Produces target\release\meeting-app.exe with the frontend embedded
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
# Usage (from WSL):
#   powershell.exe -NoProfile -ExecutionPolicy Bypass `
#     -File '\\wsl.localhost\Ubuntu\home\anl\meeting-app\scripts\build-windows-app.ps1'
#   ... -SyncOnly        # validate the sync without building
#   ... -Features vulkan # GPU build (needs the Vulkan SDK + a Vulkan GPU to run)
#
# Toolchain: Rust on PATH, VS Build Tools 2022 (MSVC), LLVM (libclang). WebView2
# runtime must be present to RUN (Evergreen ships on Win10/11).

param(
    [switch]$SyncOnly,
    [string]$Features = ''
)

$ErrorActionPreference = 'Stop'

$env:LIBCLANG_PATH = 'C:\Program Files\LLVM\bin'
$env:VULKAN_SDK    = 'C:\VulkanSDK\1.4.341.1'

$src   = '\\wsl.localhost\Ubuntu\home\anl\meeting-app'
$build = 'C:\Users\anl\meeting-app'

Set-Location C:\Users\anl
Write-Host "==> Syncing $src -> $build (including ui\dist; Node is absent on Windows)"
# Keep `target` excluded so the Windows incremental build cache survives; keep
# node_modules/.git/.claude out. NOTE: `dist` is NOT excluded here (unlike the
# test script) so the prebuilt ui\dist is mirrored and embedded by the app.
$rcArgs = @($src, $build, '/MIR',
    '/XD', 'target', '.git', '.claude', 'node_modules',
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

# Build the app binary (release). CPU-default unless -Features given.
$cargoArgs = @('build', '--release', '--bin', 'meeting-app')
if ($Features) { $cargoArgs += @('--features', $Features) }
Write-Host ("==> cargo " + ($cargoArgs -join ' '))
$ErrorActionPreference = 'Continue'
& cargo @cargoArgs 2>&1
$code = $LASTEXITCODE
$ErrorActionPreference = 'Stop'
Write-Host "==> cargo build exit $code"
if ($code -ne 0) { exit $code }

$rel = "$build\target\release"
$exe = "$rel\meeting-app.exe"
if (-not (Test-Path $exe)) { throw "build succeeded but $exe is missing" }

Write-Host "==> Built: $exe"
Get-ChildItem "$rel\*.dll" -ErrorAction SilentlyContinue | ForEach-Object {
    Write-Host ("    dll: {0}  ({1:N1} MB)" -f $_.Name, ($_.Length / 1MB))
}

# Stage the runnable artifact (exe + adjacent DLLs) and zip it. A feature build
# gets a suffix so the CPU and GPU artifacts coexist (e.g. ...-x64-vulkan.zip).
$suffix = ''
if ($Features) { $suffix = '-' + ($Features -replace '[, ]+', '-') }
$stage = "$build\dist-windows\meeting-app$suffix"
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

$zip = "$build\dist-windows\meeting-app-windows-x64$suffix.zip"
if (Test-Path $zip) { Remove-Item -Force $zip }
Compress-Archive -Path "$stage\*" -DestinationPath $zip
$zipMb = [math]::Round((Get-Item $zip).Length / 1MB, 1)
Write-Host ("==> Zipped: {0} ({1} MB)" -f $zip, $zipMb)
Write-Host "==> Done. Unzip and run meeting-app.exe (first run downloads the ASR/LLM models)."
