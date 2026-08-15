#!/usr/bin/env bash
# Single source of truth for Minutist's Linux build dependencies — used by the
# CI workflows (build/test/release), the self-hosted runner image, and
# make ci-local. Add a dependency HERE, never in an individual workflow step or
# the Dockerfile, so they cannot drift.
#
# This installs the complete BUILD/BUNDLE dependency set (a superset of every
# prior inline source), so the script alone is sufficient on a bare runner:
#   - build toolchain:   build-essential pkg-config cmake ninja-build
#                        clang libclang-dev
#   - Tauri/WebKit:      libwebkit2gtk-4.1-dev libssl-dev
#                        libayatana-appindicator3-dev librsvg2-dev
#   - audio:             libasound2-dev libopus-dev
#   - AppImage bundling: libfuse2 xdg-utils file desktop-file-utils
#   - Vulkan backend:    libvulkan-dev glslang-tools glslc
#
# glslc ships as its own package (Ubuntu 24.04+) or bundled under `shaderc`
# (from Ubuntu's universe on releases that carry it) — 22.04 and earlier carry
# neither, so this falls back to LunarG's Vulkan SDK repo, which ships the
# same binary as `shaderc` there. See the glslc install below.
#
# Idempotent: re-running is a no-op when the packages are already present.
set -euo pipefail

SUDO=""
[ "$(id -u)" -ne 0 ] && SUDO=sudo

export DEBIAN_FRONTEND=noninteractive

$SUDO apt-get update
$SUDO apt-get install -y --no-install-recommends software-properties-common
$SUDO add-apt-repository universe -y
$SUDO apt-get update

$SUDO apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    cmake \
    ninja-build \
    clang \
    libclang-dev \
    libwebkit2gtk-4.1-dev \
    libssl-dev \
    libayatana-appindicator3-dev \
    librsvg2-dev \
    libasound2-dev \
    libopus-dev \
    libfuse2 \
    xdg-utils \
    file \
    desktop-file-utils \
    libvulkan-dev \
    glslang-tools

# glslc: Ubuntu 24.04+ packages it directly as `glslc`. Older releases (22.04
# and earlier) never packaged it — under any name — so fall back to LunarG's
# Vulkan SDK repo, which ships the same binary as `shaderc`.
if $SUDO apt-get install -y --no-install-recommends glslc; then
    :
else
    if [ ! -f /etc/apt/sources.list.d/lunarg-vulkan.list ]; then
        codename="$(. /etc/os-release && echo "$VERSION_CODENAME")"
        wget -qO /tmp/lunarg-vulkan-key.asc https://packages.lunarg.com/lunarg-signing-key-pub.asc
        $SUDO gpg --batch --yes --dearmor -o /usr/share/keyrings/lunarg-vulkan-keyring.gpg /tmp/lunarg-vulkan-key.asc
        echo "deb [signed-by=/usr/share/keyrings/lunarg-vulkan-keyring.gpg] https://packages.lunarg.com/vulkan $codename main" |
            $SUDO tee /etc/apt/sources.list.d/lunarg-vulkan.list
        $SUDO apt-get update
    fi
    $SUDO apt-get install -y --no-install-recommends shaderc
fi
