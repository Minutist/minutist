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
# glslc ships from the `shaderc` source package in Ubuntu's universe
# component, so universe is enabled before the install.
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
    glslang-tools \
    glslc
