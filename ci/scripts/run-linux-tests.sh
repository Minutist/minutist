#!/usr/bin/env bash
# run-linux-tests.sh — Linux test suite shared by CI and ci-local.
#
# Runs the workspace Rust tests and the UI build+test steps.  Called directly
# by the GitHub Actions test job (after the action-managed steps that handle
# checkout, toolchain install, and caches) and by ci/scripts/ci-local.sh
# (inside the minutist-ci-runner:local container with the repo bind-mounted).
#
# Environment variables (all optional with sensible defaults):
#   CARGO_TARGET_DIR   Override the cargo target directory.  ci-local sets this
#                      to target-ci-local/ so the container build does not
#                      share artifacts with the host's target/.
#   LIBCLANG_PATH      Path to the directory containing libclang.so.  The
#                      workflow sets /usr/lib/llvm-18/lib before calling this
#                      script; ci-local passes the same value.
#   LD_LIBRARY_PATH    Prepend by the caller when sherpa-rs shared libs are
#                      pre-cached at a path the linker cannot find automatically.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# Windows-dep skew guard (issue 0040). `wmi` (via netwatch <- iroh) declares
# `windows` and `windows-core` with independent `>=0.59,<0.63` ranges, so the
# resolver can pin them to different pre-1.0 minors (e.g. windows 0.62 +
# windows-core 0.61). That breaks the Windows build but is invisible to a Linux
# compile, and the Windows/macOS CI legs are billing-blocked — so it once
# regressed silently. `wmi`/`netwatch` are pure-Rust bindings that typecheck for
# the MSVC target on a Linux host (no linker needed), so cross-checking them here
# catches a lock re-mismatch on the free Linux gate. Scoped to these two: crates
# further up the chain pull `ring`, whose cc-based build needs the MSVC
# toolchain and cannot cross-check from Linux.
echo "==> windows-dep skew guard: cross-check wmi/netwatch (x86_64-pc-windows-msvc)"
rustup target add x86_64-pc-windows-msvc
cargo check --locked --target x86_64-pc-windows-msvc -p wmi -p netwatch

echo "==> cargo test --workspace --locked"
cargo test --workspace --locked

echo "==> UI: npm ci"
cd ui
npm ci

echo "==> UI: npm run build"
npm run build

echo "==> UI: npm run test"
npm run test
