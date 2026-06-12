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

echo "==> cargo test --workspace --locked"
cargo test --workspace --locked

echo "==> UI: npm ci"
cd ui
npm ci

echo "==> UI: npm run build"
npm run build

echo "==> UI: npm run test"
npm run test
