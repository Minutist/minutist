#!/usr/bin/env bash
# ci-local.sh — run the Linux CI test suite locally in the same Docker image
# used by the self-hosted GitHub Actions runner (minutist-ci-runner:local).
#
# Invoked via `make ci-local` from the repo root.  Produces the same test
# results as a CI push without the GitHub round-trip.
#
# TOOLCHAIN PERSISTENCE
#   The Docker image ships all system dependencies (clang, libssl, etc.) but
#   not Rust or Node.js; CI installs them via actions.  Here we persist them
#   in host directories mounted as RUSTUP_HOME / CARGO_HOME:
#
#     /mnt/bulk/srv/github-runner/local-rustup  →  RUSTUP_HOME inside container
#     /mnt/bulk/srv/github-runner/local-cargo   →  CARGO_HOME  inside container
#
#   On the first run rustup-init is downloaded and stable is installed; the
#   installation is reused on subsequent runs (idempotent).
#
#   Node 20 is baked into the image (added via NodeSource).  The workflow uses
#   actions/setup-node@v4, which installs a fresh Node each job; baking it in
#   the image is equivalent here and avoids a download on every ci-local run.
#
# TARGET DIRECTORY ISOLATION
#   cargo builds into target-ci-local/ (git-ignored) rather than target/.
#   The host build target/ and the container build use different ABIs (the
#   container is ubuntu 24.04 glibc; the host may differ) and would produce
#   incompatible artifacts if they shared a directory.
#
# TMPDIR
#   Set to /tmp inside the container (separate from the bind-mounted repo) so
#   large build temporaries don't fill the host working directory.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

IMAGE="minutist-ci-runner:local"
RUSTUP_VOL="/mnt/bulk/srv/github-runner/local-rustup"
CARGO_VOL="/mnt/bulk/srv/github-runner/local-cargo"

# Ensure persistent toolchain directories exist (owned by the calling user so
# the container's uid 1000 can write — the caller must also be uid 1000, which
# matches the runner user in the image).
mkdir -p "$RUSTUP_VOL" "$CARGO_VOL"

echo "[ci-local] image:  $IMAGE"
echo "[ci-local] repo:   $REPO_ROOT"
echo "[ci-local] rustup: $RUSTUP_VOL"
echo "[ci-local] cargo:  $CARGO_VOL"
echo ""

docker run --rm \
    -v "$REPO_ROOT:$REPO_ROOT" \
    -w "$REPO_ROOT" \
    --user "$(id -u):$(id -g)" \
    -v "$RUSTUP_VOL:/opt/ci-rustup" \
    -v "$CARGO_VOL:/opt/ci-cargo" \
    -e RUSTUP_HOME=/opt/ci-rustup \
    -e CARGO_HOME=/opt/ci-cargo \
    -e CARGO_TARGET_DIR="$REPO_ROOT/target-ci-local" \
    -e LIBCLANG_PATH=/usr/lib/llvm-18/lib \
    -e TMPDIR=/tmp \
    --entrypoint bash \
    "$IMAGE" -euo pipefail -c '
# Bootstrap Rust stable if not already present.
if ! "$CARGO_HOME/bin/cargo" --version >/dev/null 2>&1; then
    echo "[ci-local] bootstrapping Rust stable (first run)..."
    # Download rustup-init to a temp location.
    curl -fsSL https://sh.rustup.rs -o /tmp/rustup-init.sh
    sh /tmp/rustup-init.sh -y --no-modify-path --default-toolchain stable --quiet
    rm /tmp/rustup-init.sh
else
    echo "[ci-local] Rust already bootstrapped: $("$CARGO_HOME/bin/cargo" --version)"
fi

export PATH="$CARGO_HOME/bin:$PATH"

# Export sherpa-rs lib path if a cached download exists (mirrors the workflow
# "Export sherpa-rs lib path" step).
lib=$(ls -d ~/.cache/sherpa-rs/x86_64-unknown-linux-gnu/*/*/lib 2>/dev/null | head -1 || true)
if [ -n "$lib" ]; then
    export LD_LIBRARY_PATH="${lib}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

exec ci/scripts/run-linux-tests.sh
'
