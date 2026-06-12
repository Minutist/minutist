#!/bin/bash
# Runner container entrypoint.
#
# First run (empty state dir): copies the staged runner distribution into the
# volume-backed state dir and registers against $REPO_URL using the one-time
# $RUNNER_TOKEN. Subsequent runs reuse the persisted registration
# (.runner/.credentials survive in the volume), so RUNNER_TOKEN is only
# needed once; re-registration is only required after explicitly removing
# the runner from GitHub.
set -euo pipefail

STATE_DIR="${RUNNER_STATE_DIR:?RUNNER_STATE_DIR must be set (same path on host and in container)}"

# Bridge docker socket group: the mounted /var/run/docker.sock carries the
# HOST's docker gid; join it so container actions can talk to the daemon.
if [ -S /var/run/docker.sock ]; then
    sock_gid=$(stat -c '%g' /var/run/docker.sock)
    if ! id -G | tr ' ' '\n' | grep -qx "$sock_gid"; then
        sudo groupadd -g "$sock_gid" docker-sock 2>/dev/null || true
        sudo usermod -aG "$sock_gid" runner
        # Re-exec with the new group membership.
        exec sudo -u runner -g "#$sock_gid" --preserve-env "$0" "$@"
    fi
fi

if [ ! -f "$STATE_DIR/config.sh" ]; then
    echo "[entrypoint] populating runner state dir from distribution"
    cp -r /opt/actions-runner-dist/. "$STATE_DIR/"
fi

cd "$STATE_DIR"

if [ ! -f .runner ]; then
    : "${REPO_URL:?REPO_URL must be set for first-run registration}"
    : "${RUNNER_TOKEN:?RUNNER_TOKEN must be set for first-run registration (one-time)}"
    echo "[entrypoint] registering runner '${RUNNER_NAME:-$(hostname)}' for $REPO_URL"
    ./config.sh \
        --url "$REPO_URL" \
        --token "$RUNNER_TOKEN" \
        --name "${RUNNER_NAME:-$(hostname)}" \
        --labels "${RUNNER_LABELS:-self-hosted,linux,x64}" \
        --work _work \
        --unattended \
        --replace
fi

exec ./run.sh
