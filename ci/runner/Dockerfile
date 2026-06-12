# Self-hosted GitHub Actions runner for the Linux CI legs.
#
# Base matches GitHub's ubuntu-latest (24.04/noble) so workflow steps behave
# identically. The workflow's apt packages are prebaked so per-job installs
# are fast no-op upgrades; the workflows still run their install steps, which
# keeps GitHub-hosted runners working unchanged.
#
# The runner distribution is staged at /opt/actions-runner-dist; the
# entrypoint copies it into the (volume-backed) state directory on first run
# so registration credentials and the runner's self-updates persist across
# container recreation. The state directory must be bind-mounted at the SAME
# absolute path as on the host: container actions (docker://) are spawned as
# sibling containers via the docker socket, and the paths the runner passes
# to them must resolve on the host daemon.

FROM ubuntu:24.04

ARG RUNNER_VERSION=2.335.1
ARG DOCKER_CLI_VERSION=27.5.1
ARG UID=1000
ARG GID=1001

ENV DEBIAN_FRONTEND=noninteractive

# Baseline tooling + everything the repo's Linux CI legs apt-get install.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl jq git sudo unzip zstd \
        software-properties-common gnupg \
    && add-apt-repository universe -y \
    && apt-get update && apt-get install -y --no-install-recommends \
        build-essential pkg-config cmake ninja-build \
        libwebkit2gtk-4.1-dev \
        libssl-dev \
        libayatana-appindicator3-dev \
        librsvg2-dev \
        libasound2-dev \
        libopus-dev \
        libfuse2 \
        libvulkan-dev glslang-tools glslc \
        file desktop-file-utils \
        clang libclang-dev \
    && rm -rf /var/lib/apt/lists/*

# Node.js 20 LTS via NodeSource. Ubuntu 24.04 noble ships Node 18 in its
# default packages, which is below the ≥20 requirement in the workflow.
# Adding NodeSource's node_20.x repo gives a version-locked 20.x package
# that matches what actions/setup-node@v4 with node-version: 20 installs on
# GitHub-hosted runners.
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

# Docker CLI only (static binary) — the daemon is the host's, reached via the
# bind-mounted socket. Needed for container actions (reuse-lint, cargo-deny).
RUN curl -fsSL "https://download.docker.com/linux/static/stable/x86_64/docker-${DOCKER_CLI_VERSION}.tgz" \
        | tar -xz --strip-components=1 -C /usr/local/bin docker/docker

# Runner user mirrors the host owner of the bind-mounted state dir. Ubuntu
# 24.04 ships a uid-1000 "ubuntu" user; remove it so the uid is free.
RUN userdel -r ubuntu 2>/dev/null || true \
    && groupadd -g ${GID} runner \
    && useradd -m -u ${UID} -g ${GID} -G sudo runner \
    && echo "runner ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/runner \
    # docker socket access: gid is taken from the mounted socket at runtime;
    # entrypoint chgrp-bridges it. Pre-create a fallback group.
    && groupadd -g 988 docker-host 2>/dev/null || true \
    && usermod -aG 988 runner 2>/dev/null || true

# Stage the runner distribution.
RUN mkdir -p /opt/actions-runner-dist \
    && curl -fsSL "https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz" \
        | tar -xz -C /opt/actions-runner-dist \
    && /opt/actions-runner-dist/bin/installdependencies.sh \
    && chown -R ${UID}:${GID} /opt/actions-runner-dist

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

USER runner
ENTRYPOINT ["/entrypoint.sh"]
