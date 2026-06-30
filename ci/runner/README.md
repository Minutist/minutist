# Self-hosted Linux CI runner

Docker-managed GitHub Actions runner serving the repo's Linux legs
(`runs-on: [self-hosted, linux, x64]`). The image mirrors `ubuntu-latest`
(24.04) with the workflows' apt dependencies prebaked, so jobs behave the
same as on GitHub-hosted runners but start faster and cost nothing.

## Setup (once per host)

```sh
mkdir -p /mnt/bulk/srv/github-runner/{state,toolcache}   # owned by uid 1000
cp .env.example .env
gh api -X POST repos/Minutist/minutist/actions/runners/registration-token \
  --jq .token   # paste into .env as RUNNER_TOKEN
docker compose up -d --build
```

### Additional hosts (more capacity / faster)

The state path defaults to `/mnt/bulk/srv/github-runner`. On a host with fast
LOCAL storage, set `RUNNER_STATE_BASE` in `.env` to a local-SSD path (the
runner's cargo `target/` + `_work` dominate build time, so local disk beats a
NAS mount), give the runner a distinct `RUNNER_NAME`, `mkdir -p
$RUNNER_STATE_BASE/{state,toolcache}` owned by uid 1000, and `docker compose up
-d --build github-runner` (a single replica per extra host is usually enough).
Run compose from this dir so `.env` is auto-loaded for `${RUNNER_STATE_BASE}`
interpolation. The workflows target the generic `[self-hosted, linux, x64]`
labels, so any such runner picks up the Linux legs — no workflow change.

Registration persists in the state volume — `RUNNER_TOKEN` is consumed once
and may be removed from `.env` afterwards. Verify with:

```sh
gh api repos/Minutist/minutist/actions/runners --jq '.runners[] | {name, status}'
```

From the repo root, `make runner-up` / `make runner-logs` / `make runner-down`
wrap the compose commands.

## Local CI runs (no GitHub round-trip)

`make ci-local` runs the same test suite inside the same image with the repo
bind-mounted — see `ci/scripts/` for the shared step scripts both paths use.

## Design notes

- **State path parity:** the state dir is mounted at the same absolute path
  inside the container as on the host. Container actions (`docker://` —
  reuse-lint, cargo-deny) are spawned as sibling containers through the
  mounted docker socket, and the bind paths the runner passes them must
  resolve on the host daemon.
- **Docker socket:** mounting `/var/run/docker.sock` gives workflow code
  effective control of the host docker daemon. Acceptable while the repo is
  private and all workflow code is ours. REVISIT BEFORE THE REPO GOES
  PUBLIC: fork PRs must never reach this runner (GitHub's default settings
  do not run fork-PR workflows on self-hosted runners without approval, but
  the policy must be confirmed at publication time — see the launch
  checklist).
- **Self-update:** the runner self-updates in place inside the state volume;
  the image pin (`RUNNER_VERSION`) only matters for first provisioning.
- **macOS/Windows legs** are unaffected and continue to need GitHub-hosted
  runners (billing) or dedicated hardware.
