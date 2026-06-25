# Deploying the Minutist headless sync hub

`minutist-hub` is the headless server daemon (planning issue 0020): a user-owned,
always-on peer that two sometimes-online devices converge through, without leaning
on the relay's deferred store-and-forward inbox. It pairs into your device mesh
like a desktop and holds meeting plaintext — but on hardware you own, so it stays
within the same trust boundary as your desktop. It is **not** the hosted relay
(which only ever brokers ciphertext).

It needs the connected-tier relay access token (`MINUTIST_SYNC_TOKEN`) — the same
token the desktop's connected build uses.

Pairing is mutual and the same on every platform:

```
minutist-hub --data-dir <dir> print-ticket        # this hub's ticket -> paste into a desktop's Sync settings
minutist-hub --data-dir <dir> add-peer <ticket>    # a desktop's ticket -> authorise it on the hub
```

A running hub re-reads its `<dir>/peers` file periodically, so `add-peer` is
honoured without a restart. Each desktop must also add the hub's ticket (sync is
mutual).

## Linux — container (Docker / Podman)

Build the image (context is the repo root; the Dockerfile-specific ignore keeps
it small):

```sh
DOCKER_BUILDKIT=1 docker build -f packaging/Dockerfile -t minutist-hub:latest .
```

Run it always-on. `--network host` keeps iroh's QUIC socket and relay path
simple; the data volume preserves the device identity across restarts:

```sh
docker run -d --name minutist-hub --restart unless-stopped --network host \
  -e MINUTIST_SYNC_TOKEN=<relay token> \
  -v minutist-hub-data:/var/lib/minutist-hub \
  minutist-hub:latest

# pair:
docker exec minutist-hub minutist-hub --data-dir /var/lib/minutist-hub print-ticket
docker exec minutist-hub minutist-hub --data-dir /var/lib/minutist-hub add-peer <desktop ticket>
```

To deploy to a host that has Docker but no Rust toolchain, build on a build host
and ship the image: `docker save minutist-hub:latest | ssh HOST docker load`.

## Linux — bare metal (systemd)

Extract a glibc-compatible binary from the builder stage (matches an LTS host),
then install the unit:

```sh
DOCKER_BUILDKIT=1 docker build -f packaging/Dockerfile --target builder -t minutist-hub-builder .
id=$(docker create minutist-hub-builder)
docker cp "$id:/src/target/release/minutist-hub" ./minutist-hub
docker rm "$id"

sudo install -m 0755 minutist-hub /usr/local/bin/minutist-hub
sudo install -m 0644 packaging/minutist-hub.service /etc/systemd/system/minutist-hub.service
sudo install -d -m 0700 /etc/minutist-hub
printf 'MINUTIST_SYNC_TOKEN=%s\n' "<relay token>" | sudo tee /etc/minutist-hub/minutist-hub.env >/dev/null
sudo chmod 0600 /etc/minutist-hub/minutist-hub.env
sudo systemctl daemon-reload && sudo systemctl enable --now minutist-hub
```

The unit runs as a `DynamicUser` with a managed `StateDirectory`
(`/var/lib/minutist-hub`). systemd sends `SIGTERM` on stop; the daemon drains
within a bounded grace window and exits. A bare host also needs `ca-certificates`
and (if the binary linked libopus dynamically) `libopus0`.

## Windows (service via WinSW)

The daemon has no Windows-specific code; `packaging/windows/install-service.ps1`
registers it as a real Windows service through [WinSW](https://github.com/winsw/winsw)
(the SCM wrapper). From an elevated PowerShell:

```powershell
packaging\windows\install-service.ps1 -BinaryPath C:\path\to\minutist-hub.exe -RelayToken <relay token>
```

It stages the binary + WinSW under `%ProgramFiles%\minutist-hub`, writes the
service config (token file locked to SYSTEM/Administrators), and starts the
service (data dir `%ProgramData%\minutist-hub`). On stop WinSW sends Ctrl+C and
the daemon drains gracefully. Remove with `uninstall-service.ps1`
(`-PurgeData` also deletes the data dir).

> Build `minutist-hub.exe` from the Windows build pipeline (the same MSVC
> toolchain used for the desktop app); the service scripts are verified on a
> Windows host, not in CI.

## GPU processing node (post-launch)

These instructions cover the **sync-hub** role only. The GPU processing node —
where the hub runs the ASR / diarize / summarise pipeline for meetings captured
on GPU-less devices — is a separate, post-launch capability of the same binary
and will document its own GPU-runtime requirements (Vulkan/CUDA, driver, model
cache) when it lands.
