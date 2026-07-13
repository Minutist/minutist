#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx>=0.27"]
# ///
"""Autonomous phone↔desktop sync e2e harness (roadmap 2.11, desktop half).

Provisions two devices under one account WITHOUT the interactive RFC 8628
device-code flow, seeds the desktop's credential file, and launches the Windows
Vulkan build — so the full sync round-trip runs hands-off from WSL.

The desktop half this file owns:

  mint    obtain a device credential for one device from the account-service
          test-mint endpoint (roadmap 2.11 server half, relay-owner). Falls back
          to a local mock so `seed` / `launch` are exercisable before the real
          endpoint lands (--mock).
  seed    write {app-data}/tunnel_device.json so the app boots already paired,
          bypassing device-code. The on-disk shape is a contract with
          src-tauri/src/tunnel.rs::StoredCredential (pinned by the
          `externally_seeded_credential_loads_as_paired` unit test there).
  launch  set MINUTIST_SYNC_TOKEN in the Windows launch context and start the
          portable Vulkan exe via powershell.exe.

The phone device (device B) is provisioned + driven by phoneapp:corona on step
(the minutist-mobile simulator); this harness provisions the desktop (device A)
and, once the mint endpoint lands, mints the phone's credential under the same
account for phoneapp to consume.

Wired against phoneapp's verified mint contract (POST /test/mint-device): the
mint call (`mint_credential`), the credential-liveness probe (`mint_liveness_ok`
via /relay-authz), and the endpoint-registered-device listing (`registered_devices`
via /v1/account/devices). Only exercisable end-to-end once the server endpoint is
deployed (currently being built inert), but the response mapping is unit-tested
offline (`selftest`).

Still open (needs a LIVE run — a booted app that has called B4 register_self):
  - the directory assertion loop (poll `registered_devices` until both devices
    appear — they are ABSENT until each peer registers its endpoint, not null);
  - the sync-completion assertion (observe the meeting artifacts land on the peer).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import subprocess
import sys
from pathlib import Path

# The account-service origin (Cloudflare-fronted; origin = telie compose stack).
DEFAULT_API_BASE = "https://api.minutist.ai"

# Windows app-data root for the app: %APPDATA%\ai.minutist. From WSL that is the
# drvfs mount below. Mirrors src-tauri/src/main.rs::app_data_root (APP_IDENTIFIER
# = "ai.minutist"). Override with --app-data for a non-default Windows user.
DEFAULT_APP_DATA = Path("/mnt/c/Users/anl/AppData/Roaming/ai.minutist")

# The portable Vulkan build the user live-tests from (CLAUDE.local.md).
DEFAULT_EXE = Path(
    "/mnt/c/Users/anl/meeting-app/dist-windows/minutist-vulkan/minutist.exe"
)

CREDENTIAL_FILE = "tunnel_device.json"


@dataclasses.dataclass(frozen=True)
class Credential:
    """The exact fields src-tauri tunnel.rs::StoredCredential persists — nothing
    else may appear in the seeded JSON or `load()` treats the file as corrupt."""

    device_credential: str  # the long-lived mdc_ device secret
    account_id: str
    device_id: str

    def to_json(self) -> str:
        # Field order/spacing is not load-significant (serde parses by key), but
        # we emit compact, key-stable JSON so the file diffs cleanly.
        return json.dumps(dataclasses.asdict(self), separators=(",", ":"))


def mock_credential(account_id: str, device_id: str) -> Credential:
    """A locally-minted stand-in so seed/launch are testable before the server
    mint endpoint exists. NOT a valid relay credential — the app boots 'paired'
    but the tunnel/account calls will be rejected. For wiring/format tests only."""
    return Credential(
        device_credential=f"mdc_mock.{account_id}.{device_id}",
        account_id=account_id,
        device_id=device_id,
    )


def mint_credential(
    api_base: str, account_id: str, mint_secret: str, label: str | None = None
) -> Credential:
    """Mint a real device credential via the account-service test-mint endpoint
    (roadmap 2.11 server half). The response mirrors /pair/poll's authorised body
    verbatim, i.e. exactly the `Credential` fields.

    `device_id` is server-generated (relay_common::credential::issue), so the
    caller does NOT supply it: call twice with the SAME `account_id` to get two
    distinct devices under one account (the ≥2-devices/1-account requirement).

    Gate behaviour (for diagnostics): 503 when TEST_MINT_SECRET is unset on the
    server; 403 on a missing/bad bearer or a non-allowlisted account_id.
    """
    import httpx

    body: dict[str, str] = {"account_id": account_id}
    if label:
        body["label"] = label
    resp = httpx.post(
        f"{api_base}/test/mint-device",
        json=body,
        headers={"Authorization": f"Bearer {mint_secret}"},
        timeout=30.0,
    )
    if resp.status_code == 503:
        sys.exit("mint endpoint disabled (TEST_MINT_SECRET unset server-side) — not deployed for tests")
    if resp.status_code == 403:
        sys.exit("mint rejected (403): bad TEST_MINT_SECRET or account_id not allowlisted")
    resp.raise_for_status()
    return _credential_from_mint(resp.json())


def _credential_from_mint(payload: dict) -> Credential:
    """Parse a /test/mint-device (or /pair/poll authorised) body into a Credential.
    Kept separate so it is unit-testable without the network."""
    return Credential(
        device_credential=payload["device_credential"],
        account_id=payload["account_id"],
        device_id=payload["device_id"],
    )


def registered_devices(api_base: str, device_credential: str) -> list[dict]:
    """GET /v1/account/devices with the mdc_ credential.

    NOTE (phoneapp, verified against relay code): this lists ONLY devices that
    have registered a sync endpoint (endpoint_id/relay_url non-null, non-revoked).
    A minted-but-unregistered device is ABSENT — not present-with-null — so a
    directory assertion for the ≥2-device count must poll until each peer app has
    booted and run B4's register_self. For an immediate post-mint liveness check,
    use `mint_liveness_ok` instead (this endpoint would return an empty list).
    """
    import httpx

    resp = httpx.get(
        f"{api_base}/v1/account/devices",
        headers={"Authorization": f"Bearer {device_credential}"},
        timeout=30.0,
    )
    resp.raise_for_status()
    data = resp.json()
    return data.get("devices", data) if isinstance(data, dict) else data


def mint_liveness_ok(api_base: str, device_credential: str) -> bool:
    """Immediate post-mint credential liveness: the minted device won't appear in
    /v1/account/devices until it registers an endpoint, so probe /relay-authz
    (which the mdc_ gates) for a quick 'the credential is valid' check."""
    import httpx

    resp = httpx.get(
        f"{api_base}/relay-authz",
        headers={"Authorization": f"Bearer {device_credential}"},
        timeout=15.0,
    )
    return resp.is_success


def seed_credential(app_data: Path, cred: Credential) -> Path:
    """Write the credential so the app boots paired. Returns the file path.

    On WSL→drvfs the 0600 mode the app uses on Unix does not map to a Windows
    ACL, and `load()` does not require it — a plain readable file is accepted."""
    app_data.mkdir(parents=True, exist_ok=True)
    path = app_data / CREDENTIAL_FILE
    path.write_text(cred.to_json(), encoding="utf-8")
    return path


def launch(exe: Path, relay_token: str) -> subprocess.Popen:
    """Launch the Windows exe with MINUTIST_SYNC_TOKEN set in ITS environment.

    A WSL env var does not cross into a Windows process unless exported via
    WSLENV, so we set it inside the powershell launch context instead — and keep
    the token out of the argv (process listing) by assigning it to $env: first."""
    if not exe.exists():
        raise FileNotFoundError(f"exe not found: {exe} (build it first)")
    win_exe = _wsl_to_win_path(exe)
    ps = (
        f"$env:MINUTIST_SYNC_TOKEN=$args[0]; "
        f"Start-Process -FilePath '{win_exe}'"
    )
    return subprocess.Popen(
        ["powershell.exe", "-NoProfile", "-Command", ps, relay_token]
    )


def _wsl_to_win_path(p: Path) -> str:
    out = subprocess.run(
        ["wslpath", "-w", str(p)], capture_output=True, text=True, check=True
    )
    return out.stdout.strip()


def cmd_seed(args: argparse.Namespace) -> int:
    if args.mock:
        if not args.device_id:
            sys.exit("--device-id is required with --mock (the real mint server-assigns it)")
        cred = mock_credential(args.account_id, args.device_id)
    else:
        cred = mint_credential(
            args.api_base, args.account_id, _require_secret(), label=args.device_id
        )
    path = seed_credential(Path(args.app_data), cred)
    kind = "mock" if args.mock else "minted"
    print(f"seeded {kind} credential (device_id={cred.device_id}) → {path}")
    return 0


def cmd_launch(args: argparse.Namespace) -> int:
    token = os.environ.get("MINUTIST_SYNC_TOKEN", "")
    if not token:
        print("MINUTIST_SYNC_TOKEN is unset — the relay will reject the connection", file=sys.stderr)
    proc = launch(Path(args.exe), token)
    print(f"launched {args.exe} (pid {proc.pid})")
    return 0


def cmd_selftest(args: argparse.Namespace) -> int:
    """Verify seed + JSON contract without a Windows build or the network."""
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        cred = mock_credential("acct-e2e", "desktop-e2e")
        path = seed_credential(Path(d), cred)
        loaded = json.loads(path.read_text(encoding="utf-8"))
        assert set(loaded) == {"device_credential", "account_id", "device_id"}, loaded
        assert loaded["account_id"] == "acct-e2e"
        assert loaded["device_id"] == "desktop-e2e"
        assert loaded["device_credential"].startswith("mdc_")

    # Mint response mapping — offline, against phoneapp's documented body shape.
    sample = {
        "device_credential": "mdc_dev-xyz.secret",
        "account_id": "sub-123",
        "device_id": "dev-xyz",
    }
    minted = _credential_from_mint(sample)
    assert minted.device_credential == "mdc_dev-xyz.secret"
    assert minted.device_id == "dev-xyz"
    # A minted credential seeds to the identical on-disk contract.
    with tempfile.TemporaryDirectory() as d:
        loaded = json.loads(seed_credential(Path(d), minted).read_text(encoding="utf-8"))
        assert set(loaded) == {"device_credential", "account_id", "device_id"}, loaded

    print("selftest OK: seed + mint-response mapping match the StoredCredential contract")
    return 0


def _require_secret() -> str:
    secret = os.environ.get("MINUTIST_TEST_MINT_SECRET", "")
    if not secret:
        sys.exit("MINUTIST_TEST_MINT_SECRET is unset (needed for the real mint endpoint)")
    return secret


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app-data", default=str(DEFAULT_APP_DATA), help="Windows app-data dir (drvfs path)")
    parser.add_argument("--api-base", default=DEFAULT_API_BASE)
    parser.add_argument("--exe", default=str(DEFAULT_EXE))
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_seed = sub.add_parser("seed", help="mint (or mock) + write tunnel_device.json")
    p_seed.add_argument("--account-id", required=True)
    p_seed.add_argument(
        "--device-id",
        help="required with --mock (the mock's device_id); with the real mint it is "
        "sent as the optional device label — the server assigns the actual device_id",
    )
    p_seed.add_argument("--mock", action="store_true", help="use a local mock credential")
    p_seed.set_defaults(func=cmd_seed)

    p_launch = sub.add_parser("launch", help="launch the Windows exe with the relay token")
    p_launch.set_defaults(func=cmd_launch)

    p_self = sub.add_parser("selftest", help="offline seed/contract self-test")
    p_self.set_defaults(func=cmd_selftest)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
