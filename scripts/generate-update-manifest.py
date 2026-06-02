#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.9"
# dependencies = []
# ///
#
# generate-update-manifest.py
#
# Fallback assembler for the tauri-plugin-updater `latest.json` manifest.
#
# In normal CI the GitHub release is produced by tauri-apps/tauri-action with
# includeUpdaterJson:true (see .github/workflows/release.yml), which generates
# latest.json for us. This script is the manual / local path: assemble the same
# manifest from already-built bundles and their detached `.sig` files when you
# are cutting a release by hand or off-CI.
#
# Cross-OS signed builds and the GPU hardware matrix are validated in CI / on
# hardware, not locally; this script only stitches the resulting artefacts
# together and performs NO signing itself.
#
# What it does:
#   - Scans one or more bundle directories for tauri updater artefacts and
#     their matching `.sig` files.
#   - Maps each artefact to a tauri-updater platform key
#     (e.g. linux-x86_64, windows-x86_64, darwin-aarch64).
#   - Emits latest.json: { version, notes, pub_date, platforms:
#     { <key>: { signature, url } } }.
#
# The updater verifies each download against its embedded `signature`, so the
# `.sig` content is copied verbatim into the manifest. The `url` is built from
# --base-url + the artefact filename (override per-file mapping if your layout
# differs).
#
# Pure stdlib; runnable via `uv run scripts/generate-update-manifest.py ...`
# or plain `python3 scripts/generate-update-manifest.py ...`.
#
# Usage:
#   uv run scripts/generate-update-manifest.py \
#       --version 1.2.3 \
#       --base-url https://releases.example.com/meeting-app/v1.2.3 \
#       --bundle-dir target/release/bundle \
#       --notes "Bug fixes and improvements" \
#       --output latest.json
#
# Multiple --bundle-dir flags may be passed (e.g. one per OS when assembling a
# cross-platform manifest from artefacts downloaded out of separate CI legs).

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

# tauri-updater artefact suffixes, longest-first so e.g. ".tar.gz" wins over
# ".gz". Each updater target produces exactly one of these per platform.
#   - Linux:   AppImage (the updater target for Linux is the AppImage)
#   - Windows: NSIS .exe / WiX .msi
#   - macOS:   .app delivered as a .app.tar.gz
UPDATER_SUFFIXES = (
    ".app.tar.gz",
    ".AppImage.tar.gz",
    ".AppImage",
    ".msi.zip",
    ".msi",
    ".nsis.zip",
    ".exe",
)


def _artefact_suffix(name: str) -> str | None:
    for suffix in UPDATER_SUFFIXES:
        if name.endswith(suffix):
            return suffix
    return None


def _platform_key(name: str) -> str | None:
    """Map a bundle filename to a tauri-updater platform key.

    tauri-updater keys are `<os>-<arch>`:
        linux-x86_64, linux-aarch64,
        windows-x86_64, windows-aarch64,
        darwin-x86_64, darwin-aarch64

    Arch is inferred from common substrings in the filename; default to x86_64
    when no arch token is present (most desktop bundles), and warn so the caller
    can correct an aarch64 build that lacks an arch token in its name.
    """
    lower = name.lower()

    if name.endswith(".AppImage") or ".appimage" in lower:
        os_key = "linux"
    elif name.endswith(".msi") or name.endswith(".exe") or ".msi.zip" in lower or ".nsis.zip" in lower:
        os_key = "windows"
    elif ".app.tar.gz" in lower or name.endswith(".dmg"):
        os_key = "darwin"
    else:
        return None

    if "aarch64" in lower or "arm64" in lower:
        arch = "aarch64"
    elif "x86_64" in lower or "x64" in lower or "amd64" in lower:
        arch = "x86_64"
    elif "i686" in lower or "x86" in lower:
        arch = "i686"
    else:
        arch = "x86_64"
        print(
            f"warning: no arch token in {name!r}; defaulting to {os_key}-{arch}. "
            "Pass a filename containing 'aarch64'/'x86_64' if this is wrong.",
            file=sys.stderr,
        )

    return f"{os_key}-{arch}"


def _iso_utc_now() -> str:
    # RFC 3339 / ISO 8601 with a trailing Z, which the updater accepts.
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _collect(bundle_dirs: list[Path]) -> dict[str, Path]:
    """Return {artefact_path: sig_path} for every signable artefact found."""
    found: dict[Path, Path] = {}
    for root in bundle_dirs:
        if not root.exists():
            print(f"warning: bundle dir does not exist: {root}", file=sys.stderr)
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix == ".sig":
                continue
            if _artefact_suffix(path.name) is None:
                continue
            sig = path.with_name(path.name + ".sig")
            if not sig.exists():
                print(
                    f"warning: no .sig for {path} (skipping; sign it or omit)",
                    file=sys.stderr,
                )
                continue
            found[path] = sig
    return found


def build_manifest(
    *,
    version: str,
    base_url: str,
    bundle_dirs: list[Path],
    notes: str,
    pub_date: str,
) -> dict:
    artefacts = _collect(bundle_dirs)
    if not artefacts:
        raise SystemExit(
            "error: no signed updater artefacts found. Expected files matching "
            f"{', '.join(UPDATER_SUFFIXES)} each with a sibling .sig."
        )

    platforms: dict[str, dict[str, str]] = {}
    base = base_url.rstrip("/")
    for artefact, sig in artefacts.items():
        key = _platform_key(artefact.name)
        if key is None:
            print(f"warning: unmapped artefact {artefact.name} (skipping)", file=sys.stderr)
            continue
        signature = sig.read_text(encoding="utf-8").strip()
        url = f"{base}/{artefact.name}"
        if key in platforms:
            print(
                f"warning: duplicate platform key {key!r}; "
                f"{artefact.name} overrides earlier {platforms[key]['url']}",
                file=sys.stderr,
            )
        platforms[key] = {"signature": signature, "url": url}

    if not platforms:
        raise SystemExit("error: found artefacts but none mapped to a platform key.")

    return {
        "version": version,
        "notes": notes,
        "pub_date": pub_date,
        "platforms": platforms,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Assemble a tauri-updater latest.json from built bundles + .sig files.",
    )
    parser.add_argument(
        "--version",
        required=True,
        help="Release version, e.g. 1.2.3 (no leading 'v').",
    )
    parser.add_argument(
        "--base-url",
        required=True,
        help="Base URL the artefacts are hosted at; filenames are appended.",
    )
    parser.add_argument(
        "--bundle-dir",
        dest="bundle_dirs",
        action="append",
        required=True,
        type=Path,
        help="Directory to scan for bundles + .sig files (repeatable).",
    )
    parser.add_argument(
        "--notes",
        default="",
        help="Release notes shown by the updater prompt.",
    )
    parser.add_argument(
        "--pub-date",
        default=None,
        help="RFC 3339 publish date; defaults to now (UTC).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write manifest here; defaults to stdout.",
    )
    args = parser.parse_args(argv)

    manifest = build_manifest(
        version=args.version.lstrip("v"),
        base_url=args.base_url,
        bundle_dirs=args.bundle_dirs,
        notes=args.notes,
        pub_date=args.pub_date or _iso_utc_now(),
    )

    text = json.dumps(manifest, indent=2, sort_keys=False) + "\n"
    if args.output is None:
        sys.stdout.write(text)
    else:
        args.output.write_text(text, encoding="utf-8")
        print(f"wrote {args.output} ({len(manifest['platforms'])} platform(s))", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
