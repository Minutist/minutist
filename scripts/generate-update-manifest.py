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
# A trailing --arch <aarch64|x86_64|i686> applies to the preceding --bundle-dir
# and is the way to key a macOS .app.tar.gz (which carries no arch token in its
# name) onto darwin-aarch64 vs darwin-x86_64.
#
# Smoke test (pure stdlib, no bundles needed):
#   python3 scripts/generate-update-manifest.py --self-test
# It exercises arch detection (incl. the macOS arm64 -> darwin-aarch64 case)
# and asserts the emitted latest.json keys for both arches; exits non-zero on
# any failure so CI can run it as a guard.

from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
import sys
from pathlib import Path

# Rust target-triple arch tokens -> tauri-updater arch component. The bundle a
# given leg produced lives under target/<triple>/release/bundle/... (or plain
# target/release/bundle/... for the host triple), so the triple in the artefact
# path is an authoritative arch signal even when the filename carries none.
# This is what lets a macOS `<App>.app.tar.gz` (which has NO arch token) be
# keyed correctly: an arm64 leg builds under aarch64-apple-darwin, an Intel leg
# under x86_64-apple-darwin.
_TRIPLE_ARCH = (
    ("aarch64", "aarch64"),
    ("arm64", "aarch64"),
    ("x86_64", "x86_64"),
    ("i686", "i686"),
)

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


def _os_key(name: str) -> str | None:
    """Map a bundle filename to its tauri-updater OS component, or None."""
    lower = name.lower()
    if name.endswith(".AppImage") or ".appimage" in lower:
        return "linux"
    if name.endswith(".msi") or name.endswith(".exe") or ".msi.zip" in lower or ".nsis.zip" in lower:
        return "windows"
    if ".app.tar.gz" in lower or name.endswith(".dmg"):
        return "darwin"
    return None


def _arch_from_text(text: str) -> str | None:
    """Infer a tauri-updater arch from a filename or a target-triple path.

    Word-boundary matched so `x86` does not match inside `x86_64`; checked
    most-specific-first. Returns None when no arch token is present.
    """
    lower = text.lower()
    if re.search(r"(?<![a-z0-9])(aarch64|arm64)(?![a-z0-9])", lower):
        return "aarch64"
    if re.search(r"(?<![a-z0-9])(x86_64|x64|amd64)(?![a-z0-9])", lower):
        return "x86_64"
    if re.search(r"(?<![a-z0-9])(i686|x86)(?![a-z0-9])", lower):
        return "i686"
    return None


def _arch_from_path(path: Path) -> str | None:
    """Infer arch from a Rust target-triple in the bundle path's directories.

    A cross-compiled / target-pinned bundle lives under
    target/<triple>/release/bundle/...; the triple (e.g. aarch64-apple-darwin)
    is an authoritative arch signal that the bare filename often lacks.
    """
    for part in path.parts:
        for token, arch in _TRIPLE_ARCH:
            if re.search(rf"(?<![a-z0-9]){re.escape(token)}(?![a-z0-9])", part.lower()):
                return arch
    return None


def _platform_key(path: Path, arch_override: str | None = None) -> str | None:
    """Map a bundle artefact to a tauri-updater platform key `<os>-<arch>`.

    tauri-updater keys are `<os>-<arch>`:
        linux-x86_64, linux-aarch64,
        windows-x86_64, windows-aarch64,
        darwin-x86_64, darwin-aarch64

    Arch resolution order (no silent x86_64 default):
        1. an explicit per-bundle --arch override, when given;
        2. an arch token in the artefact filename;
        3. the Rust target triple in the artefact's directory path
           (target/<triple>/release/bundle/...).

    macOS `.app.tar.gz` / `.dmg` artefacts usually carry NO arch token in
    their name, so relying on a filename default mapped every macOS build to
    darwin-x86_64 — arm64 clients query darwin-aarch64 and never updated. When
    none of the three sources yields an arch we return None (caller skips and
    warns) rather than guess x86_64.
    """
    os_key = _os_key(path.name)
    if os_key is None:
        return None

    arch = arch_override or _arch_from_text(path.name) or _arch_from_path(path)
    if arch is None:
        print(
            f"warning: cannot determine arch for {path.name!r} (os={os_key}); "
            "no arch token in the filename, no target triple in its path, and "
            "no --arch override. Pass --arch (e.g. aarch64) for this --bundle-dir, "
            "or build under target/<triple>/release/bundle so the triple is "
            "visible. Refusing to default to x86_64.",
            file=sys.stderr,
        )
        return None

    return f"{os_key}-{arch}"


def _iso_utc_now() -> str:
    # RFC 3339 / ISO 8601 with a trailing Z, which the updater accepts.
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _collect(bundle_dirs: list[Path]) -> dict[Path, tuple[Path, Path]]:
    """Scan for signable artefacts.

    Returns {artefact_path: (sig_path, bundle_root)} for every artefact found.
    The originating bundle root is retained so a per-bundle-dir --arch override
    can be applied to artefacts whose filename/path carry no arch token.
    """
    found: dict[Path, tuple[Path, Path]] = {}
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
            found[path] = (sig, root)
    return found


def build_manifest(
    *,
    version: str,
    base_url: str,
    bundle_dirs: list[Path],
    notes: str,
    pub_date: str,
    arch_overrides: dict[Path, str] | None = None,
) -> dict:
    artefacts = _collect(bundle_dirs)
    if not artefacts:
        raise SystemExit(
            "error: no signed updater artefacts found. Expected files matching "
            f"{', '.join(UPDATER_SUFFIXES)} each with a sibling .sig."
        )

    arch_overrides = arch_overrides or {}
    platforms: dict[str, dict[str, str]] = {}
    base = base_url.rstrip("/")
    for artefact, (sig, root) in artefacts.items():
        key = _platform_key(artefact, arch_overrides.get(root))
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


_VALID_ARCHES = ("aarch64", "x86_64", "i686")


class _BundleDirAction(argparse.Action):
    """Collect a --bundle-dir, remembering it as the target for a later --arch."""

    def __call__(self, parser, namespace, value, option_string=None):
        dirs = getattr(namespace, "bundle_dirs", None) or []
        dirs.append(value)
        namespace.bundle_dirs = dirs
        namespace._last_bundle_dir = value


class _ArchAction(argparse.Action):
    """Attach an --arch override to the most recently given --bundle-dir.

    The override is authoritative for that bundle dir's artefacts, used when a
    filename/path carries no arch token (notably macOS `.app.tar.gz`, which has
    none, so it would otherwise be mis-keyed / skipped).
    """

    def __call__(self, parser, namespace, value, option_string=None):
        if value not in _VALID_ARCHES:
            parser.error(
                f"--arch must be one of {', '.join(_VALID_ARCHES)} (got {value!r})"
            )
        last = getattr(namespace, "_last_bundle_dir", None)
        if last is None:
            parser.error("--arch must follow a --bundle-dir it applies to")
        overrides = getattr(namespace, "arch_overrides", None) or {}
        overrides[last] = value
        namespace.arch_overrides = overrides


def _self_test() -> int:
    """In-process smoke test of arch detection + manifest key emission.

    No filesystem or network access: drives _platform_key directly and
    build_manifest via temporary signed-artefact fixtures. Exits 0 on success,
    1 on the first failed assertion (printed to stderr).
    """
    import tempfile

    failures: list[str] = []

    def check(label: str, got, want) -> None:
        if got != want:
            failures.append(f"{label}: got {got!r}, want {want!r}")

    # _platform_key: arch from filename tokens.
    check(
        "linux x86_64 from name",
        _platform_key(Path("bundle/meeting-app_1.2.3_amd64.AppImage")),
        "linux-x86_64",
    )
    check(
        "windows aarch64 from name",
        _platform_key(Path("bundle/meeting-app_1.2.3_arm64-setup.exe")),
        "windows-aarch64",
    )
    # macOS .app.tar.gz has NO arch token in its name. Without a triple in the
    # path and without an override it must NOT silently key to x86_64.
    check(
        "macos no-arch refuses default",
        _platform_key(Path("bundle/meeting-app.app.tar.gz")),
        None,
    )
    # Arch from the Rust target triple in the path (the CI layout).
    check(
        "macos arm64 from triple path",
        _platform_key(Path("target/aarch64-apple-darwin/release/bundle/meeting-app.app.tar.gz")),
        "darwin-aarch64",
    )
    check(
        "macos x86_64 from triple path",
        _platform_key(Path("target/x86_64-apple-darwin/release/bundle/meeting-app.app.tar.gz")),
        "darwin-x86_64",
    )
    # Arch from an explicit override (the manual / no-triple path).
    check(
        "macos arm64 from override",
        _platform_key(Path("bundle/meeting-app.app.tar.gz"), "aarch64"),
        "darwin-aarch64",
    )
    # Word-boundary: `x86_64` must not be mis-read as `x86`/`i686`.
    check(
        "x86_64 not matched as i686",
        _platform_key(Path("bundle/app_x86_64.AppImage")),
        "linux-x86_64",
    )

    # End-to-end: assemble a manifest for both macOS arches via overrides and
    # confirm both platform keys are present with valid url/signature.
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        arm_dir = root / "mac-arm"
        intel_dir = root / "mac-intel"
        for d in (arm_dir, intel_dir):
            d.mkdir(parents=True)
            art = d / "meeting-app.app.tar.gz"
            art.write_bytes(b"fake-bundle")
            (d / "meeting-app.app.tar.gz.sig").write_text("SIG", encoding="utf-8")

        manifest = build_manifest(
            version="1.2.3",
            base_url="https://example.com/v1.2.3/",
            bundle_dirs=[arm_dir, intel_dir],
            notes="",
            pub_date="2026-01-01T00:00:00Z",
            arch_overrides={arm_dir: "aarch64", intel_dir: "x86_64"},
        )
        keys = set(manifest["platforms"])
        check("manifest has both darwin arches", keys, {"darwin-aarch64", "darwin-x86_64"})
        for k in ("darwin-aarch64", "darwin-x86_64"):
            if k in manifest["platforms"]:
                entry = manifest["platforms"][k]
                check(f"{k} signature", entry.get("signature"), "SIG")
                check(
                    f"{k} url",
                    entry.get("url"),
                    "https://example.com/v1.2.3/meeting-app.app.tar.gz",
                )
        # Must round-trip as valid JSON.
        json.loads(json.dumps(manifest))

    if failures:
        print("self-test FAILED:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1
    print("self-test OK (arch detection + dual-arch manifest keys verified)", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Assemble a tauri-updater latest.json from built bundles + .sig files.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run the in-process smoke test and exit (no bundles/network needed).",
    )
    parser.add_argument(
        "--version",
        required=False,
        help="Release version, e.g. 1.2.3 (no leading 'v').",
    )
    parser.add_argument(
        "--base-url",
        required=False,
        help="Base URL the artefacts are hosted at; filenames are appended.",
    )
    parser.add_argument(
        "--bundle-dir",
        dest="bundle_dirs",
        action=_BundleDirAction,
        type=Path,
        help="Directory to scan for bundles + .sig files (repeatable).",
    )
    parser.add_argument(
        "--arch",
        action=_ArchAction,
        help=(
            "Arch override (aarch64|x86_64|i686) for the preceding --bundle-dir. "
            "Use for artefacts with no arch token in their name or path, e.g. a "
            "macOS .app.tar.gz from an arm64 leg: "
            "--bundle-dir mac-arm/bundle --arch aarch64."
        ),
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

    if args.self_test:
        return _self_test()

    missing = [
        name
        for name, val in (
            ("--version", args.version),
            ("--base-url", args.base_url),
            ("--bundle-dir", args.bundle_dirs),
        )
        if not val
    ]
    if missing:
        parser.error(f"the following arguments are required: {', '.join(missing)}")

    manifest = build_manifest(
        version=args.version.lstrip("v"),
        base_url=args.base_url,
        bundle_dirs=args.bundle_dirs,
        notes=args.notes,
        pub_date=args.pub_date or _iso_utc_now(),
        arch_overrides=getattr(args, "arch_overrides", None),
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
