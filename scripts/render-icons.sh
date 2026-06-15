#!/usr/bin/env bash
# Render the OS app-icon set and web favicons from the single master vector
# (brand/app-icon.svg). Every raster size is rendered natively from the vector
# at its target pixel size — no size-specific geometry and no downscaling, so
# the small sizes are faithful scale-downs of the full-size mark and stay crisp
# on the Windows taskbar (including "small taskbar buttons").
#
# Requires: rsvg-convert (librsvg), magick (ImageMagick 7), python3 (Pillow not
# needed — the .icns packer uses only the stdlib).
set -euo pipefail

cd "$(dirname "$0")/.."
MASTER="brand/app-icon.svg"
ICONS="src-tauri/icons"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

render() { rsvg-convert -w "$1" -h "$1" "$MASTER" -o "$2"; }

# --- Tauri PNG set ------------------------------------------------------------
render 16  "$ICONS/16x16.png"
render 32  "$ICONS/32x32.png"
render 48  "$ICONS/48x48.png"
render 64  "$ICONS/64x64.png"
render 128 "$ICONS/128x128.png"
render 256 "$ICONS/128x128@2x.png"
render 256 "$ICONS/256x256.png"
render 512 "$ICONS/icon.png"

# --- Windows .ico -------------------------------------------------------------
# Native render at every size Windows may request. Small-taskbar-buttons mode
# and per-monitor DPI scaling pull 16/20/24/32/40/48; 256 covers Explorer's
# extra-large view. magick stores the 256 entry PNG-compressed automatically.
ICO_SIZES=(16 20 24 32 40 48 64 128 256)
ico_inputs=()
for s in "${ICO_SIZES[@]}"; do
  render "$s" "$TMP/ico-$s.png"
  ico_inputs+=("$TMP/ico-$s.png")
done
magick "${ico_inputs[@]}" "$ICONS/icon.ico"

# --- macOS .icns --------------------------------------------------------------
# ImageMagick here has no icns delegate, so pack the PNG-typed container by hand.
for s in 16 32 64 128 256 512 1024; do render "$s" "$TMP/icns-$s.png"; done
python3 - "$TMP" "$ICONS/icon.icns" <<'PY'
import struct, sys
tmp, out = sys.argv[1], sys.argv[2]
# OSType -> source PNG pixel size (PNG-encoded entries, read by modern macOS)
entries = {
    b"icp4": 16,  b"icp5": 32,  b"icp6": 64,
    b"ic07": 128, b"ic08": 256, b"ic09": 512, b"ic10": 1024,
    b"ic11": 32,  b"ic12": 64,  b"ic13": 256, b"ic14": 512,
}
blocks = b""
for ostype, size in entries.items():
    with open(f"{tmp}/icns-{size}.png", "rb") as fh:
        data = fh.read()
    blocks += ostype + struct.pack(">I", len(data) + 8) + data
payload = b"icns" + struct.pack(">I", len(blocks) + 8) + blocks
with open(out, "wb") as fh:
    fh.write(payload)
PY

# --- Web favicons (brand/) ----------------------------------------------------
for s in 16 32 48; do render "$s" "$TMP/fav-$s.png"; done
magick "$TMP/fav-16.png" "$TMP/fav-32.png" "$TMP/fav-48.png" brand/favicon.ico
render 180 brand/apple-touch-icon.png

echo "Rendered app-icon set + favicons from $MASTER"
