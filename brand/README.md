# Minutist brand assets

The Minutist logo: a fountain-pen **nib drawing a line** (the nib writes the
"minutes"). Editorial-Ink palette — oxblood ink on warm paper. The nib tip slit
is **open** (the two tines separate at the writing point).

## Colours

| Token | Light | Dark |
|-------|-------|------|
| Ink / mark | `#7a2e2e` (oxblood) | `#c06a5f` (clay) |
| Paper / tile | `#fcfaf4` | — |

## Files

| File | Use |
|------|-----|
| `logo.svg` | Master mark, transparent, oxblood. The canonical vector. |
| `logo-dark.svg` | Same mark in clay `#c06a5f` for dark backgrounds. |
| `logo-icon16.svg` | Small-size mark — open slit kept, bolder/shorter baseline for ≤24px. |
| `app-icon.svg` | Paper rounded-tile + oxblood mark (≥48px). Source for OS app icons. |
| `app-icon-small.svg` | Tile + mark, tighter framing + bold baseline for 16/32px. |
| `favicon.svg` | Web favicon — transparent, **adaptive** fill (oxblood light / clay dark). |
| `favicon.ico` | Web fallback, 16/32/48, paper tile. |
| `apple-touch-icon.png` | 180px paper-tile, for iOS home-screen. |

OS app-icon set (paper tile) lives in `../src-tauri/icons/` (`16/32/48/64/128/
128@2x/256` PNGs + `icon.ico` + `icon.icns` + `icon.png`).

## Provenance / regeneration

Design history, mocks, the live preview harness, the idealization workflow
and its instruments are kept outside the repository. The master is `logo-v2`
from that workflow (IoU 0.983 vs the chosen reference, 26 anchors, 5 named elements).
Render any SVG with `rsvg-convert`; rebuild `.ico`/`.icns` with ImageMagick
`convert`.
