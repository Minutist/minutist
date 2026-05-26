# persistence

Phase 1 minimal surface: per-meeting folder, `audio.opus`, `metadata.json`.

Phase 4 will add the libsql index, transcript/notes/summary storage, and
meeting-list queries. Nothing from Phase 4 is preempted here.

## Phase 1 surface

| Item | Description |
|---|---|
| `MeetingFolder` | On-disk handle for `{root}/{uuid}/`. Created by `MeetingFolder::create(root, id)`. |
| `MeetingWriter` | Opens the folder, accepts f32 PCM samples via `push_samples`, supports `pause`/`resume`, finalises to `audio.opus` + `metadata.json` via `finalise(meta)`. |

## Encoding parameters

- Codec: Opus via `audiopus` (FFI bindings to libopus).
- Container: Ogg (RFC 7845) via `ogg`.
- Sample rate: 16 kHz mono f32.
- Bitrate: 32 kbps.
- Frame size: 20 ms (320 samples at 16 kHz).
- Opus application mode: `Voip` (voice clarity bias).
- Complexity: 5 (mid-range CPU/quality balance).

## Pause/resume gap mechanics

**Decision: option (b) — zero-sample padding.**

On `pause()`, the encoder records the wall-clock instant and flushes any
pending partial frame (zero-padded to a full 20 ms frame). On `resume()`,
the elapsed pause duration is converted to the nearest 20 ms boundary and
that many zero-sample (silent) Opus frames are written to the stream before
any new audio frames.

**Why not option (a) — granule jump?**

A granule position jump in the Ogg container encodes the gap in the
container's timeline metadata, but the Opus decoder itself counts *packets*,
not granule positions. It decodes the packets present in the stream and does
not interpolate silence for a granule gap. Testing with `audiopus`'s own
decoder confirmed that a granule-only jump produces identical decoded sample
count to a stream without the jump (i.e. no silence, just the two audio
segments concatenated). Option (b) was chosen because it produces correct
decoded duration.

**Gap accuracy:** ±20 ms (one frame). The spec requires ±50 ms — this is
within budget.

## File layout (Phase 1)

```
{root}/
└── {uuid}/
    ├── audio.opus      — Ogg/Opus bitstream
    └── metadata.json   — MeetingMeta (serialised via serde_json)
```

No other files are written in Phase 1.

## Cross-cutting compliance

- Tracing target: `"persistence"`.
- No `tauri::*` imports — path to `{app-data}/meetings/` is passed by the
  caller (orchestrator).
- Per-crate `persistence::Error` with `thiserror`; `From<Error> for AppError`.
- No `println!` outside test code.
