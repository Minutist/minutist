# persistence

Full Phase-4 read/write surface for per-meeting storage and the libsql
meeting index. Owns `{app-data}/meetings/{uuid}/...` and `{app-data}/index.db`.

## Write surface

| Item | Description |
|---|---|
| `MeetingFolder` | On-disk handle for `{root}/{uuid}/`. Created by `MeetingFolder::create(root, id)`. Path helpers: `audio_path` / `metadata_path` / `transcript_path` / `notes_path` / `notes_md_path` / `summary_path`. |
| `MeetingWriter` | Opens the folder, accepts f32 PCM samples via `push_samples`, supports `pause`/`resume`, finalises to `audio.opus` + `metadata.json` (+ buffered `transcript.json`) via `finalise(meta)`. |
| `TranscriptWriter` | Buffered append writer for `transcript.json`. |
| `NotesStore` | Standalone reader/writer for the opaque `notes.json` + `notes.md`. |
| `summary::{write_summary, read_summary}` | Atomic `summary.md` I/O (producer lands in Phase 5). |

## Read surface (`reader` module)

Synchronous blocking `std::fs` readers; callers in an async context drive
them via `tokio::task::spawn_blocking`.

| Function | Returns |
|---|---|
| `read_metadata(meeting_dir)` | `AppResult<MeetingMeta>` |
| `read_transcript(meeting_dir)` | `AppResult<Vec<Segment>>` (absent file → empty) |
| `read_audio_pcm(meeting_dir)` | `AppResult<Vec<f32>>` — graduated Opus decoder; **pause-INCLUDING** 16 kHz mono buffer (diarization + re-transcribe source) |
| `read_meeting_state(meeting_dir)` | `AppResult<MeetingState>` — meta + transcript + optional notes (`open_meeting` payload) |

## libsql index (`index` + `migrations` modules)

`MeetingIndex` is the `index.db` meeting index — a **derived cache** over the
per-meeting folders, rebuildable from disk. libsql is async (tokio); all index
methods are `async fn` and the crate never calls `block_on`.

| Method | Description |
|---|---|
| `MeetingIndex::open(db_path)` | Open-or-create at an injected path (`":memory:"` in tests); runs the forward-only migration runner. |
| `list_meetings()` | Most-recent first (`started_at DESC`). |
| `search(query)` | Case-insensitive `LIKE` over title + excerpt (wildcards escaped). |
| `upsert(&entry)` / `delete(id)` | Keyed on `MeetingId`. |
| `rebuild_from_disk(meetings_root)` | Clears + repopulates from `{root}/{uuid}/metadata.json`. |

Migrations: a single-row `schema_version` table; `migrations::run` is
idempotent and migrates both an empty DB and a prior-schema DB forward without
data loss (additive `CREATE ... IF NOT EXISTS` steps).

`meeting_ops::{rename_meeting, delete_meeting}` keep the folder and index row
consistent (folder authoritative, index updated to match).

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
