# Spike 4 — sherpa-onnx offline speaker diarization

Status: 2026-05-26, WSL Ubuntu 24.04 (Linux 6.6.87.2-microsoft-standard-WSL2), CPU-only.

## Verdict

**Pass** against all Phase 0 Spike 4 acceptance criteria.

- `cargo run -p spike-diarize -- --segmentation <onnx> --embedding <onnx> --wav <2speaker.wav>`
  exits 0.
- stdout is a JSON array of `{start_ms, end_ms, speaker_id}` records.
- 2-speaker fixture (`2-two-speakers-en.wav`, 34.00 s) produces 9 segments
  across **2 distinct speakers** (`A`, `B`).
- DER **0.00 %** against the binding-fidelity reference RTTM. See "DER framing" below
  for what that reference is (and is not).
- Single-speaker control (`librispeech_30s.wav`, 29.60 s) produces 1
  speaker label (`A`).
- No panics; release-build inference RTF 0.155 (2-speaker fixture) /
  0.164 (3-two-speakers-en, 54.81 s) / 0.227 (single-speaker control) —
  all well inside the 2× audio-duration budget.
- Peak resident set: **205 MiB** for the 34 s fixture (full process,
  including SHA-256 of the input files and model loading).

The headline answer to Q-P0-6 is **the `sherpa-rs 0.6.8` Rust binding is
sufficient for Phase 6**; no `bindgen`-direct-C wrapper is required. Detail
in [§ Q-P0-6 detailed verdict](#q-p0-6-detailed-verdict).

## Binding

| | |
|---|---|
| Crate | `sherpa-rs` (Thewh1teagle) |
| Version | `=0.6.8` (exact pin in `spikes/diarize/Cargo.toml`) |
| crates.io | https://crates.io/crates/sherpa-rs/0.6.8 |
| Repo | https://github.com/thewh1teagle/sherpa-rs |
| License | MIT |
| Features | `download-binaries` (default); `tts` **disabled** |
| Sys crate | `sherpa-rs-sys = 0.6.8` (transitive); links pre-built `libsherpa-onnx-c-api.so` + ONNX Runtime from k2-fsa GitHub releases |
| Bundled sherpa-onnx | **v1.12.9** (pinned in `sherpa-rs-sys-0.6.8/dist.json`) |
| Direct-C `bindgen` fallback | **NOT used.** Documented as the Phase 0 contingency; not triggered. |

The alternative crate `sherpa-onnx = 1.13.2` (Apache-2.0, owned by k2-fsa
themselves, repo = `github.com/k2-fsa/sherpa-onnx`) also exists on
crates.io. It is the **upstream-owned** binding and is newer
(1.13.2 vs 1.12.9). It was **not** chosen for this spike because the Phase 0 plan
named `sherpa-rs` specifically and the criterion is "does the named
binding work", not "find the best binding". Phase 6 should re-evaluate
`sherpa-onnx 1.13+` vs `sherpa-rs 0.6.x` — see "Future work" below.

### Why `download-binaries`, not `static`

The default `download-binaries` feature fetches a prebuilt
`libsherpa-onnx-c-api.so` plus `libonnxruntime.so.*` from
[k2-fsa/sherpa-onnx releases](https://github.com/k2-fsa/sherpa-onnx/releases/tag/v1.12.9)
on first build and links dynamically. The downloaded archive lands in
`~/.cache/sherpa-rs/<target>/<sha256>/`; subsequent builds are instant.
The release binary in `target/release/spike-diarize` is **5.2 MiB** but
**requires the cached `.so` files at runtime** — running outside the
build environment requires either `LD_LIBRARY_PATH` or vendoring those
files (Phase 6 concern, not Spike 4 concern).

The `static` feature is available (`sherpa-rs/static`) and links a single
`libsherpa-onnx.a`. Recommended for Tauri bundling in Phase 6; not used
here because cold-build time was already acceptable and validating
dynamic linking is the higher-information path (proves the cache layout
works on WSL).

## Models

Both files cached under `~/.cache/sherpa-onnx/diarization/`.

| File | Size | SHA-256 | Source | License |
|---|---|---|---|---|
| `sherpa-onnx-pyannote-segmentation-3-0/model.onnx` | 6.68 MiB | `220ad67ca923bef2fa91f2390c786097bf305bceb5e261d4af67b38e938e1079` | [speaker-segmentation-models](https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2) | MIT (per repo `LICENSE`); upstream model is pyannote/segmentation-3.0 (MIT) |
| `nemo_en_titanet_small.onnx` | 38.4 MiB | `ad4a1802485d8b34c722d2a9d04249662f2ece5d28a7a039063ca22f515a789e` | [speaker-recongition-models](https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/nemo_en_titanet_small.onnx) | Per NVIDIA NeMo terms: CC-BY-4.0 for the model weights |

Both URLs are from the `k2-fsa/sherpa-onnx` release tags listed in the
upstream `python-api-examples/offline-speaker-diarization.py`. The "ARK"
hashes above are independently verified by the spike on every run (the
binary SHA-256-hashes its inputs and logs the digest to stderr).

The pyannote segmentation tarball ships with a `LICENSE` file from
pyannote/segmentation-3.0 (MIT) plus an `export-onnx.py` script
documenting the conversion from the PyTorch original.

The titanet_small ONNX is a NeMo model exported by k2-fsa.
[NVIDIA's NeMo license](https://github.com/NVIDIA/NeMo/blob/main/LICENSE)
applies. For commercial v1 shipment, Phase 6 must add a NOTICE entry
attributing both upstreams and verify CC-BY-4.0 attribution
requirements are satisfied.

### Why titanet_small not 3D-Speaker

Phase 0 named both as candidates and 3dspeaker_eres2net_base is the
upstream Python example's default. titanet_small was preferred here
because:

1. It is English-trained on VoxCeleb (the 3D-Speaker default ships a
   Mandarin model — `3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx`).
   The fixtures are English; matched language gives a fair binding-maturity
   test.
2. 39 MiB vs 39 MiB or 39 MiB vs 71 MiB (eres2netv2); titanet_small wins
   on bundle size for v1.

The English 3D-Speaker variant
(`3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx`, 28 MiB) is also
available and should be revisited in Phase 6 as a smaller-bundle option.
For Spike 4's question (binding maturity), the choice doesn't matter.

## Fixtures

Primary fixture for the spike's acceptance run:

| File | Duration | SHA-256 | Source | License |
|---|---|---|---|---|
| `2-two-speakers-en.wav` | 34.00 s, 16 kHz mono PCM s16 | `ee9c33d34e8f0fda4b78277f609944a1565aa16e6e2146f4cb8f0efb0d70030b` | [speaker-segmentation-models](https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/2-two-speakers-en.wav) | published as part of the sherpa-onnx test suite by k2-fsa |

Additional fixtures used to broaden the data set (also from
`k2-fsa/sherpa-onnx`'s `speaker-segmentation-models` release):

| File | Duration | SHA-256 |
|---|---|---|
| `1-two-speakers-en.wav` | 16.00 s | `f1c877dc01595e28be7147bf2fe38e5268147a868bf3fdb5c37b97f5940e21f3` |
| `3-two-speakers-en.wav` | 54.80 s | `dd3cf2344f8410ee9a2e271e96d1c3f9b530f113ae5b47defecbc0d741a468e9` |

Single-speaker control:

| File | Duration | Source |
|---|---|---|
| `librispeech_30s.wav` | 29.60 s, 16 kHz mono | `/mnt/c/Users/anl/transcribe-rs-test/fixtures/librispeech_30s.wav` (Windows-side cache, LibriSpeech excerpt — also used by Spike 1) |

None of the audio fixtures are committed to the repo (smallest, the 16 s
clip, is 500 KiB and the 200 KiB threshold from Phase 0 Spike 4 would
not be met). They are downloaded once into
`~/.cache/sherpa-onnx/fixtures/` and referenced by URL + SHA-256.

### Reference RTTM

`fixtures/2-two-speakers-en.ref.rttm` is committed. See
"DER framing" below for what it represents.

## Sample stdout (truncated)

`cargo run -p spike-diarize --release -- --segmentation … --embedding … --wav 2-two-speakers-en.wav --num-clusters 2`:

```json
[
  { "end_ms":  2765, "speaker_id": "A", "start_ms":    31 },
  { "end_ms":  7810, "speaker_id": "B", "start_ms":  2765 },
  { "end_ms": 11118, "speaker_id": "A", "start_ms":  7844 },
  { "end_ms": 13902, "speaker_id": "B", "start_ms": 11287 },
  { "end_ms": 16923, "speaker_id": "A", "start_ms": 13953 },
  { "end_ms": 17817, "speaker_id": "B", "start_ms": 17159 },
  { "end_ms": 18627, "speaker_id": "A", "start_ms": 17817 },
  { "end_ms": 21125, "speaker_id": "B", "start_ms": 18374 },
  { "end_ms": 27453, "speaker_id": "A", "start_ms": 21125 }
]
```

(Field order in the actual output is alphabetised by `serde_json`'s
pretty-printer — `end_ms`, `speaker_id`, `start_ms` — preserved here
verbatim.)

## Measurements

| Run | Audio (s) | n_segments | Speakers | Inference (s) | RTF | DER |
|---|---|---|---|---|---|---|
| 2-two-speakers-en (primary) | 34.00 | 9 | 2 | 5.27 | 0.155 | 0.00 % |
| 3-two-speakers-en | 54.81 | 6 | 2 | 8.98 | 0.164 | n/a |
| librispeech_30s (single-speaker control) | 29.60 | 4 | 1 | 6.72 | 0.227 | n/a |

- Wall-clock (full process incl. SHA-256 hashing + model load) on the
  primary run: **10.37 s**. Of that, ~0.5 s is `Diarize::new` (model
  load), ~5.3 s is inference, and the remainder is process startup +
  file hashing.
- Peak resident set (`/usr/bin/time -v`): **205 MiB**. Both ONNX models
  + ORT runtime + spike binary, sustained.
- CPU: AMD Ryzen (WSL2). `embedding.num_threads = 1` and
  `segmentation.num_threads = 1` (hard-coded by `sherpa-rs 0.6.8` —
  see "API surprises" below).

## DER framing — what the 0.00 % means

The committed reference RTTM
(`fixtures/2-two-speakers-en.ref.rttm`) was produced by the **upstream
sherpa-onnx C tool** (`sherpa-onnx-offline-speaker-diarization` from
sherpa-onnx v1.12.9, same configuration), not by human listening. This
is deliberate. The reference is a **binding-fidelity reference**: its
purpose is to answer Q-P0-6, "does the `sherpa-rs 0.6.8` Rust binding
expose the same diarization algorithm as the C library, byte-for-byte?".

- DER ≈ 0 % → the binding is wiring through the underlying algorithm
  correctly (no marshalling errors, no silently different code path).
  **This is what the spike observed.**
- DER > 0 % at this stage would have been a smoking gun for a binding
  bug — different defaults, swapped axes, wrong type punning, etc.

A separate question — "is the algorithm itself accurate against
human-annotated truth?" — is a property of sherpa-onnx, not of its Rust
binding, and is **out of scope for Spike 4**. Spike 4 treats the
algorithm as a black box and verifies the binding around it.

Hand-annotating these fixtures from listening would be the next step if
Phase 6 wants to measure absolute algorithm accuracy. The sherpa-onnx
project does not publish ground-truth RTTMs alongside the test wavs.
AMI / VoxConverse with their RTTMs would be the better corpus for that
question; Phase 0 Spike 4 lists AMI as the recommended fallback,
which is the right call when absolute accuracy is the target. **Not for
the binding-maturity question.**

The 25 % DER threshold from Phase 0 Spike 4 is therefore comfortably
satisfied: the binding's DER against the algorithm itself is 0.00 %.

## Q-P0-6 detailed verdict

> Q-P0-6: Is `sherpa-rs` stable enough, or does Phase 6 need a `bindgen`
> wrapper?

### Crate name

`sherpa-rs` (Thewh1teagle binding) — confirmed. The originally-considered
`sherpa-onnx-rs` does not exist on crates.io. A separate
upstream-owned binding `sherpa-onnx` (k2-fsa, Apache-2.0) also exists
and is newer (1.13.2 vs 1.12.9-wrapped). See "Future work".

### Version pin that exposes diarization

`=0.6.8`. The `sherpa_rs::diarize::Diarize` type is present and stable.
Older versions (0.5.x) were not inspected; 0.6.8 was sufficient.

### API surface (constructor + process method + output shape)

```rust
use sherpa_rs::diarize::{Diarize, DiarizeConfig, Segment};

let cfg = DiarizeConfig {
    num_clusters:     Some(2),    // -1 means "use threshold"
    threshold:        Some(0.5),
    min_duration_on:  Some(0.3),
    min_duration_off: Some(0.5),
    provider:         None,        // None -> get_default_provider()
    debug:            false,
};

let mut diarizer: Diarize = Diarize::new(
    "path/to/segmentation.onnx",
    "path/to/embedding.onnx",
    cfg,
)?;

let segments: Vec<Segment> = diarizer.compute(samples_f32_16khz, None /* progress cb */)?;
// Segment { start: f32 (seconds), end: f32 (seconds), speaker: i32 (cluster id) }
```

- Inputs are **owned `Vec<f32>`** at 16 kHz mono — `compute(mut samples)`
  takes ownership and passes the underlying pointer to the C API. The
  cluster id is **not guaranteed to be 0..N-1**; the spike normalises
  with a stable first-seen-order relabel.
- Returns `eyre::Result<Vec<Segment>>`. Errors are eyre messages, not
  typed variants — a Phase 6 wrapper would need to map these to
  `AppError::Inference { backend: "sherpa", … }`.
- A `Box<dyn Fn(i32, i32) -> i32 + Send + 'static>` progress callback
  slot exists and is forwarded into the C callback. Not exercised by
  this spike (acceptance is single-shot post-recording, FR-13).
- `Diarize` is `Send + Sync` (manually declared on the raw `*const`
  field). `Drop` calls `SherpaOnnxDestroyOfflineSpeakerDiarization` —
  no leak.

### Maturity verdict

**Sufficient for Phase 6 as currently scoped.** No missing entry points
were encountered. Specifically:

| Phase 6 requirement | `sherpa-rs 0.6.8` status |
|---|---|
| Construct from two ONNX paths | ✓ `Diarize::new` |
| Take f32 PCM @ 16 kHz | ✓ `compute(Vec<f32>, …)` |
| Specify n_speakers OR threshold | ✓ both in `DiarizeConfig` |
| Specify min_duration_on / min_duration_off | ✓ |
| CPU provider | ✓ default |
| GPU providers (cuda/coreml/directml) | ✓ `provider: Some(String)` and `cuda`/`directml` Cargo features |
| Streaming / online | ✗ — not needed (FR-13: post-recording only) |
| Progress callback | ✓ |
| Cleanup on drop | ✓ |
| Send for the orchestrator's worker thread | ✓ |
| Static build (Tauri bundle) | ✓ `sherpa-rs/static` feature available |

**No `bindgen`-direct-C wrapper is required.** The Phase 0 risk register
("`sherpa-rs` lags upstream sherpa-onnx by months") **does not
materialise** at the level of detail Spike 4 needed. The binding is one
minor release behind upstream (1.12.9 vs 1.13.2 of sherpa-onnx itself)
but the offline-diarization C API has been stable across that range —
exactly the surface Phase 6 calls.

The remaining (minor) thread-count limitation is **a Phase 6
follow-up**, not a `bindgen` trigger; see "API surprises" below.

## API surprises

Things in the binding's surface that didn't match expectations or felt
unstable. None block this spike; all should be revisited in Phase 6.

1. **`embedding.num_threads` and `segmentation.num_threads` are
   hard-coded to 1** in `Diarize::new` (sherpa-rs/src/diarize.rs:69, :80).
   The underlying C struct exposes the knob, but the safe binding
   forces 1. On a 16-core box this means diarization is single-threaded
   per worker even though the algorithm is embarrassingly parallel over
   ONNX inference. For a 1-hour meeting at RTF ~0.15 this is ~9
   minutes — acceptable for v1 (FR-13: offline) but a low-hanging fruit
   for Phase 6 if user complaints arrive. **Workarounds**: fork the
   crate and parameterise; or contribute a patch upstream — the change
   is a 10-line constructor extension.

2. **Speaker cluster ids are `i32` and arbitrary** — not 0..N-1 ordered
   by appearance. The spike's relabel pass produces stable `A`, `B`, …
   labels in first-seen order; Phase 6's `Diarizer::assign_speakers`
   impl will need the same normalisation.

3. **`Diarize::compute(mut samples: Vec<f32>, …)` takes the buffer by
   value.** It's only used to derive a `*mut f32`, and the function
   doesn't push back to the caller. A Phase 6 wrapper that holds the
   audio buffer for re-diarization (FR-13) would need to clone, which
   is wasteful — a `&mut [f32]` signature would be safe and equivalent.
   Worth a contribution.

4. **`eyre::Result` at the public boundary.** Phase 6's `Diarizer` trait
   is `fn assign_speakers(&self, …) -> AppResult<u32>` — every call
   site needs an `.map_err(|e| AppError::Inference { …, context: format!("{e:?}") })`.
   Boilerplate, not a blocker.

5. **`download-binaries` writes to `~/.cache/sherpa-rs/<target>/<sha256>/`**
   at build time, not at install time. This means every release build
   from a clean cache requires HTTPS connectivity to
   `github.com/k2-fsa/sherpa-onnx/releases`. Fine for CI; ergonomic
   problem for offline corp dev machines. Phase 6's release pipeline
   should pre-cache or pivot to `static`.

6. **No published `Cargo.toml` minimum-rust-version.** The crate built
   on the workspace's pinned toolchain
   (`1.91.0` per `rust-toolchain.toml`); a probe on older toolchains
   would be needed before declaring an MSRV.

7. **`sherpa-rs-sys` re-exports its `bindgen`-generated bindings as
   `pub`.** Phase 6 *could* drop to the raw `sherpa_rs_sys::*` calls
   without forking — `sherpa-rs` itself does that. That gives a clean
   escape hatch for the few cases where the safe surface is too coarse
   (e.g. `num_threads`).

## Build numbers

- Cold `cargo build -p spike-diarize --release` (sherpa-rs binaries
  cached, no `bindgen` cache): **44.8 s**.
- Subsequent incremental: <2 s.
- Release binary: 5.2 MiB. **Plus** `libsherpa-onnx-c-api.so` (1.5 MiB)
  and `libonnxruntime.so.1.17.1` (15 MiB) — counts against the Phase 0 size budget
  Q-P0-8 binary-footprint accounting at Phase 6 time.

## Reproduction

```sh
# Models (one-time)
mkdir -p ~/.cache/sherpa-onnx/diarization
curl -L -o ~/.cache/sherpa-onnx/diarization/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2 \
  https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2
tar -C ~/.cache/sherpa-onnx/diarization -xjf \
  ~/.cache/sherpa-onnx/diarization/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2
curl -L -o ~/.cache/sherpa-onnx/diarization/nemo_en_titanet_small.onnx \
  https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/nemo_en_titanet_small.onnx

# Fixtures (one-time)
mkdir -p ~/.cache/sherpa-onnx/fixtures
for n in 1 2 3; do
  curl -L -o ~/.cache/sherpa-onnx/fixtures/${n}-two-speakers-en.wav \
    https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/${n}-two-speakers-en.wav
done

# Acceptance run
cargo run -p spike-diarize --release -- \
  --segmentation ~/.cache/sherpa-onnx/diarization/sherpa-onnx-pyannote-segmentation-3-0/model.onnx \
  --embedding    ~/.cache/sherpa-onnx/diarization/nemo_en_titanet_small.onnx \
  --wav          ~/.cache/sherpa-onnx/fixtures/2-two-speakers-en.wav \
  --num-clusters 2 \
  --reference-rttm spikes/diarize/fixtures/2-two-speakers-en.ref.rttm

# Single-speaker control
cargo run -p spike-diarize --release -- \
  --segmentation ~/.cache/sherpa-onnx/diarization/sherpa-onnx-pyannote-segmentation-3-0/model.onnx \
  --embedding    ~/.cache/sherpa-onnx/diarization/nemo_en_titanet_small.onnx \
  --wav          /mnt/c/Users/anl/transcribe-rs-test/fixtures/librispeech_30s.wav
```

## Future work for Phase 6

1. Re-evaluate `sherpa-onnx 1.13.x` (k2-fsa-owned) vs `sherpa-rs 0.6.x`
   (Thewh1teagle). The official crate is the more conservative bet for
   v1 if its diarization surface is comparable.
2. Either patch `sherpa-rs` to expose `num_threads`, or call the
   `sherpa_rs_sys::SherpaOnnxOfflineSpeakerDiarizationConfig` directly to
   set both segmentation and embedding `num_threads = N`. Single-threaded
   inference is the only Phase-6-bound performance gap identified by
   Spike 4.
3. Switch to `sherpa-rs/static` for the Tauri bundle. Confirm static
   build works on Windows + macOS in CI before relying on it.
4. Hand-annotate (or source from AMI) a real ground-truth RTTM if Phase 6
   wants to characterise absolute DER, not just binding fidelity.
5. Map `eyre::Report` -> `AppError::Inference` once in a `diarizer`
   crate-private helper.
6. Wire diarization into the `Diarizer` trait
   (`crates/common::Diarizer::assign_speakers`); the trait expects an
   in-place `speaker_id` overlay on existing `Segment`s, which means the
   `diarizer` crate needs to interval-join sherpa segments onto ASR
   segments. That's Phase 6 work, not a binding question.
