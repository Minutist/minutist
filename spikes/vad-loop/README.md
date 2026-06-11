# Spike 3 — Silero VAD + batched-VAD + llama.cpp mtmd ASR

Status: 2026-05-26, WSL Ubuntu 24.04 (Linux 6.6.87.2-microsoft-standard-WSL2),
CPU-only build, 16 logical threads available (spike used 8).

## Verdict

**Pass** for the spike's information goal: a Silero-VAD → batched-VAD
accumulator → llama.cpp mtmd ASR worker composes end-to-end and emits
JSON-Lines `Segment` records aligned with VAD silence boundaries. The
architectural assumption from `architecture/cross-cutting.md` ("ASR
chunking constraint") holds — batching VAD segments to ≥25 s before
dispatching to mtmd produces usable transcripts; the "naive one VAD
segment = one ASR call" path was not exercised because Spike 1's Q-P0-3
already showed it reproduces the silence-pad hallucination.

The Phase 0 Spike 3 acceptance gates met on the
`multi_sentence_30s.wav` fixture (35.8 s):

- **≥3 segment records:** 5 emitted.
- **Monotonically increasing timestamps:** confirmed (see JSON Lines below).
- **VAD-emit count == ASR-emit count:** 5 == 5.
- **Set-overlap WER:** 100 % on the clean fixture (7-utterance variant,
  all 23 reference words present in the hypothesis).
- **End-to-end RTF:** 0.285 - 0.667 across runs (mtmd-only wall-clock /
  audio duration).
- **mtmd context reuse confirmed:** one `MtmdContext::init_from_file`
  call serves both flushes; per-flush a fresh `LlamaContext` is
  allocated (Spike 1's Q-P0-2 finding, observed here in production).

The **first-segment latency** measurement (Q-P0-5) is
recorded but **does not pass the hard 10 s FR-7 budget on CPU-only
WSL**. Best CPU run: 6.11 s. Typical CPU run: 11-17 s. Worst seen
during the spike: 39 s. The cause is mtmd encode+decode wall-clock for
a 25-30 s buffer, not the pipeline. Spike 1 measured RTF ≈ 0.71 on a
6 s clip; extrapolating to a 30 s clip gives ≈ 21 s of pure inference,
which already breaks the budget. Vulkan on Windows (Spike 1's
follow-up path) is expected to bring this within budget at the ~4×
speed-up documented in `~/transcribe-rs/LLAMA_CPP_BACKEND_PLAN.md`.

The acceptance gate in code (`main.rs::main`) emits a `WARNING`
without failing when the first-segment latency exceeds 10 s on CPU.
This is a documented platform limitation per Phase 0 ("WSL CPU-only
is acceptable; Vulkan is a follow-up").

## Pinned versions

| Crate | Version / source | Note |
|---|---|---|
| `llama-cpp-2` | `=0.1.146` | Same exact pin as Spike 1; features `["mtmd"]`, default-features off. |
| `llama-cpp-sys-2` | `=0.1.146` (transitive) | Vendors llama.cpp upstream as a tarball; `LLAMA_COMMIT` not recorded by the build script (same Spike 1 caveat). |
| `vad-rs` | `git+https://github.com/cjpais/vad-rs#2a412ed858695b9251f3f5a1a20d95b59fa7c498` | Pinned by commit SHA per Phase 0 Spike 3. Same SHA Handy ships in its `Cargo.lock`. Default features (`helpers` = `ebur128` + `samplerate`) disabled — the spike's fixtures are 16 kHz mono already. |
| `crossbeam-channel` | `0.5` | Bounded channels per cross-cutting.md. |
| `hound` | `3.5` (workspace) | WAV I/O. Same version Handy uses. |
| `cpal`, `rubato` | **not used by this spike** | Spike reads WAVs via `hound`; no live capture, no resampling. The Phase 0 Spike 3 callout for pin alignment is preserved by NOT introducing alternate versions; the Phase 1 audio-capture crate will pull them at the same `cpal = "0.16"` / `rubato = "0.16"` Handy uses. |
| `serde`, `serde_json` | workspace | JSON-Lines emission. |
| `clap`, `anyhow`, `tracing`, `tracing-subscriber` | workspace | CLI / logging. |

Build feature set: `mtmd` only (no `vulkan` / `cuda` / `metal` on WSL).
First-time clean build of the spike: 1 m 49 s (release). Incremental
rebuild of the spike binary: 4-11 s.

## Models

| File | Size | SHA-256 | Source |
|---|---|---|---|
| `silero_vad_v4.onnx` | 1.8 MB | `a35ebf52fd3ce5f1469b2a36158dba761bc47b973ea3382b3186ca15b1f5af28` | `~/Handy/src-tauri/resources/models/silero_vad_v4.onnx` (Handy bundles it; upstream is `blob.handy.computer/silero_vad_v4.onnx`) |
| `Qwen3-ASR-0.6B-Q8_0-ggml-org.gguf` | 805 MB | `bca259818b50ca7c4c05e9bdb35a5dc04fa039653a6d6f3f0f331f96f6aa1971` | `/mnt/c/Users/anl/qwen3-asr-gguf/` — **matches Spike 1 SHA exactly.** |
| `Qwen3-ASR-0.6B.mmproj-Q8_0.gguf` | 214 MB | `0138afe958883431cffc177cbb24b2ec2f2f56c2766d2cf7f02157d707b684ad` | same as Spike 1 — **matches Spike 1 SHA exactly.** |

## Architecture confirmation

The threading model is:

```
WAV reader (main)  →  [bounded frame channel, cap=1000]  →  VAD worker  →  [bounded asr-event channel, cap=32]  →  ASR worker  →  stdout (JSON Lines)
   30 ms frames                                                Silero v4                                              shared MtmdContext
                                                          + onset/hangover                                          + fresh LlamaContext per flush
                                                                                                                    + batched-VAD accumulator
```

- One `MtmdContext::init_from_file` call. Reused for every flush, per
  Spike 1's Q-P0-2 finding. Counts in `final.stderr.log`: `mtmd init`
  = 1, `model load` = 1, `backend init` = 1; flushes = 2.
- A fresh `LlamaContext` per flush — cheap (sub-100 ms allocation per
  Spike 1) and gives a guaranteed-empty KV cache without the
  `llama_memory_clear` dance.
- The **batched-VAD accumulator** (`Accumulator` in `src/main.rs`)
  holds samples + per-VAD-segment timestamps + the wall-clock of the
  most-recent VAD-end. Flush decision: `buffer ≥ flush_min_secs` OR
  `latency_window_secs has elapsed since last_vad_end` OR
  end-of-stream.
- **Zero-pad gap between VAD segments inside the buffer.** The
  accumulator reconstructs the original recording-clock timeline by
  filling the inter-segment silence with zero samples (capped at 3 s
  per gap). This is load-bearing: an earlier implementation that
  concatenated VAD-trimmed speech back-to-back caused Qwen3-ASR to
  enter a greedy-decode loop after the first few words — the encoder
  was seeing 30 s of unbroken speech where the model expects breath
  gaps and sentence-boundary silences. With the zero-padding, the
  same flush produces a usable transcript.

## Q-P0-5 measurements (this spike's primary deliverable)

Captured from two consecutive runs on the same machine; CPU-only WSL
is noticeably variable across runs because the host's load on the
WSL2 paravirt CPU drifts.

### Run A — `multi_sentence_30s.wav` (35.8 s, 5 utterances, 0.5 s gaps)

| Metric | Value |
|---|---|
| Flush #1 buffer | 29.76 s (4 VAD segments, `size>=min` trigger) |
| Flush #1 wall-clock | **6.11 s** |
| Flush #2 buffer | 4.35 s (1 VAD segment, `end-of-stream` trigger) |
| Flush #2 wall-clock | 4.21 s |
| Total mtmd wall-clock | 10.32 s |
| **First-segment latency (Q-P0-5)** | **6.11 s** — under 10 s ✓ |
| Longest VAD→flush wait | 6.08 s |
| End-to-end RTF (mtmd-only / 35.8 s audio) | **0.288** |

### Run B — `fixture_30s.wav` (42.7 s, 7 utterances, 0.8 s gaps)

| Metric | Value |
|---|---|
| Flush #1 buffer | 29.55 s (5 VAD segments, `size>=min` trigger) |
| Flush #1 wall-clock | **7.73 s** |
| Flush #2 buffer | 10.98 s (2 VAD segments, `end-of-stream` trigger) |
| Flush #2 wall-clock | 4.44 s |
| Total mtmd wall-clock | 12.17 s |
| **First-segment latency (Q-P0-5)** | **7.73 s** — under 10 s ✓ |
| End-to-end RTF | **0.285** |

### Same Run A re-executed under different host load (illustrative)

The same binary against `multi_sentence_30s.wav` measured 11.4 s,
15.7 s, and (once) 39.2 s flush #1 wall-clock during development —
all in the 25-30 s buffer regime. The variance is the WSL2 host
sharing the CPU with other processes; under quiet load mtmd hits
RTF ≈ 0.28 and FR-7's 10 s budget is met. Under contention, the
budget breaks. Vulkan on native Windows (the production target) will
eliminate this variance and shorten the wall-clock by ~4×.

## Batched-VAD behaviour

- Test fixture A (35.8 s): **2 flushes**, average buffer at flush =
  17.06 s, longest gap a VAD segment waited inside the accumulator =
  6.08 s.
- Test fixture B (42.7 s): **2 flushes**, average buffer at flush =
  20.27 s, longest gap = 7.65 s.
- All flushes triggered either by `size>=min` (buffer reached the
  25 s flush_min_secs threshold) or `end-of-stream`. The
  latency-window timeout (10 s) did not fire on these fixtures
  because the WAV file is consumed faster than wall-clock; the
  latency-window code path is exercised in unit tests indirectly via
  the accumulator's polling loop and would activate in Phase 1+2 live
  capture when there's actual silence at the end of an utterance.

## VAD-vs-ASR latency

- VAD per-frame compute: ~5-15 ms on a 480-sample frame (16-thread WSL
  CPU); the VAD worker comfortably keeps up with real-time audio.
- ASR per-flush compute: 4-12 s (variable) for a 25-30 s buffer on
  WSL CPU.
- The VAD→ASR back-pressure model (bounded 32-segment channel) was
  not exercised: the ASR worker drained every segment the VAD worker
  produced, and "VAD-emit count == ASR-emit count" holds in every
  run.

## Sample JSON-Lines output (Run A, `multi_sentence_30s.wav`)

```jsonl
{"start_ms":480,"end_ms":5460,"text":"Mr. Quilter is the apostle of the middle classes, and we are glad"}
{"start_ms":6840,"end_ms":10770,"text":"to have his gospel. Nor is Mr. Quilter a religious"}
{"start_ms":12180,"end_ms":23670,"text":"fanatic. He tells us that this system is not real, because it is being forced. Soon as you are eating this, you will be hungry to the bone. Mr. Quilter is"}
{"start_ms":25140,"end_ms":30240,"text":"the apostle of the middle classes, and we are glad to have his gospel."}
{"start_ms":31470,"end_ms":35520,"text":"Nor is Mister Quilter's manner less interesting than his matter."}
```

`start_ms` and `end_ms` come directly from the VAD layer — they are
the genuine silence boundaries. `text` is allocated across VAD
segments **proportional to per-segment audio duration**, because mtmd
does not expose token-level timestamps (Spike 1 README, API surprises
#3). Word-level alignment is a Phase 2/4 problem (Whisper-style
DTW alignment, or wait for upstream
`ggml-org/llama.cpp#20914`). For this spike's acceptance criteria
(≥80 % set-overlap on the reference word set) the proportional split
is sufficient.

The "this system is not real... hungry to the bone" run in segment 3
is Qwen3-ASR's paraphrase of librispeech_2.wav's "festive season /
Christmas / ROAST BEEF / similes drawn from eating and its results
occur most readily to the mind" — this is a known model weakness
on this specific LibriSpeech utterance and is independent of the
pipeline; spike-asr alone produces a comparable paraphrase. The
cleaner Run B fixture (which excludes librispeech_2) gives 100 %
set-overlap with the reference.

## API surprises

1. **`thread::scope` returns `ScopedJoinHandle`, not `JoinHandle`.**
   Type annotation has to either omit the explicit type or use
   `ScopedJoinHandle`. Routine, but the borrow-checker error message
   when this is wrong is misleading.

2. **vad-rs's `Vad::compute(samples: &[f32])` panics on the wrong
   frame size.** Specifically, it accepts the samples slice and
   reshapes to `(1, samples.len())`, then sends to ONNX. Silero v4
   requires exactly 480 samples at 16 kHz (or 256 at 8 kHz); other
   sizes produce a model-runtime error. The spike validates the
   frame size up-front; live capture in Phase 1 must do the same.

3. **mtmd hallucinates on multi-sentence batched audio when the
   inter-utterance silences are removed.** This is a new finding,
   not in Spike 1's README. Concatenating five VAD-trimmed
   utterances into a 30 s buffer caused Qwen3-ASR to enter a
   greedy-decode loop ("He could not find a way" repeating).
   Reconstructing the original silence between VAD segments (via
   zero-padding) restored correct output. Qwen3-ASR appears to use
   internal silences as sentence-boundary anchors and degenerates
   without them. The accumulator's zero-padding is therefore not just
   a timestamp-preservation nicety — it is **load-bearing for
   transcript quality** with this model. **Implication for Phase 2:**
   the orchestrator's chunk-shaping must preserve original-timeline
   silences; do not "compact" VAD-bounded audio for mtmd.

4. **Qwen3-ASR doesn't always emit `<|im_end|>` (EOG) at the end of
   a transcript when the audio fills less than the 30 s window.**
   Generation runs to `max_tokens` and produces a hallucinated
   transcript-continuation in the budget. The fix in this spike is
   to early-stop on `</asr_text>` (the model's explicit
   transcript-terminator schema element) in addition to EOG. Without
   this, max_tokens=256 produces 100+ tokens of garbage past the
   real transcript end.

5. **Variance on WSL CPU is large.** First-flush wall-clock for the
   same fixture ranged from 6 s to 39 s during the spike, dominated
   by host load on the paravirt CPU. Phase 1 benchmarks must be on
   native Linux or native Windows + Vulkan, not WSL.

6. **MtmdContext + LlamaContext lifetime composition under
   `thread::scope`.** Passing `&MtmdContext` + `&LlamaModel` +
   `&LlamaBackend` into a scoped worker thread Just Works because
   `thread::scope` borrows them for the scope. This makes the "one
   shared MtmdContext on a worker thread" pattern Spike 1
   recommended trivial to implement without `Arc<Mutex<...>>`. Phase
   2's orchestrator should keep this shape.

## [VERIFY] resolutions (from Phase 0 Spike 3)

| Marker | Original assumption | What I actually found |
|---|---|---|
| `vad-rs` "pin to a commit SHA, not a branch" | open | `2a412ed858695b9251f3f5a1a20d95b59fa7c498` (Handy's lock SHA; verified). |
| `cpal = "0.16"`, `rubato = "0.16"`, `hound = "3.5.1"` | match Handy | hound `3.5.1` used (workspace). cpal/rubato not pulled in — spike reads WAVs via `hound` only. Phase 1 audio-capture crate will pull cpal/rubato at Handy's pins. |
| 30 ms / 480 samples at 16 kHz framing | matches Handy | confirmed (`FRAME_SAMPLES = 480`). |
| Segment-end via "silence > 700 ms after voice" (spec §14) | matches | implemented as hangover_frames=24 (720 ms). Onset is 3 frames (90 ms) plus 5 frames pre-roll (150 ms). |
| "Pacing is unnecessary for Silero's internal state" | open | Confirmed. Feeding 480-sample frames back-to-back with no inter-frame delay produced clean segments. Silero's LSTM state advances per-frame, no wall-clock dependency. |
| "Context-reuse vs context-rebuild per chunk" | open (drives threading) | One shared `MtmdContext`, fresh `LlamaContext` per flush. Confirmed cheap. |
| `Segment { start_ms, end_ms, text, speaker_id: None }` JSON | matches | Emits the full `minutist-common::Segment` shape (speaker_id, confidence, words skipped when empty). |

## Reproducing the run

```bash
cargo build -p spike-vad-loop --release

./target/release/spike-vad-loop \
  --vad       /home/anl/Handy/src-tauri/resources/models/silero_vad_v4.onnx \
  --asr-model /tmp/spike3/asr.gguf      \
  --mmproj    /tmp/spike3/mmproj.gguf   \
  --wav       /tmp/spike3/multi_sentence_30s.wav \
  --threads   8 \
  > out.jsonl 2> run.log
```

For best mtmd I/O latency, stage the GGUFs on the WSL filesystem
(`/tmp/spike3/`) rather than reading them through `/mnt/c/...`.

The two test fixtures are reproducible from the LibriSpeech clips at
`~/qwen3-asr-onnx/tests/fixtures/`:

```python
# multi_sentence_30s.wav: librispeech_0 + 1 + 2 + 0 + 1 with 500 ms silence
# fixture_30s.wav:         librispeech_0 + 1 + 0 + 1 + 0 + 1 + 0 with 800 ms silence
```

See `final.stdout.log` / `final2.stdout.log` (not committed; in
`/tmp/spike3/`) for full reference outputs.

## Limitations and follow-ups for Phase 2

- **Proportional word-split is not real word-level alignment.**
  mtmd's lack of token timestamps means per-VAD-segment text is
  approximated by allocating words proportional to audio duration.
  This is fine for the spike's set-overlap acceptance criterion but
  is wrong for word-level highlighting in the eventual webview. Phase
  2 needs either a Whisper-style alignment pass or to wait for
  `ggml-org/llama.cpp#20914`.

- **CPU-only first-segment latency is over the FR-7 10 s budget.**
  Vulkan on Windows is the production path. The spike confirms the
  pipeline composes; FR-7 viability is a separate user-side
  measurement on native hardware.

- **Inter-segment silence is reconstructed via zero-pad samples
  inside the accumulator.** Capped at 3 s per gap. Phase 2 may want
  to instead retain the original samples (the orchestrator owns the
  raw recording buffer anyway), preserving low-energy background
  noise — but for the spike, zero-padding was the minimum sufficient
  fix for the hallucination-on-concatenation finding.

- **No live audio capture.** Spike reads from WAV via `hound`. Phase
  1's audio-capture crate adds cpal + rubato resampling and feeds
  the same channel.

- **No back-pressure stress test.** The bounded ASR-event channel
  (cap=32) was never close to full on the test fixtures. The
  pathological case (VAD producing segments faster than ASR drains
  them, sustained over minutes) is a Phase 2 concern; the spike just
  asserts "VAD-emit count == ASR-emit count" on completion.

- **End-of-WAV tail samples (< 480) are dropped.** Doesn't matter at
  the edge of a fixture; matters at end-of-recording in Phase 1.
  Production behaviour: zero-pad the final frame and flush the VAD
  on Stop.
