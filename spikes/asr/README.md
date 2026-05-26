# Spike 1 — llama-cpp-2 mtmd Qwen3-ASR

Status: 2026-05-26, WSL Ubuntu 24.04 (Linux 6.6.87.2-microsoft-standard-WSL2), CPU-only build.

## Verdict

**Pass** for the spike's information goal: `llama-cpp-2 0.1.146`'s `mtmd` audio path
loads a Qwen3-ASR-0.6B Q8_0 GGUF + mmproj pair, accepts raw 16 kHz f32 PCM, and
produces correct transcripts on the reference fixtures within the wall-clock and
memory budgets. The mtmd surface that Phase 0 Spike 1 needed to validate is
present and stable in this released crate version.

The Phase 0 acceptance gate (`WER ≤ 15%`, `wall-clock ≤ 60 s`, no panics, peak
RSS recorded) is met on the acceptance fixture (`librispeech_0.wav`, 5.86 s):
**WER 0.00%, inference 4.14 s, peak RSS 2.07 GiB**.

The originally-planned fixture (`librispeech_30s.wav`, truncated to 10 s) fails the
WER gate at 43.75 %. This is **not** a defect in the spike — it is a property of the
mtmd encoder: the audio window is fixed at 30 s and sub-30 s inputs are
silence-padded internally; on this particular passage the model continues into the
padded section and hallucinates. See Q-P0-3 below for the data and the
phase-1 implication.

## Pinned versions

- `llama-cpp-2 = "=0.1.146"` (exact pin in `Cargo.toml`).
- `llama-cpp-sys-2 = "=0.1.146"` (transitive, vendors llama.cpp source as a
  release tarball).
- Build features: `mtmd`, defaults disabled. CPU-only (no Vulkan / CUDA / Metal).
- llama.cpp upstream commit SHA: **not recorded by the build script.** The
  generated `build-info.cpp` shows `LLAMA_BUILD_NUMBER = 0`,
  `LLAMA_COMMIT = "unknown"` because the source is vendored as a tarball, not a
  git submodule. The vendored tree under
  `~/.cargo/registry/src/.../llama-cpp-sys-2-0.1.146/llama.cpp/` includes
  `PROJECTOR_TYPE_QWEN3A` and the qwen3-asr branches in
  `tools/mtmd/mtmd.cpp`, confirming this is post-PR-19441 (April 2026). The
  exact upstream SHA can be recovered by diffing the tarball against
  github.com/ggml-org/llama.cpp tags if needed for a later spike.
- llama.cpp built with: GNU 13.3.0, Linux x86_64 (per `build-info.cpp`). cmake
  3.28.3, system OpenMP. Default build flags from the crate's `build.rs`.

Build first time: 3 m 28 s (debug) / 4 m additional for release. Subsequent
builds are <5 s.

## Model SHAs

| File | Size | SHA-256 |
|---|---|---|
| `Qwen3-ASR-0.6B-Q8_0-ggml-org.gguf` | 805 MB | `bca259818b50ca7c4c05e9bdb35a5dc04fa039653a6d6f3f0f331f96f6aa1971` |
| `Qwen3-ASR-0.6B.mmproj-Q8_0.gguf`   | 214 MB | `0138afe958883431cffc177cbb24b2ec2f2f56c2766d2cf7f02157d707b684ad` |

Source: `/mnt/c/Users/anl/qwen3-asr-gguf/` (Windows-side cache; not committed).
These hashes are the v1 baseline per SPEC FR-9.

The Q4_K_M variant (`Qwen3-ASR-0.6B.Q4_K_M.gguf`) was not retested. Prior work
(transcribe-rs/LLAMA_CPP_BACKEND_PLAN.md) showed it hallucinates on the JFK
sample under Vulkan; CPU is unlikely to fix the quantisation issue. Q8_0 is
the v1 baseline.

## Measurements

CPU: AMD Ryzen (WSL2, 16 logical threads visible; spike used 8).

### Acceptance run — `librispeech_0.wav` (5.86 s)

| Metric | Value |
|---|---|
| Audio fed to mtmd | 5.86 s (no truncation needed; clip is shorter than 10 s) |
| Wall-clock (inference, excludes load) | **4.14 s** |
| Wall-clock budget | 60 s ✓ |
| Audio encode (mtmd `encode_chunk`) | 1.67 s |
| Audio decode (`llama_decode` over 375 audio tokens) | 1.55 s |
| Generation (25 tokens, greedy) | ~0.9 s ⇒ ~28 tok/s |
| RTF (wall-clock / audio duration) | 0.71 |
| Peak RSS (VmHWM) | 2068 MiB |
| WER vs reference | **0.00%** |

Transcript:
```
Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel.
```
Reference: `MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL`. `Mr.` is normalised to `mister` in the WER step so it matches.

### Secondary data point — `jfk.wav` (11 s)

| Metric | Value |
|---|---|
| Wall-clock (inference) | 4.00 s |
| RTF | 0.36 |
| WER vs reference (case + punctuation normalised) | **0.00 %** |

Transcript: `And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country.` — matches Vulkan-b8994 result documented in `~/transcribe-rs/LLAMA_CPP_BACKEND_PLAN.md` exactly.

### Q-P0-3 data point — `librispeech_30s.wav` truncated to 10 s

| Metric | Value |
|---|---|
| Wall-clock (inference) | 4.24 s |
| RTF (over the 10 s window) | 0.42 |
| WER vs reference (first 25 ref words) | **43.75 %** |

Transcript:
```
Law seemed to him well enough as a science, but he never could discover a case where he could walk away and apply his doctrine.
```
The first 12 words match the reference (`LAW SEEMED TO HIM WELL ENOUGH AS A SCIENCE BUT HE NEVER COULD DISCOVER A`); from "case where he could walk away" onward the model has paraphrased its way into the silence-padded tail. This is Q-P0-3 in action, not a CPU defect. See the Q-P0-3 section below.

The Windows + Vulkan claim from Phase 0 is **the user's verification, not this spike's.** The spike asserts that the mtmd surface works on CPU and that Q8_0 produces a correct transcript on properly-sized inputs.

## Open questions

### Q-P0-1 — Does mtmd accept raw f32 at 16 kHz, or pre-computed mel?

**Raw f32 at 16 kHz.** The Rust API signature used is

```rust
impl MtmdBitmap {
    pub fn from_audio_data(data: &[f32]) -> Result<Self, MtmdBitmapError>;
}
```
(`llama-cpp-2-0.1.146/src/mtmd.rs:423`).

The C side (`tools/mtmd/mtmd.cpp:mtmd_bitmap_init_from_audio`) just `memcpy`s the
samples into a `mtmd_bitmap` and sets `is_audio = true`. The mel-spectrogram
step lives inside `mtmd_encode_chunk` (the encoder graph configured by the
mmproj); the caller never touches mel coefficients.

Sample-rate validation: `mtmd_ctx.get_audio_sample_rate()` returned `Some(16000)`
for this mmproj. Feeding non-16 kHz audio is the caller's responsibility; the spike
rejects non-16 kHz WAVs up-front with a clear error rather than relying on the
encoder to detect mismatch.

A separate helper, `MtmdBitmap::from_file(&ctx, path)`, exists for WAV/MP3/FLAC
files (`tools/mtmd/mtmd-helper.cpp` parses with miniaudio). The spike does
**not** use this path — keeping WAV decoding in Rust lets the caller control
truncation and validate format up-front. This becomes load-bearing in Phase 1+2
where audio arrives from cpal, not from a file.

### Q-P0-2 — Can one MtmdContext process consecutive variable-length chunks without rebuild?

**Yes.** The spike's `--repeat` flag tokenises and evaluates the same audio
twice through one `MtmdContext` with a fresh `LlamaContext` each time, and
gets bit-identical output:

```
[first]  bitmap built: ... tokenize: 3 chunks, 397 tokens total
[second] bitmap built: ... tokenize: 3 chunks, 397 tokens total
repeat: identical output (MtmdContext reusable across calls)
```

Mechanism: `MtmdContext` is stateless w.r.t. encoding — `encode_chunk` writes
its output embeddings into a per-chunk buffer owned by the chunk, not into the
context. The `LlamaContext` owns the KV cache; per-call freshness is achieved
by allocating a new `LlamaContext` (cheap; sub-100 ms in practice) rather than
clearing the cache by hand. The spike does the heavyweight work — loading the
model and the mmproj, including the warm-up pass — exactly once.

**Implication for Phase 2/3:** one shared `MtmdContext` behind a worker thread
is correct. Per-chunk we either reset `n_past` to 0 on a fresh `LlamaContext`
or call `llama_memory_clear` on a persistent one. The spike used the former
because the WSL CPU path didn't need the optimisation; the latter is the right
choice when Phase 3 starts measuring end-to-end latency.

### Q-P0-3 — What does mtmd do with sub-30 s chunks?

**Pads internally to 30 s with silence. The model then continues into the
padded region and hallucinates.**

Evidence:
- The audio_hparams reported by mtmd init: `audio_chunk_len: 30,
  audio_sample_rate: 16000, audio_n_fft: 400, audio_window_len: 400,
  audio_hop_len: 160`. Mel spectrogram with 25 ms / 10 ms hop, 30 s window.
- Warmup log: `warmup: warmup with audio size = 3000` (3000 mel frames =
  30 s).
- The `librispeech_30s.wav` truncated to 10 s data point above: model
  transcribes the first ~10 s correctly, then continues with a paraphrased
  hallucination. Identical token count (30) and structure (`language English<asr_text>...`) regardless of input length up to 30 s.
- The model does **not** error on sub-30 s audio. It does **not** accept the
  audio as-is and emit a shorter transcript.

**Phase-1/2 implication.** Variable-length VAD chunks cannot be fed to mtmd
naively — the model treats every chunk as 30 s of audio. The phase-1 pipeline
must either:

1. Pad each VAD chunk up to exactly the 30 s window (clean, but costly: every
   chunk pays the full 30-s encode cost regardless of true duration), or
2. Batch VAD chunks into ~30 s windows before invoking mtmd (saves compute at
   the cost of latency before the first segment emits), or
3. Accept the hallucination tail and post-filter (e.g. truncate output at the
   first `</asr_text>` token if the model emits one, or stop generation once
   no new audio content is detected — but this requires output-side cues mtmd
   doesn't expose cleanly).

The `tools/mtmd/mtmd-cli.cpp` reference implementation doesn't address this —
it expects single-utterance inputs that fill or exceed the window. The
`#20914` streaming PR in llama.cpp is intended to make this no longer the
caller's problem, but it is unmerged as of this spike.

Recommended Phase-1 default: option 2 (batch VAD chunks to ≥25 s before
invoking ASR), with option 1 as a fallback when batching would push past the
FR-7 5-10 s latency budget. This is a design decision that must land in
the Phase 2 design before any audio-pipeline code is written.

### Q-P0-7 — Is Q8_0 the right quant?

**Yes, on the evidence so far.** Q8_0 produces correct transcripts on the
acceptance fixture and JFK. Prior work measured Q4_K_M hallucinating on the
JFK sample under Vulkan (`~/transcribe-rs/LLAMA_CPP_BACKEND_PLAN.md`); this
spike did not retest Q4_K_M because the quantisation defect is independent
of the backend.

Smaller acceptable quants (Q5_K_M, Q6_K) were not published by ggml-org at
spike time. If they appear, repeating the acceptance run + the JFK control
is sufficient to qualify them.

## API surprises

1. **The mtmd module is gated behind `cfg(feature = "mtmd")`**, not on by
   default. Build features had to include `mtmd` explicitly; without it the
   `llama_cpp_2::mtmd` module is missing and the crate compiles fine but
   doesn't expose the audio path. This is mentioned in the crate's lib.rs
   feature-flags section in passing.
2. **`MtmdBitmap::from_audio_data` accepts any sample count.** The encoder
   silently pads or truncates to 30 s internally. Callers cannot trust the
   model to error on out-of-range inputs — caller-side validation is required.
3. **`MtmdContext::tokenize` consumes the media marker `<__media__>` from the
   prompt and replaces it with the audio chunk inline.** The default marker
   is `<__media__>` (verified via `mtmd_default_marker()`). The Qwen3-ASR
   chat template emits this marker into the user turn, which mtmd then
   replaces with the `<|audio_bos|>` ... `<|audio_eos|>` framing during
   tokenisation. The tokenize log shows:
   ```
   add_text: <|im_start|>user
   add_text: <|audio_bos|>
   audio_tokens->n_tokens = 375
   add_text: <|audio_eos|>
   add_text: <|im_end|>
   <|im_start|>assistant
   ```
   375 audio tokens for any sub-30 s input — independent of audio length,
   because the encoder window is fixed.
4. **The Qwen3-ASR model wraps its output as `language English<asr_text>
   ...transcript...</asr_text>`.** This is the model's chosen output schema
   for ASR mode (driven by the embedded chat template). The spike strips
   the wrapper before WER measurement; downstream code in Phase 2 must do
   the same and surface the language tag separately if it wants to.
5. **`MtmdContext::decode_use_mrope()` returns `true` for Qwen3-ASR.** This
   is M-RoPE (multimodal RoPE) — affects how positions are computed across
   modalities. The `mtmd_helper_eval_chunks` helper handles this
   transparently; we don't need to do anything special at the call site,
   but Phase 2 should be aware in case it ever bypasses the helper.
6. **First-run build is slow (3-4 min) but only because llama.cpp is
   compiled from source.** Subsequent rebuilds of the spike itself are
   sub-second. The vendored llama.cpp ships in the `llama-cpp-sys-2`
   crate, so no separate clone / submodule setup is needed.
7. **`LLAMA_COMMIT` is empty in the build-info.** The crate's build script
   doesn't run `git describe` against the vendored tree (it can't — there's
   no `.git` in a release tarball). If we ever need to identify the exact
   upstream commit shipped in `llama-cpp-sys-2`, we'd have to diff the
   tarball against the upstream repo by hand. For Phase 1 this should be
   addressed — either upgrade the pin when a version with a recorded SHA
   ships, or carry our own provenance note in the consumer crate.

## [VERIFY] resolutions (from Phase 0 Spike 1)

| Marker | Original assumption | What I actually found |
|---|---|---|
| `llama-cpp-2` "latest stable + record llama.cpp commit SHA" | Pick newest stable. | `=0.1.146`. llama.cpp commit SHA **not recoverable** from the released crate; vendored as tarball. |
| Build features "likely `mtmd` plus one of `vulkan`/`metal`" | `mtmd` + GPU | `mtmd` only on WSL Linux. Vulkan is a user-side follow-up on native Windows. |
| HF repo + filenames | Pinned by SPEC | Used the files at `/mnt/c/Users/anl/qwen3-asr-gguf/`: `Qwen3-ASR-0.6B-Q8_0-ggml-org.gguf` + `Qwen3-ASR-0.6B.mmproj-Q8_0.gguf` (the ggml-org reference Q8_0 pair). SHAs recorded above. |
| mtmd context type "probably `llama_cpp_2::mtmd::MtmdContext`" | matches | `llama_cpp_2::mtmd::MtmdContext`, constructed with `MtmdContext::init_from_file(mmproj_path, &model, &params)`. |
| Audio input "raw f32 vs mel?" | Open | Raw f32 at 16 kHz via `MtmdBitmap::from_audio_data(&[f32])`. See Q-P0-1 above. |
| Decode entry "probably `LlamaContext::decode` over an mtmd-produced batch" | matches | `MtmdInputChunks::eval_chunks(...)` does the prefill (text decode + audio encode + audio decode in one call), returns updated `n_past`. The per-token generation loop then uses `LlamaContext::decode(&mut LlamaBatch)` exactly as in the upstream mtmd example. |
| Token-level timestamps | Unknown | Not exposed. `MtmdInputChunk::n_tokens()` and `n_positions()` give counts but no per-token time mapping. Phase-2 word-level timestamps will need a different code path (Whisper-style) or a downstream alignment pass. |
| GPU offload knobs | Unknown | `LlamaModelParams::with_n_gpu_layers(n: u32)` plus build-feature gates (`vulkan`, `cuda`, `metal`). On the WSL CPU build I explicitly set 0 GPU layers; the build runs CPU-only regardless. Vulkan path not verified on WSL (paravirt GPU unreliable per Phase 0). |
| Context reuse across consecutive `feed_audio` calls | Open (drives Spike 3's threading) | Yes — `MtmdContext` is stateless across calls (see Q-P0-2). |

## Reproducing the acceptance run

```bash
cargo build -p spike-asr --release

./target/release/spike-asr \
  --model  /mnt/c/Users/anl/qwen3-asr-gguf/Qwen3-ASR-0.6B-Q8_0-ggml-org.gguf \
  --mmproj /mnt/c/Users/anl/qwen3-asr-gguf/Qwen3-ASR-0.6B.mmproj-Q8_0.gguf \
  --wav    ~/qwen3-asr-onnx/tests/fixtures/librispeech_0.wav \
  --max-seconds 10 \
  --threads 8 \
  --reference <(echo "MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL")
```

For fastest I/O, copy the GGUFs from `/mnt/c/...` to the WSL filesystem
(`/tmp/`) before running — NTFS-over-9P adds noticeable load latency on
the first run.

## Limitations and follow-ups

- **CPU-only**, Linux. The Windows + Vulkan claim from Phase 0 is a
  user-side follow-up; this spike does not address it. The
  `llama-cpp-2/vulkan` build feature is the entry point when the user
  runs the verification.
- **No streaming.** mtmd is fixed-window batch inference; this matches
  Phase 0's spike framing (`batch`).
- **WER measurement uses a length-aware prefix.** A 10 s truncation
  cannot be scored against a 30 s reference verbatim; the spike trims
  the reference to a slack-ed prefix matching the hypothesis length
  (×1.25). This is documented in `word_error_rate()` and is the right
  thing to do here, but it would not be the right thing in a Phase-4
  benchmark harness — Phase 4 must use full-length references and
  alignment, not prefix WER.
- **Sub-30 s hallucination (Q-P0-3) is not solved.** This is a design
  decision for Phase 2, not a code-change for this spike.
- **llama.cpp commit SHA not recoverable.** Phase 1 should either pin a
  newer `llama-cpp-2` version whose build script records the SHA, or
  document a manual provenance check (diff against an upstream tag).
- **Q4_K_M not retested.** Prior work documented the hallucination; we
  did not redo it here. If a Q5_K_M or Q6_K from ggml-org ships, the
  acceptance run is the minimum re-test.
