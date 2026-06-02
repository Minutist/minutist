# Diarizer accuracy fixtures (committed real speech)

These fixtures back the env-var-gated accuracy test in `tests/accuracy.rs`.
They are **real speech** (per `architecture/cross-cutting.md` "Automated-testing
policy": synthetic speech-path audio must be real speech, not tones), assembled
by concatenating two committed single-speaker clips. All are 16 kHz mono s16
WAV.

Regenerate deterministically (pure Python, no external tools / no network) with:

    uv run crates/diarizer/tests/fixtures/generate_fixtures.py

## Source clips (committed)

Two GENUINELY DISTINCT LibriSpeech readers — so the gated accuracy test
measures real speaker discrimination, not a self-similar trick:

- `speaker_a.wav` — LibriSpeech reader **1089** (male), `sha256 =
  f8e9f5c202a8e1e9c7c4070a44d52d6d6ce0ae8e66693f57ab69612d33a64562`.
- `speaker_b.wav` — LibriSpeech reader **1221** (female), `sha256 =
  a433a4594df89995aa1ce28797ee7a7d5aaac2820bd9eefea7a7ce3a80259e67`.

Both are the first 80 000 samples (5.000 s) of the canonical sherpa-onnx English
`test_wavs/{0,1}.wav` (LibriSpeech test-clean excerpts, CC-BY-4.0):

| clip | source URL | original SHA-256 |
|---|---|---|
| `speaker_a.wav` | `https://huggingface.co/csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-06-26/resolve/main/test_wavs/0.wav` | `6bc58a4efdf20daac252b6b1502632601a71efe0308f6757dc1eda34891a7e4f` |
| `speaker_b.wav` | `https://huggingface.co/csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-06-26/resolve/main/test_wavs/1.wav` | `5143a6ba93c4b274e2c4ac22deb75c2c48936c853f0519add1de828b6c79cc5a` |

(Re-fetch each source, keep the first 80 000 samples, to reproduce the committed
`speaker_{a,b}.wav` bytes.) Trimming both to a common 5.000 s keeps the two
speakers balanced in the accuracy measure.

## `two_speakers_synth.wav`

Two distinct speakers concatenated with a 0.4 s silence gap:

- **Speaker A** (reader 1089) = `[0 ms, 5000 ms)`.
- 0.4 s silence gap, `[5000 ms, 5400 ms)`.
- **Speaker B** (reader 1221) = `[5400 ms, 10400 ms)`.

`sha256 = 141522837d560ca8775c1e0fa008f7422ddeaa673fcb225c1f460dd24a56c74c`

Ground truth (the label list the test scores against, one entry per ASR-style
segment): windows whose midpoint is `< 5000 ms` are speaker A, the rest are
speaker B (windows in the gap are skipped). `tests/accuracy.rs` builds a
`Vec<Segment>` tiling the timeline in ~1 s windows, runs `assign_speakers` under
the **production** `DiarizerConfig::default()` (threshold/auto-count mode — the
config the shipped app uses, since the speaker count is unknown at record time),
and asserts **≥ 80 % permutation-invariant segment accuracy** (the two distinct
labels may come back as A/B or B/A) plus a discovered count of exactly 2.

## `single_speaker_control.wav`

Speaker A repeated (same reader) with the same 0.4 s gap, `[0 ms, 10400 ms)`.

`sha256 = 0c01903dcd1cd4f28c2652ddf7196653c1d2aca10162c9897fa80bc476d2ea9a`

The control asserts the diarizer reports **exactly one** distinct label over a
single-speaker recording under the same production `DiarizerConfig::default()` —
a genuine over-segmentation guard (threshold mode must not split one speaker in
two), not an oracle `num_clusters = Some(1)` that would pass by construction.

## Why concatenated real clips rather than the Spike-4 fixtures?

The Spike-4 two-speaker WAVs (`1/2/3-two-speakers-en.wav`) are referenced by
URL + SHA only and not committed (see `spikes/diarize/fixtures/README.md`), so
they cannot run in the default checkout, and their speaker boundaries would have
to be hand-annotated. Concatenating two committed distinct single-speaker clips
gives committed, reproducible audio with exact self-authored ground truth.
