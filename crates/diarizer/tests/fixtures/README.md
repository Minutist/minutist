# Diarizer accuracy fixtures (synthetic, committed)

These fixtures back the env-var-gated accuracy test in `tests/accuracy.rs`.
They are **synthetic but real-speech** (per `architecture/cross-cutting.md`
"Automated-testing policy": synthetic speech-path audio must be real speech,
not tones). All three are 16 kHz mono s16 WAV.

Regenerate deterministically (pure Python, no external tools) with:

    uv run crates/diarizer/tests/fixtures/generate_fixtures.py

The generator derives everything from the single committed real-speech clip
`tests/fixtures/librispeech_0.wav` (workspace root, 5.855 s).

## `two_speakers_synth.wav`

Two distinct speakers concatenated with a 0.4 s silence gap:

- **Speaker A** = the original LibriSpeech clip, `[0 ms, 5855 ms)`.
- 0.4 s silence gap, `[5855 ms, 6255 ms)`.
- **Speaker B** = a pitch-shifted copy of A (resampled by 0.8 then stretched
  back to the original length; pitch + formants moved far enough that the
  speaker-embedding model clusters it as a distinct speaker), `[6255 ms,
  12110 ms)`.

`sha256 = c3ac81250a62714abd6d8cc360f36c18f737b7653cb5885727ca18ef7daec6ed`

Ground truth (the label list the test scores against, one entry per ASR-style
segment): the first half of the audio is speaker A, the second half is speaker
B. `tests/accuracy.rs` builds a `Vec<Segment>` tiling the timeline in ~1 s
windows, labels each window by which half its midpoint falls in (skipping the
gap), runs `assign_speakers`, and asserts **≥ 80 % permutation-invariant
segment accuracy** (the two distinct labels may come back as A/B or B/A).

## `single_speaker_control.wav`

Speaker A repeated (same speaker) with the same 0.4 s gap, `[0 ms, 12110 ms)`.

`sha256 = 9db57fee2393ce69ed831c02d87f80b62c6fbe5a158dddee4163a5393a9ea6f3`

The control asserts the diarizer reports **exactly one** distinct label over a
single-speaker recording (conservative clustering must not over-split one
speaker).

## Why synthetic rather than the Spike-4 fixtures?

The Spike-4 two-speaker WAVs (`1/2/3-two-speakers-en.wav`) are referenced by
URL + SHA only and not committed (see `spikes/diarize/fixtures/README.md`), so
they cannot run in the default checkout. These fixtures are committed and
reproducible, keeping the gated accuracy test self-contained.
