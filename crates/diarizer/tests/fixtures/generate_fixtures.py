#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = []
# ///
"""Deterministically (re)generate the diarizer accuracy fixtures.

Produces two committed 16 kHz mono s16 WAVs by concatenating two committed
single-speaker real-speech clips — two GENUINELY DISTINCT LibriSpeech readers,
so the gated accuracy test in `tests/accuracy.rs` measures real speaker
discrimination (not a self-similar pitch trick):

  two_speakers_synth.wav      Speaker A then a 0.4 s silence gap then Speaker B
                              (two different LibriSpeech readers). Self-authored
                              ground truth: we built the boundary, so we know it
                              exactly.
  single_speaker_control.wav  Speaker A then the same 0.4 s gap then Speaker A
                              again (one reader) — exactly one distinct speaker.

Inputs (committed alongside this script):

  speaker_a.wav   LibriSpeech reader 1089 (male),   first 5.000 s.
  speaker_b.wav   LibriSpeech reader 1221 (female), first 5.000 s.

Both source clips are the first 80 000 samples (5.000 s) of the canonical
sherpa-onnx English `test_wavs/{0,1}.wav` (LibriSpeech test-clean excerpts,
CC-BY-4.0). See `README.md` for the source URLs + original SHA-256s. The trim
to a common length keeps the two speakers balanced in the accuracy measure.

Pure-Python, no external tools / no network: re-running reproduces the
committed bytes exactly.

Usage (from the workspace root):
    uv run crates/diarizer/tests/fixtures/generate_fixtures.py
"""

import array
import hashlib
import os
import wave

HERE = os.path.dirname(os.path.abspath(__file__))

SAMPLE_RATE = 16_000
GAP_S = 0.4


def read_wav_i16(path):
    w = wave.open(path, "rb")
    assert w.getnchannels() == 1, "expect mono"
    assert w.getframerate() == SAMPLE_RATE, "expect 16 kHz"
    assert w.getsampwidth() == 2, "expect s16"
    raw = w.readframes(w.getnframes())
    w.close()
    a = array.array("h")
    a.frombytes(raw)
    return list(a)


def write_wav_i16(path, samples):
    w = wave.open(path, "wb")
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(SAMPLE_RATE)
    clamped = [max(-32768, min(32767, int(s))) for s in samples]
    w.writeframes(array.array("h", clamped).tobytes())
    w.close()


def main():
    speaker_a = read_wav_i16(os.path.join(HERE, "speaker_a.wav"))
    speaker_b = read_wav_i16(os.path.join(HERE, "speaker_b.wav"))
    gap = [0] * int(GAP_S * SAMPLE_RATE)

    two = list(speaker_a) + gap + list(speaker_b)
    single = list(speaker_a) + gap + list(speaker_a)

    a_end_ms = len(speaker_a) * 1000 // SAMPLE_RATE
    gap_end_ms = a_end_ms + int(GAP_S * 1000)
    print(f"A_END_MS={a_end_ms} GAP_END_MS={gap_end_ms} "
          f"two_total_ms={len(two) * 1000 // SAMPLE_RATE} "
          f"single_total_ms={len(single) * 1000 // SAMPLE_RATE}")

    for name, samples in (
        ("two_speakers_synth.wav", two),
        ("single_speaker_control.wav", single),
    ):
        path = os.path.join(HERE, name)
        write_wav_i16(path, samples)
        sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
        print(f"{name}: {len(samples)} samples, sha256={sha}")


if __name__ == "__main__":
    main()
