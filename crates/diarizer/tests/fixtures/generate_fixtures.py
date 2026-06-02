#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = []
# ///
"""Deterministically (re)generate the diarizer accuracy fixtures.

Produces two committed 16 kHz mono s16 WAVs from the single committed
real-speech clip `tests/fixtures/librispeech_0.wav` (workspace root):

  two_speakers_synth.wav      Speaker A (the original LibriSpeech clip) then a
                              0.4 s silence gap then Speaker B (a pitch-shifted
                              copy of A — pitch + formants moved enough that the
                              speaker-embedding model clusters it as a distinct
                              speaker, while staying real speech, not a tone).
  single_speaker_control.wav  Speaker A repeated (same speaker), with the same
                              0.4 s gap — exactly one distinct speaker.

The pitch shift resamples by a factor and stretches back to the original
length, so duration is unchanged. Re-running this script reproduces the
committed bytes exactly (pure-Python, no external tools). See `README.md` for
the ground-truth segment boundaries the gated accuracy test asserts against.

Usage (from the workspace root):
    uv run crates/diarizer/tests/fixtures/generate_fixtures.py
"""

import array
import hashlib
import math
import os
import wave

HERE = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
SRC = os.path.join(WORKSPACE_ROOT, "tests", "fixtures", "librispeech_0.wav")

SAMPLE_RATE = 16_000
GAP_S = 0.4
PITCH_FACTOR = 0.8  # < 1.0 raises pitch when re-stretched to original length.


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


def pitch_shift(samples, factor):
    n = len(samples)
    inter_len = int(round(n / factor))
    inter = []
    for i in range(inter_len):
        src = i * factor
        i0 = int(math.floor(src))
        i1 = min(i0 + 1, n - 1)
        frac = src - i0
        inter.append(samples[i0] * (1 - frac) + samples[i1] * frac)
    out = []
    m = len(inter)
    for i in range(n):
        src = i * (m - 1) / (n - 1) if n > 1 else 0
        i0 = int(math.floor(src))
        i1 = min(i0 + 1, m - 1)
        frac = src - i0
        out.append(inter[i0] * (1 - frac) + inter[i1] * frac)
    return out


def main():
    speaker_a = read_wav_i16(SRC)
    gap = [0] * int(GAP_S * SAMPLE_RATE)
    speaker_b = pitch_shift(speaker_a, PITCH_FACTOR)

    two = list(speaker_a) + gap + list(speaker_b)
    single = list(speaker_a) + gap + list(speaker_a)

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
