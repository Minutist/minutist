# Spike 4 diarization fixtures

## Audio fixtures (referenced by URL + SHA-256; not committed)

All published by `k2-fsa/sherpa-onnx` in the
`speaker-segmentation-models` release tag.

| File | Duration | SHA-256 | URL |
|---|---|---|---|
| `1-two-speakers-en.wav` | 16.00 s | `f1c877dc01595e28be7147bf2fe38e5268147a868bf3fdb5c37b97f5940e21f3` | https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/1-two-speakers-en.wav |
| `2-two-speakers-en.wav` | 34.00 s | `ee9c33d34e8f0fda4b78277f609944a1565aa16e6e2146f4cb8f0efb0d70030b` | https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/2-two-speakers-en.wav |
| `3-two-speakers-en.wav` | 54.80 s | `dd3cf2344f8410ee9a2e271e96d1c3f9b530f113ae5b47defecbc0d741a468e9` | https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/3-two-speakers-en.wav |

Format: 16 kHz mono PCM s16le, no metadata.

The primary fixture for the spike's acceptance run is
`2-two-speakers-en.wav`. The others are kept for broader sampling.

Single-speaker control: `/mnt/c/Users/anl/transcribe-rs-test/fixtures/librispeech_30s.wav`
(LibriSpeech excerpt, 29.60 s; shared with Spike 1).

## Reference RTTM

`2-two-speakers-en.ref.rttm` is committed. It is a **binding-fidelity
reference**, not a human-annotated ground truth. The full framing of
what that means lives in `spikes/diarize/README.md` § "DER framing".
The short version: a DER of 0 % against this reference means the Rust
binding (`sherpa-rs 0.6.8`) exposes the same diarization algorithm as
the upstream C tool. It does not mean the algorithm is 0 %
human-accurate on this clip.

To reproduce the reference:

```sh
SHERPA=~/.cache/sherpa-rs/<your-target>/<sha-dir>/sherpa-onnx-v1.12.9-linux-x64-shared
LD_LIBRARY_PATH=$SHERPA/lib $SHERPA/bin/sherpa-onnx-offline-speaker-diarization \
  --clustering.num-clusters=2 \
  --segmentation.pyannote-model=~/.cache/sherpa-onnx/diarization/sherpa-onnx-pyannote-segmentation-3-0/model.onnx \
  --embedding.model=~/.cache/sherpa-onnx/diarization/nemo_en_titanet_small.onnx \
  ~/.cache/sherpa-onnx/fixtures/2-two-speakers-en.wav
```

(The exact `LD_LIBRARY_PATH` directory is the `download-binaries` cache
populated by sherpa-rs-sys's build script; the SHA dir name depends on
the host target.)
