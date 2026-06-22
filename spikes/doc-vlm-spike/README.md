# doc-vlm-spike

Throwaway spike that validates the VLM fallback seam in `crates/doc-convert`.

`doc-convert`'s production path uses pure-Rust parsers for digital documents
(txt/md/xlsx/html/eml/pdf/pptx). Scanned PDFs and image-only files fall into a
`vlm_fallback` stub that currently returns `AppError::Unsupported`. This spike
determines whether Gemma-4's vision encoder is good enough to fill that seam.

**This binary is NOT wired into the app** and it does NOT add a Gemma-4-vision
entry to the production model registry (`models.json`); its downloader is
self-contained. Graduation path: if the numbers pass, the image-inference loop
here moves into `fn vlm_fallback` in `crates/doc-convert/src/lib.rs`.

---

## One-command run

Zero manual setup — no model paths, no PDFium path, no fixture, no required
flags:

```sh
cargo run -p spike-doc-vlm --release                  # CPU
cargo run -p spike-doc-vlm --release --features vulkan # GPU (Vulkan)
```

`--features metal | cuda | rocm` select the other llama.cpp backends. On a GPU
build the spike offloads all layers (`n_gpu_layers = 99`) and runs the mtmd
vision encoder on the GPU; on a CPU build it stays on CPU. Override with
`--n-gpu-layers N`.

The first run downloads the artifacts (multi-GB LM); subsequent runs reuse the
cache and skip the download.

### What the run does

1. Self-acquires the Gemma-4 E4B vision LM + mmproj GGUFs and a PDFium prebuilt
   into the cache.
2. Renders two synthetic pages from known strings (exact ground truth).
3. Runs each through Gemma-4 via mtmd IMAGE inference.
4. Scores CER + latency per page and prints a table.
5. Prints `RESULT: PASS (go)` / `RESULT: FAIL (no-go)` and sets the exit code
   (`0` PASS, non-zero FAIL).

---

## Cache location

Artifacts live under the OS cache dir, `<cache>/minutist-spike/`:

| Path | Contents |
|------|----------|
| `<cache>/minutist-spike/vlm/gemma-4-E4B-it-Q4_K_M.gguf` | LM (~5.34 GB) |
| `<cache>/minutist-spike/vlm/mmproj-gemma-4-E4B-it-Q8_0.gguf` | vision mmproj (~560 MB) |
| `<cache>/minutist-spike/pdfium/libpdfium.{so,dll,dylib}` | extracted PDFium |

`dirs::cache_dir()` resolves to `~/.cache` (Linux), `%LOCALAPPDATA%` (Windows),
`~/Library/Caches` (macOS). Delete the directory to force a re-download.

If the production app has already downloaded a Gemma-4 LM
(`<data>/minutist/models/llm/**`), the spike reuses it for the LM and only
fetches the mmproj — the app bundles the text-only LM and never carries the
vision projector.

### Sources

- LM + mmproj: `ggml-org/gemma-4-E4B-it-GGUF` via the HF resolve URL
  (`https://huggingface.co/<repo>/resolve/main/<file>`).
- PDFium: `bblanchon/pdfium-binaries` `releases/latest` (BSD-3); platform
  archive selected at runtime from `target_os` / `target_arch`.

---

## Synthetic fixtures + scoring

No binary fixture is committed and no reference text is hand-authored. Each page
is rendered in-process from a known Rust string via a pure-Rust monospaced
rasteriser (`ab_glyph` + `image`, with `assets/DejaVuSansMono.ttf` embedded via
`include_bytes!`), so the ground truth *is* the source text:

- **clean-text** — a multi-paragraph text page.
- **table** — a simple bordered table; ground truth is the canonical pipe-table
  rendering of the known cells.

These pages test the **mtmd IMAGE plumbing** plus **clean-text and simple-table
transcription accuracy**. They are deliberately easy — they are NOT a substitute
for dense real-world layout. Point `--input` at a hard document for that.

**CER** = character-level Levenshtein normalised by `max(|gt|, |pred|)`
(OmniDocBench-style; over-generation caps CER at 1.0). Both strings are
normalised first (trim + collapse whitespace runs to a single space).

**Gate** — PASS iff every synthetic page has CER < 0.15 AND latency < 30 s/page.

`assets/DejaVuSansMono.ttf` is Bitstream-Vera-derived (permissive; embeddable in
commercial software); its licence is in `assets/DejaVuSansMono.LICENSE`.

---

## Real document mode (no gate)

```sh
cargo run -p spike-doc-vlm --release -- --input scanned.pdf --pages all
cargo run -p spike-doc-vlm --release -- --input page.png
```

PDFs are rasterised via the auto-acquired PDFium (`--dpi`, `--pages` control
the raster). Real input has no ground truth: the spike prints the markdown to
stdout (diagnostics to stderr) and reports latency, but does NOT apply or affect
the PASS/FAIL gate.

---

## mtmd IMAGE inference

Mirrors the audio mtmd path in `crates/asr-runtime`, fed a page image instead of
audio samples:

```text
MtmdBitmap::from_buffer(&mtmd_ctx, png_bytes)        // image analogue of from_audio_data
  -> mtmd_ctx.tokenize(MtmdInputText{..}, &[&bitmap]) // text + media marker
  -> chunks.eval_chunks(&mtmd_ctx, &ctx, 0, 0, n_batch, true) // prefill (image encode + decode)
  -> greedy decode (LlamaSampler::greedy) until is_eog_token  // markdown out
```

`MtmdContext::init_from_file` loads the mmproj; the spike asserts
`support_vision()` and **bails** with a clear message if it is false ("passed
mmproj does not advertise vision support — did you point at the audio
projector?").

---

## Graceful failure

- Offline / download failure / truncated file / wrong URL: a clear
  `spike-doc-vlm FAILED: …` message and a non-zero exit, never a bare panic.
  Partial downloads are written to `.part` and only renamed on success.
- Wrong projector (audio mmproj): hard bail on the `support_vision()` check.
- No GPU on a GPU build: llama.cpp falls back to CPU at runtime.
- Cached artifacts (full-size LM/mmproj, extracted PDFium) skip re-download.

---

## Caveat — Gemma-4 PLE forward graph (llama.cpp #22243)

The bundled llama.cpp build (vendored by `llama-cpp-2 =0.1.146`) has a known
issue with the Gemma-4 PLE (Pre-computed Local Embeddings) forward graph,
observed with text inference; it may also affect the vision graph. **Record on
the first Vulkan run** whether vision inference completes or crashes — the
`eval_chunks` error path names this issue. If it crashes, a Gemma-3 multimodal
GGUF (`ggml-org/gemma-3-4b-it-GGUF`, also ships an mmproj, no PLE) is the
no-PLE control.

---

## Decision record (fill in after the first GPU run)

```
LM:           gemma-4-E4B-it-Q4_K_M.gguf
mmproj:       mmproj-gemma-4-E4B-it-Q8_0.gguf
llama.cpp:    (vendored by llama-cpp-2 =0.1.146)
CER:          clean-text=____  table=____
Avg latency:  ____ s/page  (Vulkan, n_gpu_layers=99)
PLE bug hit:  yes / no   (llama.cpp #22243)
Verdict:      PASS / FAIL
```

If the gate fails, re-evaluate with a doc-specialist model (PaddleOCR-VL-1.6 /
GLM-OCR / Qwen3-VL) before rebuilding this spike against that model.
