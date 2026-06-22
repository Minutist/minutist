# doc-vlm-spike

Throwaway spike that benchmarks two doc-to-markdown VLMs head-to-head to decide
which should fill the VLM fallback seam in `crates/doc-convert`.

`doc-convert`'s production path uses pure-Rust parsers for digital documents
(txt/md/xlsx/html/eml/pdf/pptx). Scanned PDFs and image-only files fall into a
`vlm_fallback` stub that currently returns `AppError::Unsupported`. This spike
compares a generic chat VLM (**Gemma-4-E4B**) against a doc-OCR specialist
(**PaddleOCR-VL-1.6**) on the SAME pages and reports which wins.

**This binary is NOT wired into the app** and it does NOT add a VLM entry to the
production model registry (`models.json`); its downloader is self-contained.
Graduation path: if a model passes, its image-inference loop moves into
`fn vlm_fallback` in `crates/doc-convert/src/lib.rs`.

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

The first run downloads the artifacts; subsequent runs reuse the cache and skip
the download.

### What the run does

1. Self-acquires **every registered model's** LM + mmproj GGUFs and a PDFium
   prebuilt into the cache.
2. Renders two synthetic pages from known strings (exact ground truth).
3. Runs **every page through every model** via mtmd IMAGE inference. The two
   LMs are loaded one at a time (not co-resident); each model's resources drop
   before the next loads.
4. Scores CER + latency per (model x page) and prints a side-by-side table.
5. Prints a per-model PASS/FAIL summary and the **winning model** on the
   synthetic pages (lower mean CER, then lower mean latency).
6. Prints `RESULT: PASS` (every model meets the gate) / `RESULT: FAIL` and sets
   the exit code (`0` PASS, non-zero FAIL).

---

## Registered models

| Model | Repo (HF resolve) | LM quant | mmproj | Prompt style |
|-------|-------------------|----------|--------|--------------|
| Gemma-4-E4B | `ggml-org/gemma-4-E4B-it-GGUF` | Q4_K_M (~5.34 GB) | Q8_0 (~560 MB) | verbose markdown instruction, **marker last**, via chat template |
| PaddleOCR-VL-1.6 | `Mungert/PaddleOCR-VL-1.6-GGUF` | Q4_K_M (~382 MB) | q8_0 `.mmproj` (~598 MB) | bare task prefix, **marker first**, ERNIE-4.5 turn |

Both are added via the `ModelSpec` abstraction in `src/models.rs` (display name,
LM/mmproj URL + cache filename + size-floor, per-model instruction, and a
per-model prompt-assembly that controls marker placement). Add a third model by
appending another `ModelSpec` to `REGISTRY`.

### PaddleOCR-VL prompt (load-bearing)

PaddleOCR-VL is a doc-OCR specialist trained on short, case- and
colon-sensitive task prefixes — it does **not** take the Gemma free-text
"convert to markdown" instruction (llama.cpp PR #18825). The spike uses:

- clean-text page -> `OCR:`
- table page -> `Table Recognition:` (its on-spec table mode; emits an
  HTML/markdown table)

The image marker must come **before** the prefix. The rendered ERNIE-4.5 turn
the spike emits is:

```
<|begin_of_sentence|>User: <__media__>OCR:\nAssistant:\n
```

`<__media__>` is the default mtmd marker (`mtmd_default_marker()`), tokenized as
a special token; `MtmdContext::tokenize` splits on it and inserts the image
chunk there, so placing it immediately before the prefix puts the image first.
Because the BOS (`<|begin_of_sentence|>`) is already literal in the string, the
ERNIE path sets `MtmdInputText.add_special = false` to avoid a double BOS, and
`parse_special = true` so the BOS/EOS and marker tokenize as specials. EOS is
`<|end_of_sentence|>`, read from GGUF metadata via `is_eog_token`. Decoding is
greedy (temperature 0), as the PaddleOCR card/PR prescribe.

Gemma, by contrast, keeps the verbose markdown instruction with the marker
appended after it, rendered through its embedded chat template. The two marker
orderings coexist because each `ModelSpec` carries its own prompt assembly.

The comparison table annotates each cell with the prompt prefix used (e.g.
`[OCR]`, `[Table Recognition]`) so the head-to-head is interpretable — note that
`OCR:` emits plain reading-order text while Gemma emits markdown, so the
clean-text CER is the most directly comparable cell.

---

## Cache location

Artifacts live under the OS cache dir, `<cache>/minutist-spike/`, one subdir per
model (keyed on the LM cache-filename stem):

| Path | Contents |
|------|----------|
| `<cache>/minutist-spike/vlm/gemma-4-E4B-it-Q4_K_M/gemma-4-E4B-it-Q4_K_M.gguf` | Gemma LM (~5.34 GB) |
| `<cache>/minutist-spike/vlm/gemma-4-E4B-it-Q4_K_M/mmproj-gemma-4-E4B-it-Q8_0.gguf` | Gemma vision mmproj (~560 MB) |
| `<cache>/minutist-spike/vlm/PaddleOCR-VL-1.6-q4_k_m/PaddleOCR-VL-1.6-q4_k_m.gguf` | PaddleOCR LM (~382 MB) |
| `<cache>/minutist-spike/vlm/PaddleOCR-VL-1.6-q4_k_m/PaddleOCR-VL-1.6-q8_0.mmproj` | PaddleOCR projector (~598 MB) |
| `<cache>/minutist-spike/pdfium/libpdfium.{so,dll,dylib}` | extracted PDFium |

`dirs::cache_dir()` resolves to `~/.cache` (Linux), `%LOCALAPPDATA%` (Windows),
`~/Library/Caches` (macOS). Delete the directory to force a re-download.

If the production app has already downloaded a Gemma-4 LM
(`<data>/minutist/models/llm/**`), the spike reuses it for the Gemma LM and only
fetches the mmproj — the app bundles the text-only LM and never carries the
vision projector. PaddleOCR is always self-acquired.

### Sources

- Gemma LM + mmproj: `ggml-org/gemma-4-E4B-it-GGUF` via the HF resolve URL.
- PaddleOCR LM + `.mmproj`: `Mungert/PaddleOCR-VL-1.6-GGUF` via the HF resolve
  URL (single repo carries both the quantized LM and a compatible projector).
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

**Gate** — for each model, PASS iff every synthetic page has CER < 0.15 AND
latency < 30 s/page. The overall `RESULT` is PASS only when every model passes;
the **winner** is the model with the lowest mean CER over the synthetic pages
(tie-broken by lower mean latency), reported even when neither passes the gate.

`assets/DejaVuSansMono.ttf` is Bitstream-Vera-derived (permissive; embeddable in
commercial software); its licence is in `assets/DejaVuSansMono.LICENSE`.

---

## Real document mode (no gate)

```sh
cargo run -p spike-doc-vlm --release -- --input scanned.pdf --pages all
cargo run -p spike-doc-vlm --release -- --input page.png
```

PDFs are rasterised once via the auto-acquired PDFium (`--dpi`, `--pages`
control the raster) and then run through **every registered model**. Real input
has no ground truth: the spike prints each model's markdown to stdout (under a
`# <model> — <input>` heading), reports latency to stderr, and does NOT apply or
affect the PASS/FAIL gate. PaddleOCR uses its general `OCR:` prefix here (the
page is not the synthetic table page).

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

`MtmdContext::init_from_file` loads each model's mmproj; the spike asserts
`support_vision()` per model and **bails** with a model-aware message if it is
false (either an audio projector was supplied, or — for PaddleOCR-VL — the
vendored llama.cpp predates PR #18825 and `llama-cpp-2` must be bumped).

---

## Graceful failure

- Offline / download failure / truncated file / wrong URL: a clear
  `spike-doc-vlm FAILED: …` message and a non-zero exit, never a bare panic.
  Partial downloads are written to `.part` and only renamed on success. All
  models are acquired up front, so a download failure on the second model fails
  before any inference runs rather than stranding a half-completed benchmark.
- Wrong projector (audio mmproj) / pre-PR llama.cpp: hard bail on the per-model
  `support_vision()` check, naming the offending model.
- No GPU on a GPU build: llama.cpp falls back to CPU at runtime.
- Cached artifacts (full-size LM/mmproj per model, extracted PDFium) skip
  re-download.

---

## Caveats

### Gemma-4 PLE forward graph (llama.cpp #22243)

The bundled llama.cpp build (vendored by `llama-cpp-2 =0.1.146`) has a known
issue with the Gemma-4 PLE (Pre-computed Local Embeddings) forward graph,
observed with text inference; it may also affect the vision graph. **Record on
the first Vulkan run** whether Gemma vision inference completes or crashes — the
`eval_chunks` error path names this issue. If it crashes, a Gemma-3 multimodal
GGUF (`ggml-org/gemma-3-4b-it-GGUF`, also ships an mmproj, no PLE) is the
no-PLE control.

### PaddleOCR-VL requires a post-PR-#18825 llama.cpp

PaddleOCR-VL uses multimodal rope (`mrope`) and the `<__media__>OCR:` template
introduced in llama.cpp PR #18825. The vendored llama.cpp must include it; an
older build either rejects the projector at `init_from_file`, reports
`support_vision() == false`, or produces garbage from missing mrope wiring. If
PaddleOCR fails to load while Gemma works, bump `llama-cpp-2` / its vendored
submodule.

---

## Decision record (fill in after the first GPU run)

```
llama.cpp:    (vendored by llama-cpp-2 =0.1.146)

Gemma-4-E4B (gemma-4-E4B-it-Q4_K_M.gguf + mmproj-gemma-4-E4B-it-Q8_0.gguf):
  CER:        clean-text=____  table=____
  Avg latency: ____ s/page  (Vulkan, n_gpu_layers=99)
  PLE bug hit: yes / no   (llama.cpp #22243)
  Verdict:    PASS / FAIL

PaddleOCR-VL-1.6 (PaddleOCR-VL-1.6-q4_k_m.gguf + PaddleOCR-VL-1.6-q8_0.mmproj):
  CER:        clean-text=____ (OCR:)  table=____ (Table Recognition:)
  Avg latency: ____ s/page  (Vulkan, n_gpu_layers=99)
  Loaded OK:  yes / no   (PR #18825 present in vendored llama.cpp?)
  Verdict:    PASS / FAIL

Winner (lower mean CER, then latency): ____
```

If neither passes, re-evaluate with a different LM quant (PaddleOCR q5_k_m /
q6_k_m) or another doc-specialist (GLM-OCR / Qwen3-VL) before rebuilding this
spike against that model.
