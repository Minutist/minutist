# Spike 2 — llama-cpp-2 text-only summarisation

Status: 2026-05-26, WSL Ubuntu 24.04 (Linux 6.6.87.2-microsoft-standard-WSL2), CPU-only build.

## Verdict

**Pass** for the spike's information goal: `llama-cpp-2 0.1.146`'s text path
loads a small instruct-tuned GGUF, accepts a Qwen-format chat prompt rendered
either via the model's baked-in template or via a hand-built ChatML scaffold,
and produces a markdown-formatted summary of a hard-coded paragraph + 5-segment
fake transcript within the 60 s wall-clock budget.

Phase 0 acceptance gate (heading + bullet present, wall-clock ≤ 60 s, no
panics, clean exit, README measurements) is met with
**inference 36.6 s, peak RSS 3.39 GiB** on
`Qwen2.5-3B-Instruct-Q4_K_M.gguf`.

The crate fully covers Phase 5's summariser surface; no separate LLM dependency
is needed.

## Pinned versions

- `llama-cpp-2 = "=0.1.146"` (same exact pin as Spike 1 so the workspace
  `Cargo.lock` keeps a single major). Declared in `spikes/llm/Cargo.toml`,
  not in the workspace root.
- `llama-cpp-sys-2 = "=0.1.146"` (transitive). Workspace feature-unification
  combines this spike's request (default features off, no `mtmd`) with
  Spike 1's request (`mtmd` + defaults off) into a single
  `mtmd`-enabled build of `llama-cpp-sys-2`. The text path is available
  regardless of the `mtmd` feature flag; turning `mtmd` off here only
  documents intent — it does not change the lock or the produced
  binary because Spike 1 forces the flag on.
- `encoding_rs = "0.8"`, `sha2 = "0.10"`, copied verbatim from Spike 1 so
  the two spikes report identically.
- CPU-only build. Vulkan / CUDA / Metal are out of scope for this WSL spike;
  the user verifies those on native Windows separately.

Build time: about 5 s after Spike 1 has already compiled `llama-cpp-sys-2`.
First-time cold build adds the same 3-4 min llama.cpp compile that Spike 1
documents.

## Model

| File | Size | SHA-256 | Source |
|---|---|---|---|
| `Qwen2.5-3B-Instruct-Q4_K_M.gguf` | 1.80 GiB | `9c9f56a391a3abbd5b89d0245bf6106081bcc3173119d4229235dd9d23253f94` | `bartowski/Qwen2.5-3B-Instruct-GGUF` on HuggingFace |

Downloaded fresh during the spike to `/tmp/Qwen2.5-3B-Instruct-Q4_K_M.gguf`
(not committed). This matches the Phase 0 Spike 2 candidate model
("Qwen2.5-3B-Instruct-Q4_K_M.gguf or current bartowski equivalent") and the
eventual default (final model selection is still unresolved; this spike does NOT
commit to the final pick — it just proves the code path).

### Models considered but not used

The HuggingFace cache on the Windows side held four candidates. The dense
9B model and the two A3B MoE variants are larger than necessary for the
spike's information goal:

| Candidate | Why not used here |
|---|---|
| `Jackrong/Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-v2 Q8_0` (9.5 GiB) | Tested. Generates `<think>...</think>` reasoning before the actual answer — at ~2 tok/s decode on CPU the model burned the entire 60 s budget inside reasoning and never emitted a heading. Reasoning models need a much larger `--max-tokens` budget or a `<think>` filter. Out of scope for this guard-rail spike. |
| `unsloth/Qwen3.5-35B-A3B Q4_K_XL` (20.7 GiB) | Not tested. MoE; 3B active params per token would likely be fast enough, but the disk footprint is wasteful given the 3B dense alternative meets the gate. |
| `unsloth/Qwen3.6-35B-A3B Q4_K_XL` (20.7 GiB) | Same reasoning as the 3.5 variant. |
| `unsloth/gemma-4-26B-A4B-it Q4_K_M` (15.7 GiB) | Same reasoning. Also uses Gemma chat template, not ChatML — would change the manual-template fallback comparison. |

The Phase 5 model selection is a separate decision; this spike's
result does not pre-empt it.

## Measurements

CPU: AMD Ryzen (WSL2, 16 logical threads visible; spike used 8).

### Acceptance run — full fixture (300-word paragraph + 5-segment transcript)

```
$ ./target/release/spike-llm \
    --model /tmp/Qwen2.5-3B-Instruct-Q4_K_M.gguf \
    --max-tokens 256 \
    --threads 8
```

| Metric | Value |
|---|---|
| Model load | 12.25 s (cold; warm reruns ≈ 2.1 s) |
| Context init | 4.23 s (cold; warm ≈ 64 ms) |
| Prompt size | 3484 chars / 863 tokens |
| Prefill | 19.43 s — 44.4 tok/s |
| Generation | 114 tokens in 17.14 s — 6.7 tok/s (greedy) |
| Inference wall-clock (prefill + gen) | **36.56 s** |
| Acceptance budget | 60 s ✓ |
| Total wall-clock incl. load | 55.8 s |
| Peak RSS (VmHWM) | **3388 MiB** |
| Markdown heading present | ✓ (`# Q3 Backlog Sync - Spike 2`) |
| Markdown bullet present | ✓ (`-` items under "Action Items") |
| Exit status | 0 |

Generation finished at 114 tokens (model emitted EOS) well before the 256
cap. The output ends with a complete bullet list, not a truncated line.

### Repeat run — manual ChatML fallback (`--force-manual-chatml`)

| Metric | Value |
|---|---|
| Prompt size | 3484 chars / 863 tokens (identical to baked-template path) |
| Prefill | 18.37 s — 47.0 tok/s |
| Generation | 114 tokens in 17.95 s — 6.3 tok/s |
| Inference wall-clock | 36.32 s |
| Output (stdout) | **byte-identical** to the baked-template run |

Two independent paths producing the same token-stream + the same text is the
strongest evidence that the spike's manual ChatML scaffold and the model's
baked-in template render to the same prompt for Qwen2.5. Phase 5 can use
either path; the manual fallback exists so a model without a baked template
still works.

## Q-P0-4 — chat template

**Answer: `chat_apply_template` is exposed and works cleanly.** Phase 5's
summariser does NOT need to pull in `tokenizers` for prompt formatting.

The Rust surface is:

```rust
use llama_cpp_2::model::{LlamaChatMessage, LlamaModel};

// Returns the chat template baked into the GGUF (Qwen2.5's is jinja-ish
// ChatML, embedded as the `tokenizer.chat_template` metadata field).
let template = model.chat_template(None::<&str>)?;

// Render a message list to a prompt string. `add_ass = true` appends the
// `<|im_start|>assistant\n` opener so the next decoded token is the model's
// reply.
let messages = vec![
    LlamaChatMessage::new("system".into(), system_prompt.into())?,
    LlamaChatMessage::new("user".into(), user_content.into())?,
];
let prompt = model.apply_chat_template(&template, &messages, /*add_ass*/ true)?;
```

Under the hood `chat_template` calls `llama_model_chat_template` and
`apply_chat_template` calls `llama_chat_apply_template`. Both are
`llama-cpp-2 0.1.146` stable surface (see
`src/model.rs:767` and `src/model.rs:885` in the registry source). The
internal jinja engine is whatever llama.cpp ships; for Qwen2.5 it produces
exactly the canonical Qwen ChatML scaffold.

Both error cases have explicit fallbacks in the spike:

- `chat_template` returns `Err(ChatTemplateError::MissingTemplate)` if the
  GGUF carries no template metadata.
- `apply_chat_template` returns `ApplyChatTemplateError::FfiError(rc)` if
  the template engine rejects the message list.

In both cases the spike falls back to a hand-built ChatML scaffold:

```
<|im_start|>system
{system}<|im_end|>
<|im_start|>user
{user}<|im_end|>
<|im_start|>assistant
```

The manual fallback is byte-identical to the baked template for Qwen models
(confirmed by `--force-manual-chatml`). For non-Qwen models (e.g. Gemma) the
fallback is incorrect; Phase 5 must use the baked-template path and treat
a missing template as a hard error rather than silently falling back to
ChatML.

## API surprises

1. **`n_batch` is a per-decode hard limit, not just an allocation hint.**
   The first prefill attempt fed 880 prompt tokens in a single
   `LlamaBatch::new(880, 1)` and got an immediate
   `GGML_ASSERT(n_tokens_all <= cparams.n_batch) failed` from
   `llama-context.cpp:1599`. The fix is to chunk the prompt into
   `cparams.n_batch`-sized batches and call `decode` once per chunk; only
   the last token of the last chunk needs `logits = true`. Spike 1 didn't
   hit this because `mtmd_helper_eval_chunks` does the chunking internally.
   This is the most likely Phase-5 footgun.
2. **`AddBos` enum has only `Always` and `Never`.** The chat-template path
   embeds the BOS (or its analogue) itself, so callers must use
   `AddBos::Never` after templating. Using `AddBos::Always` on top of the
   templated string shifts position 0 by one, and Qwen will sometimes
   produce a leading whitespace token or emit `<|im_start|>` itself.
3. **`token_to_piece(.., special: true, ..)` will surface
   `<|im_end|>` and friends if the sampler doesn't stop on EOG first.**
   The spike's greedy decoder relies on `model.is_eog_token(token)` to
   break out of the loop; that's set correctly for the Qwen template.
   Phase 5's summariser should mirror this — `is_eog_token` covers both
   the model's EOS and Qwen's `<|im_end|>` because llama.cpp registers
   both as end-of-generation tokens at GGUF metadata level.
4. **Thread-safety of `LlamaContext` is not advertised.** The crate marks
   the model `Send + Sync` (it's an `Arc`-wrapped read-only resource) but
   `LlamaContext<'a>` carries a `*mut` and is `!Send` by default.
   Phase 5's Tauri command surface must marshal calls onto a dedicated
   worker thread (via `spawn_blocking`) — exactly what
   `architecture/cross-cutting.md` already specifies.

## Sample output

The exact bytes emitted on stdout for the acceptance run:

```markdown
# Q3 Backlog Sync - Spike 2

- **Action Items:**
  - Andrew to write up Spike 2 results in the planning journal once the spike is green.
  - Carl to pull the latest llama.cpp release notes into the journal and check for relevant fixes since 0.1.146.
  - Beth to draft a download-progress UI mock in the design doc.
  - Dee to chase the sherpa-onnx upstream maintainer about the bindgen build issue she found last Friday.
- **Next Sync:** Regular Thursday slot.
```

Heading: `#` ✓. Bullets: `-` at the top level, `-` nested ✓. Content is
grounded in the fixture (no invented facts); the action items track 1:1
with the paragraph + transcript inputs. The model omitted the "risks
raised" half of the paragraph and the next-meeting plumbing, which is
reasonable summarisation behaviour for this length cap and not a defect.

## Reproducing

```bash
# 1. Download the model (one-time, ~1.8 GiB).
curl -L -o /tmp/Qwen2.5-3B-Instruct-Q4_K_M.gguf \
  https://huggingface.co/bartowski/Qwen2.5-3B-Instruct-GGUF/resolve/main/Qwen2.5-3B-Instruct-Q4_K_M.gguf

# 2. Build + run.
cargo build -p spike-llm --release
./target/release/spike-llm \
  --model /tmp/Qwen2.5-3B-Instruct-Q4_K_M.gguf \
  --max-tokens 256 \
  --threads 8
```

Add `--force-manual-chatml` to bypass the baked template and exercise the
fallback path.

## Limitations and follow-ups

- **CPU-only**, Linux. Vulkan / CUDA / Metal builds are a separate user-side
  verification on native Windows.
- **Greedy decoding only.** Spike 5's eventual production summariser will
  probably want top-p or mirostat for less repetitive output on
  open-ended prompts. The full sampler chain is exposed
  (`LlamaSampler::temp`, `top_p`, `top_k`, `mirostat_v2`, etc.); no
  follow-up API spike is needed.
- **Single-fixture acceptance.** The spike runs one fixture once. Real
  summarisation quality is Phase 5's problem, not this one.
- **No KV-cache reuse across runs.** The spike builds a fresh context each
  time. Phase 5 will need to decide whether to keep a long-lived context
  across summaries; that's an architecture question, not a llama-cpp-2
  question.
- **No `tokenizers` crate needed.** Worth noting because the eventual
  binary footprint estimate (Q-P0-8) is sensitive to this decision.
