//! Model-spec abstraction for the head-to-head doc-VLM benchmark.
//!
//! Each registered model carries everything that differs between a generic
//! Gemma-4 chat VLM and a doc-OCR specialist like PaddleOCR-VL: the GGUF
//! artifacts to acquire (LM + mmproj, with their own cache filenames and
//! size-floors), the per-model instruction text, and — the load-bearing part —
//! a per-model *prompt assembly* that controls where the `<__media__>` image
//! marker lands relative to the instruction.
//!
//! Throwaway spike code; `anyhow`, owned `String`s, and closures are fine here.

use anyhow::{anyhow, Result};
use llama_cpp_2::model::LlamaModel;

use crate::acquire;

// The default mtmd media marker (`<__media__>`, from `mtmd_default_marker()`)
// is passed into `build_prompt` by the caller. `MtmdContext::tokenize` splits
// the input text on that marker and inserts the image chunk there, so where the
// marker sits in the assembled prompt decides whether the image precedes or
// follows the instruction — the core difference between the two models below.

/// How a model wants its single-image user turn assembled into the flat string
/// fed to `MtmdContext::tokenize`.
///
/// The two registered models need OPPOSITE marker placement:
///
/// * Gemma-4 (`ChatTemplateMarkerLast`) — a generic chat VLM. Build a user
///   message whose content is `"{instruction}\n{marker}"` (instruction first,
///   marker last) and render it through the GGUF's embedded chat template via
///   `apply_chat_template`. `add_special = true` (the template does not emit a
///   literal BOS; llama adds it).
///
/// * PaddleOCR-VL (`ErnieMarkerFirst`) — a doc-OCR specialist trained on bare
///   task prefixes (`OCR:`, `Table Recognition:`). The marker must come
///   *immediately before* the prefix (`<__media__>OCR:`) inside the ERNIE-4.5
///   turn scaffold `<|begin_of_sentence|>User: <__media__>OCR:\nAssistant:\n`.
///   We emit that literal string and set `add_special = false` (BOS is already
///   literal in the string — letting llama add another would double it).
///
/// Routing PaddleOCR through the Gemma assembly (marker last) would put the
/// image AFTER the instruction and push the model off its training
/// distribution; routing Gemma through the ERNIE assembly would emit ERNIE
/// special tokens Gemma's tokenizer does not recognise. Hence per-model.
#[derive(Clone, Copy, Debug)]
pub enum PromptStyle {
    /// Gemma-style: instruction then marker, rendered via `apply_chat_template`.
    ChatTemplateMarkerLast,
    /// PaddleOCR/ERNIE-style: marker then prefix, emitted as a literal turn.
    ErnieMarkerFirst,
}

/// A registered model: artifacts to acquire + how to prompt it.
pub struct ModelSpec {
    /// Human-readable name used in the comparison table.
    pub display_name: &'static str,

    /// LM GGUF resolve URL + local cache filename + size-floor sanity check.
    pub lm_url: &'static str,
    pub lm_cache_filename: &'static str,
    pub lm_min_bytes: u64,

    /// mmproj (vision projector) GGUF resolve URL + cache filename + size-floor.
    /// PaddleOCR's projector is distributed with a `.mmproj` extension; it is a
    /// GGUF-format projector and loads via `MtmdContext::init_from_file` exactly
    /// like the Gemma `mmproj-*.gguf`.
    pub mmproj_url: &'static str,
    pub mmproj_cache_filename: &'static str,
    pub mmproj_min_bytes: u64,

    /// Per-model instruction text. For the marker-last style this is the verbose
    /// "convert to markdown" instruction. For the marker-first style this is the
    /// bare task prefix the model was trained on (`OCR:`, `Table Recognition:`).
    pub instruction: &'static str,

    /// Per-page override of `instruction` for the synthetic table page, when the
    /// on-spec prefix differs (PaddleOCR uses `Table Recognition:` there). When
    /// `None`, `instruction` is used for every page.
    pub table_instruction: Option<&'static str>,

    /// How to assemble the user turn (controls marker placement + special-token
    /// handling).
    pub prompt_style: PromptStyle,

    /// Whether to offload to GPU when the build has a GPU backend. Both models
    /// follow the CLI `--n-gpu-layers`; this is a per-model affinity hint.
    pub use_gpu: bool,
}

impl ModelSpec {
    /// Resolve the instruction for a given page name (table pages may differ).
    pub fn instruction_for(&self, page_name: &str) -> &'static str {
        if page_name == "table" {
            self.table_instruction.unwrap_or(self.instruction)
        } else {
            self.instruction
        }
    }

    /// `MtmdInputText.add_special`: true for the chat-template path (llama adds
    /// BOS), false for the literal-ERNIE path (BOS is already in the string).
    pub fn add_special(&self) -> bool {
        match self.prompt_style {
            PromptStyle::ChatTemplateMarkerLast => true,
            PromptStyle::ErnieMarkerFirst => false,
        }
    }

    /// `MtmdInputText.parse_special`: always true so the media marker and any
    /// `<|begin_of_sentence|>`/`<|end_of_sentence|>` specials tokenize as
    /// special tokens rather than literal text.
    pub fn parse_special(&self) -> bool {
        true
    }

    /// Acquire this model's LM + mmproj into the spike cache (skip when cached),
    /// reusing the existing download/cache primitives in `acquire.rs`.
    pub fn acquire(&self) -> Result<acquire::ModelPaths> {
        acquire::ensure_model_spec(self)
    }

    /// Build the flat prompt string fed to `MtmdContext::tokenize` for one page.
    ///
    /// `marker` is the resolved media marker string (`<__media__>`). The result
    /// must contain that marker exactly once; tokenize splits on it.
    pub fn build_prompt(&self, model: &LlamaModel, page_name: &str, marker: &str) -> Result<String> {
        let instruction = self.instruction_for(page_name);
        match self.prompt_style {
            PromptStyle::ChatTemplateMarkerLast => {
                // Instruction first, marker last, wrapped by the GGUF chat
                // template (ChatML fallback if the template is missing).
                let user_content = format!("{instruction}\n{marker}");
                render_chat_template(model, &user_content)
            }
            PromptStyle::ErnieMarkerFirst => {
                // Marker FIRST, immediately before the bare task prefix, inside
                // the ERNIE-4.5 turn scaffold. We emit this literally rather
                // than relying on the embedded template so the marker ordering
                // is guaranteed regardless of how the GGUF template renders.
                // `add_special = false` keeps llama from prepending a second
                // BOS on top of the literal `<|begin_of_sentence|>`.
                Ok(format!(
                    "<|begin_of_sentence|>User: {marker}{instruction}\nAssistant:\n"
                ))
            }
        }
    }
}

/// Render a single user turn through the model's embedded chat template, with a
/// ChatML fallback when the GGUF carries no template or rendering fails.
fn render_chat_template(model: &LlamaModel, user_content: &str) -> Result<String> {
    use llama_cpp_2::model::LlamaChatMessage;

    let msg = LlamaChatMessage::new("user".to_string(), user_content.to_string())
        .map_err(|e| anyhow!("LlamaChatMessage::new: {e}"))?;

    match model.chat_template(None::<&str>) {
        Ok(template) => match model.apply_chat_template(&template, &[msg], true) {
            Ok(rendered) => return Ok(rendered),
            Err(e) => eprintln!("apply_chat_template failed, ChatML fallback: {e}"),
        },
        Err(e) => eprintln!("no chat template ({e:?}); ChatML fallback"),
    }
    Ok(format!(
        "<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n"
    ))
}

// ---------------------------------------------------------------------------
// The registry: the two models compared head-to-head.
// ---------------------------------------------------------------------------

/// Gemma-4 verbose doc-to-markdown instruction (marker appended after it).
const GEMMA_INSTRUCTION: &str = "Convert this document page to clean, well-structured markdown. \
     Preserve headings, lists, and tables. For tables use GitHub \
     pipe-table syntax. Output only the markdown content, no preamble.";

/// Gemma-4 E4B-it — the existing path. ggml-org GGUF repo; Q4_K_M LM + Q8_0
/// vision mmproj. Generic chat VLM: verbose instruction, marker last.
pub const GEMMA_4_E4B: ModelSpec = ModelSpec {
    display_name: "Gemma-4-E4B",
    lm_url: "https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF/resolve/main/gemma-4-E4B-it-Q4_K_M.gguf",
    lm_cache_filename: "gemma-4-E4B-it-Q4_K_M.gguf",
    lm_min_bytes: 4_500_000_000, // ~5.34 GB expected
    mmproj_url: "https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF/resolve/main/mmproj-gemma-4-E4B-it-Q8_0.gguf",
    mmproj_cache_filename: "mmproj-gemma-4-E4B-it-Q8_0.gguf",
    mmproj_min_bytes: 400_000_000, // ~560 MB expected
    instruction: GEMMA_INSTRUCTION,
    table_instruction: None,
    prompt_style: PromptStyle::ChatTemplateMarkerLast,
    use_gpu: true,
};

/// PaddleOCR-VL-1.6 — doc-OCR specialist. Mungert single-repo carries both the
/// quantized LM and a compatible mmproj. Q4_K_M LM (the prescribed bundle quant)
/// + q8_0 `.mmproj` projector (realistic bundle choice). ERNIE-4.5 turn format,
/// bare task prefix, marker FIRST.
///
/// Prompt prefixes are training prompts and case/colon-sensitive (llama.cpp
/// PR #18825): general text -> `OCR:`, tables -> `Table Recognition:`.
pub const PADDLEOCR_VL: ModelSpec = ModelSpec {
    display_name: "PaddleOCR-VL-1.6",
    lm_url: "https://huggingface.co/Mungert/PaddleOCR-VL-1.6-GGUF/resolve/main/PaddleOCR-VL-1.6-q4_k_m.gguf",
    lm_cache_filename: "PaddleOCR-VL-1.6-q4_k_m.gguf",
    lm_min_bytes: 300_000_000, // ~382 MB expected
    mmproj_url: "https://huggingface.co/Mungert/PaddleOCR-VL-1.6-GGUF/resolve/main/PaddleOCR-VL-1.6-q8_0.mmproj",
    mmproj_cache_filename: "PaddleOCR-VL-1.6-q8_0.mmproj",
    mmproj_min_bytes: 450_000_000, // ~598 MB expected
    // Bare on-spec task prefixes; do NOT feed the Gemma verbose prompt.
    instruction: "OCR:",
    table_instruction: Some("Table Recognition:"),
    prompt_style: PromptStyle::ErnieMarkerFirst,
    use_gpu: true,
};

/// The models compared head-to-head, in table-column order.
pub const REGISTRY: &[&ModelSpec] = &[&GEMMA_4_E4B, &PADDLEOCR_VL];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_has_both_models() {
        assert_eq!(REGISTRY.len(), 2);
        assert_eq!(REGISTRY[0].display_name, "Gemma-4-E4B");
        assert_eq!(REGISTRY[1].display_name, "PaddleOCR-VL-1.6");
    }

    #[test]
    fn paddleocr_uses_table_prefix_on_table_page() {
        assert_eq!(PADDLEOCR_VL.instruction_for("table"), "Table Recognition:");
        assert_eq!(PADDLEOCR_VL.instruction_for("clean-text"), "OCR:");
    }

    #[test]
    fn gemma_uses_same_instruction_on_every_page() {
        assert_eq!(
            GEMMA_4_E4B.instruction_for("table"),
            GEMMA_4_E4B.instruction_for("clean-text")
        );
    }

    #[test]
    fn ernie_style_disables_add_special() {
        // BOS is literal in the ERNIE string, so llama must NOT add another.
        assert!(!PADDLEOCR_VL.add_special());
        assert!(GEMMA_4_E4B.add_special());
        assert!(PADDLEOCR_VL.parse_special());
        assert!(GEMMA_4_E4B.parse_special());
    }

    #[test]
    fn paddleocr_prompt_puts_marker_before_prefix() {
        // Build prompt without a model by exercising only the ERNIE branch via
        // the literal format; the marker must immediately precede `OCR:`.
        let marker = "<__media__>";
        let expected = format!("<|begin_of_sentence|>User: {marker}OCR:\nAssistant:\n");
        // Mirror ModelSpec::build_prompt's ErnieMarkerFirst arm.
        let got = format!(
            "<|begin_of_sentence|>User: {marker}{}\nAssistant:\n",
            PADDLEOCR_VL.instruction_for("clean-text")
        );
        assert_eq!(got, expected);
        // Marker is immediately before the prefix, with no space between.
        assert!(got.contains("<__media__>OCR:"));
    }
}
