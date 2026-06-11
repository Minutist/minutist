//! Gated real-model integration test for [`LlamaTurnBackend`] (SP1).
//!
//! Skips (no-op) unless `MINUTIST_LLM_MODEL_PATH` points at a bundled GGUF,
//! mirroring the `summariser` / `asr-runtime` gated tests. Run with:
//!
//! ```text
//! MINUTIST_LLM_MODEL_PATH=/path/to/model.gguf \
//!   cargo test -p chat-agent --test real_model -- --include-ignored
//! ```
//!
//! It loads the GGUF via the `summariser` substrate, lends `&LlamaModel` to a
//! `LlamaTurnBackend`, and runs ONE turn through the `TurnEngine`:
//! - a tool-less turn (no descriptors) must produce a non-empty final answer;
//! - a turn offering one tool over a question that should trigger it exercises
//!   the oaicompat tool-call path (asserted loosely — model-dependent).

use std::path::PathBuf;

use chat_agent::{CancelFlag, ChatEngine, ChatMessage, LlamaTurnBackend, LlamaTurnConfig, SamplerConfig, TurnEngine, TurnOutcome};
use summariser::{LlamaSummariser, SummariserConfig};

use agent_tools::ToolDescriptor;
use serde_json::json;

fn open_substrate() -> Option<LlamaSummariser> {
    let path = match std::env::var("MINUTIST_LLM_MODEL_PATH") {
        Ok(p) if !p.is_empty() => p,
        _ => return None,
    };
    Some(
        LlamaSummariser::open(PathBuf::from(&path), SummariserConfig::default())
            .expect("model load must succeed with a valid MINUTIST_LLM_MODEL_PATH"),
    )
}

#[test]
#[ignore = "requires MINUTIST_LLM_MODEL_PATH"]
fn real_turn_produces_a_final_answer() {
    let Some(substrate) = open_substrate() else {
        return; // no-op skip
    };
    let backend = LlamaTurnBackend::new(substrate.model(), LlamaTurnConfig::default());
    let engine = TurnEngine::new(backend);

    let history = vec![
        ChatMessage::system(
            "You are a concise meeting-notes assistant. Answer in one short sentence.",
        ),
        ChatMessage::user("Say hello and confirm you are ready."),
    ];

    let mut streamed = String::new();
    let outcome = engine
        .run_turn(
            &history,
            &[], // tool-less ⇒ forces a final answer
            &SamplerConfig::deterministic(),
            &CancelFlag::new(),
            &mut |t| streamed.push_str(t),
        )
        .expect("a tool-less turn must succeed");

    match outcome {
        TurnOutcome::Final(text) => {
            assert!(!text.trim().is_empty(), "final answer must be non-empty");
            eprintln!("real tool-less turn => {text:?}");
        }
        TurnOutcome::ToolCalls(calls) => {
            panic!("a tool-less turn must not request tools, got {calls:?}");
        }
        TurnOutcome::Cancelled { .. } => {
            panic!("an un-cancelled turn must not report cancellation");
        }
    }
}

#[test]
#[ignore = "requires MINUTIST_LLM_MODEL_PATH"]
fn real_turn_can_request_a_tool() {
    let Some(substrate) = open_substrate() else {
        return;
    };
    let backend = LlamaTurnBackend::new(substrate.model(), LlamaTurnConfig::default());
    let engine = TurnEngine::new(backend);

    let descriptors = vec![ToolDescriptor {
        name: "get_transcript",
        description: "Return the full transcript of the current meeting.",
        input_schema: json!({
            "type": "object",
            "properties": { "meeting_id": { "type": "string", "description": "Meeting UUID" } },
            "required": ["meeting_id"],
            "additionalProperties": false,
        }),
    }];

    let history = vec![
        ChatMessage::system(
            "You are a meeting assistant. When the user asks about what was said, \
             call the get_transcript tool with the meeting_id 'current'.",
        ),
        ChatMessage::user("What was said in this meeting? Use your tools."),
    ];

    // Arm the grammar backstop for the 4B model's tool-call reliability.
    let cfg = SamplerConfig {
        grammar_backstop: true,
        ..SamplerConfig::deterministic()
    };

    let outcome = engine
        .run_turn(&history, &descriptors, &cfg, &CancelFlag::new(), &mut |_| {})
        .expect("a tool-offering turn must succeed");

    // Loosely asserted: the model MAY answer directly or request the tool;
    // either is a valid engine outcome. Just record what happened so SP1's
    // tool-selection reliability bar can be judged from the test log.
    match outcome {
        TurnOutcome::ToolCalls(calls) => {
            eprintln!("real tool turn => requested {} tool call(s):", calls.len());
            for c in &calls {
                eprintln!("  {} {}", c.name, c.arguments_json);
            }
        }
        TurnOutcome::Final(text) => {
            eprintln!("real tool turn => answered directly: {text:?}");
        }
        TurnOutcome::Cancelled { .. } => {
            panic!("an un-cancelled turn must not report cancellation");
        }
    }
}
