//! Tauri command handlers for the Phase 1 IPC surface.
//!
//! Every command is annotated with both `#[tauri::command]` (wires it into
//! Tauri's invoke handler) and `#[specta::specta]` (registers its signature
//! for TypeScript generation).
//!
//! Commands are `async fn` because the orchestrator's methods are async.
//!
//! `list_devices` routes through `Orchestrator::list_devices` (which wraps
//! the cpal enumeration in `spawn_blocking`), preserving the dependency-table
//! invariant that `ipc-bridge` depends only on `orchestrator + settings +
//! common`.
//!
//! All commands return `AppResult<T>` (`Result<T, AppError>`). The `?`
//! operator converts a per-crate `Error` at the crate boundary via
//! `From<Error> for AppError`.
//!
//! ## Specta types
//!
//! `common` and `settings` derive `specta::Type` directly (gated on the
//! `specta` feature, which this crate enables on both deps). The mirror
//! layer that Phase 1 carried in `specta_types.rs` was removed in P0a;
//! commands return `common` / `settings` types directly.
//!
//! ## Tauri State
//!
//! Each command that needs the orchestrator or settings receives its handles
//! as `tauri::State<'_, IpcState>`.
//!
//! ## Module layout
//!
//! Commands are grouped by domain into submodules (`device`, `recording`,
//! `models`, `settings_commands`, `notes`, `attachments_commands`,
//! `meetings`, `collections_commands`, `summary`, `chat_commands`, `translation`,
//! `voiceprints`). Every item any submodule exposes is re-exported here at
//! the flat `commands::name` path, so callers (including
//! `tauri::generate_handler!` in `lib.rs`, and other crates reaching
//! `ipc_bridge::commands::DEFAULT_LLM_MODEL_ID`) are unaffected by this
//! internal grouping.

use std::collections::HashMap;
use std::path::Path;

use std::sync::Arc;

use agent_tools::{ToolContext, ToolOutput};
use chat_agent::{LlamaTurnBackend, LlamaTurnConfig, TurnEngine};
use minutist_common::{
    AppError, AppEvent, AppResult, AttachmentEntry, AttachmentId, AudioDevice, ChatMessage,
    ChatRole, ChatSession, ChatSessionId, Collection, CollectionId, ConversionState, MeetingId,
    MeetingListEntry, MeetingMeta, MeetingState, ModelId, ModelStatus, NotesDocument, OperationKind,
    RecordingState, Summariser, VoiceprintIdentityId,
};
use notes_crdt::NotesStore;
use persistence::{collections, meeting_ops, ChatStore};
use settings::Settings;
use summariser::{LlamaSummariser, SummariseProgress, SummariserConfig};
use tauri::State;
use tokio::sync::broadcast;

use crate::attachments::ConvertJob;
use crate::chat::{
    engine_message_from_wire, initial_history, run_chat_turn, wire_role, CHAT_N_CTX,
};
use crate::chat_runtime::ChatHandles;
use crate::live_agent::{UserChatRequest, UserReplyChunk};
use crate::output_language::resolve_output_language;
use crate::IpcState;

mod device;
mod recording;
mod models;
mod settings_commands;
mod notes;
mod attachments_commands;
mod meetings;
mod collections_commands;
mod summary;
mod chat_commands;
mod translation;
mod voiceprints;

#[cfg(test)]
mod tests;

pub use device::*;
pub use recording::*;
pub use models::*;
pub use settings_commands::*;
pub use notes::*;
pub use attachments_commands::*;
pub use meetings::*;
pub use collections_commands::*;
pub use summary::*;
pub use chat_commands::*;
pub use translation::*;
pub use voiceprints::*;

/// Append an output-language instruction to a system prompt when
/// `settings.output_language` resolves to a concrete language name.
///
/// Applies to both the summariser system prompt and the chat system prompt:
/// when [`resolve_output_language`] returns `Some(lang)`, appends
/// `"\n\nRespond entirely in {lang}."` after the full prompt (including any
/// user-customised text). Appending AFTER any user-customised prompt ensures
/// the explicit output-language setting is honoured even when the user's
/// custom prompt itself says something different. Returns the prompt unchanged
/// when the setting resolves to `None` (e.g. `"auto"` on an unmapped locale).
pub(crate) fn apply_output_language(prompt: &str, output_language_setting: &str) -> String {
    match resolve_output_language(output_language_setting) {
        Some(lang) => format!("{prompt}\n\nRespond entirely in {lang}."),
        None => prompt.to_string(),
    }
}

/// The bundled default LLM model id used when `settings.llm_model_id` is unset.
///
/// Matches the `gemma-4-e4b-it-q4_k_m` entry in `resources/models.json`
/// (`kind = "llm"`). The model is settings-selected — never hard-coded into the
/// summariser — so a user override is honoured first; this constant is only the
/// fallback. See `architecture/components.md` — `summariser` "Bundled default
/// model".
///
/// `pub` so the manifest-consistency guard test
/// (`crates/ipc-bridge/tests/default_model_manifest.rs`) can assert this id
/// stays a real `kind = Llm` entry in `resources/models.json` — turning a
/// manifest rename into a failing test rather than a silently-broken default
/// summarise path.
pub const DEFAULT_LLM_MODEL_ID: &str = "gemma-4-e4b-it-q4_k_m";

/// The bundled retrieval embedder id (BGE-M3) — hand-matched to a `kind = embed`
/// entry in `resources/models.json` (guarded by a test in `tests/`). Used by the
/// RAG write path and the `retrieve_chunks` tool.
pub const DEFAULT_EMBED_MODEL_ID: &str = "bge-m3-q8_0";

/// Resolve the LLM model id used by [`summarise_meeting`]: the user-selected
/// `settings.llm_model_id` if set, else the bundled default
/// [`DEFAULT_LLM_MODEL_ID`].
///
/// Extracted (rather than inlined in the command) so the settings-override /
/// fallback decision is unit-testable without a Tauri runtime or an
/// orchestrator (a Phase 5 design decision).
pub(crate) fn resolve_llm_model_id(settings: &Settings) -> ModelId {
    settings
        .llm_model_id
        .clone()
        .unwrap_or_else(|| ModelId::from(DEFAULT_LLM_MODEL_ID))
}
