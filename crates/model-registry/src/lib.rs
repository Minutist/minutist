//! Model registry — download, hash verification, and version metadata for
//! ML model files.
//!
//! # Responsibilities
//!
//! - Parse `resources/models.json` (manifest) at startup via [`load_manifest`].
//! - Expose [`ModelRegistry`] which resolves model files to local paths,
//!   downloading them on demand with SHA-256 verification.
//! - Emit [`AppEvent::ModelDownloadProgress`] via the shared broadcast channel
//!   at ≤ 10 Hz during downloads.
//!
//! # Filesystem layout
//!
//! ```text
//! {cache_root}/          ← `{app-data}/models/`
//!   asr/{model-id}/      ← ASR model files (e.g. Qwen3-ASR GGUFs)
//!   llm/{model-id}/      ← summarisation LLM files  (Phase 5)
//!   diarize/{model-id}/  ← diarization model files  (Phase 6)
//! ```
//!
//! `model-registry` is the **only** crate that may write under
//! `{app-data}/models/`.  See `architecture/cross-cutting.md`.
//!
//! # In-flight deduplication
//!
//! Concurrent `ensure(same_id)` calls are coalesced: the second caller blocks
//! on a shared `tokio::sync::watch` channel until the first finishes, then
//! adopts the first caller's outcome (the same `Ok` path, or the first
//! caller's real error) rather than starting a second download.
//!
//! # Tracing target
//!
//! All log calls use `target = "model-registry"` so they can be filtered
//! independently.

pub mod error;
pub mod manifest;
pub mod registry;
#[cfg(test)]
mod tests;

pub use error::Error;
pub use manifest::{parse_manifest, ManifestFile};
pub use registry::ModelRegistry;

use meeting_app_common::{AppResult, ModelManifestEntry};

/// Convenience constructor: parse the bundled manifest bytes.
///
/// `manifest_bytes` should be the contents of `resources/models.json` (e.g.
/// via `include_bytes!` in `app-main`).
pub fn load_manifest(manifest_bytes: &[u8]) -> AppResult<Vec<ModelManifestEntry>> {
    parse_manifest(manifest_bytes).map_err(Into::into)
}
