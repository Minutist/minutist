//! Model registry — the on-disk model cache, the model-manifest schema,
//! download + resume + hash verification, and version metadata exposed to
//! other components.
//!
//! `model-registry` is the **only** crate allowed to write under
//! `{app-data}/models/`. See `architecture/cross-cutting.md`.
//!
//! # Responsibilities
//!
//! - Parse `resources/models.json` (the manifest, loaded via `include_bytes!`
//!   in `app-main` and passed in as bytes) via [`load_manifest`].
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
//!   llm/{model-id}/      ← summarisation LLM files
//!   diarize/{model-id}/  ← diarization model files
//! ```
//!
//! # In-flight deduplication
//!
//! Concurrent `ensure(same_id)` calls are coalesced: the second caller blocks
//! on a shared `tokio::sync::watch` channel until the first finishes, then
//! adopts the first caller's outcome (the same `Ok` path, or the first
//! caller's real error) rather than starting a second download — so each
//! model is downloaded at most once per process lifetime regardless of how
//! many callers request it concurrently.
//!
//! # Event source
//!
//! `ModelRegistry::new` takes a `broadcast::Sender<AppEvent>` — the *same*
//! channel the orchestrator broadcasts on (`app-main` constructs the channel
//! once and shares it). The registry emits `AppEvent::ModelDownloadProgress`
//! directly onto that bus during `ensure`, so it is a first-class event
//! source in its own right, not solely a path provider — the IPC forwarder's
//! single subscription sees its progress events too.
//!
//! Progress is reported against an entry's **aggregate** byte total (the sum
//! of every file in the manifest entry), not per-file: a multi-file model
//! (e.g. an ASR GGUF + `mmproj` pair) drives one monotonic 0→100% bar rather
//! than resetting between files. A terminal `bytes_done == bytes_total` event
//! is emitted once all files verify, so a consumer's completion check fires
//! deterministically rather than depending on a throttled per-chunk emit
//! coinciding with the last byte. Verification failures (e.g. a SHA-256
//! mismatch from a stale manifest) are returned to the `ensure` caller as an
//! `Err`, not published on the broadcast bus — the caller's own seam is
//! where they get surfaced to the user.
//!
//! Manifest file URLs MUST pin an immutable commit revision; a moving ref
//! (e.g. `main`) drifts when the upstream repo is re-uploaded and silently
//! breaks hash verification.
//!
//! # Tracing target
//!
//! All log calls use `target: "model-registry"` so they can be filtered
//! independently.

pub mod error;
pub mod manifest;
pub mod registry;
#[cfg(test)]
mod tests;

pub use error::Error;
pub use manifest::{parse_manifest, ManifestFile};
pub use registry::ModelRegistry;

use minutist_common::{AppResult, ModelManifestEntry};

/// Convenience constructor: parse the bundled manifest bytes.
///
/// `manifest_bytes` should be the contents of `resources/models.json` (e.g.
/// via `include_bytes!` in `app-main`).
pub fn load_manifest(manifest_bytes: &[u8]) -> AppResult<Vec<ModelManifestEntry>> {
    parse_manifest(manifest_bytes).map_err(Into::into)
}
