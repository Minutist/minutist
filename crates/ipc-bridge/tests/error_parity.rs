//! Serde-shape parity guard between `common::AppError` and `ipc-bridge`'s
//! `IpcError` (FINDING #14).
//!
//! The TypeScript error shape has two sources of truth that must stay in sync:
//!
//! * `common::AppError` — crosses crate boundaries on the broadcast bus and
//!   reaches TypeScript via the events surface (it derives `specta::Type`
//!   behind `common`'s `specta` feature, which `ipc-bridge` enables).
//! * `ipc-bridge::IpcError` — the Tauri *command* return error, hand-mirrored
//!   to carry the same discriminants and serde shape.
//!
//! Both generate a TypeScript union in `ui/src/ipc/bindings.ts`; if their serde
//! tag/field shapes diverge, the webview's error-handling code is wrong for one
//! of the two surfaces with no compile-time signal. tauri-specta derives the TS
//! type from the serde representation, so asserting the serde JSON matches is
//! equivalent to asserting the two generated TS unions match.
//!
//! Two complementary guards:
//!
//! 1. **Exhaustiveness (compile time).** `From<AppError> for IpcError` in
//!    `src/error.rs` is an explicit per-variant match with no catch-all `_`, so
//!    a new `AppError` variant fails to compile until a matching `IpcError`
//!    variant + arm is added. The [`map_variant`] helper below repeats that
//!    exhaustive match, so this test file *also* fails to compile when a variant
//!    is added without being handled here — forcing the representative-value
//!    list to be extended.
//! 2. **Serde shape (run time).** [`every_variant_serialises_identically`]
//!    serialises a representative value of every variant through *both* types
//!    and asserts the resulting JSON is identical (same `code` tag, same
//!    fields), and that each value round-trips back through its own type.
//!
//! This file lives in `tests/` (an integration test) so it sees both the public
//! `AppError` re-exported by `meeting_app_common` and the public `IpcError`
//! exported by `ipc_bridge`. It is hook-exempt (touches no `crates/*/src`).

use ipc_bridge::IpcError;
use meeting_app_common::AppError;

/// Map an `AppError` to the `IpcError` the production `From` impl would produce,
/// via an exhaustive match (no `_` arm). This duplicates the production mapping
/// on purpose: it makes *this test file* fail to compile when an `AppError`
/// variant is added without a corresponding handling here, which is the signal
/// that [`representative_app_errors`] needs a new entry.
///
/// It also lets the test assert the `From` impl produces the same `IpcError`
/// this match expects, catching a mis-wired arm (e.g. swapped fields).
fn map_variant(e: &AppError) -> IpcError {
    match e {
        AppError::Io { context } => IpcError::Io {
            context: context.clone(),
        },
        AppError::ModelLoad { model_id, context } => IpcError::ModelLoad {
            model_id: model_id.clone(),
            context: context.clone(),
        },
        AppError::ModelNotFound { model_id } => IpcError::ModelNotFound {
            model_id: model_id.clone(),
        },
        AppError::ModelDownload { context } => IpcError::ModelDownload {
            context: context.clone(),
        },
        AppError::Inference { backend, context } => IpcError::Inference {
            backend: backend.clone(),
            context: context.clone(),
        },
        AppError::InvalidInput { context } => IpcError::InvalidInput {
            context: context.clone(),
        },
        AppError::Cancelled => IpcError::Cancelled,
        AppError::Unsupported { context } => IpcError::Unsupported {
            context: context.clone(),
        },
        AppError::Internal { context } => IpcError::Internal {
            context: context.clone(),
        },
    }
}

/// A representative value of **every** `AppError` variant, with distinct,
/// non-empty field contents so a swapped or dropped field surfaces as a JSON
/// mismatch (rather than two empty strings comparing equal by accident).
///
/// Adding an `AppError` variant forces [`map_variant`] to stop compiling until
/// the variant is handled; this list must then grow a matching entry (the
/// `expected_variant_count` assertion below is a second, cheaper reminder).
fn representative_app_errors() -> Vec<AppError> {
    vec![
        AppError::Io {
            context: "disk full while writing transcript.json".into(),
        },
        AppError::ModelLoad {
            model_id: "gemma-4-e4b-it-q4_k_m".into(),
            context: "gguf header invalid".into(),
        },
        AppError::ModelNotFound {
            model_id: "no-such-model".into(),
        },
        AppError::ModelDownload {
            context: "connection reset by peer".into(),
        },
        AppError::Inference {
            backend: "llama".into(),
            context: "context window exceeded".into(),
        },
        AppError::InvalidInput {
            context: "notes_json is not valid JSON".into(),
        },
        AppError::Cancelled,
        AppError::Unsupported {
            context: "re_summarise is not implemented".into(),
        },
        AppError::Internal {
            context: "task join failed".into(),
        },
    ]
}

/// Every `AppError` variant serialises to the same JSON as the `IpcError` it
/// maps to (same `code` tag, same fields), through both the explicit
/// [`map_variant`] mapping and the production `From<AppError> for IpcError`.
///
/// This is the run-time half of the parity guard: tauri-specta generates the TS
/// union from this serde representation, so identical JSON ⇒ identical TS type.
#[test]
fn every_variant_serialises_identically() {
    for app_err in representative_app_errors() {
        let via_match = map_variant(&app_err);
        let via_from = IpcError::from(app_err.clone());

        // The production `From` impl must agree with the test's exhaustive
        // mapping (catches a mis-wired arm — e.g. swapped `model_id`/`context`).
        let from_json = serde_json::to_value(&via_from).expect("serialise IpcError (From)");
        let match_json = serde_json::to_value(&via_match).expect("serialise IpcError (match)");
        assert_eq!(
            from_json, match_json,
            "From<AppError> for IpcError disagrees with the expected mapping for {app_err:?}"
        );

        // The AppError and the IpcError must serialise to byte-identical JSON —
        // same `code` discriminant tag, same field names and values. This is
        // the wire-shape parity that keeps the two generated TS unions in sync.
        let app_json = serde_json::to_value(&app_err).expect("serialise AppError");
        assert_eq!(
            app_json, from_json,
            "AppError and IpcError serde shapes diverge for {app_err:?}:\n  \
             AppError => {app_json}\n  IpcError => {from_json}"
        );

        // Each value must round-trip back through its own type, so the shared
        // shape is genuinely (de)serialisable on both surfaces, not just
        // serialisable.
        let app_back: AppError =
            serde_json::from_value(app_json.clone()).expect("AppError round-trips");
        assert_eq!(
            serde_json::to_value(&app_back).expect("re-serialise AppError"),
            app_json,
            "AppError did not round-trip for {app_err:?}"
        );
        let ipc_back: IpcError =
            serde_json::from_value(from_json.clone()).expect("IpcError round-trips");
        assert_eq!(
            serde_json::to_value(&ipc_back).expect("re-serialise IpcError"),
            from_json,
            "IpcError did not round-trip for {app_err:?}"
        );
    }
}

/// Cheap reminder that the representative list covers the full variant set. If a
/// variant is added, [`map_variant`] stops compiling (the primary guard); this
/// assertion is a secondary, human-readable nudge to extend
/// [`representative_app_errors`] too. Keep this count in lockstep with the
/// `AppError` enum in `crates/common/src/lib.rs`.
#[test]
fn representative_list_covers_every_variant() {
    assert_eq!(
        representative_app_errors().len(),
        9,
        "AppError has 9 variants — a count change means a variant was added or \
         removed; update representative_app_errors and map_variant to match"
    );
}
