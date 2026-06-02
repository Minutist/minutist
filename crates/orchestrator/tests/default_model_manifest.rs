//! Manifest-consistency guard for the bundled diarize models (Phase 6, W7).
//!
//! `runner.rs` hardcodes the two diarize model ids it resolves against the
//! bundled `resources/models.json`:
//!
//! - `DIARIZE_SEG_MODEL_ID = "pyannote-segmentation-3-0"`
//! - `DIARIZE_EMB_MODEL_ID = "3dspeaker-campplus-zh-cn-16k-common"`
//!
//! With no guard, renaming or dropping either manifest entry (or flipping its
//! `kind`) would silently break both the on-stop diarization pass and the
//! user-triggered re-diarize — `build_diarizer` would fail to `ensure` the model
//! at runtime — without any default-suite test failing. This test loads the
//! bundled manifest and asserts both ids are present as `kind = Diarize`, so a
//! manifest rename fails here, model-free, instead.
//!
//! It lives in `tests/` (an integration test, hook-exempt — it touches no
//! `crates/*/src`) and uses `model-registry` as a dev-dependency to parse the
//! manifest, mirroring `ipc-bridge`'s `default_model_manifest.rs`. The two ids
//! are asserted as literals because `runner::DIARIZE_SEG_MODEL_ID` /
//! `DIARIZE_EMB_MODEL_ID` are `pub(crate)` (not reachable from an integration
//! test); they are kept in lockstep with `runner.rs` by this guard's failure
//! message. The manifest is reached via the same `CARGO_MANIFEST_DIR`-relative
//! `../../resources/...` convention the orchestrator's e2e tests use.

use std::path::Path;

use meeting_app_common::{ModelId, ModelKind};

/// Literal copies of `runner::DIARIZE_SEG_MODEL_ID` / `DIARIZE_EMB_MODEL_ID`
/// (crate-private, so they cannot be imported here). If `runner.rs` renames a
/// const, update these too — the assertion below cross-checks them against the
/// bundled manifest.
const DIARIZE_SEG_MODEL_ID: &str = "pyannote-segmentation-3-0";
const DIARIZE_EMB_MODEL_ID: &str = "3dspeaker-campplus-zh-cn-16k-common";

/// Both bundled diarize model ids must exist in `resources/models.json` as
/// entries whose `kind` is `Diarize`.
#[test]
fn bundled_diarize_model_ids_are_diarize_entries_in_the_manifest() {
    let manifest_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../resources/models.json");
    let bytes = std::fs::read(&manifest_path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", manifest_path.display()));

    let entries = model_registry::load_manifest(&bytes).expect("manifest must parse");

    for id in [DIARIZE_SEG_MODEL_ID, DIARIZE_EMB_MODEL_ID] {
        let model_id = ModelId::from(id);
        let entry = entries.iter().find(|e| e.id == model_id).unwrap_or_else(|| {
            panic!(
                "diarize model id '{id}' is not present in resources/models.json — a manifest \
                 rename/drop has broken runner::build_diarizer (on-stop + re-diarize)"
            )
        });
        assert_eq!(
            entry.kind,
            ModelKind::Diarize,
            "bundled diarize model '{id}' must be a Diarize entry, got {:?}",
            entry.kind
        );
    }
}
