//! Manifest-consistency guard for the bundled default LLM (Phase 5).
//!
//! `ipc_bridge::commands::DEFAULT_LLM_MODEL_ID` is hand-matched to an entry in
//! `resources/models.json`. With no guard, renaming or removing that manifest
//! entry would silently break the default `summarise_meeting` path (the
//! fallback id would no longer resolve to a downloadable model) without any
//! test failing. This test loads the bundled manifest and asserts the default
//! id is still present as a `kind = Llm` entry, so a manifest rename fails here
//! instead.
//!
//! It lives in `tests/` (an integration test, hook-exempt — it touches no
//! `crates/*/src`) and uses `model-registry` purely as a **dev-dependency**:
//! `ipc-bridge` still resolves models only through `Orchestrator` at runtime
//! (no production `model-registry` edge; see components.md dependency table).
//! The manifest is reached via the same `CARGO_MANIFEST_DIR`-relative
//! `../../resources/...` convention the orchestrator's e2e tests use.

use std::path::Path;

use ipc_bridge::commands::DEFAULT_LLM_MODEL_ID;
use minutist_common::{ModelId, ModelKind};

/// The bundled `DEFAULT_LLM_MODEL_ID` must exist in `resources/models.json` as
/// an entry whose `kind` is `Llm`.
#[test]
fn default_llm_model_id_is_an_llm_entry_in_the_manifest() {
    let manifest_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../resources/models.json");
    let bytes = std::fs::read(&manifest_path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", manifest_path.display()));

    let entries = model_registry::load_manifest(&bytes).expect("manifest must parse");

    let default_id = ModelId::from(DEFAULT_LLM_MODEL_ID);
    let entry = entries
        .iter()
        .find(|e| e.id == default_id)
        .unwrap_or_else(|| {
            panic!(
                "DEFAULT_LLM_MODEL_ID '{DEFAULT_LLM_MODEL_ID}' is not present in \
                 resources/models.json — a manifest rename has broken the default \
                 summarise_meeting fallback"
            )
        });

    assert_eq!(
        entry.kind,
        ModelKind::Llm,
        "the default summarise model must be an LLM entry, got {:?}",
        entry.kind
    );
}

/// The bundled BGE-M3 retrieval embedder must exist in `resources/models.json` as
/// a `kind = Embed` entry. RAG Phase B's embedder hand-matches this id, so a
/// manifest rename/removal fails here instead of silently breaking retrieval.
#[test]
fn bge_m3_embed_entry_is_present_in_the_manifest() {
    let manifest_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../resources/models.json");
    let bytes = std::fs::read(&manifest_path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", manifest_path.display()));

    let entries = model_registry::load_manifest(&bytes).expect("manifest must parse");

    let embed_id = ModelId::from("bge-m3-q8_0");
    let entry = entries.iter().find(|e| e.id == embed_id).unwrap_or_else(|| {
        panic!(
            "the BGE-M3 embed entry 'bge-m3-q8_0' is not present in \
             resources/models.json — a manifest rename has broken RAG retrieval"
        )
    });

    assert_eq!(
        entry.kind,
        ModelKind::Embed,
        "the retrieval embedder must be an Embed entry, got {:?}",
        entry.kind
    );
}
