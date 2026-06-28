//! Behavioural tests for the v1 tool layer.
//!
//! Uses tempdir meeting fixtures + a `StubSummariser`. The orchestrator-backed
//! tools (relisten / reprocess) are driven through the orchestrator's
//! `test-source` seam where a real model would otherwise be required; the
//! read/compute + metadata-write tools run directly against tempdir fixtures.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use agent_tools::{ToolContext, ToolRegistry};
use minutist_common::{
    AppEvent, AppResult, AudioFormat, MeetingId, MeetingMeta, NoteBlock, Segment, Summariser,
};
use orchestrator::test_support::test_orchestrator;
use orchestrator::Orchestrator;
use persistence::MeetingIndex;
use tempfile::TempDir;
use tokio::sync::broadcast;

// ---------------------------------------------------------------------------
// Test substrate
// ---------------------------------------------------------------------------

/// A `Summariser` stub that echoes the transcript length + the instruction it
/// was handed, so a test can assert `resummarise` plumbed the instruction
/// through as the system prompt.
struct StubSummariser;

impl Summariser for StubSummariser {
    fn summarise(
        &self,
        transcript: &[Segment],
        _notes: &[NoteBlock],
        _attachments_markdown: &str,
        system_prompt: &str,
    ) -> AppResult<String> {
        Ok(format!(
            "STUB SUMMARY ({} segments) :: prompt={system_prompt}",
            transcript.len()
        ))
    }
}

/// Build a `ToolContext` over a tempdir-backed meetings root + in-memory index +
/// a test orchestrator. Returns the context plus the `TempDir` guard (kept alive
/// by the caller) and the meetings root path.
async fn make_ctx() -> (TempDir, std::path::PathBuf, ToolContext) {
    let tempdir = TempDir::new().expect("tempdir");
    let meetings_dir = tempdir.path().join("meetings");
    std::fs::create_dir_all(&meetings_dir).expect("meetings dir");

    let index = Arc::new(MeetingIndex::open(":memory:").await.expect("index"));
    let orchestrator: Arc<Orchestrator> = Arc::new(test_orchestrator(meetings_dir.clone()));
    let summariser: Arc<dyn Summariser> = Arc::new(StubSummariser);
    let (event_tx, _rx) = broadcast::channel::<AppEvent>(16);

    let ctx = ToolContext::new(
        orchestrator,
        index,
        meetings_dir.clone(),
        summariser,
        None, // no embedder (the retrieve_chunks test builds its own ctx)
        event_tx,
        None,
    );
    (tempdir, meetings_dir, ctx)
}

/// Seed a meeting folder on disk with metadata + transcript, index it, and
/// return its id. `segments` carry whatever speaker labels the test wants.
async fn seed_meeting(
    meetings_dir: &Path,
    index: &MeetingIndex,
    title: &str,
    segments: Vec<Segment>,
    speaker_names: BTreeMap<String, String>,
) -> MeetingId {
    let id = MeetingId::new();
    persistence::MeetingFolder::create(meetings_dir, id).expect("folder");
    let dir = meetings_dir.join(id.0.to_string());

    let meta = MeetingMeta {
        uuid: id,
        title: title.to_string(),
        started_at: "2026-06-10T09:00:00Z".to_string(),
        ended_at: Some("2026-06-10T09:30:00Z".to_string()),
        duration_ms: 1_800_000,
        speaker_count: speaker_names.len() as u32,
        audio_format: AudioFormat {
            codec: "opus".to_string(),
            sample_rate: 16_000,
            channels: 1,
            bitrate_kbps: Some(32),
        },
        asr_model: None,
        llm_model: None,
        diarizer: None,
        speaker_names,
        notes_format: 0,
        processing: Default::default(),
        collection_id: None,
        app_version: "0.0.0".to_string(),
    };
    persistence::write_metadata(&dir, &meta).expect("write metadata");
    if !segments.is_empty() {
        persistence::write_transcript(&dir, &segments).expect("write transcript");
    }

    // Index it so list/search tools see it.
    let entry = minutist_common::MeetingListEntry {
        id,
        title: title.to_string(),
        started_at: meta.started_at.clone(),
        duration_ms: meta.duration_ms,
        speaker_count: meta.speaker_count,
        excerpt: segments_excerpt(meetings_dir, id),
        collection_id: meta.collection_id,
    };
    index.upsert(&entry).await.expect("index upsert");
    id
}

fn segments_excerpt(meetings_dir: &Path, id: MeetingId) -> Option<String> {
    let dir = meetings_dir.join(id.0.to_string());
    persistence::read_transcript(&dir)
        .ok()
        .and_then(|s| s.first().map(|seg| seg.text.clone()))
}

fn seg(start_ms: u64, end_ms: u64, text: &str, speaker: Option<&str>) -> Segment {
    Segment {
        start_ms,
        end_ms,
        text: text.to_string(),
        speaker_id: speaker.map(|s| s.to_string()),
        confidence: None,
        words: vec![],
        shared_speakers: Vec::new(),
    }
}

fn names(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
    pairs
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}

// ---------------------------------------------------------------------------
// Registry shape: is_write / expose_over_mcp / dispatch
// ---------------------------------------------------------------------------

#[tokio::test]
async fn registry_v1_has_the_documented_tool_set() {
    let reg = ToolRegistry::v1(false);
    let names: Vec<&str> = reg.descriptors().iter().map(|d| d.name).collect();
    let expected = [
        "list_meetings",
        "search_meetings",
        "get_meeting",
        "get_transcript",
        "get_transcript_slice",
        "get_summary",
        "get_notes",
        "get_metadata",
        "get_recording_state",
        "search_within_transcript",
        "retrieve_chunks",
        "relisten_section",
        "resummarise",
        "speaker_talk_time",
        "list_attachments",
        "get_attachment_markdown",
        "set_speaker_name",
        "rename_meeting",
        "reprocess_meeting",
        "start_recording",
        "stop_recording",
        "pause_recording",
        "resume_recording",
    ];
    assert_eq!(reg.len(), expected.len(), "v1 tool count");
    for name in expected {
        assert!(reg.get(name).is_some(), "{name} must be registered");
    }
    assert!(names.contains(&"list_meetings"));
}

#[tokio::test]
async fn v1_true_adds_inter_agent_tool_only() {
    // The MCP registry (`v1(true)`) is the v1(false) set PLUS exactly one tool:
    // send_to_internal_agent. It is not a write.
    let base = ToolRegistry::v1(false);
    let mcp = ToolRegistry::v1(true);
    assert_eq!(mcp.len(), base.len() + 1, "v1(true) adds exactly one tool");
    assert!(
        mcp.get("send_to_internal_agent").is_some(),
        "v1(true) must register send_to_internal_agent"
    );
    assert!(
        base.get("send_to_internal_agent").is_none(),
        "v1(false) must NOT register send_to_internal_agent (no self-messaging)"
    );
    assert!(
        !mcp.get("send_to_internal_agent").unwrap().is_write(),
        "send_to_internal_agent is not a write"
    );
}

#[tokio::test]
async fn mcp_write_gate_projection() {
    let reg = ToolRegistry::v1(true);

    // Gate OFF: reads + the inter-agent tool, NO writes (not even the reversible
    // ones), and never the heavy ops.
    let off: Vec<&str> = reg
        .mcp_tool_descriptors_gated(false)
        .iter()
        .map(|d| d.name)
        .collect();
    assert!(off.contains(&"list_meetings"));
    assert!(off.contains(&"send_to_internal_agent"));
    assert!(!off.contains(&"set_speaker_name"));
    assert!(!off.contains(&"rename_meeting"));
    assert!(!off.contains(&"reprocess_meeting"));

    // Gate ON: the reversible writes join; the heavy op STILL never appears.
    let on: Vec<&str> = reg
        .mcp_tool_descriptors_gated(true)
        .iter()
        .map(|d| d.name)
        .collect();
    assert!(on.contains(&"set_speaker_name"));
    assert!(on.contains(&"rename_meeting"));
    assert!(!on.contains(&"reprocess_meeting"));

    // mcp_call_allowed mirrors the listing under each gate.
    assert!(!reg.mcp_call_allowed("set_speaker_name", false));
    assert!(reg.mcp_call_allowed("set_speaker_name", true));
    assert!(!reg.mcp_call_allowed("reprocess_meeting", true));
    assert!(!reg.mcp_call_allowed("unknown_tool", true));
}

#[tokio::test]
async fn record_control_tools_are_write_gated_over_mcp() {
    // The four record-control tools (#62) are `is_write` AND `expose_over_mcp`,
    // so they are WRITE-GATED exactly like `set_speaker_name`/`rename_meeting`:
    // absent + rejected when `mcp_write_tools` is OFF (the default), present +
    // callable when it is ON. This is what lets an external MCP client drive the
    // record→transcribe→read loop for E2E only when explicitly opted in.
    let reg = ToolRegistry::v1(true);
    let record_control = [
        "start_recording",
        "stop_recording",
        "pause_recording",
        "resume_recording",
    ];

    // OFF gate: absent from tools/list and rejected by mcp_call_allowed.
    let off: Vec<&str> = reg
        .mcp_tool_descriptors_gated(false)
        .iter()
        .map(|d| d.name)
        .collect();
    for name in record_control {
        assert!(
            !off.contains(&name),
            "{name} must be absent from tools/list when mcp_write_tools is off"
        );
        assert!(
            !reg.mcp_call_allowed(name, false),
            "{name} must be rejected over MCP when mcp_write_tools is off"
        );
    }

    // ON gate: present in tools/list and callable.
    let on: Vec<&str> = reg
        .mcp_tool_descriptors_gated(true)
        .iter()
        .map(|d| d.name)
        .collect();
    for name in record_control {
        assert!(
            on.contains(&name),
            "{name} must be exposed over MCP when mcp_write_tools is on"
        );
        assert!(
            reg.mcp_call_allowed(name, true),
            "{name} must be callable over MCP when mcp_write_tools is on"
        );
    }

    // They ARE in the expose-only projection (expose_over_mcp == true), unlike
    // the internal-only write (reprocess).
    let exposed: Vec<&str> = reg.mcp_tool_descriptors().iter().map(|d| d.name).collect();
    for name in record_control {
        assert!(exposed.contains(&name), "{name} must be MCP-exposable");
    }
    assert!(
        !exposed.contains(&"reprocess_meeting"),
        "the heavy op stays internal-only (expose_over_mcp == false)"
    );
}

#[tokio::test]
async fn write_flags_are_set_correctly() {
    let reg = ToolRegistry::v1(false);
    let writes = [
        "set_speaker_name",
        "rename_meeting",
        "reprocess_meeting",
        "start_recording",
        "stop_recording",
        "pause_recording",
        "resume_recording",
    ];
    for name in writes {
        assert!(
            reg.get(name).unwrap().is_write(),
            "{name} must be is_write()"
        );
    }
    // A representative read tool must NOT be a write.
    assert!(!reg.get("get_transcript").unwrap().is_write());
    assert!(!reg.get("relisten_section").unwrap().is_write());
    assert!(!reg.get("resummarise").unwrap().is_write());
}

#[tokio::test]
async fn mcp_exposure_default_safe_with_allowlist() {
    let reg = ToolRegistry::v1(false);
    let mcp: Vec<&str> = reg.mcp_tool_descriptors().iter().map(|d| d.name).collect();

    // All reads/compute are exposed.
    for name in [
        "list_meetings",
        "get_transcript",
        "relisten_section",
        "resummarise",
        "speaker_talk_time",
    ] {
        assert!(mcp.contains(&name), "{name} should be exposed over MCP");
    }
    // Allowlisted writes are exposed.
    assert!(mcp.contains(&"set_speaker_name"));
    assert!(mcp.contains(&"rename_meeting"));
    // The heavy write is internal-only.
    assert!(!mcp.contains(&"reprocess_meeting"));
}

#[tokio::test]
async fn dispatch_unknown_tool_is_invalid_input() {
    let (_t, _root, ctx) = make_ctx().await;
    let reg = ToolRegistry::v1(false);
    let err = reg
        .dispatch(&ctx, "no_such_tool", serde_json::json!({}))
        .await
        .unwrap_err();
    assert!(matches!(
        err,
        minutist_common::AppError::InvalidInput { .. }
    ));
}

#[tokio::test]
async fn retrieve_chunks_without_embedder_errors_gracefully() {
    // `make_ctx` wires no embedder (None); retrieve_chunks must surface a clean
    // InvalidInput rather than panic — chat works without retrieval available.
    let (_t, _root, ctx) = make_ctx().await;
    let reg = ToolRegistry::v1(false);
    let id = minutist_common::MeetingId::new();
    let err = reg
        .dispatch(
            &ctx,
            "retrieve_chunks",
            serde_json::json!({ "meeting_id": id.0.to_string(), "query": "anything" }),
        )
        .await
        .unwrap_err();
    assert!(matches!(
        err,
        minutist_common::AppError::InvalidInput { .. }
    ));
}

#[tokio::test]
async fn dispatch_missing_required_arg_is_invalid_input() {
    let (_t, _root, ctx) = make_ctx().await;
    let reg = ToolRegistry::v1(false);
    // search_meetings requires `query`; an empty object must be rejected by the
    // schema validation path (`query` is not the meeting_id default exception).
    let err = reg
        .dispatch(&ctx, "search_meetings", serde_json::json!({}))
        .await
        .unwrap_err();
    assert!(matches!(
        err,
        minutist_common::AppError::InvalidInput { .. }
    ));
}

#[tokio::test]
async fn dispatch_missing_meeting_without_default_is_invalid_input() {
    let (_t, _root, ctx) = make_ctx().await;
    let reg = ToolRegistry::v1(false);
    // get_transcript needs a meeting; with no `meeting_id` arg and no
    // default_meeting in the context, resolve_meeting rejects it.
    let err = reg
        .dispatch(&ctx, "get_transcript", serde_json::json!({}))
        .await
        .unwrap_err();
    assert!(matches!(
        err,
        minutist_common::AppError::InvalidInput { .. }
    ));
}

#[tokio::test]
async fn default_meeting_resolves_when_meeting_id_omitted() {
    let (_t, root, mut ctx) = make_ctx().await;
    let reg = ToolRegistry::v1(false);
    let id = seed_meeting(
        &root,
        &ctx.index,
        "Scoped",
        vec![seg(0, 1000, "scoped meeting body", None)],
        BTreeMap::new(),
    )
    .await;
    // Scope the session to this meeting (mirrors the internal-UI wiring).
    ctx.default_meeting = Some(id);

    // get_transcript with NO meeting_id arg resolves via default_meeting.
    let out = reg
        .dispatch(&ctx, "get_transcript", serde_json::json!({}))
        .await
        .unwrap();
    let arr = out.data.as_array().unwrap();
    assert_eq!(arr.len(), 1);
    assert_eq!(arr[0]["text"], "scoped meeting body");
}

// ---------------------------------------------------------------------------
// Read tools + overlay
// ---------------------------------------------------------------------------

#[tokio::test]
async fn list_and_search_meetings_via_dispatch() {
    let (_t, root, ctx) = make_ctx().await;
    let reg = ToolRegistry::v1(false);
    seed_meeting(
        &root,
        &ctx.index,
        "Launch sync",
        vec![seg(0, 1000, "kickoff", None)],
        BTreeMap::new(),
    )
    .await;
    seed_meeting(
        &root,
        &ctx.index,
        "Retro",
        vec![seg(0, 1000, "retrospective", None)],
        BTreeMap::new(),
    )
    .await;

    let out = reg
        .dispatch(&ctx, "list_meetings", serde_json::json!({}))
        .await
        .unwrap();
    let arr = out.data.as_array().unwrap();
    assert_eq!(arr.len(), 2);

    let out = reg
        .dispatch(
            &ctx,
            "search_meetings",
            serde_json::json!({ "query": "Retro" }),
        )
        .await
        .unwrap();
    let arr = out.data.as_array().unwrap();
    assert_eq!(arr.len(), 1);
    assert_eq!(arr[0]["title"], "Retro");
}

#[tokio::test]
async fn get_transcript_applies_speaker_overlay() {
    let (_t, root, ctx) = make_ctx().await;
    let reg = ToolRegistry::v1(false);
    let id = seed_meeting(
        &root,
        &ctx.index,
        "Standup",
        vec![
            seg(0, 1000, "hi", Some("A")),
            seg(1000, 2000, "hello", Some("B")),
            seg(2000, 3000, "unnamed", Some("C")),
        ],
        names(&[("A", "Alice"), ("B", "Bob")]),
    )
    .await;

    let out = reg
        .dispatch(
            &ctx,
            "get_transcript",
            serde_json::json!({ "meeting_id": id.0.to_string() }),
        )
        .await
        .unwrap();
    let arr = out.data.as_array().unwrap();
    assert_eq!(arr[0]["speaker_id"], "Alice", "A → Alice");
    assert_eq!(arr[1]["speaker_id"], "Bob", "B → Bob");
    assert_eq!(
        arr[2]["speaker_id"], "C",
        "unmapped label stays as the raw label"
    );

    // The on-disk transcript must be UNTOUCHED (overlay is presentation-only).
    let dir = root.join(id.0.to_string());
    let on_disk = persistence::read_transcript(&dir).unwrap();
    assert_eq!(on_disk[0].speaker_id.as_deref(), Some("A"));
}

#[tokio::test]
async fn get_transcript_slice_filters_by_overlap() {
    let (_t, root, ctx) = make_ctx().await;
    let reg = ToolRegistry::v1(false);
    let id = seed_meeting(
        &root,
        &ctx.index,
        "M",
        vec![
            seg(0, 1000, "a", None),
            seg(1000, 2000, "b", None),
            seg(5000, 6000, "c", None),
        ],
        BTreeMap::new(),
    )
    .await;

    let out = reg
        .dispatch(
            &ctx,
            "get_transcript_slice",
            serde_json::json!({ "meeting_id": id.0.to_string(), "start_ms": 1500, "end_ms": 5500 }),
        )
        .await
        .unwrap();
    let arr = out.data.as_array().unwrap();
    // Overlaps [1500, 5500): segment b (1000-2000) and c (5000-6000); a (0-1000) excluded.
    assert_eq!(arr.len(), 2);
    assert_eq!(arr[0]["text"], "b");
    assert_eq!(arr[1]["text"], "c");
}

#[tokio::test]
async fn search_within_transcript_is_case_insensitive() {
    let (_t, root, ctx) = make_ctx().await;
    let reg = ToolRegistry::v1(false);
    let id = seed_meeting(
        &root,
        &ctx.index,
        "M",
        vec![
            seg(0, 1000, "The Budget was approved", Some("A")),
            seg(1000, 2000, "lunch plans", None),
        ],
        names(&[("A", "Alice")]),
    )
    .await;

    let out = reg
        .dispatch(
            &ctx,
            "search_within_transcript",
            serde_json::json!({ "meeting_id": id.0.to_string(), "query": "budget" }),
        )
        .await
        .unwrap();
    let arr = out.data.as_array().unwrap();
    assert_eq!(arr.len(), 1);
    assert_eq!(arr[0]["speaker_id"], "Alice", "overlay applied to results");
    assert!(arr[0]["text"].as_str().unwrap().contains("Budget"));
}

#[tokio::test]
async fn get_notes_returns_null_when_absent() {
    let (_t, root, ctx) = make_ctx().await;
    let reg = ToolRegistry::v1(false);
    let id = seed_meeting(&root, &ctx.index, "M", vec![], BTreeMap::new()).await;
    let out = reg
        .dispatch(
            &ctx,
            "get_notes",
            serde_json::json!({ "meeting_id": id.0.to_string() }),
        )
        .await
        .unwrap();
    assert!(out.data.is_null());
}

#[tokio::test]
async fn get_notes_round_trips_saved_notes() {
    let (_t, root, ctx) = make_ctx().await;
    let reg = ToolRegistry::v1(false);
    let id = seed_meeting(&root, &ctx.index, "M", vec![], BTreeMap::new()).await;
    let doc = serde_json::json!({ "type": "doc", "content": [] });
    persistence::NotesStore::save(&root, id, &doc, "# Notes\n").expect("save notes");

    let out = reg
        .dispatch(
            &ctx,
            "get_notes",
            serde_json::json!({ "meeting_id": id.0.to_string() }),
        )
        .await
        .unwrap();
    assert_eq!(out.data["markdown"], "# Notes\n");
    assert_eq!(out.data["json"], doc);
}

#[tokio::test]
async fn get_recording_state_reports_idle() {
    let (_t, _root, ctx) = make_ctx().await;
    let reg = ToolRegistry::v1(false);
    let out = reg
        .dispatch(&ctx, "get_recording_state", serde_json::json!({}))
        .await
        .unwrap();
    assert_eq!(out.data["kind"], "idle");
}

#[tokio::test]
async fn speaker_talk_time_aggregates_with_overlay() {
    let (_t, root, ctx) = make_ctx().await;
    let reg = ToolRegistry::v1(false);
    let id = seed_meeting(
        &root,
        &ctx.index,
        "M",
        vec![
            seg(0, 1000, "x", Some("A")),    // A: 1000ms, 1 turn
            seg(1000, 3000, "y", Some("A")), // A: +2000ms, 2 turns
            seg(3000, 3500, "z", Some("B")), // B: 500ms, 1 turn
        ],
        names(&[("A", "Alice")]),
    )
    .await;

    let out = reg
        .dispatch(
            &ctx,
            "speaker_talk_time",
            serde_json::json!({ "meeting_id": id.0.to_string() }),
        )
        .await
        .unwrap();
    let arr = out.data.as_array().unwrap();
    // Sorted by raw speaker_id: A then B.
    assert_eq!(arr[0]["speaker_id"], "A");
    assert_eq!(arr[0]["display_name"], "Alice");
    assert_eq!(arr[0]["total_ms"], 3000);
    assert_eq!(arr[0]["turn_count"], 2);
    assert_eq!(arr[1]["speaker_id"], "B");
    assert_eq!(arr[1]["display_name"], "B");
    assert_eq!(arr[1]["total_ms"], 500);
}

#[tokio::test]
async fn resummarise_threads_instruction_through_as_prompt() {
    let (_t, root, ctx) = make_ctx().await;
    let reg = ToolRegistry::v1(false);
    let id = seed_meeting(
        &root,
        &ctx.index,
        "M",
        vec![seg(0, 1000, "one", None), seg(1000, 2000, "two", None)],
        BTreeMap::new(),
    )
    .await;

    let out = reg
        .dispatch(
            &ctx,
            "resummarise",
            serde_json::json!({ "meeting_id": id.0.to_string(), "instruction": "as bullet points" }),
        )
        .await
        .unwrap();
    let text = out.data["text"].as_str().unwrap();
    assert!(
        text.contains("2 segments"),
        "transcript handed to summariser"
    );
    assert!(
        text.contains("prompt=as bullet points"),
        "instruction used as system prompt"
    );

    // resummarise must NOT write summary.md.
    let dir = root.join(id.0.to_string());
    assert!(persistence::read_summary(&dir).unwrap().is_none());
}

// ---------------------------------------------------------------------------
// Attachment tools
// ---------------------------------------------------------------------------

/// Seed a Ready attachment into `meetings_dir` for `meeting_id` and return the
/// converted markdown filename so tests can pass it to `get_attachment_markdown`.
fn seed_attachment(
    meetings_dir: &std::path::Path,
    meeting_id: minutist_common::MeetingId,
    original_name: &str,
    ext: &str,
    markdown: &str,
) -> String {
    let bytes = format!("dummy content for {original_name}").into_bytes();
    let hash = persistence::save_attachment_original(meetings_dir, meeting_id, &bytes, ext)
        .expect("save original");
    let md_filename =
        persistence::save_attachment_markdown(meetings_dir, meeting_id, &hash, markdown)
            .expect("save markdown");
    let entry = minutist_common::AttachmentEntry {
        id: minutist_common::AttachmentId::new(),
        hash: hash.clone(),
        original_filename: original_name.to_string(),
        ext: ext.to_string(),
        byte_len: bytes.len() as u64,
        added_at: "2026-06-22T10:00:00Z".to_string(),
        conversion: minutist_common::ConversionState::Ready,
        converted_md_filename: Some(md_filename.clone()),
    };
    persistence::add_manifest_entry(meetings_dir, meeting_id, entry).expect("add manifest entry");
    md_filename
}

#[tokio::test]
async fn list_attachments_returns_empty_for_meeting_with_none() {
    let (_t, root, ctx) = make_ctx().await;
    let reg = ToolRegistry::v1(false);
    let id = seed_meeting(&root, &ctx.index, "M", vec![], BTreeMap::new()).await;

    let out = reg
        .dispatch(
            &ctx,
            "list_attachments",
            serde_json::json!({ "meeting_id": id.0.to_string() }),
        )
        .await
        .unwrap();
    let arr = out.data.as_array().unwrap();
    assert!(arr.is_empty(), "no attachments → empty list");
    assert_eq!(out.summary.as_deref(), Some("0 attachment(s)"));
}

#[tokio::test]
async fn list_attachments_returns_manifest_rows() {
    let (_t, root, ctx) = make_ctx().await;
    let reg = ToolRegistry::v1(false);
    let id = seed_meeting(&root, &ctx.index, "M", vec![], BTreeMap::new()).await;
    seed_attachment(&root, id, "slides.pptx", "pptx", "# Slides\n\nContent.");
    seed_attachment(&root, id, "report.pdf", "pdf", "# Report\n\nData.");

    let out = reg
        .dispatch(
            &ctx,
            "list_attachments",
            serde_json::json!({ "meeting_id": id.0.to_string() }),
        )
        .await
        .unwrap();
    let arr = out.data.as_array().unwrap();
    assert_eq!(arr.len(), 2, "two attachments must be listed");
    // Check projected fields are present and no hash field leaks.
    let row = &arr[0];
    assert!(row.get("id").is_some(), "id field present");
    assert!(
        row.get("original_filename").is_some(),
        "original_filename present"
    );
    assert!(row.get("ext").is_some(), "ext present");
    assert!(row.get("conversion").is_some(), "conversion present");
    assert!(row.get("byte_len").is_some(), "byte_len present");
    assert!(
        row.get("converted_md_filename").is_some(),
        "converted_md_filename must be projected so the agent can chain into get_attachment_markdown"
    );
    assert!(
        row.get("hash").is_none(),
        "hash must not be projected to the agent"
    );
}

#[tokio::test]
async fn list_attachments_exposes_converted_md_filename_for_chaining() {
    // Verify that the JSON rows from list_attachments carry `converted_md_filename`
    // so an agent can pass it directly to get_attachment_markdown without knowing
    // the hash. This exercises the list→get chain through the projected JSON.
    let (_t, root, ctx) = make_ctx().await;
    let reg = ToolRegistry::v1(false);
    let id = seed_meeting(&root, &ctx.index, "M", vec![], BTreeMap::new()).await;
    seed_attachment(&root, id, "agenda.md", "md", "# Agenda\n\nItems.");

    // Step 1: list_attachments — extract converted_md_filename from the JSON row.
    let list_out = reg
        .dispatch(
            &ctx,
            "list_attachments",
            serde_json::json!({ "meeting_id": id.0.to_string() }),
        )
        .await
        .unwrap();
    let arr = list_out.data.as_array().unwrap();
    assert_eq!(arr.len(), 1);
    let row = &arr[0];
    let md_filename = row
        .get("converted_md_filename")
        .and_then(|v| v.as_str())
        .expect("converted_md_filename must be present in the list row for a Ready attachment");

    // Step 2: get_attachment_markdown — use the filename extracted from the row.
    let get_out = reg
        .dispatch(
            &ctx,
            "get_attachment_markdown",
            serde_json::json!({ "meeting_id": id.0.to_string(), "filename": md_filename }),
        )
        .await
        .expect("get_attachment_markdown must succeed with filename from list_attachments row");
    assert_eq!(
        get_out.data["markdown"].as_str().unwrap(),
        "# Agenda\n\nItems.",
        "markdown content must match what was seeded"
    );
}

#[tokio::test]
async fn list_attachments_is_read_and_mcp_exposed() {
    let reg = ToolRegistry::v1(false);
    let tool = reg.get("list_attachments").expect("registered");
    assert!(!tool.is_write(), "list_attachments must not be a write");
    assert!(
        tool.expose_over_mcp(),
        "list_attachments must be MCP-exposed by default"
    );
}

#[tokio::test]
async fn get_attachment_markdown_returns_content() {
    let (_t, root, ctx) = make_ctx().await;
    let reg = ToolRegistry::v1(false);
    let id = seed_meeting(&root, &ctx.index, "M", vec![], BTreeMap::new()).await;
    let md_filename = seed_attachment(&root, id, "brief.md", "md", "# Brief\n\nKey points.");

    let out = reg
        .dispatch(
            &ctx,
            "get_attachment_markdown",
            serde_json::json!({ "meeting_id": id.0.to_string(), "filename": md_filename }),
        )
        .await
        .unwrap();
    assert_eq!(
        out.data["markdown"].as_str().unwrap(),
        "# Brief\n\nKey points."
    );
}

#[tokio::test]
async fn get_attachment_markdown_rejects_traversal_filename() {
    let (_t, root, ctx) = make_ctx().await;
    let reg = ToolRegistry::v1(false);
    let id = seed_meeting(&root, &ctx.index, "M", vec![], BTreeMap::new()).await;

    for evil in ["../secret.md", "sub/dir.md", "..", ".", "", "..\\win.md"] {
        let err = reg
            .dispatch(
                &ctx,
                "get_attachment_markdown",
                serde_json::json!({ "meeting_id": id.0.to_string(), "filename": evil }),
            )
            .await
            .expect_err(&format!("traversal filename {evil:?} must be rejected"));
        assert!(
            matches!(err, minutist_common::AppError::InvalidInput { .. }),
            "expected InvalidInput for {evil:?}, got {err:?}"
        );
    }
}

#[tokio::test]
async fn get_attachment_markdown_is_read_and_mcp_exposed() {
    let reg = ToolRegistry::v1(false);
    let tool = reg.get("get_attachment_markdown").expect("registered");
    assert!(
        !tool.is_write(),
        "get_attachment_markdown must not be a write"
    );
    assert!(
        tool.expose_over_mcp(),
        "get_attachment_markdown must be MCP-exposed by default"
    );
}

// ---------------------------------------------------------------------------
// Write tools (metadata)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn set_speaker_name_round_trips_through_metadata() {
    let (_t, root, ctx) = make_ctx().await;
    let reg = ToolRegistry::v1(false);
    let id = seed_meeting(
        &root,
        &ctx.index,
        "M",
        vec![seg(0, 1000, "x", Some("A"))],
        BTreeMap::new(),
    )
    .await;

    let out = reg
        .dispatch(
            &ctx,
            "set_speaker_name",
            serde_json::json!({ "meeting_id": id.0.to_string(), "speaker_id": "A", "name": "Alice" }),
        )
        .await
        .unwrap();
    assert_eq!(out.data["speaker_names"]["A"], "Alice");

    // Reflected in get_metadata.
    let meta = reg
        .dispatch(
            &ctx,
            "get_metadata",
            serde_json::json!({ "meeting_id": id.0.to_string() }),
        )
        .await
        .unwrap();
    assert_eq!(meta.data["speaker_names"]["A"], "Alice");

    // And in the read-time overlay.
    let tr = reg
        .dispatch(
            &ctx,
            "get_transcript",
            serde_json::json!({ "meeting_id": id.0.to_string() }),
        )
        .await
        .unwrap();
    assert_eq!(tr.data.as_array().unwrap()[0]["speaker_id"], "Alice");
}

/// Locks the MCP-boundary contract (issue 0025): routing `set_speaker_name`
/// onto `persistence::meeting_ops::set_speaker_name` means `name:""` CLEARS the
/// label (the canonical/UI behaviour) rather than inserting an empty-string name.
#[tokio::test]
async fn set_speaker_name_empty_name_clears_label() {
    let (_t, root, ctx) = make_ctx().await;
    let reg = ToolRegistry::v1(false);
    let id = seed_meeting(
        &root,
        &ctx.index,
        "M",
        vec![seg(0, 1000, "x", Some("A"))],
        BTreeMap::new(),
    )
    .await;

    // Set a name, then clear it with an empty name.
    reg.dispatch(
        &ctx,
        "set_speaker_name",
        serde_json::json!({ "meeting_id": id.0.to_string(), "speaker_id": "A", "name": "Alice" }),
    )
    .await
    .unwrap();

    let out = reg
        .dispatch(
            &ctx,
            "set_speaker_name",
            serde_json::json!({ "meeting_id": id.0.to_string(), "speaker_id": "A", "name": "" }),
        )
        .await
        .unwrap();
    assert!(
        out.data["speaker_names"].get("A").is_none(),
        "empty name must clear the label, got {:?}",
        out.data["speaker_names"]
    );

    // Cleared on disk too.
    let meta = reg
        .dispatch(
            &ctx,
            "get_metadata",
            serde_json::json!({ "meeting_id": id.0.to_string() }),
        )
        .await
        .unwrap();
    assert!(meta.data["speaker_names"].get("A").is_none());
}

#[tokio::test]
async fn concurrent_set_speaker_name_does_not_drop_a_write() {
    let (_t, root, ctx) = make_ctx().await;
    let reg = Arc::new(ToolRegistry::v1(false));
    let id = seed_meeting(&root, &ctx.index, "M", vec![], BTreeMap::new()).await;

    // Two concurrent set_speaker_name calls inserting DIFFERENT labels. Without
    // the per-meeting metadata mutex this is a read-modify-write race where one
    // insert clobbers the other (last-writer-wins drops a name).
    let ctx_a = ctx.clone();
    let ctx_b = ctx.clone();
    let reg_a = Arc::clone(&reg);
    let reg_b = Arc::clone(&reg);
    let id_s = id.0.to_string();
    let id_s2 = id_s.clone();
    let h1 = tokio::spawn(async move {
        reg_a
            .dispatch(
                &ctx_a,
                "set_speaker_name",
                serde_json::json!({ "meeting_id": id_s, "speaker_id": "A", "name": "Alice" }),
            )
            .await
    });
    let h2 = tokio::spawn(async move {
        reg_b
            .dispatch(
                &ctx_b,
                "set_speaker_name",
                serde_json::json!({ "meeting_id": id_s2, "speaker_id": "B", "name": "Bob" }),
            )
            .await
    });
    h1.await.unwrap().unwrap();
    h2.await.unwrap().unwrap();

    let dir = root.join(id.0.to_string());
    let meta = persistence::read_metadata(&dir).unwrap();
    assert_eq!(
        meta.speaker_names.get("A").map(String::as_str),
        Some("Alice")
    );
    assert_eq!(
        meta.speaker_names.get("B").map(String::as_str),
        Some("Bob"),
        "both names must survive — no dropped write"
    );
}

#[tokio::test]
async fn rename_meeting_updates_title_and_index() {
    let (_t, root, ctx) = make_ctx().await;
    let reg = ToolRegistry::v1(false);
    let id = seed_meeting(
        &root,
        &ctx.index,
        "Old title",
        vec![seg(0, 1000, "x", None)],
        BTreeMap::new(),
    )
    .await;

    reg.dispatch(
        &ctx,
        "rename_meeting",
        serde_json::json!({ "meeting_id": id.0.to_string(), "title": "New title" }),
    )
    .await
    .unwrap();

    let dir = root.join(id.0.to_string());
    assert_eq!(persistence::read_metadata(&dir).unwrap().title, "New title");

    // Index row reflects the new title.
    let found = ctx.index.search("New title").await.unwrap();
    assert_eq!(found.len(), 1);
    assert_eq!(found[0].title, "New title");
}

// ---------------------------------------------------------------------------
// S4: the metadata-write tools cap user-supplied free-text at dispatch.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn set_speaker_name_rejects_overlong_name() {
    let (_t, root, ctx) = make_ctx().await;
    let reg = ToolRegistry::v1(false);
    let id = seed_meeting(
        &root,
        &ctx.index,
        "M",
        vec![seg(0, 1000, "x", Some("A"))],
        BTreeMap::new(),
    )
    .await;

    // 513 chars > the 512-char cap.
    let overlong = "n".repeat(513);
    let err = reg
        .dispatch(
            &ctx,
            "set_speaker_name",
            serde_json::json!({ "meeting_id": id.0.to_string(), "speaker_id": "A", "name": overlong }),
        )
        .await
        .expect_err("an over-length name must be rejected");
    match err {
        minutist_common::AppError::InvalidInput { context } => {
            assert!(context.contains("too long"), "got: {context}");
        }
        other => panic!("expected InvalidInput, got {other:?}"),
    }

    // Nothing was persisted (the cap fires before the metadata write).
    let dir = root.join(id.0.to_string());
    let meta = persistence::read_metadata(&dir).unwrap();
    assert!(
        meta.speaker_names.is_empty(),
        "no speaker name may be persisted when the value is rejected"
    );
}

#[tokio::test]
async fn set_speaker_name_rejects_overlong_speaker_id() {
    let (_t, root, ctx) = make_ctx().await;
    let reg = ToolRegistry::v1(false);
    let id = seed_meeting(&root, &ctx.index, "M", vec![], BTreeMap::new()).await;

    let overlong = "A".repeat(513);
    let err = reg
        .dispatch(
            &ctx,
            "set_speaker_name",
            serde_json::json!({ "meeting_id": id.0.to_string(), "speaker_id": overlong, "name": "Alice" }),
        )
        .await
        .expect_err("an over-length speaker_id must be rejected");
    assert!(matches!(
        err,
        minutist_common::AppError::InvalidInput { .. }
    ));
}

#[tokio::test]
async fn rename_meeting_rejects_overlong_title() {
    let (_t, root, ctx) = make_ctx().await;
    let reg = ToolRegistry::v1(false);
    let id = seed_meeting(&root, &ctx.index, "Old title", vec![], BTreeMap::new()).await;

    let overlong = "t".repeat(513);
    let err = reg
        .dispatch(
            &ctx,
            "rename_meeting",
            serde_json::json!({ "meeting_id": id.0.to_string(), "title": overlong }),
        )
        .await
        .expect_err("an over-length title must be rejected");
    match err {
        minutist_common::AppError::InvalidInput { context } => {
            assert!(context.contains("too long"), "got: {context}");
        }
        other => panic!("expected InvalidInput, got {other:?}"),
    }

    // The title is unchanged on disk (the cap fires before the rename write).
    let dir = root.join(id.0.to_string());
    assert_eq!(persistence::read_metadata(&dir).unwrap().title, "Old title");
}

#[tokio::test]
async fn set_speaker_name_accepts_max_length_name() {
    // A name AT the cap (512 chars) is accepted — the boundary is inclusive.
    let (_t, root, ctx) = make_ctx().await;
    let reg = ToolRegistry::v1(false);
    let id = seed_meeting(&root, &ctx.index, "M", vec![], BTreeMap::new()).await;

    let at_cap = "n".repeat(512);
    reg.dispatch(
        &ctx,
        "set_speaker_name",
        serde_json::json!({ "meeting_id": id.0.to_string(), "speaker_id": "A", "name": at_cap.clone() }),
    )
    .await
    .expect("a name at the 512-char cap must be accepted");

    let dir = root.join(id.0.to_string());
    let meta = persistence::read_metadata(&dir).unwrap();
    assert_eq!(meta.speaker_names.get("A"), Some(&at_cap));
}

// ---------------------------------------------------------------------------
// send_to_internal_agent (the MCP-only inter-agent bridge tool) — error paths
//
// The happy path (a real reply) needs the chat engine, which lives in
// ipc-bridge; here we cover the bridge tool's OWN failure handling against a
// stub channel, with no chat turn: unavailable-when-no-bridge, busy-on-full,
// and timed-out-when-no-reply.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn send_to_internal_agent_unavailable_without_bridge() {
    // The default context (UI path) has no bridge → the tool is unavailable.
    let (_t, _root, ctx) = make_ctx().await;
    let reg = ToolRegistry::v1(true);
    let err = reg
        .dispatch(
            &ctx,
            "send_to_internal_agent",
            serde_json::json!({ "message": "hi" }),
        )
        .await
        .expect_err("no bridge → error");
    match err {
        minutist_common::AppError::InvalidInput { context } => {
            assert!(context.contains("not available"), "got: {context}");
        }
        other => panic!("expected InvalidInput, got {other:?}"),
    }
}

#[tokio::test]
async fn send_to_internal_agent_busy_on_full_queue() {
    let (_t, _root, ctx) = make_ctx().await;
    // A depth-1 bridge channel with NO receiver draining it: the first send
    // fills it, so the tool's try_send sees a full queue → "internal agent busy".
    let (tx, _rx) = tokio::sync::mpsc::channel(1);
    // Pre-fill the single slot so the tool's try_send fails as "Full".
    let (pre_reply_tx, _pre_reply_rx) = tokio::sync::oneshot::channel();
    tx.try_send((
        minutist_common::InterAgentRequest {
            session_id: None,
            meeting_id: None,
            message: "prefill".into(),
        },
        pre_reply_tx,
    ))
    .expect("prefill the queue");

    let ctx = ctx.with_inter_agent_bridge(tx);
    let reg = ToolRegistry::v1(true);
    let err = reg
        .dispatch(
            &ctx,
            "send_to_internal_agent",
            serde_json::json!({ "message": "hi" }),
        )
        .await
        .expect_err("full queue → busy");
    match err {
        minutist_common::AppError::InvalidInput { context } => {
            assert!(context.contains("busy"), "got: {context}");
        }
        other => panic!("expected InvalidInput busy, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Record-control tools (#62) — dispatch through the registry
//
// `start_recording` dispatches to `Orchestrator::start`, which opens a real
// audio device — impractical in a headless unit test — so the registry-shape +
// MCP-gate coverage above is the primary guard for it. The stop/pause/resume
// tools are dispatched here against an idle test orchestrator: they each reach
// the orchestrator and surface its state-machine rejection (`InvalidInput` when
// not in a state that permits the transition), proving the registry plumbs the
// internal-UI (no MCP gate) call straight through to the orchestrator method.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn stop_recording_dispatches_to_orchestrator_and_rejects_when_idle() {
    let (_t, _root, ctx) = make_ctx().await;
    let reg = ToolRegistry::v1(false);
    // The internal-UI path (mcp_gate None) dispatches directly through the
    // registry — no write gate. An idle orchestrator has nothing to stop, so the
    // state machine rejects it with InvalidInput (proving the call reached it).
    let err = reg
        .dispatch(&ctx, "stop_recording", serde_json::json!({}))
        .await
        .expect_err("stop while idle must be rejected by the orchestrator");
    assert!(matches!(
        err,
        minutist_common::AppError::InvalidInput { .. }
    ));
}

#[tokio::test]
async fn pause_and_resume_recording_dispatch_to_orchestrator_and_reject_when_idle() {
    let (_t, _root, ctx) = make_ctx().await;
    let reg = ToolRegistry::v1(false);

    let pause_err = reg
        .dispatch(&ctx, "pause_recording", serde_json::json!({}))
        .await
        .expect_err("pause while idle must be rejected");
    assert!(matches!(
        pause_err,
        minutist_common::AppError::InvalidInput { .. }
    ));

    let resume_err = reg
        .dispatch(&ctx, "resume_recording", serde_json::json!({}))
        .await
        .expect_err("resume while idle must be rejected");
    assert!(matches!(
        resume_err,
        minutist_common::AppError::InvalidInput { .. }
    ));
}

#[tokio::test]
async fn start_then_stop_recording_via_registry_round_trips() {
    // A full start→stop dispatch through the registry, using the orchestrator's
    // `test-source` seam to start without a real device, then stopping via the
    // `stop_recording` tool. This exercises the tool's dispatch + the
    // orchestrator's stop method end-to-end (no model present → empty transcript,
    // recording still finalises). `start_recording` itself opens a real device so
    // it cannot be the start seam here; we start the session via the test seam and
    // assert the tool can stop it.
    use audio_capture::test_source::DummyAudioSource;

    let (_t, _root, ctx) = make_ctx().await;
    let reg = ToolRegistry::v1(false);

    // Start a recording through the test-source seam (no real microphone).
    let source = DummyAudioSource::new(1600, 800);
    let streams = source.generate_streams(4, 32, 64);
    let started_id = ctx
        .orchestrator
        .start_with_streams(streams)
        .await
        .expect("test-source start should succeed");

    // Now stop it via the registry tool — the internal-UI dispatch path.
    let out = reg
        .dispatch(&ctx, "stop_recording", serde_json::json!({}))
        .await
        .expect("stop_recording must finalise the running meeting");
    let stopped_id: MeetingId =
        serde_json::from_value(out.data["meeting_id"].clone()).expect("meeting_id in result");
    assert_eq!(
        stopped_id, started_id,
        "stop_recording returns the finished meeting's id"
    );
    assert!(
        out.data.get("duration_ms").is_some(),
        "stop_recording returns the meeting duration"
    );
}
