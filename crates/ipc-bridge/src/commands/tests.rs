use super::*;
use minutist_common::{MeetingId, NoteBlock};
use notes_crdt::MeetingFolder;
use persistence::MeetingIndex;
use tempfile::TempDir;

/// `save_notes` → `load_notes` round-trip through a tempdir `meetings_dir`,
/// exercising the command bodies directly (no Tauri runtime needed).
#[test]
fn save_then_load_round_trips_through_meetings_dir() {
    let tempdir = TempDir::new().expect("tempdir");
    let root = tempdir.path();
    let meeting_id = MeetingId::new();
    // NotesStore writes into an *existing* meeting folder; create it via
    // the owning type so the layout matches production exactly.
    MeetingFolder::create(root, meeting_id).expect("create meeting folder");

    let notes_json = r#"{"type":"doc","content":[{"type":"paragraph","attrs":{"data-anchor-ms":1234},"content":[{"type":"text","text":"hello"}]}]}"#;
    let notes_markdown = "# Notes\n\nhello\n";

    save_notes_inner(root, meeting_id, notes_json, notes_markdown).expect("save");

    let loaded = load_notes_inner(root, meeting_id)
        .expect("load")
        .expect("notes present after save");

    // The markdown round-trips verbatim.
    assert_eq!(loaded.notes_markdown, notes_markdown);
    // The JSON round-trips structurally (re-serialised string may differ in
    // whitespace, so compare parsed values).
    let expected: serde_json::Value = serde_json::from_str(notes_json).unwrap();
    let actual: serde_json::Value = serde_json::from_str(&loaded.notes_json).unwrap();
    assert_eq!(actual, expected, "notes_json must round-trip structurally");
}

/// `load_notes` returns `None` when no notes have been saved for a meeting.
#[test]
fn load_returns_none_when_no_notes_saved() {
    let tempdir = TempDir::new().expect("tempdir");
    let meeting_id = MeetingId::new();
    let loaded = load_notes_inner(tempdir.path(), meeting_id).expect("load");
    assert!(loaded.is_none(), "absent notes must yield None");
}

/// Invalid `notes_json` is rejected as `AppError::InvalidInput`, not written.
#[test]
fn save_rejects_invalid_json() {
    let tempdir = TempDir::new().expect("tempdir");
    let meeting_id = MeetingId::new();
    MeetingFolder::create(tempdir.path(), meeting_id).expect("folder");
    let err = save_notes_inner(tempdir.path(), meeting_id, "not json", "")
        .expect_err("invalid JSON must error");
    assert!(matches!(err, AppError::InvalidInput { .. }));
}

/// `save_note_image`'s extension allowlist gate (`normalise_image_ext`):
/// accepts the image set (case-/dot-insensitively), rejects everything else.
#[test]
fn normalise_image_ext_accepts_allowlist_rejects_others() {
    // Accepted, normalised to lower-cased, dot-less.
    for (input, expected) in [
        ("png", "png"),
        ("PNG", "png"),
        (".jpg", "jpg"),
        ("  .JPEG  ", "jpeg"),
        ("GIF", "gif"),
        ("webp", "webp"),
    ] {
        assert_eq!(
            normalise_image_ext(input).expect("allowed ext"),
            expected,
            "ext {input:?} should normalise to {expected:?}"
        );
    }
    // Rejected as InvalidInput — non-image / executable / path-y extensions.
    for evil in ["svg", "exe", "txt", "", "png.exe", "../png", "bmp"] {
        assert!(
            matches!(
                normalise_image_ext(evil),
                Err(AppError::InvalidInput { .. })
            ),
            "ext {evil:?} must be rejected"
        );
    }
}

#[test]
fn normalise_attachment_ext_accepts_supported_rejects_others() {
    // Every `doc_convert::supported_exts` value normalises (lower-cased,
    // dot-less) and round-trips.
    for ext in doc_convert::supported_exts() {
        let upper = ext.to_ascii_uppercase();
        assert_eq!(
            normalise_attachment_ext(&format!(".{upper}")).expect("supported ext"),
            *ext,
            "ext {ext:?} should normalise from a dotted upper-case form"
        );
    }
    // Rejected — executable / unsupported document extensions. Image
    // extensions (png/jpg/jpeg/tiff) and Office docx ARE supported now (the
    // VLM OCR fallback and the OOXML walk respectively), so they are
    // exercised by the supported-exts loop above; "doc" (legacy binary
    // Word) and "rtf" remain unsupported.
    for evil in ["exe", "", "pdf.exe", "../pdf", "rtf", "doc"] {
        assert!(
            matches!(
                normalise_attachment_ext(evil),
                Err(AppError::InvalidInput { .. })
            ),
            "ext {evil:?} must be rejected"
        );
    }
}

#[test]
fn attachments_budget_scales_with_context() {
    // Budget grows with n_ctx (40% of the window × ~4 chars/token).
    let small = attachments_markdown_budget_chars(8_192);
    let large = attachments_markdown_budget_chars(32_768);
    assert!(
        large > small,
        "a larger context window must allow more chars"
    );
    assert!(small > 0);
}

#[test]
fn assemble_attachments_empty_and_within_budget() {
    // No attachments → empty (byte-identical to the no-attachment path).
    assert_eq!(assemble_attachments_markdown(Vec::new(), 100), "");

    // Within budget → full assembly under per-attachment headers, no marker.
    let parts = vec![
        ("a.txt".to_string(), "hi".to_string()),
        ("b.txt".to_string(), "yo".to_string()),
    ];
    let out = assemble_attachments_markdown(parts, 1_000);
    assert!(out.contains("## Attachment: a.txt"));
    assert!(out.contains("## Attachment: b.txt"));
    assert!(
        !out.contains("[truncated]"),
        "within budget must not truncate"
    );
}

#[test]
fn assemble_attachments_equal_share_does_not_starve_later_parts() {
    // A huge first attachment must NOT consume the whole budget: every
    // attachment gets an equal share, the trimmed one is marked, and the
    // small later attachment survives in full (the whole-string truncation
    // this replaced would have dropped it entirely).
    let parts = vec![
        ("big.txt".to_string(), "x".repeat(5_000)),
        ("small.txt".to_string(), "kept".to_string()),
    ];
    let out = assemble_attachments_markdown(parts, 400);
    assert!(
        out.contains("## Attachment: big.txt"),
        "first header survives"
    );
    assert!(
        out.contains("## Attachment: small.txt"),
        "second header survives (not starved)"
    );
    assert!(
        out.contains("kept"),
        "the small attachment's body is retained"
    );
    assert!(
        out.contains("[truncated]"),
        "the trimmed large attachment is marked"
    );
}

#[test]
fn assemble_attachments_truncates_multibyte_body_on_char_boundary() {
    // A multibyte body over a tight budget must truncate on a char boundary
    // (`chars().take`), never mid-codepoint, and stay valid UTF-8 — a future
    // refactor to byte-slicing would break this while passing the ASCII case.
    let body = "é".repeat(500); // 500 chars, 1000 bytes
    let out = assemble_attachments_markdown(vec![("doc.md".to_string(), body)], 120);
    assert!(out.contains("[truncated]"), "expected truncation marker");
    assert!(
        !out.contains('\u{FFFD}'),
        "no replacement char — no split codepoint: {out:?}"
    );
    assert!(out.contains('é'), "multibyte content preserved");
    assert!(
        std::str::from_utf8(out.as_bytes()).is_ok(),
        "output must be valid UTF-8"
    );
}

#[test]
fn check_attachment_limits_rejects_overlong_filename_and_oversize_bytes() {
    // Within limits.
    assert!(check_attachment_limits("notes.pdf", 1024).is_ok());

    // Over-long filename (by char count) → InvalidInput.
    let long = "x".repeat(MAX_ATTACHMENT_FILENAME_LEN + 1);
    assert!(matches!(
        check_attachment_limits(&long, 1),
        Err(AppError::InvalidInput { .. })
    ));

    // Oversize bytes → InvalidInput.
    assert!(matches!(
        check_attachment_limits("ok.pdf", doc_convert::MAX_INPUT_BYTES + 1),
        Err(AppError::InvalidInput { .. })
    ));

    // Exactly at the caps is allowed (the production checks are strict `>`).
    let max_name = "x".repeat(MAX_ATTACHMENT_FILENAME_LEN);
    assert!(check_attachment_limits(&max_name, doc_convert::MAX_INPUT_BYTES).is_ok());
}

/// `save_note_asset` (the persistence body the command calls) round-trips an
/// image and returns a portable bare-filename reference.
#[test]
fn save_note_asset_round_trips_via_meetings_dir() {
    let tempdir = TempDir::new().expect("tempdir");
    let root = tempdir.path();
    let meeting_id = MeetingId::new();
    MeetingFolder::create(root, meeting_id).expect("folder");

    let bytes = b"\x89PNG\r\n\x1a\n-image-bytes".to_vec();
    let filename = persistence::save_note_asset(root, meeting_id, &bytes, "png").expect("save");
    // Portable ref: a bare filename, no separators.
    assert!(!filename.contains('/') && !filename.contains('\\'));
    assert!(filename.ends_with(".png"));

    let read = persistence::read_note_asset(root, meeting_id, &filename).expect("read");
    assert_eq!(read, bytes);
}

// -----------------------------------------------------------------------
// Phase 4 meeting list/open/rename/delete round-trips (no Tauri runtime,
// no model — a synthetic meeting folder + in-memory libsql index).
// -----------------------------------------------------------------------

use minutist_common::{AudioFormat, MeetingMeta, Segment};

/// Write a synthetic meeting folder (`metadata.json` + optional
/// `transcript.json`) under `root` and return its `MeetingId`. Mirrors the
/// on-disk layout `persistence` produces so the readers + index agree.
fn write_synthetic_meeting(
    root: &Path,
    title: &str,
    started_at: &str,
    first_segment_text: Option<&str>,
) -> MeetingId {
    let meeting_id = MeetingId::new();
    let folder = MeetingFolder::create(root, meeting_id).expect("create meeting folder");

    let meta = MeetingMeta {
        uuid: meeting_id,
        title: title.to_string(),
        started_at: started_at.to_string(),
        ended_at: Some(started_at.to_string()),
        duration_ms: 60_000,
        speaker_count: 1,
        audio_format: AudioFormat {
            codec: "opus".into(),
            sample_rate: 16_000,
            channels: 1,
            bitrate_kbps: Some(32),
        },
        asr_model: None,
        llm_model: None,
        diarizer: None,
        speaker_names: std::collections::BTreeMap::new(),
        notes_format: 0,
        processing: Default::default(),
        collection_id: None,
        recording_started: true,
        app_version: "0.0.0".into(),
    };
    let meta_json = serde_json::to_vec_pretty(&meta).expect("serialise metadata");
    std::fs::write(folder.metadata_path(), meta_json).expect("write metadata.json");

    if let Some(text) = first_segment_text {
        let segments = vec![Segment {
            start_ms: 0,
            end_ms: 1_000,
            text: text.to_string(),
            speaker_id: None,
            confidence: None,
            words: Vec::new(),
            shared_speakers: Vec::new(),
        }];
        let seg_json = serde_json::to_vec_pretty(&segments).expect("serialise transcript");
        std::fs::write(folder.transcript_path(), seg_json).expect("write transcript.json");
    }

    meeting_id
}

/// Open an in-memory index seeded by rebuilding from the meeting folders.
async fn seeded_index(meetings_root: &Path) -> MeetingIndex {
    let index = MeetingIndex::open(":memory:")
        .await
        .expect("open in-memory index");
    index
        .rebuild_from_disk(meetings_root)
        .await
        .expect("rebuild index from disk");
    index
}

/// `list_meetings` returns every indexed meeting, most-recent first, with the
/// first transcript segment as the excerpt.
#[tokio::test]
async fn list_meetings_returns_indexed_rows_most_recent_first() {
    let tempdir = TempDir::new().expect("tempdir");
    let root = tempdir.path();

    let _older = write_synthetic_meeting(
        root,
        "Older meeting",
        "2026-06-01T09:00:00Z",
        Some("older excerpt"),
    );
    let _newer = write_synthetic_meeting(
        root,
        "Newer meeting",
        "2026-06-02T09:00:00Z",
        Some("newer excerpt"),
    );

    let index = seeded_index(root).await;
    let rows = index.list_meetings().await.expect("list_meetings");

    assert_eq!(rows.len(), 2, "both meetings must be indexed");
    assert_eq!(rows[0].title, "Newer meeting", "most-recent first");
    assert_eq!(rows[0].excerpt.as_deref(), Some("newer excerpt"));
    assert_eq!(rows[1].title, "Older meeting");
}

/// `open_meeting_inner` assembles a `MeetingState` matching what was written
/// to the synthetic folder (metadata + transcript; no notes saved → None).
#[test]
fn open_meeting_returns_meeting_state_matching_disk() {
    let tempdir = TempDir::new().expect("tempdir");
    let root = tempdir.path();
    let meeting_id = write_synthetic_meeting(
        root,
        "Launch sync",
        "2026-06-02T10:00:00Z",
        Some("hello world"),
    );

    let state = open_meeting_inner(root, meeting_id).expect("open_meeting");

    assert_eq!(state.meta.uuid, meeting_id);
    assert_eq!(state.meta.title, "Launch sync");
    assert_eq!(state.transcript.len(), 1);
    assert_eq!(state.transcript[0].text, "hello world");
    assert!(state.notes.is_none(), "no notes saved → None");
}

/// `open_meeting_inner` errors for a meeting folder that does not exist.
#[test]
fn open_meeting_errors_for_missing_meeting() {
    let tempdir = TempDir::new().expect("tempdir");
    let missing = MeetingId::new();
    let err =
        open_meeting_inner(tempdir.path(), missing).expect_err("missing meeting must error");
    assert!(matches!(err, AppError::Io { .. }));
}

/// `rename_meeting` rewrites `metadata.json` and refreshes the index row so a
/// subsequent `list_meetings` shows the new title.
#[tokio::test]
async fn rename_meeting_updates_disk_and_index() {
    let tempdir = TempDir::new().expect("tempdir");
    let root = tempdir.path();
    let meeting_id =
        write_synthetic_meeting(root, "Old title", "2026-06-02T11:00:00Z", Some("excerpt"));

    let index = seeded_index(root).await;
    meeting_ops::rename_meeting(root, &index, meeting_id, "New title")
        .await
        .expect("rename");

    // On-disk metadata reflects the new title.
    let meeting_dir = root.join(meeting_id.0.to_string());
    let meta = persistence::read_metadata(&meeting_dir).expect("read metadata");
    assert_eq!(meta.title, "New title");

    // Index row reflects the new title.
    let rows = index.list_meetings().await.expect("list");
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].title, "New title");
}

/// `set_speaker_name` writes the label→name mapping into `metadata.json` and
/// returns the updated map; clearing with an empty name removes the entry.
/// The index is not touched (speaker names live only in `metadata.json`).
#[tokio::test]
async fn set_speaker_name_persists_and_clears() {
    let tempdir = TempDir::new().expect("tempdir");
    let root = tempdir.path();
    let meeting_id = write_synthetic_meeting(root, "Speakers", "2026-06-14T10:00:00Z", None);

    // Upsert label "A" → "Alice".
    let names = meeting_ops::set_speaker_name(root, meeting_id, "A", "Alice")
        .await
        .expect("set A");
    assert_eq!(names.get("A").map(String::as_str), Some("Alice"));

    // Confirm on-disk metadata reflects the name.
    let folder = root.join(meeting_id.0.to_string());
    let meta = persistence::read_metadata(&folder).expect("read metadata");
    assert_eq!(
        meta.speaker_names.get("A").map(String::as_str),
        Some("Alice")
    );

    // Add a second speaker name without clobbering the first.
    let names2 = meeting_ops::set_speaker_name(root, meeting_id, "B", "Bob")
        .await
        .expect("set B");
    assert_eq!(names2.get("A").map(String::as_str), Some("Alice"));
    assert_eq!(names2.get("B").map(String::as_str), Some("Bob"));

    // Clear "A" with an empty name.
    let names3 = meeting_ops::set_speaker_name(root, meeting_id, "A", "")
        .await
        .expect("clear A");
    assert!(!names3.contains_key("A"));
    assert_eq!(names3.get("B").map(String::as_str), Some("Bob"));
}

/// Stop-upsert (FR-33): after a meeting folder is written and its index row
/// is upserted (the stop-equivalent the `stop_recording` command performs),
/// `list_meetings` returns it **in the same session** — without a
/// `rebuild_from_disk`. This is the in-session visibility guarantee that
/// `Orchestrator::stop` alone does not provide (it finalises the folder but
/// never touches the index).
#[tokio::test]
async fn stop_upsert_makes_meeting_visible_in_session() {
    let tempdir = TempDir::new().expect("tempdir");
    let root = tempdir.path();

    // A fresh, EMPTY in-memory index — no rebuild_from_disk. This models a
    // running session where the index was opened at startup and the meeting
    // recorded afterwards.
    let index = MeetingIndex::open(":memory:")
        .await
        .expect("open in-memory index");
    assert!(
        index.list_meetings().await.expect("list").is_empty(),
        "index must start empty (no rebuild)"
    );

    // Write a meeting folder + transcript, exactly as a finished recording
    // leaves on disk.
    let meeting_id = write_synthetic_meeting(
        root,
        "In-session meeting",
        "2026-06-02T13:00:00Z",
        Some("first words of the meeting"),
    );

    // The stop-equivalent: build the list entry from metadata + first
    // transcript segment, then upsert into the live index — exactly what the
    // `stop_recording` command does after `orchestrator.stop()`.
    let meta = persistence::read_metadata(&root.join(meeting_id.0.to_string()))
        .expect("read metadata");
    let entry = meeting_list_entry_for_meta(root, &meta);
    index.upsert(&entry).await.expect("upsert after stop");

    // list_meetings now returns the meeting in the SAME session.
    let rows = index.list_meetings().await.expect("list after upsert");
    assert_eq!(
        rows.len(),
        1,
        "stopped meeting must be visible without a rebuild"
    );
    assert_eq!(rows[0].id, meeting_id);
    assert_eq!(rows[0].title, "In-session meeting");
    assert_eq!(
        rows[0].excerpt.as_deref(),
        Some("first words of the meeting"),
        "excerpt must be the first transcript segment"
    );
}

/// `meeting_list_entry_for_meta` yields `excerpt: None` when the meeting has
/// no transcript (a zero-segment meeting writes no `transcript.json`).
#[test]
fn stop_upsert_entry_has_no_excerpt_without_transcript() {
    let tempdir = TempDir::new().expect("tempdir");
    let root = tempdir.path();
    let meeting_id =
        write_synthetic_meeting(root, "Silent meeting", "2026-06-02T14:00:00Z", None);

    let meta = persistence::read_metadata(&root.join(meeting_id.0.to_string()))
        .expect("read metadata");
    let entry = meeting_list_entry_for_meta(root, &meta);

    assert_eq!(entry.id, meeting_id);
    assert_eq!(entry.title, "Silent meeting");
    assert!(entry.excerpt.is_none(), "no transcript → excerpt None");
}

/// `delete_meeting` removes the on-disk folder and the index row.
#[tokio::test]
async fn delete_meeting_removes_folder_and_index_row() {
    let tempdir = TempDir::new().expect("tempdir");
    let root = tempdir.path();
    let meeting_id =
        write_synthetic_meeting(root, "Doomed", "2026-06-02T12:00:00Z", Some("excerpt"));

    let index = seeded_index(root).await;
    assert_eq!(index.list_meetings().await.expect("list").len(), 1);

    meeting_ops::delete_meeting(root, &index, meeting_id)
        .await
        .expect("delete");

    let meeting_dir = root.join(meeting_id.0.to_string());
    assert!(!meeting_dir.exists(), "folder must be removed");
    assert!(
        index.list_meetings().await.expect("list").is_empty(),
        "index row must be removed"
    );
}

// -----------------------------------------------------------------------
// Phase 5 summary wiring (no model, no Tauri runtime). The summarise inner
// path is driven by a `StubSummariser` so the read → summarise → write →
// event wiring is exercised in CI without a ~3 GB GGUF — mirroring the
// orchestrator's re_transcribe stub-backend seam.
// -----------------------------------------------------------------------
// `Summariser` is brought in via `use super::*` (the module-level
// `#[cfg(test)]` import); no separate `use` needed here.

/// A `common::Summariser` that returns a fixed markdown, recording the
/// transcript length, notes markdown, and system prompt it was handed so the
/// test can assert the inner path forwarded them.
struct StubSummariser {
    fixed_markdown: String,
    seen_transcript_len: std::sync::Mutex<Option<usize>>,
    seen_speaker_ids: std::sync::Mutex<Option<Vec<Option<String>>>>,
    seen_notes: std::sync::Mutex<Option<String>>,
    seen_prompt: std::sync::Mutex<Option<String>>,
}

impl StubSummariser {
    fn new(markdown: &str) -> Self {
        Self {
            fixed_markdown: markdown.to_string(),
            seen_transcript_len: std::sync::Mutex::new(None),
            seen_speaker_ids: std::sync::Mutex::new(None),
            seen_notes: std::sync::Mutex::new(None),
            seen_prompt: std::sync::Mutex::new(None),
        }
    }
}

impl Summariser for StubSummariser {
    fn summarise(
        &self,
        transcript: &[Segment],
        notes: &[NoteBlock],
        _attachments_markdown: &str,
        system_prompt: &str,
    ) -> Result<String, AppError> {
        *self.seen_transcript_len.lock().unwrap() = Some(transcript.len());
        // Capture the per-segment speaker labels the inner path handed us so
        // a test can assert the speaker-name overlay was applied.
        *self.seen_speaker_ids.lock().unwrap() =
            Some(transcript.iter().map(|s| s.speaker_id.clone()).collect());
        // Capture the note text the inner path read (joined in document
        // order) — empty string when no notes were taken.
        let joined = notes
            .iter()
            .map(|n| n.text.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        *self.seen_notes.lock().unwrap() = Some(joined);
        *self.seen_prompt.lock().unwrap() = Some(system_prompt.to_string());
        Ok(self.fixed_markdown.clone())
    }
}

/// Save notes for a synthetic meeting as a Tiptap document with ONE paragraph
/// per line of `text`, so [`persistence::read_note_blocks`] (which projects
/// the `notes.json` paragraphs, #70) yields those lines as un-anchored
/// [`NoteBlock`]s. Uses the same `NotesStore` path the `save_notes` command
/// uses.
fn write_synthetic_notes(root: &Path, meeting_id: MeetingId, text: &str) {
    let content: Vec<serde_json::Value> = text
        .lines()
        .map(|line| {
            serde_json::json!({
                "type": "paragraph",
                "content": [{ "type": "text", "text": line }],
            })
        })
        .collect();
    let value = serde_json::json!({ "type": "doc", "content": content });
    NotesStore::save(root, meeting_id, &value, text).expect("save notes");
}

/// `summarise_meeting_inner` reads the meeting's transcript + notes markdown,
/// runs the stub summariser, writes `summary.md`, and (separately) the
/// command emits `SummaryReady`. Here we assert the inner write + the event
/// emission, since the inner fn and the emit helper compose what the command
/// does without needing a model or Tauri runtime.
#[test]
fn summarise_inner_reads_writes_and_returns_markdown() {
    let tempdir = TempDir::new().expect("tempdir");
    let root = tempdir.path();
    let meeting_id = write_synthetic_meeting(
        root,
        "Planning sync",
        "2026-06-02T15:00:00Z",
        Some("first agenda item"),
    );
    write_synthetic_notes(root, meeting_id, "- own the resume bug");

    let stub = StubSummariser::new("## Summary\n\nWe planned the sprint.\n");
    let prompt = "You are a meeting-notes assistant.";

    let returned = summarise_meeting_inner(root, meeting_id, &stub, prompt)
        .expect("summarise inner must succeed");

    // The returned markdown is the stub's fixed output.
    assert_eq!(returned, "## Summary\n\nWe planned the sprint.\n");

    // The stub saw the transcript + notes + prompt the inner path read.
    assert_eq!(*stub.seen_transcript_len.lock().unwrap(), Some(1));
    assert_eq!(
        stub.seen_notes.lock().unwrap().as_deref(),
        Some("- own the resume bug")
    );
    assert_eq!(stub.seen_prompt.lock().unwrap().as_deref(), Some(prompt));

    // `summary.md` is persisted and readable via the get-summary inner path.
    let loaded = get_summary_inner(root, meeting_id).expect("read summary");
    assert_eq!(
        loaded.as_deref(),
        Some("## Summary\n\nWe planned the sprint.\n"),
        "summary.md must be written by the inner path"
    );
}

/// The summariser must see user-set speaker names, not the raw diarizer
/// labels: `summarise_meeting_inner` overlays `metadata.speaker_names` onto a
/// transcript copy before handing it to the summariser, while the on-disk
/// transcript keeps its raw labels.
#[test]
fn summarise_inner_overlays_speaker_names() {
    let tempdir = TempDir::new().expect("tempdir");
    let root = tempdir.path();
    let meeting_id =
        write_synthetic_meeting(root, "Sync", "2026-06-12T15:00:00Z", Some("hello"));
    let dir = root.join(meeting_id.0.to_string());

    // Label the single segment "A" and map "A" -> "Alice".
    let mut transcript = persistence::read_transcript(&dir).expect("read transcript");
    transcript[0].speaker_id = Some("A".to_string());
    persistence::write_transcript(&dir, &transcript).expect("write transcript");
    let mut meta = persistence::read_metadata(&dir).expect("read metadata");
    meta.speaker_names
        .insert("A".to_string(), "Alice".to_string());
    notes_crdt::write_metadata(&dir, &meta).expect("write metadata");

    let stub = StubSummariser::new("ok");
    summarise_meeting_inner(root, meeting_id, &stub, "p").expect("summarise");

    // The stub saw the overlaid name, not the raw label.
    assert_eq!(
        stub.seen_speaker_ids.lock().unwrap().clone(),
        Some(vec![Some("Alice".to_string())]),
        "the overlay must rewrite the segment label to the display name"
    );
    // The on-disk transcript is untouched — still the raw label.
    let on_disk = persistence::read_transcript(&dir).expect("re-read transcript");
    assert_eq!(
        on_disk[0].speaker_id.as_deref(),
        Some("A"),
        "the overlay must not mutate the stored transcript"
    );
}

/// The full `summarise_meeting` wiring sans Tauri: inner write + the same
/// `SummaryReady` event the command emits, observed on a broadcast
/// subscriber — proving the event carries the right `meeting_id`.
#[tokio::test]
async fn summarise_emits_summary_ready_for_meeting() {
    let tempdir = TempDir::new().expect("tempdir");
    let root = tempdir.path();
    let meeting_id = write_synthetic_meeting(
        root,
        "Standup",
        "2026-06-02T16:00:00Z",
        Some("status update"),
    );

    let (event_tx, mut event_rx) = broadcast::channel::<AppEvent>(8);
    let stub = StubSummariser::new("## Summary\n\nStandup notes.\n");

    let returned =
        summarise_meeting_inner(root, meeting_id, &stub, "prompt").expect("summarise");
    assert_eq!(returned, "## Summary\n\nStandup notes.\n");

    emit_summary_ready(&event_tx, meeting_id);

    let event = event_rx.recv().await.expect("an event must be broadcast");
    match event {
        AppEvent::SummaryReady { meeting_id: got } => assert_eq!(got, meeting_id),
        other => panic!("expected SummaryReady, got {other:?}"),
    }
}

/// #NN — the post-stop auto-summary lifecycle markers carry the right
/// `meeting_id`. `SummaryQueued` is broadcast when a stop plans an
/// auto-summary (so the pane shows busy immediately); `SummaryUnavailable`
/// is the terminal the deferred/failed path emits so the busy state clears.
#[tokio::test]
async fn summary_queued_and_unavailable_carry_the_meeting_id() {
    let meeting_id = MeetingId::new();
    let (event_tx, mut event_rx) = broadcast::channel::<AppEvent>(8);

    emit_summary_queued(&event_tx, meeting_id);
    match event_rx.recv().await.expect("queued event") {
        AppEvent::SummaryQueued { meeting_id: got } => assert_eq!(got, meeting_id),
        other => panic!("expected SummaryQueued, got {other:?}"),
    }

    emit_summary_unavailable(&event_tx, meeting_id);
    match event_rx.recv().await.expect("unavailable event") {
        AppEvent::SummaryUnavailable { meeting_id: got } => assert_eq!(got, meeting_id),
        other => panic!("expected SummaryUnavailable, got {other:?}"),
    }
}

/// Notes-free meeting: the inner path passes an empty notes markdown rather
/// than erroring (FR-30 — a meeting with no notes still summarises).
#[test]
fn summarise_inner_handles_meeting_without_notes() {
    let tempdir = TempDir::new().expect("tempdir");
    let root = tempdir.path();
    let meeting_id =
        write_synthetic_meeting(root, "Quiet", "2026-06-02T17:00:00Z", Some("only words"));
    // No notes saved.

    let stub = StubSummariser::new("## Summary\n");
    summarise_meeting_inner(root, meeting_id, &stub, "prompt").expect("summarise");

    assert_eq!(
        stub.seen_notes.lock().unwrap().as_deref(),
        Some(""),
        "absent notes must pass an empty markdown string"
    );
}

/// `save_summary` → `get_summary` round-trip over a tempdir, exercising the
/// command bodies directly (no Tauri runtime).
#[test]
fn save_then_get_summary_round_trips() {
    let tempdir = TempDir::new().expect("tempdir");
    let root = tempdir.path();
    let meeting_id =
        write_synthetic_meeting(root, "Edited", "2026-06-02T18:00:00Z", Some("words"));

    // No summary yet → None.
    assert!(
        get_summary_inner(root, meeting_id).expect("read").is_none(),
        "absent summary must read as None"
    );

    let edited = "## Summary\n\nUser-edited summary.\n";
    save_summary_inner(root, meeting_id, edited).expect("save summary");

    let loaded = get_summary_inner(root, meeting_id).expect("read summary");
    assert_eq!(loaded.as_deref(), Some(edited), "summary must round-trip");
}

/// `find_gguf_weights` picks the single text-weights `.gguf`, skipping an
/// `mmproj-*` projector that may sit alongside it.
#[test]
fn find_gguf_weights_skips_mmproj() {
    let tempdir = TempDir::new().expect("tempdir");
    let dir = tempdir.path();
    std::fs::write(dir.join("gemma-4-E4B-it-Q4_K_M.gguf"), b"weights").expect("write weights");
    std::fs::write(dir.join("mmproj-gemma-4-E4B-it.gguf"), b"proj").expect("write proj");
    std::fs::write(dir.join("README.md"), b"notes").expect("write readme");

    let found = find_gguf_weights(dir).expect("must find the text weights");
    assert_eq!(
        found.file_name().and_then(|n| n.to_str()),
        Some("gemma-4-E4B-it-Q4_K_M.gguf")
    );
}

/// `find_gguf_weights` errors (rather than panicking) when no weights file
/// is present.
#[test]
fn find_gguf_weights_errors_when_absent() {
    let tempdir = TempDir::new().expect("tempdir");
    let dir = tempdir.path();
    std::fs::write(dir.join("mmproj-only.gguf"), b"proj").expect("write proj");

    let err = find_gguf_weights(dir).expect_err("no text weights → error");
    assert!(matches!(err, AppError::ModelLoad { .. }));
}

// -----------------------------------------------------------------------
// LLM model-id resolution (Phase 5) — the settings-override / bundled
// default decision, unit-tested without a Tauri runtime or orchestrator.
// -----------------------------------------------------------------------

/// A set `settings.llm_model_id` resolves to that id (the user override
/// wins over the bundled default).
#[test]
fn resolve_llm_model_id_honours_settings_override() {
    let settings = Settings {
        llm_model_id: Some(ModelId::from("granite-4.1-3b-q4_k_m")),
        ..Settings::default()
    };
    assert_eq!(
        resolve_llm_model_id(&settings),
        ModelId::from("granite-4.1-3b-q4_k_m")
    );
}

/// GPU off (`gpu_acceleration = false`) MUST force CPU (`0`); GPU on MUST
/// resolve to the compile-time ceiling — itself `0` in the default CPU-only
/// build, so a CPU build is unaffected by the flag. Pure, no model.
#[test]
fn resolve_summariser_gpu_layers_off_forces_cpu() {
    assert_eq!(
        resolve_summariser_gpu_layers(false),
        0,
        "GPU off must force CPU"
    );

    let on = resolve_summariser_gpu_layers(true);
    assert_eq!(
        on,
        summariser::gpu_layers(),
        "GPU on must use the compile-time ceiling"
    );
    if cfg!(any(
        feature = "vulkan",
        feature = "metal",
        feature = "cuda",
        feature = "rocm"
    )) {
        assert_eq!(
            on,
            u32::MAX,
            "a GPU-feature build offloads all layers when on"
        );
    } else {
        assert_eq!(on, 0, "a default CPU-only build stays on CPU even when on");
    }
}

/// An unset `settings.llm_model_id` falls back to the bundled default.
#[test]
fn resolve_llm_model_id_falls_back_to_default() {
    let settings = Settings {
        llm_model_id: None,
        ..Settings::default()
    };
    assert_eq!(
        resolve_llm_model_id(&settings),
        ModelId::from(DEFAULT_LLM_MODEL_ID)
    );
}

// -----------------------------------------------------------------------
// Gated real-model test — skips when MINUTIST_LLM_MODEL_PATH is unset.
//
// To run:
//   MINUTIST_LLM_MODEL_PATH=/path/to/gemma-4-E4B-it-Q4_K_M.gguf \
//   cargo test -p ipc-bridge -- --include-ignored
// -----------------------------------------------------------------------

/// End-to-end summarise over a synthetic meeting folder using the **real**
/// Gemma-4 GGUF pointed to by `MINUTIST_LLM_MODEL_PATH`: open the model,
/// run `summarise_meeting_inner`, assert a non-empty markdown summary is
/// written, and record latency. No-op skip when the env var is unset.
#[test]
#[ignore = "requires MINUTIST_LLM_MODEL_PATH"]
fn summarise_real_model_writes_non_empty_summary() {
    let model_path = match std::env::var("MINUTIST_LLM_MODEL_PATH") {
        Ok(p) => p,
        Err(_) => return, // no-op skip path
    };

    let tempdir = TempDir::new().expect("tempdir");
    let root = tempdir.path();
    let meeting_id = write_synthetic_meeting(
        root,
        "Gated meeting",
        "2026-06-02T19:00:00Z",
        Some("Let's review the quarterly plan and assign action items."),
    );
    write_synthetic_notes(root, meeting_id, "- Decision: ship Phase 5");

    let summariser = LlamaSummariser::open(
        std::path::PathBuf::from(&model_path),
        SummariserConfig::default(),
    )
    .expect("model load must succeed with a valid path");

    let prompt =
        "You are a meeting-notes assistant. Produce a concise markdown summary with headings.";

    let start = std::time::Instant::now();
    let summary = summarise_meeting_inner(root, meeting_id, &summariser, prompt)
        .expect("summarise must succeed");
    let elapsed = start.elapsed();

    tracing::info!(
        target: "ipc-bridge",
        elapsed_ms = elapsed.as_millis() as u64,
        summary_len = summary.len(),
        "gated summarise_meeting complete"
    );

    assert!(!summary.trim().is_empty(), "summary must be non-empty");

    // `summary.md` must be on disk and match what was returned.
    let loaded = get_summary_inner(root, meeting_id)
        .expect("read summary")
        .expect("summary.md must exist after summarise");
    assert_eq!(loaded, summary, "persisted summary must match returned");
}

/// End-to-end translation pass over a 3-segment meeting using the real
/// `MINUTIST_LLM_MODEL_PATH` model: verifies that all three translations are
/// written to `translations.json` after `translate_meeting_blocking` returns,
/// exercising the batched-flush path. Skipped in CI when the env var is unset.
#[test]
#[ignore = "requires MINUTIST_LLM_MODEL_PATH"]
fn translate_meeting_blocking_writes_all_segments_to_sidecar() {
    let model_path = match std::env::var("MINUTIST_LLM_MODEL_PATH") {
        Ok(p) => p,
        Err(_) => return,
    };

    let tempdir = TempDir::new().expect("tempdir");
    let root = tempdir.path();
    let meeting_id = MeetingId::new();
    let folder = notes_crdt::MeetingFolder::create(root, meeting_id).expect("folder");

    // Write a 3-segment transcript.
    let segments = vec![
        Segment {
            start_ms: 0,
            end_ms: 1000,
            text: "Hello world.".into(),
            speaker_id: None,
            confidence: None,
            words: vec![],
            shared_speakers: vec![],
        },
        Segment {
            start_ms: 1000,
            end_ms: 2000,
            text: "This is a test.".into(),
            speaker_id: None,
            confidence: None,
            words: vec![],
            shared_speakers: vec![],
        },
        Segment {
            start_ms: 2000,
            end_ms: 3000,
            text: "Goodbye for now.".into(),
            speaker_id: None,
            confidence: None,
            words: vec![],
            shared_speakers: vec![],
        },
    ];
    let seg_json = serde_json::to_vec_pretty(&segments).expect("serialise");
    std::fs::write(folder.transcript_path(), seg_json).expect("write transcript");

    let summariser = LlamaSummariser::open(
        std::path::PathBuf::from(&model_path),
        SummariserConfig::default(),
    )
    .expect("model load");

    let (event_tx, _event_rx) = broadcast::channel::<AppEvent>(8);
    let meetings_dir = root.to_path_buf();

    translate_meeting_blocking(&meetings_dir, meeting_id, "Spanish", &summariser, &event_tx)
        .expect("translation must succeed");

    // All three segments must be in the sidecar.
    let meeting_dir = root.join(meeting_id.0.to_string());
    let all = persistence::read_translations(&meeting_dir).expect("read translations");
    let spanish = all.get("Spanish").expect("Spanish key present");
    assert_eq!(spanish.len(), 3, "all 3 segments must be persisted");
    assert!(
        !spanish[&0].trim().is_empty(),
        "segment 0 must be non-empty"
    );
    assert!(
        !spanish[&1].trim().is_empty(),
        "segment 1 must be non-empty"
    );
    assert!(
        !spanish[&2].trim().is_empty(),
        "segment 2 must be non-empty"
    );
}

// -----------------------------------------------------------------------
// Post-stop background orchestration (review Step 5): gating + ordering +
// per-pass error tolerance, verified WITHOUT a Tauri runtime or a real
// orchestrator. `post_stop_passes` is pure; `run_post_stop_passes` is driven
// by a recording closure that injects per-pass results.
// -----------------------------------------------------------------------

/// Gating + ordering: no flags → no passes; a transcript-incomplete OR a
/// diarize-enabled flag adds the ONE reprocess pass (the merged op
/// re-transcribes then diarizes internally); all → reprocess BEFORE summarise
/// (so summarise sees the final transcript, #68).
#[test]
fn post_stop_passes_gates_and_orders() {
    assert_eq!(post_stop_passes(false, false, false), vec![]);
    // Either reprocess trigger alone collapses to the single pass.
    assert_eq!(
        post_stop_passes(true, false, false),
        vec![PostStopPass::Reprocess]
    );
    assert_eq!(
        post_stop_passes(false, true, false),
        vec![PostStopPass::Reprocess]
    );
    // Both triggers still yield exactly one reprocess pass (no duplicate).
    assert_eq!(
        post_stop_passes(true, true, false),
        vec![PostStopPass::Reprocess]
    );
    assert_eq!(
        post_stop_passes(false, false, true),
        vec![PostStopPass::Summarise]
    );
    assert_eq!(
        post_stop_passes(true, true, true),
        vec![PostStopPass::Reprocess, PostStopPass::Summarise],
        "reprocess → summarise (summarise sees the final transcript)"
    );
}

/// #68 — when auto-summarise is on, the plan ends with `Summarise` (after any
/// reprocess), and when off it is absent.
#[test]
fn post_stop_passes_appends_summarise_when_enabled() {
    // Auto-summarise alone (no reprocess trigger).
    assert_eq!(
        post_stop_passes(false, false, true),
        vec![PostStopPass::Summarise],
        "auto-summarise on must add the summarise pass"
    );
    // Summarise is always LAST.
    assert_eq!(
        *post_stop_passes(true, false, true).last().unwrap(),
        PostStopPass::Summarise
    );
    // Off → never planned.
    assert!(
        !post_stop_passes(true, true, false).contains(&PostStopPass::Summarise),
        "auto-summarise off must omit the summarise pass"
    );
}

/// Producer-gate S2 — the delegate knob is truthy only for the documented
/// affirmative spellings; anything else (including a missing var) stays local.
#[test]
fn delegate_value_parses_truthy() {
    assert!(is_delegate_value(Some("1")));
    assert!(is_delegate_value(Some("true")));
    assert!(is_delegate_value(Some("yes")));
    assert!(is_delegate_value(Some("on")));
    // Case-insensitive and trimmed.
    assert!(is_delegate_value(Some("TRUE")));
    assert!(is_delegate_value(Some("On")));
    assert!(is_delegate_value(Some("  true  ")));
    assert!(!is_delegate_value(Some("0")));
    assert!(!is_delegate_value(Some("false")));
    assert!(!is_delegate_value(Some("")));
    assert!(!is_delegate_value(None));
}

/// An empty plan invokes `run_pass` zero times (no background work).
#[tokio::test]
async fn run_post_stop_passes_noop_when_empty() {
    let mut calls: Vec<PostStopPass> = Vec::new();
    run_post_stop_passes(&[], MeetingId::new(), |pass| {
        calls.push(pass);
        async { Ok(()) }
    })
    .await;
    assert!(calls.is_empty(), "no passes → run_pass never called");
}

/// All planned passes run, in order, when each succeeds — the reprocess pass
/// then the #68 auto-summarise pass LAST.
#[tokio::test]
async fn run_post_stop_passes_runs_all_in_order() {
    let passes = post_stop_passes(true, true, true);
    let mut calls: Vec<PostStopPass> = Vec::new();
    run_post_stop_passes(&passes, MeetingId::new(), |pass| {
        calls.push(pass);
        async { Ok(()) }
    })
    .await;
    assert_eq!(
        calls,
        vec![PostStopPass::Reprocess, PostStopPass::Summarise]
    );
}

/// #68 — the post-stop chain AUTO-SUMMARISES when `auto_summarise_on_stop` is
/// on: the planned passes include `Summarise`, and the `run_pass` closure
/// (here a stub summariser writing `summary.md` + emitting `SummaryReady`)
/// runs it. Verified WITHOUT a Tauri runtime, a real model, or a real
/// orchestrator — the held-summarise side effects are exercised via the
/// trait-based [`summarise_meeting_inner`] + [`emit_summary_ready`] seam.
#[tokio::test]
async fn post_stop_chain_auto_summarises_when_enabled() {
    let tempdir = TempDir::new().expect("tempdir");
    let root = tempdir.path();
    let meeting_id = write_synthetic_meeting(
        root,
        "Auto-summarised",
        "2026-06-10T09:00:00Z",
        Some("the agenda item"),
    );

    // Gating: auto-summarise on, nothing else → exactly the summarise pass.
    let passes = post_stop_passes(false, false, /* auto_summarise */ true);
    assert_eq!(passes, vec![PostStopPass::Summarise]);

    let (event_tx, mut event_rx) = broadcast::channel::<AppEvent>(16);
    let stub = StubSummariser::new("## Summary\n\nAuto-generated on stop.\n");

    // Drive the chain. The `Summarise` arm runs the SAME read → summarise →
    // write → `SummaryReady` work `run_held_summarise` performs, but through
    // the model-free stub seam so CI needs no GGUF.
    run_post_stop_passes(&passes, meeting_id, |pass| {
        let event_tx = event_tx.clone();
        let stub = &stub;
        async move {
            match pass {
                PostStopPass::Summarise => {
                    summarise_meeting_inner(root, meeting_id, stub, "prompt")?;
                    emit_summary_ready(&event_tx, meeting_id);
                    Ok(())
                }
                other => panic!("unexpected pass {other:?}"),
            }
        }
    })
    .await;

    // `summary.md` was written by the auto-summarise pass.
    let written = get_summary_inner(root, meeting_id).expect("read summary");
    assert_eq!(
        written.as_deref(),
        Some("## Summary\n\nAuto-generated on stop.\n"),
        "auto-summarise must write summary.md after stop"
    );
    // And `SummaryReady` was emitted for the meeting (clears the UI indicator).
    let event = event_rx.recv().await.expect("a SummaryReady event");
    match event {
        AppEvent::SummaryReady { meeting_id: got } => assert_eq!(got, meeting_id),
        other => panic!("expected SummaryReady, got {other:?}"),
    }
}

/// #68 — when `auto_summarise_on_stop` is off, the plan omits `Summarise`, so
/// the chain never invokes the summarise pass (no `summary.md`, no
/// `SummaryReady`).
#[tokio::test]
async fn post_stop_chain_skips_summarise_when_disabled() {
    let passes = post_stop_passes(false, false, /* auto_summarise */ false);
    assert!(
        passes.is_empty(),
        "auto-summarise off + nothing else → no passes"
    );

    let mut summarise_calls = 0u32;
    run_post_stop_passes(&passes, MeetingId::new(), |pass| {
        if pass == PostStopPass::Summarise {
            summarise_calls += 1;
        }
        async { Ok(()) }
    })
    .await;
    assert_eq!(
        summarise_calls, 0,
        "summarise pass must not run when disabled"
    );
}

/// A failed reprocess (InvalidInput = busy, OR any other error) is tolerated
/// and does NOT prevent the auto-summarise pass from being attempted — the
/// recording is already safely persisted.
#[tokio::test]
async fn run_post_stop_passes_failure_does_not_abort_later_passes() {
    for first_err in [
        AppError::InvalidInput {
            context: "busy".into(),
        },
        AppError::Internal {
            context: "boom".into(),
        },
    ] {
        // Reprocess trigger + auto-summarise on → reprocess THEN summarise.
        let passes = post_stop_passes(true, true, true);
        assert_eq!(
            passes,
            vec![PostStopPass::Reprocess, PostStopPass::Summarise]
        );
        let mut calls: Vec<PostStopPass> = Vec::new();
        // Move the error into the closure via an Option taken on first call.
        let mut first_err = Some(first_err);
        run_post_stop_passes(&passes, MeetingId::new(), |pass| {
            calls.push(pass);
            let result = match pass {
                PostStopPass::Reprocess => Err(first_err.take().expect("one reprocess call")),
                PostStopPass::Summarise => Ok(()),
            };
            async move { result }
        })
        .await;
        assert_eq!(
            calls,
            vec![PostStopPass::Reprocess, PostStopPass::Summarise],
            "summarise must still be attempted after a failed reprocess"
        );
    }
}

// -----------------------------------------------------------------------
// Chat-turn persistence (regression guards for the IMPL-4a review CRITICAL +
// WARNING-2): the user message must be persisted, and a Tool message must
// persist the FULL machine payload (not the one-line event summary).
// -----------------------------------------------------------------------

/// `wire_produced_from_delta` maps the engine-history delta to wire messages
/// carrying the FULL tool payload + the tool name (not the lossy summary),
/// AND (CQ1) the assistant-tool_calls carrier + the tool result's
/// tool_call_id so a reloaded multi-tool turn is a valid OpenAI sequence.
#[test]
fn wire_produced_from_delta_keeps_full_tool_payload() {
    let call = chat_agent::ToolCall {
        id: "call_1".to_string(),
        name: "get_summary".to_string(),
        arguments_json: "{}".to_string(),
    };
    let delta = vec![
        chat_agent::ChatMessage::assistant_tool_calls("", vec![call]),
        chat_agent::ChatMessage::tool_result(
            "call_1",
            "get_summary",
            r#"{"decisions":["ship Phase 9"]}"#,
        ),
        chat_agent::ChatMessage::assistant("We decided to ship Phase 9."),
    ];
    let wire = wire_produced_from_delta(&delta, 3);
    assert_eq!(wire.len(), 3);
    // CQ1: the assistant-tool_calls message carries the OpenAI tool-call
    // carrier (id/name/arguments), preceding the tool result.
    assert_eq!(wire[0].role, ChatRole::Assistant);
    assert_eq!(wire[0].tool_calls.len(), 1);
    assert_eq!(wire[0].tool_calls[0].id, "call_1");
    assert_eq!(wire[0].tool_calls[0].name, "get_summary");
    // The tool result carries the matching tool_call_id + full payload.
    assert_eq!(wire[1].role, ChatRole::Tool);
    assert_eq!(wire[1].tool_call_id.as_deref(), Some("call_1"));
    assert_eq!(wire[1].tool_name.as_deref(), Some("get_summary"));
    assert!(
        wire[1].content.contains("decisions"),
        "the full machine payload must be persisted, not a one-line summary"
    );
    assert_eq!(wire[1].turn_id, 3);
    assert_eq!(wire[2].role, ChatRole::Assistant);
    assert_eq!(wire[2].content, "We decided to ship Phase 9.");
    assert!(wire[2].tool_name.is_none());
    assert!(wire[2].tool_calls.is_empty());
}

/// `persist_session` saves the WHOLE in-memory session — including the user
/// message (the IMPL-4a CRITICAL: it was previously dropped) and the
/// full-payload tool message — without a reload-and-append.
#[tokio::test]
async fn persist_session_round_trips_user_and_full_tool_payload() {
    let tempdir = TempDir::new().expect("tempdir");
    let root = tempdir.path();
    let meeting_id = MeetingId::new();
    MeetingFolder::create(root, meeting_id).expect("meeting folder");
    let sid = ChatSessionId::new();

    let session = ChatSession {
        id: sid,
        meeting_id: Some(meeting_id),
        title: None,
        messages: vec![
            ChatMessage {
                role: ChatRole::User,
                content: "what was decided?".into(),
                tool_name: None,
                tool_call_id: None,
                tool_calls: Vec::new(),
                turn_id: 0,
            },
            ChatMessage {
                role: ChatRole::Tool,
                content: r#"{"decisions":["ship"]}"#.into(),
                tool_name: Some("get_summary".into()),
                tool_call_id: Some("call_1".into()),
                tool_calls: Vec::new(),
                turn_id: 0,
            },
            ChatMessage {
                role: ChatRole::Assistant,
                content: "We decided to ship.".into(),
                tool_name: None,
                tool_call_id: None,
                tool_calls: Vec::new(),
                turn_id: 0,
            },
        ],
        created_at: "2026-06-10T00:00:00Z".into(),
        updated_at: "2026-06-10T00:00:00Z".into(),
        is_live: false,
    };

    persist_session(root, Some(meeting_id), session).await;

    let loaded = ChatStore::load(root, meeting_id, sid)
        .expect("load")
        .expect("session must be persisted");
    // CRITICAL regression guard: the user message survives.
    assert!(
        loaded
            .messages
            .iter()
            .any(|m| m.role == ChatRole::User && m.content == "what was decided?"),
        "the user message must be persisted"
    );
    // WARNING-2 guard: the Tool message keeps the full payload + tool name.
    let tool = loaded
        .messages
        .iter()
        .find(|m| m.role == ChatRole::Tool)
        .expect("a tool message must be persisted");
    assert!(
        tool.content.contains("decisions"),
        "full tool payload persisted"
    );
    assert_eq!(tool.tool_name.as_deref(), Some("get_summary"));
    assert_eq!(loaded.messages.len(), 3);
}

// -----------------------------------------------------------------------
// load_or_new_session — post-Stop continuation of the live co-pilot
// session (B1): a `send_chat_message` with no `session_id` must continue
// the meeting's live session rather than mint an unrelated fresh one.
// -----------------------------------------------------------------------

#[tokio::test]
async fn load_or_new_session_continues_the_live_session_when_none_given() {
    let tempdir = TempDir::new().expect("tempdir");
    let root = tempdir.path();
    let meeting_id = MeetingId::new();
    MeetingFolder::create(root, meeting_id).expect("meeting folder");

    let live = ChatStore::load_or_create_live(root, meeting_id, "2026-07-01T00:00:00Z")
        .expect("create live session");

    let loaded = load_or_new_session(root, Some(meeting_id), None)
        .await
        .expect("load_or_new_session");
    assert_eq!(
        loaded.id, live.id,
        "no session_id given must continue the meeting's live session, not mint a fresh one"
    );
    assert!(loaded.is_live);
}

#[tokio::test]
async fn load_or_new_session_mints_fresh_session_when_no_live_session_exists() {
    let tempdir = TempDir::new().expect("tempdir");
    let root = tempdir.path();
    let meeting_id = MeetingId::new();
    MeetingFolder::create(root, meeting_id).expect("meeting folder");

    let session = load_or_new_session(root, Some(meeting_id), None)
        .await
        .expect("load_or_new_session");
    assert!(
        !session.is_live,
        "with no live session on disk, a fresh non-live session is minted"
    );
    assert!(session.messages.is_empty());
}

#[tokio::test]
async fn load_or_new_session_honours_an_explicit_session_id_over_the_live_session() {
    let tempdir = TempDir::new().expect("tempdir");
    let root = tempdir.path();
    let meeting_id = MeetingId::new();
    MeetingFolder::create(root, meeting_id).expect("meeting folder");

    // A live session exists, but the caller explicitly names a different,
    // not-yet-persisted session id — that id must be honoured, not silently
    // swapped for the live session.
    ChatStore::load_or_create_live(root, meeting_id, "2026-07-01T00:00:00Z")
        .expect("create live session");
    let explicit_sid = ChatSessionId::new();

    let session = load_or_new_session(root, Some(meeting_id), Some(explicit_sid))
        .await
        .expect("load_or_new_session");
    assert_eq!(session.id, explicit_sid);
    assert!(!session.is_live);
}

// -----------------------------------------------------------------------
// chat_turn_base_prompt — the co-pilot keeps its own persona post-Stop (B2)
// -----------------------------------------------------------------------

#[test]
fn live_session_keeps_the_copilot_persona() {
    let base = chat_turn_base_prompt(true, "Ordinary chat persona.", "Co-pilot persona.");
    assert_eq!(base, "Co-pilot persona.");
}

#[test]
fn ordinary_session_uses_the_chat_persona() {
    let base = chat_turn_base_prompt(false, "Ordinary chat persona.", "Co-pilot persona.");
    assert_eq!(base, "Ordinary chat persona.");
}

// -----------------------------------------------------------------------
// Chat prompt scoping — the agent must not ask for a meeting id when a
// meeting is open (it has `default_meeting` in scope).
// -----------------------------------------------------------------------

#[test]
fn chat_prompt_scopes_to_the_open_meeting() {
    let mid = MeetingId::new();
    let p = chat_system_prompt_for_meeting("BASE", Some(mid), Some("Standup"));
    assert!(p.starts_with("BASE"), "base prompt is preserved");
    assert!(p.contains("# Current meeting"));
    assert!(p.contains(&mid.0.to_string()), "the meeting id is named");
    assert!(p.contains("titled \"Standup\""));
    assert!(
        p.contains("NEVER ask the user"),
        "must instruct the agent not to ask for a meeting id"
    );
}

#[test]
fn chat_prompt_without_title_omits_the_titled_clause() {
    let mid = MeetingId::new();
    let p = chat_system_prompt_for_meeting("BASE", Some(mid), None);
    assert!(p.contains(&mid.0.to_string()));
    assert!(!p.contains("titled"));
    // A blank/whitespace title is treated as no title.
    let blank = chat_system_prompt_for_meeting("BASE", Some(mid), Some("   "));
    assert!(!blank.contains("titled"));
}

#[test]
fn chat_prompt_meeting_less_is_unchanged() {
    assert_eq!(
        chat_system_prompt_for_meeting("BASE", None, Some("ignored")),
        "BASE"
    );
}

// -----------------------------------------------------------------------
// apply_output_language — the prompt-injection helper
// -----------------------------------------------------------------------

#[test]
fn apply_output_language_appends_instruction_for_known_name() {
    let result = apply_output_language("Do the thing.", "French");
    assert_eq!(result, "Do the thing.\n\nRespond entirely in French.");
}

#[test]
fn apply_output_language_no_op_for_auto_with_unmappable_locale() {
    // When "auto" cannot resolve (sys_locale not available in the test
    // sandbox, or the locale maps to a known language), the helper either
    // appends a language or returns the prompt unchanged. We test the
    // explicit-name path separately; for "auto" we just verify no panic.
    let result = apply_output_language("Base prompt.", "auto");
    // Result is either the base prompt (auto → None) or extended. Either
    // is valid — we only assert the base is preserved.
    assert!(result.starts_with("Base prompt."));
}

#[test]
fn apply_output_language_no_op_for_empty_setting() {
    let result = apply_output_language("Base prompt.", "");
    assert_eq!(result, "Base prompt.");
}

#[test]
fn apply_output_language_explicit_name_appended_after_custom_prompt() {
    // The instruction is appended LAST — even if a custom prompt already
    // says something about language, the explicit setting wins.
    let result = apply_output_language("Respond in English only.", "German");
    assert!(
        result.ends_with("\n\nRespond entirely in German."),
        "output-language instruction must be appended after the full prompt"
    );
}

// -----------------------------------------------------------------------
// Live co-pilot routing — route_live_chat_message (U4 A3)
// -----------------------------------------------------------------------

/// A fake live worker task: reads one `UserChatRequest` then streams one
/// `Token` + `Done` on the request's `reply_tx`.
async fn fake_live_worker(
    mut rx: tokio::sync::mpsc::Receiver<crate::live_agent::UserChatRequest>,
) {
    if let Some(req) = rx.recv().await {
        let _ = req
            .reply_tx
            .try_send(crate::live_agent::UserReplyChunk::Token("hello".to_string()));
        let _ = req
            .reply_tx
            .send(crate::live_agent::UserReplyChunk::Done(
                "hello world".to_string(),
            ))
            .await;
    }
}

/// `route_live_chat_message` routes to the live path when a handle exists,
/// emits `ChatToken` + `ChatTurnComplete` with the live session id, and
/// clears `chat_in_flight` on completion.
#[tokio::test]
async fn route_live_chat_message_emits_token_and_complete_then_clears_guard() {
    let tmp = TempDir::new().expect("tempdir");
    let meetings_dir = tmp.path().to_path_buf();
    let mid = MeetingId::new();

    // Create the meeting folder so ChatStore::load_or_create_live can write.
    notes_crdt::MeetingFolder::create(&meetings_dir, mid).expect("create meeting folder");

    // Fake worker: reads one request and sends Token + Done.
    let (user_tx, user_rx) =
        tokio::sync::mpsc::channel::<crate::live_agent::UserChatRequest>(4);
    tokio::spawn(fake_live_worker(user_rx));

    let (event_tx, mut event_rx) = tokio::sync::broadcast::channel::<AppEvent>(32);
    let chat_in_flight: Arc<
        std::sync::Mutex<std::collections::HashSet<ChatSessionId>>,
    > = Arc::new(std::sync::Mutex::new(Default::default()));
    let chat_cancel: Arc<
        std::sync::Mutex<
            std::collections::HashMap<ChatSessionId, chat_agent::CancelFlag>,
        >,
    > = Arc::new(std::sync::Mutex::new(Default::default()));

    let live_sid = route_live_chat_message(
        &meetings_dir,
        mid,
        user_tx,
        "hello?".to_string(),
        event_tx,
        Arc::clone(&chat_in_flight),
        Arc::clone(&chat_cancel),
    )
    .await
    .expect("live route must succeed");

    // We have a live session id — it will be verified by the event assertions below.

    // Drain events until we see ChatTurnComplete.
    let mut saw_token = false;
    let mut saw_complete = false;
    let deadline =
        tokio::time::Instant::now() + tokio::time::Duration::from_secs(5);
    loop {
        match tokio::time::timeout_at(
            deadline,
            event_rx.recv(),
        )
        .await
        {
            Ok(Ok(AppEvent::ChatToken { session_id, .. })) => {
                assert_eq!(session_id, live_sid, "ChatToken must carry the live session id");
                saw_token = true;
            }
            Ok(Ok(AppEvent::ChatTurnComplete { session_id, final_text, .. })) => {
                assert_eq!(
                    session_id, live_sid,
                    "ChatTurnComplete must carry the live session id"
                );
                assert_eq!(final_text, "hello world");
                saw_complete = true;
                break;
            }
            Ok(Ok(_)) => {} // other events are ignored
            Ok(Err(_)) => break,
            Err(_) => panic!("timed out waiting for ChatTurnComplete"),
        }
    }
    assert!(saw_token, "expected at least one ChatToken event");
    assert!(saw_complete, "expected ChatTurnComplete");

    // Give the drain task time to clear the guards.
    tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
    assert!(
        chat_in_flight
            .lock()
            .expect("poisoned")
            .get(&live_sid)
            .is_none(),
        "chat_in_flight must be cleared after the terminal event"
    );
    assert!(
        chat_cancel
            .lock()
            .expect("poisoned")
            .get(&live_sid)
            .is_none(),
        "chat_cancel must be cleared after the terminal event"
    );
}

/// When no live handle is present, `route_live_chat_message` falls through
/// (this path is not exercised here; we test the non-live path by verifying
/// the command selects the fresh-context path when no handle is registered).
///
/// Specifically: when the worker channel is already closed, the function
/// emits a `ChatError` and returns `Ok(live_sid)` (does not panic or hang).
#[tokio::test]
async fn route_live_chat_message_worker_gone_emits_error_returns_ok() {
    let tmp = TempDir::new().expect("tempdir");
    let meetings_dir = tmp.path().to_path_buf();
    let mid = MeetingId::new();
    notes_crdt::MeetingFolder::create(&meetings_dir, mid).expect("create meeting folder");

    // Drop the receiver immediately — the worker is gone.
    let (user_tx, _rx) =
        tokio::sync::mpsc::channel::<crate::live_agent::UserChatRequest>(1);
    drop(_rx);

    let (event_tx, mut event_rx) = tokio::sync::broadcast::channel::<AppEvent>(8);
    let chat_in_flight: Arc<
        std::sync::Mutex<std::collections::HashSet<ChatSessionId>>,
    > = Arc::new(std::sync::Mutex::new(Default::default()));
    let chat_cancel: Arc<
        std::sync::Mutex<
            std::collections::HashMap<ChatSessionId, chat_agent::CancelFlag>,
        >,
    > = Arc::new(std::sync::Mutex::new(Default::default()));

    let live_sid = route_live_chat_message(
        &meetings_dir,
        mid,
        user_tx,
        "hello?".to_string(),
        event_tx,
        Arc::clone(&chat_in_flight),
        Arc::clone(&chat_cancel),
    )
    .await
    .expect("must return Ok even when worker is gone");

    // A ChatError should have been emitted.
    let event = tokio::time::timeout(
        tokio::time::Duration::from_secs(1),
        event_rx.recv(),
    )
    .await
    .expect("timeout waiting for ChatError")
    .expect("event");
    assert!(
        matches!(event, AppEvent::ChatError { session_id, .. } if session_id == live_sid),
        "expected ChatError for the live session id, got: {event:?}"
    );

    // Guards must be cleared (not leaked).
    assert!(
        chat_in_flight.lock().expect("poisoned").is_empty(),
        "chat_in_flight must be cleared on worker-gone path"
    );
    assert!(
        chat_cancel.lock().expect("poisoned").is_empty(),
        "chat_cancel must be cleared on worker-gone path"
    );
}

/// A fake driver that immediately sets terminal state and never drains
/// `user_msg_rx` through the normal path — simulating the context-exhausted
/// driver loop after `WorkerResult::CapacityExhausted`.
///
/// The fix adds a `select!` arm gated `if terminal` that drains
/// `user_msg_rx` and replies with `UserReplyChunk::Err`. This test asserts
/// that `route_live_chat_message`'s drain task terminates within a bounded
/// time and clears both `chat_in_flight` and `chat_cancel`, rather than
/// hanging indefinitely.
async fn fake_terminal_driver(
    mut rx: tokio::sync::mpsc::Receiver<crate::live_agent::UserChatRequest>,
) {
    // Simulate the terminal-but-alive driver: immediately reject every
    // incoming request with an Err chunk (mirrors the new `if terminal`
    // select arm in `run_driver_task`).
    while let Some(user_req) = rx.recv().await {
        let _ = user_req.reply_tx.try_send(crate::live_agent::UserReplyChunk::Err(
            "Live co-pilot paused: context window filled for this session.".to_string(),
        ));
    }
}

/// When the live driver is in terminal state, `route_live_chat_message`
/// must still clear `chat_in_flight` and `chat_cancel` within a bounded
/// time. Without the terminal-drain fix the drain task hangs on
/// `reply_rx.recv()` forever, permanently leaking both guards and leaving
/// the chat UI in "Sending…" with Stop unable to clear it.
#[tokio::test]
async fn route_live_chat_message_terminal_driver_clears_guards() {
    let tmp = TempDir::new().expect("tempdir");
    let meetings_dir = tmp.path().to_path_buf();
    let mid = MeetingId::new();
    notes_crdt::MeetingFolder::create(&meetings_dir, mid).expect("create meeting folder");

    let (user_tx, user_rx) =
        tokio::sync::mpsc::channel::<crate::live_agent::UserChatRequest>(4);
    tokio::spawn(fake_terminal_driver(user_rx));

    let (event_tx, mut event_rx) = tokio::sync::broadcast::channel::<AppEvent>(32);
    let chat_in_flight: Arc<
        std::sync::Mutex<std::collections::HashSet<ChatSessionId>>,
    > = Arc::new(std::sync::Mutex::new(Default::default()));
    let chat_cancel: Arc<
        std::sync::Mutex<
            std::collections::HashMap<ChatSessionId, chat_agent::CancelFlag>,
        >,
    > = Arc::new(std::sync::Mutex::new(Default::default()));

    let live_sid = route_live_chat_message(
        &meetings_dir,
        mid,
        user_tx,
        "hello?".to_string(),
        event_tx,
        Arc::clone(&chat_in_flight),
        Arc::clone(&chat_cancel),
    )
    .await
    .expect("live route must return Ok even when driver is terminal");

    // The terminal driver emits a ChatError via UserReplyChunk::Err.
    let event = tokio::time::timeout(
        tokio::time::Duration::from_secs(5),
        event_rx.recv(),
    )
    .await
    .expect("timed out waiting for ChatError from terminal driver")
    .expect("event channel closed");
    assert!(
        matches!(event, AppEvent::ChatError { session_id, .. } if session_id == live_sid),
        "expected ChatError for the live session id from terminal driver, got: {event:?}"
    );

    // Give the drain task time to clear the guards after receiving the error.
    tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
    assert!(
        chat_in_flight
            .lock()
            .expect("poisoned")
            .get(&live_sid)
            .is_none(),
        "chat_in_flight must be cleared after terminal driver emits Err"
    );
    assert!(
        chat_cancel
            .lock()
            .expect("poisoned")
            .get(&live_sid)
            .is_none(),
        "chat_cancel must be cleared after terminal driver emits Err"
    );
}
