//! The v1 tool catalogue (Phase 9 §2.2).
//!
//! Each tool is one zero-sized struct implementing [`crate::Tool`]. Read/compute
//! tools call the `persistence` readers (on `spawn_blocking`) or the
//! orchestrator; write tools route through an existing persistence/orchestrator
//! writer — no tool opens a file under `meetings/` for writing itself
//! (`persistence` is the sole writer; §4).
//!
//! # The speaker-name overlay
//!
//! `get_transcript` and `get_meeting` apply the [`apply_speaker_overlay`]
//! read-time mapping: a `Segment.speaker_id` of `"A"` is rewritten to the
//! display name from `MeetingMeta.speaker_names["A"]` when one is set, so the
//! agent sees "Alice" not "A". The on-disk transcript is untouched (overlay is
//! presentation-only).

use std::collections::BTreeMap;
use std::time::Duration;

use async_trait::async_trait;
use minutist_common::{
    AppError, AppResult, InterAgentRequest, MeetingId, MeetingMeta, RecordingState, Segment,
};
use serde_json::json;
use tokio::sync::oneshot;

use crate::{
    require_bounded_str, require_str, require_u64, resolve_meeting, Tool, ToolContext, ToolOutput,
};

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// The `speaker_names` overlay is the canonical helper in `minutist_common`
/// (`apply_speaker_overlay`), shared with the summariser input path; re-exported
/// here so the read tools below call it unqualified.
pub(crate) use minutist_common::apply_speaker_overlay;

/// Read a meeting's metadata on a blocking thread.
async fn read_metadata(ctx: &ToolContext, id: MeetingId) -> AppResult<MeetingMeta> {
    let dir = ctx.meeting_dir(id);
    spawn_blocking_io(move || persistence::read_metadata(&dir)).await
}

/// Read a meeting's transcript on a blocking thread.
async fn read_transcript(ctx: &ToolContext, id: MeetingId) -> AppResult<Vec<Segment>> {
    let dir = ctx.meeting_dir(id);
    spawn_blocking_io(move || persistence::read_transcript(&dir)).await
}

/// Drive a blocking `persistence` read on `spawn_blocking`, mapping a join
/// failure to `AppError::Internal`.
async fn spawn_blocking_io<T, F>(f: F) -> AppResult<T>
where
    T: Send + 'static,
    F: FnOnce() -> AppResult<T> + Send + 'static,
{
    tokio::task::spawn_blocking(f)
        .await
        .map_err(|e| AppError::Internal {
            context: format!("agent-tools blocking read join failed: {e}"),
        })?
}

/// JSON-schema object root with the given properties + required list. Avoids
/// regex `pattern` (the GBNF converter rejects PCRE shorthands).
fn schema_obj(properties: serde_json::Value, required: &[&str]) -> serde_json::Value {
    json!({
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": false,
    })
}

/// The `{meeting_id: string}` schema shared by the single-meeting tools.
fn meeting_id_schema() -> serde_json::Value {
    schema_obj(
        json!({ "meeting_id": { "type": "string", "description": "Meeting UUID" } }),
        &["meeting_id"],
    )
}

// ---------------------------------------------------------------------------
// list_meetings
// ---------------------------------------------------------------------------

pub struct ListMeetings;

#[async_trait]
impl Tool for ListMeetings {
    fn name(&self) -> &'static str {
        "list_meetings"
    }
    fn title(&self) -> &'static str {
        "List meetings"
    }
    fn description(&self) -> &'static str {
        "List all meetings, newest first."
    }
    fn input_schema(&self) -> serde_json::Value {
        schema_obj(json!({}), &[])
    }
    async fn execute(&self, ctx: &ToolContext, _args: serde_json::Value) -> AppResult<ToolOutput> {
        let entries = ctx.index.list_meetings().await?;
        let n = entries.len();
        let data = serde_json::to_value(&entries).map_err(serialise_err)?;
        Ok(ToolOutput::new(data, format!("{n} meeting(s)")))
    }
}

// ---------------------------------------------------------------------------
// search_meetings
// ---------------------------------------------------------------------------

pub struct SearchMeetings;

#[async_trait]
impl Tool for SearchMeetings {
    fn name(&self) -> &'static str {
        "search_meetings"
    }
    fn title(&self) -> &'static str {
        "Search meetings"
    }
    fn description(&self) -> &'static str {
        "Find meetings whose title or list-excerpt contains the query (does not search transcript bodies; use search_within_transcript for that)."
    }
    fn input_schema(&self) -> serde_json::Value {
        schema_obj(
            json!({ "query": { "type": "string", "description": "Substring to match against title/excerpt" } }),
            &["query"],
        )
    }
    async fn execute(&self, ctx: &ToolContext, args: serde_json::Value) -> AppResult<ToolOutput> {
        let query = require_str(&args, "query")?;
        let entries = ctx.index.search(query).await?;
        let n = entries.len();
        let data = serde_json::to_value(&entries).map_err(serialise_err)?;
        Ok(ToolOutput::new(
            data,
            format!("{n} match(es) for {query:?}"),
        ))
    }
}

// ---------------------------------------------------------------------------
// get_meeting
// ---------------------------------------------------------------------------

pub struct GetMeeting;

#[async_trait]
impl Tool for GetMeeting {
    fn name(&self) -> &'static str {
        "get_meeting"
    }
    fn title(&self) -> &'static str {
        "Get meeting"
    }
    fn description(&self) -> &'static str {
        "Full meeting state: metadata (with speaker names overlaid), transcript, notes, and the stored summary if present."
    }
    fn input_schema(&self) -> serde_json::Value {
        meeting_id_schema()
    }
    async fn execute(&self, ctx: &ToolContext, args: serde_json::Value) -> AppResult<ToolOutput> {
        let id = resolve_meeting(ctx, &args)?;
        let dir = ctx.meeting_dir(id);
        let (mut state, summary) = spawn_blocking_io(move || {
            let state = persistence::read_meeting_state(&dir)?;
            let summary = persistence::read_summary(&dir)?;
            Ok((state, summary))
        })
        .await?;

        // Overlay speaker names onto the transcript (presentation only).
        apply_speaker_overlay(&mut state.transcript, &state.meta.speaker_names);

        let n_segments = state.transcript.len();
        let data = json!({
            "meta": serde_json::to_value(&state.meta).map_err(serialise_err)?,
            "transcript": serde_json::to_value(&state.transcript).map_err(serialise_err)?,
            "notes": serde_json::to_value(&state.notes).map_err(serialise_err)?,
            "summary": summary,
        });
        Ok(ToolOutput::new(
            data,
            format!("{} — {n_segments} segment(s)", state.meta.title),
        ))
    }
}

// ---------------------------------------------------------------------------
// get_transcript
// ---------------------------------------------------------------------------

pub struct GetTranscript;

#[async_trait]
impl Tool for GetTranscript {
    fn name(&self) -> &'static str {
        "get_transcript"
    }
    fn title(&self) -> &'static str {
        "Get transcript"
    }
    fn description(&self) -> &'static str {
        "Full transcript segments for a meeting, with speaker names overlaid and per-word timestamps when available."
    }
    fn input_schema(&self) -> serde_json::Value {
        meeting_id_schema()
    }
    async fn execute(&self, ctx: &ToolContext, args: serde_json::Value) -> AppResult<ToolOutput> {
        let id = resolve_meeting(ctx, &args)?;
        let meta = read_metadata(ctx, id).await?;
        let mut transcript = read_transcript(ctx, id).await?;
        apply_speaker_overlay(&mut transcript, &meta.speaker_names);
        let n = transcript.len();
        let data = serde_json::to_value(&transcript).map_err(serialise_err)?;
        Ok(ToolOutput::new(data, format!("{n} segment(s)")))
    }
}

// ---------------------------------------------------------------------------
// get_transcript_slice
// ---------------------------------------------------------------------------

pub struct GetTranscriptSlice;

#[async_trait]
impl Tool for GetTranscriptSlice {
    fn name(&self) -> &'static str {
        "get_transcript_slice"
    }
    fn title(&self) -> &'static str {
        "Get transcript excerpt"
    }
    fn description(&self) -> &'static str {
        "Transcript segments overlapping the [start_ms, end_ms) time window (transcript-clock milliseconds), speaker names overlaid."
    }
    fn input_schema(&self) -> serde_json::Value {
        schema_obj(
            json!({
                "meeting_id": { "type": "string", "description": "Meeting UUID" },
                "start_ms": { "type": "integer", "minimum": 0, "description": "Window start (ms)" },
                "end_ms": { "type": "integer", "minimum": 0, "description": "Window end (ms)" },
            }),
            &["meeting_id", "start_ms", "end_ms"],
        )
    }
    async fn execute(&self, ctx: &ToolContext, args: serde_json::Value) -> AppResult<ToolOutput> {
        let id = resolve_meeting(ctx, &args)?;
        let start = require_u64(&args, "start_ms")?;
        let end = require_u64(&args, "end_ms")?;
        if end <= start {
            return Err(AppError::InvalidInput {
                context: format!("end_ms ({end}) must exceed start_ms ({start})"),
            });
        }
        let meta = read_metadata(ctx, id).await?;
        let mut transcript = read_transcript(ctx, id).await?;
        apply_speaker_overlay(&mut transcript, &meta.speaker_names);
        transcript.retain(|s| s.start_ms < end && s.end_ms > start);
        let n = transcript.len();
        let data = serde_json::to_value(&transcript).map_err(serialise_err)?;
        Ok(ToolOutput::new(
            data,
            format!("{n} segment(s) in [{start}, {end}) ms"),
        ))
    }
}

// ---------------------------------------------------------------------------
// get_summary
// ---------------------------------------------------------------------------

pub struct GetSummary;

#[async_trait]
impl Tool for GetSummary {
    fn name(&self) -> &'static str {
        "get_summary"
    }
    fn title(&self) -> &'static str {
        "Get summary"
    }
    fn description(&self) -> &'static str {
        "The stored summary markdown for a meeting (null when the meeting has not been summarised)."
    }
    fn input_schema(&self) -> serde_json::Value {
        meeting_id_schema()
    }
    async fn execute(&self, ctx: &ToolContext, args: serde_json::Value) -> AppResult<ToolOutput> {
        let id = resolve_meeting(ctx, &args)?;
        let dir = ctx.meeting_dir(id);
        let summary = spawn_blocking_io(move || persistence::read_summary(&dir)).await?;
        let label = if summary.is_some() {
            "summary present"
        } else {
            "no summary yet"
        };
        Ok(ToolOutput::new(json!({ "summary": summary }), label))
    }
}

// ---------------------------------------------------------------------------
// get_notes
// ---------------------------------------------------------------------------

pub struct GetNotes;

#[async_trait]
impl Tool for GetNotes {
    fn name(&self) -> &'static str {
        "get_notes"
    }
    fn title(&self) -> &'static str {
        "Get notes"
    }
    fn description(&self) -> &'static str {
        "The user's hand-typed notes for a meeting (markdown + opaque editor JSON), or null when no notes were taken."
    }
    fn input_schema(&self) -> serde_json::Value {
        meeting_id_schema()
    }
    async fn execute(&self, ctx: &ToolContext, args: serde_json::Value) -> AppResult<ToolOutput> {
        let id = resolve_meeting(ctx, &args)?;
        let root = ctx.meetings_dir.clone();
        // NotesStore::load takes (root, meeting_id) and returns Option.
        let notes = spawn_blocking_io(move || persistence::NotesStore::load(&root, id)).await?;
        let data = match notes {
            Some(n) => {
                json!({ "markdown": n.markdown, "json": n.json })
            }
            None => serde_json::Value::Null,
        };
        let label = if data.is_null() {
            "no notes"
        } else {
            "notes loaded"
        };
        Ok(ToolOutput::new(data, label))
    }
}

// ---------------------------------------------------------------------------
// get_metadata
// ---------------------------------------------------------------------------

pub struct GetMetadata;

#[async_trait]
impl Tool for GetMetadata {
    fn name(&self) -> &'static str {
        "get_metadata"
    }
    fn title(&self) -> &'static str {
        "Get meeting metadata"
    }
    fn description(&self) -> &'static str {
        "Meeting metadata, including any configured speaker_names map."
    }
    fn input_schema(&self) -> serde_json::Value {
        meeting_id_schema()
    }
    async fn execute(&self, ctx: &ToolContext, args: serde_json::Value) -> AppResult<ToolOutput> {
        let id = resolve_meeting(ctx, &args)?;
        let meta = read_metadata(ctx, id).await?;
        let title = meta.title.clone();
        let data = serde_json::to_value(&meta).map_err(serialise_err)?;
        Ok(ToolOutput::new(data, title))
    }
}

// ---------------------------------------------------------------------------
// get_recording_state
// ---------------------------------------------------------------------------

pub struct GetRecordingState;

#[async_trait]
impl Tool for GetRecordingState {
    fn name(&self) -> &'static str {
        "get_recording_state"
    }
    fn title(&self) -> &'static str {
        "Get recording state"
    }
    fn description(&self) -> &'static str {
        "The recorder's current state (Idle / Recording / Paused / Stopping / Finalising). A busy recorder reports Finalising, never an internal Offline state."
    }
    fn input_schema(&self) -> serde_json::Value {
        schema_obj(json!({}), &[])
    }
    async fn execute(&self, ctx: &ToolContext, _args: serde_json::Value) -> AppResult<ToolOutput> {
        let state: RecordingState = ctx.orchestrator.state().await;
        let label = recording_state_label(&state);
        let data = serde_json::to_value(&state).map_err(serialise_err)?;
        Ok(ToolOutput::new(data, label))
    }
}

fn recording_state_label(state: &RecordingState) -> &'static str {
    match state {
        RecordingState::Idle => "idle",
        RecordingState::Recording { .. } => "recording",
        RecordingState::Paused { .. } => "paused",
        RecordingState::Stopping { .. } => "stopping",
        RecordingState::Finalising { .. } => "finalising",
    }
}

// ---------------------------------------------------------------------------
// search_within_transcript
// ---------------------------------------------------------------------------

pub struct SearchWithinTranscript;

#[async_trait]
impl Tool for SearchWithinTranscript {
    fn name(&self) -> &'static str {
        "search_within_transcript"
    }
    fn title(&self) -> &'static str {
        "Search within transcript"
    }
    fn description(&self) -> &'static str {
        "Find a phrase in one meeting's transcript body (case-insensitive). Returns matching segments with timestamps and speaker."
    }
    fn input_schema(&self) -> serde_json::Value {
        schema_obj(
            json!({
                "meeting_id": { "type": "string", "description": "Meeting UUID" },
                "query": { "type": "string", "description": "Phrase to find (case-insensitive)" },
            }),
            &["meeting_id", "query"],
        )
    }
    async fn execute(&self, ctx: &ToolContext, args: serde_json::Value) -> AppResult<ToolOutput> {
        let id = resolve_meeting(ctx, &args)?;
        let query = require_str(&args, "query")?.to_string();
        let meta = read_metadata(ctx, id).await?;
        let mut transcript = read_transcript(ctx, id).await?;
        apply_speaker_overlay(&mut transcript, &meta.speaker_names);

        let needle = query.to_lowercase();
        let hits: Vec<serde_json::Value> = transcript
            .iter()
            .filter(|s| s.text.to_lowercase().contains(&needle))
            .map(|s| {
                json!({
                    "start_ms": s.start_ms,
                    "end_ms": s.end_ms,
                    "speaker_id": s.speaker_id,
                    "text": s.text,
                    "words": s.words,
                })
            })
            .collect();
        let n = hits.len();
        Ok(ToolOutput::new(
            json!(hits),
            format!("{n} match(es) for {query:?}"),
        ))
    }
}

// ---------------------------------------------------------------------------
// relisten_section
// ---------------------------------------------------------------------------

pub struct RelistenSection;

#[async_trait]
impl Tool for RelistenSection {
    fn name(&self) -> &'static str {
        "relisten_section"
    }
    fn title(&self) -> &'static str {
        "Re-listen to a section"
    }
    fn description(&self) -> &'static str {
        "Re-run ASR over an audio span and return what was actually said there. \
         start_ms/end_ms are transcript-clock timestamps; a window that straddles a \
         recording pause is clamped to the section before the pause. Read-only — \
         it does not modify the stored transcript."
    }
    fn input_schema(&self) -> serde_json::Value {
        schema_obj(
            json!({
                "meeting_id": { "type": "string", "description": "Meeting UUID" },
                "start_ms": { "type": "integer", "minimum": 0, "description": "Span start (transcript-clock ms)" },
                "end_ms": { "type": "integer", "minimum": 0, "description": "Span end (transcript-clock ms)" },
                "language": { "type": "string", "description": "Optional language hint (full English name, e.g. \"English\")" },
            }),
            &["meeting_id", "start_ms", "end_ms"],
        )
    }
    async fn execute(&self, ctx: &ToolContext, args: serde_json::Value) -> AppResult<ToolOutput> {
        let id = resolve_meeting(ctx, &args)?;
        let start = require_u64(&args, "start_ms")?;
        let end = require_u64(&args, "end_ms")?;
        let language = args
            .get("language")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        let segments = ctx
            .orchestrator
            .transcribe_pcm_window(id, start, end, language)
            .await?;
        let n = segments.len();
        let data = serde_json::to_value(&segments).map_err(serialise_err)?;
        Ok(ToolOutput::new(
            data,
            format!("re-listened [{start}, {end}) ms → {n} segment(s)"),
        ))
    }
}

// ---------------------------------------------------------------------------
// resummarise
// ---------------------------------------------------------------------------

/// Fixed compute cap for one `resummarise` run (S2). The held LLM over a full
/// transcript can be slow on a CPU model, so this is deliberately generous; the
/// bound exists so an MCP/bridged caller cannot pin a blocking-pool thread + the
/// held model forever on a wedged decode, not to cut a legitimate run short.
const RESUMMARISE_TIMEOUT: Duration = Duration::from_secs(300);

pub struct Resummarise;

#[async_trait]
impl Tool for Resummarise {
    fn name(&self) -> &'static str {
        "resummarise"
    }
    fn title(&self) -> &'static str {
        "Re-summarise meeting"
    }
    fn description(&self) -> &'static str {
        "Summarise or re-frame this meeting a different way following the given instruction. \
         Returns the new text only; it does NOT overwrite the stored summary."
    }
    fn input_schema(&self) -> serde_json::Value {
        schema_obj(
            json!({
                "meeting_id": { "type": "string", "description": "Meeting UUID" },
                "instruction": { "type": "string", "description": "How to summarise/re-frame (used as the summariser system prompt)" },
            }),
            &["meeting_id", "instruction"],
        )
    }
    async fn execute(&self, ctx: &ToolContext, args: serde_json::Value) -> AppResult<ToolOutput> {
        let id = resolve_meeting(ctx, &args)?;
        let instruction = require_str(&args, "instruction")?.to_string();

        // Read transcript + note blocks (with recording-clock anchors, #70) on a
        // blocking thread.
        let dir = ctx.meeting_dir(id);
        let (transcript, notes) = spawn_blocking_io(move || {
            let transcript = persistence::read_transcript(&dir)?;
            let notes = persistence::read_note_blocks(&dir)?;
            Ok((transcript, notes))
        })
        .await?;

        // Run the held summariser on a blocking thread (sync inference must not
        // run in an async handler). The instruction is the system prompt.
        //
        // Bound the run with a fixed compute timeout (S2): `resummarise` is the
        // one MCP-exposed tool that runs the held LLM over the FULL transcript,
        // and the underlying `summarise` is a single uninterruptible FFI call
        // with no progress callback, so an MCP/bridged caller could otherwise pin
        // a blocking-pool thread + the held model indefinitely on a wedged or
        // pathologically long decode. On timeout we return `AppError::Inference`
        // cleanly; the abandoned `spawn_blocking` thread's result is discarded
        // (tokio cannot cancel it).
        let summariser = ctx.summariser.clone();
        let join = tokio::task::spawn_blocking(move || {
            summariser.summarise(&transcript, &notes, &instruction)
        });
        let text = match tokio::time::timeout(RESUMMARISE_TIMEOUT, join).await {
            Ok(joined) => joined.map_err(|e| AppError::Internal {
                context: format!("resummarise spawn_blocking join failed: {e}"),
            })??,
            Err(_elapsed) => {
                return Err(AppError::Inference {
                    backend: "llm".to_string(),
                    context: format!(
                        "resummarise exceeded its {} s budget",
                        RESUMMARISE_TIMEOUT.as_secs(),
                    ),
                });
            }
        };

        Ok(ToolOutput::new(
            json!({ "text": text }),
            "re-summarised (not saved)",
        ))
    }
}

// ---------------------------------------------------------------------------
// set_speaker_name (WRITE — MCP allowlisted)
// ---------------------------------------------------------------------------

pub struct SetSpeakerName;

#[async_trait]
impl Tool for SetSpeakerName {
    fn name(&self) -> &'static str {
        "set_speaker_name"
    }
    fn title(&self) -> &'static str {
        "Set speaker name"
    }
    fn description(&self) -> &'static str {
        "Name an identified speaker (maps a diarizer label such as \"A\" to a display name). \
         Note: re-running diarization resets all speaker names."
    }
    fn input_schema(&self) -> serde_json::Value {
        schema_obj(
            json!({
                "meeting_id": { "type": "string", "description": "Meeting UUID" },
                "speaker_id": { "type": "string", "description": "Diarizer speaker label, e.g. \"A\"" },
                "name": { "type": "string", "description": "Display name for the speaker" },
            }),
            &["meeting_id", "speaker_id", "name"],
        )
    }
    fn is_write(&self) -> bool {
        true
    }
    /// Allowlisted for MCP exposure (reversible, low blast radius — §4.3).
    fn expose_over_mcp(&self) -> bool {
        true
    }
    async fn execute(&self, ctx: &ToolContext, args: serde_json::Value) -> AppResult<ToolOutput> {
        let id = resolve_meeting(ctx, &args)?;
        // Cap the free-text label + name at dispatch (S4) so an MCP/bridged
        // caller cannot persist an unbounded value into metadata.json.
        let speaker_id = require_bounded_str(&args, "speaker_id")?.to_string();
        let name = require_bounded_str(&args, "name")?.to_string();

        // Hold the per-meeting metadata mutex across the whole read-modify-write
        // so a concurrent set_speaker_name / rename_meeting cannot drop a write
        // (§4.2 class 2). The lock guards the *section*, not the disk file.
        let lock = ctx.metadata_lock(id).await;
        let _guard = lock.lock().await;

        let dir = ctx.meeting_dir(id);
        let speaker_names: BTreeMap<String, String> =
            spawn_blocking_io(move || -> AppResult<BTreeMap<String, String>> {
                let mut meta = persistence::read_metadata(&dir)?;
                meta.speaker_names.insert(speaker_id, name);
                persistence::write_metadata(&dir, &meta)?;
                Ok(meta.speaker_names)
            })
            .await?;

        let data = json!({ "speaker_names": speaker_names });
        Ok(ToolOutput::new(data, "speaker name set"))
    }
}

// ---------------------------------------------------------------------------
// rename_meeting (WRITE — MCP allowlisted)
// ---------------------------------------------------------------------------

pub struct RenameMeeting;

#[async_trait]
impl Tool for RenameMeeting {
    fn name(&self) -> &'static str {
        "rename_meeting"
    }
    fn title(&self) -> &'static str {
        "Rename meeting"
    }
    fn description(&self) -> &'static str {
        "Change a meeting's title."
    }
    fn input_schema(&self) -> serde_json::Value {
        schema_obj(
            json!({
                "meeting_id": { "type": "string", "description": "Meeting UUID" },
                "title": { "type": "string", "description": "New title" },
            }),
            &["meeting_id", "title"],
        )
    }
    fn is_write(&self) -> bool {
        true
    }
    /// Allowlisted for MCP exposure (reversible, low blast radius — §4.3).
    fn expose_over_mcp(&self) -> bool {
        true
    }
    async fn execute(&self, ctx: &ToolContext, args: serde_json::Value) -> AppResult<ToolOutput> {
        let id = resolve_meeting(ctx, &args)?;
        // Cap the title at dispatch (S4): a meeting title is a short label.
        let title = require_bounded_str(&args, "title")?.to_string();

        // rename_meeting also touches metadata.json (title) + upserts the index,
        // so it takes the SAME per-meeting metadata mutex as set_speaker_name (a
        // rename racing a set_speaker_name would otherwise lose a write — §4.2
        // class 3).
        let lock = ctx.metadata_lock(id).await;
        let _guard = lock.lock().await;

        persistence::meeting_ops::rename_meeting(&ctx.meetings_dir, &ctx.index, id, &title).await?;
        Ok(ToolOutput::new(
            json!({ "ok": true }),
            format!("renamed to {title:?}"),
        ))
    }
}

// ---------------------------------------------------------------------------
// retranscribe_meeting (WRITE — internal-only)
// ---------------------------------------------------------------------------

pub struct RetranscribeMeeting;

#[async_trait]
impl Tool for RetranscribeMeeting {
    fn name(&self) -> &'static str {
        "retranscribe_meeting"
    }
    fn title(&self) -> &'static str {
        "Re-transcribe meeting"
    }
    fn description(&self) -> &'static str {
        "Re-run full ASR over the recording, replacing the stored transcript. \
         Fails if a recording or another offline pass is in progress."
    }
    fn input_schema(&self) -> serde_json::Value {
        meeting_id_schema()
    }
    fn is_write(&self) -> bool {
        true
    }
    /// Internal-only (heavy; holding the offline claim via MCP would block the
    /// user's ability to record — §4.3). `expose_over_mcp` therefore stays
    /// `false` (the `is_write` default).
    async fn execute(&self, ctx: &ToolContext, args: serde_json::Value) -> AppResult<ToolOutput> {
        let id = resolve_meeting(ctx, &args)?;
        // Inherits the orchestrator's offline claim: InvalidInput when busy.
        ctx.orchestrator.re_transcribe(&ctx.index, id).await?;
        Ok(ToolOutput::new(json!({ "ok": true }), "re-transcribed"))
    }
}

// ---------------------------------------------------------------------------
// rediarize_meeting (WRITE — internal-only)
// ---------------------------------------------------------------------------

pub struct RediarizeMeeting;

#[async_trait]
impl Tool for RediarizeMeeting {
    fn name(&self) -> &'static str {
        "rediarize_meeting"
    }
    fn title(&self) -> &'static str {
        "Re-diarize meeting"
    }
    fn description(&self) -> &'static str {
        "Re-run speaker diarization, re-assigning speaker labels. This RESETS any \
         configured speaker names (re-lettering can change who each label is). \
         Fails if a recording or another offline pass is in progress."
    }
    fn input_schema(&self) -> serde_json::Value {
        meeting_id_schema()
    }
    fn is_write(&self) -> bool {
        true
    }
    /// Internal-only (heavy; holding the offline claim via MCP would block the
    /// user's ability to record — §4.3). `expose_over_mcp` stays `false`.
    async fn execute(&self, ctx: &ToolContext, args: serde_json::Value) -> AppResult<ToolOutput> {
        let id = resolve_meeting(ctx, &args)?;
        // Inherits the orchestrator's offline claim. The rediarize path clears
        // `speaker_names` in its metadata write (orchestrator §4.4).
        ctx.orchestrator.rediarize(&ctx.index, id).await?;
        Ok(ToolOutput::new(
            json!({ "ok": true }),
            "re-diarized (speaker names reset)",
        ))
    }
}

// ---------------------------------------------------------------------------
// speaker_talk_time
// ---------------------------------------------------------------------------

pub struct SpeakerTalkTime;

#[async_trait]
impl Tool for SpeakerTalkTime {
    fn name(&self) -> &'static str {
        "speaker_talk_time"
    }
    fn title(&self) -> &'static str {
        "Speaker talk time"
    }
    fn description(&self) -> &'static str {
        "Per-speaker total talking time and turn count, with display names overlaid."
    }
    fn input_schema(&self) -> serde_json::Value {
        meeting_id_schema()
    }
    async fn execute(&self, ctx: &ToolContext, args: serde_json::Value) -> AppResult<ToolOutput> {
        let id = resolve_meeting(ctx, &args)?;
        let meta = read_metadata(ctx, id).await?;
        let transcript = read_transcript(ctx, id).await?;

        let stats = aggregate_talk_time(&transcript, &meta.speaker_names);
        let n = stats.len();
        Ok(ToolOutput::new(json!(stats), format!("{n} speaker(s)")))
    }
}

/// Aggregate per-speaker talking time + turn count from `transcript`, overlaying
/// the configured `speaker_names` onto the `display_name` field. The raw
/// `speaker_id` (the diarizer label) is preserved; un-diarized segments
/// (`speaker_id: None`) aggregate under a `"unknown"` bucket. Results are sorted
/// by `speaker_id` for a stable order.
fn aggregate_talk_time(
    transcript: &[Segment],
    speaker_names: &BTreeMap<String, String>,
) -> Vec<serde_json::Value> {
    // (total_ms, turn_count) per raw label.
    let mut by_speaker: BTreeMap<String, (u64, u64)> = BTreeMap::new();
    for seg in transcript {
        let label = seg
            .speaker_id
            .clone()
            .unwrap_or_else(|| "unknown".to_string());
        let dur = seg.end_ms.saturating_sub(seg.start_ms);
        let entry = by_speaker.entry(label).or_insert((0, 0));
        entry.0 += dur;
        entry.1 += 1;
    }
    by_speaker
        .into_iter()
        .map(|(label, (total_ms, turn_count))| {
            let display_name = speaker_names
                .get(&label)
                .cloned()
                .unwrap_or_else(|| label.clone());
            json!({
                "speaker_id": label,
                "display_name": display_name,
                "total_ms": total_ms,
                "turn_count": turn_count,
            })
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Record-control tools (WRITE — MCP write-gated, off by default) — #62
// ---------------------------------------------------------------------------
//
// Four tools that drive the orchestrator's live recording lifecycle so an
// external MCP client can run the record→transcribe→read loop for E2E (and the
// internal UI chat can drive recording too). Each dispatches to an existing
// `Orchestrator` method — `start(device_id)` / `stop()` / `pause()` / `resume()`
// — and adds NO new dependency edge (`agent-tools → orchestrator` pre-exists).
//
// All four set `is_write() == true` AND override `expose_over_mcp() == true`, so
// they are WRITE-GATED over MCP exactly like `set_speaker_name`/`rename_meeting`
// (Group 2): absent from `mcp_tool_descriptors_gated` / rejected by
// `mcp_call_allowed` when `mcp_write_tools` is OFF (the default), and exposed +
// callable only when the user turns `mcp_write_tools` ON — so an external MCP
// client can drive the record→transcribe→read loop for end-to-end testing only
// when explicitly opted in (off by default, behind the bearer token + loopback
// bind). The internal chat agent (no MCP gate) can always dispatch them.

pub struct StartRecording;

#[async_trait]
impl Tool for StartRecording {
    fn name(&self) -> &'static str {
        "start_recording"
    }
    fn title(&self) -> &'static str {
        "Start recording"
    }
    fn description(&self) -> &'static str {
        "Start a new recording. Optionally pick an input device by id (omit for the OS default). \
         Returns the new meeting's id. Fails if a recording is already in progress."
    }
    fn input_schema(&self) -> serde_json::Value {
        // `device_id` is optional (omit → OS default). No regex `pattern` (the
        // GBNF converter rejects PCRE shorthands).
        schema_obj(
            json!({
                "device_id": { "type": "string", "description": "Optional input-device id; omit for the OS default" },
            }),
            &[],
        )
    }
    fn is_write(&self) -> bool {
        true
    }
    fn expose_over_mcp(&self) -> bool {
        true // write-gated: callable over MCP only when `mcp_write_tools` is on
    }
    async fn execute(&self, ctx: &ToolContext, args: serde_json::Value) -> AppResult<ToolOutput> {
        let device_id = args
            .get("device_id")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        let meeting_id = ctx.orchestrator.start(device_id).await?;
        let data = json!({ "meeting_id": meeting_id });
        Ok(ToolOutput::new(
            data,
            format!("recording started ({})", meeting_id.0),
        ))
    }
}

pub struct StopRecording;

#[async_trait]
impl Tool for StopRecording {
    fn name(&self) -> &'static str {
        "stop_recording"
    }
    fn title(&self) -> &'static str {
        "Stop recording"
    }
    fn description(&self) -> &'static str {
        "Stop the current recording and finalise the meeting. Returns the finished meeting's id, \
         title, and duration. Fails if no recording is in progress."
    }
    fn input_schema(&self) -> serde_json::Value {
        schema_obj(json!({}), &[])
    }
    fn is_write(&self) -> bool {
        true
    }
    fn expose_over_mcp(&self) -> bool {
        true // write-gated: callable over MCP only when `mcp_write_tools` is on
    }
    async fn execute(&self, ctx: &ToolContext, _args: serde_json::Value) -> AppResult<ToolOutput> {
        let meta = ctx.orchestrator.stop().await?;
        let data = json!({
            "meeting_id": meta.uuid,
            "title": meta.title,
            "duration_ms": meta.duration_ms,
        });
        Ok(ToolOutput::new(
            data,
            format!("recording stopped ({} — {} ms)", meta.uuid.0, meta.duration_ms),
        ))
    }
}

pub struct PauseRecording;

#[async_trait]
impl Tool for PauseRecording {
    fn name(&self) -> &'static str {
        "pause_recording"
    }
    fn title(&self) -> &'static str {
        "Pause recording"
    }
    fn description(&self) -> &'static str {
        "Pause the current recording. Fails if no recording is in progress."
    }
    fn input_schema(&self) -> serde_json::Value {
        schema_obj(json!({}), &[])
    }
    fn is_write(&self) -> bool {
        true
    }
    fn expose_over_mcp(&self) -> bool {
        true // write-gated: callable over MCP only when `mcp_write_tools` is on
    }
    async fn execute(&self, ctx: &ToolContext, _args: serde_json::Value) -> AppResult<ToolOutput> {
        ctx.orchestrator.pause().await?;
        Ok(ToolOutput::new(json!({ "ok": true }), "recording paused"))
    }
}

pub struct ResumeRecording;

#[async_trait]
impl Tool for ResumeRecording {
    fn name(&self) -> &'static str {
        "resume_recording"
    }
    fn title(&self) -> &'static str {
        "Resume recording"
    }
    fn description(&self) -> &'static str {
        "Resume a paused recording. Fails if the recording is not currently paused."
    }
    fn input_schema(&self) -> serde_json::Value {
        schema_obj(json!({}), &[])
    }
    fn is_write(&self) -> bool {
        true
    }
    fn expose_over_mcp(&self) -> bool {
        true // write-gated: callable over MCP only when `mcp_write_tools` is on
    }
    async fn execute(&self, ctx: &ToolContext, _args: serde_json::Value) -> AppResult<ToolOutput> {
        ctx.orchestrator.resume().await?;
        Ok(ToolOutput::new(json!({ "ok": true }), "recording resumed"))
    }
}

// ---------------------------------------------------------------------------
// send_to_internal_agent (MCP-only — Phase 10 §3)
// ---------------------------------------------------------------------------

/// How long the bridge waits for the internal agent's reply before giving up.
///
/// A wedged internal turn must not pin an MCP connection forever (§3 / W4). The
/// timeout surfaces as `InvalidInput { "internal agent timed out" }`. Sized for
/// a multi-tool chat turn on a CPU model (the held LLM can be slow); the MCP
/// client sees a clean error rather than a hang.
const INTER_AGENT_TIMEOUT: Duration = Duration::from_secs(180);

/// `send_to_internal_agent` — the one MCP-only tool (Phase 10 §3). Registered
/// only on the MCP registry (`ToolRegistry::v1(true)`); absent from the internal
/// chat agent's registry so the agent cannot message itself.
///
/// v1 is a SYNCHRONOUS request/reply: the call sends one [`InterAgentRequest`]
/// to the `ipc-bridge` driver task, which runs ONE chat turn on the held model
/// via the SAME `run_chat_turn` driver the UI uses, and returns the assistant's
/// final text. The tool itself holds no chat engine — it only forwards over the
/// bridge channel on the [`ToolContext`], which is what keeps `agent-tools` free
/// of a `chat-agent` edge.
pub struct SendToInternalAgent;

#[async_trait]
impl Tool for SendToInternalAgent {
    fn name(&self) -> &'static str {
        "send_to_internal_agent"
    }
    fn title(&self) -> &'static str {
        "Message the in-app assistant"
    }
    fn description(&self) -> &'static str {
        "Send a message to this app's internal meeting-notes chat agent and get its reply. \
         Optionally scope it to a meeting (meeting_id) and/or continue an existing \
         conversation (session_id). Runs one synchronous turn of the internal agent."
    }
    fn input_schema(&self) -> serde_json::Value {
        // `message` is required; `meeting_id` / `session_id` are optional scoping
        // hints. No regex `pattern` (the GBNF converter rejects PCRE shorthands).
        json!({
            "type": "object",
            "properties": {
                "message": { "type": "string", "description": "The message to send to the internal agent" },
                "meeting_id": { "type": "string", "description": "Optional meeting UUID to scope the conversation to" },
                "session_id": { "type": "string", "description": "Optional existing chat session id to continue" },
            },
            "required": ["message"],
            "additionalProperties": false,
        })
    }
    /// Not a write: it does not itself mutate disk/index/state (the turn it
    /// triggers may, through other tools, but those go through the internal
    /// agent's own write-serialization). Exposed over MCP regardless of the
    /// `mcp_write_tools` gate (it is read-like to the MCP surface).
    fn is_write(&self) -> bool {
        false
    }
    async fn execute(&self, ctx: &ToolContext, args: serde_json::Value) -> AppResult<ToolOutput> {
        let bridge = ctx
            .inter_agent_bridge()
            .ok_or_else(|| AppError::InvalidInput {
                context: "send_to_internal_agent is not available in this context".into(),
            })?;

        let message = require_str(&args, "message")?.to_string();
        // Optional scoping. A malformed id surfaces as InvalidInput.
        let meeting_id = match args.get("meeting_id").and_then(|v| v.as_str()) {
            Some(s) => Some(crate::parse_meeting_id(s)?),
            None => None,
        };
        let session_id = match args.get("session_id").and_then(|v| v.as_str()) {
            Some(s) => Some(parse_session_id(s)?),
            None => None,
        };

        let request = InterAgentRequest {
            session_id,
            meeting_id,
            message,
        };

        let (reply_tx, reply_rx) = oneshot::channel();

        // try_send (NOT send): a full queue means the internal agent is busy;
        // return promptly rather than blocking the HTTP request thread (§3 / W4).
        bridge.try_send((request, reply_tx)).map_err(|e| match e {
            tokio::sync::mpsc::error::TrySendError::Full(_) => AppError::InvalidInput {
                context: "internal agent busy".into(),
            },
            tokio::sync::mpsc::error::TrySendError::Closed(_) => AppError::Internal {
                context: "internal agent bridge is closed".into(),
            },
        })?;

        // Await the reply with a timeout so a wedged turn cannot pin the
        // connection forever (§3 / W4).
        let reply = match tokio::time::timeout(INTER_AGENT_TIMEOUT, reply_rx).await {
            Ok(Ok(result)) => result?,
            Ok(Err(_recv_err)) => {
                // The driver dropped the oneshot without replying (it panicked
                // or shut down mid-turn).
                return Err(AppError::Internal {
                    context: "internal agent dropped the request without replying".into(),
                });
            }
            Err(_elapsed) => {
                return Err(AppError::InvalidInput {
                    context: "internal agent timed out".into(),
                });
            }
        };

        let summary = format!("internal agent replied ({} chars)", reply.reply.len());
        let data = json!({
            "reply": reply.reply,
            "session_id": reply.session_id,
        });
        Ok(ToolOutput::new(data, summary))
    }
}

/// Parse a `ChatSessionId` from its hyphenated-UUID wire string. `InvalidInput`
/// on a malformed id (mirrors [`crate::parse_meeting_id`]).
fn parse_session_id(s: &str) -> AppResult<minutist_common::ChatSessionId> {
    serde_json::from_str::<minutist_common::ChatSessionId>(&format!("\"{s}\"")).map_err(|_| {
        AppError::InvalidInput {
            context: format!("invalid session_id: {s}"),
        }
    })
}

// ---------------------------------------------------------------------------
// Error mapping
// ---------------------------------------------------------------------------

/// Map a `serde_json` serialisation failure to `AppError::Internal`. The
/// payloads here are app-internal types that always serialise, so this only
/// fires on a genuine bug.
fn serialise_err(e: serde_json::Error) -> AppError {
    AppError::Internal {
        context: format!("tool result serialisation failed: {e}"),
    }
}
