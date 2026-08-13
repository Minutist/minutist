//! Summary commands (Phase 5): produce, read, and edit a meeting's `summary.md`.
//!
//! Command bodies are factored out from their `#[tauri::command]` wrappers so they are
//! unit-testable without a Tauri runtime, a real model, or the orchestrator — the
//! summarise inner takes a `&dyn Summariser` so a `StubSummariser` exercises the
//! read -> summarise -> write -> event wiring (mirroring the orchestrator's
//! re_transcribe stub-backend seam).
use super::*;


/// Run summarisation for a meeting (FR-30): produce `summary.md` and emit
/// `AppEvent::SummaryReady`.
///
/// Pipeline:
/// 1. Resolve the LLM model id — `settings.llm_model_id` if set, else the
///    bundled default [`DEFAULT_LLM_MODEL_ID`].
/// 2. Resolve the model **directory** via `Orchestrator::ensure_model_path`
///    (downloads + verifies when absent). This keeps the `model-registry` edge
///    in the orchestrator — there is **no** `orchestrator → summariser` edge;
///    the summariser is loaded here, in `ipc-bridge` (the granted
///    `ipc-bridge → summariser` edge).
/// 3. Open a [`LlamaSummariser`] over the single `.gguf` in that directory
///    (skipping any `mmproj-*`), read the transcript + notes markdown, run the
///    summary, and write `summary.md` — **all on `spawn_blocking`** because the
///    summariser's `open` + `summarise` are heavy and synchronous (the
///    threading-model rule: inference on `spawn_blocking`, never in a handler).
/// 4. Emit `AppEvent::SummaryReady { meeting_id }` on the shared `event_tx` so
///    the summary view re-reads the persisted markdown.
///
/// The model resolution (`ensure_model_path`) is awaited on the async worker
/// (it may download); only the synchronous summariser work runs on
/// `spawn_blocking`.
#[tauri::command]
#[specta::specta]
pub async fn summarise_meeting(
    meeting_id: MeetingId,
    state: State<'_, IpcState>,
) -> AppResult<()> {
    // The whole summarise body (held-model resolve → summarise-with-progress →
    // index refresh → `SummaryReady`) is extracted into [`run_held_summarise`] so
    // BOTH this user-triggered command and the post-stop auto-summarise chain
    // (#68, see [`stop_recording`]) drive the SAME path. The command surfaces an
    // error to the webview; the chain treats it as best-effort.
    run_held_summarise(&state.chat_handles(), meeting_id).await?;
    Ok(())
}

/// Resolve the held summariser, summarise a meeting (streaming determinate
/// `OperationProgress`), refresh the meeting-list index excerpt, and emit
/// `AppEvent::SummaryReady`.
///
/// Extracted from the `summarise_meeting` command (#68) so it can be invoked
/// from the post-stop background chain ([`run_post_stop_passes`]) as well as from
/// the command. Takes a [`ChatHandles`] — the same bundle the chat path uses —
/// so it shares the SAME lazily-loaded held model `Arc` (no GGUF reload).
///
/// `pub` (re-exported at the crate root) so `app-main`'s `DesktopElectionDriver`
/// (`src-tauri/src/sync.rs`, producer-gate F4-summary) can drive the same
/// summarise pass for a delegated meeting after `Orchestrator::reprocess`,
/// without a Tauri `State` — it only needs a [`ChatHandles`], which its own
/// caller constructs directly (mirroring the pattern `app-main` already uses for
/// `GemmaVlm`).
///
/// Returns the summary markdown on success. The heavy `summarise` runs on
/// `spawn_blocking`, per the threading-model rule.
pub async fn run_held_summarise(
    handles: &ChatHandles,
    meeting_id: MeetingId,
) -> AppResult<String> {
    let current = handles.settings.current();
    let event_tx = &handles.event_tx;

    // #69: surface the model LOAD as an indeterminate phase BEFORE it starts.
    // `ensure_summariser` mmaps + warms the multi-GB GGUF the FIRST time it is
    // called (and the first summarise of a session — including the post-stop
    // auto-summarise — pays it). On a warm load this flashes by; on a cold load
    // it is the bulk of the wait the user otherwise saw as a stuck 0%.
    emit_summarise_op(
        event_tx,
        meeting_id,
        None,
        "Loading the summarisation model…",
    );

    // Phase 9 (C2): use the HELD summariser substrate — loaded once on first
    // chat/summarise use and shared with the chat agent — instead of opening a
    // fresh `LlamaSummariser` per call (the old per-call GGUF reload killed
    // latency). The load resolves the model id + directory via the orchestrator
    // (keeping the `model-registry` edge there) and opens the GGUF on
    // `spawn_blocking` the FIRST time only; thereafter this is a cheap clone.
    let summariser = handles.ensure_summariser().await?;

    let meetings_dir_owned = handles.meetings_dir.clone();

    // Reference material (attachments): assemble every Ready attachment's
    // converted markdown (manifest order, each under `## Attachment: <name>`) via
    // `persistence`, then deterministically truncate it to a character budget
    // derived from the held model's `n_ctx` minus a reserve for the transcript +
    // notes + generation. The summariser renders it as a LEADING
    // `# Reference material (attachments)` section (NOT time-woven). An empty
    // result is byte-identical to the no-attachment path. The read is blocking
    // `std::fs`, so it runs on `spawn_blocking`.
    let attachments_budget = attachments_markdown_budget_chars(summariser.config().n_ctx);
    let meetings_dir_for_attach = handles.meetings_dir.clone();
    let attachment_parts = tokio::task::spawn_blocking(move || {
        persistence::read_attachments_markdown_parts(&meetings_dir_for_attach, meeting_id)
    })
    .await
    .map_err(|e| AppError::Internal {
        context: format!("attachments assembly task join failed: {e}"),
    })?
    // A failure to read the manifest must not block summarising — the summary is
    // still useful without attachments. Log and fall back to no reference
    // material (best-effort, matching the conversion worker's posture).
    .unwrap_or_else(|e| {
        tracing::warn!(
            target: "ipc-bridge",
            meeting_id = %meeting_id.0,
            "assembling attachment reference material failed: {e}; summarising without it"
        );
        Vec::new()
    });
    let attachments_markdown = assemble_attachments_markdown(attachment_parts, attachments_budget);
    // Resolve the preset-aware effective prompt (Phase 9 — D4): the user's
    // custom `summary_system_prompt` override when non-empty, else the built-in
    // prompt for the selected `summary_preset`. The output-language instruction
    // is appended last so it wins over any conflicting text in a custom prompt.
    let system_prompt = apply_output_language(
        &current.effective_summary_prompt(),
        &current.output_language,
    );

    // Heavy, synchronous summarise work on a blocking thread (the held model's
    // `summarise` builds a fresh `LlamaContext` and decodes). The held handle is
    // cloned into the closure; no GGUF reload. The held substrate is the concrete
    // `LlamaSummariser`, so we drive `summarise_with_progress` directly (the
    // `common::Summariser` trait is unchanged — see `summariser`).
    let event_tx_for_blocking = event_tx.clone();
    let summary_md = tokio::task::spawn_blocking(move || {
        summarise_meeting_with_progress(
            &meetings_dir_owned,
            meeting_id,
            summariser.as_ref(),
            &attachments_markdown,
            &system_prompt,
            &event_tx_for_blocking,
        )
    })
    .await
    .map_err(|e| AppError::Internal {
        context: format!("summarise_meeting task join failed: {e}"),
    })??;

    // Live-test UX T6: refresh the meeting-list index row so its excerpt becomes
    // the summary blurb now that `summary.md` exists (the persistence excerpt
    // derivation prefers the summary blurb over the first transcript segment). A
    // failure is logged and swallowed — the index is a derived cache the next
    // startup rebuild reconciles, so a failed refresh must not fail the summary.
    let meetings_dir_for_entry = handles.meetings_dir.clone();
    match tokio::task::spawn_blocking(move || {
        meeting_list_entry_for_meta_with_summary(&meetings_dir_for_entry, meeting_id)
    })
    .await
    {
        Ok(Ok(Some(entry))) => {
            if let Err(e) = handles.index.upsert(&entry).await {
                tracing::warn!(
                    target: "ipc-bridge",
                    meeting_id = %meeting_id.0,
                    "index excerpt refresh after summarise failed: {e}; \
                     reconciled on next startup"
                );
            }
        }
        Ok(Ok(None)) => {
            // Metadata unreadable (the meeting folder was deleted mid-summarise);
            // nothing to upsert.
        }
        Ok(Err(e)) => tracing::warn!(
            target: "ipc-bridge",
            meeting_id = %meeting_id.0,
            "building index entry after summarise failed: {e}; reconciled on next startup"
        ),
        Err(join_err) => tracing::warn!(
            target: "ipc-bridge",
            meeting_id = %meeting_id.0,
            "index entry build after summarise join failed: {join_err}; \
             reconciled on next startup"
        ),
    }

    // Emit SummaryReady so the webview re-reads `summary.md` (this also clears
    // the per-row progress indicator and refreshes the list excerpt).
    emit_summary_ready(event_tx, meeting_id);

    tracing::info!(
        target: "ipc-bridge",
        meeting_id = %meeting_id.0,
        summary_len = summary_md.len(),
        "summary generated and persisted"
    );

    Ok(summary_md)
}

/// Build the [`MeetingListEntry`] for a meeting from its metadata, deriving the
/// excerpt from `summary.md` (the T6 blurb) when one exists, else the first
/// transcript segment — via `persistence::summary_blurb` + `read_transcript`.
///
/// Returns `Ok(None)` when the metadata cannot be read (the folder is gone).
/// Blocking `std::fs` reads; the caller drives it on `spawn_blocking`.
fn meeting_list_entry_for_meta_with_summary(
    meetings_dir: &Path,
    meeting_id: MeetingId,
) -> Result<Option<MeetingListEntry>, AppError> {
    let meeting_dir = meetings_dir.join(meeting_id.0.to_string());
    let meta = match persistence::read_metadata(&meeting_dir) {
        Ok(m) => m,
        Err(_) => return Ok(None),
    };

    // Prefer the summary blurb; fall back to the first transcript segment.
    let excerpt = persistence::read_summary(&meeting_dir)
        .ok()
        .flatten()
        .and_then(|md| persistence::summary_blurb(&md))
        .or_else(|| {
            persistence::read_transcript(&meeting_dir)
                .ok()
                .and_then(|segs| segs.first().map(|s| s.text.clone()))
        });

    Ok(Some(MeetingListEntry {
        id: meta.uuid,
        title: meta.title,
        started_at: meta.started_at,
        duration_ms: meta.duration_ms,
        speaker_count: meta.speaker_count,
        excerpt,
        collection_id: meta.collection_id,
        recording_started: meta.recording_started,
        deleted_at: meta.deletion.deleted_at(),
    }))
}

/// Production summarise body that streams DETERMINATE `OperationProgress` events
/// (live-test UX T4(b)).
///
/// Mirrors [`summarise_meeting_inner`] (read transcript + notes → summarise →
/// write `summary.md`) but drives the concrete [`LlamaSummariser`]'s
/// `summarise_with_progress` so the generation loop reports `(tokens_generated,
/// max_tokens)`; the callback emits a throttled (~5 Hz) determinate
/// `AppEvent::OperationProgress`. Synchronous — the caller drives it on
/// `spawn_blocking`. (`summarise_meeting_inner` keeps the trait-based
/// `&dyn Summariser` seam for the stub unit tests, which assert the read/write
/// wiring without progress.)
fn summarise_meeting_with_progress(
    meetings_dir: &Path,
    meeting_id: MeetingId,
    summariser: &LlamaSummariser,
    attachments_markdown: &str,
    system_prompt: &str,
    event_tx: &broadcast::Sender<AppEvent>,
) -> Result<String, AppError> {
    let meeting_dir = meetings_dir.join(meeting_id.0.to_string());

    let mut transcript = persistence::read_transcript(&meeting_dir)?;
    // Overlay user-set speaker names onto a copy before summarising (same as the
    // non-progress path); the on-disk transcript keeps the raw labels.
    let meta = persistence::read_metadata(&meeting_dir)?;
    minutist_common::apply_speaker_overlay(&mut transcript, &meta.speaker_names);
    // #70: load the note paragraphs WITH their recording-clock anchors so the
    // summariser can weave them into the transcript at the time they were
    // written (rather than appending a flat markdown block).
    let notes = persistence::read_note_blocks(&meeting_dir)?;

    // #69: the model is loaded; the next opaque wait is building the
    // `LlamaContext` (cold GPU shader compile, tens of seconds on first use)
    // before the first prefill tick. Show an indeterminate "Preparing…" until
    // the prefill phase starts reporting.
    emit_summarise_op(event_tx, meeting_id, None, "Preparing the model…");

    // Throttle the callback to ~5 Hz so a fast GPU run does not flood the
    // broadcast bus (the meter already runs at ~30 Hz on the same channel), but
    // ALWAYS emit on a phase change (prefill→generate) and on completion so the
    // label flips promptly and the bar reaches 100%.
    let mut last_emit = std::time::Instant::now();
    let mut last_phase: u8 = 0;
    let summary_md = summariser.summarise_with_progress(
        &transcript,
        &notes,
        attachments_markdown,
        system_prompt,
        |progress| {
            let (phase, fraction, label): (u8, f32, &str) = match progress {
                SummariseProgress::Prefill { done, total } => (
                    1,
                    if total == 0 {
                        1.0
                    } else {
                        done as f32 / total as f32
                    },
                    "Reading the meeting…",
                ),
                SummariseProgress::Generate { done, max } => (
                    2,
                    if max == 0 {
                        1.0
                    } else {
                        done as f32 / max as f32
                    },
                    "Writing the summary…",
                ),
            };
            let now = std::time::Instant::now();
            let phase_changed = phase != last_phase;
            let complete = fraction >= 1.0;
            if phase_changed
                || complete
                || now.duration_since(last_emit) >= std::time::Duration::from_millis(200)
            {
                last_emit = now;
                last_phase = phase;
                emit_summarise_op(event_tx, meeting_id, Some(fraction), label);
            }
        },
    )?;

    persistence::write_summary(&meeting_dir, &summary_md)?;

    Ok(summary_md)
}

/// Emit an `AppEvent::OperationProgress` for the summarise op (#69).
///
/// `fraction` is `Some` for a determinate bar (clamped to `0..=1`) or `None`
/// for an indeterminate spinner (the model-load / context-prepare phases that
/// have no progress callback). `label` names the phase so the bar explains the
/// wait rather than sitting at a silent 0%.
fn emit_summarise_op(
    event_tx: &broadcast::Sender<AppEvent>,
    meeting_id: MeetingId,
    fraction: Option<f32>,
    label: &str,
) {
    let _ = event_tx.send(AppEvent::OperationProgress {
        meeting_id,
        op: OperationKind::Summarise,
        fraction: fraction.map(|f| f.clamp(0.0, 1.0)),
        label: label.to_string(),
    });
}

/// Read a meeting's persisted summary markdown, or `None` when none exists.
///
/// Routes directly to `persistence::read_summary` (a blocking `std::fs` read on
/// `spawn_blocking`).
#[tauri::command]
#[specta::specta]
pub async fn get_summary(
    meeting_id: MeetingId,
    state: State<'_, IpcState>,
) -> AppResult<Option<String>> {
    let meetings_dir = state.meetings_dir.clone();
    tokio::task::spawn_blocking(move || get_summary_inner(&meetings_dir, meeting_id))
        .await
        .map_err(|e| AppError::Internal {
            context: format!("get_summary task join failed: {e}"),
        })?
}

/// Persist an edited summary back to `summary.md` (FR-30).
///
/// Routes directly to `persistence::write_summary` (atomic tmp+rename) on
/// `spawn_blocking`.
#[tauri::command]
#[specta::specta]
pub async fn save_summary(
    meeting_id: MeetingId,
    summary_markdown: String,
    state: State<'_, IpcState>,
) -> AppResult<()> {
    let meetings_dir = state.meetings_dir.clone();
    tokio::task::spawn_blocking(move || {
        save_summary_inner(&meetings_dir, meeting_id, &summary_markdown)
    })
    .await
    .map_err(|e| AppError::Internal {
        context: format!("save_summary task join failed: {e}"),
    })?
}

// ---------------------------------------------------------------------------
// Summary command bodies — extracted so they can be unit-tested without a
// Tauri runtime, a real model, or the orchestrator. The summarise inner takes
// a `&dyn Summariser` so a `StubSummariser` exercises the read → summarise →
// write → event wiring (mirroring the orchestrator's re_transcribe stub-backend
// seam).
// ---------------------------------------------------------------------------

/// Inner body of [`summarise_meeting`]: read the meeting's transcript + notes
/// markdown, run `summariser`, and write `summary.md`. Returns the summary
/// markdown so the caller can log its length and the test can assert on it.
///
/// Synchronous (blocking `std::fs` + sync `Summariser::summarise`) — the caller
/// drives it on `spawn_blocking`. Takes `&dyn Summariser` so a stub can be
/// injected in tests.
///
/// `#[cfg(test)]`: the production command path now drives the concrete
/// [`summarise_meeting_with_progress`] (which streams `OperationProgress`, T4(b));
/// this trait-based seam is retained only for the stub unit tests that assert the
/// read → summarise → write wiring without a model or progress events.
#[cfg(test)]
pub(crate) fn summarise_meeting_inner(
    meetings_dir: &Path,
    meeting_id: MeetingId,
    summariser: &dyn Summariser,
    system_prompt: &str,
) -> Result<String, AppError> {
    let meeting_dir = meetings_dir.join(meeting_id.0.to_string());

    let mut transcript = persistence::read_transcript(&meeting_dir)?;
    // Summarise with the user-set speaker names ("Alice") rather than the raw
    // diarizer labels ("A"), matching every agent read tool. Overlay a copy —
    // the on-disk transcript keeps the raw labels. (A summary already on disk
    // keeps whatever labels it was generated with until it is regenerated.)
    let meta = persistence::read_metadata(&meeting_dir)?;
    minutist_common::apply_speaker_overlay(&mut transcript, &meta.speaker_names);
    // Note paragraphs with recording-clock anchors (#70); empty when the meeting
    // has none.
    let notes = persistence::read_note_blocks(&meeting_dir)?;

    // The trait-based test seam carries no attachment context; `""` keeps the
    // stub path byte-identical to the pre-attachments summarise behaviour. The
    // production path ([`run_held_summarise`]) assembles the real
    // attachments_markdown and drives `summarise_with_progress`.
    let summary_md = summariser.summarise(&transcript, &notes, "", system_prompt)?;

    persistence::write_summary(&meeting_dir, &summary_md)?;

    Ok(summary_md)
}

/// Inner body of [`get_summary`]: read `summary.md` via `persistence`.
pub(crate) fn get_summary_inner(
    meetings_dir: &Path,
    meeting_id: MeetingId,
) -> Result<Option<String>, AppError> {
    let meeting_dir = meetings_dir.join(meeting_id.0.to_string());
    persistence::read_summary(&meeting_dir)
}

/// Inner body of [`save_summary`]: write `summary.md` via `persistence`.
pub(crate) fn save_summary_inner(
    meetings_dir: &Path,
    meeting_id: MeetingId,
    summary_markdown: &str,
) -> Result<(), AppError> {
    let meeting_dir = meetings_dir.join(meeting_id.0.to_string());
    persistence::write_summary(&meeting_dir, summary_markdown)
}

/// Character budget for the assembled attachment reference material, derived
/// from the held model's context window (`n_ctx`).
///
/// The summariser prompt must hold the transcript, the notes, the system prompt,
/// the reference material, AND leave room for the generated summary. We reserve
/// a fixed fraction of the window for the reference material and convert tokens →
/// characters with a coarse heuristic (~4 chars/token). Semantic retrieval over
/// the reference material is deferred (#0016); this character budget is the
/// deterministic guard that keeps attachments from pushing the transcript out of
/// the window. The per-attachment split lives in
/// [`assemble_attachments_markdown`]. Pure + unit-tested.
pub(crate) fn attachments_markdown_budget_chars(n_ctx: u32) -> usize {
    // Reserve ~40% of the window for reference material; the remaining ~60%
    // covers the transcript + notes + system prompt + the generated summary.
    const REFERENCE_FRACTION: u64 = 40; // percent
    const CHARS_PER_TOKEN: u64 = 4;
    ((n_ctx as u64) * REFERENCE_FRACTION / 100 * CHARS_PER_TOKEN) as usize
}

/// Assemble the reference-material markdown from each `Ready` attachment's
/// `(original_filename, markdown)`, rendered under a `## Attachment: <name>`
/// header in manifest order.
///
/// When the untruncated assembly fits `budget_chars` it is returned unchanged
/// (an empty `parts` yields `""`, byte-identical to the no-attachment path).
/// When it would overflow, each attachment gets an EQUAL share of the budget so
/// a single large attachment cannot starve the others; that per-attachment share
/// is charged for the rendered `## Attachment: <name>` header (and surrounding
/// separators) too, so a pathologically long filename eats into its own share
/// rather than silently overrunning the budget. Whatever share remains caps the
/// BODY; a body trimmed to fit carries a visible `[truncated]` marker
/// (UTF-8-boundary safe, never mid-codepoint). Semantic retrieval is deferred
/// (#0016). Pure + unit-tested.
pub(crate) fn assemble_attachments_markdown(
    parts: Vec<(String, String)>,
    budget_chars: usize,
) -> String {
    if parts.is_empty() {
        return String::new();
    }

    const MARKER: &str = "\n\n[truncated]\n";
    let marker_len = MARKER.chars().count();

    // Char count of the fixed framing rendered around each attachment's body:
    // the `## Attachment: ` prefix, the `\n\n` after the filename, and the
    // trailing `\n` (the inter-attachment `\n` separator is counted on the
    // header side via `i > 0`). The filename itself is added per-attachment.
    let header_overhead = |filename: &str, with_separator: bool| -> usize {
        let mut n = "## Attachment: ".chars().count()
            + filename.chars().count()
            + "\n\n".chars().count()
            + "\n".chars().count();
        if with_separator {
            n += 1; // the leading '\n' between attachments
        }
        n
    };

    // Render the block, optionally capping each attachment's BODY at `body_cap`
    // chars (`None` = full body, no truncation).
    let assemble = |body_cap: Option<usize>| -> String {
        let mut out = String::new();
        for (i, (filename, md)) in parts.iter().enumerate() {
            if i > 0 {
                out.push('\n');
            }
            out.push_str("## Attachment: ");
            out.push_str(filename);
            out.push_str("\n\n");
            match body_cap {
                Some(cap) if md.chars().count() > cap => {
                    let keep = cap.saturating_sub(marker_len);
                    let trimmed: String = md.chars().take(keep).collect();
                    out.push_str(&trimmed);
                    out.push_str(MARKER);
                }
                _ => out.push_str(md),
            }
            out.push('\n');
        }
        out
    };

    let full = assemble(None);
    if full.chars().count() <= budget_chars {
        return full;
    }
    // Overflow: give every attachment an EQUAL share of the budget, then charge
    // each attachment's rendered header (and separator) against its own share so
    // a long filename cannot push the total past the budget. The remainder caps
    // the body (at least one char so the body marker still renders).
    let share = (budget_chars / parts.len()).max(1);
    let body_cap = parts
        .iter()
        .enumerate()
        .map(|(i, (filename, _))| {
            share
                .saturating_sub(header_overhead(filename, i > 0))
                .max(1)
        })
        .min()
        .unwrap_or(1);
    assemble(Some(body_cap))
}

/// Resolve the summariser `n_gpu_layers` from the runtime `gpu_acceleration`
/// setting.
///
/// GPU offload happens ONLY when BOTH (a) the build was compiled with a GPU
/// feature AND (b) the setting is on. `enabled == true` → the compile-time
/// ceiling [`summariser::gpu_layers`] (already `0` in a default CPU-only build,
/// so a CPU build is unaffected by the flag); `enabled == false` → `0` (force
/// CPU even in a GPU build). Pure + unit-tested so the wiring is verified
/// without a model. See `architecture/cross-cutting.md` — "GPU portability".
pub(crate) fn resolve_summariser_gpu_layers(enabled: bool) -> u32 {
    if enabled {
        summariser::gpu_layers()
    } else {
        0
    }
}

/// Open a [`LlamaSummariser`] over the single `.gguf` weights file in
/// `model_dir`, skipping any `mmproj-*` multimodal projector.
///
/// `n_gpu_layers` is the runtime-resolved GPU-offload count (see
/// [`resolve_summariser_gpu_layers`]); it is set on the `SummariserConfig` so
/// the summariser honours the `gpu_acceleration` setting.
///
/// The LLM is text-only (the bundled Gemma 4 GGUF ships without a projector),
/// but the helper defends against a directory that also contains an `mmproj-*`
/// file so the wrong file is never loaded. A missing or ambiguous weights file
/// is an `AppError::ModelLoad`.
pub(crate) fn open_summariser_in_dir(
    model_dir: &Path,
    n_gpu_layers: u32,
) -> Result<LlamaSummariser, AppError> {
    let gguf_path = find_gguf_weights(model_dir)?;
    let config = SummariserConfig {
        n_gpu_layers,
        ..SummariserConfig::default()
    };
    LlamaSummariser::open(gguf_path, config)
}

/// Locate the single non-`mmproj` `.gguf` file in `model_dir`.
pub(crate) fn find_gguf_weights(model_dir: &Path) -> Result<std::path::PathBuf, AppError> {
    let read_dir = std::fs::read_dir(model_dir).map_err(|e| AppError::ModelLoad {
        model_id: model_dir.display().to_string(),
        context: format!("cannot read model directory: {e}"),
    })?;

    let mut candidates: Vec<std::path::PathBuf> = Vec::new();
    for entry in read_dir.flatten() {
        let path = entry.path();
        let name = match path.file_name().and_then(|n| n.to_str()) {
            Some(n) => n,
            None => continue,
        };
        let is_gguf = path
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| e.eq_ignore_ascii_case("gguf"));
        // Skip the multimodal projector — the summariser uses the text weights.
        if is_gguf && !name.to_ascii_lowercase().starts_with("mmproj") {
            candidates.push(path);
        }
    }

    match candidates.len() {
        1 => Ok(candidates.pop().expect("len == 1")),
        0 => Err(AppError::ModelLoad {
            model_id: model_dir.display().to_string(),
            context: "no .gguf weights file found in model directory".into(),
        }),
        n => Err(AppError::ModelLoad {
            model_id: model_dir.display().to_string(),
            context: format!("expected one .gguf weights file, found {n}"),
        }),
    }
}

/// Emit `AppEvent::SummaryReady { meeting_id }` on the shared broadcast sender.
///
/// A send with no live subscribers is not an error (broadcast semantics); it is
/// logged at trace, mirroring `Orchestrator::emit`.
pub(crate) fn emit_summary_ready(event_tx: &broadcast::Sender<AppEvent>, meeting_id: MeetingId) {
    match event_tx.send(AppEvent::SummaryReady { meeting_id }) {
        Ok(n) => tracing::trace!(
            target: "ipc-bridge",
            receivers = n,
            "SummaryReady broadcast"
        ),
        Err(_) => tracing::trace!(
            target: "ipc-bridge",
            "SummaryReady dropped (no subscribers)"
        ),
    }
}

/// Emit `AppEvent::SummaryQueued { meeting_id }` — a post-stop auto-summary is
/// scheduled (the webview shows the summary pane busy until a terminal
/// `SummaryReady` / `SummaryUnavailable`). A send with no live subscribers is not
/// an error (broadcast semantics).
pub(crate) fn emit_summary_queued(event_tx: &broadcast::Sender<AppEvent>, meeting_id: MeetingId) {
    let _ = event_tx.send(AppEvent::SummaryQueued { meeting_id });
}

/// Emit `AppEvent::SummaryUnavailable { meeting_id }` — a post-stop auto-summary
/// was deferred (a new recording started) or failed, so no `summary.md` was
/// written. The webview clears the busy state and offers the manual `Summarise`
/// action. A send with no live subscribers is not an error.
pub(crate) fn emit_summary_unavailable(event_tx: &broadcast::Sender<AppEvent>, meeting_id: MeetingId) {
    let _ = event_tx.send(AppEvent::SummaryUnavailable { meeting_id });
}

