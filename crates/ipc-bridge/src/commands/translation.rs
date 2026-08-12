//! Translation commands: the translated transcript as a derived, regenerable view.
use super::*;


/// The set of supported target-language names for `translate_meeting`.
///
/// Matches the values in `output_language::SUBTAG_TO_LANGUAGE`. `translate_meeting`
/// validates its `target_language` argument against this set (case-sensitive full
/// English language names) so the LLM prompt is never built for a language the
/// picker does not expose. Values sorted alphabetically.
const SUPPORTED_TRANSLATION_LANGUAGES: &[&str] = &[
    "Arabic",
    "Chinese",
    "Dutch",
    "English",
    "French",
    "German",
    "Hindi",
    "Italian",
    "Japanese",
    "Korean",
    "Polish",
    "Portuguese",
    "Russian",
    "Spanish",
    "Turkish",
];

/// Translate every segment of a meeting's transcript into `target_language` and
/// persist the results in `translations.json` alongside the meeting folder.
///
/// The translation is a **derived** view: the verbatim transcript remains
/// authoritative; `translations.json` is regenerable from it at any time. A
/// new full `write_transcript` (re-transcribe) automatically clears the sidecar
/// so stale translations never linger — see `persistence::write_transcript`.
///
/// # Concurrency
///
/// A second call for the same `(meeting_id, target_language)` pair while one is
/// already in flight is rejected with `AppError::InvalidInput` (mirrors the
/// `chat_in_flight` guard on chat sessions).
///
/// # Progress
///
/// Emits `AppEvent::OperationProgress { op: OperationKind::Translate }` (fraction
/// = segments_done / total_segments) throttled to ~5 Hz. Emits
/// `AppEvent::TranslationReady { meeting_id, language }` on every exit path
/// (success AND error) so the webview's operation-progress indicator is always
/// cleared. On a partial-failure exit the event fires before the error is
/// returned to the caller; any completed segments remain on disk.
///
/// # Errors
///
/// - `AppError::InvalidInput` when `target_language` is not in
///   [`SUPPORTED_TRANSLATION_LANGUAGES`] or the meeting has no transcript.
/// - `AppError::InvalidInput` when translation is already in-flight for this
///   `(meeting_id, target_language)` pair.
#[tauri::command]
#[specta::specta]
pub async fn translate_meeting(
    meeting_id: MeetingId,
    target_language: String,
    state: State<'_, IpcState>,
) -> AppResult<()> {
    // Validate target_language against the supported set.
    if !SUPPORTED_TRANSLATION_LANGUAGES.contains(&target_language.as_str()) {
        return Err(AppError::InvalidInput {
            context: format!(
                "unsupported translation target language: {target_language:?}; \
                 supported: {SUPPORTED_TRANSLATION_LANGUAGES:?}"
            ),
        }
        .into());
    }

    // Reject a concurrent translate for the same (meeting_id, language) pair.
    let key = (meeting_id, target_language.clone());
    {
        let mut in_flight = state
            .translate_in_flight
            .lock()
            .expect("translate_in_flight poisoned");
        if !in_flight.insert(key.clone()) {
            return Err(AppError::InvalidInput {
                context: format!(
                    "translation already in-flight for meeting {} language {target_language:?}",
                    meeting_id.0
                ),
            }
            .into());
        }
    }

    // Emit indeterminate progress while the model loads.
    let _ = state.event_tx.send(AppEvent::OperationProgress {
        meeting_id,
        op: OperationKind::Translate,
        fraction: None,
        label: "Loading the translation model…".to_string(),
    });

    let load_result = state.ensure_summariser().await;

    // Drive the blocking work, then release the in-flight guard whether or
    // not the work succeeded (mirrors the chat_in_flight release pattern).
    let work_result = match load_result {
        Ok(summariser) => {
            let meetings_dir = state.meetings_dir.clone();
            let event_tx = state.event_tx.clone();
            let language_for_blocking = target_language.clone();
            tokio::task::spawn_blocking(move || {
                translate_meeting_blocking(
                    &meetings_dir,
                    meeting_id,
                    &language_for_blocking,
                    &summariser,
                    &event_tx,
                )
            })
            .await
            .map_err(|e| {
                AppError::Internal {
                    context: format!("translate_meeting task join failed: {e}"),
                }
            })
            .and_then(|r| r)
        }
        Err(e) => Err(e),
    };

    state
        .translate_in_flight
        .lock()
        .expect("translate_in_flight poisoned")
        .remove(&key);

    // Emit TranslationReady on every exit path so the operation-progress
    // indicator is cleared even when the pass fails mid-segment. The UI's
    // `handleEvent` only refetches translations when the meeting and language
    // match the active view; a refetch on a partial result is harmless — it
    // surfaces whatever segments completed before the error.
    let _ = state.event_tx.send(AppEvent::TranslationReady {
        meeting_id,
        language: target_language,
    });

    work_result?;

    Ok(())
}

/// Blocking body for `translate_meeting`.
///
/// Reads the transcript, translates each segment via
/// [`LlamaSummariser::translate_segment`], and merges batched results into
/// `translations.json` on the same ~200 ms cadence used for progress emission
/// (plus unconditionally on loop exit — both the normal completion and any
/// early-error exit — so partial progress always survives an interruption).
/// Batching replaces the previous per-segment flush (a full sidecar
/// read+rewrite+fsync per segment) to avoid O(n²) I/O on long meetings while
/// preserving the durability guarantee.
///
/// Emits `OperationProgress` throttled to ~5 Hz. Synchronous — the caller
/// drives this on `spawn_blocking`.
pub(crate) fn translate_meeting_blocking(
    meetings_dir: &Path,
    meeting_id: MeetingId,
    target_language: &str,
    summariser: &LlamaSummariser,
    event_tx: &tokio::sync::broadcast::Sender<AppEvent>,
) -> Result<(), AppError> {
    let meeting_dir = meetings_dir.join(meeting_id.0.to_string());
    let segments = persistence::read_transcript(&meeting_dir)?;

    let total = segments.len();
    if total == 0 {
        return Err(AppError::InvalidInput {
            context: format!(
                "meeting {} has an empty transcript; nothing to translate",
                meeting_id.0
            ),
        });
    }

    let mut pending: HashMap<usize, String> = HashMap::new();
    let mut last_flush = std::time::Instant::now();
    let mut result = Ok(());

    for (idx, segment) in segments.iter().enumerate() {
        let translated = match summariser
            .translate_segment(&segment.text, target_language)
            .map_err(|e| AppError::Internal {
                context: format!(
                    "translate_segment failed for segment {idx} of meeting {}: {e}",
                    meeting_id.0
                ),
            }) {
            Ok(t) => t,
            Err(e) => {
                result = Err(e);
                break;
            }
        };

        pending.insert(idx, translated);

        // Flush pending translations and emit progress on the same ~200 ms
        // cadence; always flush+emit on the last segment.
        let fraction = (idx + 1) as f32 / total as f32;
        let now = std::time::Instant::now();
        let is_last = idx + 1 == total;
        if is_last || now.duration_since(last_flush) >= std::time::Duration::from_millis(200) {
            last_flush = now;
            persistence::merge_translations(&meeting_dir, target_language, &pending)?;
            pending.clear();
            let _ = event_tx.send(AppEvent::OperationProgress {
                meeting_id,
                op: OperationKind::Translate,
                fraction: Some(fraction.clamp(0.0, 1.0)),
                label: format!("Translating… ({}/{total})", idx + 1),
            });
        }
    }

    // Flush any segments accumulated since the last throttled flush (covers the
    // error-exit path: partial progress on completed segments must survive).
    if !pending.is_empty() {
        persistence::merge_translations(&meeting_dir, target_language, &pending)?;
    }

    tracing::info!(
        target: "ipc-bridge",
        meeting_id = %meeting_id.0,
        language = target_language,
        segments = total,
        "translation loop complete"
    );

    result
}

/// Read the translations for a single meeting + language combination.
///
/// Returns a `HashMap<usize, String>` mapping segment index to translated text
/// for `target_language`, or an empty map when no translations exist yet for
/// that language (or when `translations.json` is absent). The webview calls this
/// on meeting open and on `TranslationReady` to populate the translated-view
/// overlay.
#[tauri::command]
#[specta::specta]
pub async fn get_translations(
    meeting_id: MeetingId,
    target_language: String,
    state: State<'_, IpcState>,
) -> AppResult<HashMap<usize, String>> {
    let meetings_dir = state.meetings_dir.clone();
    tokio::task::spawn_blocking(move || {
        let meeting_dir = meetings_dir.join(meeting_id.0.to_string());
        let mut all = persistence::read_translations(&meeting_dir)?;
        Ok::<_, AppError>(all.remove(&target_language).unwrap_or_default())
    })
    .await
    .map_err(|e| AppError::Internal {
        context: format!("get_translations task join failed: {e}"),
    })?
}

/// Get one chat session for a meeting, or `None` when it does not exist.
#[tauri::command]
#[specta::specta]
pub async fn get_chat_session(
    meeting_id: MeetingId,
    session_id: ChatSessionId,
    state: State<'_, IpcState>,
) -> AppResult<Option<ChatSession>> {
    let meetings_dir = state.meetings_dir.clone();
    tokio::task::spawn_blocking(move || ChatStore::load(&meetings_dir, meeting_id, session_id))
        .await
        .map_err(|e| AppError::Internal {
            context: format!("get_chat_session task join failed: {e}"),
        })?
}

/// List all chat sessions for a meeting, most-recently-updated first.
#[tauri::command]
#[specta::specta]
pub async fn list_chat_sessions(
    meeting_id: MeetingId,
    state: State<'_, IpcState>,
) -> AppResult<Vec<ChatSession>> {
    let meetings_dir = state.meetings_dir.clone();
    tokio::task::spawn_blocking(move || ChatStore::list(&meetings_dir, meeting_id))
        .await
        .map_err(|e| AppError::Internal {
            context: format!("list_chat_sessions task join failed: {e}"),
        })?
}

/// Delete one chat session for a meeting (idempotent).
#[tauri::command]
#[specta::specta]
pub async fn delete_chat_session(
    meeting_id: MeetingId,
    session_id: ChatSessionId,
    state: State<'_, IpcState>,
) -> AppResult<()> {
    let meetings_dir = state.meetings_dir.clone();
    tokio::task::spawn_blocking(move || ChatStore::delete(&meetings_dir, meeting_id, session_id))
        .await
        .map_err(|e| AppError::Internal {
            context: format!("delete_chat_session task join failed: {e}"),
        })?
}

