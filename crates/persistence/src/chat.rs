//! `ChatStore` — reader/writer for a meeting's chat sessions (Phase 9).
//!
//! Stores each session as `{root}/{meeting_id}/chat/{session_id}.json`, one
//! file per session, mirroring [`crate::NotesStore`]'s standalone, stateless,
//! atomic-write shape. `ChatStore` holds no open file handle; every call
//! resolves paths from the `(root, meeting_id[, session_id])` it is given.
//!
//! `persistence` is the **sole writer** under `{app-data}/meetings/`
//! (`architecture/cross-cutting.md` — Filesystem layout). The chat driver in
//! `ipc-bridge` persists a session through this store at turn end; it never
//! writes under `meetings/` itself.
//!
//! # Atomic writes
//!
//! [`ChatStore::save`] writes to a sibling temp file in the `chat/` subfolder,
//! then renames it into place — a crash mid-save leaves the previous session
//! file intact (the rename is atomic on one filesystem) and no `.tmp` residue
//! on the success path. The `chat/` subfolder is created on first save (it is a
//! child of the meeting folder, which `MeetingWriter` already owns; creating the
//! subfolder is not a write to a recording-owned file).
//!
//! `delete_meeting` removes the whole meeting folder, so a meeting's chat
//! sessions go with it — no separate chat cleanup is required.

use std::path::{Path, PathBuf};

use meeting_app_common::{AppResult, ChatSession, ChatSessionId, MeetingId};

use crate::error::Error;

/// Standalone, stateless store for a meeting's chat sessions.
///
/// Like [`crate::NotesStore`], `ChatStore` holds no state; every call resolves
/// `{root}/{meeting_id}/chat/{session_id}.json` from its arguments.
pub struct ChatStore;

impl ChatStore {
    /// The `{root}/{meeting_id}/chat/` directory for a meeting.
    fn chat_dir(root: &Path, meeting_id: MeetingId) -> PathBuf {
        root.join(meeting_id.0.to_string()).join("chat")
    }

    /// The `{root}/{meeting_id}/chat/{session_id}.json` path for one session.
    fn session_path(root: &Path, meeting_id: MeetingId, session_id: ChatSessionId) -> PathBuf {
        Self::chat_dir(root, meeting_id).join(format!("{}.json", session_id.0))
    }

    /// Atomically write `session` to its `chat/{session_id}.json` file under the
    /// meeting folder.
    ///
    /// Creates the `chat/` subfolder on first save. Writes to a sibling temp
    /// file then renames into place, so an interrupted save never leaves a
    /// truncated session file. The meeting folder itself is expected to exist
    /// (owned by [`crate::MeetingWriter`]); a save into a non-existent meeting
    /// folder errors after the `chat/` `create_dir_all` (which fails if the
    /// parent is missing) — it does not silently create the meeting folder
    /// proper, because `create_dir_all` of `…/chat` under a missing meeting
    /// folder is the caller's bug to surface.
    pub fn save(root: &Path, meeting_id: MeetingId, session: &ChatSession) -> AppResult<()> {
        let chat_dir = Self::chat_dir(root, meeting_id);
        std::fs::create_dir_all(&chat_dir).map_err(Error::Io)?;

        let path = chat_dir.join(format!("{}.json", session.id.0));
        let bytes = serde_json::to_vec_pretty(session)
            .map_err(Error::Serialise)
            .map_err(meeting_app_common::AppError::from)?;
        write_atomic(&path, &bytes)?;

        tracing::debug!(
            target: "persistence",
            meeting_id = %meeting_id.0,
            session_id = %session.id.0,
            messages = session.messages.len(),
            "chat session saved"
        );
        Ok(())
    }

    /// Load ONE chat session by id, or `Ok(None)` when it does not exist.
    pub fn load(
        root: &Path,
        meeting_id: MeetingId,
        session_id: ChatSessionId,
    ) -> AppResult<Option<ChatSession>> {
        let path = Self::session_path(root, meeting_id, session_id);
        let bytes = match std::fs::read(&path) {
            Ok(b) => b,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(Error::Io(e).into()),
        };
        let session: ChatSession = serde_json::from_slice(&bytes)
            .map_err(Error::Serialise)
            .map_err(meeting_app_common::AppError::from)?;
        Ok(Some(session))
    }

    /// Load every chat session for a meeting, most-recently-updated first.
    ///
    /// Returns an empty `Vec` when the meeting has no `chat/` folder (no session
    /// has ever been saved) — an absent folder is a legitimate empty list, not
    /// an error. A single unparseable session file is logged and skipped so one
    /// corrupt file never hides the rest.
    pub fn list(root: &Path, meeting_id: MeetingId) -> AppResult<Vec<ChatSession>> {
        let chat_dir = Self::chat_dir(root, meeting_id);
        let read_dir = match std::fs::read_dir(&chat_dir) {
            Ok(rd) => rd,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(e) => return Err(Error::Io(e).into()),
        };

        let mut sessions: Vec<ChatSession> = Vec::new();
        for entry in read_dir.flatten() {
            let path = entry.path();
            let is_json = path
                .extension()
                .and_then(|e| e.to_str())
                .is_some_and(|e| e.eq_ignore_ascii_case("json"));
            if !is_json {
                continue;
            }
            match std::fs::read(&path) {
                Ok(bytes) => match serde_json::from_slice::<ChatSession>(&bytes) {
                    Ok(session) => sessions.push(session),
                    Err(e) => tracing::warn!(
                        target: "persistence",
                        path = %path.display(),
                        "skipping unparseable chat session file: {e}"
                    ),
                },
                Err(e) => tracing::warn!(
                    target: "persistence",
                    path = %path.display(),
                    "skipping unreadable chat session file: {e}"
                ),
            }
        }

        // Most-recently-updated first (RFC 3339 strings sort lexicographically
        // in chronological order for a fixed offset, matching the meeting list).
        sessions.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
        Ok(sessions)
    }

    /// Delete ONE chat session by id. Removing an already-absent session is not
    /// an error (idempotent delete).
    pub fn delete(
        root: &Path,
        meeting_id: MeetingId,
        session_id: ChatSessionId,
    ) -> AppResult<()> {
        let path = Self::session_path(root, meeting_id, session_id);
        match std::fs::remove_file(&path) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(Error::Io(e).into()),
        }
    }
}

/// Write `bytes` to `path` atomically: write to a sibling temp file, fsync,
/// then rename into place. Mirrors `notes::write_atomic` (the same discipline);
/// kept private to this module so the chat store is self-contained.
fn write_atomic(path: &Path, bytes: &[u8]) -> AppResult<()> {
    use std::io::Write;

    let parent = path.parent().ok_or_else(|| meeting_app_common::AppError::Internal {
        context: format!("chat session path has no parent: {}", path.display()),
    })?;

    let file_name = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "session".to_string());
    let tmp_path = parent.join(format!("{file_name}.tmp"));

    let write_result = (|| -> Result<(), std::io::Error> {
        let mut file = std::fs::File::create(&tmp_path)?;
        file.write_all(bytes)?;
        file.flush()?;
        file.sync_all()?;
        Ok(())
    })();

    if let Err(e) = write_result {
        let _ = std::fs::remove_file(&tmp_path);
        return Err(Error::Io(e).into());
    }
    if let Err(e) = std::fs::rename(&tmp_path, path) {
        let _ = std::fs::remove_file(&tmp_path);
        return Err(Error::Io(e).into());
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::folder::MeetingFolder;
    use meeting_app_common::{ChatMessage, ChatRole, ToolCallRecord};
    use tempfile::TempDir;

    fn make_meeting() -> (TempDir, PathBuf, MeetingId) {
        let tempdir = TempDir::new().expect("tempdir");
        let root = tempdir.path().to_path_buf();
        let id = MeetingId::new();
        MeetingFolder::create(&root, id).expect("create meeting folder");
        (tempdir, root, id)
    }

    fn sample_session(meeting_id: MeetingId, updated_at: &str) -> ChatSession {
        ChatSession {
            id: ChatSessionId::new(),
            meeting_id: Some(meeting_id),
            title: Some("Action items".to_string()),
            messages: vec![
                ChatMessage {
                    role: ChatRole::User,
                    content: "what were the action items?".to_string(),
                    tool_name: None,
                    tool_call_id: None,
                    tool_calls: Vec::new(),
                    turn_id: 1,
                },
                ChatMessage {
                    role: ChatRole::Assistant,
                    content: String::new(),
                    tool_name: None,
                    tool_call_id: None,
                    tool_calls: vec![ToolCallRecord {
                        id: "call_1".to_string(),
                        name: "get_transcript".to_string(),
                        arguments_json: "{}".to_string(),
                    }],
                    turn_id: 1,
                },
                ChatMessage {
                    role: ChatRole::Tool,
                    content: "{\"segments\":[]}".to_string(),
                    tool_name: Some("get_transcript".to_string()),
                    tool_call_id: Some("call_1".to_string()),
                    tool_calls: Vec::new(),
                    turn_id: 1,
                },
                ChatMessage {
                    role: ChatRole::Assistant,
                    content: "the action items were …".to_string(),
                    tool_name: None,
                    tool_call_id: None,
                    tool_calls: Vec::new(),
                    turn_id: 1,
                },
            ],
            created_at: "2026-06-10T10:00:00Z".to_string(),
            updated_at: updated_at.to_string(),
        }
    }

    #[test]
    fn save_then_load_round_trips() {
        let (_tempdir, root, id) = make_meeting();
        let session = sample_session(id, "2026-06-10T10:01:00Z");

        ChatStore::save(&root, id, &session).expect("save");
        let loaded = ChatStore::load(&root, id, session.id)
            .expect("load")
            .expect("present after save");
        assert_eq!(loaded, session, "chat session must round-trip");
    }

    #[test]
    fn load_absent_session_returns_none() {
        let (_tempdir, root, id) = make_meeting();
        let loaded = ChatStore::load(&root, id, ChatSessionId::new()).expect("load");
        assert!(loaded.is_none(), "absent session must yield None");
    }

    #[test]
    fn list_absent_chat_folder_returns_empty() {
        let (_tempdir, root, id) = make_meeting();
        // No session has ever been saved → no chat/ folder.
        let sessions = ChatStore::list(&root, id).expect("list");
        assert!(sessions.is_empty(), "absent chat/ folder must yield empty list");
    }

    #[test]
    fn list_returns_sessions_most_recently_updated_first() {
        let (_tempdir, root, id) = make_meeting();
        let older = sample_session(id, "2026-06-10T10:00:00Z");
        let newer = sample_session(id, "2026-06-10T11:00:00Z");
        ChatStore::save(&root, id, &older).expect("save older");
        ChatStore::save(&root, id, &newer).expect("save newer");

        let sessions = ChatStore::list(&root, id).expect("list");
        assert_eq!(sessions.len(), 2);
        assert_eq!(sessions[0].id, newer.id, "most-recently-updated first");
        assert_eq!(sessions[1].id, older.id);
    }

    #[test]
    fn save_overwrites_same_session_in_place() {
        let (_tempdir, root, id) = make_meeting();
        let mut session = sample_session(id, "2026-06-10T10:00:00Z");
        ChatStore::save(&root, id, &session).expect("first save");

        session.messages.push(ChatMessage {
            role: ChatRole::User,
            content: "and who owns them?".to_string(),
            tool_name: None,
            tool_call_id: None,
            tool_calls: Vec::new(),
            turn_id: 2,
        });
        session.updated_at = "2026-06-10T10:05:00Z".to_string();
        ChatStore::save(&root, id, &session).expect("re-save");

        // Still exactly one session for the meeting, with the updated content.
        let sessions = ChatStore::list(&root, id).expect("list");
        assert_eq!(sessions.len(), 1, "re-save must overwrite, not append a file");
        assert_eq!(sessions[0].messages.len(), 4);
        assert_eq!(sessions[0].updated_at, "2026-06-10T10:05:00Z");
    }

    #[test]
    fn delete_removes_session_and_is_idempotent() {
        let (_tempdir, root, id) = make_meeting();
        let session = sample_session(id, "2026-06-10T10:00:00Z");
        ChatStore::save(&root, id, &session).expect("save");

        ChatStore::delete(&root, id, session.id).expect("delete");
        assert!(
            ChatStore::load(&root, id, session.id).expect("load").is_none(),
            "deleted session must be gone"
        );
        // Deleting again is not an error.
        ChatStore::delete(&root, id, session.id).expect("idempotent delete");
    }

    #[test]
    fn successful_save_leaves_no_tmp_residue() {
        let (_tempdir, root, id) = make_meeting();
        let session = sample_session(id, "2026-06-10T10:00:00Z");
        ChatStore::save(&root, id, &session).expect("save");

        let chat_dir = root.join(id.0.to_string()).join("chat");
        let residue: Vec<_> = std::fs::read_dir(&chat_dir)
            .expect("read chat dir")
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|name| name.ends_with(".tmp"))
            .collect();
        assert!(residue.is_empty(), "expected no .tmp residue, found: {residue:?}");
    }
}
