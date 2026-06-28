//! Regression test for the `metadata.json` lost-update race (task #54).
//!
//! Two writers race on the SAME meeting: `set_speaker_name` (a user field) and
//! `apply_processing_lifecycle` (the field the sync lifecycle subscriber sets).
//! Each does a read→mutate→write of `metadata.json`. Without the per-meeting
//! metadata lock the later write clobbers the field the earlier one set
//! (last-write-wins); with the lock the writes serialise — the second reads the
//! first's update before mutating — so BOTH fields survive.
//!
//! Runs on a multi-thread runtime over many independent meetings raced at once,
//! so the two writers for a meeting genuinely overlap on different worker
//! threads. Distinct meetings use distinct locks and never serialise against
//! each other; only the two writers within one meeting contend.
//!
//! Scope: this covers the `persistence::meeting_ops` + lifecycle-subscriber path
//! the lock guards. It does NOT cover the `orchestrator`'s unguarded
//! `metadata.json` RMW sites (a tracked cross-domain follow-up), which can still
//! revert `processing` — see the coverage boundary in
//! `architecture/cross-cutting.md` ("Per-meeting metadata.json write lock").

use std::collections::BTreeMap;
use std::path::Path;

use minutist_common::{AudioFormat, HostRef, MeetingId, MeetingMeta, ProcessingLifecycle};
use persistence::{read_metadata, write_metadata, MeetingFolder};
use tempfile::TempDir;

/// Seed a meeting folder with a full `metadata.json` (no speaker names,
/// `processing = Local`) and return its id.
fn seed_meeting(root: &Path) -> MeetingId {
    let id = MeetingId::new();
    let folder = MeetingFolder::create(root, id).expect("create folder");
    let meta = MeetingMeta {
        uuid: id,
        title: "Race seed".to_string(),
        started_at: "2026-06-28T09:00:00Z".to_string(),
        ended_at: None,
        duration_ms: 0,
        speaker_count: 0,
        audio_format: AudioFormat {
            codec: "opus".into(),
            sample_rate: 16_000,
            channels: 1,
            bitrate_kbps: Some(32),
        },
        asr_model: None,
        llm_model: None,
        diarizer: None,
        speaker_names: BTreeMap::new(),
        notes_format: 0,
        processing: ProcessingLifecycle::Local,
        collection_id: None,
        app_version: "0.0.0".into(),
    };
    write_metadata(folder.path(), &meta).expect("write metadata");
    id
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_metadata_rmw_keeps_both_fields() {
    let tmp = TempDir::new().expect("tempdir");
    let root = tmp.path().to_path_buf();

    // Many independent meetings, each raced by its own writer pair — more
    // meetings means more chances for an unlocked build to drop a field.
    const MEETINGS: usize = 64;
    let ids: Vec<MeetingId> = (0..MEETINGS).map(|_| seed_meeting(&root)).collect();

    let processed = ProcessingLifecycle::Processed {
        processed_by: HostRef("endpoint-test".to_string()),
        at: "2026-06-28T10:00:00Z".to_string(),
    };

    // Spawn ALL writers up front so the runtime can interleave a meeting's two
    // writers across worker threads.
    let mut handles = Vec::with_capacity(MEETINGS * 2);
    for &id in &ids {
        let r = root.clone();
        handles.push(tokio::spawn(async move {
            persistence::meeting_ops::set_speaker_name(&r, id, "A", "Alice")
                .await
                .expect("set_speaker_name");
        }));

        let r = root.clone();
        let processed = processed.clone();
        handles.push(tokio::spawn(async move {
            persistence::meeting_ops::apply_processing_lifecycle(&r, id, processed)
                .await
                .expect("apply_processing_lifecycle");
        }));
    }
    for h in handles {
        h.await.expect("writer task panicked");
    }

    // Every meeting must carry BOTH the speaker name AND the processed state —
    // neither writer clobbered the other.
    for &id in &ids {
        let folder = root.join(id.0.to_string());
        let meta = read_metadata(&folder).expect("read metadata");
        assert_eq!(
            meta.speaker_names.get("A").map(String::as_str),
            Some("Alice"),
            "meeting {}: speaker name lost — a concurrent lifecycle write clobbered it",
            id.0
        );
        assert!(
            matches!(meta.processing, ProcessingLifecycle::Processed { .. }),
            "meeting {}: processing lost — a concurrent speaker-name write clobbered it (got {:?})",
            id.0,
            meta.processing
        );
    }
}
