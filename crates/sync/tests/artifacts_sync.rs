//! Round-trip proof for the WS4-B derived-artifact sync protocol
//! ([`sync::artifacts_proto`]).
//!
//! Two `SyncEngine`s, each backed by its own temp meetings root sharing the SAME
//! `meeting_id`, reconcile a meeting's DERIVED outputs — `transcript.json` and
//! `summary.md` — over a real (relay-less, loopback) iroh connection. The bytes
//! travel over the blobs ALPN, named by an authority-stamped manifest exchanged
//! over the sync ALPN.
//!
//! Unlike the media counterpart (`media_sync.rs`, content-addressed immutable
//! blobs), a derived artifact is MUTABLE: the headline case here proves a NEWER
//! producer copy supersedes a stale consumer copy and is NOT clobbered by it
//! (`planning/DESIGN_artifacts.md` §2).

use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::path::Path;

use iroh::EndpointAddr;
use minutist_common::{HostRef, MeetingId, ProcessingLifecycle};
use notes_crdt::MeetingFolder;
use sync::identity::DeviceIdentity;
use sync::SyncEngine;

/// Loopback `EndpointAddr` for `engine` (its id + each bound port against
/// `127.0.0.1`) — mirrors the media/notes round-trip helper.
fn direct_addr(engine: &SyncEngine) -> EndpointAddr {
    let mut addr = EndpointAddr::new(engine.endpoint_id());
    for sock in engine.bound_sockets() {
        let loopback = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), sock.port());
        addr = addr.with_ip_addr(loopback);
    }
    addr
}

/// Start two relay-less engines bound to `root_a` / `root_b` and cross-inject their
/// loopback addresses (the out-of-band MUTUAL pairing the account service does).
async fn paired_engines(root_a: &Path, root_b: &Path) -> (SyncEngine, SyncEngine) {
    let id_a = DeviceIdentity::load_or_generate(root_a).expect("identity a");
    let id_b = DeviceIdentity::load_or_generate(root_b).expect("identity b");
    let a = SyncEngine::start_direct(id_a, root_a.to_path_buf())
        .await
        .expect("engine a");
    let b = SyncEngine::start_direct(id_b, root_b.to_path_buf())
        .await
        .expect("engine b");
    a.add_peer(direct_addr(&b));
    b.add_peer(direct_addr(&a));
    (a, b)
}

/// Seed a meeting that this device PRODUCED: write `transcript.json` + `summary.md`
/// and set the local `Processed { processed_by, at }` so the artifact import stamps
/// the producer authority on its manifest.
async fn seed_processed(
    root: &Path,
    id: MeetingId,
    transcript: &[u8],
    summary: &[u8],
    host: &str,
    at: &str,
) {
    // `ensure` seeds a placeholder `metadata.json` (the RMW in
    // `apply_processing_lifecycle` needs one); `create` would only make the dir.
    MeetingFolder::ensure(root, id).expect("ensure meeting folder");
    let dir = root.join(id.0.to_string());
    std::fs::write(dir.join("transcript.json"), transcript).expect("write transcript.json");
    std::fs::write(dir.join("summary.md"), summary).expect("write summary.md");
    persistence::meeting_ops::apply_processing_lifecycle(
        root,
        id,
        ProcessingLifecycle::Processed {
            processed_by: HostRef(host.to_string()),
            at: at.to_string(),
        },
    )
    .await
    .expect("set Processed");
}

/// Read a meeting artifact's on-disk bytes.
fn read_artifact(root: &Path, id: MeetingId, rel: &str) -> Vec<u8> {
    std::fs::read(root.join(id.0.to_string()).join(rel))
        .unwrap_or_else(|e| panic!("read {rel} under {root:?}: {e}"))
}

#[tokio::test]
async fn artifacts_round_trip_transcript_and_summary() {
    let dir_a = tempfile::TempDir::new().expect("tempdir a");
    let dir_b = tempfile::TempDir::new().expect("tempdir b");
    let root_a = dir_a.path();
    let root_b = dir_b.path();

    // A produced the meeting; B has only the (empty) meeting folder.
    let meeting = MeetingId::new();
    let transcript = br#"[{"speaker_id":"A","text":"hello"}]"#.repeat(4);
    let summary = b"# Summary\n\nThe meeting happened.".repeat(4);
    seed_processed(
        root_a,
        meeting,
        &transcript,
        &summary,
        "host-a",
        "2026-06-30T10:00:00Z",
    )
    .await;
    MeetingFolder::ensure(root_b, meeting).expect("ensure folder b");

    let (a, b) = paired_engines(root_a, root_b).await;

    // A initiates an artifact reconciliation with B.
    a.sync_artifacts(direct_addr(&b), meeting)
        .await
        .expect("reconcile artifacts a -> b");

    // B received both derived files, byte-identical.
    assert_eq!(
        read_artifact(root_b, meeting, "transcript.json"),
        transcript,
        "B's transcript must be byte-identical to A's"
    );
    assert_eq!(
        read_artifact(root_b, meeting, "summary.md"),
        summary,
        "B's summary must be byte-identical to A's"
    );

    a.shutdown().await.expect("shutdown a");
    b.shutdown().await.expect("shutdown b");
}

#[tokio::test]
async fn reprocess_newer_copy_supersedes_and_stale_does_not_clobber() {
    // The headline anti-clobber: A produces v1, syncs to B; A reprocesses to a
    // NEWER v2; on the next sync B takes v2 (it strictly supersedes its v1) and A's
    // v2 is NOT overwritten by B's stale v1. This is exactly the relay-clobber the
    // per-entry, byte-bound authority model exists to prevent (DESIGN §2).
    let dir_a = tempfile::TempDir::new().expect("tempdir a");
    let dir_b = tempfile::TempDir::new().expect("tempdir b");
    let root_a = dir_a.path();
    let root_b = dir_b.path();

    let meeting = MeetingId::new();
    let v1 = br#"[{"v":1}]"#.to_vec();
    let summary = b"# Summary v1".to_vec();
    seed_processed(root_a, meeting, &v1, &summary, "host-a", "2026-06-30T10:00:00Z").await;
    MeetingFolder::ensure(root_b, meeting).expect("ensure folder b");

    let (a, b) = paired_engines(root_a, root_b).await;

    // First sync: B gets v1 and records its authority (v1, host-a, T1).
    a.sync_artifacts(direct_addr(&b), meeting)
        .await
        .expect("sync v1 a -> b");
    assert_eq!(read_artifact(root_b, meeting, "transcript.json"), v1);

    // A reprocesses: a NEWER transcript v2 at T2 > T1 (summary unchanged).
    let v2 = br#"[{"v":2},{"v":2}]"#.to_vec();
    std::fs::write(
        root_a.join(meeting.0.to_string()).join("transcript.json"),
        &v2,
    )
    .expect("rewrite transcript v2");
    persistence::meeting_ops::apply_processing_lifecycle(
        root_a,
        meeting,
        ProcessingLifecycle::Processed {
            processed_by: HostRef("host-a".to_string()),
            at: "2026-06-30T11:00:00Z".to_string(),
        },
    )
    .await
    .expect("reprocess to T2");

    // Second sync: B pulls v2 (supersedes its v1); A keeps v2 (B's v1 is older, so
    // it never supersedes A's copy).
    a.sync_artifacts(direct_addr(&b), meeting)
        .await
        .expect("sync v2 a -> b");
    assert_eq!(
        read_artifact(root_b, meeting, "transcript.json"),
        v2,
        "B must take the newer producer copy"
    );
    assert_eq!(
        read_artifact(root_a, meeting, "transcript.json"),
        v2,
        "A's newer copy must not be clobbered by B's stale one"
    );

    a.shutdown().await.expect("shutdown a");
    b.shutdown().await.expect("shutdown b");
}

#[tokio::test]
async fn peer_copy_does_not_overwrite_unprovable_local_bytes() {
    // Silent-loss guard: B holds NEWER transcript bytes it cannot stamp (present on
    // disk, but the meeting is not locally Processed and there is no authority
    // record — e.g. produced before a Processed flip, or a lost authority store), so
    // B advertises nothing for it. A offers an OLDER, validly-stamped copy. The pull
    // must NOT treat "B advertises none" as "B lacks it" and overwrite B's newer
    // bytes — B keeps its local file.
    let dir_a = tempfile::TempDir::new().expect("tempdir a");
    let dir_b = tempfile::TempDir::new().expect("tempdir b");
    let root_a = dir_a.path();
    let root_b = dir_b.path();

    let meeting = MeetingId::new();
    // A: an older, stamped transcript.
    let a_bytes = br#"[{"v":"a-older"}]"#.to_vec();
    seed_processed(root_a, meeting, &a_bytes, b"# A", "host-a", "2026-06-30T10:00:00Z").await;
    // B: a newer transcript on disk, but NOT Processed and with no authority record
    // (ensure seeds metadata as Local). B will not advertise it.
    let b_bytes = br#"[{"v":"b-newer-unprovable"}]"#.to_vec();
    MeetingFolder::ensure(root_b, meeting).expect("ensure folder b");
    std::fs::write(
        root_b.join(meeting.0.to_string()).join("transcript.json"),
        &b_bytes,
    )
    .expect("write b transcript");

    let (a, b) = paired_engines(root_a, root_b).await;

    a.sync_artifacts(direct_addr(&b), meeting)
        .await
        .expect("reconcile a -> b");

    // B's unprovable local bytes are intact — NOT clobbered by A's older copy.
    assert_eq!(
        read_artifact(root_b, meeting, "transcript.json"),
        b_bytes,
        "B's newer unstampable local bytes must not be overwritten by an older peer copy"
    );
    // A keeps its own copy (B advertised nothing for A to pull).
    assert_eq!(read_artifact(root_a, meeting, "transcript.json"), a_bytes);

    a.shutdown().await.expect("shutdown a");
    b.shutdown().await.expect("shutdown b");
}

#[tokio::test]
async fn sync_is_noop_when_both_already_hold_the_artifacts() {
    // Both devices already hold byte-identical artifacts for the same meeting. A
    // full reconciliation converges cleanly (a no-op): every entry is byte-identical
    // so nothing is pulled, and both files stay unchanged.
    let dir_a = tempfile::TempDir::new().expect("tempdir a");
    let dir_b = tempfile::TempDir::new().expect("tempdir b");
    let root_a = dir_a.path();
    let root_b = dir_b.path();

    let meeting = MeetingId::new();
    let transcript = br#"[{"speaker_id":"A","text":"converged"}]"#.repeat(4);
    let summary = b"# Converged summary".to_vec();
    seed_processed(
        root_a,
        meeting,
        &transcript,
        &summary,
        "host-a",
        "2026-06-30T10:00:00Z",
    )
    .await;
    seed_processed(
        root_b,
        meeting,
        &transcript,
        &summary,
        "host-a",
        "2026-06-30T10:00:00Z",
    )
    .await;

    let (a, b) = paired_engines(root_a, root_b).await;

    a.sync_artifacts(direct_addr(&b), meeting)
        .await
        .expect("no-op artifact reconcile a -> b");

    // Both files unchanged and still byte-identical on both sides.
    assert_eq!(read_artifact(root_a, meeting, "transcript.json"), transcript);
    assert_eq!(read_artifact(root_b, meeting, "transcript.json"), transcript);
    assert_eq!(read_artifact(root_a, meeting, "summary.md"), summary);
    assert_eq!(read_artifact(root_b, meeting, "summary.md"), summary);

    a.shutdown().await.expect("shutdown a");
    b.shutdown().await.expect("shutdown b");
}
