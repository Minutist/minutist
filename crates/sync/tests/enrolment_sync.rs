//! Enrolment over a real (relay-less, loopback) iroh connection: the content-key
//! transfer of `planning/DESIGN_sync-encryption.md` §5.
//!
//! The assertions are about what a peer can READ, not about which functions
//! returned `Ok`. A transfer that reports success but leaves the two devices
//! unable to exchange notes has not enrolled anything, and a refusal that still
//! lets a rogue read a meeting has not refused anything.

use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::path::Path;

use iroh::EndpointAddr;
use minutist_common::MeetingId;
use notes_crdt::{MeetingFolder, NotesData, NotesStore};
use serde_json::json;
use sync::identity::DeviceIdentity;
use sync::{ContentKey, SyncEngine};

/// Loopback `EndpointAddr` for `engine`, mirroring the other integration tests.
fn direct_addr(engine: &SyncEngine) -> EndpointAddr {
    let mut addr = EndpointAddr::new(engine.endpoint_id());
    for sock in engine.bound_sockets() {
        addr = addr.with_ip_addr(SocketAddr::new(
            IpAddr::V4(Ipv4Addr::LOCALHOST),
            sock.port(),
        ));
    }
    addr
}

/// Two engines that can reach each other but hold DIFFERENT content keys: the
/// state a freshly-discovered peer is in before anyone confirms it.
async fn unenrolled_pair(root_a: &Path, root_b: &Path) -> (SyncEngine, SyncEngine) {
    let id_a = DeviceIdentity::load_or_generate(root_a).expect("identity a");
    let id_b = DeviceIdentity::load_or_generate(root_b).expect("identity b");
    let a = SyncEngine::start_direct(id_a, Some(ContentKey::for_tests()), root_a.to_path_buf())
        .await
        .expect("engine a");
    let b = SyncEngine::start_direct(
        id_b,
        Some(ContentKey::from_bytes([0xdd; 32])),
        root_b.to_path_buf(),
    )
    .await
    .expect("engine b");
    a.add_peer(direct_addr(&b));
    b.add_peer(direct_addr(&a));
    (a, b)
}

fn seed_notes(root: &Path, id: MeetingId, text: &str) {
    MeetingFolder::create(root, id).expect("create meeting folder");
    let doc = json!({
        "type": "doc",
        "content": [
            { "type": "paragraph",
              "content": [{ "type": "text", "text": text }] }
        ]
    });
    NotesStore::save(root, id, &doc, text).expect("seed notes.ydoc");
}

fn notes_text(root: &Path, id: MeetingId) -> Option<String> {
    NotesStore::load(root, id)
        .expect("load notes")
        .map(|NotesData { json, .. }| json.to_string())
}

/// Both devices show the same code, and it is six digits. The whole scheme rests
/// on a user comparing these across two screens.
#[tokio::test]
async fn both_devices_derive_the_same_code_for_each_other() {
    let dir_a = tempfile::TempDir::new().expect("tempdir a");
    let dir_b = tempfile::TempDir::new().expect("tempdir b");
    let (a, b) = unenrolled_pair(dir_a.path(), dir_b.path()).await;

    let from_a = a.pending_enrolments();
    let from_b = b.pending_enrolments();
    assert_eq!(
        from_a.len(),
        1,
        "A should see exactly B pending: {from_a:?}"
    );
    assert_eq!(
        from_b.len(),
        1,
        "B should see exactly A pending: {from_b:?}"
    );
    assert_eq!(
        from_a[0].safety_code, from_b[0].safety_code,
        "the code the user compares must match on both screens"
    );
    assert_eq!(from_a[0].safety_code.len(), 6);
    assert_eq!(from_a[0].peer_id, b.endpoint_id().to_string());

    a.shutdown().await.expect("shutdown a");
    b.shutdown().await.expect("shutdown b");
}

/// The happy path, asserted by content rather than by return value: before
/// enrolment the two cannot exchange notes at all; after mutual confirmation
/// they converge.
#[tokio::test]
async fn a_confirmed_peer_receives_the_key_and_can_then_sync() {
    let dir_a = tempfile::TempDir::new().expect("tempdir a");
    let dir_b = tempfile::TempDir::new().expect("tempdir b");
    let (root_a, root_b) = (dir_a.path(), dir_b.path());
    let meeting = MeetingId::new();
    seed_notes(root_a, meeting, "board minutes");

    let (a, b) = unenrolled_pair(root_a, root_b).await;

    // Baseline: different keys, so a notes exchange cannot authenticate.
    let before = a.sync_notes(b.endpoint_id(), meeting).await;
    assert!(
        matches!(before, Err(sync::Error::Unauthenticated(_))),
        "unenrolled peers must not sync: {before:?}"
    );
    assert!(
        notes_text(root_b, meeting).is_none(),
        "B must hold nothing before enrolment"
    );

    // B's user confirms A, so B will accept a key from A. A's user confirms B,
    // which records the verdict and sends the key.
    b.confirm_enrolment(&a.endpoint_id().to_string(), None)
        .expect("B records that it trusts A");
    a.confirm_and_offer(&b.endpoint_id().to_string(), None)
        .await
        .expect("A confirms B and hands over the key");

    assert!(a.is_enrolled(&b.endpoint_id().to_string()));
    assert!(b.is_enrolled(&a.endpoint_id().to_string()));

    // Now the same exchange works, and B actually has the content.
    a.sync_notes(b.endpoint_id(), meeting)
        .await
        .expect("an enrolled peer syncs");
    let converged = notes_text(root_b, meeting).expect("B has the meeting");
    assert!(
        converged.contains("board minutes"),
        "B must hold A's notes after enrolment: {converged}"
    );

    a.shutdown().await.expect("shutdown a");
    b.shutdown().await.expect("shutdown b");
}

/// A peer the receiver has NOT confirmed is refused the key, even though the
/// sender confirmed it. Both sides decide independently; neither takes the
/// other's word for having asked its user.
#[tokio::test]
async fn the_receiver_refuses_a_key_from_a_peer_it_has_not_confirmed() {
    let dir_a = tempfile::TempDir::new().expect("tempdir a");
    let dir_b = tempfile::TempDir::new().expect("tempdir b");
    let (root_a, root_b) = (dir_a.path(), dir_b.path());
    let meeting = MeetingId::new();
    seed_notes(root_a, meeting, "commercially sensitive");

    let (a, b) = unenrolled_pair(root_a, root_b).await;

    // Only A confirms. B never does, so B must refuse the offered key.
    let offered = a
        .confirm_and_offer(&b.endpoint_id().to_string(), None)
        .await;
    assert!(
        matches!(offered, Err(sync::Error::Unauthenticated(_))),
        "an unconfirmed receiver must refuse the key: {offered:?}"
    );

    // The refusal is real: B still cannot read A's notes.
    let after = a.sync_notes(b.endpoint_id(), meeting).await;
    assert!(
        matches!(after, Err(sync::Error::Unauthenticated(_))),
        "a refused peer must still not sync: {after:?}"
    );
    assert!(
        notes_text(root_b, meeting).is_none(),
        "B must hold nothing after refusing the key"
    );

    a.shutdown().await.expect("shutdown a");
    b.shutdown().await.expect("shutdown b");
}

/// Refusing an endpoint the directory offered records the decision, so the user
/// is not asked again, and the peer stays unable to read anything.
#[tokio::test]
async fn a_refused_peer_is_not_re_prompted_and_stays_locked_out() {
    let dir_a = tempfile::TempDir::new().expect("tempdir a");
    let dir_b = tempfile::TempDir::new().expect("tempdir b");
    let (root_a, root_b) = (dir_a.path(), dir_b.path());
    let meeting = MeetingId::new();
    seed_notes(root_a, meeting, "private");

    let (a, b) = unenrolled_pair(root_a, root_b).await;
    let b_id = b.endpoint_id().to_string();

    assert_eq!(a.pending_enrolments().len(), 1);
    a.refuse_enrolment(&b_id, None).expect("refuse");

    assert!(
        a.pending_enrolments().is_empty(),
        "a decided peer must not be prompted for again"
    );
    assert!(!a.is_enrolled(&b_id));

    // A must not hand the key over to a peer it refused, even if asked directly.
    let offered = a.offer_content_key(&b_id).await;
    assert!(
        offered.is_err(),
        "a refused peer must never be offered the key: {offered:?}"
    );
    assert!(notes_text(root_b, meeting).is_none());

    a.shutdown().await.expect("shutdown a");
    b.shutdown().await.expect("shutdown b");
}

/// The decision survives a restart: it is keyed on the peer's ed25519 identity
/// and persisted, not held in memory.
#[tokio::test]
async fn a_confirmation_survives_an_engine_restart() {
    let dir_a = tempfile::TempDir::new().expect("tempdir a");
    let dir_b = tempfile::TempDir::new().expect("tempdir b");
    let (root_a, root_b) = (dir_a.path(), dir_b.path());

    let (a, b) = unenrolled_pair(root_a, root_b).await;
    let b_id = b.endpoint_id().to_string();
    // Record without transferring: confirming is local and must not depend on
    // the peer being reachable at all.
    b.shutdown().await.expect("shutdown b");
    a.confirm_enrolment(&b_id, None)
        .expect("confirm records locally");
    assert!(a.is_enrolled(&b_id));
    assert_eq!(a.confirmed_peers(), vec![b_id.clone()]);
    a.shutdown().await.expect("shutdown a");

    let id_a = DeviceIdentity::load_or_generate(root_a).expect("identity a");
    let restarted =
        SyncEngine::start_direct(id_a, Some(ContentKey::for_tests()), root_a.to_path_buf())
            .await
            .expect("restart a");
    assert!(
        restarted.is_enrolled(&b_id),
        "the confirmation must outlive the process"
    );
    restarted.shutdown().await.expect("shutdown restarted");
}

/// `safety_code_for` answers for any peer, decided or not, so a settings screen
/// can show the code for an already-enrolled device.
#[tokio::test]
async fn the_code_is_readable_after_a_peer_is_enrolled() {
    let dir_a = tempfile::TempDir::new().expect("tempdir a");
    let dir_b = tempfile::TempDir::new().expect("tempdir b");
    let (a, b) = unenrolled_pair(dir_a.path(), dir_b.path()).await;
    let b_id = b.endpoint_id().to_string();

    let pending = a.pending_enrolments();
    let shown_while_pending = pending[0].safety_code.clone();
    a.refuse_enrolment(&b_id, None).expect("decide");

    assert_eq!(
        a.safety_code_for(&b_id).expect("code"),
        shown_while_pending,
        "the code must not change once a peer is decided"
    );
    assert!(a.safety_code_for("not-an-endpoint-id").is_err());

    a.shutdown().await.expect("shutdown a");
    b.shutdown().await.expect("shutdown b");
}

/// A device joining an account that already has one starts with NO key at all
/// (§3.1), rather than minting one no peer holds. It must still serve, so it can
/// be enrolled, but must sync nothing until it is.
#[tokio::test]
async fn a_keyless_device_serves_enrolment_but_syncs_nothing() {
    let dir_a = tempfile::TempDir::new().expect("tempdir a");
    let dir_b = tempfile::TempDir::new().expect("tempdir b");
    let (root_a, root_b) = (dir_a.path(), dir_b.path());
    let meeting = MeetingId::new();
    seed_notes(root_a, meeting, "quarterly numbers");

    let id_a = DeviceIdentity::load_or_generate(root_a).expect("identity a");
    let id_b = DeviceIdentity::load_or_generate(root_b).expect("identity b");
    let a = SyncEngine::start_direct(id_a, Some(ContentKey::for_tests()), root_a.to_path_buf())
        .await
        .expect("engine a");
    // B is the joining device: no key, exactly what `load_or_mint` returns for a
    // device that finds other devices already on the account.
    let b = SyncEngine::start_direct(id_b, None, root_b.to_path_buf())
        .await
        .expect("a keyless engine must still bind and serve");
    a.add_peer(direct_addr(&b));
    b.add_peer(direct_addr(&a));

    assert!(a.is_enrolled_self());
    assert!(!b.is_enrolled_self(), "B holds no key yet");

    // B cannot initiate anything: its own state, not the peer's.
    let attempted = b.sync_notes(a.endpoint_id(), meeting).await;
    assert!(
        matches!(attempted, Err(sync::Error::Unauthenticated(_))),
        "a keyless device must refuse to sync: {attempted:?}"
    );

    // Enrolment still works, which is the whole point of it serving keyless.
    b.confirm_enrolment(&a.endpoint_id().to_string(), None)
        .expect("B trusts A");
    a.confirm_and_offer(&b.endpoint_id().to_string(), None)
        .await
        .expect("A enrols B");

    assert!(b.is_enrolled_self(), "B holds the key after enrolment");

    // And now it converges.
    a.sync_notes(b.endpoint_id(), meeting)
        .await
        .expect("an enrolled peer syncs");
    assert!(notes_text(root_b, meeting)
        .expect("B has the meeting")
        .contains("quarterly numbers"));

    a.shutdown().await.expect("shutdown a");
    b.shutdown().await.expect("shutdown b");
}

/// A confirmation recorded by a DIFFERENT process reaches an already-running
/// engine, and the engine finishes the transfer on its own.
///
/// This is the headless-hub deployment: the daemon runs continuously and holds
/// the engine, while `minutist-hub confirm` is a one-shot invocation beside it.
/// The file is the channel between them, so the test writes through a bare
/// `EnrolmentStore` (what the CLI has) rather than through the engine, and never
/// lets the two share memory.
#[tokio::test]
async fn a_decision_written_by_another_process_reaches_the_running_engine() {
    use sync::{EnrolmentStore, Verdict};

    let dir_a = tempfile::TempDir::new().expect("tempdir a");
    let dir_b = tempfile::TempDir::new().expect("tempdir b");
    let (root_a, root_b) = (dir_a.path(), dir_b.path());
    let meeting = MeetingId::new();
    seed_notes(root_a, meeting, "written out of band");

    let id_a = DeviceIdentity::load_or_generate(root_a).expect("identity a");
    let id_b = DeviceIdentity::load_or_generate(root_b).expect("identity b");
    let a = SyncEngine::start_direct(id_a, Some(ContentKey::for_tests()), root_a.to_path_buf())
        .await
        .expect("engine a");
    let b = SyncEngine::start_direct(id_b, None, root_b.to_path_buf())
        .await
        .expect("engine b");
    a.add_peer(direct_addr(&b));
    b.add_peer(direct_addr(&a));

    let (a_id, b_id) = (a.endpoint_id().to_string(), b.endpoint_id().to_string());

    // Neither engine is told anything. Both decisions are written the way a
    // one-shot CLI would write them, against the same data dirs.
    EnrolmentStore::load(root_a)
        .record(&b_id, Verdict::Confirmed, None)
        .expect("A's operator confirms B out of band");
    EnrolmentStore::load(root_b)
        .record(&a_id, Verdict::Confirmed, None)
        .expect("B's operator confirms A out of band");

    // The running engines see it without being restarted or notified.
    assert!(a.is_enrolled(&b_id), "A must read the decision from disk");
    assert!(b.is_enrolled(&a_id), "B must read the decision from disk");

    // And the sweep the refresh loop drives finishes the transfer.
    assert_eq!(a.deliver_pending_keys().await, 1);
    assert!(b.is_enrolled_self(), "B holds the key after the sweep");

    // A second sweep is a no-op: delivery is recorded, so it does not re-offer.
    assert_eq!(
        a.deliver_pending_keys().await,
        0,
        "a delivered key must not be re-offered every poll"
    );

    a.sync_notes(b.endpoint_id(), meeting)
        .await
        .expect("enrolled by file alone");
    assert!(notes_text(root_b, meeting)
        .expect("B has the meeting")
        .contains("written out of band"));

    a.shutdown().await.expect("shutdown a");
    b.shutdown().await.expect("shutdown b");
}

/// The engine's view and a CLI's view of what needs deciding are the same
/// derivation, so they cannot drift.
///
/// The engine passes peer ids from its in-memory directory; a one-shot CLI
/// passes them from its own `/v1/account/devices` call. Both go through
/// `pending_from`, and this asserts they agree on the same inputs.
#[tokio::test]
async fn the_engine_and_a_cli_derive_the_same_pending_list() {
    use sync::{pending_from, EnrolmentStore, Verdict};

    let dir_a = tempfile::TempDir::new().expect("tempdir a");
    let dir_b = tempfile::TempDir::new().expect("tempdir b");
    let (root_a, root_b) = (dir_a.path(), dir_b.path());
    let (a, b) = unenrolled_pair(root_a, root_b).await;
    let b_id = b.endpoint_id().to_string();

    // What a CLI has: its own identity from a file, the peer ids from its own
    // directory call, and the record from disk. No engine.
    let own = DeviceIdentity::load_or_generate(root_a).expect("identity");
    let cli_view = |ids: &[String]| {
        pending_from(
            own.public_key().as_bytes(),
            ids,
            &EnrolmentStore::load(root_a),
        )
    };

    assert_eq!(
        cli_view(std::slice::from_ref(&b_id)),
        a.pending_enrolments()
    );

    // A decision made through the CLI path drops the peer from both views.
    EnrolmentStore::load(root_a)
        .record(&b_id, Verdict::Refused, None)
        .expect("decide");
    assert!(cli_view(std::slice::from_ref(&b_id)).is_empty());
    assert!(a.pending_enrolments().is_empty());

    // A malformed directory entry is skipped, not fatal: one bad row must not
    // hide the real devices from the user.
    let mixed = vec![
        "not-an-endpoint-id".to_string(),
        b.endpoint_id().to_string(),
    ];
    let fresh = tempfile::TempDir::new().expect("tempdir c");
    let view = pending_from(
        own.public_key().as_bytes(),
        &mixed,
        &EnrolmentStore::load(fresh.path()),
    );
    assert_eq!(view.len(), 1, "the parseable peer must survive: {view:?}");
    assert_eq!(view[0].peer_id, b_id);

    a.shutdown().await.expect("shutdown a");
    b.shutdown().await.expect("shutdown b");
}

/// Refusing a peer drops it from the directory, so it is neither dialled nor
/// admitted, and the account-refresh loop's upsert does not put it back.
///
/// The earlier refusal test asserted the record and that the key is withheld;
/// this asserts the refusal actually bites on the transport.
#[tokio::test]
async fn a_refused_peer_is_dropped_from_the_directory_and_stays_out() {
    use std::sync::Arc;
    use sync::{AccountEndpoint, RefreshSink, SyncEngineRefreshSink};

    let dir_a = tempfile::TempDir::new().expect("tempdir a");
    let dir_b = tempfile::TempDir::new().expect("tempdir b");
    let (a, b) = unenrolled_pair(dir_a.path(), dir_b.path()).await;
    let b_id = b.endpoint_id().to_string();
    let a = Arc::new(a);

    assert!(a.peer_ids().contains(&b_id), "B starts in A's directory");

    a.refuse_enrolment(&b_id, None).expect("refuse");
    assert!(
        !a.peer_ids().contains(&b_id),
        "a refused peer must leave the directory, which is what stops both the \
         outbound dial and the inbound accept"
    );
    assert!(a.is_refused(&b_id));
    assert!(!a.is_enrolled(&b_id));

    // The next directory poll re-offers it. Driven through the real sink the
    // refresh loop uses, not a test seam, so this covers the actual path.
    let was_new = {
        let sink = SyncEngineRefreshSink::new(Arc::clone(&a));
        sink.upsert_account_peer(&AccountEndpoint {
            device_id: "device-b".to_string(),
            endpoint_id: b_id.clone(),
            relay_url: "https://example.invalid".to_string(),
            direct_addrs: Vec::new(),
            label: None,
        })
    };
    assert!(!was_new, "a refused peer must not read as newly added");
    assert!(
        !a.peer_ids().contains(&b_id),
        "the next directory poll must not re-add a refused peer"
    );

    b.shutdown().await.expect("shutdown b");
    Arc::try_unwrap(a)
        .unwrap_or_else(|_| panic!("sink still holds the engine"))
        .shutdown()
        .await
        .expect("shutdown a");
}

/// A mint that fails is reported, not swallowed.
///
/// A device that has just established it is the only one on its account, and
/// then cannot write the key, is permanently unable to sync. `is_enrolled_self()`
/// is `false` in that state and also `false` in the ordinary "waiting to be
/// enrolled" state, so a caller that is only told the boolean would prompt the
/// user to go and confirm on a device that does not exist.
#[tokio::test]
async fn a_failed_mint_is_reported_rather_than_leaving_a_silently_dead_device() {
    let dir = tempfile::TempDir::new().expect("tempdir");
    let root = dir.path();
    let id = DeviceIdentity::load_or_generate(root).expect("identity");
    let engine = SyncEngine::start_direct(id, None, root.to_path_buf())
        .await
        .expect("keyless engine binds");
    assert!(!engine.is_enrolled_self());

    // A key file of the wrong length: `load` refuses it rather than minting
    // over it, because reminting would strand the device from every peer while
    // looking like a first run.
    std::fs::write(ContentKey::path(root), b"truncated").expect("seed a bad key file");

    let outcome = engine.note_account_peers(false);
    assert!(
        outcome.is_err(),
        "the caller must learn the mint failed, not just see is_enrolled_self() stay false"
    );
    assert!(!engine.is_enrolled_self());

    // The healthy path still reports success, so a caller can tell the two apart.
    std::fs::remove_file(ContentKey::path(root)).expect("clear the bad key file");
    engine
        .note_account_peers(false)
        .expect("a lone device mints and says so");
    assert!(
        engine.is_enrolled_self(),
        "minting made this device the founder"
    );

    // And once enrolled it is a no-op whatever the flag says, so a caller that
    // gets the flag wrong later cannot undo it.
    engine
        .note_account_peers(true)
        .expect("no-op once enrolled");
    assert!(engine.is_enrolled_self());

    engine.shutdown().await.expect("shutdown");
}
