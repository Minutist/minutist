//! Convergence proof for the WS4-B S3 notes-sync protocol.
//!
//! Two `SyncEngine`s, each backed by its own temp meetings root that shares the
//! SAME `meeting_id`, reconcile a meeting's Yjs notes over a real (relay-less,
//! loopback) iroh connection. The assertions compare the projected ProseMirror
//! JSON of each side's `notes.ydoc` after the exchange:
//!
//! - one-sided seed: A has notes, B is empty → B converges to A;
//! - bidirectional: A and B each have distinct edits → both converge to the same
//!   merged document (CRDT commutativity).
//!
//! The exchange runs over `SyncEngine::start_direct` with relays disabled, the
//! same loopback addressing the endpoint round-trip test uses.

use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::path::Path;

use iroh::EndpointAddr;
use minutist_common::MeetingId;
use persistence::{MeetingFolder, NotesData, NotesStore};
use serde_json::{json, Value};
use sync::identity::DeviceIdentity;
use sync::SyncEngine;

/// Loopback `EndpointAddr` for `engine` (its id + each bound port against
/// `127.0.0.1`), mirroring the endpoint round-trip test's helper.
fn direct_addr(engine: &SyncEngine) -> EndpointAddr {
    let mut addr = EndpointAddr::new(engine.endpoint_id());
    for sock in engine.bound_sockets() {
        let loopback = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), sock.port());
        addr = addr.with_ip_addr(loopback);
    }
    addr
}

/// Create the meeting folder under `root` and seed `notes.ydoc` from `doc` via
/// the authoritative `persistence` writer (the same path the editor's first
/// save takes).
fn seed_notes(root: &Path, id: MeetingId, doc: &Value, md: &str) {
    MeetingFolder::create(root, id).expect("create meeting folder");
    NotesStore::save(root, id, doc, md).expect("seed notes.ydoc");
}

/// The projected ProseMirror JSON of the meeting's `notes.ydoc` under `root`.
fn projected_json(root: &Path, id: MeetingId) -> Value {
    let NotesData { json, .. } = NotesStore::load(root, id)
        .expect("load notes")
        .expect("notes present");
    json
}

/// Start two relay-less engines bound to `root_a` / `root_b` and cross-inject
/// their loopback addresses (the out-of-band pairing the account service does).
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

/// A representative seed document.
fn seed_doc(text: &str) -> Value {
    json!({
        "type": "doc",
        "content": [
            { "type": "paragraph", "attrs": { "data-anchor-ms": 1000 },
              "content": [{ "type": "text", "text": text }] }
        ]
    })
}

#[tokio::test]
async fn one_sided_seed_converges_to_seeder() {
    let dir_a = tempfile::TempDir::new().expect("tempdir a");
    let dir_b = tempfile::TempDir::new().expect("tempdir b");
    let root_a = dir_a.path();
    let root_b = dir_b.path();

    // Same meeting id on both roots; A is seeded, B has no notes.
    let meeting = MeetingId::new();
    let doc = seed_doc("notes from device A");
    seed_notes(root_a, meeting, &doc, "# A");
    MeetingFolder::create(root_b, meeting).expect("create folder b");
    assert!(
        NotesStore::read_ydoc_state(root_b, meeting)
            .expect("read b state")
            .is_none(),
        "B must start with no notes.ydoc"
    );

    let (a, b) = paired_engines(root_a, root_b).await;

    // A initiates a reconciliation with B for the meeting.
    a.sync_notes(direct_addr(&b), meeting)
        .await
        .expect("reconcile a -> b");

    // B converged to A's document.
    assert_eq!(
        projected_json(root_b, meeting),
        projected_json(root_a, meeting),
        "B must converge to A's notes after a one-sided sync"
    );
    assert_eq!(
        projected_json(root_b, meeting),
        doc,
        "B's converged document must equal the seeded document"
    );

    a.shutdown().await.expect("shutdown a");
    b.shutdown().await.expect("shutdown b");
}

#[tokio::test]
async fn bidirectional_edits_converge_to_same_state() {
    let dir_a = tempfile::TempDir::new().expect("tempdir a");
    let dir_b = tempfile::TempDir::new().expect("tempdir b");
    let root_a = dir_a.path();
    let root_b = dir_b.path();
    let meeting = MeetingId::new();

    // Both sides start from a COMMON base so their docs share CRDT ancestry, then
    // each applies a distinct edit. Independent edits on a shared base merge
    // cleanly; two docs minted from scratch (fresh client ids) would also merge,
    // but a shared base is the realistic paired-device shape.
    let base = seed_doc("base line");
    seed_notes(root_a, meeting, &base, "# base");

    // Clone the base onto B by replaying A's whole v1 state (the pairing handshake
    // a first sync would perform), so both carry the same origin client history.
    let base_v1 = NotesStore::read_ydoc_state(root_a, meeting)
        .expect("read base state")
        .expect("base present");
    MeetingFolder::create(root_b, meeting).expect("create folder b");
    NotesStore::apply_update(root_b, meeting, &base_v1, "# base").expect("clone base onto b");

    // Distinct edits: A appends a paragraph, B appends a different paragraph.
    edit_append_paragraph(root_a, meeting, "from A");
    edit_append_paragraph(root_b, meeting, "from B");

    // Sanity: the two diverged before the sync.
    assert_ne!(
        projected_json(root_a, meeting),
        projected_json(root_b, meeting),
        "the two sides must diverge before reconciliation"
    );

    let (a, b) = paired_engines(root_a, root_b).await;
    a.sync_notes(direct_addr(&b), meeting)
        .await
        .expect("bidirectional reconcile");

    // Both sides converged to the identical merged document.
    let merged_a = projected_json(root_a, meeting);
    let merged_b = projected_json(root_b, meeting);
    assert_eq!(
        merged_a, merged_b,
        "both sides must converge to the same merged document"
    );

    // The merge retained both edits (and the base line).
    let text = all_paragraph_text(&merged_a);
    assert!(
        text.contains("base line"),
        "merged doc must keep the base line"
    );
    assert!(text.contains("from A"), "merged doc must keep A's edit");
    assert!(text.contains("from B"), "merged doc must keep B's edit");

    a.shutdown().await.expect("shutdown a");
    b.shutdown().await.expect("shutdown b");
}

/// Append a paragraph carrying `text` to the meeting's `notes.ydoc` via the
/// authoritative incremental path, preserving CRDT history (mirrors how an
/// editor edit reaches `persistence`).
///
/// The edit is a genuine **incremental** mutation: the stored state is hydrated
/// into a `yrs` doc, a new paragraph element is inserted at the end of the
/// prosemirror fragment **on that doc's existing client history**, and only the
/// delta since the pre-edit state vector is shipped to `apply_update`. Rebuilding
/// the doc from JSON instead would mint a fresh client id, so the two sides'
/// "same" base content would carry different ids and would NOT merge — it would
/// duplicate. Keeping the mutation on the hydrated doc is what makes the
/// bidirectional merge a true CRDT merge.
fn edit_append_paragraph(root: &Path, id: MeetingId, text: &str) {
    use persistence::ydoc;
    use yrs::updates::encoder::Encode;
    use yrs::{ReadTxn, Transact, XmlElementPrelim, XmlFragment, XmlTextPrelim};

    let v1 = NotesStore::read_ydoc_state(root, id)
        .expect("read state")
        .expect("state present");
    let current = ydoc::new_ydoc();
    ydoc::apply_update_v1(&current, &v1).expect("hydrate current");

    let before_sv = current.transact().state_vector().encode_v1();

    // Insert `<paragraph><text>{text}</text></paragraph>` at the fragment tail,
    // on `current`'s own client history.
    {
        let fragment = current.get_or_insert_xml_fragment(ydoc::PROSEMIRROR_FRAGMENT);
        let mut txn = current.transact_mut();
        let len = fragment.len(&txn);
        let para = fragment.insert(&mut txn, len, XmlElementPrelim::empty("paragraph"));
        para.insert(&mut txn, 0, XmlTextPrelim::new(text));
    }

    // Ship only the delta since `before_sv` so `apply_update` merges incrementally.
    let after = current
        .transact()
        .encode_state_as_update_v1(&Default::default());
    let delta = yrs::diff_updates_v1(&after, &before_sv).expect("delta since before");

    NotesStore::apply_update(root, id, &delta, "").expect("apply append edit");
}

/// Concatenate the text of every `text` node in `doc` (a coarse merged-content
/// probe for the convergence assertions).
fn all_paragraph_text(doc: &Value) -> String {
    fn walk(node: &Value, out: &mut String) {
        if let Some(arr) = node.as_array() {
            for c in arr {
                walk(c, out);
            }
            return;
        }
        if let Some(obj) = node.as_object() {
            if obj.get("type").and_then(|t| t.as_str()) == Some("text") {
                if let Some(t) = obj.get("text").and_then(|t| t.as_str()) {
                    out.push_str(t);
                    out.push('\n');
                }
            }
            if let Some(content) = obj.get("content") {
                walk(content, out);
            }
        }
    }
    let mut out = String::new();
    walk(doc, &mut out);
    out
}
