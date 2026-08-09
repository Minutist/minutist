//! Convergent authored metadata inside `notes.ydoc` (issue 0052).
//!
//! A meeting's **descriptive** metadata — dates, duration, codec, title, and the
//! speaker/model fields — is authored on the device that captured or edited it.
//! But the placeholder `metadata.json` a peer writes on sync-arrival
//! ([`crate::MeetingFolder::ensure`]) invents wrong values (`started_at = now`,
//! `duration_ms = 0`, `codec = "opus"`), and no sync protocol ever replaces them
//! (issue 0052). To converge the real values across devices with no bespoke
//! merge protocol, the authored fields are mirrored into a top-level Yjs `Map`
//! that rides inside the SAME `notes.ydoc` blob that already syncs and converges
//! — so per-field last-writer-wins falls out of the notes merge for free.
//!
//! # The load-bearing invariant
//!
//! Only an **authoritative** writer touches the map: the device that captured the
//! meeting, a user edit (rename / speaker-name), or the processing host writing
//! derived fields ([`write_descriptive`]). The sync-arrival placeholder path must
//! **never** write to it — otherwise the placeholder's invented `started_at = now`
//! would enter the CRDT and race the origin's real value. A peer that holds only
//! the placeholder READS the converged map ([`project_into_meta`]) over its
//! `metadata.json`; it does not write its own values back.
//!
//! # What is / isn't carried
//!
//! Carried (authored, converges): `title`, `started_at`, `ended_at`,
//! `duration_ms`, `speaker_count`, `audio_format`, `asr_model`, `llm_model`,
//! `diarizer`, `speaker_names`. NOT carried (per-peer or local, synced by their
//! own mechanism or intentionally device-local): `processing` (the lifecycle,
//! synced via the discovery exchange + `merge_processing`), `collection_id`
//! (local-authoritative), `notes_format`, `uuid` (the folder name), `app_version`.

use std::collections::BTreeMap;
use std::path::Path;

use minutist_common::{AppResult, AudioFormat, MeetingId, MeetingMeta, ModelDescriptor};
use serde::de::DeserializeOwned;
use serde::Serialize;
use yrs::{Any, Doc, Map, MapRef, Out, ReadTxn, Transact, TransactionMut};

/// Top-level Yjs `Map` name holding the authored descriptive metadata, alongside
/// the `prosemirror` fragment ([`crate::ydoc::PROSEMIRROR_FRAGMENT`]) in the same
/// `notes.ydoc`.
const META_MAP: &str = "meta";

// Map keys. Scalars are stored as their `yrs::Any` type; the structured fields
// (`audio_format`, the model descriptors, `speaker_names`) as a serde-JSON string
// value — each is a single independent LWW register.
const K_TITLE: &str = "title";
const K_STARTED_AT: &str = "started_at";
const K_ENDED_AT: &str = "ended_at";
const K_DURATION_MS: &str = "duration_ms";
const K_SPEAKER_COUNT: &str = "speaker_count";
const K_AUDIO_FORMAT: &str = "audio_format";
const K_ASR_MODEL: &str = "asr_model";
const K_LLM_MODEL: &str = "llm_model";
const K_DIARIZER: &str = "diarizer";
const K_SPEAKER_NAMES: &str = "speaker_names";

/// Mirror the authored descriptive fields of `meta` into `doc`'s meta map.
///
/// Call ONLY from an authoritative writer (capture, a user edit, the processing
/// host) — never the sync-arrival placeholder path (see the module invariant).
/// Each field is written as an independent map entry, so it converges per-field
/// with the notes merge. Absent optionals (`ended_at`, the models, an empty
/// `speaker_names`) are removed rather than written, so clearing a field
/// converges too.
pub fn write_descriptive(doc: &Doc, meta: &MeetingMeta) {
    let map = doc.get_or_insert_map(META_MAP);
    let mut txn = doc.transact_mut();
    map.insert(&mut txn, K_TITLE, meta.title.clone());
    map.insert(&mut txn, K_STARTED_AT, meta.started_at.clone());
    put_opt_str(&mut txn, &map, K_ENDED_AT, meta.ended_at.as_deref());
    map.insert(&mut txn, K_DURATION_MS, Any::BigInt(meta.duration_ms as i64));
    map.insert(
        &mut txn,
        K_SPEAKER_COUNT,
        Any::BigInt(i64::from(meta.speaker_count)),
    );
    put_json(&mut txn, &map, K_AUDIO_FORMAT, Some(&meta.audio_format));
    put_json(&mut txn, &map, K_ASR_MODEL, meta.asr_model.as_ref());
    put_json(&mut txn, &map, K_LLM_MODEL, meta.llm_model.as_ref());
    put_json(&mut txn, &map, K_DIARIZER, meta.diarizer.as_ref());
    let names = if meta.speaker_names.is_empty() {
        None
    } else {
        Some(&meta.speaker_names)
    };
    put_json(&mut txn, &map, K_SPEAKER_NAMES, names);
}

// --- granular single-key setters (for EDITS after capture) ------------------
//
// [`write_descriptive`] rewrites every key in one transaction, which is correct
// for the origin's one-time capture write but would make two devices' concurrent
// edits to DIFFERENT fields collide (each full write is one CRDT operation over
// every key). An edit after capture must touch only the key it changes, so
// per-field last-writer-wins actually holds. Each setter opens its own
// transaction and writes exactly one map key. Callers are the authoritative edit
// sites (rename, speaker-name, the processing host's derived-field writes).

/// Set the meeting title (a user rename).
pub fn set_title(doc: &Doc, title: &str) {
    let map = doc.get_or_insert_map(META_MAP);
    let mut txn = doc.transact_mut();
    map.insert(&mut txn, K_TITLE, title.to_string());
}

/// Set the recording end time.
pub fn set_ended_at(doc: &Doc, ended_at: Option<&str>) {
    let map = doc.get_or_insert_map(META_MAP);
    let mut txn = doc.transact_mut();
    put_opt_str(&mut txn, &map, K_ENDED_AT, ended_at);
}

/// Set the recording duration in milliseconds.
pub fn set_duration_ms(doc: &Doc, duration_ms: u64) {
    let map = doc.get_or_insert_map(META_MAP);
    let mut txn = doc.transact_mut();
    map.insert(&mut txn, K_DURATION_MS, Any::BigInt(duration_ms as i64));
}

/// Set the diarised speaker count (a processing-host derived field).
pub fn set_speaker_count(doc: &Doc, speaker_count: u32) {
    let map = doc.get_or_insert_map(META_MAP);
    let mut txn = doc.transact_mut();
    map.insert(&mut txn, K_SPEAKER_COUNT, Any::BigInt(i64::from(speaker_count)));
}

/// Set the ASR model descriptor (a processing-host derived field).
pub fn set_asr_model(doc: &Doc, model: Option<&ModelDescriptor>) {
    let map = doc.get_or_insert_map(META_MAP);
    let mut txn = doc.transact_mut();
    put_json(&mut txn, &map, K_ASR_MODEL, model);
}

/// Set the summariser LLM model descriptor (a processing-host derived field).
pub fn set_llm_model(doc: &Doc, model: Option<&ModelDescriptor>) {
    let map = doc.get_or_insert_map(META_MAP);
    let mut txn = doc.transact_mut();
    put_json(&mut txn, &map, K_LLM_MODEL, model);
}

/// Set the diarizer model descriptor (a processing-host derived field).
pub fn set_diarizer(doc: &Doc, model: Option<&ModelDescriptor>) {
    let map = doc.get_or_insert_map(META_MAP);
    let mut txn = doc.transact_mut();
    put_json(&mut txn, &map, K_DIARIZER, model);
}

/// Set the speaker-id → display-name map (a user edit or a processing-host
/// restore). Stored as one JSON entry, so this key converges block-LWW: two
/// devices renaming DIFFERENT speakers concurrently resolve to one device's whole
/// map. Per-speaker convergence (a nested map keyed by speaker id) is a possible
/// refinement if simultaneous distinct-speaker edits become a real case; the
/// dominant writer is the single processing host.
pub fn set_speaker_names(doc: &Doc, names: &BTreeMap<String, String>) {
    let map = doc.get_or_insert_map(META_MAP);
    let mut txn = doc.transact_mut();
    let value = if names.is_empty() { None } else { Some(names) };
    put_json(&mut txn, &map, K_SPEAKER_NAMES, value);
}

/// Whether `doc`'s meta map carries any authored metadata — used to skip the
/// projection for a legacy meeting whose `notes.ydoc` predates this map (no
/// authored fields to apply).
pub fn has_descriptive(doc: &Doc) -> bool {
    let map = doc.get_or_insert_map(META_MAP);
    let txn = doc.transact();
    map.len(&txn) > 0
}

/// Project `doc`'s converged meta map onto `meta`, overwriting ONLY the
/// descriptive fields the map actually carries and leaving every per-peer / local
/// field (`processing`, `collection_id`, `notes_format`, `uuid`, `app_version`)
/// untouched. A field absent from the map leaves `meta`'s value as-is, so a
/// partially-populated map never blanks a locally-known field.
///
/// Returns whether any field changed (so a caller can skip a `metadata.json`
/// rewrite when the projection is a no-op).
pub fn project_into_meta(doc: &Doc, meta: &mut MeetingMeta) -> bool {
    let map = doc.get_or_insert_map(META_MAP);
    let txn = doc.transact();
    let before = serde_json::to_value(&*meta).ok();

    if let Some(s) = get_str(&map, &txn, K_TITLE) {
        meta.title = s;
    }
    if let Some(s) = get_str(&map, &txn, K_STARTED_AT) {
        meta.started_at = s;
    }
    // `ended_at` present → Some; a missing key leaves the local value as-is (it
    // never forces None — clearing is not distinguishable from "not carried" for
    // an optional, and a recording's end time only ever gets more known).
    if let Some(s) = get_str(&map, &txn, K_ENDED_AT) {
        meta.ended_at = Some(s);
    }
    if let Some(i) = get_bigint(&map, &txn, K_DURATION_MS) {
        meta.duration_ms = i.max(0) as u64;
    }
    if let Some(i) = get_bigint(&map, &txn, K_SPEAKER_COUNT) {
        meta.speaker_count = i.max(0) as u32;
    }
    if let Some(af) = get_json::<AudioFormat>(&map, &txn, K_AUDIO_FORMAT) {
        meta.audio_format = af;
    }
    if let Some(m) = get_json::<ModelDescriptor>(&map, &txn, K_ASR_MODEL) {
        meta.asr_model = Some(m);
    }
    if let Some(m) = get_json::<ModelDescriptor>(&map, &txn, K_LLM_MODEL) {
        meta.llm_model = Some(m);
    }
    if let Some(m) = get_json::<ModelDescriptor>(&map, &txn, K_DIARIZER) {
        meta.diarizer = Some(m);
    }
    if let Some(names) = get_json::<BTreeMap<String, String>>(&map, &txn, K_SPEAKER_NAMES) {
        meta.speaker_names = names;
    }

    let after = serde_json::to_value(&*meta).ok();
    before != after
}

/// Read `{meetings_root}/{id}/notes.ydoc`, project its converged meta map onto
/// the meeting's `metadata.json` (guarded RMW under
/// [`crate::metadata_lock`]), leaving the per-peer fields untouched. A no-op that
/// returns `Ok(false)` when `notes.ydoc` is absent/undecodable or carries no meta
/// map, or when the projection changes nothing. Returns `Ok(true)` when
/// `metadata.json` was updated.
///
/// This is the sync-side apply seam: run it after a meeting's `notes.ydoc` has
/// merged in an inbound update, so the placeholder's invented fields are replaced
/// by the converged authored values.
pub fn project_ydoc_meta_into_metadata(meetings_root: &Path, id: MeetingId) -> AppResult<bool> {
    let ydoc_path = meetings_root.join(id.0.to_string()).join("notes.ydoc");
    let bytes = match std::fs::read(&ydoc_path) {
        Ok(b) => b,
        Err(_) => return Ok(false), // no notes.ydoc yet — nothing to project.
    };
    let doc = match crate::ydoc::decode_ydoc(&bytes) {
        Ok(d) => d,
        Err(_) => return Ok(false), // unreadable blob — leave metadata.json as-is.
    };
    if !has_descriptive(&doc) {
        return Ok(false);
    }
    // RMW under the per-meeting metadata lock, mutating only descriptive fields.
    let mut applied = false;
    crate::metadata::update_metadata_if_present(meetings_root, id, |meta| {
        applied = project_into_meta(&doc, meta);
        // A short state label for the RMW helper's logging; unused otherwise.
        "meta-projected"
    })?;
    Ok(applied)
}

// --- small yrs helpers -------------------------------------------------------

/// Insert `value` (a serde-serialisable field) as a JSON-string map entry, or
/// remove the key when `value` is `None` (so clearing an optional converges).
fn put_json<T: Serialize>(txn: &mut TransactionMut, map: &MapRef, key: &str, value: Option<&T>) {
    match value.and_then(|v| serde_json::to_string(v).ok()) {
        Some(s) => {
            map.insert(txn, key, s);
        }
        None => {
            map.remove(txn, key);
        }
    }
}

/// Insert an optional string entry, or remove the key when `None`.
fn put_opt_str(txn: &mut TransactionMut, map: &MapRef, key: &str, value: Option<&str>) {
    match value {
        Some(s) => {
            map.insert(txn, key, s.to_string());
        }
        None => {
            map.remove(txn, key);
        }
    }
}

fn get_str<T: ReadTxn>(map: &MapRef, txn: &T, key: &str) -> Option<String> {
    match map.get(txn, key) {
        Some(Out::Any(Any::String(s))) => Some(s.to_string()),
        _ => None,
    }
}

fn get_bigint<T: ReadTxn>(map: &MapRef, txn: &T, key: &str) -> Option<i64> {
    match map.get(txn, key) {
        Some(Out::Any(Any::BigInt(i))) => Some(i),
        _ => None,
    }
}

fn get_json<D: DeserializeOwned>(map: &MapRef, txn: &impl ReadTxn, key: &str) -> Option<D> {
    match map.get(txn, key) {
        Some(Out::Any(Any::String(s))) => serde_json::from_str(&s).ok(),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use minutist_common::ProcessingLifecycle;

    fn base_meta() -> MeetingMeta {
        MeetingMeta {
            uuid: MeetingId::new(),
            title: "Original".to_string(),
            started_at: "2026-08-01T09:00:00Z".to_string(),
            ended_at: Some("2026-08-01T10:00:00Z".to_string()),
            duration_ms: 3_600_000,
            speaker_count: 3,
            audio_format: AudioFormat {
                codec: "aac".to_string(),
                sample_rate: 44_100,
                channels: 1,
                bitrate_kbps: None,
            },
            asr_model: Some(ModelDescriptor {
                name: "parakeet".to_string(),
                quantisation: Some("int8".to_string()),
                version: "v3".to_string(),
            }),
            llm_model: None,
            diarizer: None,
            speaker_names: BTreeMap::from([("A".to_string(), "Andrew".to_string())]),
            notes_format: 1,
            processing: ProcessingLifecycle::PendingProcessing,
            collection_id: None,
            app_version: "0.1.0".to_string(),
        }
    }

    /// A placeholder like `MeetingFolder::ensure` writes: wrong started_at, zero
    /// duration, opus codec, empty title — plus a per-peer `processing`/local
    /// field the projection must NOT touch.
    fn placeholder_meta(uuid: MeetingId) -> MeetingMeta {
        MeetingMeta {
            uuid,
            title: String::new(),
            started_at: "2026-08-09T23:59:00Z".to_string(), // arrival time — wrong
            ended_at: None,
            duration_ms: 0,
            speaker_count: 0,
            audio_format: AudioFormat {
                codec: "opus".to_string(),
                sample_rate: 16_000,
                channels: 1,
                bitrate_kbps: Some(32),
            },
            asr_model: None,
            llm_model: None,
            diarizer: None,
            speaker_names: BTreeMap::new(),
            notes_format: 0,
            processing: ProcessingLifecycle::Claimed {
                claim: minutist_common::ProcessingClaim {
                    host: minutist_common::HostRef("pilap".to_string()),
                    claimed_at: "2026-08-09T23:59:00Z".to_string(),
                    lease_expires_at: "2026-08-09T23:59:30Z".to_string(),
                },
            },
            collection_id: None,
            app_version: String::new(),
        }
    }

    #[test]
    fn write_then_project_round_trips_the_authored_fields() {
        let origin = crate::ydoc::new_ydoc();
        let src = base_meta();
        write_descriptive(&origin, &src);

        // A peer starts from a wrong placeholder and projects the origin's map.
        let mut dst = placeholder_meta(src.uuid);
        let placeholder_processing = dst.processing.clone();
        assert!(project_into_meta(&origin, &mut dst));

        // Authored fields adopted:
        assert_eq!(dst.title, "Original");
        assert_eq!(dst.started_at, "2026-08-01T09:00:00Z");
        assert_eq!(dst.ended_at.as_deref(), Some("2026-08-01T10:00:00Z"));
        assert_eq!(dst.duration_ms, 3_600_000);
        assert_eq!(dst.speaker_count, 3);
        assert_eq!(dst.audio_format.codec, "aac");
        assert_eq!(dst.audio_format.sample_rate, 44_100);
        assert_eq!(dst.asr_model.as_ref().unwrap().name, "parakeet");
        assert_eq!(dst.speaker_names.get("A").map(String::as_str), Some("Andrew"));
        // Per-peer / local fields preserved — NOT clobbered by the projection:
        assert_eq!(dst.uuid, src.uuid);
        assert_eq!(dst.processing, placeholder_processing);
        assert_eq!(dst.notes_format, 0);
        assert_eq!(dst.app_version, "");
    }

    #[test]
    fn projection_is_idempotent_and_reports_no_change_second_time() {
        let origin = crate::ydoc::new_ydoc();
        let src = base_meta();
        write_descriptive(&origin, &src);
        let mut dst = placeholder_meta(src.uuid);
        assert!(project_into_meta(&origin, &mut dst), "first projection changes fields");
        assert!(
            !project_into_meta(&origin, &mut dst),
            "second projection is a no-op"
        );
    }

    /// The whole point of riding the CRDT: two devices concurrently editing
    /// DIFFERENT fields both survive the merge (per-field convergence), and a
    /// device that adopted the origin then re-edits the title wins by LWW.
    #[test]
    fn concurrent_edits_to_distinct_fields_converge() {
        // Origin authors the meeting.
        let origin = crate::ydoc::new_ydoc();
        write_descriptive(&origin, &base_meta());
        let blob = crate::ydoc::encode_ydoc(&origin);

        // Device X and device Y both start from the origin state.
        let dev_x = crate::ydoc::decode_ydoc(&blob).unwrap();
        let dev_y = crate::ydoc::decode_ydoc(&blob).unwrap();

        // X renames the title; Y sets a new speaker name — different fields, via
        // the granular setters (the real edit path), each touching one key.
        set_title(&dev_x, "X-renamed");
        {
            let mut names = base_meta().speaker_names;
            names.insert("B".to_string(), "Beth".to_string());
            set_speaker_names(&dev_y, &names);
        }

        // Merge X and Y both ways (commutative).
        let x_update = crate::ydoc::encode_ydoc(&dev_x);
        let y_update = crate::ydoc::encode_ydoc(&dev_y);
        let merged = crate::ydoc::decode_ydoc(&blob).unwrap();
        {
            // apply both updates
            use yrs::updates::decoder::Decode;
            use yrs::{Transact, Update};
            let mut txn = merged.transact_mut();
            txn.apply_update(Update::decode_v2(&x_update).unwrap()).unwrap();
            txn.apply_update(Update::decode_v2(&y_update).unwrap()).unwrap();
        }

        let mut out = placeholder_meta(base_meta().uuid);
        project_into_meta(&merged, &mut out);
        assert_eq!(out.title, "X-renamed", "X's title edit survives");
        assert_eq!(
            out.speaker_names.get("B").map(String::as_str),
            Some("Beth"),
            "Y's speaker-name edit survives"
        );
        // A's original name is still there (not lost by Y's block-write, since
        // Y wrote the whole map including A).
        assert_eq!(out.speaker_names.get("A").map(String::as_str), Some("Andrew"));
    }
}
