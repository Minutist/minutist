//! Live end-to-end check of MEDIA-BLOB sync against the DEPLOYED relay
//! (`sync.minutist.ai`).
//!
//! The media counterpart to `relay_live.rs` (which exercises notes). Two
//! `SyncEngine`s bind with `RelayMode::Custom(sync.minutist.ai)` carrying the
//! relay access token, mutually pair, then one runs `sync_media` against the
//! other **addressed by relay URL only** (no direct socket addresses). The
//! manifest exchange rides the sync ALPN through the relay and the blob bytes
//! ride the blobs ALPN through the relay, so the whole transfer is
//! relay-brokered rather than going direct over loopback. A meeting seeded on A
//! with a synthetic `audio.opus` plus one note asset must arrive on B
//! byte-identical. Passing proves, against the real deployment: relay
//! reachability for the blobs ALPN, the shared-token `AccessControl`, and the
//! content-addressed media protocol — the things the in-process `media_sync`
//! tests (which use `RelayMode::Disabled` over loopback) cannot exercise.
//!
//! GATED: skips unless `MINUTIST_SYNC_TOKEN` is set, so a normal `cargo test`
//! and CI never touch the network. Run:
//!
//! ```sh
//! MINUTIST_SYNC_TOKEN=<relay-access-token> \
//!   cargo test -p sync --features test-support --test blobs_live -- --nocapture
//! ```
//! `MINUTIST_SYNC_RELAY` overrides the relay URL (default the crate's
//! `DEFAULT_RELAY_URL`).

use std::time::Duration;

use iroh::{EndpointAddr, RelayUrl};
use minutist_common::MeetingId;
use notes_crdt::MeetingFolder;
use persistence::save_note_asset;
use sync::{DeviceIdentity, SyncConfig, SyncEngine};

#[tokio::test]
async fn media_converges_through_the_deployed_relay() {
    let token = match std::env::var("MINUTIST_SYNC_TOKEN") {
        Ok(t) if !t.is_empty() => t,
        _ => {
            eprintln!("SKIP blobs_live: set MINUTIST_SYNC_TOKEN to run the live relay test");
            return;
        }
    };
    let relay_url = std::env::var("MINUTIST_SYNC_RELAY")
        .unwrap_or_else(|_| SyncConfig::DEFAULT_RELAY_URL.into());
    eprintln!("blobs_live: using relay {relay_url}");

    let dir_a = tempfile::TempDir::new().expect("tempdir a");
    let dir_b = tempfile::TempDir::new().expect("tempdir b");
    let id_a = DeviceIdentity::load_or_generate(dir_a.path()).expect("identity a");
    let id_b = DeviceIdentity::load_or_generate(dir_b.path()).expect("identity b");

    let cfg = |dir: &std::path::Path| SyncConfig {
        relay_url: relay_url.clone(),
        relay_auth_token: Some(token.clone()),
        // Live test conflates the device-key base and the meetings root onto one
        // temp dir: the key and the per-meeting folders both sit directly under
        // it. Production keeps them distinct (base vs `base/meetings`).
        meetings_root: dir.to_path_buf(),
        backoff_policy: Default::default(),
        dns_servers: Vec::new(),
    };

    let engine_a = SyncEngine::start(cfg(dir_a.path()), id_a)
        .await
        .expect("engine A binds");
    let engine_b = SyncEngine::start(cfg(dir_b.path()), id_b)
        .await
        .expect("engine B binds");
    eprintln!(
        "blobs_live: A={} B={}",
        engine_a.endpoint_id(),
        engine_b.endpoint_id()
    );

    // Mutual pairing: each device adds the other's ticket. Sync requires it on
    // BOTH ALPNs — B's sync-accept side rejects an unpaired initiator, and B's
    // blobs-accept side rejects an unpaired blob fetch — so A must be in B's
    // directory for the relay-routed exchange below to be served.
    engine_a
        .add_peer_from_ticket(&engine_b.my_ticket())
        .expect("A pairs B");
    engine_b
        .add_peer_from_ticket(&engine_a.my_ticket())
        .expect("B pairs A");

    // Let both endpoints home to the relay (the token-gated relay handshake).
    tokio::time::sleep(Duration::from_secs(3)).await;

    // Seed a meeting on A only: a synthetic audio.opus (a few KB; need not be
    // real Opus — the transfer is byte-oriented and content-addressed) plus one
    // note asset. B holds only the empty meeting folder.
    let meeting = MeetingId::new();
    let audio = b"OggS-synthetic-opus-payload-over-the-relay".repeat(128);
    let asset_bytes = b"\x89PNG\r\n\x1a\nsynthetic-pasted-image-over-the-relay".repeat(32);
    let folder_a = MeetingFolder::create(dir_a.path(), meeting).expect("create A meeting folder");
    std::fs::write(folder_a.audio_path(), &audio).expect("seed A audio.opus");
    let asset_name =
        save_note_asset(dir_a.path(), meeting, &asset_bytes, "png").expect("seed A asset");
    MeetingFolder::create(dir_b.path(), meeting).expect("create B meeting folder");
    eprintln!(
        "blobs_live: seeded A audio={} bytes, asset {} ({} bytes)",
        audio.len(),
        asset_name,
        asset_bytes.len()
    );

    // Address B by RELAY ONLY — no direct IPs — so both the manifest stream and
    // the blob pulls are brokered by the deployed relay rather than connecting
    // directly over loopback.
    let relay: RelayUrl = relay_url.parse().expect("relay url parses");
    let b_relay_only = EndpointAddr::new(engine_b.endpoint_id()).with_relay_url(relay);

    // A dials B through the relay and reconciles the meeting's media (30s guard so
    // a relay failure surfaces as a timeout, not a hang).
    let started = std::time::Instant::now();
    tokio::time::timeout(
        Duration::from_secs(30),
        engine_a.sync_media(b_relay_only, meeting),
    )
    .await
    .expect("relay-routed media sync timed out")
    .expect("relay-routed media sync failed");
    eprintln!(
        "blobs_live: media reconciliation completed in {:?}",
        started.elapsed()
    );

    // B must now hold A's audio byte-for-byte.
    let b_audio = std::fs::read(dir_b.path().join(meeting.0.to_string()).join("audio.opus"))
        .expect("B must have audio.opus after media sync");
    assert_eq!(
        b_audio, audio,
        "B's audio must be byte-identical to A's after relay-routed sync"
    );

    // B must hold A's asset byte-for-byte under the same portable filename.
    let b_asset = persistence::read_note_asset(dir_b.path(), meeting, &asset_name)
        .expect("B must have the asset after media sync");
    assert_eq!(
        b_asset, asset_bytes,
        "B's asset must be byte-identical to A's after relay-routed sync"
    );

    engine_a.shutdown().await.expect("shutdown a");
    engine_b.shutdown().await.expect("shutdown b");
    eprintln!("blobs_live: PASS — media converged through {relay_url}");
}
