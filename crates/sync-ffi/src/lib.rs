//! UniFFI wrapper over [`sync::SyncEngine`] for the Android phone companion.
//!
//! The phone is just another paired iroh endpoint (issue 0016, Option A): this
//! crate compiles the desktop `sync` crate to an `aarch64-linux-android` `.so`
//! and exposes its transport surface to Kotlin through UniFFI, called from a
//! Capacitor plugin. The wire protocol is unchanged.
//!
//! Boundary discipline — only strings, bytes, and plain records cross the
//! UniFFI surface, and this crate takes no `iroh` dependency of its own:
//!
//! - peers are [`EndpointTicket`](iroh_tickets) strings ([`FfiSyncEngine::my_ticket`]
//!   / [`FfiSyncEngine::pair`]) and thereafter hex peer-id strings — `peer_ids` /
//!   `subscribe_peers` hand them back, and the `*_to_peer` / `*_with_peer`
//!   methods (which relay-address the peer internally — see
//!   `SyncEngine::push_all_to_peer`) take them straight back in;
//! - meetings are hyphenated UUID strings (`MeetingId`'s serde form);
//! - the discovery lifecycle is the typed [`FfiLifecycle`] enum, mapped from
//!   [`minutist_common::ProcessingLifecycle`];
//! - `sync::Error` is mapped to the coarse [`SyncFfiError`] categories so the
//!   phone can tell a transient transport failure from a fatal protocol one.
//!
//! Runtime — [`SyncEngine`] holds no tokio runtime (its accept loop is spawned
//! under an ambient one) and the phone has no Tauri/ambient runtime, so the
//! wrapper owns a multi-thread [`tokio::runtime::Runtime`]: synchronous FFI
//! methods `block_on` it (Capacitor dispatches native calls off the main
//! thread). Event subscriptions are drained on **dedicated OS threads, not
//! runtime tasks**, so a foreign callback that re-enters the wrapper — `block_on`
//! ing the runtime, as [`LifecycleListener::on_lagged`] prescribes — runs
//! off-runtime instead of panicking "cannot start a runtime from within a
//! runtime" (which would unwind across the FFI boundary and abort).

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use minutist_common::{AppError, AudioFormat, MeetingId, MeetingMeta, ProcessingLifecycle, Segment};
use notes_crdt::{MeetingFolder, NotesStore};
use sync::{DeviceIdentity, SyncConfig, SyncEngine};
use tokio::runtime::Runtime;
use tokio::sync::broadcast::{self, error::RecvError};
use uuid::Uuid;

uniffi::setup_scaffolding!();

/// Errors surfaced across the FFI boundary. The four `sync::Error` variants map
/// to coarse categories so the phone can distinguish a transient failure
/// (`Transport` / `Io` — e.g. a peer offline, worth retrying) from a fatal one
/// (`Protocol` — a malformed ticket or version mismatch). `msg` carries the
/// detail for display / logs.
#[derive(Debug, thiserror::Error, uniffi::Error)]
pub enum SyncFfiError {
    /// Endpoint bind or peer dial failed — typically transient (peer offline,
    /// relay unreachable); the caller may retry.
    #[error("transport: {msg}")]
    Transport { msg: String },
    /// A sync protocol exchange failed (malformed ticket, version mismatch) —
    /// not retryable as-is.
    #[error("protocol: {msg}")]
    Protocol { msg: String },
    /// Loading or generating the device identity failed.
    #[error("identity: {msg}")]
    Identity { msg: String },
    /// A filesystem / runtime io operation failed.
    #[error("io: {msg}")]
    Io { msg: String },
    /// A meeting-folder read/write (the phone data layer) failed — metadata /
    /// transcript / notes serialisation or storage.
    #[error("data: {msg}")]
    Data { msg: String },
    /// An argument off the boundary did not parse (e.g. a meeting-id string).
    #[error("invalid argument: {msg}")]
    InvalidArg { msg: String },
    /// A method was called after [`FfiSyncEngine::shutdown`].
    #[error("sync engine is closed")]
    Closed,
}

impl From<sync::Error> for SyncFfiError {
    fn from(e: sync::Error) -> Self {
        match e {
            sync::Error::Endpoint(msg) => SyncFfiError::Transport { msg },
            sync::Error::Protocol(msg) => SyncFfiError::Protocol { msg },
            sync::Error::Identity(msg) => SyncFfiError::Identity { msg },
            sync::Error::Io(e) => SyncFfiError::Io { msg: e.to_string() },
        }
    }
}

impl From<AppError> for SyncFfiError {
    /// The notes-crdt data-layer helpers (`MeetingFolder` / `NotesStore` /
    /// `read_metadata` / `update_metadata_if_present`) surface
    /// [`minutist_common::AppError`]. An io failure keeps the retryable `Io`
    /// category; everything else (serialisation, invalid state) is `Data`.
    fn from(e: AppError) -> Self {
        match e {
            AppError::Io { context } => SyncFfiError::Io { msg: context },
            other => SyncFfiError::Data {
                msg: other.to_string(),
            },
        }
    }
}

impl From<notes_crdt::Error> for SyncFfiError {
    fn from(e: notes_crdt::Error) -> Self {
        SyncFfiError::from(AppError::from(e))
    }
}

/// The FFI-facing projection of [`minutist_common::ProcessingLifecycle`]. A typed
/// enum (rather than a JSON string) so the Kotlin side dispatches on the variant
/// directly; the opaque `HostRef` device key is carried as its string.
#[derive(Debug, Clone, uniffi::Enum)]
pub enum FfiLifecycle {
    /// Recorded and processed on one device (never authored by the phone).
    Local,
    /// Captured here and offered for a host to adopt.
    PendingProcessing,
    /// A host has claimed the meeting and is producing its outputs.
    Claimed {
        host: String,
        claimed_at: String,
        lease_expires_at: String,
    },
    /// Processing finished; the derived outputs are authoritative.
    Processed { processed_by: String, at: String },
}

impl From<ProcessingLifecycle> for FfiLifecycle {
    fn from(lc: ProcessingLifecycle) -> Self {
        match lc {
            ProcessingLifecycle::Local => FfiLifecycle::Local,
            ProcessingLifecycle::PendingProcessing => FfiLifecycle::PendingProcessing,
            ProcessingLifecycle::Claimed { claim } => FfiLifecycle::Claimed {
                host: claim.host.0,
                claimed_at: claim.claimed_at,
                lease_expires_at: claim.lease_expires_at,
            },
            ProcessingLifecycle::Processed { processed_by, at } => FfiLifecycle::Processed {
                processed_by: processed_by.0,
                at,
            },
        }
    }
}

/// One transcript segment in the phone's view of a synced meeting. Mirrors the
/// mobile `TranscriptSegment`: `speaker_index` is 1-based into
/// [`FfiMeeting::Synced`]'s `speakers` palette (`0` when the segment carries no
/// diarised speaker).
#[derive(Debug, Clone, uniffi::Record)]
pub struct FfiSegment {
    pub speaker_index: u32,
    pub start_ms: i64,
    pub end_ms: i64,
    pub text: String,
}

/// The capture-lifecycle sub-state of a [`FfiMeeting::Captured`] meeting — the two
/// states the phone holds before results arrive (mirrors the mobile model's
/// `processing: 'pending' | 'claimed'`).
#[derive(Debug, Clone, uniffi::Enum)]
pub enum FfiCaptureState {
    /// Offered for a host to adopt (`PendingProcessing`).
    Pending,
    /// A host is producing the derived outputs (`Claimed`).
    Claimed,
}

/// The phone's view of a meeting — the two mutually-exclusive states the mobile
/// `Meeting` union carries. Projected from the meeting folder by
/// [`project_meeting`]: a meeting awaiting results (`PendingProcessing` /
/// `Claimed`) is [`Self::Captured`]; one whose outputs exist (`Processed`, or the
/// desktop-only `Local`) is [`Self::Synced`].
#[derive(Debug, Clone, uniffi::Enum)]
pub enum FfiMeeting {
    /// Recorded here (or on a delegating device), awaiting processing.
    Captured {
        id: String,
        title: String,
        started_at_ms: i64,
        duration_ms: i64,
        /// `audio.opus` is present in the folder.
        has_audio: bool,
        /// `notes.ydoc` is present in the folder.
        has_notes: bool,
        processing: FfiCaptureState,
        /// The `HostRef` string a host stamped when `Claimed` (resolution to a
        /// friendly name is the caller's — see the mobile `claimedBy` doc).
        claimed_by: Option<String>,
    },
    /// Processed and synced back: derived outputs are present and read-only here.
    Synced {
        id: String,
        title: String,
        started_at_ms: i64,
        transcript: Vec<FfiSegment>,
        summary: Option<String>,
        /// Display labels indexed from 1 (`speakers[i-1]` for `speaker_index = i`).
        speakers: Vec<String>,
        /// The authoritative `notes.ydoc` as Yjs v1 update bytes (opaque — render
        /// via a Yjs doc), or `None` when the meeting has no notes.
        notes: Option<Vec<u8>>,
    },
}

/// Foreign-implemented sink for discovery lifecycle events. The wrapper drains
/// [`SyncEngine::subscribe_lifecycle_events`] and invokes this for each
/// `(meeting, lifecycle)`. On [`Self::on_lagged`] the consumer recovers by
/// re-running discovery (the bounded channel dropped the oldest events).
#[uniffi::export(callback_interface)]
pub trait LifecycleListener: Send + Sync {
    /// A meeting's processing lifecycle was received from a peer's discovery
    /// exchange. `meeting_id` is a hyphenated UUID string.
    fn on_lifecycle(&self, meeting_id: String, lifecycle: FfiLifecycle);
    /// The bounded channel lagged; events were dropped. Recover by re-running
    /// discovery against known peers.
    fn on_lagged(&self);
}

/// Foreign-implemented sink for "peer arrived" events (an always-on consumer's
/// reciprocal-push trigger). Mirrors [`SyncEngine::subscribe_peer_events`].
#[uniffi::export(callback_interface)]
pub trait PeerListener: Send + Sync {
    /// A peer opened an authorised inbound sync connection. `peer_id` is hex.
    fn on_peer_arrived(&self, peer_id: String);
    /// The bounded channel lagged; recover by reconciling all known peers.
    fn on_lagged(&self);
}

/// A running sync endpoint for one device. Wraps [`sync::SyncEngine`] and owns
/// the tokio runtime it runs on.
#[derive(uniffi::Object)]
pub struct FfiSyncEngine {
    /// `None` after [`Self::shutdown`]. The brief lock is held only to clone the
    /// `Arc` (or `take` it on shutdown) — never across a `block_on`, so calls run
    /// concurrently on the multi-thread runtime.
    inner: Mutex<Option<Arc<SyncEngine>>>,
    rt: Arc<Runtime>,
    /// The per-meeting `{uuid}` folder root, kept for the data-layer methods
    /// (save / list / get) and the inbound lifecycle-apply — they read and write
    /// `metadata.json` directly via the C-free notes-crdt primitives, independent
    /// of the engine's transport.
    meetings_root: PathBuf,
}

#[uniffi::export]
impl FfiSyncEngine {
    /// Bind the iroh endpoint against the relay and start the accept loop.
    ///
    /// `app_data_dir` holds the `0600` device key (loaded or generated by
    /// [`DeviceIdentity::load_or_generate`] — the ed25519 secret never crosses
    /// the boundary). `meetings_root` is the directory of per-meeting `{uuid}`
    /// folders. `relay_auth_token` is the relay admission token when the account
    /// service has issued one. The relay url is held inside the engine and used
    /// to address peers for the per-meeting sync methods.
    #[uniffi::constructor]
    pub fn start(
        relay_url: String,
        relay_auth_token: Option<String>,
        meetings_root: String,
        app_data_dir: String,
        relay_ips: Vec<String>,
    ) -> Result<Arc<Self>, SyncFfiError> {
        // Route iroh's relay/handshake/net_report tracing to logcat on Android
        // (no-op on host builds). Idempotent; safe to call on every `start`.
        init_ffi_tracing();

        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .map_err(|e| SyncFfiError::Io {
                msg: format!("building tokio runtime: {e}"),
            })?;

        let identity = DeviceIdentity::load_or_generate(Path::new(&app_data_dir))?;
        let meetings_root = PathBuf::from(meetings_root);
        let config = SyncConfig {
            relay_url,
            relay_auth_token,
            meetings_root: meetings_root.clone(),
            backoff_policy: Default::default(),
            relay_ips,
        };
        let engine = rt.block_on(SyncEngine::start(config, identity))?;

        Ok(Arc::new(Self {
            inner: Mutex::new(Some(Arc::new(engine))),
            rt: Arc::new(rt),
            meetings_root,
        }))
    }

    /// This device's shareable ticket — copy it to a paired device, which feeds
    /// it to [`Self::pair`].
    pub fn my_ticket(&self) -> Result<String, SyncFfiError> {
        Ok(self.engine()?.my_ticket())
    }

    /// This device's hex endpoint id (for diagnostics / logging).
    pub fn endpoint_id(&self) -> Result<String, SyncFfiError> {
        Ok(self.engine()?.endpoint_id().to_string())
    }

    /// Register a peer from its ticket (the other device's [`Self::my_ticket`]).
    /// Sync requires MUTUAL pairing — each device pairs the other. Returns the
    /// peer's hex endpoint id.
    pub fn pair(&self, ticket: String) -> Result<String, SyncFfiError> {
        Ok(self.engine()?.add_peer_from_ticket(&ticket)?.to_string())
    }

    /// Register a peer learned from the account service (the phone's own
    /// list→add loop over `GET /v1/account/devices`), addressed by its hex
    /// endpoint id, relay URL, and published direct socket addresses
    /// ("ip:port"). Wraps [`sync::SyncEngine::add_account_peer`]; no `iroh` type
    /// crosses the boundary. Additive to [`Self::pair`] — both feed the same
    /// peer directory.
    ///
    /// `direct_addrs` lets a same-tailnet/LAN peer dial this device directly
    /// (no relay, no DNS — the crux of 0049 on Android, where the relay
    /// hostname does not resolve in-process under a full-tunnel VPN). Pass an
    /// empty list for relay-only addressing. Unparseable entries are skipped.
    pub fn add_account_peer(
        &self,
        endpoint_id: String,
        relay_url: String,
        direct_addrs: Vec<String>,
    ) -> Result<(), SyncFfiError> {
        Ok(self
            .engine()?
            .add_account_peer(&endpoint_id, &relay_url, &direct_addrs)?)
    }

    /// This device's own publishable direct socket addresses ("ip:port"), for
    /// the caller to POST to the account directory when it self-registers
    /// (0049). On the phone, register-self is driven from TypeScript via the
    /// account client (not the Rust engine), so this getter exposes the
    /// engine's filtered direct-addr set — same filter the desktop/hub applies
    /// (drops loopback/link-local/docker-bridge). Wraps
    /// [`sync::SyncEngine::publishable_direct_addrs`].
    pub fn own_direct_addrs(&self) -> Result<Vec<String>, SyncFfiError> {
        Ok(self.engine()?.publishable_direct_addrs())
    }

    /// Remove an account-sourced peer no longer present in the account's
    /// device list (reconcile — it left the account). Source-aware: a no-op
    /// (returns `false`) if `endpoint_id` was paired any other way (e.g.
    /// [`Self::pair`]). Wraps [`sync::SyncEngine::remove_account_peer`].
    pub fn remove_account_peer(&self, endpoint_id: String) -> Result<bool, SyncFfiError> {
        Ok(self.engine()?.remove_account_peer(&endpoint_id)?)
    }

    /// Hex endpoint ids of every currently-registered peer.
    pub fn peer_ids(&self) -> Result<Vec<String>, SyncFfiError> {
        // `SyncEngine::peer_ids` is already hex-string-keyed, so no per-id
        // conversion is needed here (unlike `endpoint_id` / `pair`, which map an
        // `iroh`-typed return at the boundary).
        Ok(self.engine()?.peer_ids())
    }

    /// Hyphenated UUID strings of the meetings this device holds on disk.
    pub fn local_meetings(&self) -> Result<Vec<String>, SyncFfiError> {
        Ok(self
            .engine()?
            .local_meetings()
            .into_iter()
            .map(|m| m.0.to_string())
            .collect())
    }

    /// Persist a freshly-recorded meeting as captured-unprocessed and return its
    /// new id (a hyphenated UUID string). Writes `metadata.json` (as
    /// `PendingProcessing`), optionally copies `audio_src_path` into the folder
    /// under the honest container extension — `audio.opus` for Ogg-Opus,
    /// `audio.m4a` for AAC-in-MP4 (the phone's fallback when the device has no
    /// Opus encoder), sniffed from the source bytes so metadata and filename
    /// match the real codec — and, when `notes_text` is non-empty, seeds
    /// `notes.ydoc` from it. `started_at_ms` is Unix epoch milliseconds. Does not
    /// require a started engine — purely local.
    pub fn save_captured(
        &self,
        title: String,
        started_at_ms: i64,
        duration_ms: i64,
        audio_src_path: Option<String>,
        notes_text: String,
    ) -> Result<String, SyncFfiError> {
        save_captured_to(
            &self.meetings_root,
            &title,
            started_at_ms,
            duration_ms,
            audio_src_path.as_deref(),
            &notes_text,
        )
    }

    /// Every meeting this device holds, projected to the phone model, newest
    /// first (by start time). A folder that fails to read is skipped, not fatal.
    pub fn list_meetings(&self) -> Result<Vec<FfiMeeting>, SyncFfiError> {
        let mut meetings: Vec<FfiMeeting> = notes_crdt::folder::list_meeting_ids(&self.meetings_root)
            .into_iter()
            .filter_map(|id| project_meeting(&self.meetings_root, id))
            .collect();
        meetings.sort_by_key(|m| std::cmp::Reverse(started_at_ms_of(m)));
        Ok(meetings)
    }

    /// Project a single meeting by id, or `None` if its folder is absent /
    /// unreadable.
    pub fn get_meeting(&self, meeting_id: String) -> Result<Option<FfiMeeting>, SyncFfiError> {
        let id = parse_meeting(&meeting_id)?;
        Ok(project_meeting(&self.meetings_root, id))
    }

    /// Reconcile one meeting's notes with a paired peer (blocking).
    pub fn sync_notes(&self, peer_id: String, meeting_id: String) -> Result<(), SyncFfiError> {
        let engine = self.engine()?;
        let meeting = parse_meeting(&meeting_id)?;
        self.rt
            .block_on(engine.sync_notes_to_peer(&peer_id, meeting))?;
        Ok(())
    }

    /// Reconcile one meeting's media (audio + note assets) with a paired peer.
    pub fn sync_media(&self, peer_id: String, meeting_id: String) -> Result<(), SyncFfiError> {
        let engine = self.engine()?;
        let meeting = parse_meeting(&meeting_id)?;
        self.rt
            .block_on(engine.sync_media_to_peer(&peer_id, meeting))?;
        Ok(())
    }

    /// Reconcile one meeting's derived artifacts (`transcript.json` +
    /// `summary.md`) with a paired peer — the phone-initiated pull. The exchange
    /// is bidirectional (`producer_authority`-gated): this device receives any of
    /// the peer's artifacts that supersede its own, and offers its own in return.
    /// Complements the host's push
    /// ([`DesktopElectionDriver::push_artifacts`](../../src-tauri/src/sync.rs)) so
    /// a passive or reconnecting capture device can proactively fetch a processing
    /// host's outputs rather than depend on being reachable at the host's push
    /// moment. Blocking.
    pub fn sync_artifacts(&self, peer_id: String, meeting_id: String) -> Result<(), SyncFfiError> {
        let engine = self.engine()?;
        let meeting = parse_meeting(&meeting_id)?;
        self.rt
            .block_on(engine.sync_artifacts_to_peer(&peer_id, meeting))?;
        Ok(())
    }

    /// Exchange the meeting list + lifecycle with a peer; returns the peer's
    /// meeting ids (hyphenated UUID strings). Each received lifecycle also fires
    /// on a registered [`LifecycleListener`].
    pub fn discover_with(&self, peer_id: String) -> Result<Vec<String>, SyncFfiError> {
        let engine = self.engine()?;
        let ids = self.rt.block_on(engine.discover_with_peer(&peer_id))?;
        Ok(ids.into_iter().map(|m| m.0.to_string()).collect())
    }

    /// Drain discovery lifecycle events into `listener` until shutdown. One
    /// listener per call (the subscriptions are not individually cancellable;
    /// the drain thread exits when the engine drops). See [`spawn_drain`].
    pub fn subscribe_lifecycle(
        &self,
        listener: Box<dyn LifecycleListener>,
    ) -> Result<(), SyncFfiError> {
        let rx = self.engine()?.subscribe_lifecycle_events();
        let meetings_root = self.meetings_root.clone();
        spawn_drain(rx, move |item| match item {
            Drained::Event((meeting, lifecycle)) => {
                let id_str = meeting.0.to_string();
                // Persist the inbound state to our local metadata.json FIRST,
                // then notify the UI to re-snapshot — mirroring the desktop's
                // apply_synced_lifecycle_if_present, persistence-free on the lifted
                // notes-crdt primitives (merge by precedence so a stale advert
                // cannot regress local state; a meeting we don't hold yet is
                // skipped). The drain runs on a dedicated OS thread, so this sync
                // RMW is fine.
                apply_inbound_lifecycle(&meetings_root, meeting, lifecycle.clone());
                listener.on_lifecycle(id_str, lifecycle.into());
            }
            Drained::Lagged => listener.on_lagged(),
        });
        Ok(())
    }

    /// Drain "peer arrived" events into `listener` until shutdown. One listener
    /// per call (see [`Self::subscribe_lifecycle`]).
    pub fn subscribe_peers(&self, listener: Box<dyn PeerListener>) -> Result<(), SyncFfiError> {
        // `SyncEngine::subscribe_peer_events` is already hex-string-keyed, so the
        // event is forwarded as-is (unlike the lifecycle stream, whose
        // `MeetingId` still needs its own string projection).
        let rx = self.engine()?.subscribe_peer_events();
        spawn_drain(rx, move |item| match item {
            Drained::Event(peer) => listener.on_peer_arrived(peer),
            Drained::Lagged => listener.on_lagged(),
        });
        Ok(())
    }

    /// Gracefully shut the endpoint down. Idempotent; subsequent calls return
    /// [`SyncFfiError::Closed`] from the other methods. A no-op if a call is
    /// still in flight (the engine then drops when the last reference releases).
    pub fn shutdown(&self) -> Result<(), SyncFfiError> {
        let Some(arc) = self.inner.lock().expect("sync engine lock poisoned").take() else {
            return Ok(());
        };
        match Arc::try_unwrap(arc) {
            Ok(engine) => {
                self.rt.block_on(engine.shutdown())?;
                Ok(())
            }
            // A method is mid-flight holding a clone; it drops the engine when it
            // returns. The graceful router-shutdown is skipped, which iroh's Drop
            // backstops.
            Err(_) => Ok(()),
        }
    }
}

impl FfiSyncEngine {
    /// Clone the live engine `Arc`, or [`SyncFfiError::Closed`] after shutdown.
    /// The lock is released with the guard before the caller's `block_on`.
    fn engine(&self) -> Result<Arc<SyncEngine>, SyncFfiError> {
        self.inner
            .lock()
            .expect("sync engine lock poisoned")
            .clone()
            .ok_or(SyncFfiError::Closed)
    }
}

/// One item drained from a broadcast subscription.
enum Drained<T> {
    Event(T),
    /// The bounded channel lagged; the oldest items were dropped.
    Lagged,
}

/// Drain a broadcast receiver on a DEDICATED OS THREAD (not a runtime task) so
/// the foreign `sink` callback runs OFF the tokio runtime. A callback that
/// re-enters an [`FfiSyncEngine`] method (whose `block_on` drives the runtime)
/// then lands on a non-runtime thread instead of panicking "cannot start a
/// runtime from within a runtime" — a panic that would unwind across the FFI
/// boundary and abort. The thread exits when the channel closes (the engine,
/// which holds the sender, drops).
fn spawn_drain<T, F>(mut rx: broadcast::Receiver<T>, mut sink: F)
where
    T: Clone + Send + 'static,
    F: FnMut(Drained<T>) + Send + 'static,
{
    std::thread::spawn(move || loop {
        match rx.blocking_recv() {
            Ok(event) => sink(Drained::Event(event)),
            Err(RecvError::Lagged(_)) => sink(Drained::Lagged),
            Err(RecvError::Closed) => break,
        }
    });
}

// ---------------------------------------------------------------------------
// Phone data layer (persistence-free): save / project / lifecycle-apply over the
// meeting folder, on the C-free notes-crdt primitives. Factored as free fns so
// they are testable against a tempdir root without a started `FfiSyncEngine`.
// ---------------------------------------------------------------------------

/// Install a `tracing` subscriber routing this crate's (and iroh's) output to
/// Android logcat, so the phone's relay-actor / handshake / `net_report` lines
/// are visible during diagnosis — the `.so` otherwise installs no subscriber and
/// drops all `tracing` output. Idempotent (a `Once` guards it against every
/// `start`) and Android-only; on the host it compiles to a no-op (the
/// desktop/headless consumers install their own subscriber).
#[cfg(target_os = "android")]
fn init_ffi_tracing() {
    use std::sync::Once;
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        use tracing_subscriber::layer::SubscriberExt;
        use tracing_subscriber::util::SubscriberInitExt;
        // Verbose on the relay/transport path under investigation, quiet
        // elsewhere; `RUST_LOG` overrides when the host sets one.
        let filter = tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| {
            tracing_subscriber::EnvFilter::new("info,sync=debug,iroh=debug,iroh_relay=debug")
        });
        let _ = tracing_subscriber::registry()
            .with(filter)
            .with(
                tracing_subscriber::fmt::layer()
                    .with_ansi(false)
                    .with_writer(paranoid_android::AndroidLogMakeWriter::new(
                        "minutist.sync".to_owned(),
                    )),
            )
            .try_init();
    });
}

/// Host builds install no subscriber here (consumers own theirs) — no-op.
#[cfg(not(target_os = "android"))]
fn init_ffi_tracing() {}

/// The `started_at_ms` of either [`FfiMeeting`] variant (for list ordering).
fn started_at_ms_of(m: &FfiMeeting) -> i64 {
    match m {
        FfiMeeting::Captured { started_at_ms, .. } | FfiMeeting::Synced { started_at_ms, .. } => {
            *started_at_ms
        }
    }
}

/// Unix epoch milliseconds → an RFC 3339 UTC string (the form `metadata.json`
/// stores). Falls back to the epoch on an out-of-range input rather than failing.
fn ms_to_rfc3339(ms: i64) -> String {
    chrono::DateTime::<chrono::Utc>::from_timestamp_millis(ms)
        .unwrap_or_default()
        .to_rfc3339()
}

/// An RFC 3339 timestamp → Unix epoch milliseconds, or `0` if it does not parse
/// (a meeting with an unreadable timestamp still lists, sorted oldest).
fn rfc3339_to_ms(s: &str) -> i64 {
    chrono::DateTime::parse_from_rfc3339(s)
        .map(|dt| dt.timestamp_millis())
        .unwrap_or(0)
}

/// Write a captured-unprocessed meeting (the [`FfiSyncEngine::save_captured`] body,
/// factored out for direct testing). Returns the new meeting id string.
fn save_captured_to(
    meetings_root: &Path,
    title: &str,
    started_at_ms: i64,
    duration_ms: i64,
    audio_src_path: Option<&str>,
    notes_text: &str,
) -> Result<String, SyncFfiError> {
    let id = MeetingId::new();
    let folder = MeetingFolder::ensure(meetings_root, id)?;

    // All-or-nothing: roll back the freshly-created folder if any write below
    // fails, so a partial save can't leave a useless audio-less / metadata-only
    // PendingProcessing meeting on disk. The caller retries (the platform audio
    // temp persists upstream); the meeting id is fresh, so the folder is ours to
    // remove.
    let write = (|| -> Result<(), SyncFfiError> {
        // Determine the REAL audio format up front so metadata + the stored
        // filename match the actual bytes. The phone's on-device Opus encoder is
        // often absent, so the recorder falls back to AAC-in-MP4; the previous
        // hardcoded `audio.opus` + codec:"opus" was a mislabel the desktop's
        // Ogg-Opus decoder could not read. The extension is how the file is
        // resolved on disk (`minutist_common::resolve_audio_path`), and decode
        // dispatches by it — so it must be honest.
        let (audio_ext, codec, sample_rate, channels) = match audio_src_path {
            Some(src) => sniff_captured_audio(Path::new(src))?,
            None => ("opus", "opus".to_string(), 16_000, 1),
        };
        let meta = MeetingMeta {
            uuid: id,
            title: title.to_string(),
            started_at: ms_to_rfc3339(started_at_ms),
            ended_at: None,
            duration_ms: duration_ms.max(0) as u64,
            speaker_count: 0,
            audio_format: AudioFormat {
                codec,
                sample_rate,
                channels,
                bitrate_kbps: None,
            },
            asr_model: None,
            llm_model: None,
            diarizer: None,
            speaker_names: std::collections::BTreeMap::new(),
            notes_format: 1,
            processing: ProcessingLifecycle::PendingProcessing,
            collection_id: None,
            recording_started: true,
            app_version: env!("CARGO_PKG_VERSION").to_string(),
        };
        notes_crdt::write_metadata(&folder, &meta)?;

        if let Some(src) = audio_src_path {
            std::fs::copy(src, folder.join(format!("audio.{audio_ext}"))).map_err(|e| {
                SyncFfiError::Io {
                    msg: format!("copying captured audio: {e}"),
                }
            })?;
        }

        // Create the meeting's `notes.ydoc` seeded with the authored metadata map
        // (0052) so the descriptive fields converge across devices. Universal: a
        // notes-less recording still gets a `notes.ydoc` (empty prosemirror) so the
        // map has a sync transport. When note text is present it becomes the
        // initial prosemirror content in the same doc.
        let notes_json = if notes_text.is_empty() {
            None
        } else {
            Some(serde_json::json!({
                "type": "doc",
                "content": [{
                    "type": "paragraph",
                    "content": [{ "type": "text", "text": notes_text }],
                }],
            }))
        };
        notes_crdt::meta_crdt::initialise_notes_with_meta(
            meetings_root,
            id,
            notes_json.as_ref(),
            notes_text,
            &meta,
        )?;
        Ok(())
    })();

    if let Err(e) = write {
        let _ = std::fs::remove_dir_all(&folder);
        return Err(e);
    }

    tracing::info!(target: "sync-ffi", meeting_id = %id.0, "captured meeting saved");
    Ok(id.0.to_string())
}

/// Whether a meeting folder has real note content — a non-empty prosemirror
/// document, not merely a `notes.ydoc` that exists to carry the descriptive
/// metadata map (0052: every meeting has a `notes.ydoc`, so file existence no
/// longer implies notes). Absent/undecodable `notes.ydoc` → no notes.
fn folder_has_notes_content(folder: &Path) -> bool {
    std::fs::read(folder.join("notes.ydoc"))
        .ok()
        .and_then(|bytes| notes_crdt::ydoc::decode_ydoc(&bytes).ok())
        .map(|doc| notes_crdt::ydoc::has_notes_content(&doc))
        .unwrap_or(false)
}

/// Sniff a captured audio file's real format from its container magic: returns
/// `(extension, codec label, sample_rate, channels)`. The extension + codec come
/// from the bytes, not from the recorder's own belief about the codec (that
/// belief is the mislabel bug this fixes). Rate/channels are read from the
/// container. Errors if it is not a supported audio container.
fn sniff_captured_audio(src: &Path) -> Result<(&'static str, String, u32, u16), SyncFfiError> {
    let bytes = std::fs::read(src).map_err(|e| SyncFfiError::Io {
        msg: format!("reading captured audio: {e}"),
    })?;
    if bytes.len() >= 4 && &bytes[0..4] == b"OggS" {
        // Ogg/Opus: our encoder contract is a fixed 16 kHz mono.
        return Ok(("opus", "opus".to_string(), 16_000, 1));
    }
    if bytes.len() >= 8 && &bytes[4..8] == b"ftyp" {
        // MP4/AAC (Android MediaRecorder). True rate/channels from the mp4a box;
        // fall back to the recorder's configured 44.1 kHz mono if the walk can't
        // locate it (a well-formed recorder file always can) — never a false
        // 16 kHz that would misrepresent the audio to the decoder/consumer.
        let (rate, ch) = mp4a_audio_params(&bytes).unwrap_or((44_100, 1));
        return Ok(("m4a", "aac".to_string(), rate, ch));
    }
    Err(SyncFfiError::InvalidArg {
        msg: "captured audio is neither Ogg/Opus nor MP4 — refusing to mislabel".into(),
    })
}

/// Walk an MP4 to the first `mp4a` AudioSampleEntry and return
/// `(sample_rate, channels)`. Bounded and defensive: any structural surprise
/// yields `None` and the caller falls back to the recorder's configured values.
fn mp4a_audio_params(data: &[u8]) -> Option<(u32, u16)> {
    // Return the BODY (after the 8-byte box header) of the first child box `want`.
    fn child<'a>(mut buf: &'a [u8], want: &[u8; 4]) -> Option<&'a [u8]> {
        while buf.len() >= 8 {
            let size = u32::from_be_bytes([buf[0], buf[1], buf[2], buf[3]]) as usize;
            let end = if size == 0 {
                buf.len()
            } else if size < 8 || size > buf.len() {
                return None;
            } else {
                size
            };
            if &buf[4..8] == want {
                return Some(&buf[8..end]);
            }
            if size == 0 {
                return None;
            }
            buf = &buf[end..];
        }
        None
    }
    let stsd = child(
        child(
            child(
                child(child(child(data, b"moov")?, b"trak")?, b"mdia")?,
                b"minf",
            )?,
            b"stbl",
        )?,
        b"stsd",
    )?;
    // stsd: 4 bytes version+flags, 4 bytes entry_count, then the sample entries.
    let mp4a = child(stsd.get(8..)?, b"mp4a")?;
    // AudioSampleEntry body: 8 (SampleEntry) + 8 (reserved) then channelcount(2),
    // samplesize(2), predefined(2), reserved(2), samplerate(4, 16.16 fixed).
    let channels = u16::from_be_bytes([*mp4a.get(16)?, *mp4a.get(17)?]).max(1);
    let sr = u32::from_be_bytes([
        *mp4a.get(24)?,
        *mp4a.get(25)?,
        *mp4a.get(26)?,
        *mp4a.get(27)?,
    ]);
    Some((sr >> 16, channels)) // 16.16 fixed-point → integer part
}

/// Read `transcript.json` (a `Vec<Segment>`), empty when absent or unreadable. A
/// present-but-corrupt file is logged (the meeting still projects, transcript-less)
/// rather than silently dropped; an absent file is the legitimate no-transcript
/// case and stays quiet.
fn read_transcript(folder: &Path) -> Vec<Segment> {
    match std::fs::read(folder.join("transcript.json")) {
        Ok(bytes) => serde_json::from_slice(&bytes).unwrap_or_else(|e| {
            tracing::warn!(
                target: "sync-ffi",
                folder = %folder.display(),
                error = %e,
                "corrupt transcript.json; projecting the meeting transcript-less"
            );
            Vec::new()
        }),
        Err(_) => Vec::new(),
    }
}

/// Project diarised segments into the phone's `(segments, speakers)` shape:
/// `speaker_id` labels (e.g. "A", "B") become 1-based indices in first-appearance
/// order, and `speakers[i-1]` is the user-set name (`speaker_names`) or a
/// `"Speaker {i}"` fallback. A segment with no `speaker_id` gets index `0`.
///
/// The 1-based renumber is **deliberate**: the mobile model indexes speakers from
/// 1 to align with the `--speaker-N` CSS palette tokens (see the `speakers` /
/// `speakerIndex` docs in minutist-mobile `types.ts`), so the phone shows
/// "Speaker 1" where the desktop shows the bare diarizer letter ("Speaker B").
/// NAMED speakers match cross-device via the `speaker_names` overlay; only the
/// UNNAMED fallback label differs cosmetically. Exact unnamed-label parity is a
/// UI follow-up (it would also need the palette-keying schemes aligned).
fn project_speakers(
    transcript: &[Segment],
    speaker_names: &std::collections::BTreeMap<String, String>,
) -> (Vec<FfiSegment>, Vec<String>) {
    let mut labels: Vec<String> = Vec::new();
    let mut segments = Vec::with_capacity(transcript.len());
    for seg in transcript {
        let speaker_index = match &seg.speaker_id {
            Some(label) => {
                let pos = labels.iter().position(|l| l == label).unwrap_or_else(|| {
                    labels.push(label.clone());
                    labels.len() - 1
                });
                (pos + 1) as u32
            }
            None => 0,
        };
        segments.push(FfiSegment {
            speaker_index,
            start_ms: seg.start_ms as i64,
            end_ms: seg.end_ms as i64,
            text: seg.text.clone(),
        });
    }
    let speakers = labels
        .iter()
        .enumerate()
        .map(|(i, label)| {
            speaker_names
                .get(label)
                .cloned()
                .unwrap_or_else(|| format!("Speaker {}", i + 1))
        })
        .collect();
    (segments, speakers)
}

/// Project a meeting folder into the phone's [`FfiMeeting`] model, or `None` when
/// the folder has no `metadata.json` or it cannot be read (so one bad folder does
/// not break a listing). Persistence-free: plain reads of `metadata.json` /
/// `transcript.json` / `summary.md` + the notes ydoc.
fn project_meeting(meetings_root: &Path, id: MeetingId) -> Option<FfiMeeting> {
    let folder = meetings_root.join(id.0.to_string());
    if !folder.join("metadata.json").exists() {
        return None;
    }
    let meta = match notes_crdt::read_metadata(&folder) {
        Ok(m) => m,
        Err(e) => {
            tracing::warn!(target: "sync-ffi", meeting_id = %id.0, error = %e, "skipping unreadable meeting");
            return None;
        }
    };
    let started_at_ms = rfc3339_to_ms(&meta.started_at);
    let id_str = id.0.to_string();

    match meta.processing {
        // Awaiting results → the captured-unprocessed view.
        ProcessingLifecycle::PendingProcessing | ProcessingLifecycle::Claimed { .. } => {
            let (processing, claimed_by) = match &meta.processing {
                ProcessingLifecycle::Claimed { claim } => {
                    (FfiCaptureState::Claimed, Some(claim.host.0.clone()))
                }
                _ => (FfiCaptureState::Pending, None),
            };
            Some(FfiMeeting::Captured {
                id: id_str,
                title: meta.title,
                started_at_ms,
                duration_ms: meta.duration_ms as i64,
                has_audio: minutist_common::resolve_audio_path(&folder).is_some(),
                has_notes: folder_has_notes_content(&folder),
                processing,
                claimed_by,
            })
        }
        // Processed (synced back) or the desktop-only Local: outputs exist.
        _ => {
            let transcript = read_transcript(&folder);
            let (segments, speakers) = project_speakers(&transcript, &meta.speaker_names);
            let summary = std::fs::read_to_string(folder.join("summary.md")).ok();
            let notes = NotesStore::read_ydoc_state(meetings_root, id).ok().flatten();
            Some(FfiMeeting::Synced {
                id: id_str,
                title: meta.title,
                started_at_ms,
                transcript: segments,
                summary,
                speakers,
                notes,
            })
        }
    }
}

/// Apply a peer-advertised lifecycle to our local `metadata.json`, skipping a
/// meeting we do not hold yet. Calls the shared
/// [`notes_crdt::apply_synced_lifecycle_if_present`] directly — the phone has no
/// `persistence` edge, so this is the same precedence-merge-and-skip-if-absent
/// implementation `persistence::meeting_ops::apply_synced_lifecycle_if_present`
/// re-exports for the desktop/hub, not a hand-rolled copy. Best-effort: a write
/// failure is logged, not surfaced (the next discovery sweep re-applies).
fn apply_inbound_lifecycle(meetings_root: &Path, id: MeetingId, incoming: ProcessingLifecycle) {
    let result = notes_crdt::apply_synced_lifecycle_if_present(meetings_root, id, incoming);
    if let Err(e) = result {
        tracing::warn!(target: "sync-ffi", meeting_id = %id.0, error = %e, "applying synced lifecycle failed");
    }
}

/// Parse a hyphenated UUID string off the boundary into a [`MeetingId`].
fn parse_meeting(s: &str) -> Result<MeetingId, SyncFfiError> {
    Uuid::parse_str(s)
        .map(MeetingId)
        .map_err(|e| SyncFfiError::InvalidArg {
            msg: format!("meeting id {s:?}: {e}"),
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use minutist_common::{HostRef, ProcessingClaim};
    use std::sync::atomic::{AtomicBool, Ordering};
    use tempfile::TempDir;

    #[test]
    fn sniff_captured_audio_detects_container_by_magic() {
        let dir = TempDir::new().unwrap();
        // Ogg/Opus → opus, fixed 16 kHz mono (our encoder contract).
        let ogg = dir.path().join("a.opus");
        std::fs::write(&ogg, b"OggS\x00\x02rest-of-header").unwrap();
        assert_eq!(
            sniff_captured_audio(&ogg).unwrap(),
            ("opus", "opus".to_string(), 16_000, 1)
        );
        // MP4 (`ftyp` at bytes 4..8) → m4a/aac; params fall back to 44100/1 when
        // no mp4a box is present (a header-only stub).
        let mp4 = dir.path().join("a.m4a");
        std::fs::write(&mp4, b"\x00\x00\x00\x18ftypM4A \x00\x00\x00\x00isomiso2").unwrap();
        let (ext, codec, rate, ch) = sniff_captured_audio(&mp4).unwrap();
        assert_eq!((ext, codec.as_str()), ("m4a", "aac"));
        assert_eq!((rate, ch), (44_100, 1), "fallback when the mp4a box is absent");
        // An unknown container is refused rather than mislabelled.
        let junk = dir.path().join("a.bin");
        std::fs::write(&junk, b"RIFFxxxxWAVEfmt ").unwrap();
        assert!(matches!(
            sniff_captured_audio(&junk),
            Err(SyncFfiError::InvalidArg { .. })
        ));
    }

    #[test]
    fn lifecycle_maps_every_variant_carrying_fields() {
        assert!(matches!(
            FfiLifecycle::from(ProcessingLifecycle::Local),
            FfiLifecycle::Local
        ));
        assert!(matches!(
            FfiLifecycle::from(ProcessingLifecycle::PendingProcessing),
            FfiLifecycle::PendingProcessing
        ));

        let claimed = ProcessingLifecycle::Claimed {
            claim: ProcessingClaim {
                host: HostRef("host-a".into()),
                claimed_at: "2026-06-30T00:00:00Z".into(),
                lease_expires_at: "2026-06-30T01:00:00Z".into(),
            },
        };
        match FfiLifecycle::from(claimed) {
            FfiLifecycle::Claimed {
                host,
                claimed_at,
                lease_expires_at,
            } => {
                assert_eq!(host, "host-a");
                assert_eq!(claimed_at, "2026-06-30T00:00:00Z");
                assert_eq!(lease_expires_at, "2026-06-30T01:00:00Z");
            }
            other => panic!("expected Claimed, got {other:?}"),
        }

        let processed = ProcessingLifecycle::Processed {
            processed_by: HostRef("host-b".into()),
            at: "2026-06-30T02:00:00Z".into(),
        };
        match FfiLifecycle::from(processed) {
            FfiLifecycle::Processed { processed_by, at } => {
                assert_eq!(processed_by, "host-b");
                assert_eq!(at, "2026-06-30T02:00:00Z");
            }
            other => panic!("expected Processed, got {other:?}"),
        }
    }

    #[test]
    fn parse_meeting_round_trips_a_uuid_and_rejects_garbage() {
        let id = "550e8400-e29b-41d4-a716-446655440000";
        assert_eq!(parse_meeting(id).expect("valid uuid").0.to_string(), id);
        assert!(parse_meeting("not-a-uuid").is_err());
    }

    /// Regression guard for C1: the drain must run the callback OFF the runtime,
    /// so a re-entrant `block_on` (what a real `on_lagged` handler does when it
    /// calls back into the engine) completes instead of panicking "cannot start a
    /// runtime from within a runtime". If `spawn_drain` ever reverts to a runtime
    /// task, the callback runs on a worker, the inner `block_on` panics, the flag
    /// is never set, and this test times out and fails.
    #[test]
    fn drain_runs_off_runtime_so_a_reentrant_block_on_survives() {
        let rt = Arc::new(
            tokio::runtime::Builder::new_multi_thread()
                .worker_threads(1)
                .enable_all()
                .build()
                .expect("runtime"),
        );
        let (tx, rx) = broadcast::channel::<u8>(8);
        let done = Arc::new(AtomicBool::new(false));

        let rt_in = Arc::clone(&rt);
        let done_in = Arc::clone(&done);
        spawn_drain(rx, move |item| {
            if let Drained::Event(_) = item {
                // Re-enter the runtime exactly as a callback calling back into the
                // engine would; only legal off a runtime thread.
                rt_in.block_on(async {});
                done_in.store(true, Ordering::SeqCst);
            }
        });

        tx.send(1).expect("send");
        for _ in 0..200 {
            if done.load(Ordering::SeqCst) {
                return;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        panic!("drain callback did not complete a re-entrant block_on off-runtime");
    }

    /// A diarised segment for the projection tests.
    fn seg(label: Option<&str>, start: u64, text: &str) -> Segment {
        Segment {
            start_ms: start,
            end_ms: start + 500,
            text: text.into(),
            speaker_id: label.map(String::from),
            confidence: None,
            words: vec![],
            shared_speakers: vec![],
        }
    }

    /// Write a `Processed` meeting (metadata + optional transcript/summary) for the
    /// Synced-projection tests — no engine, no ML.
    fn write_processed_meeting(
        root: &std::path::Path,
        segments: &[Segment],
        speaker_names: &[(&str, &str)],
        summary: Option<&str>,
    ) -> MeetingId {
        let id = MeetingId::new();
        let folder = MeetingFolder::ensure(root, id).expect("ensure");
        let names = speaker_names
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        let meta = MeetingMeta {
            uuid: id,
            title: "Synced one".into(),
            started_at: "2026-06-30T09:00:00+00:00".into(),
            ended_at: None,
            duration_ms: 1000,
            speaker_count: 2,
            audio_format: AudioFormat {
                codec: "opus".into(),
                sample_rate: 16_000,
                channels: 1,
                bitrate_kbps: None,
            },
            asr_model: None,
            llm_model: None,
            diarizer: None,
            speaker_names: names,
            notes_format: 1,
            processing: ProcessingLifecycle::Processed {
                processed_by: HostRef("host-x".into()),
                at: "2026-06-30T09:30:00+00:00".into(),
            },
            collection_id: None,
            recording_started: true,
            app_version: "0.0.0".into(),
        };
        notes_crdt::write_metadata(&folder, &meta).expect("write meta");
        if !segments.is_empty() {
            std::fs::write(
                folder.join("transcript.json"),
                serde_json::to_vec(segments).expect("ser segs"),
            )
            .expect("write transcript");
        }
        if let Some(s) = summary {
            std::fs::write(folder.join("summary.md"), s).expect("write summary");
        }
        id
    }

    /// `save_captured` writes a PendingProcessing meeting that projects as
    /// `Captured{Pending}`, with `has_notes`/`has_audio` reflecting the folder.
    #[test]
    fn save_captured_projects_as_pending() {
        let dir = TempDir::new().unwrap();
        let root = dir.path();
        let id_str =
            save_captured_to(root, "My meeting", 1_700_000_000_000, 5_000, None, "hello notes")
                .expect("save");
        let id = MeetingId(Uuid::parse_str(&id_str).unwrap());
        match project_meeting(root, id).expect("present") {
            FfiMeeting::Captured {
                title,
                processing,
                has_notes,
                has_audio,
                duration_ms,
                claimed_by,
                ..
            } => {
                assert_eq!(title, "My meeting");
                assert!(matches!(processing, FfiCaptureState::Pending));
                assert!(has_notes, "notes.ydoc seeded from the note text");
                assert!(!has_audio, "no audio path was passed");
                assert_eq!(duration_ms, 5_000);
                assert!(claimed_by.is_none());
            }
            other => panic!("expected Captured, got {other:?}"),
        }
    }

    /// A `Processed` meeting with a transcript + summary projects as `Synced`, with
    /// `speaker_id` labels mapped to a 1-based palette (named where set, else
    /// `"Speaker N"`) and a no-speaker segment indexed `0`.
    #[test]
    fn processed_meeting_projects_as_synced_with_speaker_palette() {
        let dir = TempDir::new().unwrap();
        let root = dir.path();
        let segs = vec![
            seg(Some("A"), 0, "hi"),
            seg(Some("B"), 500, "hello"),
            seg(Some("A"), 1000, "bye"),
            seg(None, 1500, "[noise]"),
        ];
        let id = write_processed_meeting(root, &segs, &[("A", "Alice")], Some("# Summary"));
        match project_meeting(root, id).expect("present") {
            FfiMeeting::Synced {
                transcript,
                summary,
                speakers,
                ..
            } => {
                // A → 1 (named "Alice"), B → 2 (fallback "Speaker 2").
                assert_eq!(speakers, vec!["Alice".to_string(), "Speaker 2".to_string()]);
                assert_eq!(
                    transcript.iter().map(|s| s.speaker_index).collect::<Vec<_>>(),
                    vec![1, 2, 1, 0],
                );
                assert_eq!(transcript[0].text, "hi");
                assert_eq!(summary.as_deref(), Some("# Summary"));
            }
            other => panic!("expected Synced, got {other:?}"),
        }
    }

    /// `list_meeting_ids` skips `.blobs`, and projecting + sorting newest-first
    /// orders by start time.
    #[test]
    fn list_skips_blobs_and_sorts_newest_first() {
        let dir = TempDir::new().unwrap();
        let root = dir.path();
        std::fs::create_dir_all(root.join(".blobs")).unwrap();
        save_captured_to(root, "older", 1_000, 10, None, "").unwrap();
        save_captured_to(root, "newer", 2_000, 10, None, "").unwrap();

        let ids = notes_crdt::folder::list_meeting_ids(root);
        assert_eq!(ids.len(), 2, ".blobs must be skipped");

        let mut metas: Vec<FfiMeeting> = ids
            .into_iter()
            .filter_map(|id| project_meeting(root, id))
            .collect();
        metas.sort_by_key(|m| std::cmp::Reverse(started_at_ms_of(m)));
        assert_eq!(started_at_ms_of(&metas[0]), 2_000);
        assert_eq!(started_at_ms_of(&metas[1]), 1_000);
    }

    /// Inbound lifecycle merges into the local metadata by precedence: a host's
    /// `Claimed` is applied (projects as `Captured{Claimed, claimed_by}`), and a
    /// subsequent stale `PendingProcessing` does NOT regress it.
    #[test]
    fn inbound_lifecycle_merges_by_precedence() {
        let dir = TempDir::new().unwrap();
        let root = dir.path();
        let id = MeetingId(
            Uuid::parse_str(&save_captured_to(root, "m", 1_000, 10, None, "").unwrap()).unwrap(),
        );

        apply_inbound_lifecycle(
            root,
            id,
            ProcessingLifecycle::Claimed {
                claim: ProcessingClaim {
                    host: HostRef("host-gpu".into()),
                    claimed_at: "2026-06-30T00:00:00Z".into(),
                    lease_expires_at: "2026-06-30T01:00:00Z".into(),
                },
            },
        );
        match project_meeting(root, id).expect("present") {
            FfiMeeting::Captured {
                processing,
                claimed_by,
                ..
            } => {
                assert!(matches!(processing, FfiCaptureState::Claimed));
                assert_eq!(claimed_by.as_deref(), Some("host-gpu"));
            }
            other => panic!("expected Captured/Claimed, got {other:?}"),
        }

        // A stale PendingProcessing must not walk Claimed backwards.
        apply_inbound_lifecycle(root, id, ProcessingLifecycle::PendingProcessing);
        assert!(matches!(
            project_meeting(root, id).unwrap(),
            FfiMeeting::Captured {
                processing: FfiCaptureState::Claimed,
                ..
            }
        ));
    }

    /// `save_captured` is all-or-nothing: a failed audio copy rolls back the
    /// freshly-created folder, leaving no orphan meeting (I2).
    #[test]
    fn save_captured_rolls_back_on_failed_audio_copy() {
        let dir = TempDir::new().unwrap();
        let root = dir.path();
        let result = save_captured_to(
            root,
            "m",
            1_000,
            10,
            Some("/nonexistent/path/audio.opus"),
            "notes",
        );
        assert!(result.is_err(), "a missing audio source must fail the save");
        assert!(
            notes_crdt::folder::list_meeting_ids(root).is_empty(),
            "a failed save must leave no orphan meeting folder"
        );
    }
}
