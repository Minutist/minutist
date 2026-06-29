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
//!   / [`FfiSyncEngine::pair`]) and thereafter hex peer-id strings, passed
//!   straight to `SyncEngine`'s string-keyed `*_to_peer` / `*_with_peer` methods
//!   (which relay-address the peer internally — see `SyncEngine::push_all_to`);
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

use minutist_common::{MeetingId, ProcessingLifecycle};
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
    ) -> Result<Arc<Self>, SyncFfiError> {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .map_err(|e| SyncFfiError::Io {
                msg: format!("building tokio runtime: {e}"),
            })?;

        let identity = DeviceIdentity::load_or_generate(Path::new(&app_data_dir))?;
        let config = SyncConfig {
            relay_url,
            relay_auth_token,
            meetings_root: PathBuf::from(meetings_root),
        };
        let engine = rt.block_on(SyncEngine::start(config, identity))?;

        Ok(Arc::new(Self {
            inner: Mutex::new(Some(Arc::new(engine))),
            rt: Arc::new(rt),
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

    /// Hex endpoint ids of every currently-registered peer.
    pub fn peer_ids(&self) -> Result<Vec<String>, SyncFfiError> {
        Ok(self
            .engine()?
            .peer_ids()
            .iter()
            .map(|id| id.to_string())
            .collect())
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
        spawn_drain(rx, move |item| match item {
            Drained::Event((meeting, lifecycle)) => {
                listener.on_lifecycle(meeting.0.to_string(), lifecycle.into())
            }
            Drained::Lagged => listener.on_lagged(),
        });
        Ok(())
    }

    /// Drain "peer arrived" events into `listener` until shutdown. One listener
    /// per call (see [`Self::subscribe_lifecycle`]).
    pub fn subscribe_peers(&self, listener: Box<dyn PeerListener>) -> Result<(), SyncFfiError> {
        let rx = self.engine()?.subscribe_peer_events();
        spawn_drain(rx, move |item| match item {
            Drained::Event(peer) => listener.on_peer_arrived(peer.to_string()),
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
}
