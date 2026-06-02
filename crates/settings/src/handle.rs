//! `SettingsHandle` — cheaply clonable, `Send + Sync` accessor for settings.
//!
//! Wraps an `Arc<Inner>` so cloning the handle is cheap. All callers share
//! the same underlying state; mutations are serialised by a `tokio::sync::RwLock`.
//!
//! Change broadcasts use `tokio::sync::watch`. Every subscriber receives the
//! full `Settings` snapshot — no diff, no partial updates.

use std::sync::Arc;

use tokio::sync::{watch, RwLock};

use meeting_app_common::AppResult;

use crate::{error::Error, store::SettingsStore, Settings};

struct Inner {
    /// The current settings value, protected by an RwLock.
    ///
    /// Reads are cheap (shared lock); writes are infrequent (user action).
    state: RwLock<Settings>,
    /// Backing store for persistence.
    store: Box<dyn SettingsStore>,
    /// Watch sender; keeping this here means we can call `send` from behind
    /// the Arc without an extra Arc layer on the sender.
    tx: watch::Sender<Settings>,
}

/// A cheaply clonable, thread-safe handle to the application settings.
///
/// Construction loads from the backing store (falling back to defaults on
/// missing/corrupt files).  Updates persist to the store and broadcast to all
/// watch subscribers.
#[derive(Clone)]
pub struct SettingsHandle {
    inner: Arc<Inner>,
}

impl SettingsHandle {
    /// Construct a `SettingsHandle` backed by `store`.
    ///
    /// Loads the initial settings from the store.  A missing file yields
    /// `Settings::default()` silently; a corrupt file yields
    /// `Settings::default()` with a `tracing::warn!`.
    pub fn new<S: SettingsStore + 'static>(store: S) -> AppResult<Self> {
        let initial = match store.load() {
            Ok(s) => s,
            Err(Error::Serialise(ref e)) => {
                tracing::warn!(
                    target = "settings",
                    error = %e,
                    "settings file is corrupt; falling back to defaults"
                );
                Settings::default()
            }
            Err(e) => return Err(e.into()),
        };

        let (tx, _rx) = watch::channel(initial.clone());

        Ok(Self {
            inner: Arc::new(Inner {
                state: RwLock::new(initial),
                store: Box::new(store),
                tx,
            }),
        })
    }

    /// Return a snapshot of the current settings.
    ///
    /// This is a cheap, synchronous clone of the published in-memory value; no
    /// I/O. It reads the `watch` value, which [`Self::update`] keeps current via
    /// `send_replace` regardless of whether any subscriber is alive — see the
    /// note there. (Reading the `RwLock` here instead is not an option: it is a
    /// `tokio` async lock and `current` is synchronous.)
    pub fn current(&self) -> Settings {
        self.inner.tx.borrow().clone()
    }

    /// Apply `f` to a mutable copy of the current settings, then persist and
    /// broadcast the result.
    ///
    /// The write lock is held for the whole read-modify-write so concurrent
    /// `update` calls are serialised and last-writer-wins is well defined.
    /// `store.save` is a synchronous, blocking local-file write; it is run on
    /// the blocking thread pool via [`tokio::task::spawn_blocking`] so it does
    /// **not** block the async worker thread the caller is running on (the
    /// "no blocking I/O on an async worker" rule in
    /// `architecture/cross-cutting.md` — Threading model / Persistence writes).
    /// The guard is held across that `.await`, which is correct: the worker
    /// thread is yielded back to the scheduler while the blocking task runs,
    /// and serialisation against other `update` calls is preserved.
    pub async fn update<F>(&self, f: F) -> AppResult<()>
    where
        F: FnOnce(&mut Settings) + Send,
    {
        let mut guard = self.inner.state.write().await;
        f(&mut guard);
        // Snapshot the new value to hand to the blocking save and (on success)
        // to publish. Cloning under the lock keeps it consistent with the
        // committed in-memory state.
        let new_value = guard.clone();

        // Persist first; if the save fails, don't publish the change. Run the
        // blocking fs write on the blocking pool rather than directly on this
        // async worker. We clone the `Arc<Inner>` (cheap) so the spawned task
        // owns a `'static` handle to the store; the write guard is still held
        // across the `.await`, so the read-modify-write stays atomic against
        // concurrent `update` calls.
        let inner = Arc::clone(&self.inner);
        let value_for_save = new_value.clone();
        tokio::task::spawn_blocking(move || inner.store.save(&value_for_save))
            .await
            .map_err(|e| meeting_app_common::AppError::Internal {
                context: format!("settings save task failed to join: {e}"),
            })?
            .map_err(meeting_app_common::AppError::from)?;

        // Publish via `send_replace`, NOT `send`. `send` is a no-op (returns
        // `Err`, discarded) when there are no live subscribers, which would
        // leave the value `current()` reads stale until app restart — and
        // nothing in production holds a subscriber (the orchestrator reads
        // `current().diarization_enabled` / `.input_device_id` directly).
        // `send_replace` always stores the new value (and still notifies any
        // subscribers), so `current()` is authoritative without one.
        self.inner.tx.send_replace(new_value);
        Ok(())
    }

    /// Subscribe to settings changes.
    ///
    /// The returned receiver immediately holds the current settings value.
    /// Each call to `update` on any handle clone sends the updated value.
    pub fn subscribe(&self) -> watch::Receiver<Settings> {
        self.inner.tx.subscribe()
    }
}
