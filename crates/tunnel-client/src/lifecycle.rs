//! Tunnel lifecycle handle (WS4-A S5b, tc-lifecycle).
//!
//! Wraps the supervised [`reconnect_loop`] in a start/stop handle so `app-main`
//! can run the tunnel as a background task tied to the connector toggle:
//!
//! - [`TunnelHandle::start`] spawns the reconnect loop and returns immediately.
//! - [`TunnelHandle::stop`] raises the loop's cancel signal and awaits the task,
//!   so the connector is fully torn down (in-flight requests aborted by
//!   [`run_tunnel`](crate::run_tunnel)'s `JoinSet` teardown) before the call
//!   returns — the same completion-handle discipline `app-main` uses for the
//!   MCP server.
//!
//! The handle owns a [`tokio::sync::watch`] cancel sender and the spawned task's
//! [`JoinHandle`]; dropping the handle without `stop` aborts the loop (the watch
//! sender drops, which the loop treats as a cancel).

use tokio::sync::watch;
use tokio::task::JoinHandle;

use crate::reconnect::{reconnect_loop, ConnectionState, ReconnectExit};
use crate::run::TunnelConfig;

/// A running (or stopped) tunnel background task.
///
/// Construct with [`TunnelHandle::start`]; tear down with [`TunnelHandle::stop`].
pub struct TunnelHandle {
    cancel_tx: watch::Sender<bool>,
    task: JoinHandle<ReconnectExit>,
}

impl TunnelHandle {
    /// Spawn the supervised reconnect loop for `config` and return a handle to
    /// it. `on_state` is invoked on every connection-state transition (cheap,
    /// non-async) so the caller can surface the live status; it must be `Send`
    /// because it runs on the spawned task.
    pub fn start<F>(config: TunnelConfig, on_state: F) -> Self
    where
        F: Fn(ConnectionState) + Send + 'static,
    {
        let (cancel_tx, cancel_rx) = watch::channel(false);
        let task = tokio::spawn(async move { reconnect_loop(config, cancel_rx, on_state).await });
        Self { cancel_tx, task }
    }

    /// Signal the loop to stop and await its exit. Returns how the loop exited
    /// (normally [`ReconnectExit::Cancelled`]; a terminal auth outcome is
    /// returned if the loop had already stopped on its own — e.g. the credential
    /// was revoked, which the caller surfaces as "needs re-pair").
    ///
    /// Idempotent: calling `stop` after the loop already exited returns the
    /// loop's recorded exit.
    pub async fn stop(self) -> ReconnectExit {
        // Raise the cancel signal; ignore a send error (the task may have
        // already finished, dropping its receiver).
        let _ = self.cancel_tx.send(true);
        match self.task.await {
            Ok(exit) => exit,
            Err(join_err) => {
                // The task panicked or was aborted; treat as cancelled for the
                // lifecycle (the caller is tearing down regardless).
                tracing::warn!(target: "tunnel-client", %join_err, "tunnel: lifecycle task did not exit cleanly");
                ReconnectExit::Cancelled
            }
        }
    }

    /// Whether the background task has finished (the loop exited on its own — a
    /// terminal auth outcome — rather than waiting for [`stop`](Self::stop)).
    pub fn is_finished(&self) -> bool {
        self.task.is_finished()
    }

    /// Raise the cancel signal immediately, WITHOUT awaiting the task. For a
    /// synchronous restart path that must guarantee the old loop has observed the
    /// cancel before a replacement loop is spawned and dials — closing the brief
    /// window where two loops for one account could momentarily coexist. The task
    /// is still awaited separately (via [`stop`](Self::stop)) to finish teardown;
    /// `send(true)` is idempotent, so the later `stop` is harmless.
    pub fn signal_cancel(&self) {
        let _ = self.cancel_tx.send(true);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loopback::{InternalBearer, LoopbackTarget};
    use std::sync::{Arc, Mutex};

    fn test_config(relay_url: &str) -> TunnelConfig {
        TunnelConfig {
            relay_url: relay_url.to_string(),
            device_credential: "mdc_dev.secret".to_string(),
            account_id: "acct".to_string(),
            loopback: LoopbackTarget::new("http://127.0.0.1:8765", InternalBearer::new("tok")),
        }
    }

    #[tokio::test]
    async fn start_then_stop_is_clean() {
        // The relay URL refuses to dial (cleartext remote → TunnelError::Config,
        // a transient error), so the loop sits in backoff. stop() must cancel it.
        let states: Arc<Mutex<Vec<ConnectionState>>> = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&states);
        let handle = TunnelHandle::start(test_config("ws://relay.example/tunnel"), move |s| {
            sink.lock().unwrap().push(s);
        });
        // Give the loop a moment to make its first attempt and enter backoff.
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        let exit = handle.stop().await;
        assert_eq!(exit, ReconnectExit::Cancelled);
        // It should have reported Connecting at least once.
        assert!(states
            .lock()
            .unwrap()
            .contains(&ConnectionState::Connecting));
    }
}
