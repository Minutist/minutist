//! WS4-A S4 device-side driver: dial the relay tunnel and bridge to a loopback
//! `mcp-server`.
//!
//! `tunnel-client` is a library (`run_tunnel`); this example is the binary the S4
//! live smoke runs as the app's stand-in. It reads the relay URL, the device
//! credential, the account id, and the loopback target (base URL + internal
//! bearer) from the environment and calls [`tunnel_client::run_tunnel`], looping
//! with a fixed backoff so a transient relay drop reconnects (the real reconnect
//! policy is S5; this is a smoke-loop, not that).
//!
//! Environment:
//! - `RELAY_URL`         — the relay rendezvous URL, e.g.
//!   `wss://mcp.minutist.ai/tunnel` or `ws://127.0.0.1:8482/tunnel` for a
//!   loopback relay (cleartext `ws://` is only accepted for a loopback host).
//! - `DEVICE_CREDENTIAL` — the static device credential the relay maps to the
//!   account (must equal the relay's `TUNNEL_DEVICE_CREDENTIAL`).
//! - `ACCOUNT_ID`        — the account this device serves (the relay's
//!   `TUNNEL_DEVICE_ACCOUNT`, i.e. the rauthy `sub`).
//! - `LOOPBACK_URL`      — the loopback `mcp-server` origin, e.g.
//!   `http://127.0.0.1:8765` (NO path; the relay frame carries `/mcp`).
//! - `INTERNAL_BEARER`   — the loopback server's bearer (the standalone
//!   `mcp-server` harness's `MCP_BEARER`).

use std::time::Duration;

use tunnel_client::{run_tunnel, InternalBearer, LoopbackTarget, TunnelConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info,tunnel_client=debug")),
        )
        .init();

    let relay_url = env("RELAY_URL")?;
    let device_credential = env("DEVICE_CREDENTIAL")?;
    let account_id = env("ACCOUNT_ID")?;
    let loopback_url = env("LOOPBACK_URL")?;
    let internal_bearer = env("INTERNAL_BEARER")?;

    let config = TunnelConfig {
        relay_url,
        device_credential,
        account_id,
        loopback: LoopbackTarget::new(loopback_url, InternalBearer::new(internal_bearer)),
    };

    // Smoke reconnect loop: run one session; on return (clean close or error)
    // wait a fixed beat and redial. Ctrl-C aborts. The production lifecycle +
    // backoff is S5; here a fixed delay is enough to survive a relay restart
    // during the smoke without spinning.
    loop {
        match run_tunnel(config.clone()).await {
            Ok(()) => tracing::info!("tunnel session closed cleanly; reconnecting"),
            Err(error) => tracing::warn!(%error, "tunnel session ended with error; reconnecting"),
        }
        tokio::time::sleep(Duration::from_secs(2)).await;
    }
}

fn env(key: &str) -> Result<String, String> {
    std::env::var(key).map_err(|_| format!("{key} must be set"))
}
