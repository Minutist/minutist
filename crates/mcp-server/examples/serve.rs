//! Standalone loopback `mcp-server` for WS4-A S4 live-smoke testing.
//!
//! Boots the real `mcp-server` (the same `serve(..)` `app-main` calls) on a
//! loopback port with the real `agent-tools` `v1(true)` registry, gated by a
//! bearer token read from the environment. This is the device-side endpoint the
//! `tunnel-client` replays relayed `/mcp` requests against in the S4 chain.
//!
//! It is a test/smoke harness, not shipped: the orchestrator behind the registry
//! is the `test-source` stub (no real models), which is sufficient for proving
//! `tools/list` returns the real tool set and the MCP `initialize` handshake
//! works end to end. Tool *calls* that need real meeting data will return empty
//! results against the stub backend — out of scope for S4 (the exit is the tool
//! surface, not tool execution).
//!
//! Environment:
//! - `MCP_BIND`      — loopback bind address (default `127.0.0.1:8765`).
//! - `MCP_BEARER`    — the internal bearer every request must present (required).
//! - `MCP_ALLOW_WRITES` — `1`/`true` to expose the reversible write tools
//!   (default off; reads + the inter-agent tool only).
//!
//! On bind it prints the bound address and the tool count so the harness driver
//! can confirm readiness, then serves until SIGINT.

use std::sync::Arc;

use agent_tools::{ToolContext, ToolRegistry};
use mcp_server::{serve, McpServerConfig};
use minutist_common::{AppEvent, AppResult, NoteBlock, Segment, Summariser};
use orchestrator::test_support::test_orchestrator;
use persistence::MeetingIndex;
use tokio::sync::{broadcast, watch};

/// A no-op summariser — `ToolContext::new` requires one; the S4 smoke never
/// dispatches a summarising tool.
struct StubSummariser;

impl Summariser for StubSummariser {
    fn summarise(
        &self,
        _transcript: &[Segment],
        _notes: &[NoteBlock],
        _system_prompt: &str,
    ) -> AppResult<String> {
        Ok(String::new())
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let bind = std::env::var("MCP_BIND").unwrap_or_else(|_| "127.0.0.1:8765".to_string());
    let bearer = std::env::var("MCP_BEARER")
        .map_err(|_| "MCP_BEARER must be set (the internal loopback bearer)")?;
    let allow_writes = matches!(
        std::env::var("MCP_ALLOW_WRITES").as_deref(),
        Ok("1") | Ok("true")
    );

    // A persistent (not :memory:) meetings dir under a temp path so the registry
    // backends are real; no model is loaded (test-source orchestrator).
    let scratch = std::env::temp_dir().join("minutist-s4-mcp");
    let meetings_dir = scratch.join("meetings");
    std::fs::create_dir_all(&meetings_dir)?;

    let index = Arc::new(MeetingIndex::open(":memory:").await?);
    let orchestrator = Arc::new(test_orchestrator(meetings_dir.clone()));
    let summariser: Arc<dyn Summariser> = Arc::new(StubSummariser);
    let (event_tx, _rx) = broadcast::channel::<AppEvent>(16);

    // The v1(true) registry needs an inter-agent bridge sender for
    // send_to_internal_agent; no driver is attached, so calling that tool would
    // error — fine for the S4 tools/list smoke.
    let (bridge_tx, _bridge_rx) = tokio::sync::mpsc::channel(1);
    let ctx = Arc::new(
        ToolContext::new(
            orchestrator,
            index,
            meetings_dir,
            summariser,
            event_tx,
            None,
        )
        .with_inter_agent_bridge(bridge_tx),
    );
    let registry = Arc::new(ToolRegistry::v1(true));
    let tool_count = registry.mcp_tool_descriptors_gated(allow_writes).len();

    let (_shutdown_tx, shutdown_rx) = watch::channel(false);
    let (bound, _done_rx) = serve(
        registry,
        ctx,
        McpServerConfig {
            bind_addr: bind.parse()?,
            bearer_token: bearer,
            allow_writes,
        },
        shutdown_rx,
    )
    .await?;

    // The driver greps this line to learn the server is up and how many tools it
    // exposes (so the smoke can assert the count downstream).
    println!("MCP_SERVE_READY addr={bound} tools={tool_count} allow_writes={allow_writes}");

    // Serve until Ctrl-C; keep _shutdown_tx alive so the accept loop runs.
    tokio::signal::ctrl_c().await?;
    Ok(())
}
