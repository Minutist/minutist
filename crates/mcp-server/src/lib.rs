//! `mcp-server` — the in-process Streamable HTTP MCP server (Phase 10).
//!
//! A SECOND consumer of the Phase-9 `agent-tools` registry: it adds NO tools,
//! projecting [`agent_tools::ToolRegistry::mcp_tool_descriptors_gated`] onto MCP
//! `tools/list` and [`agent_tools::ToolRegistry::dispatch`] onto `tools/call`.
//! Any tool logic, schema, or name living here rather than in `agent-tools` is a
//! reviewer finding.
//!
//! # Boundaries (binding — see `architecture/components.md`)
//!
//! Edges: `common`, `agent-tools` (+ the rmcp SDK and its hyper/http/tower leaf
//! externals). Deliberately **no** `chat-agent` edge — the inter-agent bridge
//! (`send_to_internal_agent`) reaches the chat engine through a `common`-typed
//! channel held on the [`agent_tools::ToolContext`], whose receiver + chat turn
//! live in `ipc-bridge`. Deliberately **no** `tauri`/`specta` concern — the
//! listener is spawned by `app-main` via `tauri::async_runtime::spawn`; this
//! crate takes the registry + context + a shutdown receiver + bind/token config
//! and serves until the shutdown fires.
//!
//! # Security (Phase 10 §4)
//!
//! - **Loopback bind only** — `serve` binds the caller-provided `SocketAddr`,
//!   which `app-main` sets to `127.0.0.1:{mcp_port}`. Never `0.0.0.0`.
//! - **Bearer token** — every request must carry `Authorization: Bearer <token>`
//!   (a thin wrapper service returning 401 on mismatch, BEFORE rmcp sees it; the
//!   session id is never the credential — spec MUST). See [`auth`].
//! - **Host + Origin validation** — rmcp's `StreamableHttpServerConfig` enforces
//!   the `Host` allowlist (loopback default, `rmcp >= 1.4.0` — GHSA-89vp-x53w-74fx)
//!   and we set `allowed_origins` to the loopback origins so a cross-origin
//!   browser request is a 403.
//! - **Write-tool exposure gate** — `allow_writes` (`settings.mcp_write_tools`,
//!   D3) is applied at projection time AND on call.

use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::Arc;

use agent_tools::{ToolContext, ToolRegistry};
use bytes::Bytes;
use http::{Request, Response, StatusCode};
use http_body_util::{combinators::BoxBody, BodyExt, Full};
use hyper::body::Incoming;
use hyper_util::rt::{TokioExecutor, TokioIo};
use hyper_util::server::conn::auto::Builder as ConnBuilder;
use minutist_common::{AppError, AppResult};
use rmcp::transport::streamable_http_server::session::local::LocalSessionManager;
use rmcp::transport::streamable_http_server::{StreamableHttpServerConfig, StreamableHttpService};
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;
use tower_service::Service as TowerService;

pub mod auth;
pub mod error;
pub mod handler;

pub use error::app_error_to_mcp;
pub use handler::McpToolHandler;

/// The MCP endpoint path (single Streamable HTTP endpoint, POST + GET).
pub const MCP_PATH: &str = "/mcp";

/// Configuration for one [`serve`] call.
pub struct McpServerConfig {
    /// The loopback address to bind. `app-main` sets `127.0.0.1:{mcp_port}`.
    pub bind_addr: SocketAddr,
    /// The bearer token every request must present. A high-entropy (256-bit
    /// OsRng) value generated + persisted by `app-main` to `{app-data}/mcp_token`
    /// (`0600`); keychain hardening is a follow-up. Handed here as a plain
    /// `String`, never logged.
    pub bearer_token: String,
    /// `settings.mcp_write_tools` (D3): when `false`, write tools are omitted
    /// from `tools/list` and rejected on `tools/call`.
    pub allow_writes: bool,
}

/// Serve the MCP endpoint on `config.bind_addr` until `shutdown` flips to
/// `true`. Returns the actually-bound address (so a `:0` test bind can be read
/// back).
///
/// `registry` MUST be a `ToolRegistry::v1(true)` instance (so
/// `send_to_internal_agent` is present) and `ctx` MUST carry the inter-agent
/// bridge sender (see [`ToolContext::with_inter_agent_bridge`]); `app-main`
/// builds both. The function returns once the bound address is known and the
/// accept loop has been spawned, OR an error if the bind fails.
///
/// Graceful shutdown: when `shutdown` flips, the accept loop stops taking new
/// connections and the rmcp session manager's cancellation token is fired, so
/// active sessions terminate.
pub async fn serve(
    registry: Arc<ToolRegistry>,
    ctx: Arc<ToolContext>,
    config: McpServerConfig,
    mut shutdown: watch::Receiver<bool>,
) -> AppResult<SocketAddr> {
    let listener = tokio::net::TcpListener::bind(config.bind_addr)
        .await
        .map_err(|e| AppError::Io {
            context: format!("MCP server failed to bind {}: {e}", config.bind_addr),
        })?;
    let bound = listener.local_addr().map_err(|e| AppError::Io {
        context: format!("MCP server failed to read bound addr: {e}"),
    })?;

    // rmcp's transport-level Host + Origin validation. The default
    // `allowed_hosts` is the loopback allowlist (kept; never disabled); we add
    // the loopback origins so a cross-origin browser request is a 403 (§4.3).
    // `StreamableHttpServerConfig` is `#[non_exhaustive]`; build via Default +
    // the `with_*` setters. The Default keeps the loopback `allowed_hosts`
    // allowlist (never disabled — GHSA-89vp-x53w-74fx) and stateful sessions.
    let cancel = CancellationToken::new();
    let http_config = StreamableHttpServerConfig::default()
        // Single request-response tools: return application/json directly (no
        // SSE framing overhead) while keeping stateful sessions for the
        // initialize handshake + Mcp-Session-Id routing.
        .with_json_response(true)
        .with_cancellation_token(cancel.clone())
        .with_allowed_origins(loopback_origins(bound.port()));

    // The rmcp Streamable HTTP service. `service_factory` builds a fresh handler
    // per session; all share the same `Arc<ToolRegistry>` + `Arc<ToolContext>`.
    let handler_registry = Arc::clone(&registry);
    let handler_ctx = Arc::clone(&ctx);
    let allow_writes = config.allow_writes;
    let mcp_service = StreamableHttpService::new(
        move || {
            Ok(McpToolHandler::new(
                Arc::clone(&handler_registry),
                Arc::clone(&handler_ctx),
                allow_writes,
            ))
        },
        Arc::new(LocalSessionManager::default()),
        http_config,
    );

    let token = Arc::new(config.bearer_token);

    tracing::info!(
        target: "mcp-server",
        addr = %bound,
        path = MCP_PATH,
        allow_writes,
        "MCP Streamable HTTP server listening (loopback, bearer-gated)"
    );

    // The accept loop runs as its own task so `serve` can return the bound addr
    // to `app-main` immediately (which emits the listening event). `app-main`
    // spawns `serve` itself via `tauri::async_runtime::spawn`, but the accept
    // loop must outlive this call, so it is a nested spawn.
    tokio::spawn(async move {
        loop {
            tokio::select! {
                // Shutdown signalled: stop accepting + cancel active sessions.
                changed = shutdown.changed() => {
                    // Either the sender flipped the flag or it was dropped; both
                    // mean "stop". `*borrow()` re-checks the value on a change.
                    if changed.is_err() || *shutdown.borrow() {
                        tracing::info!(target: "mcp-server", "MCP server shutting down");
                        cancel.cancel();
                        break;
                    }
                }
                accept = listener.accept() => {
                    match accept {
                        Ok((stream, peer)) => {
                            let svc = mcp_service.clone();
                            let token = Arc::clone(&token);
                            tokio::spawn(async move {
                                serve_connection(stream, peer, svc, token).await;
                            });
                        }
                        Err(e) => {
                            tracing::warn!(target: "mcp-server", "MCP accept error: {e}");
                        }
                    }
                }
            }
        }
    });

    Ok(bound)
}

/// Serve one accepted TCP connection: wrap the rmcp service in the bearer-check
/// adapter and drive it with hyper's auto (HTTP/1+2) connection builder.
async fn serve_connection(
    stream: tokio::net::TcpStream,
    peer: SocketAddr,
    mcp_service: StreamableHttpService<McpToolHandler, LocalSessionManager>,
    token: Arc<String>,
) {
    let io = TokioIo::new(stream);
    // The hyper service: bearer-check, then delegate to the rmcp tower service.
    let service = hyper::service::service_fn(move |req: Request<Incoming>| {
        let mut mcp = mcp_service.clone();
        let token = Arc::clone(&token);
        async move {
            // 401 if the bearer token is missing/wrong (before rmcp sees it).
            if !auth::bearer_ok(req.headers(), &token) {
                tracing::debug!(target: "mcp-server", %peer, "rejected MCP request: bad/missing bearer token");
                return Ok::<_, Infallible>(unauthorized_response());
            }
            // Delegate to rmcp's StreamableHttpService (a tower Service). It
            // enforces Host/Origin (403) and handles the MCP protocol; its
            // response body is already a BoxBody<Bytes, Infallible> (the
            // `BoxResponse` alias is pub(crate) in rmcp, so we name the concrete
            // type). The service is Infallible, so the error arm is unreachable.
            let resp: Response<BoxBody<Bytes, Infallible>> =
                mcp.call(req).await.unwrap_or_else(|never| match never {});
            Ok::<_, Infallible>(resp)
        }
    });

    if let Err(e) = ConnBuilder::new(TokioExecutor::new())
        .serve_connection_with_upgrades(io, service)
        .await
    {
        // A client disconnect mid-stream is normal for SSE; log at debug.
        tracing::debug!(target: "mcp-server", %peer, "MCP connection ended: {e}");
    }
}

/// A 401 Unauthorized response with the `WWW-Authenticate: Bearer` challenge.
fn unauthorized_response() -> Response<BoxBody<Bytes, Infallible>> {
    Response::builder()
        .status(StatusCode::UNAUTHORIZED)
        .header(http::header::WWW_AUTHENTICATE, "Bearer")
        .header(http::header::CONTENT_TYPE, "text/plain; charset=utf-8")
        .body(Full::new(Bytes::from_static(b"Unauthorized")).boxed())
        .expect("static 401 response is always valid")
}

/// The loopback `Origin` values to allow (Phase 10 §4.3). A browser page served
/// from another origin then gets a 403; a non-browser MCP client (no `Origin`
/// header) is unaffected (rmcp passes missing-Origin requests). Both `127.0.0.1`
/// and `localhost`, http + (defensively) https, at the bound port.
fn loopback_origins(port: u16) -> Vec<String> {
    let mut v = Vec::with_capacity(6);
    for host in ["127.0.0.1", "localhost", "[::1]"] {
        v.push(format!("http://{host}:{port}"));
        v.push(format!("https://{host}:{port}"));
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loopback_origins_cover_127_localhost_ipv6() {
        let origins = loopback_origins(8765);
        assert!(origins.contains(&"http://127.0.0.1:8765".to_string()));
        assert!(origins.contains(&"http://localhost:8765".to_string()));
        assert!(origins.contains(&"http://[::1]:8765".to_string()));
        // No bare 0.0.0.0 / wildcard origin is ever allowed.
        assert!(!origins.iter().any(|o| o.contains("0.0.0.0")));
        assert!(!origins.iter().any(|o| o == "*"));
    }

    #[test]
    fn unauthorized_response_is_401_with_challenge() {
        let resp = unauthorized_response();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
        assert_eq!(
            resp.headers().get(http::header::WWW_AUTHENTICATE).unwrap(),
            "Bearer"
        );
    }
}
