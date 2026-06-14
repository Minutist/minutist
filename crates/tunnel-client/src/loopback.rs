//! The loopback target: where relayed requests are replayed, and the internal
//! bearer applied to them.

use std::fmt;

/// The app's internal loopback bearer token. Newtyped so its `Debug` redacts the
/// value — it must never appear in logs and is attached only to the outbound
/// loopback HTTP request, never serialised into a tunnel frame.
#[derive(Clone)]
pub struct InternalBearer(String);

impl InternalBearer {
    /// Wrap the raw token. Take the value from `ipc-bridge::McpServerInfo.token`.
    pub fn new(token: impl Into<String>) -> Self {
        Self(token.into())
    }

    /// The `Authorization` header value (`Bearer <token>`) to attach to the
    /// loopback request. Crate-internal — there is no public accessor for the
    /// raw secret.
    pub(crate) fn header_value(&self) -> String {
        format!("Bearer {}", self.0)
    }
}

impl fmt::Debug for InternalBearer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("InternalBearer(<redacted>)")
    }
}

/// Where to replay relayed MCP requests: the app's loopback `mcp-server` base
/// URL plus the internal bearer it expects.
///
/// `base_url` is the scheme+authority of the loopback server, e.g.
/// `http://127.0.0.1:8765` (NO trailing slash, NO path — the relay frame carries
/// the path, e.g. `/mcp`). `app-main` builds this from `McpServerInfo`: the
/// `McpServerInfo.url` is the full `…/mcp` endpoint, so the caller strips the
/// path to the origin and keeps the token as the bearer.
#[derive(Clone, Debug)]
pub struct LoopbackTarget {
    pub base_url: String,
    pub internal_bearer: InternalBearer,
}

impl LoopbackTarget {
    /// Build a target from the loopback origin and the internal bearer.
    pub fn new(base_url: impl Into<String>, internal_bearer: InternalBearer) -> Self {
        Self {
            base_url: base_url.into(),
            internal_bearer,
        }
    }

    /// Join the origin with a request path. The relay path is already absolute
    /// (`/mcp`); a missing leading slash is tolerated.
    pub(crate) fn url_for(&self, path: &str) -> String {
        let base = self.base_url.trim_end_matches('/');
        if path.starts_with('/') {
            format!("{base}{path}")
        } else {
            format!("{base}/{path}")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn debug_redacts_bearer() {
        let bearer = InternalBearer::new("super-secret-token");
        let shown = format!("{bearer:?}");
        assert!(!shown.contains("super-secret-token"), "bearer leaked in Debug");
        assert!(shown.contains("redacted"));
    }

    #[test]
    fn debug_of_target_redacts_bearer() {
        let target = LoopbackTarget::new("http://127.0.0.1:8765", InternalBearer::new("tok"));
        let shown = format!("{target:?}");
        assert!(!shown.contains("tok"), "bearer leaked via LoopbackTarget Debug");
    }

    #[test]
    fn url_join_handles_slashes() {
        let t = LoopbackTarget::new("http://127.0.0.1:8765/", InternalBearer::new("x"));
        assert_eq!(t.url_for("/mcp"), "http://127.0.0.1:8765/mcp");
        let t = LoopbackTarget::new("http://127.0.0.1:8765", InternalBearer::new("x"));
        assert_eq!(t.url_for("mcp"), "http://127.0.0.1:8765/mcp");
    }

    #[test]
    fn header_value_is_bearer_scheme() {
        assert_eq!(InternalBearer::new("abc").header_value(), "Bearer abc");
    }
}
