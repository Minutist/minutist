//! Bearer-token authentication for the MCP transport (Phase 10 §4.2).
//!
//! Every request must carry `Authorization: Bearer <token>` matching the
//! configured token; a mismatch is a 401. This is a transport-level check that
//! runs BEFORE the request reaches rmcp's `StreamableHttpService`, so the
//! session id is never the credential (spec MUST — the session id is routing
//! state only).
//!
//! The `Host` / `Origin` 403 checks are NOT here: rmcp's
//! `StreamableHttpServerConfig` enforces them natively (the loopback
//! `allowed_hosts` default + the `allowed_origins` we configure in `lib.rs`).
//! Keeping bearer separate as a thin wrapper is what lets the 401 fire even on a
//! request that would otherwise pass Host/Origin.

use http::HeaderMap;
use subtle::ConstantTimeEq;

/// Whether `headers` carry a `Authorization: Bearer <token>` that matches
/// `expected`. Pure so it is unit-testable without a live HTTP client.
///
/// - Missing `Authorization` header → `false` (401).
/// - Non-`Bearer` scheme → `false`.
/// - Wrong token → `false`.
/// - Exact `Bearer <expected>` → `true`.
///
/// The token comparison is constant-time (`subtle::ConstantTimeEq`). The
/// loopback bind + Host/Origin checks are the primary controls (§4.3) — the
/// token is a ≥256-bit CSPRNG value, not a low-entropy password — but a plain
/// `==` still leaks the length of the matching prefix through timing to any
/// process on the same machine that can hit this endpoint, so the compare
/// itself costs nothing extra to harden.
pub fn bearer_ok(headers: &HeaderMap, expected: &str) -> bool {
    let Some(value) = headers.get(http::header::AUTHORIZATION) else {
        return false;
    };
    let Ok(value) = value.to_str() else {
        return false;
    };
    // Case-insensitive scheme per RFC 7235; the token itself is case-sensitive.
    let Some(token) = value
        .strip_prefix("Bearer ")
        .or_else(|| value.strip_prefix("bearer "))
    else {
        return false;
    };
    let token = token.trim();
    !expected.is_empty() && bool::from(token.as_bytes().ct_eq(expected.as_bytes()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use http::header::AUTHORIZATION;

    fn headers_with_auth(value: &str) -> HeaderMap {
        let mut h = HeaderMap::new();
        h.insert(AUTHORIZATION, value.parse().unwrap());
        h
    }

    #[test]
    fn correct_bearer_token_passes() {
        let h = headers_with_auth("Bearer secret-token-123");
        assert!(bearer_ok(&h, "secret-token-123"));
    }

    #[test]
    fn lowercase_bearer_scheme_passes() {
        let h = headers_with_auth("bearer secret-token-123");
        assert!(bearer_ok(&h, "secret-token-123"));
    }

    #[test]
    fn missing_header_fails() {
        let h = HeaderMap::new();
        assert!(!bearer_ok(&h, "secret-token-123"));
    }

    #[test]
    fn wrong_token_fails() {
        let h = headers_with_auth("Bearer wrong");
        assert!(!bearer_ok(&h, "secret-token-123"));
    }

    #[test]
    fn non_bearer_scheme_fails() {
        let h = headers_with_auth("Basic secret-token-123");
        assert!(!bearer_ok(&h, "secret-token-123"));
    }

    #[test]
    fn empty_expected_token_never_matches() {
        // A misconfigured empty token must NOT authenticate any request,
        // including a literal "Bearer " with no value.
        let h = headers_with_auth("Bearer ");
        assert!(!bearer_ok(&h, ""));
        let h2 = headers_with_auth("Bearer anything");
        assert!(!bearer_ok(&h2, ""));
    }

    #[test]
    fn token_with_trailing_whitespace_is_trimmed() {
        let h = headers_with_auth("Bearer secret-token-123  ");
        assert!(bearer_ok(&h, "secret-token-123"));
    }
}
