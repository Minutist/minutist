//! Device-code pairing client (WS4-A S5b, tc-pairing).
//!
//! The app links to an account with the OAuth 2.1 device authorization grant
//! (RFC 8628), driven by the relay's `account-service`. This module is the
//! app-side client of that contract:
//!
//! ```text
//!   POST {api}/pair/start {label?}  → user_code + verification_uri + device_code
//!   (the user opens the URL, logs into rauthy, approves)
//!   POST {api}/pair/poll {device_code}  (every `interval`s)
//!     → authorization_pending  (keep polling)
//!     → slow_down              (add 5s to the interval, then poll)
//!     → expired_token          (terminal — restart pairing)
//!     → access_denied          (terminal — user declined)
//!     → authorised {device_credential, account_id, device_id}
//! ```
//!
//! On `authorised` the body carries the **plaintext** `device_credential`
//! (format `mdc_<device_id>.<secret>`) returned exactly once; the caller stores
//! it and presents it verbatim as the tunnel `Hello.device_credential`.
//!
//! # Separation of concerns
//!
//! [`DeviceCodeClient`] owns only the HTTP I/O against the account-service. The
//! poll-pacing policy (when to poll, how `slow_down` adjusts the interval, and
//! the local expiry check) is the caller's loop — driven against
//! [`DeviceCodeClient::poll_once`] — so `app-main`/`ipc-bridge` can interleave
//! the wait with the lifecycle's stop signal. The pure interval-pacing rule is
//! exposed as [`next_interval`] so it is unit-testable without I/O.

use std::time::Duration;

use serde::{Deserialize, Serialize};

/// Minimum poll interval the client will honour, even if the server returns a
/// smaller `interval`. RFC 8628 recommends 5 s as the default; we floor at that
/// to avoid hammering the token endpoint.
pub const MIN_POLL_INTERVAL_SECS: u64 = 5;

/// The increment added to the poll interval on a `slow_down` response
/// (RFC 8628 §3.5 mandates 5 s).
pub const SLOW_DOWN_INCREMENT_SECS: u64 = 5;

/// What `POST /pair/start` returns: the codes + URLs to show the user, plus the
/// `device_code` the caller polls with and the server's pacing hints.
#[derive(Debug, Clone, Deserialize)]
pub struct PairingStart {
    /// The poll handle. Opaque; held by the caller and sent on every poll. Not
    /// shown to the user.
    pub device_code: String,
    /// The short code the user types into the verification page.
    pub user_code: String,
    /// The URL the user opens to enter `user_code` and approve.
    pub verification_uri: String,
    /// The URL with `user_code` pre-filled, if the server provides one. Opening
    /// this is the smoother flow (no manual code entry); falls back to
    /// `verification_uri` when absent.
    #[serde(default)]
    pub verification_uri_complete: Option<String>,
    /// Seconds until the `device_code` expires and pairing must restart.
    pub expires_in: u64,
    /// The server's recommended seconds between polls.
    pub interval: u64,
}

impl PairingStart {
    /// The URL to open in the user's browser: the pre-filled one when present,
    /// else the bare verification URL.
    pub fn open_url(&self) -> &str {
        self.verification_uri_complete
            .as_deref()
            .unwrap_or(&self.verification_uri)
    }

    /// The initial poll interval, floored at [`MIN_POLL_INTERVAL_SECS`].
    pub fn initial_interval(&self) -> Duration {
        Duration::from_secs(self.interval.max(MIN_POLL_INTERVAL_SECS))
    }
}

/// The outcome of one `POST /pair/poll`. Mirrors the relay's `status` field.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PollOutcome {
    /// The user has not authorised yet. Keep polling at the current interval.
    Pending,
    /// Polling too fast. Add [`SLOW_DOWN_INCREMENT_SECS`] to the interval, then
    /// keep polling.
    SlowDown,
    /// The user authorised; the credential was issued. Terminal success.
    Authorised(IssuedDeviceCredential),
}

/// The credential + identity the server mints on a successful pairing. The
/// `device_credential` is plaintext, returned exactly once.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IssuedDeviceCredential {
    /// The full plaintext credential (`mdc_<device_id>.<secret>`) to present as
    /// the tunnel `Hello.device_credential`. Sensitive — never logged.
    pub device_credential: String,
    /// The account the credential is bound to (the rauthy `sub`).
    pub account_id: String,
    /// The device's registry id (for the device list / unpair).
    pub device_id: String,
}

/// A pairing failure. Coarse, structured, and `thiserror`-derived so it crosses
/// the `tunnel-client` public boundary without `anyhow`.
#[derive(Debug, thiserror::Error)]
pub enum PairingError {
    /// The configured api base URL is not `https://` (and not a loopback
    /// `http://`). Pairing carries the device credential, so the channel must be
    /// encrypted off-machine — same rule as the tunnel relay scheme check.
    #[error("api_base_url must use https:// (http:// is allowed only for a loopback host)")]
    Config,
    /// The HTTP request to the account-service failed (DNS, connect, timeout).
    #[error("failed to reach the account service")]
    Transport(#[source] reqwest::Error),
    /// The account-service returned a non-success status.
    #[error("account service returned status {status}")]
    Status { status: u16 },
    /// The response body could not be decoded into the expected shape.
    #[error("undecodable response from the account service")]
    Decode(#[source] reqwest::Error),
    /// The device_code expired before the user authorised. Terminal — restart.
    #[error("pairing code expired before authorisation")]
    Expired,
    /// The user declined at the consent screen. Terminal.
    #[error("pairing was declined")]
    AccessDenied,
    /// The server returned an unrecognised `authorised` body (missing the
    /// credential fields). Treated as a hard failure rather than retried.
    #[error("malformed authorisation response")]
    MalformedAuthorisation,
}

/// The HTTP client for the device-code pairing endpoints. Holds the api base URL
/// (e.g. `https://api.minutist.ai`) and a `reqwest::Client`.
#[derive(Clone)]
pub struct DeviceCodeClient {
    http: reqwest::Client,
    api_base_url: String,
}

/// The raw `POST /pair/poll` body. `status` selects the outcome; the credential
/// fields are present only on `authorised`.
#[derive(Debug, Deserialize)]
struct PollResponse {
    status: String,
    #[serde(default)]
    device_credential: Option<String>,
    #[serde(default)]
    account_id: Option<String>,
    #[serde(default)]
    device_id: Option<String>,
}

#[derive(Debug, Serialize)]
struct StartRequest<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    label: Option<&'a str>,
}

#[derive(Debug, Serialize)]
struct PollRequest<'a> {
    device_code: &'a str,
}

impl DeviceCodeClient {
    /// Build a client for the given account-service base URL (no trailing
    /// `/pair/...` — just the origin, e.g. `https://api.minutist.ai`). Rejects a
    /// non-`https` off-machine URL before any request is sent.
    pub fn new(api_base_url: impl Into<String>) -> Result<Self, PairingError> {
        let api_base_url = api_base_url.into();
        if !api_scheme_is_acceptable(&api_base_url) {
            return Err(PairingError::Config);
        }
        let http = reqwest::Client::builder()
            .build()
            .map_err(PairingError::Transport)?;
        Ok(Self { http, api_base_url })
    }

    /// Build a client around an existing `reqwest::Client` (so the app reuses its
    /// shared client). Same scheme check as [`new`](Self::new).
    pub fn with_client(
        http: reqwest::Client,
        api_base_url: impl Into<String>,
    ) -> Result<Self, PairingError> {
        let api_base_url = api_base_url.into();
        if !api_scheme_is_acceptable(&api_base_url) {
            return Err(PairingError::Config);
        }
        Ok(Self { http, api_base_url })
    }

    fn url(&self, path: &str) -> String {
        format!("{}{path}", self.api_base_url.trim_end_matches('/'))
    }

    /// `POST /pair/start` — begin a pairing. `label` is an optional
    /// human-readable device name stored in the registry.
    pub async fn start(&self, label: Option<&str>) -> Result<PairingStart, PairingError> {
        let resp = self
            .http
            .post(self.url("/pair/start"))
            .json(&StartRequest { label })
            .send()
            .await
            .map_err(PairingError::Transport)?;
        let status = resp.status();
        if !status.is_success() {
            return Err(PairingError::Status {
                status: status.as_u16(),
            });
        }
        resp.json::<PairingStart>()
            .await
            .map_err(PairingError::Decode)
    }

    /// `POST /pair/poll` — one poll. Maps the server `status` to a
    /// [`PollOutcome`] (the non-terminal cases) or a terminal [`PairingError`]
    /// (`Expired` / `AccessDenied` / `MalformedAuthorisation`). The caller's loop
    /// sleeps `interval` between calls and applies the `slow_down` increment on
    /// [`PollOutcome::SlowDown`].
    pub async fn poll_once(&self, device_code: &str) -> Result<PollOutcome, PairingError> {
        let resp = self
            .http
            .post(self.url("/pair/poll"))
            .json(&PollRequest { device_code })
            .send()
            .await
            .map_err(PairingError::Transport)?;
        let status = resp.status();
        if !status.is_success() {
            return Err(PairingError::Status {
                status: status.as_u16(),
            });
        }
        let body = resp
            .json::<PollResponse>()
            .await
            .map_err(PairingError::Decode)?;
        classify_poll(body)
    }
}

/// Map a decoded poll body to an outcome or a terminal error. Pure, so the
/// status→outcome contract is unit-testable without HTTP.
fn classify_poll(body: PollResponse) -> Result<PollOutcome, PairingError> {
    match body.status.as_str() {
        "authorization_pending" => Ok(PollOutcome::Pending),
        "slow_down" => Ok(PollOutcome::SlowDown),
        "expired_token" => Err(PairingError::Expired),
        "access_denied" => Err(PairingError::AccessDenied),
        "authorised" | "authorized" => {
            match (body.device_credential, body.account_id, body.device_id) {
                (Some(device_credential), Some(account_id), Some(device_id))
                    if !device_credential.is_empty()
                        && !account_id.is_empty()
                        && !device_id.is_empty() =>
                {
                    Ok(PollOutcome::Authorised(IssuedDeviceCredential {
                        device_credential,
                        account_id,
                        device_id,
                    }))
                }
                _ => Err(PairingError::MalformedAuthorisation),
            }
        }
        // An unrecognised status is treated as pending — never invent a terminal
        // failure the user cannot recover from, mirroring the relay's
        // conservative classification.
        _ => Ok(PollOutcome::Pending),
    }
}

/// The next poll interval after a `slow_down`: add [`SLOW_DOWN_INCREMENT_SECS`]
/// to the current interval (RFC 8628 §3.5), never dropping below the floor.
pub fn next_interval(current: Duration) -> Duration {
    let secs = current
        .as_secs()
        .saturating_add(SLOW_DOWN_INCREMENT_SECS)
        .max(MIN_POLL_INTERVAL_SECS);
    Duration::from_secs(secs)
}

/// Whether `url` is an acceptable account-service base URL: `https://` for any
/// host, or `http://` only for a loopback host (test/dev). Mirrors the tunnel
/// `relay_scheme_is_acceptable` check.
fn api_scheme_is_acceptable(url: &str) -> bool {
    let Some((scheme, rest)) = url.split_once("://") else {
        return false;
    };
    if scheme.eq_ignore_ascii_case("https") {
        return true;
    }
    if scheme.eq_ignore_ascii_case("http") {
        let authority = rest.split('/').next().unwrap_or("");
        let host = if let Some(after) = authority.strip_prefix('[') {
            after.split(']').next().unwrap_or("")
        } else {
            authority.split(':').next().unwrap_or("")
        };
        return host == "127.0.0.1" || host == "::1" || host.eq_ignore_ascii_case("localhost");
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    fn poll(status: &str) -> PollResponse {
        PollResponse {
            status: status.to_string(),
            device_credential: None,
            account_id: None,
            device_id: None,
        }
    }

    #[test]
    fn pending_and_slow_down_map_to_outcomes() {
        assert_eq!(
            classify_poll(poll("authorization_pending")).unwrap(),
            PollOutcome::Pending
        );
        assert_eq!(
            classify_poll(poll("slow_down")).unwrap(),
            PollOutcome::SlowDown
        );
    }

    #[test]
    fn terminal_codes_map_to_errors() {
        assert!(matches!(
            classify_poll(poll("expired_token")),
            Err(PairingError::Expired)
        ));
        assert!(matches!(
            classify_poll(poll("access_denied")),
            Err(PairingError::AccessDenied)
        ));
    }

    #[test]
    fn unknown_status_is_treated_as_pending() {
        assert_eq!(classify_poll(poll("weird")).unwrap(), PollOutcome::Pending);
    }

    #[test]
    fn authorised_requires_all_credential_fields() {
        let full = PollResponse {
            status: "authorised".to_string(),
            device_credential: Some("mdc_dev.secret".to_string()),
            account_id: Some("acct".to_string()),
            device_id: Some("dev".to_string()),
        };
        match classify_poll(full).unwrap() {
            PollOutcome::Authorised(c) => {
                assert_eq!(c.device_credential, "mdc_dev.secret");
                assert_eq!(c.account_id, "acct");
                assert_eq!(c.device_id, "dev");
            }
            other => panic!("expected Authorised, got {other:?}"),
        }

        // Missing any field is a malformed authorisation, not a silent success.
        let partial = PollResponse {
            status: "authorised".to_string(),
            device_credential: Some("mdc_dev.secret".to_string()),
            account_id: None,
            device_id: Some("dev".to_string()),
        };
        assert!(matches!(
            classify_poll(partial),
            Err(PairingError::MalformedAuthorisation)
        ));
    }

    #[test]
    fn slow_down_adds_five_seconds() {
        assert_eq!(
            next_interval(Duration::from_secs(5)),
            Duration::from_secs(10)
        );
        assert_eq!(
            next_interval(Duration::from_secs(10)),
            Duration::from_secs(15)
        );
    }

    #[test]
    fn initial_interval_floors_at_minimum() {
        let start = PairingStart {
            device_code: "dc".to_string(),
            user_code: "ABCD-1234".to_string(),
            verification_uri: "https://auth.example/device".to_string(),
            verification_uri_complete: None,
            expires_in: 600,
            interval: 1,
        };
        assert_eq!(
            start.initial_interval(),
            Duration::from_secs(MIN_POLL_INTERVAL_SECS)
        );
    }

    #[test]
    fn open_url_prefers_complete_uri() {
        let mut start = PairingStart {
            device_code: "dc".to_string(),
            user_code: "ABCD-1234".to_string(),
            verification_uri: "https://auth.example/device".to_string(),
            verification_uri_complete: Some(
                "https://auth.example/device?code=ABCD-1234".to_string(),
            ),
            expires_in: 600,
            interval: 5,
        };
        assert_eq!(
            start.open_url(),
            "https://auth.example/device?code=ABCD-1234"
        );
        start.verification_uri_complete = None;
        assert_eq!(start.open_url(), "https://auth.example/device");
    }

    #[test]
    fn client_rejects_cleartext_remote_api() {
        assert!(DeviceCodeClient::new("https://api.minutist.ai").is_ok());
        assert!(DeviceCodeClient::new("http://127.0.0.1:8483").is_ok());
        assert!(DeviceCodeClient::new("http://localhost:8483").is_ok());
        assert!(matches!(
            DeviceCodeClient::new("http://api.minutist.ai"),
            Err(PairingError::Config)
        ));
        assert!(matches!(
            DeviceCodeClient::new("ftp://api.minutist.ai"),
            Err(PairingError::Config)
        ));
        assert!(matches!(
            DeviceCodeClient::new("api.minutist.ai"),
            Err(PairingError::Config)
        ));
    }
}
