//! Account device-directory HTTP client (WS4-B).
//!
//! The connected desktop/phone publishes its own sync endpoint to the account
//! service and reads the account's other devices' endpoints, so the sync engine
//! can dial its paired devices without a manual ticket exchange. This is the raw
//! HTTP client for the two device-directory endpoints; the `sync`
//! `AccountEndpointSource` trait it feeds is implemented by the caller
//! (`app-main`), which adapts these DTOs onto that trait — so this crate stays a
//! near-leaf with no dependency edge on `sync`.
//!
//! Both endpoints authenticate with the device's long-lived `mdc_` credential as
//! `Authorization: Bearer …` (the same credential the tunnel presents). It is
//! held here and redacted from `Debug`; it is never logged.

use serde::{Deserialize, Serialize};

use crate::pairing::api_scheme_is_acceptable;

/// One endpoint-registered device from `GET /v1/account/devices`. The service
/// filters to rows whose endpoint + relay are both set (an unregistered device
/// is absent from the array, never present-with-null), so every field is a plain
/// `String`.
#[derive(Debug, Clone, Deserialize)]
pub struct DeviceEndpointEntry {
    pub device_id: String,
    pub endpoint_id: String,
    pub relay_url: String,
    /// The device's published direct socket addresses ("ip:port"), for a
    /// same-network peer to dial without the relay (0049). Absent/older entries
    /// decode to an empty vec.
    #[serde(default)]
    pub direct_addrs: Vec<String>,
}

/// A device-directory request failure. `thiserror`-derived so it crosses the
/// crate's public boundary without `anyhow`.
#[derive(Debug, thiserror::Error)]
pub enum AccountDirectoryError {
    /// The configured api base URL is not `https://` (nor a loopback `http://`).
    /// The credential rides these requests, so the channel must be encrypted
    /// off-machine — the same rule as the pairing client.
    #[error("api_base_url must use https:// (http:// is allowed only for a loopback host)")]
    Config,
    /// The HTTP request failed (DNS, connect, timeout).
    #[error("failed to reach the account service")]
    Transport(#[source] reqwest::Error),
    /// `401` — the credential is missing, malformed, unknown, or revoked. The
    /// service returns a uniform 401 for all of these (no oracle), so the only
    /// remedy is to re-pair the device.
    #[error("account service rejected the device credential")]
    Unauthorised,
    /// Any other non-success status (e.g. `503` when the directory DB is
    /// unreachable — transient, safe to retry).
    #[error("account service returned status {status}")]
    Status { status: u16 },
    /// The response body could not be decoded into the expected shape.
    #[error("undecodable response from the account service")]
    Decode(#[source] reqwest::Error),
}

/// The `PUT /v1/account/devices/self/endpoint` request body. The device is
/// identified by the credential, so only the endpoint fields are sent.
#[derive(Debug, Serialize)]
struct RegisterEndpointRequest<'a> {
    endpoint_id: &'a str,
    relay_url: &'a str,
    direct_addrs: &'a [String],
}

/// HTTP client for the account device-directory, bound to one device's bearer
/// credential. `Debug` redacts the credential.
#[derive(Clone)]
pub struct AccountDirectoryClient {
    http: reqwest::Client,
    api_base_url: String,
    device_credential: String,
}

impl std::fmt::Debug for AccountDirectoryClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AccountDirectoryClient")
            .field("api_base_url", &self.api_base_url)
            .field("device_credential", &"<redacted>")
            .finish()
    }
}

impl AccountDirectoryClient {
    /// Build a client for the account-service base URL (origin only, e.g.
    /// `https://api.minutist.ai`), bound to `device_credential`. Rejects a
    /// non-`https` off-machine URL before any request is sent.
    pub fn new(
        api_base_url: impl Into<String>,
        device_credential: impl Into<String>,
    ) -> Result<Self, AccountDirectoryError> {
        let http = reqwest::Client::builder()
            .build()
            .map_err(AccountDirectoryError::Transport)?;
        Self::with_client(http, api_base_url, device_credential)
    }

    /// Build a client around an existing `reqwest::Client` (so the app reuses its
    /// shared client). Same scheme check as [`new`](Self::new).
    pub fn with_client(
        http: reqwest::Client,
        api_base_url: impl Into<String>,
        device_credential: impl Into<String>,
    ) -> Result<Self, AccountDirectoryError> {
        let api_base_url = api_base_url.into();
        if !api_scheme_is_acceptable(&api_base_url) {
            return Err(AccountDirectoryError::Config);
        }
        Ok(Self {
            http,
            api_base_url,
            device_credential: device_credential.into(),
        })
    }

    fn url(&self, path: &str) -> String {
        format!("{}{path}", self.api_base_url.trim_end_matches('/'))
    }

    /// `GET /v1/account/devices` — every endpoint-registered device on this
    /// account, including this device's own row once it has registered. The
    /// caller filters its own entry out (see `sync::peers_to_add`).
    pub async fn list_devices(&self) -> Result<Vec<DeviceEndpointEntry>, AccountDirectoryError> {
        let resp = self
            .http
            .get(self.url("/v1/account/devices"))
            .bearer_auth(&self.device_credential)
            .send()
            .await
            .map_err(AccountDirectoryError::Transport)?;
        status_check(resp.status())?;
        resp.json::<Vec<DeviceEndpointEntry>>()
            .await
            .map_err(AccountDirectoryError::Decode)
    }

    /// `PUT /v1/account/devices/self/endpoint` — publish this device's endpoint
    /// (idempotent upsert; the device is identified by the credential, not the
    /// body). `direct_addrs` are this device's published direct socket addresses
    /// ("ip:port") for same-network peers to dial without the relay (0049);
    /// empty is valid (relay-only). Success is `204 No Content` with an empty
    /// body.
    pub async fn register_self_endpoint(
        &self,
        endpoint_id: &str,
        relay_url: &str,
        direct_addrs: &[String],
    ) -> Result<(), AccountDirectoryError> {
        let resp = self
            .http
            .put(self.url("/v1/account/devices/self/endpoint"))
            .bearer_auth(&self.device_credential)
            .json(&RegisterEndpointRequest {
                endpoint_id,
                relay_url,
                direct_addrs,
            })
            .send()
            .await
            .map_err(AccountDirectoryError::Transport)?;
        status_check(resp.status())?;
        Ok(())
    }

    /// `DELETE /v1/account` — erase this account and everything the service
    /// derives from it: every device on the account (labels, credential hashes,
    /// endpoint addressing, published direct addresses) and the rauthy identity
    /// (email + credentials). The account is identified by the credential, so no
    /// body is sent. Success is `204 No Content`.
    ///
    /// A `401` ([`AccountDirectoryError::Unauthorised`]) means the credential no
    /// longer resolves — the account is already gone — which the caller may treat
    /// as erasure already complete when tearing down local state.
    pub async fn delete_account(&self) -> Result<(), AccountDirectoryError> {
        let resp = self
            .http
            .delete(self.url("/v1/account"))
            .bearer_auth(&self.device_credential)
            .send()
            .await
            .map_err(AccountDirectoryError::Transport)?;
        status_check(resp.status())?;
        Ok(())
    }
}

/// Map a response status to the coarse error. `401` is the service's uniform
/// auth-failure code; any other non-2xx is surfaced by code (the caller retries
/// `503`, re-pairs on `Unauthorised`).
fn status_check(status: reqwest::StatusCode) -> Result<(), AccountDirectoryError> {
    if status.is_success() {
        return Ok(());
    }
    if status == reqwest::StatusCode::UNAUTHORIZED {
        return Err(AccountDirectoryError::Unauthorised);
    }
    Err(AccountDirectoryError::Status {
        status: status.as_u16(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_non_https_off_machine_base() {
        assert!(matches!(
            AccountDirectoryClient::new("http://api.minutist.ai", "mdc_x"),
            Err(AccountDirectoryError::Config)
        ));
        assert!(AccountDirectoryClient::new("https://api.minutist.ai", "mdc_x").is_ok());
        // Loopback http is allowed (mirrors the pairing-client rule).
        assert!(AccountDirectoryClient::new("http://127.0.0.1:8483", "mdc_x").is_ok());
    }

    #[test]
    fn debug_redacts_the_credential() {
        let c =
            AccountDirectoryClient::new("https://api.minutist.ai", "mdc_super.secret").unwrap();
        let dbg = format!("{c:?}");
        assert!(!dbg.contains("secret"), "credential must not appear in Debug");
        assert!(dbg.contains("<redacted>"));
    }

    #[test]
    fn status_check_maps_statuses() {
        assert!(matches!(
            status_check(reqwest::StatusCode::UNAUTHORIZED),
            Err(AccountDirectoryError::Unauthorised)
        ));
        assert!(matches!(
            status_check(reqwest::StatusCode::SERVICE_UNAVAILABLE),
            Err(AccountDirectoryError::Status { status: 503 })
        ));
        assert!(status_check(reqwest::StatusCode::NO_CONTENT).is_ok());
        assert!(status_check(reqwest::StatusCode::OK).is_ok());
    }

    // ---- Wire-contract tests against a mock account-service (B1 schema) ----

    use wiremock::matchers::{body_json, header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn client_for(server: &MockServer) -> AccountDirectoryClient {
        // `MockServer::uri()` is a loopback `http://` origin — accepted by the
        // scheme check (same rule as the pairing client).
        AccountDirectoryClient::new(server.uri(), "mdc_test.secret").unwrap()
    }

    #[tokio::test]
    async fn list_devices_parses_bare_array_and_sends_bearer() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/account/devices"))
            .and(header("authorization", "Bearer mdc_test.secret"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!([
                {"device_id": "dev-a", "endpoint_id": "ep-a", "relay_url": "https://sync.example/r", "direct_addrs": ["100.82.58.55:41641", "192.168.0.9:41641"]},
                {"device_id": "dev-b", "endpoint_id": "ep-b", "relay_url": "https://sync.example/r"}
            ])))
            .expect(1)
            .mount(&server)
            .await;

        let devices = client_for(&server).list_devices().await.unwrap();
        assert_eq!(devices.len(), 2);
        assert_eq!(devices[0].device_id, "dev-a");
        assert_eq!(devices[0].endpoint_id, "ep-a");
        assert_eq!(
            devices[0].direct_addrs,
            vec!["100.82.58.55:41641".to_string(), "192.168.0.9:41641".to_string()]
        );
        assert_eq!(devices[1].relay_url, "https://sync.example/r");
        // An entry that omits direct_addrs decodes to an empty vec (older
        // client / relay-only device).
        assert!(devices[1].direct_addrs.is_empty());
    }

    #[tokio::test]
    async fn register_self_sends_put_body_and_accepts_204() {
        let server = MockServer::start().await;
        Mock::given(method("PUT"))
            .and(path("/v1/account/devices/self/endpoint"))
            .and(header("authorization", "Bearer mdc_test.secret"))
            .and(body_json(serde_json::json!({
                "endpoint_id": "ep-self",
                "relay_url": "https://sync.example/r",
                "direct_addrs": ["100.82.58.55:41641"]
            })))
            .respond_with(ResponseTemplate::new(204))
            .expect(1)
            .mount(&server)
            .await;

        client_for(&server)
            .register_self_endpoint(
                "ep-self",
                "https://sync.example/r",
                &["100.82.58.55:41641".to_string()],
            )
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn delete_account_sends_delete_with_bearer_and_accepts_204() {
        let server = MockServer::start().await;
        Mock::given(method("DELETE"))
            .and(path("/v1/account"))
            .and(header("authorization", "Bearer mdc_test.secret"))
            .respond_with(ResponseTemplate::new(204))
            .expect(1)
            .mount(&server)
            .await;

        client_for(&server).delete_account().await.unwrap();
    }

    #[tokio::test]
    async fn delete_account_maps_401_to_unauthorised() {
        let server = MockServer::start().await;
        Mock::given(method("DELETE"))
            .and(path("/v1/account"))
            .respond_with(ResponseTemplate::new(401))
            .mount(&server)
            .await;
        assert!(matches!(
            client_for(&server).delete_account().await,
            Err(AccountDirectoryError::Unauthorised)
        ));
    }

    #[tokio::test]
    async fn maps_401_to_unauthorised_and_503_to_status() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/account/devices"))
            .respond_with(ResponseTemplate::new(401))
            .mount(&server)
            .await;
        assert!(matches!(
            client_for(&server).list_devices().await,
            Err(AccountDirectoryError::Unauthorised)
        ));

        let server503 = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/account/devices"))
            .respond_with(ResponseTemplate::new(503))
            .mount(&server503)
            .await;
        assert!(matches!(
            client_for(&server503).list_devices().await,
            Err(AccountDirectoryError::Status { status: 503 })
        ));
    }
}
