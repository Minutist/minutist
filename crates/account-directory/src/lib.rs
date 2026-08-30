//! Adapts `tunnel_client::AccountDirectoryClient` (an HTTP client) onto
//! `sync::AccountEndpointSource` (the trait `sync` depends on instead of an
//! HTTP client), so `sync` and `tunnel-client` never take an edge on each
//! other. `app-main` (its account-refresh wiring) and `headless` (its
//! account-discovery startup path) — the two account-mediated-discovery
//! consumers — both depend on this crate instead of each carrying their own
//! copy of the adapter: before this crate existed, the two implementations
//! were identical apart from an `AppError` import qualifier.
//!
//! A leaf on top of both `sync` and `tunnel-client`. It exists only to serve
//! the account-mediated discovery loop, so it is part of the connected feature
//! surface — every consumer that wires it in does so behind the same
//! `connected` Cargo feature as `sync` / `tunnel-client` / `mcp-server` /
//! `election`; the free build has no relay and no account directory to adapt.

use minutist_common::{AppError, AppResult};
use sync::{AccountEndpoint, AccountEndpointSource};

/// The account-directory adapter. Construct with an already-authenticated
/// [`tunnel_client::AccountDirectoryClient`] and pass as `Arc<dyn
/// AccountEndpointSource>` wherever `sync` needs one.
pub struct AccountDirectorySource {
    client: tunnel_client::AccountDirectoryClient,
}

impl AccountDirectorySource {
    pub fn new(client: tunnel_client::AccountDirectoryClient) -> Self {
        Self { client }
    }
}

#[async_trait::async_trait]
impl AccountEndpointSource for AccountDirectorySource {
    async fn list_endpoints(&self) -> AppResult<Vec<AccountEndpoint>> {
        let devices = self
            .client
            .list_devices()
            .await
            .map_err(|e| AppError::Internal {
                context: format!("account directory list: {e}"),
            })?;
        Ok(devices
            .into_iter()
            .map(|d| AccountEndpoint {
                device_id: d.device_id,
                endpoint_id: d.endpoint_id,
                relay_url: d.relay_url,
                direct_addrs: d.direct_addrs,
                // The one place an account-published label crosses the network
                // into the app, so the one place it is normalised.
                label: sync::sanitise_label(&d.label),
            })
            .collect())
    }

    async fn register_self(&self, endpoint: &AccountEndpoint) -> AppResult<()> {
        self.client
            .register_self_endpoint(
                &endpoint.endpoint_id,
                &endpoint.relay_url,
                &endpoint.direct_addrs,
                endpoint.label.as_deref(),
            )
            .await
            .map_err(|e| AppError::Internal {
                context: format!("account directory register-self: {e}"),
            })
    }
}
