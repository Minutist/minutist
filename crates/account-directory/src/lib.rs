//! Adapts `tunnel_client::AccountDirectoryClient` (an HTTP client) onto
//! `sync::AccountEndpointSource` (the trait `sync` depends on instead of an
//! HTTP client), so `sync` and `tunnel-client` never take an edge on each
//! other. `app-main` and `headless` — the two account-mediated-discovery
//! consumers (B4) — both depend on this crate instead of each carrying their
//! own copy of the adapter.

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
            })
            .collect())
    }

    async fn register_self(&self, endpoint: &AccountEndpoint) -> AppResult<()> {
        self.client
            .register_self_endpoint(
                &endpoint.endpoint_id,
                &endpoint.relay_url,
                &endpoint.direct_addrs,
            )
            .await
            .map_err(|e| AppError::Internal {
                context: format!("account directory register-self: {e}"),
            })
    }
}
