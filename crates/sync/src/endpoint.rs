//! The iroh endpoint that owns device-to-device sync.
//!
//! [`SyncEngine`] holds a single [`iroh::Endpoint`] bound with:
//!
//! - `RelayMode::Custom` pinning the self-hosted relay ([`SyncConfig::relay_url`]),
//! - the device [`DeviceIdentity`]'s secret key,
//! - a `MemoryLookup` for out-of-band peer addressing (peers learned from the
//!   account service rather than DNS/pkarr discovery),
//! - the notes ([`crate::notes_proto::SYNC_ALPN`]) and blobs
//!   (`iroh_blobs::ALPN`) protocols multiplexed on one endpoint via the iroh
//!   `Router`.
//!
//! No network activity happens at construction beyond the bind; dialling a peer
//! and the protocol exchanges are S2/S3/S4.

use iroh::{Endpoint, RelayMode, RelayUrl};

use crate::identity::DeviceIdentity;
use crate::{Error, Result, SyncConfig};

/// Owns the iroh endpoint and the registered sync protocols for one device.
pub struct SyncEngine {
    #[allow(dead_code)] // TODO(S2): drives connect()/accept() once protocols exist.
    endpoint: Endpoint,
    config: SyncConfig,
}

impl SyncEngine {
    /// Build the endpoint from `config` and the device `identity`, pinning the
    /// configured relay and registering out-of-band addressing.
    ///
    /// TODO(S2): build the real endpoint —
    /// `Endpoint::builder(presets::N0).secret_key(identity).relay_mode(RelayMode::custom([relay]))
    /// .alpns(vec![SYNC_ALPN.into(), iroh_blobs::ALPN.into()]).address_lookup(memory_lookup).bind()`
    /// — then spawn the `Router` accepting both ALPNs. Stubbed so the crate
    /// compiles without binding a socket; [`Self::parse_relay_url`] is already
    /// exercised to keep the iroh types load-bearing.
    pub async fn start(config: SyncConfig, _identity: DeviceIdentity) -> Result<Self> {
        let _relay = Self::parse_relay_url(&config.relay_url)?;
        Err(Error::Endpoint(
            "SyncEngine::start not implemented (S2)".into(),
        ))
    }

    /// Parse the configured relay URL and wrap it as a single-relay
    /// `RelayMode::Custom`. Pure helper, used at bind time (S2).
    fn parse_relay_url(url: &str) -> Result<RelayMode> {
        let relay: RelayUrl = url
            .parse()
            .map_err(|e| Error::Endpoint(format!("invalid relay url {url:?}: {e}")))?;
        Ok(RelayMode::Custom([relay].into_iter().collect()))
    }

    /// The relay URL this engine is configured to pin.
    pub fn relay_url(&self) -> &str {
        &self.config.relay_url
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relay_url_parses_into_custom_mode() {
        let mode = SyncEngine::parse_relay_url(SyncConfig::DEFAULT_RELAY_URL)
            .expect("default relay url must parse");
        assert!(matches!(mode, RelayMode::Custom(_)));
    }

    #[test]
    fn empty_relay_url_is_rejected() {
        assert!(SyncEngine::parse_relay_url("").is_err());
    }
}
