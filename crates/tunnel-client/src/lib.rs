//! App-side account client for the connected tier.
//!
//! Two HTTP clients against the account-service, both bearer/device-credential
//! authed against `mdc_`-prefixed device credentials:
//!
//! - [`DeviceCodeClient`] (in `pairing`) is the app-side RFC 8628 device-code
//!   client used to sign a device in and obtain its `mdc_` credential.
//! - [`AccountDirectoryClient`] (in `account`) is the account device-directory
//!   client (`GET /v1/account/devices`, `PUT
//!   /v1/account/devices/self/endpoint`, `DELETE /v1/account`), used once
//!   signed in to publish this device's sync endpoint and discover its peers.
//!
//! Both live here (not in `sync`) so `sync` keeps no HTTP-client edge; the
//! caller (`app-main` / `headless`, via the `account-directory` adapter crate)
//! adapts [`DeviceEndpointEntry`] onto `sync::AccountEndpointSource`.
//!
//! # Injection, not coupling
//!
//! The crate takes no workspace edge — it is a near-leaf consumer of
//! third-party crates only. `headless`'s edge to it is unconditional (a seeded
//! hub is always account-capable); `app-main`'s is gated by the `connected`
//! Cargo feature (the free build has no account/sync tier).

mod account;
mod pairing;

pub use account::{AccountDirectoryClient, AccountDirectoryError, DeviceEndpointEntry};
pub use pairing::{
    next_interval, DeviceCodeClient, IssuedDeviceCredential, PairingError, PairingStart,
    PollOutcome, MIN_POLL_INTERVAL_SECS, SLOW_DOWN_INCREMENT_SECS,
};
