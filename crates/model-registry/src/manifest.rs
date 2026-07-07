//! Manifest file loading — `resources/models.json`.

use serde::Deserialize;

use minutist_common::ModelManifestEntry;

use crate::error::Error;

/// The only `version` value [`parse_manifest`] accepts. Bump this — and
/// migrate `resources/models.json` and any manifest-shape assumptions in this
/// crate — together whenever the manifest schema changes; a manifest built
/// for a different schema must fail loudly at startup rather than be parsed
/// under the wrong assumptions.
pub const SUPPORTED_MANIFEST_VERSION: u32 = 1;

/// Top-level shape of `resources/models.json`.
#[derive(Debug, Deserialize)]
pub struct ManifestFile {
    pub version: u32,
    pub models: Vec<ModelManifestEntry>,
}

/// Parse the manifest from a JSON byte slice, rejecting one whose `version`
/// isn't [`SUPPORTED_MANIFEST_VERSION`].
pub fn parse_manifest(bytes: &[u8]) -> Result<Vec<ModelManifestEntry>, Error> {
    let mf: ManifestFile = serde_json::from_slice(bytes)?;
    if mf.version != SUPPORTED_MANIFEST_VERSION {
        return Err(Error::UnsupportedManifestVersion {
            found: mf.version,
            supported: SUPPORTED_MANIFEST_VERSION,
        });
    }
    Ok(mf.models)
}
