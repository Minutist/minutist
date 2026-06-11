//! Manifest file loading — `resources/models.json`.

use serde::Deserialize;

use minutist_common::ModelManifestEntry;

use crate::error::Error;

/// Top-level shape of `resources/models.json`.
#[derive(Debug, Deserialize)]
pub struct ManifestFile {
    pub version: u32,
    pub models: Vec<ModelManifestEntry>,
}

/// Parse the manifest from a JSON byte slice.
pub fn parse_manifest(bytes: &[u8]) -> Result<Vec<ModelManifestEntry>, Error> {
    let mf: ManifestFile = serde_json::from_slice(bytes)?;
    Ok(mf.models)
}
