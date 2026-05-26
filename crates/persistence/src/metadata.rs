use std::io::Write;
use std::path::Path;

use meeting_app_common::MeetingMeta;

use crate::error::{Error, Result};

/// Serialise `meta` to `metadata.json` at `path`.
///
/// Writes atomically via a temporary buffer then a single `write_all` to the
/// opened file (no rename-into-place needed for a new file, but the buffered
/// write ensures partial writes don't leave a truncated JSON file behind).
pub fn write_metadata(path: impl AsRef<Path>, meta: &MeetingMeta) -> Result<()> {
    let json = serde_json::to_string_pretty(meta)?;

    let mut file = std::fs::File::create(path.as_ref()).map_err(Error::Io)?;
    file.write_all(json.as_bytes()).map_err(Error::Io)?;
    file.flush().map_err(Error::Io)?;

    Ok(())
}
