//! `AppError -> McpError` mapping (Phase 10 §2.2).
//!
//! This is the ONE place `rmcp` error types are constructed from the shared
//! [`minutist_common::AppError`]. The `agent-tools` registry returns
//! `AppError`; the MCP `tools/call` handler maps it to a JSON-RPC / MCP
//! [`rmcp::ErrorData`] per the Phase 10 mapping table.
//!
//! Real variants only — there is **no** `ContextOverflow`. Overflow and
//! "recorder busy" both surface as `AppError::InvalidInput`, which maps to
//! `-32602 invalid params` (the natural "your request could not be served as
//! given" code).

use minutist_common::AppError;
use rmcp::ErrorData as McpError;

/// Map an [`AppError`] to an MCP [`McpError`] (`rmcp::ErrorData`).
///
/// | `AppError` | MCP error |
/// |---|---|
/// | `InvalidInput { context }` | `-32602` invalid params, message = context |
/// | `ModelNotFound`/`ModelLoad`/`ModelDownload`/`Inference` | `-32603` internal error |
/// | `Cancelled` | `-32603` internal error ("operation cancelled") |
/// | `Unsupported { context }` | `-32603` internal error ("unsupported: …") |
/// | `Io`/`Internal` | `-32603` internal error |
///
/// `InvalidInput` is the only one that becomes `invalid_params`; everything
/// else is `internal_error` (the result of a server-side operation failing,
/// not a malformed request). The `Display` of each variant is used verbatim as
/// the message so the cause is preserved for the caller's logs.
pub fn app_error_to_mcp(err: &AppError) -> McpError {
    match err {
        AppError::InvalidInput { context } => McpError::invalid_params(context.clone(), None),
        // Unsupported is closest to "the server can't do this" — surface as an
        // internal error with the context (method-not-found would imply the MCP
        // *method* is unknown, which is not the case: the tool exists, the
        // operation it maps to is unsupported).
        AppError::Unsupported { context } => {
            McpError::internal_error(format!("unsupported: {context}"), None)
        }
        // Everything else is a server-side failure → internal error. Use the
        // `Display` impl so the structured cause (model id, backend, context) is
        // preserved in the message.
        other => McpError::internal_error(other.to_string(), None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rmcp::model::ErrorCode;

    #[test]
    fn invalid_input_maps_to_invalid_params() {
        let e = app_error_to_mcp(&AppError::InvalidInput {
            context: "bad meeting id".into(),
        });
        assert_eq!(e.code, ErrorCode::INVALID_PARAMS);
        assert!(e.message.contains("bad meeting id"));
    }

    #[test]
    fn model_errors_map_to_internal_error() {
        for e in [
            AppError::ModelNotFound {
                model_id: "m".into(),
            },
            AppError::ModelLoad {
                model_id: "m".into(),
                context: "c".into(),
            },
            AppError::ModelDownload {
                context: "c".into(),
            },
            AppError::Inference {
                backend: "b".into(),
                context: "c".into(),
            },
        ] {
            let mapped = app_error_to_mcp(&e);
            assert_eq!(
                mapped.code,
                ErrorCode::INTERNAL_ERROR,
                "{e:?} must map to internal error"
            );
        }
    }

    #[test]
    fn cancelled_maps_to_internal_error() {
        let e = app_error_to_mcp(&AppError::Cancelled);
        assert_eq!(e.code, ErrorCode::INTERNAL_ERROR);
        assert!(e.message.to_lowercase().contains("cancelled"));
    }

    #[test]
    fn unsupported_maps_to_internal_error_with_prefix() {
        let e = app_error_to_mcp(&AppError::Unsupported {
            context: "reprocess over MCP".into(),
        });
        assert_eq!(e.code, ErrorCode::INTERNAL_ERROR);
        assert!(e.message.contains("unsupported"));
        assert!(e.message.contains("reprocess over MCP"));
    }

    #[test]
    fn io_and_internal_map_to_internal_error() {
        for e in [
            AppError::Io {
                context: "disk".into(),
            },
            AppError::Internal {
                context: "bug".into(),
            },
        ] {
            let mapped = app_error_to_mcp(&e);
            assert_eq!(mapped.code, ErrorCode::INTERNAL_ERROR);
        }
    }
}
