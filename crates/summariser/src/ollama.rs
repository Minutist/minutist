//! Optional external-LLM dispatcher (FR-29), gated behind the
//! `external-ollama` cargo feature.
//!
//! Dispatches summarisation to a local Ollama / LM-Studio-compatible server
//! over its OpenAI-style `/api/chat` endpoint instead of loading a GGUF
//! in-process. Selected by settings when the user prefers an external model;
//! the bundled [`LlamaSummariser`](crate::LlamaSummariser) remains the default.
//!
//! Uses `reqwest`'s **blocking** client so the synchronous
//! [`Summariser`](meeting_app_common::Summariser) trait is satisfied without an
//! async runtime — mirroring the "called from `spawn_blocking`" contract.

use std::time::Duration;

use meeting_app_common::{AppResult, Segment, Summariser};
use serde::{Deserialize, Serialize};

use crate::{render_user_content, strip_think_block, Error};

/// Summariser backed by an external Ollama-compatible chat endpoint.
pub struct OllamaSummariser {
    /// Base URL of the server, e.g. `http://127.0.0.1:11434`.
    base_url: String,
    /// Model name as registered on the server, e.g. `gemma3:4b`.
    model: String,
    client: reqwest::blocking::Client,
}

impl OllamaSummariser {
    /// Construct a dispatcher targeting `base_url` and `model`.
    ///
    /// # Errors
    ///
    /// Returns `AppError::Inference` if the HTTP client cannot be built.
    pub fn new(base_url: impl Into<String>, model: impl Into<String>) -> AppResult<Self> {
        let client = reqwest::blocking::Client::builder()
            // External LLMs on a long transcript can take minutes; size the
            // timeout generously rather than failing a slow-but-valid run.
            .timeout(Duration::from_secs(600))
            .build()
            .map_err(|e| Error::Inference(format!("ollama client build: {e}")))?;
        Ok(Self {
            base_url: base_url.into(),
            model: model.into(),
            client,
        })
    }
}

#[derive(Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: Vec<ChatMessage<'a>>,
    /// Single-shot, no token streaming — we want the whole summary back.
    stream: bool,
}

#[derive(Serialize)]
struct ChatMessage<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Deserialize)]
struct ChatResponse {
    message: ChatResponseMessage,
}

#[derive(Deserialize)]
struct ChatResponseMessage {
    content: String,
}

/// Assemble the `/api/chat` endpoint URL from a base URL, tolerating a trailing
/// slash so `http://host:11434` and `http://host:11434/` both yield the same
/// endpoint. Pure (no I/O) so the normalisation can be unit-tested.
fn chat_url(base_url: &str) -> String {
    format!("{}/api/chat", base_url.trim_end_matches('/'))
}

/// Build the system + user chat request body. Pure (no I/O) so the serde shape
/// (roles/content, `stream: false`) can be asserted without a live server.
fn build_chat_request<'a>(
    model: &'a str,
    system_prompt: &'a str,
    user_content: &'a str,
) -> ChatRequest<'a> {
    ChatRequest {
        model,
        messages: vec![
            ChatMessage {
                role: "system",
                content: system_prompt,
            },
            ChatMessage {
                role: "user",
                content: user_content,
            },
        ],
        stream: false,
    }
}

/// Map a non-2xx HTTP status to the crate `Error::Inference` carried for the
/// ollama backend. Pure so the status→error mapping can be unit-tested without
/// issuing a request.
fn inference_error_for_status(status: reqwest::StatusCode) -> Error {
    Error::Inference(format!("ollama returned HTTP {status}"))
}

impl Summariser for OllamaSummariser {
    fn summarise(
        &self,
        transcript: &[Segment],
        notes_markdown: &str,
        system_prompt: &str,
    ) -> AppResult<String> {
        let user_content = render_user_content(transcript, notes_markdown);
        let body = build_chat_request(&self.model, system_prompt, &user_content);

        let url = chat_url(&self.base_url);
        let resp = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .map_err(|e| Error::Inference(format!("ollama request: {e}")))?;

        if !resp.status().is_success() {
            return Err(inference_error_for_status(resp.status()).into());
        }

        let parsed: ChatResponse = resp
            .json()
            .map_err(|e| Error::Inference(format!("ollama response decode: {e}")))?;

        Ok(strip_think_block(&parsed.message.content))
    }
}

// ---------------------------------------------------------------------------
// Tests — the pure URL/serde/status-mapping seams (no live server). The only
// untested line in `summarise` is the `reqwest` `send()` itself.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use meeting_app_common::AppError;

    /// A base URL without a trailing slash yields the `/api/chat` endpoint.
    #[test]
    fn chat_url_appends_endpoint_without_trailing_slash() {
        assert_eq!(
            chat_url("http://127.0.0.1:11434"),
            "http://127.0.0.1:11434/api/chat"
        );
    }

    /// A base URL with a trailing slash is normalised (no doubled slash).
    #[test]
    fn chat_url_normalises_trailing_slash() {
        assert_eq!(
            chat_url("http://127.0.0.1:11434/"),
            "http://127.0.0.1:11434/api/chat"
        );
    }

    /// The request body serialises to the expected JSON: model, the two
    /// system/user messages with roles + content, and `stream: false`.
    #[test]
    fn chat_request_serialises_to_expected_json() {
        let req = build_chat_request("gemma3:4b", "you are an assistant", "the transcript");
        let value = serde_json::to_value(&req).expect("serialise request");

        let expected = serde_json::json!({
            "model": "gemma3:4b",
            "messages": [
                { "role": "system", "content": "you are an assistant" },
                { "role": "user", "content": "the transcript" }
            ],
            "stream": false
        });
        assert_eq!(value, expected);
    }

    /// A canned `/api/chat` response JSON deserialises to the message content.
    #[test]
    fn chat_response_deserialises_message_content() {
        let json = "{\"message\":{\"role\":\"assistant\",\"content\":\"## Summary\\n\\n- a point\"}}";
        let parsed: ChatResponse = serde_json::from_str(json).expect("deserialise response");
        assert_eq!(parsed.message.content, "## Summary\n\n- a point");
    }

    /// A non-2xx HTTP status maps to `Error::Inference`, which converts to
    /// `AppError::Inference` carrying the `"summariser"` backend label.
    #[test]
    fn non_2xx_status_maps_to_inference_error() {
        let err = inference_error_for_status(reqwest::StatusCode::INTERNAL_SERVER_ERROR);
        match &err {
            Error::Inference(context) => assert!(
                context.contains("500"),
                "context must carry the status: {context}"
            ),
            other => panic!("expected Error::Inference, got {other:?}"),
        }

        let app: AppError = err.into();
        match app {
            AppError::Inference { backend, context } => {
                assert_eq!(backend, "summariser", "backend label must be set");
                assert!(context.contains("500"));
            }
            other => panic!("expected AppError::Inference, got {other:?}"),
        }
    }
}
