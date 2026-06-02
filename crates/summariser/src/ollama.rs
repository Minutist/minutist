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

impl Summariser for OllamaSummariser {
    fn summarise(
        &self,
        transcript: &[Segment],
        notes_markdown: &str,
        system_prompt: &str,
    ) -> AppResult<String> {
        let user_content = render_user_content(transcript, notes_markdown);
        let body = ChatRequest {
            model: &self.model,
            messages: vec![
                ChatMessage {
                    role: "system",
                    content: system_prompt,
                },
                ChatMessage {
                    role: "user",
                    content: &user_content,
                },
            ],
            stream: false,
        };

        let url = format!("{}/api/chat", self.base_url.trim_end_matches('/'));
        let resp = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .map_err(|e| Error::Inference(format!("ollama request: {e}")))?;

        if !resp.status().is_success() {
            return Err(Error::Inference(format!(
                "ollama returned HTTP {}",
                resp.status()
            ))
            .into());
        }

        let parsed: ChatResponse = resp
            .json()
            .map_err(|e| Error::Inference(format!("ollama response decode: {e}")))?;

        Ok(strip_think_block(&parsed.message.content))
    }
}
