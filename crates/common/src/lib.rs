//! Shared types for meeting-app spikes.
//!
//! Types here are tentative. Spike findings will refine them before any
//! migration into application crates.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Segment {
    pub start_ms: u64,
    pub end_ms: u64,
    pub text: String,
    pub speaker_id: Option<String>,
}

#[derive(Debug, Clone)]
pub struct AudioChunk {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub start_ms: u64,
}
