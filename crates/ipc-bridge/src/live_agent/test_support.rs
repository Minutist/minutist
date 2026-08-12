//! Stub backends for unit tests that exercise the full worker loop without a
//! real model. Only compiled in `#[cfg(test)]`. Production code always uses
//! `LlamaLiveBackend`.

use super::*;
use chat_agent::{
    CancelFlag, ConversationalTurn, Error as ChatError, LiveSessionBackend, RawTurn,
    SamplerConfig,
};

/// A stub that always returns a short non-NOOP reply and counts `prefill_prefix`
/// calls. Used to verify the single-seed guarantee.
pub(crate) struct WorkerBackend {
    pub(crate) prefill_counter: Arc<std::sync::atomic::AtomicU32>,
}

impl WorkerBackend {
    pub(crate) fn new() -> Self {
        Self {
            prefill_counter: Arc::new(std::sync::atomic::AtomicU32::new(0)),
        }
    }

    pub(crate) fn prefill_counter(&self) -> Arc<std::sync::atomic::AtomicU32> {
        Arc::clone(&self.prefill_counter)
    }
}

impl LiveSessionBackend for WorkerBackend {
    fn prefill_prefix(
        &mut self,
        _prefix_text: &str,
        _cancel: &CancelFlag,
    ) -> Result<usize, ChatError> {
        self.prefill_counter
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(0)
    }

    fn refresh(
        &mut self,
        _tail_text: &str,
        _cfg: &SamplerConfig,
        _cancel: &CancelFlag,
        _token_cb: &mut dyn FnMut(&str),
    ) -> Result<RawTurn, ChatError> {
        Ok(RawTurn {
            text: "stub reply".to_string(),
            tool_calls: Vec::new(),
            cancelled: false,
        })
    }

    fn reset_to_prefix(&mut self) -> Result<(), ChatError> {
        Ok(())
    }

    fn has_room_for(&self, _estimated_tokens: usize, _max_gen: usize) -> bool {
        true
    }

    fn n_past(&self) -> i32 {
        0
    }
}

impl ConversationalTurn for WorkerBackend {
    fn converse(
        &mut self,
        _role: &str,
        _content: &str,
        _cfg: &SamplerConfig,
        cancel: &CancelFlag,
        _token_cb: &mut dyn FnMut(&str),
    ) -> Result<RawTurn, ChatError> {
        if cancel.is_cancelled() {
            return Ok(RawTurn {
                text: String::new(),
                tool_calls: Vec::new(),
                cancelled: true,
            });
        }
        Ok(RawTurn {
            text: "stub reply".to_string(),
            tool_calls: Vec::new(),
            cancelled: false,
        })
    }
}

/// A stub backend whose `converse` returns `Error::ContextOverflow`, for
/// testing the overflow classification path.
pub(crate) struct OverflowBackend;

impl LiveSessionBackend for OverflowBackend {
    fn prefill_prefix(
        &mut self,
        _prefix_text: &str,
        _cancel: &CancelFlag,
    ) -> Result<usize, ChatError> {
        Ok(0)
    }

    fn refresh(
        &mut self,
        _tail_text: &str,
        _cfg: &SamplerConfig,
        _cancel: &CancelFlag,
        _token_cb: &mut dyn FnMut(&str),
    ) -> Result<RawTurn, ChatError> {
        Err(ChatError::ContextOverflow(
            "stub: n_past=30000 would exceed n_ctx=32768".to_string(),
        ))
    }

    fn reset_to_prefix(&mut self) -> Result<(), ChatError> {
        Ok(())
    }

    fn has_room_for(&self, _estimated_tokens: usize, _max_gen: usize) -> bool {
        false
    }

    fn n_past(&self) -> i32 {
        30_000
    }
}

impl ConversationalTurn for OverflowBackend {
    fn converse(
        &mut self,
        _role: &str,
        _content: &str,
        _cfg: &SamplerConfig,
        _cancel: &CancelFlag,
        _token_cb: &mut dyn FnMut(&str),
    ) -> Result<RawTurn, ChatError> {
        Err(ChatError::ContextOverflow(
            "stub: n_past=30000 would exceed n_ctx=32768".to_string(),
        ))
    }
}

/// A stub backend that records the content strings it is asked to decode
/// (so a test can assert what reached the model) and returns a short
/// non-NOOP reply.
pub(crate) struct CapturingBackend {
    pub(crate) tails: Arc<std::sync::Mutex<Vec<String>>>,
}

impl CapturingBackend {
    pub(crate) fn new() -> Self {
        Self {
            tails: Arc::new(std::sync::Mutex::new(Vec::new())),
        }
    }

    pub(crate) fn tails(&self) -> Arc<std::sync::Mutex<Vec<String>>> {
        Arc::clone(&self.tails)
    }
}

impl LiveSessionBackend for CapturingBackend {
    fn prefill_prefix(
        &mut self,
        _prefix_text: &str,
        _cancel: &CancelFlag,
    ) -> Result<usize, ChatError> {
        Ok(0)
    }

    fn refresh(
        &mut self,
        tail_text: &str,
        _cfg: &SamplerConfig,
        _cancel: &CancelFlag,
        _token_cb: &mut dyn FnMut(&str),
    ) -> Result<RawTurn, ChatError> {
        self.tails.lock().unwrap().push(tail_text.to_string());
        Ok(RawTurn {
            text: "stub reply".to_string(),
            tool_calls: Vec::new(),
            cancelled: false,
        })
    }

    fn reset_to_prefix(&mut self) -> Result<(), ChatError> {
        Ok(())
    }

    fn has_room_for(&self, _estimated_tokens: usize, _max_gen: usize) -> bool {
        true
    }

    fn n_past(&self) -> i32 {
        0
    }
}

impl ConversationalTurn for CapturingBackend {
    fn converse(
        &mut self,
        _role: &str,
        content: &str,
        _cfg: &SamplerConfig,
        _cancel: &CancelFlag,
        _token_cb: &mut dyn FnMut(&str),
    ) -> Result<RawTurn, ChatError> {
        self.tails.lock().unwrap().push(content.to_string());
        Ok(RawTurn {
            text: "stub reply".to_string(),
            tool_calls: Vec::new(),
            cancelled: false,
        })
    }
}

/// A stub whose `converse` returns the NOOP sentinel — for testing transcript
/// suppression.
pub(crate) struct NoopBackend;

impl LiveSessionBackend for NoopBackend {
    fn prefill_prefix(
        &mut self,
        _prefix_text: &str,
        _cancel: &CancelFlag,
    ) -> Result<usize, ChatError> {
        Ok(0)
    }

    fn refresh(
        &mut self,
        _tail_text: &str,
        _cfg: &SamplerConfig,
        _cancel: &CancelFlag,
        _token_cb: &mut dyn FnMut(&str),
    ) -> Result<RawTurn, ChatError> {
        Ok(RawTurn {
            text: COPILOT_NOOP_SENTINEL.to_string(),
            tool_calls: Vec::new(),
            cancelled: false,
        })
    }

    fn reset_to_prefix(&mut self) -> Result<(), ChatError> {
        Ok(())
    }

    fn has_room_for(&self, _estimated_tokens: usize, _max_gen: usize) -> bool {
        true
    }

    fn n_past(&self) -> i32 {
        0
    }
}

impl ConversationalTurn for NoopBackend {
    fn converse(
        &mut self,
        _role: &str,
        _content: &str,
        _cfg: &SamplerConfig,
        _cancel: &CancelFlag,
        _token_cb: &mut dyn FnMut(&str),
    ) -> Result<RawTurn, ChatError> {
        Ok(RawTurn {
            text: COPILOT_NOOP_SENTINEL.to_string(),
            tool_calls: Vec::new(),
            cancelled: false,
        })
    }
}

/// A stub backend that simulates a nearly-full context. It reports
/// `has_room_for = false` until `reset_to_prefix` is called, after which
/// it returns `true`. The `reset_counter` tracks how many times
/// `reset_to_prefix` has been called. `converse` records its content so
/// tests can inspect whether the recap header was prepended.
pub(crate) struct NearFullBackend {
    pub(crate) reset_counter: Arc<std::sync::atomic::AtomicU32>,
    pub(crate) converse_calls: Arc<std::sync::Mutex<Vec<String>>>,
    was_reset: bool,
}

impl NearFullBackend {
    pub(crate) fn new() -> Self {
        Self {
            reset_counter: Arc::new(std::sync::atomic::AtomicU32::new(0)),
            converse_calls: Arc::new(std::sync::Mutex::new(Vec::new())),
            was_reset: false,
        }
    }

    pub(crate) fn reset_counter(&self) -> Arc<std::sync::atomic::AtomicU32> {
        Arc::clone(&self.reset_counter)
    }

    pub(crate) fn converse_calls(&self) -> Arc<std::sync::Mutex<Vec<String>>> {
        Arc::clone(&self.converse_calls)
    }
}

impl LiveSessionBackend for NearFullBackend {
    fn prefill_prefix(
        &mut self,
        _prefix_text: &str,
        _cancel: &CancelFlag,
    ) -> Result<usize, ChatError> {
        Ok(10)
    }

    fn refresh(
        &mut self,
        _tail_text: &str,
        _cfg: &SamplerConfig,
        _cancel: &CancelFlag,
        _token_cb: &mut dyn FnMut(&str),
    ) -> Result<RawTurn, ChatError> {
        Ok(RawTurn::default())
    }

    fn reset_to_prefix(&mut self) -> Result<(), ChatError> {
        self.reset_counter
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.was_reset = true;
        Ok(())
    }

    fn has_room_for(&self, _estimated_tokens: usize, _max_gen: usize) -> bool {
        // Simulate a full context until the first reset.
        self.was_reset
    }

    fn n_past(&self) -> i32 {
        if self.was_reset { 10 } else { 30_000 }
    }
}

impl ConversationalTurn for NearFullBackend {
    fn converse(
        &mut self,
        _role: &str,
        content: &str,
        _cfg: &SamplerConfig,
        _cancel: &CancelFlag,
        _token_cb: &mut dyn FnMut(&str),
    ) -> Result<RawTurn, ChatError> {
        self.converse_calls.lock().unwrap().push(content.to_string());
        Ok(RawTurn {
            text: "eviction reply".to_string(),
            tool_calls: Vec::new(),
            cancelled: false,
        })
    }
}
