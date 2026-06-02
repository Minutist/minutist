//! The process-wide shared `LlamaBackend` singleton.
//!
//! `LlamaBackend::init()` is GLOBAL — llama.cpp permits it once per process;
//! a second call returns an error. Both `asr-runtime` and `summariser` load
//! GGUF models, and in the assembled app both run in the SAME process (record
//! then summarise). If each crate owned its own `OnceLock<LlamaBackend>`, the
//! second crate to initialise would hit the global already-init error while its
//! own cell stayed empty, and fail — silently breaking record-then-summarise.
//! They MUST funnel through this single cell.

use std::sync::{Mutex, OnceLock};

use llama_cpp_2::llama_backend::LlamaBackend;

use crate::{AppError, AppResult};

static BACKEND: OnceLock<LlamaBackend> = OnceLock::new();

/// Return the process-wide [`LlamaBackend`], initialising it exactly once.
///
/// Safe to call from any crate or thread; the global `LlamaBackend::init()`
/// runs once and every caller shares the returned `&'static` reference.
pub fn shared_llama_backend() -> AppResult<&'static LlamaBackend> {
    if let Some(backend) = BACKEND.get() {
        return Ok(backend);
    }
    // Double-checked locking: serialise the one-time init so two racing callers
    // don't both call the global `init()` (the loser would get an error).
    // Stable Rust has no fallible `OnceLock::get_or_try_init`, hence the lock.
    static INIT: Mutex<()> = Mutex::new(());
    let _guard = INIT.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some(backend) = BACKEND.get() {
        return Ok(backend);
    }
    let backend = LlamaBackend::init().map_err(|e| AppError::ModelLoad {
        model_id: "llama-backend".to_string(),
        context: format!("llama.cpp backend init failed: {e}"),
    })?;
    let _ = BACKEND.set(backend);
    Ok(BACKEND.get().expect("backend was just set under the init lock"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_backend_is_a_singleton() {
        // Two calls in one process must succeed and yield the SAME instance —
        // this is the exact invariant the old per-crate OnceLocks violated
        // (the second init() would fail). Backend init is CPU-side and needs no
        // model or GPU.
        let first = shared_llama_backend().expect("first init must succeed");
        let second = shared_llama_backend().expect("second call must reuse, not re-init");
        assert!(
            std::ptr::eq(first, second),
            "shared_llama_backend must return one process-wide instance"
        );
    }
}
