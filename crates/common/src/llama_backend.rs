//! The process-wide shared `LlamaBackend` singleton.
//!
//! `LlamaBackend::init()` is GLOBAL — llama.cpp permits it once per process;
//! a second call returns an error. `asr-runtime`, `summariser`, and `embedder`
//! all load GGUF models, and in the assembled app they run in the SAME process
//! (record, then summarise + embed for retrieval). If each crate owned its own
//! `OnceLock<LlamaBackend>`, the
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
    Ok(BACKEND
        .get()
        .expect("backend was just set under the init lock"))
}

/// Serialise the FIRST model load in this process, then run every later one
/// unguarded.
///
/// `asr-runtime` and `summariser` each load their own GGUF model, and the app
/// spawns their prewarm/preload independently at startup
/// (`src-tauri/src/main.rs`'s `setup()`), so two loads can race on a fresh
/// process. On the Vulkan backend that race is live, not theoretical:
/// `ggml_vk_get_device()` (vendored `llama.cpp/ggml/src/ggml-vulkan/
/// ggml-vulkan.cpp`) publishes a device's `shared_ptr` into the shared
/// `vk_instance.devices[idx]` slot *before* the object's fields (buffer-type
/// dispatch table included) are fully populated, with no lock — a second
/// thread reading that slot mid-populate can observe a null function pointer
/// and crash (`EXCEPTION_ACCESS_VIOLATION_EXEC`, seen live on an AMD Radeon
/// 890M iGPU; see `planning/research/vulkan-igpu-crash-investigation-2026-08-22.md`).
/// The sibling lazy-init `ggml_backend_vk_reg_get_device()` in the same file
/// already takes a mutex for its analogous one-time init, so this looks like
/// an upstream oversight rather than deliberate design — not something we can
/// fix by patching around it cheaply, but we CAN stop triggering the race:
/// once any one load has completed (crossed the racy window), every later
/// `ggml_vk_get_device()` call just reads the now-fully-populated slot, no
/// matter how concurrent. Only the first-ever call needs to be alone.
pub fn serialize_first_model_load<T>(load: impl FnOnce() -> T) -> T {
    static DONE: OnceLock<()> = OnceLock::new();
    if DONE.get().is_some() {
        return load();
    }
    static LOCK: Mutex<()> = Mutex::new(());
    let _guard = LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    if DONE.get().is_some() {
        // Another thread took the lock first and already finished.
        drop(_guard);
        return load();
    }
    let result = load();
    let _ = DONE.set(());
    result
}

/// List the ggml backend devices (name/type/memory), or an empty vec when the
/// GPU backend is not compiled in. The crate-internal entry point behind
/// [`crate::probe_primary_gpu`]; kept here so the `llama_cpp_2` use stays inside
/// the `llama-backend`-gated module.
pub(crate) fn list_gpu_devices() -> Vec<crate::GpuProbe> {
    use llama_cpp_2::LlamaBackendDeviceType as T;
    // Backend init registers the GPU devices; without it the list is empty.
    if shared_llama_backend().is_err() {
        return Vec::new();
    }
    llama_cpp_2::list_llama_ggml_backend_devices()
        .into_iter()
        .filter_map(|d| {
            let is_gpu = matches!(d.device_type, T::Gpu | T::IntegratedGpu);
            is_gpu.then(|| crate::GpuProbe {
                total_bytes: d.memory_total as u64,
                free_bytes: d.memory_free as u64,
                is_integrated: d.device_type == T::IntegratedGpu,
                name: if d.description.is_empty() {
                    d.name
                } else {
                    d.description
                },
            })
        })
        .collect()
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
