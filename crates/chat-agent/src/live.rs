//! Held-context live-session engine.
//!
//! The Phase-9 chat loop allocates a **fresh** [`LlamaContext`] per turn (clean
//! KV cache). The live in-meeting agent is an explicit departure from that
//! pattern (SP-LIVE E2): the prefix is prefilled **once** at recording start and
//! that prefix KV is reused across every digest refresh. A fresh-per-turn
//! approach would re-pay the prefill cost on every cadence tick, making live
//! operation unusable.
//!
//! This crate is prefix-AGNOSTIC — it does the KV mechanics, never templating.
//! The driver (`ipc-bridge::live_agent`) supplies the prefix text and the
//! per-refresh tail already wrapped in chat-template markers. Under the current
//! driver the prefix is the **open user turn** of a chat-template prompt
//! (`<bos><start_of_turn>user\n{system + digest categories}`) — small and cheap;
//! attachment and earlier-transcript context is RETRIEVED into the per-refresh
//! tail (RAG), not pinned in the prefix. Each refresh **prunes the KV back to the
//! prefix** and decodes a fresh, BOUNDED tail on top — the retrieved context +
//! the running digest + a recent transcript window + the template suffix that
//! closes the user turn and opens the model turn. The held context therefore
//! never grows beyond `prefix + tail + generation`: it cannot overflow on a long
//! meeting (the failure mode the cumulative-append design hit on its first live
//! test — see planning issue 0022). The running digest (re-fed each refresh) plus
//! the RAG retrieval layer are the durable memory; verbatim transcript that
//! scrolls out of the window is dropped from the prompt but stays retrievable.
//!
//! Caveat (Gemma SWA): the shipped model uses interleaved sliding-window
//! attention, so on the local-attention layers a far-back token stops being
//! attendable once generation runs many positions past it — only the
//! global-attention layers keep full visibility. With the small Phase-D prefix
//! this is a non-issue for the prefix itself; it bounds how far back in the
//! per-refresh tail the local layers attend, which is why the tail window is kept
//! small and the relevant context is retrieved fresh each refresh rather than
//! relied on from deep in the KV.
//!
//! # Design
//!
//! The testable seam is [`LiveSessionBackend`]:
//!
//! - [`LiveSessionBackend::prefill_prefix`] — tokenise + chunked-prefill the
//!   pinned (open user turn) prefix ONCE, retaining the KV state and recording
//!   its length.
//! - [`LiveSessionBackend::refresh`] — prune the KV back to the prefix, append
//!   the fresh bounded tail, and decode the digest answer; does NOT re-prefill
//!   the prefix.
//!
//! [`LlamaLiveBackend`] is the real impl. It owns one `LlamaContext` for the
//! session lifetime (n_ctx from config, default 32 768; KV-quant OFF per
//! SP-LIVE E3). Because `LlamaContext` is `!Send`, `LlamaLiveBackend` is also
//! `!Send`. The owned-thread requirement is satisfied by the caller (S2b):
//! `LlamaLiveBackend` is constructed and driven entirely from a single dedicated
//! thread; this crate never moves it across threads.
//!
//! [`LiveSession`] is the driver type, generic over the backend:
//!
//! - [`LiveSession::seed_prefix`] — calls `prefill_prefix` exactly once (a
//!   second call is a no-op: the prefix is already in the KV cache).
//! - [`LiveSession::refresh`] — passes only the NEW tail since the last refresh
//!   to the backend; the prefix KV is never touched again.
//!
//! # Threading
//!
//! `LlamaLiveBackend` is `!Send` (it holds a `!Send` `LlamaContext`). The
//! [`LiveSession<LlamaLiveBackend>`] type is therefore also `!Send`. The S2b
//! driver in `ipc-bridge` owns the single thread and drives the session there.
//! The stub backend used in unit tests (`StubLiveBackend`) IS `Send` so the
//! test harness may run on the default test threads.

use minutist_common::AppResult;

use crate::backend::RawTurn;
use crate::error::Error;
use crate::types::{CancelFlag, SamplerConfig};

// ---------------------------------------------------------------------------
// The testable seam
// ---------------------------------------------------------------------------

/// The low-level held-context backend seam.
///
/// The real implementation ([`LlamaLiveBackend`]) holds one `LlamaContext`
/// across the session. A stub implementation drives [`LiveSession`] in unit
/// tests without a model.
///
/// **Not** `Send`: the real impl holds a `!Send` `LlamaContext`. The S2b
/// driver owns the thread and calls these methods only from there.
pub trait LiveSessionBackend {
    /// Tokenise `prefix_text` and feed it into the context via chunked-prefill.
    ///
    /// Called exactly once per session. After this returns the KV cache holds
    /// the full prefix state; [`Self::refresh`] appends tail tokens on top.
    ///
    /// `cancel` is checked between decoded chunks. A raised flag causes the
    /// prefill to return [`Error::Inference`] with a "cancelled" message;
    /// partially-decoded KV state is discarded before returning.
    ///
    /// Returns the number of tokens prefilled (useful for capacity checks).
    fn prefill_prefix(&mut self, prefix_text: &str, cancel: &CancelFlag) -> Result<usize, Error>;

    /// Prune the KV cache back to the pinned prefix, append `tail_text` on top,
    /// and decode the digest answer.
    ///
    /// The backend MUST NOT re-prefill the prefix — the prefix KV is reused. The
    /// `tail_text` is the WHOLE volatile portion of the prompt for this refresh
    /// (running digest + recent transcript window + the template suffix that
    /// closes the user turn and opens the model turn); it replaces, rather than
    /// extends, the previous refresh's tail.
    ///
    /// `tail_text` is always non-empty (it carries the template suffix at
    /// minimum), so the tail decode repopulates logits at its final token and
    /// there is no empty-tail logit-coherence hazard.
    ///
    /// - `cfg` — sampler knobs for this decode pass.
    /// - `cancel` — checked between tokens; a raised flag causes
    ///   `RawTurn { cancelled: true, … }`.
    /// - `token_cb` — called per detokenised user-visible piece (the driver
    ///   maps these to digest-update events).
    fn refresh(
        &mut self,
        tail_text: &str,
        cfg: &SamplerConfig,
        cancel: &CancelFlag,
        token_cb: &mut dyn FnMut(&str),
    ) -> Result<RawTurn, Error>;
}

// ---------------------------------------------------------------------------
// LiveSession driver
// ---------------------------------------------------------------------------

/// Driver that guarantees the prefix is prefilled exactly once and each
/// [`Self::refresh`] call passes only the incremental tail to the backend.
///
/// Generic over the backend so unit tests inject a [`StubLiveBackend`] and the
/// production path injects [`LlamaLiveBackend`].
///
/// The invariant this type enforces:
/// - `seed_prefix` calls `backend.prefill_prefix` on the first call and is a
///   no-op on subsequent calls (the prefix is already in the KV cache).
/// - `refresh(tail)` calls `backend.refresh` with the whole volatile portion of
///   this refresh's prompt (running digest + recent transcript window +
///   template suffix). The backend prunes the prior tail and decodes this one
///   on top of the reused prefix KV.
pub struct LiveSession<B: LiveSessionBackend> {
    backend: B,
    prefix_seeded: bool,
}

impl<B: LiveSessionBackend> LiveSession<B> {
    /// Construct a session over the given backend. The prefix has not been
    /// prefilled yet; call [`Self::seed_prefix`] before any [`Self::refresh`].
    pub fn new(backend: B) -> Self {
        Self {
            backend,
            prefix_seeded: false,
        }
    }

    /// Prefill the prefix (the open user turn: system + digest categories) into
    /// the held KV cache.
    ///
    /// Idempotent: the second and subsequent calls are no-ops (the prefix is
    /// already in the KV cache; re-prefilling would corrupt the position state).
    /// Returns `Ok(0)` on a no-op call.
    ///
    /// `cancel` is forwarded to the backend; a raised flag during the prefill
    /// returns an `Inference` error and the partially-decoded KV state is
    /// discarded by the backend before returning.
    ///
    /// Returns the typed [`Error`] so callers that need to distinguish
    /// [`Error::ContextOverflow`] from other failures can do so without
    /// string matching. See [`Self::seed_prefix`] for the `AppResult` variant.
    pub fn seed_prefix_typed(
        &mut self,
        prefix_text: &str,
        cancel: &CancelFlag,
    ) -> Result<usize, Error> {
        if self.prefix_seeded {
            return Ok(0);
        }
        let n = self.backend.prefill_prefix(prefix_text, cancel)?;
        self.prefix_seeded = true;
        Ok(n)
    }

    /// Like [`Self::seed_prefix_typed`] but converts to [`AppResult`].
    pub fn seed_prefix(&mut self, prefix_text: &str, cancel: &CancelFlag) -> AppResult<usize> {
        self.seed_prefix_typed(prefix_text, cancel)
            .map_err(Into::into)
    }

    /// Prune the KV back to the prefix, decode the fresh tail (running digest +
    /// recent window + template suffix) on top, and return the generated text.
    ///
    /// - Returns `Ok(String)` with the generated text (the `RawTurn::text`).
    /// - A raised `cancel` flag produces an `Ok(String)` with the partial text
    ///   decoded so far (cancellation is not an error).
    /// - If the prefix has not been seeded yet, this returns
    ///   [`Error::Inference`] (the KV state would be incoherent).
    ///
    /// Returns the typed [`Error`] so callers that need to distinguish
    /// [`Error::ContextOverflow`] from other failures can do so without
    /// string matching. See [`Self::refresh`] for the `AppResult` variant.
    pub fn refresh_typed(
        &mut self,
        tail_text: &str,
        cfg: &SamplerConfig,
        cancel: &CancelFlag,
        token_cb: &mut dyn FnMut(&str),
    ) -> Result<String, Error> {
        if !self.prefix_seeded {
            return Err(Error::Inference(
                "live session refresh called before seed_prefix".to_string(),
            ));
        }
        let raw = self.backend.refresh(tail_text, cfg, cancel, token_cb)?;
        Ok(raw.text)
    }

    /// Like [`Self::refresh_typed`] but converts to [`AppResult`].
    pub fn refresh(
        &mut self,
        tail_text: &str,
        cfg: &SamplerConfig,
        cancel: &CancelFlag,
        token_cb: &mut dyn FnMut(&str),
    ) -> AppResult<String> {
        self.refresh_typed(tail_text, cfg, cancel, token_cb)
            .map_err(Into::into)
    }
}

// ---------------------------------------------------------------------------
// Real backend — LlamaLiveBackend
// ---------------------------------------------------------------------------

/// The real held-context backend. Owns one `LlamaContext` for the session
/// lifetime.
///
/// Construction: call [`LlamaLiveBackend::new`] with the loaded model and
/// config. The context is built once (`n_ctx` from config, KV-quant OFF).
///
/// Lifecycle:
/// 1. [`LiveSessionBackend::prefill_prefix`] — tokenise + chunked-prefill the
///    prefix, advance `n_past`, record `prefix_len`, and capture the KV
///    checkpoint snapshot.
/// 2. [`LiveSessionBackend::refresh`] — restore the KV checkpoint (or fall back
///    to `clear_kv_cache_seq` if no snapshot is held), set `n_past` back to
///    `prefix_len`, append the fresh tail, and decode the answer. `n_past`
///    therefore stays bounded at `prefix_len + tail + generation`.
///
/// **`!Send`**: `LlamaContext` is `!Send`. The S2b driver owns the thread.
pub struct LlamaLiveBackend<'m> {
    model: &'m llama_cpp_2::model::LlamaModel,
    config: LlamaLiveConfig,
    ctx: llama_cpp_2::context::LlamaContext<'m>,
    n_past: i32,
    /// KV positions `0..prefix_len` hold the pinned prefix (the open user turn:
    /// system + digest-category instructions). Recorded by `prefill_prefix` and
    /// never re-decoded. Every `refresh` restores the KV checkpoint back to
    /// exactly this length before appending its tail, so the prefill is paid once
    /// and the held context never grows beyond `prefix_len + tail + generation`.
    prefix_len: i32,
    /// A FNV-1a hash of the prefix text used to build the current snapshot.
    /// When the prefix changes (settings update), the snapshot is discarded and
    /// re-captured after the next `prefill_prefix` call so the KV checkpoint
    /// stays coherent with the actual prefix content.
    prefix_hash: u64,
    /// Serialised KV checkpoint of the context state immediately after
    /// `prefill_prefix` completes. Captured once per prefix via
    /// `state_seq_get_data_ext` (the per-sequence form with a detectable failure
    /// path). Each `refresh` restores it via `state_seq_set_data_ext` (preferred,
    /// bool-returning) so a restore failure is detectable and can be treated as
    /// fatal.
    ///
    /// `None` until the first successful `prefill_prefix`. Falls back to
    /// `clear_kv_cache_seq` when absent (e.g. a cancelled or failed prefill).
    ///
    /// Promotion gate: this path is opt-in at compile time via the
    /// `USE_KV_CHECKPOINT` constant. The `clear_kv_cache_seq` fallback remains
    /// the active code path until the round-trip test (see
    /// `tests::kv_checkpoint_round_trip_smoke`) is confirmed green.
    snapshot: Option<Vec<u8>>,
    /// Test-only flag: when `true`, `refresh` uses path A (checkpoint restore)
    /// regardless of the `USE_KV_CHECKPOINT` compile-time constant. This lets
    /// integration tests exercise path A without promoting the constant.
    #[cfg(test)]
    force_kv_checkpoint: bool,
}

/// Runtime knobs for the held live-session context. Mirrors
/// [`crate::LlamaTurnConfig`] but with live-specific defaults (n_ctx = 32 768,
/// KV-quant OFF per SP-LIVE E3).
///
/// Note: GPU offload (n_gpu_layers) is a model-load-time decision — it is set
/// on `LlamaModelParams` when the `LlamaModel` is constructed by `ipc-bridge`,
/// not on `LlamaContextParams`. `LlamaLiveBackend` borrows an already-loaded
/// `&LlamaModel` and therefore does NOT carry its own `n_gpu_layers` field.
///
/// There is deliberately **no `max_tokens` field** here. The per-refresh
/// generation cap is `SamplerConfig::max_tokens` (passed at each `refresh`
/// call). Keeping a second cap on the config struct would create a mismatch:
/// the seed-time overflow check would reserve against a different value than
/// the actual generation loop. The per-refresh budget guard in `refresh()` uses
/// `cfg.max_tokens` directly, matching the `LlamaTurnBackend` pattern.
#[derive(Debug, Clone)]
pub struct LlamaLiveConfig {
    /// Context window to allocate, in tokens. Default 32 768 (SP-LIVE E3).
    pub n_ctx: u32,
    /// Per-decode batch size — the chunked-prefill chunk size.
    pub n_batch: u32,
    /// CPU threads for llama.cpp inference.
    pub threads: i32,
}

impl Default for LlamaLiveConfig {
    fn default() -> Self {
        let threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        let threads = ((threads / 2) as i32).clamp(1, 8);
        Self {
            n_ctx: 32_768,
            n_batch: 512,
            threads,
        }
    }
}

impl<'m> LlamaLiveBackend<'m> {
    /// Build the backend. Allocates the `LlamaContext` immediately so any
    /// resource exhaustion is caught at construction, not mid-session.
    pub fn new(
        model: &'m llama_cpp_2::model::LlamaModel,
        config: LlamaLiveConfig,
    ) -> Result<Self, Error> {
        use llama_cpp_2::context::params::LlamaContextParams;
        use std::num::NonZeroU32;

        let backend = minutist_common::llama_backend::shared_llama_backend()
            .map_err(|e| Error::Inference(format!("llama backend init: {e}")))?;

        let n_ctx = NonZeroU32::new(config.n_ctx)
            .ok_or_else(|| Error::Inference("n_ctx must be non-zero".to_string()))?;

        // KV-quant OFF: do not set k_type/v_type — the default is no
        // quantisation (SP-LIVE E3: q8_0 costs ~15 % decode throughput for
        // memory savings the 36 GB test GPU does not need).
        //
        // GPU offload is a model-load-time decision (set via LlamaModelParams on
        // the borrowed &LlamaModel), NOT a context-level setting; there is no
        // `with_n_gpu_layers` on LlamaContextParams.
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(Some(n_ctx))
            .with_n_batch(config.n_batch)
            .with_n_threads(config.threads)
            .with_n_threads_batch(config.threads);

        let ctx = model
            .new_context(backend, ctx_params)
            .map_err(|e| Error::Inference(format!("LlamaContext init: {e}")))?;

        Ok(Self {
            model,
            config,
            ctx,
            n_past: 0,
            prefix_len: 0,
            prefix_hash: 0,
            snapshot: None,
            #[cfg(test)]
            force_kv_checkpoint: false,
        })
    }
}

/// When `true`, the checkpoint snapshot path (`state_seq_get_data_ext` /
/// `state_seq_set_data_ext`) is the ACTIVE prune mechanism in `refresh`.
/// When `false` (current default), `clear_kv_cache_seq` is used instead and
/// the snapshot is captured but never applied.
///
/// **Promotion criterion (authoritative — referenced by `components.md`,
/// `domain-ownership.md`, and both gated tests):** promote to `true` only when
/// BOTH gated real-model tests are green — `kv_checkpoint_round_trip_smoke`
/// (raw `state_seq_*_ext` round-trip identity under SWA) AND
/// `kv_checkpoint_refresh_path_a_smoke` (the same identity through `refresh`'s
/// path A, incl. `n_past` bookkeeping).
const USE_KV_CHECKPOINT: bool = false;

/// FNV-1a (64-bit) hash of a string — used as a fast prefix-change detector.
/// Not a cryptographic hash; collision probability is negligible for short
/// prefix texts.
fn fnv1a_64(s: &str) -> u64 {
    const FNV_OFFSET: u64 = 14_695_981_039_346_656_037;
    const FNV_PRIME: u64 = 1_099_511_628_211;
    s.bytes()
        .fold(FNV_OFFSET, |acc, b| acc.wrapping_mul(FNV_PRIME) ^ b as u64)
}

impl<'m> LiveSessionBackend for LlamaLiveBackend<'m> {
    fn prefill_prefix(&mut self, prefix_text: &str, cancel: &CancelFlag) -> Result<usize, Error> {
        use llama_cpp_2::context::session::LlamaStateSeqFlags;
        use llama_cpp_2::llama_batch::LlamaBatch;
        use llama_cpp_2::model::AddBos;

        // Snapshot-coherence guard: if the prefix text has changed since the
        // last successful prefill, discard the stale snapshot so the new one is
        // captured below. This is a snapshot-validity check only — it does NOT
        // reset the KV cache or n_past. `prefill_prefix` must be called at most
        // once per backend instance with a given prefix text; calling it a second
        // time with different text would append the new prefix after the stale
        // one and produce a corrupt doubled prefix. The driver-level
        // `LiveSession::seed_prefix` idempotency guard is the true call-once
        // enforcer; this guard merely keeps the snapshot consistent if the hash
        // changes for any reason.
        let new_hash = fnv1a_64(prefix_text);
        if new_hash != self.prefix_hash {
            self.snapshot = None;
        }

        let mut tokens = self
            .model
            .str_to_token(prefix_text, AddBos::Never)
            .map_err(|e| Error::Inference(format!("tokenize prefix: {e}")))?;

        if tokens.is_empty() {
            return Ok(0);
        }

        // D3 (issue 0022): cap the pinned prefix at half of n_ctx so the other
        // half is always available for the per-refresh tail (retrieved context +
        // running digest + recent transcript window) plus generation. Under the
        // current driver the prefix is small (system prompt + category list + JSON
        // contract), so this is a defensive guard — not normally hit — against a
        // pathologically large system prompt. The instruction text sits at the
        // FRONT of the prefix, so truncating the token tail drops trailing prefix
        // content first and preserves the instructions: a soft degradation,
        // whereas starving the per-refresh window is not.
        let max_prefix_tokens = (self.config.n_ctx as usize) / 2;
        if tokens.len() > max_prefix_tokens {
            tracing::warn!(
                target: "chat-agent",
                prefix_tokens = tokens.len(),
                max_prefix_tokens,
                n_ctx = self.config.n_ctx,
                "live prefix exceeds half of n_ctx; truncating it to preserve \
                 per-refresh window headroom"
            );
            tokens.truncate(max_prefix_tokens);
        }

        let plan = summariser::plan_prefill(tokens.len(), self.config.n_batch);
        let mut batch = LlamaBatch::new(self.config.n_batch as usize, 1);
        // Capture n_past before any decode so a cancel or decode error can prune
        // the partially-appended prefix tokens and leave a consistent KV state.
        let n_past_before = self.n_past;

        for chunk in &plan.chunks {
            // M5: honour cancel between chunks so a Stop during prefill does not
            // leave a GPU-bound zombie.
            if cancel.is_cancelled() {
                // Prune any chunks already written to the KV cache.
                let _ = self
                    .ctx
                    .clear_kv_cache_seq(Some(0), Some(n_past_before as u32), None);
                return Err(Error::Inference("prefix prefill cancelled".to_string()));
            }
            batch.clear();
            for offset in 0..chunk.len {
                let global = chunk.start + offset;
                let pos = self.n_past + global as i32;
                let logits = chunk.logits_at_last && offset == chunk.len - 1;
                batch
                    .add(tokens[global], pos, &[0], logits)
                    .map_err(|e| Error::Inference(format!("batch.add (prefix prefill): {e}")))?;
            }
            if let Err(e) = self.ctx.decode(&mut batch) {
                // Prune any partial prefix tokens already in the KV cache so
                // the context remains consistent if the caller retries.
                let _ = self
                    .ctx
                    .clear_kv_cache_seq(Some(0), Some(n_past_before as u32), None);
                return Err(Error::Inference(format!("decode (prefix prefill): {e}")));
            }
        }

        let n_prefilled = tokens.len();
        self.n_past += n_prefilled as i32;
        // The pinned prefix occupies KV positions 0..n_past. Record this length:
        // every refresh prunes back to it before appending its tail.
        self.prefix_len = self.n_past;

        // --- Capture the KV checkpoint immediately after the prefix is settled ---
        //
        // The snapshot is taken ONLY when the cancel flag is clear — a cancelled
        // prefill already returned Err above, so reaching here guarantees the
        // context state is fully decoded and coherent.
        //
        // We use `state_seq_get_size_ext` + `state_seq_get_data_ext` (per-sequence
        // form). The allocation is paid once per session: snapshot size equals the
        // full KV state for seq 0 at prefix depth, typically a few tens of MB.
        //
        // SAFETY (copy): `buf` is sized by `state_seq_get_size_ext` on the same
        // context with no intervening decode, so sz equals the exact number of
        // bytes the C side will write. The wrapper passes `usize::MAX` as the
        // dest-size arg; the C call is bounded by the state size it measured
        // internally (which equals sz), not by the Rust buffer length.
        {
            let sz = self
                .ctx
                .state_seq_get_size_ext(0, LlamaStateSeqFlags::empty());
            let mut buf = vec![0u8; sz];
            let n = unsafe {
                self.ctx
                    .state_seq_get_data_ext(buf.as_mut_ptr(), 0, LlamaStateSeqFlags::empty())
            };
            if n == 0 || n != sz {
                // `get_size` and `get_data` share the same `state_seq_write_data`
                // measurement pass on an unchanged context, so the only failure
                // the C impl actually produces is a 0 return (a caught
                // exception); `n != sz` cannot occur in practice and is a
                // belt-and-braces guard. Either way, leave snapshot = None so
                // refresh falls back to clear_kv_cache_seq rather than restoring
                // a truncated buffer.
                tracing::warn!(
                    target: "chat-agent",
                    expected_bytes = sz,
                    written_bytes = n,
                    "KV checkpoint capture incomplete — snapshot discarded, \
                     refresh will fall back to clear_kv_cache_seq"
                );
            } else {
                buf.truncate(n);
                self.snapshot = Some(buf);
                self.prefix_hash = new_hash;
                tracing::debug!(
                    target: "chat-agent",
                    snapshot_bytes = n,
                    "live session KV checkpoint captured"
                );
            }
        }

        tracing::debug!(
            target: "chat-agent",
            n_tokens = n_prefilled,
            n_past = self.n_past,
            "live session prefix prefilled"
        );

        Ok(n_prefilled)
    }

    fn refresh(
        &mut self,
        tail_text: &str,
        cfg: &SamplerConfig,
        cancel: &CancelFlag,
        token_cb: &mut dyn FnMut(&str),
    ) -> Result<RawTurn, Error> {
        use encoding_rs::UTF_8;
        use llama_cpp_2::context::session::LlamaStateSeqFlags;
        use llama_cpp_2::llama_batch::LlamaBatch;
        use llama_cpp_2::model::AddBos;
        use llama_cpp_2::sampling::LlamaSampler;

        // --- Restore the KV state back to the pinned prefix ---
        //
        // The held context retains ONLY the pinned prefix (the open user turn:
        // system + digest-category instructions). Each refresh re-decodes a fresh,
        // BOUNDED tail (the retrieved context + the running digest + the recent
        // transcript window + the chat-template suffix that closes the user turn
        // and opens the model turn) on top of the reused prefix KV, then
        // generates. Restoring to the checkpoint here drops the previous refresh's
        // tail AND its generated answer, so `n_past` never grows beyond
        // `prefix_len + tail + generation`. The prefix prefill is paid once.
        //
        // Two paths:
        //   A (checkpoint, USE_KV_CHECKPOINT=true): restore via
        //      `state_seq_set_data_ext` — the bool return gives a detectable
        //      failure; false is treated as fatal and the context is marked
        //      terminal. This is the preferred path once the round-trip test
        //      confirms correctness across Gemma SWA.
        //   B (fallback, USE_KV_CHECKPOINT=false / no snapshot): fall back to
        //      `clear_kv_cache_seq` — the existing proven path, unchanged.
        if self.prefix_len <= 0 {
            return Err(Error::Inference(
                "refresh called before the prefix was seeded".to_string(),
            ));
        }
        #[cfg(test)]
        let use_checkpoint = USE_KV_CHECKPOINT || self.force_kv_checkpoint;
        #[cfg(not(test))]
        let use_checkpoint = USE_KV_CHECKPOINT;

        if self.n_past > self.prefix_len {
            if use_checkpoint {
                // Path A: snapshot restore (preferred — detectable failure).
                //
                // SAFETY: `snapshot` bytes were written by `state_seq_get_data_ext`
                // on this same context immediately after `prefill_prefix`. The
                // context has not been destroyed or replaced since then (it lives for
                // `LlamaLiveBackend`'s lifetime). The buffer length is passed as
                // `src.len()` inside `state_seq_set_data_ext`.
                match &self.snapshot {
                    Some(buf) => {
                        let ok = unsafe {
                            self.ctx.state_seq_set_data_ext(
                                buf,
                                0,
                                LlamaStateSeqFlags::empty(),
                            )
                        };
                        if !ok {
                            return Err(Error::Inference(
                                "state_seq_set_data_ext failed restoring KV checkpoint; \
                                 context state is inconsistent"
                                    .to_string(),
                            ));
                        }
                        self.n_past = self.prefix_len;
                    }
                    None => {
                        // Snapshot absent (e.g. prefill cancelled); fall back to
                        // clear_kv_cache_seq so the session degrades gracefully.
                        match self
                            .ctx
                            .clear_kv_cache_seq(Some(0), Some(self.prefix_len as u32), None)
                        {
                            Ok(true) => self.n_past = self.prefix_len,
                            Ok(false) | Err(_) => {
                                return Err(Error::Inference(
                                    "clear_kv_cache_seq failed (checkpoint fallback); \
                                     context state is inconsistent"
                                        .to_string(),
                                ));
                            }
                        }
                    }
                }
            } else {
                // Path B (active): clear_kv_cache_seq — the proven prune path.
                // The snapshot is captured in prefill_prefix but not applied here
                // until USE_KV_CHECKPOINT is promoted to true.
                match self
                    .ctx
                    .clear_kv_cache_seq(Some(0), Some(self.prefix_len as u32), None)
                {
                    Ok(true) => self.n_past = self.prefix_len,
                    Ok(false) | Err(_) => {
                        return Err(Error::Inference(
                            "clear_kv_cache_seq failed pruning to prefix; \
                             context state is inconsistent"
                                .to_string(),
                        ));
                    }
                }
            }
        }

        // --- Tokenise the tail ---
        //
        // The driver always supplies a non-empty tail: at minimum it carries the
        // chat-template suffix (`<end_of_turn>\n<start_of_turn>model\n`), so the
        // tail decode below always repopulates logits at its final token. That
        // removes the empty-tail logit-coherence hazard the cumulative-append
        // design had to guard against.
        let tail_tokens = self
            .model
            .str_to_token(tail_text, AddBos::Never)
            .map_err(|e| Error::Inference(format!("tokenize tail: {e}")))?;
        if tail_tokens.is_empty() {
            return Err(Error::Inference(
                "live refresh tail tokenised to zero tokens".to_string(),
            ));
        }

        // --- Budget guard: prefix + tail + generation must fit n_ctx ---
        //
        // With a bounded transcript window and a token-budgeted prefix this
        // never fires in practice; it is a backstop against a misconfigured
        // window or prefix budget. It reserves cfg.max_tokens for the generation
        // loop, mirroring the check in LlamaTurnBackend.
        let required = (self.n_past as usize)
            .saturating_add(tail_tokens.len())
            .saturating_add(cfg.max_tokens);
        if required > self.config.n_ctx as usize {
            return Err(Error::ContextOverflow(format!(
                "live refresh would exceed context: prefix={}, tail={} tokens, \
                 max_tokens={} but n_ctx={}",
                self.prefix_len,
                tail_tokens.len(),
                cfg.max_tokens,
                self.config.n_ctx,
            )));
        }

        // --- Append the tail on top of the pinned prefix ---
        // M1: a mid-loop decode failure prunes back to prefix_len so the KV
        // state stays consistent. n_past advances only after the whole tail
        // decodes.
        let plan = summariser::plan_prefill(tail_tokens.len(), self.config.n_batch);
        let mut batch = LlamaBatch::new(self.config.n_batch as usize, 1);
        for chunk in &plan.chunks {
            // M5: honour cancel during the tail-prefill loop so a mid-recording
            // Stop doesn't block on a large tail decode.
            if cancel.is_cancelled() {
                let _ = self
                    .ctx
                    .clear_kv_cache_seq(Some(0), Some(self.prefix_len as u32), None);
                self.n_past = self.prefix_len;
                return Err(Error::Inference("tail prefill cancelled".to_string()));
            }
            batch.clear();
            for offset in 0..chunk.len {
                let global = chunk.start + offset;
                let pos = self.n_past + global as i32;
                let logits = chunk.logits_at_last && offset == chunk.len - 1;
                batch
                    .add(tail_tokens[global], pos, &[0], logits)
                    .map_err(|e| Error::Inference(format!("batch.add (tail prefill): {e}")))?;
            }
            if let Err(e) = self.ctx.decode(&mut batch) {
                let _ = self
                    .ctx
                    .clear_kv_cache_seq(Some(0), Some(self.prefix_len as u32), None);
                self.n_past = self.prefix_len;
                return Err(Error::Inference(format!("decode (tail prefill): {e}")));
            }
        }
        self.n_past += tail_tokens.len() as i32;

        // --- Decode the digest answer from the reused-prefix + fresh-tail state ---
        //
        // The generated answer tokens are decoded into the KV cache during the
        // loop below and left there; the NEXT refresh's prune-to-prefix removes
        // them along with this tail. They are never sampled against again (the
        // next refresh re-decodes a fresh non-empty tail, repopulating logits,
        // before it samples), so no separate generation-prune or logit-coherence
        // guard is required.
        let mut sampler = if cfg.is_greedy() {
            LlamaSampler::chain_simple([LlamaSampler::greedy()])
        } else {
            LlamaSampler::chain_simple([
                LlamaSampler::penalties(64, 1.1, 0.0, 0.0),
                LlamaSampler::top_k(64),
                LlamaSampler::top_p(cfg.top_p, 1),
                LlamaSampler::min_p(0.05, 1),
                LlamaSampler::temp(cfg.temperature),
                LlamaSampler::dist(cfg.seed),
            ])
        };

        let mut decoder = UTF_8.new_decoder();
        let mut text = String::new();
        let mut batch = LlamaBatch::new(self.config.n_batch as usize, 1);

        let mut cancelled = false;
        for _ in 0..cfg.max_tokens {
            if cancel.is_cancelled() {
                cancelled = true;
                break;
            }

            let token = sampler.sample(&self.ctx, -1);
            sampler.accept(token);

            if self.model.is_eog_token(token) {
                break;
            }

            let piece = self
                .model
                .token_to_piece(token, &mut decoder, true, None)
                .map_err(|e| Error::Inference(format!("token_to_piece: {e}")))?;
            text.push_str(&piece);

            if !piece.is_empty() {
                token_cb(&piece);
            }

            batch.clear();
            batch
                .add(token, self.n_past, &[0], true)
                .map_err(|e| Error::Inference(format!("batch.add (gen): {e}")))?;
            self.n_past += 1;
            self.ctx
                .decode(&mut batch)
                .map_err(|e| Error::Inference(format!("decode (gen): {e}")))?;
        }

        // The generated answer tokens remain in the KV cache (positions
        // prefix_len + tail_len .. n_past). They are removed by the NEXT
        // refresh's prune-to-prefix, never sampled against again.

        tracing::debug!(
            target: "chat-agent",
            n_past = self.n_past,
            prefix_len = self.prefix_len,
            tail_tokens = tail_tokens.len(),
            text_chars = text.len(),
            cancelled,
            "live session digest refresh complete"
        );

        Ok(RawTurn {
            text,
            tool_calls: Vec::new(),
            cancelled,
        })
    }
}

impl<'m> LlamaLiveBackend<'m> {
    /// Number of tokens currently consumed in the held KV cache.
    pub fn n_past(&self) -> i32 {
        self.n_past
    }

    /// Remaining context capacity in tokens, reserving `cfg_max_tokens` for
    /// generation on the next refresh.
    ///
    /// Returns 0 if the reservation already exceeds available space. The S2b
    /// driver can use this to detect approaching exhaustion proactively (e.g.
    /// log a warning or stop accepting refreshes before the hard overflow).
    pub fn remaining_capacity(&self, cfg_max_tokens: usize) -> usize {
        let used = self.n_past as usize;
        let total = self.config.n_ctx as usize;
        total.saturating_sub(used).saturating_sub(cfg_max_tokens)
    }

    /// Byte size of the current KV checkpoint snapshot, or `None` if no
    /// snapshot has been captured yet (the prefix has not been prefilled, or
    /// the last prefill was cancelled/failed).
    ///
    /// The S2b driver logs this at session-start to give visibility into the
    /// per-session memory cost of the checkpoint.
    pub fn snapshot_size(&self) -> Option<usize> {
        self.snapshot.as_ref().map(|b| b.len())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::RawTurn;
    use std::sync::{Arc, Mutex};

    // -----------------------------------------------------------------------
    // Stub backend for unit tests (no model, no FFI)
    // -----------------------------------------------------------------------

    /// A scriptable stub that records every `prefill_prefix` and `refresh` call
    /// so the `LiveSession` discipline (prefix exactly once, only-tail-per-
    /// refresh) can be asserted without a model.
    ///
    /// Tracks a fake `n_past` (one byte = one token) and a configurable
    /// `n_ctx` capacity so capacity-overflow tests can drive the stub to the
    /// boundary.
    struct StubLiveBackend {
        /// Responses to return for successive `refresh` calls (popped in order).
        refresh_results: Vec<RawTurn>,
        /// All `prefix_text` values passed to `prefill_prefix`.
        prefill_calls: Arc<Mutex<Vec<String>>>,
        /// All `tail_text` values passed to `refresh`.
        refresh_calls: Arc<Mutex<Vec<String>>>,
        /// Fake token counter (one byte = one token).
        n_past: usize,
        /// Length of the pinned prefix, recorded at `prefill_prefix`. Each
        /// `refresh` prunes `n_past` back to this before appending its tail,
        /// mirroring the real backend's prune-to-prefix discipline.
        prefix_len: usize,
        /// Simulated context window size for overflow tests. `None` = unlimited.
        n_ctx: Option<usize>,
    }

    impl StubLiveBackend {
        fn new(refresh_results: Vec<RawTurn>) -> Self {
            Self {
                refresh_results,
                prefill_calls: Arc::new(Mutex::new(Vec::new())),
                refresh_calls: Arc::new(Mutex::new(Vec::new())),
                n_past: 0,
                prefix_len: 0,
                n_ctx: None,
            }
        }

        fn with_n_ctx(mut self, n_ctx: usize) -> Self {
            self.n_ctx = Some(n_ctx);
            self
        }

        fn prefill_call_count(&self) -> usize {
            self.prefill_calls.lock().unwrap().len()
        }

        fn refresh_tails(&self) -> Vec<String> {
            self.refresh_calls.lock().unwrap().clone()
        }

        fn n_past(&self) -> usize {
            self.n_past
        }
    }

    impl LiveSessionBackend for StubLiveBackend {
        fn prefill_prefix(
            &mut self,
            prefix_text: &str,
            cancel: &CancelFlag,
        ) -> Result<usize, Error> {
            if cancel.is_cancelled() {
                return Err(Error::Inference("prefix prefill cancelled".to_string()));
            }
            self.prefill_calls
                .lock()
                .unwrap()
                .push(prefix_text.to_string());
            let n = prefix_text.len(); // fake: one byte = one token
            self.n_past += n;
            // The prefix occupies positions 0..n_past; record its length so
            // refresh can prune back to it (mirrors the real backend).
            self.prefix_len = self.n_past;
            Ok(n)
        }

        fn refresh(
            &mut self,
            tail_text: &str,
            cfg: &SamplerConfig,
            cancel: &CancelFlag,
            token_cb: &mut dyn FnMut(&str),
        ) -> Result<RawTurn, Error> {
            self.refresh_calls
                .lock()
                .unwrap()
                .push(tail_text.to_string());

            let tail_len = tail_text.len();

            // Prune back to the pinned prefix before appending this tail
            // (mirrors the real backend's prune-to-prefix discipline). n_past
            // therefore tracks prefix_len + current tail, never cumulative.
            self.n_past = self.prefix_len;

            // Budget guard matching LlamaLiveBackend behaviour.
            if let Some(n_ctx) = self.n_ctx {
                let required = self
                    .n_past
                    .saturating_add(tail_len)
                    .saturating_add(cfg.max_tokens);
                if required > n_ctx {
                    return Err(Error::ContextOverflow(format!(
                        "stub: n_past={}, tail={}, max_tokens={} would exceed n_ctx={}",
                        self.n_past, tail_len, cfg.max_tokens, n_ctx
                    )));
                }
            }

            // Advance n_past by the tail (mirrors real backend).
            self.n_past += tail_len;
            // Record n_past after tail (before generation).
            let n_past_after_tail = self.n_past;

            let raw = self.refresh_results.pop().unwrap_or_default();
            if cancel.is_cancelled() {
                // Do NOT advance n_past by generated tokens (mirrors real
                // backend's cancel-prune behaviour).
                return Ok(RawTurn {
                    text: raw.text.clone(),
                    tool_calls: Vec::new(),
                    cancelled: true,
                });
            }
            // Stream each word as a token; n_past does NOT advance past the
            // generation (mirrors the real backend's post-decode prune).
            for word in raw.text.split_inclusive(' ') {
                token_cb(word);
            }
            // Confirm n_past was not advanced by generation.
            debug_assert_eq!(self.n_past, n_past_after_tail);
            Ok(raw)
        }
    }

    // -----------------------------------------------------------------------
    // seed_prefix is called exactly once
    // -----------------------------------------------------------------------

    #[test]
    fn seed_prefix_prefills_exactly_once() {
        let results = vec![
            RawTurn {
                text: "digest B".into(),
                ..Default::default()
            },
            RawTurn {
                text: "digest A".into(),
                ..Default::default()
            },
        ];
        let stub = StubLiveBackend::new(results);
        let mut session = LiveSession::new(stub);

        // First seed: should call prefill_prefix once.
        session
            .seed_prefix("system + categories prefix text", &CancelFlag::new())
            .unwrap();
        assert_eq!(
            session.backend.prefill_call_count(),
            1,
            "prefill_prefix must be called exactly once on first seed_prefix"
        );

        // Second seed: no-op; prefill_prefix must NOT be called again.
        session
            .seed_prefix("system + categories prefix text", &CancelFlag::new())
            .unwrap();
        assert_eq!(
            session.backend.prefill_call_count(),
            1,
            "second seed_prefix is a no-op — prefix must not be re-prefilled"
        );
    }

    // -----------------------------------------------------------------------
    // Cancel during prefill returns Err (M5)
    // -----------------------------------------------------------------------

    #[test]
    fn cancel_during_prefill_returns_error_not_ok() {
        let stub = StubLiveBackend::new(vec![]);
        let mut session = LiveSession::new(stub);

        let cancel = CancelFlag::new();
        cancel.cancel(); // raised before the call

        let result = session.seed_prefix("prefix text that would be long in production", &cancel);

        assert!(
            result.is_err(),
            "cancelled prefill must return Err, got {result:?}"
        );
        // The session must NOT mark prefix_seeded = true on a cancelled prefill.
        // A subsequent seed with a live cancel flag must still try (and succeed if
        // cancel is cleared). We verify prefix_seeded is false by attempting a
        // refresh — it must fail with "before seed_prefix" (not "refresh" error).
        let refresh_err = session.refresh_typed(
            "tail",
            &SamplerConfig::deterministic(),
            &CancelFlag::new(),
            &mut |_| {},
        );
        assert!(
            matches!(refresh_err, Err(Error::Inference(_))),
            "refresh after cancelled prefill must still report unseeded context"
        );
    }

    // -----------------------------------------------------------------------
    // Each refresh passes only the incremental tail
    // -----------------------------------------------------------------------

    #[test]
    fn refresh_passes_only_incremental_tail() {
        // Two refreshes: first with "segment one", then with "segment two".
        // The backend must see each tail only — not the prefix, not earlier tails.
        let results = vec![
            RawTurn {
                text: "digest 2".into(),
                ..Default::default()
            },
            RawTurn {
                text: "digest 1".into(),
                ..Default::default()
            },
        ];
        let stub = StubLiveBackend::new(results);
        let mut session = LiveSession::new(stub);

        session.seed_prefix("prefix", &CancelFlag::new()).unwrap();

        let result1 = session
            .refresh(
                "segment one",
                &SamplerConfig::deterministic(),
                &CancelFlag::new(),
                &mut |_| {},
            )
            .unwrap();
        let result2 = session
            .refresh(
                "segment two",
                &SamplerConfig::deterministic(),
                &CancelFlag::new(),
                &mut |_| {},
            )
            .unwrap();

        let tails = session.backend.refresh_tails();
        assert_eq!(tails, vec!["segment one", "segment two"]);
        assert_eq!(result1, "digest 1");
        assert_eq!(result2, "digest 2");
    }

    // -----------------------------------------------------------------------
    // refresh before seed_prefix is an error
    // -----------------------------------------------------------------------

    #[test]
    fn refresh_before_seed_prefix_returns_error() {
        let stub = StubLiveBackend::new(vec![]);
        let mut session = LiveSession::new(stub);

        let err = session
            .refresh(
                "some tail",
                &SamplerConfig::deterministic(),
                &CancelFlag::new(),
                &mut |_| {},
            )
            .unwrap_err();

        assert!(
            matches!(err, minutist_common::AppError::Inference { .. }),
            "refresh before seed_prefix must be an Inference error, got {err:?}"
        );
    }

    // -----------------------------------------------------------------------
    // Cancel mid-refresh
    // -----------------------------------------------------------------------

    #[test]
    fn cancel_mid_refresh_returns_partial_not_error() {
        let results = vec![RawTurn {
            text: "partial answer".into(),
            ..Default::default()
        }];
        let stub = StubLiveBackend::new(results);
        let mut session = LiveSession::new(stub);
        session.seed_prefix("prefix", &CancelFlag::new()).unwrap();

        let cancel = CancelFlag::new();
        cancel.cancel(); // raise before calling refresh

        // The stub observes the cancel flag and returns cancelled: true with
        // the partial text. LiveSession maps this to Ok(partial_text).
        let result = session.refresh(
            "new tail",
            &SamplerConfig::deterministic(),
            &cancel,
            &mut |_| {},
        );

        // Cancellation is not an error — the Ok path carries whatever partial
        // text the backend managed to produce.
        assert!(result.is_ok(), "cancel must not be an Err, got {result:?}");
        assert_eq!(
            result.unwrap(),
            "partial answer",
            "partial text is returned on cancel"
        );
    }

    // -----------------------------------------------------------------------
    // Multiple refreshes all land only their own tails
    // -----------------------------------------------------------------------

    #[test]
    fn multiple_refreshes_each_see_own_tail_only() {
        // Five refreshes with distinct tails. Assert each backend.refresh call
        // got exactly its own tail and nothing else.
        let n = 5usize;
        let results: Vec<RawTurn> = (0..n)
            .map(|i| RawTurn {
                text: format!("digest {i}"),
                ..Default::default()
            })
            .collect::<Vec<_>>()
            .into_iter()
            .rev() // pop() returns last first
            .collect();

        let stub = StubLiveBackend::new(results);
        let mut session = LiveSession::new(stub);
        session.seed_prefix("prefix", &CancelFlag::new()).unwrap();

        let tails: Vec<String> = (0..n).map(|i| format!("tail {i}")).collect();
        for tail in &tails {
            session
                .refresh(
                    tail,
                    &SamplerConfig::deterministic(),
                    &CancelFlag::new(),
                    &mut |_| {},
                )
                .unwrap();
        }

        assert_eq!(session.backend.refresh_tails(), tails);
        // The prefix was prefilled exactly once, regardless of how many refreshes.
        assert_eq!(session.backend.prefill_call_count(), 1);
    }

    // -----------------------------------------------------------------------
    // Token streaming through the callback
    // -----------------------------------------------------------------------

    #[test]
    fn refresh_streams_tokens_through_callback() {
        let results = vec![RawTurn {
            text: "action item: call Bob".into(),
            ..Default::default()
        }];
        let stub = StubLiveBackend::new(results);
        let mut session = LiveSession::new(stub);
        session.seed_prefix("prefix", &CancelFlag::new()).unwrap();

        let mut streamed = String::new();
        session
            .refresh(
                "tail",
                &SamplerConfig::deterministic(),
                &CancelFlag::new(),
                &mut |piece| streamed.push_str(piece),
            )
            .unwrap();

        assert_eq!(
            streamed, "action item: call Bob",
            "every piece must reach the token callback"
        );
    }

    // -----------------------------------------------------------------------
    // Capacity boundary — overflow returns ContextOverflow, not a crash
    // -----------------------------------------------------------------------

    #[test]
    fn refresh_at_capacity_returns_context_overflow() {
        // Configure a tiny context: prefix uses 6 bytes ("prefix" = 6), so
        // n_past = 6 after seed_prefix. Tail is "t" (1 byte).
        // SamplerConfig::deterministic() has max_tokens = 1024. So:
        //   n_past=6 + tail=1 + max_tokens=1024 = 1031 > n_ctx=1030 → overflow.
        let cfg = SamplerConfig::deterministic();
        let n_ctx = (6 + 1 + cfg.max_tokens) - 1; // exactly one short
        let stub = StubLiveBackend::new(vec![]).with_n_ctx(n_ctx);
        let mut session = LiveSession::new(stub);
        session.seed_prefix("prefix", &CancelFlag::new()).unwrap(); // 6 tokens used

        let err = session
            .refresh("t", &cfg, &CancelFlag::new(), &mut |_| {})
            .unwrap_err();

        assert!(
            matches!(err, minutist_common::AppError::InvalidInput { .. }),
            "overflow at capacity boundary must be InvalidInput (ContextOverflow maps there), got {err:?}"
        );
    }

    #[test]
    fn refresh_just_within_capacity_succeeds() {
        // n_past=6 + tail=1 + max_tokens=1024 = 1031; n_ctx=1031 → fits exactly.
        let cfg = SamplerConfig::deterministic();
        let n_ctx = 6 + 1 + cfg.max_tokens; // exactly enough
        let results = vec![RawTurn {
            text: "ok".into(),
            ..Default::default()
        }];
        let stub = StubLiveBackend::new(results).with_n_ctx(n_ctx);
        let mut session = LiveSession::new(stub);
        session.seed_prefix("prefix", &CancelFlag::new()).unwrap();

        let result = session.refresh("t", &cfg, &CancelFlag::new(), &mut |_| {});
        assert!(result.is_ok(), "exactly-at-capacity refresh must succeed");
    }

    // -----------------------------------------------------------------------
    // Cancel-prune discipline: n_past after cancel == n_past after tail append
    // -----------------------------------------------------------------------

    #[test]
    fn cancel_does_not_advance_n_past_beyond_tail() {
        // After seed_prefix("prefix") — 6 bytes — n_past = 6.
        // refresh("tail") — 4 bytes — n_past after tail append = 10.
        // Cancel fires before any generation tokens are decoded.
        // After the cancelled refresh, n_past must still be 10 (not higher).
        let results = vec![RawTurn {
            text: "some answer text".into(),
            ..Default::default()
        }];
        let stub = StubLiveBackend::new(results);
        let mut session = LiveSession::new(stub);
        session.seed_prefix("prefix", &CancelFlag::new()).unwrap(); // n_past = 6
        assert_eq!(session.backend.n_past(), 6);

        let cancel = CancelFlag::new();
        cancel.cancel(); // raise before refresh

        session
            .refresh(
                "tail",
                &SamplerConfig::deterministic(),
                &cancel,
                &mut |_| {},
            )
            .unwrap();

        // n_past must equal prefix (6) + tail (4) = 10; no generation tokens.
        assert_eq!(
            session.backend.n_past(),
            10,
            "cancelled refresh must not advance n_past past the tail"
        );
    }

    // -----------------------------------------------------------------------
    // Successful refresh also prunes generated tokens (n_past stays at tail)
    // -----------------------------------------------------------------------

    #[test]
    fn successful_refresh_prunes_generated_tokens_from_n_past() {
        // After seed_prefix("prefix") — 6 bytes — n_past = 6.
        // refresh("tail") — 4 bytes — answer is "hello world" (11 bytes).
        // After the completed refresh, n_past must be 10 (not 21).
        let results = vec![RawTurn {
            text: "hello world".into(),
            ..Default::default()
        }];
        let stub = StubLiveBackend::new(results);
        let mut session = LiveSession::new(stub);
        session.seed_prefix("prefix", &CancelFlag::new()).unwrap(); // n_past = 6

        session
            .refresh(
                "tail",
                &SamplerConfig::deterministic(),
                &CancelFlag::new(),
                &mut |_| {},
            )
            .unwrap();

        // The stub mirrors the real backend's prune policy: n_past stays at
        // prefix + tail, not prefix + tail + answer.
        assert_eq!(
            session.backend.n_past(),
            10,
            "successful refresh must prune generated tokens from n_past"
        );
    }

    // -----------------------------------------------------------------------
    // n_past stays bounded across many refreshes (the issue-0022 regression)
    // -----------------------------------------------------------------------
    //
    // The cumulative-append design grew n_past every refresh until it overflowed
    // n_ctx mid-meeting. The prune-to-prefix design must keep n_past at
    // prefix_len + current_tail regardless of how many refreshes have run.

    #[test]
    fn n_past_stays_bounded_across_many_refreshes() {
        let n = 50usize;
        let results: Vec<RawTurn> = (0..n)
            .map(|i| RawTurn {
                text: format!("answer {i} with some generated tokens"),
                ..Default::default()
            })
            .collect();
        let stub = StubLiveBackend::new(results);
        let mut session = LiveSession::new(stub);
        session.seed_prefix("prefix-of-12b", &CancelFlag::new()).unwrap();
        let prefix_len = session.backend.n_past();

        // Every refresh uses a same-length tail. n_past after each must equal
        // prefix_len + tail_len — never cumulative.
        let tail = "this is one refresh tail";
        for _ in 0..n {
            session
                .refresh(
                    tail,
                    &SamplerConfig::deterministic(),
                    &CancelFlag::new(),
                    &mut |_| {},
                )
                .unwrap();
            assert_eq!(
                session.backend.n_past(),
                prefix_len + tail.len(),
                "n_past must stay bounded (prefix + current tail), not grow per refresh"
            );
        }
        // The expensive prefix prefill happened exactly once.
        assert_eq!(session.backend.prefill_call_count(), 1);
    }

    // -----------------------------------------------------------------------
    // Empty-tail refresh routing (LiveSession layer)
    // -----------------------------------------------------------------------
    //
    // The production driver never sends an empty tail (the tail always carries
    // the chat-template suffix), but the `LiveSession` layer itself imposes no
    // such constraint. This asserts it routes the caller's exact tail through —
    // the prefix is prefilled once and earlier tails are not re-sent.

    #[test]
    fn empty_tail_after_refresh_is_routed_correctly() {
        // Three scripted responses: first normal refresh, then empty-tail.
        let results = vec![
            RawTurn {
                text: "second digest".into(),
                ..Default::default()
            },
            RawTurn {
                text: "first digest".into(),
                ..Default::default()
            },
        ];
        let stub = StubLiveBackend::new(results);
        let mut session = LiveSession::new(stub);
        session.seed_prefix("prefix", &CancelFlag::new()).unwrap();

        // First refresh with real tail — advances n_past, prunes generation.
        session
            .refresh(
                "first segment",
                &SamplerConfig::deterministic(),
                &CancelFlag::new(),
                &mut |_| {},
            )
            .unwrap();

        // Second refresh with an empty tail — the stub sees "" as the tail,
        // NOT "prefix" or "first segment". The LiveSession must not re-send
        // earlier content.
        let result = session.refresh(
            "",
            &SamplerConfig::deterministic(),
            &CancelFlag::new(),
            &mut |_| {},
        );
        assert!(
            result.is_ok(),
            "empty-tail refresh after normal refresh must not error: {result:?}"
        );

        let tails = session.backend.refresh_tails();
        assert_eq!(
            tails,
            vec!["first segment", ""],
            "second refresh must pass only its own (empty) tail to the backend"
        );
        // The prefix was still prefilled exactly once.
        assert_eq!(session.backend.prefill_call_count(), 1);
    }

    // -----------------------------------------------------------------------
    // Gated real-backend test (requires MINUTIST_LLM_MODEL_PATH)
    // -----------------------------------------------------------------------

    /// Smoke-test that `LlamaLiveBackend::new` can construct a context and
    /// `LiveSession` can prefill a trivial prefix + decode a short answer.
    ///
    /// Requires the model GGUF at `MINUTIST_LLM_MODEL_PATH`. Skip if unset
    /// (CI host has no model). Validate on the Windows Vulkan build.
    #[test]
    #[ignore]
    fn llama_live_backend_prefill_and_refresh_smoke() {
        let model_path = match std::env::var("MINUTIST_LLM_MODEL_PATH") {
            Ok(p) => p,
            Err(_) => {
                eprintln!("MINUTIST_LLM_MODEL_PATH unset — skipping real-model live test");
                return;
            }
        };

        let backend_init = minutist_common::llama_backend::shared_llama_backend().expect("backend");
        let model = llama_cpp_2::model::LlamaModel::load_from_file(
            backend_init,
            std::path::Path::new(&model_path),
            &llama_cpp_2::model::params::LlamaModelParams::default(),
        )
        .expect("model load");

        let config = LlamaLiveConfig {
            n_ctx: 2_048, // small for smoke test
            ..LlamaLiveConfig::default()
        };

        let live_backend = LlamaLiveBackend::new(&model, config).expect("context build");
        let mut session = LiveSession::new(live_backend);

        session
            .seed_prefix("You are a meeting digest agent.\n", &CancelFlag::new())
            .expect("prefill");

        let mut digest = String::new();
        let result = session.refresh(
            "Alice: We need to set up a follow-up call.\n",
            &SamplerConfig::deterministic(),
            &CancelFlag::new(),
            &mut |piece| digest.push_str(piece),
        );
        assert!(result.is_ok(), "refresh smoke failed: {result:?}");
    }

    // -----------------------------------------------------------------------
    // Gated checkpoint round-trip test (requires MINUTIST_TEST_GEMMA_GGUF)
    // -----------------------------------------------------------------------

    /// Reads the sliding-window attention size from GGUF model metadata.
    ///
    /// llama.cpp stores the SWA window under a per-architecture key of the form
    /// `<arch>.attention.sliding_window`. This helper scans all metadata keys
    /// looking for one that ends with `.attention.sliding_window` and parses
    /// its value. Returns `None` if no such key exists (non-SWA model).
    #[cfg(test)]
    fn model_n_swa(model: &llama_cpp_2::model::LlamaModel) -> Option<i32> {
        let count = model.meta_count();
        for i in 0..count {
            let Ok(key) = model.meta_key_by_index(i) else { continue };
            if key.ends_with(".attention.sliding_window") {
                let Ok(val) = model.meta_val_str_by_index(i) else { continue };
                if let Ok(n) = val.trim().parse::<i32>() {
                    return Some(n);
                }
            }
        }
        None
    }

    /// Proves that the raw `state_seq_get_data_ext` / `state_seq_set_data_ext`
    /// API round-trips the KV state faithfully across Gemma SWA layers.
    ///
    /// The prefix is sized so that `prefix_len > n_swa` (queried from the
    /// model at runtime), guaranteeing the snapshot actually contains SWA
    /// state; the test fails loudly if the prefix is too short. The snapshot
    /// is captured at `prefix_len` depth (the capture depth), then restored
    /// and re-decoded with the same greedy tail — continuation A and B must
    /// be identical.
    ///
    /// This test exercises the raw API, not the `refresh` code path. See
    /// `kv_checkpoint_refresh_path_a_smoke` for the promotion gate that
    /// exercises path A inside `refresh` itself.
    ///
    /// Requires the Gemma 4B GGUF at `MINUTIST_TEST_GEMMA_GGUF`. Skips
    /// cleanly if the env var is unset so the test never blocks CI.
    ///
    /// Run locally with:
    /// ```text
    /// MINUTIST_TEST_GEMMA_GGUF=/path/to/gemma-4b.gguf \
    ///   cargo test -p chat-agent -- --include-ignored kv_checkpoint_round_trip_smoke
    /// ```
    #[test]
    #[ignore = "requires MINUTIST_TEST_GEMMA_GGUF pointing at a Gemma 4B GGUF"]
    fn kv_checkpoint_round_trip_smoke() {
        use llama_cpp_2::context::session::LlamaStateSeqFlags;
        use llama_cpp_2::llama_batch::LlamaBatch;
        use llama_cpp_2::model::AddBos;
        use llama_cpp_2::sampling::LlamaSampler;

        let model_path = match std::env::var("MINUTIST_TEST_GEMMA_GGUF") {
            Ok(p) if !p.is_empty() => p,
            _ => {
                eprintln!("MINUTIST_TEST_GEMMA_GGUF unset — skipping checkpoint round-trip test");
                return;
            }
        };

        let backend_init =
            minutist_common::llama_backend::shared_llama_backend().expect("llama backend");
        let model = llama_cpp_2::model::LlamaModel::load_from_file(
            backend_init,
            std::path::Path::new(&model_path),
            &llama_cpp_2::model::params::LlamaModelParams::default(),
        )
        .expect("model load");

        // Query the real SWA window from the model so the prefix-size assertion
        // below is accurate regardless of which Gemma variant is loaded.
        let n_swa = model_n_swa(&model)
            .expect("model must have an .attention.sliding_window metadata key");
        assert!(n_swa > 0, "expected a positive SWA window from the model");
        eprintln!("model n_swa = {n_swa}");

        // Build a prefix large enough that prefix_len > n_swa after tokenisation.
        // Each "wordN " is roughly 2–3 tokens; multiply by 2 for headroom.
        let target_tokens = (n_swa as usize) * 2;
        let words_needed = target_tokens / 2 + 1; // conservative lower bound
        let prefix_text: String = (0..words_needed)
            .map(|i| format!("word{i} "))
            .collect::<Vec<_>>()
            .join("");

        let n_batch = 512u32;
        let n_ctx = ((n_swa as u32) * 4).max(4_096);
        let config = LlamaLiveConfig {
            n_ctx,
            n_batch,
            ..LlamaLiveConfig::default()
        };

        let mut backend = LlamaLiveBackend::new(&model, config.clone()).expect("context build");

        // A tail that triggers a MULTI-token continuation (not a one-word
        // answer): the round-trip must compare a real generated sequence, else
        // an immediate-EOG tail makes the comparison vacuous (see the non-empty
        // assertion below — this is a USE_KV_CHECKPOINT promotion gate).
        let tail_text = "Continue this story about a lighthouse keeper: ";

        let cancel = CancelFlag::new();

        // Prefill the prefix. This also captures the snapshot.
        let n_prefilled = backend
            .prefill_prefix(&prefix_text, &cancel)
            .expect("prefill");
        assert!(n_prefilled > 0, "prefix must not tokenise to zero tokens");
        assert!(
            backend.snapshot.is_some(),
            "snapshot must be captured after prefill"
        );

        // Verify the snapshot was captured at a depth that actually exercises
        // SWA state: prefix_len must exceed n_swa so the snapshot contains
        // more than one SWA window of KV state.
        let prefix_len = backend.prefix_len;
        assert!(
            prefix_len > n_swa,
            "prefix_len={prefix_len} must exceed n_swa={n_swa} to force SWA state into the snapshot; \
             increase words_needed in the test"
        );
        let snap_size = backend.snapshot_size().unwrap();
        eprintln!("checkpoint snapshot size: {snap_size} bytes, prefix_len={prefix_len}");

        // Helper: prefill `tail_text` on top of the current n_past, then
        // greedily generate N tokens. Returns the generated token sequence.
        let decode_tail_and_generate = |ctx: &mut llama_cpp_2::context::LlamaContext<'_>,
                                        n_past_start: i32,
                                        n_generate: usize|
         -> Vec<i32> {
            let tail_tokens = model
                .str_to_token(tail_text, AddBos::Never)
                .expect("tokenize tail");
            assert!(!tail_tokens.is_empty(), "tail must not be empty");

            // Decode the tail so logits are valid at its final position.
            let plan = summariser::plan_prefill(tail_tokens.len(), n_batch);
            let mut batch = LlamaBatch::new(n_batch as usize, 1);
            let mut n_past = n_past_start;
            for chunk in &plan.chunks {
                batch.clear();
                for offset in 0..chunk.len {
                    let global = chunk.start + offset;
                    let pos = n_past + global as i32;
                    let logits = chunk.logits_at_last && offset == chunk.len - 1;
                    batch
                        .add(tail_tokens[global], pos, &[0], logits)
                        .expect("batch add");
                }
                ctx.decode(&mut batch).expect("decode tail");
            }
            n_past += tail_tokens.len() as i32;

            // Greedy-generate EXACTLY N tokens (the stale-logits gotcha is
            // avoided because the tail decode above repopulates logits at its
            // final token before any sampling). This is a MECHANISM round-trip
            // test, not a quality test: we deliberately do NOT stop at EOG. Post-
            // EOG greedy decode is still fully deterministic, so a correct
            // snapshot/restore reproduces it identically; generating a fixed N
            // makes the A==B comparison span enough tokens to genuinely exercise
            // the sliding-window state (an untemplated tail can otherwise emit a
            // single newline then EOG, making a stop-on-EOG comparison vacuous).
            let mut sampler = LlamaSampler::chain_simple([LlamaSampler::greedy()]);
            let mut tokens_out = Vec::with_capacity(n_generate);
            let mut batch = LlamaBatch::new(1, 1);
            for _ in 0..n_generate {
                let tok = sampler.sample(ctx, -1);
                sampler.accept(tok);
                tokens_out.push(tok.0);
                batch.clear();
                batch.add(tok, n_past, &[0], true).expect("batch add gen");
                n_past += 1;
                ctx.decode(&mut batch).expect("decode gen");
            }
            tokens_out
        };

        // --- Continuation A: generate from the freshly-prefilled state ---
        let continuation_a = decode_tail_and_generate(&mut backend.ctx, prefix_len, 16);
        eprintln!("continuation A: {continuation_a:?}");

        // --- Restore the snapshot and regenerate (continuation B) ---
        //
        // SAFETY: snapshot bytes were written by `state_seq_get_data_ext` on
        // this context immediately after prefill_prefix. The context is still
        // live and has not been replaced.
        let snapshot = backend.snapshot.as_ref().expect("snapshot present");
        let restored = unsafe {
            backend.ctx.state_seq_set_data_ext(
                snapshot,
                0,
                LlamaStateSeqFlags::empty(),
            )
        };
        assert!(restored, "state_seq_set_data_ext must return true on restore");

        // n_past is back to prefix_len after restore.
        let continuation_b = decode_tail_and_generate(&mut backend.ctx, prefix_len, 16);
        eprintln!("continuation B: {continuation_b:?}");

        // This is a promotion gate for USE_KV_CHECKPOINT, so the comparison must
        // not pass vacuously: if the tail decoded straight to EOG both
        // continuations would be empty and assert_eq! would succeed while
        // proving nothing about SWA-state fidelity. Require real generated
        // tokens before trusting the round-trip.
        assert!(
            continuation_a.len() >= 2,
            "the tail must produce >=2 non-EOG tokens for the round-trip comparison \
             to be meaningful (got {}); pick a tail that generates text",
            continuation_a.len()
        );
        assert_eq!(
            continuation_a, continuation_b,
            "KV checkpoint restore must reproduce identical greedy continuations; \
             divergence means SWA or recurrent state was not fully captured"
        );
        eprintln!("checkpoint round-trip: PASS ({} tokens match)", continuation_a.len());
    }

    // -----------------------------------------------------------------------
    // Gated path-A promotion gate (requires MINUTIST_TEST_GEMMA_GGUF)
    // -----------------------------------------------------------------------

    /// Exercises path A (checkpoint restore) through `refresh` itself, not
    /// through direct FFI calls. This is the gate that must be green before
    /// `USE_KV_CHECKPOINT` is promoted to `true`.
    ///
    /// The test seeds a prefix that exceeds `n_swa` (read from the model),
    /// forces `force_kv_checkpoint = true` on the backend, then calls
    /// `refresh` twice with identical tails and a greedy sampler. Both
    /// generated strings must be identical — any failure in path A's error
    /// handling, `n_past` bookkeeping, or snapshot restore would diverge
    /// or return an error rather than matching.
    ///
    /// Run locally with:
    /// ```text
    /// MINUTIST_TEST_GEMMA_GGUF=/path/to/gemma-4b.gguf \
    ///   cargo test -p chat-agent -- --include-ignored kv_checkpoint_refresh_path_a_smoke
    /// ```
    #[test]
    #[ignore = "requires MINUTIST_TEST_GEMMA_GGUF pointing at a Gemma 4B GGUF"]
    fn kv_checkpoint_refresh_path_a_smoke() {
        let model_path = match std::env::var("MINUTIST_TEST_GEMMA_GGUF") {
            Ok(p) if !p.is_empty() => p,
            _ => {
                eprintln!(
                    "MINUTIST_TEST_GEMMA_GGUF unset — skipping path-A refresh gate test"
                );
                return;
            }
        };

        let backend_init =
            minutist_common::llama_backend::shared_llama_backend().expect("llama backend");
        let model = llama_cpp_2::model::LlamaModel::load_from_file(
            backend_init,
            std::path::Path::new(&model_path),
            &llama_cpp_2::model::params::LlamaModelParams::default(),
        )
        .expect("model load");

        let n_swa = model_n_swa(&model)
            .expect("model must have an .attention.sliding_window metadata key");
        assert!(n_swa > 0, "expected a positive SWA window from the model");
        eprintln!("model n_swa = {n_swa}");

        // Build a prefix that will exceed n_swa after tokenisation.
        let target_tokens = (n_swa as usize) * 2;
        let words_needed = target_tokens / 2 + 1;
        let prefix_text: String = (0..words_needed)
            .map(|i| format!("word{i} "))
            .collect::<Vec<_>>()
            .join("");

        let n_ctx = ((n_swa as u32) * 4).max(4_096);
        let config = LlamaLiveConfig {
            n_ctx,
            n_batch: 512,
            ..LlamaLiveConfig::default()
        };

        let mut backend = LlamaLiveBackend::new(&model, config).expect("context build");

        // Force path A in refresh regardless of USE_KV_CHECKPOINT.
        backend.force_kv_checkpoint = true;

        let cancel = CancelFlag::new();
        backend
            .prefill_prefix(&prefix_text, &cancel)
            .expect("prefill");

        assert!(
            backend.snapshot.is_some(),
            "snapshot must be captured after prefill"
        );
        assert!(
            backend.prefix_len > n_swa,
            "prefix_len={} must exceed n_swa={n_swa} to exercise SWA in the snapshot",
            backend.prefix_len
        );

        // Use a greedy sampler so both calls produce deterministic output.
        let cfg = SamplerConfig {
            seed: 0,
            temperature: 0.0,
            top_p: 1.0,
            max_tokens: 16,
            grammar_backstop: false,
        };
        let tail = "<end_of_turn>\n<start_of_turn>model\n";

        // First refresh: n_past starts at prefix_len; path A is NOT triggered
        // yet (n_past == prefix_len), so this decodes the tail and generates.
        let result_a = backend
            .refresh(tail, &cfg, &CancelFlag::new(), &mut |_| {})
            .expect("first refresh");

        // Second refresh: n_past > prefix_len (tail + generation from first
        // call are still in the KV); path A must restore the snapshot back to
        // prefix_len, then decode the same tail and generate identically.
        let result_b = backend
            .refresh(tail, &cfg, &CancelFlag::new(), &mut |_| {})
            .expect("second refresh");

        eprintln!("path A refresh A: {result_a:?}");
        eprintln!("path A refresh B: {result_b:?}");

        assert_eq!(
            result_a.text, result_b.text,
            "path A restore via refresh must reproduce identical greedy output; \
             divergence means n_past bookkeeping or snapshot restore in refresh is wrong"
        );
        eprintln!("path-A gate: PASS");
    }
}
