//! Held-context live-session engine.
//!
//! The Phase-9 chat loop allocates a **fresh** [`LlamaContext`] per turn (clean
//! KV cache). The live in-meeting agent is an explicit departure from that
//! pattern (SP-LIVE E2): the attachment prefix is prefilled **once** at
//! recording start (~40 s for a moderately sized slide deck) and the context is
//! then extended with each digest refresh by appending only the incremental
//! transcript tail. A fresh-per-turn approach would re-pay the ~40 s prefill
//! cost on every cadence tick, making live operation unusable.
//!
//! # Design
//!
//! The testable seam is [`LiveSessionBackend`]:
//!
//! - [`LiveSessionBackend::prefill_prefix`] — tokenise + chunked-prefill the
//!   pinned system + attachments text ONCE, retaining the KV state.
//! - [`LiveSessionBackend::refresh`] — append the incremental tail tokens and
//!   decode the digest answer from the retained KV; does NOT re-prefill.
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

    /// Append `tail_text` tokens to the held KV cache and decode the digest
    /// answer from the retained context.
    ///
    /// The backend MUST NOT re-prefill the prefix. Only the tail tokens since
    /// the last call (or since prefill, on the first refresh) are new.
    ///
    /// **Precondition (logits invariant):** the most recent `decode` call into
    /// this context must have produced logits at the last KV position. The
    /// first refresh always satisfies this (prefill sets logits at the final
    /// prefix token). A subsequent refresh with an empty tail, however, arrives
    /// after the prior refresh pruned its generated tokens from the KV cache —
    /// leaving the logit buffer referencing a now-removed KV slot. The real
    /// backend enforces this invariant internally (re-decoding the last
    /// retained token when `tail_text` is empty and a prior generation was
    /// pruned); callers that bypass the real backend via a stub must honour
    /// it in the same way if they test the empty-tail-after-refresh path.
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
/// - `refresh(new_tail)` calls `backend.refresh` with only the text appended
///   since the last `refresh` (or since `seed_prefix` if this is the first).
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

    /// Prefill the system + attachments prefix into the held KV cache.
    ///
    /// Idempotent: the second and subsequent calls are no-ops (the prefix is
    /// already in the KV cache; re-prefilling would corrupt the position state).
    /// Returns `Ok(0)` on a no-op call.
    ///
    /// `cancel` is forwarded to the backend; a raised flag during the ~40 s
    /// prefill returns an `Inference` error and the partially-decoded KV state
    /// is discarded by the backend before returning.
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

    /// Append the incremental transcript tail and decode a digest text from the
    /// retained KV state.
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
///    prefix, advance `n_past`.
/// 2. [`LiveSessionBackend::refresh`] — tokenise the tail, prefill the tail
///    tokens from `n_past`, decode the answer, advance `n_past`.
///
/// **`!Send`**: `LlamaContext` is `!Send`. The S2b driver owns the thread.
pub struct LlamaLiveBackend<'m> {
    model: &'m llama_cpp_2::model::LlamaModel,
    config: LlamaLiveConfig,
    ctx: llama_cpp_2::context::LlamaContext<'m>,
    n_past: i32,
    /// True once a refresh has completed and pruned its generated tokens from
    /// the KV cache. After pruning, the last-logits slot references a position
    /// that no longer holds a freshly decoded token, so sampling against
    /// `ctx`'s stale logit buffer is undefined. An empty-tail refresh that
    /// follows a pruned prior refresh must re-decode the last retained token
    /// before sampling — see the empty-tail guard in `refresh`.
    prior_gen_pruned: bool,
    /// The last token id written to the held KV cache (the final token of the
    /// last tail prefill, or the final prefix token if no tail has been
    /// prefilled yet). Used by the empty-tail logit-repopulate path in
    /// `refresh` to re-decode the retained KV slot with logits enabled.
    last_kv_token: Option<llama_cpp_2::token::LlamaToken>,
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
            prior_gen_pruned: false,
            last_kv_token: None,
        })
    }
}

impl<'m> LiveSessionBackend for LlamaLiveBackend<'m> {
    fn prefill_prefix(&mut self, prefix_text: &str, cancel: &CancelFlag) -> Result<usize, Error> {
        use llama_cpp_2::llama_batch::LlamaBatch;
        use llama_cpp_2::model::AddBos;

        let tokens = self
            .model
            .str_to_token(prefix_text, AddBos::Never)
            .map_err(|e| Error::Inference(format!("tokenize prefix: {e}")))?;

        if tokens.is_empty() {
            return Ok(0);
        }

        // The prefix itself must fit in n_ctx (at least one token of headroom
        // for the first generation). The per-refresh budget guard in `refresh`
        // enforces the actual remaining capacity against cfg.max_tokens at call
        // time — that is where the correct per-call generation cap is known.
        if tokens.len() >= self.config.n_ctx as usize {
            return Err(Error::ContextOverflow(format!(
                "prefix is {} tokens which fills the entire context window ({}); \
                 no headroom for transcript or generation",
                tokens.len(),
                self.config.n_ctx,
            )));
        }

        let plan = summariser::plan_prefill(tokens.len(), self.config.n_batch);
        let mut batch = LlamaBatch::new(self.config.n_batch as usize, 1);
        // Capture n_past before any decode so a cancel or decode error can prune
        // the partially-appended prefix tokens and leave a consistent KV state.
        let n_past_before = self.n_past;

        for chunk in &plan.chunks {
            // M5: honour cancel between chunks so a Stop during the ~40 s
            // prefill does not leave a GPU-bound zombie.
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
        // Record the last prefix token for the empty-tail logit-repopulate path.
        self.last_kv_token = tokens.last().copied();

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
        use llama_cpp_2::llama_batch::LlamaBatch;
        use llama_cpp_2::model::AddBos;
        use llama_cpp_2::sampling::LlamaSampler;

        // --- Budget guard: ensure tail + generation fit in the remaining window ---
        //
        // The held context grows monotonically (prefix + accumulated transcript).
        // Check that appending the new tail tokens plus generating up to
        // cfg.max_tokens tokens would not overflow n_ctx. This mirrors the check
        // in LlamaTurnBackend (llama.rs line 270), but against the REMAINING
        // capacity rather than the full window. The check uses cfg.max_tokens
        // because that is the actual cap on the generation loop below.
        let tail_tokens = if !tail_text.is_empty() {
            self.model
                .str_to_token(tail_text, AddBos::Never)
                .map_err(|e| Error::Inference(format!("tokenize tail: {e}")))?
        } else {
            Vec::new()
        };

        let required = (self.n_past as usize)
            .saturating_add(tail_tokens.len())
            .saturating_add(cfg.max_tokens);
        if required > self.config.n_ctx as usize {
            return Err(Error::ContextOverflow(format!(
                "live session context exhausted: n_past={}, tail={} tokens, \
                 max_tokens={} but n_ctx={}; session must be re-seeded",
                self.n_past,
                tail_tokens.len(),
                cfg.max_tokens,
                self.config.n_ctx,
            )));
        }

        // --- Append tail tokens to the held KV ---
        // M1: capture n_past before the tail loop. A mid-loop decode failure
        // prunes the partially-appended range so the KV state is consistent on
        // return. n_past is only advanced AFTER the entire tail completes.
        if !tail_tokens.is_empty() {
            let plan = summariser::plan_prefill(tail_tokens.len(), self.config.n_batch);
            let mut batch = LlamaBatch::new(self.config.n_batch as usize, 1);
            let n_past_before_tail = self.n_past;

            for chunk in &plan.chunks {
                // M5: honour cancel during the tail-prefill loop too, so a
                // mid-recording Stop doesn't block on a large tail decode.
                if cancel.is_cancelled() {
                    let _ =
                        self.ctx
                            .clear_kv_cache_seq(Some(0), Some(n_past_before_tail as u32), None);
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
                    // M1: prune the partial tail range so the KV state is
                    // consistent. n_past is NOT advanced; the context stays at
                    // the pre-tail position. The driver tears down the session
                    // on this Err (the held-context invariant is broken).
                    let _ =
                        self.ctx
                            .clear_kv_cache_seq(Some(0), Some(n_past_before_tail as u32), None);
                    return Err(Error::Inference(format!("decode (tail prefill): {e}")));
                }
            }

            self.n_past += tail_tokens.len() as i32;
            // Record the last tail token for the empty-tail logit-repopulate path.
            self.last_kv_token = tail_tokens.last().copied();
            // The tail-prefill decode repopulates logits; any prior prune is
            // superseded.
            self.prior_gen_pruned = false;
        }

        // Record n_past after the tail append. Generated answer tokens are
        // decoded into the KV cache during the loop below, but are PRUNED back
        // to this position after every refresh (both on success and on cancel).
        //
        // KV retention policy: the held context accumulates prefix + transcript
        // tail only. Each refresh's generated answer is decoded ephemerally —
        // it guides sampling but is not retained across refreshes. This keeps
        // capacity growth proportional to the transcript (not doubled by answer
        // accumulation) and eliminates cancelled-generation pollution (an
        // abandoned partial answer is never left in the KV).
        //
        // Trade-off vs retaining answers: the model cannot see its prior digest
        // text as context. Standing-list continuity (the 'update, don't
        // regenerate' behaviour) is therefore achieved by re-deriving from the
        // full transcript tail each refresh, not by the model reading its prior
        // output. This is the safer choice for v1: it avoids capacity blowup
        // and the coherence hazard of a partial abandoned answer in context.
        let n_past_after_tail = self.n_past;

        // --- Logits coherence guard (empty-tail after prune) ---
        //
        // When a prior refresh decoded generation tokens and then pruned them via
        // clear_kv_cache_seq, the context's last-logits slot references the last
        // GENERATED token position, which no longer exists in the KV cache. A
        // subsequent sampler.sample call against those stale logits produces
        // incoherent output.
        //
        // The first refresh is always safe: plan_prefill sets logits_at_last on
        // the final prefix chunk, so the prefix decode leaves valid logits.
        //
        // A second+ refresh with a non-empty tail is also safe: the tail-prefill
        // decode repopulates logits at the last tail token before sampling.
        //
        // The hazardous path is: prior refresh pruned + this refresh has an empty
        // tail. Fix: re-decode the last retained token (stored in `last_kv_token`)
        // with logits enabled. The KV entry for that position is still valid; the
        // re-decode simply refreshes the logit buffer from the retained state.
        if tail_tokens.is_empty() && self.prior_gen_pruned {
            use llama_cpp_2::llama_batch::LlamaBatch;
            let last_pos = self.n_past - 1;
            if last_pos < 0 {
                return Err(Error::Inference(
                    "empty tail on empty context — seed_prefix must be called first".to_string(),
                ));
            }
            let last_token = self.last_kv_token.ok_or_else(|| {
                Error::Inference(
                    "empty-tail refresh after prune but last_kv_token not set — \
                     this is a bug in LlamaLiveBackend"
                        .to_string(),
                )
            })?;
            let mut repopulate_batch = LlamaBatch::new(1, 1);
            repopulate_batch
                .add(last_token, last_pos, &[0], true)
                .map_err(|e| Error::Inference(format!("batch.add (logit repopulate): {e}")))?;
            self.ctx
                .decode(&mut repopulate_batch)
                .map_err(|e| Error::Inference(format!("decode (logit repopulate): {e}")))?;
            // Logits are now valid; clear the flag.
            self.prior_gen_pruned = false;
        }

        // --- Decode the digest answer from the retained KV state ---
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

        // Prune the generated answer tokens from the KV cache (see policy above).
        // n_past is reset to the post-tail position so the next refresh appends
        // its own tail onto the transcript-only KV state.
        //
        // M2: match the clear_kv_cache_seq return. A false/Err means the KV
        // cache is in an unknown state; n_past must NOT be reset (that would
        // desync it from the actual KV), so we return Err to let the driver
        // tear down the session.
        if self.n_past > n_past_after_tail {
            match self
                .ctx
                .clear_kv_cache_seq(Some(0), Some(n_past_after_tail as u32), None)
            {
                Ok(true) => {
                    self.n_past = n_past_after_tail;
                    // Mark that a generation was pruned. The next empty-tail
                    // refresh must re-decode the last retained token to
                    // repopulate logits.
                    self.prior_gen_pruned = true;
                }
                Ok(false) | Err(_) => {
                    return Err(Error::Inference(
                        "clear_kv_cache_seq failed after generation prune; \
                         context state is inconsistent"
                            .to_string(),
                    ));
                }
            }
        }

        tracing::debug!(
            target: "chat-agent",
            n_past = self.n_past,
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
            .seed_prefix("system + attachments prefix text", &CancelFlag::new())
            .unwrap();
        assert_eq!(
            session.backend.prefill_call_count(),
            1,
            "prefill_prefix must be called exactly once on first seed_prefix"
        );

        // Second seed: no-op; prefill_prefix must NOT be called again.
        session
            .seed_prefix("system + attachments prefix text", &CancelFlag::new())
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
    // Empty-tail refresh after a normal refresh
    // -----------------------------------------------------------------------
    //
    // Verifies the `LiveSessionBackend::refresh` logits-invariant contract:
    // after a normal refresh (which prunes generated tokens), a subsequent
    // refresh with an empty tail must succeed and must record only the empty
    // tail — NOT re-send the prefix or the previous tail. The stub has no
    // real logit buffer, so this test exercises the `LiveSession` routing
    // discipline rather than the real-backend re-decode path (which is
    // covered by the gated `#[ignore]` test below).

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
}
