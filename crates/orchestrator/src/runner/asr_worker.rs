//! The ASR worker task: drains flush payloads from the runner's queue, transcribes each with the resolved ASR backend, and reports segments back to the runner.

use super::*;
use super::drain_loop::*;


/// The ASR worker runs as a dedicated `spawn_blocking` task. It drains
/// `FlushPayload`s from the flush queue, transcribes each with the ASR
/// runtime, emits `AppEvent::TranscriptSegment` events, and sends transcript
/// segments back to the runner via `writer_cmd_tx` for persistence.
///
/// `prebuilt_backend`: when `Some`, the provided backend is used directly
/// (test injection path). When `None`, the `AsrRuntime` is lazy-initialised
/// on the first flush from the `model_registry` (production path).
///
/// `n_gpu_layers`: the runtime-resolved GPU-offload count (from the
/// `gpu_acceleration` setting via [`resolve_gpu_layers`]) applied to the lazily
/// built production `AsrRuntime`.
///
/// `language`: the runtime-resolved ASR language hint (from the
/// `transcription_language` setting via [`resolve_transcription_language`]);
/// `None` = auto-detect. Passed by value into the one-shot `init_asr_backend`.
///
/// One flush is in-flight at a time: the worker pops one payload, processes
/// it fully, then waits for the next notification.
#[allow(clippy::too_many_arguments)]
pub(super) fn run_asr_worker(
    flush_queue: FlushQueue,
    prebuilt_backend: Option<Box<dyn AsrBackend + Send>>,
    model_registry: Arc<ModelRegistry>,
    n_gpu_layers: u32,
    language: Option<String>,
    engine: AsrEngine,
    event_tx: broadcast::Sender<AppEvent>,
    writer_cmd_tx: mpsc::Sender<WriterCommand>,
) {
    // Guard that sets `worker_exited` when this function returns for any reason
    // (clean exit or unwind). This ensures `wait_all_processed` is never wedged
    // by a dead worker that can no longer decrement `in_flight`.
    struct WorkerExitGuard<'a>(&'a Arc<AtomicBool>);
    impl Drop for WorkerExitGuard<'_> {
        fn drop(&mut self) {
            self.0.store(true, Ordering::Release);
        }
    }
    let _exit_guard = WorkerExitGuard(&flush_queue.worker_exited);

    // Use a single-threaded tokio runtime to allow async calls (model_registry.ensure,
    // Notify::notified) from within this spawn_blocking context.
    let rt = match tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
    {
        Ok(r) => r,
        Err(e) => {
            tracing::error!(target: "orchestrator", "ASR worker: failed to build runtime: {e}");
            return;
        }
    };

    // Production path: lazily-initialised ASR backend (Parakeet or a Qwen tier,
    // chosen by `engine`) wrapped in an Option.
    // Test path: the prebuilt backend is used directly.
    let mut lazy_runtime: Option<Box<dyn AsrBackend + Send>> = None;

    // Determine which backend to use: either the prebuilt one (test) or a lazy
    // production one. We use a closure-style dispatch below to avoid lifetimes
    // across the Option<Box<dyn AsrBackend>>.
    //
    // The prebuilt backend is moved into its own Option so we can mutably borrow
    // it separately from lazy_runtime.
    let mut prebuilt: Option<Box<dyn AsrBackend + Send>> = prebuilt_backend;

    // `pending_closed`: when true, skip the `notified()` wait and drain the
    // remaining queue immediately (runner has exited, no more notify_one calls).
    let mut pending_closed = false;

    loop {
        if !pending_closed {
            // Wait for a payload to be available, or the runner to signal closed.
            rt.block_on(flush_queue.notify.notified());
        }

        // Drain the queue fully before waiting on the next notification.
        // `Notify` coalesces any `notify_one()` calls that land while this
        // worker is busy processing into a single stored permit, so popping
        // only one item per wakeup would park the rest of a burst until some
        // UNRELATED later push notifies again (B3b) — instead loop popping
        // until the deque reports empty, then break out to wait again.
        loop {
            // Pop the oldest pending payload.
            let (payload, is_closed) = {
                let mut deque = flush_queue
                    .deque
                    .lock()
                    .expect("flush queue mutex poisoned");
                let p = deque.pop_front();
                let closed = flush_queue.closed.load(Ordering::Acquire);
                (p, closed)
            };

            // Once the runner signals closed we keep draining without waiting.
            if is_closed {
                pending_closed = true;
            }

            let payload = match payload {
                Some(p) => {
                    // Track that we have a payload in-flight; decremented via
                    // InFlightGuard below regardless of how this iteration exits.
                    flush_queue.in_flight.fetch_add(1, Ordering::AcqRel);
                    p
                }
                None => {
                    if pending_closed {
                        // Queue is empty and runner has exited — we're done.
                        tracing::debug!(
                            target: "orchestrator",
                            "ASR worker: flush queue closed and drained; exiting"
                        );
                        return;
                    }
                    // Deque drained; go back to waiting for the next notification.
                    break;
                }
            };

            // Guard that decrements `in_flight` when dropped (on any loop
            // iteration exit path: return, continue, or falling through).
            struct InFlightGuard<'a>(&'a Arc<AtomicUsize>);
            impl Drop for InFlightGuard<'_> {
                fn drop(&mut self) {
                    self.0.fetch_sub(1, Ordering::AcqRel);
                }
            }
            let _in_flight_guard = InFlightGuard(&flush_queue.in_flight);

            // Determine the backend to use.
            let backend: &mut dyn AsrBackend = if let Some(ref mut pb) = prebuilt {
                pb.as_mut()
            } else {
                // Lazy-initialise the ASR backend on the first flush (production
                // path). The engine (Parakeet vs a Qwen tier) was resolved at start
                // from the transcription-language setting.
                if lazy_runtime.is_none() {
                    // `init_asr_backend` runs at most once (lazy init), but the
                    // borrow checker can't see that across the loop, so clone the
                    // owned hint into the one-shot call.
                    let result = rt.block_on(init_asr_backend(
                        &model_registry,
                        engine,
                        n_gpu_layers,
                        language.clone(),
                    ));
                    match result {
                        Ok(Some(runtime)) => {
                            lazy_runtime = Some(runtime);
                        }
                        Ok(None) => {
                            // Model not available; skip this flush, keep draining.
                            continue;
                        }
                        Err(e) => {
                            tracing::warn!(target: "orchestrator", "ASR backend init failed: {e}");
                            let _ = event_tx.send(AppEvent::ErrorOccurred { error: e });
                            continue;
                        }
                    }
                }
                lazy_runtime.as_mut().expect("just initialised").as_mut()
            };

            // Wrap the per-flush call in `catch_unwind` so a panic inside
            // `transcribe_chunk` is caught, converted to `AppError::Internal`, and
            // emitted as `AppEvent::ErrorOccurred`. The worker then continues to the
            // next flush — one bad flush must not kill the worker or the recording.
            //
            // Per `architecture/cross-cutting.md`: "A panic inside a `spawn_blocking`
            // task must abort the parent orchestrator task and surface as a
            // recoverable `AppError`. The app does not exit on a single bad recording."
            //
            // `AssertUnwindSafe` is required because `&mut dyn AsrBackend`,
            // `broadcast::Sender`, and `mpsc::Sender` are not `UnwindSafe` by default.
            // We uphold the invariant: on unwind we do not use the backend again in
            // the same call (the closure is consumed), and the senders are only used
            // for a `send` before we return from the closure, so there is no risk of
            // leaving them in a broken intermediate state after an unwind.
            let call_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                process_flush_with_backend(
                    payload,
                    backend,
                    &event_tx,
                    &writer_cmd_tx,
                    &flush_queue.incomplete,
                )
            }));

            match call_result {
                Ok(Ok(())) => {}
                Ok(Err(e)) => {
                    // Backend returned an error; surface it and continue.
                    tracing::warn!(target: "orchestrator", "ASR flush error: {e}");
                    let _ = event_tx.send(AppEvent::ErrorOccurred { error: e });
                }
                Err(panic_payload) => {
                    // Backend panicked; extract the message if possible.
                    let msg = if let Some(s) = panic_payload.downcast_ref::<&str>() {
                        s.to_string()
                    } else if let Some(s) = panic_payload.downcast_ref::<String>() {
                        s.clone()
                    } else {
                        "unknown panic payload".to_string()
                    };
                    tracing::warn!(
                        target: "orchestrator",
                        "ASR worker caught panic in transcribe_chunk: {msg}; \
                         continuing to next flush"
                    );
                    let _ = event_tx.send(AppEvent::ErrorOccurred {
                        error: AppError::Internal {
                            context: format!("ASR worker panicked: {msg}"),
                        },
                    });
                }
            }
            // `_in_flight_guard` drops here, decrementing in_flight.
        }
    }
}

/// Resolve the ASR `n_gpu_layers` from the runtime `gpu_acceleration` setting.
///
/// GPU offload happens ONLY when BOTH (a) the build was compiled with a GPU
/// feature AND (b) the setting is on. `enabled == true` → the compile-time
/// ceiling [`asr_runtime::default_n_gpu_layers`] (which is already `0` in a
/// default CPU-only build, so a CPU build is unaffected by the flag);
/// `enabled == false` → `0` (force CPU even in a GPU build). Pure + unit-tested
/// so the wiring is verified without a model. See `architecture/cross-cutting.md`
/// — "GPU portability".
pub(crate) fn resolve_gpu_layers(enabled: bool) -> u32 {
    if enabled {
        asr_runtime::default_n_gpu_layers()
    } else {
        0
    }
}

/// Resolve the ASR language hint from the `transcription_language` setting.
///
/// Mirrors [`resolve_gpu_layers`]: a pure, model-free mapping from the settings
/// String to the `AsrRuntimeConfig.language: Option<String>` the runtime
/// consumes. The reserved sentinel `"auto"` (case-insensitive), the empty
/// string, and whitespace-only all map to `None` → no prefix → auto-detect
/// (byte-identical to the pre-feature behaviour). Any other value is a full
/// English language name, trimmed and forwarded verbatim → prefix-force.
pub(crate) fn resolve_transcription_language(setting: &str) -> Option<String> {
    let t = setting.trim();
    if t.is_empty() || t.eq_ignore_ascii_case("auto") {
        None
    } else {
        Some(t.to_string())
    }
}

/// Map the resolved [`AsrEngine`] to its `resources/models.json` model id.
pub(crate) fn engine_model_id(engine: AsrEngine) -> &'static str {
    match engine {
        AsrEngine::ParakeetEuV3 => "parakeet-tdt-0.6b-v3-int8",
        AsrEngine::Qwen06B => "qwen3-asr-0.6b-q8_0",
        AsrEngine::Qwen17B => "qwen3-asr-1.7b-q8_0",
    }
}

/// Lazily initialise the ASR backend chosen by `engine` on the first flush.
///
/// `engine` is resolved at recording start from the `transcription_language`
/// setting (and the GPU-model opt-in) via [`common::asr_engine_for_language`]:
/// Parakeet (`asr-parakeet`, sherpa-onnx) for the languages it covers, else a
/// Qwen tier (`asr-runtime`, llama-cpp-2 mtmd).
///
/// `n_gpu_layers` + `language` only affect the Qwen path — Parakeet is
/// multilingual auto-detect and runs on the ONNX-Runtime CPU EP.
///
/// Returns `Ok(None)` if the selected model is not yet available (caller should
/// skip the flush). Returns `Ok(Some(backend))` when initialisation succeeded.
pub(super) async fn init_asr_backend(
    model_registry: &ModelRegistry,
    engine: AsrEngine,
    n_gpu_layers: u32,
    language: Option<String>,
) -> AppResult<Option<Box<dyn AsrBackend + Send>>> {
    // Is a given model id locally `Available` (downloaded + hash-verified)? A
    // synchronous, no-network check — the live path must never download or block.
    let is_available = |id: &ModelId| -> bool {
        model_registry
            .list_models()
            .into_iter()
            .find(|s| &s.id == id)
            .map(|s| matches!(s.status, ModelStatusState::Available { .. }))
            .unwrap_or(false)
    };

    // Resolve the effective engine. The routed engine is preferred, but if its
    // model isn't downloaded yet we fall back to any available ASR engine rather
    // than silently producing no transcript (e.g. a fresh English install whose
    // onboarding fetched only the Qwen model still transcribes via Qwen until
    // Parakeet is downloaded). Preference keeps the broad Qwen-0.6B as the safety
    // net, then Parakeet, then the GPU Qwen.
    let routed_id = ModelId::from(engine_model_id(engine));
    let (engine, model_id) = if is_available(&routed_id) {
        (engine, routed_id)
    } else {
        let fallback = [AsrEngine::Qwen06B, AsrEngine::ParakeetEuV3, AsrEngine::Qwen17B]
            .into_iter()
            .filter(|&e| e != engine)
            .find(|&e| is_available(&ModelId::from(engine_model_id(e))));
        match fallback {
            Some(fb) => {
                tracing::warn!(
                    target: "orchestrator",
                    "routed ASR model {} not downloaded; falling back to {}",
                    routed_id.0,
                    engine_model_id(fb)
                );
                (fb, ModelId::from(engine_model_id(fb)))
            }
            None => {
                tracing::debug!(
                    target: "orchestrator",
                    "no ASR model available (routed {}); skipping flush",
                    routed_id.0
                );
                return Ok(None);
            }
        }
    };

    let model_dir = model_registry.ensure(&model_id).await.map_err(|e| {
        tracing::warn!(target: "orchestrator", "model ensure failed: {e}");
        e
    })?;

    match engine {
        AsrEngine::ParakeetEuV3 => {
            // sherpa-onnx offline transducer; reads the 4 model files by name.
            let backend = asr_parakeet::ParakeetBackend::new(
                asr_parakeet::ParakeetConfig::new(model_dir.clone()),
            )?;
            tracing::info!(
                target: "orchestrator",
                "Parakeet ASR initialised from {}",
                model_dir.display()
            );
            Ok(Some(Box::new(backend)))
        }
        AsrEngine::Qwen06B | AsrEngine::Qwen17B => {
            // llama-cpp-2 mtmd. The convention is the first .gguf file in the
            // model dir; mmproj is the file whose name contains "mmproj".
            let gguf_path = find_file_in_dir(&model_dir, |name| {
                name.ends_with(".gguf") && !name.contains("mmproj")
            })?;
            let mmproj_path = find_file_in_dir(&model_dir, |name| name.contains("mmproj"))?;

            let config = AsrRuntimeConfig {
                n_gpu_layers,
                language,
                ..AsrRuntimeConfig::default()
            };
            match AsrRuntime::new(&gguf_path, &mmproj_path, config) {
                Ok(runtime) => {
                    tracing::info!(
                        target: "orchestrator",
                        "Qwen ASR runtime initialised from {}",
                        model_dir.display()
                    );
                    Ok(Some(Box::new(runtime)))
                }
                Err(e) => {
                    tracing::warn!(target: "orchestrator", "ASR runtime init failed: {e}");
                    Err(e)
                }
            }
        }
    }
}

/// Process one flush payload with the provided ASR backend.
pub(super) fn process_flush_with_backend(
    payload: FlushPayload,
    backend: &mut dyn AsrBackend,
    event_tx: &broadcast::Sender<AppEvent>,
    writer_cmd_tx: &mpsc::Sender<WriterCommand>,
    incomplete: &AtomicBool,
) -> AppResult<()> {
    if payload.vad_segments.is_empty() {
        return Ok(());
    }

    let chunk = AudioChunk {
        samples: payload.samples,
        sample_rate: SAMPLE_RATE_HZ as u32,
        start_ms: payload.vad_segments.first().map(|(s, _)| *s).unwrap_or(0),
        end_ms: payload.vad_segments.last().map(|(_, e)| *e).unwrap_or(0),
    };

    let chunk_segments = match backend.transcribe_chunk(&chunk) {
        Ok(segs) => segs,
        Err(e) => {
            tracing::warn!(target: "orchestrator", "transcribe_chunk failed: {e}");
            return Err(e);
        }
    };

    // Combine all returned segment text into one string for proportional split.
    let combined_text: String = chunk_segments
        .iter()
        .map(|s| s.text.as_str())
        .collect::<Vec<_>>()
        .join(" ");

    // Re-split proportionally across the VAD sub-segments. Each emitted
    // sub-Segment inherits the live speaker label of its originating VAD
    // segment (Phase B) — the proportional re-split preserves one output
    // Segment per input vad_segment in order, so the 1:1 label mapping holds.
    let sub_segments =
        emit_segments_proportional(&combined_text, &payload.vad_segments, &payload.speaker_ids);

    for seg in sub_segments {
        // Emit broadcast event.
        let _ = event_tx.send(AppEvent::TranscriptSegment {
            meeting_id: payload.meeting_id,
            segment: seg.clone(),
        });
        // Send to runner for persistence write. This segment was already
        // broadcast above (the live transcript view has it), so a dropped send
        // here means it silently vanishes from `transcript.json` unless flagged:
        // set `incomplete` so the post-stop re-transcribe restores it from
        // `audio.opus` (the same repair the drop-oldest queue path drives).
        if let Err(e) = writer_cmd_tx.try_send(WriterCommand::WriteSegment(seg)) {
            incomplete.store(true, Ordering::Release);
            tracing::warn!(
                target: "orchestrator",
                "writer command channel full or closed; transcript segment dropped from persistence: {e}"
            );
        }
    }

    Ok(())
}

/// Wait for the ASR worker to finish processing all queued flushes.
///
/// After dispatching the final flush before `stop`, the runner calls this to
/// ensure the ASR worker has processed all pending items and sent back all
/// `WriterCommand::WriteSegment` commands before `finalise()` is called.
///
/// Polls until both the queue is empty AND no payload is in-flight, draining
/// writer commands as they arrive. Times out after 30 s (covers the slowest
/// expected inference on the target hardware).
pub(super) fn wait_for_asr_worker_drain(
    flush_queue: &FlushQueue,
    writer_cmd_rx: &mut mpsc::Receiver<WriterCommand>,
    writer: &mut MeetingWriter,
) {
    let timeout = Duration::from_secs(30);
    let poll_interval = Duration::from_millis(20);
    let drained = flush_queue.wait_all_processed(timeout, poll_interval);
    if !drained {
        // Tail loss: the remaining queued/in-flight audio could not be
        // transcribed in time. Mark incomplete so ipc-bridge re-transcribes the
        // complete audio in the background (the audio itself is fully captured).
        flush_queue.incomplete.store(true, Ordering::Release);
        tracing::warn!(
            target: "orchestrator",
            "ASR worker did not drain within 30 s; finalising now — a background \
             re-transcribe of the complete audio will repair the transcript"
        );
    }
    // Drain any writer commands that arrived while we were waiting.
    drain_writer_commands(writer_cmd_rx, writer);
}

/// Drain any pending `WriterCommand`s from the ASR worker and apply them.
pub(super) fn drain_writer_commands(
    rx: &mut mpsc::Receiver<WriterCommand>,
    writer: &mut MeetingWriter,
) {
    while let Ok(cmd) = rx.try_recv() {
        match cmd {
            WriterCommand::WriteSegment(seg) => {
                if let Err(e) = writer.write_transcript_segment(seg) {
                    tracing::warn!(
                        target: "orchestrator",
                        "write_transcript_segment failed: {e}"
                    );
                }
            }
        }
    }
}

/// Run the end-of-stream finalisation shared by the Recording-stop and
/// Paused-stop branches (TIMELINE-DRIFT #6).
///
/// Both stop paths must flush the VAD end-of-stream (so the last in-progress
/// utterance is closed and dispatched) and drain the accumulator before
/// finalising the writer. Previously only the Recording-stop branch did this;
/// stopping while Paused finalised directly and silently lost the last
/// utterance. This helper centralises the flush → accumulator-drain → ASR-worker
/// wait → transcript-write-drain → `writer.finalise` sequence so the two
/// branches cannot diverge again.
///
/// The caller is responsible for any sample/meter draining that is specific to
/// its branch (the Recording branch drains the live sample stream first; the
/// Paused branch has no pending samples to drain).
#[allow(clippy::too_many_arguments)]
pub(super) fn finalise_on_stop(
    meta: Box<MeetingMeta>,
    reply: oneshot::Sender<AppResult<MeetingMeta>>,
    vad_opt: &mut Option<VadChunker>,
    acc: &mut Accumulator,
    flush_queue: &FlushQueue,
    writer_cmd_rx: &mut mpsc::Receiver<WriterCommand>,
    mut writer: MeetingWriter,
    meeting_id: MeetingId,
    online_diarizer: Option<&Arc<OnlineDiarizer>>,
) {
    // End-of-stream VAD flush closes any in-progress segment.
    if let Some(vad) = vad_opt {
        match vad.flush_end_of_stream() {
            Ok(events) => {
                for ev in events {
                    if let VadEvent::SegmentEnd { start_ms, end_ms, samples } = ev {
                        // Live label from the un-padded slice (Phase B); this
                        // end-of-stream flush is a SegmentEnd site too.
                        let label = live_segment_label(online_diarizer, &samples);
                        acc.append(start_ms, end_ms, &samples, label);
                    }
                }
            }
            Err(e) => {
                tracing::warn!(
                    target: "orchestrator",
                    "VAD flush_end_of_stream error: {e}"
                );
            }
        }
    }

    // Flush any remaining accumulator content before finalising.
    if !acc.is_empty() {
        let (samples, vad_segments, speaker_ids) = acc.drain();
        let payload = FlushPayload {
            samples,
            vad_segments,
            speaker_ids,
            meeting_id,
        };
        // Final stop-time drain: use the inner enqueue directly. If it drops
        // under a full queue the `incomplete` flag is still set (→ post-stop
        // re-transcribe), but no `AsrBackpressure` event is emitted — the
        // recording is ending, so the co-pilot cooldown it drives is moot.
        let _ = dispatch_flush_inner(flush_queue, payload);
    }

    // Wait for the ASR worker to process any remaining flushes so that
    // transcript.json is fully written before finalise.
    wait_for_asr_worker_drain(flush_queue, writer_cmd_rx, &mut writer);

    // Drain any remaining transcript write commands.
    drain_writer_commands(writer_cmd_rx, &mut writer);

    let unboxed = *meta;
    let result = writer.finalise(unboxed.clone()).map(|_| unboxed);
    let _ = reply.send(result);
    tracing::info!(target: "orchestrator", "runner: exiting cleanly");
}

/// Find a file in `dir` matching `predicate`. Returns `AppError::Internal` if
/// the directory cannot be read or no matching file is found.
pub(crate) fn find_file_in_dir(
    dir: &std::path::Path,
    predicate: impl Fn(&str) -> bool,
) -> AppResult<std::path::PathBuf> {
    let entries = std::fs::read_dir(dir).map_err(|e| AppError::Internal {
        context: format!("read_dir {}: {e}", dir.display()),
    })?;
    for entry in entries.flatten() {
        let path = entry.path();
        if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
            if predicate(name) {
                return Ok(path);
            }
        }
    }
    Err(AppError::Internal {
        context: format!("no matching file found in {}", dir.display()),
    })
}

