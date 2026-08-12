//! The blocking drain loop: pulls sample batches + VAD events from the capture streams, feeds the batched-VAD accumulator, dispatches flushes to the ASR worker, and applies live diarization labels (Phase B).

use super::*;
use super::asr_worker::*;
use super::audio_helpers::*;

// ---------------------------------------------------------------------------
// Live diarization label (Phase B)
// ---------------------------------------------------------------------------

/// Compute the live provisional speaker label for one VAD segment's un-padded
/// samples (Phase B).
///
/// Returns `None` when no live diarizer is wired (`online_diarizer == None`) or
/// when `assign_segment` fails for this segment (FFI error, mutex poison, empty
/// or invalid segment). A failure is logged at `warn` and degrades to `None` for
/// THAT segment only — subsequent segments keep trying, and recording /
/// transcription are never affected. The label MUST be computed here, at
/// SegmentEnd, from the original un-padded per-segment slice: the accumulator's
/// `MAX_GAP_MS` zero-pad cap makes per-segment sample boundaries unrecoverable
/// from the flushed buffer downstream.
pub(super) fn live_segment_label(
    online_diarizer: Option<&Arc<OnlineDiarizer>>,
    samples: &[f32],
) -> Option<String> {
    let diarizer = online_diarizer?;
    // Live diarization is strictly additive and runs INLINE on the runner's
    // drain-loop thread (the one that also drains the sample channel and writes
    // audio.opus). A panic in the sherpa FFI (or any unwinding panic in
    // `assign_segment`) must therefore NOT escape this function and abort the
    // drain loop — that would stop recording + transcription. `catch_unwind`
    // contains it; the `Mutex` inside `OnlineDiarizer` poisons on a
    // panic-while-locked, so any subsequent segment cleanly returns `Err` (and
    // hence `None`) rather than re-panicking. Either way: no live label for the
    // affected segment, recording unaffected. (`AssertUnwindSafe` is sound here
    // because we discard the diarizer's state on a panic — we never observe a
    // torn value; we only map the outcome to `Option<String>`.)
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        diarizer.assign_segment(samples, SAMPLE_RATE_HZ as u32)
    }));
    match outcome {
        Ok(Ok(label)) => Some(label),
        Ok(Err(e)) => {
            tracing::warn!(
                target: "orchestrator",
                "live diarization assign_segment failed: {e}"
            );
            None
        }
        Err(_) => {
            tracing::error!(
                target: "orchestrator",
                "live diarization panicked; continuing without a live label"
            );
            None
        }
    }
}

// ---------------------------------------------------------------------------
// Blocking drain loop (runner)
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
pub(super) fn run_drain_loop(
    mut streams: AudioStreams,
    mut writer: MeetingWriter,
    event_tx: broadcast::Sender<AppEvent>,
    mut cmd_rx: mpsc::Receiver<RunnerCommand>,
    flush_queue: FlushQueue,
    mut writer_cmd_rx: mpsc::Receiver<WriterCommand>,
    meeting_id: MeetingId,
    online_diarizer: Option<Arc<OnlineDiarizer>>,
) {
    // Ensure the worker is signalled when this function exits for any reason.
    // `FlushQueueGuard` calls `close()` when dropped.
    struct FlushQueueGuard<'a>(&'a FlushQueue);
    impl Drop for FlushQueueGuard<'_> {
        fn drop(&mut self) {
            self.0.close();
        }
    }
    let _guard = FlushQueueGuard(&flush_queue);
    let mut writer_paused = false;
    let mut acc = Accumulator::new();
    let latency_window = Duration::from_secs_f32(LATENCY_WINDOW_SECS);

    // Throttle for `AppEvent::RecordingClock`. The notes editor stamps
    // paragraph anchors from this event (see `architecture/cross-cutting.md`
    // "Notes paragraph-anchor clock"); ~5 Hz is plenty for that and avoids
    // flooding the broadcast channel. We only emit on the sample-batch receive
    // path below — the paused branch never reaches that path, so the clock is
    // naturally not advanced while paused, matching the pause-EXCLUDING sample
    // clock carried by `batch.end_ms`.
    //
    // Initialised so the first received batch emits immediately rather than
    // waiting out the first throttle window.
    let clock_emit_interval = Duration::from_millis(RECORDING_CLOCK_MIN_INTERVAL_MS);
    let mut last_clock_emit: Option<Instant> = None;

    // Attempt to open the VAD chunker. If the model file is missing we log
    // a warning and operate without VAD (no ASR transcription).
    let mut vad_opt: Option<VadChunker> = {
        let model_path = vad_chunker::default_model_path();
        match VadChunker::open(&model_path, VadConfig::default()) {
            Ok(c) => {
                tracing::info!(
                    target: "orchestrator",
                    "VAD chunker opened from {}",
                    model_path.display()
                );
                Some(c)
            }
            Err(e) => {
                tracing::warn!(
                    target: "orchestrator",
                    "VAD chunker unavailable (model missing?): {e}; ASR disabled for this session"
                );
                None
            }
        }
    };

    loop {
        // --- 1. Poll commands (non-blocking) ---
        loop {
            match cmd_rx.try_recv() {
                Ok(RunnerCommand::WriterPause) => {
                    if let Err(e) = writer.pause() {
                        tracing::warn!(
                            target: "orchestrator",
                            "MeetingWriter::pause failed: {e}"
                        );
                    }
                    writer_paused = true;
                    tracing::debug!(target: "orchestrator", "runner: writer paused");
                }
                Ok(RunnerCommand::WriterResume) => {
                    if let Err(e) = writer.resume() {
                        tracing::warn!(
                            target: "orchestrator",
                            "MeetingWriter::resume failed: {e}"
                        );
                    }
                    writer_paused = false;
                    tracing::debug!(target: "orchestrator", "runner: writer resumed");
                }
                Ok(RunnerCommand::Stop { meta, reply }) => {
                    tracing::info!(
                        target: "orchestrator",
                        meeting_id = %meta.uuid.0,
                        "runner: stop command received; draining remaining samples"
                    );
                    if !writer_paused {
                        drain_samples_through_vad(
                            &mut streams.samples,
                            &mut writer,
                            &mut vad_opt,
                            &mut acc,
                            online_diarizer.as_ref(),
                            "stop drain",
                        );
                    }
                    drain_meter(&mut streams.meter, &event_tx);

                    finalise_on_stop(
                        meta,
                        reply,
                        &mut vad_opt,
                        &mut acc,
                        &flush_queue,
                        &mut writer_cmd_rx,
                        writer,
                        meeting_id,
                        online_diarizer.as_ref(),
                    );
                    return;
                }
                Err(mpsc::error::TryRecvError::Empty) => break,
                Err(mpsc::error::TryRecvError::Disconnected) => {
                    tracing::warn!(
                        target: "orchestrator",
                        "runner: command channel disconnected; exiting"
                    );
                    return;
                }
            }
        }

        // --- 2. If writer is paused, block on the next command ---
        if writer_paused {
            match cmd_rx.blocking_recv() {
                Some(RunnerCommand::WriterPause) => {
                    // Already paused — no-op.
                }
                Some(RunnerCommand::WriterResume) => {
                    if let Err(e) = writer.resume() {
                        tracing::warn!(
                            target: "orchestrator",
                            "MeetingWriter::resume failed: {e}"
                        );
                    }
                    writer_paused = false;
                    tracing::debug!(target: "orchestrator", "runner: writer resumed");
                }
                Some(RunnerCommand::Stop { meta, reply }) => {
                    tracing::info!(
                        target: "orchestrator",
                        meeting_id = %meta.uuid.0,
                        "runner: stop while writer paused; draining queued samples then finalising"
                    );
                    // Drain any sample batches that were queued before the pause
                    // arrived: push each to persistent audio AND feed it to the VAD.
                    // Batches accepted by the channel before the WriterPause command
                    // would otherwise be stranded here (the paused loop blocks on
                    // cmd_rx and never reads streams.samples), so the VAD would
                    // never build an in-progress segment for them. `finalise_on_stop`
                    // then flushes the end-of-stream to close that segment.
                    drain_samples_through_vad(
                        &mut streams.samples,
                        &mut writer,
                        &mut vad_opt,
                        &mut acc,
                        online_diarizer.as_ref(),
                        "paused-stop drain",
                    );
                    drain_meter(&mut streams.meter, &event_tx);
                    finalise_on_stop(
                        meta,
                        reply,
                        &mut vad_opt,
                        &mut acc,
                        &flush_queue,
                        &mut writer_cmd_rx,
                        writer,
                        meeting_id,
                        online_diarizer.as_ref(),
                    );
                    return;
                }
                None => {
                    tracing::warn!(
                        target: "orchestrator",
                        "runner: command channel closed while paused; exiting"
                    );
                    return;
                }
            }
            continue;
        }

        // --- 3. Receive one sample batch ---
        match streams.samples.try_recv() {
            Ok(batch) => {
                // Push to persistent audio.
                push_batch(&mut writer, &batch);

                // Emit a throttled RecordingClock (~5 Hz) so the notes editor
                // can stamp paragraph anchors on the pause-EXCLUDING sample
                // clock. `batch.end_ms` is exactly that clock (it advances only
                // while samples are flowing; the paused branch never reaches
                // here). See `architecture/cross-cutting.md` — "Notes
                // paragraph-anchor clock".
                let now = Instant::now();
                let should_emit_clock = last_clock_emit
                    .map(|t| now.duration_since(t) >= clock_emit_interval)
                    .unwrap_or(true);
                if should_emit_clock {
                    let _ = event_tx.send(AppEvent::RecordingClock {
                        meeting_id,
                        clock_ms: batch.end_ms,
                    });
                    last_clock_emit = Some(now);
                }

                // Feed to VAD if available.
                if let Some(ref mut vad) = vad_opt {
                    match vad.process_samples(&batch.samples, batch.start_ms) {
                        Ok(events) => {
                            for ev in events {
                                if let VadEvent::SegmentEnd { start_ms, end_ms, samples } = ev {
                                    // Live label from the still-un-padded
                                    // per-segment slice (Phase B).
                                    let label =
                                        live_segment_label(online_diarizer.as_ref(), &samples);
                                    acc.append(start_ms, end_ms, &samples, label);
                                }
                            }
                        }
                        Err(e) => {
                            tracing::warn!(
                                target: "orchestrator",
                                "VAD process_samples error: {e}"
                            );
                        }
                    }
                }

                // Size-triggered flush.
                if acc.duration_secs() >= FLUSH_MIN_SECS {
                    let (samples, vad_segments, speaker_ids) = acc.drain();
                    let payload = FlushPayload {
                        samples,
                        vad_segments,
                        speaker_ids,
                        meeting_id,
                    };
                    dispatch_flush(&flush_queue, payload, &event_tx, meeting_id);
                }
            }
            Err(mpsc::error::TryRecvError::Empty) => {
                // Nothing queued; check latency window, then sleep briefly.
                let should_flush_latency = !acc.is_empty()
                    && acc
                        .last_vad_end_at
                        .map(|t| t.elapsed() >= latency_window)
                        .unwrap_or(false);

                if should_flush_latency {
                    tracing::debug!(target: "orchestrator", "runner: latency-window flush");
                    let (samples, vad_segments, speaker_ids) = acc.drain();
                    let payload = FlushPayload {
                        samples,
                        vad_segments,
                        speaker_ids,
                        meeting_id,
                    };
                    dispatch_flush(&flush_queue, payload, &event_tx, meeting_id);
                } else {
                    std::thread::sleep(POLL_INTERVAL);
                }
            }
            Err(mpsc::error::TryRecvError::Disconnected) => {
                tracing::info!(
                    target: "orchestrator",
                    "runner: samples channel closed; draining meter and waiting for stop"
                );
                drain_meter(&mut streams.meter, &event_tx);
                if let Some(RunnerCommand::Stop { meta, reply }) = cmd_rx.blocking_recv() {
                    // End-of-stream VAD flush.
                    if let Some(ref mut vad) = vad_opt {
                        match vad.flush_end_of_stream() {
                            Ok(events) => {
                                for ev in events {
                                    if let VadEvent::SegmentEnd { start_ms, end_ms, samples } = ev
                                    {
                                        // Live label from the un-padded slice (Phase B).
                                        let label = live_segment_label(
                                            online_diarizer.as_ref(),
                                            &samples,
                                        );
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
                    if !acc.is_empty() {
                        let (samples, vad_segments, speaker_ids) = acc.drain();
                        let payload = FlushPayload {
                            samples,
                            vad_segments,
                            speaker_ids,
                            meeting_id,
                        };
                        dispatch_flush(&flush_queue, payload, &event_tx, meeting_id);
                    }
                    // Wait for the ASR worker to process any remaining flushes.
                    wait_for_asr_worker_drain(
                        &flush_queue,
                        &mut writer_cmd_rx,
                        &mut writer,
                    );
                    // Drain any remaining transcript write commands.
                    drain_writer_commands(&mut writer_cmd_rx, &mut writer);
                    let unboxed = *meta;
                    let result = writer.finalise(unboxed.clone()).map(|_| unboxed);
                    let _ = reply.send(result);
                }
                tracing::info!(target: "orchestrator", "runner: exiting after samples closed");
                return;
            }
        }

        // --- 4. Forward any pending meter frames ---
        drain_meter_nonblocking(&mut streams.meter, &event_tx);

        // --- 5. Drain any transcript write commands from the ASR worker ---
        drain_writer_commands(&mut writer_cmd_rx, &mut writer);
    }
}

// ---------------------------------------------------------------------------
// Flush dispatch helper
// ---------------------------------------------------------------------------

/// Push a flush payload to the ASR worker's bounded drop-oldest queue.
///
/// If the queue is already at capacity (`FLUSH_CHANNEL_CAP`), the **oldest**
/// pending flush (the entry at the front of the deque) is removed and
/// discarded, then the new payload is pushed to the back.
///
/// This is self-healing and log-only — NOT surfaced to the UI as an error (it
/// can fire repeatedly under sustained load, e.g. CPU-only ASR). Audio is always
/// preserved in `audio.opus`; the dropped flush's transcript is restored by the
/// post-stop re-transcribe that the `incomplete` flag triggers.
fn dispatch_flush(
    flush_queue: &FlushQueue,
    payload: FlushPayload,
    event_tx: &broadcast::Sender<AppEvent>,
    meeting_id: MeetingId,
) {
    if dispatch_flush_inner(flush_queue, payload) {
        // A pending flush was dropped under backpressure. Emit a NON-error
        // signal (NOT `ErrorOccurred`): the live co-pilot driver observes it and
        // pauses its transcript-turn cadence so its own decodes do not compound
        // the backpressure; the webview ignores it. The dropped audio survives
        // in `audio.opus` and the `incomplete` flag drives the post-stop
        // re-transcribe. Send is best-effort (no subscribers → dropped), like
        // the other event emissions in this loop.
        let _ = event_tx.send(AppEvent::AsrBackpressure { meeting_id });
    }
}

/// Public wrapper for `dispatch_flush` used by tests. Returns whether a pending
/// flush was dropped under backpressure.
///
/// Only available under the `test-source` feature (or in `#[cfg(test)]`).
#[cfg(any(test, feature = "test-source"))]
pub(crate) fn dispatch_flush_pub(flush_queue: &FlushQueue, payload: FlushPayload) -> bool {
    dispatch_flush_inner(flush_queue, payload)
}

/// Enqueue `payload`, dropping the oldest pending flush first if the bounded
/// queue is already full. Returns `true` iff a pending flush was dropped
/// (backpressure), so the caller can emit [`AppEvent::AsrBackpressure`].
pub(super) fn dispatch_flush_inner(flush_queue: &FlushQueue, payload: FlushPayload) -> bool {
    let mut deque = flush_queue
        .deque
        .lock()
        .expect("flush queue mutex poisoned");

    let mut dropped = false;
    if deque.len() >= FLUSH_CHANNEL_CAP {
        // Drop the OLDEST pending flush (the front of the queue).
        let _dropped = deque.pop_front();
        // Mark the live transcript incomplete so ipc-bridge runs a background
        // re-transcribe of the complete audio after stop (the dropped flush's
        // audio survives in audio.opus; only its transcript was lost).
        flush_queue.incomplete.store(true, Ordering::Release);
        dropped = true;
        // Self-healing backpressure: the audio survives in `audio.opus` and the
        // `incomplete` flag above triggers a full re-transcribe after stop, so
        // this is a log-only WARN — NOT surfaced to the UI as an error (it can
        // fire repeatedly under sustained load, e.g. CPU-only ASR). The caller
        // additionally emits a non-error `AsrBackpressure` event for the co-pilot.
        tracing::warn!(
            target: "orchestrator",
            "ASR flush queue full (backpressure); dropping oldest pending flush \
             (audio.opus unaffected; restored by the post-stop re-transcribe)"
        );
    }

    deque.push_back(payload);
    drop(deque);

    // Signal the worker that a new payload is available.
    flush_queue.notify.notify_one();
    tracing::debug!(target: "orchestrator", "runner: flush dispatched to ASR worker");

    dropped
}

