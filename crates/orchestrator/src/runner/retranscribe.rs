//! Offline re-transcription: re-runs the live pipeline's batched-VAD + ASR-dispatch machinery over an already-decoded PCM buffer.

use super::*;

/// Re-run transcription over a fully-decoded PCM buffer, reusing the live
/// pipeline's batched-VAD [`Accumulator`] + ASR-dispatch machinery.
///
/// This is the offline counterpart of [`run_drain_loop`]: instead of draining a
/// live capture stream, it feeds an already-decoded `pcm` buffer (the
/// pause-INCLUDING 16 kHz mono samples from
/// `persistence::reader::read_audio_pcm`) through the **same** `VadChunker` →
/// [`Accumulator`] → [`process_flush_with_backend`]-equivalent path. The 30 s
/// encoder-window constraint and the silence-preservation rule therefore hold
/// identically — the accumulator zero-pads inter-utterance gaps (capped at
/// `MAX_GAP_MS`) exactly as the live path does.
///
/// Differences from the live path:
/// - No flush queue / ASR worker thread: the work runs synchronously on the
///   caller's `spawn_blocking` thread, one accumulator flush at a time, so the
///   produced segments can be collected in order and returned to the caller for
///   `transcript.json` rewrite + index upsert.
/// - `AppEvent::TranscriptSegment` is emitted as each segment is produced
///   (same event the live path emits), so the webview's transcript pane
///   appends them live.
///
/// # Pause-EXCLUDING timeline (TIMELINE-DRIFT #4)
///
/// `audio.opus` is recorded **pause-INCLUDING**: the encoder pads every pause
/// with synthesised silence, so `read_audio_pcm` returns a buffer whose duration
/// equals wall-clock recording time. The live transcript, however, is on the
/// **pause-EXCLUDING** capture-sample clock (`Segment::start_ms` —
/// `architecture/cross-cutting.md` "Notes paragraph-anchor clock"): the capture
/// forwarder's sample clock does not advance while paused, so the live VAD never
/// sees the pause silence and post-pause segments continue contiguously on the
/// pause-excluding timeline.
///
/// To produce timestamps that match the live ones, the offline feeder must
/// reproduce that pause-excluding clock. Pause boundaries are **not** persisted
/// anywhere (`MeetingMeta` has no pause map), but the encoder-synthesised pause
/// padding is recoverable from the audio itself: it is a contiguous run of
/// near-silent samples far longer than any natural inter-utterance gap (a user
/// pause is typically many seconds; the live accumulator never zero-pads more
/// than `MAX_GAP_MS`). The feeder therefore detects long near-silent runs
/// (`PAUSE_*` constants below), **skips** them, and feeds only the non-pause
/// audio to the VAD with a clock that advances over kept audio only — exactly
/// reconstructing the pause-excluding capture clock. Combined with the
/// VAD-realignment fix (#3) this yields `Segment::start_ms` matching the live
/// path within frame tolerance.
///
/// Natural quiet gaps (mic live, room quiet) are NOT skipped: they are below the
/// `PAUSE_MIN_MS` length threshold, so they stay on the timeline just as the
/// live capture clock counts them.
///
/// Returns all produced segments in start-time order.
pub(crate) fn re_transcribe_buffer(
    pcm: &[f32],
    backend: &mut dyn AsrBackend,
    event_tx: &broadcast::Sender<AppEvent>,
    meeting_id: MeetingId,
) -> AppResult<Vec<Segment>> {
    // Same VAD model + config the live path uses. If the VAD model is missing
    // we cannot segment the audio, so re-transcribe yields no segments.
    let model_path = vad_chunker::default_model_path();
    let mut vad = VadChunker::open(&model_path, VadConfig::default()).map_err(|e| {
        tracing::warn!(
            target: "orchestrator",
            "re_transcribe: VAD chunker unavailable ({e}); cannot segment audio"
        );
        e
    })?;

    let mut acc = Accumulator::new();
    let mut produced: Vec<Segment> = Vec::new();

    // Build the pause-excluding feed: the kept (non-pause) sample ranges, each
    // tagged with its start offset on the pause-EXCLUDING clock.
    let kept = pause_excluding_segments(pcm);

    // Live-test UX T4(a): determinate progress = kept samples FED to the VAD so
    // far / total kept samples. Counted on the kept feed (not raw `pcm`) so the
    // skipped pause padding does not stall the bar; emitted per accumulator flush
    // (the natural cadence at which produced segments stream out).
    let total_kept_samples: usize = kept.iter().map(|r| r.src_end - r.src_start).sum();
    let mut samples_fed: usize = 0;
    // Emit a 0.0 start so the UI shows the bar immediately rather than waiting
    // for the first flush (which can be seconds of inference away).
    emit_retranscribe_progress(event_tx, meeting_id, samples_fed, total_kept_samples);

    // 1600 samples = 100 ms at 16 kHz — the same batch granularity the live
    // feeder uses, so VAD framing is identical to the live path.
    const BATCH_SAMPLES: usize = 1600;
    let n_regions = kept.len();
    for (region_idx, region) in kept.iter().enumerate() {
        let region_pcm = &pcm[region.src_start..region.src_end];
        // `start_ms` runs on the pause-EXCLUDING clock: it begins at the
        // region's excluding-offset and advances only over kept audio. At a
        // region boundary (a skipped pause) it continues from where the
        // previous region left off, so the VAD sees a contiguous pause-excluding
        // timeline — matching the live capture clock.
        let mut start_ms = region.excl_start_ms;
        for chunk in region_pcm.chunks(BATCH_SAMPLES) {
            let duration_ms = (chunk.len() as u64 * 1000) / SAMPLE_RATE_HZ;
            match vad.process_samples(chunk, start_ms) {
                Ok(events) => {
                    for ev in events {
                        if let VadEvent::SegmentEnd { start_ms, end_ms, samples } = ev {
                            // Offline re-transcribe is a distinct path: the
                            // on-stop SherpaDiarizer pass owns labels here, so no
                            // live label is assigned (Phase B).
                            acc.append(start_ms, end_ms, &samples, None);
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!(target: "orchestrator", "re_transcribe VAD error: {e}");
                }
            }
            start_ms += duration_ms;
            samples_fed += chunk.len();

            // Size-triggered flush, identical threshold to the live path.
            if acc.duration_secs() >= FLUSH_MIN_SECS {
                let (samples, vad_segments, speaker_ids) = acc.drain();
                transcribe_one_flush(
                    samples,
                    vad_segments,
                    speaker_ids,
                    backend,
                    event_tx,
                    meeting_id,
                    &mut produced,
                )?;
                emit_retranscribe_progress(
                    event_tx,
                    meeting_id,
                    samples_fed,
                    total_kept_samples,
                );
            }
        }

        // Pause boundary (between kept regions): close any in-progress VAD
        // segment here and reset the chunker before the next region. The
        // skipped ≥`PAUSE_MIN_MS` silence WOULD have closed the segment in the
        // live path (its hangover fires on ≥720 ms of silence); because the
        // offline feeder skips that silence to reconstruct the pause-EXCLUDING
        // clock, it must close the segment explicitly — otherwise the VAD,
        // seeing the pre-pause region's last speech sample immediately followed
        // by the post-pause region's first, MERGES the two utterances into one
        // segment with the pre-pause start time (TIMELINE-DRIFT #4: the offline
        // segmentation must match the live path's, which splits at the pause).
        if region_idx + 1 < n_regions {
            match vad.flush_end_of_stream() {
                Ok(events) => {
                    for ev in events {
                        if let VadEvent::SegmentEnd { start_ms, end_ms, samples } = ev {
                            acc.append(start_ms, end_ms, &samples, None);
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!(
                        target: "orchestrator",
                        "re_transcribe region-boundary flush error: {e}"
                    );
                }
            }
            vad.reset();
        }
    }

    // End-of-stream VAD flush closes any in-progress segment.
    match vad.flush_end_of_stream() {
        Ok(events) => {
            for ev in events {
                if let VadEvent::SegmentEnd { start_ms, end_ms, samples } = ev {
                    // Offline path: no live label (the on-stop pass owns it).
                    acc.append(start_ms, end_ms, &samples, None);
                }
            }
        }
        Err(e) => {
            tracing::warn!(target: "orchestrator", "re_transcribe flush_end_of_stream error: {e}");
        }
    }

    // Final flush of whatever remains in the accumulator.
    if !acc.is_empty() {
        let (samples, vad_segments, speaker_ids) = acc.drain();
        transcribe_one_flush(
            samples,
            vad_segments,
            speaker_ids,
            backend,
            event_tx,
            meeting_id,
            &mut produced,
        )?;
    }

    // Terminal 1.0 so the bar visibly completes; the subsequent
    // `AppEvent::TranscriptReady` clears the per-row indicator.
    emit_retranscribe_progress(event_tx, meeting_id, total_kept_samples, total_kept_samples);

    Ok(produced)
}

/// Determinate progress fraction for an offline re-transcribe: kept samples fed
/// to the VAD so far / total kept samples (live-test UX T4(a)). Pure +
/// unit-tested. Clamped to `0.0..=1.0`; a zero total (empty/all-pause audio)
/// reports `1.0` (nothing to do is "done").
pub(crate) fn re_transcribe_fraction(samples_fed: usize, total_kept_samples: usize) -> f32 {
    if total_kept_samples == 0 {
        return 1.0;
    }
    (samples_fed as f32 / total_kept_samples as f32).clamp(0.0, 1.0)
}

/// Emit a determinate `AppEvent::OperationProgress` for the re-transcribe op.
fn emit_retranscribe_progress(
    event_tx: &broadcast::Sender<AppEvent>,
    meeting_id: MeetingId,
    samples_fed: usize,
    total_kept_samples: usize,
) {
    let _ = event_tx.send(AppEvent::OperationProgress {
        meeting_id,
        op: minutist_common::OperationKind::ReTranscribe,
        fraction: Some(re_transcribe_fraction(samples_fed, total_kept_samples)),
        label: "Re-transcribing…".to_string(),
    });
}

/// Transcribe a single accumulator flush synchronously and append the produced
/// segments to `out`, emitting an `AppEvent::TranscriptSegment` for each.
///
/// Mirrors [`process_flush_with_backend`] (same `AudioChunk` construction +
/// proportional re-split) but collects the segments for the offline caller
/// rather than queueing writer commands.
fn transcribe_one_flush(
    samples: Vec<f32>,
    vad_segments: Vec<(u64, u64)>,
    speaker_ids: Vec<Option<String>>,
    backend: &mut dyn AsrBackend,
    event_tx: &broadcast::Sender<AppEvent>,
    meeting_id: MeetingId,
    out: &mut Vec<Segment>,
) -> AppResult<()> {
    if vad_segments.is_empty() {
        return Ok(());
    }

    let chunk = AudioChunk {
        samples,
        sample_rate: SAMPLE_RATE_HZ as u32,
        start_ms: vad_segments.first().map(|(s, _)| *s).unwrap_or(0),
        end_ms: vad_segments.last().map(|(_, e)| *e).unwrap_or(0),
    };

    let chunk_segments = backend.transcribe_chunk(&chunk)?;

    let combined_text: String = chunk_segments
        .iter()
        .map(|s| s.text.as_str())
        .collect::<Vec<_>>()
        .join(" ");

    let sub_segments = emit_segments_proportional(&combined_text, &vad_segments, &speaker_ids);

    for seg in sub_segments {
        let _ = event_tx.send(AppEvent::TranscriptSegment {
            meeting_id,
            segment: seg.clone(),
        });
        out.push(seg);
    }

    Ok(())
}

/// Resolve and construct the production ASR backend for the re-transcribe path.
///
/// Reuses the live-path engine routing + model-resolution logic
/// ([`init_asr_backend`]): the model for the chosen `engine` must already be
/// `Available` in the registry. `n_gpu_layers` + `language` only affect the Qwen
/// tiers (Parakeet ignores them). Returns `Ok(None)` when the model is not
/// available (the caller surfaces this as an error, since an explicit
/// user-triggered re-transcribe with no model is a failure, unlike the live
/// path's best-effort skip).
pub(crate) async fn build_asr_backend_for_retranscribe(
    model_registry: &ModelRegistry,
    engine: AsrEngine,
    n_gpu_layers: u32,
    language: Option<String>,
) -> AppResult<Option<Box<dyn AsrBackend + Send>>> {
    init_asr_backend(model_registry, engine, n_gpu_layers, language).await
}
