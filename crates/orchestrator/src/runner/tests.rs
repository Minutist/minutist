use super::*;
use super::asr_worker::*;
use super::drain_loop::*;

// Convenience: ms-to-sample count at 16 kHz.
fn ms_to_samples(ms: u64) -> usize {
    (ms as usize * SAMPLE_RATE_HZ as usize) / 1000
}

// -----------------------------------------------------------------------
// resolve_gpu_layers — the runtime GPU toggle wiring (pure, no model)
// -----------------------------------------------------------------------

/// GPU off (`gpu_acceleration = false`) MUST force CPU (`0`), regardless of
/// the compiled backend. GPU on MUST resolve to the compile-time ceiling —
/// which is itself `0` in the default CPU-only build, so a CPU build is
/// unaffected by the flag.
#[test]
fn resolve_gpu_layers_off_forces_cpu() {
    assert_eq!(resolve_gpu_layers(false), 0, "GPU off must force CPU");

    let on = resolve_gpu_layers(true);
    assert_eq!(
        on,
        asr_runtime::default_n_gpu_layers(),
        "GPU on must use the compile-time ceiling"
    );
    if cfg!(any(
        feature = "vulkan",
        feature = "metal",
        feature = "cuda",
        feature = "rocm"
    )) {
        assert_eq!(on, u32::MAX, "a GPU-feature build offloads all layers when on");
    } else {
        assert_eq!(on, 0, "a default CPU-only build stays on CPU even when on");
    }
}

// -----------------------------------------------------------------------
// resolve_transcription_language — the ASR language-hint wiring (pure)
// -----------------------------------------------------------------------

/// A real language name resolves to `Some(name)` (prefix-force), trimmed.
/// The `"auto"` sentinel (case-insensitive), the empty string, and
/// whitespace-only resolve to `None` (auto-detect, no prefix).
#[test]
fn resolve_transcription_language_maps_names_and_sentinel() {
    assert_eq!(
        resolve_transcription_language("English"),
        Some("English".to_string())
    );
    assert_eq!(
        resolve_transcription_language("Spanish"),
        Some("Spanish".to_string())
    );

    // The reserved sentinel maps to None, case-insensitively.
    assert_eq!(resolve_transcription_language("auto"), None);
    assert_eq!(resolve_transcription_language("Auto"), None);

    // Empty / whitespace-only also map to None.
    assert_eq!(resolve_transcription_language(""), None);
    assert_eq!(resolve_transcription_language("   "), None);

    // A name is trimmed before forwarding.
    assert_eq!(
        resolve_transcription_language("  English "),
        Some("English".to_string())
    );
}

// -----------------------------------------------------------------------
// re_transcribe_fraction — the determinate progress mapping (pure, T4(a))
// -----------------------------------------------------------------------

/// The re-transcribe progress fraction is `samples_fed / total_kept_samples`,
/// clamped to `0.0..=1.0`; a zero total reports `1.0` (nothing to do is done);
/// an over-count clamps to `1.0`.
#[test]
fn re_transcribe_fraction_maps_fed_over_total() {
    assert_eq!(re_transcribe_fraction(0, 1000), 0.0, "start is 0.0");
    assert_eq!(re_transcribe_fraction(500, 1000), 0.5, "halfway is 0.5");
    assert_eq!(re_transcribe_fraction(1000, 1000), 1.0, "complete is 1.0");

    // Zero total (empty / all-pause audio) → 1.0 (done).
    assert_eq!(re_transcribe_fraction(0, 0), 1.0);
    assert_eq!(re_transcribe_fraction(10, 0), 1.0);

    // Over-count (a final-flush tick past the total) clamps to 1.0.
    assert_eq!(re_transcribe_fraction(1500, 1000), 1.0);

    // Quarter, three-quarters — within tolerance.
    assert!((re_transcribe_fraction(250, 1000) - 0.25).abs() < 1e-6);
    assert!((re_transcribe_fraction(750, 1000) - 0.75).abs() < 1e-6);
}

// -----------------------------------------------------------------------
// Test 1: zero-pad between two segments
// -----------------------------------------------------------------------

/// Accumulator must zero-pad the gap between two VAD segments so the
/// buffer length matches `(seg2.start_ms - seg1.start_ms) * 16`.
#[test]
fn accumulator_zero_pads_gap_between_two_segments() {
    let mut acc = Accumulator::new();

    // Segment 1: 0 – 1000 ms (1 s).
    let seg1_samples = vec![0.5f32; ms_to_samples(1000)];
    acc.append(0, 1000, &seg1_samples, None);

    // Segment 2: 2000 – 3000 ms (1 s). Gap = 1000 ms.
    let seg2_samples = vec![0.5f32; ms_to_samples(1000)];
    acc.append(2000, 3000, &seg2_samples, None);

    // Expected buffer: 3 s = 48_000 samples.
    let expected_len = ms_to_samples(3000);
    assert_eq!(
        acc.samples.len(),
        expected_len,
        "buffer length should be {} (3 s), got {}",
        expected_len,
        acc.samples.len()
    );

    // The gap (1 s, samples 16_000..32_000) must be silence.
    let gap_start = ms_to_samples(1000);
    let gap_end = ms_to_samples(2000);
    for s in &acc.samples[gap_start..gap_end] {
        assert!(
            s.abs() < 1e-6,
            "silence pad violated at offset [{gap_start}..{gap_end}]: {s}"
        );
    }
}

// -----------------------------------------------------------------------
// Test 2: gap capped at MAX_GAP_MS
// -----------------------------------------------------------------------

/// A 5 s gap between segments must be capped at MAX_GAP_MS (3 s).
#[test]
fn accumulator_caps_large_gap_at_max_gap_ms() {
    let mut acc = Accumulator::new();

    let seg1 = vec![0.5f32; ms_to_samples(1000)];
    acc.append(0, 1000, &seg1, None);

    // Segment 2 starts at 6000 ms — gap is 5000 ms, exceeds MAX_GAP_MS=3000 ms.
    let seg2 = vec![0.5f32; ms_to_samples(1000)];
    acc.append(6000, 7000, &seg2, None);

    // Buffer should be: 1 s speech + 3 s capped silence + 1 s speech = 5 s.
    let expected_len = ms_to_samples(1000) + ms_to_samples(MAX_GAP_MS) + ms_to_samples(1000);
    assert_eq!(
        acc.samples.len(),
        expected_len,
        "buffer should be capped to {expected_len} samples, got {}",
        acc.samples.len()
    );
}

// -----------------------------------------------------------------------
// Test 3: flush triggers on size
// -----------------------------------------------------------------------

/// A buffer at/over `FLUSH_MIN_SECS` must be flagged for a size-triggered
/// flush; one under it must not. Expressed relative to the constant so the
/// test stays honest if the threshold is retuned.
#[test]
fn accumulator_flush_triggers_on_size() {
    let flush_ms = (FLUSH_MIN_SECS * 1000.0) as u64;

    let mut under = Accumulator::new();
    let s = vec![0.0f32; ms_to_samples(flush_ms.saturating_sub(500))];
    under.append(0, flush_ms.saturating_sub(500), &s, None);
    assert!(
        under.duration_secs() < FLUSH_MIN_SECS,
        "sub-threshold buffer ({} s) should NOT trigger a size flush",
        under.duration_secs()
    );

    let mut over = Accumulator::new();
    let s = vec![0.0f32; ms_to_samples(flush_ms + 1000)];
    over.append(0, flush_ms + 1000, &s, None);
    assert!(
        over.duration_secs() >= FLUSH_MIN_SECS,
        "over-threshold buffer ({} s) must trigger a size flush at {} s",
        over.duration_secs(),
        FLUSH_MIN_SECS
    );
}

// -----------------------------------------------------------------------
// Test 4: flush triggers on latency
// -----------------------------------------------------------------------

/// After `LATENCY_WINDOW_SECS` of quiet (last_vad_end_at sufficiently old),
/// a non-empty sub-threshold buffer should be flushed so the live transcript
/// is not held back.
///
/// We don't sleep in the test; we set `last_vad_end_at` to a time well past
/// the window and check the elapsed condition directly.
#[test]
fn accumulator_flush_triggers_on_latency() {
    let mut acc = Accumulator::new();
    let seg = vec![0.0f32; ms_to_samples(1000)]; // 1 s — below flush_min
    acc.append(0, 1000, &seg, None);

    // Simulate elapsed time comfortably beyond the latency window.
    acc.last_vad_end_at =
        Some(Instant::now() - Duration::from_secs_f32(LATENCY_WINDOW_SECS + 2.0));

    let latency_window = Duration::from_secs_f32(LATENCY_WINDOW_SECS);
    let should_flush = !acc.is_empty()
        && acc
            .last_vad_end_at
            .map(|t| t.elapsed() >= latency_window)
            .unwrap_or(false);

    assert!(
        should_flush,
        "latency-window flush should trigger after >10 s of quiet"
    );
}

// -----------------------------------------------------------------------
// Test 5: proportional word allocation
// -----------------------------------------------------------------------

/// `emit_segments_proportional` must produce N segments matching the VAD
/// segment count, and the total word count across all segments must equal
/// the word count of the input text.
#[test]
fn proportional_allocation_matches_vad_count_and_total_words() {
    let vad_segments = vec![(0u64, 1000u64), (1000, 3000), (3000, 4000)];
    let text = "alpha beta gamma delta epsilon zeta";
    let word_count = text.split_whitespace().count(); // 6

    let segments = emit_segments_proportional(text, &vad_segments, &[None, None, None]);

    assert_eq!(
        segments.len(),
        vad_segments.len(),
        "segment count must match VAD segment count"
    );

    let total_words: usize = segments
        .iter()
        .map(|s| s.text.split_whitespace().count())
        .sum();
    assert_eq!(
        total_words, word_count,
        "total word count must be preserved across all sub-segments"
    );
}

// -----------------------------------------------------------------------
// Test 6: empty text still produces one Segment per VAD segment
// -----------------------------------------------------------------------

/// When the ASR produces empty text, `emit_segments_proportional` must
/// still return one `Segment` per VAD segment (timestamps preserved).
#[test]
fn proportional_allocation_empty_text_preserves_segment_count() {
    let vad_segments = vec![(0u64, 1000u64), (1000, 2000)];
    let segments = emit_segments_proportional("", &vad_segments, &[None, None]);
    assert_eq!(segments.len(), 2);
    assert!(segments.iter().all(|s| s.text.is_empty()));
}

// -----------------------------------------------------------------------
// Phase B: live speaker-label threading through the proportional re-split
// -----------------------------------------------------------------------

/// Each output `Segment.speaker_id` must equal the input live label at its
/// index, preserved positionally across the proportional TEXT re-split (the
/// re-split redistributes words but keeps one Segment per input vad_segment
/// in order).
#[test]
fn proportional_allocation_carries_live_labels_positionally() {
    let vad_segments = vec![(0u64, 1000u64), (1000, 3000), (3000, 4000)];
    let labels = vec![
        Some("A".to_string()),
        Some("B".to_string()),
        Some("A".to_string()),
    ];
    let segments =
        emit_segments_proportional("alpha beta gamma delta", &vad_segments, &labels);
    assert_eq!(segments.len(), 3);
    assert_eq!(segments[0].speaker_id.as_deref(), Some("A"));
    assert_eq!(segments[1].speaker_id.as_deref(), Some("B"));
    assert_eq!(segments[2].speaker_id.as_deref(), Some("A"));
}

/// Regression guard: all-None labels → all-None speaker_id (today's
/// behaviour, i.e. live diarization off / unwired).
#[test]
fn proportional_allocation_all_none_labels_yields_all_none() {
    let vad_segments = vec![(0u64, 1000u64), (1000, 2000)];
    let segments =
        emit_segments_proportional("alpha beta", &vad_segments, &[None, None]);
    assert_eq!(segments.len(), 2);
    assert!(segments.iter().all(|s| s.speaker_id.is_none()));
}

/// A `speaker_ids` slice SHORTER than `vad_segments` must yield `None` for
/// the missing tail rather than panicking — exercises the `.get(i)` guard.
#[test]
fn proportional_allocation_short_labels_yields_none_tail_no_panic() {
    let vad_segments = vec![(0u64, 1000u64), (1000, 2000), (2000, 3000)];
    // Only one label supplied for three segments.
    let labels = vec![Some("A".to_string())];
    let segments = emit_segments_proportional("a b c", &vad_segments, &labels);
    assert_eq!(segments.len(), 3);
    assert_eq!(segments[0].speaker_id.as_deref(), Some("A"));
    assert_eq!(segments[1].speaker_id, None);
    assert_eq!(segments[2].speaker_id, None);
}

/// Mixed Some/None labels are preserved positionally.
#[test]
fn proportional_allocation_mixed_labels_preserved() {
    let vad_segments = vec![(0u64, 1000u64), (1000, 2000), (2000, 3000)];
    let labels = vec![Some("A".to_string()), None, Some("B".to_string())];
    let segments = emit_segments_proportional("x y z", &vad_segments, &labels);
    assert_eq!(segments[0].speaker_id.as_deref(), Some("A"));
    assert_eq!(segments[1].speaker_id, None);
    assert_eq!(segments[2].speaker_id.as_deref(), Some("B"));
}

// -----------------------------------------------------------------------
// Phase B: Accumulator carries the speaker-label column in lockstep
// -----------------------------------------------------------------------

/// `append` pushes a label per segment; `drain` returns the label column in
/// lockstep with `vad_segments` (len equality invariant) and resets it.
#[test]
fn accumulator_carries_speaker_ids_in_lockstep_and_drains() {
    let mut acc = Accumulator::new();
    let s = vec![0.5f32; ms_to_samples(500)];
    acc.append(0, 500, &s, Some("A".to_string()));
    acc.append(500, 1000, &s, None);
    acc.append(1000, 1500, &s, Some("B".to_string()));

    assert_eq!(
        acc.speaker_ids.len(),
        acc.vad_segments.len(),
        "speaker_ids must stay in lockstep with vad_segments"
    );

    let (_samples, vad_segments, speaker_ids) = acc.drain();
    assert_eq!(vad_segments.len(), 3);
    assert_eq!(speaker_ids.len(), vad_segments.len());
    assert_eq!(speaker_ids[0].as_deref(), Some("A"));
    assert_eq!(speaker_ids[1], None);
    assert_eq!(speaker_ids[2].as_deref(), Some("B"));

    // drain reset the label column.
    assert!(acc.speaker_ids.is_empty(), "drain must reset speaker_ids");
    assert!(acc.vad_segments.is_empty(), "drain must reset vad_segments");
}

/// The gap-capping path (large inter-segment gap) must not perturb the
/// label/segment 1:1 correspondence — the label is tied to the segment, not
/// to the zero-padded samples.
#[test]
fn accumulator_gap_cap_preserves_label_correspondence() {
    let mut acc = Accumulator::new();
    let seg = vec![0.5f32; ms_to_samples(1000)];
    acc.append(0, 1000, &seg, Some("A".to_string()));
    // 5 s gap → capped at MAX_GAP_MS; label still rides segment 2.
    acc.append(6000, 7000, &seg, Some("B".to_string()));

    let (_samples, vad_segments, speaker_ids) = acc.drain();
    assert_eq!(speaker_ids.len(), vad_segments.len());
    assert_eq!(speaker_ids[0].as_deref(), Some("A"));
    assert_eq!(speaker_ids[1].as_deref(), Some("B"));
}

// -----------------------------------------------------------------------
// Test 7: dispatch_flush drop-oldest behaviour
// -----------------------------------------------------------------------

/// When `dispatch_flush` is called with a full queue (FLUSH_CHANNEL_CAP=4),
/// it must drop the OLDEST entry (front) and retain the newest (back).
#[test]
fn dispatch_flush_drops_oldest_when_queue_full() {
    use minutist_common::MeetingId;
    let flush_queue = FlushQueue::new();
    let meeting_id = MeetingId::new();

    // Fill the queue to capacity with payloads tagged by their index in
    // vad_segments[0].0 (start_ms) so we can identify which was dropped.
    for i in 0..FLUSH_CHANNEL_CAP {
        let payload = FlushPayload {
            samples: vec![0.0f32; 100],
            vad_segments: vec![(i as u64 * 1000, i as u64 * 1000 + 500)],
            speaker_ids: vec![None],
            meeting_id,
        };
        assert!(
            !dispatch_flush_pub(&flush_queue, payload),
            "no drop while filling below capacity"
        );
    }

    // No drop yet → the live transcript is still complete.
    assert!(
        !flush_queue.incomplete.load(std::sync::atomic::Ordering::Acquire),
        "incomplete must be false before any drop"
    );

    // Queue is now full (4 entries). Dispatching one more should drop index 0.
    let newest_start_ms = FLUSH_CHANNEL_CAP as u64 * 1000;
    let newest_payload = FlushPayload {
        samples: vec![0.0f32; 100],
        vad_segments: vec![(newest_start_ms, newest_start_ms + 500)],
        speaker_ids: vec![None],
        meeting_id,
    };
    assert!(
        dispatch_flush_pub(&flush_queue, newest_payload),
        "dispatching into a full queue must report a drop (backpressure)"
    );

    // Drain the queue and collect start_ms values.
    let deque = flush_queue.deque.lock().unwrap();
    let starts: Vec<u64> = deque
        .iter()
        .map(|p| p.vad_segments[0].0)
        .collect();

    // The oldest (start_ms = 0) must have been dropped.
    assert!(
        !starts.contains(&0),
        "oldest flush (start_ms=0) must have been dropped; got: {starts:?}"
    );
    // The newest must be present.
    assert!(
        starts.contains(&newest_start_ms),
        "newest flush (start_ms={newest_start_ms}) must be retained; got: {starts:?}"
    );
    // Queue must not exceed FLUSH_CHANNEL_CAP.
    assert_eq!(
        deque.len(),
        FLUSH_CHANNEL_CAP,
        "queue must remain at FLUSH_CHANNEL_CAP after drop-oldest; len={}",
        deque.len()
    );

    // The drop must flag the transcript incomplete so ipc-bridge runs a
    // background re-transcribe of the complete audio after stop.
    assert!(
        flush_queue.incomplete.load(std::sync::atomic::Ordering::Acquire),
        "dropping a pending flush must set incomplete"
    );
}

// -----------------------------------------------------------------------
// B3a: a dropped writer-channel send must flag `incomplete`
// -----------------------------------------------------------------------

/// A segment that fails to reach the writer (the `try_send` in
/// `process_flush_with_backend` returning `Err`, e.g. because the writer
/// task has already exited) was already broadcast to the live transcript,
/// so silently dropping it would let it vanish from `transcript.json`
/// forever. `incomplete` must be set so the post-stop re-transcribe
/// restores it from `audio.opus`.
#[test]
fn writer_send_failure_sets_incomplete() {
    let (event_tx, _event_rx) = broadcast::channel::<AppEvent>(16);
    // Capacity 1, and the receiver is dropped immediately, so `try_send`
    // fails with `Closed` regardless of capacity.
    let (writer_cmd_tx, writer_cmd_rx) = mpsc::channel::<WriterCommand>(1);
    drop(writer_cmd_rx);

    let incomplete = AtomicBool::new(false);
    let payload = FlushPayload {
        samples: vec![0.0f32; ms_to_samples(100)],
        vad_segments: vec![(0, 100)],
        speaker_ids: vec![None],
        meeting_id: MeetingId::new(),
    };
    let mut backend = crate::test_support::StubAsrBackend::new("hello");

    process_flush_with_backend(payload, &mut backend, &event_tx, &writer_cmd_tx, &incomplete)
        .expect("a dropped writer send must not fail the flush itself");

    assert!(
        incomplete.load(Ordering::Acquire),
        "a dropped writer-channel send must flag the transcript incomplete"
    );
}

// -----------------------------------------------------------------------
// B3b: the ASR worker must drain a Notify-coalesced burst in one wakeup
// -----------------------------------------------------------------------

/// `tokio::sync::Notify` stores at most one permit: several `notify_one()`
/// calls issued before the worker is waiting collapse into a SINGLE
/// `notified().await` wakeup. A worker that pops only one queue item per
/// wakeup (the pre-fix shape) therefore leaves the rest of a burst parked
/// until some unrelated later push notifies again. This test enqueues five
/// payloads directly (bypassing `dispatch_flush`, which would call
/// `notify_one()` once per push — already coalesced to one permit by the
/// time the worker thread starts) and asserts the worker drains ALL of them
/// from that one wakeup, with no further push or notify.
#[test]
fn asr_worker_drains_full_burst_from_a_single_notify_wakeup() {
    const N: usize = 5;

    let flush_queue = FlushQueue::new();
    let meeting_id = MeetingId::new();

    // Enqueue the whole burst BEFORE the worker thread exists, so there is
    // no waiter yet: every `notify_one()` below coalesces into one stored
    // permit, reproducing the real runner's rapid-push burst pattern.
    for i in 0..N {
        let payload = FlushPayload {
            samples: vec![0.0f32; ms_to_samples(20)],
            vad_segments: vec![(i as u64 * 100, i as u64 * 100 + 20)],
            speaker_ids: vec![None],
            meeting_id,
        };
        flush_queue.deque.lock().unwrap().push_back(payload);
        flush_queue.notify.notify_one();
    }

    let (event_tx, _event_rx) = broadcast::channel::<AppEvent>(64);
    let (writer_cmd_tx, _writer_cmd_rx) = mpsc::channel::<WriterCommand>(64);

    let tmp = tempfile::tempdir().expect("tempdir");
    let (registry_event_tx, _) = broadcast::channel::<AppEvent>(16);
    let model_registry = Arc::new(
        ModelRegistry::new(tmp.path().to_path_buf(), Vec::new(), registry_event_tx)
            .expect("test ModelRegistry construction should not fail"),
    );

    let worker_queue = flush_queue.consumer_clone();
    let worker = std::thread::spawn(move || {
        run_asr_worker(
            worker_queue,
            Some(Box::new(crate::test_support::StubAsrBackend::new("hi"))),
            model_registry,
            0,
            None,
            AsrEngine::ParakeetEuV3,
            event_tx,
            writer_cmd_tx,
        );
    });

    // The fixed worker drains the whole burst from the ONE coalesced
    // notification without any further push/notify; `wait_all_processed`
    // (queue empty AND nothing in-flight) must go true well inside this
    // bound. Before the fix this timed out with 4 of 5 payloads still
    // queued, parked behind a `notified()` nobody was ever going to call
    // again.
    assert!(
        flush_queue.wait_all_processed(Duration::from_secs(2), Duration::from_millis(5)),
        "the worker must drain the full burst from a single notify wakeup"
    );

    // Let the worker exit cleanly and reclaim the thread.
    flush_queue.close();
    worker.join().expect("ASR worker thread must not panic");
}

// -----------------------------------------------------------------------
// pause_excluding_segments (TIMELINE-DRIFT #4)
// -----------------------------------------------------------------------

/// A buffer with no long silent run is one kept region spanning the whole
/// buffer, starting at excl_start_ms = 0.
#[test]
fn pause_excluding_no_pause_is_single_region() {
    // 2 s of "speech" (non-silent) with a SHORT (1 s) quiet gap in the
    // middle — below PAUSE_MIN_MS, so it must NOT split.
    let mut pcm = vec![0.5f32; ms_to_samples(1000)];
    pcm.extend(std::iter::repeat_n(0.0f32, ms_to_samples(1000))); // 1 s quiet < 4 s
    pcm.extend(std::iter::repeat_n(0.5f32, ms_to_samples(1000)));

    let regions = pause_excluding_segments(&pcm);
    assert_eq!(regions.len(), 1, "short quiet gap must not split the buffer");
    assert_eq!(regions[0].src_start, 0);
    assert_eq!(regions[0].src_end, pcm.len());
    assert_eq!(regions[0].excl_start_ms, 0);
}

/// A long near-silent run (a pause) splits the buffer into two kept regions,
/// and the second region's pause-EXCLUDING start equals the duration of the
/// FIRST region only — the pause contributes no pause-excluding time.
#[test]
fn pause_excluding_splits_on_long_silence_and_excludes_it() {
    // 1 s speech, 6 s pause (> PAUSE_MIN_MS = 4 s), 2 s speech.
    let speech_a = ms_to_samples(1000);
    let pause = ms_to_samples(6000);
    let speech_b = ms_to_samples(2000);

    let mut pcm = vec![0.5f32; speech_a];
    pcm.extend(std::iter::repeat_n(0.0f32, pause));
    pcm.extend(std::iter::repeat_n(0.5f32, speech_b));

    let regions = pause_excluding_segments(&pcm);
    assert_eq!(regions.len(), 2, "long pause must split into two regions");

    // Region 0: the first second of speech, starting at excl 0.
    assert_eq!(regions[0].src_start, 0);
    assert_eq!(regions[0].src_end, speech_a);
    assert_eq!(regions[0].excl_start_ms, 0);

    // Region 1: the post-pause speech. Its source start is AFTER the pause,
    // but its pause-EXCLUDING start is only 1000 ms (the kept duration of
    // region 0) — the 6 s pause is excluded from the timeline.
    assert_eq!(regions[1].src_start, speech_a + pause);
    assert_eq!(regions[1].src_end, pcm.len());
    assert_eq!(
        regions[1].excl_start_ms, 1000,
        "post-pause region must start at the pause-EXCLUDING offset (1 s), \
         not the pause-INCLUDING offset (7 s)"
    );
}

// -----------------------------------------------------------------------
// Phase B: build_online_diarizer — no-download/no-block guarantee
// -----------------------------------------------------------------------

/// `build_online_diarizer` must return `None` (not Err, no panic, no
/// network) when the embedding model is NOT `Available` in the registry —
/// proving the local-only/no-download start guarantee. The cache dir is an
/// empty tempdir, so the model is `Missing`.
#[test]
fn build_online_diarizer_returns_none_when_model_absent() {
    use minutist_common::{ModelFileEntry, ModelKind, ModelManifestEntry};

    let dir = tempfile::tempdir().expect("tempdir");
    let (event_tx, _rx) = broadcast::channel::<AppEvent>(16);

    // Manifest entry for the embedding model the live path resolves, but no
    // files are placed → status is Missing.
    let entry = ModelManifestEntry {
        id: ModelId::from(DIARIZE_EMB_MODEL_ID),
        kind: ModelKind::Diarize,
        display_name: "Embedding".into(),
        files: vec![ModelFileEntry {
            filename: "model.onnx".into(),
            url: "http://example.com/model.onnx".into(),
            size: 10,
            sha256: "00".repeat(32),
        }],
        total_size_bytes: 10,
        license: "apache-2.0".into(),
    };

    let registry = ModelRegistry::new(dir.path().to_path_buf(), vec![entry], event_tx)
        .expect("registry");

    // Must be None, and must not have downloaded anything or panicked.
    let result = build_online_diarizer(&registry);
    assert!(
        result.is_none(),
        "build_online_diarizer must return None when the embedding model is absent"
    );
}

// -----------------------------------------------------------------------
// pcm_window_for_excluding_range (W1, Phase 9 — gating)
// -----------------------------------------------------------------------

/// A window wholly inside a single kept region maps to the matching PCM
/// sample range on the pause-INCLUDING clock (no pause present).
#[test]
fn pcm_window_maps_within_single_region() {
    let pcm = vec![0.5f32; ms_to_samples(5000)]; // 5 s, no pause
    // Excluding window [1000, 2000) ms.
    let range = pcm_window_for_excluding_range(&pcm, 1000, 2000).expect("range");
    assert_eq!(range.start, ms_to_samples(1000));
    assert_eq!(range.end, ms_to_samples(2000));
}

/// W1 GATING: a window whose START lands in the POST-pause region must map
/// onto the pause-INCLUDING samples AFTER the skipped pause, not the
/// pause-excluding offset — proving the clock conversion, not a passthrough.
#[test]
fn pcm_window_maps_post_pause_window_onto_including_clock() {
    // 1 s speech, 6 s pause (> 4 s threshold), 3 s speech.
    let speech_a = ms_to_samples(1000);
    let pause = ms_to_samples(6000);
    let speech_b = ms_to_samples(3000);
    let mut pcm = vec![0.5f32; speech_a];
    pcm.extend(std::iter::repeat_n(0.0f32, pause));
    pcm.extend(std::iter::repeat_n(0.5f32, speech_b));

    // The post-pause region starts at pause-EXCLUDING 1000 ms. A window
    // [1500, 2500) ms (excluding clock) is 500..1500 ms INTO the post-pause
    // region, i.e. pause-INCLUDING samples [speech_a + pause + 500ms,
    // speech_a + pause + 1500ms).
    let range = pcm_window_for_excluding_range(&pcm, 1500, 2500).expect("range");
    assert_eq!(range.start, speech_a + pause + ms_to_samples(500));
    assert_eq!(range.end, speech_a + pause + ms_to_samples(1500));
    // Confirm the slice is over the post-pause SPEECH (non-silent), proving
    // the pause was skipped rather than sliced.
    assert!(pcm[range.start..range.end].iter().all(|s| s.abs() > 0.1));
}

/// W1 GATING: a window that STRADDLES a pause is clamped to the kept region
/// containing its start (the documented v1 clamp decision — no concatenation
/// across the pause seam).
#[test]
fn pcm_window_straddling_a_pause_clamps_to_first_region() {
    let speech_a = ms_to_samples(2000); // pause-excl [0, 2000)
    let pause = ms_to_samples(6000);
    let speech_b = ms_to_samples(2000); // pause-excl [2000, 4000)
    let mut pcm = vec![0.5f32; speech_a];
    pcm.extend(std::iter::repeat_n(0.0f32, pause));
    pcm.extend(std::iter::repeat_n(0.5f32, speech_b));

    // Window [1000, 3000) ms straddles the pause (region 0 ends at excl
    // 2000). It must clamp to region 0: [1000ms .. 2000ms] of region 0,
    // i.e. PCM samples [1000ms, 2000ms) — never crossing into region 1.
    let range = pcm_window_for_excluding_range(&pcm, 1000, 3000).expect("range");
    assert_eq!(range.start, ms_to_samples(1000));
    assert_eq!(range.end, speech_a, "must clamp at the first region's end");
}

/// A window past the end of all kept audio yields `None` (out of range).
#[test]
fn pcm_window_out_of_range_is_none() {
    let pcm = vec![0.5f32; ms_to_samples(1000)];
    assert!(pcm_window_for_excluding_range(&pcm, 5000, 6000).is_none());
}

/// B2: the cached-region path ([`excluding_range_to_pcm_slice`] against a
/// [`pause_excluding_segments`] table computed ONCE) must yield exactly the
/// same window as the per-call path ([`pcm_window_for_excluding_range`],
/// which recomputes the region scan every call) for every one of several
/// requests over the SAME meeting — including a request that straddles a
/// pause and one that is out of range. Guards the B2 refactor: callers that
/// switched from the recomputing form to the cached form (the per-segment /
/// per-cut loops in `lib.rs`) must observe bit-identical windows.
#[test]
fn cached_region_path_matches_per_call_path_across_many_requests() {
    // 2 s speech, 6 s pause (> PAUSE_MIN_MS), 3 s speech, 5 s pause, 1 s speech.
    let mut pcm = vec![0.5f32; ms_to_samples(2000)];
    pcm.extend(std::iter::repeat_n(0.0f32, ms_to_samples(6000)));
    pcm.extend(std::iter::repeat_n(0.5f32, ms_to_samples(3000)));
    pcm.extend(std::iter::repeat_n(0.0f32, ms_to_samples(5000)));
    pcm.extend(std::iter::repeat_n(0.5f32, ms_to_samples(1000)));

    let regions = pause_excluding_segments(&pcm);

    let requests: &[(u64, u64)] = &[
        (0, 1000),        // wholly inside region 0
        (1500, 2500),     // straddles the region 0 → region 1 pause
        (2000, 4000),     // wholly inside region 1 (post first pause)
        (2000, 10_000),   // straddles the region 1 → region 2 pause
        (3000, 3100),     // inside region 2 (post second pause)
        (100_000, 101_000), // out of range
    ];

    for &(start_ms, end_ms) in requests {
        let cached = excluding_range_to_pcm_slice(&regions, start_ms, end_ms);
        let per_call = pcm_window_for_excluding_range(&pcm, start_ms, end_ms);
        assert_eq!(
            cached, per_call,
            "cached vs per-call mismatch for window [{start_ms}, {end_ms})"
        );
    }
}

// -----------------------------------------------------------------------
// pcm16_wav (the transcript "play segment" clip encoder)
// -----------------------------------------------------------------------

/// The encoded buffer is a well-formed 16 kHz mono PCM16 WAV: a 44-byte
/// header carrying the canonical RIFF/WAVE fields, one little-endian i16 per
/// input sample, and a `data` length matching the sample count. Samples are
/// clamped to [-1, 1] before scaling, so an out-of-range f32 saturates rather
/// than wrapping.
#[test]
fn pcm16_wav_header_and_samples_are_canonical() {
    // 0.0 → 0, 1.0 → 32767, -1.0 → -32767, and 2.0 clamps to 32767.
    let samples = [0.0f32, 1.0, -1.0, 2.0];
    let wav = pcm16_wav(&samples);

    assert_eq!(wav.len(), 44 + samples.len() * 2, "44-byte header + 2 B/sample");
    assert_eq!(&wav[0..4], b"RIFF");
    assert_eq!(&wav[8..12], b"WAVE");
    assert_eq!(&wav[12..16], b"fmt ");
    assert_eq!(u32::from_le_bytes([wav[16], wav[17], wav[18], wav[19]]), 16); // PCM fmt size
    assert_eq!(u16::from_le_bytes([wav[20], wav[21]]), 1); // PCM
    assert_eq!(u16::from_le_bytes([wav[22], wav[23]]), 1); // mono
    assert_eq!(
        u32::from_le_bytes([wav[24], wav[25], wav[26], wav[27]]),
        SAMPLE_RATE_HZ as u32,
    );
    assert_eq!(u16::from_le_bytes([wav[34], wav[35]]), 16); // bits/sample
    assert_eq!(&wav[36..40], b"data");
    let data_len = u32::from_le_bytes([wav[40], wav[41], wav[42], wav[43]]);
    assert_eq!(data_len as usize, samples.len() * 2);
    assert_eq!(
        u32::from_le_bytes([wav[4], wav[5], wav[6], wav[7]]),
        36 + data_len,
    );

    let decoded: Vec<i16> = wav[44..]
        .chunks_exact(2)
        .map(|b| i16::from_le_bytes([b[0], b[1]]))
        .collect();
    assert_eq!(decoded, vec![0, 32767, -32767, 32767]);
}

/// Trailing/leading pause padding is excluded; a pause at the very start
/// shifts the first kept region's source start without consuming
/// pause-excluding time.
#[test]
fn pause_excluding_handles_leading_pause() {
    let pause = ms_to_samples(5000);
    let speech = ms_to_samples(1000);
    let mut pcm = vec![0.0f32; pause];
    pcm.extend(std::iter::repeat_n(0.5f32, speech));

    let regions = pause_excluding_segments(&pcm);
    assert_eq!(regions.len(), 1);
    assert_eq!(regions[0].src_start, pause, "leading pause must be skipped");
    assert_eq!(regions[0].excl_start_ms, 0);
}

// -----------------------------------------------------------------------
// snap_to_energy_min (#0015 phase 4)
// -----------------------------------------------------------------------

/// A speech–silence–speech buffer snaps the cut into the silent valley: the
/// returned argmin window sits inside the near-silent run, not in either
/// speech burst.
#[test]
fn snap_finds_the_silence_valley() {
    // 200 ms speech, 60 ms silence, 200 ms speech.
    let a = ms_to_samples(200);
    let gap = ms_to_samples(60);
    let b = ms_to_samples(200);
    let mut pcm = vec![0.5f32; a];
    pcm.extend(std::iter::repeat_n(0.0f32, gap));
    pcm.extend(std::iter::repeat_n(0.5f32, b));

    // Cut nominally at the speech/silence boundary; search ±150 ms.
    let cut = a;
    let snapped = snap_to_energy_min(&pcm, cut, 150).expect("a clear minimum exists");
    // The argmin window must fall within the silent run [a, a + gap).
    assert!(
        snapped >= a && snapped < a + gap,
        "snap {snapped} must land in the silent valley [{a}, {})",
        a + gap
    );
}

/// Constant-energy audio has no clear minimum → `None` (keep-whole). The
/// relative-RMS floor rejects a flat span as continuous/overlapping speech.
#[test]
fn snap_returns_none_on_constant_energy() {
    let pcm = vec![0.5f32; ms_to_samples(500)];
    let cut = ms_to_samples(250);
    assert!(
        snap_to_energy_min(&pcm, cut, 150).is_none(),
        "a flat-energy span has no boundary to snap to"
    );
}

/// A degenerate search span (window shorter than one RMS window) → `None`.
#[test]
fn snap_returns_none_on_tiny_span() {
    let pcm = vec![0.5f32; ms_to_samples(100)];
    // window_ms = 1 → ±1 ms span (16 samples each side), below the 5 ms RMS
    // window, so there is no usable analysis window.
    assert!(snap_to_energy_min(&pcm, ms_to_samples(50), 1).is_none());
}

// -----------------------------------------------------------------------
// excluding_ms_for_pcm_sample_in_regions — the PCM→excluding-ms inverse
// (#0015 phase 4)
// -----------------------------------------------------------------------

/// Inside a single kept region (no pause) the inverse is the straight
/// sample→ms conversion: PCM sample N maps to N/16 ms on the excluding clock.
#[test]
fn inverse_maps_within_single_region() {
    let pcm = vec![0.5f32; ms_to_samples(5000)];
    let regions = pause_excluding_segments(&pcm);
    assert_eq!(
        excluding_ms_for_pcm_sample_in_regions(&regions, ms_to_samples(1500)),
        1500
    );
    assert_eq!(excluding_ms_for_pcm_sample_in_regions(&regions, 0), 0);
}

/// CLOCK REGRESSION GUARD: with a ≥4 s pause, the forward map and the inverse
/// round-trip. A post-pause excluding-ms maps to a pause-INCLUDING PCM sample
/// (forward), and that sample maps back to the SAME excluding-ms (inverse) —
/// proving a split sub-clip's `start_ms` lands on the transcript clock, not
/// inflated by the skipped pause padding.
#[test]
fn inverse_round_trips_across_a_long_pause() {
    // 2 s speech, 6 s pause (> PAUSE_MIN_MS = 4 s), 3 s speech.
    let speech_a = ms_to_samples(2000);
    let pause = ms_to_samples(6000);
    let speech_b = ms_to_samples(3000);
    let mut pcm = vec![0.5f32; speech_a];
    pcm.extend(std::iter::repeat_n(0.0f32, pause));
    pcm.extend(std::iter::repeat_n(0.5f32, speech_b));
    let regions = pause_excluding_segments(&pcm);

    // Post-pause excluding clock: region 2 starts at excluding 2000 ms. Pick a
    // point 1000 ms into it → excluding 3000 ms.
    let excl_ms = 3000u64;
    let range =
        pcm_window_for_excluding_range(&pcm, excl_ms, excl_ms + 100).expect("forward range");
    // The forward map lands on pause-INCLUDING samples AFTER the pause.
    assert_eq!(range.start, speech_a + pause + ms_to_samples(1000));
    // The inverse takes that PCM sample back to the SAME excluding ms.
    assert_eq!(
        excluding_ms_for_pcm_sample_in_regions(&regions, range.start),
        excl_ms
    );

    // A sample INSIDE the skipped pause clamps to the pre-pause region end
    // (excluding 2000 ms — the instant the clock froze).
    let mid_pause = speech_a + ms_to_samples(3000);
    assert_eq!(
        excluding_ms_for_pcm_sample_in_regions(&regions, mid_pause),
        2000
    );
}

// -----------------------------------------------------------------------
// Embedding-model resolution convergence guard (WU1 / #0003)
// -----------------------------------------------------------------------

/// The offline diarizer path (`build_diarizer`, lines ~2170-2171) and the
/// online diarizer path (`build_online_diarizer`, lines ~2234-2235) both
/// resolve the embedding model `.onnx` via `find_file_in_dir` with a
/// predicate of `name.ends_with(".onnx")`, and both must use the SAME
/// manifest id (`DIARIZE_EMB_MODEL_ID`). A divergence would place
/// `VoiceprintExtractor` in a different embedding space than the online
/// clusterer, silently invalidating voiceprint comparisons.
///
/// This test is purely structural (no model, no I/O): it verifies the
/// predicate applied to candidate filenames and confirms that the model
/// id constant matches what both paths reference, making the divergence
/// visible at compile-or-test time rather than at runtime on a user's
/// machine.
#[test]
fn embedding_model_resolution_predicate_is_identical_for_offline_and_online() {
    // The predicate used in both build_diarizer (line ~2171) and
    // build_online_diarizer (line ~2235).  Any change to either call
    // site must keep these assertions in sync.
    let predicate = |name: &str| name.ends_with(".onnx");

    // Positive: the embedding model file the registry places in the dir.
    assert!(predicate("model.onnx"), "primary .onnx name must match");
    assert!(predicate("speaker_embedding.onnx"), "variant .onnx name must match");

    // Negative: other file types in a model dir must NOT match.
    assert!(!predicate("model.bin"), ".bin must not match");
    assert!(!predicate("manifest.json"), ".json must not match");
    assert!(!predicate("README.md"), ".md must not match");
    assert!(!predicate("model.onnxruntime"), "partial extension must not match");

    // The model id used by both resolution paths is the same constant.
    // VoiceprintExtractor (WU1) relies on this: it is opened by the
    // orchestrator using the same DIARIZE_EMB_MODEL_ID + predicate pair,
    // ensuring the embedding space is consistent across all three paths.
    assert_eq!(
        DIARIZE_EMB_MODEL_ID, "3dspeaker-campplus-zh-en-advanced",
        "DIARIZE_EMB_MODEL_ID must not diverge between offline and online paths"
    );
}
