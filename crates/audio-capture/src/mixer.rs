//! Two-source audio mixer — sums the microphone's 16 kHz mono stream with the
//! system/call (loopback) stream into the single `samples` stream the
//! orchestrator drains.
//!
//! Design (per `architecture/cross-cutting.md` — "Threading model"):
//!   • Each capture source (mic + loopback) resamples to 16 kHz mono
//!     independently and pushes [`AudioFrameBatch`]es into its own bounded
//!     internal channel (the same drop-oldest, RT-callback-safe discipline as
//!     the single-source path).
//!   • A `spawn_blocking` mixer task drains both channels, SUMS them
//!     sample-wise, clamps to `[-1.0, 1.0]`, and forwards `AudioFrameBatch`es
//!     into the public `samples` channel — the same shape the mic-only path
//!     emits, so the orchestrator/runner are unchanged.
//!   • Sources drift: if one lags, the mixer zero-fills the missing source so
//!     the timeline keeps advancing. Transcription tolerates small drift, so a
//!     sample-accurate lock-step is not required.
//!
//! The mixing math is factored into the pure [`MixState`] so it can be
//! unit-tested with synthetic two-stream input (the real capture devices
//! cannot be driven in a unit test).

use crate::manager::AudioFrameBatch;

/// Maximum lead (in 16 kHz samples) one source may hold over an *empty* other
/// source before that other source is treated as starved (idle) and zero-filled
/// mid-stream. 30 ms @ 16 kHz.
///
/// This is the steady-state safety valve the end-of-stream `drain_remaining`
/// flush cannot provide: WASAPI loopback delivers ~no samples when no system
/// audio is playing, so without this bound a silent loopback would buffer the
/// microphone indefinitely and emit nothing — starving transcription AND the
/// meter. Only triggers when the lagging source is genuinely empty; a merely
/// slow source (both buffers non-empty) is still paired exactly by
/// `drain_paired` with no silence injected.
///
/// 30 ms (≈ one VAD frame) so that, on the common idle-loopback path, the mixer
/// emits ~30 batches/sec — driving the audio level meter at ~30 Hz and keeping
/// mic latency low. A larger cap made the meter update in laggy ~10 Hz bursts.
pub(crate) const MAX_SKEW_SAMPLES: usize = crate::resample::TARGET_RATE as usize * 30 / 1000;

/// Sum two equal-length mono buffers sample-wise, clamping each result to
/// `[-1.0, 1.0]`.
///
/// `b` shorter than `a` is treated as zero-filled past its end (and vice
/// versa): the longer buffer's tail passes through clamped. This is the
/// zero-fill-on-lag rule — a momentarily-absent source contributes silence
/// rather than stalling the mix.
pub(crate) fn sum_clamp(a: &[f32], b: &[f32]) -> Vec<f32> {
    let n = a.len().max(b.len());
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let x = a.get(i).copied().unwrap_or(0.0) + b.get(i).copied().unwrap_or(0.0);
        out.push(x.clamp(-1.0, 1.0));
    }
    out
}

/// Per-source pending sample buffer + the running output sample clock.
///
/// The mixer accumulates samples from each source into a FIFO and emits a
/// mixed batch whenever it can pair up as many samples as the shorter of the
/// two buffers currently holds. On flush (a source has ended) it drains
/// whatever remains, zero-filling the exhausted source.
pub(crate) struct MixState {
    mic: Vec<f32>,
    sys: Vec<f32>,
    /// 16 kHz mono samples emitted so far (drives the output batch timestamps).
    out_clock_samples: u64,
}

impl MixState {
    pub(crate) fn new() -> Self {
        Self {
            mic: Vec::new(),
            sys: Vec::new(),
            out_clock_samples: 0,
        }
    }

    /// Append mic samples to the pending buffer.
    pub(crate) fn push_mic(&mut self, samples: &[f32]) {
        self.mic.extend_from_slice(samples);
    }

    /// Append system/loopback samples to the pending buffer.
    pub(crate) fn push_sys(&mut self, samples: &[f32]) {
        self.sys.extend_from_slice(samples);
    }

    /// Mix and drain the samples both sources have in common (`min(len)`),
    /// leaving any surplus from the longer source buffered for the next call.
    /// Returns `None` when there is nothing to emit yet.
    ///
    /// This is the steady-state path: it never zero-fills, so a temporarily
    /// slow source simply holds back the mix by its own backlog rather than
    /// injecting silence. The `drain_remaining` flush handles a source that has
    /// stopped entirely.
    pub(crate) fn drain_paired(&mut self) -> Option<AudioFrameBatch> {
        let n = self.mic.len().min(self.sys.len());
        if n == 0 {
            return None;
        }
        let mixed = sum_clamp(&self.mic[..n], &self.sys[..n]);
        self.mic.drain(..n);
        self.sys.drain(..n);
        Some(self.emit(mixed))
    }

    /// Mid-stream starvation valve: when one source's buffer is empty and the
    /// other has accumulated more than [`MAX_SKEW_SAMPLES`], drain that lead and
    /// emit it with the starved source zero-filled, so an idle source (e.g.
    /// loopback when nothing is playing) cannot block the live source forever.
    ///
    /// Returns `None` unless a source is genuinely starved beyond the cap — the
    /// normal pairing in [`drain_paired`] handles everything else. Call this
    /// after `drain_paired` has consumed the common prefix (so at most one
    /// buffer is non-empty).
    pub(crate) fn drain_starved(&mut self) -> Option<AudioFrameBatch> {
        let lead = if self.sys.is_empty() && self.mic.len() > MAX_SKEW_SAMPLES {
            std::mem::take(&mut self.mic)
        } else if self.mic.is_empty() && self.sys.len() > MAX_SKEW_SAMPLES {
            std::mem::take(&mut self.sys)
        } else {
            return None;
        };
        // The starved source contributes silence; clamp the survivor for parity
        // with the paired path.
        Some(self.emit(sum_clamp(&lead, &[])))
    }

    /// Flush any buffered surplus from either source at end-of-stream,
    /// zero-filling the exhausted source. Returns `None` when both buffers are
    /// empty.
    pub(crate) fn drain_remaining(&mut self) -> Option<AudioFrameBatch> {
        if self.mic.is_empty() && self.sys.is_empty() {
            return None;
        }
        let mixed = sum_clamp(&self.mic, &self.sys);
        self.mic.clear();
        self.sys.clear();
        Some(self.emit(mixed))
    }

    /// Wrap a mixed sample run in an [`AudioFrameBatch`] and advance the clock.
    fn emit(&mut self, samples: Vec<f32>) -> AudioFrameBatch {
        let start_ms = self.out_clock_samples * 1000 / crate::resample::TARGET_RATE as u64;
        self.out_clock_samples += samples.len() as u64;
        let end_ms = self.out_clock_samples * 1000 / crate::resample::TARGET_RATE as u64;
        AudioFrameBatch {
            samples,
            start_ms,
            end_ms,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sum_clamp_adds_sample_wise() {
        // Use power-of-two fractions so the sums are exactly representable.
        let a = [0.25, 0.5, -0.5];
        let b = [0.25, 0.25, 0.125];
        assert_eq!(sum_clamp(&a, &b), vec![0.5, 0.75, -0.375]);
    }

    #[test]
    fn sum_clamp_saturates_at_unit() {
        // 0.8 + 0.7 = 1.5 → clamps to 1.0; -0.8 + -0.7 = -1.5 → -1.0.
        let a = [0.8, -0.8, 0.5];
        let b = [0.7, -0.7, 0.5];
        assert_eq!(sum_clamp(&a, &b), vec![1.0, -1.0, 1.0]);
    }

    #[test]
    fn sum_clamp_zero_fills_shorter_source() {
        // `b` is shorter: its missing tail contributes silence, so `a`'s tail
        // passes through clamped.
        let a = [0.3, 0.4, 0.9, -0.9];
        let b = [0.2, 0.2];
        assert_eq!(sum_clamp(&a, &b), vec![0.5, 0.6, 0.9, -0.9]);
    }

    #[test]
    fn drain_paired_mixes_known_two_stream_input() {
        let mut m = MixState::new();
        m.push_mic(&[0.1, 0.2, 0.3, 0.4]);
        m.push_sys(&[0.5, 0.5, 0.5, 0.5]);
        let batch = m.drain_paired().expect("a mixed batch");
        assert_eq!(batch.samples, vec![0.6, 0.7, 0.8, 0.9]);
        assert_eq!(batch.start_ms, 0);
        // 4 samples at 16 kHz → end clock still rounds to 0 ms (sub-ms run),
        // but the sample clock advanced so the next batch starts after it.
        assert!(m.drain_paired().is_none(), "nothing buffered after a full drain");
    }

    #[test]
    fn drain_paired_only_consumes_the_common_prefix() {
        // mic has more samples than sys; only the 2 paired samples are emitted,
        // and mic's surplus stays buffered (NOT zero-filled mid-stream).
        let mut m = MixState::new();
        m.push_mic(&[0.1, 0.2, 0.3, 0.4]);
        m.push_sys(&[0.5, 0.5]);
        let batch = m.drain_paired().expect("a mixed batch");
        assert_eq!(batch.samples, vec![0.6, 0.7]);
        // The surplus mic samples are held back, not silence-padded.
        assert!(m.drain_paired().is_none());
        // When sys catches up, the held-back mic samples mix with the new sys.
        m.push_sys(&[0.0, 0.0]);
        let batch2 = m.drain_paired().expect("a second mixed batch");
        assert_eq!(batch2.samples, vec![0.3, 0.4]);
    }

    #[test]
    fn drain_remaining_zero_fills_a_lagging_source_at_end_of_stream() {
        // The loopback source ends having delivered fewer samples than the mic.
        // On flush the mic surplus is emitted, zero-filling the absent loopback
        // so the timeline still advances.
        let mut m = MixState::new();
        m.push_mic(&[0.1, 0.2, 0.3]);
        m.push_sys(&[0.5]);
        let paired = m.drain_paired().expect("paired prefix");
        assert_eq!(paired.samples, vec![0.6]);
        let flushed = m.drain_remaining().expect("flushed mic tail");
        assert_eq!(flushed.samples, vec![0.2, 0.3], "mic tail zero-filled for sys");
        assert!(m.drain_remaining().is_none());
    }

    #[test]
    fn drain_starved_flushes_mic_when_sys_idle_beyond_cap() {
        // Regression: WASAPI loopback delivers nothing when no system audio
        // plays, so `sys` stays empty. The mic must still be emitted (zero-fill
        // sys) instead of buffering forever — the bug behind "mic not picked up
        // with capture_system_audio on".
        let mut m = MixState::new();
        m.push_mic(&vec![0.3; MAX_SKEW_SAMPLES + 100]);
        // Nothing on sys → drain_paired emits nothing.
        assert!(m.drain_paired().is_none(), "no common prefix with empty sys");
        let batch = m.drain_starved().expect("mic lead flushed past the cap");
        assert_eq!(batch.samples.len(), MAX_SKEW_SAMPLES + 100);
        assert!(batch.samples.iter().all(|&s| (s - 0.3).abs() < 1e-6));
        assert!(m.drain_starved().is_none(), "buffer drained");
    }

    #[test]
    fn drain_starved_holds_below_cap() {
        // A small mic lead with idle sys is NOT force-flushed — bounded latency,
        // gives the other source a chance to catch up first.
        let mut m = MixState::new();
        m.push_mic(&vec![0.3; MAX_SKEW_SAMPLES - 1]);
        assert!(m.drain_starved().is_none(), "under cap → no forced flush");
    }

    #[test]
    fn drain_starved_noop_when_both_sources_active() {
        // Both non-empty → pairing handles it; the valve must not fire (no
        // silence injected for a merely-slow source).
        let mut m = MixState::new();
        m.push_mic(&vec![0.3; MAX_SKEW_SAMPLES + 50]);
        m.push_sys(&[0.1; 10]);
        assert!(
            m.drain_starved().is_none(),
            "valve must not fire while both buffers hold samples",
        );
    }

    #[test]
    fn drain_starved_flushes_sys_when_mic_idle_beyond_cap() {
        // Symmetric: a dead mic must not block call/loopback audio either.
        let mut m = MixState::new();
        m.push_sys(&vec![0.2; MAX_SKEW_SAMPLES + 5]);
        let batch = m.drain_starved().expect("sys lead flushed past the cap");
        assert_eq!(batch.samples.len(), MAX_SKEW_SAMPLES + 5);
        assert!(batch.samples.iter().all(|&s| (s - 0.2).abs() < 1e-6));
    }

    #[test]
    fn timestamps_advance_monotonically_across_batches() {
        let mut m = MixState::new();
        // 16_000 mic + sys samples → exactly 1000 ms, emitted in two halves.
        m.push_mic(&vec![0.0; 8_000]);
        m.push_sys(&vec![0.0; 8_000]);
        let b1 = m.drain_paired().expect("first half");
        assert_eq!(b1.start_ms, 0);
        assert_eq!(b1.end_ms, 500);
        m.push_mic(&vec![0.0; 8_000]);
        m.push_sys(&vec![0.0; 8_000]);
        let b2 = m.drain_paired().expect("second half");
        assert_eq!(b2.start_ms, 500, "next batch starts where the last ended");
        assert_eq!(b2.end_ms, 1000);
    }
}
