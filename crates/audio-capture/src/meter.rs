//! Audio level meter.
//!
//! Computes peak and RMS over a rolling window of ~33 ms (≈512 samples at
//! 16 kHz). Emits one `AudioMeterFrame` per window.

use meeting_app_common::AudioMeterFrame;

/// Accumulates samples and emits meter readings at a fixed window size.
pub(crate) struct LevelMeter {
    window_samples: usize,
    buf: Vec<f32>,
}

impl LevelMeter {
    /// Create a new meter.
    ///
    /// `window_samples` is the number of 16 kHz mono samples per meter
    /// window. At 16 kHz, 512 samples ≈ 32 ms ≈ 30 Hz emission rate.
    pub fn new(window_samples: usize) -> Self {
        assert!(window_samples > 0, "meter window must be > 0 samples");
        Self {
            window_samples,
            buf: Vec::with_capacity(window_samples),
        }
    }

    /// Feed samples into the meter.
    ///
    /// For every complete window accumulated, `emit` is called with the
    /// computed `AudioMeterFrame`. Partial windows are held until more
    /// samples arrive.
    pub fn push(&mut self, samples: &[f32], mut emit: impl FnMut(AudioMeterFrame)) {
        let mut remaining = samples;
        while !remaining.is_empty() {
            let space = self.window_samples - self.buf.len();
            let take = space.min(remaining.len());
            self.buf.extend_from_slice(&remaining[..take]);
            remaining = &remaining[take..];

            if self.buf.len() == self.window_samples {
                emit(compute_frame(&self.buf));
                self.buf.clear();
            }
        }
    }

    /// Flush any partial window, emitting a frame if there are buffered
    /// samples. Called on stop/pause to drain the tail.
    pub fn flush(&mut self, mut emit: impl FnMut(AudioMeterFrame)) {
        if !self.buf.is_empty() {
            emit(compute_frame(&self.buf));
            self.buf.clear();
        }
    }
}

fn compute_frame(samples: &[f32]) -> AudioMeterFrame {
    let mut peak = 0.0f32;
    let mut sum_sq = 0.0f64;
    for &s in samples {
        let abs = s.abs();
        if abs > peak {
            peak = abs;
        }
        sum_sq += (s as f64) * (s as f64);
    }
    let rms = (sum_sq / samples.len() as f64).sqrt() as f32;
    AudioMeterFrame { peak, rms }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    /// 1 kHz sine at amplitude 0.5 over a 33 ms window.
    /// Expected: peak ≈ 0.5, rms ≈ 0.5 / √2 ≈ 0.3536. Tolerance ±0.02.
    #[test]
    fn meter_math_sine_1khz() {
        const SAMPLE_RATE: u32 = 16_000;
        const AMPLITUDE: f32 = 0.5;
        const FREQ: f32 = 1_000.0;
        const WINDOW: usize = 512; // ~32 ms at 16 kHz

        let samples: Vec<f32> = (0..WINDOW)
            .map(|i| AMPLITUDE * (2.0 * PI * FREQ * i as f32 / SAMPLE_RATE as f32).sin())
            .collect();

        let mut meter = LevelMeter::new(WINDOW);
        let mut frames = Vec::new();
        meter.push(&samples, |f| frames.push(f));

        assert_eq!(frames.len(), 1, "expected exactly one meter frame");
        let frame = frames[0];

        let expected_peak = AMPLITUDE;
        let expected_rms = AMPLITUDE / 2.0_f32.sqrt();
        let tol = 0.02;

        assert!(
            (frame.peak - expected_peak).abs() <= tol,
            "peak {:.4} not within {tol} of {expected_peak}",
            frame.peak
        );
        assert!(
            (frame.rms - expected_rms).abs() <= tol,
            "rms {:.4} not within {tol} of {expected_rms}",
            frame.rms
        );
    }
}
