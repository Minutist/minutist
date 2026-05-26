//! Sample-rate conversion using rubato `FftFixedIn`.
//!
//! Converts arbitrary-rate mono PCM to 16 kHz mono f32. If the input rate
//! already matches 16 kHz, the resampler is a no-op pass-through.

use rubato::{FftFixedIn, Resampler};

use crate::error::Error;

/// Target sample rate mandated by the workspace (16 kHz mono).
pub const TARGET_RATE: u32 = 16_000;

/// Input block size fed to rubato's FFT resampler in one call. 1024
/// samples provides reasonable latency/quality trade-off and matches
/// Handy's tested value.
const CHUNK_IN: usize = 1024;

pub(crate) struct StreamResampler {
    inner: Option<FftFixedIn<f32>>,
    in_buf: Vec<f32>,
    pending_out: Vec<f32>,
}

impl StreamResampler {
    /// Construct a resampler for `in_rate → TARGET_RATE`.
    ///
    /// If `in_rate == TARGET_RATE` the resampler is a pass-through.
    pub fn new(in_rate: u32) -> Result<Self, Error> {
        let inner = if in_rate != TARGET_RATE {
            let r = FftFixedIn::<f32>::new(in_rate as usize, TARGET_RATE as usize, CHUNK_IN, 1, 1)
                .map_err(|e| Error::Resampler {
                    context: e.to_string(),
                })?;
            Some(r)
        } else {
            None
        };

        Ok(Self {
            inner,
            in_buf: Vec::with_capacity(CHUNK_IN),
            pending_out: Vec::new(),
        })
    }

    /// Push mono samples at the input rate; calls `emit` with each batch of
    /// resampled 16 kHz samples produced.
    pub fn push(&mut self, mut src: &[f32], emit: &mut impl FnMut(&[f32])) {
        if self.inner.is_none() {
            emit(src);
            return;
        }

        while !src.is_empty() {
            let space = CHUNK_IN - self.in_buf.len();
            let take = space.min(src.len());
            self.in_buf.extend_from_slice(&src[..take]);
            src = &src[take..];

            if self.in_buf.len() == CHUNK_IN {
                if let Ok(out) = self
                    .inner
                    .as_mut()
                    .unwrap()
                    .process(&[&self.in_buf[..]], None)
                {
                    emit(&out[0]);
                }
                self.in_buf.clear();
            }
        }
    }

    /// Flush any partially-filled input buffer, zero-padding to `CHUNK_IN`.
    /// Called on stop to drain the tail.
    pub fn finish(&mut self, emit: &mut impl FnMut(&[f32])) {
        if let Some(ref mut r) = self.inner {
            if !self.in_buf.is_empty() {
                self.in_buf.resize(CHUNK_IN, 0.0);
                if let Ok(out) = r.process(&[&self.in_buf[..]], None) {
                    emit(&out[0]);
                }
                self.in_buf.clear();
            }
        }
        self.pending_out.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn passthrough_when_rates_match() {
        let mut r = StreamResampler::new(TARGET_RATE).unwrap();
        let input: Vec<f32> = (0..256).map(|i| i as f32).collect();
        let mut got = Vec::new();
        r.push(&input, &mut |chunk| got.extend_from_slice(chunk));
        // Pass-through: all samples forwarded immediately, order preserved.
        assert_eq!(got, input);
    }

    #[test]
    fn resampler_constructs_at_48k() {
        let r = StreamResampler::new(48_000);
        assert!(r.is_ok(), "rubato failed to construct 48k->16k resampler");
    }
}
