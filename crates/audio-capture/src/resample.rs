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
        })
    }

    /// Push mono samples at the input rate; calls `emit` with each batch of
    /// resampled 16 kHz samples produced.
    ///
    /// Returns `Err` if rubato rejects a full input block (e.g. non-finite
    /// samples); the failed block is dropped (not emitted) but any other full
    /// blocks in `src` are still processed. The caller logs a warning rather
    /// than propagating further — a resampler hiccup must not stop capture.
    pub fn push(&mut self, mut src: &[f32], emit: &mut impl FnMut(&[f32])) -> Result<(), Error> {
        if self.inner.is_none() {
            emit(src);
            return Ok(());
        }

        let mut last_err = None;
        while !src.is_empty() {
            let space = CHUNK_IN - self.in_buf.len();
            let take = space.min(src.len());
            self.in_buf.extend_from_slice(&src[..take]);
            src = &src[take..];

            if self.in_buf.len() == CHUNK_IN {
                let result = self
                    .inner
                    .as_mut()
                    .unwrap()
                    .process(&[&self.in_buf[..]], None);
                self.in_buf.clear();
                if let Err(e) = handle_process_result(result, emit) {
                    last_err = Some(e);
                }
            }
        }
        last_err.map_or(Ok(()), Err)
    }

    /// Flush any partially-filled input buffer, zero-padding to `CHUNK_IN`.
    /// Called on stop to drain the tail.
    pub fn finish(&mut self, emit: &mut impl FnMut(&[f32])) -> Result<(), Error> {
        if let Some(ref mut r) = self.inner {
            if !self.in_buf.is_empty() {
                self.in_buf.resize(CHUNK_IN, 0.0);
                let result = r.process(&[&self.in_buf[..]], None);
                self.in_buf.clear();
                return handle_process_result(result, emit);
            }
        }
        Ok(())
    }
}

/// Turn one rubato `process()` result into `Result<(), Error>`: on success,
/// call `emit` with the resampled block; on failure, produce `Error::Resampler`
/// without calling `emit` — a failed block is dropped rather than silently
/// forwarded or lost without trace.
fn handle_process_result(
    result: rubato::ResampleResult<Vec<Vec<f32>>>,
    emit: &mut impl FnMut(&[f32]),
) -> Result<(), Error> {
    match result {
        Ok(out) => {
            emit(&out[0]);
            Ok(())
        }
        Err(e) => Err(Error::Resampler {
            context: e.to_string(),
        }),
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
        r.push(&input, &mut |chunk| got.extend_from_slice(chunk))
            .expect("pass-through push must not error");
        // Pass-through: all samples forwarded immediately, order preserved.
        assert_eq!(got, input);
    }

    #[test]
    fn resampler_constructs_at_48k() {
        let r = StreamResampler::new(48_000);
        assert!(r.is_ok(), "rubato failed to construct 48k->16k resampler");
    }

    /// A rubato process() failure must surface as `Error::Resampler` — not be
    /// silently swallowed — and `emit` must not be called for the failed block.
    #[test]
    fn process_error_surfaces_as_resampler_error_without_emitting() {
        let result: rubato::ResampleResult<Vec<Vec<f32>>> =
            Err(rubato::ResampleError::WrongNumberOfInputChannels {
                expected: 1,
                actual: 2,
            });
        let mut emitted = false;
        let err = handle_process_result(result, &mut |_| emitted = true)
            .expect_err("a rubato error must propagate, not be discarded");
        assert!(
            matches!(err, Error::Resampler { .. }),
            "expected Error::Resampler, got {err:?}"
        );
        assert!(!emitted, "emit must not run for a failed block");
    }

    /// A successful process() result still emits exactly the resampled block.
    #[test]
    fn process_success_emits_and_returns_ok() {
        let result: rubato::ResampleResult<Vec<Vec<f32>>> = Ok(vec![vec![1.0, 2.0, 3.0]]);
        let mut got = Vec::new();
        handle_process_result(result, &mut |chunk| got.extend_from_slice(chunk))
            .expect("a successful process() result must not error");
        assert_eq!(got, vec![1.0, 2.0, 3.0]);
    }
}
