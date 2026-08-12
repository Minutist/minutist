//! Small audio/writer helpers shared by the drain loop and the offline re-transcribe path.

use super::*;
use super::drain_loop::live_segment_label;

/// end-of-stream flush in `finalise_on_stop` can close an in-progress segment
/// from the tail audio. Shared by both stop paths (Recording and Paused) so
/// their drain semantics cannot diverge: batches accepted by the channel
/// before a pause would otherwise be stranded (the paused loop blocks on
/// cmd_rx and never reads `streams.samples`).
#[allow(clippy::too_many_arguments)]
pub(super) fn drain_samples_through_vad(
    samples_rx: &mut mpsc::Receiver<AudioFrameBatch>,
    writer: &mut MeetingWriter,
    vad_opt: &mut Option<VadChunker>,
    acc: &mut Accumulator,
    online_diarizer: Option<&Arc<OnlineDiarizer>>,
    context: &str,
) {
    while let Ok(batch) = samples_rx.try_recv() {
        push_batch(writer, &batch);
        if let Some(ref mut vad) = vad_opt {
            match vad.process_samples(&batch.samples, batch.start_ms) {
                Ok(events) => {
                    for ev in events {
                        if let VadEvent::SegmentEnd {
                            start_ms,
                            end_ms,
                            samples,
                        } = ev
                        {
                            let label = live_segment_label(online_diarizer, &samples);
                            acc.append(start_ms, end_ms, &samples, label);
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!(
                        target: "orchestrator",
                        "VAD process_samples during {context}: {e}"
                    );
                }
            }
        }
    }
}

pub(super) fn push_batch(writer: &mut MeetingWriter, batch: &AudioFrameBatch) {
    if let Err(e) = writer.push_samples(&batch.samples) {
        tracing::error!(
            target: "orchestrator",
            "push_samples failed: {e}"
        );
    }
}

pub(super) fn drain_meter(rx: &mut mpsc::Receiver<AudioMeterFrame>, event_tx: &broadcast::Sender<AppEvent>) {
    while let Ok(mf) = rx.try_recv() {
        broadcast_meter(event_tx, mf);
    }
}

pub(super) fn drain_meter_nonblocking(
    rx: &mut mpsc::Receiver<AudioMeterFrame>,
    event_tx: &broadcast::Sender<AppEvent>,
) {
    drain_meter(rx, event_tx);
}

fn broadcast_meter(tx: &broadcast::Sender<AppEvent>, frame: AudioMeterFrame) {
    let event = AppEvent::AudioMeter { frame };
    match tx.send(event) {
        Ok(_) => {}
        Err(broadcast::error::SendError(_)) => {
            // No subscribers — silent drop is correct.
        }
    }
}
