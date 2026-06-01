/**
 * Webview-side `AppEvent` view that anticipates the `recording_clock` variant.
 *
 * The `AppEvent::RecordingClock { meeting_id, clock_ms }` variant exists in
 * `crates/common` (wire tag `"recording_clock"`) but is NOT yet present in the
 * generated `ui/src/ipc/bindings.ts` — Stream S3 wires it through `ipc-bridge`
 * and regenerates the bindings at integration time.
 *
 * Rather than hand-edit the generated bindings file (forbidden — see
 * `architecture/domain-ownership.md`), this module augments the generated
 * `AppEvent` union locally so the store's `handleEvent` can switch on the
 * forthcoming variant in a type-safe way. Once S3 regenerates `bindings.ts`,
 * the generated union will already include `recording_clock`; the extra member
 * here is then a harmless duplicate of the same shape and this file collapses
 * to `export type AppEvent = GeneratedAppEvent`.
 */
import type { AppEvent as GeneratedAppEvent, MeetingId } from "./bindings";

/**
 * The live recording clock advanced. `clock_ms` is the capture-sample,
 * pause-**excluding** offset from the start of the recording (same timeline as
 * `Segment::start_ms`). Source of truth for notes paragraph anchors.
 */
export type RecordingClockEvent = {
  kind: "recording_clock";
  meeting_id: MeetingId;
  clock_ms: number;
};

/**
 * The generated `AppEvent` union plus the not-yet-generated `recording_clock`
 * variant. Use this type at the webview event boundary.
 */
export type AppEvent = GeneratedAppEvent | RecordingClockEvent;
