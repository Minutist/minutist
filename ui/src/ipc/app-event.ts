/**
 * Webview-side `AppEvent` view.
 *
 * The `recording_clock` variant (`AppEvent::RecordingClock { meeting_id,
 * clock_ms }`) is now present in the generated `bindings.ts` — Stream S3 wired
 * it through `ipc-bridge` and regenerated the bindings. The local augmentation
 * that anticipated it is therefore redundant and has been removed; this module
 * re-exports the generated union verbatim so the rest of the webview keeps a
 * single import site for the event type.
 */
export type { AppEvent } from "./bindings";
