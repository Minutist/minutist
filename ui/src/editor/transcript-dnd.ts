/**
 * Native HTML5 drag-and-drop bridge for transcript segments (FR-24).
 *
 * The transcript pane marks each segment row `draggable` and, on dragstart,
 * writes the segment as JSON onto a private `dataTransfer` MIME type. The notes
 * editor listens for `drop` and, when the payload carries our MIME type,
 * inserts a `transcriptChip` node at the drop position.
 *
 * Native DnD (rather than a JS DnD library) keeps the dependency surface
 * minimal — the segment payload is small and self-describing, and ProseMirror's
 * own drop handling is bypassed only when our MIME type is present (so internal
 * editor drags are unaffected).
 */
import type { Editor } from "@tiptap/core";
import type { Segment } from "../ipc/bindings";
import type { TranscriptChipAttrs } from "./transcript-chip";

/** Private MIME type carrying a dragged transcript segment. */
export const TRANSCRIPT_SEGMENT_MIME = "application/x-meeting-app-segment";

/** The JSON payload written to `dataTransfer` for a dragged segment. */
export type DraggedSegment = {
  startMs: number;
  endMs: number;
  speakerId: string | null;
  text: string;
};

/** Map a `Segment` to the drag payload. */
export function segmentToDragPayload(segment: Segment): DraggedSegment {
  return {
    startMs: segment.start_ms,
    endMs: segment.end_ms,
    speakerId: segment.speaker_id ?? null,
    text: segment.text,
  };
}

/**
 * Write a transcript segment onto a dragstart `DataTransfer`.
 *
 * Sets both the private JSON type (consumed by the editor drop handler) and a
 * `text/plain` fallback (the bare text) so a drag into a non-editor target
 * still yields something sensible.
 */
export function writeSegmentDrag(
  dataTransfer: DataTransfer,
  segment: Segment,
): void {
  const payload = segmentToDragPayload(segment);
  dataTransfer.setData(TRANSCRIPT_SEGMENT_MIME, JSON.stringify(payload));
  dataTransfer.setData("text/plain", segment.text);
  dataTransfer.effectAllowed = "copy";
}

/** Parse a dragged-segment payload from a drop `DataTransfer`, or `null`. */
export function readSegmentDrag(
  dataTransfer: DataTransfer | null,
): DraggedSegment | null {
  if (!dataTransfer) return null;
  const raw = dataTransfer.getData(TRANSCRIPT_SEGMENT_MIME);
  if (!raw) return null;
  try {
    const parsed = JSON.parse(raw) as Partial<DraggedSegment>;
    if (
      typeof parsed.startMs !== "number" ||
      typeof parsed.endMs !== "number" ||
      typeof parsed.text !== "string"
    ) {
      return null;
    }
    return {
      startMs: parsed.startMs,
      endMs: parsed.endMs,
      speakerId: typeof parsed.speakerId === "string" ? parsed.speakerId : null,
      text: parsed.text,
    };
  } catch {
    return null;
  }
}

/** Convert a parsed drag payload to chip-node attributes. */
export function dragPayloadToChipAttrs(
  payload: DraggedSegment,
): TranscriptChipAttrs {
  return {
    startMs: payload.startMs,
    endMs: payload.endMs,
    speakerId: payload.speakerId,
    text: payload.text,
  };
}

/**
 * Insert a transcript chip into `editor` from a drop event.
 *
 * Returns `true` if the drop carried a transcript segment and a chip was
 * inserted (the caller should `preventDefault` so the browser does not also
 * paste the `text/plain` fallback); `false` for any other drop (left to
 * ProseMirror / the browser).
 *
 * When the drop has client coordinates, the chip is inserted at the document
 * position under the cursor; otherwise it is appended at the current selection.
 */
export function handleSegmentDrop(
  editor: Editor,
  event: DragEvent,
): boolean {
  const payload = readSegmentDrag(event.dataTransfer);
  if (!payload) return false;

  const attrs = dragPayloadToChipAttrs(payload);

  // Resolve the drop position from the pointer when possible so the chip lands
  // where the user released it; fall back to the current selection otherwise.
  const coords =
    typeof event.clientX === "number" && typeof event.clientY === "number"
      ? editor.view.posAtCoords({ left: event.clientX, top: event.clientY })
      : null;

  const chain = editor.chain().focus();
  if (coords) {
    chain.setTextSelection(coords.pos);
  }
  chain.insertTranscriptChip(attrs).run();
  return true;
}
