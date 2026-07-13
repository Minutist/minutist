/** Public surface of the notes editor component. */
export { Editor } from "./Editor";
export { ParagraphAnchor, ANCHOR_ATTR } from "./paragraph-anchor";
export type { AnchorClockSource } from "./paragraph-anchor";
export { buildEditorExtensions } from "./extensions";
export { useAutosave, activeMeetingId } from "./useAutosave";
export {
  buildClipboardPayload,
  stripAnchorAttributes,
  wrapHtmlDocument,
} from "./clipboard";
export type { ClipboardPayload } from "./clipboard";
export { TranscriptChip, CHIP_NODE_NAME } from "./transcript-chip";
export type { TranscriptChipAttrs } from "./transcript-chip";
export { NoteImage, resolveImageSrc } from "./note-image";
export type { MeetingIdSource } from "./note-image";
export { AttachmentRef, ATTACHMENT_REF_NODE_NAME, isImageExt } from "./attachment-ref";
export type { AttachmentRefAttrs } from "./attachment-ref";
export {
  handleAttachmentPaste,
  handleAttachmentDrop,
  filesFromDataTransfer,
} from "./attachment-drop";
export { NotesHoverBridge } from "./hover-bridge";
export type { HoverAnchorReporter } from "./hover-bridge";
export {
  TRANSCRIPT_SEGMENT_MIME,
  writeSegmentDrag,
  readSegmentDrag,
  segmentToDragPayload,
  dragPayloadToChipAttrs,
  handleSegmentDrop,
} from "./transcript-dnd";
export type { DraggedSegment } from "./transcript-dnd";
export {
  nearestAnchoredElement,
  scrollToNearestAnchor,
} from "./scroll-to-anchor";
