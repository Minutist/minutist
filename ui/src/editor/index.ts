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
