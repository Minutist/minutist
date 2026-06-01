/**
 * HTML clipboard serialiser (FR-17).
 *
 * Produces the `text/html` payload that, when written to the system clipboard,
 * lets a paste into Microsoft Word (or any rich-text target) retain the notes'
 * formatting (headings, bold/italic, lists, tables, links).
 *
 * The Word paste itself is a manual fidelity check;
 * this module exists so the serialiser is deterministic and unit-tested. The
 * `data-anchor-ms` attributes the paragraph-anchor extension stamps are
 * stripped from the copied HTML — anchors are an internal cross-reference
 * concern, not something the user wants pasted into Word.
 */

/** A clipboard payload: the two MIME types we write on copy. */
export type ClipboardPayload = {
  "text/html": string;
  "text/plain": string;
};

/**
 * Wrap raw body HTML in a minimal, self-contained HTML document.
 *
 * Word's HTML importer is tolerant but pastes most reliably from a full
 * `<html><head><meta charset><body>…` document with a UTF-8 charset. The body
 * markup is the editor's own serialised HTML (already valid ProseMirror DOM).
 */
export function wrapHtmlDocument(bodyHtml: string): string {
  return (
    `<!DOCTYPE html><html><head>` +
    `<meta charset="utf-8">` +
    `</head><body>${bodyHtml}</body></html>`
  );
}

/**
 * Remove `data-anchor-ms` attributes from a fragment of HTML.
 *
 * Implemented as a string transform (no DOM dependency) so it is identical in
 * jsdom and in the Tauri webview, and trivially unit-testable. Matches the
 * attribute with single- or double-quoted values and with surrounding
 * whitespace, and collapses the residual double-space.
 */
export function stripAnchorAttributes(html: string): string {
  return html
    .replace(/\s*data-anchor-ms=(?:"[^"]*"|'[^']*')/g, "")
    .replace(/<(\w+)\s+>/g, "<$1>");
}

/**
 * Build the clipboard payload from the editor's serialised HTML and plain text.
 *
 * @param html  The editor's `getHTML()` output (ProseMirror DOM serialisation).
 * @param text  The editor's plain-text projection (`state.doc.textBetween`,
 *              typically newline-joined). Used for the `text/plain` fallback.
 */
export function buildClipboardPayload(
  html: string,
  text: string,
): ClipboardPayload {
  const cleaned = stripAnchorAttributes(html);
  return {
    "text/html": wrapHtmlDocument(cleaned),
    "text/plain": text,
  };
}
