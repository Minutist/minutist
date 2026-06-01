/**
 * Shared helpers for headless Tiptap editor behaviour tests.
 *
 * The editor runs against jsdom (no live recording, CI-safe). Typing is
 * simulated faithfully: each character is fed through the editor view's
 * `handleTextInput` prop (the same path ProseMirror uses for real keystrokes),
 * so both markdown input rules and the paragraph-anchor `appendTransaction`
 * fire exactly as they would on a real keypress.
 */
import { Editor } from "@tiptap/core";
import { TextSelection } from "@tiptap/pm/state";

/** Type a single character at the current selection, firing input rules. */
export function typeChar(editor: Editor, ch: string): void {
  const { state, view } = editor;
  const { from, to } = state.selection;
  // `handleTextInput` props take a `deflt` thunk (the default text-insertion
  // transaction) as their 5th argument; provide it so the prop chain matches
  // the real ProseMirror keystroke path.
  const deflt = () => view.state.tr.insertText(ch, from, to);
  const handled = view.someProp("handleTextInput", (f) =>
    f(view, from, to, ch, deflt),
  );
  if (!handled) {
    view.dispatch(view.state.tr.insertText(ch, from, to));
  }
}

/** Type a string, character by character. */
export function typeText(editor: Editor, text: string): void {
  for (const ch of text) typeChar(editor, ch);
}

/** Move the cursor to the end of the document. */
export function placeCursorAtEnd(editor: Editor): void {
  const { state, view } = editor;
  const end = Math.max(0, state.doc.content.size - 1);
  view.dispatch(state.tr.setSelection(TextSelection.create(state.doc, end)));
}

/** Press Enter (split the current block into a new paragraph). */
export function pressEnter(editor: Editor): void {
  editor.commands.splitBlock();
}

/**
 * Collect the `data-anchor-ms` attribute of every paragraph in document order.
 * `null` entries are unanchored paragraphs.
 */
export function paragraphAnchors(editor: Editor): Array<number | null> {
  const anchors: Array<number | null> = [];
  editor.state.doc.descendants((node) => {
    if (node.type.name === "paragraph") {
      const value = node.attrs["data-anchor-ms"];
      anchors.push(typeof value === "number" ? value : null);
    }
  });
  return anchors;
}
