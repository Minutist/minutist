/**
 * Notes-editor context-menu entry tests (issue #0034).
 *
 * Drives a headless Tiptap editor (no live recording, CI-safe) and asserts
 * the entries `buildEditorMenuEntries` returns reflect the editor's actual
 * selection/mark state, and that invoking an entry's `onSelect` produces the
 * expected document change. Cut/Copy/Paste are NOT exercised here — they
 * delegate to `document.execCommand`, which jsdom does not implement
 * meaningfully; that wiring is a thin pass-through with nothing of its own to
 * assert beyond "it calls execCommand", which would test the mock, not the
 * behaviour.
 */
import { describe, it, expect, beforeEach } from "vitest";

import { Editor } from "@tiptap/core";
import { TextSelection } from "@tiptap/pm/state";
import { buildEditorExtensions } from "../editor/extensions";
import { buildEditorMenuEntries } from "../editor/editor-context-menu";
import type { ContextMenuEntry } from "../shell/ContextMenu";
import { placeCursorAtEnd, typeText } from "./editor-test-utils";

function makeEditor(): Editor {
  return new Editor({
    extensions: buildEditorExtensions({
      clockSource: () => ({ recording: false, clockMs: null }),
    }),
    content: "<p></p>",
  });
}

/** Select the whole current paragraph's text (simple single-block docs only). */
function selectAll(editor: Editor): void {
  const { state, view } = editor;
  view.dispatch(state.tr.setSelection(TextSelection.create(state.doc, 1, state.doc.content.size - 1)));
}

function findItem(entries: ContextMenuEntry[], label: string) {
  const entry = entries.find((e) => "label" in e && e.label === label);
  if (!entry || entry.kind === "submenu" || entry.kind === "divider") {
    throw new Error(`no plain item entry "${label}"`);
  }
  return entry;
}

describe("buildEditorMenuEntries", () => {
  let editor: Editor;

  beforeEach(() => {
    editor = makeEditor();
    typeText(editor, "hello world");
    placeCursorAtEnd(editor);
  });

  it("Cut/Copy are disabled with no selection, enabled with one", () => {
    const collapsed = buildEditorMenuEntries(editor);
    expect(findItem(collapsed, "Cut").disabled).toBe(true);
    expect(findItem(collapsed, "Copy").disabled).toBe(true);

    selectAll(editor);
    const withSelection = buildEditorMenuEntries(editor);
    expect(findItem(withSelection, "Cut").disabled).toBe(false);
    expect(findItem(withSelection, "Copy").disabled).toBe(false);
  });

  it("Paste is always enabled", () => {
    expect(findItem(buildEditorMenuEntries(editor), "Paste").disabled).toBeFalsy();
  });

  it("Bold toggles and reflects the active mark", () => {
    selectAll(editor);
    expect(findItem(buildEditorMenuEntries(editor), "Bold").checked).toBe(false);

    findItem(buildEditorMenuEntries(editor), "Bold").onSelect();
    expect(editor.isActive("bold")).toBe(true);
    expect(findItem(buildEditorMenuEntries(editor), "Bold").checked).toBe(true);
  });

  it("Italic toggles and reflects the active mark", () => {
    selectAll(editor);
    findItem(buildEditorMenuEntries(editor), "Italic").onSelect();
    expect(editor.isActive("italic")).toBe(true);
  });

  it("Bulleted list toggles the block type", () => {
    findItem(buildEditorMenuEntries(editor), "Bulleted list").onSelect();
    expect(editor.isActive("bulletList")).toBe(true);
  });

  it("Heading submenu applies the chosen level and marks it current", () => {
    const heading = buildEditorMenuEntries(editor).find(
      (e) => e.kind === "submenu" && e.label === "Heading",
    );
    if (!heading || heading.kind !== "submenu") throw new Error("no Heading submenu");
    const h2 = heading.items.find((i) => i.label === "Heading 2");
    if (!h2) throw new Error("no Heading 2 item");
    expect(h2.current).toBe(false);

    h2.onSelect();
    expect(editor.isActive("heading", { level: 2 })).toBe(true);

    const afterHeading = buildEditorMenuEntries(editor).find(
      (e) => e.kind === "submenu" && e.label === "Heading",
    );
    if (!afterHeading || afterHeading.kind !== "submenu") throw new Error("gone");
    const h2Again = afterHeading.items.find((i) => i.label === "Heading 2");
    expect(h2Again?.current).toBe(true);
  });

  it("offers 'Add link…' (disabled without a selection) when no link is active", () => {
    const collapsed = findItem(buildEditorMenuEntries(editor), "Add link…");
    expect(collapsed.disabled).toBe(true);

    selectAll(editor);
    expect(findItem(buildEditorMenuEntries(editor), "Add link…").disabled).toBe(false);
  });

  it("offers 'Remove link' instead of 'Add link…' once a link mark is active", () => {
    selectAll(editor);
    editor.chain().focus().setLink({ href: "https://example.com" }).run();
    expect(editor.isActive("link")).toBe(true);

    const entries = buildEditorMenuEntries(editor);
    expect(entries.some((e) => "label" in e && e.label === "Add link…")).toBe(false);
    findItem(entries, "Remove link").onSelect();
    expect(editor.isActive("link")).toBe(false);
  });

  it("includes a divider between clipboard and formatting entries", () => {
    const entries = buildEditorMenuEntries(editor);
    expect(entries.some((e) => e.kind === "divider")).toBe(true);
  });
});
