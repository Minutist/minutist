/**
 * Behaviour tests for the editor's markdown-shortcut input rules (FR-15/16/20).
 *
 * Drives a headless Tiptap editor with simulated keystrokes and asserts the
 * document transforms while typing — not snapshots.
 */
import { describe, it, expect, vi, beforeEach } from "vitest";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn(), Channel: vi.fn() }));

import { Editor } from "@tiptap/core";
import { buildEditorExtensions } from "../editor/extensions";
import { typeText, placeCursorAtEnd } from "./editor-test-utils";

function makeEditor(): Editor {
  return new Editor({
    extensions: buildEditorExtensions({
      clockSource: () => ({ recording: false, clockMs: null }),
    }),
    content: "<p></p>",
  });
}

describe("editor markdown input rules", () => {
  let editor: Editor;

  beforeEach(() => {
    editor = makeEditor();
    placeCursorAtEnd(editor);
  });

  it("`# ` becomes a level-1 heading", () => {
    typeText(editor, "# Heading");
    expect(editor.getHTML()).toContain("<h1");
    expect(editor.getHTML()).toContain("Heading");
  });

  it("`## ` becomes a level-2 heading", () => {
    typeText(editor, "## Sub");
    expect(editor.getHTML()).toContain("<h2");
  });

  it("`- ` becomes a bullet list", () => {
    typeText(editor, "- item");
    const html = editor.getHTML();
    expect(html).toContain("<ul");
    expect(html).toContain("<li");
  });

  it("`1. ` becomes an ordered list", () => {
    typeText(editor, "1. item");
    const html = editor.getHTML();
    expect(html).toContain("<ol");
    expect(html).toContain("<li");
  });

  it("`> ` becomes a blockquote", () => {
    typeText(editor, "> quoted");
    expect(editor.getHTML()).toContain("<blockquote");
  });

  it("`**bold** ` produces a bold mark", () => {
    typeText(editor, "**bold** ");
    expect(editor.getHTML()).toContain("<strong>bold</strong>");
  });

  it("`*italic* ` produces an italic mark", () => {
    typeText(editor, "*italic* ");
    expect(editor.getHTML()).toContain("<em>italic</em>");
  });

  it("Typography rewrites `--` into an em dash", () => {
    typeText(editor, "a--b ");
    // Typography emDash input rule turns `--` into the em-dash character.
    expect(editor.getText()).toContain("—");
  });
});
