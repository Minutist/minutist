/**
 * Fixture generator for the cross-language Yjs interop test (W-1b).
 *
 * Run once to regenerate the committed binary fixture consumed by the Rust
 * integration test `crates/persistence/tests/yjs_interop.rs`:
 *
 *   cd ui && REGEN_YJS_FIXTURE=1 npx vitest run src/__tests__/gen-yjs-interop-fixture.test.ts
 *
 * Without REGEN_YJS_FIXTURE set (i.e. a normal `npm test` run) the test still
 * drives the real binding and asserts the emitted bytes, but does NOT overwrite
 * the committed fixture — yjs assigns a random Y.Doc clientID per run, so an
 * unconditional write would rewrite `notes_yjs_v1.bin` with different bytes
 * every run and leave the working tree dirty.
 *
 * Fixture produced (path relative to repo root):
 *   crates/persistence/tests/fixtures/notes_yjs_v1.bin   — lib0 v1 bytes
 *
 * The binary is produced by the REAL yjs + @tiptap/extension-collaboration
 * libraries (not yrs, not hand-encoded) so it captures what the live editor
 * actually emits and what `apply_update_v1` on the Rust side must accept.
 *
 * The companion `notes_yjs_v1.json` is the RUST-CANONICAL expected JSON
 * produced by `ydoc_to_json` after decoding the binary. It is hand-maintained
 * because two known encoding differences exist between the editor's `getJSON()`
 * and the Rust projection:
 *
 *   1. Numeric attrs: yjs stores element attrs as IEEE 754 doubles; yrs
 *      decodes them as `Any::Number` (float); `any_to_json` emits float JSON
 *      (e.g. `12400.0`). The JSON file uses integer literals because
 *      `normalize_numbers` in the Rust test coerces whole-valued floats to
 *      integers before comparison.
 *
 *   2. Null attrs: yjs stores `data-anchor-ms: null` (the ParagraphAnchor
 *      default) as an attribute value. In yrs semantics, `Any::Null` in the
 *      attribute map is a tombstone — the attribute is considered absent —
 *      so `element.attributes(txn)` does not iterate it and `ydoc_to_json`
 *      omits it. The JSON file therefore has no `attrs` key on plain
 *      paragraphs, even though the editor's `getJSON()` would emit
 *      `{"attrs": {"data-anchor-ms": null}}`. This is lossless: the
 *      ParagraphAnchor extension defaults `data-anchor-ms` to `null` when
 *      absent on import.
 *
 * When regenerating (e.g. after a yjs / @tiptap version bump):
 *   1. Re-run this generator (updates `notes_yjs_v1.bin`).
 *   2. Run `cargo test -p persistence --test yjs_interop` to verify.
 *   3. If the Rust test fails, inspect the divergence and update
 *      `notes_yjs_v1.json` to match the new Rust projection.
 *   4. Commit `.bin`, `.json`, and this file together.
 *   5. Bump the version comment below.
 *
 * Versions at fixture-generation time:
 *   yjs                              13.6.31
 *   @tiptap/extension-collaboration   3.24.0
 *
 * This file is in src/__tests__/ so vitest resolves the jsdom environment and
 * the same module resolution / mocks used by the other editor tests apply.
 */

import { it, expect } from "vitest";
import * as fs from "node:fs";
import * as path from "node:path";
import * as Y from "yjs";
import { Editor } from "@tiptap/core";
import { buildEditorExtensions } from "../editor/extensions";

// ---------------------------------------------------------------------------
// Tauri stub — note-image.ts imports `convertFileSrc` from @tauri-apps/api/core
// which is unavailable in jsdom.  The generator doesn't exercise image nodes
// so the stub is a no-op.
// ---------------------------------------------------------------------------
vi.mock("@tauri-apps/api/core", () => ({
  invoke: vi.fn(),
  Channel: vi.fn(),
  convertFileSrc: (src: string) => src,
}));

// ---------------------------------------------------------------------------
// The canonical fixture document.
//
// Three node types chosen to exercise the load-bearing interop surfaces:
//   1. heading        — element + text round-trip
//   2. paragraph      — bold+italic mark combination (marks sorted in output)
//   3. transcriptChip — atom node with custom numeric + string attrs
//
// Do NOT change this document without also updating notes_yjs_v1.json and the
// assertions in crates/persistence/tests/yjs_interop.rs.
// ---------------------------------------------------------------------------
const FIXTURE_DOC = {
  type: "doc",
  content: [
    {
      type: "heading",
      attrs: { level: 2 },
      content: [{ type: "text", text: "Interop heading" }],
    },
    {
      type: "paragraph",
      content: [
        { type: "text", text: "plain " },
        {
          type: "text",
          marks: [{ type: "bold" }, { type: "italic" }],
          text: "bold-italic",
        },
        { type: "text", text: " end" },
      ],
    },
    {
      type: "transcriptChip",
      attrs: {
        startMs: 12400,
        endMs: 21300,
        speakerId: "A",
        text: "a quoted segment",
      },
    },
  ],
};

it("generates yjs interop binary fixture", () => {
  // Produce lib0 v1 bytes from the REAL yjs + tiptap binding.
  const ydoc = new Y.Doc();
  const editor = new Editor({
    extensions: buildEditorExtensions({
      clockSource: () => ({ recording: false, clockMs: null }),
      collabDoc: ydoc,
    }),
  });
  editor.commands.setContent(FIXTURE_DOC);
  const v1Bytes = Y.encodeStateAsUpdate(ydoc);
  editor.destroy();

  // Sanity: the binary must contain the prosemirror fragment name and our node
  // types, confirming the editor binding wrote into the Y.Doc.
  const binStr = Buffer.from(v1Bytes).toString("binary");
  expect(binStr).toContain("prosemirror");
  expect(binStr).toContain("transcriptChip");
  expect(binStr).toContain("Interop heading");
  expect(v1Bytes.length).toBeGreaterThan(100);

  // Only (re)write the committed fixture when explicitly regenerating. yjs
  // assigns a random clientID per Y.Doc, so an unconditional write would churn
  // the committed `notes_yjs_v1.bin` on every `npm test`. The Rust test consumes
  // the committed bytes; regenerate deliberately with REGEN_YJS_FIXTURE=1.
  if (process.env.REGEN_YJS_FIXTURE) {
    const fixturesDir = path.resolve(
      __dirname,
      "../../../crates/persistence/tests/fixtures",
    );
    fs.mkdirSync(fixturesDir, { recursive: true });
    const binPath = path.join(fixturesDir, "notes_yjs_v1.bin");
    fs.writeFileSync(binPath, v1Bytes);
    console.log(`wrote ${v1Bytes.length} bytes → ${binPath}`);
  }
});
