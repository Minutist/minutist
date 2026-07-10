//! Cross-language Yjs interop fixture test (W-1b).
//!
//! Proves that lib0 v1 bytes produced by the REAL JavaScript yjs library
//! (via `@tiptap/extension-collaboration`) are accepted by [`apply_update_v1`]
//! and that [`ydoc_to_json`] projects them back to the expected ProseMirror JSON.
//!
//! # Fixture provenance
//!
//! `fixtures/notes_yjs_v1.bin` and `fixtures/notes_yjs_v1.json` were produced
//! by `ui/src/__tests__/gen-yjs-interop-fixture.test.ts` using:
//!
//!   yjs                              13.6.31
//!   @tiptap/extension-collaboration   3.24.0
//!
//! To regenerate after a yjs or @tiptap version bump, delete both fixture files
//! and run:
//!
//!   cd ui && npx vitest run src/__tests__/gen-yjs-interop-fixture.test.ts
//!
//! The generator must be re-run and both fixture files committed together with
//! the version bump in the generator's header comment.
//!
//! # Document structure
//!
//! The fixture encodes a document containing:
//!   - a level-2 heading "Interop heading"
//!   - a paragraph with "plain ", then "bold-italic" with bold+italic marks,
//!     then " end"
//!   - a `transcriptChip` atom node with startMs=12400, endMs=21300,
//!     speakerId="A", text="a quoted segment"
//!
//! These three node types cover the load-bearing interop surfaces: element +
//! text round-trip, sorted multi-mark combination, and atom node with custom
//! numeric + string attributes.
//!
//! # Known encoding asymmetry (not a bug)
//!
//! yjs stores element attributes as IEEE 754 doubles (`Any::Number` / float tag
//! in lib0 v1), while the Rust `json_to_ydoc` path stores JSON integers as
//! `Any::BigInt`.  As a result, a yjs-produced update delivers `startMs` etc. as
//! `Any::Number(12400.0)` to yrs, which `any_to_json` maps to a JSON float.
//! The fixture JSON (produced by the editor's `getJSON()`) uses integer `12400`.
//! Both represent the same numeric value; the `normalize_numbers` helper in this
//! test coerces whole-valued floats to integers so the full-document comparison
//! is value-semantic rather than representation-sensitive.  Functionally this is
//! harmless: JavaScript reads `12400.0` from `notes.json` identically to `12400`.

use notes_crdt::ydoc::{apply_update_v1, new_ydoc, ydoc_to_json};
use serde_json::Value;

/// Raw lib0 v1 bytes from the real JavaScript yjs library.
const FIXTURE_BIN: &[u8] = include_bytes!("fixtures/notes_yjs_v1.bin");

/// Editor-canonical ProseMirror JSON the Rust projection must reproduce.
const FIXTURE_JSON: &str = include_str!("fixtures/notes_yjs_v1.json");

/// Apply the yjs-produced lib0 v1 fixture to a fresh yrs doc and verify the
/// JSON projection matches the editor-canonical expected JSON.
///
/// This is the cross-language seam: bytes produced by `Y.encodeStateAsUpdate`
/// (JavaScript) decoded and projected by `apply_update_v1` + `ydoc_to_json`
/// (Rust/yrs).  Specifically asserts that:
///
/// - `apply_update_v1` accepts the bytes without error
/// - the heading text "Interop heading" survives
/// - the bold+italic mark combination on "bold-italic" survives (marks sorted
///   alphabetically by type, matching `format_to_marks`)
/// - the `transcriptChip` node type survives with all its attrs intact
///   (startMs=12400, endMs=21300, speakerId="A", text="a quoted segment")
#[test]
fn yjs_bytes_apply_and_project_to_expected_json() {
    let doc = new_ydoc();
    apply_update_v1(&doc, FIXTURE_BIN).expect("apply yjs-produced v1 update");

    let actual = ydoc_to_json(&doc);

    // --- spot checks (maximally informative on failure) ---

    // Heading: element + text survival.
    let heading_text = actual["content"][0]["content"][0]["text"]
        .as_str()
        .expect("heading text node");
    assert_eq!(heading_text, "Interop heading");

    // Paragraph: bold+italic combination.  Marks must be sorted alphabetically
    // (bold < italic) by `format_to_marks`.
    let bold_italic_run = &actual["content"][1]["content"][1];
    assert_eq!(bold_italic_run["text"], "bold-italic");
    let marks = bold_italic_run["marks"]
        .as_array()
        .expect("marks array on bold-italic run");
    assert_eq!(marks.len(), 2, "bold+italic = 2 marks");
    assert_eq!(marks[0]["type"], "bold");
    assert_eq!(marks[1]["type"], "italic");

    // TranscriptChip: atom node type + all custom attrs.
    // Numeric attrs come back as f64 from yjs (see "Known encoding asymmetry"
    // above), so compare via `as_f64` rather than integer literal equality.
    let chip = &actual["content"][2];
    assert_eq!(chip["type"], "transcriptChip", "chip node type");
    assert_eq!(
        chip["attrs"]["startMs"].as_f64(),
        Some(12400.0),
        "chip startMs"
    );
    assert_eq!(
        chip["attrs"]["endMs"].as_f64(),
        Some(21300.0),
        "chip endMs"
    );
    assert_eq!(chip["attrs"]["speakerId"], "A", "chip speakerId");
    assert_eq!(
        chip["attrs"]["text"],
        "a quoted segment",
        "chip text"
    );

    // --- full document equality (value-semantic) ---
    //
    // Both JSON trees are normalized: whole-valued floats are coerced to
    // integers so that `12400.0` (Rust projection of a yjs Number attr) and
    // `12400` (editor's getJSON() integer) compare equal.  All other values
    // are compared exactly.
    let actual_norm = normalize_numbers(actual);
    let expected: Value =
        serde_json::from_str(FIXTURE_JSON).expect("parse fixture JSON");
    let expected_norm = normalize_numbers(expected);

    assert_eq!(
        actual_norm, expected_norm,
        "full document JSON must match fixture (numbers value-normalized)"
    );
}

/// Recursively normalize a JSON value: convert any float `Number` whose value
/// is a whole integer into an integer `Number`.  This makes the comparison
/// insensitive to the `Any::Number(12400.0)` vs integer-`12400` difference
/// that arises from yjs storing all element attrs as IEEE 754 doubles.
fn normalize_numbers(v: Value) -> Value {
    match v {
        Value::Number(n) => {
            if let Some(f) = n.as_f64() {
                if f.fract() == 0.0 && f.abs() < 9.007_199_254_740_992e15 {
                    // Whole integer within safe-integer range → use i64 form.
                    return Value::Number((f as i64).into());
                }
            }
            Value::Number(n)
        }
        Value::Array(arr) => Value::Array(arr.into_iter().map(normalize_numbers).collect()),
        Value::Object(map) => Value::Object(
            map.into_iter()
                .map(|(k, v)| (k, normalize_numbers(v)))
                .collect(),
        ),
        other => other,
    }
}
