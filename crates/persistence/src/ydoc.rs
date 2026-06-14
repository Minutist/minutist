//! Yjs (`yrs`) CRDT representation of the notes document — the authoritative
//! `notes.ydoc` blob and its conversions to/from ProseMirror JSON.
//!
//! # The one relaxation of the opacity guarantee
//!
//! [`crate::NotesStore`] keeps `notes.json` opaque (see its module docs). This
//! module is the **single, narrow** place that relaxes that: deriving a
//! ProseMirror JSON projection from the Yjs document requires knowing the
//! document is ProseMirror-shaped. It does so **generically** — it walks the
//! Yjs `XmlFragment` tree node-by-node and never models a specific Tiptap node
//! type. Element tags, attributes, text marks, and nesting all round-trip by
//! structure, so unknown/custom nodes (the transcript-chip atom, note images,
//! future nodes) survive losslessly exactly as they do through the opaque JSON
//! path. Do **not** introduce a typed Tiptap node model here.
//!
//! # Mapping (matches y-prosemirror conventions)
//!
//! The document is stored under a single top-level `XmlFragment` named
//! [`PROSEMIRROR_FRAGMENT`]. The ProseMirror `doc` node itself is not
//! represented — its children are the fragment's top-level children, mirroring
//! y-prosemirror so the editor-side Yjs binding (a later work unit) interops
//! without conversion. Each non-text node becomes an `XmlElement` whose tag is
//! the ProseMirror node `type` and whose `attrs` become XML attributes (stored
//! as typed [`yrs::Any`] so numbers, booleans, nulls, arrays, and nested
//! objects round-trip). A text node becomes an `XmlText`; its `marks` become
//! the text's formatting attributes (keyed by mark `type`, valued by the mark's
//! `attrs`).
//!
//! # Encoding
//!
//! The durable blob is the lib0 **v2** whole-state encoding
//! (`encode_state_as_update_v2`), the size-optimised whole-document form
//! (`planning/DESIGN_notes-crdt.md` D-O2.4 / OQ-C).

use std::collections::HashMap;
use std::sync::Arc;

use serde_json::{Map, Value};
use yrs::types::xml::{XmlElementRef, XmlFragmentRef, XmlOut};
use yrs::types::text::YChange;
use yrs::types::Attrs;
use yrs::updates::decoder::Decode;
use yrs::{
    Any, Doc, Out, ReadTxn, StateVector, Text, Transact, TransactionMut, Update, Xml,
    XmlElementPrelim, XmlFragment, XmlTextPrelim, XmlTextRef,
};

/// The top-level `XmlFragment` name holding the notes document. Matches
/// y-prosemirror's default (`"prosemirror"`) so the editor-side binding interops.
pub const PROSEMIRROR_FRAGMENT: &str = "prosemirror";

/// Build a fresh Yjs [`Doc`] from a ProseMirror-JSON document (the seed /
/// import path — `prosemirrorJSONToYDoc`-equivalent).
///
/// The top-level `content` array becomes the children of the
/// [`PROSEMIRROR_FRAGMENT`] fragment. A document with no `content` array yields
/// an empty fragment. The new doc has fresh client history — safe only as a
/// one-time origin for a never-synced document (see D-O2.7).
pub fn json_to_ydoc(doc_json: &Value) -> Doc {
    let doc = Doc::new();
    let fragment = doc.get_or_insert_xml_fragment(PROSEMIRROR_FRAGMENT);
    if let Some(content) = doc_json.get("content").and_then(|c| c.as_array()) {
        let mut txn = doc.transact_mut();
        for (i, node) in content.iter().enumerate() {
            write_node_into_fragment(&mut txn, &fragment, i as u32, node);
        }
    }
    doc
}

/// Project a Yjs [`Doc`] back to a ProseMirror-JSON document
/// (`yDocToProsemirrorJSON`-equivalent).
///
/// Walks the [`PROSEMIRROR_FRAGMENT`] fragment generically and reconstructs the
/// `{ "type": "doc", "content": [...] }` shape the editor and the summariser's
/// `note_blocks_from_json` consume.
pub fn ydoc_to_json(doc: &Doc) -> Value {
    let fragment = doc.get_or_insert_xml_fragment(PROSEMIRROR_FRAGMENT);
    let txn = doc.transact();
    let content = read_fragment_children(&txn, &fragment);
    let mut obj = Map::new();
    obj.insert("type".to_string(), Value::String("doc".to_string()));
    obj.insert("content".to_string(), Value::Array(content));
    Value::Object(obj)
}

/// Encode a Yjs [`Doc`]'s whole state as the durable lib0 v2 blob written to
/// `notes.ydoc`.
pub fn encode_ydoc(doc: &Doc) -> Vec<u8> {
    doc.transact()
        .encode_state_as_update_v2(&StateVector::default())
}

/// Decode a `notes.ydoc` lib0 v2 blob into a Yjs [`Doc`].
///
/// Returns an error string when the blob is not a valid lib0 v2 update; the
/// caller maps it into the crate error type.
pub fn decode_ydoc(bytes: &[u8]) -> Result<Doc, String> {
    let doc = Doc::new();
    // Touch the fragment so it exists even for an empty/edge-case blob.
    let _ = doc.get_or_insert_xml_fragment(PROSEMIRROR_FRAGMENT);
    let update = Update::decode_v2(bytes).map_err(|e| format!("decode notes.ydoc v2: {e}"))?;
    {
        let mut txn = doc.transact_mut();
        txn.apply_update(update)
            .map_err(|e| format!("apply notes.ydoc update: {e}"))?;
    }
    Ok(doc)
}

// ---------------------------------------------------------------------------
// JSON → Yjs (write)
// ---------------------------------------------------------------------------

/// Insert one ProseMirror node as the `index`-th child of an `XmlFragment`.
fn write_node_into_fragment(
    txn: &mut TransactionMut,
    parent: &XmlFragmentRef,
    index: u32,
    node: &Value,
) {
    if is_text_node(node) {
        let text = parent.insert(txn, index, XmlTextPrelim::new(""));
        write_text_body(txn, &text, node);
        return;
    }
    let tag = node_type(node);
    let element = parent.insert(txn, index, XmlElementPrelim::empty(tag));
    write_element_body(txn, &element, node);
}

/// Insert one ProseMirror node as the `index`-th child of an `XmlElement`.
fn write_node_into_element(
    txn: &mut TransactionMut,
    parent: &XmlElementRef,
    index: u32,
    node: &Value,
) {
    if is_text_node(node) {
        let text = parent.insert(txn, index, XmlTextPrelim::new(""));
        write_text_body(txn, &text, node);
        return;
    }
    let tag = node_type(node);
    let element = parent.insert(txn, index, XmlElementPrelim::empty(tag));
    write_element_body(txn, &element, node);
}

/// Write an element node's `attrs` (as XML attributes) and recurse into its
/// `content`.
fn write_element_body(txn: &mut TransactionMut, element: &XmlElementRef, node: &Value) {
    if let Some(attrs) = node.get("attrs").and_then(|a| a.as_object()) {
        for (key, value) in attrs {
            element.insert_attribute(txn, key.clone(), json_to_any(value));
        }
    }
    if let Some(content) = node.get("content").and_then(|c| c.as_array()) {
        for (i, child) in content.iter().enumerate() {
            write_node_into_element(txn, element, i as u32, child);
        }
    }
}

/// Write a text node's string plus its marks (as text-format attributes).
fn write_text_body(txn: &mut TransactionMut, text: &XmlTextRef, node: &Value) {
    let body = node.get("text").and_then(|t| t.as_str()).unwrap_or("");
    let mut attrs = Attrs::new();
    if let Some(marks) = node.get("marks").and_then(|m| m.as_array()) {
        for mark in marks {
            let mark_type = mark.get("type").and_then(|t| t.as_str()).unwrap_or("");
            // A mark with no `attrs` round-trips as an empty object value, which
            // the read path renders back as a mark with no `attrs` key.
            let mark_attrs = mark.get("attrs").cloned().unwrap_or(Value::Object(Map::new()));
            attrs.insert(mark_type.into(), json_to_any(&mark_attrs));
        }
    }
    text.insert_with_attributes(txn, 0, body, attrs);
}

// ---------------------------------------------------------------------------
// Yjs → JSON (read)
// ---------------------------------------------------------------------------

/// Read all children of an `XmlFragment` as ProseMirror JSON nodes.
fn read_fragment_children<T: ReadTxn>(txn: &T, fragment: &XmlFragmentRef) -> Vec<Value> {
    let mut out = Vec::new();
    for child in fragment.children(txn) {
        read_node(txn, &child, &mut out);
    }
    out
}

/// Read all children of an `XmlElement` as ProseMirror JSON nodes.
fn read_element_children<T: ReadTxn>(txn: &T, element: &XmlElementRef) -> Vec<Value> {
    let mut out = Vec::new();
    for child in element.children(txn) {
        read_node(txn, &child, &mut out);
    }
    out
}

/// Append the ProseMirror JSON for one Yjs XML node to `out`.
///
/// A text node may expand to **several** ProseMirror text nodes — one per
/// contiguous run of identical marks — which is why this appends rather than
/// returns one value.
fn read_node<T: ReadTxn>(txn: &T, node: &XmlOut, out: &mut Vec<Value>) {
    match node {
        XmlOut::Element(element) => {
            let mut obj = Map::new();
            obj.insert("type".to_string(), Value::String(element.tag().to_string()));
            let mut attrs = Map::new();
            for (key, value) in element.attributes(txn) {
                attrs.insert(key.to_string(), out_to_json(&value));
            }
            if !attrs.is_empty() {
                obj.insert("attrs".to_string(), Value::Object(attrs));
            }
            let content = read_element_children(txn, element);
            if !content.is_empty() {
                obj.insert("content".to_string(), Value::Array(content));
            }
            out.push(Value::Object(obj));
        }
        XmlOut::Text(text) => {
            // `diff` returns the text split into runs of identical formatting.
            for chunk in text.diff(txn, YChange::identity) {
                let body = match &chunk.insert {
                    Out::Any(Any::String(s)) => s.to_string(),
                    _ => String::new(),
                };
                let mut obj = Map::new();
                obj.insert("type".to_string(), Value::String("text".to_string()));
                obj.insert("text".to_string(), Value::String(body));
                if let Some(format) = &chunk.attributes {
                    let marks = format_to_marks(format);
                    if !marks.is_empty() {
                        obj.insert("marks".to_string(), Value::Array(marks));
                    }
                }
                out.push(Value::Object(obj));
            }
        }
        // A nested XmlFragment is not produced by the writer; skip defensively.
        XmlOut::Fragment(_) => {}
    }
}

/// Convert a text run's formatting attributes back into ProseMirror `marks`.
///
/// yrs stores a text run's formatting in an [`Attrs`] (a `HashMap`), whose
/// iteration order is non-deterministic. Emit the marks sorted by type so the
/// derived `notes.json` is stable across re-derives — a run with several marks
/// (e.g. bold + italic) must not reorder on disk on every save. ProseMirror
/// re-normalises mark order on import, so canonicalising to sorted order here is
/// lossless for the editor and removes the on-disk churn.
fn format_to_marks(format: &Attrs) -> Vec<Value> {
    let mut entries: Vec<_> = format.iter().collect();
    entries.sort_by(|a, b| a.0.as_ref().cmp(b.0.as_ref()));
    let mut marks = Vec::new();
    for (key, value) in entries {
        let mut mark = Map::new();
        mark.insert("type".to_string(), Value::String(key.to_string()));
        let mark_attrs = any_to_json(value);
        // Only emit an `attrs` key when the mark actually carries attributes,
        // so a bare mark (e.g. `bold`) round-trips to `{ "type": "bold" }`.
        if let Value::Object(obj) = &mark_attrs {
            if !obj.is_empty() {
                mark.insert("attrs".to_string(), mark_attrs);
            }
        }
        marks.push(Value::Object(mark));
    }
    marks
}

// ---------------------------------------------------------------------------
// JSON value ↔ yrs Any
// ---------------------------------------------------------------------------

/// Convert a `serde_json::Value` into a typed [`yrs::Any`] for storage as an
/// XML / text-format attribute. Integers map to `BigInt`, other numbers to
/// `Number`, preserving the JSON value shape on read-back.
fn json_to_any(value: &Value) -> Any {
    match value {
        Value::Null => Any::Null,
        Value::Bool(b) => Any::Bool(*b),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Any::BigInt(i)
            } else {
                Any::Number(n.as_f64().unwrap_or(0.0))
            }
        }
        Value::String(s) => Any::String(s.as_str().into()),
        Value::Array(arr) => Any::Array(arr.iter().map(json_to_any).collect::<Vec<_>>().into()),
        Value::Object(obj) => {
            let map: HashMap<String, Any> = obj
                .iter()
                .map(|(k, v)| (k.clone(), json_to_any(v)))
                .collect();
            Any::Map(Arc::new(map))
        }
    }
}

/// Convert a typed [`yrs::Any`] back into a `serde_json::Value`.
fn any_to_json(any: &Any) -> Value {
    match any {
        Any::Null | Any::Undefined => Value::Null,
        Any::Bool(b) => Value::Bool(*b),
        Any::Number(n) => serde_json::Number::from_f64(*n)
            .map(Value::Number)
            .unwrap_or(Value::Null),
        Any::BigInt(i) => Value::Number((*i).into()),
        Any::String(s) => Value::String(s.to_string()),
        Any::Buffer(_) => Value::Null,
        Any::Array(arr) => Value::Array(arr.iter().map(any_to_json).collect()),
        Any::Map(map) => Value::Object(
            map.iter()
                .map(|(k, v)| (k.clone(), any_to_json(v)))
                .collect(),
        ),
    }
}

/// Convert a read-back attribute [`Out`] into JSON. Attributes are always
/// stored as [`Any`], so any other `Out` variant is unexpected and maps to null.
fn out_to_json(out: &Out) -> Value {
    match out {
        Out::Any(any) => any_to_json(any),
        _ => Value::Null,
    }
}

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

fn is_text_node(node: &Value) -> bool {
    node.get("type").and_then(|t| t.as_str()) == Some("text")
}

fn node_type(node: &Value) -> &str {
    node.get("type").and_then(|t| t.as_str()).unwrap_or("")
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// `JSON → yrs doc → re-derived JSON` for one document. The CRDT analogue
    /// of the `NotesStore` opacity test.
    fn round_trip(input: &Value) -> Value {
        let doc = json_to_ydoc(input);
        ydoc_to_json(&doc)
    }

    /// Assert a document survives `JSON → yrs → JSON` unchanged, and also
    /// survives the durable v2 encode/decode hop (so the on-disk blob is faithful).
    fn assert_round_trips(input: Value) {
        assert_eq!(round_trip(&input), input, "JSON -> yrs -> JSON must be lossless");

        // Encode to the durable blob, decode, re-derive — the on-disk path.
        let doc = json_to_ydoc(&input);
        let bytes = encode_ydoc(&doc);
        let decoded = decode_ydoc(&bytes).expect("decode v2 blob");
        assert_eq!(
            ydoc_to_json(&decoded),
            input,
            "JSON -> yrs -> v2 blob -> yrs -> JSON must be lossless"
        );
    }

    /// The full Tiptap editor schema in one document: StarterKit blocks +
    /// marks, Link (a mark with attrs), lists, blockquote, code block, headings,
    /// the ParagraphAnchor `data-anchor-ms` attr, the TranscriptChip atom, the
    /// NoteImage node, and a Table with header/row/cell. The CRDT round-trip
    /// must preserve all of it, including the custom node types (the analogue of
    /// the opacity guarantee).
    fn full_schema_doc() -> Value {
        json!({
            "type": "doc",
            "content": [
                { "type": "heading", "attrs": { "level": 1 },
                  "content": [{ "type": "text", "text": "Heading one" }] },
                // ParagraphAnchor: the data-anchor-ms paragraph attr, plus marks.
                { "type": "paragraph", "attrs": { "data-anchor-ms": 12345 },
                  "content": [
                    { "type": "text", "text": "plain " },
                    { "type": "text", "marks": [{ "type": "bold" }], "text": "bold" },
                    { "type": "text", "text": " " },
                    { "type": "text", "marks": [{ "type": "italic" }], "text": "italic" },
                    { "type": "text",
                      "marks": [{ "type": "bold" }, { "type": "italic" }],
                      "text": "both" },
                    { "type": "text", "text": " " },
                    { "type": "text",
                      "marks": [{ "type": "code" }], "text": "inline code" },
                    { "type": "text", "text": " see " },
                    { "type": "text",
                      "marks": [{ "type": "link",
                                  "attrs": { "href": "https://example.test",
                                             "target": "_blank",
                                             "rel": "noopener" } }],
                      "text": "this link" }
                  ] },
                // Empty paragraph (no content key).
                { "type": "paragraph" },
                { "type": "bulletList", "content": [
                    { "type": "listItem", "content": [
                        { "type": "paragraph",
                          "content": [{ "type": "text", "text": "bullet one" }] } ] },
                    { "type": "listItem", "content": [
                        { "type": "paragraph",
                          "content": [{ "type": "text", "text": "bullet two" }] } ] }
                ] },
                { "type": "orderedList", "attrs": { "start": 1 }, "content": [
                    { "type": "listItem", "content": [
                        { "type": "paragraph",
                          "content": [{ "type": "text", "text": "first" }] } ] }
                ] },
                { "type": "blockquote", "content": [
                    { "type": "paragraph",
                      "content": [{ "type": "text", "text": "quoted" }] } ] },
                { "type": "codeBlock", "attrs": { "language": "rust" },
                  "content": [{ "type": "text", "text": "fn main() {}" }] },
                // TranscriptChip: an atom/leaf block node with rich custom attrs.
                { "type": "transcriptChip",
                  "attrs": { "segmentStartMs": 42000, "segmentEndMs": 45500,
                             "speakerId": "A", "text": "spoken words",
                             "nested": { "arr": [1, 2, 3], "flag": true, "none": null } } },
                // NoteImage: references a content-hash asset by bare filename.
                { "type": "noteImage",
                  "attrs": { "src": "deadbeef.png", "alt": "a diagram", "title": null } },
                // Table with a header row and a body row.
                { "type": "table", "content": [
                    { "type": "tableRow", "content": [
                        { "type": "tableHeader",
                          "attrs": { "colspan": 1, "rowspan": 1, "colwidth": null },
                          "content": [{ "type": "paragraph",
                            "content": [{ "type": "text", "text": "Name" }] }] },
                        { "type": "tableHeader",
                          "attrs": { "colspan": 1, "rowspan": 1, "colwidth": null },
                          "content": [{ "type": "paragraph",
                            "content": [{ "type": "text", "text": "Value" }] }] }
                    ] },
                    { "type": "tableRow", "content": [
                        { "type": "tableCell",
                          "attrs": { "colspan": 1, "rowspan": 1, "colwidth": null },
                          "content": [{ "type": "paragraph",
                            "content": [{ "type": "text", "text": "alpha" }] }] },
                        { "type": "tableCell",
                          "attrs": { "colspan": 2, "rowspan": 1, "colwidth": null },
                          "content": [{ "type": "paragraph",
                            "content": [{ "type": "text", "text": "beta" }] }] }
                    ] }
                ] }
            ]
        })
    }

    #[test]
    fn full_editor_schema_round_trips_through_yrs() {
        assert_round_trips(full_schema_doc());
    }

    #[test]
    fn transcript_chip_atom_round_trips() {
        assert_round_trips(json!({
            "type": "doc",
            "content": [
                { "type": "transcriptChip",
                  "attrs": { "segmentStartMs": 1000, "segmentEndMs": 2000,
                             "speakerId": "B", "text": "hi" } }
            ]
        }));
    }

    #[test]
    fn note_image_round_trips() {
        assert_round_trips(json!({
            "type": "doc",
            "content": [
                { "type": "noteImage",
                  "attrs": { "src": "abc123.jpg", "alt": "x", "width": 640 } }
            ]
        }));
    }

    #[test]
    fn paragraph_anchor_attr_round_trips() {
        assert_round_trips(json!({
            "type": "doc",
            "content": [
                { "type": "paragraph", "attrs": { "data-anchor-ms": 98765 },
                  "content": [{ "type": "text", "text": "anchored" }] }
            ]
        }));
    }

    #[test]
    fn table_round_trips() {
        assert_round_trips(json!({
            "type": "doc",
            "content": [
                { "type": "table", "content": [
                    { "type": "tableRow", "content": [
                        { "type": "tableHeader",
                          "attrs": { "colspan": 1, "rowspan": 1, "colwidth": null },
                          "content": [{ "type": "paragraph",
                            "content": [{ "type": "text", "text": "h" }] }] },
                        { "type": "tableCell",
                          "attrs": { "colspan": 1, "rowspan": 1, "colwidth": null },
                          "content": [{ "type": "paragraph",
                            "content": [{ "type": "text", "text": "c" }] }] }
                    ] }
                ] }
            ]
        }));
    }

    #[test]
    fn overlapping_and_adjacent_marks_round_trip() {
        // Adjacent runs with distinct marks must stay distinct; a run that
        // shares marks with its neighbour is one ProseMirror text node, exactly
        // as ProseMirror itself normalises.
        assert_round_trips(json!({
            "type": "doc",
            "content": [
                { "type": "paragraph", "content": [
                    { "type": "text", "marks": [{ "type": "bold" }], "text": "A" },
                    { "type": "text", "marks": [{ "type": "italic" }], "text": "B" },
                    { "type": "text",
                      "marks": [{ "type": "link",
                                  "attrs": { "href": "http://x" } }], "text": "C" }
                ] }
            ]
        }));
    }

    #[test]
    fn empty_document_round_trips() {
        // An empty doc (no content) yields an empty fragment and the canonical
        // `{ "type": "doc", "content": [] }` shape.
        let out = round_trip(&json!({ "type": "doc" }));
        assert_eq!(out, json!({ "type": "doc", "content": [] }));
    }

    #[test]
    fn decode_rejects_garbage_blob() {
        // A non-v2 blob must error, not panic.
        assert!(decode_ydoc(&[0xff, 0x00, 0x13, 0x37]).is_err());
    }
}
