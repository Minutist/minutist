/**
 * Tests for the HTML clipboard serialiser (FR-17).
 *
 * Verifies the `text/html` payload shape: a self-contained UTF-8 document that
 * wraps the editor's body HTML, with internal `data-anchor-ms` attributes
 * stripped, plus the plain-text fallback.
 */
import { describe, it, expect } from "vitest";
import {
  buildClipboardPayload,
  stripAnchorAttributes,
  wrapHtmlDocument,
} from "../editor/clipboard";

describe("stripAnchorAttributes", () => {
  it("removes a double-quoted data-anchor-ms attribute", () => {
    expect(stripAnchorAttributes('<p data-anchor-ms="1500">hi</p>')).toBe(
      "<p>hi</p>",
    );
  });

  it("removes a single-quoted data-anchor-ms attribute", () => {
    expect(stripAnchorAttributes("<p data-anchor-ms='0'>x</p>")).toBe(
      "<p>x</p>",
    );
  });

  it("preserves other attributes on the same element", () => {
    expect(
      stripAnchorAttributes('<p class="note" data-anchor-ms="42">t</p>'),
    ).toBe('<p class="note">t</p>');
  });

  it("strips anchors from multiple paragraphs", () => {
    const input =
      '<p data-anchor-ms="1000">a</p><p data-anchor-ms="2000">b</p>';
    expect(stripAnchorAttributes(input)).toBe("<p>a</p><p>b</p>");
  });

  it("leaves HTML without anchors unchanged", () => {
    const input = "<h1>Title</h1><p><strong>bold</strong></p>";
    expect(stripAnchorAttributes(input)).toBe(input);
  });
});

describe("wrapHtmlDocument", () => {
  it("produces a full UTF-8 HTML document", () => {
    const out = wrapHtmlDocument("<p>body</p>");
    expect(out).toContain("<!DOCTYPE html>");
    expect(out).toContain('<meta charset="utf-8">');
    expect(out).toContain("<body><p>body</p></body>");
  });
});

describe("buildClipboardPayload", () => {
  it("produces a text/html and text/plain pair", () => {
    const payload = buildClipboardPayload("<p>hello</p>", "hello");
    expect(payload).toHaveProperty("text/html");
    expect(payload).toHaveProperty("text/plain");
    expect(payload["text/plain"]).toBe("hello");
  });

  it("wraps the body HTML and strips anchors from the html payload", () => {
    const html =
      '<h1>Meeting</h1><p data-anchor-ms="3200">a note</p>';
    const payload = buildClipboardPayload(html, "Meeting\na note");

    const out = payload["text/html"];
    expect(out).toContain("<!DOCTYPE html>");
    expect(out).toContain('<meta charset="utf-8">');
    expect(out).toContain("<h1>Meeting</h1>");
    expect(out).toContain("<p>a note</p>");
    // No internal anchors leak into the Word-bound payload.
    expect(out).not.toContain("data-anchor-ms");
    // Plain text untouched.
    expect(payload["text/plain"]).toBe("Meeting\na note");
  });

  it("retains rich formatting markup (bold, lists, links, tables)", () => {
    const html =
      '<p><strong>bold</strong> <em>italic</em></p>' +
      "<ul><li>one</li><li>two</li></ul>" +
      '<p><a href="https://example.com">link</a></p>' +
      "<table><tr><th>H</th></tr><tr><td>C</td></tr></table>";
    const out = buildClipboardPayload(html, "")["text/html"];

    expect(out).toContain("<strong>bold</strong>");
    expect(out).toContain("<em>italic</em>");
    expect(out).toContain("<ul><li>one</li><li>two</li></ul>");
    expect(out).toContain('href="https://example.com"');
    expect(out).toContain("<table>");
  });
});
