/**
 * Tiptap extension set for the notes editor.
 *
 * Bundles the WYSIWYG building blocks (FR-15/16/20):
 *   - StarterKit — paragraph, headings, bold/italic/strike/code, lists,
 *     blockquote, code block, horizontal rule, link, history, etc. Each of
 *     these ships markdown-shortcut input rules that transform while typing
 *     (e.g. `# ` → heading, `- ` → bullet list, `> ` → blockquote, `**x**`
 *     → bold). StarterKit's own Link is disabled so the standalone
 *     `@tiptap/extension-link` (configured below) owns linking.
 *   - Typography — smart-quote / dash / ellipsis input rules.
 *   - Link — autolink + paste-to-link.
 *   - Table family — Table + TableRow + TableHeader + TableCell with resizing.
 *   - Markdown (tiptap-markdown) — round-trips the document to/from markdown so
 *     autosave can persist `notes.md` and copy can include a markdown view.
 *   - ParagraphAnchor — stamps `data-anchor-ms` on first keystroke while
 *     recording (see ./paragraph-anchor).
 *   - AnchorMarginalia — presentation-only decoration that renders each
 *     anchored paragraph's timestamp as quiet left-margin marginalia
 *     (see ./anchor-marginalia); adds no attrs and dispatches no transactions.
 *   - TranscriptChip — first-class block node for a dragged-in transcript
 *     segment (FR-24/25; see ./transcript-chip). Survives notes.json round-trip
 *     and exports to markdown as a fenced quotation.
 *   - NoteImage — pasted/dropped images stored as FILES in the meeting folder
 *     (see ./note-image). The node's `src` is a PORTABLE filename ref in
 *     notes.json; a node view converts it to a `meetingasset:` URL at display
 *     time, so the meeting folder stays copyable across machines.
 *   - NotesHoverBridge — presentation-only plugin reporting the hovered
 *     paragraph's `data-anchor-ms` (FR-22 read side; see ./hover-bridge); adds
 *     no attrs and dispatches no transactions, so it cannot affect anchoring.
 */
import StarterKit from "@tiptap/starter-kit";
import { Link } from "@tiptap/extension-link";
import { Typography } from "@tiptap/extension-typography";
import { Table } from "@tiptap/extension-table";
import { TableRow } from "@tiptap/extension-table-row";
import { TableHeader } from "@tiptap/extension-table-header";
import { TableCell } from "@tiptap/extension-table-cell";
import { Markdown } from "tiptap-markdown";
import type { Extensions } from "@tiptap/core";
import { ParagraphAnchor } from "./paragraph-anchor";
import type { AnchorClockSource } from "./paragraph-anchor";
import { AnchorMarginalia } from "./anchor-marginalia";
import { TranscriptChip } from "./transcript-chip";
import { NotesHoverBridge } from "./hover-bridge";
import type { HoverAnchorReporter } from "./hover-bridge";
import { NoteImage } from "./note-image";
import type { MeetingIdSource } from "./note-image";

export type BuildExtensionsOptions = {
  /** Supplies the recording state + pause-excluding clock to ParagraphAnchor. */
  clockSource: AnchorClockSource;
  /**
   * Optional FR-22 hover reporter — receives the hovered paragraph's
   * `data-anchor-ms` (or `null`). Defaults to a no-op when omitted (e.g. in
   * editor tests that don't exercise cross-reference).
   */
  onHoverAnchor?: HoverAnchorReporter;
  /**
   * Supplies the current meeting id to {@link NoteImage} so a stored portable
   * image ref resolves to its `meetingasset:` URL at render time. Defaults to a
   * no-op (`null`) when omitted; bare refs then render as-is (unresolved).
   */
  meetingIdSource?: MeetingIdSource;
};

/**
 * Construct the ordered extension list for the editor.
 *
 * StarterKit is configured with `link: false` so the standalone Link extension
 * (listed explicitly in the Phase 3 deliverable) is the single source of link
 * behaviour and avoids a duplicate-extension warning.
 */
export function buildEditorExtensions(
  options: BuildExtensionsOptions,
): Extensions {
  return [
    StarterKit.configure({
      // The standalone Link extension owns linking; disable the bundled one.
      link: false,
    }),
    Link.configure({
      openOnClick: false,
      autolink: true,
    }),
    Typography,
    Table.configure({ resizable: true }),
    TableRow,
    TableHeader,
    TableCell,
    Markdown.configure({
      html: true,
      transformCopiedText: false,
      transformPastedText: true,
    }),
    ParagraphAnchor.configure({ clockSource: options.clockSource }),
    // Presentation-only: renders anchor timestamps as left-margin marginalia.
    // Adds no node attrs and dispatches no transactions, so it cannot affect
    // ParagraphAnchor's stamping logic.
    AnchorMarginalia,
    // First-class dragged-in transcript segment (FR-24/25).
    TranscriptChip,
    // Pasted/dropped note images, stored as files + referenced portably.
    NoteImage.configure({
      meetingIdSource: options.meetingIdSource ?? (() => null),
    }),
    // FR-22 read side: reports the hovered paragraph's anchor. Presentation-only
    // (no doc mutation), so it cannot affect ParagraphAnchor's stamping.
    NotesHoverBridge.configure({
      onHoverAnchor: options.onHoverAnchor ?? (() => {}),
    }),
  ];
}
