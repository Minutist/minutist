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
 *   - NoteImage — renders images already embedded as note-assets in EXISTING
 *     meetings (see ./note-image), back-compat only: no NEW drop/paste creates
 *     this node any more (#0038 moved that path onto AttachmentRef below). The
 *     node's `src` is a PORTABLE filename ref in notes.json; a node view
 *     converts it to a `meetingasset:` URL at display time.
 *   - AttachmentRef — a file dropped/pasted into notes (ANY type, #0038) is
 *     registered as a full meeting attachment (manifest, pane, summariser
 *     markdown) and this node is the inline reference left in the notes body
 *     (see ./attachment-ref): an image thumbnail, or a file-type card for
 *     everything else, both expanding on click.
 *   - NotesHoverBridge — presentation-only plugin reporting the hovered
 *     paragraph's `data-anchor-ms` (FR-22 read side; see ./hover-bridge); adds
 *     no attrs and dispatches no transactions, so it cannot affect anchoring.
 */
import StarterKit from "@tiptap/starter-kit";
import { Collaboration } from "@tiptap/extension-collaboration";
import type * as Y from "yjs";
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
import { AttachmentRef } from "./attachment-ref";
import { NOTES_FRAGMENT } from "./notes-collab";

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
   * Supplies the open/recording meeting's start wall-clock (epoch ms) to
   * {@link AnchorMarginalia} for the derived-time fallback on notes that predate
   * the stored per-note wall-clock. Defaults to `null` (elapsed shown for those).
   */
  startedAtMs?: () => number | null;
  /**
   * Supplies the current meeting id to {@link NoteImage} so a stored portable
   * image ref resolves to its `meetingasset:` URL at render time. Defaults to a
   * no-op (`null`) when omitted; bare refs then render as-is (unresolved).
   */
  meetingIdSource?: MeetingIdSource;
  /**
   * Expand a non-image `AttachmentRef` (opens it in the host OS default
   * application, via the attachments store's existing open affordance).
   * Defaults to a no-op when omitted (e.g. editor unit tests).
   */
  onOpenAttachment?: (attachmentId: string) => void;
  /**
   * The per-meeting Yjs document the editor binds to via
   * `@tiptap/extension-collaboration` (B6 WU7). When supplied, the document
   * flows through this `Y.Doc` rather than `getJSON()` → `save_notes`, and the
   * Collaboration extension owns undo/redo — so StarterKit's built-in history
   * is DISABLED to avoid a double/broken undo stack.
   *
   * When omitted (e.g. editor unit tests that don't exercise collaboration),
   * StarterKit's history is kept and content flows through the legacy JSON path.
   */
  collabDoc?: Y.Doc;
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
  const collab = options.collabDoc;
  return [
    StarterKit.configure({
      // The standalone Link extension owns linking; disable the bundled one.
      link: false,
      // When collaborating, the Collaboration extension provides undo/redo via
      // the Yjs UndoManager; StarterKit's own history would fight it (a broken,
      // doubled undo stack), so disable it. Without a collab doc, keep history.
      ...(collab ? { undoRedo: false as const } : {}),
    }),
    // Bind the editor to the per-meeting Y.Doc (B6 WU7). y-prosemirror binds at
    // the ProseMirror schema level, so the custom nodes round-trip unchanged.
    // The fragment name is pinned to match `persistence::ydoc`.
    ...(collab
      ? [Collaboration.configure({ document: collab, field: NOTES_FRAGMENT })]
      : []),
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
    // Presentation-only: renders anchor timestamps (local time-of-day) as
    // left-margin marginalia. Adds no node attrs and dispatches no transactions,
    // so it cannot affect ParagraphAnchor's stamping logic.
    AnchorMarginalia.configure({
      startedAtMs: options.startedAtMs ?? (() => null),
    }),
    // First-class dragged-in transcript segment (FR-24/25).
    TranscriptChip,
    // Back-compat rendering only for note-images already embedded in existing
    // meetings; no new drop/paste creates this node (see AttachmentRef below).
    NoteImage.configure({
      meetingIdSource: options.meetingIdSource ?? (() => null),
    }),
    // A file dropped/pasted into notes becomes an attachment; this is the
    // inline reference (thumbnail or file-card) left in the notes body.
    AttachmentRef.configure({
      meetingIdSource: options.meetingIdSource ?? (() => null),
      onOpenAttachment: options.onOpenAttachment ?? (() => {}),
    }),
    // FR-22 read side: reports the hovered paragraph's anchor. Presentation-only
    // (no doc mutation), so it cannot affect ParagraphAnchor's stamping.
    NotesHoverBridge.configure({
      onHoverAnchor: options.onHoverAnchor ?? (() => {}),
    }),
  ];
}
