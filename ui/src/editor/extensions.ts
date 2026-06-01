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

export type BuildExtensionsOptions = {
  /** Supplies the recording state + pause-excluding clock to ParagraphAnchor. */
  clockSource: AnchorClockSource;
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
  ];
}
