/**
 * Attachment-reference Tiptap node (#0038) — a file dropped/pasted into the
 * notes editor becomes a normal meeting ATTACHMENT (the `add_attachment`
 * pipeline: manifest entry, attachments pane, markdown conversion fed to the
 * summariser), and this node is the inline REFERENCE left in the notes body.
 *
 * # Single storage, portable reference
 *
 * The attachment's bytes live once, under the meeting's `attachments/`
 * directory (owned by `persistence::attachments`). This node's attrs carry a
 * PORTABLE reference to that attachment — `attachmentId` (the manifest row's
 * id) plus `filename`/`originalFilename`/`ext`/`byteLen` (copied from the
 * `AttachmentEntry` returned by `add_attachment` at insert time, so the node
 * renders without re-reading the manifest) — never a URL and never a second
 * copy of the bytes. This is what `editor.getJSON()` serialises into
 * `notes.json`, so it round-trips the same portable way `NoteImage` does (see
 * `./note-image`).
 *
 * At DISPLAY time the node view converts `filename` into a working webview URL
 * via `convertFileSrc(<meetingId>/<filename>, ATTACHMENT_SCHEME)`, resolved
 * against the editor's current meeting id (supplied by {@link MeetingIdSource},
 * the same pattern as `NoteImage`'s). The `attachment:` protocol handler
 * (`app-main` + `ipc_bridge::resolve_attachment_asset`) serves the original
 * bytes from `{meetings_dir}/<meeting_id>/attachments/<filename>`.
 *
 * # Two node types coexist (back-compat)
 *
 * `NoteImage` (the pre-existing node for images pasted/dropped before this
 * issue landed) is UNCHANGED — existing meetings keep rendering their
 * note-asset images via `meetingasset:` exactly as before. This node only
 * covers NEW drops/pastes, which now always route through the attachment
 * pipeline (any file type, not just images) rather than the note-asset path.
 *
 * # Thumbnail vs file-card
 *
 * An image-extension attachment renders as an `<img>` thumbnail (CSS-capped
 * height; the served bytes are the original — no separate thumbnail generation
 * in this first cut). Any other extension renders as a file-type card (an
 * extension badge, filename, and human-readable size).
 *
 * # Expand on click
 *
 * Image → an in-app lightbox overlay showing the full image (dismissed by
 * clicking it or pressing Escape). Non-image → the pane's existing
 * open-in-OS-default-app affordance (`ipc/attachments.ts`'s
 * `openAttachmentById`), reused via {@link AttachmentRefOptions.onOpenAttachment}
 * rather than re-implemented here.
 */
import { Node, mergeAttributes } from "@tiptap/core";
import { convertFileSrc } from "@tauri-apps/api/core";
import { ATTACHMENT_SCHEME } from "../ipc/attachments";
import { formatBytes } from "../lib/format";

/** Supplies the editor's current meeting id (or `null` when none is open). */
export type MeetingIdSource = () => string | null;

export const ATTACHMENT_REF_NODE_NAME = "attachmentRef";

/** The portable attrs an `AttachmentRef` node carries. */
export type AttachmentRefAttrs = {
  attachmentId: string;
  /** The attachment's content-addressed on-disk name (`<hash>.<ext>`). */
  filename: string;
  /** The user-visible original filename, for the file-card / lightbox alt. */
  originalFilename: string;
  /** Lower-cased, dot-less extension — selects thumbnail vs file-card. */
  ext: string;
  byteLen: number;
};

export type AttachmentRefOptions = {
  /** Reads the current meeting id so the portable ref resolves to its bytes. */
  meetingIdSource: MeetingIdSource;
  /**
   * Expand a non-image attachment. Wired to the attachments store's
   * open-in-OS-default-app affordance; a no-op default so the extension is
   * safe to construct unwired (mirrors `NoteImage`'s default meeting-id
   * source).
   */
  onOpenAttachment: (attachmentId: string) => void;
};

declare module "@tiptap/core" {
  interface Commands<ReturnType> {
    attachmentRef: {
      /** Insert an attachment-reference node carrying the given portable ref. */
      insertAttachmentRef: (attrs: AttachmentRefAttrs) => ReturnType;
    };
  }
}

/** The extensions a browser/webview `<img>` can render natively as a thumbnail. */
const IMAGE_EXTS = new Set(["png", "jpg", "jpeg", "gif", "webp"]);

/** True when `ext` should render as an image thumbnail rather than a file card. */
export function isImageExt(ext: string): boolean {
  return IMAGE_EXTS.has(ext.toLowerCase());
}

/**
 * Show a full-size lightbox overlay for `src`. Dismissed by a click anywhere
 * on the overlay or the Escape key. A minimal in-app implementation — no
 * dependency beyond the DOM — since this is the only place in the editor that
 * needs a full-screen image preview.
 */
function openLightbox(src: string, alt: string): void {
  if (!src) return;
  const overlay = document.createElement("div");
  overlay.className = "attachment-ref__lightbox";
  overlay.setAttribute("role", "dialog");
  overlay.setAttribute("aria-modal", "true");

  const img = document.createElement("img");
  img.src = src;
  img.alt = alt;
  overlay.appendChild(img);

  const close = () => {
    overlay.remove();
    document.removeEventListener("keydown", onKey, true);
  };
  const onKey = (e: KeyboardEvent) => {
    if (e.key === "Escape") close();
  };
  overlay.addEventListener("click", close);
  document.addEventListener("keydown", onKey, true);
  document.body.appendChild(overlay);
}

/** Read a numeric DOM attribute, defaulting to 0 when absent/NaN. */
function numAttr(element: HTMLElement, name: string): number {
  const raw = element.getAttribute(name);
  if (raw === null) return 0;
  const parsed = Number.parseInt(raw, 10);
  return Number.isNaN(parsed) ? 0 : parsed;
}

export const AttachmentRef = Node.create<AttachmentRefOptions>({
  name: ATTACHMENT_REF_NODE_NAME,

  group: "block",
  // An atom: no editable inner content, so the reference cannot drift from the
  // attachment it points to (mirrors TranscriptChip).
  atom: true,
  selectable: true,
  draggable: true,

  addOptions() {
    return {
      meetingIdSource: () => null,
      onOpenAttachment: () => {},
    };
  },

  addAttributes() {
    return {
      attachmentId: {
        default: "",
        parseHTML: (element) => element.getAttribute("data-attachment-id") ?? "",
        renderHTML: (attributes) => ({
          "data-attachment-id": String(attributes.attachmentId ?? ""),
        }),
      },
      filename: {
        default: "",
        parseHTML: (element) => element.getAttribute("data-filename") ?? "",
        renderHTML: (attributes) => ({
          "data-filename": String(attributes.filename ?? ""),
        }),
      },
      originalFilename: {
        default: "",
        parseHTML: (element) =>
          element.getAttribute("data-original-filename") ?? "",
        renderHTML: (attributes) => ({
          "data-original-filename": String(attributes.originalFilename ?? ""),
        }),
      },
      ext: {
        default: "",
        parseHTML: (element) => element.getAttribute("data-ext") ?? "",
        renderHTML: (attributes) => ({
          "data-ext": String(attributes.ext ?? ""),
        }),
      },
      byteLen: {
        default: 0,
        parseHTML: (element) => numAttr(element, "data-byte-len"),
        renderHTML: (attributes) => ({
          "data-byte-len": String(attributes.byteLen ?? 0),
        }),
      },
    };
  },

  parseHTML() {
    return [{ tag: "div[data-attachment-ref]" }];
  },

  // A static (non-node-view) HTML rendering so the reference serialises to
  // HTML (clipboard / round-trip) even outside the live editor. The
  // interactive thumbnail/card is provided by the node view below.
  renderHTML({ HTMLAttributes, node }) {
    const attrs = node.attrs as AttachmentRefAttrs;
    return [
      "div",
      mergeAttributes(HTMLAttributes, {
        "data-attachment-ref": "",
        class: "attachment-ref",
      }),
      ["span", { class: "attachment-ref__name" }, attrs.originalFilename],
    ];
  },

  addNodeView() {
    const meetingIdSource = this.options.meetingIdSource;
    const onOpenAttachment = this.options.onOpenAttachment;
    return ({ node }) => {
      const attrs = node.attrs as AttachmentRefAttrs;
      const dom = document.createElement("div");
      dom.setAttribute("data-attachment-ref", "");
      dom.className = "attachment-ref";

      if (isImageExt(attrs.ext)) {
        const meetingId = meetingIdSource();
        const src =
          meetingId !== null
            ? convertFileSrc(`${meetingId}/${attrs.filename}`, ATTACHMENT_SCHEME)
            : "";
        const img = document.createElement("img");
        img.className = "attachment-ref__thumb";
        img.src = src;
        img.alt = attrs.originalFilename;
        img.title = `${attrs.originalFilename} — click to expand`;
        img.addEventListener("click", () => openLightbox(src, attrs.originalFilename));
        dom.appendChild(img);
      } else {
        const card = document.createElement("div");
        card.className = "attachment-ref__card";
        card.setAttribute("role", "button");
        card.tabIndex = 0;
        card.title = `Open ${attrs.originalFilename}`;

        const icon = document.createElement("span");
        icon.className = "attachment-ref__icon";
        icon.setAttribute("aria-hidden", "true");
        icon.textContent = attrs.ext ? attrs.ext.toUpperCase() : "FILE";

        const text = document.createElement("div");
        text.className = "attachment-ref__text";
        const name = document.createElement("span");
        name.className = "attachment-ref__name";
        name.textContent = attrs.originalFilename;
        const size = document.createElement("span");
        size.className = "attachment-ref__size";
        size.textContent = formatBytes(attrs.byteLen);
        text.append(name, size);

        card.append(icon, text);
        const activate = () => onOpenAttachment(attrs.attachmentId);
        card.addEventListener("click", activate);
        card.addEventListener("keydown", (event) => {
          if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            activate();
          }
        });
        dom.appendChild(card);
      }

      return { dom };
    };
  },

  addCommands() {
    return {
      insertAttachmentRef:
        (attrs: AttachmentRefAttrs) =>
        ({ commands }) =>
          commands.insertContent({
            type: ATTACHMENT_REF_NODE_NAME,
            attrs,
          }),
    };
  },

  // tiptap-markdown hook: export the reference as a markdown link so it
  // survives the `notes.md` export in a readable form (mirrors TranscriptChip's
  // fenced-quotation export). The link target is not a resolvable URL outside
  // the app — it documents the reference, matching the fenced-transcript
  // export's role (metadata capture, not a clickable external link).
  addStorage() {
    return {
      markdown: {
        serialize(
          state: { write: (text: string) => void; closeBlock: (node: unknown) => void },
          node: { attrs: Record<string, unknown> },
        ) {
          const attrs = node.attrs as AttachmentRefAttrs;
          state.write(`[${attrs.originalFilename}](attachment:${attrs.attachmentId})`);
          state.closeBlock(node);
        },
        parse: {
          // Round-trip into the editor is via `notes.json` (the opaque store),
          // not the markdown export, so no markdown-it parse rule is needed.
        },
      },
    };
  },
});
