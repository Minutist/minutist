/**
 * Attachments pane — meeting reference material.
 *
 * An always-available optional column (not mode-gated like summary/chat): the
 * user attaches documents (spreadsheets, slides, PDFs, web pages, emails) that
 * are converted to markdown in the background and fed to the summariser as
 * leading "Reference material". Usable before / during / after recording.
 *
 * Editorial Ink language: rows read as entries on a sheet of warm paper, one
 * oxblood accent on the primary "Add files" action. Each row carries the
 * filename, an extension badge, and a conversion-state affordance (a subtle
 * indeterminate bar while converting, a quiet check when ready, an inline
 * "couldn't convert" notice with an "Open anyway" fallback when it failed).
 *
 * All mutations route through `useAttachmentsStore` (which wraps the
 * `../ipc/attachments` seam); the component holds only local drag-over UI state.
 */
import { useEffect, useRef, useState } from "react";
import { useAttachmentsStore } from "../state/attachments";
import { MAX_ATTACHMENT_BYTES } from "../ipc/attachments";
import type { AttachmentEntry, MeetingId } from "../ipc/bindings";
import "./AttachmentsPane.css";

/**
 * The extensions the backend converter accepts (mirrors
 * `doc_convert::supported_exts()`). Used to build the file-picker `accept` list
 * and to reject an unsupported drop before the round-trip.
 *
 * Image extensions (`png`, `jpg`, `jpeg`, `tiff`) route to the VLM OCR path
 * (Gemma-4 vision) rather than a pure-Rust converter.
 */
const SUPPORTED_EXTS = [
  "txt",
  "md",
  "csv",
  "tsv",
  "json",
  "yaml",
  "yml",
  "xml",
  "log",
  "xlsx",
  "ods",
  "html",
  "htm",
  "eml",
  "pdf",
  "pptx",
  "docx",
  "png",
  "jpg",
  "jpeg",
  "tiff",
] as const;

const ACCEPT = SUPPORTED_EXTS.map((e) => `.${e}`).join(",");

/** Lower-cased, dot-less extension of a file's name, or `null` if none. */
function extOf(file: File): string | null {
  const dot = file.name.lastIndexOf(".");
  if (dot < 0 || dot === file.name.length - 1) return null;
  return file.name.slice(dot + 1).toLowerCase();
}

/** Human-readable file size (decimal units, matching desktop file managers). */
function formatBytes(bytes: number): string {
  if (bytes < 1000) return `${bytes} B`;
  const units = ["kB", "MB", "GB"];
  let value = bytes / 1000;
  let unit = 0;
  while (value >= 1000 && unit < units.length - 1) {
    value /= 1000;
    unit += 1;
  }
  return `${value < 10 ? value.toFixed(1) : Math.round(value)} ${units[unit]}`;
}

type AttachmentsPaneProps = {
  /** The meeting whose attachments are shown. */
  meetingId: MeetingId;
};

export function AttachmentsPane({ meetingId }: AttachmentsPaneProps) {
  const attachments = useAttachmentsStore((s) => s.attachments);
  const loading = useAttachmentsStore((s) => s.loading);
  const adding = useAttachmentsStore((s) => s.adding);
  const lastError = useAttachmentsStore((s) => s.lastError);
  const loadedMeetingId = useAttachmentsStore((s) => s.meetingId);
  const read = useAttachmentsStore((s) => s.read);
  const add = useAttachmentsStore((s) => s.add);
  const open = useAttachmentsStore((s) => s.open);
  const remove = useAttachmentsStore((s) => s.remove);

  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [dragOver, setDragOver] = useState(false);
  // A non-null message while a dropped/picked file is rejected for an
  // unsupported type — surfaced inline rather than swallowed.
  const [rejectNote, setRejectNote] = useState<string | null>(null);

  // Load the persisted manifest when the open meeting changes.
  useEffect(() => {
    void read(meetingId);
  }, [meetingId, read]);

  // Only render rows once they belong to THIS meeting (a fast switch may leave a
  // prior meeting's list in the store for one render before `read` commits).
  const rows = loadedMeetingId === meetingId ? attachments : [];

  function addFiles(files: FileList | File[]) {
    setRejectNote(null);
    const list = Array.from(files);
    const rejectedType: string[] = [];
    const rejectedSize: string[] = [];
    for (const file of list) {
      const ext = extOf(file);
      if (ext === null || !SUPPORTED_EXTS.includes(ext as never)) {
        rejectedType.push(file.name);
        continue;
      }
      if (file.size > MAX_ATTACHMENT_BYTES) {
        rejectedSize.push(file.name);
        continue;
      }
      void add(meetingId, file, ext);
    }
    const notes: string[] = [];
    if (rejectedType.length > 0) {
      notes.push(
        `Can't attach ${rejectedType.join(", ")} — unsupported file type.`,
      );
    }
    if (rejectedSize.length > 0) {
      notes.push(
        `Can't attach ${rejectedSize.join(", ")} — exceeds the 50 MiB limit.`,
      );
    }
    if (notes.length > 0) setRejectNote(notes.join(" "));
  }

  function onPick() {
    fileInputRef.current?.click();
  }

  function onInputChange(e: React.ChangeEvent<HTMLInputElement>) {
    if (e.target.files && e.target.files.length > 0) addFiles(e.target.files);
    // Reset so picking the same file again re-fires `change`.
    e.target.value = "";
  }

  function onDrop(e: React.DragEvent) {
    e.preventDefault();
    setDragOver(false);
    if (e.dataTransfer.files.length > 0) addFiles(e.dataTransfer.files);
  }

  function onDragOver(e: React.DragEvent) {
    // Required so the drop event fires; show the drop affordance.
    e.preventDefault();
    e.dataTransfer.dropEffect = "copy";
    if (!dragOver) setDragOver(true);
  }

  function onDragLeave(e: React.DragEvent) {
    // Only clear when leaving the pane itself, not a child element.
    if (e.currentTarget === e.target) setDragOver(false);
  }

  const hasRows = rows.length > 0;

  return (
    <section
      className={`attachments-pane ink-reveal${dragOver ? " attachments-pane--drag" : ""}`}
      aria-label="Meeting attachments"
      onDrop={onDrop}
      onDragOver={onDragOver}
      onDragLeave={onDragLeave}
    >
      <header className="attachments-pane__header">
        <h2 className="attachments-pane__heading">Attachments</h2>
        <div className="attachments-pane__actions">
          <button
            type="button"
            className="attachments-pane__action attachments-pane__action--primary"
            onClick={onPick}
            disabled={adding > 0}
          >
            {adding > 0 ? "Adding…" : "Add files"}
          </button>
        </div>
        <input
          ref={fileInputRef}
          type="file"
          className="attachments-pane__file-input"
          accept={ACCEPT}
          multiple
          aria-hidden="true"
          tabIndex={-1}
          onChange={onInputChange}
        />
      </header>

      <p className="attachments-pane__hint">
        Reference material — spreadsheets, slides, PDFs, web pages, emails,
        images — fed to the summary as context.
      </p>

      {lastError && (
        <p className="attachments-pane__error" role="alert">
          {lastError}
        </p>
      )}
      {rejectNote && (
        <p className="attachments-pane__error" role="alert">
          {rejectNote}
        </p>
      )}

      {loading && !hasRows ? (
        <p className="attachments-pane__status" role="status">
          <span className="attachments-pane__spinner" aria-hidden="true" />
          Loading attachments…
        </p>
      ) : hasRows ? (
        <ul className="attachments-pane__list">
          {rows.map((entry) => (
            <AttachmentRow
              key={entry.id}
              entry={entry}
              onOpen={() => void open(meetingId, entry)}
              onRemove={() => void remove(meetingId, entry.id)}
            />
          ))}
        </ul>
      ) : (
        <div className="attachments-pane__empty">
          <div
            className="attachments-pane__dropzone"
            role="button"
            tabIndex={0}
            onClick={onPick}
            onKeyDown={(e) => {
              if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                onPick();
              }
            }}
          >
            <span className="attachments-pane__drop-mark" aria-hidden="true">
              +
            </span>
            <p className="attachments-pane__drop-lead">
              Drop files here, or <span className="attachments-pane__drop-link">browse</span>
            </p>
            <p className="attachments-pane__drop-meta">
              {SUPPORTED_EXTS.join(" · ")}
            </p>
          </div>
        </div>
      )}
    </section>
  );
}

type AttachmentRowProps = {
  entry: AttachmentEntry;
  onOpen: () => void;
  onRemove: () => void;
};

function AttachmentRow({ entry, onOpen, onRemove }: AttachmentRowProps) {
  const conversion = entry.conversion;
  const failed = conversion.state === "failed";
  return (
    <li className="attachments-pane__row">
      <div className="attachments-pane__row-main">
        <span className="attachments-pane__ext" aria-hidden="true">
          {entry.ext}
        </span>
        <div className="attachments-pane__row-text">
          <span className="attachments-pane__name" title={entry.original_filename}>
            {entry.original_filename}
          </span>
          <span className="attachments-pane__meta">
            {formatBytes(entry.byte_len)}
            {" · "}
            <ConversionLabel entry={entry} />
          </span>
        </div>
        <div className="attachments-pane__row-actions">
          <button
            type="button"
            className="attachments-pane__row-btn"
            onClick={onOpen}
          >
            {failed ? "Open anyway" : "Open"}
          </button>
          <button
            type="button"
            className="attachments-pane__row-btn attachments-pane__row-btn--remove"
            aria-label={`Remove ${entry.original_filename}`}
            onClick={onRemove}
          >
            Remove
          </button>
        </div>
      </div>
      {entry.conversion.state === "pending" && (
        <div
          className="attachments-pane__progress"
          role="progressbar"
          aria-label="Converting attachment"
        >
          <span className="attachments-pane__progress-fill" />
        </div>
      )}
      {conversion.state === "failed" && (
        <p className="attachments-pane__row-error">
          Couldn&rsquo;t convert: {conversion.reason}
        </p>
      )}
    </li>
  );
}

/** The per-row conversion-state label (pending / ready / failed). */
function ConversionLabel({ entry }: { entry: AttachmentEntry }) {
  switch (entry.conversion.state) {
    case "pending":
      return (
        <span className="attachments-pane__state attachments-pane__state--pending">
          Converting…
        </span>
      );
    case "ready":
      return (
        <span className="attachments-pane__state attachments-pane__state--ready">
          <span aria-hidden="true">✓ </span>Ready
        </span>
      );
    case "failed":
      return (
        <span className="attachments-pane__state attachments-pane__state--failed">
          Conversion failed
        </span>
      );
  }
}
