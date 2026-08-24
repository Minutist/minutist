/**
 * Transcript-row context-menu entries (issue #0034).
 *
 * A pure function so the entry list (labels, disabled/checked state) is
 * unit-testable without mounting the virtualised pane. `selectedText` is
 * captured by the caller at the `contextmenu` event itself (before the menu
 * steals focus) — see `TranscriptPane.tsx`'s `onContextMenu` handler — rather
 * than re-read here, since opening the menu can otherwise race the DOM
 * selection.
 */
import type { ContextMenuEntry } from "../shell/ContextMenu";

export function buildTranscriptMenuEntries(opts: {
  selectedText: string | null;
  canPlay: boolean;
  isPlaying: boolean;
  onCopy: (text: string) => void;
  onJump: () => void;
  onPlayToggle: () => void;
}): ContextMenuEntry[] {
  const entries: ContextMenuEntry[] = [
    {
      label: "Copy",
      disabled: opts.selectedText === null,
      onSelect: () => opts.onCopy(opts.selectedText ?? ""),
    },
    { kind: "divider" },
    { label: "Jump to linked paragraph", onSelect: opts.onJump },
  ];
  if (opts.canPlay) {
    entries.push({
      label: opts.isPlaying ? "Stop playback" : "Play this segment’s audio",
      onSelect: opts.onPlayToggle,
    });
  }
  return entries;
}
