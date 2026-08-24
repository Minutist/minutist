/**
 * Transcript-row context-menu entry tests (issue #0034).
 */
import { describe, it, expect, vi } from "vitest";
import { buildTranscriptMenuEntries } from "../transcript/transcript-context-menu";

function findItem(entries: ReturnType<typeof buildTranscriptMenuEntries>, label: string) {
  const entry = entries.find((e) => "label" in e && e.label === label);
  if (!entry || entry.kind === "submenu" || entry.kind === "divider") {
    throw new Error(`no plain item entry "${label}"`);
  }
  return entry;
}

describe("buildTranscriptMenuEntries", () => {
  it("Copy is disabled with no selected text, and calls onCopy with it when present", () => {
    const onCopy = vi.fn();
    const disabled = buildTranscriptMenuEntries({
      selectedText: null,
      canPlay: false,
      isPlaying: false,
      onCopy,
      onJump: vi.fn(),
      onPlayToggle: vi.fn(),
    });
    expect(findItem(disabled, "Copy").disabled).toBe(true);

    const enabled = buildTranscriptMenuEntries({
      selectedText: "hello",
      canPlay: false,
      isPlaying: false,
      onCopy,
      onJump: vi.fn(),
      onPlayToggle: vi.fn(),
    });
    const copy = findItem(enabled, "Copy");
    expect(copy.disabled).toBe(false);
    copy.onSelect();
    expect(onCopy).toHaveBeenCalledWith("hello");
  });

  it("always offers Jump to linked paragraph", () => {
    const onJump = vi.fn();
    const entries = buildTranscriptMenuEntries({
      selectedText: null,
      canPlay: false,
      isPlaying: false,
      onCopy: vi.fn(),
      onJump,
      onPlayToggle: vi.fn(),
    });
    findItem(entries, "Jump to linked paragraph").onSelect();
    expect(onJump).toHaveBeenCalledOnce();
  });

  it("omits the play entry entirely when canPlay is false", () => {
    const entries = buildTranscriptMenuEntries({
      selectedText: null,
      canPlay: false,
      isPlaying: false,
      onCopy: vi.fn(),
      onJump: vi.fn(),
      onPlayToggle: vi.fn(),
    });
    expect(
      entries.some(
        (e) =>
          "label" in e &&
          (e.label === "Play this segment’s audio" || e.label === "Stop playback"),
      ),
    ).toBe(false);
  });

  it("labels the play entry by isPlaying, and it toggles", () => {
    const onPlayToggle = vi.fn();
    const idle = buildTranscriptMenuEntries({
      selectedText: null,
      canPlay: true,
      isPlaying: false,
      onCopy: vi.fn(),
      onJump: vi.fn(),
      onPlayToggle,
    });
    findItem(idle, "Play this segment’s audio").onSelect();
    expect(onPlayToggle).toHaveBeenCalledOnce();

    const playing = buildTranscriptMenuEntries({
      selectedText: null,
      canPlay: true,
      isPlaying: true,
      onCopy: vi.fn(),
      onJump: vi.fn(),
      onPlayToggle,
    });
    expect(findItem(playing, "Stop playback")).toBeTruthy();
  });
});
