/**
 * Settings-drawer tests.
 *
 * The capture / processing controls moved out of the top bar into a drawer.
 * These cover the drawer's own behaviour: open/closed gating, the controls it
 * surfaces, and the dismissal affordances (Done, Escape, scrim). The per-setting
 * persistence is unchanged and stays covered by the store-seam tests
 * (Diarization / GpuAcceleration / SystemAudio / TranscriptionLanguage /
 * DevicePersistence) — the drawer routes through those same seams.
 */
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, act } from "@testing-library/react";
import { SettingsDrawer } from "../shell/SettingsDrawer";
import { useRecordingStore } from "../state/recording";
import type { Settings } from "../ipc/bindings";

const BASE_SETTINGS: Settings = {
  input_device_id: null,
  theme: "system",
  data_directory: null,
  start_hidden: false,
};

function seedStore() {
  act(() => {
    useRecordingStore.setState({
      settings: BASE_SETTINGS,
      devices: [{ id: "mic-1", name: "Built-in Mic", is_default: true }],
      selectedDeviceId: null,
    });
  });
}

describe("SettingsDrawer", () => {
  beforeEach(seedStore);

  it("renders nothing when closed", () => {
    const { container } = render(
      <SettingsDrawer open={false} onClose={() => {}} onAbout={() => {}} />,
    );
    expect(container).toBeEmptyDOMElement();
  });

  it("surfaces the appearance + capture + processing controls when open", () => {
    render(<SettingsDrawer open onClose={() => {}} onAbout={() => {}} />);
    // Appearance: colour theme + writing-paper rules.
    expect(screen.getByLabelText("Colour theme")).toBeInTheDocument();
    expect(screen.getByText("Ruled writing paper")).toBeInTheDocument();
    // Capture + processing.
    expect(screen.getByLabelText("Input device")).toBeInTheDocument();
    expect(screen.getByLabelText("Transcription language")).toBeInTheDocument();
    // Live-test UX T5: the diarization toggle is relabelled "Identify speakers"
    // (the wire name `diarization_enabled` is unchanged).
    expect(screen.getByText("Identify speakers")).toBeInTheDocument();
    expect(screen.getByText("GPU acceleration")).toBeInTheDocument();
    expect(screen.getByText("Capture call / system audio")).toBeInTheDocument();
    // Output language dropdown (Processing section).
    expect(screen.getByLabelText("Output language")).toBeInTheDocument();
  });

  it("reflects the persisted appearance defaults (system theme, ruled paper on)", () => {
    render(<SettingsDrawer open onClose={() => {}} onAbout={() => {}} />);
    // BASE_SETTINGS omits both fields → the schema defaults apply: theme falls
    // back to "system" and the writing-paper rules read as on.
    const theme = screen.getByLabelText("Colour theme") as HTMLSelectElement;
    expect(theme.value).toBe("system");
    const ruled = screen
      .getByText("Ruled writing paper")
      .closest("label")!
      .querySelector("input") as HTMLInputElement;
    expect(ruled.checked).toBe(true);
  });

  it("calls onClose on the Done button and on Escape", () => {
    const onClose = vi.fn();
    render(<SettingsDrawer open onClose={onClose} onAbout={() => {}} />);
    fireEvent.click(screen.getByRole("button", { name: "Close settings" }));
    expect(onClose).toHaveBeenCalledTimes(1);
    fireEvent.keyDown(document, { key: "Escape" });
    expect(onClose).toHaveBeenCalledTimes(2);
  });

  it("dismisses on a scrim click but not on a click inside the panel", () => {
    const onClose = vi.fn();
    const { container } = render(<SettingsDrawer open onClose={onClose} onAbout={() => {}} />);
    // Click inside the dialog panel — stopPropagation keeps it open.
    fireEvent.click(screen.getByRole("dialog"));
    expect(onClose).not.toHaveBeenCalled();
    // Click the scrim (the overlay root) — dismisses.
    fireEvent.click(container.firstChild as Element);
    expect(onClose).toHaveBeenCalledTimes(1);
  });
});
