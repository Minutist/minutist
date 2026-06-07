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

  it("surfaces the capture + processing controls when open", () => {
    render(<SettingsDrawer open onClose={() => {}} onAbout={() => {}} />);
    expect(screen.getByLabelText("Input device")).toBeInTheDocument();
    expect(screen.getByLabelText("Transcription language")).toBeInTheDocument();
    expect(screen.getByText("Diarize speakers on stop")).toBeInTheDocument();
    expect(screen.getByText("GPU acceleration")).toBeInTheDocument();
    expect(screen.getByText("Capture call / system audio")).toBeInTheDocument();
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
