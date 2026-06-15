/**
 * Settings drawer.
 *
 * A right-anchored slide-in panel that holds the rarely-changed capture /
 * inference configuration that previously lived inline in the top bar: input
 * device, transcription-language hint, diarize-on-stop, GPU acceleration, and
 * system-audio capture. Pulling these out of the masthead keeps the chrome a
 * quiet editorial bar (no overflow) while leaving each setting one click away.
 *
 * Presentational only: it adds no Tauri command. Each control routes through
 * the existing `useRecordingStore` seams (the same ones the inline controls
 * used), so the persistence behaviour — and its tests — are unchanged.
 *
 * Dialog affordances mirror {@link About}: focus moves to the close control on
 * open, Escape and a scrim click dismiss. Rendered in the Editorial Ink
 * language using `theme.css` tokens only.
 */
import { useEffect, useRef, lazy, Suspense, Component } from "react";
import type { ReactNode } from "react";
import { useRecordingStore } from "../state/recording";
import { readDiarizationEnabled } from "../state/diarization-settings";
import { readGpuAcceleration } from "../state/gpu-acceleration-settings";
import { readPreloadSummariser } from "../state/preload-summariser-settings";
import { readCaptureSystemAudio } from "../state/system-audio-settings";
import { readNotesPaperRules } from "../state/notes-paper-settings";
import { readTheme } from "../state/onboarding-settings";
import type { GpuAcceleration, Theme } from "../ipc/bindings";
import { DevicePicker } from "./DevicePicker";
import { LanguagePicker } from "./LanguagePicker";
import { OutputLanguagePicker } from "./OutputLanguagePicker";
import "./SettingsDrawer.css";

// Error boundary for the lazy MCP pane. A chunk-load failure is silently
// suppressed: the pane is omitted rather than unmounting the whole React tree.
class McpPaneErrorBoundary extends Component<
  { children: ReactNode },
  { failed: boolean }
> {
  constructor(props: { children: ReactNode }) {
    super(props);
    this.state = { failed: false };
  }
  static getDerivedStateFromError() {
    return { failed: true };
  }
  override render() {
    if (this.state.failed) return null;
    return this.props.children;
  }
}

// McpSettingsPane is present only in the connected build.  `import.meta.env.VITE_CONNECTED`
// is replaced with a string literal by Vite's `define` at bundle time (see
// vite.config.ts), so the false branch and its `import()` call are
// dead-code-eliminated, dropping McpSettingsPane, mcp-settings.ts, and
// mcp-server-info.ts from the free bundle.
//
// `npm run dev` and vitest both default to VITE_CONNECTED="1" (see
// vite.config.ts), so the pane renders and tests pass unchanged.
//
// Verification: VITE_CONNECTED= npm run build && grep -r "Enable MCP server" dist/
// → no matches expected in the free bundle.
const LazyMcpSettingsPane =
  import.meta.env.VITE_CONNECTED === "1"
    ? lazy(() =>
        import("./McpSettingsPane").then((m) => ({ default: m.McpSettingsPane })),
      )
    : null;

// ConnectionSettingsPane (the connector / relay tunnel) is likewise present only
// in the connected build, gated by the same VITE_CONNECTED literal so the false
// branch (and its `import()`) is dead-code-eliminated, dropping the pane and its
// state modules (connector-settings.ts, tunnel-status.ts) from the free bundle.
const LazyConnectionSettingsPane =
  import.meta.env.VITE_CONNECTED === "1"
    ? lazy(() =>
        import("./ConnectionSettingsPane").then((m) => ({
          default: m.ConnectionSettingsPane,
        })),
      )
    : null;

export type SettingsDrawerProps = {
  /** Whether the drawer is shown. */
  open: boolean;
  /** Called when the drawer should close (scrim click, Close button, Esc). */
  onClose: () => void;
  /** Open the About dialog (the affordance lives in this drawer's footer). */
  onAbout: () => void;
};

export function SettingsDrawer({ open, onClose, onAbout }: SettingsDrawerProps) {
  const closeRef = useRef<HTMLButtonElement>(null);
  const settings = useRecordingStore((s) => s.settings);
  const setDiarizationEnabled = useRecordingStore(
    (s) => s.setDiarizationEnabled,
  );
  const setGpuAcceleration = useRecordingStore((s) => s.setGpuAcceleration);
  const setPreloadSummariser = useRecordingStore(
    (s) => s.setPreloadSummariser,
  );
  const setCaptureSystemAudio = useRecordingStore(
    (s) => s.setCaptureSystemAudio,
  );
  const setTheme = useRecordingStore((s) => s.setTheme);
  const setNotesPaperRules = useRecordingStore((s) => s.setNotesPaperRules);

  const diarizationEnabled = readDiarizationEnabled(settings);
  const gpuAcceleration = readGpuAcceleration(settings);
  const preloadSummariser = readPreloadSummariser(settings);
  const captureSystemAudio = readCaptureSystemAudio(settings);
  const theme = readTheme(settings);
  const notesPaperRules = readNotesPaperRules(settings);

  // Focus the close control on open, and close on Escape — the minimum dialog
  // affordances; mirrors the About dialog.
  useEffect(() => {
    if (!open) return;
    closeRef.current?.focus();
    function onKeyDown(e: KeyboardEvent) {
      if (e.key === "Escape") {
        e.stopPropagation();
        onClose();
      }
    }
    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, [open, onClose]);

  if (!open) return null;

  return (
    <div className="settings-drawer-overlay" onClick={onClose}>
      <aside
        className="settings-drawer"
        role="dialog"
        aria-modal="true"
        aria-labelledby="settings-title"
        // Keep clicks inside the drawer from bubbling to the dismiss handler.
        onClick={(e) => e.stopPropagation()}
      >
        <header className="settings-drawer__head">
          <h2 className="settings-drawer__title" id="settings-title">
            Settings
          </h2>
          <button
            ref={closeRef}
            type="button"
            className="settings-drawer__close"
            aria-label="Close settings"
            onClick={onClose}
          >
            Done
          </button>
        </header>

        <section className="settings-drawer__group" aria-label="Appearance">
          <h3 className="settings-drawer__group-title">Appearance</h3>
          <div className="settings-drawer__field">
            <label htmlFor="settings-theme">Colour theme</label>
            <select
              id="settings-theme"
              value={theme}
              disabled={settings === null}
              onChange={(e) => void setTheme(e.target.value as Theme)}
            >
              <option value="system">Match system</option>
              <option value="light">Light</option>
              <option value="dark">Dark</option>
            </select>
          </div>
          <label
            className="settings-drawer__toggle"
            title="Show faint horizontal writing-paper rules behind the notes. The oxblood margin rule that separates the timestamp gutter from the text is always shown."
          >
            <input
              type="checkbox"
              checked={notesPaperRules}
              disabled={settings === null}
              onChange={(e) => void setNotesPaperRules(e.target.checked)}
            />
            <span>Ruled writing paper</span>
          </label>
        </section>

        <section className="settings-drawer__group" aria-label="Capture">
          <h3 className="settings-drawer__group-title">Capture</h3>
          <DevicePicker />
          <LanguagePicker />
          <label
            className="settings-drawer__toggle"
            title="Capture the call / system audio and mix it with your microphone so all participants are transcribed. Turn this off if your microphone also picks up the call from your speakers (echo)."
          >
            <input
              type="checkbox"
              checked={captureSystemAudio}
              disabled={settings === null}
              onChange={(e) => void setCaptureSystemAudio(e.target.checked)}
            />
            <span>Capture call / system audio</span>
          </label>
        </section>

        <section className="settings-drawer__group" aria-label="Processing">
          <h3 className="settings-drawer__group-title">Processing</h3>
          <label
            className="settings-drawer__toggle"
            title="When a recording stops, identify who spoke each line and label them Speaker A / B / C."
          >
            <input
              type="checkbox"
              checked={diarizationEnabled}
              disabled={settings === null}
              onChange={(e) => void setDiarizationEnabled(e.target.checked)}
            />
            <span>Identify speakers</span>
          </label>
          {/* TODO: a live "auto-result" indicator (what Auto actually resolved
              on this machine) is deferred — it needs a new IPC payload and is
              out of scope here. The app-startup GPU-probe log
              (`IpcState::log_gpu_probe`) is the diagnostic for now. */}
          <div className="settings-drawer__field">
            <label htmlFor="settings-gpu-acceleration">GPU acceleration</label>
            <select
              id="settings-gpu-acceleration"
              value={gpuAcceleration}
              disabled={settings === null}
              onChange={(e) =>
                void setGpuAcceleration(e.target.value as GpuAcceleration)
              }
            >
              <option value="auto">Auto</option>
              <option value="on">On</option>
              <option value="off">Off</option>
            </select>
            <p className="settings-drawer__hint">
              Auto detects your GPU memory and runs models on GPU when they fit,
              otherwise CPU.
            </p>
          </div>
          <label
            className="settings-drawer__toggle"
            title="Load the summary & chat model when the app starts (if it is already downloaded) and keep it ready, so the first Summarise or chat is instant. Turn off to load it on first use and keep idle memory lower."
          >
            <input
              type="checkbox"
              checked={preloadSummariser}
              disabled={settings === null}
              onChange={(e) => void setPreloadSummariser(e.target.checked)}
            />
            <span>Keep the summary &amp; chat model loaded</span>
          </label>
          <OutputLanguagePicker />
        </section>

        {LazyConnectionSettingsPane && (
          <McpPaneErrorBoundary>
            <Suspense fallback={null}>
              <LazyConnectionSettingsPane />
            </Suspense>
          </McpPaneErrorBoundary>
        )}

        {LazyMcpSettingsPane && (
          <McpPaneErrorBoundary>
            <Suspense fallback={null}>
              <LazyMcpSettingsPane />
            </Suspense>
          </McpPaneErrorBoundary>
        )}

        <footer className="settings-drawer__footer">
          <button
            type="button"
            className="settings-drawer__about"
            aria-haspopup="dialog"
            onClick={onAbout}
          >
            About Minutist
          </button>
        </footer>
      </aside>
    </div>
  );
}
