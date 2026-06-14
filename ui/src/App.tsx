import { useEffect } from "react";
import { useAppEventBridge } from "./shell/event-listener";
import { MainWindow } from "./shell/MainWindow";
import { Onboarding } from "./shell/Onboarding";
import { useRecordingStore } from "./state/recording";
import { useModelsStore } from "./state/models";
import { useReportProblemStore } from "./state/report-problem";
import {
  isOnboardingResolved,
  readOnboardingCompleted,
  readTheme,
} from "./state/onboarding-settings";

/**
 * Application root.
 *
 * Mounts the global Tauri event bridge at the top of the tree so it is never
 * accidentally unmounted by conditional rendering in child components, then
 * gates the UI on the persisted first-run flag (Phase 7):
 *
 *   - settings still loading  → render nothing app-specific (no flash);
 *   - `onboarding_completed`  false → first-run onboarding flow;
 *   - `onboarding_completed`  true  → the main app.
 *
 * Settings (the gate input) and the model list (needed by the onboarding model
 * step) are fetched here so the gate can be evaluated before `MainWindow`
 * mounts. The recording store is the single settings seam; the onboarding flow
 * persists completion through it.
 */
export function App() {
  useAppEventBridge();

  const settings = useRecordingStore((s) => s.settings);
  const refreshSettings = useRecordingStore((s) => s.refreshSettings);
  const refreshModels = useModelsStore((s) => s.refreshModels);
  const captureWebviewError = useReportProblemStore((s) => s.captureWebviewError);

  useEffect(() => {
    void refreshSettings();
    void refreshModels();
  }, [refreshSettings, refreshModels]);

  // Webview unhandled-error path (#0014): surface an uncaught error / promise
  // rejection in the main-window error pane (which offers "Report a problem"),
  // feeding the same report flow as a backend error. Mounted once at the root.
  useEffect(() => {
    const onError = (e: ErrorEvent) => {
      captureWebviewError(e.message || String(e.error));
    };
    const onRejection = (e: PromiseRejectionEvent) => {
      const reason = e.reason;
      captureWebviewError(
        reason instanceof Error ? reason.message : String(reason),
      );
    };
    window.addEventListener("error", onError);
    window.addEventListener("unhandledrejection", onRejection);
    return () => {
      window.removeEventListener("error", onError);
      window.removeEventListener("unhandledrejection", onRejection);
    };
  }, [captureWebviewError]);

  // Reflect the persisted colour-scheme preference onto the document root so
  // the `theme.css` `:root[data-theme="dark"]` overrides apply. `"dark"` /
  // `"light"` are explicit; `"system"` follows the OS `prefers-color-scheme`
  // (and tracks live changes to it). The light-first `:root` defaults apply
  // whenever the attribute is absent. UI-only; adds no command.
  const theme = readTheme(settings);
  useEffect(() => {
    const root = document.documentElement;
    const apply = (dark: boolean) => {
      if (dark) root.setAttribute("data-theme", "dark");
      else root.removeAttribute("data-theme");
    };

    if (theme === "dark") {
      apply(true);
      return;
    }
    if (theme === "light") {
      apply(false);
      return;
    }

    // "system": resolve from the OS preference and track live changes. Guard
    // `matchMedia` for environments that lack it (jsdom in unit tests) — there,
    // fall back to the light-first default.
    const media =
      typeof window !== "undefined" && typeof window.matchMedia === "function"
        ? window.matchMedia("(prefers-color-scheme: dark)")
        : null;
    if (!media) {
      apply(false);
      return;
    }
    apply(media.matches);
    const onChange = (e: MediaQueryListEvent) => apply(e.matches);
    media.addEventListener("change", onChange);
    return () => media.removeEventListener("change", onChange);
  }, [theme]);

  // Hold the UI neutral until settings resolve so a returning user is never
  // flashed the onboarding flow during the load round-trip.
  if (!isOnboardingResolved(settings)) {
    return null;
  }

  if (!readOnboardingCompleted(settings)) {
    return <Onboarding />;
  }

  return <MainWindow />;
}
