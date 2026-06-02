import { useEffect } from "react";
import { useAppEventBridge } from "./shell/event-listener";
import { MainWindow } from "./shell/MainWindow";
import { Onboarding } from "./shell/Onboarding";
import { useRecordingStore } from "./state/recording";
import { useModelsStore } from "./state/models";
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

  useEffect(() => {
    void refreshSettings();
    void refreshModels();
  }, [refreshSettings, refreshModels]);

  // Reflect the persisted colour-scheme preference onto the document root so
  // the `theme.css` `:root[data-theme="dark"]` overrides apply. `"system"`
  // (and the unresolved state) clears the attribute, falling back to the
  // light-first `:root` defaults. UI-only; adds no command.
  const theme = readTheme(settings);
  useEffect(() => {
    const root = document.documentElement;
    if (theme === "dark") {
      root.setAttribute("data-theme", "dark");
    } else {
      root.removeAttribute("data-theme");
    }
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
