/**
 * First-run onboarding flow (Phase 7).
 *
 * A three-step, full-window flow rendered in the Editorial Ink language (warm
 * paper sheet, Fraunces display, single oxblood accent, hairline rules — all
 * via `theme.css` tokens; no hard-coded colour/type). It gates the main app:
 * the shell renders this when `settings.onboarding_completed` is false and
 * reveals `MainWindow` once the final step persists it true.
 *
 *   1. Welcome      — product intro.
 *   2. Model        — REUSES `ModelDownloadStatus` so the user downloads the ASR
 *                     model (or skips to do it later). Download is not
 *                     reimplemented here.
 *   3. Quick settings — a minimal tour: theme preference + the diarization
 *                     toggle, reusing the existing settings read/write seam.
 *                     Finish persists `onboarding_completed = true`.
 *
 * Persistence rides the existing settings seam: the recording store owns the
 * `settings` snapshot and rounds every write through `commands.updateSettings`
 * (the same path `diarization-settings` uses). This view never adds a Tauri
 * command nor hand-rolls a raw `invoke` (rule A9).
 */
import { useRecordingStore } from "../state/recording";
import { useOnboardingStore } from "../state/onboarding";
import { readDiarizationEnabled } from "../state/diarization-settings";
import { readTheme } from "../state/onboarding-settings";
import {
  ModelDownloadCard,
  PARAKEET_MODEL_ID,
  ASR_MODEL_ID,
  LLM_MODEL_ID,
} from "./ModelDownloadStatus";
import type { Theme } from "../ipc/bindings";
import "./Onboarding.css";

const THEME_OPTIONS: { value: Theme; label: string }[] = [
  { value: "system", label: "Follow system" },
  { value: "light", label: "Light" },
  { value: "dark", label: "Dark" },
];

function WelcomeStep() {
  return (
    <div className="onboarding__body">
      <h1 className="onboarding__title">Welcome to Minutist</h1>
      <p className="onboarding__lede">
        A calm, local-first place for meeting notes. Write as you would on
        paper; the transcript, speakers, and summary gather quietly alongside —
        all on this machine, nothing sent away.
      </p>
      <p className="onboarding__prose">
        Three short steps and you are set up. Let&rsquo;s get the speech model in
        place, then choose a couple of preferences.
      </p>
    </div>
  );
}

function ModelStep() {
  return (
    <div className="onboarding__body">
      <h1 className="onboarding__title">Models</h1>
      <p className="onboarding__lede">
        Local models do the work: a speech model for the transcript and a
        language model for summaries. The English &amp; European speech model
        provides word-level timings; a separate multilingual model covers
        Chinese, Japanese, Korean, Arabic, and more. Download what you need
        now, or fetch it later from the main window.
      </p>
      {/* Per-model cards — each self-tracks progress / ready / retry. */}
      <div className="onboarding__models">
        <ModelDownloadCard
          modelId={PARAKEET_MODEL_ID}
          fallbackName="Speech model — English & European"
        />
        <ModelDownloadCard
          modelId={ASR_MODEL_ID}
          fallbackName="Speech model — other languages"
        />
        <ModelDownloadCard
          modelId={LLM_MODEL_ID}
          fallbackName="Summarisation model"
        />
      </div>
    </div>
  );
}

function SettingsStep() {
  const settings = useRecordingStore((s) => s.settings);
  const setTheme = useRecordingStore((s) => s.setTheme);
  const setDiarizationEnabled = useRecordingStore(
    (s) => s.setDiarizationEnabled,
  );

  const theme = readTheme(settings);
  const diarizationEnabled = readDiarizationEnabled(settings);
  const settingsPending = settings === null;

  return (
    <div className="onboarding__body">
      <h1 className="onboarding__title">A couple of preferences</h1>
      <p className="onboarding__lede">
        You can change these any time from the main window.
      </p>

      <fieldset className="onboarding__field">
        <legend className="onboarding__field-label">Appearance</legend>
        <div className="onboarding__theme-options">
          {THEME_OPTIONS.map((opt) => (
            <label key={opt.value} className="onboarding__theme-option">
              <input
                type="radio"
                name="onboarding-theme"
                value={opt.value}
                checked={theme === opt.value}
                disabled={settingsPending}
                onChange={() => void setTheme(opt.value)}
              />
              <span>{opt.label}</span>
            </label>
          ))}
        </div>
      </fieldset>

      <div className="onboarding__field">
        <label className="onboarding__toggle">
          <input
            type="checkbox"
            checked={diarizationEnabled}
            disabled={settingsPending}
            onChange={(e) => void setDiarizationEnabled(e.target.checked)}
          />
          <span>
            Identify speakers after recording
            <span className="onboarding__hint">
              Identifies who spoke each line when a recording stops. On by
              default.
            </span>
          </span>
        </label>
      </div>
    </div>
  );
}

export function Onboarding() {
  const step = useOnboardingStore((s) => s.step);
  const next = useOnboardingStore((s) => s.next);
  const back = useOnboardingStore((s) => s.back);
  const settings = useRecordingStore((s) => s.settings);
  const setOnboardingCompleted = useRecordingStore(
    (s) => s.setOnboardingCompleted,
  );

  const isFirst = step === "welcome";
  const isLast = step === "settings";
  // Finish is disabled until settings have loaded, so the persist round-trip
  // never clobbers settings with a partial object (mirrors the main-window
  // diarize-toggle guard).
  const finishDisabled = settings === null;

  function onPrimary() {
    if (isLast) {
      void setOnboardingCompleted(true);
    } else {
      next();
    }
  }

  const stepNumber = step === "welcome" ? 1 : step === "model" ? 2 : 3;

  return (
    <div className="onboarding ink-reveal" role="dialog" aria-label="Welcome">
      <div className="onboarding__sheet">
        {step === "welcome" && <WelcomeStep />}
        {step === "model" && <ModelStep />}
        {step === "settings" && <SettingsStep />}

        <footer className="onboarding__footer">
          <ol
            className="onboarding__steps"
            aria-label={`Step ${stepNumber} of 3`}
          >
            {[1, 2, 3].map((n) => (
              <li
                key={n}
                className={
                  n === stepNumber
                    ? "onboarding__step onboarding__step--active"
                    : "onboarding__step"
                }
                aria-hidden="true"
              />
            ))}
          </ol>

          <div className="onboarding__actions">
            {!isFirst && (
              <button
                type="button"
                className="onboarding__btn onboarding__btn--quiet"
                onClick={back}
              >
                Back
              </button>
            )}
            <button
              type="button"
              className="onboarding__btn onboarding__btn--primary"
              onClick={onPrimary}
              disabled={isLast && finishDisabled}
            >
              {isLast ? "Finish" : "Continue"}
            </button>
          </div>
        </footer>
      </div>
    </div>
  );
}
