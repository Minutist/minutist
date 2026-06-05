/**
 * About dialog (Phase 7, S6).
 *
 * Satisfies the acceptance item "About dialog lists the bundled-model SPDX
 * licenses + NOTICE/attribution". The bundled-model rows are derived from the
 * models store (`useModelsStore().models`), whose `ModelStatus` entries carry
 * `display_name` + `license` sourced verbatim from `resources/models.json`.
 * This keeps the list in lockstep with the manifest — no hand-kept mirror to
 * drift. The remaining attribution data (version, OSS components, NOTICE) is
 * static (`about-content.ts`); it is not in the manifest. It adds no Tauri
 * command and issues no raw `invoke`.
 *
 * Rendered in the Editorial Ink language: a centered paper sheet on the warm
 * desk field, reusing the same sheet treatment as the onboarding flow
 * (`--sheet`, `--shadow-sheet`, hairline edge), the single oxblood accent, and
 * `theme.css` tokens only — no hard-coded colour / type / radius.
 */
import { useEffect, useMemo, useRef } from "react";
import {
  APP_NAME,
  APP_VERSION,
  OSS_COMPONENTS,
  NOTICE_LINE,
  spdxDisplay,
} from "./about-content";
import { useModelsStore } from "../state/models";
import "./About.css";

/** Stable ordering for the bundled-model list: by kind, then id. */
const KIND_ORDER: Record<string, number> = { asr: 0, llm: 1, diarize: 2 };

export type AboutProps = {
  /** Called when the dialog should close (overlay click, Close button, Esc). */
  onClose: () => void;
};

export function About({ onClose }: AboutProps) {
  const closeRef = useRef<HTMLButtonElement>(null);
  const models = useModelsStore((s) => s.models);

  // Derive the bundled-model rows from the manifest-backed store. Sorted
  // stably (kind, then id) so the list is deterministic across renders.
  const bundledModels = useMemo(
    () =>
      [...models]
        .sort((a, b) => {
          const ka = KIND_ORDER[a.kind] ?? 99;
          const kb = KIND_ORDER[b.kind] ?? 99;
          return ka - kb || a.id.localeCompare(b.id);
        })
        .map((m) => ({
          id: m.id,
          displayName: m.display_name,
          spdx: spdxDisplay(m.license),
        })),
    [models],
  );

  // Move focus to the Close control on open and close on Escape — the minimum
  // dialog affordances; mirrors `role="dialog"` usage in the onboarding flow.
  useEffect(() => {
    closeRef.current?.focus();
    function onKeyDown(e: KeyboardEvent) {
      if (e.key === "Escape") {
        e.stopPropagation();
        onClose();
      }
    }
    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, [onClose]);

  return (
    <div
      className="about-overlay"
      // Clicking the desk field (outside the sheet) dismisses.
      onClick={onClose}
    >
      <div
        className="about ink-reveal"
        role="dialog"
        aria-modal="true"
        aria-labelledby="about-title"
        // Keep clicks inside the sheet from bubbling to the dismiss handler.
        onClick={(e) => e.stopPropagation()}
      >
        <header className="about__head">
          <h1 className="about__title" id="about-title">
            {APP_NAME}
          </h1>
          <p className="about__version">Version {APP_VERSION}</p>
          <p className="about__lede">
            A calm, local-first place for meeting notes. Transcription,
            speakers, and summary run on this machine using bundled open models.
          </p>
        </header>

        <section className="about__section" aria-labelledby="about-models">
          <h2 className="about__section-title" id="about-models">
            Bundled models
          </h2>
          <ul className="about__list">
            {bundledModels.map((model) => (
              <li key={model.id} className="about__row">
                <span className="about__row-name">{model.displayName}</span>
                <span className="about__row-license">{model.spdx}</span>
              </li>
            ))}
          </ul>
          <p className="about__notice">{NOTICE_LINE}</p>
        </section>

        <section className="about__section" aria-labelledby="about-oss">
          <h2 className="about__section-title" id="about-oss">
            Built with
          </h2>
          <ul className="about__list">
            {OSS_COMPONENTS.map((component) => (
              <li key={component.name} className="about__row">
                <span className="about__row-name">{component.name}</span>
                <span className="about__row-license">{component.spdx}</span>
              </li>
            ))}
          </ul>
        </section>

        <footer className="about__footer">
          <button
            ref={closeRef}
            type="button"
            className="about__btn about__btn--primary"
            onClick={onClose}
          >
            Close
          </button>
        </footer>
      </div>
    </div>
  );
}
