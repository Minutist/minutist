/**
 * Static attribution data for the About dialog (Phase 7, S6 acceptance:
 * "About dialog lists the bundled-model SPDX licenses + NOTICE/attribution").
 *
 * Sourcing note — the bundled-model rows are NO LONGER mirrored here. The
 * generated `ModelStatus` binding (`ui/src/ipc/bindings.ts`) now carries a
 * `license` field (sourced verbatim from each `resources/models.json` entry),
 * so `About.tsx` derives the model list (id + display_name + license) directly
 * from the models store. `resources/models.json` is the single source of truth
 * for that list; there is no hand-kept per-model mirror on the UI side.
 *
 * The constants below are NOT in the manifest and remain static here:
 * the app version, the major OSS components the app is built on, and the
 * NOTICE line.
 */

/**
 * Application version. `ui/package.json` (and the Cargo workspace) is at
 * `0.0.0` pre-release; this constant mirrors it rather than fabricating a
 * version. `package.json` sits outside the TS `src` include and JSON-module
 * resolution is not enabled, so a hand-kept constant is the pragmatic source.
 */
export const APP_NAME = "meeting-app";
export const APP_VERSION = "0.0.0";

/**
 * Normalise a lowercase SPDX-ish licence identifier (as carried by the model
 * manifest, e.g. `apache-2.0`, `mit`) to its canonical SPDX display form
 * (`Apache-2.0`, `MIT`). Unknown values fall back to the raw string so a new
 * manifest licence still renders something rather than silently dropping.
 */
export function spdxDisplay(license: string): string {
  const known: Record<string, string> = {
    "apache-2.0": "Apache-2.0",
    mit: "MIT",
    openrail: "OpenRAIL",
  };
  return known[license.toLowerCase()] ?? license;
}

export type OssComponent = {
  name: string;
  /** SPDX identifier(s) for display. */
  spdx: string;
};

/** Major open-source components the application is built on. */
export const OSS_COMPONENTS: OssComponent[] = [
  { name: "Tauri", spdx: "Apache-2.0 OR MIT" },
  { name: "llama.cpp", spdx: "MIT" },
  { name: "sherpa-onnx", spdx: "Apache-2.0" },
  { name: "Tiptap", spdx: "MIT" },
  { name: "React", spdx: "MIT" },
];

/**
 * One-line NOTICE statement: the full MIT and Apache-2.0 license texts and
 * attribution NOTICE ship with the application.
 */
export const NOTICE_LINE =
  "The full MIT and Apache-2.0 license texts and the accompanying NOTICE / " +
  "attribution files ship with the application.";
