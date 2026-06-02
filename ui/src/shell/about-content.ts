/**
 * Static attribution data for the About dialog (Phase 7, S6 acceptance:
 * "About dialog lists the bundled-model SPDX licenses + NOTICE/attribution").
 *
 * Sourcing note — bundled-model licenses are STATIC here, not read from the
 * models store. The generated `ModelStatus` binding (`ui/src/ipc/bindings.ts`)
 * carries only `{ id, kind, display_name, status }`; it has no `license`
 * field. The license metadata lives solely in `resources/models.json`
 * (each entry's `license`), which the webview never receives over IPC. So the
 * single available source of truth for licenses on the UI side is a static
 * mirror of that file — kept in step with it by hand. The `id`/`display_name`
 * values below match `resources/models.json` exactly so a drift is easy to
 * spot in review.
 */

/**
 * Application version. `ui/package.json` (and the Cargo workspace) is at
 * `0.0.0` pre-release; this constant mirrors it rather than fabricating a
 * version. `package.json` sits outside the TS `src` include and JSON-module
 * resolution is not enabled, so a hand-kept constant is the pragmatic source.
 */
export const APP_NAME = "meeting-app";
export const APP_VERSION = "0.0.0";

export type BundledModel = {
  /** Matches `resources/models.json` `id`. */
  id: string;
  /** Matches `resources/models.json` `display_name`. */
  displayName: string;
  /** SPDX identifier (uppercased for display). */
  spdx: string;
};

/**
 * Bundled / on-demand-downloaded models, mirroring `resources/models.json`.
 * SPDX identifiers are the canonical forms of that file's `license` values
 * (`apache-2.0` → `Apache-2.0`, `mit` → `MIT`).
 */
export const BUNDLED_MODELS: BundledModel[] = [
  {
    id: "qwen3-asr-0.6b-q8_0",
    displayName: "Qwen3-ASR 0.6B (Q8_0)",
    spdx: "Apache-2.0",
  },
  {
    id: "gemma-4-e4b-it-q4_k_m",
    displayName: "Gemma 4 E4B Instruct (Q4_K_M)",
    spdx: "Apache-2.0",
  },
  {
    id: "pyannote-segmentation-3-0",
    displayName: "pyannote segmentation 3.0",
    spdx: "MIT",
  },
  {
    id: "3dspeaker-campplus-zh-cn-16k-common",
    displayName: "3D-Speaker CAM++ (zh-cn 16k common)",
    spdx: "Apache-2.0",
  },
];

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
