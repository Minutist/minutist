/**
 * Zustand store for the Phase-7 first-run onboarding flow.
 *
 * Owns only the transient *navigation* state — which step is showing — and the
 * step-progression actions. Onboarding completion is NOT stored here: it lives
 * in the persisted `settings.onboarding_completed` field, read/written through
 * the existing settings seam (the recording store's `settings` snapshot +
 * `setOnboardingCompleted`, which rounds through `commands.updateSettings`).
 * Keeping completion in settings — not in this store — means the gate is driven
 * by the same source of truth the backend persists, so a returning user is
 * never re-onboarded because of a stale in-memory flag.
 *
 * Mirrors the store conventions in `ui/src/state/` (a `create<T>()` with a flat
 * state shape and action methods).
 */
import { create } from "zustand";

/** The three onboarding steps, in order. */
export const ONBOARDING_STEPS = ["welcome", "model", "settings"] as const;
export type OnboardingStep = (typeof ONBOARDING_STEPS)[number];

export type OnboardingStore = {
  /** The currently-displayed step. */
  step: OnboardingStep;
  /** Advance to the next step; a no-op on the final step. */
  next: () => void;
  /** Go back to the previous step; a no-op on the first step. */
  back: () => void;
  /** Jump to a specific step (used by tests and any future step index). */
  goTo: (step: OnboardingStep) => void;
  /** Reset to the first step (used when the flow is (re)entered). */
  reset: () => void;
};

function stepIndex(step: OnboardingStep): number {
  return ONBOARDING_STEPS.indexOf(step);
}

export const useOnboardingStore = create<OnboardingStore>((set, get) => ({
  step: "welcome",

  next: () => {
    const idx = stepIndex(get().step);
    const nextIdx = Math.min(idx + 1, ONBOARDING_STEPS.length - 1);
    set({ step: ONBOARDING_STEPS[nextIdx] });
  },

  back: () => {
    const idx = stepIndex(get().step);
    const prevIdx = Math.max(idx - 1, 0);
    set({ step: ONBOARDING_STEPS[prevIdx] });
  },

  goTo: (step) => set({ step }),

  reset: () => set({ step: "welcome" }),
}));
