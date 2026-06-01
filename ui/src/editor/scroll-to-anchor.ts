/**
 * Scroll-to-nearest-anchor helper (FR-23, notes side).
 *
 * Given a target anchor ms (a clicked transcript segment's `start_ms`, on the
 * pause-EXCLUDING recording clock) and the editor's root DOM element, finds the
 * paragraph whose `data-anchor-ms` is nearest the target and scrolls it into
 * view. "Nearest" is the same start_ms-distance metric the cross-ref store uses
 * for the FR-22 direction, so the two directions are symmetric.
 *
 * Pure DOM read + `scrollIntoView`; it never mutates the ProseMirror document,
 * so it cannot disturb anchor stamping.
 */
import { ANCHOR_ATTR } from "./paragraph-anchor";

/**
 * Find the anchored element in `root` whose `data-anchor-ms` is nearest
 * `targetMs`, or `null` when there are no anchored paragraphs.
 *
 * Exported for unit testing the mapping independent of `scrollIntoView` (which
 * jsdom does not implement).
 */
export function nearestAnchoredElement(
  root: HTMLElement,
  targetMs: number,
): HTMLElement | null {
  const candidates = root.querySelectorAll<HTMLElement>(`[${ANCHOR_ATTR}]`);
  let best: HTMLElement | null = null;
  let bestDistance = Number.POSITIVE_INFINITY;
  candidates.forEach((el) => {
    const raw = el.getAttribute(ANCHOR_ATTR);
    if (raw === null) return;
    const ms = Number.parseInt(raw, 10);
    if (Number.isNaN(ms)) return;
    const distance = Math.abs(ms - targetMs);
    if (distance < bestDistance) {
      bestDistance = distance;
      best = el;
    }
  });
  return best;
}

/**
 * Scroll the paragraph nearest `targetMs` into view. Returns the element that
 * was scrolled to (or `null` when none matched), so callers can apply a
 * transient highlight if they wish.
 */
export function scrollToNearestAnchor(
  root: HTMLElement,
  targetMs: number,
): HTMLElement | null {
  const el = nearestAnchoredElement(root, targetMs);
  if (!el) return null;
  // `scrollIntoView` is a browser API; guard so the call is a no-op under jsdom.
  if (typeof el.scrollIntoView === "function") {
    el.scrollIntoView({ behavior: "smooth", block: "center" });
  }
  return el;
}
