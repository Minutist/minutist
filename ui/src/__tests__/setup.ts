import "@testing-library/jest-dom";
import { vi } from "vitest";

/**
 * jsdom does not implement `ResizeObserver`, which `react-resizable-panels`
 * (used by the two-pane layout) reads from `ownerDocument.defaultView`. Provide
 * a no-op polyfill so component tests can mount the panel group. The real Tauri
 * webview (Chromium/WebKit) ships `ResizeObserver` natively, so this stub is
 * test-only.
 */
if (typeof globalThis.ResizeObserver === "undefined") {
  class ResizeObserverStub {
    observe(): void {}
    unobserve(): void {}
    disconnect(): void {}
  }
  globalThis.ResizeObserver =
    ResizeObserverStub as unknown as typeof ResizeObserver;
}

/**
 * `@tanstack/react-virtual` (TranscriptPane's row virtualiser) measures the
 * scroll container's real layout box (`offsetWidth`/`offsetHeight`) to decide
 * the visible window; jsdom never lays anything out, so every element reports
 * a permanent 0×0 and the real virtualiser would render zero rows regardless
 * of `estimateSize`/`overscan`/`initialRect`. Component tests exist to verify
 * TranscriptPane's OWN wiring (row content, order, click-to-jump, highlight,
 * drag-start) — not the third-party library's viewport math, which has its
 * own test suite — so replace it here with a measurement-free fake that
 * always renders the full range. This is the only module in `ui/src` that
 * imports `@tanstack/react-virtual`, so the mock cannot affect anything else.
 */
vi.mock("@tanstack/react-virtual", () => ({
  useVirtualizer: <T,>(options: {
    count: number;
    estimateSize: (index: number) => number;
    getItemKey?: (index: number) => T;
  }) => {
    const sizes = Array.from({ length: options.count }, (_, i) =>
      options.estimateSize(i),
    );
    const starts = sizes.reduce<number[]>((acc, size, i) => {
      acc.push(i === 0 ? 0 : acc[i - 1] + sizes[i - 1]);
      return acc;
    }, []);
    return {
      getVirtualItems: () =>
        sizes.map((size, index) => ({
          index,
          key: options.getItemKey ? options.getItemKey(index) : index,
          start: starts[index],
          size,
        })),
      getTotalSize: () => sizes.reduce((sum, s) => sum + s, 0),
      scrollToIndex: () => {},
      scrollToOffset: () => {},
      measureElement: () => {},
    };
  },
}));
