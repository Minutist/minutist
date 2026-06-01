import "@testing-library/jest-dom";

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
