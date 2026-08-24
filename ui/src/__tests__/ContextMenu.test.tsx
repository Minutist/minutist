/**
 * Themed context menu tests (issue #0034).
 *
 * Focuses on the mount/unmount focus management: it must move focus into the
 * menu on open and restore it on close, WITHOUT letting the browser's default
 * focus-driven scroll fire. A plain `.focus()` on a menu item — even though
 * the menu itself is `position: fixed` at the cursor — has the browser scroll
 * the nearest scrollable ancestor (the meeting list) to "reveal" it, which in
 * practice snapped the list back to the top on every right-click. Regression
 * coverage for that: assert every `.focus()` call in this component passes
 * `{ preventScroll: true }`.
 */
import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { ContextMenu } from "../shell/ContextMenu";
import type { ContextMenuEntry } from "../shell/ContextMenu";

function entries(onSelect: () => void = vi.fn()): ContextMenuEntry[] {
  return [
    { label: "Open", onSelect },
    { label: "Delete", onSelect: vi.fn(), danger: true },
  ];
}

describe("ContextMenu", () => {
  it("focuses the first item on mount with preventScroll", () => {
    const focusSpy = vi.spyOn(HTMLElement.prototype, "focus");
    render(<ContextMenu x={10} y={10} entries={entries()} onClose={vi.fn()} />);

    expect(focusSpy).toHaveBeenCalledWith({ preventScroll: true });
    expect(screen.getByRole("menuitem", { name: "Open" })).toHaveFocus();
    focusSpy.mockRestore();
  });

  it("restores the previously-focused element with preventScroll on unmount", () => {
    const trigger = document.createElement("button");
    document.body.appendChild(trigger);
    trigger.focus();

    const focusSpy = vi.spyOn(HTMLElement.prototype, "focus");
    const { unmount } = render(
      <ContextMenu x={10} y={10} entries={entries()} onClose={vi.fn()} />,
    );
    focusSpy.mockClear();
    unmount();

    expect(focusSpy).toHaveBeenCalledWith({ preventScroll: true });
    expect(trigger).toHaveFocus();
    focusSpy.mockRestore();
    trigger.remove();
  });

  it("ArrowDown/ArrowUp move focus between items with preventScroll", () => {
    render(<ContextMenu x={10} y={10} entries={entries()} onClose={vi.fn()} />);

    const focusSpy = vi.spyOn(HTMLElement.prototype, "focus");
    fireEvent.keyDown(document, { key: "ArrowDown" });
    expect(screen.getByRole("menuitem", { name: "Delete" })).toHaveFocus();
    expect(focusSpy).toHaveBeenCalledWith({ preventScroll: true });

    fireEvent.keyDown(document, { key: "ArrowUp" });
    expect(screen.getByRole("menuitem", { name: "Open" })).toHaveFocus();
    focusSpy.mockRestore();
  });

  it("Escape calls onClose", () => {
    const onClose = vi.fn();
    render(<ContextMenu x={10} y={10} entries={entries()} onClose={onClose} />);
    fireEvent.keyDown(document, { key: "Escape" });
    expect(onClose).toHaveBeenCalledOnce();
  });

  it("clicking an entry calls onSelect then onClose", () => {
    const onSelect = vi.fn();
    const onClose = vi.fn();
    render(
      <ContextMenu x={10} y={10} entries={entries(onSelect)} onClose={onClose} />,
    );
    fireEvent.click(screen.getByRole("menuitem", { name: "Open" }));
    expect(onSelect).toHaveBeenCalledOnce();
    expect(onClose).toHaveBeenCalledOnce();
  });

  it("clicking the backdrop calls onClose", () => {
    const onClose = vi.fn();
    render(<ContextMenu x={10} y={10} entries={entries()} onClose={onClose} />);
    fireEvent.click(screen.getByRole("button", { name: "Close menu" }));
    expect(onClose).toHaveBeenCalledOnce();
  });
});
