/**
 * Themed right-click context menu (issue #0034 — meeting-list slice).
 *
 * A small floating menu anchored at an arbitrary viewport point (typically a
 * `contextmenu` event's cursor position). Replaces the native WebView2 menu on
 * the surface that renders it; the caller is responsible for suppressing the
 * native menu (`preventDefault()` on `contextmenu`) on that surface only — this
 * component has no opinion about where it is invoked from.
 *
 * Entries are one level deep: a leaf action, or a submenu that expands inline
 * (used for "Move to…"'s folder list). Dismisses on outside click, Escape, or
 * after any leaf entry is chosen. Renders with `theme.css` tokens only, so it
 * matches the light/dark theme automatically.
 */
import { useEffect, useLayoutEffect, useRef, useState } from "react";
import "./ContextMenu.css";

export type ContextMenuEntry =
  | {
      kind?: "item";
      label: string;
      onSelect: () => void;
      danger?: boolean;
      disabled?: boolean;
    }
  | {
      kind: "submenu";
      label: string;
      items: { label: string; onSelect: () => void; current?: boolean }[];
      emptyLabel?: string;
    };

export type ContextMenuProps = {
  /** Viewport coordinates for the menu's top-left corner (before clamping). */
  x: number;
  y: number;
  entries: ContextMenuEntry[];
  onClose: () => void;
};

export function ContextMenu({ x, y, entries, onClose }: ContextMenuProps) {
  const menuRef = useRef<HTMLDivElement | null>(null);
  const itemRefs = useRef<(HTMLButtonElement | null)[]>([]);
  const [submenuIndex, setSubmenuIndex] = useState<number | null>(null);
  const [pos, setPos] = useState({ x, y });

  // The menu's own size is unknown until it mounts, so clamp to the viewport
  // (with a small margin) in a layout effect rather than up front — a
  // right-click near the window's edge would otherwise render off-screen.
  useLayoutEffect(() => {
    const el = menuRef.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    const margin = 8;
    const maxX = Math.max(margin, window.innerWidth - rect.width - margin);
    const maxY = Math.max(margin, window.innerHeight - rect.height - margin);
    setPos({ x: Math.min(x, maxX), y: Math.min(y, maxY) });
  }, [x, y]);

  // Move focus into the menu on open, and restore it to whatever had focus
  // beforehand (the row) on close, so keyboard/screen-reader users land back
  // where they started.
  useEffect(() => {
    const previouslyFocused = document.activeElement as HTMLElement | null;
    itemRefs.current.find((el) => el !== null)?.focus();
    return () => {
      previouslyFocused?.focus();
    };
  }, []);

  useEffect(() => {
    function focusableItems(): HTMLButtonElement[] {
      return itemRefs.current.filter((el): el is HTMLButtonElement => el !== null);
    }

    function onKeyDown(e: KeyboardEvent) {
      if (e.key === "Escape") {
        onClose();
        return;
      }
      const items = focusableItems();
      if (items.length === 0) return;
      const currentIndex = items.indexOf(document.activeElement as HTMLButtonElement);
      if (e.key === "ArrowDown") {
        e.preventDefault();
        items[(currentIndex + 1) % items.length].focus();
      } else if (e.key === "ArrowUp") {
        e.preventDefault();
        items[(currentIndex - 1 + items.length) % items.length].focus();
      }
    }
    document.addEventListener("keydown", onKeyDown, true);
    return () => document.removeEventListener("keydown", onKeyDown, true);
  }, [onClose]);

  return (
    <>
      {/* Full-viewport transparent backdrop, behind the menu, that closes it on
          any outside click — mirrors `MeetingList.tsx`'s `MoveMenu` popover. */}
      <button
        type="button"
        className="context-menu__backdrop"
        aria-label="Close menu"
        tabIndex={-1}
        onClick={onClose}
      />
      <div
        ref={menuRef}
        className="context-menu"
        role="menu"
        style={{ left: pos.x, top: pos.y }}
      >
        {entries.map((entry, i) =>
          entry.kind === "submenu" ? (
            <div key={entry.label} className="context-menu__submenu-wrap">
              <button
                type="button"
                role="menuitem"
                aria-haspopup="menu"
                aria-expanded={submenuIndex === i}
                ref={(el) => {
                  itemRefs.current[i] = el;
                }}
                className="context-menu__item"
                onClick={() => setSubmenuIndex((cur) => (cur === i ? null : i))}
              >
                {entry.label}
              </button>
              {submenuIndex === i && (
                <div className="context-menu__submenu" role="menu">
                  {entry.items.length === 0 ? (
                    <span className="context-menu__empty">
                      {entry.emptyLabel ?? "Nothing here"}
                    </span>
                  ) : (
                    entry.items.map((sub) => (
                      <button
                        key={sub.label}
                        type="button"
                        role="menuitem"
                        aria-current={sub.current}
                        className="context-menu__item"
                        onClick={() => {
                          sub.onSelect();
                          onClose();
                        }}
                      >
                        {sub.label}
                      </button>
                    ))
                  )}
                </div>
              )}
            </div>
          ) : (
            <button
              key={entry.label}
              type="button"
              role="menuitem"
              ref={(el) => {
                itemRefs.current[i] = el;
              }}
              disabled={entry.disabled}
              className={
                entry.danger
                  ? "context-menu__item context-menu__item--danger"
                  : "context-menu__item"
              }
              onClick={() => {
                entry.onSelect();
                onClose();
              }}
            >
              {entry.label}
            </button>
          ),
        )}
      </div>
    </>
  );
}
