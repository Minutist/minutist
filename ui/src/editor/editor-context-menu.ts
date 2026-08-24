/**
 * Notes-editor context-menu entries (issue #0034).
 *
 * A pure function of the live Tiptap editor: cut/copy/paste (delegated to the
 * browser's own clipboard events via `document.execCommand`, so they reuse
 * the SAME `handleDOMEvents.copy`/`cut`/`paste` handling `Editor.tsx` already
 * wires for the keyboard-shortcut path — no separate clipboard-payload logic
 * to keep in sync) plus the common StarterKit formatting toggles.
 *
 * `execCommand` is deprecated for many purposes but still dispatches a real,
 * trusted `ClipboardEvent` in the Chromium engine WebView2 embeds, which is
 * exactly what ProseMirror's own copy/cut/paste handling (and this app's
 * override) listens for — using it here is not reaching for a legacy API for
 * its own sake, it is the one remaining way to trigger the native clipboard
 * pipeline from a menu click rather than a real keyboard shortcut.
 */
import type { Editor as TiptapEditor } from "@tiptap/core";
import type { ContextMenuEntry } from "../shell/ContextMenu";

function refocus(editor: TiptapEditor) {
  editor.chain().focus().run();
}

export function buildEditorMenuEntries(editor: TiptapEditor): ContextMenuEntry[] {
  const hasSelection = !editor.state.selection.empty;
  const isLink = editor.isActive("link");

  return [
    {
      label: "Cut",
      disabled: !hasSelection,
      onSelect: () => {
        refocus(editor);
        document.execCommand("cut");
      },
    },
    {
      label: "Copy",
      disabled: !hasSelection,
      onSelect: () => {
        refocus(editor);
        document.execCommand("copy");
      },
    },
    {
      label: "Paste",
      onSelect: () => {
        refocus(editor);
        document.execCommand("paste");
      },
    },
    { kind: "divider" },
    {
      label: "Bold",
      checked: editor.isActive("bold"),
      onSelect: () => editor.chain().focus().toggleBold().run(),
    },
    {
      label: "Italic",
      checked: editor.isActive("italic"),
      onSelect: () => editor.chain().focus().toggleItalic().run(),
    },
    {
      label: "Bulleted list",
      checked: editor.isActive("bulletList"),
      onSelect: () => editor.chain().focus().toggleBulletList().run(),
    },
    {
      kind: "submenu",
      label: "Heading",
      items: [1, 2, 3].map((level) => ({
        label: `Heading ${level}`,
        current: editor.isActive("heading", { level }),
        onSelect: () =>
          editor
            .chain()
            .focus()
            .toggleHeading({ level: level as 1 | 2 | 3 })
            .run(),
      })),
    },
    isLink
      ? {
          label: "Remove link",
          onSelect: () => editor.chain().focus().unsetLink().run(),
        }
      : {
          label: "Add link…",
          disabled: !hasSelection,
          onSelect: () => {
            const url = window.prompt("Link URL", "https://");
            if (!url) return;
            editor.chain().focus().setLink({ href: url }).run();
          },
        },
  ];
}
