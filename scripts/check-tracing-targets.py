# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Lint `tracing::*!` macro calls for the target-directive convention.

Check A (all crates): flags the field form `target = "..."` — it silently
defeats `RUST_LOG` filtering, because `tracing`'s `target` is a macro-time
directive (`target: "..."`), not a runtime field. Check B (all crates):
flags any `tracing::*!` call with no `target:` directive at all. Check C
(all crates): flags a `target: "..."` directive whose STRING VALUE isn't
the calling crate's own name — the actual rule text
(`architecture/cross-cutting.md` — "Logging": "Each component uses a
static target matching the crate name"), not just "has some target".

All three checks run against a MASKED copy of each macro span (string-literal
and comment CONTENTS blanked, delimiters kept) so a `target =` / `target:`
occurrence inside a log message or comment cannot trip Check A or suppress a
real Check B/C violation. See `find_macro_spans`.

Assumption: this only matches fully-qualified `tracing::<level>!(...)` calls
(the convention this codebase uses everywhere); a bare imported `info!(...)`
would not be matched.

Only `crates/*/src/` and `src-tauri/src/` are scanned — `spikes/` and any
`tests/` directory are excluded (spikes are throwaway; integration tests
under a `tests/` directory are not part of the shipped crate's logging
surface this lint governs).
"""

import argparse
import re
import sys
from pathlib import Path

# Crates whose logging target deliberately differs from their crate/directory
# name. `headless`'s daemon is user-facing as `minutist-hub` (see
# architecture/cross-cutting.md — "Headless server daemon"); nobody filters
# logs by the internal crate name "headless", so its every call targets "hub"
# consistently. Extend only with the same rationale — a name users actually
# filter logs by, not a typo.
TARGET_OVERRIDES = {"headless": "hub"}

MACRO_RE = re.compile(r"tracing::(?:trace|debug|info|warn|error|event)!\s*\(")

# Check A: the field form `target = "..."` (defeats RUST_LOG filtering).
TARGET_FIELD_RE = re.compile(r"(?<!\w)target\s*=\s*\"")
# Check B/C: the directive form `target: "..."` — captures the string value
# for Check C's crate-name comparison.
TARGET_DIRECTIVE_RE = re.compile(r"(?<!\w)target\s*:\s*\"([^\"]*)\"")

ESCAPED_CHAR_LITERAL_RE = re.compile(
    r"'\\\\(?:x[0-9A-Fa-f]{2}|u\{[0-9A-Fa-f]+\}|.)'"
)


def find_macro_spans(text: str) -> list[tuple[int, str, str]]:
    """Return (1-based line number, MASKED call span text, ORIGINAL call span
    text) for every `tracing::<level>!(...)` invocation in `text`.

    Scans forward from the opening paren counting depth, skipping over
    string literals (with `\\"` escapes), char literals, and `//` line
    comments, so parens inside those don't miscount.

    The returned span is *masked*: the contents strictly inside a string
    literal and inside a `//` comment are replaced with spaces (newlines
    preserved so line offsets are unchanged), while the delimiters (`"`, the
    `//`) are kept. This is deliberate — the target regexes run against the
    masked span so that a `target =` / `target:` occurrence inside a log
    MESSAGE (string content) or a comment cannot masquerade as the real
    macro-level directive/field. A genuine `target = "..."` / `target: "..."`
    stays visible because the key and the opening quote are code, not string
    content. Char literals are copied verbatim (they cannot contain `target`).
    Masking preserves length 1:1, so the reported line number (of the macro
    call itself) is unaffected.
    """
    spans = []
    n = len(text)
    for m in MACRO_RE.finditer(text):
        start = m.start()
        i = m.end() - 1  # index of the opening '('
        # The `tracing::<level>!` prefix up to the '(' is code — copy verbatim.
        masked: list[str] = list(text[start:i])
        depth = 0
        while i < n:
            c = text[i]
            if c == '"':
                masked.append('"')  # opening delimiter kept
                i += 1
                while i < n:
                    if text[i] == "\\":
                        # Escape sequence: both chars are string content, blank
                        # them (preserve an actual newline for line-continuation).
                        masked.append(" ")
                        if i + 1 < n:
                            masked.append("\n" if text[i + 1] == "\n" else " ")
                        i += 2
                        continue
                    if text[i] == '"':
                        masked.append('"')  # closing delimiter kept
                        i += 1
                        break
                    # String content: blank (preserve newline).
                    masked.append("\n" if text[i] == "\n" else " ")
                    i += 1
                continue
            if c == "/" and i + 1 < n and text[i + 1] == "/":
                nl = text.find("\n", i)
                end = n if nl == -1 else nl
                # Keep the `//` delimiter; blank the comment body to end-of-line.
                masked.append("/")
                masked.append("/")
                masked.extend(" " * (end - (i + 2)))
                i = end
                continue
            if c == "'":
                esc = ESCAPED_CHAR_LITERAL_RE.match(text, i)
                if esc:
                    masked.extend(text[i : esc.end()])
                    i = esc.end()
                    continue
                if i + 2 < n and text[i + 1] != "'" and text[i + 2] == "'":
                    masked.extend(text[i : i + 3])
                    i += 3
                    continue
                # Otherwise this is a lifetime (`'a`), not a char literal —
                # leave it alone.
                masked.append(c)
                i += 1
                continue
            if c == "(":
                depth += 1
                masked.append(c)
                i += 1
                continue
            if c == ")":
                depth -= 1
                masked.append(c)
                i += 1
                if depth == 0:
                    break
                continue
            masked.append(c)
            i += 1
        line_no = text.count("\n", 0, start) + 1
        spans.append((line_no, "".join(masked), text[start:i]))
    return spans


def rust_files(repo_root: Path) -> list[Path]:
    src_dirs = []
    crates_dir = repo_root / "crates"
    if crates_dir.is_dir():
        for crate_dir in sorted(crates_dir.iterdir()):
            src_dir = crate_dir / "src"
            if src_dir.is_dir():
                src_dirs.append(src_dir)
    tauri_src = repo_root / "src-tauri" / "src"
    if tauri_src.is_dir():
        src_dirs.append(tauri_src)

    files = []
    for src_dir in src_dirs:
        for p in sorted(src_dir.rglob("*.rs")):
            if "spikes" in p.parts or "tests" in p.parts:
                continue
            files.append(p)
    return files


def crate_name_for(path: Path, repo_root: Path) -> str | None:
    parts = path.relative_to(repo_root).parts
    if parts[0] == "crates":
        return parts[1]
    if parts[0] == "src-tauri":
        return "app-main"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Repository root (default: parent of this script's directory).",
    )
    args = parser.parse_args()
    repo_root: Path = args.root

    violations: list[str] = []
    files_scanned = 0
    calls_checked = 0

    for path in rust_files(repo_root):
        files_scanned += 1
        text = path.read_text(encoding="utf-8")
        rel = path.relative_to(repo_root)
        crate = crate_name_for(path, repo_root)

        for line_no, masked_span, original_span in find_macro_spans(text):
            calls_checked += 1
            if TARGET_FIELD_RE.search(masked_span):
                violations.append(
                    f"VIOLATION (target=): {rel}:{line_no} uses field syntax "
                    f"'target =' instead of the directive 'target:'"
                )
                continue
            directive = TARGET_DIRECTIVE_RE.search(masked_span)
            if directive is None:
                violations.append(
                    f"VIOLATION (no target): {rel}:{line_no} tracing call in "
                    f"{crate} has no target:"
                )
            else:
                # The masked match's offsets align 1:1 with original_span (masking
                # preserves length), so re-read the real string value from there —
                # the masked group is blanked spaces, not the actual target.
                value = original_span[directive.start(1) : directive.end(1)]
                expected = TARGET_OVERRIDES.get(crate, crate)
                if crate is not None and value != expected:
                    violations.append(
                        f'VIOLATION (wrong target): {rel}:{line_no} tracing call '
                        f'in {crate} targets "{value}", expected "{expected}"'
                    )

    if violations:
        for v in violations:
            print(v)
        return 1

    print(
        f"tracing-target check OK: {files_scanned} files, "
        f"{calls_checked} tracing calls verified"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
