# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Verify every intra-workspace crate dependency edge is declared in the
dependency table in `architecture/components.md`.

Algorithm: read `[workspace].members` from the root `Cargo.toml` (excluding
`spikes/*`) to get the set of canonical crate names (directory basename,
with the one alias `src-tauri` -> `app-main`). Parse the markdown table under
"## Dependency rule" in `architecture/components.md` into `allowed[crate] =
{permitted workspace deps}`. Then, for each workspace crate's `Cargo.toml`,
walk its `[dependencies]` and `[target.*.dependencies]` tables (never
`[dev-dependencies]` / `[build-dependencies]` — those are not runtime edges),
resolve each `{ path = "..." }` entry to a canonical crate name by path
basename, and flag any edge whose target is not in `allowed[crate]`.

Limitation: intra-workspace edges are resolved via `{ path = "..." }` entries
only. A workspace-internal crate pulled in via `{ workspace = true }` (i.e.
declared with a path under the root `[workspace.dependencies]`) would not be
detected. That pattern is not used today — every internal edge is a direct
path dependency — but if it is adopted, this checker must be extended to
resolve workspace-dependency aliases back to their root path.
"""

import argparse
import re
import sys
import tomllib
from pathlib import Path

# Footnote markers used in the components.md table to annotate optional /
# feature-gated edges (e.g. `mcp-server`†). They always sit outside the
# backtick-quoted crate name, but tokens are stripped of them defensively.
FOOTNOTE_MARKERS = "†‡§※‖¶"

TABLE_HEADER = "| Crate | First phase | May depend on |"


def workspace_crates(repo_root: Path) -> dict[str, Path]:
    """Return {canonical_name: crate_dir} for every non-spike workspace member."""
    root_toml = repo_root / "Cargo.toml"
    with root_toml.open("rb") as f:
        data = tomllib.load(f)
    members = data["workspace"]["members"]

    crates: dict[str, Path] = {}
    for member in members:
        if member.startswith("spikes/"):
            continue
        member_dir = repo_root / member
        if member == "src-tauri":
            canonical = "app-main"
        elif member.startswith("crates/"):
            canonical = Path(member).name
        else:
            # Not a crate/src-tauri member (shouldn't happen given the current
            # workspace layout); skip rather than guess a canonical name.
            continue
        crates[canonical] = member_dir
    return crates


def parse_dependency_table(components_md: Path) -> dict[str, set[str]]:
    """Parse the "May depend on" table into {crate: {allowed workspace deps}}."""
    lines = components_md.read_text(encoding="utf-8").splitlines()

    try:
        start = next(i for i, l in enumerate(lines) if l.strip() == TABLE_HEADER)
    except StopIteration:
        print(
            f"ERROR: could not find the dependency table header {TABLE_HEADER!r} "
            f"in {components_md}",
            file=sys.stderr,
        )
        sys.exit(2)

    allowed: dict[str, set[str]] = {}
    # Skip the header row and the `|---|---|---|` separator, then read data
    # rows until the table ends (a line that no longer starts with '|').
    for line in lines[start + 2 :]:
        if not line.strip().startswith("|"):
            break
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != 3:
            continue
        crate_cell, _phase_cell, deps_cell = cells

        name_match = re.search(r"`([^`]+)`", crate_cell)
        if not name_match:
            continue
        crate_name = name_match.group(1).strip()

        deps: set[str] = set()
        for tok in re.findall(r"`([^`]+)`", deps_cell):
            tok = tok.strip().strip(FOOTNOTE_MARKERS).strip()
            if tok:
                deps.add(tok)
        allowed[crate_name] = deps

    return allowed


def runtime_path_deps(cargo_toml: Path, workspace_names: set[str]) -> set[str]:
    """Return the set of workspace-crate names this Cargo.toml depends on at
    runtime (i.e. via [dependencies] or [target.*.dependencies] tables with a
    `path = "..."` entry resolving to another workspace member)."""
    with cargo_toml.open("rb") as f:
        data = tomllib.load(f)

    dep_tables = []
    if "dependencies" in data:
        dep_tables.append(data["dependencies"])
    for cfg_table in data.get("target", {}).values():
        if "dependencies" in cfg_table:
            dep_tables.append(cfg_table["dependencies"])

    edges: set[str] = set()
    for dep_table in dep_tables:
        for value in dep_table.values():
            if isinstance(value, dict) and "path" in value:
                target = Path(value["path"]).name
                if target in workspace_names:
                    edges.add(target)
    return edges


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

    crates = workspace_crates(repo_root)
    workspace_names = set(crates)
    allowed = parse_dependency_table(repo_root / "architecture" / "components.md")

    violations: list[str] = []
    total_edges = 0

    for canonical, crate_dir in sorted(crates.items()):
        if canonical not in allowed:
            violations.append(
                f"VIOLATION: {canonical} has no row in the components.md "
                f"dependency table ({crate_dir / 'Cargo.toml'})"
            )
            continue

        cargo_toml = crate_dir / "Cargo.toml"
        edges = runtime_path_deps(cargo_toml, workspace_names)
        for dep in sorted(edges):
            total_edges += 1
            if dep not in allowed[canonical]:
                violations.append(
                    f"VIOLATION: {canonical} depends on {dep} — not in the "
                    f"components.md dependency table ({cargo_toml})"
                )

    if violations:
        for v in violations:
            print(v)
        return 1

    print(f"dep-table check OK: {len(crates)} crates, {total_edges} edges verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
