#!/usr/bin/env bash
#
# Render the C4 architecture diagrams from the Structurizr DSL.
#
# Two-step pipeline:
#   1. structurizr/structurizr (Docker)  — DSL → Mermaid (.mmd files)
#   2. minlag/mermaid-cli       (Docker) — Mermaid (.mmd) → SVG
#
# Outputs live alongside the DSL in architecture/. The .mmd files
# are intermediate; SVGs are the rendered artefacts checked in alongside
# the DSL.
#
# Requires Docker. No host-side Java / Node toolchains needed.

set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
arch_dir="$repo_root/architecture"
dsl="$arch_dir/workspace.dsl"

if [ ! -f "$dsl" ]; then
    echo "error: $dsl not found" >&2
    exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
    echo "error: docker not found in PATH" >&2
    exit 1
fi

uid="$(id -u)"
gid="$(id -g)"

echo "==> Exporting DSL to Mermaid"
docker run --rm \
    -v "$repo_root:$repo_root" \
    -w "$repo_root" \
    --user "$uid:$gid" \
    structurizr/structurizr \
    export \
    -w architecture/workspace.dsl \
    -f mermaid \
    -o architecture

echo "==> Converting Mermaid to SVG"
shopt -s nullglob
for mmd in "$arch_dir"/structurizr-*.mmd; do
    base="$(basename "$mmd" .mmd)"
    # Drop the "structurizr-" prefix the exporter adds.
    svg_name="${base#structurizr-}.svg"
    echo "    $svg_name"
    docker run --rm \
        --user "$uid:$gid" \
        -v "$arch_dir:/data" \
        minlag/mermaid-cli \
        -i "/data/$(basename "$mmd")" \
        -o "/data/$svg_name" \
        >/dev/null
done

# The .mmd files are intermediate and excluded via .gitignore. Keep them
# on disk so a diff is available when iterating; only the .svg renders are
# committed alongside the DSL.
echo "==> Done. SVGs in $arch_dir"
