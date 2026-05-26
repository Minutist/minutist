# meeting-app — architecture

Live architecture documentation. Authoritative for component boundaries,
interfaces, and ownership.

## Read order

1. [`system-context.md`](system-context.md) — C4 Level 1. What the system
   talks to.
2. [`containers.md`](containers.md) — C4 Level 2. The runtime units that
   make up the app.
3. [`components.md`](components.md) — C4 Level 3. The Rust crates inside
   the core, the React components inside the webview.
4. [`cross-cutting.md`](cross-cutting.md) — threading, errors, logging,
   model lifecycle, IPC contract. Concerns that touch every component.
5. [`domain-ownership.md`](domain-ownership.md) — which crate belongs to
   which agent role, what dependencies are allowed, what isn't.

## Authoritative artefacts

- [`workspace.dsl`](workspace.dsl) — Structurizr DSL. Source of truth
  for the diagrams. Edit this; the SVGs are derived.
- `L1_SystemContext.svg`, `L2_Containers.svg`,
  `L3_CoreComponents.svg`, `L3_WebviewComponents.svg` — rendered views.
  Regenerate via [`../scripts/render-architecture.sh`](../scripts/render-architecture.sh).

## Live-doc convention

These docs are not historical artefacts. They describe the architecture
**as it is in the current commit**. Two enforcement mechanisms keep
them honest:

1. **Pre-commit hook.** `.githooks/pre-commit` fails any commit that
   touches a `crates/`, `src-tauri/`, or `ui/src/` file without also
   touching `architecture/`. The hook is installable via
   `git config core.hooksPath .githooks` — see the repo root README.
2. **Reviewer prompt.** Every `principal-code-reviewer` pass is briefed
   with `components.md`, `domain-ownership.md`, and `cross-cutting.md`
   as context. The reviewer is expected to flag boundary violations and
   stale docs as review findings.

If a change makes these docs wrong, the change is the part that needs
to update them — not the docs that get archived.

## What is **not** here

- Implementation details that live in code: function signatures, error
  enum variants, exact trait method shapes. The docs name the trait;
  the code defines it.
- Decisions that are still open, and the historical reasoning behind
  closed ones. Those live in the product specification and engineering
  journal, which are kept outside the repository.

## When to update

Any change that:

- adds or removes a crate
- changes a crate's responsibility
- adds, removes, or renames a public trait or shared type in `common`
- alters a dependency edge between components
- changes the threading model, error-propagation contract, or IPC surface

…must update the relevant files here in the same commit. The pre-commit
hook checks the cheap part of this (architecture path touched); the
reviewer checks the substantive part.
