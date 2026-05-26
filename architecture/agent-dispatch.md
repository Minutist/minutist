# Agent dispatch

How to hand work to sub-agents inside this project. This document
operationalises [`domain-ownership.md`](domain-ownership.md): it tells the
main session **which** agent to dispatch for **which** role, with **what**
context, under **which** isolation model.

## Why this exists

The architecture defines abstract roles (`audio-engineer`,
`ml-runtime-engineer`, etc.). The Agent tool dispatches concrete subagent
types (`software:scrum-developer`, `software:scrum-sqe`, etc.). The two
don't map automatically; this doc fixes that mapping and the operational
rules that make parallel dispatch safe.

## Prerequisites for parallel dispatch

Before fanning out work to multiple sub-agents in a phase:

1. **`crates/common` is locked for the phase.** All shared types and
   trait signatures the phase will use already exist as stubs in
   `common`. Adding a `pub` item in `common` mid-phase invalidates every
   running sub-agent's working tree.
2. **The phase plan identifies independent work-streams.** Two streams
   are independent if they touch disjoint crate sets and neither needs
   the other's output to compile. The `scrum-planner` produces this
   identification.
3. **Each stream has a join target.** A named crate or interaction that
   the streams converge into. Usually the orchestrator-integration step
   for live-pipeline phases.

If any of these is missing, do not fan out. Run serial.

## Role-to-subagent mapping

| Architecture role | Subagent type | Owns crates / paths |
|---|---|---|
| architecture-owner | (main session, human-in-loop) | `crates/common`, `architecture/**` |
| audio-engineer | `software:scrum-developer` | `crates/audio-capture`, `crates/vad-chunker` |
| ml-runtime-engineer | `software:scrum-developer` | `crates/asr-runtime`, `crates/diarizer`, `crates/summariser`, `crates/model-registry` |
| data-engineer | `software:scrum-developer` | `crates/persistence`, `crates/settings` |
| systems-engineer | `software:scrum-developer` | `crates/orchestrator`, `crates/ipc-bridge`, `src-tauri/**` |
| frontend-engineer | `software:scrum-developer` | `ui/src/**` |
| test author (any role) | `software:scrum-sqe` | tests within the role's crates |
| independent verdict | `software:scrum-tester` | (read-only across all crates) |
| code review | `principal-code-reviewer` | (read-only across all crates) |
| acceptance / PO | `software:scrum-po` | (read-only across all crates) |

A single agent can hold multiple roles when work is sequential; multiple
agents can share a role when the role's crates have no internal
dependency (e.g., `audio-capture` and `vad-chunker` could parallelise
within `audio-engineer` once `common` exposes `AudioChunk`).

## Isolation model

Parallel agents work in **separate worktrees** (`isolation: "worktree"`
on the Agent call). Each worktree is an isolated copy of the repo on a
branch.

Why worktrees:

- Lockfile races. Two `cargo build`s on the same worktree race on
  `Cargo.lock`.
- File contention. Two simultaneous edits of `Cargo.toml` clash.
- Test parallelism. Each worktree gets its own `target/` directory.

Conventions:

- One worktree per stream. Each agent edits only inside its owned
  crates, plus `architecture/components.md` to add a 1-line per-crate
  note (required by the pre-commit hook).
- **No agent edits `crates/common`.** That's architecture-owner work.
  If a stream discovers it needs a new shared type, it stops and reports
  back to the main session.
- **No agent edits the workspace `Cargo.toml`.** Workspace-level
  dependency additions are coordinated by the main session before
  fan-out.
- When all streams complete, the main session reviews each worktree
  diff, then merges them into `main` sequentially. The orchestrator
  integration step is the last serial step.

If a phase has only one work-stream, dispatch on the main checkout — no
worktree needed.

## Dispatch prompt templates

The templates below are the *prompt body*; combine each with the agent's
own brief (which agent type to invoke, scope, deliverables).

### Template — production-crate development

```
You are implementing one crate of the meeting-app project: <CRATE_NAME>.

Required reading before any edits:
1. architecture/README.md
2. architecture/components.md — specifically the <CRATE_NAME> section
   and the dependency table
3. architecture/domain-ownership.md — specifically the <ROLE> section
   and the parallel-work rules
4. architecture/cross-cutting.md — binding rules (tokio threading,
   thiserror per-crate + AppError at boundaries, no anyhow in public
   signatures, bounded channels, no println!, tracing target = crate name)
5. crates/common/src/lib.rs — the locked interface contracts

Scope:
- You may edit files under crates/<CRATE_NAME>/** only.
- You may NOT edit:
  - crates/common/** (architecture-owner only)
  - Any other crate
  - The workspace Cargo.toml
- You MUST also add a 1-line change to architecture/components.md
  (the <CRATE_NAME> entry or its dependency-table row) to satisfy the
  pre-commit hook. If your work makes the existing description
  inaccurate, fix it; otherwise add a short clarifying note.

Deliverable:
- <DELIVERABLE — phase-specific, e.g. "audio-capture exposes
  AudioFrame stream with cpal-backed device-selection helper">
- Tests in crates/<CRATE_NAME>/src or crates/<CRATE_NAME>/tests
  covering the public surface.
- All cross-cutting rules satisfied.
- cargo build -p <CRATE_PACKAGE_NAME> succeeds.
- cargo test -p <CRATE_PACKAGE_NAME> passes.

Report:
- Summary of the public surface added.
- Any cross-crate interaction you needed but couldn't make (flagging
  this is fine; do not work around the dependency table).
- Any case where the locked interface in common didn't fit — these
  are architecture issues, escalate rather than work around.
```

### Template — webview component development

```
You are implementing UI for the meeting-app project: <COMPONENT>.

Required reading:
1. architecture/README.md
2. architecture/components.md — Webview components section
3. architecture/cross-cutting.md — IPC contract section

Scope:
- You may edit files under ui/src/<COMPONENT>/** only.
- You may NOT hand-edit ui/src/ipc/bindings.ts — it's generated.
- You MAY add new imports from ui/src/ipc/bindings.ts.
- You MUST also touch architecture/components.md (the Webview section).

Deliverable:
- <DELIVERABLE — phase-specific>
- Component tests where applicable.
- bun run build succeeds.

Report:
- New React components and their props.
- IPC command/event additions needed but not yet in bindings.ts
  (flag as a backend-side dependency; do not stub them locally).
```

### Template — SQE pass on a developer-completed crate

```
The crate crates/<CRATE_NAME> was just developed by another agent
against the locked common interfaces.

Required reading:
1. architecture/components.md — <CRATE_NAME> section
2. architecture/cross-cutting.md — Testing section
3. The dev report from the previous agent

Goal:
- Write or extend tests in crates/<CRATE_NAME> that exercise the
  public surface against realistic inputs.
- Tests must be "behaviour tests" — exercising what the code does,
  not what it is. No mirror-impl tests that just restate the code.

Constraints:
- Tests live with the crate (src/ #[cfg(test)] or tests/ for
  integration tests).
- Use real implementations where they're cheap (e.g., a 1 second
  recorded WAV). Mock only at external boundaries (network, slow
  inference).
- All cross-cutting rules apply to test code too, except for the
  unbounded-channels rule (tests can use std channels freely).

Report:
- New tests added and what they cover.
- Coverage gaps you saw but didn't address (these are findings for
  the developer agent, not bugs in your work).
```

## Fan-out planning

The phase planner is required to identify work-streams.
The output format is:

```
Phase N — work streams:

  Stream A — <role> — <crate(s)>:
    deliverable: ...
    depends on: <locked common types / traits>
    blocks: <which streams join after this>

  Stream B — <role> — <crate(s)>:
    deliverable: ...
    depends on: ...
    blocks: ...

  Join: <role> — <crate>:
    deliverable: wire streams together; integration tests; commit.
    depends on: A, B
```

The main session uses this to issue concurrent Agent() calls, one per
stream, in a single message.

## What stays serial

Some work is irreducibly sequential and must not be fanned out:

- Architecture-owner changes to `crates/common` or `architecture/`.
- Workspace `Cargo.toml` edits (deps, members, profile).
- Orchestrator wiring of independently-developed components into the
  live pipeline.
- Final phase commit + reviewer + PO sign-off.
- Anything that touches the IPC bindings generation step.

## Anti-patterns to refuse

If an agent's report contains any of these, the main session must
unwind the work rather than accept it:

- A new `pub` item in `crates/common` that wasn't in the locked
  interfaces.
- Workspace `Cargo.toml` edits.
- Imports from another crate not in the agent's allowed-deps list.
- `tauri::*` imports outside `ipc-bridge` or `app-main`.
- Filesystem writes outside the agent's owned scope.
- A "small refactor" in another crate to make the agent's work easier.
- `SKIP_ARCH_CHECK=1` in a commit message — agents do not bypass the
  drift guard.

## When to update this doc

Update when:

- A new subagent type becomes available in the workflow.
- The role-to-subagent mapping changes.
- A dispatch template needs to change because of a recurring failure
  pattern.
- The isolation model changes (we move from worktrees to something
  else, or vice versa).
