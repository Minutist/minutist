/**
 * Voiceprint management pane.
 *
 * Lists every enrolled speaker identity with per-condition gallery summaries,
 * and provides rename, delete, clear-all, and "these are the same person"
 * merge affordances.
 *
 * Presented as a section inside the SettingsDrawer; rendered only when the
 * voiceprint enrolment setting is on (the pane is useless without data).
 *
 * Merge flow (§2.9.4): the user picks two identities via checkboxes, presses
 * "Merge selected", chooses which name survives, and confirms. The UI calls
 * `rename_voiceprint_identity` to set the surviving name first, then
 * `merge_voiceprint_identities` to fold. The operation is destructive once
 * cap-and-merge collapses contributions; the confirmation dialog warns the
 * user.
 *
 * Embedding bytes never cross the IPC boundary (§2.2); only the CentroidInfo
 * metadata (sample_count, condition_label) travels to the UI.
 */
import { useState, useEffect } from "react";
import { useVoiceprintsStore } from "../state/voiceprints";
import "./VoiceprintPane.css";

type MergeState =
  | { kind: "idle" }
  | { kind: "picking" }
  | { kind: "confirming"; keepId: string; mergedId: string; keepName: string; mergedName: string }
  | { kind: "saving" };

/** Voiceprint management section, rendered inside the SettingsDrawer. */
export function VoiceprintPane() {
  const identities = useVoiceprintsStore((s) => s.identities);
  const loading = useVoiceprintsStore((s) => s.loading);
  const error = useVoiceprintsStore((s) => s.lastError);
  const refresh = useVoiceprintsStore((s) => s.refresh);
  const renameIdentity = useVoiceprintsStore((s) => s.rename);
  const removeIdentity = useVoiceprintsStore((s) => s.remove);
  const mergeIdentities = useVoiceprintsStore((s) => s.merge);
  const clearAllIdentities = useVoiceprintsStore((s) => s.clearAll);

  // Rename: the identity currently being renamed and the draft name.
  const [renameId, setRenameId] = useState<string | null>(null);
  const [renameDraft, setRenameDraft] = useState("");

  // Merge flow state.
  const [merge, setMerge] = useState<MergeState>({ kind: "idle" });
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());

  // Surviving name when the user is confirming a merge.
  const [survivingName, setSurvivingName] = useState("");

  useEffect(() => {
    void refresh();
  }, [refresh]);

  // -------------------------------------------------------------------------
  // Rename handlers
  // -------------------------------------------------------------------------

  function beginRename(id: string, currentName: string) {
    setRenameId(id);
    setRenameDraft(currentName);
  }

  async function commitRename() {
    if (!renameId) return;
    const trimmed = renameDraft.trim();
    if (!trimmed) {
      setRenameId(null);
      return;
    }
    try {
      await renameIdentity(renameId, trimmed);
    } finally {
      setRenameId(null);
    }
  }

  // -------------------------------------------------------------------------
  // Delete handlers
  // -------------------------------------------------------------------------

  async function deleteIdentity(id: string) {
    if (!confirm("Delete this speaker voiceprint? This cannot be undone.")) return;
    await removeIdentity(id);
  }

  async function clearAll() {
    if (
      !confirm(
        "Delete all enrolled speaker voiceprints? This clears the entire local library and cannot be undone.",
      )
    )
      return;
    await clearAllIdentities();
  }

  // -------------------------------------------------------------------------
  // Merge flow
  // -------------------------------------------------------------------------

  function toggleSelected(id: string) {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else if (next.size < 2) {
        next.add(id);
      }
      return next;
    });
  }

  function beginMergePicking() {
    setSelectedIds(new Set());
    setMerge({ kind: "picking" });
  }

  function beginMergeConfirm() {
    const [a, b] = [...selectedIds];
    if (!a || !b) return;
    const nameA = identities.find((i) => i.identityId === a)?.displayName ?? a;
    const nameB = identities.find((i) => i.identityId === b)?.displayName ?? b;
    setSurvivingName(nameA);
    setMerge({ kind: "confirming", keepId: a, mergedId: b, keepName: nameA, mergedName: nameB });
  }

  function cancelMerge() {
    setSelectedIds(new Set());
    setMerge({ kind: "idle" });
  }

  async function commitMerge() {
    if (merge.kind !== "confirming") return;
    const { keepId, mergedId } = merge;
    setMerge({ kind: "saving" });
    try {
      // Rename keep_id to the chosen surviving name first (if it differs).
      const currentKeepName = identities.find((i) => i.identityId === keepId)?.displayName ?? "";
      if (survivingName.trim() && survivingName.trim() !== currentKeepName) {
        await renameIdentity(keepId, survivingName.trim());
      }
      // merge() reloads the list to get the post-merge gallery state.
      await mergeIdentities(keepId, mergedId);
    } finally {
      setSelectedIds(new Set());
      setMerge({ kind: "idle" });
    }
  }

  // -------------------------------------------------------------------------
  // Render
  // -------------------------------------------------------------------------

  const isPicking = merge.kind === "picking";
  const isConfirming = merge.kind === "confirming";
  const isSaving = merge.kind === "saving";

  return (
    <section className="settings-drawer__group vp-pane" aria-label="Speaker voiceprints">
      <h3 className="settings-drawer__group-title">Speaker voiceprints</h3>

      {error && <p className="vp-pane__error">{error}</p>}

      {loading && <p className="vp-pane__hint">Loading…</p>}

      {!loading && identities.length === 0 && (
        <p className="vp-pane__hint">
          No enrolled speakers yet. Rename a speaker in a transcript to enrol them (requires
          &ldquo;Learn speaker voices&rdquo; to be on).
        </p>
      )}

      {identities.length > 0 && (
        <>
          <ul className="vp-pane__list">
            {identities.map((identity) => {
              const isRenaming = renameId === identity.identityId;
              const isChecked = selectedIds.has(identity.identityId);
              const totalSamples = identity.centroids.reduce((s, c) => s + c.sampleCount, 0);

              return (
                <li key={identity.identityId} className="vp-pane__identity">
                  {isPicking && (
                    <input
                      type="checkbox"
                      className="vp-pane__checkbox"
                      checked={isChecked}
                      disabled={!isChecked && selectedIds.size >= 2}
                      onChange={() => toggleSelected(identity.identityId)}
                      aria-label={`Select ${identity.displayName} for merge`}
                    />
                  )}

                  <div className="vp-pane__identity-body">
                    {isRenaming ? (
                      <form
                        className="vp-pane__rename-form"
                        onSubmit={(e) => {
                          e.preventDefault();
                          void commitRename();
                        }}
                      >
                        <input
                          className="vp-pane__rename-input"
                          type="text"
                          value={renameDraft}
                          autoFocus
                          onChange={(e) => setRenameDraft(e.target.value)}
                          onBlur={() => void commitRename()}
                          onKeyDown={(e) => e.key === "Escape" && setRenameId(null)}
                          aria-label="New name"
                        />
                        <button type="submit" className="vp-pane__btn vp-pane__btn--primary">
                          Save
                        </button>
                        <button
                          type="button"
                          className="vp-pane__btn"
                          onClick={() => setRenameId(null)}
                        >
                          Cancel
                        </button>
                      </form>
                    ) : (
                      <span className="vp-pane__name">{identity.displayName}</span>
                    )}

                    <span className="vp-pane__meta">
                      {identity.centroids.length === 1
                        ? `${totalSamples} sample${totalSamples !== 1 ? "s" : ""}`
                        : `${identity.centroids.length} conditions, ${totalSamples} samples`}
                    </span>
                  </div>

                  {!isPicking && !isRenaming && (
                    <div className="vp-pane__actions">
                      <button
                        type="button"
                        className="vp-pane__btn"
                        onClick={() => beginRename(identity.identityId, identity.displayName)}
                      >
                        Rename
                      </button>
                      <button
                        type="button"
                        className="vp-pane__btn vp-pane__btn--danger"
                        onClick={() => void deleteIdentity(identity.identityId)}
                      >
                        Delete
                      </button>
                    </div>
                  )}
                </li>
              );
            })}
          </ul>

          <div className="vp-pane__toolbar">
            {merge.kind === "idle" && (
              <>
                <button type="button" className="vp-pane__btn" onClick={beginMergePicking}>
                  Merge two people
                </button>
                <button
                  type="button"
                  className="vp-pane__btn vp-pane__btn--danger"
                  onClick={() => void clearAll()}
                >
                  Clear all
                </button>
              </>
            )}

            {isPicking && (
              <>
                <span className="vp-pane__hint">
                  {selectedIds.size < 2
                    ? `Select two speakers to merge (${selectedIds.size}/2 selected)`
                    : "Ready to merge"}
                </span>
                <button
                  type="button"
                  className="vp-pane__btn vp-pane__btn--primary"
                  disabled={selectedIds.size !== 2}
                  onClick={beginMergeConfirm}
                >
                  Merge selected
                </button>
                <button type="button" className="vp-pane__btn" onClick={cancelMerge}>
                  Cancel
                </button>
              </>
            )}
          </div>

          {isConfirming && merge.kind === "confirming" && (
            <div className="vp-pane__confirm-box" role="alertdialog" aria-modal="false">
              <p className="vp-pane__confirm-title">
                Merge <strong>{merge.keepName}</strong> and <strong>{merge.mergedName}</strong>?
              </p>
              <p className="vp-pane__hint">
                This combines both voiceprint galleries. Once merged, the operation cannot be
                reversed. Choose which name to keep:
              </p>
              <div className="vp-pane__confirm-names">
                <label className="vp-pane__radio-label">
                  <input
                    type="radio"
                    name="surviving-name"
                    value={merge.keepName}
                    checked={survivingName === merge.keepName}
                    onChange={() => setSurvivingName(merge.keepName)}
                  />
                  {merge.keepName}
                </label>
                <label className="vp-pane__radio-label">
                  <input
                    type="radio"
                    name="surviving-name"
                    value={merge.mergedName}
                    checked={survivingName === merge.mergedName}
                    onChange={() => setSurvivingName(merge.mergedName)}
                  />
                  {merge.mergedName}
                </label>
              </div>
              <div className="vp-pane__confirm-btns">
                <button
                  type="button"
                  className="vp-pane__btn vp-pane__btn--primary"
                  disabled={isSaving}
                  onClick={() => void commitMerge()}
                >
                  {isSaving ? "Merging…" : "Confirm merge"}
                </button>
                <button
                  type="button"
                  className="vp-pane__btn"
                  disabled={isSaving}
                  onClick={cancelMerge}
                >
                  Cancel
                </button>
              </div>
            </div>
          )}
        </>
      )}
    </section>
  );
}
