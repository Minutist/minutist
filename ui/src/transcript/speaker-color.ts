/**
 * Speaker -> palette-slot mapping for live diarization (Phase C).
 *
 * A pure, stateless function of the diarizer's speaker_id. The colour is the
 * SAME in the live store and after the on-stop diarization relabel (which
 * re-reads the meeting through a separate code path), because the slot is
 * derived only from the id string — there is no per-component or per-store
 * registry to drift between the two paths.
 */

// Palette size: the number of --speaker-N vars defined in theme.css. A unit
// test asserts the two agree, so changing one without the other fails CI.
export const SPEAKER_PALETTE_SIZE = 8;

/**
 * Deterministic, stable mapping from a diarizer speaker_id to a 1-based
 * palette slot. Pure function of the id: the same id always yields the
 * same slot, in the live store and after the on-stop relabel.
 *
 * Single-letter labels "A".."Z" (the diarizer's first-seen alpha labels)
 * map by ordinal: A->1, B->2, ... Z->26, cycling the palette past its size.
 * Any other id shape (multi-char, numeric, unexpected) falls back to a
 * stable string hash so it still gets a fixed colour rather than throwing.
 */
export function speakerColorIndex(speakerId: string): number {
  let ordinal: number;
  if (/^[A-Z]$/.test(speakerId)) {
    ordinal = speakerId.charCodeAt(0) - 65; // 'A' -> 0
  } else {
    // Stable djb2-ish hash so non-alpha ids are deterministic too.
    let h = 0;
    for (let i = 0; i < speakerId.length; i++) {
      h = (h * 31 + speakerId.charCodeAt(i)) | 0;
    }
    ordinal = Math.abs(h);
  }
  return (ordinal % SPEAKER_PALETTE_SIZE) + 1; // 1-based: --speaker-1..N
}
