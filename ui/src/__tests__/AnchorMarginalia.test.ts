/**
 * Unit tests for `formatAnchorMark` — the notes-gutter timestamp formatter.
 *
 * Mirrors the boundary-case coverage the sibling `formatTimestamp` has in
 * `TranscriptPane.test.tsx`. `formatAnchorMark` deliberately differs: no
 * centiseconds, unpadded leading minute, and an `H:MM:SS` rollover past an hour
 * (so a long meeting's stamp stays inside the narrow gutter). The negative
 * clamp guards against a bad/early anchor value.
 */
import { describe, it, expect } from "vitest";
import { formatAnchorMark } from "../editor/anchor-marginalia";

describe("formatAnchorMark", () => {
  it("formats sub-minute offsets as M:SS", () => {
    expect(formatAnchorMark(0)).toBe("0:00");
    expect(formatAnchorMark(5_000)).toBe("0:05");
    expect(formatAnchorMark(59_999)).toBe("0:59");
  });

  it("formats sub-hour offsets with the full (unpadded) minute count", () => {
    expect(formatAnchorMark(75_400)).toBe("1:15");
    expect(formatAnchorMark(600_000)).toBe("10:00");
    expect(formatAnchorMark(59 * 60_000 + 59_000)).toBe("59:59");
  });

  it("rolls into H:MM:SS past one hour (padding minutes/seconds)", () => {
    expect(formatAnchorMark(3_600_000)).toBe("1:00:00");
    expect(formatAnchorMark(3_723_456)).toBe("1:02:03");
    expect(formatAnchorMark(10 * 3_600_000)).toBe("10:00:00");
  });

  it("clamps a negative offset to zero", () => {
    expect(formatAnchorMark(-5_000)).toBe("0:00");
  });
});
