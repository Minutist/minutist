#!/usr/bin/env bash
# Prove a change touched comments only.
#
# Two independent checks:
#   1. every added and removed line in the diff is blank or a `//` line;
#   2. with comment and blank lines stripped, each changed file is byte-identical
#      at both revisions.
#
# A trailing comment edited on a code line fails check 1, which is deliberate: the
# whole line moves, so it cannot be distinguished from a code edit and must be
# reviewed by hand.
#
# By default this compares the WORKING TREE against <baseline-ref>, so uncommitted
# work is covered. Pass --committed to compare HEAD instead. Comparing only HEAD
# silently ignores uncommitted edits, which is exactly when the check matters
# least, so the working tree is the default.
#
# Usage: verify-comment-only-change.sh [--committed] <baseline-ref> [path...]
set -uo pipefail

RANGE_END=""            # empty => working tree
if [ "${1:-}" = "--committed" ]; then
  RANGE_END="HEAD"
  shift
fi
BASE="${1:?usage: $0 [--committed] <baseline-ref> [path...]}"
shift
PATHS=("${@:-crates/sync}")

if [ -n "$RANGE_END" ]; then
  DIFF_ARGS=("$BASE".."$RANGE_END")
  echo "comparing $BASE..HEAD (committed only)"
else
  DIFF_ARGS=("$BASE")
  echo "comparing $BASE..working-tree"
fi

fail=0
offending=$(mktemp)
trap 'rm -f "$offending"' EXIT

git diff -U0 "${DIFF_ARGS[@]}" -- "${PATHS[@]}" \
  | grep -E '^[+-]' \
  | grep -vE '^(\+\+\+|---)' \
  | sed -E 's/^[+-]//' \
  | grep -vE '^[[:space:]]*(//|$)' > "$offending" || true

n=$(wc -l < "$offending")
if [ "$n" -gt 0 ]; then
  echo "FAIL: $n non-comment line(s) changed:"
  head -40 "$offending" | sed 's/^/    /'
  fail=1
else
  echo "PASS: every changed line is a comment or blank."
  echo "  $(git diff --shortstat "${DIFF_ARGS[@]}" -- "${PATHS[@]}")"
fi

strip() { grep -vE '^[[:space:]]*(///|//!|//)' | grep -vE '^[[:space:]]*$'; }

for f in $(git diff --name-only "${DIFF_ARGS[@]}" -- "${PATHS[@]}"); do
  [ -f "$f" ] || continue
  if [ -n "$RANGE_END" ]; then
    now=$(git show "$RANGE_END:$f" | strip)
  else
    now=$(strip < "$f")
  fi
  if ! diff -q <(git show "$BASE:$f" | strip) <(printf '%s\n' "$now") >/dev/null; then
    echo "FAIL: code differs in $f"
    fail=1
  fi
done
[ "$fail" -eq 0 ] && echo "PASS: code byte-identical in every changed file."
exit "$fail"
