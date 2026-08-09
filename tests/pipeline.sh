#!/usr/bin/env bash
# End-to-end pipeline test: .wva -> warpvm-as -> .wvm -> warpvm run -> check.
#
# usage: pipeline.sh <warpvm-as> <warpvm> <input.wva> <expected-substring> <scratch.wvm>
set -euo pipefail

AS="$1"
RUNNER="$2"
SRC="$3"
EXPECT="$4"
OUT="$5"

"$AS" "$SRC" -o "$OUT"
"$RUNNER" run "$OUT" | tee "$OUT.log"

grep -q "status=HALTED" "$OUT.log"
grep -q "$EXPECT" "$OUT.log"
echo "pipeline: PASS ($(basename "$SRC"))"
