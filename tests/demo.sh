#!/usr/bin/env bash
# v0.1 capstone: assemble demo.wva, boot 64 resident VMs, verify the 32-lane
# computation + ring messaging + live inspectability.
#
# usage: demo.sh <warpvm-as> <warpvm> <demo.wva> <scratch-prefix>
set -euo pipefail

AS="$1"
RUNNER="$2"
SRC="$3"
PREFIX="$4"

WVM="$PREFIX.wvm"
OUT="$PREFIX.demo.log"

"$AS" "$SRC" -o "$WVM"
"$RUNNER" demo "$WVM" --vms 64 > "$OUT"

grep -q "64/64 correct" "$OUT"
grep -q "ticking" "$OUT"
grep -q "demo: PASS" "$OUT"

echo "demo: PASS (64 VMs, ring messaging, live inspection)"
