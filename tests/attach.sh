#!/usr/bin/env bash
# Slice 6 end-to-end: assemble a resident program, attach, pause, step, and
# inspect registers/memory/disasm over the control plane.
#
# usage: attach.sh <warpvm-as> <warpvm> <input.wva> <scratch-prefix>
set -euo pipefail

AS="$1"
RUNNER="$2"
SRC="$3"
PREFIX="$4"

WVM="$PREFIX.wvm"
OUT="$PREFIX.attach.log"

"$AS" "$SRC" -o "$WVM"

printf 'list\npause\nstatus\nstep\nstep 2\npc\nregs\nmem 0 4\ndisasm 0 7\nquit\n' \
  | "$RUNNER" attach "$WVM" --vms 4 > "$OUT"

grep -q "RUNNING" "$OUT"                 # VMs booted
grep -q "paused" "$OUT"                  # pause reached the control point
grep -q "stepped -> status=PAUSED" "$OUT" # single-step executed
grep -q "PAUSED fault=OK" "$OUT"         # status inspection
grep -q "(uniform)" "$OUT"               # register dump
grep -q "shutdown: ok" "$OUT"            # clean teardown

echo "attach: PASS ($(basename "$SRC"))"
