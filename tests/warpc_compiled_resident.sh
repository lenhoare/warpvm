#!/usr/bin/env bash
# Prove that the v0.1.5 capstone is continuously resident in direct PTX while
# frames, messages, status, and explicit control remain live.
set -euo pipefail

WARPC="$1"
RUNNER="$2"
SRC="$3"
PREFIX="$4"

WVM="$PREFIX.wvm"
LOG="$PREFIX.log"

"$WARPC" "$SRC" -o "$WVM"
{
    sleep 2
    printf 'list\nvm 0\nframe\nmem 96 7\npause\nstatus\nresume\nquit\n'
} | "$RUNNER" attach "$WVM" --vms 64 --compiled | tee "$LOG"

test "$(grep -c 'RUNNING   OK' "$LOG")" -eq 64
grep -Eq 'frame_seq=[1-9][0-9]*' "$LOG"
grep -q 'status=PAUSED fault=OK' "$LOG"
grep -q '^vm-0> resuming$' "$LOG"
grep -q 'shutdown: ok' "$LOG"

RECEIVED="$(awk '/\[ *96\]:/ { for (i = 1; i <= NF; ++i) if ($i ~ /^96\]:$/) { print $(i + 3); exit } }' "$LOG")"
test -n "$RECEIVED"
test "$RECEIVED" -gt 0

echo "warpc_compiled_resident PASS (64 VMs, received=$RECEIVED)"
