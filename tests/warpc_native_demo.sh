#!/usr/bin/env bash
# Exercise the v0.1.5 live capstone as 64 persistent, communicating VMs.
set -euo pipefail

WARPC="$1"
RUNNER="$2"
SRC="$3"
PREFIX="$4"

WVM="$PREFIX.wvm"
LOG="$PREFIX.log"

"$WARPC" "$SRC" -o "$WVM"
{
    sleep 5
    printf 'list\nvm 0\nframe\nmem 96 7\nquit\n'
} | "$RUNNER" attach "$WVM" --vms 64 | tee "$LOG"

RUNNING_COUNT="$(grep -c 'RUNNING   OK' "$LOG")"
test "$RUNNING_COUNT" -eq 64
grep -Eq 'frame_seq=[1-9][0-9]*' "$LOG"

# Globals 96..102 are frame, flash, received count, last payload/metadata,
# current reduction, and ballot. Read the third value independently of the
# interactive prompt prefix.
RECEIVED="$(awk '/\[ *96\]:/ { for (i = 1; i <= NF; ++i) if ($i ~ /^96\]:$/) { print $(i + 3); exit } }' "$LOG")"
test -n "$RECEIVED"
test "$RECEIVED" -gt 0

echo "warpc_native_demo PASS (64 VMs, received=$RECEIVED)"
