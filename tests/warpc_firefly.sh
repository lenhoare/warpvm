#!/usr/bin/env bash
# Run the Warp C fireflies as 64 real resident VMs and prove mailbox activity.
set -euo pipefail

WARPC="$1"
RUNNER="$2"
SRC="$3"
PREFIX="$4"

WVM="$PREFIX.wvm"
LOG="$PREFIX.log"

"$WARPC" "$SRC" -o "$WVM"
{
    sleep 4
    printf 'list\nvm 0\nframe\nmem 0 10\nquit\n'
} | "$RUNNER" attach "$WVM" --vms 64 | tee "$LOG"

RUNNING_COUNT="$(grep -c 'RUNNING   OK' "$LOG")"
test "$RUNNING_COUNT" -eq 64
grep -Eq 'frame_seq=[1-9][0-9]*' "$LOG"

# Global word 7 is messages_received. In the `mem 0 10` row it is the eighth
# numeric value, or awk field 11 after the interactive prompt/address label.
RECEIVED="$(awk '/\[    0\]:/ { print $11; exit }' "$LOG")"
test -n "$RECEIVED"
test "$RECEIVED" -gt 0

echo "warpc_firefly PASS (64 VMs, received=$RECEIVED)"
echo "warpc_warp_messaging PASS"
