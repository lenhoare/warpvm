#!/usr/bin/env bash
# v0.1.5 Slice A/B: WARP is lane ID, divergent, and has no backing object.
set -euo pipefail

WARPC="$1"
RUNNER="$2"
VALUES_SRC="$3"
NO_STORAGE_SRC="$4"
PREFIX="$5"

VALUES_WVM="$PREFIX.wvm"
NO_STORAGE_WVM="$PREFIX.no_storage.wvm"
ASM_LOG="$PREFIX.no_storage.wva"
FRAME_LOG="$PREFIX.memory.log"

"$WARPC" "$NO_STORAGE_SRC" -o "$NO_STORAGE_WVM" --emit-asm > "$ASM_LOG"
test "$(grep -c 'LANEID' "$ASM_LOG")" -eq 1
if grep -Eq '^[[:space:]]+(LOAD|STORE|LDW)[[:space:]]' "$ASM_LOG"; then
    echo "WARP unexpectedly used memory" >&2
    exit 1
fi

"$WARPC" "$VALUES_SRC" -o "$VALUES_WVM"
"$RUNNER" compiled_run "$VALUES_WVM" | tee "$PREFIX.compiled.log"
grep -q "state=PASS memory=PASS framebuffer=PASS frame_seq=PASS r0=42" \
    "$PREFIX.compiled.log"

{
    sleep 1
    printf 'status\nmem 0 64\nquit\n'
} | "$RUNNER" attach "$VALUES_WVM" --vms 1 | tee "$FRAME_LOG"
grep -q "status=HALTED fault=OK" "$FRAME_LOG"

VALUES="$(awk '
    /\[[[:space:]]*[0-9]+\]:/ {
        start = 1
        while (start <= NF && $start !~ /]:$/) ++start
        for (i = start + 1; i <= NF; ++i) {
            if (count < 32) {
                printf "%s%s", count ? " " : "", $i
                ++count
            }
        }
    }
    END { print "" }
' "$FRAME_LOG")"
EXPECTED="$(seq 0 31 | paste -sd ' ' -)"
test "$VALUES" = "$EXPECTED"

echo "warpc_warp_values PASS"
echo "warpc_warp_no_storage PASS"
echo "warpc_warp_uniformity PASS"
echo "warpc_warp_array PASS"
