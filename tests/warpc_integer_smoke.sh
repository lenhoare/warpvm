#!/usr/bin/env bash
# Compile the Slice A integer program, then check both execution backends.
set -euo pipefail

WARPC="$1"
RUNNER="$2"
SRC="$3"
PREFIX="$4"

WVM="$PREFIX.wvm"
INTERPRETED_LOG="$PREFIX.interpreted.log"
COMPILED_LOG="$PREFIX.compiled.log"

"$WARPC" "$SRC" -o "$WVM"
"$RUNNER" run "$WVM" | tee "$INTERPRETED_LOG"
grep -q "status=HALTED fault=OK" "$INTERPRETED_LOG"
grep -q "r0  = 42" "$INTERPRETED_LOG"

"$RUNNER" compiled_run "$WVM" | tee "$COMPILED_LOG"
grep -q "state=PASS memory=PASS framebuffer=PASS frame_seq=PASS r0=42" \
    "$COMPILED_LOG"
grep -q "compiled halt equivalence: PASS" "$COMPILED_LOG"

echo "warpc_integer_smoke PASS"
