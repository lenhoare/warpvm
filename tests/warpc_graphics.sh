#!/usr/bin/env bash
# Slice F: compile graphics C, compare both backends, then inspect the frame.
set -euo pipefail

WARPC="$1"
RUNNER="$2"
SRC="$3"
PREFIX="$4"

WVM="$PREFIX.wvm"
INTERPRETED_LOG="$PREFIX.interpreted.log"
COMPILED_LOG="$PREFIX.compiled.log"
FRAME_LOG="$PREFIX.frame.log"

"$WARPC" "$SRC" -o "$WVM"
"$RUNNER" run "$WVM" | tee "$INTERPRETED_LOG"
grep -q "status=HALTED fault=OK" "$INTERPRETED_LOG"
grep -q "r0  = 42" "$INTERPRETED_LOG"

"$RUNNER" compiled_run "$WVM" | tee "$COMPILED_LOG"
grep -q "state=PASS memory=PASS framebuffer=PASS frame_seq=PASS r0=42" \
    "$COMPILED_LOG"
grep -q "compiled halt equivalence: PASS" "$COMPILED_LOG"

{
    sleep 3
    printf 'status\nframe\npixel 0 0\npixel 16 0\npixel 32 127\npixel 48 64\npixel 127 127\nquit\n'
} | "$RUNNER" attach "$WVM" --vms 1 | tee "$FRAME_LOG"

grep -q "status=HALTED fault=OK" "$FRAME_LOG"
grep -q "resolution=128x128" "$FRAME_LOG"
grep -q "format=ARGB8888" "$FRAME_LOG"
grep -q "frame_seq=1" "$FRAME_LOG"
grep -q "pixel(0,0) = 0xffff0000" "$FRAME_LOG"
grep -q "pixel(16,0) = 0xff00ff00" "$FRAME_LOG"
grep -q "pixel(32,127) = 0xff0000ff" "$FRAME_LOG"
grep -q "pixel(48,64) = 0xffffffff" "$FRAME_LOG"
grep -q "pixel(127,127) = 0xffffffff" "$FRAME_LOG"

echo "warpc_graphics PASS"
