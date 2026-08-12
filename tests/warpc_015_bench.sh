#!/usr/bin/env bash
set -euo pipefail

WARPC="$1"
RUNNER="$2"
SEQUENTIAL_SRC="$3"
WARP_SRC="$4"
PREFIX="$5"

"$WARPC" "$SEQUENTIAL_SRC" -o "$PREFIX.sequential.wvm"
"$WARPC" "$WARP_SRC" -o "$PREFIX.warp.wvm"
"$RUNNER" warpc_bench "$PREFIX.sequential.wvm" "$PREFIX.warp.wvm" | \
    tee "$PREFIX.log"
grep -q "warpc_015_bench PASS" "$PREFIX.log"
