#!/usr/bin/env bash
# Program 01 benchmark smoke test: assemble WarpLife, pass the CPU/GPU
# semantic gate, and exercise all CPU/GPU interpreter/native timing paths.
#
# usage: warplife_bench.sh <warpvm-as> <warpvm> <warplife.wva> <scratch-prefix>
set -euo pipefail

AS="$1"
RUNNER="$2"
SRC="$3"
PREFIX="$4"

WVM="$PREFIX.wvm"
"$AS" "$SRC" -o "$WVM"
"$RUNNER" life_bench "$WVM" --vms 1 --ms 200
