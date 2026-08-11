#!/usr/bin/env bash
# Program 01 end-to-end test: assemble WarpLife, run its deterministic blinker
# and toroidal still-life VMs, and verify packed RAM + framebuffer output.
#
# usage: warplife.sh <warpvm-as> <warpvm> <warplife.wva> <scratch-prefix>
set -euo pipefail

AS="$1"
RUNNER="$2"
SRC="$3"
PREFIX="$4"

WVM="$PREFIX.wvm"
"$AS" "$SRC" -o "$WVM"
"$RUNNER" life_test "$WVM"
