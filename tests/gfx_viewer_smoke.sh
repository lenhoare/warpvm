#!/usr/bin/env bash
# v0.1.1 end-to-end viewer smoke test (headless, SDL-free): assemble the
# graphics demo, run it on the resident GPU runtime, and verify a published
# frame can be copied to the host with expected sample pixel values.
#
# usage: gfx_viewer_smoke.sh <warpvm-as> <warpvm> <graphics.wva> <scratch-prefix>
set -euo pipefail

AS="$1"
RUNNER="$2"
SRC="$3"
PREFIX="$4"

WVM="$PREFIX.wvm"
"$AS" "$SRC" -o "$WVM"
"$RUNNER" gfxsmoke "$WVM"
