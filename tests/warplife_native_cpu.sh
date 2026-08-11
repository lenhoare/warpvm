#!/usr/bin/env bash
# Assemble WarpLife and compare the complete packed world and ARGB framebuffer
# against the independent handwritten byte-per-cell native CPU implementation.
set -euo pipefail

AS="$1"
RUNNER="$2"
SRC="$3"
PREFIX="$4"

WVM="$PREFIX.wvm"
"$AS" "$SRC" -o "$WVM"
"$RUNNER" life_native_cpu_equiv "$WVM"
