#!/usr/bin/env bash
# Assemble canonical WarpLife bytecode, then verify the generated PTX backend
# at full architectural checkpoints and across both mode-transition directions.
set -euo pipefail

AS="$1"
RUNNER="$2"
SRC="$3"
PREFIX="$4"

WVM="$PREFIX.wvm"
"$AS" "$SRC" -o "$WVM"
"$RUNNER" compiled_life "$WVM"
