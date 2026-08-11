#!/usr/bin/env bash
# Assemble WarpLife and compare deterministic packed worlds produced by the
# CPU and GPU WarpVM interpreters for several logical VM IDs.
set -euo pipefail

AS="$1"
RUNNER="$2"
SRC="$3"
PREFIX="$4"

WVM="$PREFIX.wvm"
"$AS" "$SRC" -o "$WVM"
"$RUNNER" life_equiv "$WVM"
