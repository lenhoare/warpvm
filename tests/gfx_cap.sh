#!/usr/bin/env bash
# v0.1.1 graphics capstone: assemble graphics.wva, boot 64 resident VMs, and
# verify every VM renders a correct, VMID-distinct 128x128 image (host copy),
# while all remain independently running.
#
# usage: gfx_cap.sh <warpvm-as> <warpvm> <graphics.wva> <scratch-prefix>
set -euo pipefail

AS="$1"
RUNNER="$2"
SRC="$3"
PREFIX="$4"

WVM="$PREFIX.wvm"
"$AS" "$SRC" -o "$WVM"
"$RUNNER" gfx_cap "$WVM" --vms 64
