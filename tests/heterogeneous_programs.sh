#!/usr/bin/env bash
# Compile four real graphics programs and run them concurrently through the
# shared-program heterogeneous interpreted runtime.
set -euo pipefail

WARPC="$1"
RUNNER="$2"
ROOT="$3"
PREFIX="$4"

PLASMA="${PREFIX}.plasma.wvm"
MANDELBROT="${PREFIX}.mandelbrot.wvm"
WAVE="${PREFIX}.wave.wvm"
SANDPILE="${PREFIX}.sandpile.wvm"
LOG="${PREFIX}.log"

"${WARPC}" "${ROOT}/testprojects/plasma/plasma.wc" -o "${PLASMA}"
"${WARPC}" "${ROOT}/testprojects/mandelbrot/mandelbrot.wc" \
  -o "${MANDELBROT}"
"${WARPC}" "${ROOT}/testprojects/wave/wave.wc" -o "${WAVE}"
"${WARPC}" "${ROOT}/testprojects/sandpile/sandpile.wc" -o "${SANDPILE}"

"${RUNNER}" hetero_smoke \
  "${PLASMA}" "${MANDELBROT}" "${WAVE}" "${SANDPILE}" | tee "${LOG}"

grep -q "4 VMs, 4 registry programs, 4 device uploads" "${LOG}"
grep -q "all_publish_frames: PASS" "${LOG}"
grep -q "isolated_stop:      PASS" "${LOG}"
grep -q "heterogeneous smoke: PASS" "${LOG}"

echo "heterogeneous_programs PASS"
