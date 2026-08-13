#!/usr/bin/env bash
# Compile four graphics programs, create a heterogeneous population through
# the public supervisor command language, then leave an interactive prompt.
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
build_dir=${WARPVM_BUILD_DIR:-"${repo_root}/build"}
warpc=${build_dir}/tools-rust/release/warpc
warpvm=${build_dir}/runtime/warpvm
output_dir=${build_dir}/supervisor_demo
startup=${output_dir}/population.wvs
engine=${WARPVM_ENGINE:-compiled}

if [[ ! -x "${warpc}" || ! -x "${warpvm}" ]]; then
  echo "error: build WarpVM first with: cmake --build ${build_dir}" >&2
  exit 2
fi

mkdir -p "${output_dir}"
"${warpc}" "${repo_root}/testprojects/plasma/plasma.wc" \
  -o "${output_dir}/plasma.wvm"
"${warpc}" "${repo_root}/testprojects/mandelbrot/mandelbrot.wc" \
  -o "${output_dir}/mandelbrot.wvm"
"${warpc}" "${repo_root}/testprojects/wave/wave.wc" \
  -o "${output_dir}/wave.wvm"
"${warpc}" "${repo_root}/testprojects/sandpile/sandpile.wc" \
  -o "${output_dir}/sandpile.wvm"

printf '%s\n' \
  "# Programs are immutable registry objects; loading does not create a VM." \
  "program load plasma ${output_dir}/plasma.wvm" \
  "program load mandelbrot ${output_dir}/mandelbrot.wvm" \
  "program load wave ${output_dir}/wave.wvm" \
  "program load sandpile ${output_dir}/sandpile.wvm" \
  "launch 8 ${engine}" \
  "vm create plasma" \
  "vm create mandelbrot" \
  "vm create wave" \
  "vm create sandpile" \
  "vm start 0" \
  "vm start 1" \
  "vm start 2" \
  "vm start 3" \
  "wait 0 RUNNING 5000" \
  "wait 1 RUNNING 5000" \
  "wait 2 RUNNING 5000" \
  "wait 3 RUNNING 5000" \
  "list" > "${startup}"

echo "The ${engine} population will remain resident at the interactive prompt."
echo "Try: view, list, vm stop 1, vm reset 1, vm start 1"
echo "A hand-editable example is also checked in at programs/population.wvs."
exec "${warpvm}" supervise --script "${startup}" --interactive
