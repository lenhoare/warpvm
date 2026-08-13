#!/usr/bin/env bash
# Build and display four visibly different Warp C programs in one resident,
# interpreted WarpVM population.
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
build_dir=${WARPVM_BUILD_DIR:-"${repo_root}/build"}
warpc=${build_dir}/tools-rust/release/warpc
warpvm=${build_dir}/runtime/warpvm
output_dir=${build_dir}/heterogeneous_demo

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

exec "${warpvm}" hetero_view \
  "${output_dir}/plasma.wvm" \
  "${output_dir}/mandelbrot.wvm" \
  "${output_dir}/wave.wvm" \
  "${output_dir}/sandpile.wvm"
