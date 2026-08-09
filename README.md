# WarpVM

An experimental GPU-native virtual-computer architecture built around NVIDIA
CUDA warps: each virtual machine is a persistent, independently addressable
32-lane computer with its own state, memory, program, and mailbox.

See [project_spec.md](project_spec.md) for the full vision and
[docs/isa.md](docs/isa.md) for the v0.1 instruction set contract.

## Layout

```text
runtime/    C++/CUDA — kernel, interpreter, host runtime, CLI
tools/      Rust — assembler, disassembler (lands in slice 2)
programs/   .wva example programs
docs/       architecture + ISA documentation
tests/      host-side tests
benchmarks/ performance experiments
```

## Build

Requires CUDA toolkit (nvcc) and CMake ≥ 3.24. Default CUDA architecture is
sm_86 (RTX 3060); override with `-DCMAKE_CUDA_ARCHITECTURES=<cc>`.

```sh
cmake -S . -B build
cmake --build build -j
ctest --test-dir build --output-on-failure
```

## Slice status

| Slice | Content | Status |
|---|---|---|
| 0 | repo skeleton, ISA contract (`docs/isa.md`) | done |
| 1 | one warp, lane-wise arithmetic + reduction to host | done |
| 2 | interpreter (MOV/ADD/HALT), Rust assembler/disassembler | — |
| 3 | many VMs, stable IDs, private RAM | — |
| 4 | warp-native ops, stride-32 loops | — |
| 5 | persistent kernel control plane, `warpvm list` | — |
| 6 | `warpvm attach`, live inspection | — |
| 7 | messaging | — |
