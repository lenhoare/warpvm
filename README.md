# WarpVM

An experimental GPU-native virtual-computer architecture built around NVIDIA
CUDA warps: each virtual machine is a persistent, independently addressable
32-lane computer with its own state, memory, program, and mailbox.

See [project_spec.md](project_spec.md) for the full vision and
[docs/isa.md](docs/isa.md) for the v0.1 instruction set contract.

## Layout

```text
runtime/    C++/CUDA — kernel, interpreter, host runtime, CLI
tools/      Rust — assembler (`warpvm-as`) and disassembler (`warpvm-dis`)
programs/   .wva example programs
docs/       architecture + ISA documentation
tests/      host-side tests
benchmarks/ performance experiments
```

## Build

Requires a CUDA toolkit (nvcc) and CMake ≥ 3.24.

The CUDA target architecture is build configuration, not a project default.
Pass `-DCMAKE_CUDA_ARCHITECTURES=<arch>` or set the `CUDAARCHS` environment
variable; left unset, the toolkit's own default is used (sm_75 on CUDA 13.2),
which may not match your GPU. The configured list must include a native
cubin for the GPU you run on.

```sh
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=86   # e.g. RTX 3060
cmake --build build -j
ctest --test-dir build --output-on-failure
```

## Language tools

The Rust assembler/disassembler build as part of `cmake --build` and land in
`build/tools-rust/release/`:

```sh
build/tools-rust/release/warpvm-as programs/hello.wva -o hello.wvm
build/tools-rust/release/warpvm-dis hello.wvm
build/runtime/warpvm run hello.wvm
```

## Persistent machines (slice 5)

`run` executes a program to completion. To keep VMs **resident** — boot them,
watch them run, pause/resume/reset them from the host — use `serve`, which
launches the persistent kernel and prints a live `list`:

```sh
build/tools-rust/release/warpvm-as programs/heartbeat.wva -o heartbeat.wvm
build/runtime/warpvm serve heartbeat.wvm --vms 12 --for 5
```

`programs/heartbeat.wva` never halts (it is a small event loop), so it stays
`RUNNING` and its instruction counter advances until you shut it down. The
control plane (host↔GPU commands + status + log) is documented in
[docs/architecture.md](docs/architecture.md).

## Attach and single-step (slice 6)

`attach` boots resident VMs and drops into an interactive console to inspect
and drive one of them — pause, single-step, dump registers/memory, disassemble:

```sh
build/runtime/warpvm attach heartbeat.wvm --vms 4
vm-0> pause
vm-0> step          # retire exactly one instruction
vm-0> regs          # r0..r15 (live, from the paused VM)
vm-0> mem 0 8       # VM RAM
vm-0> disasm        # program around the pc, current line marked
vm-0> resume
vm-0> quit
```

## Slice status

| Slice | Content | Status |
|---|---|---|
| 0 | repo skeleton, ISA contract (`docs/isa.md`) | done |
| 1 | one warp, lane-wise arithmetic + reduction to host | done |
| 2 | interpreter (MOV/ADD/HALT), Rust assembler/disassembler | done |
| 3 | many VMs, stable IDs, private RAM | done |
| 4 | warp-native ops, control flow, stride-32 loops | done |
| 5 | persistent kernel control plane, `serve` + live `list`, log | done |
| 6 | `warpvm attach`, live inspection, single-step | done |
| 7 | messaging | — |
