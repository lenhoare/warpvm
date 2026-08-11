# WarpVM

An experimental GPU-native virtual-computer architecture built around NVIDIA
CUDA warps: each virtual machine is a persistent, independently addressable
32-lane computer with its own state, memory, program, and mailbox.

See [project_spec.md](project_spec.md) for the full vision and
[docs/isa.md](docs/isa.md) for the v0.1 instruction set contract.

## Layout

```text
runtime/    C++/CUDA — kernel, interpreter, host runtime, CLI
tools/      Rust — assembler, disassembler, and Warp C compiler (`warpc`)
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

The Rust assembler, disassembler, and Warp C compiler build as part of
`cmake --build` and land in `build/tools-rust/release/`:

```sh
build/tools-rust/release/warpvm-as programs/hello.wva -o hello.wvm
build/tools-rust/release/warpvm-dis hello.wvm
build/runtime/warpvm run hello.wvm
```

Warp C currently supports signed `int`, `unsigned`, word-sized unsigned
`char`, local variables, integer expressions, structured conditionals and
loops, `switch`, named functions, prototypes, up to four parameters,
assignment, pointers, fixed arrays, structs, word-per-character strings,
globals, `sizeof`, `return`, the `warp_lane_id()` / `warp_vm_id()` intrinsics,
and the built-in `<warp.h>` framebuffer API, including nested divergent
`if` / `else`. It emits inspectable
WarpVM assembly and assembles it in-process to canonical `.wvm`:

```sh
build/tools-rust/release/warpc programs/warpc/integer_smoke.wc \
  -o build/warpc_integer_smoke.wvm --emit-asm --dump-uniformity
build/runtime/warpvm run build/warpc_integer_smoke.wvm
build/runtime/warpvm compiled_run build/warpc_integer_smoke.wvm

build/tools-rust/release/warpc programs/warpc/hello_pixels.wc \
  -o build/hello_pixels.wvm
build/runtime/warpvm view build/hello_pixels.wvm --vm 0
```

The current subset and integer semantics are documented in
[`docs/warpc.md`](docs/warpc.md).

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

## Graphics viewer (v0.1.1)

The SDL viewer keeps the persistent kernel running while it presents either
one enlarged VM or a tiled set of resident VMs. The 64-VM demo uses an 8×8
grid. Sequence counters trigger a single batched framebuffer-pool transfer and
one atlas upload, avoiding per-VM copy and texture-update overhead:

```sh
build/tools-rust/release/warpvm-as programs/graphics.wva -o build/graphics.wvm
build/runtime/warpvm view build/graphics.wvm --vm 0
build/runtime/warpvm view build/graphics.wvm --vms 64
```

## Program 01: WarpLife

`programs/warplife.wva` is a complete 128×128 toroidal Conway's Life program
written directly in WarpVM assembly. Each VM keeps two bit-packed 512-word
worlds, evolves 32 cells per warp batch, renders through ordinary framebuffer
stores, and runs independently forever. Its frequently used nine-instruction
`load_cell` sequence is inlined: this grows static code from 153 to 215 words
but removes 9,216 interpreted `CALL`/`RET` bytecodes per generation.

VM 0 is a deterministic blinker, VM 1 is a toroidal four-corner still life,
and the remaining VMs use deterministic `VMID`-derived worlds:

```sh
build/tools-rust/release/warpvm-as programs/warplife.wva -o build/warplife.wvm
build/runtime/warpvm life_test build/warplife.wvm
build/runtime/warpvm view build/warplife.wvm --vms 64
build/runtime/warpvm life_equiv build/warplife.wvm
build/runtime/warpvm life_bench build/warplife.wvm --ms 2000 --workers 4
build/runtime/warpvm life_profile build/warplife.wvm --ms 1000
```

The v0.1.2 benchmark runs the unchanged `.wvm` through both the persistent
CUDA interpreter and a logical 32-lane CPU interpreter. It also measures
single- and multi-worker native CPU Life plus native CUDA. The default CPU
worker count is the host's reported hardware concurrency; `--workers`
overrides it. Every benchmark run first performs deterministic, full-world
CPU/GPU equivalence checks for VM IDs 0, 1, 2, and 37.

Concrete architectural issues discovered while writing applications are
recorded in [`notes.md`](notes.md). The RTX 3060 timing methodology and first
1/8/32/64/256-VM results are in
[`benchmarks/warplife.md`](benchmarks/warplife.md). The one-VM opcode, phase,
control-poll, and memory breakdown is in
[`benchmarks/warplife_profile.md`](benchmarks/warplife_profile.md).

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
| 7 | messaging (`SEND`/`TRY_RECV`, mailboxes) | done |
| ★ | **v0.1 demo**: 64 resident VMs compute + ring-message + stay live | done |
| gfx-A | framebuffer memory: memory-mapped `LOAD`/`STORE`, reset-clear, isolation | done |
| gfx-B | `FLIP` publication + `frame_seq`, predefined `VIDEO_*` symbols | done |
| gfx-C | SDL single-VM viewer, host-copy (`gfxsmoke`), attach `frame`/`pixel` | done |
| gfx-D | SDL multi-VM tiled viewer | done |
| gfx-★ | **v0.1.1 capstone**: 64 VMs render distinct animated 128×128 images | done |
| program-01 | **WarpLife**: packed toroidal Life, deterministic tests, 64-world grid | done |
| v0.1.2 | logical CPU interpreter, CPU/GPU equivalence, five-engine WarpLife benchmark | done |
| v0.1.3 | direct `.wvm` to PTX compiled execution and mixed-mode equivalence | done |
| v0.1.4-A | Warp C lexer/parser, typed integer expressions, direct `.wvm` output | done |
| v0.1.4-B | uniform structured control flow, loop jumps, and switch fall-through | done |
| v0.1.4-C | function prototypes/calls, parameters, returns, and lane-private stack ABI | done |
| v0.1.4-D | word-addressed pointers, arrays, structs, strings, globals, and automatic memory frames | done |
| v0.1.4-E | uniformity propagation, lane/VM intrinsics, and masked divergent `if` / `else` | done |
| v0.1.4-F | built-in `warp.h`, ARGB/framebuffer helpers, and `FLIP` | done |

## v0.1 milestone

The capstone boots 64 resident machines that each run a 32-lane computation
(`LANEID` + `REDUCE_ADD`), send the result to the next machine in a ring,
receive their neighbour's, and keep running so they remain inspectable:

```sh
build/tools-rust/release/warpvm-as programs/demo.wva -o demo.wvm
build/runtime/warpvm demo demo.wvm --vms 64
#   demo: ring exchange   PASS (64/64 correct)
#   demo: still RUNNING   PASS (64/64)
#   demo: vm 37 live       status=RUNNING instrs 0 -> 1025 (ticking)
#   demo: PASS
```

To poke at a live machine by hand: `warpvm attach demo.wvm --vms 64`, then
`vm 37`, `regs`, `mem 0 8`, `disasm`, `step`, `resume`.
