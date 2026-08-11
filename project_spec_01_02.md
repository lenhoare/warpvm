# WarpVM v0.1.2 — CPU Interpreter and Benchmark Baseline

## 1. Purpose

Add a CPU implementation of the existing WarpVM interpreter so that the GPU WarpVM can be benchmarked against the same virtual-machine architecture running on a conventional CPU.

This slice is **not** a CPU port of the whole WarpVM runtime and is **not** an attempt to build an optimized CPU virtual machine.

The goal is to answer one clean question:

> How much useful throughput does the GPU substrate provide when the exact same WarpVM bytecode and VM semantics are executed on the CPU?

The existing GPU implementation remains the reference implementation for VM semantics.

---

## 2. Scope

Implement a host-side WarpVM interpreter capable of loading and executing existing `.wvm` binaries.

It must support enough of the current ISA and runtime state to execute the existing WarpLife benchmark unchanged.

The slice should provide:

1. a CPU WarpVM interpreter,
2. deterministic execution of the same `.wvm` program used by the GPU benchmark,
3. a single-threaded multi-VM benchmark,
4. a parallel CPU benchmark using available host CPU cores,
5. output directly comparable with the existing WarpLife GPU benchmark,
6. correctness checks showing that CPU and GPU executions produce equivalent results.

Do not change WarpLife merely to make the CPU implementation easier.

---

## 3. Design Principle

The CPU interpreter should implement the **logical WarpVM machine**, not emulate CUDA implementation details.

A WarpVM instance still has:

- one VM ID,
- one program counter,
- 32 logical lanes,
- the existing vector and scalar register state,
- active-lane state/masks,
- VM-local memory,
- status/fault state,
- framebuffer state where used,
- the same bytecode semantics as the GPU interpreter.

On the CPU, the 32 lanes may simply be represented as ordinary arrays and evaluated with straightforward loops.

Correctness and comparability are more important than CPU-specific optimization.

---

## 4. Interpreter

Add a CPU interpreter alongside the CUDA interpreter rather than replacing or conditionally rewriting the existing GPU path.

A basic execution loop should conceptually be:

```text
while VM is RUNNING:
    fetch instruction at pc
    decode instruction
    execute instruction over the logical lanes
    update pc / VM state
    increment instruction counter
```

Use the existing bytecode definitions and instruction encodings directly. Avoid duplicating opcode constants or creating a second independently maintained ISA definition where practical.

### Requirements

- The same `.wvm` file must run on both CPU and GPU implementations.
- Instruction behaviour must match the GPU interpreter.
- Integer overflow, comparisons, masks, branches, loads/stores and lane semantics must be equivalent.
- Faults should be represented consistently with the GPU runtime where practical.
- No Life-specific instruction or hard-coded Life implementation may be added.

---

## 5. CPU Execution Modes

Implement two deliberately simple execution modes.

### 5.1 Single-threaded round-robin

Run all requested VMs on one host thread.

This mode exists to establish the cost of the WarpVM abstraction on a normal scalar CPU without parallel CPU execution obscuring the result.

A simple scheduler is sufficient. For example, execute a bounded instruction quantum for each runnable VM before moving to the next.

The quantum should be fixed and documented. Avoid tuning it separately for WarpLife.

### 5.2 Parallel host execution

Add a second mode that distributes VMs across host CPU worker threads.

Requirements:

- default worker count should correspond sensibly to available host hardware concurrency,
- allow the worker count to be overridden from the CLI,
- assign VMs stably to workers for the duration of a benchmark,
- avoid creating one OS thread per VM unless there is a compelling implementation reason,
- no work-stealing or sophisticated scheduler is required for this slice.

A fixed worker pool with static VM partitioning is preferred.

---

## 6. WarpLife Benchmark

Extend the existing WarpLife benchmark so that it can measure:

```text
GPU WarpVM
CPU WarpVM, 1 worker
CPU WarpVM, N workers
Native CPU reference
Native CUDA reference (existing)
```

The critical comparison for this slice is:

```text
CPU WarpVM vs GPU WarpVM
```

The native CPU result is also useful because it separates:

- VM/interpreter overhead,
- CPU versus GPU substrate differences,
- native-algorithm performance.

### Native CPU reference

Implement a conventional CPU version of the same Life workload:

- same deterministic initial states,
- 128×128 toroidal world,
- synchronous generations,
- same cell-update definition,
- same rendering work where rendering is included in the existing benchmark,
- no host display or SDL presentation during timing.

The native CPU reference need not be aggressively hand-vectorized. It should be a reasonable optimized Release-build implementation that a competent C++ compiler can optimize normally.

Document whether it is single-threaded or multi-threaded. Prefer providing both if trivial, but do not expand the slice substantially merely to do so.

---

## 7. Benchmark Matrix

At minimum run the same VM counts as the current baseline:

```text
1
8
32
64
256
```

Use the existing benchmark duration unless a longer duration is required to obtain meaningful progress measurements.

For 256 VMs, retain the longer supplementary sample if short samples produce zero-generation minima.

Record at least:

```text
VM count
worker count / execution mode
elapsed seconds
average generations/sec/VM
minimum generations/sec/VM
maximum generations/sec/VM
aggregate cell updates/sec
aggregate Mcell/sec
```

Also calculate useful ratios such as:

```text
GPU WarpVM / single-thread CPU WarpVM
GPU WarpVM / parallel CPU WarpVM
native CPU / CPU WarpVM
native CUDA / GPU WarpVM
```

The benchmark report should make the numerator and denominator explicit so that a ratio cannot be misread.

---

## 8. Correctness

Before trusting timing results, demonstrate that CPU and GPU WarpVM executions are equivalent.

For deterministic WarpLife seeds, run a fixed number of generations and compare at least one of:

- full world state,
- framebuffer contents,
- deterministic checksum/hash of the resulting world.

Prefer a checksum plus an optional full comparison on failure.

Test multiple VM IDs/seeds rather than only VM 0.

The benchmark must fail clearly if the CPU implementation produces a different result from the GPU interpreter.

---

## 9. CLI

Keep the interface small and consistent with the existing runtime tooling.

Exact syntax may follow the existing CLI structure, but the user should be able to request something equivalent to:

```bash
warpvm bench warplife --engine gpu
warpvm bench warplife --engine cpu --workers 1
warpvm bench warplife --engine cpu --workers 12
warpvm bench warplife --engine native-cpu
```

If the current CLI architecture makes a separate benchmark executable cleaner, that is acceptable. Do not perform a broad CLI refactor for this slice.

---

## 10. Performance Rules

This is a baseline implementation.

Do **not** add CPU-specific optimizations that materially alter the VM abstraction merely to improve the benchmark number.

In particular, do not:

- recognize WarpLife bytecode and shortcut it,
- fuse arbitrary VM instruction sequences into Life operations,
- replace the interpreter with generated native code,
- special-case known program counters,
- use a separate optimized representation of the Life world only on CPU.

Normal compiler optimization, efficient data structures, avoiding unnecessary allocations, and sensible threading are allowed.

The purpose is to measure the current architecture honestly.

---

## 11. Suggested Structure

Fit this into the current repository structure rather than forcing these exact names, but conceptually separate:

```text
runtime/
    cpu_interpreter.*
    gpu_interpreter / existing CUDA runtime

bench/
    warplife_bench.*
    native_cpu_life.*
    existing native CUDA reference
```

Where possible, share:

- opcode definitions,
- bytecode loader,
- VM constants,
- benchmark seed generation,
- result formatting,
- correctness/checksum helpers.

Avoid coupling the CPU interpreter to CUDA headers unless genuinely necessary.

---

## 12. Tests

Add focused tests for CPU execution of representative current instructions, especially:

- vector arithmetic,
- scalar/control operations,
- active-lane masking,
- branches,
- loads/stores,
- packed-cell operations used by WarpLife,
- framebuffer writes,
- halt/fault behaviour.

Where practical, use the same small bytecode fixtures against both CPU and GPU engines and compare final VM state.

At minimum, WarpLife must produce matching deterministic output between CPU and GPU engines before benchmark results are accepted.

---

## 13. Deliverable

Produce a benchmark report, preferably alongside the existing WarpLife baseline, containing:

1. machine/CPU details,
2. compiler/build mode,
3. worker counts,
4. CPU WarpVM results,
5. GPU WarpVM results,
6. native CPU results,
7. existing native CUDA results where available,
8. explicit ratios,
9. correctness verification result,
10. short observations about scaling.

Do not draw architectural conclusions beyond what the measurements support.

The most important questions are:

> How much faster or slower is GPU WarpVM than the exact same WarpVM interpreter model on one CPU thread?

and

> How does that relationship change as the number of independent VMs rises from 1 to 256?

---

## 14. Definition of Done

v0.1.2 is complete when:

- existing `.wvm` WarpLife bytecode runs through a real CPU WarpVM interpreter,
- CPU and GPU WarpVM produce matching deterministic results,
- single-threaded CPU WarpVM benchmarks run for 1, 8, 32, 64 and 256 VMs,
- parallel CPU WarpVM benchmarks run using a configurable worker count,
- a native CPU Life baseline is available,
- results are reported in the same units as the existing GPU/native-CUDA baseline,
- no Life-specific shortcuts have been introduced into WarpVM,
- existing GPU tests and functionality still pass.

Keep the slice narrow. Optimization of either interpreter belongs in later versions after this comparison exists.
