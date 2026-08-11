# WarpVM Project Spec v0.1.3
## Dual-Mode WarpVM: Interpreted + Native GPU Execution

**Status:** Proposed implementation direction  
**Version target:** v0.1.3  
**Date:** 2026-08-11

---

## 1. Purpose

WarpVM already defines a useful abstract machine:

- one WarpVM corresponds conceptually to one 32-lane GPU warp;
- each WarpVM has its own program;
- each WarpVM has its own registers, memory, stack, execution state, and framebuffer region;
- multiple WarpVMs execute independently and concurrently;
- programs are currently represented as WarpVM bytecode (`.wvm`) and executed by a persistent GPU interpreter.

The current interpreter is correct, but profiling shows that instruction-by-instruction interpretation imposes a large cost on the GPU. Attempts to reduce individual interpreter costs have produced useful but modest gains.

v0.1.3 should therefore **not redefine WarpVM around CUDA** and should **not discard the interpreter**.

Instead, it should establish a second execution backend:

> A WarpVM program may execute either as interpreted WarpVM bytecode or as native GPU code compiled from that same bytecode.

The `.wvm` program remains the canonical representation of the program.

Interpretation and compilation are execution strategies, not different machine models.

---

## 2. Core Architectural Principle

The following must remain true regardless of execution mode:

```text
1 WarpVM
    = 1 logical 32-lane machine
    = 1 independent program
    + independent VM state
    + WarpVM memory semantics
    + WarpVM messaging semantics
    + WarpVM framebuffer semantics
```

Execution may be:

```text
.wvm
  |
  +--> WarpVM interpreter --> GPU execution
  |
  +--> WarpVM compiler --> PTX/native GPU code --> GPU execution
```

Both paths must implement the same observable WarpVM semantics.

The important abstraction is therefore:

```text
WarpVM program + WarpVM state
```

not:

```text
WarpVM interpreter
```

The interpreter is only one backend.

---

## 3. Key Design Goal

Each VM must ultimately be capable of selecting its execution strategy independently.

For example:

```text
VM 0   interpreted
VM 1   compiled
VM 2   compiled
VM 3   interpreted
...
VM 111 compiled
```

This is preferable to making compilation a global population state.

A population-wide compiled kernel may be useful as an optimization later, but it must not become the conceptual foundation of the system.

The architectural target is:

> **112 individual computers, each of which may currently be interpreted or compiled.**

The execution mode is a property of a particular VM/program version, not of the WarpVM system as a whole.

---

## 4. Canonical Program Representation

`.wvm` bytecode remains canonical.

Compiled native code is derived state.

Conceptually:

```text
VM {
    program_bytecode
    program_hash

    architectural_state

    execution_mode:
        INTERPRETED
        COMPILED

    compiled_artifact:
        optional
}
```

If the bytecode changes, any compiled artifact derived from the old bytecode becomes invalid.

Example:

```text
program mutation
    |
    +--> invalidate compiled artifact
    +--> retain new .wvm as canonical program
    +--> VM can immediately continue through interpreter
```

Compilation may then occur later.

---

## 5. Why Preserve the Interpreter?

The interpreter is not merely a slow fallback.

It provides capabilities that are valuable to the architecture:

- immediate execution of newly generated or mutated code;
- no compile latency for short-lived programs;
- debugging;
- instruction stepping;
- tracing;
- introspection;
- rapid evolutionary mutation;
- execution of cold or rarely used code;
- a semantic reference implementation for the compiled backend.

This produces a useful lifecycle:

```text
new / mutated program
        |
        v
   interpreted
        |
        | survives / becomes hot
        v
     compile
        |
        v
     compiled
        |
        | program changes
        v
   interpreted again
```

This model should be treated as a primary architectural feature rather than merely a compatibility mechanism.

---

## 6. Why Compile?

Current profiling shows that the GPU interpreter spends substantial time repeatedly implementing virtual-machine machinery:

- bytecode fetch;
- opcode dispatch;
- virtual register selection;
- VM PC maintenance;
- stack/call machinery;
- architectural state loads/stores;
- per-instruction semantic checks.

The native GPU should instead eventually execute operations corresponding directly to the WarpVM program.

Conceptually, interpreted execution performs:

```text
fetch WarpVM instruction
decode
select operands
perform operation
update VM state
repeat
```

Compiled execution should translate the WarpVM instruction stream ahead of execution so that the hardware runs native GPU instructions directly.

The objective is not to make WarpVM become PTX.

The relationship is:

```text
WarpVM bytecode : PTX/SASS
        as
source/IR        : target machine code
```

PTX is an implementation backend.

WarpVM remains the language/machine contract.

---

## 7. Physical Mapping

The compiled backend should preserve the fundamental WarpVM mapping:

```text
WarpVM lane 0  -> GPU lane/thread 0
WarpVM lane 1  -> GPU lane/thread 1
...
WarpVM lane 31 -> GPU lane/thread 31
```

The preferred first implementation should therefore compile one WarpVM program into code intended to execute cooperatively across exactly 32 GPU threads.

The runtime is responsible for preserving the WarpVM model even if CUDA schedules those warps across SMs in any order.

No semantics may depend on a particular physical SM.

---

## 8. Compiled Program Model

For v0.1.3, compilation should be whole-program and deliberately simple.

Input:

```text
.wvm bytecode
```

Output:

```text
PTX or another CUDA-loadable native artifact
```

The compiler does not need to be a sophisticated optimizer.

The first goal is to establish whether:

> removing the interpreter while preserving the existing WarpVM ISA produces a major performance improvement.

This is an experiment before it is an optimization project.

The existing WarpLife program should be the first compiled target.

Do not initially redesign WarpLife or add Life-specific instructions.

---

## 9. First Compiler Scope

The compiler should support only the subset of the ISA required by WarpLife plus the minimum runtime machinery necessary to execute it correctly.

Expand coverage incrementally.

The compiler should translate:

- arithmetic instructions;
- logical instructions;
- loads/stores;
- branches;
- WarpVM lane operations used by WarpLife;
- framebuffer writes;
- required VM memory accesses;
- required control flow.

Unsupported instructions should produce a clear compiler error rather than silently falling back inside compiled code.

Fallback between modes should occur at VM execution boundaries, not instruction-by-instruction in v0.1.3.

---

## 10. Safe Execution Boundaries

Do not attempt arbitrary mid-instruction migration between interpreted and compiled execution.

Define explicit safe points where VM architectural state is coherent and ownership may move between backends.

Initially acceptable safe points include:

- VM start;
- VM yield;
- host command boundary;
- evolutionary generation boundary;
- explicit runtime checkpoint;
- program replacement/mutation;
- compiled kernel completion.

At a safe point:

```text
compiled -> interpreted
```

or:

```text
interpreted -> compiled
```

must preserve all architectural state required for equivalent continuation.

---

## 11. Per-VM Mode Switching

The runtime should expose an internal operation conceptually equivalent to:

```text
set_execution_mode(vm_id, INTERPRETED)
set_execution_mode(vm_id, COMPILED)
```

The first implementation does not need to expose this as public user-facing syntax.

Mode transitions should be possible independently for different VMs.

Example:

```text
VM 4 mutates
    -> compiled artifact invalidated
    -> VM 4 returns to interpreter

VM 5 unchanged
    -> remains compiled

VM 6 is newly created
    -> starts interpreted

VM 7 has stable bytecode
    -> remains compiled
```

This independence is an important design requirement.

---

## 12. Compilation Cache

Compiled code should be cached by program identity.

Minimum key:

```text
hash(.wvm bytecode)
```

Potential extended key:

```text
hash(
    .wvm bytecode
    + ISA version
    + compiler version
    + target compute capability
    + relevant compile flags
)
```

If multiple VMs contain identical bytecode, they should be able to share one compiled program artifact while retaining independent VM state.

Example:

```text
VM 2  --\
VM 19 ----> compiled artifact A
VM 43 --/

each VM has separate state
```

This is particularly important for evolutionary populations where identical genomes/programs may occur repeatedly.

---

## 13. Evolutionary Execution Model

A major intended use is an evolving population of WarpVM programs.

Compilation should fit naturally into that model.

Example:

```text
mutation created
      |
      v
execute interpreted
      |
      +---- dies quickly --------> never compiled
      |
      +---- survives / gets hot
                    |
                    v
                 compile
                    |
                    v
           execute natively
                    |
                    v
             program mutates
                    |
                    v
       invalidate compiled version
                    |
                    v
             interpret again
```

Compilation therefore need not happen once per global population generation.

It may happen independently for individual VMs according to program stability or runtime policy.

Population-wide synchronization remains available where the evolutionary algorithm naturally requires it, but compilation should not impose an unnecessary global barrier.

---

## 14. Hotness Policy

v0.1.3 does not need an advanced JIT policy.

Initially support explicit/manual compilation.

Example internal policy:

```text
if compiled artifact exists:
    may run compiled
else:
    run interpreted
```

Then optionally add a simple hotness threshold:

```text
if executions(program_hash) >= threshold:
    request compilation
```

Do not spend significant effort tuning automatic hotness heuristics in v0.1.3.

The architecture matters more than policy.

---

## 15. Runtime Scheduling

The first implementation may use separate CUDA launches for:

- interpreted VM groups;
- compiled VM groups/programs.

It is acceptable if mode transitions require returning to the runtime scheduler.

The runtime should group work where useful, but this grouping must not leak into WarpVM semantics.

Example implementation:

```text
runtime scheduler
    |
    +--> interpreter kernel for interpreted VMs
    |
    +--> compiled kernel A for VMs using program A
    |
    +--> compiled kernel B for VMs using program B
```

This is still one population of independent WarpVMs.

The scheduler is merely choosing execution machinery.

---

## 16. State Representation

A compiled VM and interpreted VM must agree on the externally observable architectural state.

Where possible, use a shared canonical state representation for:

- architectural registers;
- memory;
- stack;
- PC / continuation metadata where relevant;
- messages;
- framebuffer;
- VM status;
- host-visible control state.

Compiled code may keep hot temporary values in physical GPU registers during execution.

At a safe point it must materialize enough architectural state for:

- debugging;
- migration back to interpreter;
- host inspection;
- checkpointing;
- correctness comparison.

Do not require compiled execution to materialize the complete VM state after every WarpVM instruction.

That would recreate interpreter-like overhead.

---

## 17. Program Counter Semantics

Compiled execution does not need to maintain the architectural PC continuously if doing so would impose unnecessary cost.

Instead:

- preserve exact WarpVM control-flow semantics;
- maintain sufficient continuation metadata at safe points;
- materialize a meaningful WarpVM PC when execution returns to the runtime/interpreter.

If precise instruction-level debugging is requested, run that VM interpreted.

This is an intentional distinction between:

```text
semantic equivalence
```

and:

```text
identical internal implementation
```

---

## 18. Correctness Requirement

The interpreted backend remains the reference.

For every compiled test program, compare CPU/interpreted GPU/compiled GPU state at defined checkpoints.

For WarpLife, continue the existing full-world equivalence checks.

At minimum compare:

- complete world memory;
- generation count;
- framebuffer where applicable;
- architectural state required at checkpoint;
- status/termination state.

Compiled execution must not be accepted merely because its visual output appears correct.

---

## 19. WarpLife Validation Experiment

WarpLife should be the first end-to-end experiment.

Use the existing optimized WarpLife bytecode as input.

Do not initially alter its algorithm.

Benchmark:

```text
A. CPU WarpVM interpreter
B. GPU WarpVM interpreter
C. GPU compiled WarpVM
D. Native handwritten CUDA reference
```

At VM counts:

```text
1
8
32
64
256
```

Record:

- generations/sec/VM;
- aggregate cell updates/sec;
- latency;
- GPU occupancy/residency where meaningful;
- compile time;
- generated PTX size;
- generated native code size;
- correctness.

The important comparison is:

```text
compiled WarpVM / handwritten native CUDA
```

and:

```text
compiled WarpVM / interpreted GPU WarpVM
```

---

## 20. Success Criteria

v0.1.3 is successful if it demonstrates all of the following:

1. An existing `.wvm` program can execute through both interpreted and compiled GPU backends.

2. Both backends preserve the WarpVM 32-lane machine semantics.

3. A VM can change execution mode at a defined safe point without losing architectural state.

4. Two VMs may use different execution modes within the same overall WarpVM runtime session.

5. Identical programs may share one compiled artifact while maintaining independent VM state.

6. Compiled WarpLife is substantially faster than interpreted GPU WarpLife.

7. Correctness remains bit-equivalent at defined checkpoints.

The initial experiment does **not** require compiled WarpLife to match handwritten native CUDA performance.

---

## 21. Important Negative Result

If compiled WarpLife remains dramatically slower than native CUDA even after interpreter overhead has been removed, that is valuable evidence.

It would indicate that one or more of the following are fundamental:

- WarpVM ISA granularity is too fine;
- architectural register semantics inhibit efficient GPU code;
- memory semantics impose excessive overhead;
- control flow maps poorly to GPU execution;
- lane semantics need redesign;
- higher-granularity warp-native operations are required.

Only after obtaining this compiled baseline should the project make a major decision about redesigning the ISA.

This avoids changing two variables at once.

---

## 22. Higher-Granularity ISA

Do not make a high-granularity ISA the primary v0.1.3 task.

However, keep it as the likely next architectural experiment.

Possible future generic warp-native operations include:

```text
shuffle
gather
scatter
scan
reduce
map
packed-vector arithmetic
packed comparisons
stencil/neighbour operations
```

Avoid application-specific instructions such as:

```text
LIFE_STEP
```

The purpose of such future instructions would be to make each WarpVM instruction represent a useful unit of GPU computation, not to hard-code benchmarks.

---

## 23. Compiler Implementation Strategy

Prefer the smallest credible compiler.

A reasonable first pipeline is:

```text
.wvm
  |
decode bytecode
  |
construct simple CFG
  |
emit PTX text
  |
PTX compile/load
  |
launch native WarpVM kernel
```

Do not build a sophisticated SSA optimizer unless required.

Use direct lowering wherever possible.

The existing bytecode is already low-level enough that the first backend should be primarily translation plus runtime glue.

---

## 24. Compiled Function ABI

Define a stable internal ABI between runtime and compiled WarpVM programs.

Conceptually:

```text
compiled_program(
    vm_state*,
    vm_memory*,
    framebuffer*,
    messaging_state*,
    control_state*
)
```

The exact representation may differ.

The ABI must allow:

- multiple independent VM instances to execute the same compiled program;
- runtime-selected VM state;
- eventual migration between execution modes;
- compatibility checking by compiler/runtime version.

Keep the ABI minimal.

---

## 25. Do Not Couple Program Identity to VM Identity

This distinction is essential:

```text
VM identity != program identity
```

A VM owns state.

A program describes behaviour.

Therefore:

```text
many VMs -> one compiled program
```

must be supported.

Likewise:

```text
one VM -> program A -> mutate -> program B
```

must be supported without creating a new conceptual machine.

---

## 26. Debugging Model

Interpreted execution should remain the preferred debugging mode.

Features such as:

- single-step;
- exact WarpVM PC;
- opcode tracing;
- register tracing;
- fault diagnosis;

should not initially be duplicated in compiled mode.

Instead:

```text
debug requested
    -> transition VM to interpreter at safe point
```

Compiled mode may initially provide only coarse checkpoint/state inspection.

This is a feature of the dual-mode architecture, not a deficiency.

---

## 27. Fault Semantics

Compiled code must preserve WarpVM-visible fault behaviour where practical.

Do not necessarily reproduce the interpreter's per-opcode fault-vote implementation.

The interpreter implementation mechanism is not part of the abstract machine.

Compiled mode may use more efficient native checks provided that externally visible WarpVM behaviour remains compatible.

Where exact instruction-level fault timing is impossible or undesirable, define the difference explicitly and test it.

---

## 28. Host Control

Do not poll host-visible control memory on every compiled WarpVM instruction.

Compiled programs should poll for control only at defined safe points or suitable coarse intervals.

The runtime must retain the ability to:

- stop a VM;
- request yield/checkpoint;
- inspect state;
- transition mode.

But compiled execution must not recreate the interpreter's hot-path control overhead.

---

## 29. Persistence

"Persistent WarpVM" refers to the logical machine and its continuing state.

It does **not** require one CUDA kernel invocation to remain alive forever.

A VM may conceptually persist across:

```text
interpreter launch
-> runtime checkpoint
-> compiled launch
-> host interaction
-> another compiled launch
```

provided its WarpVM state and identity persist.

Do not confuse CUDA kernel lifetime with VM lifetime.

---

## 30. Non-Goals for v0.1.3

Do not attempt:

- full optimizing compiler;
- arbitrary instruction-level OSR;
- LLVM integration;
- complex tracing JIT;
- speculative compilation;
- whole-population superkernel generation;
- runtime code mutation while native code is currently executing;
- fully asynchronous per-instruction migration;
- high-level language design;
- major ISA redesign;
- Life-specific acceleration;
- removal of the interpreter.

These may be explored later.

---

## 31. Suggested Implementation Slices

### Slice 1 — Minimal PTX backend

Compile a tiny arithmetic WarpVM program.

Requirements:

- 32-lane mapping preserved;
- input state loaded;
- program executes;
- output state matches interpreter.

### Slice 2 — Control flow

Add:

- conditional branches;
- loops;
- required PC/continuation handling.

Compare complete state with interpreter.

### Slice 3 — WarpLife ISA subset

Implement every instruction currently required by WarpLife.

WarpLife runs compiled for one VM.

### Slice 4 — WarpLife correctness

Run existing full-world CPU/GPU equivalence tests against compiled backend.

### Slice 5 — Performance baseline

Benchmark interpreted GPU vs compiled GPU vs native CUDA.

Do not optimize generated PTX yet.

### Slice 6 — Shared compiled program

Run multiple VM instances using the same compiled WarpLife artifact and independent state.

### Slice 7 — Mixed mode

Run at least:

```text
VM 0 interpreted
VM 1 compiled
```

within the same runtime session.

Verify independent progress and state.

### Slice 8 — Mode transition

At a safe point:

```text
interpreted -> compiled
```

then later:

```text
compiled -> interpreted
```

Verify state continuity.

### Slice 9 — Compilation cache

Cache artifact by program hash.

Verify multiple identical VMs reuse it.

---

## 32. Benchmark Discipline

Do not optimize from intuition alone.

For every experimental change:

- preserve a baseline;
- measure one-VM latency;
- measure high-occupancy throughput;
- retain correctness checks;
- distinguish compile time from execution time;
- distinguish host scheduling overhead from kernel runtime;
- preserve the handwritten CUDA reference.

Failed experiments should remain documented where useful.

---

## 33. Decision Point After v0.1.3

After the compiled WarpLife benchmark, make the architectural decision using evidence.

### Outcome A — compiled WarpVM approaches native CUDA

Then the existing WarpVM ISA is viable.

Focus on:

- compilation quality;
- runtime scheduling;
- mixed interpreted/compiled execution;
- evolutionary workloads;
- richer applications.

### Outcome B — compiled WarpVM improves dramatically but remains substantially behind native CUDA

Investigate generated PTX/SASS and identify semantic costs.

Then selectively introduce generic warp-native ISA operations.

### Outcome C — compiled WarpVM remains poor

Then the current ISA or machine semantics are likely mismatched to GPU hardware.

Use the result to redesign WarpVM around higher-granularity warp-native computation.

The experiment is still useful because it identifies whether interpretation or architecture is the true limiting factor.

---

## 34. Conceptual Summary

WarpVM should become:

> **A persistent 32-lane virtual computer architecture whose programs have a canonical bytecode representation and may execute either through an interpreter or as compiled native GPU code.**

The individual VM remains the unit of identity.

The program remains independent from the VM state.

The compiled artifact remains independent from both and may be shared.

Therefore:

```text
VM
  has state
  has a program

Program
  exists canonically as .wvm
  may have a compiled native representation

Runtime
  chooses how each VM executes
  may change that choice at safe points
```

This preserves the original "many little computers" idea while allowing the GPU to execute stable programs without paying an instruction-interpreter cost forever.

The immediate question for v0.1.3 is deliberately narrow:

> **Can the exact same WarpVM program become dramatically faster when compiled directly to native GPU execution, while preserving the one-WarpVM-per-warp machine model?**

Answer that before redesigning the ISA.
