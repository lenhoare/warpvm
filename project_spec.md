# Project Spec: WarpVM

## 1. Project Summary

WarpVM is an experimental GPU-native virtual-computer architecture designed around NVIDIA CUDA warps.

Instead of emulating a conventional scalar CPU on a GPU, WarpVM treats a 32-lane CUDA warp as the natural unit of computation. Each virtual machine is a persistent, independently addressable 32-lane computer with its own state, memory, program, mailbox, and logical identity.

The project has two goals:

1. Explore what a computer architecture looks like when 32-way parallel execution is the default rather than an optimisation.
2. Provide a substrate for highly parallel, irregular machine-learning workloads such as genetic algorithms, genetic programming, evolutionary search, swarm systems, agent populations, and asynchronous island models.

A future extension may use a large language model as a high-level evolutionary adviser: the GPU performs large-scale cheap search and evaluation, while an LLM occasionally proposes structured mutations, new search directions, operators, constraints, or candidate programs.

---

## 2. Core Design Principle

The machine must be designed around the hardware that actually exists.

A WarpVM machine is not a pretend scalar CPU that happens to run on CUDA.

A WarpVM machine is:

- one logical virtual computer,
- implemented by one CUDA warp,
- composed of 32 execution lanes,
- naturally SIMD/SIMT in its instruction semantics,
- persistent,
- independently addressable,
- capable of communication with other machines,
- observable and controllable from the host.

The central programming rule is:

> Parallel operations are normal; scalar operations are exceptional.

For example, a normal arithmetic instruction operates across all active lanes:

```text
ADD r0, r1, r2
```

conceptually means:

```text
for each active lane i:
    r0[i] = r1[i] + r2[i]
```

A scalar result should normally be produced through an explicit reduction, lane selection, broadcast, or similar operation.

---

## 3. Non-Goals

Version 0.x is not intended to provide:

- full hardware virtualisation,
- POSIX compatibility,
- Linux compatibility,
- x86, ARM, 68000, or RISC-V emulation,
- strong security isolation between mutually hostile tenants,
- exact control over which SM hosts a VM,
- general-purpose GPU acceleration for conventional applications,
- a replacement for CUDA, cuBLAS, PyTorch, or Triton,
- high-performance dense neural-network training.

The project should remain deliberately small until the core architecture has been demonstrated.

---

## 4. Hardware Model

### 4.1 Virtual Machine Unit

One WarpVM machine maps to one CUDA warp.

Each VM therefore has:

- 32 execution lanes,
- one logical VM ID,
- one program counter,
- vector registers with one value per lane,
- optional scalar/control registers,
- an active-lane mask,
- private VM memory,
- mailbox/message queues,
- status/control flags.

The physical SM on which a VM executes is not part of VM identity.

```text
VM identity != warp identity != SM identity
```

A VM must remain logically identical if its execution is later stopped, restored, or migrated.

---

### 4.2 Warp Alignment

VMs must be formed from 32 consecutive CUDA threads aligned to hardware warp boundaries.

Thread-block geometry must guarantee that a VM never straddles two hardware warps.

Example for a 256-thread block:

```text
threads   0-31   -> VM 0
threads  32-63   -> VM 1
threads  64-95   -> VM 2
...
threads 224-255  -> VM 7
```

The runtime must not depend on a particular SM assignment.

---

### 4.3 Persistent Kernel

The intended execution model is a persistent CUDA kernel.

At startup:

1. host allocates VM state,
2. programs and initial data are copied to GPU memory,
3. a persistent kernel is launched,
4. warp workers enter their VM execution loops,
5. the kernel remains resident while VMs execute,
6. the host communicates through shared control/message structures.

The first implementation should favour simplicity over maximum occupancy.

---

## 5. Proposed VM State

A minimal VM should contain:

```text
vm_id
status
pc
active_mask

vector_registers[]
scalar_registers[]

memory_base
memory_size

mailbox_in
mailbox_out

instruction_counter
fault_code
debug_flags
```

Recommended v0.1 starting point:

- 16 vector registers
- 8 scalar registers
- 32-bit integer values
- 32-bit fixed-width bytecode instructions
- 16 KB or 64 KB VM-local memory
- one small inbound mailbox
- one outbound message slot or ring buffer

Exact values should remain configurable.

---

## 6. Memory Model

### 6.1 VM-Local Memory

Each VM receives a logically private region of GPU global memory.

The runtime may physically store VM state using structure-of-arrays layout where this improves coalescing.

Logical model:

```text
VM 0 -> RAM region 0
VM 1 -> RAM region 1
...
```

A VM cannot directly address another VM's private RAM in v0.1.

Communication occurs through explicit messaging.

---

### 6.2 Register Model

Vector registers contain 32 values:

```text
r0[0..31]
r1[0..31]
...
```

Each hardware lane owns or operates on its corresponding element.

Scalar registers are reserved for genuinely scalar control data such as:

- program counter,
- loop base,
- VM ID,
- message destination,
- aggregate results.

Scalar operations must be explicit in the ISA where practical.

---

### 6.3 Shared Global Structures

The runtime may expose:

- global message routing table,
- VM status table,
- host command queue,
- VM event/log ring,
- optional shared read-only data,
- optional population/genome store for ML workloads.

These are runtime services, not directly writable arbitrary shared memory unless later added deliberately.

---

## 7. Instruction Set Philosophy

The ISA should be small, regular, and explicitly warp-native.

The first assembler exists primarily to make the machine understandable and debuggable.

### 7.1 Arithmetic

Candidate instructions:

```text
MOV
ADD
SUB
MUL
DIV
MOD
MIN
MAX
ABS
NEG
```

Unless marked scalar, these operate lane-wise.

---

### 7.2 Bitwise / Integer

```text
AND
OR
XOR
NOT
SHL
SHR
```

---

### 7.3 Comparisons and Masks

Comparisons should produce masks or predicates rather than immediately causing divergent branches.

```text
CMPEQ
CMPNE
CMPLT
CMPLE
CMPGT
CMPGE
MASK
NOTMASK
ANDMASK
ORMASK
```

Predicated execution syntax may be used:

```text
@p ADD r0, r1, r2
@!p SUB r0, r1, r2
```

---

### 7.4 Lane and Warp Operations

These are fundamental, not optional accelerator features.

```text
LANEID
BROADCAST
SHUFFLE
SHUFFLE_XOR
BALLOT
ANY
ALL
REDUCE_ADD
REDUCE_MIN
REDUCE_MAX
PREFIX_SUM        # later
```

The exact mapping should favour CUDA warp primitives wherever possible.

---

### 7.5 Memory

```text
LOAD
STORE
GATHER            # later
SCATTER           # later
```

Initial addressing should be simple and bounded.

All VM memory accesses must remain inside the VM's allocated region.

---

### 7.6 Control Flow

```text
JMP
JMP_IF_ANY
JMP_IF_ALL
CALL
RET
HALT
YIELD
```

Avoid designing the ISA around arbitrary lane-divergent scalar branching.

Masked execution should be preferred.

---

### 7.7 Messaging

```text
SEND
RECV
TRY_RECV
BROADCAST_MSG
```

Messages should initially be small fixed-size records.

Example:

```text
destination_vm
message_type
payload[0..N]
```

Host-visible messaging should use the same conceptual model where possible.

---

### 7.8 Runtime / Introspection

Potential instructions or services:

```text
VMID
CLOCK
RAND
LOG
FAULT
```

Random-number generation is especially useful for evolutionary workloads and should eventually support independent reproducible streams per VM/lane.

---

## 8. Loop Semantics

WarpVM should encourage programmers to process data in chunks of 32.

A conventional loop:

```c
for (i = 0; i < N; ++i)
    work(i);
```

should naturally map to:

```text
base = 0

while base < N:
    i = base + lane_id
    active = i < N
    @active work(i)
    base += 32
```

Thus a loop over 1024 independent elements has 32 warp iterations rather than 1024 scalar iterations.

Multiples of 32 are naturally efficient, but non-multiples are supported with masks.

---

## 9. Assembler

### 9.1 Purpose

The assembler is the first programming environment for WarpVM.

It should:

- expose the real machine model,
- make lane-parallel behaviour obvious,
- produce compact bytecode,
- support labels,
- support constants,
- allow small programs to be written by hand,
- include a disassembler,
- include useful diagnostics.

---

### 9.2 Example

```text
.const N 1000

    S_MOV   s0, 0              ; base

loop:
    LANEID  r0
    S_BCAST r1, s0
    ADD     r2, r0, r1         ; indices base..base+31
    CMP_LT  p0, r2, N

    @p0 LOAD  r3, [r2]
    @p0 MUL   r3, r3, r3
    @p0 STORE [r2], r3

    S_ADD   s0, s0, 32
    S_CMP_LT p1, s0, N
    JMP_IF  p1, loop

    HALT
```

Exact syntax is not fixed by this document.

---

## 10. Higher-Level Languages

### 10.1 Stage 1: Assembly

Required for v0.1.

---

### 10.2 Stage 2: Forth-Like Interactive Environment

A tiny Forth-like environment is attractive because it provides:

- tiny implementation,
- interactive execution,
- easy introspection,
- natural REPL behaviour,
- a useful bootstrap environment,
- a good fit for the planned attach/debug console.

This should be considered after the base VM is stable.

---

### 10.3 Stage 3: Tiny C-Like Compiler

A restricted C-like language should target WarpVM bytecode.

Important semantic difference:

- vector values should be natural/default,
- scalar values should be explicit,
- loops should naturally stride by warp width,
- reductions and masks should be first-class operations.

Possible illustrative syntax:

```c
int x = a + b;          // 32 lane-wise additions
scalar int s = sum(x);  // explicit reduction
```

This syntax is illustrative only.

---

### 10.4 Stage 4: Rust / LLVM

Only after the machine architecture stabilises.

Possible later goals:

- `no_std` Rust support,
- LLVM backend,
- direct PTX generation,
- JIT/specialised execution for hot VM programs.

These are explicitly not v0.1 goals.

---

## 11. Host Runtime

The host-side runtime should initially be written in C++/CUDA C++ or another language with straightforward CUDA integration.

Responsibilities:

- initialise GPU,
- allocate VM state,
- load bytecode,
- launch persistent kernel,
- route host commands,
- read logs and VM status,
- deliver messages,
- stop/restart VMs,
- snapshot VM state,
- expose interactive control.

---

## 12. Interactive Control

A key usability goal is that every VM feels like an addressable little computer.

Actual SSH is unnecessary initially.

Provide a host CLI such as:

```text
warpvm list
warpvm status 73
warpvm attach 73
warpvm pause 73
warpvm resume 73
warpvm reset 73
warpvm load 73 program.wvm
warpvm snapshot 73 out.snap
warpvm send 73 42 1234
```

Inside `attach`:

```text
vm-73> regs
vm-73> sregs
vm-73> mem 0x0000 128
vm-73> pc
vm-73> disasm
vm-73> messages
vm-73> step
vm-73> continue
vm-73> halt
vm-73> log
```

Later this interface may be exposed through an SSH-compatible façade, but the first version should use a simple local terminal protocol.

---

## 13. VM Scheduling and Cooperation

WarpVM does not require control over which SM executes a given VM.

The system should use stable logical VM IDs independent of physical placement.

VMs are expected to be cooperative.

Every VM interpreter loop should periodically observe runtime control flags so that the host can request:

- pause,
- terminate,
- snapshot,
- debug break.

The implementation must not assume CPU-style hard preemption of individual VM instruction streams.

---

## 14. Fault Handling

A malformed or mutated VM program must not be able to corrupt the entire runtime.

At minimum detect:

- invalid opcode,
- out-of-range jump,
- invalid memory access,
- invalid register,
- stack overflow if a stack is introduced,
- illegal message operation,
- execution-budget limit where configured.

Faulted VM:

```text
status = FAULTED
fault_code = ...
```

The warp should then stop executing that VM program while remaining controllable from the host where possible.

This sandboxing is especially important for genetic programming.

---

## 15. Evolutionary ML Architecture

Evolutionary computation is the first serious workload target.

### 15.1 Island Model

Each WarpVM machine can represent an evolutionary island.

An island may contain:

- current genome or program,
- candidate variants,
- fitness history,
- mutation parameters,
- local archive,
- communication state.

One warp can evaluate 32 variants, environments, data samples, or rollouts simultaneously.

---

### 15.2 Asynchronous Evolution

Avoid mandatory global generations.

Each island should be able to run continuously:

```text
select
mutate / recombine
evaluate 32-way
retain/promote
communicate
repeat
```

Useful genomes may migrate between neighbouring or selected VMs.

This allows:

- asynchronous evolution,
- local adaptation,
- heterogeneous strategies,
- reduced global synchronization,
- persistent state,
- continuous search.

---

### 15.3 Candidate Workloads

Priority experiments:

1. genetic algorithms,
2. evolutionary strategies,
3. island-model evolution,
4. genetic programming,
5. novelty search,
6. swarm intelligence,
7. cellular/evolutionary systems,
8. population-based optimisation,
9. simple reinforcement-learning populations,
10. program synthesis/search.

Dense neural-network training is not a primary target.

---

## 16. LLM-Assisted Evolution

The LLM is not part of the inner evaluation loop.

It acts as an occasional high-level search adviser.

Suggested architecture:

```text
GPU WarpVM population
        |
        v
CPU population controllers
        |
        v
selected summaries / elites / failures
        |
        v
LLM adviser
        |
        v
structured interventions
        |
        v
GPU population
```

Potential LLM interventions:

- propose mutations,
- propose new genomes,
- identify useful patterns among successful candidates,
- recommend preserving one trait while varying another,
- change search ranges,
- propose new fitness components,
- create new crossover operators,
- introduce subgoals,
- generate novel program fragments,
- diagnose stagnation,
- seed new islands.

The LLM should operate infrequently relative to GPU evaluation because its role is expensive semantic guidance, not brute-force search.

---

## 17. CPU Controller Layer

A later architecture may use CPU cores as richer supervisory computers.

Responsibilities could include:

- managing groups of WarpVM islands,
- aggregating fitness statistics,
- handling persistent storage,
- deciding migration topology,
- selecting candidates for LLM review,
- injecting LLM interventions,
- controlling experiments,
- monitoring faults.

The exact number of CPU controllers should be configurable and should not be baked into the GPU VM architecture.

---

## 18. Message Topologies

v0.1:

- direct VM-to-VM messaging by logical VM ID.

Later experiments:

- ring,
- grid,
- random graph,
- small-world graph,
- hierarchical groups,
- dynamic neighbourhoods,
- broadcast groups,
- CPU-controller domains.

Topology should be a runtime property rather than hard-coded into VM identity.

---

## 19. Performance Questions

The project should measure rather than assume performance.

Important experiments:

### 19.1 VM Count

Benchmark:

- 32 VMs
- 64 VMs
- 112 VMs
- 128 VMs
- 256 VMs
- 512 VMs
- larger populations where feasible

112 is an interesting starting point on an RTX 3060 because `3584 / 32 = 112`, but it is not a fundamental architectural limit.

---

### 19.2 VM State Size

Measure the effect of:

- register count,
- local RAM size,
- interpreter register pressure,
- shared-memory use,
- mailbox size.

---

### 19.3 Divergence

Create workloads ranging from:

- identical VM programs,
- same program with different data,
- moderately different control flow,
- entirely different programs.

Measure throughput degradation.

This is one of the project's most interesting empirical questions.

---

### 19.4 Interpreter Overhead

Compare:

1. bytecode interpreter,
2. specialised interpreter,
3. generated PTX/CUDA for fixed programs,
4. native handwritten CUDA baseline.

---

### 19.5 Evolutionary Workloads

Compare WarpVM against:

- CPU-only implementation,
- conventional batch CUDA kernels,
- host-orchestrated CUDA implementation,
- persistent-kernel implementation without the VM abstraction.

The goal is to find workloads where persistent state, reduced host orchestration, or warp-native structure outweigh interpreter overhead.

---

## 20. v0.1 Deliverable

Version 0.1 should be intentionally small.

### Required

- CUDA persistent kernel
- one VM per warp
- stable logical VM IDs
- at least 64 simultaneous VMs
- 16 vector registers
- basic scalar/control state
- private VM RAM
- fixed-width bytecode
- assembler
- disassembler
- arithmetic instructions
- comparisons/masks
- lane ID
- broadcast
- at least one reduction
- load/store
- jumps
- halt/yield
- VM-to-host logging
- VM-to-VM fixed-size messaging
- host CLI
- `list`
- `status`
- `attach`
- register inspection
- memory inspection
- program loading
- basic fault detection

### Demonstration

Run at least 64 VMs simultaneously.

Each VM should:

1. execute its own bytecode program,
2. maintain independent state,
3. perform a 32-lane computation,
4. send at least one message,
5. remain inspectable while the persistent kernel is running.

The user should be able to run:

```text
warpvm attach 37
```

and inspect a live GPU-resident virtual computer.

That is the milestone that proves the concept.

---

## 21. v0.2 Evolution Demo

Implement a simple asynchronous island-model genetic algorithm.

Suggested toy problem:

- each VM owns a candidate genome,
- each lane evaluates one mutation or one independent trial,
- reduction computes aggregate fitness,
- VM retains the best candidate,
- high-performing genomes occasionally migrate to another VM,
- no global generation barrier.

Measure:

- evaluations per second,
- VM utilisation,
- divergence cost,
- messaging overhead,
- convergence against CPU and conventional CUDA implementations.

---

## 22. v0.3 Genetic Programming Demo

Represent programs directly as WarpVM bytecode or a safe restricted subset.

Allow mutation of:

- opcodes,
- operands,
- constants,
- instruction insertion/deletion,
- control flow within bounded rules.

Evaluate many candidate programs in the sandbox.

This turns the machine itself into an evolutionary substrate.

---

## 23. v0.4 LLM Adviser

Add an optional host-side LLM interface.

The controller periodically sends the LLM:

- elite candidates,
- fitness trends,
- novelty archive samples,
- failure/stagnation summaries,
- mutation history.

The LLM returns structured JSON-like interventions, for example:

```text
mutation_bias
preserve_features
vary_features
new_seed_candidates
new_operator
new_objective_weight
island_restart
```

All LLM output must be validated before injection into the population.

The evolutionary system must remain functional with the LLM disabled.

---

## 24. Implementation Suggestions

Suggested initial stack:

- CUDA C++ for kernel/runtime
- C++ host CLI
- CMake
- Python only for analysis/benchmark scripts if useful
- custom assembler implemented in C++ or Rust
- simple binary `.wvm` bytecode format
- textual `.wva` assembly format

Possible names:

```text
warpvm          runtime/CLI
.wva            WarpVM assembly
.wvm            WarpVM bytecode
.wvs            WarpVM snapshot
```

Naming is provisional.

---

## 25. Repository Layout

Suggested:

```text
warpvm/
├── CMakeLists.txt
├── README.md
├── project_spec.md
├── docs/
│   ├── architecture.md
│   ├── isa.md
│   ├── memory_model.md
│   └── experiments.md
├── src/
│   ├── gpu/
│   │   ├── kernel.cu
│   │   ├── interpreter.cuh
│   │   ├── vm_state.cuh
│   │   └── messaging.cuh
│   ├── host/
│   │   ├── runtime.cpp
│   │   ├── cli.cpp
│   │   └── debugger.cpp
│   └── asm/
│       ├── assembler.cpp
│       └── disassembler.cpp
├── include/
├── programs/
│   ├── hello.wva
│   ├── vector_add.wva
│   ├── messaging.wva
│   └── loop32.wva
├── tests/
└── benchmarks/
```

---

## 26. First Development Slices

### Slice 1 — One Warp

- launch one 32-thread warp,
- create one VM state,
- execute a tiny hard-coded instruction stream,
- prove lane-wise arithmetic,
- return result to host.

### Slice 2 — Interpreter

- define bytecode,
- fetch/decode/execute loop,
- implement MOV, ADD, HALT,
- load program from host.

### Slice 3 — Multiple VMs

- launch multiple aligned warps,
- assign stable VM IDs,
- give each independent state and RAM,
- verify no cross-VM corruption.

### Slice 4 — Warp-Native Instructions

- LANEID,
- mask comparison,
- broadcast,
- reduction,
- 32-stride loop demo.

### Slice 5 — Control Plane

- persistent kernel,
- status table,
- host pause/halt commands,
- logs,
- `warpvm list`.

### Slice 6 — Attach

- `warpvm attach N`,
- inspect registers,
- inspect memory,
- inspect PC,
- basic breakpoint/pause mechanism.

### Slice 7 — Messaging

- fixed-size mailbox,
- SEND,
- TRY_RECV,
- demonstrate VM 7 sending a value to VM 23.

### Slice 8 — Evolution

- 32 candidate variants per VM,
- reduction to fitness,
- local selection,
- asynchronous migration.

---

## 27. Success Criteria

The project is successful if it answers interesting questions, even if it never becomes a production system.

Minimum success:

- a warp behaves convincingly as an addressable virtual computer,
- many such machines coexist,
- programming them feels naturally parallel,
- interactive inspection works,
- messaging works,
- an evolutionary workload can exploit the architecture.

Stronger success:

- WarpVM demonstrates a measurable advantage for at least one irregular, persistent, population-based workload compared with a straightforward conventional implementation.

Research-level success:

- the architecture reveals a useful programming abstraction for SIMT-native persistent computation that is valuable beyond this project.

---

## 28. Guiding Principle

Do not optimise the machine into looking like an ordinary CPU.

If a design decision makes WarpVM more familiar but wastes the 32-lane nature of the hardware, reconsider it.

The project exists to answer:

> What should a programmable computer look like if 32-way parallel execution is its primitive operation rather than an accelerator feature?

