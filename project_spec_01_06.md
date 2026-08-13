# WarpVM Project Spec 01.06 — CPU Supervisor and Heterogeneous VM Programs

## 1. The model to preserve

WarpVM now needs a small **CPU-side supervisor/control plane**.

This is deliberately **not** an operating system inside the WarpVM ISA and it should not add process-management instructions to the VM.

The intended architecture is:

```text
human / startup script
          |
          v
CPU WarpVM supervisor
          |
          v
persistent GPU WarpVM population
   |      |      |      |
  VM0    VM1    VM2    VM3 ...
```

The CPU supervisor owns orchestration:

- load and unload program images,
- create/delete VM instances,
- assign programs to VM slots,
- start, stop, resume and reset VMs,
- choose interpreted or compiled execution,
- inspect status and state,
- service control requests originating from WarpVM programs.

The GPU VMs remain small parallel computers. They execute programs, hold state, communicate and make requests. They do not contain an OS.

The central rule is:

> **Program identity, VM identity and execution engine are separate concepts.**

A VM remains the same logical computer whether its program is being interpreted or executed through a compiled backend.

---

## 2. Why this slice exists

The runtime can currently execute one program on one or many WarpVMs.

That is useful, but it still resembles the conventional CUDA model:

```text
one program
    |
    +-- VM 0
    +-- VM 1
    +-- VM 2
    +-- ...
```

WarpVM should instead support a heterogeneous resident population:

```text
VM 0 -> plasma.wvm
VM 1 -> mandelbrot.wvm
VM 2 -> gossip.wvm
VM 3 -> histogram.wvm
VM 4 -> gossip.wvm
VM 5 -> controller.wvm
...
```

Multiple VMs may share one program image, while other VMs execute entirely different programs.

This is an important architectural milestone because it makes the project's central abstraction visible:

> The GPU is hosting a population of persistent computers, not merely applying one kernel/program to many pieces of data.

CUDA can of course express heterogeneous behaviour manually. WarpVM's differentiator is that **independent program identity, persistent machine state, lifecycle, messaging and execution mode are runtime concepts rather than application-specific CUDA plumbing**.

---

## 3. Program objects

The supervisor should maintain a registry of loaded programs.

Conceptually:

```text
Program
    program_id
    name
    wvm_image
    wvm_word_count
    optional compiled implementation
    metadata / diagnostics
    reference count
```

Program images should normally be immutable and shareable.

If 64 VMs run `gossip.wvm`, the bytecode should exist once and all 64 VMs may reference it.

```text
Program GOSSIP
      ^   ^   ^
      |   |   |
     VM2 VM8 VM31
```

Loading a program does **not** create a VM.

Deleting/unloading a program should be rejected while live VM instances still reference it, unless an explicit forced behaviour is later designed.

---

## 4. VM instances

A VM instance is the running-machine state, not the code image.

Conceptually each slot contains:

```text
VmInstance
    vm_id
    lifecycle_state
    program_id
    execution_engine

    pc
    registers
    scalar registers
    active mask
    RAM
    framebuffer
    mailbox state
    fault state
    counters
    continuation/call state
    other existing architectural state
```

The program should no longer be an implicit global property of the resident population.

Each VM must be able to resolve instruction fetch through its own `program_id` / program reference.

Different VMs may therefore have the same logical PC while fetching entirely unrelated instructions.

---

## 5. VM lifecycle

Keep the lifecycle deliberately small.

Recommended states:

```text
EMPTY
READY
RUNNING
STOPPED
HALTED
FAULTED
```

Suggested meanings:

### EMPTY

No VM instance occupies this logical slot.

### READY

A VM has been created, has a program assigned and architectural state initialised, but is not currently executing.

### RUNNING

The VM is eligible to execute through its selected engine.

### STOPPED

Execution is intentionally paused while architectural state is preserved.

### HALTED

The program executed its normal HALT behaviour.

### FAULTED

Execution stopped because of an architectural/runtime fault. Existing diagnostics and fault information remain inspectable.

Lifecycle transitions should be explicit and testable.

For example:

```text
EMPTY -> create -> READY -> start -> RUNNING
RUNNING -> stop -> STOPPED -> resume -> RUNNING
RUNNING -> HALT -> HALTED
RUNNING -> fault -> FAULTED
STOPPED/HALTED/FAULTED -> reset -> READY
any non-EMPTY state -> delete -> EMPTY
```

Do not silently destroy RAM/register/framebuffer state on `stop`.

`reset` is the operation that deliberately restores initial machine state.

---

## 6. Execution engine is per VM

Execution mode must not be a global property of the entire resident WarpVM population.

Conceptually:

```text
VM 0 -> plasma       -> COMPILED
VM 1 -> gossip       -> INTERPRETED
VM 2 -> mandelbrot   -> COMPILED
VM 3 -> controller   -> INTERPRETED
```

Define at least:

```text
INTERPRETED
COMPILED
```

The architectural state of the VM is authoritative in either case.

Compilation is an implementation of a program, not a different kind of VM.

A program object may therefore conceptually contain both:

```text
Program A
    canonical .wvm image
    optional native/compiled implementation
```

The `.wvm` program remains the canonical executable definition.

### Initial implementation rule

Do not force the first heterogeneous-program implementation to solve every compiled-dispatch problem at once.

It is acceptable to stage the work:

1. heterogeneous interpreted VMs first;
2. supervisor lifecycle and program registry;
3. heterogeneous compiled dispatch;
4. mixed interpreted/compiled VMs simultaneously.

However, the data structures and public control model must be designed from the beginning so that `execution_engine` belongs to the VM, not globally to the kernel/runtime.

### Engine changes

Long-term target:

```text
stop VM
change execution engine
resume same VM
```

must preserve the VM's architectural identity and state.

If live engine migration at an arbitrary PC requires additional backend work, implement it as a later sub-slice rather than compromising the model.

---

## 7. CPU supervisor API

The supervisor should expose one coherent operation set which can later be driven by several frontends.

Core operations:

```text
program_load(path, name) -> program_id
program_list()
program_unload(program_id)

vm_create(vm_id, program_id)   or allocate next free vm_id
vm_delete(vm_id)
vm_start(vm_id)
vm_stop(vm_id)
vm_resume(vm_id)
vm_reset(vm_id)
vm_set_engine(vm_id, engine)
vm_status(vm_id)
vm_list()
```

Existing inspection operations should continue to work:

```text
registers
scalar registers
memory
pc
disassembly
messages
fault state
framebuffer
instruction counters
```

The implementation may use a C++ class/API internally, but the semantics should not depend on the CLI syntax.

---

## 8. Human CLI and startup scripts

A human should be able to configure a WarpVM population without writing CUDA or host C++.

Illustrative CLI:

```text
warpvm program load plasma.wvm --name plasma
warpvm program load gossip.wvm --name gossip
warpvm program load mandelbrot.wvm --name mandelbrot

warpvm vm create 0 plasma
warpvm vm create 1 gossip
warpvm vm create 2 mandelbrot
warpvm vm create 3 gossip

warpvm vm engine 0 compiled
warpvm vm engine 1 interpreted
warpvm vm engine 2 compiled
warpvm vm engine 3 interpreted

warpvm vm start 0
warpvm vm start 1
warpvm vm start 2
warpvm vm start 3

warpvm vm stop 1
warpvm vm resume 1
warpvm vm delete 3

warpvm list
warpvm status 2
```

Exact command spelling is not architectural. Prefer compatibility with the existing `warpvm list/status/attach/...` CLI where practical.

### Startup script

Provide a simple batch/startup mechanism so a population can be declared and launched repeatably.

The first version may simply execute supervisor commands line by line:

```text
program load plasma.wvm plasma
program load gossip.wvm gossip

vm create 0 plasma
vm create 1 gossip
vm create 2 gossip

vm engine 0 compiled
vm engine 1 interpreted
vm engine 2 interpreted

vm start 0
vm start 1
vm start 2
```

Do not invent a large configuration language in this slice.

---

## 9. VM-originated control requests

Eventually WarpVM programs should be able to ask the CPU supervisor to perform the same kinds of orchestration that a human or startup script can request.

Examples:

```text
create another VM running worker.wvm
stop VM 17
reset VM 23
request compiled execution for VM 8
query VM 9 status
delete a finished child VM
```

These are **CPU-side supervisor operations**, not new WarpVM process-management opcodes.

A WarpVM program should submit a structured request through a host-control channel, for example using the existing VM-to-host communication mechanism or a reserved runtime control queue/mailbox.

Conceptually:

```text
VM program
    |
    | control request
    v
CPU supervisor
    |
    | performs lifecycle/program operation
    v
GPU runtime
    |
    | result / acknowledgement
    v
requesting VM
```

The exact binary request format may be small and fixed-width initially.

The important design rule is:

> Human commands, startup scripts and VM-originated requests should converge on the same supervisor operations rather than implementing three separate orchestration systems.

Do not add arbitrary host execution or a general syscall layer in this slice.

---

## 10. GPU/runtime changes

The persistent runtime must stop assuming one common program image for every VM.

At minimum, each VM needs enough program identity to fetch from the correct image.

Possible implementation:

```text
VmDesc
    ... existing state ...
    program_id
```

with a program table containing:

```text
ProgramDesc
    bytecode_base
    bytecode_words
    compiled_id / compiled entry metadata
```

The interpreter then conceptually performs:

```text
program = program_table[vm.program_id]
instruction = program.bytecode_base[vm.pc]
```

rather than indexing one global bytecode image.

Multiple VM descriptors may reference the same `ProgramDesc`.

Do not duplicate bytecode per VM merely to obtain program independence.

---

## 11. Compiled-program dispatch

Compiled execution must ultimately obey the same program/VM model.

The implementation mechanism is deliberately not fixed by this specification.

Possible implementations include:

- a resident native dispatch table over loaded compiled programs;
- generated resident code containing the currently compiled program set;
- multiple native kernels managed beneath one supervisor;
- a hybrid interpreter/native dispatcher.

The architectural requirement is stronger than any one implementation:

> A compiled implementation belongs to a program; a VM chooses an execution engine; neither changes VM identity.

Avoid exposing implementation constraints such as one global `--compiled` population mode as the long-term API.

The host supervisor may internally rebuild, cache, select or dispatch compiled implementations as necessary.

---

## 12. Scheduling and stop semantics

The existing cooperative persistent-kernel model remains valid.

`vm_stop()` means the CPU supervisor requests a cooperative stop and waits until the VM reaches a runtime-observable safe state.

Do not pretend that WarpVM has CPU-style hard preemption if the GPU/runtime cannot provide it.

The supervisor should distinguish:

```text
stop requested
stopped/acknowledged
```

where necessary.

Deleting or resetting a running VM must first bring it to a safe stopped state.

---

## 13. First demonstration

The visual demonstration should make heterogeneous programs undeniable.

Run at least four simultaneous VM instances executing visibly different programs, for example:

```text
VM 0 -> plasma
VM 1 -> mandelbrot
VM 2 -> cellular automaton / wave simulation
VM 3 -> messaging or another graphics workload
```

Display their private 128x128 framebuffers together in the viewer.

Acceptance criteria:

- all four programs are loaded once into the program registry;
- each VM references its assigned program independently;
- each VM retains independent RAM/registers/PC/framebuffer/mailboxes;
- stopping one VM does not stop the others;
- resetting/deleting one VM does not corrupt the others;
- two VMs may share one program while retaining independent state;
- interpreter execution works with heterogeneous programs;
- existing same-program-many-VM behaviour remains valid.

A later acceptance step should demonstrate mixed engines, e.g.:

```text
VM 0 plasma      compiled
VM 1 mandelbrot  interpreted
VM 2 gossip      compiled
VM 3 gossip      interpreted
```

while all remain part of the same resident WarpVM population.

---

## 14. Suggested implementation sequence

### Slice A — Program registry

- load multiple `.wvm` images;
- assign stable program IDs;
- share one image across many VMs;
- unload safely.

### Slice B — Per-VM program identity

- add `program_id` / program reference to VM state;
- interpreter fetches through the VM's program;
- run heterogeneous interpreted VMs concurrently.

### Slice C — CPU supervisor lifecycle

- create/delete;
- start/stop/resume/reset;
- status/list;
- preserve state correctly across stop/resume.

### Slice D — Startup script

- drive the same supervisor API from a simple command file;
- reproduce a heterogeneous population deterministically.

### Slice E — Per-VM execution engine

- make engine selection VM-local runtime metadata;
- support heterogeneous compiled programs;
- support interpreted and compiled VMs simultaneously;
- preserve canonical `.wvm` semantics and execution equivalence.

### Slice F — VM-originated supervisor requests

- define a small structured control-request protocol;
- route requests to the same CPU supervisor API;
- return acknowledgements/results to the requesting VM;
- demonstrate a controller VM creating/stopping another VM.

Keep each slice independently testable.

---

## 15. Tests

Add regressions for at least:

1. two VMs executing different `.wvm` programs;
2. two VMs sharing the same `.wvm` program;
3. independent PCs for different programs;
4. independent RAM/register state;
5. independent framebuffer state;
6. independent HALT/FAULT states;
7. stop one VM while another continues;
8. reset one VM without disturbing another;
9. delete/recreate a slot;
10. reject unloading a referenced program;
11. startup-script population creation;
12. existing homogeneous `--vms N` behaviour;
13. interpreter/compiled equivalence once heterogeneous compiled execution lands;
14. mixed interpreted/compiled population once supported;
15. VM-originated control request round-trip once that slice lands.

The full existing test suite must remain green after every sub-slice.

---

## 16. Explicit non-goals for 01.06

Do not turn this into a conventional operating system.

Not required:

- scheduler priorities;
- users/permissions;
- filesystems inside WarpVM;
- virtual memory;
- processes within one WarpVM;
- POSIX APIs;
- arbitrary host syscalls;
- dynamic linker;
- security isolation between hostile programs;
- GPU-side process-management ISA instructions;
- sophisticated startup/configuration language.

The CPU already exists and is the natural place for orchestration.

---

## 17. Architectural outcome

After 01.06, the important mental model should be:

```text
CPU SUPERVISOR

Programs:
    P0 plasma.wvm       [WVM + optional native implementation]
    P1 gossip.wvm       [WVM + optional native implementation]
    P2 mandelbrot.wvm   [WVM + optional native implementation]

VM population:
    VM0 -> P0 -> RUNNING -> COMPILED
    VM1 -> P1 -> RUNNING -> INTERPRETED
    VM2 -> P2 -> STOPPED -> INTERPRETED
    VM3 -> P1 -> RUNNING -> COMPILED
    VM4 -> EMPTY
```

The supervisor controls the population from the CPU.

The WarpVMs remain GPU-native computers with independent architectural state.

Programs may be shared; machines are not.

Execution engines may differ; machine identity does not.

That separation is the foundation for later controller/worker populations, evolutionary systems, self-managing experiments and host-assisted dynamic compilation without turning WarpVM itself into a large operating-system architecture.
