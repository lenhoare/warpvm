# WarpVM Project Spec 01.07 — Stable VM Identity and Heterogeneous Compiled Populations

## 1. Purpose

This document records two architectural decisions that refine the CPU
supervisor design in `project_spec_01_06.md`:

1. a WarpVM needs a stable logical identity that is separate from its current
   resident GPU slot and physical warp placement;
2. heterogeneous compiled programs should ultimately be able to coexist in a
   resident population, with a combined compiled kernel as the leading design
   to investigate.

These decisions preserve the central WarpVM model:

> The GPU hosts a population of persistent, independently addressable
> computers, not a batch of interchangeable work items.

This is an architectural addendum and direction-setting document. The
heterogeneous interpreted supervisor should still be implemented before the
compiled-population design is finalized.

---

## 2. Separate the four identities

The runtime must not conflate logical machine identity, resident storage,
physical GPU placement and installed program.

| Concept | Meaning | Lifetime |
|---|---|---|
| logical VM ID | Stable architectural address used by `WARP`, messaging and supervisor commands | Entire logical VM lifetime |
| resident slot ID | Index into device state, mailbox, framebuffer and control arrays | Recyclable implementation detail |
| physical CUDA warp | The GPU warp currently executing a resident slot | Transient and controlled by CUDA |
| program ID | Program currently installed in the logical VM | Replaceable under supervisor control |

Only the logical VM ID is a machine address.

A program must never need to know its resident array index, block number, SM,
or physical warp placement.

---

## 3. Stable logical VM addresses

Programs communicate with peers by logical VM ID.

If a program learns that its peer is VM 37, messages sent to address 37 must
continue to refer to that same logical computer for its entire lifetime.

Deleting VM 37 must not allow an unrelated future VM occupying the same
resident slot to receive messages intended for VM 37.

The initial supervisor policy should therefore be:

- allocate logical VM IDs monotonically;
- do not reuse an ID during one supervisor lifetime;
- preserve the ID across stop, resume and reset;
- preserve the ID when deliberately installing a new program in the same
  logical VM;
- retire the ID permanently when the logical VM is deleted;
- give a newly created VM a new ID even if it reuses the deleted VM's resident
  slot.

"Permanent" in this first design means for the lifetime of the running
supervisor universe. Persistent identities across supervisor restarts can be
added later through stable names or UUIDs resolved to architectural VM IDs.

---

## 4. Current addressing constraint

The current message representation stores the sender VM ID in the low 16 bits
of the message header. `SEND` accepts a 32-bit destination value, but current
execution validates and uses it directly as a mailbox-array index.

The narrowest compatible supervisor design therefore uses:

```text
logical VM address space:       0 .. 65535
simultaneously resident VMs:    currently at most 256
resident slot space:            0 .. configured_capacity-1
```

This provides 65,536 logical identities per supervisor lifetime without an ISA
or message-format change.

If exhaustion ever becomes realistic, address width and persistent namespaces
can be revisited deliberately. It should not be solved prematurely by reusing
addresses and weakening message safety.

---

## 5. Logical-to-resident routing

The supervisor owns a routing directory:

```text
logical_vm_id -> resident_slot_id | INVALID
```

The device needs a read-only or supervisor-updated representation of the same
mapping for message delivery.

Execution then behaves as follows:

- `WARP` returns the stable logical VM ID;
- `SEND` resolves its logical destination through the routing directory;
- an invalid or retired destination follows the defined message-failure
  behaviour;
- `TRY_RECV` consumes the mailbox belonging to the executing resident slot;
- a transmitted message records the sender's logical VM ID;
- framebuffer, RAM, state and control arrays remain indexed by resident slot;
- CLI, viewer and inspection operations accept logical VM IDs and resolve them
  internally.

The extra routing-table read occurs only when sending a message. It does not
add dispatch work to normal VM instructions and is small relative to the
atomic mailbox operation itself.

Interpreter and compiled execution must use precisely the same routing and
message semantics.

---

## 6. Resident capacity remains explicit

CUDA cannot add another warp to a kernel that is already running.

The supervisor should therefore launch a configured resident slot capacity and
manage logical VMs within that capacity.

Initially:

- an empty resident slot may be assigned to a newly created logical VM;
- deleting a logical VM returns its slot to the empty pool but retires its
  logical ID;
- creating a new VM in that slot assigns a fresh logical ID;
- growing beyond the configured capacity requires a controlled population
  quiesce and relaunch;
- the relaunch must preserve all live VM state that the operation promises to
  preserve.

The configured capacity should reflect the intended population. WarpVM should
not launch hundreds of permanently idle resident warps by default because idle
warps still consume scheduling and residency resources.

Physical placement remains irrelevant. A logical VM may execute on a different
physical CUDA warp after a relaunch without changing identity.

---

## 7. Program replacement without identity replacement

Stable VM identity is particularly important for evolutionary and genetic
programming workloads.

A logical computer may retain its network identity while its installed program
changes:

```text
logical VM 37
    address:       preserved
    resident slot: may be preserved or moved
    program:       generation N -> generation N+1
    entry PC:      reset to the new program's entry point
```

Program replacement must occur only through a safe supervisor transition,
normally while the VM is stopped.

Installing a different program cannot preserve the old bytecode PC: instruction
addresses in unrelated programs have no common meaning. The initial operation
should therefore be a cold program rebind that resets execution state and
starts at the new entry point.

The policy for other state must be explicit. Useful eventual forms may include:

- cold replacement: clear registers, stack, RAM, framebuffer and mailbox;
- lineage replacement: reset execution state but preserve selected RAM;
- explicit mailbox preservation or clearing.

The first implementation should choose one simple, safe default rather than
implicitly preserving stale state. Clearing the mailbox on a cold rebind avoids
delivering requests meant for the previous program incarnation.

Stop/resume of the same program remains a different operation and preserves the
machine exactly.

---

## 8. Why one heterogeneous compiled kernel is promising

The current resident compiled engine produces a CUDA kernel specialized for one
WVM program and launches that program across its VM population.

The leading design for compiled heterogeneity is to generate one resident
kernel containing all active compiled programs and perform one warp-uniform
program selection:

```text
slot = resident_slot()
program = slot_program_id[slot]

switch program:
    PLASMA:       enter compiled plasma body
    MANDELBROT:   enter compiled mandelbrot body
    CREATURE_A:   enter compiled creature A body
    CREATURE_B:   enter compiled creature B body
```

This does not turn compiled execution back into interpretation.

The selected program ID is uniform across the 32 lanes of a WarpVM, so:

- the program switch causes no within-warp divergence;
- different physical warps may select different compiled bodies;
- the selection can occur when a VM enters or resumes its program, rather than
  once per VM instruction;
- after selection, the warp executes normal compiled native code;
- each compiled program uses the same architectural state, RAM, framebuffer,
  messaging and control ABI.

Independent program counters across physical warps are normal CUDA execution.
The fact that two warps in one block enter different program bodies is not a
semantic problem.

---

## 9. Why a combined kernel may be preferable to one kernel per program

Launching a separate persistent CUDA kernel for every program looks simple but
has an important risk: an earlier persistent kernel may occupy enough device
resources to prevent later kernels from becoming resident.

A combined population kernel offers:

- one residency plan for the whole population;
- one global resident-slot namespace;
- straightforward shared messaging;
- direct use of logical-to-slot routing;
- no competition between independently persistent kernel grids;
- a single point for cooperative stop, inspection and relaunch.

This is not yet a final implementation mandate. It is the strongest design to
prototype once heterogeneous interpreted execution and supervisor lifecycle are
stable.

---

## 10. Compiled population epochs

Genetic workloads naturally produce changing sets of programs. WarpVM can
support this through compiled population epochs:

```text
generation N executes
        |
        +-- CPU prepares and compiles generation N+1
        |
generation boundary
        |
        +-- stop affected VMs
        +-- install new heterogeneous compiled kernel
        +-- rebind programs while preserving logical VM identities
        +-- relaunch population
```

Compilation should ideally happen asynchronously while the current generation
continues to run.

Compiled artifacts should be cached by program-content hash. If several VMs run
the same program, that compiled body should appear only once in the generated
population kernel.

The registry should distinguish:

- program identity and metadata;
- immutable WVM image;
- cached compiled program body;
- membership in a particular population-kernel epoch.

Rebuilding or replacing a population kernel changes the execution
implementation, not the logical identity of any VM.

---

## 11. Scale limits and fallback designs

A combined kernel is credible for 64 relatively small programs, but it is not
assumed to scale without limit.

Potential pressure points include:

- generated PTX and native code size;
- JIT compilation latency;
- instruction-cache locality when many unrelated programs run concurrently;
- relocation and label-management complexity;
- large switch or dispatch-tree cost.

If measurement shows these becoming significant, possible refinements include:

- group programs into a small number of population kernels;
- group by program family or code locality;
- build one kernel per evolutionary cohort;
- batch newly mutated programs into the next kernel epoch;
- retain interpretation temporarily for programs awaiting compilation;
- cache and reuse previously generated population combinations where useful.

These are optimization choices. None require abandoning stable logical VM
identity or the heterogeneous resident-machine model.

---

## 12. Mixed interpreted and compiled execution

The architectural requirement remains that execution engine is independent of
VM identity.

Eventually a population may contain:

```text
VM 11 -> program A -> interpreted
VM 12 -> program B -> compiled
VM 13 -> program C -> compiled
VM 14 -> program D -> interpreted
```

The exact CUDA launch structure for this mixed population should be chosen from
evidence. Separate persistent kernels can have residency hazards, while a
single combined implementation requires a common generated/runtime entry path.

This question should not delay the initial heterogeneous interpreted
supervisor. The supervisor data model must represent the desired engine from
the beginning, but unsupported combinations may be rejected clearly until the
compiled slice lands.

---

## 13. Required supervisor invariants

The supervisor should enforce:

1. logical VM IDs are opaque architectural addresses, not slot indices;
2. an active logical ID maps to at most one resident slot;
3. a resident slot hosts at most one logical VM;
4. retired logical IDs never become valid again during the supervisor lifetime;
5. `WARP` and message metadata expose logical IDs only;
6. device arrays and physical execution placement expose slot IDs only to the
   runtime;
7. program replacement never changes logical VM identity;
8. changing to an unrelated program restarts at the new program's entry point;
9. interpreter and compiled engines observe identical VM and message identity;
10. a failed route can never fall through to a recycled slot;
11. population relaunch may change physical placement but not architectural
    identity;
12. unloading a program remains forbidden while any VM or active compiled
    population artifact references it.

---

## 14. Suggested implementation order

This document does not replace the slices in `project_spec_01_06.md`; it sharpens
their order and identity model.

### Slice A — Identity types and supervisor model

- introduce distinct logical VM ID, resident slot ID and program ID types;
- add the host-side logical-to-slot directory;
- allocate monotonically increasing logical IDs;
- represent retired and invalid routes;
- test allocation, deletion and non-reuse without involving CUDA execution.

### Slice B — Shared program registry and heterogeneous interpreted population

- upload each immutable program image once;
- allow resident descriptors to point to different shared program images;
- bind each active slot to one logical VM and one program;
- make the existing homogeneous `--vms N` path use the same machinery;
- run the first four-program tiled demonstration.

### Slice C — Architectural identity in execution and messaging

- split runtime `slot_id` from architectural `vm_id`;
- make `WARP` return logical identity;
- route `SEND` through the device directory;
- keep receive mailboxes indexed by resident slot;
- preserve logical sender IDs in message metadata;
- add stale-route and recycled-slot regression tests.

### Slice D — Lifecycle, capacity and safe program rebind

- create and delete logical VMs within configured resident capacity;
- stop, resume and reset independently;
- define cold program-replacement state clearing;
- preserve identity across deliberate program replacement;
- support controlled quiesce/relaunch for capacity growth later.

### Slice E — Long-lived CLI, startup script and supervisor-aware viewer

- send program and VM commands to a persistent supervisor;
- address machines by logical VM ID;
- display logical VM ID, slot, program and lifecycle separately;
- keep framebuffer tiles associated with logical machines even if slots move.

### Slice F — Heterogeneous compiled-kernel prototype

- compile several WVM programs into one resident kernel;
- perform one warp-uniform program dispatch;
- use global resident slots and stable logical VM routing;
- measure JIT time, code size, instruction-cache behaviour and throughput;
- compare against homogeneous compiled execution.

### Slice G — Compiled population epochs and genetic-program replacement

- cache program bodies by content hash;
- compile the next population asynchronously where practical;
- stop and rebind at a safe generation boundary;
- preserve logical identities while installing new genetic programs;
- verify messaging continuity and selected-state preservation policy.

### Slice H — Mixed engines and VM-originated supervisor control

- select the mixed-engine launch structure from prototype evidence;
- retain identical architectural behaviour across engines;
- add the structured host request protocol described by the supervisor design.

---

## 15. Essential regression tests

Identity and routing tests should include:

1. `WARP` returns logical VM ID rather than resident slot;
2. two VMs in different slots exchange messages using logical addresses;
3. delete a VM, recycle its slot, and prove a message to the retired ID cannot
   reach the replacement;
4. stop/resume preserves identity and pending state;
5. reset preserves identity while applying the documented reset policy;
6. program replacement preserves identity and starts at the new entry point;
7. sender metadata contains logical identity;
8. interpreter and compiled messaging produce identical results;
9. viewer and CLI continue addressing the same logical VM after a slot move;
10. capacity exhaustion reports a supervisor resource error rather than
    silently reusing an identity.

The heterogeneous compiled prototype should verify:

1. at least four distinct compiled programs execute concurrently;
2. each physical warp selects exactly its assigned program;
3. the initial switch is warp-uniform;
4. shared program bodies are emitted once;
5. RAM and framebuffer remain isolated by resident slot;
6. messaging uses logical identity across different compiled program bodies;
7. stop/resume and fault reporting remain per VM;
8. the homogeneous compiled path does not regress materially;
9. logical identity survives a compiled population-kernel replacement;
10. results match the interpreted engine.

---

## 16. Architectural outcome

The intended population model is:

```text
logical address directory

    VM 37 -> slot 2 -> program creature_17 -> COMPILED
    VM 41 -> slot 0 -> program plasma      -> COMPILED
    VM 52 -> slot 3 -> program controller  -> INTERPRETED
    VM 63 -> slot 1 -> program creature_17 -> COMPILED

physical CUDA placement: deliberately unspecified
```

If VM 37 receives a mutated program in the next generation, it remains VM 37.
If VM 37 is deleted and slot 2 is later reused, the new machine receives a new
logical address and cannot receive messages sent to 37.

Compiled heterogeneity is expected to be implemented by selecting a compiled
program body once per warp within a resident population kernel, subject to
measurement and prototype validation.

The resulting principle is:

> A WarpVM's address belongs to the logical computer, not to a GPU array index,
> physical warp, program image or compiled-kernel incarnation.

That separation supports long-lived peer relationships, changing genetic
programs, safe slot reuse and genuinely heterogeneous resident computation.
