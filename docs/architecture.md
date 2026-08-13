# WarpVM Architecture

Living document. Records runtime decisions as the slices land.
The instruction contract lives in [isa.md](isa.md).

## Execution model

- One persistent CUDA kernel hosts all VMs. One warp = one VM; thread-block
  geometry is a multiple of 32 so no VM ever straddles a hardware warp.
  Blocks of 256 threads (8 VMs) where the VM count allows, single-warp
  blocks otherwise.
- Logical `vm_id` is a stable architectural address and is independent of the
  resident slot indexing descriptor/state arrays. A device routing directory
  maps logical IDs to current slots. Nothing in VM identity or state references
  SM, block, physical-warp, or resident-slot placement.
- All warps run the **same interpreter code**; each fetches from its own
  program. Within a warp the opcode stream is uniform, so fetch/decode is
  divergence-free. Divergence between warps is free (independent scheduling).
- The interpreter keeps vector register slices in per-thread registers
  (lane *i* owns element *i* of every vector register). Scalar registers and
  predicate masks are uniform values replicated in every lane; they stay in
  sync by construction because all lanes execute the same uniform operations.

## Fault uniformity

Per-lane conditions (e.g. one lane's LOAD address out of bounds while other
lanes are fine) are resolved to a warp-uniform decision with
`__ballot_sync`: if any lane faults, the whole VM faults, taking the lowest
faulting lane's code. This keeps the fetch loop converged even with
scattered per-lane accesses — essential for GP-mutated programs. Partial
effects a dying VM already wrote into its own RAM are tolerated; effects on
other VMs are impossible by construction (disjoint RAM regions).

## Host ↔ kernel control channel

The persistent kernel never exits until the host asks. Communication uses
**pinned, mapped host memory** (`cudaHostAllocMapped`):

```text
host → GPU : per-VM command word (run/pause/resume/reset/step/deactivate/exit),
             global shutdown flag, program reload descriptors
GPU → host : per-VM status word, fault code, pc, instruction counter,
             log ring
```

GPU-side reads of mapped memory are plain loads over PCIe — acceptable at
control-plane rates. Data-plane transfers (program load, snapshots) use
regular `cudaMemcpy` while the target VM is held at a control point.

Two constraints learned the hard way:

- **Launch on a non-blocking stream.** The resident kernel must be launched on
  a stream created with `cudaStreamNonBlocking`. If it runs on the legacy
  default stream, every host `cudaMemcpy`/read implicitly waits on the kernel
  and deadlocks (the kernel never finishes). Non-blocking streams are exempt
  from legacy default-stream synchronisation, so inspection copies proceed
  while the kernel stays resident.
- **The command channel is one word per VM.** A new command overwrites an
  unconsumed one, so the host must wait for each command to take effect
  (status transition or the per-VM `seq` counter) before sending the next. At
  boot, the runtime waits for every VM to leave `IDLE` before accepting
  further commands.

Every interpreter iteration checks the VM's command word only when the VM
reaches a control point (`HALT`, `YIELD`, backward branch). Running VMs are
not hard-preempted; cooperation points are frequent enough for interactive
control.

## Attach and single-step

`warpvm attach` boots resident VMs and drives one through a console:
`pause / step / resume / reset / regs / sregs / mem / pc / disasm / log`.

With `attach ... --compiled`, inspection, pause/resume, reset, memory,
framebuffer, and status use the same console. Single-instruction stepping is
currently interpreter-only.

- Inspection reads the VM's spilled `VmState` and its private RAM with
  `cudaMemcpy` while the kernel is resident (valid because a stopped VM's
  state is stable).
- **Single-step**: the host sends `kCmdStep` to a paused VM. The warp sets a
  `step` flag and re-enters the interpreter, which retires exactly one
  instruction and re-pauses at the new pc. Because the status stays `PAUSED`
  across a step, the warp bumps a per-VM `seq` counter when the step
  completes; the host waits on `seq` to detect completion.

## Inspection: spill-on-pause

Register state lives in per-thread registers, invisible to the host. When a
VM enters `PAUSED`/`DEBUG`/`HALTED`/`FAULTED`, the warp **spills** its
register file, predicates, pc and call stack into the VM's global-memory
state block, then spins on its command word. On `RESUME`/`RUN` it reloads
state and continues. This makes `attach` (regs/mem/pc/step) possible
without terminating the kernel.

## Memory

- VM RAM: private global-memory region per VM, word-addressed. The standard
  default is 65,536 words (256 KiB) per VM. All LOAD/STORE bounds-check
  against `[0, mem_size_words)`; violations fault the VM.
- VM state blocks: structure-of-arrays layout preferred where it improves
  coalescing (spill/reload, status scans).

## Program registry and heterogeneous populations

The CPU-side `ProgramRegistry` assigns stable, non-reused program IDs to
immutable WVM code/literal images. Loading a program and creating a VM are
separate operations. Registry reference counts reject unload while VM
definitions still reference an image.

When a resident population is initialized, each referenced registry program is
uploaded to the GPU exactly once. Multiple resident `VmDesc` entries may point
to that shared code/literal allocation, while RAM, architectural state,
framebuffer and mailbox remain private to each resident slot. Consequently two
VMs may share a program without sharing machine state, and two VMs at the same
PC may fetch unrelated instructions from different programs.

Legacy homogeneous commands use the same path: identical code/literal images
are interned into one program allocation and slots receive logical IDs `0..N-1`
for compatibility. Static heterogeneous bindings are interpreted in this
slice; compiled program selection remains a later population-kernel feature.

The host list view exposes logical VM ID, resident slot and program name as
separate columns. `hetero_view` loads a list of WVM paths into one registry and
shows their private framebuffers in one tiled viewer.

## Supervisor lifecycle and resident capacity

The CPU `Supervisor` launches a configured number of resident slots once. Each
slot's RAM, framebuffer, state, mailbox and control storage is allocated before
the persistent kernel starts; an empty slot's warp remains idle and consumes no
program identity. CUDA cannot grow this population in place, so exceeding the
configured capacity is an explicit resource error and later capacity growth
will use a controlled quiesce/relaunch.

The lifecycle is `EMPTY -> READY -> RUNNING`, with cooperative
`RUNNING -> STOPPED -> RUNNING`, autonomous `HALTED`/`FAULTED`, reset back to
`READY`, and deletion back to `EMPTY`. Stop/resume preserves the complete
machine. Reset restores registers, control state, initial RAM, framebuffer and
mailbox. Cold program replacement uses the same safe transition, starts at PC
zero with cleared machine state, and preserves the logical VM ID.

Deletion does not terminate a resident CUDA warp. A `DEACTIVATE` command makes
the target warp withdraw its logical route, drain in-flight mailbox producers,
clear the mailbox, acknowledge through the control sequence, and return to its
idle command loop. The host can then rebind the descriptor to a new program and
logical ID; the next cold `RUN` reloads both rather than using stale values
cached when the kernel launched. Other VMs continue running during the entire
transition.

Programs used by this initial resident epoch are loaded before launch and their
device code allocations remain until shutdown. Program unload is still allowed
after its last VM reference disappears; adding new device program images while
the kernel is resident is deferred to the controlled population-epoch work.

### Long-lived supervisor frontend

`warpvm supervise` is a line-oriented frontend over the same `Supervisor` C++
API used by tests. Interactive input and startup files share one parser and one
operation implementation; a failed startup-file command reports its line and
aborts instead of leaving a partially configured unattended population.

The frontend separates its phases explicitly: load immutable program objects,
`launch <capacity>`, then create and operate logical VMs. Commands cover
create/delete/start/stop/resume/reset, cold program replacement, engine
metadata, deterministic state waits, and existing register/RAM/disassembly/
framebuffer/log inspection. All VM operands are logical IDs and are resolved to
slots at the point of use.

The supervisor viewer snapshots the requested logical IDs as tile identities,
but resolves each ID through the live route directory on every refresh. A
logical machine therefore remains attached to its tile if its resident slot
changes; deletion blanks that logical tile rather than showing an unrelated VM
which later occupies the slot. Closing the viewer returns to the command loop
without stopping the resident population.

## Messaging

Fixed 16-byte messages. The header packs `src_vm` (low 16 bits) and
`msg_type` (high 16 bits); of the 3 payload words, v0.1 `SEND` transmits only
`payload[0]` (`payload[1..2]` reserved). Each VM has a 16-slot inbound ring in
a global mailbox array. Each slot carries a publication sequence: a producer
atomically reserves `head`, writes the complete message, performs a device
release fence, then publishes the slot sequence. The single owner observes
that sequence before reading and releases the slot for the next ring cycle.
This prevents both over-reservation and observing a slot before its message is
complete. v0.1: mailbox full ⇒ `FAULT_MSG` (no blocking). `TRY_RECV` is
non-blocking and returns a
got-message predicate.

Each mailbox also carries its logical owner and an in-flight producer count.
`SEND` pins the mailbox and verifies that the owner still equals its requested
logical destination before reserving a ring entry. Slot retirement invalidates
the route and owner and waits for this count to reach zero before clearing the
ring. A sender that raced with route withdrawal therefore either completes for
the old owner before recycling or detects the owner mismatch and faults; it can
never publish into the replacement VM's mailbox. The interpreted and generated
PTX implementations use the same protocol.

Message destinations and `src_vm` are stable logical VM IDs, not resident
mailbox-array indices. `SEND` resolves the destination through the supervisor's
device routing directory and then writes the mailbox belonging to that resident
slot. `TRY_RECV` consumes the executing slot's own mailbox. A retired or absent
logical ID has an invalid route and faults with `FAULT_MSG`; recycling its old
slot cannot redirect stale messages to the replacement VM. The v0.1 header
therefore defines a 65,536-address logical namespace, independently of the
currently configured resident capacity.

## Graphics (v0.1.1)

Each VM owns a 128×128×32-bit framebuffer (16,384 words). Physical storage is
one flat device allocation (`num_vms × kVideoWords`), exposed logically
through a fixed memory-mapped region (`VIDEO_BASE = 0x00100000`) decoded in
`LOAD`/`STORE`. The ABI never reveals the physical layout.

- **Drawing is warp-native**: per-lane addresses let one predicated `STORE`
  write up to 32 pixels. No `DRAW`/`PIXEL` opcode exists — pixels are memory.
- **Publication**: `FLIP` bumps the VM's `frame_seq` (in the mapped Control
  block, like `status`/`instrs`) exactly once, non-blockingly. The host polls
  `frame_seq` to find new frames. No double buffering in v0.1.1 — a host copy
  may observe early next-frame writes (acceptable; tearing is a later concern).
- **Reset clears only the reset VM's framebuffer** to opaque black; the clear
  runs device-side in `InitVmCtx` (all 32 lanes cooperate). Pause/resume
  preserves contents.
- **Inspection**: `ReadFramebuffer` copies a VM's framebuffer to the host
  (`cudaMemcpy` on the non-blocking stream, so it doesn't deadlock against the
  resident kernel). The attach console gains `frame` (resolution/format/seq)
  and `pixel <x> <y>`.
- **Presentation is a host concern** (windows, scaling, grids) and stays out
  of the ISA. The SDL viewer supports an enlarged single-VM view and a tiled
  multi-VM view. Per-VM `frame_seq` changes trigger one contiguous pool copy
  and one composed-atlas upload, rather than separate device and SDL transfers
  for every VM. Nearest-neighbour sampling and the framebuffer aspect ratio
  are preserved as the window is resized. Automated pixel validation remains headless
  (assemble → `.wvm` → resident runtime → host copy → pixel checks).

## Scheduling

Logical `vm_id` is stable; resident slot and physical warp/SM placement are not
part of VM identity. Resident slots are recyclable implementation resources,
but a retired logical ID is not reused during a supervisor epoch. v0.1 does not
yet migrate a running VM, but state layout must not preclude it: everything a
VM needs (program, RAM, state, mailbox) is reachable through its resident slot
and supervisor metadata.

The direct PTX engine has the same resident topology. Generated program code
runs continuously in one warp per VM and receives pointers to the canonical
descriptor/state array, VM RAM, framebuffer, mapped control plane, and shared
mailbox pool. Backward branches and explicit `YIELD` instructions are safe
control points; they publish live PC/instruction progress and observe
pause/shutdown without ending normal execution. `FLIP`, `SEND`, and
`TRY_RECV` operate directly on resident device state.

Pausing spills registers and architectural state, then waits inside the same
resident kernel. Resume continues from the saved PC; reset reconstructs the
power-on register state and clears the framebuffer. HALT and faults publish
their terminal state but the warp remains available for reset/run/exit until
the runtime shuts down. Host/device transfers are therefore inspection and
presentation operations, not the VM scheduler. The older checkpoint launcher
remains only a bounded equivalence and benchmark harness.

## Display-GPU constraint

Development runs on the display GPU (RTX 3060). Consequences:

- runs are bounded by default (iteration budgets, explicit `--forever` opt-in);
- kernel teardown must always be reachable via the global shutdown flag;
- avoid occupying all SMs in interactive demos.
