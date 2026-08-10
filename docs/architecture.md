# WarpVM Architecture

Living document. Records runtime decisions as the slices land.
The instruction contract lives in [isa.md](isa.md).

## Execution model

- One persistent CUDA kernel hosts all VMs. One warp = one VM; thread-block
  geometry is a multiple of 32 so no VM ever straddles a hardware warp.
  Blocks of 256 threads (8 VMs) where the VM count allows, single-warp
  blocks otherwise.
- Logical `vm_id` is the warp's slot index into the descriptor/state arrays.
  Nothing in the VM's identity or state references SM or block placement.
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
host → GPU : per-VM command word (run/pause/resume/halt/reset/step/exit),
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

- VM RAM: private global-memory region per VM, word-addressed. All LOAD/
  STORE bounds-check against `[0, mem_size_words)`; violations fault the VM.
- VM state blocks: structure-of-arrays layout preferred where it improves
  coalescing (spill/reload, status scans).

## Messaging

Fixed 16-byte messages. The header packs `src_vm` (low 16 bits) and
`msg_type` (high 16 bits); of the 3 payload words, v0.1 `SEND` transmits only
`payload[0]` (`payload[1..2]` reserved). Each VM has a 16-slot inbound ring in
a global mailbox array: `SEND` claims a slot with an atomic on the
destination's `head`, the owner consumes from `tail`. v0.1: mailbox full ⇒
`FAULT_MSG` (no blocking). `TRY_RECV` is non-blocking and returns a
got-message predicate.

The claim-then-write / read-then-advance scheme has a small producer/consumer
race window (a receiver could read a slot a sender is still filling). v0.1
accepts this for cooperative traffic; a two-phase commit or per-slot flag is a
later hardening if messaging becomes load-bearing.

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
  of the ISA. The SDL window viewer is deferred; the automated path is
  headless (assemble → `.wvm` → resident runtime → host copy → pixel checks).

## Scheduling

Logical `vm_id` is stable; physical warp/SM placement is not part of VM
identity. v0.1 does not migrate VMs, but state layout must not preclude it:
everything a VM needs (program, RAM, state, mailbox) is reachable by
pointer from the VM state block.

## Display-GPU constraint

Development runs on the display GPU (RTX 3060). Consequences:

- runs are bounded by default (iteration budgets, explicit `--forever` opt-in);
- kernel teardown must always be reachable via the global shutdown flag;
- avoid occupying all SMs in interactive demos.
