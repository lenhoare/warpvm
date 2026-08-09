# WarpVM Architecture

Living document. Records runtime decisions as the slices land.
The instruction contract lives in [isa.md](isa.md).

## Execution model

- One persistent CUDA kernel hosts all VMs. One warp = one VM; thread-block
  geometry is a multiple of 32 so no VM ever straddles a hardware warp.
- All warps run the **same interpreter code**; each fetches from its own
  program. Within a warp the opcode stream is uniform, so fetch/decode is
  divergence-free. Divergence between warps is free (independent scheduling).
- The interpreter keeps vector register slices in per-thread registers
  (lane *i* owns element *i* of every vector register). Scalar registers and
  predicate masks are uniform values replicated in every lane; they stay in
  sync by construction because all lanes execute the same uniform operations.

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

Every interpreter iteration checks the VM's command word only when the VM
reaches a control point (`HALT`, `YIELD`, debug request). Running VMs are
not hard-preempted; cooperation points are frequent enough for interactive
control.

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

Fixed 16-byte messages (`src_vm`, `msg_type`, 3×u32 payload). Each VM has a
16-slot inbound ring; `SEND` claims a slot with an atomic on the
destination's `head`. v0.1: mailbox full ⇒ `FAULT_MSG` (no blocking).

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
