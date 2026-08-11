# WarpVM Architectural Findings

This is a short evidence log for issues discovered while writing real WarpVM
programs. It is not an ISA wishlist. Each entry records the program that
exposed the issue, the concrete assembly required today, and a possible
general remedy to evaluate only after measuring the workaround.

## 1. Predicate masks cannot be stored as numeric values

**Discovered by:** Program 01, WarpLife

**Use case:** Thirty-two lanes calculate the next state of 32 adjacent Life
cells. The resulting lane mask is exactly the packed 32-bit word required by
the next-generation buffer.

The implemented instruction is:

```text
BALLOT p0, rState
```

It writes the mask to predicate register `p0`. Predicate registers can guard
instructions, combine with other predicates, and control jumps, but their
32-bit mask cannot be moved into a vector or scalar register. Consequently it
cannot be passed to `STORE`.

Current correctness workaround:

```text
BALLOT      p0, rState
MOV_I       rBits, 0
@p0 MOV_I   rBits, 1
SHL         rBits, rBits, rLane
REDUCE_OR   rPacked, rBits
@pLane0 STORE rAddress, rPacked
```

The workaround is general and preserves the current ISA, but packing costs
four instructions after `BALLOT` instead of making the ballot result directly
available as a value. WarpLife executes this sequence once per packed word:
512 times, or 2,048 workaround instructions, per generation.

Possible general remedies to evaluate later:

- make `BALLOT` write its uniform 32-bit result to a vector register;
- add a predicate-to-vector/scalar mask move;
- support a typed cross-register-class move that covers this and future
  predicate-mask consumers.

Do not choose among these until WarpLife is correct and the workaround's
frequency and measured cost are known.

## 2. Maximum residency does not imply uniform progress

**Discovered by:** Program 01 benchmark

**Classification:** scheduling/runtime observation; no ISA change proposed

The runtime is configured for at most 256 VMs. CUDA's occupancy API currently
estimates 672 resident VM-warp slots for this kernel on the RTX 3060, so all
256 configured machines fit by the resource calculation. In a ten-second
viewer-disabled WarpLife sample,
however, individual VMs ranged from 0.30 to 2.40 generations/second around a
1.42 average. Every VM remained `RUNNING` and made progress, but progress was
not evenly distributed near maximum occupancy.

Keep "maximum resident VM count" separate from "maximum useful/fair VM
count" in future benchmarks. Possible investigations include launch geometry,
block placement, interpreter register pressure, and longer measurement
windows. This result does not presently motivate an ISA change.

## 3. Interpreter overhead dominates the first application baseline

**Discovered by:** Program 01 benchmark

**Classification:** execution-strategy observation; no ISA change proposed

With evolution and framebuffer rendering included on both sides, the
correctness-first WarpLife interpreter path is roughly 2,900–5,000 times
slower than the conventional native CUDA reference across the tested VM
counts. The native and WarpVM implementations use the same dimensions,
toroidal rule, deterministic seeds, and synchronous generation semantics.

This gap is not evidence for Life-specific opcodes. WarpLife intentionally
uses nine straightforward packed-cell loads per batch and executes many
bytecode dispatches. Before considering broad ISA additions, measure:

- simulation and rendering separately;
- the `BALLOT` workaround in isolation;
- packed-word or `SHUFFLE`-based application optimizations;
- interpreter dispatch versus specialized/JIT execution.

## 4. Partially guarded scalar operations are underspecified

**Discovered by:** v0.1.2 CPU interpreter semantic audit

**Classification:** ISA semantic ambiguity; clarification required

The ISA describes scalar registers as uniform and says inactive lanes retain
their old destination. The CUDA interpreter physically replicates scalar
registers in every lane, however, and applies a guard independently in each
lane. A partially populated predicate can therefore update only some physical
copies of a scalar register. A later `S_BCAST` can expose those different
copies as a non-uniform vector value.

WarpLife uses scalar instructions only unguarded, so v0.1.2 equivalence and
benchmarks are unaffected. The CPU interpreter treats the lane-0 copy as the
architectural scalar value, matching the value spilled by the GPU debugger.

Possible resolutions to decide in a later ISA revision:

- declare scalar instructions unguardable;
- require their guard to be warp-uniform (`0` or `0xFFFFFFFF`), faulting
  otherwise;
- explicitly define lane-replicated scalar behavior, although that weakens
  the scalar/vector distinction.

The first or second option better preserves the stated uniform scalar model.

## 5. The direct CPU interpreter currently beats the GPU interpreter

**Discovered by:** v0.1.2 WarpLife CPU/GPU benchmark

**Classification:** execution-strategy observation; no ISA change proposed

The same 153-word WarpLife binary and logical 32-lane state were executed by
both interpreters after full packed-world equivalence checks. On the current
machine, one CPU worker delivered 1.06–23.1 times the aggregate throughput of
GPU WarpVM across 256 down to 1 VMs. Four CPU workers delivered 3.5–7.5 times
the GPU WarpVM throughput. The GPU/CPU ratios printed by the harness are the
reciprocals (0.941–0.043 and 0.283–0.133 respectively) because they are
explicitly reported as `GPU WarpVM / CPU WarpVM`.

This does not mean a CPU is generally better at Life: native CUDA remains
much faster than native CPU as the workload grows. It says that the current
instruction-by-instruction persistent CUDA interpreter has not yet recovered
enough substrate parallelism to offset its dispatch and warp-wide execution
costs. Native CPU is itself about 36–52 times faster than CPU WarpVM, showing
that VM interpretation is also the dominant cost on the host.

Before proposing ISA changes, useful next separations are bytecode dispatch
cost, application instruction count, packed memory work, and framebuffer
rendering. Repeated runs with controlled clocks and a CPU allocation exposing
all physical cores are also needed before treating these ratios as stable
hardware limits.

## 6. Subroutine returns triggered unintended host-control polling (resolved)

**Discovered by:** one-VM WarpLife dynamic profile

**Classification:** runtime control-point issue; no ISA change proposed

The persistent interpreter describes backward branches as loop control
points, but its implemented test applies to every instruction that jumps to a
lower PC. WarpLife's `load_cell` subroutine is above its callers, so each
`RET` returns to a lower PC and triggers `CheckInterrupt`. This produces 4,608
mapped-host control polls per generation. Actual loop backedges and `YIELD`
account for only 1,024 more.

The production interpreter now distinguishes taken backward branch opcodes
from subroutine transfers. It polls at backward `JMP`/`JMP_IF_*` instructions
and `YIELD`, but not `CALL` or `RET`. Polls fell from 5,632 to 1,024 per
generation. The retained build reduced exact-frame generation time from
39.586 to 34.876 ms (11.9%), while the standard two-second benchmark improved
from 25.0 to 28.0 generations/second. Full CPU/GPU world equivalence and all
20 regression tests pass.

## 7. Universal fault voting has a measurable per-opcode cost

**Discovered by:** one-VM WarpLife safety-ablation profile

**Classification:** interpreter strategy; fault semantics must be preserved

After every bytecode, including operations that cannot fault, all lanes execute
a warp `BALLOT` to discover per-lane faults and may shuffle the selected fault
code. WarpLife therefore executes 86,030 implicit fault ballots per generation
in addition to its explicit ISA operations.

A benchmark-only no-fault-vote kernel reduced generation time from 39.586 ms
to 36.532 ms, about 7.7% or 35.5 ns per retired bytecode. This kernel is not
semantically acceptable as a normal runtime. Possible safe investigations are
voting only after fault-capable opcodes, keeping a warp-uniform fault path for
uniform decode errors, or proving which operations cannot create lane-local
faults. This is a runtime optimization question, not evidence for changing
the visible ISA fault contract.

Two selective-vote implementations were subsequently tested. Both retained
votes for lane-local memory and guarded literal faults, and new focused tests
confirmed those faults still became warp-uniform. However, opcode
classification and opcode-local vote placement changed generated control flow
enough that measured WarpLife performance was equal to or worse than the
simple universal vote. The selective optimization was therefore reverted;
the focused fault tests remain. Revisit this only with SASS/hardware-counter
evidence or a design that avoids adding dispatch-path work.

After production `load_cell` inlining, WarpLife retires 76,814 bytecodes and
therefore executes the same number of implicit votes per generation. A fresh
ablation measured 2.246 ms total, or about 29.2 ns per bytecode. The precise
number varies between runs; the conclusion and rejected implementation are
unchanged.

## 8. Small assembly subroutines can cost more than their static compactness

**Discovered by:** one-VM WarpLife instruction census

**Classification:** application optimization; no ISA change proposed

The outlined `load_cell` routine contains nine useful instructions and is
called nine times for every one of the 512 packed words evolved per
generation. That compact 153-word program consequently retires 4,608 `CALL`
and 4,608 `RET` instructions per generation.

Inlining the body at all nine static call sites grows WarpLife to 215 words
but reduces dynamic work from 86,030 to 76,814 bytecodes (10.7%). A controlled
same-run transform measured 36.113 ms outlined versus 31.814 ms inlined
(11.9% lower latency); the subsequent one-second production profile measured
30.643 ms. The standard two-second GPU benchmark rose from 28.0 to 32.5
generations/second (16.1%). CPU/GPU packed worlds remain identical for VM IDs
0, 1, 2, and 37, the dedicated WarpLife checks pass, and all 20 regression
tests pass.

The inlining is retained in `programs/warplife.wva`. This is useful guidance
for current assembly authors, not yet evidence that the ISA needs a macro,
function-inlining facility, or fused cell-load operation.

## 9. One-VM dispatch exposes serial interpreter latency

**Discovered by:** matched microprograms and production-kernel inspection

**Classification:** interpreter execution strategy; no ISA change proposed

After subtracting the measured fault-vote cost, a matched NOP stream still
costs about 288 ns per interpreted operation. The production `sm_86` kernel
uses 72 registers per thread and a 256-byte stack. Its opcode switch compiles
to a compare-and-branch tree, and dynamically selected virtual registers
produce local-memory `LDL`/`STL` traffic in the generated SASS. With only one
VM, only one interpreter warp is active and none of this serial dependency
chain can be hidden behind other VM warps.

This identifies a genuine interpreter floor but not a single safe fix.
Scalarized register storage, a different dispatch structure, bytecode
predecode, and JIT/specialized execution are plausible controlled experiments.
Any retained change must also be measured at many VMs: extra native registers
or code size could improve one-warp latency while reducing occupancy or
instruction-cache behaviour at the workload WarpVM is designed for. Nsight
Compute was installed, but driver policy denied hardware counters
(`ERR_NVGPUCTRPERM`), so static SASS alone is not sufficient attribution.
