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

## 10. Named registers trade local traffic for selection logic

**Discovered by:** four-way register-file/dispatch experiment

**Classification:** rejected interpreter implementation; no ISA change

The dynamically indexed `uint32_t vregs[16]` does generate local-memory
instructions. A benchmark-only interpreter specialization loaded the sixteen
architectural values into named `r0` through `r15` C++ locals and implemented
dynamic reads and writes with fully inlined switches. Full CPU/GPU world
equivalence passed for all four deterministic VM IDs.

The change reduced static SASS `LDL` occurrences from 250 to 147 and `STL`
from 388 to 316, confirming that more state reached physical SM registers.
But physical register use rose from 72 to 118 per thread, reducing estimated
residency from 672 to 448 VM warps. More importantly, selecting a named value
expanded the generated kernel from 772 to 7,162 branches and from 597 to 5,090
integer comparisons. One-VM generation time rose from 30.353 to 39.410 ms
(29.8% higher latency). This implementation is rejected.

A second specialization translated sparse ISA opcodes to a dense 0–81 range
through constant memory before the switch. It kept 72 registers and measured
30.108 ms, an apparent 0.8% advantage; a shorter independent run showed 1.2%.
Generated SASS nevertheless remained a compare/branch opcode tree. The sole
`BRX` is the unrelated host-command switch. Dense translation removed only
four static branches and six comparisons, so it remains a benchmark-only
near-neutral result rather than a production optimization.

The shared-memory alternative was evaluated next and is recorded in finding
11. A genuinely different dispatcher likely requires explicit hierarchical
handler groups, device indirect calls, generated PTX, or specialization/JIT.
Each must retain general ISA semantics and be checked at both one and many
VMs.

## 11. Shared registers need a register-major bank layout

**Discovered by:** shared-memory register-file experiment

**Classification:** benchmark-only near-neutral interpreter implementation;
no ISA change

A dynamically indexed virtual register file can be placed in shared memory,
but its layout matters. Each 256-thread block needs 16 KiB for sixteen
registers per thread. The initially natural `[thread][register]` layout makes
the lanes of a warp access words 16 apart when they request the same dynamic
register, producing 16-way bank conflicts. Transposing it to
`[register][thread]` gives the warp 32 consecutive, conflict-free addresses.

The VM-entry and VM-exit copies must also remain compact loops. Fully
unrolling those copies raised allocation to 119 native registers per thread.
Preventing unrolling restored the unconstrained shared kernel to 72 registers,
a 256-byte stack, and the baseline estimate of 672 resident VM slots. A
three-block launch bound used 80 registers and a 304-byte stack while
retaining the same estimated residency.

The corrected shared implementation reduces static SASS local-memory
instructions from 250 `LDL` / 388 `STL` to 137 / 302, adding 120 `LDS` and 90
`STS`. Unlike named-register scalarization, it does not expand the selection
logic: its 775 `BRA` and 601 `ISETP` counts remain close to the baseline's 772
and 597. Complete CPU/GPU packed-world equivalence passed for VM IDs 0, 1, 2,
and 37.

In the final median-of-three one-second run, the occupancy-constrained shared
kernel reduced one-VM generation time from 30.781 to 30.372 ms, a 1.35%
throughput improvement. Adding dense opcode translation reached 30.354 ms, a
1.41% improvement, so the dispatcher change contributed essentially nothing.
At 64 VMs, the combined result was 26.15 generations/s/VM versus 26.12 for
baseline, while other repeated samples moved by a few percent in either
direction. The shared register file therefore demonstrates the mechanism but
does not provide a robust enough end-to-end gain to replace the production
array implementation.

## 12. Opcode selection is measurable but wants workload specialization

**Discovered by:** SASS control-flow inspection, matched empty handlers, and
profile-guided dispatch kernels

**Classification:** benchmark-only interpreter specialization; no ISA change

The production opcode switch becomes a balanced SASS compare/branch tree.
Its root splits at opcode `0x37`, and common WarpLife instructions normally
traverse roughly five or six dependent decisions. `NOP` (`0x00`) and
`STEP_TRAP` (`0x77`) have identical empty handlers in normal execution, so
matched bodies isolate their different paths through that tree. Across runs,
the high-opcode path cost 7–17 ns more per bytecode; a one-second sample
measured 10.1 ns. Tree position matters, but it is only a fraction of the
roughly 290 ns NOP interpreter floor.

A benchmark kernel then checked `ADD` and `AND_I` before entering the general
switch. Those two opcodes make up 40.0% of WarpLife's 76,814-bytecode steady
state generation. ptxas retained two explicit equality checks, while native
allocation remained 72 registers per thread with a 256-byte local stack and
672 estimated resident VM slots.

In-kernel `clock64()` timing removes the frequency drift seen in host timings.
The two-handler path reduced one-VM cost from 62,608,603 to 60,588,772 cycles
per generation, a 3.3% throughput gain. An independent run showed the same
3.1% gain. At 64 VMs, median cost fell from 80,314,984 to 79,447,861 cycles
per VM-generation, a smaller 1.1% gain because other warps hide more serial
latency.

Extending the front end with `ADD_I` and `SHL_I` raised dynamic coverage to
55.3% but measured 60,625,395 cycles, indistinguishable from the two-handler
version. The extra comparisons and duplicated code consume the additional
saving. Static code grows from 77,184 bytes / 4,824 instructions at baseline
to 78,080 / 4,880 for two handlers and 78,848 / 4,928 for four.

This is useful attribution, not a globally safe production optimization.
Hard-coding WarpLife's hot pair makes every other opcode pay up to two extra
tests. The result argues for collecting opcode profiles from several real VM
programs or for program-specific/JIT dispatch specialization. It does not
argue for an ISA encoding change: dense remapping already showed that changing
opcode numbers alone does not make ptxas emit indexed dispatch.

## 13. Compilation removes the interpreter floor, but not the machine-model gap

**Discovered by:** v0.1.3 direct `.wvm` to PTX WarpLife experiment

**Classification:** retained dual-mode backend; possible future ISA and
execution-model work

The unchanged 215-word WarpLife bytecode now compiles to a 32-lane PTX kernel.
Complete architectural state, both 16K-word RAM images, both 128x128
framebuffers, frame counters, two VMs sharing one artifact, simultaneous
mixed-mode execution, and transitions in both directions between interpreted
and compiled execution are bit-equivalent at YIELD checkpoints. An exact-key
process-local compilation cache also reuses an artifact for identical `.wvm`
content.

The retained fault-checking backend generates 72,427 bytes of PTX and a
64,168-byte `sm_86` cubin. `ptxas` uses 54 registers per thread with no local
spills. At one VM it reaches 574.8 generations/s versus 32.0 for the GPU
interpreter, an 18.0x improvement. At 256 VMs it reaches 282.9 generations/s/VM
versus 8.04, a 35.2x improvement. Opcode interpretation was therefore a major
cost, and the dual-mode direction is strongly validated.

Compilation does not close the native CUDA gap. The native reference remains
115.6x faster per VM at one VM and 10.0x faster per VM at 256 VMs. WarpLife
serializes 512 packed-word batches inside one VM warp, whereas the native
reference exposes cells across many CUDA threads. The remaining difference is
therefore not evidence for another dispatch tweak. It reflects useful work,
the current low-level ISA stream, and especially the one-warp-per-VM execution
model's available parallelism. Before changing the ISA, inspect compiled SASS
and separate instruction/addressing cost from the deliberate per-VM
parallelism limit. Generic higher-granularity warp operations remain a
plausible later experiment; no Life-specific opcode is proposed.

## 14. Compiled WarpVM crosses the handwritten CPU at population scale

**Discovered by:** full-world native CPU equivalence and unified five-engine
benchmark

**Classification:** positive architecture result; CPU reference remains a
straightforward scalar baseline

The handwritten native CPU implementation is now an exact semantic gate, not
only a timing reference. At generation 3, all 512 packed world words and all
16,384 ARGB framebuffer pixels match canonical WarpLife for VM IDs 0, 1, 2,
and 37. The independent packed-world hashes match in every case.

At one VM, the scalar native CPU is 53.3x faster than compiled WarpVM: one GPU
warp has too little parallel work to compete with a CPU core running a direct
cell loop. The result reverses as the population grows. At 64 VMs, compiled
WarpVM is 1.82x the one-worker native CPU throughput. At 256 VMs it is 4.82x
one worker and 1.91x the four-worker run on the two-core/four-thread WSL guest.
Aggregate compiled throughput reaches 1,175 Mcell/s versus 617 Mcell/s for the
four-worker native CPU.

This is the intended WarpVM scaling shape: poor single-VM CPU competition but
strong many-independent-VM throughput. It is a confidence-building result,
not a claim against optimized CPU Life generally. Host disassembly confirms
the current native CPU inner loop is scalar; it uses a clear byte-per-cell
implementation and was built without architecture-specific AVX2 intrinsics.
An optimized SIMD or bit-parallel CPU reference would be a useful later bar.

Compiled SASS and controlled phase bypasses locate the remaining GPU time.
The full kernel uses 54 registers and zero local spills; dynamic virtual
register traffic has disappeared. One generation measures 1.782 ms: evolution
is 1.573 ms (88.2%), rendering 0.208 ms (11.7%), and launch/YIELD/state
materialization only 0.007 ms (0.4%). Actual framebuffer writes contribute
about 0.078 ms. Replacing all nine static evolution RAM loads gives only a
0.319 ms upper bound, so most remaining cost is the real serialized
neighbour/address/arithmetic chain across 512 packed batches.
