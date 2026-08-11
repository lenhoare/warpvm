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

## 15. Existing lane operations are enough to expose word-level parallelism

**Discovered by:** packed word-per-lane WarpLife implementation and matched
five-engine benchmark

**Classification:** retained program and compiled-backend completion; no ISA
change

`programs/warplife_words.wva` maps one packed 32-bit world word to each lane.
A warp evolves 1,024 cells per batch and completes a 512-word world in 16
batches rather than assigning one lane to one cell for 512 batches. Existing
`LOAD`, `STORE`, `LANEID`, `SHUFFLE`, shifts, and Boolean operations express
the complete algorithm. Horizontal word boundaries wrap within each
four-lane 128-cell row, and a four-plane bit-sliced binary accumulator counts
eight neighbours for 32 cells simultaneously.

The program is exactly equivalent at generation 3 for VM IDs 0, 1, 2, and 37
against the GPU interpreter, CPU interpreter, and independent native CPU
implementation. All 512 packed words and all 16,384 framebuffer pixels match.
Compiled checkpoint sharing, mixed execution, and transitions in both
directions also pass. The minimal PTX backend now lowers the already-defined
`SHUFFLE` and `SHUFFLE_XOR` instructions; this completes backend coverage used
by the program and does not alter the ISA.

The exact steady-state census falls from 76,814 to 10,552 bytecodes per
generation (7.28x fewer). Evolution alone falls from 68,097 to 1,835
bytecodes (37.1x fewer). Evolution RAM lane-loads fall from 147,456 to 1,536
(96x fewer), while the final 512 packed-word stores are unchanged. The
remaining 8,711 rendering bytecodes now account for 82.6% of the stream.

In matched one-second runs, compiled throughput improves by 7.1–7.4x across
1–256 VMs. At 64 VMs it reaches 3,476 generations/s/VM and 3,645 Mcell/s
aggregate. At 256 VMs it reaches 2,020 generations/s/VM and 8,472 Mcell/s,
only 1.19x behind the separately handwritten native CUDA reference in the
same run. The GPU interpreter also improves, by 6.3x at one VM and 2.1x at 64
VMs.

Compiled phase bypasses measure a full generation at 0.232 ms. Evolution is
only 0.024 ms (10.3%), rendering 0.173 ms (74.6%), and checkpoint/launch state
0.021 ms (8.9%), with a 0.014 ms cross-run residual. Removing all three static
evolution loads saves at most 0.008 ms. There is therefore no measured reason
to add `VADD32`, `VLOAD32`, a Life instruction, or another evolution primitive:
ordinary vector operations and per-lane addresses already supply those
semantics. If more speed is wanted, first redesign framebuffer expansion with
the current ISA and measure it. A general graphics conversion primitive should
only follow evidence from several programs, not this one application.

The packed compiled kernel uses 58 native registers per thread with no stack,
spills, or local-memory instructions. Its 3,368 static `sm_86` SASS
instructions contain the expected `SHFL.IDX` operations. This confirms that
the speedup did not exchange bytecode work for hidden local-memory traffic.

## 16. C signed integers can be lowered without changing the ISA

**Discovered by:** Warp C v0.1.4 Slice A

**Classification:** compiler/backend implementation; no ISA change

WarpVM's arithmetic comparisons, division, remainder, and right shift are
unsigned, while Warp C defines `int` as signed 32-bit. The first compiler
lowers signed comparison by XOR-biasing both operands with `0x80000000`,
division and remainder through magnitudes plus sign reconstruction, and
arithmetic right shift through logical shift and explicit sign fill. Signed
overflow retains the machine's modulo-2^32 behaviour.

The integer smoke program exercises these paths together with mixed
signed/unsigned conversion and short-circuit evaluation. It halts with
`r0 = 42` in the interpreter and matches complete architectural state, RAM,
framebuffer, and frame sequence through direct PTX compilation. This exposed
three existing instructions absent from the minimal compiled backend:
`DIV`, `MOD`, and `ABS`, plus the `NOTMASK` used by short-circuit control flow.
All four now have direct PTX lowering.

The comprehensive 40-condition smoke program is 424 words with six literals.
That is intentionally not treated as an optimization baseline: the current
register-only allocator and explicit signed sequences favour transparent
semantics. If signed-heavy real programs later show a measured material cost,
signed ISA operations may be reconsidered then; Slice A provides no such
evidence.

## 17. Structured uniform C control flow maps directly to the current ISA

**Discovered by:** Warp C v0.1.4 Slice B

**Classification:** compiler implementation; no ISA change

Uniform `if`, `while`, `do/while`, and `for` lower to the existing comparison
and jump instructions. Compiler-maintained target stacks preserve C nesting:
`break` selects the nearest loop or switch, whereas `continue` searches past
switches for the nearest loop and enters a `for` at its step expression.

`switch` currently emits one equality test per non-default case followed by a
jump to the matching source-order label. The body is then emitted in source
order, so fall-through needs no special instruction. Constant-expression
evaluation rejects non-constant and duplicate case values before assembly.
The control-flow and switch acceptance programs are 143 and 108 words and
both return 42 with exact interpreter/PTX architectural equivalence.

This linear switch strategy is deliberately inspectable and adequate for the
first language slice. Jump tables or a dedicated multi-way branch should only
be considered if real programs show large switches to be important. The
current results provide no reason to change the ISA.

## 18. The existing call stack plus ordinary RAM supports a C ABI

**Discovered by:** Warp C v0.1.4 Slice C

**Classification:** compiler and compiled-backend implementation; no ISA
change

Warp C now passes up to four arguments in `r0`–`r3`, returns a value in `r0`,
and uses the architectural `CALL` / `RET` stack for continuations. Live caller
registers are preserved in private RAM through a uniform `s7` stack pointer.
Each vector spill occupies 32 adjacent words, one per lane, so the ABI already
preserves future divergent values rather than relying on Slice C's currently
uniform programs.

The 441-word acceptance program covers prototypes, two- and four-argument
functions, void returns, calls nested inside argument expressions, and calls
from one non-main function to another. It returns 42 in the interpreter and
matches complete state, RAM, framebuffer, and frame sequence in compiled
execution. A focused backend program also ends with the same retained call
stack words, zero call depth, and `r0 = 18` in both engines.

Direct PTX execution previously lacked `CALL` and `RET` because the compiled
WarpLife programs had inlined their only subroutine. The backend now keeps the
eight architectural continuation words and depth in native registers and
returns through a common PC dispatcher. Stack overflow and underflow retain
the existing `FAULT_STACK` contract.

The conservative caller-save lowering emits substantial code and RAM traffic;
that is visible in the 441-word result. It is a correctness-first baseline,
not evidence for `PUSH`, `POP`, or a C-specific call opcode. Liveness-aware
preservation, scalar placement for uniform values, and selective inlining are
compiler optimizations to measure before proposing any ISA addition.

## 19. Word-addressed C aggregates fit ordinary RAM instructions

**Discovered by:** Warp C v0.1.4 Slice D

**Classification:** compiler memory-layout implementation; no ISA or WVM
format change

Warp C pointers now address logical VM words directly. Arrays and structures
are consecutive words with one-word alignment, so `int *`, `char *`, and
structure pointer arithmetic lower to ordinary integer address arithmetic.
The memory acceptance program covers global storage, address-of, dereference,
pointer increment, indexing, two-word structure layout, and both `.` and `->`.
It is 130 words and returns 42 with exact interpreter/PTX RAM equivalence.

Automatic aggregates and address-taken scalars need lane-private identity even
though Slice D programs remain logically uniform. Their stack frames therefore
use a lane-major physical layout: one lane's complete C frame is contiguous.
This preserves both per-lane isolation and the required `p + 1` physical word
increment. Register-only scalar locals are unchanged. Existing caller-save
spills retain their separate word-major layout because C pointers cannot refer
to them.

Strings use one 32-bit word per character and a zero terminator. A Warp C
`strlen`/`strcmp` acceptance program checks `"hello"` as six words, pointer
iteration, comparison, and a globally initialized string pointer. It is 401
words, returns 42 in both engines, and their complete RAM images match.
Nonzero global and literal data is emitted as lane-0 startup stores, so static
data required neither byte-addressing nor an extension to the WVM file.

The generated automatic-object address sequence currently broadcasts `s7`,
multiplies the lane ID by the per-lane frame size, and adds the object offset.
That is transparent and correct but intentionally not treated as optimized.
Future compiler work can retain frame bases or common addresses in registers;
these results provide no evidence for byte loads/stores, `PUSH`/`POP`, string
instructions, or another memory opcode.

## 20. Existing predicate guards express structured lane divergence

**Discovered by:** Warp C v0.1.4 Slice E

**Classification:** compiler control-flow implementation; no ISA change

`warp_lane_id()` exposes the already-defined `LANEID` value as divergent Warp
C data, while `warp_vm_id()` lowers directly to `VMID` and remains uniform
within one machine. Semantic propagation follows arithmetic, assignments, and
transitive function calls. Ordinary uniform loops and branches retain their
existing jump lowering.

A divergent `if` needs no architectural active-mask stack. The compiler
ballots its condition into `p3`, emits the then instructions under `@p3`,
complements the same predicate for the else instructions, and resumes
unguarded emission at reconvergence. One nested level intersects the child
condition with its parent in `p2`; `p0` and `p1` remain available for ordinary
comparisons and temporary mask combinations. This is the normal predicated
SIMT shape expressed explicitly in inspectable WarpVM bytecode.

The 47-word uniformity program verifies that VM ID and ordinary counters stay
uniform while lane arithmetic becomes divergent; its uniform branch remains a
jump and its lane-varying branch becomes a ballot. The 90-word divergent
program splits lanes at 16, splits each half again, performs masked
lane-private RAM accesses, and reconverges every lane to 42. Both match full
interpreter/PTX state, predicates, and RAM.

Slice E deliberately diagnoses more than two nested divergent masks,
divergent loop/switch control, function calls or returns inside a divergent
region, and divergent short-circuit expressions. Correctly generalizing those
constructs may motivate predicate spilling or a compiler-managed mask stack,
but the current programs do not justify an opcode. Two-level structured
divergence already works using `BALLOT`, `NOTMASK`, `ANDMASK`, and ordinary
instruction guards.

## 21. The graphics C API is a lowering layer, not a new instruction family

**Discovered by:** Warp C v0.1.4 Slice F

**Classification:** compiler and one-shot runtime integration; no ISA change

The four useful source operations map cleanly onto facilities already present.
`warp_framebuffer()` is the constant memory-mapped base, `warp_set_pixel()` is
ordinary `y * 128 + x` address arithmetic plus `STORE`, and `warp_argb()` is
mask/shift/or arithmetic. Only `warp_flip()` is intrinsically VM-wide, and it
lowers directly to the existing unguarded `FLIP`. The compiler rejects a flip
inside divergent control rather than assigning ambiguous per-lane publication
semantics; divergent pixel stores remain naturally expressible.

The 188-word `hello_pixels.wc` acceptance program fills a 128x128 four-colour
image, reads back every framebuffer word, publishes exactly one frame, and
returns 42. The GPU interpreter, logical CPU interpreter, and direct PTX
backend agree on execution and the full framebuffer; persistent inspection
also confirms exact sampled ARGB values and `frame_seq = 1`.

This test exposed one host-path omission: the old one-shot GPU `run` wrapper
created `VmDesc` records without framebuffer storage even though persistent
VMs and the compiled path already supplied it. Allocating the canonical
framebuffer in that wrapper made the architectural memory map consistent. No
opcode, WVM format, or interpreter address-decoding change was needed.
