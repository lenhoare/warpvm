# WarpLife One-VM Profile

Date: 2026-08-11

This profile concentrates on one WarpVM running WarpLife on the RTX 3060. The
program was initially measured as a 153-word outlined implementation and is
now a 215-word implementation with `load_cell` inlined. The normal interpreter
remains the semantic reference. Every run first passes the full packed-world
CPU/GPU equivalence gate for VM IDs 0, 1, 2, and 37.

The profiler combines three forms of evidence:

- an exact dynamic bytecode census from one steady-state generation;
- phase-elimination variants that preserve the original loop structure;
- matched synthetic `.wvm` programs with equal instruction and branch counts.

The timing variants are generated in memory by the benchmark and are not
application replacements. Benchmark-only kernel variants can suppress the
per-instruction fault vote or poll host control only at `YIELD`; normal runtime
commands never launch those variants. Nsight Compute 2026.1 was also tried,
but hardware counters were unavailable because the driver denied performance-
counter access (`ERR_NVGPUCTRPERM`).

## Dynamic work per generation

```text
Retired bytecodes                 76,814
Evolution bytecodes              68,097  (88.7%)
Rendering bytecodes               8,711  (11.3%)
Publish/control bytecodes             6
Per-op warp fault ballots        76,814
Mapped-host control polls         1,024
RAM LOAD instructions             5,120
RAM load lane accesses          163,840
Summed unique RAM addresses       8,192
RAM store lane accesses             512
Framebuffer lane writes          16,384
```

The original profile exposed 5,632 control polls:

```text
RET from load_cell                4,608
Taken evolution-loop branch         511
Taken rendering-loop branch         511
Generation-backedge JMP               1
YIELD                                  1
```

The interpreter treated every transfer to a lower PC as a control point.
Consequently every `RET` from the forward `load_cell` subroutine read mapped
host control state even though it was not a loop backedge.

This has now been corrected. Normal execution polls only taken backward
`JMP`/`JMP_IF_*` instructions and `YIELD`, leaving 1,024 polls per generation:

```text
Taken evolution-loop branch         511
Taken rendering-loop branch         511
Generation-backedge JMP               1
YIELD                                  1
```

The largest opcode counts are:

| Opcode | Count | Share |
|---|---:|---:|
| `ADD` | 15,872 | 20.7% |
| `AND_I` | 14,848 | 19.3% |
| `ADD_I` | 6,144 | 8.0% |
| `SHL_I` | 5,632 | 7.3% |
| `LOAD` | 5,120 | 6.7% |
| `SHR_I` | 5,120 | 6.7% |
| `SHR` | 5,120 | 6.7% |

The outlined version also retired 4,608 `CALL` and 4,608 `RET` instructions
per generation. Inlining the nine-instruction `load_cell` body at its nine
static call sites removes those 9,216 dynamic bytecodes. Static code grows
from 153 to 215 words, while dynamic work falls from 86,030 to 76,814
bytecodes per generation (10.7%).

This is primarily a large stream of small integer/addressing instructions,
not a framebuffer-bandwidth workload.

## Timing results

Results are medians of three exact-frame measurements targeting one second
each. The initial profile before correcting `RET` polling was:

| Experiment | ms/generation | Difference from normal |
|---|---:|---:|
| Normal WarpLife | 39.586 | — |
| Evolution + publication; rendering bypassed | 35.347 | 4.239 ms |
| Rendering + publication; evolution bypassed | 4.012 | 35.574 ms |
| Normal instruction stream, framebuffer `STORE` changed to `NOP` | 38.888 | 0.698 ms |
| No per-op fault vote, benchmark only | 36.532 | 3.054 ms / 7.7% |
| Host polling only at `YIELD`, benchmark only | 32.928 | 6.658 ms / 16.8% |
| Both safety ablations, benchmark only | 29.488 | 10.098 ms / 25.5% |

Differential phase estimates are approximately 35.6 ms evolution and 4.2 ms
rendering. Their 0.23 ms overlap/residual is normal cross-run and non-additive
variation and is not assigned to the six publication instructions.

Actual framebuffer memory writes account for about 0.70 ms, or 1.8%. Most of
the rendering phase is therefore its 8,711 interpreted address, unpacking,
loop, and colour-selection bytecodes rather than framebuffer bandwidth.

Suppressing backward polling leaves the `YIELD` poll intact so the persistent
kernel can still stop. The observed saving is about 1.18 microseconds for each
of the 5,631 removed polls. Suppressing the fault vote saves about 35.5 ns per
retired bytecode.

The retained build after the control-point correction measures:

| Experiment | ms/generation | Difference from normal |
|---|---:|---:|
| Normal WarpLife | 34.876 | — |
| Evolution + publication; rendering bypassed | 30.578 | 4.299 ms |
| Rendering + publication; evolution bypassed | 4.087 | 30.790 ms |
| Normal instruction stream, framebuffer `STORE` changed to `NOP` | 34.644 | 0.232 ms |
| No per-op fault vote, benchmark only | 32.105 | 2.772 ms / 7.9% |
| Host polling only at `YIELD`, benchmark only | 33.213 | 1.663 ms / 4.8% |
| Both safety ablations, benchmark only | 30.680 | 4.196 ms / 12.0% |

Normal generation latency therefore fell from 39.586 to 34.876 ms, an 11.9%
reduction, while the standard two-second benchmark moved from 25.0 to 28.0
generations/second (12% higher throughput). Independent short and long runs
put the gain in roughly the 12–16% range. The CPU/GPU full-world equivalence
gate and all regression tests continue to pass.

The remaining 1,023 backward-branch polls cost about 1.66 ms in this run. They
are intentional responsiveness points rather than accidental subroutine
traffic.

Before changing the source, the profiler performed the transformation in
memory, compared the outlined and inlined CPU worlds through generation 3 for
VM IDs 0, 2, and 37, then timed both under the same normal runtime. A short
median-of-three experiment measured 36.113 ms outlined and 31.814 ms inlined,
a 4.299 ms or 11.9% reduction.

The production WarpLife assembly now includes that measured inlining. A final
one-second, median-of-three profile measured:

| Experiment | ms/generation | Difference from normal |
|---|---:|---:|
| Normal WarpLife, inlined | 30.643 | — |
| Evolution + publication | 26.501 | 4.143 ms rendering estimate |
| Rendering + publication | 4.033 | 26.610 ms evolution estimate |
| Framebuffer `STORE` changed to `NOP` | 30.647 | within run noise |
| No per-op fault vote, benchmark only | 28.397 | 2.246 ms / 7.3% |
| Host polling only at `YIELD`, benchmark only | 29.234 | 1.409 ms / 4.6% |
| Both safety ablations, benchmark only | 26.874 | 3.769 ms / 12.3% |

Against the immediately preceding 34.876 ms outlined profile, inlining saves
4.233 ms (12.1%). The standard two-second benchmark improved from 28.0 to
32.5 generations/second (16.1%); ordinary run-to-run variation accounts for
the difference between the two comparison methods. The CPU interpreter also
rose slightly from 576.95 to 579.94 generations/second in the standard run.

## Matched microprograms

Each synthetic frame performs 1,024 loop iterations, 1,024 backward polls,
and one `YIELD` poll. Increasing an unrolled NOP body while keeping control
work fixed gives:

```text
0 NOPs/iteration                 2.498 ms/frame
32 NOPs/iteration               12.880 ms/frame
128 NOPs/iteration              44.103 ms/frame
Incremental interpreted NOP      317.4 ns/op
Inferred control poll             1.485 us/poll
```

Equal-count opcode bodies add approximately the following over NOP:

| Opcode body | Additional ns/op |
|---|---:|
| `ADD` | 40.2 |
| two-address RAM `LOAD` | 128.0 |
| RAM `STORE` | 106.8 |
| `REDUCE_OR` | 128.2 |
| contiguous framebuffer `STORE` | 125.1 |

The fault-vote contribution is 29.2 ns/op. Subtracting it leaves an
approximate 288 ns/op fetch/decode/dispatch/NOP floor. This is an estimate
rather than a hardware-counter attribution, but it is consistent with the
26.9 ms still required after both safety ablations.

Static inspection provides supporting, though not stall-level, evidence. The
`sm_86` production kernel uses 72 registers per thread and a 256-byte stack.
The opcode switch compiles to a compare-and-branch decision tree rather than
one indirect dispatch, while dynamically indexed virtual-register accesses
produce local `LDL`/`STL` instructions. At one VM only one VM warp is active,
so the GPU has no independent interpreter warps with which to hide that serial
latency. Nsight hardware counters would be needed to apportion the remaining
floor reliably between dispatch branches, local accesses, dependencies, and
instruction-fetch effects.

## Interpretation

The priorities indicated by this one-VM result are:

1. The `RET` control-point correction is implemented and retained. It removed
   4,608 unnecessary polls per generation and improved one-VM throughput by
   about 12–16% without an ISA change.
2. Unconditional warp fault voting remains a measured opportunity, but is not
   yet an obvious production change. Two selective-vote implementations kept
   fault semantics and passed focused tests, yet generated code was equal to
   or slower than the simple unconditional vote. Both were rejected rather
   than retaining an optimization that only looked better structurally.
3. `load_cell` inlining is implemented and retained. It removes 9,216
   bytecodes per generation and improves one-VM generation latency by about
   12%, with no ISA or interpreter specialization.
4. Treat framebuffer bandwidth as low priority for one VM. Removing the actual
   writes saves under 2%; deleting rendering entirely saves about 11%.
5. The remaining dispatch floor is real, but no similarly obvious production
   edit follows from static SASS alone. Scalarizing virtual registers,
   predecoding bytecode, or changing dispatch structure should be controlled
   experiments and must also be checked at high VM counts because register
   growth can reduce occupancy.

These are baseline observations, not final architecture conclusions. Nsight
stall counters, repeated clock-controlled runs, and equivalent measurements
at higher VM counts remain useful follow-ups.
