# Packed word-per-lane WarpLife

Date: 2026-08-11

This experiment keeps `programs/warplife.wva` as the reference and adds
`programs/warplife_words.wva`. Both programs implement the same deterministic
128×128 toroidal Life worlds and full ARGB framebuffer rendering. The packed
version uses only the existing ISA and assigns one 32-cell packed word to each
lane, allowing one warp to evolve 1,024 cells per batch.

## Correctness gate

At generation 3, VM IDs 0, 1, 2, and 37 match across GPU WarpVM, CPU WarpVM,
and the independent handwritten native CPU implementation. Each comparison
covers all 512 packed world words and all 16,384 framebuffer pixels. The known
cross-word blinker and toroidal-corner still-life tests also pass. The compiled
backend passes shared-artifact checkpoints, interpreted-to-compiled and
compiled-to-interpreted transitions, and a simultaneous mixed-mode session.

## Exact dynamic census

| Metric per steady-state generation | Reference | Word-per-lane | Change |
|---|---:|---:|---:|
| Retired bytecodes | 76,814 | 10,552 | 7.28× fewer |
| Evolution bytecodes | 68,097 | 1,835 | 37.1× fewer |
| Rendering bytecodes | 8,711 | 8,711 | unchanged |
| Host-control polls | 1,024 | 528 | 48.4% fewer |
| RAM load lane accesses | 163,840 | 17,920 | 9.14× fewer |
| Evolution RAM load lane accesses | 147,456 | 1,536 | 96× fewer |
| Summed unique RAM load addresses | 8,192 | 2,048 | 4× fewer |
| RAM store lane accesses | 512 | 512 | unchanged |
| Framebuffer lane writes | 16,384 | 16,384 | unchanged |

The packed program executes 96 `SHUFFLE` bytecodes per generation. Its largest
remaining opcode counts are `ADD` 1,616, `S_BCAST` 1,104, `MOV` 1,028, and the
unchanged render loop's 512-iteration families. Rendering is now 82.6% of all
retired bytecodes.

## Matched one-second benchmark

The environment is the same RTX 3060 / i5-12400F WSL system documented in
`benchmarks/warplife.md`. Each row below comes from adjacent full benchmark
runs. Rates include evolution and complete framebuffer rendering; SDL and host
framebuffer copies are disabled.

| VMs | GPU interpreter reference | GPU interpreter packed | Speedup | Compiled reference | Compiled packed | Speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 32.00 | 200.98 | 6.28× | 583.71 | 4,321.22 | 7.40× |
| 8 | 30.00 | 192.11 | 6.40× | 496.97 | 3,542.44 | 7.13× |
| 32 | 28.50 | 102.21 | 3.59× | 491.13 | 3,559.39 | 7.25× |
| 64 | 26.06 | 54.75 | 2.10× | 491.37 | 3,476.02 | 7.07× |
| 256 | 8.18 | 14.53 | 1.78× | 284.83 | 2,019.97 | 7.09× |

All rate columns are generations/s/VM. Packed compiled aggregate throughput is
70.8, 464.3, 1,866.1, 3,644.9, and 8,472.4 Mcell/s at the five populations.

Against the native references in the same packed-program run:

| VMs | Compiled WarpVM / native CPU (1 worker) | Compiled WarpVM / native CPU (4 workers) | Native CUDA / compiled WarpVM |
|---:|---:|---:|---:|
| 1 | 0.137× | 0.137× | 16.6× |
| 8 | 1.264× | 0.401× | 8.5× |
| 32 | 5.427× | 2.236× | 4.3× |
| 64 | 11.565× | 4.022× | 1.3× |
| 256 | 35.142× | 11.331× | 1.2× |

The native CPU is a clear scalar byte-per-cell reference, not an optimized
SIMD or bit-parallel CPU implementation.

## Compiled phase attribution

Median of three runs, 500 YIELD checkpoints per sample:

| Component | Time per generation | Share of full time |
|---|---:|---:|
| Full program | 0.2318 ms | 100% |
| Evolution differential | 0.0238 ms | 10.3% |
| Rendering differential | 0.1730 ms | 74.6% |
| Checkpoint/launch state | 0.0206 ms | 8.9% |
| Cross-run residual | 0.0143 ms | 6.2% |

Replacing the framebuffer store with `NOP` reduces the full run by 0.0368 ms.
Replacing all three static evolution `LOAD` sites with zero-valued `MOV_I`
reduces it by only 0.0081 ms. The compiled program emits 64,602 bytes of PTX.
For `sm_86`, `ptxas` uses 58 registers per thread with a zero-byte stack and
no spills or local memory. The cubin contains 3,368 static SASS instructions,
including the expected `SHFL.IDX` sites. The largest mnemonic groups are
`PRMT` 555, `ISETP.NE.U32.AND` 467, `IMAD.MOV.U32` 409, `IADD3` 273,
`SEL` 271, and `BRA` 270; there are no `LDL` or `STL` instructions.

## ISA conclusion

No new instruction is justified by this experiment. `LOAD` and `STORE`
already accept independent per-lane addresses; arithmetic and Boolean
instructions already operate lane-wise; `LANEID` and `SHUFFLE` provide the
needed cross-lane mapping. Prefixing duplicate forms with `V` would add names,
not capability.

The dominant remaining work is framebuffer expansion. The next sensible
experiment is a renderer mapped around packed words using the current ISA.
Only after several graphics workloads show the same irreducible sequence
should a general conversion or packed-pixel primitive be considered.
