# Compiled WarpLife SASS and Native CPU Attribution

Date: 2026-08-11

## Correctness gate

The handwritten native CPU implementation uses an independent byte-per-cell
128x128 toroidal world and direct ARGB rendering. It now compares the complete
result, rather than checking only the two known seed shapes.

At generation 3, all 512 packed world words and all 16,384 framebuffer pixels
match the WarpVM CPU interpreter for VM IDs 0, 1, 2, and 37:

| VM | WarpVM checksum | Native CPU checksum | World | Framebuffer |
|---:|---:|---:|:---:|:---:|
| 0 | `9e4ccd619a0ef56a` | `9e4ccd619a0ef56a` | PASS | PASS |
| 1 | `0f6d80073ff2d013` | `0f6d80073ff2d013` | PASS | PASS |
| 2 | `e31359b622c31b43` | `e31359b622c31b43` | PASS | PASS |
| 37 | `14bbba62c62b9de3` | `14bbba62c62b9de3` | PASS | PASS |

This gate now runs before the central `life_bench` command accepts timing
results.

## Unified compiled/native CPU result

Each row is a same-run one-second sample. The WSL guest exposes an i5-12400F
as two cores/four hardware threads. The native CPU loop is handwritten scalar
C++ over byte cells; host disassembly contains scalar integer operations and
no SIMD inner loop.

| VMs | Compiled WarpVM Mcell/s | Native CPU 1 worker Mcell/s | Native CPU 4 workers Mcell/s | Compiled / CPU 1 | Compiled / CPU 4 |
|---:|---:|---:|---:|---:|---:|
| 1 | 9.516 | 507.288 | — | 0.019x | — |
| 8 | 64.752 | 426.490 | 1,116.041 | 0.152x | 0.058x |
| 32 | 251.789 | 329.501 | 946.261 | 0.764x | 0.266x |
| 64 | 507.441 | 278.790 | 801.417 | 1.820x | 0.633x |
| 256 | 1,175.441 | 243.685 | 616.547 | 4.824x | 1.906x |

The result has the desired population-compute shape. A single direct CPU loop
is overwhelmingly faster than one active GPU warp. Compiled WarpVM overtakes
one CPU worker between 32 and 64 VMs and overtakes all four exposed hardware
threads at 256 VMs. This comparison does not claim superiority over an AVX2,
bit-parallel, or heavily tuned CPU Life implementation.

## Compiled phase attribution

Every variant retains the 215-word program layout and one kernel invocation
per YIELD. Results are medians of three samples, each containing 500 resident
checkpoints.

| Variant | ms/generation | Generated PTX bytes |
|---|---:|---:|
| Full | 1.7823 | 72,427 |
| Evolution + publication | 1.5796 | 72,417 |
| Rendering + publication | 0.2148 | 72,429 |
| Publication/checkpoint only | 0.0070 | 72,429 |
| Framebuffer `STORE` changed to `NOP` | 1.7043 | 71,607 |
| Nine static evolution `LOAD`s changed to `MOV_I` | 1.4636 | 65,554 |

Differential estimates:

```text
Evolution                         1.5726 ms   88.2%
Rendering                         0.2079 ms   11.7%
Launch + YIELD + state checkpoint 0.0070 ms    0.4%
Phase-sum residual               -0.0051 ms
Actual framebuffer STORE delta    0.0780 ms
Evolution RAM LOAD upper bound    0.3187 ms
```

The RAM-load ablation changes values and dependency chains, so 0.3187 ms is
an upper bound rather than a pure load-latency measurement.

## Static SASS

The retained `sm_86` cubin contains 3,736 static SASS instructions. Resource
usage is 54 registers per thread, zero stack, zero shared memory, and zero
local memory. In particular, there are no `LDL` or `STL` instructions: the
compiled program has eliminated the interpreter's dynamically indexed local
register file.

The largest static opcode groups are:

| SASS group | Static count |
|---|---:|
| `PRMT` | 576 |
| `ISETP.NE.U32.AND` | 515 |
| `IMAD.MOV.U32` | 422 |
| `SEL` | 364 |
| `IADD3` | 314 |
| `BRA` | 291 |
| `IMAD.X` | 275 |
| `SHF.R.U32.HI` | 204 |
| `SGXT.U32` | 192 |
| `LDG.E` / `LDG.E.64` | 39 / 7 |
| `STG.E` / `STG.E.64` | 53 / 1 |

Static counts should not be mistaken for dynamic attribution: the bytecode
CFG contains loops, and ptxas factors some common generated paths into
stackless calls. Entry continuation dispatch is currently a linear chain over
215 PCs, but the measured complete checkpoint-only path is just 7 microseconds
per generation. Optimizing that chain cannot materially close the remaining
gap.

The evidence points to the 512 serial packed-word evolution batches and their
neighbour/address arithmetic. Native CUDA spreads cells across many more
threads; compilation cannot infer that extra parallelism without changing the
machine model or adding higher-granularity generic operations.
