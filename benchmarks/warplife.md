# WarpLife Benchmark Baseline

Date: 2026-08-11

## Environment

```text
GPU:          NVIDIA GeForce RTX 3060, 12,288 MiB
Driver:       591.86
CUDA toolkit: 13.2 (nvcc 13.2.78)
CUDA target:  sm_86
CPU:          12th Gen Intel Core i5-12400F
CPU exposed:  2 cores / 4 hardware threads (WSL virtual machine)
C++ compiler: GCC 13.3.0
Build:        Release
Viewer:       disabled
```

The tested `.wvm` was assembled from `programs/warplife.wva` and executed
through the real persistent WarpVM interpreter. No hard-coded Life shortcut
was used.

The native reference uses the same deterministic initial states, 128×128
toroidal world, synchronous generations, and ARGB8888 framebuffer expansion.
It is a conventional byte-per-cell CUDA implementation with one evolution
kernel and one render kernel per generation. Neither measurement includes
host framebuffer copying or SDL presentation.

`cell updates/sec` counts simulated cells once per generation. Rendering is
included in elapsed time but does not add to the cell-update count.

## Results

Two-second measurement per implementation and VM count:

| VMs | WarpVM gen/s/VM, average (min–max) | WarpVM aggregate Mcell/s | Native gen/s/VM | Native aggregate Mcell/s | Native/WarpVM |
|---:|---:|---:|---:|---:|---:|
| 1 | 24.50 (24.50–24.50) | 0.401 | 77,053.99 | 1,262.453 | 3,145.2× |
| 8 | 21.56 (21.50–22.00) | 2.826 | 62,629.98 | 8,209.037 | 2,904.7× |
| 32 | 9.37 (9.00–9.50) | 4.915 | 45,405.19 | 23,805.397 | 4,843.4× |
| 64 | 5.08 (3.50–8.50) | 5.324 | 24,567.28 | 25,760.658 | 4,838.2× |
| 256 | 1.40 (0.00–3.00) | 5.882 | 7,006.48 | 29,387.305 | 4,996.4× |

The original report clamped the occupancy result to the runtime's configured
256-VM maximum. The raw CUDA occupancy calculation for the current build is
672 resident VM-warp slots (3 blocks/SM × 28 SMs × 8 VMs/block). Thus 256 is
the software limit, not the hardware occupancy estimate. At that count, a
two-second window is shorter than the slowest VM's generation time. A
supplementary ten-second sample confirmed forward progress for every VM:

```text
256 VMs
WarpVM:  1.42 gen/s/VM average (0.30–2.40 min–max)
         5.957 aggregate Mcell/s
Native:  7,021.85 gen/s/VM
         29,451.781 aggregate Mcell/s
Ratio:   4,943.9x
```

## Initial observations

- Aggregate WarpVM throughput scales from 0.401 Mcell/s at one VM to roughly
  5.3–6.0 Mcell/s at 64–256 VMs, a 13–15× increase.
- Per-VM throughput decreases as more persistent interpreters share the GPU.
- Progress becomes uneven near maximum occupancy even though all 256 VMs are
  resident and remain `RUNNING`.
- The correctness-first program performs nine general packed-cell loads per
  32-cell batch and deliberately avoids shuffle/packed-word optimization.
- The native comparison establishes the scale of interpreter and instruction
  sequence overhead. It does not justify adding Life-specific instructions.

This is a baseline, not a statistically rigorous performance study. Future
measurements should include repeated trials, clock/power controls, simulation-
only and render-only breakdowns, and optimized WarpLife variants.

## v0.1.2 CPU interpreter comparison

The CPU interpreter executes the exact 153-word `warplife.wvm` bytecode with
32 logical lanes represented as ordinary arrays. Single-worker scheduling is
round-robin; parallel scheduling uses four fixed workers and stable
interleaved VM ownership. Both use a fixed 4,096-instruction quantum. No
Life-specific shortcut, bytecode fusion, JIT, or alternate CPU world
representation is present in the interpreter.

Before timing, the harness compared all 512 packed world words between CPU
and GPU at the same generation for VM IDs 0, 1, 2, and 37. All four full
comparisons passed. The native CPU checks the blinker and toroidal still-life
seeds after every timed run. Evolution and complete ARGB rendering are timed
for every implementation; SDL and host framebuffer copies are not.

Two-second requested samples produced the following final run. `s` is actual
elapsed time, and each rate cell is `average gen/s/VM (min–max); aggregate
Mcell/s`.

| VMs | GPU WarpVM | CPU WarpVM, 1 worker | CPU WarpVM, 4 workers |
|---:|---:|---:|---:|
| 1 | 2.000s; 25.00 (25.00–25.00); 0.410 | 2.000s; 576.95 (576.95–576.95); 9.453 | same one-worker run |
| 8 | 2.000s; 22.00 (22.00–22.00); 2.883 | 2.000s; 64.99 (64.99–64.99); 8.519 | 2.008s; 165.68 (135.93–216.59); 21.716 |
| 32 | 2.000s; 8.48 (7.50–9.50); 4.448 | 2.001s; 16.99 (16.99–16.99); 8.909 | 2.005s; 43.65 (32.92–56.87); 22.884 |
| 64 | 2.000s; 4.94 (4.50–9.50); 5.177 | 2.000s; 8.00 (8.00–8.00); 8.387 | 2.004s; 20.21 (17.47–24.46); 21.196 |
| 256 | 2.000s; 1.41 (0.00–2.50); 5.923 | 2.000s; 1.50 (1.50–1.50); 6.291 | 2.002s; 4.99 (4.50–5.49); 20.949 |

| VMs | Native CPU, 1 worker | Native CPU, 4 workers | Native CUDA |
|---:|---:|---:|---:|
| 1 | 2.000s; 29,914.22; 490.115 | same one-worker run | 0.911s; 46,088.44; 755.113 |
| 8 | 2.000s; 3,280.30 (3,280.17–3,280.67); 429.955 | 2.002s; 8,337.53 (7,361.52–8,908.74); 1,092.816 | 0.576s; 33,101.55; 4,338.686 |
| 32 | 2.000s; 698.55 (698.44–698.94); 366.242 | 2.004s; 1,817.99 (1,708.86–2,020.79); 953.152 | 1.791s; 44,653.11; 23,411.088 |
| 64 | 2.001s; 329.38 (329.30–329.80); 345.381 | 2.002s; 912.23 (870.08–945.99); 956.546 | 0.490s; 25,082.87; 26,301.293 |
| 256 | 2.003s; 60.05 (59.91–60.41); 251.848 | 2.002s; 178.70 (172.29–189.27); 749.518 | 0.193s; 4,761.55; 19,971.371 |

Native CUDA uses a calibrated generation count, hence its actual elapsed time
can differ from the requested two seconds. All comparisons use aggregate cell
throughput. Ratios are explicitly `left / right`:

| VMs | GPU WarpVM / CPU WarpVM (1) | GPU WarpVM / CPU WarpVM (4) | Native CPU (1) / CPU WarpVM (1) | Native CPU (4) / CPU WarpVM (4) | Native CUDA / GPU WarpVM |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.043× | 0.043× | 51.8× | 51.8× | 1,843.6× |
| 8 | 0.338× | 0.133× | 50.5× | 50.3× | 1,504.7× |
| 32 | 0.499× | 0.194× | 41.1× | 41.7× | 5,263.2× |
| 64 | 0.617× | 0.244× | 41.2× | 45.1× | 5,080.3× |
| 256 | 0.941× | 0.283× | 40.0× | 35.8× | 3,372.1× |

The CPU WarpVM is faster in this baseline: one worker is about 1.06–23.1×
the GPU WarpVM throughput across these counts, while four workers reach about
3.5–7.5× GPU WarpVM throughput. This is evidence about the current
instruction-by-instruction interpreters, not about CPU versus GPU Life in
general: native CUDA remains far faster than native CPU at larger VM counts,
and the native/interpreted gap is about 36–52× on the CPU itself.

The four-worker runs use the four logical CPUs exposed to this WSL guest,
which correspond to two reported cores with SMT. Progress variation in both
parallel CPU and GPU modes means repeated controlled trials are needed before
drawing scheduler or hardware conclusions.

## Subsequent one-VM application optimization

The tables above remain the historical 153-word outlined baseline. After the
interpreter's `RET` polling correction, the standard one-VM GPU result was
28.0 generations/second. Inlining the nine-instruction `load_cell` routine at
all nine call sites then changed the program from 153 to 215 static words but
removed 9,216 dynamic bytecodes per generation. A fresh two-second run was:

| Implementation | gen/s/VM | Aggregate Mcell/s |
|---|---:|---:|
| GPU WarpVM | 32.50 | 0.532 |
| CPU WarpVM, 1 worker | 579.94 | 9.502 |
| Native CPU, 1 worker | 30,732.59 | 503.523 |
| Native CUDA | 59,269.43 | 971.070 |

The full census and exact-frame measurements are in
[`warplife_profile.md`](warplife_profile.md). The deterministic CPU/GPU
equivalence gate and all regressions passed before these results were
accepted.
