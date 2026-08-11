# Compiled WarpLife Baseline

Date: 2026-08-11

This is the first v0.1.3 baseline for direct translation of the existing
WarpLife `.wvm` program to PTX. The bytecode remains canonical and is not
rewritten for the benchmark. One logical VM still maps to one hardware warp.

## Correctness gate

Before benchmarking, the compiled backend passed all of the following:

- exact CPU-interpreter/compiled state for straight-line arithmetic, a native
  control-flow loop, guarded memory, scalar loop control, and a memory fault;
- complete state, 16K-word RAM, 128x128 framebuffer, and frame-counter equality
  for WarpLife VMs 0 and 1 at YIELD;
- two independent VMs using one compiled program artifact;
- simultaneous interpreted VM 0 and compiled VM 1 progress in one runtime
  session;
- interpreted-to-compiled continuation at YIELD;
- compiled-to-interpreted continuation at YIELD.
- identical canonical program content hits a process-local compilation cache.

The compiled kernel has a stable checkpoint ABI for state, private RAM,
framebuffer, and frame sequence. Dynamic shifts explicitly preserve WarpVM's
modulo-32 semantics; invalid memory accesses fault the entire VM at the same
PC without retiring the faulting instruction.

## Artifact

```text
WarpLife bytecode words       215
Generated PTX bytes        72,427
sm_86 cubin bytes          64,168
ptxas registers/thread         54
ptxas spill loads/stores         0 / 0
Observed cold JIT             217-355 ms
Observed warm JIT               <1 ms
```

The emitted PTX has real bytecode labels and native branches. Architectural
registers are loaded once at kernel entry and materialized only at HALT/YIELD,
not after every virtual instruction.

## Throughput

The compiled interval keeps state, RAM, and framebuffer resident on the GPU.
It excludes allocation and host/device copies, and includes one host kernel
launch plus native execution per YIELD checkpoint. Each count ran for roughly
one second after an untimed initialization/warm-up generation. Interpreter and
native figures are the existing one-second `life_bench` measurements from the
same retained source and device.

| VMs | GPU interpreted gen/s/VM | GPU compiled gen/s/VM | Native CUDA gen/s/VM | Compiled / interpreted | Native / compiled | Compiled aggregate Mcell/s |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 32.00 | 574.8 | 66,433.5 | 18.0x | 115.6x | 9.4 |
| 8 | 30.00 | 498.6 | 62,439.9 | 16.6x | 125.2x | 65.3 |
| 32 | 28.59 | 492.0 | 48,159.1 | 17.2x | 97.9x | 257.9 |
| 64 | 25.69 | 489.4 | 25,865.4 | 19.0x | 52.9x | 513.2 |
| 256 | 8.04 | 282.9 | 2,830.7 | 35.2x | 10.0x | 1,186.7 |

At one VM, compiled GPU WarpLife and the single-worker CPU WarpVM interpreter
are coincidentally almost equal (574.8 versus 573.9 generations/s). They reach
that rate by very different execution paths; it is not evidence that the GPU
backend has saturated.

## Interpretation

This is a strong positive result for dual-mode WarpVM: removing fetch/decode,
dynamic opcode dispatch, and dynamic virtual-register selection improves the
unchanged program by 17-35x.

It is also an important limit result. Native CUDA remains 10-125x faster,
depending on VM count. The native reference distributes cells across many
threads, while one WarpVM intentionally serializes 512 packed-word batches
inside one VM warp. Compilation removes the interpreter mechanism but cannot
create parallelism outside the defined virtual machine. The next attribution
work should inspect generated SASS and distinguish low-level instruction and
addressing overhead from this deliberate machine-model limit before proposing
ISA changes.
