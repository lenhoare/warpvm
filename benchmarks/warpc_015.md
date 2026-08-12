# Warp C v0.1.5 cooperative-data benchmark

Measured on an NVIDIA GeForce RTX 3060 (12 GiB), driver 591.86. Each sample
processes 262,144 logical 32-bit words. Direct-compiled timings are the median
of five kernel-only launches; the interpreter column is the exact number of
retired WarpVM bytecodes in the logical CPU interpreter. Every compiled sample
is checked against the interpreter's complete architectural state and RAM.

| operation | words | interpreter sequential | interpreter warp | bytecode speedup | compiled sequential (ms) | compiled warp (ms) | compiled speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| copy | 32 | 4,055,106 | 737,346 | 5.50x | 97.0141 | 27.1650 | 3.57x |
| copy | 128 | 3,962,950 | 301,126 | 13.16x | 101.3892 | 9.7314 | 10.42x |
| copy | 1,024 | 3,936,070 | 173,894 | 22.63x | 93.4438 | 4.4582 | 20.96x |
| copy | 4,096 | 3,933,191 | 160,263 | 24.54x | 94.3904 | 3.7875 | 24.92x |
| fill | 32 | 3,792,962 | 729,154 | 5.20x | 75.2879 | 30.9609 | 2.43x |
| fill | 128 | 3,700,806 | 286,790 | 12.90x | 73.8215 | 9.8631 | 7.48x |
| fill | 1,024 | 3,673,926 | 157,766 | 23.29x | 73.7423 | 3.6951 | 19.96x |
| fill | 4,096 | 3,671,047 | 143,943 | 25.50x | 73.3135 | 3.0370 | 24.14x |
| add | 32 | 5,890,114 | 360,514 | 16.34x | 137.5703 | 7.2149 | 19.07x |
| add | 128 | 5,797,958 | 268,358 | 21.61x | 141.6294 | 5.9037 | 23.99x |
| add | 1,024 | 5,771,078 | 241,478 | 23.90x | 140.2951 | 5.4679 | 25.66x |
| add | 4,096 | 5,768,199 | 238,599 | 24.18x | 136.9275 | 5.1838 | 26.41x |

The comparison is deliberately source-level. The sequential program uses an
ordinary shared-memory C loop, so all 32 hardware lanes execute the same
loads/stores redundantly. The cooperative version distributes indices as
`base + WARP`; `warp_memcpy` and `warp_memset` use the same pattern with a
predicated final partial batch. It therefore measures the benefit available
to a Warp C author who expresses independent word work across the machine's
native lanes, not a comparison with optimized host `memcpy`.

The short 32-word copy/fill cases retain visible loop/setup cost. From 1,024
words onward the direct-compiled speedups settle near 20x--26x, close enough
to the 32-lane ceiling to show that the language interface exposes the GPU's
useful parallelism without a specialized memory opcode.

Reproduce with:

```sh
build/tools-rust/release/warpc programs/warpc/bench_sequential.wc \
  -o build/bench_sequential.wvm
build/tools-rust/release/warpc programs/warpc/bench_warp_native.wc \
  -o build/bench_warp_native.wvm
build/runtime/warpvm warpc_bench \
  build/bench_sequential.wvm build/bench_warp_native.wvm
```
