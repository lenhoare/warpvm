# Continuously resident compiled messaging and graphics

Measured on an NVIDIA GeForce RTX 3060 (12 GiB), driver 591.86, using 64
copies of `programs/warpc/warp_native_demo.wc`. Each VM continuously combines
cooperative memory, reduction, ballot, a complete 128x128 framebuffer render,
`FLIP`, and two-neighbour mailbox traffic.

The timed interval begins only after every VM has consumed its boot command.
Compilation/JIT, allocation, host/device setup, and shutdown are excluded.
Both engines retain all VM state on the GPU throughout the interval.

| engine | average frames/s per VM | total messages observed after 2 s |
|---|---:|---:|
| persistent bytecode interpreter | 43.46 | 74 |
| continuously resident compiled PTX | 792.55 | 1,354 |

The compiled resident engine was **18.24x faster** on this application. The
frame count is an architectural `FLIP` publication rate, not the SDL display
refresh rate (the viewer remains capped near 60 Hz). Message counts rise with
the application's frame-based pulse schedule, demonstrating that messaging
continued throughout native execution rather than being emulated between
kernel launches.

Reproduce with:

```sh
build/runtime/warpvm resident_bench build/warp_native_demo.wvm \
  --vms 64 --ms 2000
```
