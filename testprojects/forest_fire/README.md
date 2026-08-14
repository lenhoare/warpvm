# Warp C forest fire

`forest_test.wc` is the deterministic CALL-ABI regression. Its non-inlined
`paint(int state)` helper receives three interleaved lane states and verifies
the exact colour produced by every lane. `forest.wc` is the persistent visual
version; each lane owns one pixel in every 32-pixel batch.

Build and run the deterministic check with:

```sh
warpc forest_test.wc -o forest_test.wvm
warpvm run forest_test.wvm
warpvm compiled_run forest_test.wvm
```

Run the visual version with `warpvm view forest.wvm --vms 64`, adding
`--compiled` for the resident native engine.
