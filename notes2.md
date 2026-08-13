# Warp C guinea-pig notes

Observations from writing programs in `testprojects/` against the v0.1.5
language, including the call-ABI and inlining slices. This is a programming
and compiler-feeling log, not an ISA wishlist. Each item names the program
that exposed it.

Constraint in force for this batch: **signed `int` only** in program source.
`unsigned` was not used, even where it would have been the cheaper loop type.

Halt-and-check programs were run through `warpvm run` and `warpvm compiled_run`.
Visual programs compile; they are meant for `warpvm view --vms 64` (and
`--compiled`).

| Program | Path | Result |
|---|---|---|
| 32-bin histogram | `testprojects/histogram/histogram.wc` | halt `r0=42`, dual-engine PASS |
| 32×32 matvec | `testprojects/matvec/matvec.wc` | halt `r0=42`, dual-engine PASS |
| 1D stencil | `testprojects/stencil/stencil.wc` | halt `r0=42`, dual-engine PASS |
| dartboard | `testprojects/darts/darts.wc` | halt `r0=42`, dual-engine PASS (includes `FLIP`) |
| Rule 30 test | `testprojects/rule30/rule30_test.wc` | halt `r0=42`, dual-engine PASS |
| Rule 30 visual | `testprojects/rule30/rule30.wc` | compiles; persistent |
| particles test | `testprojects/particles/particles_test.wc` | halt `r0=42`, dual-engine PASS |
| particles visual | `testprojects/particles/particles.wc` | compiles; persistent |
| sandpile test | `testprojects/sandpile/sandpile_test.wc` | halt `r0=42`, dual-engine PASS |
| sandpile visual | `testprojects/sandpile/sandpile.wc` | compiles; persistent |
| 2D wave test | `testprojects/wave/wave_test.wc` | halt `r0=42`, dual-engine PASS |
| 2D wave visual | `testprojects/wave/wave.wc` | compiles; persistent |

---

## 1. Passing `WARP` into a non-inlined call used to be an internal error

**Discovered by:** histogram

`count_chunk(WARP, 1)` failed with `internal error: argument register was not
saved` because `WARP` lives in `r13`, which was not in the caller-save set.

**Status:** fixed in the call-ABI slice. Histogram now binds `sample = WARP`
only as a comment of the old workaround. Stencil calls `blur3(WARP)` directly
and compiles.

**Recommendation:** keep a regression that passes `WARP` (and other unowned
registers such as the result of an inlined helper) into a non-inlined
function.

---

## 2. The cheaper call frames are visible and welcome

**Discovered by:** histogram, then matvec / stencil / sandpile after the fix

Matvec calls are a 96-word frame. Stencil's `blur3(WARP)` is also 96 words
(1 live vector save + argument + return). Sandpile's `topple` is 128 words
because `guard` stays live across the call.

That is the difference between "I will write a function" and "I will paste a
loop into `main`". The architecture wants small helpers; the ABI now mostly
allows them.

**Recommendation:** keep printing the `call … call_frame_words=` comment. It
is the most useful compiler transcript for this kind of testing.

---

## 3. A local that is ever divergent is divergent everywhere

**Discovered by:** darts, particles_test

`int y = WARP * 4;` later followed by `for (y = 0; y < 128; ++y)` is a compile
error: `for condition must be uniform`. The assignment of `125` to a local
that previously held `WARP` does not restore uniformity, so
`if (x != 126 || vx != -1)` then fails as a divergent short-circuit.

This is the sharpest Warp C edge I hit. It is defensible (function-level
uniformity is simple) and it is surprising (C programmers reuse names).

**Workaround:** different locals for divergent data and uniform control
(`dart_x` vs `row`, `wall_x` vs `x`). Bitwise `&` / `|` when a predicate is
genuinely lane-varying.

**Recommendation:** a diagnostic that names the earlier assignment which made
the local divergent would save a lot of staring. Per-assignment uniformity
would be even better, but the diagnostic alone would make the current rule
feel designed rather than accidental.

---

## 4. Shuffle first, select second

**Discovered by:** stencil

`warp_shuffle(v, WARP - 1)` on lane 0 reads lane 31, because the ISA masks the
source with 31 and `-1 & 31 == 31`. That is correct hardware and wrong physics
for a replicate-edge stencil.

The shuffle must happen under uniform control (collectives are rejected
inside divergence). The edge fix is a later `?:`:

```c
left = warp_shuffle(v, WARP - 1);
left = (WARP == 0) ? v : left;
```

That pattern is the right Warp C spelling. The generated code for the `?:` is
a ballot plus two predicated moves, which feels heavy for "keep or replace
one value".

**Recommendation:** value-select `?:` on integers should lower to predicated
`MOV`, not a ballot. The ballot is the honest lowering of a divergent *branch*;
it is not required for a select.

---

## 5. Lane-0 stores pay the same ballot tax

**Discovered by:** matvec

`acc` from `warp_reduce_add` is already uniform, yet

```c
if (WARP == 0)
    y[row] = acc;
```

becomes compare, materialise 0/1, `BALLOT`, predicated broadcasts, predicated
store. Honest (one writer) and noisy.

Writing `y[row] = acc` from every lane stores the same uniform word 32 times
to one address. That is fewer instructions and worse memory behaviour.

**Recommendation:** same as §4. A predicated scalar store of a uniform value
under `WARP == 0` should be cheap. Until it is, I would rather keep the lane-0
write than 32-way conflicts.

---

## 6. Signed loop compares are the tax of the signed-only constraint

**Discovered by:** histogram (avoided with `unsigned`), then every later program
(accepted)

`for (int row = 0; row < 32; ++row)` XORs both sides with `0x80000000` on
every trip. Histogram got ~12% fewer retired instructions by switching the
counters to `unsigned`. This batch did not.

An equality trick (`do { …; row = row + 1; } while (row != 32);`) would dodge
the bias without changing type. I did not use it: it is cute, and it would
hide the cost you wanted to keep visible.

**Recommendation:** leave signed compares as they are until a real inner loop
shows up in a compiled profile. A peephole for `0 <= x < N` with constant
non-negative `N` would be the un-cute fix.

---

## 7. Callee parameter uniformity is a lie that happens to work

**Discovered by:** stencil `blur3`

`--emit-asm` labels `blur3`'s `v` as `Uniform -> r0` even though the only call
is `blur3(WARP)`. The body still emits `SHUFFLE` of `r0`, so the divergent
argument is actually shuffled correctly. If the compiler had believed itself
and `S_GET` lane 0, the stencil would have been wrong.

**Recommendation:** parameter uniformity should be the join of call sites, or
conservatively divergent. Never lower a "uniform" parameter through lane 0
unless every call site is uniform.

---

## 8. Inlining is picky in a way that is hard to predict

**Discovered by:** darts vs stencil vs sandpile

`in_circle` (arithmetic + `<=`) inlined into the dartboard fill.
`blur3` (two shuffles + two `?:`) did not, despite having no statements that
look like control flow. `height_colour` is a candidate; `topple` correctly
stays a call (nested loops).

Inlined `in_circle` pushed darts to `vector_peak=11/13` and 8 frame spill
words. Still the right trade: 16,384 circle tests should not `CALL`.

**Recommendation:** treat `?:` as an expression for the inliner, not as
control flow. Publish the inliner rules in `docs/warpc.md` as a bullet list
a programmer can apply without reading `sema.rs`.

---

## 9. Automatic locals *are* the 32-wide array

**Discovered by:** particles

Because automatic storage is lane-private, 32 particles are four `int`s in
`main` (`x y vx vy`), not `g_x[32]`. The only cross-lane traffic is an
explicit shuffle loop for collisions. That felt like the machine, not like a
workaround.

Globals remain the right place for data the lanes share (Rule 30's tape,
sandpile's lattice, matvec's `A`).

**Recommendation:** document this in `docs/warpc.md` next to the existing
"automatic arrays are lane-private" sentence, with a particle/agent example.
It is the happiest surprise in the language.

---

## 10. Divergent `&&` / `||` are banned; bitwise is the idiom

**Discovered by:** particles, histogram tails

`hit = (k != WARP) & (x == ox) & (y == oy)` then `if (hit)`. That is the
spelling. It is fine. It is not C, and it should be in the first page of
Warp C examples.

---

## 11. Signed `& 127` wrap works, and is the right torus

**Discovered by:** Rule 30

`(i - 1) & 127` on signed `i==0` is `-1 & 127 == 127` in two's complement.
No `unsigned`, no extra `?:` for the left edge of a 128-cell ring.

**Recommendation:** keep this. It is one of the places where 32-bit words and
power-of-two sizes make Warp C nicer than byte-machine C.

---

## 12. Conflicting stores have a clean algorithmic way out

**Discovered by:** sandpile

Four neighbours donating into one cell is an illegal 32-lane write conflict
if expressed as stores. Each cell instead reads whether *its* neighbours are
toppling:

```c
recv = (n >= 4) + (s >= 4) + (e >= 4) + (w >= 4);
out  = h - ((h >= 4) ? 4 : 0) + recv;
```

No atomics, no lane-0 reduction, one store per cell. This is the same idea as
the histogram (lane owns the output, shuffle/load the inputs).

**Recommendation:** show this pattern next to `warp_ballot` in the docs. It
will come up in every CA and every stencil with fan-in.

---

## 13. 16,384 words of RAM is a real design limit

**Discovered by:** choosing project 7

A 128×128 `int` field is all of VM RAM. Two generations of Life-sized state
do not fit unless packed. Sandpile is 64×64 × 2 buffers = 8,192 words, then
2×2 pixels onto the 128×128 framebuffer.

The framebuffer does not help: it is display, not working storage, and it is
not double-buffered.

**Recommendation:** this is fine for v0.1. Programmers need the number in
`warp.h` (`WARP_RAM_WORDS 16384`) so they stop discovering it by `FAULT_MEM`.

The 64×64 wave in `testprojects/wave/` is the constructive answer: three
fields of 4,096 words (12,288), rotated by pointer, painted 2×2. Same
physics, quarter the lattice, full framebuffer.

---

## 14. The signed-only rule collides with the platform header

**Discovered by:** every visual program

`warp_vm_id`, `warp_argb`, `warp_framebuffer`, `warp_send`, and
`warp_memset` are unsigned. Graphics colours have the alpha bit set, so they
are negative as signed `int`. I used `int vm = warp_vm_id();` and
`warp_argb(255, r, g, b)` with `int` channels, and never declared `unsigned`.

That is cheating at the ABI boundary. It is also the only way to paint.

**Recommendation:** if signed-only is a compiler-testing constraint, keep it
for user types and let `warp.h` be the unsigned island. If it is a language
direction, the graphics API should take `int` (or a `colour` word) so
programs do not have to launder `0xFF000000`.

---

## 15. `topple` is the first helper that actually spills

**Discovered by:** sandpile

```
allocation topple: vector_peak=13/13 vector_homes=7 scalar_homes=2 spills=5
```

One 64×64 sweep with five neighbour loads, four compares, and an accumulate
fills the register file. The program is still correct. It is the first time
the allocator's spill path showed up in these tests, which is useful.

**Recommendation:** keep sandpile (or any 2D stencil with a halo) in the
compiler corpus. Histogram and matvec never spilled.

---

## 16. Collectives as closed forms beat magic numbers

**Discovered by:** matvec, Rule 30, sandpile

`warp_reduce_add(WARP)` and `warp_reduce_add(WARP * WARP)` are 496 and 10416.
Rule 30's weighted checksum is four chunk reductions. Sandpile's grain count
is a 32-wide reduce over the whole lattice. I would rather check those than
paste a table of cells.

---

## How the architecture felt

Good:

- `out[WARP] = f(in[WARP])` and `warp_reduce_add(row[WARP] * x[WARP])` are
  the native sentences of the machine.
- 32 particles as four locals is delightful.
- Shuffle-neighbour, ballot/reduce, and "each lane owns one output" cover
  histogram, matvec, stencil, Rule 30, collisions, and sandpile without any
  new ISA.
- Dual-engine halt checks are a high-trust way to work.

Awkward:

- Function-wide uniformity (§3) is the main way Warp C stops feeling like C.
- `?:` and `if (WARP == 0)` are semantically right and lower too loudly (§4, §5).
- Signed loops, unsigned `warp.h`, and 16 KiW RAM are constraints, not bugs.
- Clearing a 128×128 frame with `warp_set_pixel` is 512 warp stores. Fine,
  but `warp_memset` being `unsigned *` kept it out of the signed-only batch.

---

## Suggested next compiler work (priority)

1. Diagnostic (or per-assignment analysis) for §3 — this will keep biting.
2. Cheap integer select for `?:` and `if (WARP==0) x = uniform;` — §4, §5.
3. Call-site join for parameter uniformity — §7, silent correctness risk.
4. Let the inliner treat `?:` as an expression — §8.
5. `WARP_RAM_WORDS` in `warp.h` — §13.

None of these need a new opcode.

---

## 17. A 64×64 wave fits, and pointer rotation beats a fourth copy

**Discovered by:** writing the smaller wave after sandpile

`u_new = 2u - u_prev + (n+s+e+w-4u) >> 3` on a torus, pebble of 256 at
(32,32), four steps, then check a Python oracle: centre `-6`, cardinals
`170`, diagonal `85`, sum `1254`. Dual-engine PASS.

Arithmetic `>> 3` is the cheap Courant-1/8 step. Signed `/ 8` would truncate
toward zero and disagree on negative laps; the shift is both faster and the
thing I matched.

`sum(lap)` is zero on a torus, but `sum(lap >> 3)` is not, so the discrete
wave injects energy (256 → 1254 in four steps). Fine for a ripple demo;
a long-running simulation would want damping or a conservative rounding.

`wave_step` spills (`vector_peak=13/13 … spills=3`), like sandpile: 2D halo
loads fill the register file. The call frame is 224 words because the three
buffer pointers and the step counter stay live. Rotating

```c
tmp = prev; prev = u; u = nxt; nxt = tmp;
```

avoids copying 4,096 words after every step. That felt like the right use of
C pointers on this machine.

---

## How to look at the pictures

```text
build/tools-rust/release/warpc testprojects/rule30/rule30.wc -o /tmp/rule30.wvm
build/runtime/warpvm view /tmp/rule30.wvm --vms 64
build/runtime/warpvm view /tmp/rule30.wvm --vms 64 --compiled

build/tools-rust/release/warpc testprojects/particles/particles.wc -o /tmp/particles.wvm
build/runtime/warpvm view /tmp/particles.wvm --vms 64

build/tools-rust/release/warpc testprojects/sandpile/sandpile.wc -o /tmp/sandpile.wvm
build/runtime/warpvm view /tmp/sandpile.wvm --vms 64
```

Rule 30 scrolls a triangle per VM (seed shifted by `warp_vm_id`). Particles
are 32 coloured crosses bouncing with all-pairs overlap reversal. Sandpile
drops one grain a frame on the diagonal and topples in 8 parallel sweeps;
give it a few seconds to grow the identity.

```text
build/tools-rust/release/warpc testprojects/wave/wave.wc -o /tmp/wave.wvm
build/runtime/warpvm view /tmp/wave.wvm --vms 64
build/runtime/warpvm view /tmp/wave.wvm --vms 64 --compiled
```

The wave is a 64×64 torus painted 2×2. Each VM drops pebbles on a different
diagonal cell; amber is positive height, blue is negative.
