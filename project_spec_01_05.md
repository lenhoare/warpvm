# WarpVM Project Spec v0.1.5
## Warp C — Warp-Native Data Operations

**Status:** Proposed implementation direction  
**Version target:** v0.1.5  
**Date:** 2026-08-12

---

## 1. Purpose

WarpVM v0.1.4 established Warp C as a small C frontend targeting canonical WarpVM `.wvm` programs, with explicit access to lane-parallel execution and the existing framebuffer/runtime facilities.

v0.1.5 should make the 32-lane nature of the machine substantially more natural to program.

The central addition is a fundamental Warp C architectural value:

```c
WARP
```

which behaves as though the programmer had the sequence:

```text
0, 1, 2, ... 31
```

available everywhere, while physically each lane simply sees its own lane ID.

This should allow common Warp C code to be written in an array-oriented style:

```c
dst[WARP] = src[WARP] * 2;
```

rather than:

```c
for (int i = 0; i < 32; ++i)
    dst[i] = src[i] * 2;
```

The slice should also expose the existing general warp operations cleanly from Warp C and add a very small set of source-level cooperative memory helpers such as `warp_memcpy()` and `warp_memset()`.

The objective is not to add a large vector library or redesign the ISA.

The objective is:

> **Make the natural 32-wide data operation the pleasant, obvious way to write Warp C.**

---

## 2. Architectural Principle

Warp C should increasingly reflect the machine beneath it.

A WarpVM machine is physically one CUDA warp. Therefore the 32 physical execution lanes should not feel like an accelerator bolted onto ordinary C.

They should feel like the native width of the computer.

The programming rule for v0.1.5 is:

> **Scalar C remains available for control; 32-wide array operations become the normal way to express data work.**

Examples:

```c
x[WARP] = 0;
```

```c
x[WARP] += velocity[WARP];
```

```c
out[WARP] = a[WARP] + b[WARP];
```

```c
pixels[base + WARP] = colour[WARP];
```

For data larger than one warp, ordinary scalar control advances in chunks of 32:

```c
for (int base = 0; base < n; base += 32) {
    int i = base + WARP;
    if (i < n)
        dst[i] = src[i] * 2;
}
```

The loop controls chunks.

The lanes process elements inside each chunk.

---

## 3. `WARP` Is Not an Array

This distinction is fundamental.

Do **not** implement:

```c
const int WARP[32] = {0, 1, 2, ... 31};
```

There must be:

- no 32-word object in VM RAM;
- no hidden global array;
- no load from memory when `WARP` is used;
- no array indexing operation required to obtain the lane number.

Instead `WARP` is a built-in lane-varying integer expression.

Conceptually:

```text
lane 0  evaluates WARP as 0
lane 1  evaluates WARP as 1
lane 2  evaluates WARP as 2
...
lane 31 evaluates WARP as 31
```

Physically it lowers to the existing WarpVM lane-ID mechanism.

The closest architectural meaning is:

```text
WARP := current physical/logical lane number within this WarpVM
```

No new VM opcode is required.

---

## 4. Language Semantics of `WARP`

`WARP` should be a compiler-known predefined identifier.

It has Warp C type:

```c
int
```

and compiler uniformity class:

```text
DIVERGENT
```

because its value differs between lanes.

It is immutable.

Valid:

```c
int lane = WARP;
int x = base + WARP;
a[WARP] = 7;
a[base + WARP] = b[base + WARP];
int first_half = WARP < 16;
```

Invalid:

```c
WARP = 3;
++WARP;
&WARP;
```

`WARP` must not allocate storage and therefore has no address.

### 4.1 Constant terminology

`WARP` is an architectural constant in the sense that a lane's ID does not change while the WarpVM executes.

It is **not** a C compile-time integer constant expression because its runtime value differs by lane.

Therefore it must not be accepted where C requires a compile-time constant, for example:

```c
int a[WARP];        // reject
case WARP:          // reject
```

### 4.2 Name reservation

Treat `WARP` as a reserved predefined identifier in Warp C.

User declarations must not shadow it.

A clear diagnostic should be produced for:

```c
int WARP;
```

---

## 5. Relationship to `warp_lane_id()`

Keep the existing:

```c
warp_lane_id()
```

for compatibility and for code where the explicit function-style spelling improves clarity.

The two expressions must have identical observable values:

```c
WARP == warp_lane_id()
```

for every active lane.

Preferred new Warp C style should generally use:

```c
WARP
```

for indexing and lane-local arithmetic.

Example:

```c
fb[y * 128 + base + WARP] = colour;
```

rather than:

```c
int lane = warp_lane_id();
fb[y * 128 + base + lane] = colour;
```

The compiler may lower both through the same internal node or intrinsic.

---

## 6. Important Non-Feature: No New Vector Type

v0.1.5 should **not** introduce a 32-element C vector type.

Do not make `WARP` have a type such as:

```text
int32x32
```

Do not introduce general language semantics in which every expression creates a stored or abstract 32-element vector object.

The existing SIMT model already provides the desired behaviour.

This source:

```c
a[WARP] = WARP * WARP;
```

means physically:

```text
lane 0:  a[0]  = 0  * 0
lane 1:  a[1]  = 1  * 1
lane 2:  a[2]  = 2  * 2
...
lane 31: a[31] = 31 * 31
```

The useful programmer illusion is 32-wide array calculation.

The implementation remains one scalar expression per physical lane.

This keeps Warp C small and maps directly onto WarpVM.

---

## 7. Ordinary C Loops Remain Ordinary C Loops

v0.1.5 must preserve the v0.1.4 rule that conventional C loops retain conventional sequential semantics.

This:

```c
for (int i = 0; i < 128; ++i)
    f(i);
```

still means 128 sequential loop iterations inside one VM.

Do not silently auto-vectorize it into four 32-wide operations as a language rule.

The explicitly warp-native spelling is:

```c
for (int base = 0; base < 128; base += 32)
    f(base + WARP);
```

or, for direct array work:

```c
for (int base = 0; base < 128; base += 32)
    dst[base + WARP] = src[base + WARP];
```

Automatic vectorization may be investigated later as a compiler optimization, but it must not be required for the v0.1.5 programming model.

---

## 8. `WARP` and Divergent Control

Because `WARP` is divergent, expressions derived from it are normally divergent.

Examples:

```text
WARP                   -> divergent
WARP + 32              -> divergent
WARP < 16              -> divergent predicate
base + WARP            -> divergent if base is uniform
array[base + WARP]      -> lane-varying address
```

Existing Warp C divergent `if/else` lowering should therefore make this natural:

```c
if (WARP < 16)
    a[WARP] = 1;
else
    a[WARP] = 2;
```

No special syntax is required.

The compiler should reuse the existing mask/reconvergence machinery.

---

## 9. Warp-Native Intrinsic Surface

v0.1.5 should expose the general WarpVM warp operations cleanly through `warp.h` / compiler builtins.

Required surface where the corresponding architectural operation already exists:

```c
int      warp_broadcast(int value, int lane);
int      warp_shuffle(int value, int lane);
int      warp_shuffle_xor(int value, int mask);
unsigned warp_ballot(int predicate);
int      warp_any(int predicate);
int      warp_all(int predicate);
int      warp_reduce_add(int value);
int      warp_reduce_min(int value);
int      warp_reduce_max(int value);
```

Unsigned overloads or equivalent compiler handling may be provided where already natural in Warp C.

Exact internal lowering must follow the existing WarpVM ISA rather than inventing parallel C-only machinery.

### 9.1 Result semantics

At the Warp C level:

```text
warp_ballot()       -> uniform 32-bit mask
warp_any()          -> uniform boolean/int result
warp_all()          -> uniform boolean/int result
warp_reduce_add()   -> uniform reduced result
warp_reduce_min()   -> uniform reduced result
warp_reduce_max()   -> uniform reduced result
```

`warp_shuffle()` and `warp_shuffle_xor()` may produce lane-varying results.

`warp_broadcast(value, lane)` broadcasts one selected lane's value across the warp when the selected lane is uniform.

If the current ISA has narrower operand restrictions than the API above, the compiler should diagnose unsupported forms rather than silently synthesize surprising semantics.

---

## 10. Reduction Examples

Examples should be small enough that Warp C programmers can understand the physical operation immediately.

Sum 32 values:

```c
int total = warp_reduce_add(a[WARP]);
```

Find the largest lane-local score:

```c
int best = warp_reduce_max(score[WARP]);
```

Determine whether any lane found a match:

```c
int found = warp_any(candidate[WARP] == target);
```

Build a lane bitmask:

```c
unsigned mask = warp_ballot(value[WARP] > threshold);
```

These are core operations of the machine, not optional accelerator-library calls.

---

## 11. Cooperative Memory Operations

Add a tiny source-level warp memory library.

Required:

```c
void warp_memcpy(unsigned *dst, unsigned *src, unsigned words);
void warp_memset(unsigned *dst, unsigned value, unsigned words);
```

If the compiler/library already has suitable generic pointer handling, signatures may be made more C-like, but do not expand the type system merely for these functions.

The essential semantics are 32-lane cooperative processing.

Conceptual implementation of `warp_memcpy()`:

```c
void warp_memcpy(unsigned *dst, unsigned *src, unsigned words)
{
    for (unsigned base = 0; base < words; base += 32) {
        unsigned i = base + WARP;
        if (i < words)
            dst[i] = src[i];
    }
}
```

Conceptual implementation of `warp_memset()`:

```c
void warp_memset(unsigned *dst, unsigned value, unsigned words)
{
    for (unsigned base = 0; base < words; base += 32) {
        unsigned i = base + WARP;
        if (i < words)
            dst[i] = value;
    }
}
```

These should initially be ordinary Warp C source/library code.

Do **not** add `MEMCPY` or `MEMSET` opcodes for v0.1.5.

---

## 12. Coalesced Memory Intent

The preferred common case for `warp_memcpy()` and direct `base + WARP` indexing is:

```text
lane 0  accesses word base + 0
lane 1  accesses word base + 1
...
lane 31 accesses word base + 31
```

This maps the source-language structure onto the GPU's natural adjacent-lane memory access pattern.

The architecture should not require the programmer to know CUDA transaction details, but the API should make the efficient layout obvious.

The source-level rule is simply:

> **Adjacent lanes should normally operate on adjacent words.**

Do not add a complex cache/coalescing model to Warp C in this slice.

---

## 13. Sequential `memcpy()` vs `warp_memcpy()`

Retain any existing ordinary small-library `memcpy()` semantics.

The distinction should be explicit:

```text
memcpy()       ordinary sequential C-style copy
warp_memcpy()  explicit 32-lane cooperative copy
```

This gives the programmer control without hidden auto-vectorization.

A v0.1.5 benchmark should measure both on representative aligned VM-memory blocks.

The purpose is to validate that the 32-lane source structure produces the expected benefit, not to promise a particular speedup in advance.

---

## 14. Memory Bounds and Tail Handling

`warp_memcpy()` and `warp_memset()` must work for sizes that are not multiples of 32.

Example:

```c
warp_memcpy(dst, src, 100);
```

should perform:

```text
32 + 32 + 32 + 4 words
```

using a predicate/mask for the final partial warp.

The functions must preserve existing WarpVM memory isolation and fault semantics.

No lane may access beyond the requested logical range merely because the hardware width is 32.

---

## 15. Existing Messaging Remains Orthogonal

WarpVM already has VM-to-VM messaging, and Warp C programs may already use the C bindings added for the firefly demonstration.

v0.1.5 should preserve those functions and verify they continue to work with `WARP`-based code.

Do not redesign the mailbox system merely because this slice is adding warp-native data operations.

In particular, do not conflate:

```text
lane cooperation inside one VM
```

with:

```text
message passing between different VMs
```

They are different levels of parallelism.

A useful mental model is:

```text
inside VM:     WARP / shuffle / ballot / reduce / cooperative memory
between VMs:   SEND / TRY_RECV and existing Warp C wrappers
```

Collective inter-VM messaging may be investigated later if real programs justify it.

---

## 16. No New ISA by Default

v0.1.5 should primarily expose machinery that already exists and compose it better from Warp C.

`WARP` lowers to existing lane ID support.

The warp intrinsics lower to existing warp operations.

`warp_memcpy()` and `warp_memset()` are ordinary Warp C source-level loops using lane-varying addresses.

Therefore the default expected number of new WarpVM opcodes for this slice is:

```text
0
```

If implementation reveals that an operation repeatedly lowers to a materially expensive general sequence, document the evidence before changing the ISA.

The existing rule from v0.1.4 remains in force.

---

## 17. Compiler Changes

The compiler should add a dedicated AST/semantic representation for `WARP` rather than pretending it is a normal variable declaration.

Conceptually:

```text
BuiltinExpr::WarpLane
```

or equivalent.

Required compiler behaviour:

1. parse `WARP` as a predefined identifier;
2. assign type `int`;
3. mark it divergent;
4. reject assignment/address-taking/shadowing;
5. lower it directly to existing lane ID functionality;
6. allow it in ordinary arithmetic, comparisons, pointer arithmetic and subscripts;
7. preserve divergence through existing uniformity analysis;
8. emit inspectable WarpVM assembly.

Do not implement `WARP` by injecting a hidden global declaration.

---

## 18. `warp.h` Additions

The platform header should expose the warp-native surface in one obvious place.

Conceptually:

```c
#ifndef WARP_H
#define WARP_H

/* Existing graphics/runtime declarations remain. */

int      warp_lane_id(void);
unsigned warp_vm_id(void);

int      warp_broadcast(int value, int lane);
int      warp_shuffle(int value, int lane);
int      warp_shuffle_xor(int value, int mask);
unsigned warp_ballot(int predicate);
int      warp_any(int predicate);
int      warp_all(int predicate);
int      warp_reduce_add(int value);
int      warp_reduce_min(int value);
int      warp_reduce_max(int value);

void warp_memcpy(unsigned *dst, unsigned *src, unsigned words);
void warp_memset(unsigned *dst, unsigned value, unsigned words);

#endif
```

`WARP` itself should preferably be compiler-predefined rather than a textual macro in `warp.h`.

This avoids making the language's central lane coordinate depend on preprocessing.

---

## 19. Preferred Warp C Idioms

The documentation/examples should begin establishing a characteristic Warp C style.

### 19.1 One warp of values

Prefer:

```c
out[WARP] = a[WARP] + b[WARP];
```

rather than:

```c
for (int i = 0; i < 32; ++i)
    out[i] = a[i] + b[i];
```

### 19.2 More than 32 values

Prefer:

```c
for (int base = 0; base < n; base += 32) {
    int i = base + WARP;
    if (i < n)
        out[i] = a[i] + b[i];
}
```

### 19.3 Reduction

Prefer:

```c
int total = warp_reduce_add(a[WARP]);
```

rather than manually storing and sequentially summing 32 lane values.

### 19.4 Lane masks

Prefer:

```c
unsigned active = warp_ballot(score[WARP] > cutoff);
```

when the desired result is a compact warp-wide mask.

---

## 20. Graphics Example Using `WARP`

The existing framebuffer is a good visual validation tool.

A complete 128-pixel row can be processed four chunks at a time:

```c
for (int base = 0; base < 128; base += 32) {
    int x = base + WARP;
    fb[y * 128 + x] = colour(x, y);
}
```

This should become the preferred Warp C graphics idiom.

A simple animation can therefore be written with scalar outer control and 32-wide inner work:

```c
for (;;) {
    for (int y = 0; y < 128; ++y) {
        for (int base = 0; base < 128; base += 32) {
            int x = base + WARP;
            fb[y * 128 + x] = make_colour(x, y, frame);
        }
    }

    warp_flip();
    frame++;
}
```

No graphics vector API is needed.

---

## 21. Array-Oriented Programming Direction

v0.1.5 should deliberately test a broader programming idea:

> **On WarpVM, many loops over individual elements should naturally become array expressions indexed by `WARP`.**

This is not APL-style implicit whole-array evaluation and it is not compiler auto-vectorization.

It is a direct spelling of the physical machine width.

For a 32-element local problem:

```c
state[WARP] = update(state[WARP]);
```

For a 1024-element problem:

```c
for (int base = 0; base < 1024; base += 32)
    state[base + WARP] = update(state[base + WARP]);
```

This should be treated as a language-design experiment worth evaluating through real programs.

Do not add more syntax until this simple model has been used substantially.

---

## 22. Uniform Scalar Control + Divergent Data

A characteristic Warp C program should often have this shape:

```text
uniform scalar control
        |
        v
choose chunk / phase / message / state transition
        |
        v
32 lane-varying data operations
        |
        v
optional reduction / ballot
        |
        v
uniform decision
```

Example:

```c
for (int base = 0; base < n; base += 32) {
    int i = base + WARP;
    int score = 0;

    if (i < n)
        score = evaluate(item[i]);

    int best = warp_reduce_max(score);

    if (WARP == 0) {
        /* lane-specific action where appropriate */
    }
}
```

The compiler should preserve the existing distinction between uniform and divergent values so this style lowers clearly.

---

## 23. Lane 0 Usage

`WARP == 0` is a natural way to select one lane when a genuinely single-lane side effect is required.

Example:

```c
if (WARP == 0)
    do_one_lane_operation();
```

However, do not encourage lane 0 as a substitute for understanding uniform operations.

Where an operation is architecturally warp-uniform or has a dedicated scalar path, use that mechanism.

The purpose of `WARP` is primarily data parallelism, not turning the other 31 lanes off.

---

## 24. Benchmark Requirements

v0.1.5 should produce a small benchmark report rather than assuming the new source style is faster.

At minimum measure:

### 24.1 Copy

Compare:

```text
ordinary sequential memcpy
warp_memcpy
```

for representative word counts such as:

```text
32
128
1024
4096
16384
```

where VM memory limits permit.

### 24.2 Fill

Compare:

```text
ordinary sequential memset/fill
warp_memset
```

at similar sizes.

### 24.3 Direct arithmetic

Compare a sequential C loop over 32 elements with:

```c
out[WARP] = a[WARP] + b[WARP];
```

### 24.4 Engines

Where practical run representative tests through:

```text
interpreted WarpVM
compiled WarpVM
```

The compiled path is especially important because these operations are intended to map naturally onto real GPU execution.

Do not promise a speedup before measuring it.

---

## 25. Correctness Requirements

Every new intrinsic and library helper must be validated through actual WarpVM execution.

At minimum verify:

- interpreted and compiled execution agree;
- `WARP` produces exactly `0..31` across lanes;
- `WARP == warp_lane_id()` for every lane;
- lane-varying addressing writes exactly the intended 32 words;
- partial tails do not write beyond the requested range;
- reductions produce correct uniform results;
- ballot bit positions correspond exactly to lane IDs;
- shuffles move the expected lane values;
- existing messaging continues to operate correctly;
- framebuffer behaviour remains unchanged;
- existing Warp C and WarpVM tests remain green.

---

## 26. Required Tests

Add focused tests approximately equivalent to the following.

### 26.1 `WARP` values

```text
warpc_warp_values PASS
```

Write `WARP` into 32 consecutive words and verify:

```text
0 1 2 ... 31
```

### 26.2 No storage

Inspect generated assembly and verify `WARP` does not cause loads from a hidden 32-word constant array.

Checkpoint:

```text
warpc_warp_no_storage PASS
```

### 26.3 Divergence propagation

Verify:

```text
WARP                divergent
WARP + uniform      divergent
WARP < 16           divergent predicate
```

Checkpoint:

```text
warpc_warp_uniformity PASS
```

### 26.4 Array store

Compile:

```c
a[WARP] = WARP * WARP;
```

and verify all 32 results.

Checkpoint:

```text
warpc_warp_array PASS
```

### 26.5 Tail mask

Run `warp_memcpy()` and `warp_memset()` with lengths:

```text
0
1
31
32
33
63
64
65
100
```

Verify exact boundaries.

Checkpoint:

```text
warpc_warp_memory PASS
```

### 26.6 Collectives

Test:

```text
broadcast
shuffle
shuffle_xor
ballot
any
all
reduce_add
reduce_min
reduce_max
```

Checkpoint:

```text
warpc_collectives PASS
```

### 26.7 Messaging compatibility

Run a small Warp C program that uses both `WARP`-based computation and the existing VM messaging wrappers.

Checkpoint:

```text
warpc_warp_messaging PASS
```

### 26.8 Dual-mode equivalence

Run the representative v0.1.5 programs interpreted and compiled and compare observable state.

Checkpoint:

```text
warpc_015_dual_mode PASS
```

---

## 27. Suggested Development Slices

### Slice A — `WARP`

Add compiler support for the predefined lane expression.

Implement:

- parsing/name resolution;
- type `int`;
- divergent classification;
- direct lowering to lane ID;
- assignment/address/shadow diagnostics.

Checkpoint:

```text
warpc_warp_values PASS
warpc_warp_no_storage PASS
```

### Slice B — Array-oriented examples

Add programs using:

```c
a[WARP]
a[base + WARP]
```

Verify direct lane-varying loads/stores and tail masks.

Checkpoint:

```text
warpc_warp_array PASS
```

### Slice C — Existing warp collectives

Expose the current general warp operations through `warp.h` / builtins.

Checkpoint:

```text
warpc_collectives PASS
```

### Slice D — Cooperative memory library

Implement source-level:

```text
warp_memcpy
warp_memset
```

Checkpoint:

```text
warpc_warp_memory PASS
```

### Slice E — Graphics demonstration

Rewrite/add a small animated Warp C graphics program whose inner drawing work uses `base + WARP` directly.

Checkpoint:

```text
warpc_warp_graphics PASS
```

### Slice F — Messaging integration

Run lane-parallel computation and existing VM-to-VM messaging in the same program.

Checkpoint:

```text
warpc_warp_messaging PASS
```

### Slice G — Benchmark and dual-mode acceptance

Measure sequential vs warp-native data operations and verify interpreted/compiled equivalence.

Checkpoint:

```text
warpc_015_bench PASS
warpc_015_dual_mode PASS
```

---

## 28. Capstone Program

Add a small program such as:

```text
programs/warp_native_demo.wc
```

The exact visual design is unimportant, but it should be animated and make the programming model obvious.

Requirements:

- use `WARP` directly;
- use `base + WARP` to process framebuffer/data arrays;
- use at least one reduction or ballot;
- use `warp_memcpy()` or `warp_memset()` for a real purpose;
- use the existing VM messaging API at least once;
- use `warp_vm_id()` so VMs are visibly or behaviourally distinct;
- run continuously;
- display through the existing viewer;
- run interpreted;
- run compiled;
- permit different VMs to use different execution modes as already supported.

The program should be understandable from its Warp C source without requiring generated assembly to see where the parallelism comes from.

The visual source should contain characteristic lines such as:

```c
int x = base + WARP;
fb[y * 128 + x] = colour;
```

and:

```c
int total = warp_reduce_add(value[WARP]);
```

---

## 29. Deliberate Non-Goals for v0.1.5

Do not add merely for this slice:

- a general 32-element vector type;
- implicit whole-array arithmetic;
- APL-style array semantics;
- automatic loop vectorization;
- new C syntax for map/reduce;
- new memory-copy opcodes;
- arbitrary shared writable memory between VMs;
- a new mailbox architecture;
- broadcast-message opcodes merely for convenience;
- general atomics exposed to Warp C;
- shared-memory/scratchpad architectural memory;
- asynchronous global-to-shared copy;
- Tensor Core / matrix instructions;
- floating point;
- packed 8/16-bit vector types;
- CUDA APIs visible from Warp C;
- direct Warp C-to-PTX compilation;
- cuBLAS integration;
- LLVM;
- a sophisticated optimizer.

These may be investigated in later architectural slices.

---

## 30. Later GPU-Native Experiments

The following are intentionally **not** part of v0.1.5, but should remain visible as likely next experiments after the simple warp-native C surface has been measured.

### 30.1 Atomics / shared structures

Potential future operations:

```text
atomic add
atomic exchange
compare-and-swap
```

These become useful if VMs share work queues, counters, population structures or other writable global data.

Do not add them merely because CUDA provides them.

### 30.2 Fast per-VM scratch memory

Investigate whether a small explicit WarpVM scratchpad mapped to GPU shared memory would materially accelerate repeated local working sets.

This has occupancy/resource consequences and must be benchmark-driven.

### 30.3 Asynchronous prefetch

If a scratchpad exists and real workloads benefit, investigate cooperative prefetch / asynchronous copy as a later abstraction.

### 30.4 Matrix/Tensor operations

Tensor-core matrix multiply-accumulate is especially interesting because a tensor operation is itself naturally warp-cooperative.

A future WarpVM may expose this as a small matrix coprocessor-like facility rather than making matrix arithmetic the default language model.

Do not introduce matrix or floating-point semantics in v0.1.5.

---

## 31. Performance Philosophy

The purpose of this slice is not merely syntax cleanup.

It should test whether making the **warp width explicit in ordinary C source** gives both:

1. a cleaner programming model;
2. a better mapping to the hardware.

However, preserve the existing benchmark discipline:

- measure rather than assume;
- compare sequential and warp-native source;
- compare interpreted and compiled execution;
- preserve correctness checks;
- inspect generated assembly/PTX where a result is surprising;
- avoid adding opcodes to rescue a benchmark before understanding the generated code.

A disappointing result is useful if it reveals where WarpVM or the compiler still prevents natural GPU execution.

---

## 32. Success Criteria

WarpVM v0.1.5 is complete when all of the following are true:

1. Warp C has a predefined `WARP` expression.

2. `WARP` evaluates as lane IDs `0..31` with no backing array or memory load.

3. `WARP` is typed as `int` and classified as divergent.

4. `WARP` cannot be assigned, addressed or shadowed.

5. `warp_lane_id()` remains compatible and produces identical values.

6. Code such as:

```c
a[WARP] = b[WARP] + c[WARP];
```

executes correctly across all 32 lanes.

7. Code such as:

```c
a[base + WARP]
```

supports efficient 32-word chunk processing and correct partial tails.

8. Existing warp operations are exposed cleanly through Warp C where the ISA already supports them.

9. Ballot, vote and reduction results have clear Warp C semantics.

10. `warp_memcpy()` and `warp_memset()` exist as source-level 32-lane cooperative helpers.

11. No `MEMCPY`/`MEMSET` VM opcode is introduced merely for this slice.

12. Existing Warp C messaging bindings continue to work alongside `WARP`-based computation.

13. Existing graphics and framebuffer behaviour remain unchanged.

14. Representative v0.1.5 programs run correctly interpreted and compiled.

15. Existing WarpVM/Warp C tests remain green.

16. A benchmark compares ordinary sequential data loops with explicit `WARP`-based operations.

17. No shared-memory, atomic or Tensor Core architectural extension is added without a separate measured experiment.

---

## 33. Conceptual Summary

After v0.1.5, the characteristic Warp C expression should be:

```c
out[WARP] = f(in[WARP]);
```

not because Warp C has secretly become a vector language, but because one WarpVM already **is** a 32-lane computer.

At source level the programmer may think:

```text
WARP = [0, 1, 2, ... 31]
```

At machine level there is no such array.

Instead:

```text
lane 0  sees 0
lane 1  sees 1
...
lane 31 sees 31
```

and all lanes execute together.

For larger data:

```c
for (int base = 0; base < n; base += 32)
    out[base + WARP] = f(in[base + WARP]);
```

This gives Warp C a simple emerging style:

```text
scalar control
32-wide array work
warp collectives
explicit VM messaging
```

The architecture remains small.

`.wvm` remains canonical.

The interpreter and native compiler remain interchangeable execution backends.

And Warp C begins to express the actual shape of the GPU machine rather than merely borrowing C syntax for it.
