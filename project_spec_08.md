# WarpVM Project Spec 08 — Explicit Uniform Values and Warp C Corrections

## 1. Purpose

This slice corrects the most important language and compiler issues found while
writing the second Warp C guinea-pig corpus in `testprojects/`.

The central language decision is:

> An ordinary Warp C value is a 32-lane warp value. A value is uniform only
> when the program says so or the compiler can establish it directly from an
> intrinsically uniform expression.

Warp C must never silently collapse an ordinary function argument to lane 0.
Uniformity is an explicit contract at declarations and function boundaries,
not a speculative whole-program guess.

This slice also contains the small lowering, inlining, diagnostic and
documentation corrections exposed by the same program corpus. It requires no
new WarpVM opcode.

---

## 2. Guinea-pig projects are the regression corpus

The existing programs in `testprojects/` were created and the following issues / improvements discussed

| Notes2 finding | Project source | Use in this slice |
|---|---|---|
| 1. Passing `WARP` to a non-inlined call | `testprojects/histogram/histogram.wc`; `testprojects/stencil/stencil.wc` | Preserve direct varying arguments such as `count_chunk(WARP, 1)` and `blur3(WARP)` across the call ABI. |
| 2. Call-frame cost | `testprojects/histogram/histogram.wc`; `testprojects/matvec/matvec.wc`; `testprojects/stencil/stencil.wc`; `testprojects/sandpile/sandpile.wc` | Preserve `call_frame_words=` output and compare vector/scalar homes after adding `uniform`. |
| 3. Function-wide value shape | `testprojects/darts/darts.wc`; `testprojects/particles/particles_test.wc` | Declare loop-control locals `uniform`; retain separate varying names such as `dart_x`; verify the diagnostic on deliberate role reuse. |
| 4. Expensive integer `?:` | `testprojects/stencil/stencil.wc` | Use the replicate-edge `blur3` selection to prove cheap predicated selection and correct lane-0 physics. |
| 5. Expensive lane-0 store | `testprojects/matvec/matvec.wc` | Use the dot-product result store to prove exact-lane lowering without a general ballot branch. |
| 6. Signed loop compares | `testprojects/histogram/histogram.wc` and the full corpus | Retain as a measurement baseline; no signed-compare optimization is required in this slice. |
| 7. Incorrect callee uniformity | `testprojects/stencil/stencil.wc`; `testprojects/forest_fire/forest.wc`; `testprojects/forest_fire/forest_test.wc` | Make `blur3` and `paint` ordinary warp-valued helpers; prove that no varying parameter is collapsed through lane 0. |
| 8. Ternary blocks inlining | `testprojects/darts/darts.wc`; `testprojects/stencil/stencil.wc`; `testprojects/sandpile/sandpile.wc` | Keep `in_circle` as the existing baseline, allow an eligible ternary helper such as `blur3`/colour selection to inline, and leave loop-heavy `topple` non-inlined. |
| 9. Locals are natural 32-wide data | `testprojects/particles/particles.wc`; `testprojects/particles/particles_test.wc` | Preserve the four-local, 32-particle model and document it from the working program. |
| 10. Varying predicate composition | `testprojects/particles/particles.wc`; `testprojects/particles/particles_test.wc`; `testprojects/histogram/histogram.wc` | Preserve `&`/`|` mask composition and test the diagnostic for varying `&&`/`||`. |
| 11. Power-of-two signed wrapping | `testprojects/rule30/rule30.wc`; `testprojects/rule30/rule30_test.wc` | Preserve `(i - 1) & 127` behaviour and the Rule 30 checksum oracle. |
| 12. Lane-owned output avoids conflicts | `testprojects/sandpile/sandpile.wc`; `testprojects/sandpile/sandpile_test.wc` | Preserve the gather-neighbours/one-owner-store formulation and document it. |
| 13 and 17. Published 64K-word RAM size | `testprojects/wave/wave.wc`; `testprojects/wave/wave_test.wc` | Compile the three-field 128×128 wave against `WARP_RAM_WORDS == 65536` and retain its four-step numeric oracle. |
| 14. Signed source versus platform ABI | `testprojects/darts/darts.wc` and every persistent visual source in the canonical list below | Continue compiling signed-only user programs through the existing unsigned graphics boundary. |
| 15. Spill path | `testprojects/sandpile/sandpile.wc`; `testprojects/sandpile/sandpile_test.wc` | Preserve `topple` as the register-allocation and spill regression. |
| 16. Closed-form collectives | `testprojects/matvec/matvec.wc`; `testprojects/rule30/rule30_test.wc`; `testprojects/sandpile/sandpile_test.wc` | Preserve reduction-based numeric oracles rather than replacing them with tables of expected cells. |

The canonical source list is:

```text
testprojects/histogram/histogram.wc
testprojects/matvec/matvec.wc
testprojects/stencil/stencil.wc
testprojects/darts/darts.wc
testprojects/rule30/rule30_test.wc
testprojects/rule30/rule30.wc
testprojects/particles/particles_test.wc
testprojects/particles/particles.wc
testprojects/sandpile/sandpile_test.wc
testprojects/sandpile/sandpile.wc
testprojects/wave/wave_test.wc
testprojects/wave/wave.wc
testprojects/forest_fire/forest_test.wc
testprojects/forest_fire/forest.wc
```

The halt-and-check variants must continue to halt with `r0=42` under both
`warpvm run` and `warpvm compiled_run`. Visual variants are compile-and-run
tests under `warpvm view`, including `--compiled` where currently supported.

For every work item, the implementation report must name the project files
actually exercised and record whether the evidence was:

- a dual-engine numeric result;
- an emitted-assembly assertion;
- a compiler diagnostic assertion;
- a persistent visual smoke run;
- or a documented, deliberately deferred measurement.

---

## 3. Motivation

Warp C is an array-first language presented through a restricted C-like
surface syntax.

Semantically:

```text
int x
```

normally denotes:

```text
x = [x0, x1, x2, ... x31]
```

Arithmetic and bitwise operations apply lane-wise without ambiguity:

```text
a + b
a * b
a & b
a | b
a ^ b
~a
```

Comparisons produce 32-lane predicates. Cross-lane operations such as
shuffle, sum, product, minimum, maximum, ballot, any and all remain explicit.

Some warp values happen to contain the same value in every active lane:

```text
[7, 7, 7, ... 7]
```

These values are **uniform**. The compiler may represent a proven-uniform
value once in scalar storage, but that is a backend representation choice. It
must not change the source-level meaning of ordinary warp-valued functions.

The guinea-pig forest-fire program exposed the present danger. A helper such
as:

```c
int paint(int state) {
    return (state == 2) ? FIRE : TREE;
}
```

was called with 32 different states. The compiler nevertheless classified the
callee parameter as uniform, read lane 0 with `S_GET`, and coloured each whole
32-pixel chunk from that one lane. The program compiled and ran but produced
four vertical columns rather than a two-dimensional forest.

This is a silent correctness failure. Explicit uniform contracts remove the
ambiguity.

---

## 4. Language rule: ordinary values are warp-valued

Unqualified values are varying-capable warp values:

```c
int state;
int paint(int state);
```

Each may contain a different value in every lane. A function call passes the
whole warp value:

```text
paint: int[32] -> int[32]
```

This remains true even when a particular call happens to supply equal values
in all lanes. Call-site uniformity may later enable specialization, but it
must not alter the declared meaning of the callee.

`WARP` is varying by definition:

```text
WARP = [0, 1, 2, ... 31]
```

Automatic locals remain lane-private. This means that four ordinary locals:

```c
int x;
int y;
int vx;
int vy;
```

naturally describe 32 independent particles without declaring four explicit
32-element arrays.

---

## 5. The `uniform` qualifier

Warp C adds `uniform` as a value-shape qualifier:

```c
uniform int row;
uniform int limit;
```

A `uniform int` is guaranteed to have one value shared by all active lanes.
It may therefore live in a scalar register or scalar stack home.

Examples:

```c
int paint(int state);

int scale(int values, uniform int factor) {
    return values * factor;
}

uniform int choose_mode(uniform int configuration) {
    return configuration & 3;
}
```

The qualifier applies independently to:

- automatic variables;
- function parameters;
- function return values;
- pointer values where supported by the current Warp C type subset.

For declarations such as:

```c
uniform int *p;
```

`uniform` describes the pointer value: every lane uses the same address. It
does not change the element type or introduce C `const`/`volatile` semantics.

There is no `$` suffix. In BASIC-family languages `$` conventionally denotes
a string and would give the wrong signal.

### 5.1 Assignment rules

The required rules are:

```text
uniform destination <- uniform expression       allowed
uniform destination <- varying expression       compile error
varying destination <- uniform expression       allowed; logically broadcast
varying destination <- varying expression       allowed
```

For example:

```c
uniform int row = 0;          // valid
int x = WARP;                 // valid
int y = row;                  // valid: uniform value used as a warp value
uniform int bad = WARP;       // compile error
```

Uniformity is a stable declaration property. A varying local does not become
uniform merely because it is later assigned `0`, and a uniform local may never
be assigned a varying expression.

This deliberately rejects variable-role reuse:

```c
int y = WARP * 4;
y = 0;
for (y = 0; y < 128; y = y + 1) { ... }  // varying loop condition: error
```

The intended spelling uses different names and declares the control value:

```c
int dart_y = WARP * 4;

for (uniform int row = 0; row < 128; row = row + 1) {
    ...
}
```

Function-wide value shape is a deliberate Warp C feature. This slice must not
implement per-assignment uniformity dataflow.

### 5.2 Expression classification

Within one function, expression shape is determined locally:

- integer literals are uniform;
- a reference to a `uniform` declaration is uniform;
- `WARP` is varying;
- a reference to an ordinary declaration is varying-capable;
- unary arithmetic or bitwise operations preserve the operand's shape;
- a lane-wise binary operation is uniform only when both operands are uniform;
- comparisons follow the same rule for their predicate result;
- shuffle and lane-indexed operations are varying unless their intrinsic
  contract proves otherwise;
- reduction results are uniform;
- built-ins must declare their result and parameter shapes explicitly.

The compiler may perform additional local constant folding, but correctness
must not depend on interprocedural uniformity inference.

---

## 6. Function calls and the ABI

Function signatures are authoritative.

### 6.1 Parameter rules

An ordinary parameter receives a warp value:

```c
int paint(int state);
```

A uniform parameter explicitly requires a uniform argument:

```c
int scale(int value, uniform int factor);
```

At a call:

```text
varying parameter <- varying argument    allowed
varying parameter <- uniform argument    allowed; logically broadcast
uniform parameter <- uniform argument    allowed
uniform parameter <- varying argument    compile error
```

Parameter shape must not be inferred by joining call sites. There is no need
for an interprocedural fixed-point analysis in this slice.

The compiler may eventually clone or specialize a varying function for an
all-uniform call as an optimization. Such specialization must remain
semantically invisible and is out of scope here.

### 6.2 Return rules

An ordinary return type accepts either varying or uniform expressions and
produces a warp value.

A `uniform` return type requires every return expression to be uniform:

```c
uniform int configuration(uniform int seed) {
    return seed & 7;
}
```

A varying return expression from a uniform-returning function is a compile
error.

### 6.3 Backend representation

The ABI must preserve declared shape across both interpreter bytecode and
generated PTX:

- ordinary parameters use vector argument storage;
- uniform parameters may use scalar argument storage;
- ordinary returns use the vector return convention;
- uniform returns may use the scalar return convention;
- call-frame accounting must include the correct homes for each class;
- a varying parameter must never be materialized by reading lane 0;
- `S_GET` or equivalent lane extraction is legal only when uniformity has been
  established by the language rules;
- passing `WARP` or another unowned vector register to a non-inlined varying
  parameter must remain correct.

Interpreter and compiled execution must observe identical semantics.

### 6.4 Functions still execute as warp functions

`uniform` qualifies values. It does not introduce a separate scalar execution
mode and does not mean that a function body executes outside the warp.

One-time VM effects, divergent effects and future scalar-control syntax remain
separate language-design questions. This slice must not infer effect
multiplicity merely from a function's return type.

---

## 7. Uniform control and mask operations

The current Warp C control model remains in force for this corrective slice:

- `for` and `while` conditions that control the whole warp must be uniform;
- collective calls must occur under uniform control;
- divergent `if` continues to use the current predicated/masked semantics;
- `&&` and `||` are permitted only where their short-circuit control is
  uniform;
- lane-varying predicate composition uses `&` and `|`.

Examples:

```c
uniform int row;

for (row = 0; row < 128; row = row + 1) {
    int x = row * 32 + WARP;
    int live = (x >= 0) & (x < 4096);

    if (live) {
        ...
    }
}
```

The documentation must explain the distinction on its first page:

```text
ordinary arithmetic: 32 lane-wise results
comparison:           32-lane predicate
& and |:              lane-wise predicate composition
&& and ||:            uniform short-circuit control only
```

---

## 8. Cheap integer selection

The guinea-pig stencil exposed unnecessarily expensive lowering for a pure
integer ternary:

```c
left = (WARP == 0) ? value : left;
```

The existing lowering constructs branch machinery using a ballot and
predicated broadcasts. A side-effect-free integer value selection does not
need that machinery.

For a pure ternary:

```c
result = condition ? true_value : false_value;
```

the compiler should:

1. evaluate the condition as a lane predicate;
2. evaluate the two pure value operands according to existing expression
   rules;
3. select with predicated vector moves;
4. avoid `BALLOT`, reconvergence scaffolding and scalar broadcasts.

The initial optimization may be deliberately narrow. It need only cover
integer operands whose evaluation has no calls, stores, messaging, control
effects or other side effects. More general conditional expressions may retain
the existing correct lowering.

Uniform ternaries should continue to use the cheapest uniform lowering.

The project-level regression is `testprojects/stencil/stencil.wc`. Its
`blur3(WARP)` path must remain a real helper call in at least one test variant,
and its replicate-edge oracle must prove that lane 0 retains the centre value
rather than accepting lane 31 after the masked shuffle.

---

## 9. Cheap exact-lane side effects

The matvec corpus exposed the same ballot tax around a conventional one-lane
store:

```c
if (WARP == 0) {
    y[row] = total;
}
```

For an exact-lane predicate such as `WARP == constant`, the compiler already
knows the active mask. It should lower a simple store or assignment directly
under that lane predicate rather than materializing a general divergent branch
through `BALLOT` and broadcasts.

The first implementation should cover:

- an equality comparison between `WARP` and a compile-time lane in `0..31`;
- a simple body containing one supported assignment or store;
- no `else` arm;
- no calls, collectives or nested control flow.

Unsupported cases retain the general lowering.

This is an optimization only. It must not change the conflict rules for
multiple lanes storing through the same address.

The project-level regression is `testprojects/matvec/matvec.wc`. Its lane-0
write of the uniform dot-product result must preserve the existing `r0=42`
dual-engine oracle while assembly inspection proves that this exact-lane case
does not create a general ballot/reconvergence sequence.

---

## 10. Inlining pure ternaries

The current inliner rejects helpers containing `?:` even when the conditional
is a pure value expression. This makes small warp-native helpers unpredictable:

```c
int clamp_edge(int neighbour, int centre) {
    return (WARP == 0) ? centre : neighbour;
}
```

A side-effect-free ternary must be treated as an expression rather than as
disqualifying control flow for inlining.

The existing size, recursion and register-pressure safeguards remain. This
change merely permits otherwise eligible functions containing pure ternaries
to be considered.

The inliner rules must be summarized as a short programmer-readable list in
`docs/warpc.md`.

Use three projects to keep the boundary honest:

- `testprojects/darts/darts.wc`: `in_circle` remains the known arithmetic
  helper that should inline;
- `testprojects/stencil/stencil.wc`: a small pure helper containing `?:` must
  become eligible;
- `testprojects/sandpile/sandpile.wc`: loop-heavy `topple` should remain a
  normal call and continue exercising call frames and spills.

---

## 11. Diagnostics

Diagnostics must make Warp C's stable value-shape rule feel intentional.

Required cases include:

### 11.1 Varying expression assigned to uniform storage

```c
uniform int row = WARP;
```

Diagnostic requirements:

- identify `row` as uniform;
- identify `WARP` or the relevant subexpression as varying;
- point to the invalid assignment;
- do not report only a later secondary failure.

Add the focused negative case to the compiler tests, then compile
`testprojects/darts/darts.wc` and
`testprojects/particles/particles_test.wc` with their scalar-control roles
declared `uniform`.

### 11.2 Varying value used for uniform control

```c
int y = WARP * 4;
y = 0;
for (; y < 128; y = y + 1) { ... }
```

The diagnostic should:

- state that the loop condition is varying;
- name `y`;
- point to its varying declaration;
- where useful, point to the expression that establishes its varying nature;
- suggest a separately named `uniform` loop variable.

Derive this regression from the original `y` reuse in
`testprojects/darts/darts.wc`, and retain a second form based on the wall
coordinate reuse in `testprojects/particles/particles_test.wc`.

### 11.3 Varying argument passed to a uniform parameter

The diagnostic should name:

- the argument expression;
- the callee and parameter;
- the parameter's `uniform` declaration.

Use a deliberately misdeclared form of the `paint` call from
`testprojects/forest_fire/forest_test.wc` as the negative regression. The real
`paint` helper remains ordinary and must compile with per-lane state.

### 11.4 Illegal short-circuit control

For varying `&&` or `||`, recommend the lane-wise `&` or `|` spelling when the
operands are suitable predicates.

Base the examples on the collision predicate in
`testprojects/particles/particles_test.wc` and the tail predicate in
`testprojects/histogram/histogram.wc`.

---

## 12. Header and documentation corrections

### 12.1 Publish RAM constants

`warp.h` must expose:

```c
#define WARP_RAM_WORDS 65536
```

It should remain consistent with the runtime's configured v0.1 VM RAM size.
If byte counts are already published, their relationship to word counts must
be explicit.

Compile `testprojects/wave/wave_test.wc` and `testprojects/wave/wave.wc`
against this public constant. Their three 128×128 fields occupy 49,152 words
and are the concrete proof that the published 65,536-word limit is usable.

### 12.2 Document automatic locals as the natural 32-element array

`docs/warpc.md` should show that:

```c
int x;
int y;
int vx;
int vy;
```

represent 32 lane-private particle records. Globals and shared VM RAM remain
appropriate for data that must be indexed across lanes or retained as an
explicit lattice.

Take the example directly from `testprojects/particles/particles.wc`, and
contrast it with the explicit shared arrays in Rule 30, matvec and sandpile.

### 12.3 Document lane-owned output

The documentation should include the histogram/stencil/sandpile rule:

> Assign each lane ownership of one output and gather, shuffle or load the
> required inputs. Do not express fan-in as conflicting stores when the same
> update can be written as one owner reading its contributors.

Use the neighbour-gather update from
`testprojects/sandpile/sandpile_test.wc` as the main example, with histogram
bin ownership and stencil output ownership as shorter companion examples.

### 12.4 Preserve useful compiler transcripts

`--emit-asm` must continue printing:

```text
call ... call_frame_words=...
```

Allocation summaries and spill counts should remain visible. They proved
valuable when deciding whether a helper was affordable and when sandpile first
exercised the spill path.

Check call-frame comments for histogram, matvec and stencil. Check that
`testprojects/sandpile/sandpile.wc` continues to report the `topple` allocation
and a non-zero spill path unless an independently correct allocator improvement
removes the need.

### 12.5 Clarify the signed-only experiment

The guinea-pig batch deliberately used signed `int` in program source. This
does not require removing the existing unsigned platform ABI in this slice.
Document that platform headers may remain an unsigned boundary while the
signed-only corpus continues to test signed lowering.

Compile every visual project, particularly wave and forest fire, without
adding user-source `unsigned` declarations merely to satisfy `warp_argb`,
`warp_framebuffer`, `warp_memset` or related platform declarations.

---

## 13. Required regressions

### 13.1 Uniform type checking

Add positive and negative compiler tests proving:

1. literals initialize uniform locals;
2. uniform arithmetic remains uniform;
3. combining a uniform and varying operand produces a varying result;
4. `WARP` cannot initialize or be assigned to a uniform local;
5. uniform arguments may be passed to either uniform or varying parameters;
6. varying arguments are rejected for uniform parameters;
7. a varying return is rejected from a uniform-returning function;
8. uniform loop control compiles;
9. varying loop control fails with the required provenance diagnostic;
10. nested helper calls preserve declared parameter and return shapes.

### 13.2 Forest-fire correctness

In `testprojects/forest_fire/forest.wc`, restore `paint` as a non-inlined helper
receiving ordinary varying parameters. Mirror the deterministic calculation in
`testprojects/forest_fire/forest_test.wc`.
Prove that:

- each lane's state independently determines its colour;
- the output does not collapse into 32-pixel columns;
- generated assembly does not use `S_GET` to materialize `paint`'s varying
  parameters;
- interpreted and compiled results agree.

The regression should use deterministic data or a framebuffer checksum/pixel
oracle rather than visual inspection alone.

### 13.3 Call ABI

Retain `count_chunk(WARP, 1)` in
`testprojects/histogram/histogram.wc` and `blur3(WARP)` in
`testprojects/stencil/stencil.wc`. Verify correct results and call-frame
accounting in both engines.

Include mixed signatures such as:

```c
int scale_bias(int value, uniform int scale, uniform int bias);
```

and verify isolated vector and scalar argument homes.

### 13.4 Selection lowering

Compile the replicate-edge ternaries in `testprojects/stencil/stencil.wc` and
inspect emitted assembly. The selection must use predicated moves and must not
contain a `BALLOT` introduced solely for the ternary.

Verify boundary-shuffle behaviour such as lane 0 retaining its centre value.

### 13.5 Exact-lane store lowering

Compile the lane-zero result store in `testprojects/matvec/matvec.wc`. Verify:

- exactly the selected lane performs the store;
- no general ballot/reconvergence sequence is emitted solely for the branch;
- neighbouring memory is unchanged;
- interpreter and compiled execution agree.

### 13.6 Inlining

Use `in_circle` in `testprojects/darts/darts.wc` as the existing positive
baseline. Make an otherwise eligible pure-ternary stencil or colour helper a
second positive case, and keep `topple` in
`testprojects/sandpile/sandpile.wc` as the intentional non-inlined case. Prove
that the inlined result matches a forced non-inlined form.

### 13.7 Corpus validation

Update the existing guinea-pig programs to declare genuinely uniform control
locals and parameters. Then run:

- histogram;
- matvec;
- stencil;
- darts;
- Rule 30;
- particles;
- sandpile;
- wave;
- forest fire.

All halt-and-check programs must pass under both interpreted and compiled
execution. All persistent visual programs must compile in both modes where
currently supported.

---

## 14. Implementation order

Implement in this order:

1. add `uniform` to the lexer, parser, AST and displayed types;
2. make ordinary locals and parameters varying-capable by default;
3. implement local expression-shape checking and assignment rules;
4. encode declared parameter and return shapes in function signatures;
5. update call checking, call frames and both execution backends;
6. land the deterministic forest-fire and call-ABI correctness regressions;
7. improve shape/provenance diagnostics;
8. update the guinea-pig corpus with explicit uniform control declarations;
9. implement cheap pure integer selection;
10. implement the narrow exact-lane store optimization;
11. permit pure ternaries in otherwise eligible inline candidates;
12. add the RAM constant and documentation examples;
13. run the full dual-engine and CTest suites.

Correctness work through step 6 must land before the code-quality
optimizations.

---

## 15. Explicit non-goals

This slice must not add:

- per-assignment uniformity analysis;
- call-site joining or whole-program uniformity inference;
- function cloning or uniform-call specialization;
- a new scalar execution engine;
- new WarpVM opcodes;
- general predicated-side-effect optimization beyond the stated narrow case;
- a signed-comparison peephole without profile evidence;
- a change to the unsigned platform ABI;
- additional VM RAM;
- general conflicting-store resolution or atomics.

Two larger language ideas arose from the same architectural discussion:

```c
for WARP { ... }
for WARP * 4 { ... }
if any (predicate) { ... }
if all (predicate) { ... }
```

These express the deeper Architecture 1 view in which loops become lanes and
control explicitly consumes warp predicates. They deserve a separate language
design slice. This specification neither implements nor forecloses them.

---

## 16. Acceptance criteria

This slice is complete when:

1. unqualified Warp C parameters reliably carry 32 lane values;
2. `uniform` declarations, parameters and returns parse and type-check;
3. varying-to-uniform assignments and calls fail at compile time with useful
   source locations;
4. the forest-fire helper produces per-lane colours with no lane-0 collapse;
5. interpreter and compiled execution agree on mixed uniform/varying calls;
6. no varying parameter is lowered through `S_GET` or equivalent lane
   extraction;
7. pure integer `?:` avoids unnecessary ballot-based branch lowering;
8. the supported exact-lane store avoids unnecessary general branch lowering;
9. pure ternary helpers can inline when otherwise eligible;
10. `WARP_RAM_WORDS` and the required programming idioms are documented;
11. the complete existing test suite and updated guinea-pig corpus pass;
12. `git diff --check` is clean.

---

## 17. Architectural outcome

After this slice, Warp C has an explicit and stable answer to its central
scalar/array question:

```text
ordinary int          one value per lane: 32 values
uniform int           one value shared across the warp
ordinary function     consumes and returns warp values
uniform annotation    an explicit equality contract and optimization license
```

The compiler remains free to place proven-uniform values in scalar registers,
but it may never manufacture uniformity by taking lane 0 from an ordinary warp
value.

The resulting principle is:

> Warp-valued computation is the language default. Uniformity is exceptional,
> explicit and checked.
