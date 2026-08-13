# Warp C

`warpc` is WarpVM's from-scratch C frontend. It compiles `.wc` source through
a lexer, parser, typed semantic tree, uniformity analysis, and direct WarpVM
assembly lowering. The generated assembly is passed to the existing Rust
assembler library in-process, producing a canonical `.wvm` file.

This document describes the implemented v0.1.4 Slice A through F compiler and
the v0.1.5 warp-native language/library slice.

## Command line

```text
warpc input.wc -o output.wvm
      [--emit-asm] [--emit-ast] [--dump-uniformity]
```

The dump switches may be combined with `-o`. If no output file is requested,
at least one dump switch is required.

## Implemented source subset

Programs may contain structure definitions, global declarations, and one or
more function declarations and definitions, with exactly one defined
`int main(void)` entry point. Bodies may contain nested blocks, local
declarations, expression statements, structured control flow, function calls,
and `return`. Ordinary scalar locals use WarpVM vector registers; aggregates
and address-taken locals use lane-private RAM.

Implemented types and memory objects:

- `int`: signed 32-bit two's-complement value;
- `unsigned`: unsigned 32-bit value;
- `char`: unsigned value occupying one 32-bit WarpVM word;
- `void`: accepted for the `main` parameter list;
- word-addressed pointers, fixed-size one-dimensional arrays, and structures;
- globals, address-taken automatic scalars, automatic arrays and structures;
- one-word-per-character, zero-terminated string literals.

Implemented expressions include decimal and hexadecimal integer literals,
`u` suffixes, character and string literals and escapes, unary `+ - ! ~`,
arithmetic, bitwise and logical operators, comparisons, shifts, comma, simple
and compound assignment, pre/post increment and decrement, and the
right-associative conditional operator `condition ? a : b`. Normal C
precedence and associativity apply. `&&`, `||`, and `?:` evaluate only the
selected expression. Memory
expressions include `&`, `*`, pointer arithmetic, subscripting, `.`, `->`, and
`sizeof` on both types and expressions. Pointer arithmetic scales by the
pointed-to object's word size; `char *` and `int *` therefore advance by one
word.

Implemented statements include:

- `if` / `else`, including the normal nearest-`if` dangling-else rule;
- `while`, `do` / `while`, and all three-clause forms of `for`;
- a declaration or expression in the `for` initializer and omitted loop
  clauses;
- `break` and `continue` with nearest enclosing-target semantics;
- `switch`, `case`, `default`, fall-through, and nested switches;
- compile-time integer expressions in case labels.

Functions support zero to four fixed parameters, forward prototypes, integer
and `void` return types, locals, nested calls, and calls appearing inside
larger expressions. Definition parameters require names; prototype parameter
names are optional. Recursion is rejected, and static call chains may not
exceed WarpVM's eight-entry architectural call stack.

The compiler conservatively inlines small, straight-line scalar helpers. An
eligible helper has at most six statements and 48 expression/statement AST
nodes, ends in one value-return statement, and contains no control flow,
nested calls, arrays, structures, pointers, or address-taking. Actual
arguments are first bound once to fresh caller-local slots; helper locals are
also fresh at every call site, so side effects, parameter assignment, and name
shadowing retain normal call semantics. Inlined result expressions keep their
ordinary uniform/divergent analysis and may therefore execute inside existing
divergent masks. Helpers outside this deliberately small policy continue to
use the unchanged WarpVM CALL/RET ABI. Recursive helpers are never candidates
and remain rejected by normal call-graph validation.

Slice B requires controlling expressions to be uniform. This is checked in
the typed semantic layer rather than inferred from the current lowering.
Slice E relaxes that restriction specifically for structured `if` / `else`.

The deliberately small declarator subset currently supports one array suffix,
ordinary pointer stars, and top-level named structure definitions. Brace
initializers, structure assignment/return, casts, and variadic functions are
not yet accepted. Character
arrays may be initialized directly from strings; scalar globals require
constant integer or string-pointer initializers.

The compiler injects the Warp intrinsics directly:

- `warp_lane_id()` returns lane 0–31 and is divergent;
- `warp_vm_id()` returns the current VM ID and is uniform within a VM.

v0.1.5 also makes `WARP` a predefined `int` expression containing the current
lane ID. It is divergent, cannot be declared, assigned, incremented, or have
its address taken, and has no backing memory object. Generated entry code
initializes the reserved lane register once. `WARP` is deliberately not an
integer constant expression, so it cannot be used as an array bound or case
label. Automatic arrays remain private to each lane; cooperative arrays should
therefore be globals or reached through pointers to shared VM RAM.

Slice F also accepts `#include <warp.h>` as its one built-in include. This is
deliberately not a general preprocessor: other `#` directives are diagnosed.
The interface and its matching documentation header at `include/warp.h`
provide:

- `WARP_VIDEO_WIDTH`, `WARP_VIDEO_HEIGHT`, `WARP_VIDEO_WORDS`, and
  `WARP_VIDEO_BASE` as unsigned constant expressions;
- `warp_framebuffer()` returning the ordinary word-addressed ARGB8888 buffer;
- `warp_argb(a, r, g, b)` masking and packing four 8-bit components;
- `warp_set_pixel(x, y, colour)` lowering to address arithmetic and `STORE`;
- `warp_flip()` lowering to the existing unguarded `FLIP` instruction.

The same built-in header exposes the existing VM mailboxes as
`warp_send(destination, type, payload)` and non-blocking
`warp_try_recv(&payload, &metadata)`. Successful receives return nonzero and
write metadata as `(type << 16) | source_vm`. Both operations are VM-wide and
therefore rejected inside divergent control flow. Mailboxes exist on resident
VMs. A program using them may run through `view`, `serve`, or `attach` in the
interpreter, or through the continuously resident PTX engine selected with
`--compiled`. The one-shot compiled runner supports bounded messaging tests,
but it is not the normal live-machine execution model.

No implicit clipping or bounds check is added to `warp_set_pixel`; invalid
coordinates retain the architecture's normal memory-fault behaviour.
`warp_flip()` is rejected inside divergent control because publication is one
VM-wide event, while framebuffer stores may be lane-predicated normally.

Warp C also provides type-generic integer `min(a, b)` and `max(a, b)`
builtins. Mixed arguments use the usual integer conversions. Unsigned values
lower directly to the existing `MIN`/`MAX` instructions; signed values use a
sign-bit bias around those same instructions. No additional VM opcodes are
required.

## Warp-wide collectives and cooperative memory

The v0.1.5 `<warp.h>` interface exposes the existing warp-wide execution
machinery as C values:

- `warp_broadcast(value, lane)`, `warp_shuffle(value, lane)`, and
  `warp_shuffle_xor(value, mask)`;
- `warp_ballot(predicate)`, `warp_any(predicate)`, and `warp_all(predicate)`;
- signed/unsigned sum, minimum, and maximum reductions;
- unsigned AND, OR, and XOR reductions.

All lanes must reach a collective under uniform control, though its value and,
for `warp_shuffle`, source-lane arguments may vary by lane. A broadcast source
lane must be uniform. `warp_shuffle_xor` currently requires a compile-time mask
from 0 to 31. Signed minimum and maximum use an explicit sign-bit transform
around the ISA's unsigned reductions. Ballot and C-valued votes are synthesized
with lane shifts, predicate selection, and bitwise reductions because WarpVM
predicate registers do not move directly into vector registers. These
restrictions and lowerings are compiler contracts, not new ISA operations.

`warp_shuffle(value, lane)` is inferred uniform when `lane` is uniform: every
lane then reads the same `value[lane & 31]`, even if `value` is divergent.
A divergent source-lane expression keeps the result divergent, and
`warp_shuffle_xor` remains lane-relative.

`warp_memcpy(dst, src, words)` and `warp_memset(dst, value, words)` are ordinary
Warp C routines injected on demand by the compiler. They process uniform
32-word batches at `base + WARP` and predicate the final partial batch with an
ordinary divergent `if`. Lengths zero and either side of the 32-lane boundary
therefore need no scalar fallback. Because these are source routines rather
than opaque intrinsics, `--emit-asm` leaves their calls, loops, loads, stores,
and tail masks inspectable.

## Integer semantics

Signed operations wrap modulo 2^32. Signed division truncates toward zero and
signed remainder follows the dividend. Right shift of `int` is arithmetic;
right shift of `unsigned` and `char` is logical. Mixed signed/unsigned binary
expressions convert to unsigned. `char` promotes to unsigned in arithmetic.

WarpVM's comparison, division, remainder, and right-shift instructions are
unsigned. The compiler preserves the source semantics with explicit lowering:
sign-bit bias for signed comparisons, magnitude/sign reconstruction for
division and remainder, and sign-fill construction for arithmetic shifts. No
new ISA operation is required.

## Current allocation and ABI

Warp C's first calling convention is deliberately small:

- `r0`–`r3` carry up to four vector arguments;
- `r0` carries a non-void vector return value;
- `r0`–`r12` hold C locals and temporaries and are caller-saved;
- `s0`–`s6` hold liveness-allocated uniform C locals and are caller-saved;
- `r13` permanently holds the lane ID;
- `r14` is the compiler's stack-address temporary;
- `r15` remains the assembler scratch register;
- `s7` is the uniform RAM stack pointer;
- `p0`–`p3` are temporary condition masks;
- architectural `CALL` / `RET` preserve return continuations.

Entry initializes `s7` to `WARP_RAM_SIZE_WORDS`, currently word address
65,536. The software stack grows down.
Every saved vector value occupies 32 consecutive words: slot lane `i` belongs
to lane `i`. A call frame currently stores every allocator-active caller
vector and scalar register in ascending register order, followed by dedicated
argument slots and, for a non-void call, one return-value slot. Arguments are
loaded into `r0`–`r3`; after `RET`, the return is parked in its stack slot,
caller registers are restored, and the result is reloaded into a free
temporary. Uniform scalar saves currently replicate the same value into all
32 lane words. This layout is deliberately conservative and preserves
genuinely lane-varying values correctly, but allocator-active intervals can be
larger than the values genuinely live across a particular call. A second
backward pass therefore annotates every real call with definition-sensitive
semantic live-out. CALL lowering saves only those register homes plus any
already-evaluated expression temporaries that must survive a nested or later
call; live values whose authoritative home is already in the function's RAM
frame need no duplicate call save.

Before assembly lowering, a small structured lifetime pass records every
local's first and last required program point. Lifetimes are conservatively
extended through loops and across both control-flow paths. Non-overlapping
locals reuse the same home: uniform locals prefer `s0`–`s6`, while divergent
locals use vector homes. Allocation begins with eight vector homes, leaving
five transient registers, and retries a function with seven or six homes only
when its actual expression lowering needs more transient capacity.

When genuinely overlapping local pressure exceeds those homes, scalar locals
spill into the existing lane-private function frame. Arrays, structures, and
address-taken scalars use that frame unconditionally. A frame is
lane-major: all words of lane 0's C objects are consecutive, followed by lane
1's, and so on. Consequently an automatic `p + 1` is a physical address
increment of one, while the same logical automatic object remains private to
each lane. Function-local frames and allocator spill slots coexist with the
older word-major caller-save spill frames; the latter are never exposed as C
pointers.

`--emit-asm` prints each local's uniformity and chosen home, leaves all reloads
and stores visible, and appends an allocation summary to each function, for
example:

```text
; allocation main: vector_peak=10/13 vector_homes=8 scalar_homes=2 spills=7 frame_words=7
```

Globals and literal strings grow upward from RAM word zero and are shared by
the lanes of one VM. Generated entry code initializes nonzero data before
calling `main`; this avoided any WVM image-format or runtime-loader change.
Arrays and structure fields are consecutive with one-word alignment and no
padding. `sizeof(char)`, `sizeof(int)`, `sizeof(unsigned)`, and
`sizeof(pointer)` are all one. Strings use one 32-bit word per character plus
a zero word.

Returning from `main` moves the result to `r0` and executes `HALT`; other
functions use `RET`. A non-void function that reaches its closing brace
currently returns zero deterministically. Uninitialized locals are likewise
zeroed for deterministic execution; neither behaviour is a promise about an
eventual broader C implementation.

## Uniformity and divergent control

The typed semantic tree records every expression and local as `Uniform` or
`Divergent`. Literals, `warp_vm_id()`, and ordinary loop counters begin
uniform. `warp_lane_id()` is divergent, and binary expressions join their
operands. Initializers, assignments, increments, and function arguments
propagate the result. An order-independent call-graph summary marks a function
result divergent when that function, or a transitive callee, uses the lane ID;
forward prototypes cannot hide divergence.

Uniform `if` statements retain the original comparison-and-jump lowering.
For a divergent condition, the compiler ballots the materialized condition
into `p3`, emits the then side guarded by `@p3`, complements the mask for the
else side, and reconverges simply by returning to unguarded emission. A nested
divergent `if` intersects its condition with the parent mask in `p2`. `p0` and
`p1` remain comparison/temporary masks. Both branch bodies are linear in the
bytecode and only their selected lanes write registers or memory.

Slice E deliberately supports two nested divergent mask levels. It diagnoses
deeper nesting, divergent loop conditions and switches, calls or returns
inside divergent regions, divergent `break` / `continue`, and divergent
short-circuit expressions. Those constructs need either more predicate
storage or mask-aware control-transfer semantics; they are rejected rather
than guessed. Uniform loops, switches, calls, returns, and short-circuit
expressions are unchanged.

## Verification

`warpc_integer_smoke` checks signed division, remainder, arithmetic shift and
comparison; mixed unsigned comparison; word-sized character values; compound
assignment; and short-circuit evaluation. It must halt with `r0 = 42` in the
interpreter, then match complete architectural state, RAM, framebuffer, and
frame sequence in the PTX-compiled backend.

`warpc_control_flow` covers both arms of conditionals, dangling `else`, all
three loop forms, declaration-scoped `for`, and loop `break` / `continue`.
`warpc_switch` covers dispatch, no-match default, signed case values,
fall-through, default before a later case, and switch/loop jump nesting. Both
must also return 42 with exact interpreter/PTX equivalence.

`warpc_functions` covers forward prototypes, two- and four-argument calls,
void functions, register preservation, calls nested in expressions, and calls
from one non-main function to another. It returns 42 and compares registers,
scalar state, predicates, architectural call stack, RAM, framebuffer, and
frame sequence between the interpreter and PTX backend.

`warpc_memory` covers a global scalar and array, address-of and dereference,
an address-taken scalar, pointer increment, fixed arrays, structure layout,
`.` and `->`, and aggregate `sizeof`. `warpc_strings` implements source-level
`strlen` and `strcmp`, iterates pointers, checks the exact six-word
`"hello\0"` shape and zero terminator, and reads a globally initialized string
pointer. Both return 42 and pass exact interpreter/PTX state and RAM
equivalence.

`warpc_uniformity` checks direct `VMID` lowering, lane-ID propagation through
arithmetic, an ordinary uniform loop counter, a uniform branch that remains a
normal jump, and a divergent branch that becomes a ballot mask.
`warpc_divergent_if` covers both sides of an outer lane split, nested splits in
each half, masked lane-private RAM stores/loads, complementary else masks, and
reconvergence to `r0 = 42` in all lanes. Both match complete interpreter/PTX
state, predicate masks, and RAM.

`warpc_graphics` compiles `programs/warpc/hello_pixels.wc`, fills and reads back
all 16,384 framebuffer words, and publishes exactly one frame. The one-shot
GPU interpreter and PTX backend must both halt with `r0 = 42` and exact state,
RAM, framebuffer, and frame-sequence equivalence. A persistent-runtime check
then verifies the 128x128 ARGB8888 format, `frame_seq = 1`, and known red,
green, blue, and white pixels through the attach inspection path.

`warpc_firefly` runs 64 persistent copies of the Warp C firefly demo. Its
uniform 512-iteration render loop combines each batch with `WARP`,
so every framebuffer store writes 32 distinct pixels. The acceptance check
requires all 64 VMs to remain running without faults, a positive frame
sequence, and a positive received-message counter in VM-private RAM.

The v0.1.5 acceptance set additionally proves exact `WARP` values with no
backing load/store; collective results including signed extrema and exact
ballot masks; cooperative copy/fill tails at lengths 0, 1, 31, 32, 33, 63,
64, 65, and 100; and full-frame cooperative graphics. A bounded,
messaging-free native demo combines all four features and must match complete
interpreter/PTX state, RAM, framebuffer, and frame sequence. Its persistent
counterpart boots 64 communicating VMs in either engine, and the live checks
require all to remain healthy, publish frames, receive a real neighbour
message, and respond to explicit pause/resume control.

The paired sequential/cooperative data benchmark checks every direct-compiled
sample against the logical interpreter before recording timings. On the RTX
3060, large-array direct-compiled copy/fill/add improve by approximately
20x--26x. The full methodology, the smaller cases, and the important caveat
that the shared sequential C loop is redundantly executed by all lanes are in
[`benchmarks/warpc_015.md`](../benchmarks/warpc_015.md).
