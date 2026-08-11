# WarpVM Project Spec v0.1.4
## Warp C — Small Graphics-Enabled C Frontend

**Status:** Proposed implementation direction  
**Version target:** v0.1.4  
**Date:** 2026-08-11

---

## 1. Purpose

WarpVM v0.1.3 established that a canonical `.wvm` program can execute either through the interpreter or as compiled native GPU code while preserving the same architectural machine state and semantics.

v0.1.4 should add the first practical high-level language for that machine:

> **Warp C: a deliberately small C implementation designed specifically for WarpVM.**

Warp C should be written from scratch rather than porting an existing C compiler.

The compiler should preserve the character of WarpVM instead of reshaping WarpVM into a conventional byte-addressed scalar CPU merely to satisfy existing compiler assumptions.

The primary source-to-execution path becomes:

```text
Warp C source
    .wc
     |
     v
Warp C compiler
     |
     v
WarpVM assembly / bytecode
    .wvm
     |
     +--> interpreter
     |
     +--> existing WarpVM native compiler --> PTX / cubin
```

`.wvm` remains the canonical executable program representation.

Warp C is a frontend, not a replacement VM, runtime, or native backend.

---

## 2. Core Design Principle

Warp C should feel recognisably like small C while respecting the machine that actually exists.

The guiding rule is:

> **Adapt C slightly to WarpVM; do not deform WarpVM into a conventional CPU just to make C conventional.**

In particular:

- WarpVM is a 32-bit word-addressed machine;
- its natural arithmetic unit is a 32-lane warp;
- ordinary program values are often uniform across all lanes;
- lane-varying values are useful and intentional;
- framebuffer pixels are ordinary 32-bit memory words;
- compiled and interpreted execution must remain semantically equivalent;
- memory is plentiful enough that saving a few kilobytes is not worth complicating common computation.

The first compiler should therefore favour semantic clarity, direct lowering, and inspectable output over completeness or clever optimization.

---

## 3. Why Write Warp C From Scratch?

A port of an existing tiny C compiler would provide a ready-made parser and type checker, but most such compilers assume a conventional scalar, byte-addressed target.

WarpVM deliberately differs in several important ways:

- the smallest addressable memory unit is currently a 32-bit word;
- `char` can naturally occupy one complete machine word;
- pointers naturally advance in word units;
- ordinary scalar values may be held in scalar state or replicated uniformly across lanes;
- lane-varying values are a first-class machine property;
- warp shuffle, broadcast, reduction, lane ID and related operations already exist;
- the framebuffer is memory-mapped into each VM;
- `.wvm`, rather than native machine code, is the desired compiler target.

The frontend is small enough that these semantics are clearer to implement directly than to retrofit into a compiler designed around another machine.

Existing small C compilers may be consulted as references for parsing, precedence, declarators and C semantics, but their architecture should not be imported wholesale.

---

## 4. Language Identity

Call the language **Warp C**.

Recommended source extension:

```text
.wc
```

This deliberately distinguishes Warp C source from host/ISO C source because the memory model and type sizes are not host-C ABI compatible.

Recommended compiler command:

```text
warpc input.wc -o output.wvm
```

The exact executable name may follow repository conventions.

Useful optional modes:

```text
warpc input.wc --emit-asm
warpc input.wc --emit-ast
warpc input.wc --dump-uniformity
```

Do not build a large driver or build system in this slice.

---

## 5. C Model for v1

Warp C v1 should support a deliberately useful subset rather than attempting ISO C completeness.

### 5.1 Fundamental types

Required:

```c
void
int
unsigned
char
```

Semantics:

```text
int       signed 32-bit two's-complement value
unsigned  unsigned 32-bit value
char      one 32-bit WarpVM word
pointer   one 32-bit WarpVM word address
```

Warp C's smallest addressable C unit is therefore one 32-bit WarpVM word.

A coherent C-style interpretation is:

```text
CHAR_BIT       = 32
sizeof(char)   = 1
sizeof(int)    = 1
sizeof(unsigned)= 1
sizeof(pointer)= 1
```

`sizeof` values are measured in Warp C addressable units, not host bytes.

This makes pointer arithmetic natural on the existing machine without introducing byte-addressing instructions.

### 5.2 Intentionally unsupported primitive types

Do not initially implement:

```text
signed/unsigned char as distinct 8-bit storage
short
long
long long
float
double
long double
```

Reject unsupported types clearly rather than silently giving them host-C semantics.

---

## 6. Character and String Representation

Strings should remain inside WarpVM memory.

Do **not** move ordinary string computation to the host CPU as part of language semantics.

A Warp C string literal:

```c
"hello"
```

is stored conceptually as:

```text
'h'  'e'  'l'  'l'  'o'  0
```

with one 32-bit word per character.

Advantages:

- no byte extraction;
- no packed-character masking;
- no read/modify/write byte stores;
- natural `char *` pointer arithmetic;
- natural array indexing;
- no byte alignment cases;
- simple interpreter and compiler lowering.

The memory cost is deliberately accepted. Strings are expected to be relatively uncommon and VM memory is not sufficiently constrained to justify complicating ordinary computation to save a few kilobytes.

Required character features:

```c
char c = 'A';
char s[] = "hello";
char *p = s;
p++;
if (*p == 'e') { ... }
```

String literals must be zero terminated.

---

## 7. Expressions and Operators

Support the normal useful integer expression set.

### 7.1 Arithmetic

```text
+  -  *  /  %
unary +  unary -
++  --
```

### 7.2 Bitwise

```text
&  |  ^  ~
<<  >>
```

### 7.3 Comparison

```text
==  !=
<   <=   >   >=
```

### 7.4 Logical

```text
!  &&  ||
```

Preserve C short-circuit semantics.

### 7.5 Assignment

```text
=
+= -= *= /= %=
&= |= ^=
<<= >>=
```

### 7.6 Addressing

```text
&x
*p
p[i]
structure.member
pointer->member
```

### 7.7 Other useful expressions

Support:

```text
casts
sizeof
?:
comma where naturally required by C syntax
```

Do not build constant-expression machinery beyond what is needed for declarations, array sizes, `case` labels and simple compile-time constants.

---

## 8. Statements and Control Flow

Warp C v1 should include:

```text
expression statements
blocks
if / else
while
do / while
for
break
continue
return
switch / case / default
```

`switch` is explicitly part of v1.

### 8.1 Switch lowering

No new WarpVM opcode is required for `switch`.

The first compiler may lower a switch to ordinary comparisons and branches.

Example:

```c
switch (x) {
    case 0: a(); break;
    case 1: b(); break;
    default: c(); break;
}
```

may become conceptually:

```text
compare x, 0
branch case_0
compare x, 1
branch case_1
jump default
```

Requirements:

- integer controlling expression;
- compile-time integer case labels;
- `default` optional;
- normal C fall-through;
- `break` exits the switch;
- duplicate case values are compile errors.

Jump-table generation is not required in v0.1.4.

---

## 9. Declarations and Aggregate Types

Support:

```text
local variables
global variables
function parameters
function prototypes
pointers
fixed-size arrays
structs
```

`enum` is desirable if inexpensive because it pairs naturally with `switch`, but it is not required to block the first end-to-end compiler milestone.

`typedef` may be added if it remains a small parser feature; it is not more important than correctness of the basic compiler.

Do not implement unions, bitfields, flexible array members or complex C declarator edge cases merely for standards completeness.

---

## 10. Functions

Support ordinary named functions with fixed argument lists.

Example:

```c
int square(int x)
{
    return x * x;
}
```

Required:

- parameters;
- local variables;
- return values;
- nested calls;
- forward prototypes;
- `void` functions.

Define one small Warp C calling convention over the existing WarpVM register/stack model.

The exact register assignment should follow the implemented ISA and current runtime rather than being invented independently by this spec.

The ABI should define at least:

```text
argument passing
return value location
caller/callee saved temporary state
stack pointer convention
local stack layout
return continuation
```

Do not add `PUSH` or `POP` instructions merely for C. Ordinary `LOAD`, `STORE`, arithmetic and the existing call/return machinery should be used where sufficient.

Recursion is not required for v0.1.4. If it falls out naturally from the calling convention and stack implementation, it may work, but do not expand the slice specifically to support or optimize it.

---

## 11. Program Entry

Use conventional C-style entry:

```c
int main(void)
{
    ...
}
```

Returning from `main` should halt the VM cleanly.

Persistent programs may simply remain in a loop:

```c
int main(void)
{
    for (;;) {
        ...
        warp_flip();
    }
}
```

No host process model, command-line arguments, environment variables, files or operating-system ABI are required.

---

## 12. Uniform and Lane-Varying Values

This is the key WarpVM-specific compiler concept.

Ordinary C source should remain familiar. The programmer should not need a special scalar integer type merely to write:

```c
for (int i = 0; i < 100; ++i) {
    work(i);
}
```

The compiler should track whether expressions are:

```text
UNIFORM    same logical value in every lane
DIVERGENT  potentially different value per lane
```

This is a compiler property rather than a new C type qualifier in v1.

### 12.1 Basic propagation

Examples:

```text
literal                     -> uniform
global constant             -> uniform
uniform + uniform           -> uniform
uniform < uniform           -> uniform
warp_lane_id()              -> divergent
uniform + divergent         -> divergent
divergent & divergent       -> divergent
load through uniform addr   -> uniform where memory semantics permit
load through divergent addr -> divergent
```

The implementation may refine these rules as needed.

### 12.2 Physical representation of uniform values

Uniform values may be represented using whichever existing mechanism produces the cleanest code:

- scalar WarpVM registers where suitable;
- replicated vector-register values;
- lane 0 plus `BROADCAST` when useful;
- constants folded away entirely.

The language must not expose which representation was selected.

This allows common scalar C control such as loop counters and pointers to use the existing machine without requiring a parallel duplicate scalar ISA.

### 12.3 Ordinary C loops

Do **not** silently reinterpret a normal C loop as a 32-at-a-time loop.

This:

```c
for (int i = 0; i < 128; ++i)
    f(i);
```

has ordinary sequential C iteration semantics inside one VM.

Warp parallelism should initially be explicit through Warp intrinsics.

Automatic vectorization is a later compiler optimization, not a v1 language semantic.

---

## 13. Divergent Control Flow

Warp C should permit useful explicit lane-parallel programming without requiring the programmer to write assembly.

Example:

```c
int lane = warp_lane_id();

if (lane < 16)
    x = a;
else
    x = b;
```

The compiler should lower a divergent `if/else` through the existing predicate/mask machinery and reconverge at the end of the structured statement.

### 13.1 v1 restriction on divergent loops

For v0.1.4, loop continuation conditions should normally be uniform.

A loop such as:

```c
while (warp_lane_id() < x) {
    ...
}
```

where different lanes require different iteration counts may be rejected with a clear diagnostic if implementing correct masked divergent looping would substantially expand the compiler.

This restriction may be lifted later.

### 13.2 `switch`

The v1 `switch` controlling expression should be uniform.

Divergent `switch` may be added later using masks if a real program motivates it.

---

## 14. Memory Semantics

Warp C pointers are logical WarpVM word addresses.

Pointer arithmetic is therefore naturally expressed in addressable words.

Examples:

```c
int *p;
p + 1       // next int word

char *s;
s + 1       // next 32-bit Warp C character
```

Arrays and structs should be laid out as consecutive WarpVM words with simple alignment.

For v1, all supported fundamental objects are one word, so alignment can simply be one word.

Struct size is the sum of its fields plus only such padding as the compiler genuinely requires. Prefer no unnecessary padding.

### 14.1 Uniform memory accesses

When all lanes access the same address and the value is uniform, the compiler should avoid 32 redundant physical memory operations where practical.

It may use:

```text
lane-0 access + broadcast
```

or an existing scalar path where one exists.

This is particularly useful for:

- ordinary scalar variables spilled to memory;
- string traversal;
- pointer-based scalar data structures;
- sequential framebuffer drawing.

### 14.2 Divergent memory accesses

A lane-varying pointer naturally maps to the existing lane-wise `LOAD` / `STORE` semantics.

This is the mechanism that makes explicit Warp C parallelism useful.

If multiple lanes attempt conflicting writes to the same address with different values, behaviour should not silently depend on GPU race ordering.

For v0.1.4 either:

- diagnose cases the compiler can prove conflicting, or
- document such writes as invalid Warp C lane-parallel code.

Do not invent complex conflict-resolution semantics in this slice.

---

## 15. Graphics Model

Graphics requires no new C-specific or VM-specific drawing instruction.

WarpVM already provides each VM with:

```text
128 x 128 pixels
32-bit ARGB8888
one 32-bit word per pixel
memory-mapped framebuffer
```

The architectural constants remain:

```text
VIDEO_BASE   = 0x00100000
VIDEO_WIDTH  = 128
VIDEO_HEIGHT = 128
VIDEO_WORDS  = 16384
```

Pixel `(x, y)` is:

```text
VIDEO_BASE + y * 128 + x
```

A normal C pointer store should therefore be sufficient to draw a pixel.

Example low-level Warp C:

```c
#define VIDEO_BASE ((unsigned *)0x00100000)

void set_pixel(int x, int y, unsigned colour)
{
    VIDEO_BASE[y * 128 + x] = colour;
}
```

No `PIXEL`, `DRAW`, `LINE`, `RECT`, sprite or texture opcode should be added.

---

## 16. `warp.h`

Provide a tiny Warp C platform header.

Recommended interface:

```c
#ifndef WARP_H
#define WARP_H

#define WARP_VIDEO_WIDTH   128
#define WARP_VIDEO_HEIGHT  128
#define WARP_VIDEO_WORDS   16384
#define WARP_VIDEO_BASE    0x00100000u

unsigned *warp_framebuffer(void);
void warp_flip(void);

int      warp_lane_id(void);
unsigned warp_vm_id(void);

unsigned warp_argb(unsigned a, unsigned r,
                   unsigned g, unsigned b);

void warp_set_pixel(int x, int y, unsigned colour);

#endif
```

These may be implemented as a mixture of:

- compiler builtins;
- inline Warp C functions;
- ordinary library functions.

Prefer ordinary source-level implementation whenever no special instruction semantics are required.

For example, `warp_set_pixel()` should ultimately be an ordinary framebuffer store.

`warp_flip()` is special only because it lowers to the existing `FLIP` instruction.

`warp_lane_id()` and `warp_vm_id()` expose existing architectural values.

---

## 17. Pixel Example

A simple ordinary C program should work:

```c
#include <warp.h>

int main(void)
{
    unsigned *fb = warp_framebuffer();

    for (int y = 0; y < WARP_VIDEO_HEIGHT; ++y) {
        for (int x = 0; x < WARP_VIDEO_WIDTH; ++x) {
            unsigned colour;

            switch ((x / 16) & 3) {
                case 0:  colour = 0xFFFF0000u; break;
                case 1:  colour = 0xFF00FF00u; break;
                case 2:  colour = 0xFF0000FFu; break;
                default: colour = 0xFFFFFFFFu; break;
            }

            fb[y * WARP_VIDEO_WIDTH + x] = colour;
        }
    }

    warp_flip();
    return 0;
}
```

This intentionally demonstrates that:

- normal integer loops work;
- `switch` works;
- pointer arithmetic works;
- framebuffer pixels are ordinary memory;
- `FLIP` is available from C;
- no graphics syscall or GPU API is required.

---

## 18. Warp-Parallel Graphics Example

Warp C should also make the machine's native parallelism available explicitly.

Example:

```c
#include <warp.h>

int main(void)
{
    unsigned *fb = warp_framebuffer();
    int lane = warp_lane_id();

    for (int y = 0; y < 128; ++y) {
        for (int base = 0; base < 128; base += 32) {
            int x = base + lane;
            unsigned colour = warp_argb(255, x * 2, y * 2, x ^ y);
            fb[y * 128 + x] = colour;
        }
    }

    warp_flip();
    return 0;
}
```

Here:

```text
lane 0  -> pixel base + 0
lane 1  -> pixel base + 1
...
lane 31 -> pixel base + 31
```

A lane-wise `STORE` can therefore write 32 distinct pixels concurrently.

The language has not acquired a graphics vector API; this falls naturally out of pointer arithmetic plus WarpVM lane semantics.

---

## 19. Warp Intrinsics

Expose a small set of existing WarpVM architectural operations through compiler builtins or `warp.h` intrinsics.

At minimum:

```text
warp_lane_id()
warp_vm_id()
warp_flip()
```

Where implementation is straightforward, also expose current general warp operations rather than hiding them from C:

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

Exact names should be consistent and unsurprising, for example:

```c
warp_broadcast(x, lane)
warp_shuffle(x, lane)
warp_shuffle_xor(x, mask)
warp_any(predicate)
warp_all(predicate)
warp_reduce_add(x)
```

Do not add new WarpVM opcodes merely to make the C API look richer.

The C intrinsic set should initially be a thin spelling layer over instructions that already exist.

---

## 20. Preprocessing

Do not build a full C preprocessor in v0.1.4.

Provide only the minimum needed for pleasant small programs.

Required or strongly preferred:

```text
#include "file.h"
#include <warp.h>
object-like #define
```

Comments must support:

```text
// line comments
/* block comments */
```

Function-like macros, conditional compilation, token pasting, stringification and the full macro-expansion model are not required.

If even the minimal preprocessor threatens to dominate the slice, the compiler may initially inject `warp.h` builtins directly and support local constant declarations instead of general preprocessing.

Do not let preprocessing delay the core compiler.

---

## 21. Small Standard Library

Provide a tiny source-level library only where useful.

Reasonable first functions:

```text
strlen
strcmp
strcpy
memcpy
memset
```

These should operate on Warp C words and characters according to Warp C semantics.

They do not need hand-written ISA implementations.

Do not add a host CPU string side channel.

No full libc is required.

In particular, v0.1.4 does not require:

```text
printf
stdio
files
malloc/free
locale
time
host environment access
```

Host-visible output can continue to use existing WarpVM runtime/message mechanisms in later work.

---

## 22. Compiler Architecture

Prefer a small conventional frontend.

Recommended pipeline:

```text
source text
   |
   v
lexer
   |
   v
recursive-descent parser
   |
   v
typed AST
   |
   v
semantic analysis
   |
   +--> scope/name resolution
   +--> type checking
   +--> constant evaluation
   +--> uniform/divergent analysis
   |
   v
simple WarpVM code generation
   |
   v
.wvm
```

A sophisticated IR is not required initially.

A small internal three-address form may be introduced if it clearly simplifies register allocation or control flow, but do not build SSA merely because modern compilers often use it.

The compiler's job is primarily to turn a small C program into clear WarpVM code.

---

## 23. Reuse the Existing Assembler

Do not independently reimplement the WarpVM instruction encoding unless necessary.

Preferred development path:

```text
Warp C AST
   |
   v
WarpVM assembly text / assembler API
   |
   v
existing assembler
   |
   v
.wvm
```

The compiler should preferably support:

```text
--emit-asm
```

so generated assembly can be inspected directly.

Once the compiler is stable, direct bytecode emission may be added through shared assembler/encoding code if useful.

Do not fork opcode definitions.

---

## 24. Register Allocation

Keep the first allocator simple.

The compiler should understand at least three kinds of values:

```text
compile-time constants
uniform runtime values
lane-varying runtime values
```

Use the existing scalar/vector register resources sensibly.

Spill to the VM stack/RAM when necessary.

Correctness is more important than minimizing spills in the first slice, but generated code should not deliberately force all locals through memory.

Uniformity information should survive long enough to avoid obvious unnecessary lane-wise loads/stores and broadcasts.

---

## 25. ISA Policy

The first Warp C implementation should target the **current ISA**.

Do not begin v0.1.4 by adding instructions for anticipated C needs.

In particular, do not automatically add:

```text
byte load/store
PUSH / POP
C-specific call instructions
string instructions
graphics instructions
switch/jump-table instructions
```

The current architecture already has the essential ingredients for integer C:

- arithmetic and bitwise operations;
- comparisons and predicates;
- control flow;
- load/store;
- call/return machinery;
- scalar/control state;
- lane operations;
- framebuffer memory;
- `FLIP`.

If a real compiler lowering exposes a missing recurring operation, document:

1. the C construct;
2. the generated WarpVM instruction sequence;
3. why that sequence is materially awkward or expensive;
4. the proposed general opcode;
5. uses beyond C or one benchmark.

Only then consider an ISA addition.

---

## 26. Interaction With v0.1.3 Native Compilation

Warp C should not emit PTX directly.

The intended pipeline is:

```text
program.wc
   |
   v
program.wvm
   |
   +--> interpreted WarpVM
   |
   +--> existing .wvm -> PTX compiler
             |
             v
         native GPU execution
```

This preserves all v0.1.3 properties:

- `.wvm` remains canonical;
- the interpreter remains the semantic reference;
- a Warp C program may execute interpreted or compiled;
- a VM may transition execution mode at safe points;
- multiple VMs running identical Warp C programs may share one compiled artifact;
- debugging may transition back to the interpreter.

Warp C must not bypass this architecture merely for convenience.

---

## 27. Correctness Strategy

Every non-trivial compiler fixture should be validated through actual WarpVM execution.

Preferred path:

```text
.wc
 -> Warp C compiler
 -> .wvm
 -> interpreted GPU
 -> compiled GPU
```

At defined checkpoints compare complete observable state where practical.

For deterministic tests include:

- return/termination state;
- relevant registers;
- RAM;
- framebuffer;
- fault state;
- frame sequence where graphics is used.

The compiled backend and interpreter must remain equivalent.

Do not accept compiler output merely because the displayed image looks correct.

---

## 28. Required Compiler Tests

Add focused tests for at least the following.

### 28.1 Lexer / parser

- precedence;
- associativity;
- declarations;
- pointers;
- arrays;
- functions;
- comments;
- character literals;
- string literals;
- `switch` syntax.

### 28.2 Integer arithmetic

Test signed and unsigned:

```text
+ - * / %
bitwise
shifts
comparisons
```

### 28.3 Control flow

Test:

```text
if/else
while
do/while
for
break
continue
return
switch
fall-through
default
```

### 28.4 Functions

Test:

- zero/multiple arguments;
- nested calls;
- locals;
- return values;
- stack preservation.

### 28.5 Pointers / arrays / structs

Test:

- address-of;
- dereference;
- pointer increment;
- array indexing;
- struct field layout;
- pointers to structs.

### 28.6 Strings

Verify exact 32-bit-word layout for:

```c
"hello"
```

and test at least:

```text
strlen
strcmp
pointer iteration
zero termination
```

### 28.7 Uniformity

Test that:

```text
constant expressions remain uniform
ordinary loop counters remain uniform
warp_lane_id() becomes divergent
divergence propagates through arithmetic
uniform branches lower normally
divergent if/else lowers through masks
```

### 28.8 Graphics

Compile a C program that:

- writes exact known pixels;
- fills the full framebuffer;
- calls `warp_flip()`;
- produces exact expected framebuffer words;
- increments `frame_seq` correctly.

### 28.9 Parallel graphics

Use `warp_lane_id()` so one store instruction-level operation writes 32 distinct pixel addresses.

Verify exact pixel results.

### 28.10 Dual-mode equivalence

Run representative Warp C programs through both:

```text
interpreted .wvm
compiled .wvm
```

and compare complete relevant state.

---

## 29. Demonstration Programs

Add at least two small Warp C programs.

### 29.1 `hello_pixels.wc`

A conventional C-style graphics demo using:

- nested `for` loops;
- arithmetic;
- `switch`;
- `warp_set_pixel()` or direct framebuffer indexing;
- `warp_flip()`.

Its purpose is to demonstrate that Warp C is pleasant for ordinary programming.

### 29.2 `parallel_pixels.wc`

A warp-native version using:

- `warp_lane_id()`;
- lane-varying framebuffer addresses;
- 32-pixel stores;
- a simple animation or colour field;
- `warp_flip()`.

Its purpose is to demonstrate that Warp C does not hide the underlying 32-lane machine.

Optionally use `warp_vm_id()` so many VMs display visibly different patterns in the existing grid viewer.

---

## 30. End-to-End Graphics Acceptance Test

The capstone should run through the real toolchain:

```text
parallel_pixels.wc
       |
       v
     warpc
       |
       v
parallel_pixels.wvm
       |
       +--> interpreted execution
       |
       +--> compiled execution
       |
       v
existing WarpVM viewer
```

Run multiple VMs concurrently.

The viewer should visibly show independent 128×128 displays generated by Warp C programs.

At least one test should use many VMs and `warp_vm_id()` so the result makes VM independence obvious.

---

## 31. Diagnostics

The compiler should fail clearly rather than guessing.

Diagnostics should include source location for:

- syntax errors;
- unknown identifiers;
- incompatible pointer use;
- invalid dereference;
- unsupported type;
- duplicate `case`;
- non-constant case label;
- unsupported divergent loop condition where applicable;
- invalid intrinsic usage;
- register/stack resource exhaustion if encountered.

Do not spend the slice building IDE-grade diagnostics, but errors should be understandable enough to program Warp C without inspecting compiler source.

---

## 32. Debuggability

Generated WarpVM assembly should remain inspectable.

Useful compiler options:

```text
--emit-asm
--emit-ast
--dump-uniformity
```

If a Warp C program faults, the developer should be able to:

1. inspect the generated `.wvm` / assembly;
2. run it interpreted;
3. attach using existing WarpVM debugging tools;
4. inspect registers, memory, PC and framebuffer;
5. transition away from compiled execution when exact instruction-level debugging is required.

Do not build a source-level C debugger in v0.1.4.

---

## 33. Performance Philosophy

Warp C v0.1.4 is primarily a usability and semantic milestone, not an optimizing-compiler competition.

However, avoid obviously pathological lowering.

In particular:

- keep ordinary uniform loop counters out of RAM where practical;
- avoid 32 identical framebuffer stores for one ordinary scalar pixel operation where practical;
- use lane-varying `LOAD` / `STORE` directly when addresses differ by lane;
- preserve existing warp operations as direct intrinsics;
- do not insert host round trips for strings or ordinary C operations;
- do not materialize architectural state more often than existing v0.1.3 execution semantics require.

Do not add sophisticated auto-vectorization in this slice.

The programmer can explicitly expose lane parallelism using Warp intrinsics.

---

## 34. Suggested Development Slices

### Slice A — Lexer, parser and integer expressions

Implement:

```text
int
unsigned
char
variables
integer/character literals
arithmetic
bitwise
comparisons
assignment
```

Compile a trivial program to WarpVM and execute it.

Checkpoint:

```text
warpc_integer_smoke PASS
```

### Slice B — Structured control flow

Add:

```text
if/else
while
do/while
for
break
continue
switch/case/default
```

Checkpoint:

```text
warpc_control_flow PASS
warpc_switch       PASS
```

### Slice C — Functions and stack ABI

Add:

```text
functions
parameters
locals
return values
CALL/RET lowering
```

Checkpoint:

```text
warpc_functions PASS
```

### Slice D — Pointers, arrays, structs and strings

Add:

```text
pointers
arrays
structs
32-bit char/string literals
sizeof
```

Checkpoint:

```text
warpc_memory  PASS
warpc_strings PASS
```

### Slice E — Uniformity analysis

Track:

```text
uniform
divergent
```

Add direct Warp intrinsics beginning with:

```text
warp_lane_id()
warp_vm_id()
```

Support divergent `if/else` through masks.

Checkpoint:

```text
warpc_uniformity PASS
warpc_divergent_if PASS
```

### Slice F — Graphics

Add `warp.h` support for:

```text
framebuffer constants
warp_framebuffer()
warp_set_pixel()
warp_argb()
warp_flip()
```

Compile and run `hello_pixels.wc`.

Checkpoint:

```text
warpc_graphics PASS
```

### Slice G — Warp-parallel graphics

Compile and run `parallel_pixels.wc` using lane-varying framebuffer stores.

Verify 32-lane writes and display multiple independent VM screens.

Checkpoint:

```text
warpc_parallel_graphics PASS
```

### Slice H — Dual-mode acceptance

Run the same Warp C programs:

```text
interpreted
compiled
mixed-mode across multiple VMs
```

Verify exact state and framebuffer equivalence.

Checkpoint:

```text
warpc_dual_mode PASS
```

---

## 35. Deliberate Non-Goals for v0.1.4

Do not add merely for this slice:

- full ISO C conformance;
- host C ABI compatibility;
- 8-bit byte addressing;
- packed 8-bit strings;
- `short`, `long`, `long long`;
- floating point;
- full preprocessor;
- function-like macro system;
- varargs;
- function pointers;
- unions;
- bitfields;
- complex declarator edge cases;
- dynamic allocation / `malloc`;
- full libc;
- stdio / `printf`;
- files;
- OS calls;
- CPU offload for strings;
- source-level C debugger;
- LLVM;
- direct C-to-PTX compilation;
- sophisticated optimization passes;
- automatic loop vectorization;
- unrestricted divergent loops;
- divergent `switch`;
- new graphics opcodes;
- C-specific VM opcodes;
- byte load/store instructions merely for conventional C expectations.

These may be reconsidered later if actual programs justify them.

---

## 36. ISA Change Rule

v0.1.4 should begin with the assumption:

> **The existing WarpVM ISA is sufficient until a real compiler lowering proves otherwise.**

This follows the lesson from WarpLife: the architecture already contained more useful parallel machinery than the first application mapping exploited.

Before accepting any new instruction, require a concrete case showing that the current ISA produces a repeated, general and materially expensive sequence.

Likely future candidates such as sequential/strided loads, funnel shifts or three-input Boolean operations remain interesting, but they are not Warp C prerequisites.

---

## 37. Success Criteria

WarpVM v0.1.4 is complete when all of the following are true:

1. A Warp C compiler exists and is implemented specifically for WarpVM rather than ported wholesale from an existing compiler.

2. `.wc` source compiles to canonical `.wvm` programs.

3. The compiler supports at least:

```text
int
unsigned
char
pointers
arrays
structs
functions
if/else
while
do/while
for
break/continue
switch/case/default
return
```

4. `char` and strings use one 32-bit WarpVM word per character with zero termination.

5. Pointer arithmetic follows WarpVM word addressing without adding byte-addressed VM memory.

6. Ordinary scalar C loops work naturally using compiler-tracked uniform values.

7. The compiler distinguishes uniform and lane-varying values internally.

8. `warp_lane_id()` allows explicit lane-parallel programs.

9. Structured divergent `if/else` works for lane-varying predicates.

10. Graphics is available through normal memory access to the existing 128×128 ARGB8888 framebuffer.

11. A Warp C program can set a pixel with an ordinary pointer/array store.

12. `warp_flip()` lowers to the existing `FLIP` instruction.

13. A lane-parallel Warp C graphics program can write 32 distinct pixels concurrently using the current ISA.

14. The existing viewer displays Warp C-generated frames without graphics-specific compiler/runtime bypasses.

15. Representative Warp C programs produce equivalent state and framebuffer output through interpreted and compiled WarpVM execution.

16. Existing WarpVM tests remain green.

17. No new VM opcode is added without a measured or clearly demonstrated need discovered during implementation.

---

## 38. Capstone

The preferred v0.1.4 capstone is deliberately visual and simple.

Write an animated Warp C program that:

- uses ordinary C functions;
- uses integer loop counters;
- contains at least one `switch`;
- stores and uses at least one string;
- writes directly to the framebuffer;
- uses `warp_lane_id()` for a genuinely 32-lane drawing operation;
- uses `warp_vm_id()` so different VMs produce distinct images;
- calls `warp_flip()` repeatedly;
- runs correctly interpreted;
- runs correctly compiled;
- can have different VMs in different execution modes simultaneously;
- appears in the existing single-VM and multi-VM viewers.

The visual result should make the architecture understandable without introducing a graphics API.

---

## 39. Conceptual Summary

After v0.1.4 the programming stack should be:

```text
                 Warp C source (.wc)
                        |
                        v
                   Warp C compiler
                        |
                        v
                 WarpVM bytecode (.wvm)
                        |
             +----------+----------+
             |                     |
             v                     v
        interpreter          native compiler
                                   |
                                   v
                              PTX / cubin
             |                     |
             +----------+----------+
                        |
                        v
                    WarpVM state
                        |
        +---------------+---------------+
        |               |               |
       RAM           mailbox        framebuffer
                                    128 x 128
```

The important abstraction remains the same:

> **Warp C programs run on WarpVM, not on CUDA.**

CUDA is an execution substrate below the machine.

Graphics is simply memory.

Strings are simply 32-bit words.

Ordinary scalar C values are compiler-tracked uniform values.

Explicit lane-varying operations expose the 32-lane machine when useful.

And `.wvm` remains the stable boundary connecting source language, interpreter, native compilation, debugging, mutation and execution.
