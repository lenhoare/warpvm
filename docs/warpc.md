# Warp C

`warpc` is WarpVM's from-scratch C frontend. It compiles `.wc` source through
a lexer, parser, typed semantic tree, uniformity analysis, and direct WarpVM
assembly lowering. The generated assembly is passed to the existing Rust
assembler library in-process, producing a canonical `.wvm` file.

This document describes the implemented v0.1.4 Slice A through D checkpoints.
Later slices in `project_spec_01_04.md` will add lane divergence and graphics.

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
- `void`: accepted for the `main` parameter list.
- word-addressed pointers, fixed-size one-dimensional arrays, and structures;
- globals, address-taken automatic scalars, automatic arrays and structures;
- one-word-per-character, zero-terminated string literals.

Implemented expressions include decimal and hexadecimal integer literals,
`u` suffixes, character and string literals and escapes, unary `+ - ! ~`,
arithmetic, bitwise and logical operators, comparisons, shifts, comma, simple
and compound assignment, and pre/post increment and decrement. Normal C
precedence and associativity apply. `&&` and `||` short-circuit. Memory
expressions include `&`, `*`, pointer arithmetic, subscripting, `.`, `->`, and `sizeof` on both
types and expressions. Pointer arithmetic scales by the pointed-to object's
word size; `char *` and `int *` therefore advance by one word.

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

Slice B requires controlling expressions to be uniform. This is checked in
the typed semantic layer rather than inferred from the current lowering.
Divergent conditions will be added with explicit mask lowering in Slice E.

The deliberately small declarator subset currently supports one array suffix,
ordinary pointer stars, and top-level named structure definitions. Brace
initializers, structure assignment/return, casts, conditional expressions,
variadic functions, and graphics intrinsics are not yet accepted. Character
arrays may be initialized directly from strings; scalar globals require
constant integer or string-pointer initializers.

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
- `r13` permanently holds the lane ID;
- `r14` is the compiler's stack-address temporary;
- `r15` remains the assembler scratch register;
- `s7` is the uniform RAM stack pointer;
- `p0`–`p3` are temporary condition masks;
- architectural `CALL` / `RET` preserve return continuations.

Entry initializes `s7` to word address 16,384. The software stack grows down.
Every saved vector value occupies 32 consecutive words: slot lane `i` belongs
to lane `i`. A call frame stores live caller registers in ascending register
order and, for non-void calls, one return-value slot after them. Arguments are
then loaded into `r0`–`r3`; after `RET`, the return is parked in its stack slot,
caller registers are restored, and the result is reloaded into a free
temporary. This layout is somewhat verbose but already preserves genuinely
lane-varying values correctly.

Ordinary scalar automatic locals remain in registers. Arrays, structures, and
address-taken scalars use a function-local RAM frame below `s7`. A frame is
lane-major: all words of lane 0's C objects are consecutive, followed by lane
1's, and so on. Consequently an automatic `p + 1` is a physical address
increment of one, while the same logical automatic object remains private to
each lane. Function-local frames coexist with the older word-major caller-save
spill frames; the latter are never exposed as C pointers.

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

The compiler already records every expression and local as `Uniform` or
`Divergent`. All Slice A–D inputs are logically uniform. The representation
decision is kept explicit so later `warp_lane_id()` and divergent control-flow
lowering do not require replacing the frontend.

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
