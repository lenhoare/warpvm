# Warp C

`warpc` is WarpVM's from-scratch C frontend. It compiles `.wc` source through
a lexer, parser, typed semantic tree, uniformity analysis, and direct WarpVM
assembly lowering. The generated assembly is passed to the existing Rust
assembler library in-process, producing a canonical `.wvm` file.

This document describes the implemented v0.1.4 Slice A and B checkpoints.
Later slices in `project_spec_01_04.md` will add functions, memory types, lane
divergence, and graphics.

## Command line

```text
warpc input.wc -o output.wvm
      [--emit-asm] [--emit-ast] [--dump-uniformity]
```

The dump switches may be combined with `-o`. If no output file is requested,
at least one dump switch is required.

## Implemented source subset

The current entry point is exactly one `int main(void)` function. Its body may
contain nested blocks, local declarations, expression statements, structured
control flow, and `return`. Locals are held in WarpVM vector registers and are
replicated uniform values in these slices.

Implemented types:

- `int`: signed 32-bit two's-complement value;
- `unsigned`: unsigned 32-bit value;
- `char`: unsigned value occupying one 32-bit WarpVM word;
- `void`: accepted for the `main` parameter list.

Implemented expressions include decimal and hexadecimal integer literals,
`u` suffixes, character literals and escapes, unary `+ - ! ~`, arithmetic,
bitwise and logical operators, comparisons, shifts, comma, simple and compound
assignment, and pre/post increment and decrement. Normal C precedence and
associativity apply. `&&` and `||` short-circuit.

Implemented statements include:

- `if` / `else`, including the normal nearest-`if` dangling-else rule;
- `while`, `do` / `while`, and all three-clause forms of `for`;
- a declaration or expression in the `for` initializer and omitted loop
  clauses;
- `break` and `continue` with nearest enclosing-target semantics;
- `switch`, `case`, `default`, fall-through, and nested switches;
- compile-time integer expressions in case labels.

Slice B requires controlling expressions to be uniform. This is checked in
the typed semantic layer rather than inferred from the current lowering.
Divergent conditions will be added with explicit mask lowering in Slice E.

The compiler deliberately does not yet accept functions, pointers, arrays,
structs, strings, casts, or graphics intrinsics.

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

Registers `r0` through `r14` hold locals and temporaries. `r15` is reserved as
assembler scratch space. A returned value is moved to `r0` before `HALT`.
Uninitialized locals are currently zeroed for deterministic execution; this
is a compiler convenience, not a promise about the eventual full-language C
semantics.

The compiler already records every expression and local as `Uniform` or
`Divergent`. All Slice A/B inputs are uniform. The representation decision is
kept explicit so later `warp_lane_id()` and divergent control-flow lowering do
not require replacing the frontend.

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
