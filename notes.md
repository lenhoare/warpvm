# WarpVM Architectural Findings

This is a short evidence log for issues discovered while writing real WarpVM
programs. It is not an ISA wishlist. Each entry records the program that
exposed the issue, the concrete assembly required today, and a possible
general remedy to evaluate only after measuring the workaround.

## 1. Predicate masks cannot be stored as numeric values

**Discovered by:** Program 01, WarpLife

**Use case:** Thirty-two lanes calculate the next state of 32 adjacent Life
cells. The resulting lane mask is exactly the packed 32-bit word required by
the next-generation buffer.

The implemented instruction is:

```text
BALLOT p0, rState
```

It writes the mask to predicate register `p0`. Predicate registers can guard
instructions, combine with other predicates, and control jumps, but their
32-bit mask cannot be moved into a vector or scalar register. Consequently it
cannot be passed to `STORE`.

Current correctness workaround:

```text
BALLOT      p0, rState
MOV_I       rBits, 0
@p0 MOV_I   rBits, 1
SHL         rBits, rBits, rLane
REDUCE_OR   rPacked, rBits
@pLane0 STORE rAddress, rPacked
```

The workaround is general and preserves the current ISA, but packing costs
four instructions after `BALLOT` instead of making the ballot result directly
available as a value. WarpLife executes this sequence once per packed word:
512 times, or 2,048 workaround instructions, per generation.

Possible general remedies to evaluate later:

- make `BALLOT` write its uniform 32-bit result to a vector register;
- add a predicate-to-vector/scalar mask move;
- support a typed cross-register-class move that covers this and future
  predicate-mask consumers.

Do not choose among these until WarpLife is correct and the workaround's
frequency and measured cost are known.
