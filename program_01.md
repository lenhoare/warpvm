# WarpVM Program 01 — WarpLife

## 1. Purpose

`WarpLife` is the first substantial application written directly in WarpVM assembly.

It implements Conway's Game of Life on each VM as an independently evolving
128 × 128 cellular automaton and displays the result through the existing
128 × 128 ARGB8888 framebuffer.

The purpose is not primarily to build a Game of Life program. The purpose is
to exercise WarpVM as a real programmable 32-lane computer and expose awkward
or missing parts of the ISA, assembler, memory model, graphics path, and
runtime before a higher-level language is introduced.

The program should remain an ordinary `.wva` application. Do not add
Life-specific ISA instructions or runtime services.

---

## 2. Architectural Basis

This program assumes the existing WarpVM architecture and graphics slice:

- one logical VM maps to one 32-lane CUDA warp;
- vector arithmetic operates lane-wise;
- scalar operations are explicit;
- VM-local RAM is private;
- memory addresses are 32-bit word addresses;
- `LANEID` exposes lane identity;
- `VMID` exposes VM identity;
- comparisons produce predicates/masks;
- `BALLOT` is available as a warp-native packing primitive;
- framebuffer writes use ordinary `STORE`;
- each VM owns a 128 × 128 ARGB8888 framebuffer;
- framebuffer base is `VIDEO_BASE` / `0x00100000`;
- `FLIP` publishes a frame without blocking VM execution.

No changes to those architectural rules are part of Program 01.

If the current implemented assembler syntax differs from examples in this
document, use the implemented syntax rather than changing the assembler merely
to match this file.

---

## 3. Simulation

Each VM runs one independent Conway's Game of Life universe.

Fixed dimensions:

```text
WORLD_WIDTH   = 128
WORLD_HEIGHT  = 128
WORLD_CELLS   = 16384
```

Cell states are binary:

```text
0 = dead
1 = alive
```

Conway's rule is:

```text
alive next generation if:
    neighbour_count == 3
or:
    currently_alive && neighbour_count == 2
```

All other cells are dead in the next generation.

Updates are synchronous: every generation must be calculated entirely from
generation N before generation N+1 becomes current.

---

## 4. State Representation

### 4.1 Do Not Store One 32-bit Word Per Cell

A naïve 128 × 128 array of 32-bit cells requires:

```text
16384 words = 65536 bytes
```

and synchronous Life requires two generations, giving 128 KiB.

That is larger than the intended v0.1 VM-local RAM configuration and would
make this example unnecessarily dependent on enlarging the machine.

Program 01 therefore deliberately uses a warp-native packed representation.

### 4.2 Bit-Packed Worlds

Store one cell per bit.

Each 32-bit RAM word contains 32 consecutive horizontal cells:

```text
bit 0  -> x = 0
bit 1  -> x = 1
...
bit 31 -> x = 31
```

for the relevant 32-cell horizontal group.

Each row therefore contains:

```text
128 / 32 = 4 words
```

and one world contains:

```text
128 rows × 4 words = 512 words = 2048 bytes
```

Use two buffers:

```text
WORLD_A_BASE
WORLD_B_BASE

512 words each
1024 words total = 4096 bytes
```

This provides true synchronous double buffering while remaining comfortably
inside VM-local RAM.

Suggested initial layout:

```text
0x0000 ... 0x01FF    world A
0x0200 ... 0x03FF    world B
```

Exact locations may be adjusted if existing programs/runtime reserve low RAM.

---

## 5. Warp-Native Work Mapping

The fundamental simulation batch is **32 adjacent cells**.

For a row and 32-cell horizontal group:

```text
lane 0  handles first cell
lane 1  handles second cell
...
lane 31 handles thirty-second cell
```

Thus one warp batch evolves 32 cells concurrently.

A complete 128 × 128 generation contains:

```text
128 rows × 4 horizontal groups = 512 warp batches
```

Each batch should:

1. derive the cell's `(x, y)` from the batch position plus `LANEID`;
2. read the current cell;
3. read/extract its eight neighbours;
4. count neighbours lane-wise;
5. calculate the next-state predicate;
6. use `BALLOT` to pack the 32 lane results into one 32-bit word;
7. store that packed word once into the destination world.

The packed representation is therefore not merely a memory optimisation:
`BALLOT` naturally converts the 32 simultaneously calculated lane results
into the exact physical representation required for the next generation.

---

## 6. Reading Packed Cells

For logical cell index:

```text
index = y * 128 + x
```

derive:

```text
word_index = index >> 5
bit_index  = index & 31
```

Read:

```text
word = LOAD source_world[word_index]
cell = (word >> bit_index) & 1
```

The same operation is used for neighbour cells.

Addresses and bit positions may differ by lane. This is intended behaviour.

The implementation should favour straightforward correctness first. Do not
prematurely replace neighbour loads with clever `SHUFFLE` tricks.

A later optimisation may exploit the fact that adjacent lanes usually access
the same or neighbouring packed words.

---

## 7. Boundary Rule

Program 01 uses a **toroidal world**.

Coordinates wrap:

```text
x = -1   -> 127
x = 128  -> 0

y = -1   -> 127
y = 128  -> 0
```

The choice avoids special dead-edge behaviour and makes every cell follow the
same conceptual rule.

Because 128 is a power of two, wrapping may be implemented as:

```text
x_wrapped = x & 127
y_wrapped = y & 127
```

where convenient.

---

## 8. Writing the Next Generation

After all 32 lanes calculate their next state, produce a predicate:

```text
p_alive_next
```

Pack it:

```text
next_word = BALLOT(p_alive_next)
```

Only one lane should store the packed result.

Preferred pattern:

```text
LANEID
compare lane == 0 -> p_lane0
@p_lane0 STORE destination[word_index], next_word
```

Use the current ISA's actual `BALLOT` result/register conventions.

Do not allow 32 lanes to perform independent read-modify-write operations on
the same packed destination word.

---

## 9. Display

The simulation representation remains bit-packed in ordinary RAM.

The framebuffer is display output only.

After a generation has been completed and the new world becomes current,
render it to the 128 × 128 framebuffer.

For each 32-cell batch:

1. each lane obtains its cell state;
2. each lane calculates its framebuffer address;
3. each lane selects an ARGB8888 colour;
4. all 32 lanes `STORE` their own pixel.

Initial colours:

```text
dead  = 0xFF000000
alive = 0xFFFFFFFF
```

A later aesthetic revision may assign a VM-specific alive colour using
`VMID`, but the first correctness version should use black and white.

After the whole image is written:

```text
FLIP
```

The simulation must continue immediately after `FLIP`.

---

## 10. Initial State

The first version must be deterministic and require no new RNG facility.

Each VM should begin from a reproducible seed derived from `VMID`.

Preferred initial test hierarchy:

### 10.1 Single-VM correctness seed

Provide at least one known pattern with predictable behaviour, such as:

```text
blinker
```

This is used for automated correctness tests.

### 10.2 Multi-VM visual seed

For the interactive demonstration, generate a deterministic pseudo-random-like
bit pattern from:

```text
VMID
word_index
```

using only existing integer/bitwise instructions.

Different VMs must visibly start from different worlds.

The generator does not need cryptographic or statistically excellent random
properties. It only needs to be deterministic and produce useful Life
starting populations.

If the existing `RAND` instruction is already implemented and reproducible,
it may be used, but Program 01 must not require adding `RAND`.

---

## 11. Main Program Loop

Conceptually:

```text
initialise world A
clear world B

current = A
next    = B

forever:
    evolve current -> next
    render next -> framebuffer
    FLIP
    swap current, next
```

The swap should exchange scalar base addresses or equivalent control state;
do not physically copy the worlds.

The VM should remain resident indefinitely unless halted by the host/runtime.

---

## 12. Multi-VM Demonstration

Run the same `.wvm` program on multiple VMs.

Initial target:

```text
64 VMs
```

Each VM:

- executes identical bytecode;
- derives its distinct initial state from `VMID`;
- owns independent RAM;
- owns an independent framebuffer;
- evolves without synchronising with other VMs;
- publishes frames independently.

The existing multi-VM grid viewer should show 64 simultaneously evolving
Life worlds.

No inter-VM messaging is required for Program 01.

---

## 13. Correctness Tests

### 13.1 Packed Cell Addressing

Verify selected `(x, y)` coordinates map to the expected packed word and bit.

Include boundaries:

```text
(0, 0)
(31, 0)
(32, 0)
(127, 0)
(0, 127)
(127, 127)
```

### 13.2 Pack With `BALLOT`

Construct a known 32-lane predicate pattern and verify `BALLOT` produces the
expected 32-bit word.

### 13.3 Toroidal Neighbours

Verify neighbours wrap correctly across all four world edges and corners.

### 13.4 Still Life

Seed a standard 2 × 2 block.

After one or more generations, verify it is unchanged.

### 13.5 Oscillator

Seed a standard blinker.

Verify:

```text
generation 0 = horizontal
generation 1 = vertical
generation 2 = horizontal
```

### 13.6 Buffer Separation

Verify generation N is not modified while generation N+1 is being calculated.

### 13.7 Render Mapping

For a known packed world, verify selected framebuffer pixels are exactly:

```text
dead  -> 0xFF000000
alive -> 0xFFFFFFFF
```

### 13.8 Multiple VM Isolation

Run multiple VMs with different seeds and verify no VM changes another VM's
RAM or framebuffer.

### 13.9 Persistent Execution

Verify a running VM advances through many generations while remaining
`RUNNING` and host-inspectable.

---

## 14. Benchmarking

Program 01 is also the first application-level performance benchmark.

Measure at minimum:

```text
generations per second per VM
cell updates per second per VM
aggregate cell updates per second
```

where:

```text
cell_updates = generations × 16384
```

Measure at several VM counts where practical, for example:

```text
1
8
32
64
maximum stable/resident count
```

Record whether the viewer is enabled because host framebuffer copying and
presentation may affect measurements.

### 14.1 Interpreter Benchmark

The primary WarpVM number is simulation throughput through the real path:

```text
.wva
 -> assembler
 -> .wvm
 -> persistent CUDA interpreter
```

Do not benchmark a special hard-coded interpreter shortcut.

### 14.2 Native CUDA Comparison

After the assembly implementation is correct, add a small conventional CUDA
Game of Life implementation using the same:

```text
128 × 128 dimensions
toroidal boundaries
initial states
generation count
```

Use it only as a benchmark reference.

The native implementation is not part of WarpVM and must not influence the
VM ISA merely to improve the comparison.

Report:

```text
native CUDA cell updates / second
WarpVM cell updates / second
slowdown ratio
```

The result should be treated as an empirical architecture measurement, not as
a pass/fail target.

---

## 15. Development Slices

### Slice A — Packed State

- define memory layout;
- create deterministic initial world;
- implement packed cell extraction;
- verify packing/addressing tests.

Checkpoint:

```text
packed_state PASS
```

### Slice B — One Generation

- implement eight-neighbour counting;
- implement Conway rule;
- `BALLOT` 32 results;
- store next-generation packed words;
- verify still-life and oscillator tests.

Checkpoint:

```text
life_step PASS
```

### Slice C — Continuous Simulation

- alternate A/B buffers;
- loop indefinitely;
- verify persistent generation advancement.

Checkpoint:

```text
life_persistent PASS
```

### Slice D — Display

- expand packed cells to framebuffer pixels;
- publish with `FLIP`;
- verify render mapping;
- display one VM in the viewer.

Checkpoint:

```text
life_view PASS
```

### Slice E — Many Worlds

- seed by `VMID`;
- run 64 independent VMs;
- verify isolation;
- display the grid.

Checkpoint:

```text
life_64 PASS
```

### Slice F — Benchmark

- add generation/cell-update measurements;
- measure scaling with VM count;
- implement equivalent native CUDA reference;
- record slowdown ratio.

Checkpoint:

```text
life_benchmark PASS
```

---

## 16. Deliberate Non-Goals

Program 01 must not initially add:

- a C compiler or other high-level language;
- new Life-specific opcodes;
- architectural double-buffered graphics;
- larger VM RAM merely to make the program easier;
- interactive mouse/keyboard editing;
- user-selectable Life rules;
- sprites, text, fonts, or UI widgets;
- inter-VM communication;
- evolutionary search;
- stochastic simulation requirements;
- sophisticated graphics;
- premature `SHUFFLE`-based neighbour optimisation.

Those are possible later experiments.

---

## 17. Questions This Program Should Answer

By the end of Program 01 we should know substantially more about:

1. Is hand-written WarpVM assembly practical for a real program?
2. Does the current scalar/vector/predicate register model feel sufficient?
3. Are labels, constants and control flow pleasant enough for nested loops?
4. Is packed data awkward or natural on a 32-lane machine?
5. Does `BALLOT` feel like a genuine architectural primitive?
6. Are lane-varying memory addresses sufficiently expressive?
7. Is framebuffer output cheap enough for persistent visual workloads?
8. How much interpreter overhead exists on a realistic integer workload?
9. How does throughput scale as many independent VMs become resident?
10. What ISA or assembler changes, if any, are justified by actual use rather
   than anticipation?

Any proposed ISA change discovered during this project should be documented
with the concrete assembly sequence that motivated it before being accepted.

---

## 18. Success Criteria

Program 01 is complete when:

- the application is written in WarpVM assembly;
- one VM correctly executes Conway's Life at 128 × 128;
- state is stored in two bit-packed VM-local buffers;
- each 32-lane batch evolves 32 cells concurrently;
- `BALLOT` packs next-generation cells;
- the framebuffer displays the current world;
- `FLIP` publishes continuously;
- multiple VMs evolve independent worlds simultaneously;
- the grid viewer visibly demonstrates this independence;
- known Life patterns pass deterministic correctness tests;
- application-level throughput is measured;
- a native CUDA reference comparison produces a concrete slowdown ratio;
- no higher-level compiler is required.

The most important outcome is not graphical polish or benchmark victory.

The program succeeds if it teaches us what WarpVM assembly and the WarpVM
machine model are actually like to program.

---

## 19. Guiding Principle

Do not make WarpVM look like a scalar CPU merely to make Conway's Life easy.

Use the program to discover the natural implementation on a machine whose
primitive unit of computation is 32 lanes.

In particular, the central operation:

```text
32 cells calculated in parallel
        ↓
      BALLOT
        ↓
one packed 32-bit next-generation word
```

should be treated as the characteristic idea of Program 01.
