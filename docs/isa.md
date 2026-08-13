# WarpVM ISA — v0.1

Status: **contract for v0.1** (slices 1–7). Both the C++ interpreter and the
Rust assembler/disassembler implement this document. Changes require bumping
the `.wvm` version.

---

## 1. Machine model

One WarpVM machine = one CUDA warp = 32 lanes executing lockstep.

Per-VM state:

| Item | Size | Notes |
|---|---|---|
| `vm_id` | u32 | stable logical ID, independent of SM/warp placement |
| `status` | u32 | see §7 |
| `pc` | u32 | word index into the VM's program |
| `exec_mask` | u32 | global lane mask, ANDed into every guard. v0.1: always `0xFFFFFFFF` (reserved for future use) |
| vector registers `r0–r15` | 16 × 32 × u32 | one u32 per lane |
| scalar registers `s0–s7` | 8 × u32 | uniform; logically one value, physically replicated on all lanes |
| predicate registers `p0–p3` | 4 × u32 | lane masks: bit *i* = lane *i*. Uniform (all lanes hold the same mask) |
| call stack | 8 × u32 | return addresses; fixed depth |
| `mem_base`, `mem_size_words` | — | private RAM region, word-addressed |
| mailbox in/out | — | see §8 |
| `rng_state` | u32 | xorshift32 seed for RAND |
| `instruction_counter` | u64 | retired instructions |
| `fault_code` | u32 | see §7 |

Value type for v0.1 is **u32 integer only**. All arithmetic is unsigned 32-bit
with wraparound. Floating point is deliberately out of scope for v0.1.

## 2. Instruction encoding

All instructions are **32 bits, fixed width**. Programs are arrays of u32
words; `pc` is a word index.

```text
 31           25 24        21 20      17 16      13 12              0
┌──────────────┬────────────┬──────────┬──────────┬──────────────────┐
│ opcode   (7) │ guard  (4) │ rd   (4) │ rs1  (4) │ lo          (13) │
└──────────────┴────────────┴──────────┴──────────┴──────────────────┘
```

Two forms, fixed per opcode:

- **R-form**: `lo` holds `rs2` in bits [3:0]; bits [12:4] must be zero.
- **I-form**: `lo` holds a signed 13-bit immediate (`−4096 … 4095`),
  sign-extended to u32.

Field meanings depend on the opcode's operand class:

- vector opcodes: `rd`/`rs1`/`rs2` address `r0–r15`;
- scalar opcodes (`S_*`): they address `s0–s7` (bits [2:0], bit 3 must be 0);
- comparison opcodes: `rd` addresses a predicate `p0–p3` (bits [1:0], bits [3:2] must be 0).

### 2.1 Guard field (4 bits)

| Value | Meaning |
|---|---|
| 0 | always (no guard) |
| 1–4 | `@p0` … `@p3` |
| 5–8 | `@!p0` … `@!p3` |

predicate index = `(guard − 1) & 3`, inverted = `guard ≥ 5`.

Effective per-lane activity for an instruction:

```text
active(lane) = ((pred_mask >> lane) & 1)        // guard ≠ 0
active(lane) = 1                                // guard = 0
```

Masked semantics: inactive lanes keep their old `rd` value, and inactive
lanes perform **no** memory access and **no** message side effects. Guards
never cause intra-warp divergence — implementations must use predicated
selection, not branches.

`exec_mask` is ANDed into the predicate before evaluation (no effect in
v0.1 where it is all-ones).

## 3. Register conventions

- `r15` is the assembler's default **scratch register** for materialising
  out-of-range constants (see §9). Programs may reclaim it with `.scratch rN`.
- Call stack depth is 8. `CALL` pushes, `RET` pops.

## 4. Opcode map

Unassigned opcodes fault (`FAULT_OPCODE`). Reserved ranges must not be
emitted by the assembler.

### 4.0 Misc

| Op | Name | Form | Semantics |
|---|---|---|---|
| 0x00 | `NOP` | R | do nothing |

### 4.1 Vector arithmetic / bitwise (lane-wise)

| Op | Name | Form | Semantics |
|---|---|---|---|
| 0x01 | `MOV`   | R | `rd = rs1` |
| 0x02 | `ADD`   | R | `rd = rs1 + rs2` |
| 0x03 | `SUB`   | R | `rd = rs1 − rs2` |
| 0x04 | `MUL`   | R | `rd = rs1 × rs2` (low 32 bits) |
| 0x05 | `DIV`   | R | `rd = rs1 / rs2`; result 0 where `rs2 == 0` |
| 0x06 | `MOD`   | R | `rd = rs1 % rs2`; result 0 where `rs2 == 0` |
| 0x07 | `MIN`   | R | `rd = min(rs1, rs2)` (unsigned) |
| 0x08 | `MAX`   | R | `rd = max(rs1, rs2)` (unsigned) |
| 0x09 | `AND`   | R | `rd = rs1 & rs2` |
| 0x0A | `OR`    | R | `rd = rs1 \| rs2` |
| 0x0B | `XOR`   | R | `rd = rs1 ^ rs2` |
| 0x0C | `SHL`   | R | `rd = rs1 << (rs2 & 31)` |
| 0x0D | `SHR`   | R | `rd = rs1 >> (rs2 & 31)` (logical) |
| 0x10 | `MOV_I` | I | `rd = imm` |
| 0x11 | `ADD_I` | I | `rd = rs1 + imm` |
| 0x12 | `SUB_I` | I | `rd = rs1 − imm` |
| 0x13 | `MUL_I` | I | `rd = rs1 × imm` |
| 0x14 | `AND_I` | I | `rd = rs1 & imm` |
| 0x15 | `OR_I`  | I | `rd = rs1 \| imm` |
| 0x16 | `XOR_I` | I | `rd = rs1 ^ imm` |
| 0x17 | `SHL_I` | I | `rd = rs1 << (imm & 31)` |
| 0x18 | `SHR_I` | I | `rd = rs1 >> (imm & 31)` |
| 0x20 | `ABS`   | R | `rd = abs(rs1)` (interpreted as i32) |
| 0x21 | `NEG`   | R | `rd = −rs1` (two's complement) |
| 0x22 | `NOT`   | R | `rd = ~rs1` |

Division by zero is **not** a fault (see §10, GP rationale).

### 4.2 Comparisons

Comparisons write a predicate mask, never a register value.

| Op | Name | Form | Semantics |
|---|---|---|---|
| 0x28 | `CMP_EQ` | R | `pD[i] = rs1[i] == rs2[i]` |
| 0x29 | `CMP_NE` | R | `pD[i] = rs1[i] != rs2[i]` |
| 0x2A | `CMP_LT` | R | `pD[i] = rs1[i] <  rs2[i]` (unsigned) |
| 0x2B | `CMP_LE` | R | `pD[i] = rs1[i] <= rs2[i]` (unsigned) |
| 0x2C | `CMP_GT` | R | `pD[i] = rs1[i] >  rs2[i]` (unsigned) |
| 0x2D | `CMP_GE` | R | `pD[i] = rs1[i] >= rs2[i]` (unsigned) |
| 0x30 | `CMP_EQ_I` | I | as above, `rs2` replaced by `imm` |
| 0x31 | `CMP_NE_I` | I | |
| 0x32 | `CMP_LT_I` | I | |
| 0x33 | `CMP_LE_I` | I | |
| 0x34 | `CMP_GT_I` | I | |
| 0x35 | `CMP_GE_I` | I | |

Comparisons ignore their own guard for the *computation* (all lanes are
compared) but the write of `pD` happens only if the guard is satisfied by
the current predicate state — like every other write. In practice
comparisons are written unguarded.

### 4.3 Predicate-mask operations

Operands in the `rd`/`rs1`/`rs2` positions address `p0–p3`.

| Op | Name | Form | Semantics |
|---|---|---|---|
| 0x38 | `NOTMASK` | R | `pD = ~pS1` |
| 0x39 | `ANDMASK` | R | `pD = pS1 & pS2` |
| 0x3A | `ORMASK`  | R | `pD = pS1 \| pS2` |
| 0x3B | `BALLOT`  | R | `pD[i] = rs1[i] != 0` |
| 0x3C | `ANY`     | R | `pD = (pS1 != 0) ? 0xFFFFFFFF : 0` |
| 0x3D | `ALL`     | R | `pD = (pS1 == 0xFFFFFFFF) ? 0xFFFFFFFF : 0` |

### 4.4 Lane / warp operations

| Op | Name | Form | Semantics |
|---|---|---|---|
| 0x40 | `LANEID`      | R | `rd[i] = i` (`rs1` ignored) |
| 0x41 | `BROADCAST`   | I | `rd[i] = rs1[imm & 31]` |
| 0x42 | `SHUFFLE`     | R | `rd[i] = rs1[rs2[i] & 31]` |
| 0x43 | `SHUFFLE_XOR` | I | `rd[i] = rs1[i ^ (imm & 31)]` |
| 0x44 | `REDUCE_ADD`  | R | all lanes: `rd = Σ rs1[i]` |
| 0x45 | `REDUCE_MIN`  | R | all lanes: `rd = min rs1[i]` |
| 0x46 | `REDUCE_MAX`  | R | all lanes: `rd = max rs1[i]` |
| 0x47 | `REDUCE_AND`  | R | all lanes: `rd = & rs1[i]` |
| 0x48 | `REDUCE_OR`   | R | all lanes: `rd = \| rs1[i]` |
| 0x49 | `REDUCE_XOR`  | R | all lanes: `rd = ^ rs1[i]` |
| 0x4A | `VMID`        | R | all lanes: `rd = vm_id` (stable logical address, not resident slot) |
| 0x4B | `CLOCK`       | R | all lanes: `rd =` coarse tick counter (SM-local cycles, approximate) |
| 0x4C | `RAND`        | R | `rd[i] =` per-lane xorshift-derived value; advances VM rng state |

Reductions ignore guards for the computation (all 32 lanes contribute);
the guarded write rule applies to `rd` as usual.

### 4.5 Memory

Addresses are **word indices**. `LOAD`/`STORE` decode an address as:

```text
if addr < ram_size_words:                 access private RAM
else if VIDEO_BASE <= addr < VIDEO_END:   access this VM's framebuffer
else:                                     FAULT_MEM
```

Per-lane, potentially scattered. Private-RAM addresses are `0 …
mem_size_words − 1`.

The standard VM configuration uses `RAM_SIZE_WORDS = 65536` private 32-bit
words (256 KiB) per VM. Runtimes may still construct deliberately smaller
memories for bounds and fault tests.

| Op | Name | Form | Semantics |
|---|---|---|---|
| 0x50 | `LOAD`  | R | active lanes: `rd[i] = mem[rs1[i]]` |
| 0x51 | `STORE` | R | active lanes: `mem[rd[i]] = rs1[i]` |

Out-of-range access by any active lane faults the whole VM (`FAULT_MEM`).

#### Architectural display (v0.1.1)

Every VM owns one fixed framebuffer, memory-mapped into its logical address
space and accessed with the ordinary `LOAD`/`STORE` above (no special pixel
opcodes). One pixel is one ordinary 32-bit word `0xAARRGGBB`.

```text
VIDEO_BASE = 0x00100000   (word address)
VIDEO_WIDTH = VIDEO_HEIGHT = 128
VIDEO_WORDS = 16384
VIDEO_END  = VIDEO_BASE + VIDEO_WORDS
```

Pixel `(x, y)` is at `VIDEO_BASE + y*128 + x`. Because each lane may supply a
different address, a single predicated `STORE` writes up to 32 pixels in
parallel. On reset the framebuffer is cleared to `0xFF000000` (opaque black);
pause/resume preserves it. Presentation is a host concern (see `FLIP`).

### 4.6 Logging / runtime services

| Op | Name | Form | Semantics |
|---|---|---|---|
| 0x58 | `LOG`   | R | append `{vm_id, tag=rs2[0], value=rs1[0]}` to the VM log ring |
| 0x59 | `LOG_I` | I | append `{vm_id, tag=imm, value=rs1[0]}` to the VM log ring |
| 0x5A | `FLIP`  | R | publish the framebuffer (v0.1.1) |

The log ring lives in host-visible memory; writes are fire-and-forget and
may drop entries when full.

`FLIP` (v0.1.1) publishes the current framebuffer: all stores retired before
it belong to the published frame, lane 0 increments the VM's `frame_seq`
exactly once, and execution continues without blocking on the host renderer.
`FLIP` is **unguardable** — a non-default guard faults (`FAULT_OPERAND`) — so a
publication is always a single warp-uniform event. The host watches
`frame_seq` to detect newer frames. There is no double buffering in v0.1.1.

### 4.7 Messaging

Message record (fixed size, 16 bytes). `payload[1..2]` are reserved (zero) in
v0.1; `SEND` transmits `payload[0]` only.

```text
u32 header      // src_vm in low 16 bits, msg_type in high 16 bits
u32 payload[3]
```

Each VM has an inbound mailbox: a ring of 16 message slots. Multiple producers
reserve positions through `head`; the owner consumes through `tail`. A
per-slot publication sequence makes reservation, complete-message visibility,
and slot reuse distinct ordered events.

| Op | Name | Form | Semantics |
|---|---|---|---|
| 0x60 | `SEND`     | R | lane 0 posts `{src=vm_id, type=rs1[0], payload=rs2[0]}` to VM `rd[0]` |
| 0x61 | `TRY_RECV` | R | lane 0 consumes one pending message |

`SEND rd, rType, rPayload`: destination is `rd[0]`. Faults (`FAULT_MSG`) if
the destination VM does not exist or its mailbox is full.

VM addresses are stable logical IDs. They are independent of the destination's
resident GPU slot and are not reused within one supervisor lifetime. The
runtime resolves a destination ID to its current mailbox slot; consequently a
message to a retired ID cannot be delivered to a later VM that reuses the same
resident storage. The low 16-bit `src_vm` field currently bounds the logical
address namespace to 65,536 IDs per supervisor epoch.

`TRY_RECV pGot, rPayload, rMeta`: non-blocking. If a message is pending,
`pGot` = all lanes, `rPayload` = `payload[0]`, `rMeta` = header
(`msg_type<<16 | src_vm`); otherwise `pGot` = 0 and the registers are
unchanged. Poll with `NOTMASK` + `JMP_IF_ANY` to wait for a message.

`RECV` (blocking) and `BROADCAST_MSG` remain reserved (`0x62–0x6F`) for a
later slice.

### 4.8 Control flow

Jump/call targets are absolute word indices (I-form immediate).

| Op | Name | Form | Semantics |
|---|---|---|---|
| 0x70 | `JMP`         | I | `pc = imm` |
| 0x71 | `JMP_IF_ANY`  | I | if guard predicate non-zero: `pc = imm` |
| 0x72 | `JMP_IF_ALL`  | I | if guard predicate all-ones: `pc = imm` |
| 0x73 | `CALL`        | I | push `pc+1`; `pc = imm` |
| 0x74 | `RET`         | R | pop into `pc` |
| 0x75 | `HALT`        | R | VM stops executing; warp spins on host control flags (reload/reset/exit possible) |
| 0x76 | `YIELD`       | R | cooperation point: warp checks host control flags (pause/snapshot/step requests) |
| 0x77 | `STEP_TRAP`   | R | reserved for debugger single-step; behaves as NOP unless debug-active |

`JMP_IF_ANY`/`JMP_IF_ALL` use the **guard field** as their condition
predicate; their own guard should be 0 (assembler enforces).

After any non-jump instruction, `pc` advances by 1. Execution that reaches
`pc == code_len` without a `HALT` faults (`FAULT_JUMP`).

### 4.9 Scalar operations

| Op | Name | Form | Semantics |
|---|---|---|---|
| 0x78 | `S_MOV`      | R | `sD = sS1` |
| 0x79 | `S_MOV_I`    | I | `sD = imm` |
| 0x7A | `S_ADD`      | R | `sD = sS1 + sS2` |
| 0x7B | `S_ADD_I`    | I | `sD = sS1 + imm` |
| 0x7C | `S_LDW`      | I | `sD = literals[imm]` |
| 0x7D | `S_CMP_LT`   | R | `pD = (sS1 < sS2) ? 0xFFFFFFFF : 0` |
| 0x7E | `S_CMP_LT_I` | I | `pD = (sS1 < imm) ? 0xFFFFFFFF : 0` |

`rd`/`rs1` fields address `s0–s7` (except comparisons, whose `rd`
addresses `p0–p3`).

### 4.10 Literal pool load

| Op | Name | Form | Semantics |
|---|---|---|---|
| 0x19 | `LDW` | I | all lanes: `rd = literals[imm]` (`imm` unsigned, `0 … literals_len−1`) |

Used to materialise constants that do not fit the 13-bit immediate.

### 4.11 Cross-class move

| Op | Name | Form | Semantics |
|---|---|---|---|
| 0x1A | `S_BCAST` | R | vector op: `rd[i] = sS1[0]` (`rs1` addresses a scalar reg) |
| 0x1B | `S_GET`   | R | scalar op: `sD = rs1[0]` (`rs1` addresses a vector reg, lane 0) |

## 5. Faults

A fault stops the VM's program. The warp sets `status = FAULTED`, records
`fault_code`, spills state to VM memory and remains controllable by the
host. Faults must never corrupt other VMs or the runtime.

| Code | Name | Cause |
|---|---|---|
| 0 | `OK` | no fault |
| 1 | `FAULT_OPCODE` | unassigned opcode |
| 2 | `FAULT_OPERAND` | register/predicate field out of range, or reserved bits set |
| 3 | `FAULT_JUMP` | jump/call/ret target outside `[0, code_len)` |
| 4 | `FAULT_MEM` | active-lane load/store outside VM RAM |
| 5 | `FAULT_STACK` | call stack overflow or underflow |
| 6 | `FAULT_MSG` | illegal message operation (bad destination, mailbox error) |
| 7 | `FAULT_BUDGET` | execution-budget limit reached (when configured) |

## 6. Status values

| Value | Name | Meaning |
|---|---|---|
| 0 | `IDLE` | no program loaded |
| 1 | `RUNNING` | executing normally |
| 2 | `PAUSED` | host-paused at a control point |
| 3 | `HALTED` | executed `HALT` |
| 4 | `FAULTED` | stopped by a fault |
| 5 | `DEBUG` | held at a debug barrier (attach/step) |

## 7. Program and literal layout

A loaded program consists of:

```text
code:     u32[code_len]       // instructions, entry at word 0
literals: u32[literals_len]   // constant pool, referenced by LDW / S_LDW
```

`code_len` max for v0.1: 4096 words (16 KB). `literals_len` max: 256.

## 8. Mailboxes

Each VM has an inbound ring of 16 logical message slots with u32 `head`
(multi-producer reservation), `tail` (single owner), and one publication
sequence per physical slot. A producer writes the message before publishing
the sequence with release ordering; the receiver observes publication with
acquire ordering before reading and releasing the slot. Outbound traffic goes
directly to the destination's inbound ring via `SEND`. Full mailbox: `SEND`
faults (`FAULT_MSG`) in v0.1 (no blocking semantics yet).

## 9. Assembly format (`.wva`)

- Line comments start with `;`.
- Labels: `name:` at the start of a line (alphanumerics + `_`, not starting
  with a digit).
- Directives:
  - `.const NAME VALUE` — appends VALUE to the literal pool and binds NAME.
  - `.scratch rN` — choose the assembler scratch register (default `r15`).
- Registers: `r0–r15`, `s0–s7`, `p0–p3`.
- Immediates: decimal, `0x…` hex, `0b…` binary, or a `.const` name.
- Guard prefix: `@p0` … `@p3`, `@!p0` … `@!p3`.
- Operand forms: `OP rd, rs1, rs2` / `OP rd, rs1, IMM` / memory uses plain
  registers: `LOAD r3, r2` / `STORE r2, r3`.
- The assembler chooses R-form vs I-form by operand kind: `ADD r0, r1, r2`
  and `ADD r0, r1, #5` are both legal; explicit `_I` mnemonics are accepted
  too. Immediates may be written bare or `#`-prefixed.

Assembler behaviour for constants in immediate position:

- fits signed 13 bits → emitted inline (I-form);
- otherwise → materialised into the scratch register via `LDW` from the
  literal pool, then the R-form is emitted. `.const` names automatically
  enter the literal pool.

Example (squares of `0 … N−1` in VM RAM):

```text
.const N 1000

    S_MOV_I  s0, 0              ; base

loop:
    LANEID   r0
    S_BCAST  r1, s0
    ADD      r2, r0, r1         ; indices base..base+31
    CMP_LT   p0, r2, #N

    @p0 LOAD  r3, r2
    @p0 MUL   r3, r3, r3
    @p0 STORE r2, r3

    S_ADD_I  s0, s0, 32
    S_CMP_LT_I p1, s0, #N
    JMP_IF_ANY p1, loop

    HALT
```

`JMP_IF_ANY p1, loop` encodes `p1` in the guard field and `loop` as the
13-bit immediate target.

## 10. `.wvm` binary format

Little-endian, all offsets u32-aligned.

```text
offset  size  field
0       4     magic    = 0x304D5657  ('W','V','M','0' as bytes)
4       4     version  = 1
8       4     flags    = 0 (reserved)
12      4     code_len      (words)
16      4     literals_len  (words)
20      4     entry    = 0 (v0.1)
24      8     reserved (zero)
32      …     code      u32[code_len]
…       …     literals  u32[literals_len]
```

## 11. Design notes / rationale

- **u32 word addressing** keeps LOAD/STORE alignment trivial and bounds
  checks to a single compare.
- **DIV/MOD by zero yields 0 instead of faulting** so mutated/GP-generated
  programs degrade instead of dying en masse; faults stay reserved for
  structural violations that threaten the runtime.
- **Predicates are ballot masks**, uniform across the warp. This keeps
  predicate state inspectable (four u32s) and makes `JMP_IF_ANY/ALL`
  single compares.
- **Fixed 32-bit encoding** keeps fetch/decode branch-free and makes the
  machine trivially GP-mutable (every program is a u32 array).
