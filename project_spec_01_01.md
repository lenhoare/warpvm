# WarpVM v0.1.1 Project Spec — Graphics Slice

## 1. Purpose

WarpVM v0.1.1 adds a minimal architectural display to every WarpVM virtual machine.

The goal is not to build a game console, a Pico-8 clone, a general GPU API, or a windowing system. The goal is to give every persistent WarpVM computer a small, directly programmable visual output device that fits naturally into the existing 32-bit machine model.

A WarpVM machine becomes:

```text
WarpVM
├── 32 execution lanes
├── vector/scalar/predicate registers
├── private RAM
├── mailbox
├── control/status
└── 128 × 128 × 32-bit framebuffer
```

The framebuffer is part of the architectural machine. Windows, scaling, presentation, and host UI are runtime concerns.

---

## 2. Fixed Architectural Display

Every VM has exactly one display with these v0.1.1 properties:

```text
width:        128 pixels
height:       128 pixels
pixel format: 32-bit ARGB8888
storage:      one 32-bit word per pixel
size:         16,384 words = 65,536 bytes per VM
```

For 64 VMs the total framebuffer allocation is 4 MiB.

The resolution is deliberately fixed for the architecture rather than configurable per VM.

### 2.1 Pixel Encoding

A framebuffer word uses:

```text
0xAARRGGBB
```

where each channel is eight bits.

Examples:

```text
0xFFFF0000  opaque red
0xFF00FF00  opaque green
0xFF0000FF  opaque blue
0xFFFFFFFF  opaque white
0xFF000000  opaque black
```

For v0.1.1 the host renderer MAY ignore alpha and treat all pixels as opaque, but the framebuffer representation remains ARGB8888 so the ABI does not need to change if alpha becomes useful later.

No palette is imposed by the machine.

---

## 3. Design Principles

### 3.1 Stay 32-bit

WarpVM v0.1 is a 32-bit integer machine. Graphics must not introduce byte, halfword, packed-pixel, or special colour types merely to save a negligible amount of memory.

One pixel is one ordinary 32-bit value.

### 3.2 Graphics memory is memory

The VM should not require a special `PIXEL`, `DRAW`, or `STORE_PIXEL` opcode.

The framebuffer is memory-mapped into the VM's logical address space and is accessed through the existing `LOAD` and `STORE` instructions.

### 3.3 Warp-native drawing should emerge naturally

Because addresses may differ by lane, a single predicated `STORE` can write up to 32 pixels in parallel.

A natural pattern is:

```text
LANEID r0
... calculate 32 pixel addresses ...
... calculate 32 colours ...
STORE [rAddr], rColour
```

Thirty-two lanes cover one quarter of a 128-pixel scanline per instruction-level batch.

### 3.4 Presentation is not part of the VM ISA

The VM owns pixels, not windows.

The host runtime owns:

- display windows,
- enlargement/scaling,
- VM selection,
- grid/compositor views,
- refresh scheduling,
- conversion to the host graphics API.

---

## 4. Logical Memory Map

### 4.1 Addressing Unit

Preserve the existing WarpVM memory convention: addresses supplied to `LOAD` and `STORE` are logical 32-bit word addresses.

Do not introduce byte addressing in this slice.

### 4.2 Framebuffer Region

Reserve a fixed architectural word-address region for video memory.

Recommended initial layout:

```text
0x00000000 ... RAM_SIZE_WORDS-1   VM private RAM

0x00100000 ... 0x00103FFF         framebuffer
```

Constants:

```text
RAM_SIZE_WORDS = 65536
VIDEO_BASE_WORD = 0x00100000
VIDEO_WIDTH     = 128
VIDEO_HEIGHT    = 128
VIDEO_WORDS     = 16384
VIDEO_END_WORD  = VIDEO_BASE_WORD + VIDEO_WORDS
```

Pixel `(x, y)` is located at:

```text
VIDEO_BASE_WORD + y * 128 + x
```

The deliberate gap between RAM and video memory leaves room for future architectural MMIO regions without coupling framebuffer location to configured RAM size.

If the existing assembler's immediate encoding cannot directly materialise `VIDEO_BASE_WORD`, normal literal-pool materialisation should be used. No ISA change is required.

### 4.3 Address Decoder

`LOAD` and `STORE` should resolve an address as follows:

```text
if addr < ram_size_words:
    access private RAM
else if VIDEO_BASE_WORD <= addr < VIDEO_END_WORD:
    access this VM's framebuffer
else:
    FAULT_MEM
```

All existing per-lane bounds checking and warp-uniform fault handling remain in force.

A bad framebuffer address must fault exactly like any other invalid VM memory access.

One VM must never be able to address another VM's framebuffer through this mapping.

---

## 5. Physical Runtime Storage

Each VM receives a private framebuffer allocation of 16,384 `uint32_t` words.

The physical layout may be either:

```text
framebuffer[vm_count][16384]
```

or an equivalent flat allocation:

```text
framebuffers + vm_slot * 16384
```

The logical ABI must not expose the physical layout.

`VmDesc` should gain whatever framebuffer pointer/base information the interpreter needs, keyed by logical VM slot in the same way as program, literal and RAM state.

### 5.1 Reset Behaviour

On VM reset, its framebuffer must be cleared to:

```text
0xFF000000
```

opaque black.

Resetting one VM must not alter any other VM's framebuffer.

Pausing and resuming a VM must preserve its framebuffer contents.

---

## 6. Frame Publication

Direct framebuffer writes and frame presentation are separate concepts.

A VM may modify video memory at any time. The host should only consider a frame explicitly published when the VM requests presentation.

### 6.1 `FLIP` Instruction

Add one new v0.1.1 instruction:

```text
FLIP
```

Semantics:

1. all framebuffer stores retired before `FLIP` are considered part of the frame being published;
2. lane 0 increments the VM's `frame_seq` counter exactly once;
3. the instruction retires normally;
4. execution continues immediately;
5. `FLIP` does not block waiting for the host renderer.

`FLIP` has no operands and is warp-uniform.

The host uses `frame_seq` to determine whether a VM has published a newer frame.

### 6.2 No Double Buffering Yet

v0.1.1 uses one framebuffer per VM.

A host copy may therefore observe subsequent writes made after `FLIP` if the VM immediately begins drawing the next frame. This is acceptable for the first graphics slice.

Do not add architectural double buffering, page flipping, fences, vsync, or host acknowledgements yet.

If tearing becomes materially visible in experiments, double buffering can be evaluated in a later slice.

### 6.3 Control/Status Plane

Expose at least:

```text
frame_seq
```

per VM in host-visible status/control state.

A useful optional host-side field is:

```text
last_presented_frame_seq
```

but that is renderer state rather than VM architectural state.

---

## 7. ISA / Assembler / Disassembler Changes

### 7.1 New Opcode

Allocate one currently free opcode for:

```text
FLIP
```

Update:

- `docs/isa.md`,
- CUDA interpreter decode/execute,
- Rust assembler,
- Rust disassembler,
- C++ disassembler used by attach,
- round-trip assembler/disassembler tests.

The exact opcode number should follow the existing opcode allocation scheme rather than being chosen ad hoc by this spec.

### 7.2 Assembly Constants

The assembler should make the architectural display constants conveniently available.

Preferred approach: predefined symbols rather than new directives.

At minimum:

```text
VIDEO_BASE
VIDEO_WIDTH
VIDEO_HEIGHT
VIDEO_WORDS
```

If automatic predefined symbols complicate the assembler unnecessarily, provide a small include/example convention instead for v0.1.1. Do not expand the assembler language substantially for this feature.

---

## 8. Host Graphics Viewer

Add a simple viewer capable of displaying WarpVM framebuffers while the persistent kernel remains resident.

The viewer is a development/runtime tool, not part of the bytecode format.

### 8.1 Required Modes

The first implementation should support:

1. **single-VM view** — display one chosen VM enlarged with nearest-neighbour scaling;
2. **multi-VM grid** — show a useful set of VM displays simultaneously.

A 128×128 framebuffer should always be scaled with nearest-neighbour sampling by default so individual VM pixels remain crisp.

### 8.2 Suggested CLI

Exact syntax may adapt to the existing CLI, but a reasonable interface is:

```text
warpvm view <program.wvm>
warpvm view <program.wvm> --vms 64
warpvm view <program.wvm> --vm 37
```

Alternatively, if integration with `serve` is cleaner:

```text
warpvm serve <program.wvm> --view
```

Do not introduce a daemon or cross-process attachment architecture merely for this slice.

The viewer may initially live in-process with the persistent runtime, matching the current `serve`/`attach` architecture.

### 8.3 Host Update Loop

The host should:

1. watch per-VM `frame_seq` values;
2. copy only framebuffers whose sequence has advanced where practical;
3. upload/copy those frames to the host rendering surface;
4. remain bounded and responsive on a display GPU;
5. not pause the VMs to display their output.

Correctness matters more than optimising copy bandwidth in v0.1.1.

### 8.4 Host Graphics Library

Use a small conventional host library if useful rather than implementing window creation or platform graphics manually.

SDL3 is a natural candidate if it integrates cleanly with the existing CMake/C++ runtime, but the architectural specification does not depend on SDL.

The renderer must not leak host-library concepts into the WarpVM ISA.

---

## 9. Attach / Inspection

Graphics should integrate with the existing inspection philosophy.

Recommended addition to the attach console:

```text
frame
```

which reports at least:

```text
resolution=128x128
format=ARGB8888
frame_seq=<n>
```

Optionally add:

```text
pixel <x> <y>
```

for textual inspection of one framebuffer word.

A GUI viewer does not need to be embedded inside the attach console for v0.1.1.

---

## 10. Demonstration Program

Add:

```text
programs/graphics.wva
```

It should prove that graphics is genuinely warp-native rather than a lane-0 side channel.

Recommended demonstration:

1. use `LANEID` to generate 32 pixel positions concurrently;
2. fill the entire 128×128 framebuffer using 32-lane stores;
3. calculate colours algorithmically from `x`, `y`, `LANEID`, and/or `VMID`;
4. execute `FLIP`;
5. continue changing the image and publishing frames while resident.

For multiple VMs, each screen should be visibly distinct using `VMID` so the grid view immediately demonstrates independent framebuffers.

A simple animated colour/pattern field is preferable to adding fonts or sprite assets.

---

## 11. Required Tests

All existing v0.1 tests must remain green.

Add at least the following graphics tests.

### 11.1 Framebuffer Isolation

Run multiple VMs.

Each VM writes a VM-specific pattern to the same logical framebuffer addresses.

Verify every VM's physical framebuffer contains only its own expected values.

### 11.2 32-Lane Pixel Store

One instruction sequence should cause all 32 active lanes to write different framebuffer addresses.

Verify all 32 pixels and ensure neighbouring pixels remain untouched.

### 11.3 Predicated Pixel Store

Use a predicate mask so only selected lanes write pixels.

Verify inactive lanes do not modify their target pixels.

### 11.4 Bounds Fault

Attempt a framebuffer access at:

```text
VIDEO_END_WORD
```

Verify:

```text
FAULT_MEM
```

and ensure neighbouring VMs are unaffected.

### 11.5 RAM Compatibility

Verify ordinary RAM addresses still behave exactly as in v0.1 after the address decoder is extended.

### 11.6 Reset

Write non-black pixels, reset the VM, and verify its entire framebuffer returns to opaque black without affecting another VM.

### 11.7 Pause / Resume

Write/publish a frame, pause, inspect/copy it, resume, and verify framebuffer state survives the pause.

### 11.8 `FLIP` Sequence

Execute multiple `FLIP` instructions and verify `frame_seq` advances exactly once per retired `FLIP`.

Predicated execution, if the generic guard encoding allows `FLIP` to be guarded, must still produce a single warp-uniform publication event according to the ISA's validation rules. If that semantic is awkward, declare `FLIP` unguardable and reject non-default guards with `FAULT_OPERAND`.

Preferred v0.1.1 choice: **make `FLIP` unguardable**.

### 11.9 Toolchain Round Trip

Assemble and disassemble a program containing `FLIP`; reassembly must reproduce identical bytecode.

### 11.10 End-to-End Viewer Smoke Test

Run the graphics demo through the real assembler → `.wvm` → persistent GPU runtime path and verify at least one published frame can be copied to the host with expected corner/sample pixel values.

The automated test does not need to validate pixels visually on screen.

---

## 12. Deliberate Non-Goals for v0.1.1

Do not add:

- 8-bit or 16-bit VM numeric types,
- indexed/palette colour as an architectural restriction,
- sprites,
- tiles,
- text rendering,
- fonts,
- lines/rectangles/circles as ISA instructions,
- triangle rasterisation,
- 3D graphics,
- textures,
- shaders,
- host GPU command submission from a VM,
- configurable display resolution,
- multiple displays per VM,
- double buffering,
- vsync,
- keyboard/mouse/gamepad input,
- audio,
- cross-process graphics attach,
- networking changes.

These may later exist as libraries, runtime features, or separate architectural slices.

In particular, drawing primitives should preferably become ordinary WarpVM library code rather than ISA instructions.

---

## 13. Likely Implementation Files

This is me thinking out loud, please adapt as best as you see fit:

```text
docs/isa.md
docs/architecture.md
runtime/src/gpu/interpreter.cuh
runtime/src/gpu/vm_state.cuh      # or current VmDesc/VmState location
runtime/src/host/...              # framebuffer copy/viewer
runtime/src/.../control...        # frame_seq publication
runtime/src/.../disasm...         # C++ attach disassembler
tools/warpvm-asm/...              # FLIP + optional display symbols
programs/graphics.wva
tests/...
CMakeLists.txt                    # host viewer dependency if required
```

---

## 14. Suggested Development Slices

### Graphics Slice A — Framebuffer Hardware

- add per-VM framebuffer storage;
- extend memory address decode;
- clear framebuffer on reset;
- test isolation, bounds, predicated writes and RAM compatibility.

Checkpoint:

```text
framebuffer_memory   PASS
framebuffer_isolation PASS
```

### Graphics Slice B — Publication

- add `FLIP` opcode;
- add `frame_seq`;
- assembler/disassembler support;
- test exact sequence increments and round-trip stability.

Checkpoint:

```text
flip_sequence        PASS
graphics_roundtrip   PASS
```

### Graphics Slice C — Host Viewer

- copy published framebuffer to host;
- single-VM nearest-neighbour viewer;
- multi-VM grid;
- ensure resident kernel remains responsive.

Checkpoint:

```text
viewer_smoke         PASS
```

### Graphics Slice D — Capstone

- create `graphics.wva`;
- run at least 64 resident VMs;
- every VM generates a distinct 128×128 image using 32-lane pixel stores;
- every VM publishes frames while remaining independently running and inspectable;
- display all 64 in the host grid;
- attach/pause one VM without disturbing the others.

---

## 15. v0.1.1 Success Criteria

v0.1.1 is complete when:

1. every VM owns an isolated 128×128×32-bit framebuffer;
2. framebuffer pixels are ordinary 32-bit VM values;
3. normal `LOAD`/`STORE` can access video memory through a fixed memory-mapped region;
4. 32 lanes can write 32 distinct pixels concurrently;
5. invalid video addresses fault safely;
6. `FLIP` publishes a frame non-blockingly through a monotonically increasing sequence number;
7. resetting a VM clears only its own framebuffer;
8. the host can show one VM enlarged and many VMs in a grid;
9. the persistent kernel and existing pause/step/message behaviour remain intact;
10. an end-to-end assembly program visibly animates many independent VM displays.

The capstone should make the machine visibly understandable:

> dozens of persistent 32-lane computers executing simultaneously on the GPU, each with its own independent 128×128 display.

---

## 16. Guiding Rule

Keep the graphics architecture as primitive as the rest of WarpVM.

The machine provides **pixels and parallel computation**.

Higher-level graphics should be written *on* WarpVM, not baked *into* WarpVM.
