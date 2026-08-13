// Device-side fetch/decode/execute interpreter.
//
// All 32 lanes of the warp run this together. The opcode stream is uniform
// (every lane holds the same pc), so fetch/decode never diverges. Vector
// register slices live per-thread: lane i owns element i of every register.
// Scalar registers and predicate masks are uniform values replicated in
// every lane; they stay in sync by construction.
//
// Guarded instructions never branch per lane — inactive lanes keep their
// old destination value and perform no memory/message side effects.
// Warp-collective operations (ballot/shuffle/reduce) are computed on all
// lanes regardless of the guard; only the write is masked.
#pragma once

#include "control.cuh"
#include "warpvm.cuh"

namespace wvm {

struct VmCtx {
  const uint32_t* code;
  uint32_t code_len;
  const uint32_t* literals;
  uint32_t literals_len;
  uint32_t* mem;
  uint32_t mem_size_words;
  uint32_t* fb;   // framebuffer base (kVideoWords words), nullptr if absent
  VmId vm_id;       // stable architectural identity returned by VMID
  VmSlot slot_id;   // resident array index; never visible to the program
  uint32_t lane;

  Mailbox* mailboxes;        // resident-slot-indexed inbound mailboxes
  const VmSlot* vm_routes;   // logical VM address -> resident slot
  uint32_t num_slots;        // resident slot count for route validation

  uint32_t pc;
  uint32_t vregs[kVectorRegs];  // this lane's slice of each vector register
  uint32_t sregs[kScalarRegs];  // uniform, replicated
  uint32_t preds[kPredRegs];    // uniform lane masks, replicated
  uint32_t call_stack[kCallDepth];
  uint32_t call_depth;
  uint32_t rng_state;
  uint64_t instr_count;
  uint64_t last_pub;  // instr_count at last control-plane progress publish
  bool step;          // single-step: retire one instruction then re-pause
  uint32_t fault;
};

// Compile-time experiment: keep the architectural vector registers as named
// scalar locals instead of a dynamically indexed array. Dynamic reads and
// writes become select/switch logic, giving ptxas the opportunity to allocate
// the sixteen live values to physical registers. The normal specialization
// continues to access VmCtx::vregs directly.
template <bool kScalarized>
struct VectorRegisterFile;

template <>
struct VectorRegisterFile<false> {
  uint32_t* values;
  uint32_t stride;

  __device__ __forceinline__ explicit VectorRegisterFile(uint32_t* source,
                                                         uint32_t step = 1)
      : values(source), stride(step) {}
  __device__ __forceinline__ uint32_t Get(uint32_t index) const {
    return values[index * stride];
  }
  __device__ __forceinline__ void Set(uint32_t index, uint32_t value) {
    values[index * stride] = value;
  }
  __device__ __forceinline__ void Spill(uint32_t*) const {}
};

template <>
struct VectorRegisterFile<true> {
  uint32_t r0, r1, r2, r3, r4, r5, r6, r7;
  uint32_t r8, r9, r10, r11, r12, r13, r14, r15;

  __device__ __forceinline__ explicit VectorRegisterFile(
      const uint32_t* source, uint32_t = 1)
      : r0(source[0]), r1(source[1]), r2(source[2]), r3(source[3]),
        r4(source[4]), r5(source[5]), r6(source[6]), r7(source[7]),
        r8(source[8]), r9(source[9]), r10(source[10]), r11(source[11]),
        r12(source[12]), r13(source[13]), r14(source[14]), r15(source[15]) {}

  __device__ __forceinline__ uint32_t Get(uint32_t index) const {
    switch (index) {
      case 0: return r0;
      case 1: return r1;
      case 2: return r2;
      case 3: return r3;
      case 4: return r4;
      case 5: return r5;
      case 6: return r6;
      case 7: return r7;
      case 8: return r8;
      case 9: return r9;
      case 10: return r10;
      case 11: return r11;
      case 12: return r12;
      case 13: return r13;
      case 14: return r14;
      default: return r15;
    }
  }

  __device__ __forceinline__ void Set(uint32_t index, uint32_t value) {
    switch (index) {
      case 0: r0 = value; break;
      case 1: r1 = value; break;
      case 2: r2 = value; break;
      case 3: r3 = value; break;
      case 4: r4 = value; break;
      case 5: r5 = value; break;
      case 6: r6 = value; break;
      case 7: r7 = value; break;
      case 8: r8 = value; break;
      case 9: r9 = value; break;
      case 10: r10 = value; break;
      case 11: r11 = value; break;
      case 12: r12 = value; break;
      case 13: r13 = value; break;
      case 14: r14 = value; break;
      default: r15 = value; break;
    }
  }

  __device__ __forceinline__ void Spill(uint32_t* destination) const {
    destination[0] = r0;
    destination[1] = r1;
    destination[2] = r2;
    destination[3] = r3;
    destination[4] = r4;
    destination[5] = r5;
    destination[6] = r6;
    destination[7] = r7;
    destination[8] = r8;
    destination[9] = r9;
    destination[10] = r10;
    destination[11] = r11;
    destination[12] = r12;
    destination[13] = r13;
    destination[14] = r14;
    destination[15] = r15;
  }
};

__device__ inline void VmFault(VmCtx& ctx, Fault f) { ctx.fault = f; }

__device__ inline bool GuardActive(const VmCtx& ctx, uint32_t guard) {
  if (guard == 0) return true;
  const uint32_t idx = (guard - 1) & 3u;
  const bool inv = guard >= 5u;
  const bool bit = ((ctx.preds[idx] >> ctx.lane) & 1u) != 0u;
  return inv ? !bit : bit;
}

// Warp-uniform mask selected by a guard field (used by JMP_IF_*).
__device__ inline uint32_t GuardMask(const VmCtx& ctx, uint32_t guard) {
  if (guard == 0 || guard > 8u) return 0u;
  const uint32_t m = ctx.preds[(guard - 1) & 3u];
  return guard >= 5u ? ~m : m;
}

__device__ inline int32_t SignExt13(uint32_t lo) {
  return (lo & 0x1000u) != 0u ? static_cast<int32_t>(lo | 0xFFFFE000u)
                               : static_cast<int32_t>(lo);
}

__device__ inline uint32_t Xorshift32(uint32_t s) {
  s ^= s << 13;
  s ^= s >> 17;
  s ^= s << 5;
  return s;
}

__device__ inline uint32_t WarpReduceAdd(uint32_t v) {
#pragma unroll
  for (int off = kLanes / 2; off > 0; off >>= 1)
    v += __shfl_xor_sync(kFullMask, v, off);
  return v;
}
__device__ inline uint32_t WarpReduceMin(uint32_t v) {
#pragma unroll
  for (int off = kLanes / 2; off > 0; off >>= 1) {
    const uint32_t o = __shfl_xor_sync(kFullMask, v, off);
    v = o < v ? o : v;
  }
  return v;
}
__device__ inline uint32_t WarpReduceMax(uint32_t v) {
#pragma unroll
  for (int off = kLanes / 2; off > 0; off >>= 1) {
    const uint32_t o = __shfl_xor_sync(kFullMask, v, off);
    v = o > v ? o : v;
  }
  return v;
}
__device__ inline uint32_t WarpReduceAnd(uint32_t v) {
#pragma unroll
  for (int off = kLanes / 2; off > 0; off >>= 1)
    v &= __shfl_xor_sync(kFullMask, v, off);
  return v;
}
__device__ inline uint32_t WarpReduceOr(uint32_t v) {
#pragma unroll
  for (int off = kLanes / 2; off > 0; off >>= 1)
    v |= __shfl_xor_sync(kFullMask, v, off);
  return v;
}
__device__ inline uint32_t WarpReduceXor(uint32_t v) {
#pragma unroll
  for (int off = kLanes / 2; off > 0; off >>= 1)
    v ^= __shfl_xor_sync(kFullMask, v, off);
  return v;
}

// Scalar-register field check (s0-s7 live in 3 bits of a 4-bit field).
__device__ inline bool BadScalarField(uint32_t f) { return (f & 8u) != 0u; }
// Predicate field check (p0-p3 in 2 bits).
__device__ inline bool BadPredField(uint32_t f) { return f >= kPredRegs; }

// Dense opcode numbers used by the alternative-dispatch experiment. The ISA
// encoding is intentionally grouped but sparse; ptxas lowers its sparse
// switch to a large compare/branch tree. A constant-memory translation makes
// the execute switch contiguous so ptxas can choose indexed branch dispatch.
__host__ __device__ constexpr uint32_t DenseOpcodeValue(uint32_t op) {
  return op <= 0x0Du ? op
       : op >= 0x10u && op <= 0x1Bu ? op - 2u
       : op >= 0x20u && op <= 0x22u ? op - 6u
       : op >= 0x28u && op <= 0x2Du ? op - 11u
       : op >= 0x30u && op <= 0x35u ? op - 13u
       : op >= 0x38u && op <= 0x3Du ? op - 15u
       : op >= 0x40u && op <= 0x4Cu ? op - 17u
       : op >= 0x50u && op <= 0x51u ? op - 20u
       : op >= 0x58u && op <= 0x5Au ? op - 26u
       : op >= 0x60u && op <= 0x61u ? op - 31u
       : op >= 0x70u && op <= 0x7Eu ? op - 45u
       : 0xFFu;
}

static __device__ __constant__ uint8_t kDenseOpcodeMap[128] = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 255, 255,
    14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 255, 255, 255, 255,
    26, 27, 28, 255, 255, 255, 255, 255, 29, 30, 31, 32, 33, 34, 255, 255,
    35, 36, 37, 38, 39, 40, 255, 255, 41, 42, 43, 44, 45, 46, 255, 255,
    47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 255, 255, 255,
    60, 61, 255, 255, 255, 255, 255, 255, 62, 63, 64, 255, 255, 255, 255, 255,
    65, 66, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 255};

template <bool kAlternative>
__host__ __device__ constexpr uint32_t DispatchCase(uint32_t op) {
  return kAlternative ? DenseOpcodeValue(op) : op;
}

// Execute until HALT, fault, or — when a control plane is attached — a
// pause/shutdown request at a control point (YIELD, backward jumps).
// Returns why execution stopped. With ctrl == nullptr this is the plain
// run-to-completion path used by `warpvm run` and the slice self-tests.
template <bool kResolveFaultVotes = true, bool kPollBackwardControl = true,
          bool kScalarizedVectorRegs = false,
          bool kAlternativeDispatch = false,
          bool kSharedVectorRegs = false,
          int kHotOpcodeCount = 0,
          bool kProfileFrameCycles = false>
__device__ StopReason VmRun(VmCtx& ctx, Control* ctrl = nullptr,
                            uint32_t vm_id = 0,
                            uint32_t* shared_vregs = nullptr) {
  if constexpr (kSharedVectorRegs) {
#pragma unroll 1
    for (uint32_t reg = 0; reg < kVectorRegs; ++reg)
      shared_vregs[reg * kPersistentBlockThreads] = ctx.vregs[reg];
  }
  VectorRegisterFile<kScalarizedVectorRegs> vregs(
      kSharedVectorRegs ? shared_vregs : ctx.vregs,
      kSharedVectorRegs ? kPersistentBlockThreads : 1);
  StopReason reason = kStopHalted;
  while (ctx.fault == kFaultOk) {
    if (ctx.pc >= ctx.code_len) {
      // Ran off the end without HALT.
      VmFault(ctx, kFaultJump);
      break;
    }
    const uint32_t this_pc = ctx.pc;
    const uint32_t instr = ctx.code[ctx.pc];
    const uint32_t op = (instr >> kOpcodeShift) & kOpcodeMask;
    const uint32_t guard = (instr >> kGuardShift) & kGuardMask;
    const uint32_t rd = (instr >> kRdShift) & kRegFieldMask;
    const uint32_t rs1 = (instr >> kRs1Shift) & kRegFieldMask;
    const uint32_t lo = instr & kLoMask;
    const uint32_t rs2 = lo & 0xFu;
    if (guard > 8u) {
      VmFault(ctx, kFaultOperand);
    }
    const bool active = GuardActive(ctx, guard);
    const uint32_t imm = static_cast<uint32_t>(SignExt13(lo));

    bool stop = false;
    bool jumped = false;

    const uint32_t dispatch_op =
        kAlternativeDispatch ? kDenseOpcodeMap[op] : op;
    bool hot_dispatched = false;
    if constexpr (kHotOpcodeCount >= 1) {
      // Profile-guided experiment: handle WarpLife's hottest opcodes before
      // the general ptxas-generated decision tree. The first two account for
      // 40.0% of its dynamic stream and the first four account for 55.3%.
      if (op == kAdd) {
        if (active) vregs.Set(rd, vregs.Get(rs1) + vregs.Get(rs2));
        hot_dispatched = true;
      } else if constexpr (kHotOpcodeCount >= 2) {
        if (op == kAndI) {
          if (active) vregs.Set(rd, vregs.Get(rs1) & imm);
          hot_dispatched = true;
        } else if constexpr (kHotOpcodeCount >= 4) {
          if (op == kAddI) {
            if (active) vregs.Set(rd, vregs.Get(rs1) + imm);
            hot_dispatched = true;
          } else if (op == kShlI) {
            if (active) vregs.Set(rd, vregs.Get(rs1) << (lo & 31u));
            hot_dispatched = true;
          }
        }
      }
    }
#define VM_CASE(opcode) case DispatchCase<kAlternativeDispatch>(opcode)
    if (!hot_dispatched) {
      switch (dispatch_op) {
      VM_CASE(kNop):
        break;

      // ---- vector arithmetic / bitwise (lane-wise) ----------------------
      VM_CASE(kMov):
        if (active) vregs.Set(rd, vregs.Get(rs1));
        break;
      VM_CASE(kAdd):
        if (active) vregs.Set(rd, vregs.Get(rs1) + vregs.Get(rs2));
        break;
      VM_CASE(kSub):
        if (active) vregs.Set(rd, vregs.Get(rs1) - vregs.Get(rs2));
        break;
      VM_CASE(kMul):
        if (active) vregs.Set(rd, vregs.Get(rs1) * vregs.Get(rs2));
        break;
      VM_CASE(kDiv):
        // Division by zero yields 0 (GP-friendliness, isa.md §11).
        if (active) {
          const uint32_t b = vregs.Get(rs2);
          vregs.Set(rd, b != 0u ? vregs.Get(rs1) / b : 0u);
        }
        break;
      VM_CASE(kMod):
        if (active) {
          const uint32_t b = vregs.Get(rs2);
          vregs.Set(rd, b != 0u ? vregs.Get(rs1) % b : 0u);
        }
        break;
      VM_CASE(kMin):
        if (active)
          vregs.Set(rd, vregs.Get(rs1) < vregs.Get(rs2) ? vregs.Get(rs1)
                                                        : vregs.Get(rs2));
        break;
      VM_CASE(kMax):
        if (active)
          vregs.Set(rd, vregs.Get(rs1) > vregs.Get(rs2) ? vregs.Get(rs1)
                                                        : vregs.Get(rs2));
        break;
      VM_CASE(kAnd):
        if (active) vregs.Set(rd, vregs.Get(rs1) & vregs.Get(rs2));
        break;
      VM_CASE(kOr):
        if (active) vregs.Set(rd, vregs.Get(rs1) | vregs.Get(rs2));
        break;
      VM_CASE(kXor):
        if (active) vregs.Set(rd, vregs.Get(rs1) ^ vregs.Get(rs2));
        break;
      VM_CASE(kShl):
        if (active)
          vregs.Set(rd, vregs.Get(rs1) << (vregs.Get(rs2) & 31u));
        break;
      VM_CASE(kShr):
        if (active)
          vregs.Set(rd, vregs.Get(rs1) >> (vregs.Get(rs2) & 31u));
        break;

      VM_CASE(kMovI):
        if (active) vregs.Set(rd, imm);
        break;
      VM_CASE(kAddI):
        if (active) vregs.Set(rd, vregs.Get(rs1) + imm);
        break;
      VM_CASE(kSubI):
        if (active) vregs.Set(rd, vregs.Get(rs1) - imm);
        break;
      VM_CASE(kMulI):
        if (active) vregs.Set(rd, vregs.Get(rs1) * imm);
        break;
      VM_CASE(kAndI):
        if (active) vregs.Set(rd, vregs.Get(rs1) & imm);
        break;
      VM_CASE(kOrI):
        if (active) vregs.Set(rd, vregs.Get(rs1) | imm);
        break;
      VM_CASE(kXorI):
        if (active) vregs.Set(rd, vregs.Get(rs1) ^ imm);
        break;
      VM_CASE(kShlI):
        if (active) vregs.Set(rd, vregs.Get(rs1) << (lo & 31u));
        break;
      VM_CASE(kShrI):
        if (active) vregs.Set(rd, vregs.Get(rs1) >> (lo & 31u));
        break;

      VM_CASE(kAbs):
        if (active) {
          const uint32_t v = vregs.Get(rs1);
          vregs.Set(rd, static_cast<int32_t>(v) < 0 ? 0u - v : v);
        }
        break;
      VM_CASE(kNeg):
        if (active) vregs.Set(rd, 0u - vregs.Get(rs1));
        break;
      VM_CASE(kNot):
        if (active) vregs.Set(rd, ~vregs.Get(rs1));
        break;

      // ---- comparisons (write predicate masks) ---------------------------
      VM_CASE(kCmpEq):
      VM_CASE(kCmpNe):
      VM_CASE(kCmpLt):
      VM_CASE(kCmpLe):
      VM_CASE(kCmpGt):
      VM_CASE(kCmpGe):
      VM_CASE(kCmpEqI):
      VM_CASE(kCmpNeI):
      VM_CASE(kCmpLtI):
      VM_CASE(kCmpLeI):
      VM_CASE(kCmpGtI):
      VM_CASE(kCmpGeI): {
        if (BadPredField(rd)) {
          VmFault(ctx, kFaultOperand);
          break;
        }
        const uint32_t a = vregs.Get(rs1);
        const uint32_t b = (op >= kCmpEqI) ? imm : vregs.Get(rs2);
        bool cond;
        switch (op) {
          case kCmpEq: case kCmpEqI: cond = a == b; break;
          case kCmpNe: case kCmpNeI: cond = a != b; break;
          case kCmpLt: case kCmpLtI: cond = a < b; break;
          case kCmpLe: case kCmpLeI: cond = a <= b; break;
          case kCmpGt: case kCmpGtI: cond = a > b; break;
          default:                   cond = a >= b; break;
        }
        const uint32_t mask = __ballot_sync(kFullMask, cond);
        if (active) ctx.preds[rd] = mask;
        break;
      }

      // ---- predicate-mask operations --------------------------------------
      VM_CASE(kNotMask):
        if (BadPredField(rd) || BadPredField(rs1)) {
          VmFault(ctx, kFaultOperand);
          break;
        }
        if (active) ctx.preds[rd] = ~ctx.preds[rs1];
        break;
      VM_CASE(kAndMask):
        if (BadPredField(rd) || BadPredField(rs1) || BadPredField(rs2)) {
          VmFault(ctx, kFaultOperand);
          break;
        }
        if (active) ctx.preds[rd] = ctx.preds[rs1] & ctx.preds[rs2];
        break;
      VM_CASE(kOrMask):
        if (BadPredField(rd) || BadPredField(rs1) || BadPredField(rs2)) {
          VmFault(ctx, kFaultOperand);
          break;
        }
        if (active) ctx.preds[rd] = ctx.preds[rs1] | ctx.preds[rs2];
        break;
      VM_CASE(kBallot): {
        if (BadPredField(rd)) {
          VmFault(ctx, kFaultOperand);
          break;
        }
        const uint32_t mask = __ballot_sync(kFullMask, vregs.Get(rs1) != 0u);
        if (active) ctx.preds[rd] = mask;
        break;
      }
      VM_CASE(kAny):
        if (BadPredField(rd) || BadPredField(rs1)) {
          VmFault(ctx, kFaultOperand);
          break;
        }
        if (active) ctx.preds[rd] = ctx.preds[rs1] != 0u ? kFullMask : 0u;
        break;
      VM_CASE(kAll):
        if (BadPredField(rd) || BadPredField(rs1)) {
          VmFault(ctx, kFaultOperand);
          break;
        }
        if (active)
          ctx.preds[rd] = ctx.preds[rs1] == kFullMask ? kFullMask : 0u;
        break;

      // ---- lane / warp operations -----------------------------------------
      VM_CASE(kLaneId):
        if (active) vregs.Set(rd, ctx.lane);
        break;
      VM_CASE(kBroadcast): {
        const uint32_t v = __shfl_sync(kFullMask, vregs.Get(rs1), lo & 31u);
        if (active) vregs.Set(rd, v);
        break;
      }
      VM_CASE(kShuffle): {
        const uint32_t v =
            __shfl_sync(kFullMask, vregs.Get(rs1), vregs.Get(rs2) & 31u);
        if (active) vregs.Set(rd, v);
        break;
      }
      VM_CASE(kShuffleXor): {
        const uint32_t v =
            __shfl_xor_sync(kFullMask, vregs.Get(rs1), lo & 31u);
        if (active) vregs.Set(rd, v);
        break;
      }
      VM_CASE(kReduceAdd): {
        const uint32_t v = WarpReduceAdd(vregs.Get(rs1));
        if (active) vregs.Set(rd, v);
        break;
      }
      VM_CASE(kReduceMin): {
        const uint32_t v = WarpReduceMin(vregs.Get(rs1));
        if (active) vregs.Set(rd, v);
        break;
      }
      VM_CASE(kReduceMax): {
        const uint32_t v = WarpReduceMax(vregs.Get(rs1));
        if (active) vregs.Set(rd, v);
        break;
      }
      VM_CASE(kReduceAnd): {
        const uint32_t v = WarpReduceAnd(vregs.Get(rs1));
        if (active) vregs.Set(rd, v);
        break;
      }
      VM_CASE(kReduceOr): {
        const uint32_t v = WarpReduceOr(vregs.Get(rs1));
        if (active) vregs.Set(rd, v);
        break;
      }
      VM_CASE(kReduceXor): {
        const uint32_t v = WarpReduceXor(vregs.Get(rs1));
        if (active) vregs.Set(rd, v);
        break;
      }
      VM_CASE(kVmid):
        if (active) vregs.Set(rd, ctx.vm_id);
        break;
      VM_CASE(kClock): {
        const uint32_t t = static_cast<uint32_t>(clock64());
        if (active) vregs.Set(rd, t);
        break;
      }
      VM_CASE(kRand): {
        // Uniform state advance; per-lane decorrelation.
        ctx.rng_state = Xorshift32(ctx.rng_state);
        uint32_t v = ctx.rng_state ^ (ctx.lane * 0x9E3779B9u);
        v = Xorshift32(v | 1u);
        if (active) vregs.Set(rd, v);
        break;
      }

      // ---- memory -----------------------------------------------------------
      // Address decode (docs/isa.md §4): private RAM first, then the
      // memory-mapped framebuffer, else FAULT_MEM. Per-lane addresses; any
      // lane's fault faults the whole VM (resolved by the ballot below).
      VM_CASE(kLoad):
        if (active) {
          const uint32_t addr = vregs.Get(rs1);
          if (addr < ctx.mem_size_words) {
            vregs.Set(rd, ctx.mem[addr]);
          } else if (ctx.fb != nullptr && addr >= kVideoBaseWord &&
                     addr < kVideoEndWord) {
            vregs.Set(rd, ctx.fb[addr - kVideoBaseWord]);
          } else {
            VmFault(ctx, kFaultMem);
          }
        }
        break;
      VM_CASE(kStore):
        if (active) {
          const uint32_t addr = vregs.Get(rd);
          if (addr < ctx.mem_size_words) {
            ctx.mem[addr] = vregs.Get(rs1);
          } else if (ctx.fb != nullptr && addr >= kVideoBaseWord &&
                     addr < kVideoEndWord) {
            ctx.fb[addr - kVideoBaseWord] = vregs.Get(rs1);
          } else {
            VmFault(ctx, kFaultMem);
          }
        }
        break;

      VM_CASE(kLdw):
        if (active) {
          if (lo >= ctx.literals_len) {
            VmFault(ctx, kFaultOperand);
            break;
          }
          vregs.Set(rd, ctx.literals[lo]);
        }
        break;

      // ---- logging (lane 0 appends to the host-visible ring) ---------------
      VM_CASE(kLog):
      VM_CASE(kLogI):
        if (active && ctrl != nullptr && ctx.lane == 0) {
          const uint32_t tag = (op == kLogI) ? imm : vregs.Get(rs2);
          LogAppend(ctrl, ctx.vm_id, tag, vregs.Get(rs1));
        }
        break;

      // ---- frame publication (v0.1.1) --------------------------------------
      // FLIP publishes the framebuffer: lane 0 bumps frame_seq exactly once.
      // Unguardable — a publication is a single warp-uniform event.
      VM_CASE(kFlip):
        if (guard != 0) {
          VmFault(ctx, kFaultOperand);
          break;
        }
        if (ctrl != nullptr && ctx.lane == 0) {
          if constexpr (kProfileFrameCycles) {
            const uint64_t now = clock64();
            const uint64_t previous =
                ReadOnce64(&ctrl->profile_frame_clock[ctx.slot_id]);
            if (previous != 0)
              WriteOnce64(&ctrl->profile_frame_cycles[ctx.slot_id],
                          now - previous);
            WriteOnce64(&ctrl->profile_frame_clock[ctx.slot_id], now);
          }
          WriteOnce(&ctrl->frame_seq[ctx.slot_id],
                    ReadOnce(&ctrl->frame_seq[ctx.slot_id]) + 1u);
        }
        break;

      // ---- messaging -------------------------------------------------------
      // SEND rDest, rType, rPayload: lane 0 posts a message to VM rDest[0].
      VM_CASE(kSend): {
        uint32_t err = 0;
        if (active && ctx.lane == 0) {
          if (ctx.mailboxes == nullptr) {
            err = 1;
          } else {
            const VmId dest = vregs.Get(rd);
            if (dest >= kVmIdCount || ctx.vm_routes == nullptr) {
              err = 1;
            } else {
              const VmSlot dest_slot = ctx.vm_routes[dest];
              if (dest_slot >= ctx.num_slots) {
                err = 1;
              } else {
                Message m;
                m.header = (ctx.vm_id & 0xFFFFu) |
                           ((vregs.Get(rs1) & 0xFFFFu) << 16);
                m.payload[0] = vregs.Get(rs2);
                m.payload[1] = 0;
                m.payload[2] = 0;
                if (!MailboxTrySend(ctx.mailboxes[dest_slot], m)) err = 1;
              }
            }
          }
        }
        err = __shfl_sync(kFullMask, err, 0);
        if (err != 0u) VmFault(ctx, kFaultMsg);
        break;
      }
      // TRY_RECV pGot, rPayload, rMeta: lane 0 consumes one pending message.
      // On success pGot = all lanes, rPayload = payload[0],
      // rMeta = header (type<<16 | src); else pGot = 0.
      VM_CASE(kTryRecv): {
        if (BadPredField(rd)) {
          VmFault(ctx, kFaultOperand);
          break;
        }
        uint32_t got = 0, payload = 0, meta = 0;
        if (active && ctx.lane == 0 && ctx.mailboxes != nullptr) {
          Message m;
          if (MailboxTryReceive(ctx.mailboxes[ctx.slot_id], m)) {
            got = 1;
            payload = m.payload[0];
            meta = m.header;
          }
        }
        got = __shfl_sync(kFullMask, got, 0);
        payload = __shfl_sync(kFullMask, payload, 0);
        meta = __shfl_sync(kFullMask, meta, 0);
        if (active) {
          ctx.preds[rd] = got != 0u ? kFullMask : 0u;
          if (got != 0u) {
            vregs.Set(rs1, payload);
            vregs.Set(rs2, meta);
          }
        }
        break;
      }

      // ---- scalar operations (uniform) --------------------------------------
      VM_CASE(kSMov):
        if (BadScalarField(rd) || BadScalarField(rs1)) {
          VmFault(ctx, kFaultOperand);
          break;
        }
        if (active) ctx.sregs[rd] = ctx.sregs[rs1];
        break;
      VM_CASE(kSMovI):
        if (BadScalarField(rd)) {
          VmFault(ctx, kFaultOperand);
          break;
        }
        if (active) ctx.sregs[rd] = imm;
        break;
      VM_CASE(kSAdd):
        if (BadScalarField(rd) || BadScalarField(rs1) || BadScalarField(rs2)) {
          VmFault(ctx, kFaultOperand);
          break;
        }
        if (active) ctx.sregs[rd] = ctx.sregs[rs1] + ctx.sregs[rs2];
        break;
      VM_CASE(kSAddI):
        if (BadScalarField(rd) || BadScalarField(rs1)) {
          VmFault(ctx, kFaultOperand);
          break;
        }
        if (active) ctx.sregs[rd] = ctx.sregs[rs1] + imm;
        break;
      VM_CASE(kSLdw):
        if (BadScalarField(rd)) {
          VmFault(ctx, kFaultOperand);
          break;
        }
        if (active) {
          if (lo >= ctx.literals_len) {
            VmFault(ctx, kFaultOperand);
            break;
          }
          ctx.sregs[rd] = ctx.literals[lo];
        }
        break;
      VM_CASE(kSCmpLt):
        if (BadPredField(rd) || BadScalarField(rs1) || BadScalarField(rs2)) {
          VmFault(ctx, kFaultOperand);
          break;
        }
        if (active)
          ctx.preds[rd] = ctx.sregs[rs1] < ctx.sregs[rs2] ? kFullMask : 0u;
        break;
      VM_CASE(kSCmpLtI):
        if (BadPredField(rd) || BadScalarField(rs1)) {
          VmFault(ctx, kFaultOperand);
          break;
        }
        if (active) ctx.preds[rd] = ctx.sregs[rs1] < imm ? kFullMask : 0u;
        break;
      VM_CASE(kSBcast):
        if (BadScalarField(rs1)) {
          VmFault(ctx, kFaultOperand);
          break;
        }
        if (active) vregs.Set(rd, ctx.sregs[rs1]);
        break;
      VM_CASE(kSGet): {
        if (BadScalarField(rd)) {
          VmFault(ctx, kFaultOperand);
          break;
        }
        const uint32_t v = __shfl_sync(kFullMask, vregs.Get(rs1), 0);
        if (active) ctx.sregs[rd] = v;
        break;
      }

      // ---- control flow -------------------------------------------------------
      VM_CASE(kJmp):
        if (lo >= ctx.code_len) {
          VmFault(ctx, kFaultJump);
          break;
        }
        ctx.pc = lo;
        jumped = true;
        break;
      VM_CASE(kJmpIfAny): {
        const uint32_t m = GuardMask(ctx, guard);
        if (m != 0u) {
          if (lo >= ctx.code_len) {
            VmFault(ctx, kFaultJump);
            break;
          }
          ctx.pc = lo;
          jumped = true;
        }
        break;
      }
      VM_CASE(kJmpIfAll): {
        const uint32_t m = GuardMask(ctx, guard);
        if (m == kFullMask) {
          if (lo >= ctx.code_len) {
            VmFault(ctx, kFaultJump);
            break;
          }
          ctx.pc = lo;
          jumped = true;
        }
        break;
      }
      VM_CASE(kCall):
        if (lo >= ctx.code_len) {
          VmFault(ctx, kFaultJump);
          break;
        }
        if (ctx.call_depth >= kCallDepth) {
          VmFault(ctx, kFaultStack);
          break;
        }
        ctx.call_stack[ctx.call_depth++] = ctx.pc + 1;
        ctx.pc = lo;
        jumped = true;
        break;
      VM_CASE(kRet):
        if (ctx.call_depth == 0u) {
          VmFault(ctx, kFaultStack);
          break;
        }
        ctx.pc = ctx.call_stack[--ctx.call_depth];
        jumped = true;
        break;
      VM_CASE(kHalt):
        stop = true;
        break;
      VM_CASE(kYield):
        // Cooperation point: publish live progress, then honour a pending
        // pause/shutdown when a control plane is attached.
        if (ctrl != nullptr) {
          if (ctx.instr_count - ctx.last_pub >= 1024u) {
            ctx.last_pub = ctx.instr_count;
            PublishStatus(ctrl, vm_id, ctx.lane, kRunning, 0, ctx.pc,
                          ctx.instr_count);
          }
          StopReason ir;
          if (CheckInterrupt(ctrl, vm_id, ctx.lane, &ir)) {
            reason = ir;
            stop = true;
          }
        }
        break;
      VM_CASE(kStepTrap):
        break;  // debugger hook; NOP unless debug-active

      default:
        VmFault(ctx, kFaultOpcode);
        break;
      }
    }
#undef VM_CASE

    // Resolve per-lane fault conditions to a warp-uniform decision: any
    // lane's fault faults the whole VM, taking the lowest faulting lane's
    // code. Keeps control flow converged even with scattered accesses.
    if constexpr (kResolveFaultVotes) {
      const uint32_t fault_ballot =
          __ballot_sync(kFullMask, ctx.fault != kFaultOk);
      if (fault_ballot != 0u) {
        ctx.fault =
            __shfl_sync(kFullMask, ctx.fault, __ffs(fault_ballot) - 1);
        break;
      }
    }
    ++ctx.instr_count;
    if (stop) break;

    // Control point: taken backward branch instructions are loop backedges,
    // so check for pause/shutdown there too. CALL/RET may also transfer to a
    // lower PC, but they are subroutine mechanics rather than loop control
    // points and must not poll mapped host state on every return.
    const bool backward_branch =
        jumped && ctx.pc <= this_pc &&
        (op == kJmp || op == kJmpIfAny || op == kJmpIfAll);
    if (backward_branch && ctrl != nullptr) {
      if (ctx.instr_count - ctx.last_pub >= 1024u) {
        ctx.last_pub = ctx.instr_count;
        PublishStatus(ctrl, vm_id, ctx.lane, kRunning, 0, ctx.pc,
                      ctx.instr_count);
      }
      if constexpr (kPollBackwardControl) {
        StopReason ir;
        if (CheckInterrupt(ctrl, vm_id, ctx.lane, &ir)) {
          reason = ir;
          break;
        }
      }
    }

    if (!jumped) ++ctx.pc;

    // Single-step: the one instruction has retired; re-pause at the new pc.
    if (ctx.step) {
      ctx.step = false;
      reason = kStopPaused;
      break;
    }
  }
  if constexpr (kSharedVectorRegs) {
#pragma unroll 1
    for (uint32_t reg = 0; reg < kVectorRegs; ++reg)
      ctx.vregs[reg] = shared_vregs[reg * kPersistentBlockThreads];
  } else {
    vregs.Spill(ctx.vregs);
  }
  if (ctx.fault != kFaultOk) return kStopFaulted;
  return reason;
}

}  // namespace wvm
