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
  uint32_t vm_id;
  uint32_t lane;

  Mailbox* mailboxes;   // global mailbox array (nullptr if messaging absent)
  uint32_t num_vms;     // for validating SEND destinations

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

// Execute until HALT, fault, or — when a control plane is attached — a
// pause/shutdown request at a control point (YIELD, backward jumps).
// Returns why execution stopped. With ctrl == nullptr this is the plain
// run-to-completion path used by `warpvm run` and the slice self-tests.
__device__ StopReason VmRun(VmCtx& ctx, Control* ctrl = nullptr,
                            uint32_t vm_id = 0) {
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

    switch (op) {
      case kNop:
        break;

      // ---- vector arithmetic / bitwise (lane-wise) ----------------------
      case kMov:
        if (active) ctx.vregs[rd] = ctx.vregs[rs1];
        break;
      case kAdd:
        if (active) ctx.vregs[rd] = ctx.vregs[rs1] + ctx.vregs[rs2];
        break;
      case kSub:
        if (active) ctx.vregs[rd] = ctx.vregs[rs1] - ctx.vregs[rs2];
        break;
      case kMul:
        if (active) ctx.vregs[rd] = ctx.vregs[rs1] * ctx.vregs[rs2];
        break;
      case kDiv:
        // Division by zero yields 0 (GP-friendliness, isa.md §11).
        if (active) {
          const uint32_t b = ctx.vregs[rs2];
          ctx.vregs[rd] = b != 0u ? ctx.vregs[rs1] / b : 0u;
        }
        break;
      case kMod:
        if (active) {
          const uint32_t b = ctx.vregs[rs2];
          ctx.vregs[rd] = b != 0u ? ctx.vregs[rs1] % b : 0u;
        }
        break;
      case kMin:
        if (active)
          ctx.vregs[rd] =
              ctx.vregs[rs1] < ctx.vregs[rs2] ? ctx.vregs[rs1] : ctx.vregs[rs2];
        break;
      case kMax:
        if (active)
          ctx.vregs[rd] =
              ctx.vregs[rs1] > ctx.vregs[rs2] ? ctx.vregs[rs1] : ctx.vregs[rs2];
        break;
      case kAnd:
        if (active) ctx.vregs[rd] = ctx.vregs[rs1] & ctx.vregs[rs2];
        break;
      case kOr:
        if (active) ctx.vregs[rd] = ctx.vregs[rs1] | ctx.vregs[rs2];
        break;
      case kXor:
        if (active) ctx.vregs[rd] = ctx.vregs[rs1] ^ ctx.vregs[rs2];
        break;
      case kShl:
        if (active) ctx.vregs[rd] = ctx.vregs[rs1] << (ctx.vregs[rs2] & 31u);
        break;
      case kShr:
        if (active) ctx.vregs[rd] = ctx.vregs[rs1] >> (ctx.vregs[rs2] & 31u);
        break;

      case kMovI:
        if (active) ctx.vregs[rd] = imm;
        break;
      case kAddI:
        if (active) ctx.vregs[rd] = ctx.vregs[rs1] + imm;
        break;
      case kSubI:
        if (active) ctx.vregs[rd] = ctx.vregs[rs1] - imm;
        break;
      case kMulI:
        if (active) ctx.vregs[rd] = ctx.vregs[rs1] * imm;
        break;
      case kAndI:
        if (active) ctx.vregs[rd] = ctx.vregs[rs1] & imm;
        break;
      case kOrI:
        if (active) ctx.vregs[rd] = ctx.vregs[rs1] | imm;
        break;
      case kXorI:
        if (active) ctx.vregs[rd] = ctx.vregs[rs1] ^ imm;
        break;
      case kShlI:
        if (active) ctx.vregs[rd] = ctx.vregs[rs1] << (lo & 31u);
        break;
      case kShrI:
        if (active) ctx.vregs[rd] = ctx.vregs[rs1] >> (lo & 31u);
        break;

      case kAbs:
        if (active) {
          const uint32_t v = ctx.vregs[rs1];
          ctx.vregs[rd] = static_cast<int32_t>(v) < 0 ? 0u - v : v;
        }
        break;
      case kNeg:
        if (active) ctx.vregs[rd] = 0u - ctx.vregs[rs1];
        break;
      case kNot:
        if (active) ctx.vregs[rd] = ~ctx.vregs[rs1];
        break;

      // ---- comparisons (write predicate masks) ---------------------------
      case kCmpEq:
      case kCmpNe:
      case kCmpLt:
      case kCmpLe:
      case kCmpGt:
      case kCmpGe:
      case kCmpEqI:
      case kCmpNeI:
      case kCmpLtI:
      case kCmpLeI:
      case kCmpGtI:
      case kCmpGeI: {
        if (BadPredField(rd)) {
          VmFault(ctx, kFaultOperand);
          break;
        }
        const uint32_t a = ctx.vregs[rs1];
        const uint32_t b = (op >= kCmpEqI) ? imm : ctx.vregs[rs2];
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
      case kNotMask:
        if (BadPredField(rd) || BadPredField(rs1)) {
          VmFault(ctx, kFaultOperand);
          break;
        }
        if (active) ctx.preds[rd] = ~ctx.preds[rs1];
        break;
      case kAndMask:
        if (BadPredField(rd) || BadPredField(rs1) || BadPredField(rs2)) {
          VmFault(ctx, kFaultOperand);
          break;
        }
        if (active) ctx.preds[rd] = ctx.preds[rs1] & ctx.preds[rs2];
        break;
      case kOrMask:
        if (BadPredField(rd) || BadPredField(rs1) || BadPredField(rs2)) {
          VmFault(ctx, kFaultOperand);
          break;
        }
        if (active) ctx.preds[rd] = ctx.preds[rs1] | ctx.preds[rs2];
        break;
      case kBallot: {
        if (BadPredField(rd)) {
          VmFault(ctx, kFaultOperand);
          break;
        }
        const uint32_t mask = __ballot_sync(kFullMask, ctx.vregs[rs1] != 0u);
        if (active) ctx.preds[rd] = mask;
        break;
      }
      case kAny:
        if (BadPredField(rd) || BadPredField(rs1)) {
          VmFault(ctx, kFaultOperand);
          break;
        }
        if (active) ctx.preds[rd] = ctx.preds[rs1] != 0u ? kFullMask : 0u;
        break;
      case kAll:
        if (BadPredField(rd) || BadPredField(rs1)) {
          VmFault(ctx, kFaultOperand);
          break;
        }
        if (active)
          ctx.preds[rd] = ctx.preds[rs1] == kFullMask ? kFullMask : 0u;
        break;

      // ---- lane / warp operations -----------------------------------------
      case kLaneId:
        if (active) ctx.vregs[rd] = ctx.lane;
        break;
      case kBroadcast: {
        const uint32_t v = __shfl_sync(kFullMask, ctx.vregs[rs1], lo & 31u);
        if (active) ctx.vregs[rd] = v;
        break;
      }
      case kShuffle: {
        const uint32_t v =
            __shfl_sync(kFullMask, ctx.vregs[rs1], ctx.vregs[rs2] & 31u);
        if (active) ctx.vregs[rd] = v;
        break;
      }
      case kShuffleXor: {
        const uint32_t v =
            __shfl_xor_sync(kFullMask, ctx.vregs[rs1], lo & 31u);
        if (active) ctx.vregs[rd] = v;
        break;
      }
      case kReduceAdd: {
        const uint32_t v = WarpReduceAdd(ctx.vregs[rs1]);
        if (active) ctx.vregs[rd] = v;
        break;
      }
      case kReduceMin: {
        const uint32_t v = WarpReduceMin(ctx.vregs[rs1]);
        if (active) ctx.vregs[rd] = v;
        break;
      }
      case kReduceMax: {
        const uint32_t v = WarpReduceMax(ctx.vregs[rs1]);
        if (active) ctx.vregs[rd] = v;
        break;
      }
      case kReduceAnd: {
        const uint32_t v = WarpReduceAnd(ctx.vregs[rs1]);
        if (active) ctx.vregs[rd] = v;
        break;
      }
      case kReduceOr: {
        const uint32_t v = WarpReduceOr(ctx.vregs[rs1]);
        if (active) ctx.vregs[rd] = v;
        break;
      }
      case kReduceXor: {
        const uint32_t v = WarpReduceXor(ctx.vregs[rs1]);
        if (active) ctx.vregs[rd] = v;
        break;
      }
      case kVmid:
        if (active) ctx.vregs[rd] = ctx.vm_id;
        break;
      case kClock: {
        const uint32_t t = static_cast<uint32_t>(clock64());
        if (active) ctx.vregs[rd] = t;
        break;
      }
      case kRand: {
        // Uniform state advance; per-lane decorrelation.
        ctx.rng_state = Xorshift32(ctx.rng_state);
        uint32_t v = ctx.rng_state ^ (ctx.lane * 0x9E3779B9u);
        v = Xorshift32(v | 1u);
        if (active) ctx.vregs[rd] = v;
        break;
      }

      // ---- memory -----------------------------------------------------------
      // Address decode (docs/isa.md §4): private RAM first, then the
      // memory-mapped framebuffer, else FAULT_MEM. Per-lane addresses; any
      // lane's fault faults the whole VM (resolved by the ballot below).
      case kLoad:
        if (active) {
          const uint32_t addr = ctx.vregs[rs1];
          if (addr < ctx.mem_size_words) {
            ctx.vregs[rd] = ctx.mem[addr];
          } else if (ctx.fb != nullptr && addr >= kVideoBaseWord &&
                     addr < kVideoEndWord) {
            ctx.vregs[rd] = ctx.fb[addr - kVideoBaseWord];
          } else {
            VmFault(ctx, kFaultMem);
          }
        }
        break;
      case kStore:
        if (active) {
          const uint32_t addr = ctx.vregs[rd];
          if (addr < ctx.mem_size_words) {
            ctx.mem[addr] = ctx.vregs[rs1];
          } else if (ctx.fb != nullptr && addr >= kVideoBaseWord &&
                     addr < kVideoEndWord) {
            ctx.fb[addr - kVideoBaseWord] = ctx.vregs[rs1];
          } else {
            VmFault(ctx, kFaultMem);
          }
        }
        break;

      case kLdw:
        if (active) {
          if (lo >= ctx.literals_len) {
            VmFault(ctx, kFaultOperand);
            break;
          }
          ctx.vregs[rd] = ctx.literals[lo];
        }
        break;

      // ---- logging (lane 0 appends to the host-visible ring) ---------------
      case kLog:
      case kLogI:
        if (active && ctrl != nullptr && ctx.lane == 0) {
          const uint32_t tag = (op == kLogI) ? imm : ctx.vregs[rs2];
          LogAppend(ctrl, ctx.vm_id, tag, ctx.vregs[rs1]);
        }
        break;

      // ---- frame publication (v0.1.1) --------------------------------------
      // FLIP publishes the framebuffer: lane 0 bumps frame_seq exactly once.
      // Unguardable — a publication is a single warp-uniform event.
      case kFlip:
        if (guard != 0) {
          VmFault(ctx, kFaultOperand);
          break;
        }
        if (ctrl != nullptr && ctx.lane == 0) {
          WriteOnce(&ctrl->frame_seq[ctx.vm_id],
                    ReadOnce(&ctrl->frame_seq[ctx.vm_id]) + 1u);
        }
        break;

      // ---- messaging -------------------------------------------------------
      // SEND rDest, rType, rPayload: lane 0 posts a message to VM rDest[0].
      case kSend: {
        uint32_t err = 0;
        if (active && ctx.lane == 0) {
          if (ctx.mailboxes == nullptr) {
            err = 1;
          } else {
            const uint32_t dest = ctx.vregs[rd];
            if (dest >= ctx.num_vms) {
              err = 1;
            } else {
              Mailbox& mb = ctx.mailboxes[dest];
              const uint32_t h = mb.head;
              const uint32_t t = mb.tail;
              if (h - t >= kMailboxSlots) {
                err = 1;  // mailbox full
              } else {
                const uint32_t slot =
                    atomicAdd(const_cast<uint32_t*>(&mb.head), 1u);
                Message m;
                m.header = (ctx.vm_id & 0xFFFFu) |
                           ((ctx.vregs[rs1] & 0xFFFFu) << 16);
                m.payload[0] = ctx.vregs[rs2];
                m.payload[1] = 0;
                m.payload[2] = 0;
                mb.slots[slot % kMailboxSlots] = m;
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
      case kTryRecv: {
        if (BadPredField(rd)) {
          VmFault(ctx, kFaultOperand);
          break;
        }
        uint32_t got = 0, payload = 0, meta = 0;
        if (ctx.lane == 0 && ctx.mailboxes != nullptr) {
          Mailbox& mb = ctx.mailboxes[ctx.vm_id];
          const uint32_t h = mb.head;
          const uint32_t t = mb.tail;
          if (h != t) {
            const Message m = mb.slots[t % kMailboxSlots];
            mb.tail = t + 1;
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
            ctx.vregs[rs1] = payload;
            ctx.vregs[rs2] = meta;
          }
        }
        break;
      }

      // ---- scalar operations (uniform) --------------------------------------
      case kSMov:
        if (BadScalarField(rd) || BadScalarField(rs1)) {
          VmFault(ctx, kFaultOperand);
          break;
        }
        if (active) ctx.sregs[rd] = ctx.sregs[rs1];
        break;
      case kSMovI:
        if (BadScalarField(rd)) {
          VmFault(ctx, kFaultOperand);
          break;
        }
        if (active) ctx.sregs[rd] = imm;
        break;
      case kSAdd:
        if (BadScalarField(rd) || BadScalarField(rs1) || BadScalarField(rs2)) {
          VmFault(ctx, kFaultOperand);
          break;
        }
        if (active) ctx.sregs[rd] = ctx.sregs[rs1] + ctx.sregs[rs2];
        break;
      case kSAddI:
        if (BadScalarField(rd) || BadScalarField(rs1)) {
          VmFault(ctx, kFaultOperand);
          break;
        }
        if (active) ctx.sregs[rd] = ctx.sregs[rs1] + imm;
        break;
      case kSLdw:
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
      case kSCmpLt:
        if (BadPredField(rd) || BadScalarField(rs1) || BadScalarField(rs2)) {
          VmFault(ctx, kFaultOperand);
          break;
        }
        if (active)
          ctx.preds[rd] = ctx.sregs[rs1] < ctx.sregs[rs2] ? kFullMask : 0u;
        break;
      case kSCmpLtI:
        if (BadPredField(rd) || BadScalarField(rs1)) {
          VmFault(ctx, kFaultOperand);
          break;
        }
        if (active) ctx.preds[rd] = ctx.sregs[rs1] < imm ? kFullMask : 0u;
        break;
      case kSBcast:
        if (BadScalarField(rs1)) {
          VmFault(ctx, kFaultOperand);
          break;
        }
        if (active) ctx.vregs[rd] = ctx.sregs[rs1];
        break;
      case kSGet: {
        if (BadScalarField(rd)) {
          VmFault(ctx, kFaultOperand);
          break;
        }
        const uint32_t v = __shfl_sync(kFullMask, ctx.vregs[rs1], 0);
        if (active) ctx.sregs[rd] = v;
        break;
      }

      // ---- control flow -------------------------------------------------------
      case kJmp:
        if (lo >= ctx.code_len) {
          VmFault(ctx, kFaultJump);
          break;
        }
        ctx.pc = lo;
        jumped = true;
        break;
      case kJmpIfAny: {
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
      case kJmpIfAll: {
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
      case kCall:
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
      case kRet:
        if (ctx.call_depth == 0u) {
          VmFault(ctx, kFaultStack);
          break;
        }
        ctx.pc = ctx.call_stack[--ctx.call_depth];
        jumped = true;
        break;
      case kHalt:
        stop = true;
        break;
      case kYield:
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
      case kStepTrap:
        break;  // debugger hook; NOP unless debug-active

      default:
        VmFault(ctx, kFaultOpcode);
        break;
    }

    // Resolve per-lane fault conditions to a warp-uniform decision: any
    // lane's fault faults the whole VM, taking the lowest faulting lane's
    // code. Keeps control flow converged even with scattered accesses.
    const uint32_t fault_ballot =
        __ballot_sync(kFullMask, ctx.fault != kFaultOk);
    if (fault_ballot != 0u) {
      ctx.fault = __shfl_sync(kFullMask, ctx.fault, __ffs(fault_ballot) - 1);
      break;
    }
    ++ctx.instr_count;
    if (stop) break;

    // Control point: backward branches are loop heads, so check for a
    // pause/shutdown there too. pc already holds the taken target; stopping
    // here resumes at that target. Publish throttled progress for the host.
    if (jumped && ctrl != nullptr && ctx.pc <= this_pc) {
      if (ctx.instr_count - ctx.last_pub >= 1024u) {
        ctx.last_pub = ctx.instr_count;
        PublishStatus(ctrl, vm_id, ctx.lane, kRunning, 0, ctx.pc,
                      ctx.instr_count);
      }
      StopReason ir;
      if (CheckInterrupt(ctrl, vm_id, ctx.lane, &ir)) {
        reason = ir;
        break;
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
  if (ctx.fault != kFaultOk) return kStopFaulted;
  return reason;
}

}  // namespace wvm
