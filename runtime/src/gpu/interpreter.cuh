// Device-side fetch/decode/execute interpreter.
//
// All 32 lanes of the warp run this together. The opcode stream is uniform
// (every lane holds the same pc), so fetch/decode never diverges. Vector
// register slices live per-thread: lane i owns element i of every register.
// Scalar registers and predicate masks are uniform values replicated in
// every lane; they stay in sync by construction.
//
// Guarded instructions never branch per lane — inactive lanes keep their
// old rd value and perform no memory/message side effects.
#pragma once

#include "warpvm.cuh"

namespace wvm {

struct VmCtx {
  const uint32_t* code;
  uint32_t code_len;
  const uint32_t* literals;
  uint32_t literals_len;
  uint32_t vm_id;
  uint32_t lane;

  uint32_t pc;
  uint32_t vregs[kVectorRegs];  // this lane's slice of each vector register
  uint32_t sregs[kScalarRegs];  // uniform, replicated
  uint32_t preds[kPredRegs];    // uniform lane masks, replicated
  uint64_t instr_count;
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

__device__ inline int32_t SignExt13(uint32_t lo) {
  return (lo & 0x1000u) != 0u ? static_cast<int32_t>(lo | 0xFFFFE000u)
                               : static_cast<int32_t>(lo);
}

__device__ void VmRun(VmCtx& ctx) {
  while (ctx.fault == kFaultOk) {
    if (ctx.pc >= ctx.code_len) {
      // Ran off the end without HALT.
      VmFault(ctx, kFaultJump);
      break;
    }
    const uint32_t instr = ctx.code[ctx.pc];
    const uint32_t op = (instr >> kOpcodeShift) & kOpcodeMask;
    const uint32_t guard = (instr >> kGuardShift) & kGuardMask;
    const uint32_t rd = (instr >> kRdShift) & kRegFieldMask;
    const uint32_t rs1 = (instr >> kRs1Shift) & kRegFieldMask;
    const uint32_t lo = instr & kLoMask;
    const uint32_t rs2 = lo & 0xFu;
    const bool active = GuardActive(ctx, guard);

    bool stop = false;
    switch (op) {
      case kNop:
        break;

      case kMov:
        if (active) ctx.vregs[rd] = ctx.vregs[rs1];
        break;
      case kAdd:
        if (active) ctx.vregs[rd] = ctx.vregs[rs1] + ctx.vregs[rs2];
        break;
      case kMovI:
        if (active) ctx.vregs[rd] = static_cast<uint32_t>(SignExt13(lo));
        break;
      case kAddI:
        if (active)
          ctx.vregs[rd] =
              ctx.vregs[rs1] + static_cast<uint32_t>(SignExt13(lo));
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

      case kHalt:
        stop = true;
        break;

      default:
        VmFault(ctx, kFaultOpcode);
        break;
    }

    if (ctx.fault != kFaultOk) break;  // faulting instruction does not retire
    ++ctx.instr_count;
    if (stop) break;
    ++ctx.pc;
  }
}

}  // namespace wvm
