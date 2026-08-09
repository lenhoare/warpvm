// VM execution kernel. Slice 2: one warp runs one program to HALT/fault,
// spills final state to global memory, exits. Later slices turn this into
// the persistent multi-VM kernel.
#include "gpu/interpreter.cuh"
#include "gpu/vm_state.cuh"

namespace wvm {

__global__ void VmKernel(const uint32_t* code, uint32_t code_len,
                         const uint32_t* literals, uint32_t literals_len,
                         VmState* state) {
  VmCtx ctx{};
  ctx.code = code;
  ctx.code_len = code_len;
  ctx.literals = literals;
  ctx.literals_len = literals_len;
  ctx.vm_id = 0;
  ctx.lane = threadIdx.x & (kLanes - 1);

  VmRun(ctx);

  // Spill register state for host inspection.
  for (int r = 0; r < kVectorRegs; ++r)
    state->vregs[r * kLanes + ctx.lane] = ctx.vregs[r];

  if (ctx.lane == 0) {
    state->vm_id = ctx.vm_id;
    state->status = ctx.fault != kFaultOk ? kFaulted : kHalted;
    state->fault_code = ctx.fault;
    state->pc = ctx.pc;
    state->instruction_counter = ctx.instr_count;
    for (int r = 0; r < kScalarRegs; ++r) state->sregs[r] = ctx.sregs[r];
    for (int r = 0; r < kPredRegs; ++r) state->preds[r] = ctx.preds[r];
  }
}

}  // namespace wvm
