// VM execution kernel. Slice 3: one warp per VM, many VMs per launch,
// each with its own descriptor (program, literals, private RAM). Every
// warp runs to HALT/fault and spills final state. Slice 5 turns this into
// the persistent kernel with a host control plane.
#include "gpu/interpreter.cuh"
#include "gpu/vm_state.cuh"

namespace wvm {

__device__ static void SpillState(const VmCtx& ctx, VmState* state) {
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
    for (int r = 0; r < kCallDepth; ++r)
      state->call_stack[r] = ctx.call_stack[r];
    state->call_depth = ctx.call_depth;
    state->rng_state = ctx.rng_state;
  }
}

// One warp per VM. vm_id is the warp's slot index — a stable logical ID
// independent of which SM the warp is scheduled on. Launch geometry is a
// multiple of 32 threads, so a VM never straddles two hardware warps.
__global__ void VmArrayKernel(const VmDesc* descs, VmState* states) {
  const uint32_t global = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t vm_id = global >> 5;
  const VmDesc d = descs[vm_id];

  VmCtx ctx{};
  ctx.code = d.code;
  ctx.code_len = d.code_len;
  ctx.literals = d.literals;
  ctx.literals_len = d.literals_len;
  ctx.mem = d.mem;
  ctx.mem_size_words = d.mem_size_words;
  ctx.vm_id = vm_id;
  ctx.lane = global & (kLanes - 1);
  // Deterministic per-VM rng seed; a later slice may let the host set it.
  ctx.rng_state = vm_id * 0x9E3779B9u + 0x1234567u;

  VmRun(ctx);
  SpillState(ctx, &states[vm_id]);
}

}  // namespace wvm
