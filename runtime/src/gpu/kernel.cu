// VM execution kernels.
//
// VmArrayKernel   — slice 3: run each VM to HALT/fault, spill, exit. Used by
//                   `warpvm run` and the early-slice self-tests.
// PersistentKernel — slice 5: one warp per VM stays resident, driven by the
//                   host↔GPU control plane (run / pause / resume / reset /
//                   exit, global shutdown).
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

// Reset a VM context to its power-on state for the given descriptor.
__device__ static void InitVmCtx(VmCtx& ctx, const VmDesc& d, uint32_t vm_id,
                                 uint32_t lane) {
  ctx = VmCtx{};
  ctx.code = d.code;
  ctx.code_len = d.code_len;
  ctx.literals = d.literals;
  ctx.literals_len = d.literals_len;
  ctx.mem = d.mem;
  ctx.mem_size_words = d.mem_size_words;
  ctx.vm_id = vm_id;
  ctx.lane = lane;
  // Deterministic per-VM rng seed; a later slice may let the host set it.
  ctx.rng_state = vm_id * 0x9E3779B9u + 0x1234567u;
}

__global__ void VmArrayKernel(const VmDesc* descs, VmState* states) {
  const uint32_t global = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t vm_id = global >> 5;
  const VmDesc d = descs[vm_id];

  VmCtx ctx{};
  InitVmCtx(ctx, d, vm_id, global & (kLanes - 1));
  VmRun(ctx);
  SpillState(ctx, &states[vm_id]);
}

// ---------------------------------------------------------------------------
// Persistent kernel
// ---------------------------------------------------------------------------

// Run a VM through any number of pause/resume cycles until it halts, faults,
// is reset, or the kernel must exit. Returns false when the warp should leave
// the kernel (global shutdown or EXIT).
__device__ static bool RunVmUntilStop(VmCtx& ctx, const VmDesc& d,
                                      Control* ctrl, uint32_t vm_id,
                                      uint32_t lane, VmState* state) {
  for (;;) {
    PublishStatus(ctrl, vm_id, lane, kRunning, ctx.fault, ctx.pc,
                  ctx.instr_count);
    const StopReason reason = VmRun(ctx, ctrl, vm_id);
    SpillState(ctx, state);

    if (reason == kStopShutdown) return false;

    const uint32_t status = reason == kStopHalted   ? kHalted
                            : reason == kStopFaulted ? kFaulted
                                                     : kPaused;
    PublishStatus(ctrl, vm_id, lane, status, ctx.fault, ctx.pc,
                  ctx.instr_count);

    if (reason != kStopPaused) return true;  // halted/faulted: await host

    // Paused: hold at the control point until the host acts.
    for (;;) {
      if (ReadOnce(&ctrl->shutdown)) return false;
      const uint32_t next = ConsumeCmd(ctrl, vm_id, lane);
      if (next == kCmdRun) break;  // resume from the paused pc
      if (next == kCmdReset) {
        InitVmCtx(ctx, d, vm_id, lane);
        PublishStatus(ctrl, vm_id, lane, kIdle, 0, 0, 0);
        return true;
      }
      if (next == kCmdExit) return false;
      __nanosleep(1000);  // kCmdNone (or an in-band command): keep waiting
    }
    // Loop: VmRun continues from the paused pc.
  }
}

__global__ void PersistentKernel(const VmDesc* descs, VmState* states,
                                 Control* ctrl, uint32_t num_vms) {
  const uint32_t global = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t vm_id = global >> 5;
  const uint32_t lane = global & (kLanes - 1);
  if (vm_id >= num_vms) return;

  const VmDesc d = descs[vm_id];
  VmCtx ctx{};
  InitVmCtx(ctx, d, vm_id, lane);
  PublishStatus(ctrl, vm_id, lane, kIdle, 0, 0, 0);

  bool alive = true;
  while (alive) {
    if (ReadOnce(&ctrl->shutdown)) break;
    const uint32_t cmd = ConsumeCmd(ctrl, vm_id, lane);
    switch (cmd) {
      case kCmdRun:
        InitVmCtx(ctx, d, vm_id, lane);  // RUN = reset + execute
        alive = RunVmUntilStop(ctx, d, ctrl, vm_id, lane, &states[vm_id]);
        break;
      case kCmdReset:
        InitVmCtx(ctx, d, vm_id, lane);
        PublishStatus(ctrl, vm_id, lane, kIdle, 0, 0, 0);
        break;
      case kCmdExit:
        alive = false;
        break;
      case kCmdNone:
      default:
        __nanosleep(1000);  // idle: wait for the host
        break;
    }
  }

  // Final state so the host can observe a clean exit.
  if (lane == 0) WriteOnce(&ctrl->status[vm_id], kIdle);
}

}  // namespace wvm
