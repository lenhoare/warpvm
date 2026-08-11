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
                                 uint32_t lane, Mailbox* mailboxes,
                                 uint32_t num_vms) {
  ctx = VmCtx{};
  ctx.code = d.code;
  ctx.code_len = d.code_len;
  ctx.literals = d.literals;
  ctx.literals_len = d.literals_len;
  ctx.mem = d.mem;
  ctx.mem_size_words = d.mem_size_words;
  ctx.fb = d.fb;
  ctx.vm_id = vm_id;
  ctx.lane = lane;
  ctx.mailboxes = mailboxes;
  ctx.num_vms = num_vms;
  // Deterministic per-VM rng seed; a later slice may let the host set it.
  ctx.rng_state = vm_id * 0x9E3779B9u + 0x1234567u;
  // Reset clears only this VM's framebuffer to opaque black. All 32 lanes
  // cooperate on the clear (reset is not a hot path).
  if (ctx.fb != nullptr) {
    for (uint32_t i = lane; i < kVideoWords; i += kLanes)
      ctx.fb[i] = kVideoResetColor;
  }
}

__global__ void VmArrayKernel(const VmDesc* descs, VmState* states) {
  const uint32_t global = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t vm_id = global >> 5;
  const VmDesc d = descs[vm_id];

  VmCtx ctx{};
  InitVmCtx(ctx, d, vm_id, global & (kLanes - 1), nullptr, 0);
  VmRun(ctx);
  SpillState(ctx, &states[vm_id]);
}

// ---------------------------------------------------------------------------
// Persistent kernel
// ---------------------------------------------------------------------------

// Run a VM through any number of pause/resume/step cycles until it halts,
// faults, is reset, or the kernel must exit. Returns false when the warp
// should leave the kernel (global shutdown or EXIT).
template <bool kResolveFaultVotes, bool kPollBackwardControl>
__device__ static bool RunVmUntilStop(VmCtx& ctx, const VmDesc& d,
                                      Control* ctrl, uint32_t vm_id,
                                      uint32_t lane, VmState* state,
                                      Mailbox* mailboxes, uint32_t num_vms) {
  for (;;) {
    PublishStatus(ctrl, vm_id, lane, kRunning, ctx.fault, ctx.pc,
                  ctx.instr_count);
    const StopReason reason =
        VmRun<kResolveFaultVotes, kPollBackwardControl>(ctx, ctrl, vm_id);
    SpillState(ctx, state);

    if (reason == kStopShutdown) return false;
    if (reason == kStopHalted) {
      PublishStatus(ctrl, vm_id, lane, kHalted, ctx.fault, ctx.pc,
                    ctx.instr_count);
      return true;
    }
    if (reason == kStopFaulted) {
      PublishStatus(ctrl, vm_id, lane, kFaulted, ctx.fault, ctx.pc,
                    ctx.instr_count);
      return true;
    }

    // reason == kStopPaused: hold at the control point until the host acts.
    PublishStatus(ctrl, vm_id, lane, kPaused, ctx.fault, ctx.pc,
                  ctx.instr_count);
    for (;;) {
      if (ReadOnce(&ctrl->shutdown)) return false;
      const uint32_t next = ConsumeCmd(ctrl, vm_id, lane);
      if (next == kCmdRun) break;  // resume: outer loop re-runs VmRun
      if (next == kCmdStep) {
        // Retire exactly one instruction, then re-pause. Bump seq so the host
        // can detect completion (status may stay PAUSED throughout).
        ctx.step = true;
        const StopReason r =
            VmRun<kResolveFaultVotes, kPollBackwardControl>(ctx, ctrl, vm_id);
        ctx.step = false;
        SpillState(ctx, state);
        if (r == kStopShutdown) return false;
        if (r == kStopHalted) {
          PublishStatus(ctrl, vm_id, lane, kHalted, ctx.fault, ctx.pc,
                        ctx.instr_count);
          if (lane == 0)
            WriteOnce(&ctrl->seq[vm_id], ReadOnce(&ctrl->seq[vm_id]) + 1u);
          return true;
        }
        if (r == kStopFaulted) {
          PublishStatus(ctrl, vm_id, lane, kFaulted, ctx.fault, ctx.pc,
                        ctx.instr_count);
          if (lane == 0)
            WriteOnce(&ctrl->seq[vm_id], ReadOnce(&ctrl->seq[vm_id]) + 1u);
          return true;
        }
        PublishStatus(ctrl, vm_id, lane, kPaused, ctx.fault, ctx.pc,
                      ctx.instr_count);
        if (lane == 0)
          WriteOnce(&ctrl->seq[vm_id], ReadOnce(&ctrl->seq[vm_id]) + 1u);
        continue;  // still paused: wait for the next command
      }
      if (next == kCmdReset) {
        InitVmCtx(ctx, d, vm_id, lane, mailboxes, num_vms);
        PublishStatus(ctrl, vm_id, lane, kIdle, 0, 0, 0);
        return true;
      }
      if (next == kCmdExit) return false;
      __nanosleep(1000);  // kCmdNone (or stray command): keep waiting
    }
    // kCmdRun consumed: loop and continue executing from the paused pc.
  }
}

template <bool kResolveFaultVotes, bool kPollBackwardControl>
__device__ void PersistentKernelBody(const VmDesc* descs, VmState* states,
                                     Control* ctrl, uint32_t num_vms,
                                     Mailbox* mailboxes) {
  const uint32_t global = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t vm_id = global >> 5;
  const uint32_t lane = global & (kLanes - 1);
  if (vm_id >= num_vms) return;

  const VmDesc d = descs[vm_id];
  VmCtx ctx{};
  InitVmCtx(ctx, d, vm_id, lane, mailboxes, num_vms);
  PublishStatus(ctrl, vm_id, lane, kIdle, 0, 0, 0);

  bool alive = true;
  while (alive) {
    if (ReadOnce(&ctrl->shutdown)) break;
    const uint32_t cmd = ConsumeCmd(ctrl, vm_id, lane);
    switch (cmd) {
      case kCmdRun:
        InitVmCtx(ctx, d, vm_id, lane, mailboxes, num_vms);  // RUN = reset+run
        alive = RunVmUntilStop<kResolveFaultVotes, kPollBackwardControl>(
            ctx, d, ctrl, vm_id, lane, &states[vm_id], mailboxes, num_vms);
        break;
      case kCmdReset:
        InitVmCtx(ctx, d, vm_id, lane, mailboxes, num_vms);
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

__global__ void PersistentKernel(const VmDesc* descs, VmState* states,
                                 Control* ctrl, uint32_t num_vms,
                                 Mailbox* mailboxes) {
  PersistentKernelBody<true, true>(descs, states, ctrl, num_vms, mailboxes);
}

// Benchmark-only kernels. They deliberately relax fault voting and/or
// backward-branch host polling, but still poll at YIELD so shutdown works.
// Normal runtime commands never launch these variants.
__global__ void PersistentNoFaultVoteKernel(const VmDesc* descs,
                                            VmState* states, Control* ctrl,
                                            uint32_t num_vms,
                                            Mailbox* mailboxes) {
  PersistentKernelBody<false, true>(descs, states, ctrl, num_vms, mailboxes);
}

__global__ void PersistentYieldPollKernel(const VmDesc* descs,
                                          VmState* states, Control* ctrl,
                                          uint32_t num_vms,
                                          Mailbox* mailboxes) {
  PersistentKernelBody<true, false>(descs, states, ctrl, num_vms, mailboxes);
}

__global__ void PersistentMinimalProfileKernel(const VmDesc* descs,
                                               VmState* states, Control* ctrl,
                                               uint32_t num_vms,
                                               Mailbox* mailboxes) {
  PersistentKernelBody<false, false>(descs, states, ctrl, num_vms, mailboxes);
}

}  // namespace wvm
