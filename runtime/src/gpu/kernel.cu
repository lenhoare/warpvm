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
__device__ static void InitVmCtx(VmCtx& ctx, const VmDesc& d, VmId vm_id,
                                 VmSlot slot_id, uint32_t lane,
                                 Mailbox* mailboxes, uint32_t num_slots) {
  ctx = VmCtx{};
  ctx.code = d.code;
  ctx.code_len = d.code_len;
  ctx.literals = d.literals;
  ctx.literals_len = d.literals_len;
  ctx.mem = d.mem;
  ctx.mem_size_words = d.mem_size_words;
  ctx.fb = d.fb;
  ctx.vm_id = vm_id;
  ctx.slot_id = slot_id;
  ctx.lane = lane;
  ctx.mailboxes = mailboxes;
  ctx.vm_routes = d.vm_routes;
  ctx.num_slots = num_slots;
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
  const VmSlot slot_id = global >> 5;
  const VmDesc d = descs[slot_id];

  VmCtx ctx{};
  InitVmCtx(ctx, d, slot_id, slot_id, global & (kLanes - 1), nullptr, 0);
  VmRun(ctx);
  SpillState(ctx, &states[slot_id]);
}

// ---------------------------------------------------------------------------
// Persistent kernel
// ---------------------------------------------------------------------------

// Run a VM through any number of pause/resume/step cycles until it halts,
// faults, is reset, or the kernel must exit. Returns false when the warp
// should leave the kernel (global shutdown or EXIT).
template <bool kResolveFaultVotes, bool kPollBackwardControl,
          bool kScalarizedVectorRegs = false,
          bool kAlternativeDispatch = false,
          bool kSharedVectorRegs = false,
          int kHotOpcodeCount = 0,
          bool kProfileFrameCycles = false>
__device__ static bool RunVmUntilStop(VmCtx& ctx, const VmDesc& d,
                                      Control* ctrl, VmSlot slot_id,
                                      uint32_t lane, VmState* state,
                                      Mailbox* mailboxes, uint32_t num_slots,
                                      uint32_t* shared_vregs = nullptr) {
  for (;;) {
    PublishStatus(ctrl, slot_id, lane, kRunning, ctx.fault, ctx.pc,
                  ctx.instr_count);
    const StopReason reason =
        VmRun<kResolveFaultVotes, kPollBackwardControl, kScalarizedVectorRegs,
              kAlternativeDispatch, kSharedVectorRegs,
              kHotOpcodeCount, kProfileFrameCycles>(ctx, ctrl, slot_id,
                                                    shared_vregs);
    SpillState(ctx, state);

    if (reason == kStopShutdown) return false;
    if (reason == kStopHalted) {
      PublishStatus(ctrl, slot_id, lane, kHalted, ctx.fault, ctx.pc,
                    ctx.instr_count);
      return true;
    }
    if (reason == kStopFaulted) {
      PublishStatus(ctrl, slot_id, lane, kFaulted, ctx.fault, ctx.pc,
                    ctx.instr_count);
      return true;
    }

    // reason == kStopPaused: hold at the control point until the host acts.
    PublishStatus(ctrl, slot_id, lane, kPaused, ctx.fault, ctx.pc,
                  ctx.instr_count);
    for (;;) {
      if (ReadOnce(&ctrl->shutdown)) return false;
      const uint32_t next = ConsumeCmd(ctrl, slot_id, lane);
      if (next == kCmdRun) break;  // resume: outer loop re-runs VmRun
      if (next == kCmdStep) {
        // Retire exactly one instruction, then re-pause. Bump seq so the host
        // can detect completion (status may stay PAUSED throughout).
        ctx.step = true;
        const StopReason r =
            VmRun<kResolveFaultVotes, kPollBackwardControl,
                  kScalarizedVectorRegs, kAlternativeDispatch,
                  kSharedVectorRegs, kHotOpcodeCount, kProfileFrameCycles>(
                ctx, ctrl, slot_id, shared_vregs);
        ctx.step = false;
        SpillState(ctx, state);
        if (r == kStopShutdown) return false;
        if (r == kStopHalted) {
          PublishStatus(ctrl, slot_id, lane, kHalted, ctx.fault, ctx.pc,
                        ctx.instr_count);
          if (lane == 0)
            WriteOnce(&ctrl->seq[slot_id],
                      ReadOnce(&ctrl->seq[slot_id]) + 1u);
          return true;
        }
        if (r == kStopFaulted) {
          PublishStatus(ctrl, slot_id, lane, kFaulted, ctx.fault, ctx.pc,
                        ctx.instr_count);
          if (lane == 0)
            WriteOnce(&ctrl->seq[slot_id],
                      ReadOnce(&ctrl->seq[slot_id]) + 1u);
          return true;
        }
        PublishStatus(ctrl, slot_id, lane, kPaused, ctx.fault, ctx.pc,
                      ctx.instr_count);
        if (lane == 0)
          WriteOnce(&ctrl->seq[slot_id],
                    ReadOnce(&ctrl->seq[slot_id]) + 1u);
        continue;  // still paused: wait for the next command
      }
      if (next == kCmdReset) {
        InitVmCtx(ctx, d, ctx.vm_id, slot_id, lane, mailboxes, num_slots);
        PublishStatus(ctrl, slot_id, lane, kIdle, 0, 0, 0);
        return true;
      }
      if (next == kCmdExit) return false;
      __nanosleep(1000);  // kCmdNone (or stray command): keep waiting
    }
    // kCmdRun consumed: loop and continue executing from the paused pc.
  }
}

template <bool kResolveFaultVotes, bool kPollBackwardControl,
          bool kScalarizedVectorRegs = false,
          bool kAlternativeDispatch = false,
          bool kSharedVectorRegs = false,
          int kHotOpcodeCount = 0,
          bool kProfileFrameCycles = false>
__device__ void PersistentKernelBody(const VmDesc* descs, VmState* states,
                                     Control* ctrl, uint32_t num_vms,
                                     Mailbox* mailboxes,
                                     uint32_t* block_shared_vregs = nullptr) {
  const uint32_t global = blockIdx.x * blockDim.x + threadIdx.x;
  const VmSlot slot_id = global >> 5;
  const uint32_t lane = global & (kLanes - 1);
  if (slot_id >= num_vms) return;
  uint32_t* shared_vregs =
      kSharedVectorRegs
          ? block_shared_vregs + threadIdx.x
          : nullptr;

  const VmDesc d = descs[slot_id];
  const VmId vm_id = states[slot_id].vm_id;
  VmCtx ctx{};
  InitVmCtx(ctx, d, vm_id, slot_id, lane, mailboxes, num_vms);
  PublishStatus(ctrl, slot_id, lane, kIdle, 0, 0, 0);

  bool alive = true;
  while (alive) {
    if (ReadOnce(&ctrl->shutdown)) break;
    const uint32_t cmd = ConsumeCmd(ctrl, slot_id, lane);
    switch (cmd) {
      case kCmdRun:
        InitVmCtx(ctx, d, vm_id, slot_id, lane, mailboxes,
                  num_vms);  // RUN = reset+run
        alive = RunVmUntilStop<kResolveFaultVotes, kPollBackwardControl,
                               kScalarizedVectorRegs, kAlternativeDispatch,
                               kSharedVectorRegs, kHotOpcodeCount,
                               kProfileFrameCycles>(
            ctx, d, ctrl, slot_id, lane, &states[slot_id], mailboxes, num_vms,
            shared_vregs);
        break;
      case kCmdReset:
        InitVmCtx(ctx, d, vm_id, slot_id, lane, mailboxes, num_vms);
        PublishStatus(ctrl, slot_id, lane, kIdle, 0, 0, 0);
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
  if (lane == 0) WriteOnce(&ctrl->status[slot_id], kIdle);
}

__global__ void PersistentKernel(const VmDesc* descs, VmState* states,
                                 Control* ctrl, uint32_t num_vms,
                                 Mailbox* mailboxes) {
  PersistentKernelBody<true, true>(descs, states, ctrl, num_vms, mailboxes);
}

__global__ void PersistentScalarRegsKernel(const VmDesc* descs,
                                           VmState* states, Control* ctrl,
                                           uint32_t num_vms,
                                           Mailbox* mailboxes) {
  PersistentKernelBody<true, true, true, false>(descs, states, ctrl, num_vms,
                                                mailboxes);
}

__global__ void PersistentDenseDispatchKernel(const VmDesc* descs,
                                              VmState* states, Control* ctrl,
                                              uint32_t num_vms,
                                              Mailbox* mailboxes) {
  PersistentKernelBody<true, true, false, true>(descs, states, ctrl, num_vms,
                                                mailboxes);
}

__global__ void PersistentHotDispatchKernel(const VmDesc* descs,
                                            VmState* states, Control* ctrl,
                                            uint32_t num_vms,
                                            Mailbox* mailboxes) {
  PersistentKernelBody<true, true, false, false, false, 2>(
      descs, states, ctrl, num_vms, mailboxes);
}

__global__ void PersistentHot4DispatchKernel(const VmDesc* descs,
                                             VmState* states, Control* ctrl,
                                             uint32_t num_vms,
                                             Mailbox* mailboxes) {
  PersistentKernelBody<true, true, false, false, false, 4>(
      descs, states, ctrl, num_vms, mailboxes);
}

__global__ void PersistentCycleProfileKernel(const VmDesc* descs,
                                             VmState* states, Control* ctrl,
                                             uint32_t num_vms,
                                             Mailbox* mailboxes) {
  PersistentKernelBody<true, true, false, false, false, 0, true>(
      descs, states, ctrl, num_vms, mailboxes);
}

__global__ void PersistentHotCycleProfileKernel(const VmDesc* descs,
                                                VmState* states,
                                                Control* ctrl,
                                                uint32_t num_vms,
                                                Mailbox* mailboxes) {
  PersistentKernelBody<true, true, false, false, false, 2, true>(
      descs, states, ctrl, num_vms, mailboxes);
}

__global__ void PersistentHot4CycleProfileKernel(const VmDesc* descs,
                                                 VmState* states,
                                                 Control* ctrl,
                                                 uint32_t num_vms,
                                                 Mailbox* mailboxes) {
  PersistentKernelBody<true, true, false, false, false, 4, true>(
      descs, states, ctrl, num_vms, mailboxes);
}

__global__ void PersistentScalarDenseKernel(const VmDesc* descs,
                                            VmState* states, Control* ctrl,
                                            uint32_t num_vms,
                                            Mailbox* mailboxes) {
  PersistentKernelBody<true, true, true, true>(descs, states, ctrl, num_vms,
                                               mailboxes);
}

__global__ void PersistentSharedRegsKernel(const VmDesc* descs,
                                           VmState* states, Control* ctrl,
                                           uint32_t num_vms,
                                           Mailbox* mailboxes) {
  __shared__ uint32_t
      shared_vregs[kPersistentBlockThreads * kVectorRegs];
  PersistentKernelBody<true, true, false, false, true>(
      descs, states, ctrl, num_vms, mailboxes, shared_vregs);
}

__global__ __launch_bounds__(kPersistentBlockThreads, 3)
void PersistentSharedRegsThreeBlockKernel(const VmDesc* descs,
                                          VmState* states, Control* ctrl,
                                          uint32_t num_vms,
                                          Mailbox* mailboxes) {
  __shared__ uint32_t
      shared_vregs[kPersistentBlockThreads * kVectorRegs];
  PersistentKernelBody<true, true, false, false, true>(
      descs, states, ctrl, num_vms, mailboxes, shared_vregs);
}

__global__ __launch_bounds__(kPersistentBlockThreads, 3)
void PersistentSharedDenseThreeBlockKernel(const VmDesc* descs,
                                           VmState* states, Control* ctrl,
                                           uint32_t num_vms,
                                           Mailbox* mailboxes) {
  __shared__ uint32_t
      shared_vregs[kPersistentBlockThreads * kVectorRegs];
  PersistentKernelBody<true, true, false, true, true>(
      descs, states, ctrl, num_vms, mailboxes, shared_vregs);
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
