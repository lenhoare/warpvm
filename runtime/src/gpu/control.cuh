// Host ↔ GPU control plane (docs/architecture.md).
//
// Allocated in pinned, mapped host memory so the resident kernel can read
// commands and write status with plain loads/stores over PCIe, while the
// host accesses the same words directly. All words are volatile on both
// sides; no caching assumptions.
#pragma once

#include <cstdint>

#include "warpvm.cuh"

namespace wvm {

constexpr uint32_t kMaxVms = 256;
constexpr uint32_t kLogCapacity = 1024;
// Persistent kernels use this width so the shared register-file layout,
// launch bounds, occupancy queries, and host launch geometry stay identical.
constexpr int kPersistentBlockThreads = 256;

// Host → GPU commands, one per VM. The warp consumes its command word
// (writes it back to kCmdNone) exactly once via ConsumeCmd.
enum VmCmd : uint32_t {
  kCmdNone = 0,
  kCmdRun = 1,     // execute (or resume executing) the program
  kCmdPause = 2,   // stop at the next control point
  kCmdReset = 3,   // reinitialise VM state (pc=0, registers zeroed)
  kCmdExit = 4,    // this VM's warp leaves the kernel
  kCmdStep = 5,    // paused VM: retire exactly one instruction, re-pause
};

// Why a VM's execution loop stopped.
enum StopReason : uint32_t {
  kStopHalted = 0,
  kStopFaulted = 1,
  kStopPaused = 2,
  kStopShutdown = 3,
};

struct LogEntry {
  uint32_t vm_id;
  uint32_t tag;
  uint32_t value;
};

struct Control {
  // ---- host → GPU ----
  volatile uint32_t shutdown;        // all warps exit
  volatile uint32_t cmd[kMaxVms];    // per-VM command word

  // ---- GPU → host ----
  volatile uint32_t status[kMaxVms]; // live Status values
  volatile uint32_t fault[kMaxVms];
  volatile uint32_t pc[kMaxVms];
  volatile uint64_t instrs[kMaxVms];
  volatile uint32_t seq[kMaxVms];    // bumped by the warp on each completed step
  volatile uint32_t frame_seq[kMaxVms];  // bumped by FLIP (frame publication)
  // Benchmark-only frame timestamps. Normal kernels leave these zero.
  volatile uint64_t profile_frame_clock[kMaxVms];
  volatile uint64_t profile_frame_cycles[kMaxVms];
  volatile uint32_t log_head;        // total entries appended
  LogEntry log[kLogCapacity];        // ring buffer
};

// ---- device-side accessors ------------------------------------------------

#ifdef __CUDACC__
__device__ inline uint32_t ReadOnce(const volatile uint32_t* p) { return *p; }
__device__ inline uint64_t ReadOnce64(const volatile uint64_t* p) { return *p; }
__device__ inline void WriteOnce(volatile uint32_t* p, uint32_t v) { *p = v; }
__device__ inline void WriteOnce64(volatile uint64_t* p, uint64_t v) { *p = v; }

// Publish a VM's execution snapshot to the host (lane 0 only inside).
__device__ inline void PublishStatus(Control* ctrl, uint32_t vm_id,
                                     uint32_t lane, uint32_t status,
                                     uint32_t fault, uint32_t pc,
                                     uint64_t instrs) {
  if (lane != 0) return;
  WriteOnce(&ctrl->fault[vm_id], fault);
  WriteOnce(&ctrl->pc[vm_id], pc);
  WriteOnce64(&ctrl->instrs[vm_id], instrs);
  WriteOnce(&ctrl->status[vm_id], status);  // status last: it's the flag
}

// Lane 0 reads the command word and clears it; result broadcast to all lanes.
__device__ inline uint32_t ConsumeCmd(Control* ctrl, uint32_t vm_id,
                                      uint32_t lane) {
  uint32_t cmd = kCmdNone;
  if (lane == 0) {
    cmd = ReadOnce(&ctrl->cmd[vm_id]);
    if (cmd != kCmdNone) WriteOnce(&ctrl->cmd[vm_id], kCmdNone);
  }
  return __shfl_sync(kFullMask, cmd, 0);
}

// Peek at the pending command without consuming it (uniform result).
__device__ inline uint32_t PeekCmd(Control* ctrl, uint32_t vm_id,
                                   uint32_t lane) {
  uint32_t cmd = kCmdNone;
  if (lane == 0) cmd = ReadOnce(&ctrl->cmd[vm_id]);
  return __shfl_sync(kFullMask, cmd, 0);
}

// Called at control points (YIELD, backward jumps). Uniform decision: lane 0
// reads over PCIe once, broadcasts the verdict. Consumes a pending pause.
__device__ inline bool CheckInterrupt(Control* ctrl, uint32_t vm_id,
                                      uint32_t lane, StopReason* reason) {
  uint32_t sh = 0;
  uint32_t pause = 0;
  if (lane == 0) {
    sh = ReadOnce(&ctrl->shutdown);
    if (!sh && ReadOnce(&ctrl->cmd[vm_id]) == kCmdPause) {
      WriteOnce(&ctrl->cmd[vm_id], kCmdNone);
      pause = 1;
    }
  }
  sh = __shfl_sync(kFullMask, sh, 0);
  pause = __shfl_sync(kFullMask, pause, 0);
  if (sh != 0u) {
    *reason = kStopShutdown;
    return true;
  }
  if (pause != 0u) {
    *reason = kStopPaused;
    return true;
  }
  return false;
}

// Append a log entry. Single-lane call (lane 0). Fire-and-forget ring; the
// host distinguishes entries by the monotonic log_head counter.
__device__ inline void LogAppend(Control* ctrl, uint32_t vm_id, uint32_t tag,
                                 uint32_t value) {
  const uint32_t slot = atomicAdd(const_cast<uint32_t*>(&ctrl->log_head), 1u);
  ctrl->log[slot % kLogCapacity] = LogEntry{vm_id, tag, value};
}
#endif

}  // namespace wvm
