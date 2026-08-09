// Host-visible VM state: the spill target a warp writes when it stops
// (HALT/fault) or pauses. Live execution state is per-thread registers;
// see docs/architecture.md.
#pragma once

#include <cstdint>

#include "warpvm.cuh"

namespace wvm {

struct VmState {
  uint32_t vm_id;
  uint32_t status;
  uint32_t pc;
  uint32_t fault_code;
  uint64_t instruction_counter;
  uint32_t vregs[kVectorRegs * kLanes];  // [r * kLanes + lane]
  uint32_t sregs[kScalarRegs];
  uint32_t preds[kPredRegs];
};

inline const char* StatusName(uint32_t s) {
  switch (s) {
    case kIdle: return "IDLE";
    case kRunning: return "RUNNING";
    case kPaused: return "PAUSED";
    case kHalted: return "HALTED";
    case kFaulted: return "FAULTED";
    case kDebug: return "DEBUG";
    default: return "?";
  }
}

inline const char* FaultName(uint32_t f) {
  switch (f) {
    case kFaultOk: return "OK";
    case kFaultOpcode: return "OPCODE";
    case kFaultOperand: return "OPERAND";
    case kFaultJump: return "JUMP";
    case kFaultMem: return "MEM";
    case kFaultStack: return "STACK";
    case kFaultMsg: return "MSG";
    case kFaultBudget: return "BUDGET";
    default: return "?";
  }
}

}  // namespace wvm
