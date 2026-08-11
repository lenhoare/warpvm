#include "host/cpu_interpreter.h"

#include <algorithm>
#include <cstdio>
#include <chrono>

namespace wvm {
namespace {

int32_t SignExt13(uint32_t lo) {
  return (lo & 0x1000u) ? static_cast<int32_t>(lo | 0xFFFFE000u)
                        : static_cast<int32_t>(lo);
}

uint32_t Xorshift32(uint32_t value) {
  value ^= value << 13;
  value ^= value >> 17;
  value ^= value << 5;
  return value;
}

bool BadScalar(uint32_t field) { return (field & 8u) != 0; }
bool BadPred(uint32_t field) { return field >= kPredRegs; }

uint32_t ActiveMask(const CpuVm& vm, uint32_t guard) {
  if (guard == 0) return kFullMask;
  const uint32_t mask = vm.preds[(guard - 1) & 3u];
  return guard >= 5 ? ~mask : mask;
}

void WritePred(CpuVm& vm, uint32_t destination, uint32_t value,
               uint32_t active) {
  vm.preds[destination] =
      (vm.preds[destination] & ~active) | (value & active);
}

template <typename Function>
void WriteVector(CpuVm& vm, uint32_t destination, uint32_t active,
                 Function&& value) {
  for (uint32_t lane = 0; lane < kLanes; ++lane)
    if ((active >> lane) & 1u) vm.vregs[destination][lane] = value(lane);
}

}  // namespace

void CpuVm::Init(uint32_t id, const WvmFile& file, uint32_t memory_words) {
  vm_id = id;
  code = file.code;
  literals = file.literals;
  memory.assign(memory_words, 0u);
  framebuffer.assign(kVideoWords, kVideoResetColor);
  Reset();
}

void CpuVm::Reset() {
  status = kRunning;
  pc = 0;
  fault = kFaultOk;
  instruction_counter = 0;
  frame_seq = 0;
  rng_state = vm_id * 0x9E3779B9u + 0x1234567u;
  vregs = {};
  sregs = {};
  preds = {};
  call_stack = {};
  call_depth = 0;
  std::fill(framebuffer.begin(), framebuffer.end(), kVideoResetColor);
}

bool CpuVm::Step() {
  if (status != kRunning) return false;
  if (pc >= code.size()) {
    fault = kFaultJump;
    status = kFaulted;
    return false;
  }

  const uint32_t instruction = code[pc];
  const uint32_t op = (instruction >> kOpcodeShift) & kOpcodeMask;
  const uint32_t guard = (instruction >> kGuardShift) & kGuardMask;
  const uint32_t rd = (instruction >> kRdShift) & kRegFieldMask;
  const uint32_t rs1 = (instruction >> kRs1Shift) & kRegFieldMask;
  const uint32_t lo = instruction & kLoMask;
  const uint32_t rs2 = lo & 0xFu;
  const uint32_t imm = static_cast<uint32_t>(SignExt13(lo));
  if (guard > 8) {
    fault = kFaultOperand;
    status = kFaulted;
    return false;
  }
  const uint32_t active = ActiveMask(*this, guard);
  bool jumped = false;
  bool halted = false;

  auto set_fault = [&](uint32_t value) { fault = value; };
  auto binary = [&](auto function) {
    WriteVector(*this, rd, active, [&](uint32_t lane) {
      return function(vregs[rs1][lane], vregs[rs2][lane]);
    });
  };
  auto binary_imm = [&](auto function) {
    WriteVector(*this, rd, active, [&](uint32_t lane) {
      return function(vregs[rs1][lane], imm);
    });
  };

  switch (op) {
    case kNop: break;
    case kMov:
      WriteVector(*this, rd, active,
                  [&](uint32_t lane) { return vregs[rs1][lane]; });
      break;
    case kAdd: binary([](uint32_t a, uint32_t b) { return a + b; }); break;
    case kSub: binary([](uint32_t a, uint32_t b) { return a - b; }); break;
    case kMul: binary([](uint32_t a, uint32_t b) { return a * b; }); break;
    case kDiv:
      binary([](uint32_t a, uint32_t b) { return b ? a / b : 0u; });
      break;
    case kMod:
      binary([](uint32_t a, uint32_t b) { return b ? a % b : 0u; });
      break;
    case kMin: binary([](uint32_t a, uint32_t b) { return std::min(a, b); }); break;
    case kMax: binary([](uint32_t a, uint32_t b) { return std::max(a, b); }); break;
    case kAnd: binary([](uint32_t a, uint32_t b) { return a & b; }); break;
    case kOr:  binary([](uint32_t a, uint32_t b) { return a | b; }); break;
    case kXor: binary([](uint32_t a, uint32_t b) { return a ^ b; }); break;
    case kShl:
      binary([](uint32_t a, uint32_t b) { return a << (b & 31u); });
      break;
    case kShr:
      binary([](uint32_t a, uint32_t b) { return a >> (b & 31u); });
      break;

    case kMovI:
      WriteVector(*this, rd, active, [&](uint32_t) { return imm; });
      break;
    case kAddI: binary_imm([](uint32_t a, uint32_t b) { return a + b; }); break;
    case kSubI: binary_imm([](uint32_t a, uint32_t b) { return a - b; }); break;
    case kMulI: binary_imm([](uint32_t a, uint32_t b) { return a * b; }); break;
    case kAndI: binary_imm([](uint32_t a, uint32_t b) { return a & b; }); break;
    case kOrI:  binary_imm([](uint32_t a, uint32_t b) { return a | b; }); break;
    case kXorI: binary_imm([](uint32_t a, uint32_t b) { return a ^ b; }); break;
    case kShlI:
      WriteVector(*this, rd, active, [&](uint32_t lane) {
        return vregs[rs1][lane] << (lo & 31u);
      });
      break;
    case kShrI:
      WriteVector(*this, rd, active, [&](uint32_t lane) {
        return vregs[rs1][lane] >> (lo & 31u);
      });
      break;
    case kAbs:
      WriteVector(*this, rd, active, [&](uint32_t lane) {
        const uint32_t value = vregs[rs1][lane];
        return static_cast<int32_t>(value) < 0 ? 0u - value : value;
      });
      break;
    case kNeg:
      WriteVector(*this, rd, active,
                  [&](uint32_t lane) { return 0u - vregs[rs1][lane]; });
      break;
    case kNot:
      WriteVector(*this, rd, active,
                  [&](uint32_t lane) { return ~vregs[rs1][lane]; });
      break;

    case kCmpEq: case kCmpNe: case kCmpLt: case kCmpLe: case kCmpGt:
    case kCmpGe: case kCmpEqI: case kCmpNeI: case kCmpLtI: case kCmpLeI:
    case kCmpGtI: case kCmpGeI: {
      if (BadPred(rd)) { set_fault(kFaultOperand); break; }
      uint32_t result = 0;
      for (uint32_t lane = 0; lane < kLanes; ++lane) {
        const uint32_t a = vregs[rs1][lane];
        const uint32_t b = op >= kCmpEqI ? imm : vregs[rs2][lane];
        bool condition = false;
        switch (op) {
          case kCmpEq: case kCmpEqI: condition = a == b; break;
          case kCmpNe: case kCmpNeI: condition = a != b; break;
          case kCmpLt: case kCmpLtI: condition = a < b; break;
          case kCmpLe: case kCmpLeI: condition = a <= b; break;
          case kCmpGt: case kCmpGtI: condition = a > b; break;
          default: condition = a >= b; break;
        }
        if (condition) result |= 1u << lane;
      }
      WritePred(*this, rd, result, active);
      break;
    }
    case kNotMask:
      if (BadPred(rd) || BadPred(rs1)) set_fault(kFaultOperand);
      else WritePred(*this, rd, ~preds[rs1], active);
      break;
    case kAndMask:
      if (BadPred(rd) || BadPred(rs1) || BadPred(rs2))
        set_fault(kFaultOperand);
      else WritePred(*this, rd, preds[rs1] & preds[rs2], active);
      break;
    case kOrMask:
      if (BadPred(rd) || BadPred(rs1) || BadPred(rs2))
        set_fault(kFaultOperand);
      else WritePred(*this, rd, preds[rs1] | preds[rs2], active);
      break;
    case kBallot: {
      if (BadPred(rd)) { set_fault(kFaultOperand); break; }
      uint32_t result = 0;
      for (uint32_t lane = 0; lane < kLanes; ++lane)
        if (vregs[rs1][lane]) result |= 1u << lane;
      WritePred(*this, rd, result, active);
      break;
    }
    case kAny:
      if (BadPred(rd) || BadPred(rs1)) set_fault(kFaultOperand);
      else WritePred(*this, rd, preds[rs1] ? kFullMask : 0u, active);
      break;
    case kAll:
      if (BadPred(rd) || BadPred(rs1)) set_fault(kFaultOperand);
      else WritePred(*this, rd,
                     preds[rs1] == kFullMask ? kFullMask : 0u, active);
      break;

    case kLaneId:
      WriteVector(*this, rd, active, [](uint32_t lane) { return lane; });
      break;
    case kBroadcast: {
      const uint32_t value = vregs[rs1][lo & 31u];
      WriteVector(*this, rd, active, [&](uint32_t) { return value; });
      break;
    }
    case kShuffle: {
      std::array<uint32_t, kLanes> values{};
      for (uint32_t lane = 0; lane < kLanes; ++lane)
        values[lane] = vregs[rs1][vregs[rs2][lane] & 31u];
      WriteVector(*this, rd, active,
                  [&](uint32_t lane) { return values[lane]; });
      break;
    }
    case kShuffleXor: {
      std::array<uint32_t, kLanes> values{};
      for (uint32_t lane = 0; lane < kLanes; ++lane)
        values[lane] = vregs[rs1][lane ^ (lo & 31u)];
      WriteVector(*this, rd, active,
                  [&](uint32_t lane) { return values[lane]; });
      break;
    }
    case kReduceAdd: case kReduceMin: case kReduceMax: case kReduceAnd:
    case kReduceOr: case kReduceXor: {
      uint32_t result = (op == kReduceAnd) ? kFullMask : vregs[rs1][0];
      if (op == kReduceAdd || op == kReduceOr || op == kReduceXor) result = 0;
      for (uint32_t lane = 0; lane < kLanes; ++lane) {
        const uint32_t value = vregs[rs1][lane];
        if (op == kReduceAdd) result += value;
        else if (op == kReduceMin) result = std::min(result, value);
        else if (op == kReduceMax) result = std::max(result, value);
        else if (op == kReduceAnd) result &= value;
        else if (op == kReduceOr) result |= value;
        else result ^= value;
      }
      WriteVector(*this, rd, active, [&](uint32_t) { return result; });
      break;
    }
    case kVmid:
      WriteVector(*this, rd, active, [&](uint32_t) { return vm_id; });
      break;
    case kClock: {
      const uint32_t tick = static_cast<uint32_t>(
          std::chrono::steady_clock::now().time_since_epoch().count());
      WriteVector(*this, rd, active, [&](uint32_t) { return tick; });
      break;
    }
    case kRand:
      rng_state = Xorshift32(rng_state);
      WriteVector(*this, rd, active, [&](uint32_t lane) {
        return Xorshift32((rng_state ^ (lane * 0x9E3779B9u)) | 1u);
      });
      break;

    case kLoad:
      for (uint32_t lane = 0; lane < kLanes && fault == kFaultOk; ++lane) {
        if (((active >> lane) & 1u) == 0) continue;
        const uint32_t address = vregs[rs1][lane];
        if (address < memory.size()) vregs[rd][lane] = memory[address];
        else if (address >= kVideoBaseWord && address < kVideoEndWord)
          vregs[rd][lane] = framebuffer[address - kVideoBaseWord];
        else set_fault(kFaultMem);
      }
      break;
    case kStore:
      for (uint32_t lane = 0; lane < kLanes && fault == kFaultOk; ++lane) {
        if (((active >> lane) & 1u) == 0) continue;
        const uint32_t address = vregs[rd][lane];
        if (address < memory.size()) memory[address] = vregs[rs1][lane];
        else if (address >= kVideoBaseWord && address < kVideoEndWord)
          framebuffer[address - kVideoBaseWord] = vregs[rs1][lane];
        else set_fault(kFaultMem);
      }
      break;
    case kLdw:
      if (lo >= literals.size()) set_fault(kFaultOperand);
      else WriteVector(*this, rd, active,
                       [&](uint32_t) { return literals[lo]; });
      break;
    case kLog: case kLogI:
      break;
    case kFlip:
      if (guard) set_fault(kFaultOperand);
      else ++frame_seq;
      break;
    case kSend: case kTryRecv:
      set_fault(kFaultMsg);  // Messaging is outside the v0.1.2 CPU scope.
      break;

    // Scalar state is architecturally uniform. Partial scalar guards are
    // underspecified (notes.md); lane 0 selects the spilled logical value.
    case kSMov:
      if (BadScalar(rd) || BadScalar(rs1)) set_fault(kFaultOperand);
      else if (active & 1u) sregs[rd] = sregs[rs1];
      break;
    case kSMovI:
      if (BadScalar(rd)) set_fault(kFaultOperand);
      else if (active & 1u) sregs[rd] = imm;
      break;
    case kSAdd:
      if (BadScalar(rd) || BadScalar(rs1) || BadScalar(rs2))
        set_fault(kFaultOperand);
      else if (active & 1u) sregs[rd] = sregs[rs1] + sregs[rs2];
      break;
    case kSAddI:
      if (BadScalar(rd) || BadScalar(rs1)) set_fault(kFaultOperand);
      else if (active & 1u) sregs[rd] = sregs[rs1] + imm;
      break;
    case kSLdw:
      if (BadScalar(rd) || lo >= literals.size()) set_fault(kFaultOperand);
      else if (active & 1u) sregs[rd] = literals[lo];
      break;
    case kSCmpLt: case kSCmpLtI:
      if (BadPred(rd) || BadScalar(rs1) ||
          (op == kSCmpLt && BadScalar(rs2))) {
        set_fault(kFaultOperand);
      } else {
        const uint32_t rhs = op == kSCmpLtI ? imm : sregs[rs2];
        WritePred(*this, rd, sregs[rs1] < rhs ? kFullMask : 0u, active);
      }
      break;
    case kSBcast:
      if (BadScalar(rs1)) set_fault(kFaultOperand);
      else WriteVector(*this, rd, active,
                       [&](uint32_t) { return sregs[rs1]; });
      break;
    case kSGet:
      if (BadScalar(rd)) set_fault(kFaultOperand);
      else if (active & 1u) sregs[rd] = vregs[rs1][0];
      break;

    case kJmp:
      if (lo >= code.size()) set_fault(kFaultJump);
      else { pc = lo; jumped = true; }
      break;
    case kJmpIfAny:
      if (active != 0) {
        if (lo >= code.size()) set_fault(kFaultJump);
        else { pc = lo; jumped = true; }
      }
      break;
    case kJmpIfAll:
      if (active == kFullMask) {
        if (lo >= code.size()) set_fault(kFaultJump);
        else { pc = lo; jumped = true; }
      }
      break;
    case kCall:
      if (lo >= code.size()) set_fault(kFaultJump);
      else if (call_depth >= kCallDepth) set_fault(kFaultStack);
      else {
        call_stack[call_depth++] = pc + 1;
        pc = lo;
        jumped = true;
      }
      break;
    case kRet:
      if (call_depth == 0) set_fault(kFaultStack);
      else { pc = call_stack[--call_depth]; jumped = true; }
      break;
    case kHalt: halted = true; break;
    case kYield: case kStepTrap: break;
    default: set_fault(kFaultOpcode); break;
  }

  if (fault != kFaultOk) {
    status = kFaulted;
    return false;
  }
  ++instruction_counter;
  if (halted) {
    status = kHalted;
    return true;
  }
  if (!jumped) ++pc;
  return true;
}

uint64_t CpuVm::RunQuantum(uint64_t budget) {
  const uint64_t before = instruction_counter;
  while (status == kRunning && instruction_counter - before < budget) Step();
  return instruction_counter - before;
}

int RunCpuInterpreterTests() {
  bool execution = true;
  WvmFile file;
  file.literals = {kVideoBaseWord};
  file.code = {
      enc_r(kLaneId, 0, 0, 0, 0),
      enc_i(kAddI, 0, 1, 0, 1),
      enc_i(kCmpLtI, 0, 0, 0, 16),
      enc_i(kMovI, 0, 2, 0, 0),
      enc_i(kAddI, 1, 2, 1, 10),
      enc_i(kAddI, 5, 2, 1, 100),
      enc_r(kBallot, 0, 1, 2, 0),
      enc_r(kReduceAdd, 0, 3, 0, 0),
      enc_i(kSMovI, 0, 0, 0, 5),
      enc_r(kSBcast, 0, 4, 0, 0),
      enc_r(kSGet, 0, 1, 0, 0),
      enc_r(kStore, 0, 0, 2, 0),
      enc_r(kLoad, 0, 5, 0, 0),
      enc_i(kLdw, 0, 6, 0, 0),
      enc_r(kAdd, 0, 6, 6, 0),
      enc_r(kStore, 0, 6, 2, 0),
      enc_r(kFlip, 0, 0, 0, 0),
      enc_r(kHalt, 0, 0, 0, 0),
  };
  CpuVm vm;
  vm.Init(7, file, 64);
  vm.RunQuantum(1000);
  execution &= vm.status == kHalted && vm.fault == kFaultOk;
  execution &= vm.preds[0] == 0x0000FFFFu && vm.preds[1] == kFullMask;
  execution &= vm.sregs[0] == 5 && vm.sregs[1] == 0;
  execution &= vm.frame_seq == 1;
  for (uint32_t lane = 0; lane < kLanes; ++lane) {
    const uint32_t expected = lane < 16 ? lane + 11 : lane + 101;
    execution &= vm.vregs[2][lane] == expected;
    execution &= vm.vregs[3][lane] == 496;
    execution &= vm.vregs[4][lane] == 5;
    execution &= vm.vregs[5][lane] == expected;
    execution &= vm.memory[lane] == expected;
    execution &= vm.framebuffer[lane] == expected;
  }

  bool control = true;
  WvmFile control_file;
  control_file.code = {
      enc_i(kSMovI, 0, 0, 0, 0),       // 0
      enc_i(kSAddI, 0, 0, 0, 1),       // 1 loop
      enc_i(kSCmpLtI, 0, 0, 0, 3),     // 2
      enc_i(kJmpIfAny, 1, 0, 0, 1),    // 3
      enc_i(kCall, 0, 0, 0, 6),        // 4
      enc_r(kHalt, 0, 0, 0, 0),        // 5
      enc_i(kSAddI, 0, 0, 0, 10),      // 6 subroutine
      enc_r(kRet, 0, 0, 0, 0),         // 7
  };
  CpuVm control_vm;
  control_vm.Init(0, control_file, 8);
  control_vm.RunQuantum(1000);
  control &= control_vm.status == kHalted && control_vm.sregs[0] == 13;
  control &= control_vm.call_depth == 0 && control_vm.pc == 5;

  WvmFile mem_fault_file;
  mem_fault_file.code = {enc_i(kMovI, 0, 0, 0, 100),
                         enc_r(kLoad, 0, 1, 0, 0),
                         enc_r(kHalt, 0, 0, 0, 0)};
  CpuVm mem_fault_vm;
  mem_fault_vm.Init(0, mem_fault_file, 8);
  mem_fault_vm.RunQuantum(100);
  const bool mem_fault = mem_fault_vm.status == kFaulted &&
                         mem_fault_vm.fault == kFaultMem &&
                         mem_fault_vm.instruction_counter == 1;

  WvmFile stack_fault_file;
  stack_fault_file.code = {enc_r(kRet, 0, 0, 0, 0)};
  CpuVm stack_fault_vm;
  stack_fault_vm.Init(0, stack_fault_file, 8);
  stack_fault_vm.RunQuantum(10);
  const bool stack_fault = stack_fault_vm.status == kFaulted &&
                           stack_fault_vm.fault == kFaultStack;

  std::printf("cpu_interpreter: vector_mask_memory %s\n",
              execution ? "PASS" : "FAIL");
  std::printf("cpu_interpreter: scalar_control     %s\n",
              control ? "PASS" : "FAIL");
  std::printf("cpu_interpreter: memory_fault       %s\n",
              mem_fault ? "PASS" : "FAIL");
  std::printf("cpu_interpreter: stack_fault        %s\n",
              stack_fault ? "PASS" : "FAIL");
  const bool pass = execution && control && mem_fault && stack_fault;
  std::printf("cpu_interpreter: %s\n", pass ? "PASS" : "FAIL");
  return pass ? 0 : 1;
}

}  // namespace wvm
