// Logical host implementation of the WarpVM v0.1 machine.
//
// Executes the same .wvm words as the CUDA interpreter while representing the
// 32 lanes as ordinary arrays. No bytecode specialization or application-
// specific fusion belongs in this baseline interpreter.
#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "gpu/warpvm.cuh"
#include "host/wvm_file.h"

namespace wvm {

constexpr uint32_t kCpuVmQuantum = 4096;

struct CpuVm {
  uint32_t vm_id = 0;
  uint32_t status = kIdle;
  uint32_t pc = 0;
  uint32_t fault = kFaultOk;
  uint64_t instruction_counter = 0;
  uint32_t frame_seq = 0;
  uint32_t rng_state = 0;

  std::array<std::array<uint32_t, kLanes>, kVectorRegs> vregs{};
  std::array<uint32_t, kScalarRegs> sregs{};
  std::array<uint32_t, kPredRegs> preds{};
  std::array<uint32_t, kCallDepth> call_stack{};
  uint32_t call_depth = 0;

  std::vector<uint32_t> code;
  std::vector<uint32_t> literals;
  std::vector<uint32_t> memory;
  std::vector<uint32_t> framebuffer;

  void Init(uint32_t id, const WvmFile& file, uint32_t memory_words = 16384);
  void Reset();
  bool Step();  // true iff an instruction retired
  uint64_t RunQuantum(uint64_t budget = kCpuVmQuantum);
};

int RunCpuInterpreterTests();

}  // namespace wvm
