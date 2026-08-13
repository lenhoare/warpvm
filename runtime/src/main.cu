// WarpVM host entry point.
//
// Commands:
//   slice1          slice-1 warp arithmetic demo (direct CUDA)
//   slice2          slice-2 interpreter self-tests (embedded programs)
//   run <file.wvm>  load a .wvm program, run it on one warp, print state

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <cuda_runtime.h>

#include "gpu/vm_state.cuh"
#include "gpu/warpvm.cuh"
#include "host/disasm.h"
#include "host/cpu_interpreter.h"
#include "host/persistent.h"
#include "host/ptx_compiler.h"
#include "host/supervisor.h"
#include "host/supervisor_cli.h"
#include "host/vm_image.h"
#include "host/wvm_file.h"

namespace wvm {
__global__ void Slice1Kernel(uint32_t* lane_out, uint32_t* sum_out);
__global__ void VmArrayKernel(const VmDesc* descs, VmState* states);
int ViewSingleVm(const char* path, uint32_t vm_index,
                 bool compiled);  // host/view_sdl.cu
int ViewVmGrid(const char* path, uint32_t n_vms,
               bool compiled);  // host/view_sdl.cu
int ViewHeterogeneousGrid(
    const std::vector<std::string>& paths);  // host/view_sdl.cu
int RunLifeBenchmark(const char* path,
                     const std::vector<uint32_t>& vm_counts,
                     int duration_ms,
                     uint32_t cpu_workers);             // host/life_bench.cu
int RunLifeCensus(const char* path);                     // host/life_bench.cu
int RunLifeProfile(const char* path, int duration_ms);  // host/life_bench.cu
int RunCpuGpuLifeEquivalence(const char* path);         // host/life_bench.cu
int RunNativeCpuLifeEquivalence(const char* path);      // host/life_bench.cu
}  // namespace wvm

namespace {

#define CUDA_CHECK(expr)                                                    \
  do {                                                                      \
    cudaError_t err_ = (expr);                                              \
    if (err_ != cudaSuccess) {                                              \
      std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(err_));                               \
      std::exit(1);                                                         \
    }                                                                       \
  } while (0)

// ---- slice 1 ---------------------------------------------------------------

__global__ void Slice1Kernel(uint32_t* lane_out, uint32_t* sum_out) {
  const uint32_t lane = threadIdx.x & (wvm::kLanes - 1);
  const uint32_t x = lane;
  const uint32_t y = x * x + 3u * x + 7u;
  lane_out[lane] = y;

  uint32_t s = y;
#pragma unroll
  for (int off = wvm::kLanes / 2; off > 0; off >>= 1)
    s += __shfl_down_sync(wvm::kFullMask, s, off);
  if (lane == 0) *sum_out = s;
}

int RunSlice1() {
  uint32_t* d_lane = nullptr;
  uint32_t* d_sum = nullptr;
  CUDA_CHECK(cudaMalloc(&d_lane, wvm::kLanes * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&d_sum, sizeof(uint32_t)));

  Slice1Kernel<<<1, wvm::kLanes>>>(d_lane, d_sum);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<uint32_t> lane(wvm::kLanes);
  uint32_t sum = 0;
  CUDA_CHECK(cudaMemcpy(lane.data(), d_lane, wvm::kLanes * sizeof(uint32_t),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&sum, d_sum, sizeof(uint32_t), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_lane));
  CUDA_CHECK(cudaFree(d_sum));

  bool ok = true;
  uint32_t expect_sum = 0;
  for (uint32_t i = 0; i < wvm::kLanes; ++i) {
    const uint32_t expect = i * i + 3u * i + 7u;
    expect_sum += expect;
    if (lane[i] != expect) {
      std::printf("MISMATCH lane %u: got %u want %u\n", i, lane[i], expect);
      ok = false;
    }
  }
  if (sum != expect_sum) {
    std::printf("MISMATCH sum: got %u want %u\n", sum, expect_sum);
    ok = false;
  }
  std::printf("warp sum = %u (expect %u)\n", sum, expect_sum);
  std::printf(ok ? "slice1: PASS\n" : "slice1: FAIL\n");
  return ok ? 0 : 1;
}

// ---- VM execution ----------------------------------------------------------

bool ExecVmArray(const std::vector<wvm::VmImage>& images,
                 std::vector<wvm::VmState>& states,
                 std::vector<std::vector<uint32_t>>& mem_out) {
  const size_t n = images.size();
  std::vector<uint32_t*> d_code(n, nullptr), d_lit(n, nullptr), d_mem(n,
                                                                      nullptr),
      d_fb(n, nullptr);
  std::vector<wvm::VmDesc> descs(n);

  for (size_t i = 0; i < n; ++i) {
    const wvm::VmImage& img = images[i];
    CUDA_CHECK(cudaMalloc(&d_code[i], img.code.size() * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(d_code[i], img.code.data(),
                          img.code.size() * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));
    if (!img.literals.empty()) {
      CUDA_CHECK(
          cudaMalloc(&d_lit[i], img.literals.size() * sizeof(uint32_t)));
      CUDA_CHECK(cudaMemcpy(d_lit[i], img.literals.data(),
                            img.literals.size() * sizeof(uint32_t),
                            cudaMemcpyHostToDevice));
    }
    if (img.mem_size_words > 0) {
      CUDA_CHECK(cudaMalloc(&d_mem[i],
                            img.mem_size_words * sizeof(uint32_t)));
      CUDA_CHECK(cudaMemset(d_mem[i], 0,
                            img.mem_size_words * sizeof(uint32_t)));
      const size_t seed_words =
          std::min<size_t>(img.mem_init.size(), img.mem_size_words);
      if (seed_words > 0)
        CUDA_CHECK(cudaMemcpy(d_mem[i], img.mem_init.data(),
                              seed_words * sizeof(uint32_t),
                              cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMalloc(&d_fb[i], wvm::kVideoWords * sizeof(uint32_t)));
    descs[i] = wvm::VmDesc{d_code[i],
                           static_cast<uint32_t>(img.code.size()),
                           d_lit[i],
                           static_cast<uint32_t>(img.literals.size()),
                           d_mem[i],
                           img.mem_size_words,
                           d_fb[i]};
  }

  wvm::VmDesc* d_descs = nullptr;
  wvm::VmState* d_states = nullptr;
  CUDA_CHECK(cudaMalloc(&d_descs, n * sizeof(wvm::VmDesc)));
  CUDA_CHECK(cudaMemcpy(d_descs, descs.data(), n * sizeof(wvm::VmDesc),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMalloc(&d_states, n * sizeof(wvm::VmState)));
  CUDA_CHECK(cudaMemset(d_states, 0, n * sizeof(wvm::VmState)));

  // Warp-aligned geometry: blocks of 8 warps where the VM count allows,
  // otherwise one warp per block. A VM never straddles a hardware warp.
  const int block = (n % 8 == 0) ? 256 : 32;
  const int grid = static_cast<int>(n * wvm::kLanes) / block;
  wvm::VmArrayKernel<<<grid, block>>>(d_descs, d_states);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  states.resize(n);
  CUDA_CHECK(cudaMemcpy(states.data(), d_states, n * sizeof(wvm::VmState),
                        cudaMemcpyDeviceToHost));
  mem_out.resize(n);
  for (size_t i = 0; i < n; ++i) {
    mem_out[i].resize(images[i].mem_size_words);
    if (images[i].mem_size_words > 0)
      CUDA_CHECK(cudaMemcpy(mem_out[i].data(), d_mem[i],
                            images[i].mem_size_words * sizeof(uint32_t),
                            cudaMemcpyDeviceToHost));
  }

  CUDA_CHECK(cudaFree(d_descs));
  CUDA_CHECK(cudaFree(d_states));
  for (size_t i = 0; i < n; ++i) {
    CUDA_CHECK(cudaFree(d_code[i]));
    CUDA_CHECK(cudaFree(d_lit[i]));
    CUDA_CHECK(cudaFree(d_mem[i]));
    CUDA_CHECK(cudaFree(d_fb[i]));
  }
  return true;
}

// Single-VM convenience wrapper used by `run` and the slice-2 self-tests.
bool ExecProgram(const std::vector<uint32_t>& code,
                 const std::vector<uint32_t>& literals, wvm::VmState& out) {
  wvm::VmImage img;
  img.code = code;
  img.literals = literals;
  img.mem_size_words = wvm::kRamSizeWords;
  std::vector<wvm::VmState> states;
  std::vector<std::vector<uint32_t>> mem;
  if (!ExecVmArray({img}, states, mem)) return false;
  out = states[0];
  return true;
}

wvm::VmState StateFromCpu(const wvm::CpuVm& cpu) {
  wvm::VmState state{};
  state.vm_id = cpu.vm_id;
  state.status = cpu.status;
  state.pc = cpu.pc;
  state.fault_code = cpu.fault;
  state.instruction_counter = cpu.instruction_counter;
  for (uint32_t reg = 0; reg < wvm::kVectorRegs; ++reg)
    for (uint32_t lane = 0; lane < wvm::kLanes; ++lane)
      state.vregs[reg * wvm::kLanes + lane] = cpu.vregs[reg][lane];
  std::copy(cpu.sregs.begin(), cpu.sregs.end(), state.sregs);
  std::copy(cpu.preds.begin(), cpu.preds.end(), state.preds);
  std::copy(cpu.call_stack.begin(), cpu.call_stack.end(), state.call_stack);
  state.call_depth = cpu.call_depth;
  state.rng_state = cpu.rng_state;
  return state;
}

bool SameArchitecturalState(const wvm::VmState& left,
                            const wvm::VmState& right,
                            std::string& difference) {
  auto mismatch = [&](const std::string& field) {
    difference = field;
    return false;
  };
  if (left.vm_id != right.vm_id) return mismatch("vm_id");
  if (left.status != right.status) return mismatch("status");
  if (left.pc != right.pc) return mismatch("pc");
  if (left.fault_code != right.fault_code) return mismatch("fault_code");
  if (left.instruction_counter != right.instruction_counter)
    return mismatch("instruction_counter");
  for (uint32_t reg = 0; reg < wvm::kVectorRegs; ++reg)
    for (uint32_t lane = 0; lane < wvm::kLanes; ++lane)
      if (left.vregs[reg * wvm::kLanes + lane] !=
          right.vregs[reg * wvm::kLanes + lane])
        return mismatch("r" + std::to_string(reg) + " lane " +
                        std::to_string(lane));
  for (uint32_t reg = 0; reg < wvm::kScalarRegs; ++reg)
    if (left.sregs[reg] != right.sregs[reg])
      return mismatch("s" + std::to_string(reg));
  for (uint32_t reg = 0; reg < wvm::kPredRegs; ++reg)
    if (left.preds[reg] != right.preds[reg])
      return mismatch("p" + std::to_string(reg));
  for (uint32_t index = 0; index < wvm::kCallDepth; ++index)
    if (left.call_stack[index] != right.call_stack[index])
      return mismatch("call_stack[" + std::to_string(index) + "]");
  if (left.call_depth != right.call_depth) return mismatch("call_depth");
  if (left.rng_state != right.rng_state) return mismatch("rng_state");
  return true;
}

bool RunCpuToNextYield(wvm::CpuVm& vm) {
  const uint32_t target_frame = vm.frame_seq + 1;
  uint64_t steps = 0;
  while (vm.status == wvm::kRunning && vm.frame_seq < target_frame &&
         steps < 1000000) {
    vm.Step();
    ++steps;
  }
  if (vm.status != wvm::kRunning || vm.frame_seq != target_frame ||
      vm.pc >= vm.code.size())
    return false;
  const uint32_t op =
      (vm.code[vm.pc] >> wvm::kOpcodeShift) & wvm::kOpcodeMask;
  return op == wvm::kYield && vm.Step();
}

void RestoreCpuAtCheckpoint(wvm::CpuVm& cpu, const wvm::VmState& state,
                            const uint32_t* memory,
                            const uint32_t* framebuffer,
                            uint32_t frame_seq) {
  cpu.vm_id = state.vm_id;
  cpu.status = wvm::kRunning;
  cpu.pc = state.pc;
  cpu.fault = state.fault_code;
  cpu.instruction_counter = state.instruction_counter;
  for (uint32_t reg = 0; reg < wvm::kVectorRegs; ++reg)
    for (uint32_t lane = 0; lane < wvm::kLanes; ++lane)
      cpu.vregs[reg][lane] =
          state.vregs[reg * wvm::kLanes + lane];
  std::copy_n(state.sregs, wvm::kScalarRegs, cpu.sregs.begin());
  std::copy_n(state.preds, wvm::kPredRegs, cpu.preds.begin());
  std::copy_n(state.call_stack, wvm::kCallDepth, cpu.call_stack.begin());
  cpu.call_depth = state.call_depth;
  cpu.rng_state = state.rng_state;
  std::copy_n(memory, cpu.memory.size(), cpu.memory.begin());
  std::copy_n(framebuffer, cpu.framebuffer.size(), cpu.framebuffer.begin());
  cpu.frame_seq = frame_seq;
}

int RunCompiledSlice1() {
  bool ok = true;
  std::string err;
  std::string difference;

  // Fresh power-on state: compare the exact state produced by the reference
  // CPU bytecode interpreter with the PTX-compiled form of the same program.
  wvm::WvmFile arithmetic;
  arithmetic.code = {
      wvm::enc_r(wvm::kLaneId, 0, 0, 0, 0),
      wvm::enc_i(wvm::kMovI, 0, 1, 0, 7),
      wvm::enc_r(wvm::kAdd, 0, 2, 0, 1),
      wvm::enc_i(wvm::kMulI, 0, 3, 2, 3),
      wvm::enc_i(wvm::kXorI, 0, 4, 3, -1),
      wvm::enc_i(wvm::kMovI, 0, 5, 0, -21),
      wvm::enc_r(wvm::kAbs, 0, 6, 5, 0),
      wvm::enc_r(wvm::kDiv, 0, 7, 6, 1),
      wvm::enc_r(wvm::kMod, 0, 8, 6, 1),
      wvm::enc_r(wvm::kHalt, 0, 0, 0, 0),
  };
  wvm::CpuVm arithmetic_cpu;
  arithmetic_cpu.Init(0, arithmetic);
  arithmetic_cpu.RunQuantum();
  const wvm::VmState interpreted = StateFromCpu(arithmetic_cpu);

  wvm::PtxCompiledProgram compiled;
  if (!compiled.Compile(arithmetic, err)) {
    std::printf("compiled slice 1: PTX compile FAIL: %s\n", err.c_str());
    return 1;
  }
  std::vector<wvm::VmState> compiled_states(1);
  compiled_states[0].status = wvm::kRunning;
  compiled_states[0].rng_state = 0x1234567u;
  if (!compiled.Launch(compiled_states, err)) {
    std::printf("compiled slice 1: launch FAIL: %s\n", err.c_str());
    return 1;
  }
  const bool fresh_equal =
      SameArchitecturalState(interpreted, compiled_states[0], difference);
  std::printf("compiled slice 1: CPU interpreted/compiled state %s",
              fresh_equal ? "PASS" : "FAIL");
  if (!fresh_equal) std::printf(" (%s)", difference.c_str());
  std::printf("\n");
  ok &= fresh_equal;
  std::printf("compiled slice 1: PTX %zu bytes, JIT %.3f ms\n",
              compiled.ptx().size(), compiled.jit_milliseconds());

  // Seeded canonical state: prove that compiled execution consumes existing
  // VM state rather than assuming every architectural register starts zero.
  wvm::WvmFile resumed;
  resumed.code = {
      wvm::enc_i(wvm::kAddI, 0, 5, 5, 9),
      wvm::enc_i(wvm::kXorI, 0, 6, 5, 0x55),
      wvm::enc_r(wvm::kHalt, 0, 0, 0, 0),
  };
  wvm::CpuVm cpu;
  cpu.Init(0, resumed);
  for (uint32_t lane = 0; lane < wvm::kLanes; ++lane)
    cpu.vregs[5][lane] = 1000u + lane * 17u;
  cpu.RunQuantum();
  const wvm::VmState cpu_state = StateFromCpu(cpu);

  if (!compiled.Compile(resumed, err)) {
    std::printf("compiled slice 1: seeded PTX compile FAIL: %s\n",
                err.c_str());
    return 1;
  }
  std::vector<wvm::VmState> resumed_states(1);
  resumed_states[0].status = wvm::kRunning;
  resumed_states[0].rng_state = 0x1234567u;
  for (uint32_t lane = 0; lane < wvm::kLanes; ++lane)
    resumed_states[0].vregs[5 * wvm::kLanes + lane] =
        1000u + lane * 17u;
  if (!compiled.Launch(resumed_states, err)) {
    std::printf("compiled slice 1: seeded launch FAIL: %s\n", err.c_str());
    return 1;
  }
  difference.clear();
  const bool seeded_equal =
      SameArchitecturalState(cpu_state, resumed_states[0], difference);
  std::printf("compiled slice 1: seeded CPU/compiled state %s",
              seeded_equal ? "PASS" : "FAIL");
  if (!seeded_equal) std::printf(" (%s)", difference.c_str());
  std::printf("\n");
  ok &= seeded_equal;

  // A backward bytecode branch must become native control flow while keeping
  // WarpVM's warp-wide ANY semantics and exact dynamic retirement count.
  wvm::WvmFile control;
  control.code = {
      wvm::enc_r(wvm::kLaneId, 0, 0, 0, 0),
      wvm::enc_i(wvm::kMovI, 0, 1, 0, 0),
      wvm::enc_i(wvm::kAddI, 0, 1, 1, 1),
      wvm::enc_r(wvm::kCmpLt, 0, 0, 1, 0),
      wvm::enc_i(wvm::kJmpIfAny, 1, 0, 0, 2),
      wvm::enc_r(wvm::kHalt, 0, 0, 0, 0),
  };
  wvm::CpuVm control_cpu;
  control_cpu.Init(0, control);
  control_cpu.RunQuantum();
  const wvm::VmState control_reference = StateFromCpu(control_cpu);
  if (!compiled.Compile(control, err)) {
    std::printf("compiled slice 2: control-flow PTX compile FAIL: %s\n",
                err.c_str());
    return 1;
  }
  std::vector<wvm::VmState> control_states(1);
  control_states[0].status = wvm::kRunning;
  control_states[0].rng_state = 0x1234567u;
  if (!compiled.Launch(control_states, err)) {
    std::printf("compiled slice 2: control-flow launch FAIL: %s\n",
                err.c_str());
    return 1;
  }
  difference.clear();
  const bool control_equal =
      SameArchitecturalState(control_reference, control_states[0], difference);
  std::printf("compiled slice 2: loop/predicate state %s",
              control_equal ? "PASS" : "FAIL");
  if (!control_equal) std::printf(" (%s)", difference.c_str());
  std::printf(" (instrs=%llu)\n",
              static_cast<unsigned long long>(
                  control_states[0].instruction_counter));
  ok &= control_equal;

  // Nested CALL/RET uses the architectural eight-entry continuation stack.
  // Compare the retained (popped) stack words as well as registers and depth.
  wvm::WvmFile calls;
  calls.code = {
      wvm::enc_i(wvm::kMovI, 0, 0, 0, 5),
      wvm::enc_i(wvm::kCall, 0, 0, 0, 4),
      wvm::enc_i(wvm::kAddI, 0, 0, 0, 1),
      wvm::enc_r(wvm::kHalt, 0, 0, 0, 0),
      wvm::enc_i(wvm::kCall, 0, 0, 0, 7),
      wvm::enc_i(wvm::kAddI, 0, 0, 0, 2),
      wvm::enc_r(wvm::kRet, 0, 0, 0, 0),
      wvm::enc_i(wvm::kMulI, 0, 0, 0, 3),
      wvm::enc_r(wvm::kRet, 0, 0, 0, 0),
  };
  wvm::CpuVm calls_cpu;
  calls_cpu.Init(0, calls);
  calls_cpu.RunQuantum();
  if (!compiled.Compile(calls, err)) {
    std::printf("compiled slice C: CALL/RET PTX compile FAIL: %s\n",
                err.c_str());
    return 1;
  }
  std::vector<wvm::VmState> call_states(1);
  call_states[0].status = wvm::kRunning;
  call_states[0].rng_state = 0x1234567u;
  if (!compiled.Launch(call_states, err)) {
    std::printf("compiled slice C: CALL/RET launch FAIL: %s\n", err.c_str());
    return 1;
  }
  difference.clear();
  const bool calls_equal = SameArchitecturalState(
      StateFromCpu(calls_cpu), call_states[0], difference);
  std::printf("compiled slice C: nested CALL/RET state %s (r0=%u depth=%u)",
              calls_equal ? "PASS" : "FAIL", call_states[0].vregs[0],
              call_states[0].call_depth);
  if (!calls_equal) std::printf(" (%s)", difference.c_str());
  std::printf("\n");
  ok &= calls_equal;

  auto check_compiled_stack_fault = [&](const char* name,
                                        const wvm::WvmFile& program) {
    wvm::CpuVm reference;
    reference.Init(0, program);
    reference.RunQuantum();
    if (!compiled.Compile(program, err)) {
      std::printf("compiled slice C: %s compile FAIL: %s\n", name,
                  err.c_str());
      return false;
    }
    std::vector<wvm::VmState> states(1);
    states[0].status = wvm::kRunning;
    states[0].rng_state = 0x1234567u;
    if (!compiled.Launch(states, err)) {
      std::printf("compiled slice C: %s launch FAIL: %s\n", name,
                  err.c_str());
      return false;
    }
    difference.clear();
    const bool equal = SameArchitecturalState(
        StateFromCpu(reference), states[0], difference);
    std::printf("compiled slice C: %s %s", name,
                equal ? "PASS" : "FAIL");
    if (!equal) std::printf(" (%s)", difference.c_str());
    std::printf("\n");
    return equal;
  };
  wvm::WvmFile ret_underflow;
  ret_underflow.code = {
      wvm::enc_r(wvm::kRet, 0, 0, 0, 0),
      wvm::enc_r(wvm::kHalt, 0, 0, 0, 0),
  };
  ok &= check_compiled_stack_fault("RET underflow", ret_underflow);
  wvm::WvmFile call_overflow;
  call_overflow.code = {
      wvm::enc_i(wvm::kCall, 0, 0, 0, 0),
      wvm::enc_r(wvm::kHalt, 0, 0, 0, 0),
  };
  ok &= check_compiled_stack_fault("CALL overflow", call_overflow);

  // A compact loop32 form exercises scalar loop control, guarded vector
  // execution, private RAM, and a backward loop.
  wvm::WvmFile loop32;
  loop32.code = {
      wvm::enc_i(wvm::kSMovI, 0, 0, 0, 0),
      wvm::enc_r(wvm::kLaneId, 0, 0, 0, 0),
      wvm::enc_r(wvm::kSBcast, 0, 1, 0, 0),
      wvm::enc_r(wvm::kAdd, 0, 2, 0, 1),
      wvm::enc_i(wvm::kCmpLtI, 0, 0, 2, 100),
      wvm::enc_r(wvm::kStore, 1, 2, 2, 0),
      wvm::enc_i(wvm::kSAddI, 0, 0, 0, 32),
      wvm::enc_i(wvm::kSCmpLtI, 0, 1, 0, 100),
      wvm::enc_i(wvm::kJmpIfAny, 2, 0, 0, 1),
      wvm::enc_i(wvm::kMovI, 0, 4, 0, 10),
      wvm::enc_r(wvm::kLoad, 0, 7, 4, 0),
      wvm::enc_r(wvm::kHalt, 0, 0, 0, 0),
  };
  wvm::CpuVm loop_cpu;
  loop_cpu.Init(0, loop32, wvm::kRamSizeWords);
  loop_cpu.RunQuantum();
  const wvm::VmState loop_reference = StateFromCpu(loop_cpu);
  if (!compiled.Compile(loop32, err)) {
    std::printf("compiled slice 3: loop32 PTX compile FAIL: %s\n",
                err.c_str());
    return 1;
  }
  std::vector<wvm::VmState> loop_states(1);
  loop_states[0].status = wvm::kRunning;
  loop_states[0].rng_state = 0x1234567u;
  std::vector<uint32_t> loop_memory(wvm::kRamSizeWords, 0);
  std::vector<uint32_t> no_framebuffer;
  std::vector<uint32_t> no_frame_seq;
  if (!compiled.Launch(loop_states, loop_memory, wvm::kRamSizeWords, no_framebuffer,
                       no_frame_seq, err)) {
    std::printf("compiled slice 3: loop32 launch FAIL: %s\n", err.c_str());
    return 1;
  }
  difference.clear();
  const bool loop_state_equal =
      SameArchitecturalState(loop_reference, loop_states[0], difference);
  const bool loop_memory_equal = loop_memory == loop_cpu.memory;
  std::printf("compiled slice 3: loop32 state/memory %s",
              loop_state_equal && loop_memory_equal ? "PASS" : "FAIL");
  if (!loop_state_equal) std::printf(" (state: %s)", difference.c_str());
  if (!loop_memory_equal) std::printf(" (memory differs)");
  std::printf("\n");
  ok &= loop_state_equal && loop_memory_equal;

  // Invalid per-lane addresses fault the whole VM at the same bytecode PC;
  // the faulting memory instruction is not counted as retired.
  wvm::WvmFile memory_fault;
  memory_fault.code = {
      wvm::enc_i(wvm::kMovI, 0, 0, 0, 1000),
      wvm::enc_r(wvm::kLoad, 0, 1, 0, 0),
      wvm::enc_r(wvm::kHalt, 0, 0, 0, 0),
  };
  wvm::CpuVm fault_cpu;
  fault_cpu.Init(0, memory_fault, 16);
  fault_cpu.RunQuantum();
  if (!compiled.Compile(memory_fault, err)) {
    std::printf("compiled slice 3: memory-fault PTX compile FAIL: %s\n",
                err.c_str());
    return 1;
  }
  std::vector<wvm::VmState> fault_states(1);
  fault_states[0].status = wvm::kRunning;
  fault_states[0].rng_state = 0x1234567u;
  std::vector<uint32_t> fault_memory(16, 0);
  if (!compiled.Launch(fault_states, fault_memory, 16, no_framebuffer,
                       no_frame_seq, err)) {
    std::printf("compiled slice 3: memory-fault launch FAIL: %s\n",
                err.c_str());
    return 1;
  }
  difference.clear();
  const bool fault_equal = SameArchitecturalState(
      StateFromCpu(fault_cpu), fault_states[0], difference);
  std::printf("compiled slice 3: memory-fault semantics %s",
              fault_equal ? "PASS" : "FAIL");
  if (!fault_equal) std::printf(" (%s)", difference.c_str());
  std::printf("\n");
  ok &= fault_equal;

  // Messaging is device-resident state shared by every compiled VM in the
  // launch. VM 0 sends while VM 1 polls; the result never passes through the
  // host between those operations.
  wvm::WvmFile messaging;
  messaging.code = {
      wvm::enc_r(wvm::kVmid, 0, 0, 0, 0),
      wvm::enc_i(wvm::kCmpEqI, 0, 0, 0, 0),
      wvm::enc_i(wvm::kJmpIfAny, 1, 0, 0, 6),
      wvm::enc_i(wvm::kCmpEqI, 0, 1, 0, 1),
      wvm::enc_i(wvm::kJmpIfAny, 2, 0, 0, 11),
      wvm::enc_r(wvm::kHalt, 0, 0, 0, 0),
      wvm::enc_i(wvm::kMovI, 0, 1, 0, 1),
      wvm::enc_i(wvm::kMovI, 0, 2, 0, 7),
      wvm::enc_i(wvm::kMovI, 0, 3, 0, 1234),
      wvm::enc_r(wvm::kSend, 0, 1, 2, 3),
      wvm::enc_r(wvm::kHalt, 0, 0, 0, 0),
      wvm::enc_r(wvm::kTryRecv, 0, 2, 4, 5),
      wvm::enc_r(wvm::kNotMask, 0, 3, 2, 0),
      wvm::enc_i(wvm::kJmpIfAny, 4, 0, 0, 11),
      wvm::enc_i(wvm::kMovI, 0, 6, 0, 0),
      wvm::enc_r(wvm::kStore, 0, 6, 4, 0),
      wvm::enc_r(wvm::kHalt, 0, 0, 0, 0),
  };
  if (!compiled.Compile(messaging, err)) {
    std::printf("compiled messaging: PTX compile FAIL: %s\n", err.c_str());
    return 1;
  }
  std::vector<wvm::VmState> messaging_states(4);
  for (uint32_t vm = 0; vm < messaging_states.size(); ++vm) {
    messaging_states[vm].vm_id = vm;
    messaging_states[vm].status = wvm::kRunning;
    messaging_states[vm].rng_state = vm * 0x9E3779B9u + 0x1234567u;
  }
  std::vector<uint32_t> messaging_memory(4 * 16, 0);
  if (!compiled.Launch(messaging_states, messaging_memory, 16,
                       no_framebuffer, no_frame_seq, err)) {
    std::printf("compiled messaging: launch FAIL: %s\n", err.c_str());
    return 1;
  }
  bool messaging_ok = messaging_memory[16] == 1234;
  for (const wvm::VmState& state : messaging_states)
    messaging_ok &= state.status == wvm::kHalted &&
                    state.fault_code == wvm::kFaultOk;
  std::printf("compiled messaging: resident ring delivery %s\n",
              messaging_ok ? "PASS" : "FAIL");
  ok &= messaging_ok;

  // Architectural IDs need not equal resident slots. Exercise the same
  // logical route in both one-shot native execution and the resident native
  // engine; control/status remains slot-indexed in the resident kernel.
  wvm::WvmFile logical_messaging;
  logical_messaging.code = {
      wvm::enc_r(wvm::kVmid, 0, 0, 0, 0),
      wvm::enc_i(wvm::kCmpEqI, 0, 0, 0, 37),
      wvm::enc_i(wvm::kJmpIfAny, 1, 0, 0, 6),
      wvm::enc_i(wvm::kCmpEqI, 0, 1, 0, 91),
      wvm::enc_i(wvm::kJmpIfAny, 2, 0, 0, 11),
      wvm::enc_r(wvm::kHalt, 0, 0, 0, 0),
      wvm::enc_i(wvm::kMovI, 0, 1, 0, 91),
      wvm::enc_i(wvm::kMovI, 0, 2, 0, 7),
      wvm::enc_i(wvm::kMovI, 0, 3, 0, 2345),
      wvm::enc_r(wvm::kSend, 0, 1, 2, 3),
      wvm::enc_r(wvm::kHalt, 0, 0, 0, 0),
      wvm::enc_r(wvm::kTryRecv, 0, 2, 4, 5),
      wvm::enc_r(wvm::kNotMask, 0, 3, 2, 0),
      wvm::enc_i(wvm::kJmpIfAny, 4, 0, 0, 11),
      wvm::enc_i(wvm::kMovI, 0, 6, 0, 0),
      wvm::enc_r(wvm::kStore, 0, 6, 4, 0),
      wvm::enc_i(wvm::kMovI, 0, 6, 0, 1),
      wvm::enc_r(wvm::kStore, 0, 6, 5, 0),
      wvm::enc_r(wvm::kFlip, 0, 0, 0, 0),
      wvm::enc_r(wvm::kHalt, 0, 0, 0, 0),
  };
  if (!compiled.Compile(logical_messaging, err)) {
    std::printf("compiled logical messaging: compile FAIL: %s\n",
                err.c_str());
    return 1;
  }
  std::vector<wvm::VmState> logical_states(2);
  logical_states[0].vm_id = 37;
  logical_states[1].vm_id = 91;
  for (wvm::VmState& state : logical_states) {
    state.status = wvm::kRunning;
    state.rng_state = state.vm_id * 0x9E3779B9u + 0x1234567u;
  }
  std::vector<uint32_t> logical_memory(32, 0);
  if (!compiled.Launch(logical_states, logical_memory, 16, no_framebuffer,
                       no_frame_seq, err)) {
    std::printf("compiled logical messaging: launch FAIL: %s\n",
                err.c_str());
    return 1;
  }
  const bool logical_compiled_ok =
      logical_states[0].status == wvm::kHalted &&
      logical_states[1].status == wvm::kHalted &&
      logical_states[0].vm_id == 37 && logical_states[1].vm_id == 91 &&
      logical_memory[16] == 2345 &&
      logical_memory[17] == ((7u << 16) | 37u);
  std::printf("compiled messaging: logical IDs one-shot %s\n",
              logical_compiled_ok ? "PASS" : "FAIL");
  ok &= logical_compiled_ok;

  std::vector<wvm::VmImage> logical_images(2);
  for (wvm::VmImage& image : logical_images) {
    image.code = logical_messaging.code;
    image.mem_size_words = 16;
  }
  const std::vector<wvm::LogicalVmId> logical_ids = {{37}, {91}};
  wvm::PersistentRuntime logical_runtime;
  wvm::PtxResidentProgram logical_resident;
  const bool resident_started =
      logical_runtime.Init(logical_images, logical_ids, err) &&
      logical_runtime.EnsureStream(err) &&
      logical_resident.Compile(logical_messaging, err) &&
      logical_resident.Launch(
          reinterpret_cast<CUdeviceptr>(logical_runtime.DeviceStates()), 2,
          reinterpret_cast<CUdeviceptr>(logical_runtime.DeviceDescs()),
          reinterpret_cast<CUdeviceptr>(logical_runtime.DeviceControl()),
          reinterpret_cast<CUdeviceptr>(logical_runtime.DeviceMailboxes()),
          reinterpret_cast<CUstream>(logical_runtime.Stream()), err);
  if (!resident_started) {
    std::printf("compiled logical resident: setup FAIL: %s\n", err.c_str());
    return 1;
  }
  logical_runtime.BootAll();
  logical_runtime.WaitStatus(0, wvm::kHalted, 2000);
  logical_runtime.WaitStatus(1, wvm::kHalted, 2000);
  std::vector<uint32_t> resident_logical_memory;
  const bool logical_resident_ok =
      logical_runtime.Status(0) == wvm::kHalted &&
      logical_runtime.Status(1) == wvm::kHalted &&
      logical_runtime.ReadMem(1, 0, 2, resident_logical_memory) &&
      resident_logical_memory.size() == 2 &&
      resident_logical_memory[0] == 2345 &&
      resident_logical_memory[1] == ((7u << 16) | 37u) &&
      logical_runtime.FrameSeq(1) == 1;
  std::printf("compiled messaging: logical IDs resident %s\n",
              logical_resident_ok ? "PASS" : "FAIL");
  ok &= logical_resident_ok;
  logical_runtime.ShutdownAll();
  ok &= logical_runtime.Sync() == cudaSuccess;

  // A retired logical address must not alias a replacement occupying slot 1.
  wvm::WvmFile stale_logical_destination;
  stale_logical_destination.code = {
      wvm::enc_r(wvm::kVmid, 0, 0, 0, 0),
      wvm::enc_i(wvm::kCmpEqI, 0, 0, 0, 37),
      wvm::enc_i(wvm::kJmpIfAny, 1, 0, 0, 4),
      wvm::enc_r(wvm::kHalt, 0, 0, 0, 0),
      wvm::enc_i(wvm::kMovI, 0, 1, 0, 1),
      wvm::enc_i(wvm::kMovI, 0, 2, 0, 9),
      wvm::enc_i(wvm::kMovI, 0, 3, 0, 999),
      wvm::enc_r(wvm::kSend, 0, 1, 2, 3),
      wvm::enc_r(wvm::kHalt, 0, 0, 0, 0),
  };
  if (!compiled.Compile(stale_logical_destination, err)) {
    std::printf("compiled retired route: compile FAIL: %s\n", err.c_str());
    return 1;
  }
  std::vector<wvm::VmState> stale_states(2);
  stale_states[0].vm_id = 37;
  stale_states[1].vm_id = 91;
  for (wvm::VmState& state : stale_states)
    state.status = wvm::kRunning;
  std::vector<uint32_t> stale_memory(32, 0);
  if (!compiled.Launch(stale_states, stale_memory, 16, no_framebuffer,
                       no_frame_seq, err)) {
    std::printf("compiled retired route: launch FAIL: %s\n", err.c_str());
    return 1;
  }
  const bool stale_compiled_ok =
      stale_states[0].status == wvm::kFaulted &&
      stale_states[0].fault_code == wvm::kFaultMsg &&
      stale_states[1].status == wvm::kHalted;
  std::printf("compiled messaging: retired route rejected %s\n",
              stale_compiled_ok ? "PASS" : "FAIL");
  ok &= stale_compiled_ok;

  // A false guard suppresses the receive side effect; the following
  // unguarded receive must still consume the self-sent message.
  wvm::WvmFile guarded_receive;
  guarded_receive.code = {
      wvm::enc_r(wvm::kVmid, 0, 0, 0, 0),
      wvm::enc_i(wvm::kMovI, 0, 1, 0, 0),
      wvm::enc_i(wvm::kMovI, 0, 2, 0, 7),
      wvm::enc_i(wvm::kMovI, 0, 3, 0, 1234),
      wvm::enc_r(wvm::kSend, 0, 1, 2, 3),
      wvm::enc_i(wvm::kCmpEqI, 0, 0, 0, 1),
      wvm::enc_r(wvm::kTryRecv, 1, 1, 4, 5),
      wvm::enc_r(wvm::kTryRecv, 0, 2, 6, 7),
      wvm::enc_r(wvm::kHalt, 0, 0, 0, 0),
  };
  if (!compiled.Compile(guarded_receive, err)) {
    std::printf("compiled messaging guard: compile FAIL: %s\n", err.c_str());
    return 1;
  }
  std::vector<wvm::VmState> guarded_states(1);
  guarded_states[0].status = wvm::kRunning;
  guarded_states[0].rng_state = 0x1234567u;
  std::vector<uint32_t> guarded_memory(16, 0);
  if (!compiled.Launch(guarded_states, guarded_memory, 16,
                       no_framebuffer, no_frame_seq, err)) {
    std::printf("compiled messaging guard: launch FAIL: %s\n", err.c_str());
    return 1;
  }
  const bool guard_ok = guarded_states[0].status == wvm::kHalted &&
                        guarded_states[0].preds[1] == 0 &&
                        guarded_states[0].preds[2] == wvm::kFullMask &&
                        guarded_states[0].vregs[6 * wvm::kLanes] == 1234;
  std::printf("compiled messaging: guarded receive %s\n",
              guard_ok ? "PASS" : "FAIL");
  ok &= guard_ok;

  // Fill, drain, and reuse every ring slot twice. This checks FIFO payloads,
  // metadata, wraparound publication sequences, and the exact 16-slot limit.
  wvm::WvmFile mailbox_fifo;
  auto& fifo_code = mailbox_fifo.code;
  fifo_code.push_back(wvm::enc_i(wvm::kMovI, 0, 0, 0, 0));  // self dest
  fifo_code.push_back(wvm::enc_i(wvm::kMovI, 0, 1, 0, 9));  // type
  fifo_code.push_back(wvm::enc_i(wvm::kMovI, 0, 2, 0, 1));  // payload
  fifo_code.push_back(wvm::enc_i(wvm::kMovI, 0, 3, 0, 0));  // address
  fifo_code.push_back(wvm::enc_i(wvm::kMovI, 0, 7, 0, 0));  // sum
  for (uint32_t cycle = 0; cycle < 2; ++cycle) {
    for (uint32_t message = 0; message < wvm::kMailboxSlots; ++message) {
      fifo_code.push_back(wvm::enc_r(wvm::kSend, 0, 0, 1, 2));
      fifo_code.push_back(wvm::enc_i(wvm::kAddI, 0, 2, 2, 1));
    }
    for (uint32_t message = 0; message < wvm::kMailboxSlots; ++message) {
      fifo_code.push_back(wvm::enc_r(wvm::kTryRecv, 0, 0, 4, 5));
      fifo_code.push_back(wvm::enc_r(wvm::kAdd, 0, 7, 7, 4));
      fifo_code.push_back(wvm::enc_r(wvm::kStore, 0, 3, 4, 0));
      fifo_code.push_back(wvm::enc_i(wvm::kAddI, 0, 3, 3, 1));
    }
  }
  fifo_code.push_back(wvm::enc_r(wvm::kHalt, 0, 0, 0, 0));
  if (!compiled.Compile(mailbox_fifo, err)) {
    std::printf("compiled mailbox FIFO: compile FAIL: %s\n", err.c_str());
    return 1;
  }
  std::vector<wvm::VmState> fifo_states(1);
  fifo_states[0].status = wvm::kRunning;
  fifo_states[0].rng_state = 0x1234567u;
  std::vector<uint32_t> fifo_memory(64, 0);
  if (!compiled.Launch(fifo_states, fifo_memory, 64,
                       no_framebuffer, no_frame_seq, err)) {
    std::printf("compiled mailbox FIFO: launch FAIL: %s\n", err.c_str());
    return 1;
  }
  bool fifo_ok = fifo_states[0].status == wvm::kHalted &&
                 fifo_states[0].vregs[7 * wvm::kLanes] == 528 &&
                 fifo_states[0].vregs[5 * wvm::kLanes] == (9u << 16);
  for (uint32_t word = 0; word < 32; ++word)
    fifo_ok &= fifo_memory[word] == word + 1;
  std::printf("compiled messaging: FIFO capacity/reuse %s\n",
              fifo_ok ? "PASS" : "FAIL");
  ok &= fifo_ok;

  wvm::WvmFile mailbox_full;
  mailbox_full.code = {
      wvm::enc_i(wvm::kMovI, 0, 0, 0, 0),
      wvm::enc_i(wvm::kMovI, 0, 1, 0, 1),
      wvm::enc_i(wvm::kMovI, 0, 2, 0, 99),
  };
  for (uint32_t message = 0; message <= wvm::kMailboxSlots; ++message)
    mailbox_full.code.push_back(wvm::enc_r(wvm::kSend, 0, 0, 1, 2));
  mailbox_full.code.push_back(wvm::enc_r(wvm::kHalt, 0, 0, 0, 0));
  if (!compiled.Compile(mailbox_full, err)) {
    std::printf("compiled mailbox full: compile FAIL: %s\n", err.c_str());
    return 1;
  }
  std::vector<wvm::VmState> full_states(1);
  full_states[0].status = wvm::kRunning;
  full_states[0].rng_state = 0x1234567u;
  std::vector<uint32_t> full_memory(16, 0);
  if (!compiled.Launch(full_states, full_memory, 16,
                       no_framebuffer, no_frame_seq, err)) {
    std::printf("compiled mailbox full: launch FAIL: %s\n", err.c_str());
    return 1;
  }
  const bool full_ok = full_states[0].status == wvm::kFaulted &&
                       full_states[0].fault_code == wvm::kFaultMsg;
  std::printf("compiled messaging: full-mailbox fault %s\n",
              full_ok ? "PASS" : "FAIL");
  ok &= full_ok;

  wvm::WvmFile invalid_destination;
  invalid_destination.code = {
      wvm::enc_i(wvm::kMovI, 0, 0, 0, 1),
      wvm::enc_i(wvm::kMovI, 0, 1, 0, 1),
      wvm::enc_i(wvm::kMovI, 0, 2, 0, 99),
      wvm::enc_r(wvm::kSend, 0, 0, 1, 2),
      wvm::enc_r(wvm::kHalt, 0, 0, 0, 0),
  };
  if (!compiled.Compile(invalid_destination, err)) {
    std::printf("compiled invalid destination: compile FAIL: %s\n",
                err.c_str());
    return 1;
  }
  std::vector<wvm::VmState> invalid_states(1);
  invalid_states[0].status = wvm::kRunning;
  invalid_states[0].rng_state = 0x1234567u;
  std::vector<uint32_t> invalid_memory(16, 0);
  if (!compiled.Launch(invalid_states, invalid_memory, 16,
                       no_framebuffer, no_frame_seq, err)) {
    std::printf("compiled invalid destination: launch FAIL: %s\n",
                err.c_str());
    return 1;
  }
  const bool invalid_ok = invalid_states[0].status == wvm::kFaulted &&
                          invalid_states[0].fault_code == wvm::kFaultMsg;
  std::printf("compiled messaging: invalid-destination fault %s\n",
              invalid_ok ? "PASS" : "FAIL");
  ok &= invalid_ok;

  // Unsupported bytecode must stop at compilation, never become a partial
  // instruction-level fallback inside native execution.
  wvm::WvmFile unsupported;
  unsupported.code = {
      wvm::enc_r(wvm::kRand, 0, 0, 0, 0),
      wvm::enc_r(wvm::kHalt, 0, 0, 0, 0),
  };
  err.clear();
  const bool rejected = !compiled.Compile(unsupported, err) &&
                        err.find("unsupported") != std::string::npos;
  std::printf("compiled slice 1: unsupported opcode rejection %s",
              rejected ? "PASS" : "FAIL");
  if (!rejected) std::printf(" (%s)", err.c_str());
  std::printf("\n");
  ok &= rejected;

  wvm::PtxCompilationCache cache;
  std::shared_ptr<wvm::PtxCompiledProgram> cached_first;
  std::shared_ptr<wvm::PtxCompiledProgram> cached_second;
  bool first_hit = true;
  bool second_hit = false;
  const bool cache_ok =
      cache.GetOrCompile(arithmetic, cached_first, first_hit, err) &&
      cache.GetOrCompile(arithmetic, cached_second, second_hit, err) &&
      !first_hit && second_hit && cached_first == cached_second &&
      cache.size() == 1;
  std::printf("compiled slice 9: program-identity cache %s\n",
              cache_ok ? "PASS" : "FAIL");
  ok &= cache_ok;

  std::printf(ok ? "compiled backend: PASS\n" :
                   "compiled backend: FAIL\n");
  return ok ? 0 : 1;
}

int EmitCompiledPtx(const char* input, const char* output) {
  wvm::WvmFile file;
  std::string err;
  if (!wvm::LoadWvm(input, file, err)) {
    std::fprintf(stderr, "error: %s: %s\n", input, err.c_str());
    return 2;
  }
  std::string ptx;
  if (!wvm::TranslateWvmToPtx(file, ptx, err)) {
    std::fprintf(stderr, "error: %s\n", err.c_str());
    return 1;
  }
  std::ofstream stream(output, std::ios::binary | std::ios::trunc);
  if (!stream || !stream.write(ptx.data(), ptx.size())) {
    std::fprintf(stderr, "error: cannot write %s\n", output);
    return 1;
  }
  std::printf("compiled PTX: %zu bytes -> %s\n", ptx.size(), output);
  return 0;
}

int RunCompiledHaltEquivalence(const char* path) {
  constexpr uint32_t kMemoryWords = wvm::kRamSizeWords;
  wvm::WvmFile file;
  std::string err;
  if (!wvm::LoadWvm(path, file, err)) {
    std::fprintf(stderr, "error: %s: %s\n", path, err.c_str());
    return 2;
  }

  wvm::CpuVm cpu;
  cpu.Init(0, file, kMemoryWords);
  while (cpu.status == wvm::kRunning &&
         cpu.instruction_counter < 10000000u)
    cpu.RunQuantum();
  if (cpu.status != wvm::kHalted) {
    std::printf("compiled halt equivalence: CPU did not halt "
                "(status=%s fault=%s pc=%u)\n",
                wvm::StatusName(cpu.status), wvm::FaultName(cpu.fault), cpu.pc);
    return 1;
  }

  wvm::PtxCompiledProgram compiled;
  if (!compiled.Compile(file, err)) {
    std::printf("compiled halt equivalence: compile FAIL: %s\n", err.c_str());
    return 1;
  }
  std::vector<wvm::VmState> states(1);
  states[0].status = wvm::kRunning;
  states[0].rng_state = 0x1234567u;
  std::vector<uint32_t> memory(kMemoryWords, 0);
  std::vector<uint32_t> framebuffer(wvm::kVideoWords,
                                    wvm::kVideoResetColor);
  std::vector<uint32_t> frame_seq(1, 0);
  if (!compiled.Launch(states, memory, kMemoryWords, framebuffer, frame_seq,
                       err)) {
    std::printf("compiled halt equivalence: launch FAIL: %s\n", err.c_str());
    return 1;
  }

  std::string difference;
  const bool state_equal =
      SameArchitecturalState(StateFromCpu(cpu), states[0], difference);
  const bool memory_equal = memory == cpu.memory;
  const bool framebuffer_equal = framebuffer == cpu.framebuffer;
  const bool frame_equal = frame_seq[0] == cpu.frame_seq;
  const bool pass = state_equal && memory_equal && framebuffer_equal &&
                    frame_equal;
  std::printf("compiled halt equivalence: state=%s memory=%s framebuffer=%s "
              "frame_seq=%s r0=%u%s%s\n",
              state_equal ? "PASS" : "FAIL",
              memory_equal ? "PASS" : "FAIL",
              framebuffer_equal ? "PASS" : "FAIL",
              frame_equal ? "PASS" : "FAIL", states[0].vregs[0],
              state_equal ? "" : " difference=", state_equal ? "" : difference.c_str());
  std::printf(pass ? "compiled halt equivalence: PASS\n" :
                     "compiled halt equivalence: FAIL\n");
  return pass ? 0 : 1;
}

int RunCompiledResident(const char* path, uint32_t num_vms,
                        int seconds = 3) {
  if (num_vms == 0 || num_vms > wvm::kMaxVms) {
    std::fprintf(stderr, "compiled_resident: VM count must be 1..%u\n",
                 wvm::kMaxVms);
    return 2;
  }
  wvm::WvmFile file;
  std::string err;
  if (!wvm::LoadWvm(path, file, err)) {
    std::fprintf(stderr, "compiled_resident: %s\n", err.c_str());
    return 2;
  }
  std::vector<wvm::VmImage> images(num_vms);
  for (wvm::VmImage& image : images) {
    image.code = file.code;
    image.literals = file.literals;
    image.mem_size_words = wvm::kRamSizeWords;
  }
  wvm::PersistentRuntime runtime;
  wvm::PtxResidentProgram program;
  if (!runtime.Init(images, err) || !runtime.EnsureStream(err) ||
      !program.Compile(file, err) ||
      !program.Launch(
          reinterpret_cast<CUdeviceptr>(runtime.DeviceStates()), num_vms,
          reinterpret_cast<CUdeviceptr>(runtime.DeviceDescs()),
          reinterpret_cast<CUdeviceptr>(runtime.DeviceControl()),
          reinterpret_cast<CUdeviceptr>(runtime.DeviceMailboxes()),
          reinterpret_cast<CUstream>(runtime.Stream()), err)) {
    std::fprintf(stderr, "compiled_resident: setup failed: %s\n", err.c_str());
    return 1;
  }
  runtime.BootAll();
  for (uint32_t vm = 0; vm < num_vms; ++vm) {
    for (int waited = 0; waited < 2000 && runtime.Status(vm) == wvm::kIdle;
         ++waited)
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  std::this_thread::sleep_for(std::chrono::seconds(seconds));

  bool running = true;
  for (uint32_t vm = 0; vm < num_vms; ++vm)
    running &= runtime.Status(vm) == wvm::kRunning &&
               runtime.Fault(vm) == wvm::kFaultOk;
  const uint32_t frame_before_pause = runtime.FrameSeq(0);
  std::vector<uint32_t> memory;
  const bool memory_ok = runtime.ReadMem(0, 96, 7, memory) &&
                         memory.size() == 7;
  const uint32_t received = memory_ok ? memory[2] : 0;

  const bool paused = runtime.Pause(0);
  wvm::VmState paused_state{};
  const bool state_ok = paused && runtime.ReadState(0, paused_state) &&
                        paused_state.status == wvm::kPaused;
  runtime.SendCmd(0, wvm::kCmdRun);
  std::this_thread::sleep_for(std::chrono::milliseconds(250));
  const bool resumed = runtime.Status(0) == wvm::kRunning &&
                       runtime.FrameSeq(0) > frame_before_pause;

  runtime.ShutdownAll();
  const cudaError_t sync = runtime.Sync();
  const bool pass = running && frame_before_pause > 0 &&
                    (num_vms == 1 || received > 0) && state_ok && resumed &&
                    sync == cudaSuccess;
  std::printf("compiled resident: VMs=%u running=%s frames=%u received=%u "
              "pause=%s resume=%s shutdown=%s JIT=%.3fms\n",
              num_vms, running ? "PASS" : "FAIL", frame_before_pause,
              received, state_ok ? "PASS" : "FAIL",
              resumed ? "PASS" : "FAIL",
              sync == cudaSuccess ? "PASS" : "FAIL",
              program.jit_milliseconds());
  std::printf(pass ? "compiled resident: PASS\n" :
                     "compiled resident: FAIL\n");
  return pass ? 0 : 1;
}

int RunWarpCDataBenchmark(const char* sequential_path,
                          const char* warp_path) {
  constexpr uint32_t kMemoryWords = wvm::kRamSizeWords;
  constexpr uint64_t kInstructionLimit = 100000000u;
  struct Measurement {
    uint64_t instructions = 0;
    double milliseconds = 0.0;
  };

  wvm::WvmFile sequential_file, warp_file;
  std::string err;
  if (!wvm::LoadWvm(sequential_path, sequential_file, err) ||
      !wvm::LoadWvm(warp_path, warp_file, err)) {
    std::fprintf(stderr, "warpc_bench: load failed: %s\n", err.c_str());
    return 2;
  }
  wvm::PtxCompiledProgram sequential_compiled, warp_compiled;
  if (!sequential_compiled.Compile(sequential_file, err) ||
      !warp_compiled.Compile(warp_file, err)) {
    std::fprintf(stderr, "warpc_bench: PTX compile failed: %s\n", err.c_str());
    return 1;
  }

  auto measure = [&](const wvm::WvmFile& file,
                     const wvm::PtxCompiledProgram& compiled,
                     uint32_t benchmark_case, uint32_t operation,
                     Measurement& measurement) -> bool {
    wvm::CpuVm cpu;
    cpu.Init(0, file, kMemoryWords);
    cpu.memory[0] = benchmark_case;
    cpu.memory[1] = operation;
    while (cpu.status == wvm::kRunning &&
           cpu.instruction_counter < kInstructionLimit)
      cpu.RunQuantum();
    if (cpu.status != wvm::kHalted || cpu.vregs[0][0] != 42) {
      err = "logical interpreter did not halt with r0=42";
      return false;
    }
    measurement.instructions = cpu.instruction_counter;

    std::array<double, 5> samples{};
    for (double& sample : samples) {
      std::vector<wvm::VmState> states(1);
      states[0].status = wvm::kRunning;
      states[0].rng_state = 0x1234567u;
      std::vector<uint32_t> memory(kMemoryWords, 0);
      memory[0] = benchmark_case;
      memory[1] = operation;
      std::vector<uint32_t> no_framebuffer;
      std::vector<uint32_t> no_frame_seq;
      if (!compiled.LaunchCheckpoints(states, memory, kMemoryWords,
                                      no_framebuffer, no_frame_seq, 1,
                                      sample, err))
        return false;
      std::string difference;
      if (!SameArchitecturalState(StateFromCpu(cpu), states[0], difference) ||
          memory != cpu.memory) {
        err = "interpreter/PTX mismatch: " + difference;
        return false;
      }
    }
    std::sort(samples.begin(), samples.end());
    measurement.milliseconds = samples[samples.size() / 2];
    return true;
  };

  const std::array<const char*, 3> operation_names = {"copy", "fill", "add"};
  const std::array<uint32_t, 4> word_counts = {32, 128, 1024, 4096};
  bool ok = true;
  std::printf("Warp C v0.1.5 data benchmark\n");
  std::printf("  each sample processes 262144 logical words\n");
  std::printf("  interpreted column is exact retired WarpVM bytecodes; "
              "compiled time is median of 5 kernel-only launches\n\n");
  std::printf("  op    words   interpreter bytecodes (seq/warp/speedup)"
              "   compiled ms (seq/warp/speedup)\n");
  for (uint32_t operation = 0; operation < operation_names.size(); ++operation) {
    for (uint32_t benchmark_case = 0; benchmark_case < word_counts.size();
         ++benchmark_case) {
      Measurement sequential, warp;
      const bool measured =
          measure(sequential_file, sequential_compiled, benchmark_case,
                  operation, sequential) &&
          measure(warp_file, warp_compiled, benchmark_case, operation, warp);
      ok &= measured;
      if (!measured) {
        std::printf("  %-5s %5u   FAIL: %s\n", operation_names[operation],
                    word_counts[benchmark_case], err.c_str());
        continue;
      }
      const double instruction_speedup =
          static_cast<double>(sequential.instructions) / warp.instructions;
      const double compiled_speedup =
          sequential.milliseconds / warp.milliseconds;
      std::printf("  %-5s %5u   %9llu / %-9llu / %6.2fx"
                  "       %8.4f / %-8.4f / %6.2fx\n",
                  operation_names[operation], word_counts[benchmark_case],
                  static_cast<unsigned long long>(sequential.instructions),
                  static_cast<unsigned long long>(warp.instructions),
                  instruction_speedup, sequential.milliseconds,
                  warp.milliseconds, compiled_speedup);
    }
  }
  std::printf(ok ? "warpc_015_bench PASS\n" : "warpc_015_bench FAIL\n");
  return ok ? 0 : 1;
}

int RunCompiledLife(const char* path) {
  constexpr uint32_t kVms = 2;
  constexpr uint32_t kMemoryWords = wvm::kRamSizeWords;
  wvm::WvmFile file;
  std::string err;
  if (!wvm::LoadWvm(path, file, err)) {
    std::fprintf(stderr, "error: %s: %s\n", path, err.c_str());
    return 2;
  }

  wvm::PtxCompiledProgram compiled;
  if (!compiled.Compile(file, err)) {
    std::printf("compiled WarpLife: compile FAIL: %s\n", err.c_str());
    return 1;
  }

  std::vector<wvm::CpuVm> cpu(kVms);
  for (uint32_t vm = 0; vm < kVms; ++vm) {
    cpu[vm].Init(vm, file, kMemoryWords);
    if (!RunCpuToNextYield(cpu[vm])) {
      std::printf("compiled WarpLife: CPU checkpoint FAIL (vm %u)\n", vm);
      return 1;
    }
  }

  std::vector<wvm::VmState> states(kVms);
  std::vector<uint32_t> memory(kVms * kMemoryWords, 0);
  std::vector<uint32_t> framebuffers(kVms * wvm::kVideoWords,
                                     wvm::kVideoResetColor);
  std::vector<uint32_t> frame_seq(kVms, 0);
  for (uint32_t vm = 0; vm < kVms; ++vm) {
    states[vm].vm_id = vm;
    states[vm].status = wvm::kRunning;
    states[vm].rng_state = vm * 0x9E3779B9u + 0x1234567u;
  }
  if (!compiled.Launch(states, memory, kMemoryWords, framebuffers,
                       frame_seq, err)) {
    std::printf("compiled WarpLife: first checkpoint launch FAIL: %s\n",
                err.c_str());
    return 1;
  }

  bool first_equal = true;
  std::string difference;
  for (uint32_t vm = 0; vm < kVms; ++vm) {
    wvm::VmState expected = StateFromCpu(cpu[vm]);
    expected.status = wvm::kPaused;
    difference.clear();
    const bool state_equal =
        SameArchitecturalState(expected, states[vm], difference);
    const bool memory_equal =
        std::equal(cpu[vm].memory.begin(), cpu[vm].memory.end(),
                   memory.begin() + vm * kMemoryWords);
    const bool framebuffer_equal = std::equal(
        cpu[vm].framebuffer.begin(), cpu[vm].framebuffer.end(),
        framebuffers.begin() + vm * wvm::kVideoWords);
    const bool seq_equal = frame_seq[vm] == cpu[vm].frame_seq;
    if (!state_equal || !memory_equal || !framebuffer_equal || !seq_equal) {
      std::printf("compiled WarpLife: vm %u differs:", vm);
      if (!state_equal) std::printf(" state(%s)", difference.c_str());
      if (!memory_equal) {
        for (uint32_t i = 0; i < kMemoryWords; ++i) {
          if (cpu[vm].memory[i] != memory[vm * kMemoryWords + i]) {
            std::printf(" memory[%u]=%08x/%08x", i, cpu[vm].memory[i],
                        memory[vm * kMemoryWords + i]);
            break;
          }
        }
      }
      if (!framebuffer_equal) {
        for (uint32_t i = 0; i < wvm::kVideoWords; ++i) {
          if (cpu[vm].framebuffer[i] !=
              framebuffers[vm * wvm::kVideoWords + i]) {
            std::printf(" framebuffer[%u]=%08x/%08x", i,
                        cpu[vm].framebuffer[i],
                        framebuffers[vm * wvm::kVideoWords + i]);
            break;
          }
        }
      }
      if (!seq_equal)
        std::printf(" frame_seq=%u/%u", cpu[vm].frame_seq, frame_seq[vm]);
      std::printf("\n");
    }
    first_equal &= state_equal && memory_equal && framebuffer_equal &&
                   seq_equal;
  }
  std::printf("compiled WarpLife: two-VM shared-artifact checkpoint %s\n",
              first_equal ? "PASS" : "FAIL");
  if (!first_equal) return 1;

  // compiled -> interpreted: restore VM 0's exact checkpoint into the CPU
  // reference engine, then retire one more generation there.
  wvm::CpuVm compiled_to_cpu;
  compiled_to_cpu.Init(0, file, kMemoryWords);
  RestoreCpuAtCheckpoint(compiled_to_cpu, states[0], memory.data(),
                         framebuffers.data(), frame_seq[0]);
  if (!RunCpuToNextYield(compiled_to_cpu)) {
    std::printf("compiled WarpLife: compiled->interpreted continuation FAIL\n");
    return 1;
  }

  // interpreted -> compiled: materialize the CPU checkpoint for both VMs and
  // let the same shared native artifact retire their next generation.
  std::vector<wvm::VmState> transitioned_states(kVms);
  std::vector<uint32_t> transitioned_memory(kVms * kMemoryWords);
  std::vector<uint32_t> transitioned_framebuffers(kVms * wvm::kVideoWords);
  std::vector<uint32_t> transitioned_seq(kVms);
  for (uint32_t vm = 0; vm < kVms; ++vm) {
    transitioned_states[vm] = StateFromCpu(cpu[vm]);
    transitioned_states[vm].status = wvm::kPaused;
    std::copy(cpu[vm].memory.begin(), cpu[vm].memory.end(),
              transitioned_memory.begin() + vm * kMemoryWords);
    std::copy(cpu[vm].framebuffer.begin(), cpu[vm].framebuffer.end(),
              transitioned_framebuffers.begin() + vm * wvm::kVideoWords);
    transitioned_seq[vm] = cpu[vm].frame_seq;
    if (!RunCpuToNextYield(cpu[vm])) {
      std::printf("compiled WarpLife: second CPU checkpoint FAIL (vm %u)\n",
                  vm);
      return 1;
    }
  }
  if (!compiled.Launch(transitioned_states, transitioned_memory,
                       kMemoryWords, transitioned_framebuffers,
                       transitioned_seq, err)) {
    std::printf("compiled WarpLife: interpreted->compiled launch FAIL: %s\n",
                err.c_str());
    return 1;
  }

  bool transition_equal = true;
  for (uint32_t vm = 0; vm < kVms; ++vm) {
    wvm::VmState expected = StateFromCpu(cpu[vm]);
    expected.status = wvm::kPaused;
    difference.clear();
    transition_equal &=
        SameArchitecturalState(expected, transitioned_states[vm], difference);
    transition_equal &= std::equal(
        cpu[vm].memory.begin(), cpu[vm].memory.end(),
        transitioned_memory.begin() + vm * kMemoryWords);
    transition_equal &= std::equal(
        cpu[vm].framebuffer.begin(), cpu[vm].framebuffer.end(),
        transitioned_framebuffers.begin() + vm * wvm::kVideoWords);
    transition_equal &= transitioned_seq[vm] == cpu[vm].frame_seq;
  }
  wvm::VmState c2i_expected = StateFromCpu(cpu[0]);
  c2i_expected.status = wvm::kPaused;
  wvm::VmState c2i_actual = StateFromCpu(compiled_to_cpu);
  c2i_actual.status = wvm::kPaused;
  difference.clear();
  const bool reverse_equal =
      SameArchitecturalState(c2i_expected, c2i_actual, difference) &&
      compiled_to_cpu.memory == cpu[0].memory &&
      compiled_to_cpu.framebuffer == cpu[0].framebuffer &&
      compiled_to_cpu.frame_seq == cpu[0].frame_seq;

  // Mixed session: VM 0 remains live in the persistent interpreter while an
  // independently identified VM 1 reaches a compiled YIELD checkpoint on a
  // separate non-blocking stream.
  wvm::VmImage interpreted_image;
  interpreted_image.code = file.code;
  interpreted_image.literals = file.literals;
  interpreted_image.mem_size_words = kMemoryWords;
  wvm::PersistentRuntime interpreted_runtime;
  bool mixed_equal = interpreted_runtime.Init({interpreted_image}, err) &&
                     interpreted_runtime.Launch(err);
  const bool mixed_started = mixed_equal;
  if (mixed_equal) interpreted_runtime.BootAll();
  for (int waited = 0;
       mixed_equal && interpreted_runtime.FrameSeq(0) < 1 && waited < 10000;
       ++waited)
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  const bool interpreted_ready = interpreted_runtime.FrameSeq(0) >= 1 &&
                                 interpreted_runtime.Status(0) == wvm::kRunning;
  mixed_equal &= interpreted_ready;
  const uint32_t interpreted_before = interpreted_runtime.FrameSeq(0);

  std::vector<wvm::VmState> mixed_state(1);
  mixed_state[0].vm_id = 1;
  mixed_state[0].status = wvm::kRunning;
  mixed_state[0].rng_state = 1u * 0x9E3779B9u + 0x1234567u;
  std::vector<uint32_t> mixed_memory(kMemoryWords, 0);
  std::vector<uint32_t> mixed_framebuffer(wvm::kVideoWords,
                                           wvm::kVideoResetColor);
  std::vector<uint32_t> mixed_seq(1, 0);
  bool mixed_launch = false;
  if (mixed_equal)
    mixed_launch = compiled.Launch(mixed_state, mixed_memory, kMemoryWords,
                                   mixed_framebuffer, mixed_seq, err);
  mixed_equal &= mixed_launch;
  std::this_thread::sleep_for(std::chrono::milliseconds(250));
  const uint32_t interpreted_after = interpreted_runtime.FrameSeq(0);
  const bool interpreted_progress =
      interpreted_after > interpreted_before &&
      interpreted_runtime.Status(0) == wvm::kRunning;
  mixed_equal &= interpreted_progress;

  wvm::CpuVm mixed_reference;
  mixed_reference.Init(1, file, kMemoryWords);
  mixed_equal &= RunCpuToNextYield(mixed_reference);
  wvm::VmState mixed_expected = StateFromCpu(mixed_reference);
  mixed_expected.status = wvm::kPaused;
  difference.clear();
  const bool mixed_state_equal =
      SameArchitecturalState(mixed_expected, mixed_state[0], difference);
  const bool mixed_memory_equal = mixed_memory == mixed_reference.memory;
  const bool mixed_framebuffer_equal =
      mixed_framebuffer == mixed_reference.framebuffer;
  const bool mixed_seq_equal = mixed_seq[0] == mixed_reference.frame_seq;
  mixed_equal &= mixed_state_equal && mixed_memory_equal &&
                 mixed_framebuffer_equal && mixed_seq_equal;
  interpreted_runtime.ShutdownAll();
  mixed_equal &= interpreted_runtime.Sync() == cudaSuccess;

  std::printf("compiled WarpLife: interpreted->compiled transition %s\n",
              transition_equal ? "PASS" : "FAIL");
  std::printf("compiled WarpLife: compiled->interpreted transition %s\n",
              reverse_equal ? "PASS" : "FAIL");
  std::printf("compiled WarpLife: simultaneous mixed-mode session %s\n",
              mixed_equal ? "PASS" : "FAIL");
  if (!mixed_equal) {
    std::printf("  mixed diagnostics: start=%d ready=%d launch=%d "
                "progress=%d frames=%u->%u state=%d(%s) memory=%d "
                "framebuffer=%d seq=%d error=%s\n",
                mixed_started, interpreted_ready, mixed_launch,
                interpreted_progress, interpreted_before, interpreted_after,
                mixed_state_equal, difference.c_str(), mixed_memory_equal,
                mixed_framebuffer_equal, mixed_seq_equal, err.c_str());
  }
  std::printf("compiled WarpLife: PTX %zu bytes, JIT %.3f ms\n",
              compiled.ptx().size(), compiled.jit_milliseconds());
  const bool ok = transition_equal && reverse_equal && mixed_equal;
  std::printf(ok ? "compiled WarpLife: PASS\n" :
                   "compiled WarpLife: FAIL\n");
  return ok ? 0 : 1;
}

int RunCompiledLifeBenchmark(const char* path) {
  constexpr uint32_t kMemoryWords = wvm::kRamSizeWords;
  constexpr uint32_t kCells = 128 * 128;
  struct Case { uint32_t vms; uint32_t checkpoints; };
  const Case cases[] = {
      {1, 500}, {8, 500}, {32, 500}, {64, 500}, {256, 300}};

  wvm::WvmFile file;
  std::string err;
  if (!wvm::LoadWvm(path, file, err)) {
    std::fprintf(stderr, "error: %s: %s\n", path, err.c_str());
    return 2;
  }
  wvm::PtxCompiledProgram compiled;
  if (!compiled.Compile(file, err)) {
    std::printf("compiled WarpLife benchmark: compile FAIL: %s\n",
                err.c_str());
    return 1;
  }
  std::printf("compiled WarpLife benchmark: PTX %zu bytes, JIT %.3f ms\n",
              compiled.ptx().size(), compiled.jit_milliseconds());
  std::printf("VMs  checkpoints  ms/checkpoint  gen/s/VM  aggregate Mcell/s\n");

  for (const Case& benchmark : cases) {
    std::vector<wvm::VmState> states(benchmark.vms);
    std::vector<uint32_t> memory(
        static_cast<size_t>(benchmark.vms) * kMemoryWords, 0);
    std::vector<uint32_t> framebuffers(
        static_cast<size_t>(benchmark.vms) * wvm::kVideoWords,
        wvm::kVideoResetColor);
    std::vector<uint32_t> frame_seq(benchmark.vms, 0);
    for (uint32_t vm = 0; vm < benchmark.vms; ++vm) {
      states[vm].vm_id = vm;
      states[vm].status = wvm::kRunning;
      states[vm].rng_state = vm * 0x9E3779B9u + 0x1234567u;
    }
    // First checkpoint performs deterministic initialization and warms the
    // generated kernel before the resident timing interval.
    if (!compiled.Launch(states, memory, kMemoryWords, framebuffers,
                         frame_seq, err)) {
      std::printf("compiled WarpLife benchmark: %u-VM warm-up FAIL: %s\n",
                  benchmark.vms, err.c_str());
      return 1;
    }
    double elapsed_ms = 0.0;
    if (!compiled.LaunchCheckpoints(
            states, memory, kMemoryWords, framebuffers, frame_seq,
            benchmark.checkpoints, elapsed_ms, err)) {
      std::printf("compiled WarpLife benchmark: %u-VM run FAIL: %s\n",
                  benchmark.vms, err.c_str());
      return 1;
    }
    bool frames_ok = true;
    for (uint32_t seq : frame_seq)
      frames_ok &= seq == benchmark.checkpoints + 1;
    if (!frames_ok) {
      std::printf("compiled WarpLife benchmark: %u-VM frame count FAIL\n",
                  benchmark.vms);
      return 1;
    }
    const double seconds = elapsed_ms / 1000.0;
    const double generations_per_vm = benchmark.checkpoints / seconds;
    const double aggregate_mcells =
        generations_per_vm * benchmark.vms * kCells / 1.0e6;
    std::printf("%3u  %11u  %13.4f  %8.1f  %17.1f\n",
                benchmark.vms, benchmark.checkpoints,
                elapsed_ms / benchmark.checkpoints, generations_per_vm,
                aggregate_mcells);
  }
  return 0;
}

struct CompiledProfileResult {
  double milliseconds = 0.0;
  size_t ptx_bytes = 0;
  double jit_milliseconds = 0.0;
};

bool MeasureCompiledVariant(const wvm::WvmFile& file,
                            CompiledProfileResult& measurement,
                            std::string& err) {
  constexpr uint32_t kMemoryWords = wvm::kRamSizeWords;
  constexpr uint32_t kCheckpoints = 500;
  wvm::PtxCompiledProgram compiled;
  if (!compiled.Compile(file, err)) return false;
  measurement.ptx_bytes = compiled.ptx().size();
  measurement.jit_milliseconds = compiled.jit_milliseconds();

  std::array<double, 3> samples{};
  for (double& sample : samples) {
    std::vector<wvm::VmState> states(1);
    states[0].status = wvm::kRunning;
    states[0].rng_state = 0x1234567u;
    std::vector<uint32_t> memory(kMemoryWords, 0);
    std::vector<uint32_t> framebuffer(wvm::kVideoWords,
                                       wvm::kVideoResetColor);
    std::vector<uint32_t> frame_seq(1, 0);
    if (!compiled.Launch(states, memory, kMemoryWords, framebuffer, frame_seq,
                         err))
      return false;
    double elapsed_ms = 0.0;
    if (!compiled.LaunchCheckpoints(states, memory, kMemoryWords, framebuffer,
                                    frame_seq, kCheckpoints, elapsed_ms, err))
      return false;
    if (frame_seq[0] != kCheckpoints + 1) {
      err = "compiled profile frame count mismatch";
      return false;
    }
    sample = elapsed_ms / kCheckpoints;
  }
  std::sort(samples.begin(), samples.end());
  measurement.milliseconds = samples[1];
  return true;
}

int RunCompiledLifeProfile(const char* path) {
  wvm::WvmFile file;
  std::string err;
  if (!wvm::LoadWvm(path, file, err)) {
    std::fprintf(stderr, "error: %s: %s\n", path, err.c_str());
    return 2;
  }
  uint32_t evolve_pc = 0;
  uint32_t render_pc = 0;
  uint32_t framebuffer_store_pc = 0;
  uint32_t publish_pc = 0;
  uint32_t expected_evolve_op = 0;
  if (file.code.size() == 215) {
    evolve_pc = 51;
    render_pc = 185;
    framebuffer_store_pc = 205;
    publish_pc = 209;
    expected_evolve_op = wvm::kSMovI;
  } else if (file.code.size() == 206) {
    evolve_pc = 51;
    render_pc = 176;
    framebuffer_store_pc = 196;
    publish_pc = 200;
    expected_evolve_op = wvm::kLaneId;
  }
  if (evolve_pc == 0 ||
      ((file.code[evolve_pc] >> wvm::kOpcodeShift) & wvm::kOpcodeMask) !=
          expected_evolve_op ||
      ((file.code[render_pc] >> wvm::kOpcodeShift) & wvm::kOpcodeMask) !=
          wvm::kLdw ||
      ((file.code[framebuffer_store_pc] >> wvm::kOpcodeShift) &
       wvm::kOpcodeMask) != wvm::kStore) {
    std::fprintf(stderr,
                 "error: compiled profile PCs do not match WarpLife\n");
    return 1;
  }

  wvm::WvmFile evolve_publish = file;
  evolve_publish.code[render_pc] =
      wvm::enc_i(wvm::kJmp, 0, 0, 0, publish_pc);
  wvm::WvmFile render_publish = file;
  render_publish.code[evolve_pc] =
      wvm::enc_i(wvm::kJmp, 0, 0, 0, render_pc);
  wvm::WvmFile publish_only = file;
  publish_only.code[evolve_pc] =
      wvm::enc_i(wvm::kJmp, 0, 0, 0, publish_pc);
  wvm::WvmFile no_framebuffer_write = file;
  no_framebuffer_write.code[framebuffer_store_pc] =
      wvm::enc_r(wvm::kNop, 0, 0, 0, 0);
  wvm::WvmFile no_evolution_loads = file;
  uint32_t removed_loads = 0;
  for (uint32_t pc = evolve_pc; pc < render_pc; ++pc) {
    const uint32_t instruction = no_evolution_loads.code[pc];
    const uint32_t op =
        (instruction >> wvm::kOpcodeShift) & wvm::kOpcodeMask;
    if (op != wvm::kLoad) continue;
    const uint32_t guard =
        (instruction >> wvm::kGuardShift) & wvm::kGuardMask;
    const uint32_t rd =
        (instruction >> wvm::kRdShift) & wvm::kRegFieldMask;
    no_evolution_loads.code[pc] =
        wvm::enc_i(wvm::kMovI, guard, rd, 0, 0);
    ++removed_loads;
  }

  struct Variant {
    const char* name;
    const wvm::WvmFile* file;
    CompiledProfileResult result;
  };
  std::array<Variant, 6> variants{{
      {"full", &file, {}},
      {"evolution + publish", &evolve_publish, {}},
      {"render + publish", &render_publish, {}},
      {"publish/checkpoint only", &publish_only, {}},
      {"framebuffer STORE -> NOP", &no_framebuffer_write, {}},
      {"evolution LOAD -> MOV_I", &no_evolution_loads, {}},
  }};
  for (Variant& variant : variants) {
    if (!MeasureCompiledVariant(*variant.file, variant.result, err)) {
      std::fprintf(stderr, "error: %s: %s\n", variant.name, err.c_str());
      return 1;
    }
  }

  const double full = variants[0].result.milliseconds;
  const double evolve_publish_ms = variants[1].result.milliseconds;
  const double render_publish_ms = variants[2].result.milliseconds;
  const double publish_ms = variants[3].result.milliseconds;
  const double evolve_ms = evolve_publish_ms - publish_ms;
  const double render_ms = render_publish_ms - publish_ms;
  std::printf("compiled WarpLife one-VM profile\n");
  std::printf("  median of 3, 500 YIELD checkpoints per sample\n");
  std::printf("  static evolution LOAD sites replaced: %u\n\n",
              removed_loads);
  std::printf("  variant                       ms/generation  PTX bytes\n");
  for (const Variant& variant : variants)
    std::printf("  %-29s %13.4f  %9zu\n", variant.name,
                variant.result.milliseconds, variant.result.ptx_bytes);
  std::printf("\n  differential attribution\n");
  std::printf("    evolution:             %8.4f ms  (%5.1f%%)\n",
              evolve_ms, 100.0 * evolve_ms / full);
  std::printf("    rendering:             %8.4f ms  (%5.1f%%)\n",
              render_ms, 100.0 * render_ms / full);
  std::printf("    checkpoint/launch:     %8.4f ms  (%5.1f%%)\n",
              publish_ms, 100.0 * publish_ms / full);
  std::printf("    phase sum residual:    %+8.4f ms\n",
              full - evolve_ms - render_ms - publish_ms);
  std::printf("    framebuffer STORE delta:%7.4f ms\n",
              full - variants[4].result.milliseconds);
  std::printf("    evolution LOAD upper bound:%6.4f ms\n",
              full - variants[5].result.milliseconds);
  std::printf("compiled WarpLife profile: PASS\n");
  return 0;
}

void PrintState(const wvm::VmState& st) {
  std::printf("vm %u: status=%s fault=%s pc=%u instrs=%llu\n", st.vm_id,
              wvm::StatusName(st.status), wvm::FaultName(st.fault_code),
              st.pc, static_cast<unsigned long long>(st.instruction_counter));

  for (int r = 0; r < wvm::kVectorRegs; ++r) {
    const uint32_t* v = &st.vregs[r * wvm::kLanes];
    bool uniform = true;
    for (int i = 1; i < wvm::kLanes; ++i)
      if (v[i] != v[0]) uniform = false;
    if (uniform) {
      std::printf("r%-2d = %u (uniform)\n", r, v[0]);
    } else {
      std::printf("r%-2d = [%u, %u, %u, %u, ...] (varied)\n", r, v[0], v[1],
                  v[2], v[3]);
    }
  }

  std::printf("sregs:");
  for (int r = 0; r < wvm::kScalarRegs; ++r) std::printf(" %u", st.sregs[r]);
  std::printf("\npreds:");
  for (int r = 0; r < wvm::kPredRegs; ++r)
    std::printf(" %08x", st.preds[r]);
  std::printf("\n");
}

// ---- commands ----------------------------------------------------------------

int CmdRun(const char* path) {
  wvm::WvmFile file;
  std::string err;
  if (!wvm::LoadWvm(path, file, err)) {
    std::fprintf(stderr, "error: %s: %s\n", path, err.c_str());
    return 2;
  }
  std::printf("program: %zu words, %zu literals\n", file.code.size(),
              file.literals.size());

  wvm::VmState st{};
  ExecProgram(file.code, file.literals, st);
  PrintState(st);
  return st.status == wvm::kHalted ? 0 : 1;
}

void PrintVmTable(const wvm::PersistentRuntime& rt);

// `warpvm serve <file.wvm> [--vms N] [--for SECONDS]` — launch the persistent
// kernel with N copies of the program, boot them, print a `list` table once a
// second, then shut down. Bounded by design (display-GPU friendly).
bool LaunchPersistentEngine(wvm::PersistentRuntime& runtime,
                            const wvm::WvmFile& file, bool compiled,
                            std::unique_ptr<wvm::PtxResidentProgram>& artifact,
                            std::string& err) {
  if (!compiled) return runtime.Launch(err);
  artifact = std::make_unique<wvm::PtxResidentProgram>();
  return runtime.EnsureStream(err) && artifact->Compile(file, err) &&
         artifact->Launch(
             reinterpret_cast<CUdeviceptr>(runtime.DeviceStates()),
             runtime.num_vms(),
             reinterpret_cast<CUdeviceptr>(runtime.DeviceDescs()),
             reinterpret_cast<CUdeviceptr>(runtime.DeviceControl()),
             reinterpret_cast<CUdeviceptr>(runtime.DeviceMailboxes()),
             reinterpret_cast<CUstream>(runtime.Stream()), err);
}

struct ResidentFramebufferSnapshot {
  std::vector<uint32_t> status;
  std::vector<uint32_t> fault;
  std::vector<uint32_t> pc;
  std::vector<uint32_t> frame_seq;
  std::vector<std::vector<uint32_t>> memory;
  std::vector<uint32_t> framebuffers;
};

bool RunResidentToTerminal(const wvm::WvmFile& file, uint32_t num_vms,
                           uint32_t memory_words, bool compiled,
                           ResidentFramebufferSnapshot& snapshot,
                           std::string& err) {
  std::vector<wvm::VmImage> images(num_vms);
  for (wvm::VmImage& image : images) {
    image.code = file.code;
    image.literals = file.literals;
    image.mem_size_words = memory_words;
  }

  wvm::PersistentRuntime runtime;
  std::unique_ptr<wvm::PtxResidentProgram> artifact;
  if (!runtime.Init(images, err) ||
      !LaunchPersistentEngine(runtime, file, compiled, artifact, err))
    return false;

  runtime.BootAll();
  bool terminal = false;
  for (int waited = 0; waited < 5000 && !terminal; ++waited) {
    terminal = true;
    for (uint32_t vm = 0; vm < num_vms; ++vm) {
      const uint32_t status = runtime.Status(vm);
      terminal &= status == wvm::kHalted || status == wvm::kFaulted;
    }
    if (!terminal) std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  bool copied = terminal;
  snapshot.status.resize(num_vms);
  snapshot.fault.resize(num_vms);
  snapshot.pc.resize(num_vms);
  snapshot.frame_seq.resize(num_vms);
  snapshot.memory.resize(num_vms);
  for (uint32_t vm = 0; vm < num_vms; ++vm) {
    snapshot.status[vm] = runtime.Status(vm);
    snapshot.fault[vm] = runtime.Fault(vm);
    snapshot.pc[vm] = runtime.Pc(vm);
    snapshot.frame_seq[vm] = runtime.FrameSeq(vm);
    copied &= runtime.ReadMem(vm, 0, memory_words, snapshot.memory[vm]);
  }
  copied &= runtime.ReadFramebuffers(0, num_vms, snapshot.framebuffers);

  runtime.ShutdownAll();
  const cudaError_t sync = runtime.Sync();
  if (!terminal) {
    err = compiled ? "compiled resident test timed out"
                   : "interpreted resident test timed out";
    return false;
  }
  if (!copied) {
    err = "resident test state copy failed";
    return false;
  }
  if (sync != cudaSuccess) {
    err = std::string("resident test shutdown failed: ") +
          cudaGetErrorString(sync);
    return false;
  }
  return true;
}

int RunCompiledResidentFramebufferTests() {
  constexpr uint32_t kVms = 4;
  constexpr uint32_t kMemoryWords = 64;
  wvm::WvmFile program;
  program.literals = {wvm::kVideoBaseWord};
  program.code = {
      wvm::enc_r(wvm::kLaneId, 0, 0, 0, 0),
      wvm::enc_i(wvm::kLdw, 0, 1, 0, 0),
      wvm::enc_r(wvm::kAdd, 0, 1, 1, 0),
      wvm::enc_r(wvm::kVmid, 0, 2, 0, 0),
      wvm::enc_i(wvm::kAddI, 0, 2, 2, 1),
      wvm::enc_i(wvm::kShlI, 0, 2, 2, 8),
      wvm::enc_r(wvm::kAdd, 0, 2, 2, 0),
      wvm::enc_r(wvm::kStore, 0, 1, 2, 0),
      wvm::enc_r(wvm::kStore, 0, 0, 2, 0),
      wvm::enc_r(wvm::kLoad, 0, 3, 1, 0),
      wvm::enc_i(wvm::kAddI, 0, 4, 0, 32),
      wvm::enc_r(wvm::kStore, 0, 4, 3, 0),
      wvm::enc_r(wvm::kFlip, 0, 0, 0, 0),
      wvm::enc_r(wvm::kHalt, 0, 0, 0, 0),
  };

  std::string err;
  std::string resident_ptx;
  if (!wvm::TranslateWvmToResidentPtx(program, resident_ptx, err)) {
    std::printf("resident framebuffer: translation FAIL: %s\n", err.c_str());
    return 1;
  }
  const std::string correct_guard = "setp.ne.u64 %p6, %rd9, 0;";
  const std::string stale_guard = "setp.ne.u64 %p6, %rd8, 0;";
  size_t guard_count = 0;
  for (size_t pos = resident_ptx.find(correct_guard); pos != std::string::npos;
       pos = resident_ptx.find(correct_guard, pos + correct_guard.size()))
    ++guard_count;
  const bool translation_ok = guard_count == 4 &&
                              resident_ptx.find(stale_guard) ==
                                  std::string::npos;

  ResidentFramebufferSnapshot interpreted;
  ResidentFramebufferSnapshot resident;
  if (!RunResidentToTerminal(program, kVms, kMemoryWords, false, interpreted,
                             err) ||
      !RunResidentToTerminal(program, kVms, kMemoryWords, true, resident,
                             err)) {
    std::printf("resident framebuffer: launch FAIL: %s\n", err.c_str());
    return 1;
  }

  bool valid_store = true;
  bool vm0 = true;
  bool isolated = true;
  bool lane_varying = true;
  bool ram_unchanged = true;
  for (uint32_t vm = 0; vm < kVms; ++vm) {
    valid_store &= resident.status[vm] == wvm::kHalted &&
                   resident.fault[vm] == wvm::kFaultOk &&
                   resident.frame_seq[vm] == 1;
    const uint32_t* framebuffer =
        resident.framebuffers.data() + static_cast<size_t>(vm) *
                                           wvm::kVideoWords;
    for (uint32_t lane = 0; lane < wvm::kLanes; ++lane) {
      const uint32_t expected = ((vm + 1) << 8) + lane;
      lane_varying &= framebuffer[lane] == expected;
      ram_unchanged &= resident.memory[vm][lane] == expected &&
                       resident.memory[vm][32 + lane] == expected;
      if (vm == 0) vm0 &= framebuffer[lane] == 0x100u + lane;
      for (uint32_t other = 0; other < kVms; ++other) {
        if (other == vm) continue;
        const uint32_t* other_framebuffer =
            resident.framebuffers.data() +
            static_cast<size_t>(other) * wvm::kVideoWords;
        isolated &= framebuffer[lane] != other_framebuffer[lane];
      }
    }
    for (uint32_t pixel = wvm::kLanes; pixel < wvm::kVideoWords; ++pixel)
      isolated &= framebuffer[pixel] == wvm::kVideoResetColor;
  }

  bool engine_equal = resident.framebuffers == interpreted.framebuffers &&
                      resident.status == interpreted.status &&
                      resident.fault == interpreted.fault &&
                      resident.frame_seq == interpreted.frame_seq;
  for (uint32_t vm = 0; vm < kVms; ++vm)
    engine_equal &= resident.memory[vm] == interpreted.memory[vm];

  wvm::PtxCompiledProgram one_shot;
  std::vector<wvm::VmState> one_shot_states(kVms);
  for (uint32_t vm = 0; vm < kVms; ++vm) {
    one_shot_states[vm].vm_id = vm;
    one_shot_states[vm].status = wvm::kRunning;
    one_shot_states[vm].rng_state = vm * 0x9E3779B9u + 0x1234567u;
  }
  std::vector<uint32_t> one_shot_memory(kVms * kMemoryWords, 0);
  std::vector<uint32_t> one_shot_framebuffers(
      kVms * wvm::kVideoWords, wvm::kVideoResetColor);
  std::vector<uint32_t> one_shot_frame_seq(kVms, 0);
  bool one_shot_ok = one_shot.Compile(program, err) &&
                     one_shot.Launch(one_shot_states, one_shot_memory,
                                     kMemoryWords, one_shot_framebuffers,
                                     one_shot_frame_seq, err);
  if (one_shot_ok) {
    one_shot_ok &= one_shot_framebuffers == interpreted.framebuffers &&
                   one_shot_frame_seq == interpreted.frame_seq;
    for (uint32_t vm = 0; vm < kVms; ++vm) {
      one_shot_ok &= one_shot_states[vm].status == wvm::kHalted &&
                     one_shot_states[vm].fault_code == wvm::kFaultOk;
      one_shot_ok &= std::equal(
          one_shot_memory.begin() + vm * kMemoryWords,
          one_shot_memory.begin() + (vm + 1) * kMemoryWords,
          interpreted.memory[vm].begin());
    }
  }

  wvm::WvmFile invalid;
  invalid.literals = {wvm::kVideoEndWord};
  invalid.code = {
      wvm::enc_i(wvm::kLdw, 0, 0, 0, 0),
      wvm::enc_i(wvm::kMovI, 0, 1, 0, 1),
      wvm::enc_r(wvm::kStore, 0, 0, 1, 0),
      wvm::enc_r(wvm::kHalt, 0, 0, 0, 0),
  };
  ResidentFramebufferSnapshot invalid_result;
  bool invalid_fault = RunResidentToTerminal(
      invalid, 1, kMemoryWords, true, invalid_result, err);
  if (invalid_fault) {
    invalid_fault = invalid_result.status[0] == wvm::kFaulted &&
                    invalid_result.fault[0] == wvm::kFaultMem &&
                    invalid_result.pc[0] == 2 &&
                    std::all_of(invalid_result.framebuffers.begin(),
                                invalid_result.framebuffers.end(),
                                [](uint32_t value) {
                                  return value == wvm::kVideoResetColor;
                                });
  }

  std::printf("resident framebuffer: translated base guard  %s\n",
              translation_ok ? "PASS" : "FAIL");
  std::printf("resident framebuffer: valid STORE / HALT     %s\n",
              valid_store ? "PASS" : "FAIL");
  std::printf("resident framebuffer: VM 0                  %s\n",
              vm0 ? "PASS" : "FAIL");
  std::printf("resident framebuffer: per-VM isolation      %s\n",
              isolated ? "PASS" : "FAIL");
  std::printf("resident framebuffer: 32 lane-varying STORE %s\n",
              lane_varying ? "PASS" : "FAIL");
  std::printf("resident framebuffer: ordinary RAM STORE    %s\n",
              ram_unchanged ? "PASS" : "FAIL");
  std::printf("resident framebuffer: interpreter equality  %s\n",
              engine_equal ? "PASS" : "FAIL");
  std::printf("resident framebuffer: one-shot equality     %s\n",
              one_shot_ok ? "PASS" : "FAIL");
  std::printf("resident framebuffer: VIDEO_END FAULT_MEM   %s\n",
              invalid_fault ? "PASS" : "FAIL");
  if (!one_shot_ok && !err.empty())
    std::printf("resident framebuffer: one-shot detail: %s\n", err.c_str());

  const bool ok = translation_ok && valid_store && vm0 && isolated &&
                  lane_varying && ram_unchanged && engine_equal &&
                  one_shot_ok && invalid_fault;
  std::printf(ok ? "resident framebuffer tests: PASS\n"
                 : "resident framebuffer tests: FAIL\n");
  return ok ? 0 : 1;
}

struct ResidentMeasurement {
  double frames_per_second = 0.0;
  uint64_t messages_received = 0;
  bool healthy = false;
};

bool MeasureResidentEngine(const wvm::WvmFile& file, uint32_t num_vms,
                           int duration_ms, bool compiled,
                           ResidentMeasurement& measurement,
                           std::string& err) {
  std::vector<wvm::VmImage> images(num_vms);
  for (wvm::VmImage& image : images) {
    image.code = file.code;
    image.literals = file.literals;
    image.mem_size_words = wvm::kRamSizeWords;
  }
  wvm::PersistentRuntime runtime;
  std::unique_ptr<wvm::PtxResidentProgram> artifact;
  if (!runtime.Init(images, err) ||
      !LaunchPersistentEngine(runtime, file, compiled, artifact, err))
    return false;
  runtime.BootAll();
  for (uint32_t vm = 0; vm < num_vms; ++vm) {
    for (int waited = 0; waited < 5000 && runtime.Status(vm) == wvm::kIdle;
         ++waited)
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  std::vector<uint32_t> before(num_vms);
  for (uint32_t vm = 0; vm < num_vms; ++vm) before[vm] = runtime.FrameSeq(vm);
  const auto start = std::chrono::steady_clock::now();
  std::this_thread::sleep_for(std::chrono::milliseconds(duration_ms));
  const auto stop = std::chrono::steady_clock::now();
  uint64_t frames = 0;
  measurement.healthy = true;
  for (uint32_t vm = 0; vm < num_vms; ++vm) {
    frames += runtime.FrameSeq(vm) - before[vm];
    measurement.healthy &= runtime.Status(vm) == wvm::kRunning &&
                           runtime.Fault(vm) == wvm::kFaultOk;
    std::vector<uint32_t> words;
    if (runtime.ReadMem(vm, 98, 1, words) && !words.empty())
      measurement.messages_received += words[0];
  }
  const double seconds =
      std::chrono::duration<double>(stop - start).count();
  measurement.frames_per_second =
      static_cast<double>(frames) / num_vms / seconds;
  runtime.ShutdownAll();
  if (runtime.Sync() != cudaSuccess) {
    err = "resident benchmark shutdown failed";
    return false;
  }
  return true;
}

int RunResidentBenchmark(const char* path, uint32_t num_vms,
                         int duration_ms) {
  wvm::WvmFile file;
  std::string err;
  if (!wvm::LoadWvm(path, file, err)) {
    std::fprintf(stderr, "resident_bench: %s\n", err.c_str());
    return 2;
  }
  ResidentMeasurement interpreted, compiled;
  if (!MeasureResidentEngine(file, num_vms, duration_ms, false, interpreted,
                             err) ||
      !MeasureResidentEngine(file, num_vms, duration_ms, true, compiled,
                             err)) {
    std::fprintf(stderr, "resident_bench: %s\n", err.c_str());
    return 1;
  }
  const double speedup = compiled.frames_per_second /
                         interpreted.frames_per_second;
  const bool pass = interpreted.healthy && compiled.healthy &&
                    interpreted.frames_per_second > 0 &&
                    compiled.frames_per_second > 0 &&
                    compiled.messages_received > 0;
  std::printf("resident messaging/graphics benchmark: %u VMs, %.3fs\n",
              num_vms, duration_ms / 1000.0);
  std::printf("  interpreted       %9.2f average frames/s  messages=%llu\n",
              interpreted.frames_per_second,
              static_cast<unsigned long long>(interpreted.messages_received));
  std::printf("  compiled resident %9.2f average frames/s  messages=%llu\n",
              compiled.frames_per_second,
              static_cast<unsigned long long>(compiled.messages_received));
  std::printf("  compiled speedup  %9.2fx\n", speedup);
  std::printf(pass ? "resident_bench PASS\n" : "resident_bench FAIL\n");
  return pass ? 0 : 1;
}

int CmdServe(const char* path, uint32_t n_vms, int seconds, bool compiled) {
  wvm::WvmFile file;
  std::string err;
  if (!wvm::LoadWvm(path, file, err)) {
    std::fprintf(stderr, "error: %s: %s\n", path, err.c_str());
    return 2;
  }
  if (n_vms == 0 || n_vms > wvm::kMaxVms) {
    std::fprintf(stderr, "error: --vms must be 1..%u\n", wvm::kMaxVms);
    return 2;
  }

  std::vector<wvm::VmImage> images(n_vms);
  for (uint32_t i = 0; i < n_vms; ++i) {
    images[i].code = file.code;
    images[i].literals = file.literals;
    images[i].mem_size_words = wvm::kRamSizeWords;
  }

  wvm::PersistentRuntime rt;
  std::unique_ptr<wvm::PtxResidentProgram> compiled_artifact;
  if (!rt.Init(images, err) ||
      !LaunchPersistentEngine(rt, file, compiled, compiled_artifact, err)) {
    std::fprintf(stderr, "error: %s\n", err.c_str());
    return 1;
  }
  rt.BootAll();
  std::printf("serving %u %s VMs from %s for %ds\n", n_vms,
              compiled ? "compiled resident" : "interpreted", path,
              seconds);

  for (int t = 1; t <= seconds; ++t) {
    std::this_thread::sleep_for(std::chrono::seconds(1));
    std::printf("t=%ds\n", t);
    PrintVmTable(rt);
  }

  rt.ShutdownAll();
  const cudaError_t e = rt.Sync();
  std::printf("shutdown: %s\n", e == cudaSuccess ? "ok" : cudaGetErrorString(e));
  return e == cudaSuccess ? 0 : 1;
}

int CmdHeterogeneousSmoke(const std::vector<std::string>& paths) {
  if (paths.empty() || paths.size() > wvm::kMaxVms) {
    std::fprintf(stderr, "error: hetero_smoke requires 1..%u programs\n",
                 wvm::kMaxVms);
    return 2;
  }

  wvm::ProgramRegistry registry;
  wvm::VmDirectory directory(static_cast<uint32_t>(paths.size()));
  std::vector<wvm::VmBinding> bindings(paths.size());
  std::vector<std::pair<std::string, wvm::LoadedProgramId>> loaded;
  std::string err;
  for (wvm::VmSlot slot = 0; slot < paths.size(); ++slot) {
    wvm::LoadedProgramId program_id;
    bool found = false;
    for (const auto& item : loaded) {
      if (item.first == paths[slot]) {
        program_id = item.second;
        found = true;
        break;
      }
    }
    if (!found) {
      const std::string name = "program-" + std::to_string(loaded.size());
      if (!registry.Load(paths[slot], name, program_id, err)) {
        std::fprintf(stderr, "error: %s: %s\n", paths[slot].c_str(),
                     err.c_str());
        return 2;
      }
      loaded.emplace_back(paths[slot], program_id);
    }
    wvm::LogicalVmId vm_id;
    if (!directory.Create(wvm::ResidentSlotId{slot}, vm_id, err) ||
        !registry.Retain(program_id, err)) {
      std::fprintf(stderr, "error: %s\n", err.c_str());
      return 1;
    }
    bindings[slot].vm_id = vm_id;
    bindings[slot].program_id = program_id;
  }

  wvm::PersistentRuntime rt;
  if (!rt.Init(registry, bindings, err) || !rt.Launch(err)) {
    std::fprintf(stderr, "hetero_smoke: %s\n", err.c_str());
    return 1;
  }
  rt.BootAll();

  bool published = false;
  for (int waited = 0; waited < 15000 && !published; ++waited) {
    published = true;
    for (wvm::VmSlot slot = 0; slot < paths.size(); ++slot) {
      if (rt.Status(slot) == wvm::kFaulted) break;
      published &= rt.FrameSeq(slot) > 0;
    }
    if (!published)
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  bool healthy = published;
  for (wvm::VmSlot slot = 0; slot < paths.size(); ++slot)
    healthy &= rt.Status(slot) == wvm::kRunning;

  // Stop one machine and prove another continues publishing. This exercises
  // heterogeneous lifecycle isolation without changing reset/delete policy.
  bool isolated_stop = true;
  if (paths.size() > 1 && healthy) {
    isolated_stop = rt.Pause(1, 5000);
    const uint32_t stopped_frame = rt.FrameSeq(1);
    std::vector<uint32_t> other_frames(paths.size());
    for (wvm::VmSlot slot = 0; slot < paths.size(); ++slot)
      other_frames[slot] = rt.FrameSeq(slot);
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    bool other_progress = false;
    for (wvm::VmSlot slot = 0; slot < paths.size(); ++slot) {
      if (slot != 1 && rt.FrameSeq(slot) > other_frames[slot])
        other_progress = true;
    }
    isolated_stop &= rt.Status(1) == wvm::kPaused &&
                     rt.FrameSeq(1) == stopped_frame && other_progress;
    rt.SendCmd(1, wvm::kCmdRun);
    isolated_stop &= rt.WaitStatus(1, wvm::kRunning, 5000);
  }

  std::printf("heterogeneous smoke: %zu VMs, %zu registry programs, %zu "
              "device uploads\n",
              paths.size(), registry.size(), rt.device_program_count());
  PrintVmTable(rt);
  std::printf("  all_publish_frames: %s\n", healthy ? "PASS" : "FAIL");
  std::printf("  isolated_stop:      %s\n",
              isolated_stop ? "PASS" : "FAIL");

  rt.ShutdownAll();
  const bool shutdown = rt.Sync() == cudaSuccess;
  const bool pass = healthy && isolated_stop && shutdown;
  std::printf(pass ? "heterogeneous smoke: PASS\n"
                   : "heterogeneous smoke: FAIL\n");
  return pass ? 0 : 1;
}

void PrintVectorRegs(const wvm::VmState& st) {
  for (int r = 0; r < wvm::kVectorRegs; ++r) {
    const uint32_t* v = &st.vregs[r * wvm::kLanes];
    bool uniform = true;
    for (int i = 1; i < wvm::kLanes; ++i)
      if (v[i] != v[0]) uniform = false;
    if (uniform)
      std::printf("r%-2d = %u (uniform)\n", r, v[0]);
    else
      std::printf("r%-2d = [%u, %u, %u, %u, ...] (varied)\n", r, v[0], v[1],
                  v[2], v[3]);
  }
}

void PrintScalarRegs(const wvm::VmState& st) {
  for (int r = 0; r < wvm::kScalarRegs; ++r)
    std::printf("s%d = %u\n", r, st.sregs[r]);
}

void PrintPreds(const wvm::VmState& st) {
  for (int r = 0; r < wvm::kPredRegs; ++r)
    std::printf("p%d = %08x\n", r, st.preds[r]);
}

std::vector<std::string> Tokenize(const std::string& line) {
  std::vector<std::string> toks;
  std::istringstream iss(line);
  std::string t;
  while (iss >> t) toks.push_back(t);
  return toks;
}

// `warpvm attach <file.wvm> [--vms N]` — launch the persistent kernel, boot
// the VMs, then run an interactive console reading commands from stdin.
int CmdAttach(const char* path, uint32_t n_vms, bool compiled) {
  wvm::WvmFile file;
  std::string err;
  if (!wvm::LoadWvm(path, file, err)) {
    std::fprintf(stderr, "error: %s: %s\n", path, err.c_str());
    return 2;
  }
  if (n_vms == 0 || n_vms > wvm::kMaxVms) {
    std::fprintf(stderr, "error: --vms must be 1..%u\n", wvm::kMaxVms);
    return 2;
  }
  std::vector<wvm::VmImage> images(n_vms);
  for (uint32_t i = 0; i < n_vms; ++i) {
    images[i].code = file.code;
    images[i].literals = file.literals;
    images[i].mem_size_words = wvm::kRamSizeWords;
  }
  wvm::PersistentRuntime rt;
  std::unique_ptr<wvm::PtxResidentProgram> compiled_artifact;
  if (!rt.Init(images, err) ||
      !LaunchPersistentEngine(rt, file, compiled, compiled_artifact, err)) {
    std::fprintf(stderr, "error: %s\n", err.c_str());
    return 1;
  }
  rt.BootAll();
  // The command channel is a single word per VM, so we must let each warp
  // consume its RUN (leave IDLE) before the console can send anything else —
  // otherwise a later command would overwrite the unconsumed RUN.
  for (uint32_t i = 0; i < n_vms; ++i) {
    for (int w = 0; w < 2000 && rt.Status(i) == wvm::kIdle; ++w)
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  std::printf("attached: %u %s VMs from %s\n", n_vms,
              compiled ? "compiled resident" : "interpreted", path);
  std::printf("type 'help' for commands, 'quit' to exit\n");

  uint32_t cur = 0;
  auto ensure_paused = [&](uint32_t vm) -> bool {
    if (rt.Status(vm) == wvm::kPaused) return true;
    return rt.Pause(vm);
  };

  std::string line;
  while (true) {
    std::printf("vm-%u> ", cur);
    std::fflush(stdout);
    if (!std::getline(std::cin, line)) break;  // EOF
    const std::vector<std::string> tok = Tokenize(line);
    if (tok.empty()) continue;
    const std::string& c = tok[0];

    if (c == "quit" || c == "exit") break;
    else if (c == "help") {
      std::printf("  list                 status of all VMs\n");
      std::printf("  vm <id>              select VM to inspect\n");
      std::printf("  status               status/fault/pc/instrs of current VM\n");
      std::printf("  pause | halt         pause current VM\n");
      std::printf("  resume | continue    resume current VM\n");
      std::printf("  step [n]             single-step current VM n times\n");
      std::printf("  reset                reset current VM\n");
      std::printf("  regs | sregs | preds register dumps (pause first)\n");
      std::printf("  pc                   program counter + current instr\n");
      std::printf("  frame                display resolution/format/frame_seq\n");
      std::printf("  pixel <x> <y>        read one framebuffer word\n");
      std::printf("  disasm [start] [n]   disassemble around pc\n");
      std::printf("  mem <addr> <count>   dump VM RAM words\n");
      std::printf("  log [n]              recent log entries\n");
      std::printf("  quit                 shut down and exit\n");
    } else if (c == "list") {
      PrintVmTable(rt);
    } else if (c == "vm") {
      if (tok.size() < 2) { std::printf("usage: vm <id>\n"); continue; }
      const uint32_t id = static_cast<uint32_t>(std::atoi(tok[1].c_str()));
      if (id >= n_vms) { std::printf("vm id out of range\n"); continue; }
      cur = id;
    } else if (c == "status") {
      std::printf("vm %u: status=%s fault=%s pc=%u instrs=%llu\n", cur,
                  wvm::StatusName(rt.Status(cur)),
                  wvm::FaultName(rt.Fault(cur)), rt.Pc(cur),
                  (unsigned long long)rt.Instrs(cur));
    } else if (c == "pause" || c == "halt") {
      std::printf("%s\n", rt.Pause(cur) ? "paused" : "pause timed out");
    } else if (c == "resume" || c == "continue" || c == "run") {
      rt.SendCmd(cur, wvm::kCmdRun);
      std::printf("resuming\n");
    } else if (c == "step") {
      if (compiled) {
        std::printf("single-step requires the interpreter engine\n");
        continue;
      }
      if (!ensure_paused(cur)) { std::printf("could not pause\n"); continue; }
      int n = tok.size() >= 2 ? std::atoi(tok[1].c_str()) : 1;
      if (n < 1) n = 1;
      uint32_t s = wvm::kPaused;
      for (int i = 0; i < n; ++i) {
        s = rt.Step(cur);
        if (s != wvm::kPaused) break;
      }
      std::printf("stepped -> status=%s pc=%u instrs=%llu\n",
                  wvm::StatusName(s), rt.Pc(cur),
                  (unsigned long long)rt.Instrs(cur));
    } else if (c == "reset") {
      if (!ensure_paused(cur)) { std::printf("could not pause\n"); continue; }
      rt.SendCmd(cur, wvm::kCmdReset);
      std::printf("resetting\n");
    } else if (c == "regs" || c == "sregs" || c == "preds") {
      wvm::VmState st{};
      if (!rt.ReadState(cur, st)) { std::printf("read failed\n"); continue; }
      if (rt.Status(cur) != wvm::kPaused && rt.Status(cur) != wvm::kHalted &&
          rt.Status(cur) != wvm::kFaulted)
        std::printf("(VM not stopped — snapshot may be stale)\n");
      if (c == "regs") PrintVectorRegs(st);
      else if (c == "sregs") PrintScalarRegs(st);
      else PrintPreds(st);
    } else if (c == "pc") {
      const uint32_t pc = rt.Pc(cur);
      std::printf("pc=%u status=%s\n", pc, wvm::StatusName(rt.Status(cur)));
      const auto& code = rt.Code(cur);
      if (pc < code.size())
        std::printf("  %u: %s\n", pc,
                    wvm::DisasmWord(code[pc], rt.Literals(cur)).c_str());
    } else if (c == "disasm") {
      const auto& code = rt.Code(cur);
      const auto& lits = rt.Literals(cur);
      uint32_t start = rt.Pc(cur) >= 2 ? rt.Pc(cur) - 2 : 0;
      uint32_t count = 8;
      if (tok.size() >= 2) start = static_cast<uint32_t>(std::atoi(tok[1].c_str()));
      if (tok.size() >= 3) count = static_cast<uint32_t>(std::atoi(tok[2].c_str()));
      const uint32_t pc = rt.Pc(cur);
      for (uint32_t i = start; i < code.size() && i < start + count; ++i)
        std::printf("%s %4u: %s\n", i == pc ? ">" : " ", i,
                    wvm::DisasmWord(code[i], lits).c_str());
    } else if (c == "mem") {
      if (tok.size() < 3) { std::printf("usage: mem <addr> <count>\n"); continue; }
      const uint32_t addr = static_cast<uint32_t>(std::atoi(tok[1].c_str()));
      const uint32_t count = static_cast<uint32_t>(std::atoi(tok[2].c_str()));
      std::vector<uint32_t> words;
      if (!rt.ReadMem(cur, addr, count, words)) {
        std::printf("mem read failed (out of range?)\n");
        continue;
      }
      for (size_t i = 0; i < words.size(); i += 8) {
        std::printf("  [%5zu]:", addr + i);
        for (size_t j = i; j < i + 8 && j < words.size(); ++j)
          std::printf(" %10u", words[j]);
        std::printf("\n");
      }
    } else if (c == "frame") {
      std::printf("resolution=%ux%u\n", wvm::kVideoWidth, wvm::kVideoHeight);
      std::printf("format=ARGB8888\n");
      std::printf("frame_seq=%u\n", rt.FrameSeq(cur));
    } else if (c == "pixel") {
      if (tok.size() < 3) { std::printf("usage: pixel <x> <y>\n"); continue; }
      const uint32_t x = static_cast<uint32_t>(std::atoi(tok[1].c_str()));
      const uint32_t y = static_cast<uint32_t>(std::atoi(tok[2].c_str()));
      if (x >= wvm::kVideoWidth || y >= wvm::kVideoHeight) {
        std::printf("pixel out of range\n");
        continue;
      }
      std::vector<uint32_t> fb;
      if (!rt.ReadFramebuffer(cur, fb)) {
        std::printf("framebuffer read failed\n");
        continue;
      }
      const uint32_t v = fb[y * wvm::kVideoWidth + x];
      std::printf("pixel(%u,%u) = 0x%08x\n", x, y, v);
    } else if (c == "log") {
      const int n = tok.size() >= 2 ? std::atoi(tok[1].c_str()) : 16;
      const wvm::LogSnapshot snap = rt.ReadLog();
      const size_t start = snap.entries.size() > (size_t)n
                               ? snap.entries.size() - (size_t)n : 0;
      for (size_t i = start; i < snap.entries.size(); ++i) {
        const auto& e = snap.entries[i];
        std::printf("  vm=%u tag=%u value=%u\n", e.vm_id, e.tag, e.value);
      }
      std::printf("(%u total entries)\n", snap.head);
    } else {
      std::printf("unknown command '%s' (try 'help')\n", c.c_str());
    }
  }

  rt.ShutdownAll();
  const cudaError_t e = rt.Sync();
  std::printf("shutdown: %s\n", e == cudaSuccess ? "ok" : cudaGetErrorString(e));
  return e == cudaSuccess ? 0 : 1;
}

bool CheckState(const char* name, const wvm::VmState& st, uint32_t status,
                uint32_t fault, uint32_t r0_expect) {
  const bool ok = st.status == status && st.fault_code == fault &&
                  st.vregs[0 * wvm::kLanes] == r0_expect;
  std::printf("  %-22s %s (status=%s fault=%s r0=%u)\n", name,
              ok ? "PASS" : "FAIL", wvm::StatusName(st.status),
              wvm::FaultName(st.fault_code), st.vregs[0 * wvm::kLanes]);
  return ok;
}

int RunSlice2() {
  using wvm::enc_i;
  using wvm::enc_r;
  bool ok = true;
  wvm::VmState st{};

  // MOV/ADD/HALT end to end: r0=42 uniform, r1=42, r2=84.
  {
    const std::vector<uint32_t> prog = {
        enc_i(wvm::kMovI, 0, 0, 0, 7),    // MOV_I r0, 7
        enc_i(wvm::kAddI, 0, 0, 0, 35),   // ADD_I r0, r0, 35
        enc_r(wvm::kMov, 0, 1, 0, 0),     // MOV   r1, r0
        enc_r(wvm::kAdd, 0, 2, 0, 1),     // ADD   r2, r0, r1
        enc_r(wvm::kHalt, 0, 0, 0, 0),    // HALT
    };
    ExecProgram(prog, {}, st);
    ok &= CheckState("mov_add_halt", st, wvm::kHalted, wvm::kFaultOk, 42);
    ok &= st.vregs[1 * wvm::kLanes] == 42 && st.vregs[2 * wvm::kLanes] == 84 &&
          st.instruction_counter == 5;
  }

  // LDW from the literal pool.
  {
    const std::vector<uint32_t> prog = {
        enc_i(wvm::kLdw, 0, 3, 0, 0),     // LDW r3, lit[0]
        enc_r(wvm::kHalt, 0, 0, 0, 0),    // HALT
    };
    const std::vector<uint32_t> lits = {0xDEADBEEFu};
    ExecProgram(prog, lits, st);
    ok &= CheckState("ldw_literal", st, wvm::kHalted, wvm::kFaultOk, 0);
    ok &= st.vregs[3 * wvm::kLanes] == 0xDEADBEEFu;
  }

  // Invalid opcode faults the VM.
  {
    const std::vector<uint32_t> prog = {enc_r(0x7Fu, 0, 0, 0, 0)};
    ExecProgram(prog, {}, st);
    ok &= CheckState("invalid_opcode_faults", st, wvm::kFaulted,
                     wvm::kFaultOpcode, 0);
    ok &= st.instruction_counter == 0;
  }

  // Falling off the end without HALT faults.
  {
    const std::vector<uint32_t> prog = {enc_i(wvm::kMovI, 0, 0, 0, 1)};
    ExecProgram(prog, {}, st);
    ok &= CheckState("fall_off_end_faults", st, wvm::kFaulted, wvm::kFaultJump,
                     1);
  }

  // A fault raised by only one guarded lane must still fault the whole VM.
  // This protects the per-lane fault-vote contract during future tuning.
  {
    const std::vector<uint32_t> prog = {
        enc_r(wvm::kLaneId, 0, 0, 0, 0),       // r0 = lane
        enc_i(wvm::kCmpEqI, 0, 0, 0, 0),       // p0 = lane 0 only
        enc_i(wvm::kLdw, 1, 1, 0, 0),          // @p0 r1 = bad RAM address
        enc_r(wvm::kLoad, 1, 2, 1, 0),         // @p0 LOAD -> lane-local fault
        enc_r(wvm::kHalt, 0, 0, 0, 0),
    };
    ExecProgram(prog, {wvm::kRamSizeWords}, st);
    ok &= CheckState("guarded_lane_mem_fault", st, wvm::kFaulted,
                     wvm::kFaultMem, 0);
    ok &= st.instruction_counter == 3;
  }

  // A guarded literal miss is also lane-local before the retained vote.
  {
    const std::vector<uint32_t> prog = {
        enc_r(wvm::kLaneId, 0, 0, 0, 0),
        enc_i(wvm::kCmpEqI, 0, 0, 0, 0),
        enc_i(wvm::kLdw, 1, 1, 0, 0),          // empty literal pool
        enc_r(wvm::kHalt, 0, 0, 0, 0),
    };
    ExecProgram(prog, {}, st);
    ok &= CheckState("guarded_literal_fault", st, wvm::kFaulted,
                     wvm::kFaultOperand, 0);
    ok &= st.instruction_counter == 2;
  }

  std::printf(ok ? "slice2: PASS\n" : "slice2: FAIL\n");
  return ok ? 0 : 1;
}

int RunSlice3() {
  using wvm::enc_i;
  using wvm::enc_r;
  bool ok = true;

  // Program: r0 = mem[0] + delta; mem[1] = r0. 6 words.
  auto make_prog = [](uint32_t delta) {
    return std::vector<uint32_t>{
        enc_i(wvm::kMovI, 0, 1, 0, 0),      // MOV_I r1, 0   (addr)
        enc_r(wvm::kLoad, 0, 0, 1, 0),      // LOAD  r0, r1
        enc_i(wvm::kAddI, 0, 0, 0, static_cast<int32_t>(delta)),
        enc_i(wvm::kMovI, 0, 2, 0, 1),      // MOV_I r2, 1   (addr)
        enc_r(wvm::kStore, 0, 2, 0, 0),     // STORE r2, r0
        enc_r(wvm::kHalt, 0, 0, 0, 0),      // HALT
    };
  };
  const std::vector<uint32_t> prog_even = make_prog(7);
  const std::vector<uint32_t> prog_odd = make_prog(11);

  // ---- 64 VMs, stable IDs, private RAM, distinct programs/data -----------
  {
    const uint32_t N = 64;
    std::vector<wvm::VmImage> images(N);
    for (uint32_t i = 0; i < N; ++i) {
      images[i].code = (i % 2 == 0) ? prog_even : prog_odd;
      images[i].mem_size_words = 8;
      images[i].mem_init = {1000u * i + 5u};
    }
    std::vector<wvm::VmState> states;
    std::vector<std::vector<uint32_t>> mem;
    ExecVmArray(images, states, mem);

    uint32_t bad = 0;
    for (uint32_t i = 0; i < N; ++i) {
      const uint32_t seed = 1000u * i + 5u;
      const uint32_t delta = (i % 2 == 0) ? 7u : 11u;
      const wvm::VmState& st = states[i];
      bool good = st.vm_id == i && st.status == wvm::kHalted &&
                  st.fault_code == wvm::kFaultOk &&
                  st.instruction_counter == 6;
      for (int lane = 0; lane < wvm::kLanes; ++lane)
        good &= st.vregs[0 * wvm::kLanes + lane] == seed + delta;
      good &= mem[i][0] == seed && mem[i][1] == seed + delta;
      for (uint32_t w = 2; w < 8; ++w) good &= mem[i][w] == 0;
      if (!good) {
        if (bad < 4)
          std::printf("  vm %u WRONG: status=%s fault=%s vm_id=%u r0=%u "
                      "mem[0]=%u mem[1]=%u (want seed=%u delta=%u)\n",
                      i, wvm::StatusName(st.status),
                      wvm::FaultName(st.fault_code), st.vm_id,
                      st.vregs[0], mem[i][0], mem[i][1], seed, delta);
        ++bad;
      }
    }
    std::printf("  %-22s %s (%u/%u VMs correct)\n", "64vm_independent",
                bad == 0 ? "PASS" : "FAIL", N - bad, N);
    ok &= bad == 0;
  }

  // ---- Fault isolation: VM 5 has 1-word RAM, neighbours must be fine -----
  {
    const uint32_t N = 16;
    const uint32_t bad_vm = 5;
    std::vector<wvm::VmImage> images(N);
    for (uint32_t i = 0; i < N; ++i) {
      images[i].code = prog_even;
      images[i].mem_size_words = (i == bad_vm) ? 1 : 8;
      images[i].mem_init = {1000u * i + 5u};
    }
    std::vector<wvm::VmState> states;
    std::vector<std::vector<uint32_t>> mem;
    ExecVmArray(images, states, mem);

    bool isolated = states[bad_vm].status == wvm::kFaulted &&
                    states[bad_vm].fault_code == wvm::kFaultMem;
    uint32_t bad = 0;
    for (uint32_t i = 0; i < N; ++i) {
      if (i == bad_vm) continue;
      const uint32_t seed = 1000u * i + 5u;
      const wvm::VmState& st = states[i];
      bool good = st.vm_id == i && st.status == wvm::kHalted &&
                  st.vregs[0 * wvm::kLanes] == seed + 7u &&
                  mem[i][1] == seed + 7u;
      if (!good) ++bad;
    }
    std::printf("  %-22s %s (vm %u -> %s/%s, %u neighbours disturbed)\n",
                "fault_isolation", (isolated && bad == 0) ? "PASS" : "FAIL",
                bad_vm, wvm::StatusName(states[bad_vm].status),
                wvm::FaultName(states[bad_vm].fault_code), bad);
    ok &= isolated && bad == 0;
  }

  std::printf(ok ? "slice3: PASS\n" : "slice3: FAIL\n");
  return ok ? 0 : 1;
}

int RunSlice4() {
  using wvm::enc_i;
  using wvm::enc_r;
  bool ok = true;
  wvm::VmState st{};

  // Reductions over lane ids: sum 0..31 = 496, max = 31, xor = 0.
  {
    const std::vector<uint32_t> prog = {
        enc_r(wvm::kLaneId, 0, 0, 0, 0),      // LANEID r0
        enc_r(wvm::kReduceAdd, 0, 1, 0, 0),   // REDUCE_ADD r1, r0
        enc_r(wvm::kReduceMax, 0, 2, 0, 0),   // REDUCE_MAX r2, r0
        enc_r(wvm::kReduceXor, 0, 3, 0, 0),   // REDUCE_XOR r3, r0
        enc_r(wvm::kHalt, 0, 0, 0, 0),
    };
    ExecProgram(prog, {}, st);
    const bool pass = st.status == wvm::kHalted &&
                      st.vregs[1 * wvm::kLanes] == 496 &&
                      st.vregs[2 * wvm::kLanes] == 31 &&
                      st.vregs[3 * wvm::kLanes] == 0;
    std::printf("  %-22s %s (sum=%u max=%u xor=%u)\n", "reductions",
                pass ? "PASS" : "FAIL", st.vregs[1 * wvm::kLanes],
                st.vregs[2 * wvm::kLanes], st.vregs[3 * wvm::kLanes]);
    ok &= pass;
  }

  // Broadcast / shuffle: pull specific lanes.
  {
    const std::vector<uint32_t> prog = {
        enc_r(wvm::kLaneId, 0, 0, 0, 0),      // LANEID r0
        enc_i(wvm::kBroadcast, 0, 1, 0, 5),   // BROADCAST r1, r0, #5 -> 5
        enc_i(wvm::kMovI, 0, 2, 0, 3),        // MOV_I r2, 3
        enc_r(wvm::kShuffle, 0, 3, 0, 2),     // SHUFFLE r3, r0, r2 -> 3
        enc_i(wvm::kShuffleXor, 0, 4, 0, 1),  // SHUFFLE_XOR r4, r0, #1
        enc_r(wvm::kHalt, 0, 0, 0, 0),
    };
    ExecProgram(prog, {}, st);
    const bool pass = st.status == wvm::kHalted &&
                      st.vregs[1 * wvm::kLanes] == 5 &&
                      st.vregs[3 * wvm::kLanes] == 3 &&
                      st.vregs[4 * wvm::kLanes + 0] == 1 &&  // lane0 ^ 1 = 1
                      st.vregs[4 * wvm::kLanes + 1] == 0;    // lane1 ^ 1 = 0
    std::printf("  %-22s %s (bcast=%u shuf=%u xor[0]=%u xor[1]=%u)\n",
                "broadcast_shuffle", pass ? "PASS" : "FAIL",
                st.vregs[1 * wvm::kLanes], st.vregs[3 * wvm::kLanes],
                st.vregs[4 * wvm::kLanes + 0], st.vregs[4 * wvm::kLanes + 1]);
    ok &= pass;
  }

  // Compare -> predicate mask -> guarded execution; ballot reads it back.
  {
    const std::vector<uint32_t> prog = {
        enc_r(wvm::kLaneId, 0, 0, 0, 0),       // LANEID r0
        enc_i(wvm::kCmpLtI, 0, 0, 0, 16),      // CMP_LT p0, r0, #16 (lanes 0-15)
        enc_i(wvm::kMovI, 0, 1, 0, 7),         // MOV_I r1, 7
        enc_i(wvm::kAddI, 1, 1, 1, 1),         // @p0 ADD_I r1, r1, 1
        enc_r(wvm::kBallot, 0, 1, 1, 0),       // BALLOT p1, r1 (all nonzero)
        enc_r(wvm::kHalt, 0, 0, 0, 0),
    };
    ExecProgram(prog, {}, st);
    const bool pass = st.status == wvm::kHalted &&
                      st.preds[0] == 0x0000FFFFu &&
                      st.preds[1] == 0xFFFFFFFFu &&
                      st.vregs[1 * wvm::kLanes + 0] == 8 &&   // active lane -> 8
                      st.vregs[1 * wvm::kLanes + 15] == 8 &&
                      st.vregs[1 * wvm::kLanes + 16] == 7 &&  // inactive -> stays 7
                      st.vregs[1 * wvm::kLanes + 31] == 7;
    std::printf("  %-22s %s (p0=%08x p1=%08x r1[0]=%u r1[31]=%u)\n",
                "cmp_mask_guard", pass ? "PASS" : "FAIL", st.preds[0],
                st.preds[1], st.vregs[1 * wvm::kLanes + 0],
                st.vregs[1 * wvm::kLanes + 31]);
    ok &= pass;
  }

  // Scalar loop counter driven by JMP_IF_ANY.
  {
    const std::vector<uint32_t> prog = {
        enc_i(wvm::kSMovI, 0, 0, 0, 0),        // S_MOV_I s0, 0
        enc_i(wvm::kSAddI, 0, 0, 0, 1),        // loop: S_ADD_I s0, s0, 1
        enc_i(wvm::kSCmpLtI, 0, 0, 0, 5),      // S_CMP_LT_I p0, s0, 5
        enc_i(wvm::kJmpIfAny, 1, 0, 0, 1),     // JMP_IF_ANY p0, loop(=1)
        enc_r(wvm::kHalt, 0, 0, 0, 0),
    };
    ExecProgram(prog, {}, st);
    const bool pass = st.status == wvm::kHalted && st.sregs[0] == 5 &&
                      st.instruction_counter == 17;
    std::printf("  %-22s %s (s0=%u instrs=%llu)\n", "scalar_loop",
                pass ? "PASS" : "FAIL", st.sregs[0],
                (unsigned long long)st.instruction_counter);
    ok &= pass;
  }

  // CALL / RET round trip.
  {
    const std::vector<uint32_t> prog = {
        enc_i(wvm::kMovI, 0, 0, 0, 10),        // 0: MOV_I r0, 10
        enc_i(wvm::kCall, 0, 0, 0, 4),         // 1: CALL sub(=4)
        enc_i(wvm::kAddI, 0, 0, 0, 1),         // 2: ADD_I r0, r0, 1
        enc_r(wvm::kHalt, 0, 0, 0, 0),         // 3: HALT
        enc_i(wvm::kMulI, 0, 0, 0, 2),         // 4: sub: MUL_I r0, r0, 2
        enc_r(wvm::kRet, 0, 0, 0, 0),          // 5: RET
    };
    ExecProgram(prog, {}, st);
    const bool pass = st.status == wvm::kHalted &&
                      st.vregs[0 * wvm::kLanes] == 21 && st.call_depth == 0;
    std::printf("  %-22s %s (r0=%u depth=%u)\n", "call_ret",
                pass ? "PASS" : "FAIL", st.vregs[0 * wvm::kLanes],
                st.call_depth);
    ok &= pass;
  }

  // RET on an empty call stack faults.
  {
    const std::vector<uint32_t> prog = {enc_r(wvm::kRet, 0, 0, 0, 0),
                                        enc_r(wvm::kHalt, 0, 0, 0, 0)};
    ExecProgram(prog, {}, st);
    ok &= CheckState("ret_empty_stack", st, wvm::kFaulted, wvm::kFaultStack, 0);
  }

  // Jump target past the end faults.
  {
    const std::vector<uint32_t> prog = {enc_i(wvm::kJmp, 0, 0, 0, 100),
                                        enc_r(wvm::kHalt, 0, 0, 0, 0)};
    ExecProgram(prog, {}, st);
    ok &= CheckState("jmp_out_of_range", st, wvm::kFaulted, wvm::kFaultJump, 0);
  }

  // Comparison into an invalid predicate field faults.
  {
    const std::vector<uint32_t> prog = {enc_r(wvm::kCmpLt, 0, 5, 0, 0),
                                        enc_r(wvm::kHalt, 0, 0, 0, 0)};
    ExecProgram(prog, {}, st);
    ok &= CheckState("bad_pred_field", st, wvm::kFaulted, wvm::kFaultOperand, 0);
  }

  std::printf(ok ? "slice4: PASS\n" : "slice4: FAIL\n");
  return ok ? 0 : 1;
}

void PrintVmTable(const wvm::PersistentRuntime& rt) {
  std::printf("  %-6s %-4s %-16s %-9s %-8s %-6s %s\n", "vm_id", "slot",
              "program", "status", "fault", "pc", "instrs");
  for (wvm::VmSlot slot = 0; slot < rt.num_vms(); ++slot) {
    std::printf("  %-6u %-4u %-16.16s %-9s %-8s %-6u %llu\n",
                rt.VmIdAtSlot(slot), slot,
                rt.ProgramNameAtSlot(slot).c_str(),
                wvm::StatusName(rt.Status(slot)),
                wvm::FaultName(rt.Fault(slot)), rt.Pc(slot),
                (unsigned long long)rt.Instrs(slot));
  }
}

int RunSlice5() {
  using wvm::enc_i;
  using wvm::enc_r;
  bool ok = true;
  auto sleep_ms = [](int ms) {
    std::this_thread::sleep_for(std::chrono::milliseconds(ms));
  };
  auto report = [&ok](const char* name, bool pass, const char* detail = "") {
    if (detail[0])
      std::printf("  %-22s %s (%s)\n", name, pass ? "PASS" : "FAIL", detail);
    else
      std::printf("  %-22s %s\n", name, pass ? "PASS" : "FAIL");
    ok &= pass;
  };

  // Heartbeat: infinite loop, counter in r5, stored to mem[0], logged, with
  // a YIELD and a backward JMP as pause control points.
  const std::vector<uint32_t> heartbeat = {
      enc_i(wvm::kMovI, 0, 5, 0, 0),   // 0: MOV_I r5, 0
      enc_i(wvm::kAddI, 0, 5, 5, 1),   // 1: beat: ADD_I r5, r5, 1
      enc_i(wvm::kMovI, 0, 0, 0, 0),   // 2: MOV_I r0, 0
      enc_r(wvm::kStore, 0, 0, 5, 0),  // 3: STORE r0, r5
      enc_i(wvm::kLogI, 0, 0, 5, 7),   // 4: LOG_I r5, #7
      enc_r(wvm::kYield, 0, 0, 0, 0),  // 5: YIELD
      enc_i(wvm::kJmp, 0, 0, 0, 1),    // 6: JMP beat
  };

  const uint32_t N = 8;
  std::vector<wvm::VmImage> images(N);
  for (uint32_t i = 0; i < N; ++i) {
    images[i].code = heartbeat;
    images[i].mem_size_words = 8;
  }

  wvm::PersistentRuntime rt;
  std::string err;
  if (!rt.Init(images, err) || !rt.Launch(err)) {
    std::fprintf(stderr, "slice5: setup failed: %s\n", err.c_str());
    return 1;
  }

  // Boot all VMs; they should reach RUNNING.
  rt.BootAll();
  bool all_running = true;
  for (uint32_t i = 0; i < N; ++i)
    all_running &= rt.WaitStatus(i, wvm::kRunning, 2000);
  report("boot_all_running", all_running);

  // Running VMs make progress: instruction counter advances.
  const uint64_t a0 = rt.Instrs(0);
  sleep_ms(20);
  const uint64_t b0 = rt.Instrs(0);
  char buf[96];
  std::snprintf(buf, sizeof(buf), "instrs %llu -> %llu",
                (unsigned long long)a0, (unsigned long long)b0);
  report("running_progress", b0 > a0, buf);

  // Pause VM 0: it halts at a control point and its counter freezes.
  rt.SendCmd(0, wvm::kCmdPause);
  const bool paused = rt.WaitStatus(0, wvm::kPaused, 2000);
  const uint64_t p1 = rt.Instrs(0);
  sleep_ms(20);
  const uint64_t p2 = rt.Instrs(0);
  std::snprintf(buf, sizeof(buf), "frozen at %llu", (unsigned long long)p2);
  report("pause_freezes", paused && p2 == p1, buf);

  // Resume VM 0: counter advances again.
  rt.SendCmd(0, wvm::kCmdRun);
  const bool resumed = rt.WaitStatus(0, wvm::kRunning, 2000);
  sleep_ms(20);
  const uint64_t r0 = rt.Instrs(0);
  report("resume_progress", resumed && r0 > p2);

  // Reset VM 0 (pause first, then reset): returns to IDLE.
  rt.SendCmd(0, wvm::kCmdPause);
  rt.WaitStatus(0, wvm::kPaused, 2000);
  rt.SendCmd(0, wvm::kCmdReset);
  report("reset_to_idle", rt.WaitStatus(0, wvm::kIdle, 2000));

  // Log ring received entries tagged 7 from the heartbeats.
  const wvm::LogSnapshot snap = rt.ReadLog();
  bool has_tag7 = false;
  for (const auto& e : snap.entries)
    if (e.tag == 7) has_tag7 = true;
  std::snprintf(buf, sizeof(buf), "%u entries", snap.head);
  report("vm_log", snap.head > 0 && has_tag7, buf);

  // `warpvm list` view: vm 0 idle after reset, others still running.
  std::printf("  --- warpvm list ---\n");
  PrintVmTable(rt);

  // Shutdown: every warp exits and the kernel completes.
  rt.ShutdownAll();
  const cudaError_t sync_err = rt.Sync();
  report("clean_shutdown", sync_err == cudaSuccess);

  std::printf(ok ? "slice5: PASS\n" : "slice5: FAIL\n");
  return ok ? 0 : 1;
}

int RunHeterogeneousProgramTests() {
  using wvm::enc_i;
  using wvm::enc_r;
  bool ok = true;
  auto report = [&ok](const char* name, bool pass) {
    std::printf("  %-28s %s\n", name, pass ? "PASS" : "FAIL");
    ok &= pass;
  };

  auto make_program = [&](uint32_t sentinel, uint32_t colour,
                          uint32_t extra_nops) {
    wvm::WvmFile file;
    file.literals = {sentinel, wvm::kVideoBaseWord, colour};
    file.code = {
        enc_i(wvm::kLdw, 0, 0, 0, 0),
        enc_i(wvm::kMovI, 0, 1, 0, 0),
        enc_r(wvm::kStore, 0, 1, 0, 0),
        enc_r(wvm::kVmid, 0, 2, 0, 0),
        enc_i(wvm::kMovI, 0, 3, 0, 1),
        enc_r(wvm::kStore, 0, 3, 2, 0),
        enc_i(wvm::kLdw, 0, 4, 0, 1),
        enc_i(wvm::kLdw, 0, 5, 0, 2),
        enc_r(wvm::kStore, 0, 4, 5, 0),
        enc_r(wvm::kFlip, 0, 0, 0, 0),
    };
    for (uint32_t i = 0; i < extra_nops; ++i)
      file.code.push_back(enc_r(wvm::kNop, 0, 0, 0, 0));
    file.code.push_back(enc_r(wvm::kHalt, 0, 0, 0, 0));
    return file;
  };

  wvm::ProgramRegistry registry;
  wvm::LoadedProgramId red, green, blue;
  std::string err;
  const bool loaded =
      registry.Add("red", "red.wvm",
                   make_program(101, 0xFFFF2020u, 0), red, err) &&
      registry.Add("green", "green.wvm",
                   make_program(202, 0xFF20FF20u, 1), green, err) &&
      registry.Add("blue", "blue.wvm",
                   make_program(303, 0xFF2020FFu, 2), blue, err);
  report("registry_loads_three", loaded && registry.size() == 3 &&
                                     registry.Find("green") != nullptr);
  if (!loaded) return 1;

  const bool retained = registry.Retain(red, err) &&
                        registry.Retain(green, err) &&
                        registry.Retain(red, err) &&
                        registry.Retain(blue, err);
  const bool unload_rejected = retained && !registry.Unload(red, err) &&
                               registry.Find(red) != nullptr;
  report("referenced_unload_rejected", unload_rejected);

  const std::vector<wvm::VmBinding> bindings = {
      {wvm::LogicalVmId{10}, red},
      {wvm::LogicalVmId{21}, green},
      {wvm::LogicalVmId{35}, red},
      {wvm::LogicalVmId{48}, blue},
  };
  wvm::PersistentRuntime rt;
  if (!rt.Init(registry, bindings, err) || !rt.Launch(err)) {
    std::fprintf(stderr, "heterogeneous programs: setup failed: %s\n",
                 err.c_str());
    return 1;
  }
  report("three_device_program_uploads", rt.device_program_count() == 3);
  report("per_slot_program_identity",
         rt.ProgramIdAtSlot(0) == red.value &&
             rt.ProgramIdAtSlot(1) == green.value &&
             rt.ProgramIdAtSlot(2) == red.value &&
             rt.ProgramIdAtSlot(3) == blue.value &&
             rt.ProgramNameAtSlot(2) == "red");

  wvm::VmDesc desc0{}, desc1{}, desc2{};
  const bool shared_upload = rt.ReadDescriptor(0, desc0) &&
                             rt.ReadDescriptor(1, desc1) &&
                             rt.ReadDescriptor(2, desc2) &&
                             desc0.code == desc2.code &&
                             desc0.literals == desc2.literals &&
                             desc0.code != desc1.code;
  report("shared_immutable_device_image", shared_upload);

  rt.BootAll();
  bool halted = true;
  for (wvm::VmSlot slot = 0; slot < bindings.size(); ++slot)
    halted &= rt.WaitStatus(slot, wvm::kHalted, 2000);
  report("heterogeneous_concurrent_halt", halted);

  const uint32_t expected_sentinels[] = {101, 202, 101, 303};
  const uint32_t expected_colours[] = {0xFFFF2020u, 0xFF20FF20u,
                                       0xFFFF2020u, 0xFF2020FFu};
  bool isolated = true;
  for (wvm::VmSlot slot = 0; slot < bindings.size(); ++slot) {
    std::vector<uint32_t> memory;
    std::vector<uint32_t> framebuffer;
    isolated &= rt.ReadMem(slot, 0, 2, memory) && memory.size() == 2 &&
                memory[0] == expected_sentinels[slot] &&
                memory[1] == bindings[slot].vm_id.value &&
                rt.ReadFramebuffer(slot, framebuffer) &&
                framebuffer[0] == expected_colours[slot] &&
                rt.FrameSeq(slot) == 1;
  }
  report("private_state_and_framebuffers", isolated);

  wvm::VmState state0{}, state1{}, state3{};
  const bool independent_fetch =
      rt.ReadState(0, state0) && rt.ReadState(1, state1) &&
      rt.ReadState(3, state3) && state0.pc != state1.pc &&
      state1.pc != state3.pc;
  report("independent_program_fetch_pc", independent_fetch);

  rt.ShutdownAll();
  report("heterogeneous_clean_shutdown", rt.Sync() == cudaSuccess);

  bool released = registry.Release(red, err) && registry.Release(green, err) &&
                  registry.Release(red, err) && registry.Release(blue, err);
  const wvm::ProgramId retired_red = red.value;
  released &= registry.Unload(red, err) && registry.Find(red) == nullptr;
  wvm::LoadedProgramId replacement;
  released &= registry.Add("replacement", "replacement.wvm",
                           make_program(404, 0xFFFFFFFFu, 0), replacement,
                           err) &&
              replacement.value != retired_red;
  report("program_release_unload_nonreuse", released);

  std::printf(ok ? "heterogeneous programs: PASS\n"
                 : "heterogeneous programs: FAIL\n");
  return ok ? 0 : 1;
}

int RunSupervisorLifecycleTests() {
  using wvm::enc_i;
  using wvm::enc_r;
  bool ok = true;
  auto report = [&](const char* name, bool pass) {
    std::printf("  %-30s %s\n", name, pass ? "PASS" : "FAIL");
    ok &= pass;
  };
  auto image = [](std::initializer_list<uint32_t> code,
                  std::initializer_list<uint32_t> literals = {}) {
    wvm::WvmFile result;
    result.code.assign(code);
    result.literals.assign(literals);
    return result;
  };

  // Increments RAM[0], yields, and repeats. The backward jump is also a
  // control point, so stop/delete acknowledgement is prompt.
  const wvm::WvmFile counter = image({
      enc_i(wvm::kMovI, 0, 0, 0, 0),
      enc_r(wvm::kLoad, 0, 1, 0, 0),
      enc_i(wvm::kAddI, 0, 1, 1, 1),
      enc_r(wvm::kStore, 0, 0, 1, 0),
      enc_r(wvm::kYield, 0, 0, 0, 0),
      enc_i(wvm::kJmp, 0, 0, 0, 1),
  });
  const wvm::WvmFile halter = image({
      enc_r(wvm::kVmid, 0, 0, 0, 0),
      enc_i(wvm::kMovI, 0, 1, 0, 0),
      enc_r(wvm::kStore, 0, 1, 0, 0),
      enc_r(wvm::kHalt, 0, 0, 0, 0),
  });
  const wvm::WvmFile faulter = image({
      enc_r(wvm::kRet, 0, 0, 0, 0),
  });
  // A replacement VM probes its fresh mailbox once and records whether it
  // inherited a message intended for the prior logical owner of its slot.
  const wvm::WvmFile receiver = image({
      enc_i(wvm::kMovI, 0, 0, 0, 0),
      enc_r(wvm::kTryRecv, 0, 0, 1, 2),
      enc_i(wvm::kJmpIfAny, 1, 0, 0, 6),
      enc_i(wvm::kMovI, 0, 3, 0, 0),
      enc_r(wvm::kStore, 0, 0, 3, 0),
      enc_r(wvm::kHalt, 0, 0, 0, 0),
      enc_i(wvm::kMovI, 0, 3, 0, 1),
      enc_r(wvm::kStore, 0, 0, 3, 0),
      enc_r(wvm::kHalt, 0, 0, 0, 0),
  });
  // Logical destination 1 will have been retired when this program runs.
  const wvm::WvmFile stale_sender = image({
      enc_i(wvm::kMovI, 0, 0, 0, 1),
      enc_i(wvm::kMovI, 0, 1, 0, 7),
      enc_i(wvm::kMovI, 0, 2, 0, 999),
      enc_r(wvm::kSend, 0, 0, 1, 2),
      enc_r(wvm::kHalt, 0, 0, 0, 0),
  });
  const wvm::WvmFile race_sender = image({
      enc_r(wvm::kVmid, 0, 0, 0, 0),
      enc_i(wvm::kAddI, 0, 0, 0, 1),  // destination is next-created VM
      enc_i(wvm::kMovI, 0, 1, 0, 12),
      enc_i(wvm::kMovI, 0, 2, 0, 777),
      enc_r(wvm::kSend, 0, 0, 1, 2),
      enc_i(wvm::kMovI, 0, 3, 0, 0),
      enc_i(wvm::kAddI, 0, 3, 3, 1),
      enc_i(wvm::kCmpLtI, 0, 0, 3, 4000),
      enc_i(wvm::kJmpIfAny, 1, 0, 0, 6),
      enc_r(wvm::kYield, 0, 0, 0, 0),
      enc_i(wvm::kJmp, 0, 0, 0, 4),
  });
  const wvm::WvmFile drain_receiver = image({
      enc_r(wvm::kTryRecv, 0, 0, 0, 1),
      enc_r(wvm::kYield, 0, 0, 0, 0),
      enc_i(wvm::kJmp, 0, 0, 0, 0),
  });

  wvm::Supervisor supervisor;
  wvm::LoadedProgramId counter_id, halt_id, fault_id, receiver_id, sender_id,
      race_sender_id, drain_id;
  std::string err;
  const bool programs_loaded =
      supervisor.ProgramAdd("counter", counter, counter_id, err) &&
      supervisor.ProgramAdd("halter", halter, halt_id, err) &&
      supervisor.ProgramAdd("faulter", faulter, fault_id, err) &&
      supervisor.ProgramAdd("receiver", receiver, receiver_id, err) &&
      supervisor.ProgramAdd("stale-sender", stale_sender, sender_id, err) &&
      supervisor.ProgramAdd("race-sender", race_sender, race_sender_id, err) &&
      supervisor.ProgramAdd("drain-receiver", drain_receiver, drain_id, err);
  report("program_registry_before_launch", programs_loaded);
  if (!programs_loaded || !supervisor.Launch(3, err)) {
    std::fprintf(stderr, "supervisor lifecycle setup failed: %s\n",
                 err.c_str());
    return 1;
  }

  wvm::LogicalVmId counter_vm, halt_vm, fault_vm;
  const bool created =
      supervisor.VmCreate(counter_id, counter_vm, err, 16, {41}) &&
      supervisor.VmCreate(halt_id, halt_vm, err, 16) &&
      supervisor.VmCreate(fault_id, fault_vm, err, 16);
  report("empty_create_ready", created && counter_vm.value == 0 &&
                                      halt_vm.value == 1 &&
                                      fault_vm.value == 2 &&
                                      supervisor.Find(counter_vm)->lifecycle ==
                                          wvm::VmLifecycle::kReady);
  wvm::LogicalVmId over_capacity;
  report("fixed_capacity_enforced",
         !supervisor.VmCreate(halt_id, over_capacity, err));
  report("referenced_program_not_unloaded",
         !supervisor.ProgramUnload(counter_id, err));

  bool counter_cycle = supervisor.VmStart(counter_vm, err);
  counter_cycle &= supervisor.runtime().WaitStatus(0, wvm::kRunning, 2000);
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  counter_cycle &= supervisor.VmStop(counter_vm, err);
  std::vector<uint32_t> before_pause, during_pause, after_resume;
  counter_cycle &= supervisor.runtime().ReadMem(0, 0, 1, before_pause);
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  counter_cycle &= supervisor.runtime().ReadMem(0, 0, 1, during_pause);
  counter_cycle &= supervisor.VmResume(counter_vm, err);
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  counter_cycle &= supervisor.VmStop(counter_vm, err);
  counter_cycle &= supervisor.runtime().ReadMem(0, 0, 1, after_resume);
  report("stop_preserves_resume_continues",
         counter_cycle && before_pause[0] > 41 &&
             during_pause[0] == before_pause[0] &&
             after_resume[0] > during_pause[0]);

  bool reset_ok = supervisor.VmReset(counter_vm, err);
  std::vector<uint32_t> reset_memory;
  reset_ok &= supervisor.runtime().ReadMem(0, 0, 1, reset_memory);
  report("reset_restores_initial_state",
         reset_ok && reset_memory[0] == 41 &&
             supervisor.Find(counter_vm)->lifecycle ==
                 wvm::VmLifecycle::kReady);

  bool terminal_states = supervisor.VmStart(halt_vm, err) &&
                         supervisor.runtime().WaitStatus(1, wvm::kHalted,
                                                         2000) &&
                         supervisor.Find(halt_vm)->lifecycle ==
                             wvm::VmLifecycle::kHalted &&
                         supervisor.VmStart(fault_vm, err) &&
                         supervisor.runtime().WaitStatus(2, wvm::kFaulted,
                                                         2000) &&
                         supervisor.Find(fault_vm)->lifecycle ==
                             wvm::VmLifecycle::kFaulted;
  report("halted_and_faulted_states", terminal_states);

  bool neighbour_isolation = supervisor.VmStart(counter_vm, err);
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  const uint64_t peer_before = supervisor.runtime().Instrs(0);
  neighbour_isolation &= supervisor.VmDelete(halt_vm, err);
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  const uint64_t peer_after = supervisor.runtime().Instrs(0);
  report("delete_does_not_stop_neighbour",
         neighbour_isolation && peer_after > peer_before &&
             supervisor.runtime().SlotForVmId(halt_vm.value) ==
                 wvm::kInvalidVmSlot);

  wvm::LogicalVmId replacement_vm;
  bool recycled = supervisor.VmCreate(receiver_id, replacement_vm, err, 16);
  recycled &= replacement_vm.value > fault_vm.value &&
              supervisor.Find(replacement_vm)->slot.value == 1 &&
              supervisor.runtime().SlotForVmId(halt_vm.value) ==
                  wvm::kInvalidVmSlot;
  report("slot_reuse_gets_fresh_identity", recycled);

  bool stale_safe = supervisor.VmDelete(fault_vm, err);
  wvm::LogicalVmId sender_vm;
  stale_safe &= supervisor.VmCreate(sender_id, sender_vm, err, 16);
  stale_safe &= supervisor.VmStart(sender_vm, err);
  stale_safe &= supervisor.runtime().WaitStatus(2, wvm::kFaulted, 2000);
  stale_safe &= supervisor.runtime().Fault(2) == wvm::kFaultMsg;
  stale_safe &= supervisor.VmStart(replacement_vm, err);
  stale_safe &= supervisor.runtime().WaitStatus(1, wvm::kHalted, 2000);
  std::vector<uint32_t> replacement_memory;
  stale_safe &= supervisor.runtime().ReadMem(1, 0, 1, replacement_memory);
  report("retired_messages_not_retargeted",
         stale_safe && replacement_memory[0] == 0);

  const wvm::VmId preserved_id = replacement_vm.value;
  bool rebound = supervisor.VmSetProgram(replacement_vm, halt_id, err);
  rebound &= supervisor.Find(replacement_vm)->vm_id.value == preserved_id &&
             supervisor.Find(replacement_vm)->program_id.value ==
                 halt_id.value &&
             supervisor.Find(replacement_vm)->lifecycle ==
                 wvm::VmLifecycle::kReady;
  rebound &= supervisor.VmStart(replacement_vm, err) &&
             supervisor.runtime().WaitStatus(1, wvm::kHalted, 2000);
  std::vector<uint32_t> rebound_memory;
  rebound &= supervisor.runtime().ReadMem(1, 0, 1, rebound_memory);
  report("cold_rebind_preserves_identity",
         rebound && rebound_memory[0] == preserved_id);

  bool cleared_population = supervisor.VmDelete(replacement_vm, err) &&
                            supervisor.VmDelete(sender_vm, err) &&
                            supervisor.VmDelete(counter_vm, err);
  wvm::LogicalVmId race_sender_vm, drain_vm;
  const bool race_created =
      cleared_population &&
      supervisor.VmCreate(race_sender_id, race_sender_vm, err, 16) &&
      supervisor.VmCreate(drain_id, drain_vm, err, 16) &&
      drain_vm.value == race_sender_vm.value + 1;
  const bool receiver_running =
      race_created && supervisor.VmStart(drain_vm, err) &&
      supervisor.runtime().WaitStatus(1, wvm::kRunning, 2000);
  const bool race_setup = receiver_running &&
                          supervisor.VmStart(race_sender_vm, err);
  const bool both_running =
      race_setup && supervisor.runtime().WaitStatus(0, wvm::kRunning, 2000);
  const bool race_delete = both_running && supervisor.VmDelete(drain_vm, err);
  const bool sender_rejected =
      race_delete && supervisor.runtime().WaitStatus(0, wvm::kFaulted, 2000) &&
      supervisor.runtime().Fault(0) == wvm::kFaultMsg;
  wvm::LogicalVmId race_replacement;
  const bool replacement_created =
      sender_rejected &&
      supervisor.VmCreate(receiver_id, race_replacement, err, 16) &&
      supervisor.Find(race_replacement)->slot.value == 1 &&
      supervisor.VmStart(race_replacement, err) &&
      supervisor.runtime().WaitStatus(1, wvm::kHalted, 2000);
  std::vector<uint32_t> race_replacement_memory;
  const bool replacement_read = replacement_created &&
      supervisor.runtime().ReadMem(1, 0, 1, race_replacement_memory);
  const bool racing_retirement =
      replacement_read && race_replacement_memory[0] == 0;
  if (!racing_retirement || race_replacement_memory.empty() ||
      race_replacement_memory[0] != 0) {
    std::printf("    race detail: setup=%u running=%u delete=%u reject=%u "
                "create=%u read=%u sender_status=%s fault=%s replacement=%u "
                "error=%s\n",
                race_setup, both_running, race_delete, sender_rejected,
                replacement_created, replacement_read,
                wvm::StatusName(supervisor.runtime().Status(0)),
                wvm::FaultName(supervisor.runtime().Fault(0)),
                race_replacement_memory.empty() ? 0xFFFFFFFFu
                                                : race_replacement_memory[0],
                err.c_str());
  }
  report("concurrent_retire_send_is_safe",
         racing_retirement && race_replacement_memory[0] == 0);

  bool cleanup = supervisor.VmDelete(race_replacement, err) &&
                 supervisor.VmDelete(race_sender_vm, err) &&
                 supervisor.ProgramUnload(sender_id, err) &&
                 supervisor.ProgramUnload(counter_id, err) &&
                 supervisor.ProgramUnload(halt_id, err) &&
                 supervisor.ProgramUnload(fault_id, err) &&
                 supervisor.ProgramUnload(receiver_id, err) &&
                 supervisor.ProgramUnload(race_sender_id, err) &&
                 supervisor.ProgramUnload(drain_id, err);
  report("delete_releases_program_refs", cleanup);
  supervisor.Shutdown();
  std::printf(ok ? "supervisor lifecycle: PASS\n"
                 : "supervisor lifecycle: FAIL\n");
  return ok ? 0 : 1;
}

int RunSlice7() {
  using wvm::enc_i;
  using wvm::enc_r;
  bool ok = true;

  // Logical identities survive slot reuse: retiring VM 0 frees its resident
  // slot, but the replacement receives a fresh address and the old route
  // remains invalid.
  {
    wvm::VmDirectory directory(2);
    wvm::LogicalVmId first, peer, replacement;
    std::string directory_err;
    const bool created =
        directory.Create(wvm::ResidentSlotId{0}, first, directory_err) &&
        directory.Create(wvm::ResidentSlotId{1}, peer, directory_err);
    const wvm::VmId retired_id = first.value;
    const bool recycled =
        created && directory.Retire(first, directory_err) &&
        directory.Create(wvm::ResidentSlotId{0}, replacement, directory_err);
    const bool identity_ok =
        recycled && replacement.value != retired_id &&
        directory.SlotFor(wvm::LogicalVmId{retired_id}) ==
            wvm::kInvalidVmSlot &&
        directory.SlotFor(replacement) == 0 &&
        directory.IdFor(wvm::ResidentSlotId{1}) == peer.value;
    std::printf("  %-22s %s\n", "stable_identity_directory",
                identity_ok ? "PASS" : "FAIL");
    ok &= identity_ok;
  }

  // VM 0 sends payload 1234 to VM 1; VM 1 receives it and stores mem[0].
  // All other VMs halt immediately.
  const std::vector<uint32_t> prog = {
      enc_r(wvm::kVmid, 0, 0, 0, 0),        // 0: VMID r0
      enc_i(wvm::kCmpEqI, 0, 0, 0, 0),      // 1: CMP_EQ p0, r0, #0
      enc_i(wvm::kJmpIfAny, 1, 0, 0, 6),    // 2: JMP_IF_ANY p0, sender(6)
      enc_i(wvm::kCmpEqI, 0, 1, 0, 1),      // 3: CMP_EQ p1, r0, #1
      enc_i(wvm::kJmpIfAny, 2, 0, 0, 11),   // 4: JMP_IF_ANY p1, receiver(11)
      enc_r(wvm::kHalt, 0, 0, 0, 0),        // 5: HALT (other VMs)
      enc_i(wvm::kMovI, 0, 1, 0, 1),        // 6: sender: MOV_I r1, 1 (dest)
      enc_i(wvm::kMovI, 0, 2, 0, 7),        // 7: MOV_I r2, 7 (type)
      enc_i(wvm::kMovI, 0, 3, 0, 1234),     // 8: MOV_I r3, 1234 (payload)
      enc_r(wvm::kSend, 0, 1, 2, 3),        // 9: SEND r1, r2, r3
      enc_r(wvm::kHalt, 0, 0, 0, 0),        // 10: HALT
      enc_r(wvm::kTryRecv, 0, 2, 4, 5),     // 11: recv: TRY_RECV p2, r4, r5
      enc_r(wvm::kNotMask, 0, 3, 2, 0),     // 12: NOTMASK p3, p2
      enc_i(wvm::kJmpIfAny, 4, 0, 0, 11),   // 13: JMP_IF_ANY p3, recv(11)
      enc_i(wvm::kMovI, 0, 6, 0, 0),        // 14: MOV_I r6, 0 (addr)
      enc_r(wvm::kStore, 0, 6, 4, 0),       // 15: STORE r6, r4 (mem[0]=payload)
      enc_r(wvm::kHalt, 0, 0, 0, 0),        // 16: HALT
  };

  const uint32_t N = 4;
  std::vector<wvm::VmImage> images(N);
  for (uint32_t i = 0; i < N; ++i) {
    images[i].code = prog;
    images[i].mem_size_words = 16;
  }
  wvm::PersistentRuntime rt;
  std::string err;
  if (!rt.Init(images, err) || !rt.Launch(err)) {
    std::fprintf(stderr, "slice7: setup failed: %s\n", err.c_str());
    return 1;
  }
  rt.BootAll();

  // Wait for every VM to halt (sender/receiver handshake completes).
  bool all_halted = true;
  for (int waited = 0;; ++waited) {
    all_halted = true;
    for (uint32_t i = 0; i < N; ++i) {
      const uint32_t s = rt.Status(i);
      if (s != wvm::kHalted && s != wvm::kFaulted) all_halted = false;
    }
    if (all_halted || waited > 5000) break;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  if (!all_halted) {
    std::printf("  %-22s FAIL (timeout waiting for halt)\n", "messaging");
    rt.ShutdownAll();
    rt.Sync();
    return 1;
  }

  // VM 1 must have received VM 0's payload (1234) in mem[0]; no faults.
  std::vector<uint32_t> mem;
  const bool read_ok = rt.ReadMem(1, 0, 1, mem);
  const bool delivered = read_ok && !mem.empty() && mem[0] == 1234;
  bool no_faults = true;
  for (uint32_t i = 0; i < N; ++i)
    no_faults &= rt.Status(i) == wvm::kHalted;
  char buf[96];
  std::snprintf(buf, sizeof(buf), "vm1 mem[0]=%u", read_ok && !mem.empty() ? mem[0] : 0);
  std::printf("  %-22s %s (%s)\n", "vm_send_recv",
              (delivered && no_faults) ? "PASS" : "FAIL", buf);
  ok &= delivered && no_faults;

  rt.ShutdownAll();
  const cudaError_t e = rt.Sync();
  ok &= e == cudaSuccess;

  // A false guard suppresses consumption, so the subsequent unguarded
  // receive must still observe the self-sent message.
  const std::vector<uint32_t> guarded_prog = {
      enc_i(wvm::kMovI, 0, 0, 0, 0),
      enc_i(wvm::kMovI, 0, 1, 0, 7),
      enc_i(wvm::kMovI, 0, 2, 0, 1234),
      enc_r(wvm::kSend, 0, 0, 1, 2),
      enc_i(wvm::kCmpEqI, 0, 0, 0, 1),
      enc_r(wvm::kTryRecv, 1, 1, 3, 4),
      enc_r(wvm::kTryRecv, 0, 2, 5, 6),
      enc_i(wvm::kMovI, 0, 7, 0, 0),
      enc_r(wvm::kStore, 0, 7, 5, 0),
      enc_r(wvm::kHalt, 0, 0, 0, 0),
  };
  std::vector<wvm::VmImage> guarded_images(1);
  guarded_images[0].code = guarded_prog;
  guarded_images[0].mem_size_words = 16;
  wvm::PersistentRuntime guarded_rt;
  if (!guarded_rt.Init(guarded_images, err) || !guarded_rt.Launch(err)) {
    std::printf("  %-22s FAIL (%s)\n", "guarded_receive", err.c_str());
    return 1;
  }
  guarded_rt.BootAll();
  guarded_rt.WaitStatus(0, wvm::kHalted, 2000);
  std::vector<uint32_t> guarded_mem;
  const bool guarded_ok = guarded_rt.Status(0) == wvm::kHalted &&
                          guarded_rt.ReadMem(0, 0, 1, guarded_mem) &&
                          guarded_mem[0] == 1234;
  std::printf("  %-22s %s\n", "guarded_receive",
              guarded_ok ? "PASS" : "FAIL");
  ok &= guarded_ok;
  guarded_rt.ShutdownAll();
  ok &= guarded_rt.Sync() == cudaSuccess;

  // Logical addresses are not resident slots. Slot 0 (VM 37) sends to VM 91,
  // which currently occupies slot 1; the receiver observes logical sender 37.
  const std::vector<uint32_t> logical_prog = {
      enc_r(wvm::kVmid, 0, 0, 0, 0),
      enc_i(wvm::kCmpEqI, 0, 0, 0, 37),
      enc_i(wvm::kJmpIfAny, 1, 0, 0, 6),
      enc_i(wvm::kCmpEqI, 0, 1, 0, 91),
      enc_i(wvm::kJmpIfAny, 2, 0, 0, 11),
      enc_r(wvm::kHalt, 0, 0, 0, 0),
      enc_i(wvm::kMovI, 0, 1, 0, 91),
      enc_i(wvm::kMovI, 0, 2, 0, 7),
      enc_i(wvm::kMovI, 0, 3, 0, 2345),
      enc_r(wvm::kSend, 0, 1, 2, 3),
      enc_r(wvm::kHalt, 0, 0, 0, 0),
      enc_r(wvm::kTryRecv, 0, 2, 4, 5),
      enc_r(wvm::kNotMask, 0, 3, 2, 0),
      enc_i(wvm::kJmpIfAny, 4, 0, 0, 11),
      enc_i(wvm::kMovI, 0, 6, 0, 0),
      enc_r(wvm::kStore, 0, 6, 4, 0),
      enc_i(wvm::kMovI, 0, 6, 0, 1),
      enc_r(wvm::kStore, 0, 6, 5, 0),
      enc_r(wvm::kFlip, 0, 0, 0, 0),
      enc_r(wvm::kHalt, 0, 0, 0, 0),
  };
  std::vector<wvm::VmImage> logical_images(2);
  for (wvm::VmImage& image : logical_images) {
    image.code = logical_prog;
    image.mem_size_words = 16;
  }
  const std::vector<wvm::LogicalVmId> logical_ids = {{37}, {91}};
  wvm::PersistentRuntime logical_rt;
  if (!logical_rt.Init(logical_images, logical_ids, err) ||
      !logical_rt.Launch(err)) {
    std::printf("  %-22s FAIL (%s)\n", "logical_message_route",
                err.c_str());
    return 1;
  }
  logical_rt.BootAll();
  logical_rt.WaitStatus(0, wvm::kHalted, 2000);
  logical_rt.WaitStatus(1, wvm::kHalted, 2000);
  std::vector<uint32_t> logical_mem;
  wvm::VmState logical_sender{}, logical_receiver{};
  const bool logical_ok =
      logical_rt.Status(0) == wvm::kHalted &&
      logical_rt.Status(1) == wvm::kHalted &&
      logical_rt.ReadMem(1, 0, 2, logical_mem) &&
      logical_mem.size() == 2 && logical_mem[0] == 2345 &&
      logical_mem[1] == ((7u << 16) | 37u) &&
      logical_rt.FrameSeq(1) == 1 &&
      logical_rt.ReadState(0, logical_sender) &&
      logical_rt.ReadState(1, logical_receiver) &&
      logical_sender.vm_id == 37 && logical_receiver.vm_id == 91 &&
      logical_rt.SlotForVmId(37) == 0 &&
      logical_rt.SlotForVmId(91) == 1;
  std::printf("  %-22s %s\n", "logical_message_route",
              logical_ok ? "PASS" : "FAIL");
  ok &= logical_ok;
  logical_rt.ShutdownAll();
  ok &= logical_rt.Sync() == cudaSuccess;

  // Address 1 represents a retired VM while its former slot is occupied by
  // logical VM 91. A stale send must fault rather than reach slot 1.
  const std::vector<uint32_t> stale_prog = {
      enc_r(wvm::kVmid, 0, 0, 0, 0),
      enc_i(wvm::kCmpEqI, 0, 0, 0, 37),
      enc_i(wvm::kJmpIfAny, 1, 0, 0, 4),
      enc_r(wvm::kHalt, 0, 0, 0, 0),
      enc_i(wvm::kMovI, 0, 1, 0, 1),
      enc_i(wvm::kMovI, 0, 2, 0, 9),
      enc_i(wvm::kMovI, 0, 3, 0, 999),
      enc_r(wvm::kSend, 0, 1, 2, 3),
      enc_r(wvm::kHalt, 0, 0, 0, 0),
  };
  std::vector<wvm::VmImage> stale_images(2);
  for (wvm::VmImage& image : stale_images) {
    image.code = stale_prog;
    image.mem_size_words = 16;
  }
  wvm::PersistentRuntime stale_rt;
  if (!stale_rt.Init(stale_images, logical_ids, err) ||
      !stale_rt.Launch(err)) {
    std::printf("  %-22s FAIL (%s)\n", "retired_route_rejected",
                err.c_str());
    return 1;
  }
  stale_rt.BootAll();
  stale_rt.WaitStatus(0, wvm::kFaulted, 2000);
  stale_rt.WaitStatus(1, wvm::kHalted, 2000);
  const bool stale_ok = stale_rt.Status(0) == wvm::kFaulted &&
                        stale_rt.Fault(0) == wvm::kFaultMsg &&
                        stale_rt.Status(1) == wvm::kHalted;
  std::printf("  %-22s %s\n", "retired_route_rejected",
              stale_ok ? "PASS" : "FAIL");
  ok &= stale_ok;
  stale_rt.ShutdownAll();
  ok &= stale_rt.Sync() == cudaSuccess;

  std::printf(ok ? "slice7: PASS\n" : "slice7: FAIL\n");
  return ok ? 0 : 1;
}

// v0.1.1 Graphics Slice A: framebuffer memory. Verifies the memory-mapped
// framebuffer through the extended address decoder: 32-lane + predicated
// stores, isolation between VMs, bounds faults, RAM compatibility, reset.
int RunSliceGfxA() {
  using wvm::enc_i;
  using wvm::enc_r;
  bool ok = true;
  auto wait_stopped = [](wvm::PersistentRuntime& rt, uint32_t n) -> bool {
    for (int waited = 0; waited < 5000; ++waited) {
      bool done = true;
      for (uint32_t i = 0; i < n; ++i) {
        const uint32_t s = rt.Status(i);
        if (s != wvm::kHalted && s != wvm::kFaulted) {
          done = false;
          break;
        }
      }
      if (done) return true;
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return false;
  };

  const uint32_t LIT_BASE = 0;  // literal 0 = VIDEO_BASE_WORD
  const uint32_t LIT_OPAQUE = 1;

  // ---- 32-lane store + isolation: each VM paints row 0 its own colour ----
  {
    const std::vector<uint32_t> lits = {wvm::kVideoBaseWord, 0xFF000000u};
    const std::vector<uint32_t> prog = {
        enc_i(wvm::kLdw, 0, 2, 0, LIT_BASE),      // r2 = VIDEO_BASE
        enc_i(wvm::kLdw, 0, 4, 0, LIT_OPAQUE),    // r4 = 0xFF000000
        enc_r(wvm::kVmid, 0, 0, 0, 0),            // r0 = vm_id
        enc_r(wvm::kLaneId, 0, 1, 0, 0),          // r1 = lane
        enc_r(wvm::kAdd, 0, 2, 2, 1),             // r2 = VIDEO_BASE + lane
        enc_i(wvm::kMovI, 0, 3, 0, 1),            // r3 = 1
        enc_r(wvm::kAdd, 0, 3, 3, 0),             // r3 = vm_id + 1
        enc_r(wvm::kOr, 0, 3, 3, 4),              // r3 = 0xFF000000|(vm_id+1)
        enc_r(wvm::kStore, 0, 2, 3, 0),           // fb[lane] = colour
        enc_r(wvm::kHalt, 0, 0, 0, 0),
    };
    const uint32_t N = 4;
    std::vector<wvm::VmImage> images(N);
    for (uint32_t i = 0; i < N; ++i) {
      images[i].code = prog;
      images[i].literals = lits;
      images[i].mem_size_words = 8;
    }
    wvm::PersistentRuntime rt;
    std::string err;
    bool pass = rt.Init(images, err) && rt.Launch(err);
    if (pass) {
      rt.BootAll();
      pass = wait_stopped(rt, N);
      for (uint32_t i = 0; i < N && pass; ++i) {
        std::vector<uint32_t> fb;
        if (!rt.ReadFramebuffer(i, fb)) {
          pass = false;
          break;
        }
        const uint32_t expect = 0xFF000000u | (i + 1);
        for (uint32_t x = 0; x < 32; ++x)
          if (fb[x] != expect) pass = false;        // 32-lane store
        for (uint32_t p = 32; p < wvm::kVideoWords; ++p)
          if (fb[p] != wvm::kVideoResetColor) pass = false;  // isolation/clear
      }
    }
    std::printf("  %-22s %s\n", "framebuffer_isolation", pass ? "PASS" : "FAIL");
    ok &= pass;
    rt.ShutdownAll();
    rt.Sync();
  }

  // ---- predicated store: only lanes 0..15 write ----
  {
    const std::vector<uint32_t> lits = {wvm::kVideoBaseWord, 0xFFFFFFFFu};
    const std::vector<uint32_t> prog = {
        enc_i(wvm::kLdw, 0, 2, 0, LIT_BASE),      // r2 = VIDEO_BASE
        enc_i(wvm::kLdw, 0, 3, 0, LIT_OPAQUE),    // r3 = white
        enc_r(wvm::kLaneId, 0, 1, 0, 0),          // r1 = lane
        enc_r(wvm::kAdd, 0, 2, 2, 1),             // r2 = VIDEO_BASE + lane
        enc_i(wvm::kCmpLtI, 0, 0, 1, 16),         // p0 = lane < 16
        enc_r(wvm::kStore, 1, 2, 3, 0),           // @p0 STORE fb[lane]=white
        enc_r(wvm::kHalt, 0, 0, 0, 0),
    };
    std::vector<wvm::VmImage> images(1);
    images[0].code = prog;
    images[0].literals = lits;
    images[0].mem_size_words = 8;
    wvm::PersistentRuntime rt;
    std::string err;
    bool pass = rt.Init(images, err) && rt.Launch(err);
    if (pass) {
      rt.BootAll();
      pass = wait_stopped(rt, 1);
      std::vector<uint32_t> fb;
      if (pass && rt.ReadFramebuffer(0, fb)) {
        for (uint32_t x = 0; x < 16; ++x)
          if (fb[x] != 0xFFFFFFFFu) pass = false;         // active lanes wrote
        for (uint32_t x = 16; x < 32; ++x)
          if (fb[x] != wvm::kVideoResetColor) pass = false;  // inactive untouched
      } else {
        pass = false;
      }
    }
    std::printf("  %-22s %s\n", "predicated_pixel_store", pass ? "PASS" : "FAIL");
    ok &= pass;
    rt.ShutdownAll();
    rt.Sync();
  }

  // ---- bounds fault at VIDEO_END_WORD; neighbour unaffected ----
  {
    const std::vector<uint32_t> lits = {wvm::kVideoEndWord};
    const std::vector<uint32_t> prog = {
        enc_r(wvm::kVmid, 0, 0, 0, 0),            // r0 = vm_id
        enc_i(wvm::kCmpEqI, 0, 0, 0, 0),          // p0 = (vm_id == 0)
        enc_i(wvm::kJmpIfAny, 1, 0, 0, 4),        // -> dofault(4)
        enc_r(wvm::kHalt, 0, 0, 0, 0),            // other VMs halt
        enc_i(wvm::kLdw, 0, 2, 0, 0),             // dofault: r2 = VIDEO_END
        enc_i(wvm::kMovI, 0, 3, 0, 0),            // r3 = 0
        enc_r(wvm::kStore, 0, 2, 3, 0),           // STORE at VIDEO_END -> fault
        enc_r(wvm::kHalt, 0, 0, 0, 0),
    };
    std::vector<wvm::VmImage> images(2);
    for (uint32_t i = 0; i < 2; ++i) {
      images[i].code = prog;
      images[i].literals = lits;
      images[i].mem_size_words = 8;
    }
    wvm::PersistentRuntime rt;
    std::string err;
    bool pass = rt.Init(images, err) && rt.Launch(err);
    if (pass) {
      rt.BootAll();
      pass = wait_stopped(rt, 2);
      pass &= rt.Status(0) == wvm::kFaulted && rt.Fault(0) == wvm::kFaultMem;
      pass &= rt.Status(1) == wvm::kHalted;  // neighbour unaffected
    }
    std::printf("  %-22s %s\n", "video_bounds_fault", pass ? "PASS" : "FAIL");
    ok &= pass;
    rt.ShutdownAll();
    rt.Sync();
  }

  // ---- RAM compatibility: ordinary RAM still works ----
  {
    const std::vector<uint32_t> prog = {
        enc_i(wvm::kMovI, 0, 0, 0, 5),   // r0 = 5
        enc_i(wvm::kMovI, 0, 1, 0, 0),   // r1 = addr 0
        enc_r(wvm::kStore, 0, 1, 0, 0),  // mem[0] = 5
        enc_r(wvm::kLoad, 0, 2, 1, 0),   // r2 = mem[0]
        enc_r(wvm::kHalt, 0, 0, 0, 0),
    };
    std::vector<wvm::VmImage> images(1);
    images[0].code = prog;
    images[0].mem_size_words = 8;
    wvm::PersistentRuntime rt;
    std::string err;
    bool pass = rt.Init(images, err) && rt.Launch(err);
    if (pass) {
      rt.BootAll();
      pass = wait_stopped(rt, 1);
      std::vector<uint32_t> mem;
      if (pass && rt.ReadMem(0, 0, 1, mem)) pass = !mem.empty() && mem[0] == 5;
      else pass = false;
    }
    std::printf("  %-22s %s\n", "ram_compatibility", pass ? "PASS" : "FAIL");
    ok &= pass;
    rt.ShutdownAll();
    rt.Sync();
  }

  // ---- reset clears only the reset VM's framebuffer ----
  {
    const std::vector<uint32_t> lits = {wvm::kVideoBaseWord, 0xAABBCCDDu};
    const std::vector<uint32_t> prog = {
        enc_i(wvm::kLdw, 0, 2, 0, LIT_BASE),     // r2 = VIDEO_BASE
        enc_i(wvm::kLdw, 0, 3, 0, LIT_OPAQUE),   // r3 = 0xAABBCCDD
        enc_r(wvm::kLaneId, 0, 1, 0, 0),         // r1 = lane
        enc_r(wvm::kAdd, 0, 2, 2, 1),            // r2 = VIDEO_BASE + lane
        enc_r(wvm::kStore, 0, 2, 3, 0),          // fb[lane] = 0xAABBCCDD
        enc_r(wvm::kHalt, 0, 0, 0, 0),
    };
    std::vector<wvm::VmImage> images(2);
    for (uint32_t i = 0; i < 2; ++i) {
      images[i].code = prog;
      images[i].literals = lits;
      images[i].mem_size_words = 8;
    }
    wvm::PersistentRuntime rt;
    std::string err;
    bool pass = rt.Init(images, err) && rt.Launch(err);
    if (pass) {
      rt.BootAll();
      pass = wait_stopped(rt, 2);
      // Reset VM 0 only.
      rt.SendCmd(0, wvm::kCmdReset);
      bool idle = false;
      for (int w = 0; w < 2000; ++w) {
        if (rt.Status(0) == wvm::kIdle) { idle = true; break; }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
      pass &= idle;
      std::vector<uint32_t> fb0, fb1;
      if (pass && rt.ReadFramebuffer(0, fb0) && rt.ReadFramebuffer(1, fb1)) {
        for (uint32_t p = 0; p < wvm::kVideoWords; ++p)
          if (fb0[p] != wvm::kVideoResetColor) pass = false;  // VM0 cleared
        for (uint32_t x = 0; x < 32; ++x)
          if (fb1[x] != 0xAABBCCDDu) pass = false;  // VM1 untouched
      } else {
        pass = false;
      }
    }
    std::printf("  %-22s %s\n", "reset_clears_fb", pass ? "PASS" : "FAIL");
    ok &= pass;
    rt.ShutdownAll();
    rt.Sync();
  }

  std::printf(ok ? "gfxA: PASS\n" : "gfxA: FAIL\n");
  return ok ? 0 : 1;
}

// v0.1.1 Graphics Slice B: frame publication. FLIP bumps frame_seq exactly
// once per retired FLIP; FLIP is unguardable.
int RunSliceGfxB() {
  using wvm::enc_i;
  using wvm::enc_r;
  bool ok = true;
  auto wait_stopped = [](wvm::PersistentRuntime& rt, uint32_t n) -> bool {
    for (int waited = 0; waited < 5000; ++waited) {
      bool done = true;
      for (uint32_t i = 0; i < n; ++i) {
        const uint32_t s = rt.Status(i);
        if (s != wvm::kHalted && s != wvm::kFaulted) {
          done = false;
          break;
        }
      }
      if (done) return true;
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return false;
  };

  // ---- flip_sequence: frame_seq advances exactly once per FLIP ----
  {
    const std::vector<uint32_t> prog = {
        enc_r(wvm::kFlip, 0, 0, 0, 0),     // frame_seq -> 1
        enc_r(wvm::kFlip, 0, 0, 0, 0),     // -> 2
        enc_i(wvm::kMovI, 0, 0, 0, 5),     // (work between flips)
        enc_r(wvm::kFlip, 0, 0, 0, 0),     // -> 3
        enc_r(wvm::kHalt, 0, 0, 0, 0),
    };
    std::vector<wvm::VmImage> images(1);
    images[0].code = prog;
    images[0].mem_size_words = 8;
    wvm::PersistentRuntime rt;
    std::string err;
    bool pass = rt.Init(images, err) && rt.Launch(err);
    if (pass) {
      rt.BootAll();
      pass = wait_stopped(rt, 1);
      pass &= rt.Status(0) == wvm::kHalted && rt.FrameSeq(0) == 3;
    }
    std::printf("  %-22s %s (frame_seq=%u)\n", "flip_sequence",
                pass ? "PASS" : "FAIL", rt.FrameSeq(0));
    ok &= pass;
    rt.ShutdownAll();
    rt.Sync();
  }

  // ---- flip_unguardable: a guarded FLIP faults with FAULT_OPERAND ----
  {
    const std::vector<uint32_t> prog = {
        enc_r(wvm::kFlip, 1, 0, 0, 0),   // @p0 FLIP -> FAULT_OPERAND
        enc_r(wvm::kHalt, 0, 0, 0, 0),
    };
    std::vector<wvm::VmImage> images(1);
    images[0].code = prog;
    images[0].mem_size_words = 8;
    wvm::PersistentRuntime rt;
    std::string err;
    bool pass = rt.Init(images, err) && rt.Launch(err);
    if (pass) {
      rt.BootAll();
      pass = wait_stopped(rt, 1);
      pass &= rt.Status(0) == wvm::kFaulted &&
              rt.Fault(0) == wvm::kFaultOperand;
    }
    std::printf("  %-22s %s\n", "flip_unguardable", pass ? "PASS" : "FAIL");
    ok &= pass;
    rt.ShutdownAll();
    rt.Sync();
  }

  std::printf(ok ? "gfxB: PASS\n" : "gfxB: FAIL\n");
  return ok ? 0 : 1;
}

// v0.1.1 viewer smoke test (headless, SDL-free): run a graphics program on
// one resident VM, wait for a published frame, copy the framebuffer to the
// host and check sample pixels. graphics.wva paints
// colour = 0xFF000000 | ((x+vmid*4)&0xFF)<<16 | (y&0xFF)<<8 | (frame&0xFF);
// the red/green channels are frame-independent, so masking out blue gives a
// deterministic check even though the image is animating.
int RunGfxSmoke(const char* path) {
  wvm::WvmFile file;
  std::string err;
  if (!wvm::LoadWvm(path, file, err)) {
    std::fprintf(stderr, "error: %s: %s\n", path, err.c_str());
    return 2;
  }
  std::vector<wvm::VmImage> images(1);
  images[0].code = file.code;
  images[0].literals = file.literals;
  images[0].mem_size_words = 8;
  wvm::PersistentRuntime rt;
  if (!rt.Init(images, err) || !rt.Launch(err)) {
    std::fprintf(stderr, "error: %s\n", err.c_str());
    return 1;
  }
  rt.BootAll();

  // Wait for at least one published frame.
  bool published = false;
  for (int w = 0; w < 5000; ++w) {
    if (rt.FrameSeq(0) >= 1) {
      published = true;
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  bool pass = published;
  if (pass) {
    std::vector<uint32_t> fb;
    pass = rt.ReadFramebuffer(0, fb);
    if (pass) {
      auto masked = [&](uint32_t x, uint32_t y) -> uint32_t {
        return fb[y * wvm::kVideoWidth + x] & 0xFFFFFF00u;
      };
      const uint32_t vmid = 0;
      const uint32_t e00 = 0xFF000000u | (((0 + vmid * 4) & 0xFF) << 16);
      const uint32_t eX0 = 0xFF000000u | (((127 + vmid * 4) & 0xFF) << 16);
      const uint32_t e0Y = 0xFF000000u | ((127 & 0xFF) << 8);
      pass &= masked(0, 0) == e00;
      pass &= masked(127, 0) == eX0;
      pass &= masked(0, 127) == e0Y;
    }
  }
  std::printf("viewer_smoke: %s (frame_seq=%u)\n", pass ? "PASS" : "FAIL",
              rt.FrameSeq(0));
  rt.ShutdownAll();
  rt.Sync();
  return pass ? 0 : 1;
}

// v0.1.1 graphics capstone: run N resident VMs on graphics.wvm, each
// producing a VMID-distinct 128x128 image via 32-lane stores + FLIP, and
// verify every framebuffer (via host copy) is correct and distinct.
int RunGfxCap(const char* path, uint32_t n_vms) {
  wvm::WvmFile file;
  std::string err;
  if (!wvm::LoadWvm(path, file, err)) {
    std::fprintf(stderr, "error: %s: %s\n", path, err.c_str());
    return 2;
  }
  if (n_vms < 2 || n_vms > wvm::kMaxVms) {
    std::fprintf(stderr, "error: --vms must be 2..%u\n", wvm::kMaxVms);
    return 2;
  }
  std::vector<wvm::VmImage> images(n_vms);
  for (uint32_t i = 0; i < n_vms; ++i) {
    images[i].code = file.code;
    images[i].literals = file.literals;
    images[i].mem_size_words = 8;
  }
  wvm::PersistentRuntime rt;
  if (!rt.Init(images, err) || !rt.Launch(err)) {
    std::fprintf(stderr, "error: %s\n", err.c_str());
    return 1;
  }
  rt.BootAll();
  std::printf("gfx_cap: %u resident VMs rendering graphics.wvm\n", n_vms);

  // Wait until every VM has published at least one frame.
  bool all_published = false;
  for (int w = 0; w < 10000; w += 5) {
    all_published = true;
    for (uint32_t i = 0; i < n_vms; ++i)
      if (rt.FrameSeq(i) < 1) { all_published = false; break; }
    if (all_published) break;
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  if (!all_published) {
    std::printf("gfx_cap: FAIL (not all VMs published a frame)\n");
    rt.ShutdownAll();
    rt.Sync();
    return 1;
  }

  // Verify each VM's image against the formula and check they're distinct.
  // red(x,y) = (x + vmid*4)&0xFF, green(x,y) = y&0xFF; blue = frame (masked
  // out). pixel(127,0): red=(127+vmid*4), green=0. pixel(0,127):
  // red=(vmid*4), green=127.
  std::vector<uint32_t> framebuffers;
  if (!rt.ReadFramebuffers(0, n_vms, framebuffers)) {
    std::printf("gfx_cap: FAIL (bulk framebuffer copy failed)\n");
    rt.ShutdownAll();
    rt.Sync();
    return 1;
  }

  uint32_t bad = 0;
  std::vector<uint32_t> reds;
  reds.reserve(n_vms);
  for (uint32_t i = 0; i < n_vms; ++i) {
    const uint32_t* fb =
        framebuffers.data() + static_cast<size_t>(i) * wvm::kVideoWords;
    const uint32_t red_x0 = (127u + i * 4u) & 0xFFu;   // pixel (127,0)
    const uint32_t red_0y = (0u + i * 4u) & 0xFFu;     // pixel (0,127)
    const uint32_t expect_x0 = 0xFF000000u | (red_x0 << 16);
    const uint32_t expect_0y = 0xFF000000u | (red_0y << 16) | (127u << 8);
    const uint32_t px_x0 = fb[127] & 0xFFFFFF00u;
    const uint32_t px_0y = fb[127 * wvm::kVideoWidth] & 0xFFFFFF00u;
    if (px_x0 != expect_x0 || px_0y != expect_0y) ++bad;
    reds.push_back(red_x0);
  }
  // All reds must be pairwise distinct (vmid*4 spans 0..252 for 64 VMs).
  uint32_t distinct = 0;
  for (uint32_t i = 0; i < reds.size(); ++i) {
    bool dup = false;
    for (uint32_t j = 0; j < i; ++j)
      if (reds[j] == reds[i]) { dup = true; break; }
    if (!dup) ++distinct;
  }
  // Every VM should still be running (redrawing) — independently alive.
  uint32_t running = 0;
  for (uint32_t i = 0; i < n_vms; ++i)
    if (rt.Status(i) == wvm::kRunning) ++running;

  const bool pass = bad == 0 && distinct == n_vms && running == n_vms;
  std::printf("gfx_cap: images correct    %s (%u/%u)\n",
              bad == 0 ? "PASS" : "FAIL", n_vms - bad, n_vms);
  std::printf("gfx_cap: images distinct   %s (%u/%u)\n",
              distinct == n_vms ? "PASS" : "FAIL", distinct, n_vms);
  std::printf("gfx_cap: still RUNNING     %s (%u/%u)\n",
              running == n_vms ? "PASS" : "FAIL", running, n_vms);
  std::printf(pass ? "gfx_cap: PASS\n" : "gfx_cap: FAIL\n");
  rt.ShutdownAll();
  rt.Sync();
  return pass ? 0 : 1;
}

// Program 01 correctness test. VM 0 starts as a blinker spanning packed words
// at x=63/64; VM 1 starts as a toroidal 2x2 block across all four corners.
// Pause after publication so both packed RAM and rendered pixels are stable.
int RunLifeTest(const char* path) {
  wvm::WvmFile file;
  std::string err;
  if (!wvm::LoadWvm(path, file, err)) {
    std::fprintf(stderr, "error: %s: %s\n", path, err.c_str());
    return 2;
  }

  constexpr uint32_t kLifeVms = 2;
  constexpr uint32_t kWorldWords = 512;
  std::vector<wvm::VmImage> images(kLifeVms);
  for (wvm::VmImage& image : images) {
    image.code = file.code;
    image.literals = file.literals;
    image.mem_size_words = wvm::kRamSizeWords;
  }

  wvm::PersistentRuntime rt;
  if (!rt.Init(images, err) || !rt.Launch(err)) {
    std::fprintf(stderr, "error: %s\n", err.c_str());
    return 1;
  }
  rt.BootAll();

  bool published = false;
  for (int waited = 0; waited < 10000; ++waited) {
    if (rt.FrameSeq(0) >= 2 && rt.FrameSeq(1) >= 2) {
      published = true;
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  bool paused = published;
  for (uint32_t vm = 0; vm < kLifeVms && paused; ++vm)
    paused &= rt.Pause(vm);

  bool packed_state = paused;
  bool life_step = paused;
  bool toroidal = paused;
  bool buffer_separation = paused;
  bool render_mapping = paused;
  bool isolation = paused;
  uint32_t stopped_seq[kLifeVms] = {rt.FrameSeq(0), rt.FrameSeq(1)};

  std::vector<uint32_t> worlds[kLifeVms];
  std::vector<uint32_t> frames[kLifeVms];
  wvm::VmState states[kLifeVms]{};
  for (uint32_t vm = 0; vm < kLifeVms && paused; ++vm) {
    packed_state &= rt.ReadState(vm, states[vm]);
    const uint32_t current_base = states[vm].sregs[0];
    buffer_separation &=
        (current_base == 0 || current_base == kWorldWords) &&
        states[vm].sregs[1] != current_base;
    packed_state &=
        rt.ReadMem(vm, current_base, kWorldWords, worlds[vm]);
    render_mapping &= rt.ReadFramebuffer(vm, frames[vm]);
  }

  std::vector<uint32_t> expected0(kWorldWords, 0);
  const bool horizontal = (stopped_seq[0] & 1u) == 0;
  if (horizontal) {
    expected0[257] = 0x80000000u;  // (63,64)
    expected0[258] = 0x00000003u;  // (64,64), (65,64)
  } else {
    expected0[254] = 0x00000001u;  // (64,63)
    expected0[258] = 0x00000001u;  // (64,64)
    expected0[262] = 0x00000001u;  // (64,65)
  }

  std::vector<uint32_t> expected1(kWorldWords, 0);
  expected1[0] = 0x00000001u;     // (0,0)
  expected1[3] = 0x80000000u;     // (127,0)
  expected1[508] = 0x00000001u;   // (0,127)
  expected1[511] = 0x80000000u;   // (127,127)

  if (worlds[0].size() == kWorldWords) life_step &= worlds[0] == expected0;
  else life_step = false;
  if (worlds[1].size() == kWorldWords) toroidal &= worlds[1] == expected1;
  else toroidal = false;
  packed_state &= life_step && toroidal;
  isolation &= worlds[0] != worlds[1];

  auto pixel_matches = [](const std::vector<uint32_t>& frame,
                          const std::vector<uint32_t>& world) {
    if (frame.size() != wvm::kVideoWords || world.size() != kWorldWords)
      return false;
    for (uint32_t y = 0; y < wvm::kVideoHeight; ++y) {
      for (uint32_t x = 0; x < wvm::kVideoWidth; ++x) {
        const uint32_t word = world[y * 4 + (x >> 5)];
        const bool alive = ((word >> (x & 31)) & 1u) != 0;
        const uint32_t expected = alive ? 0xFFFFFFFFu : 0xFF000000u;
        if (frame[y * wvm::kVideoWidth + x] != expected) return false;
      }
    }
    return true;
  };
  render_mapping &= pixel_matches(frames[0], expected0);
  render_mapping &= pixel_matches(frames[1], expected1);

  // Resume both universes and require another independently published frame.
  for (uint32_t vm = 0; vm < kLifeVms; ++vm) rt.SendCmd(vm, wvm::kCmdRun);
  bool persistent = false;
  for (int waited = 0; waited < 10000; ++waited) {
    if (rt.FrameSeq(0) > stopped_seq[0] &&
        rt.FrameSeq(1) > stopped_seq[1] &&
        rt.Status(0) == wvm::kRunning && rt.Status(1) == wvm::kRunning) {
      persistent = true;
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  const bool pass = packed_state && life_step && toroidal &&
                    buffer_separation && render_mapping && isolation &&
                    persistent;
  std::printf("warplife: packed_state       %s\n",
              packed_state ? "PASS" : "FAIL");
  std::printf("warplife: life_step          %s (VM 0 seq=%u, %s)\n",
              life_step ? "PASS" : "FAIL", stopped_seq[0],
              horizontal ? "horizontal" : "vertical");
  std::printf("warplife: toroidal_stilllife %s\n",
              toroidal ? "PASS" : "FAIL");
  std::printf("warplife: buffer_separation  %s\n",
              buffer_separation ? "PASS" : "FAIL");
  std::printf("warplife: render_mapping     %s\n",
              render_mapping ? "PASS" : "FAIL");
  std::printf("warplife: vm_isolation       %s\n",
              isolation ? "PASS" : "FAIL");
  std::printf("warplife: persistent         %s\n",
              persistent ? "PASS" : "FAIL");
  std::printf(pass ? "warplife: PASS\n" : "warplife: FAIL\n");

  rt.ShutdownAll();
  const cudaError_t shutdown = rt.Sync();
  return pass && shutdown == cudaSuccess ? 0 : 1;
}

// v0.1 capstone: boot 64+ resident VMs that each do a 32-lane computation,
// exchange a message around a ring, and keep running so they stay
// inspectable. Verifies the whole milestone, then inspects one VM live.
int CmdDemo(const char* path, uint32_t n_vms) {
  wvm::WvmFile file;
  std::string err;
  if (!wvm::LoadWvm(path, file, err)) {
    std::fprintf(stderr, "error: %s: %s\n", path, err.c_str());
    return 2;
  }
  if (n_vms < 2 || n_vms > wvm::kMaxVms) {
    std::fprintf(stderr, "error: --vms must be 2..%u\n", wvm::kMaxVms);
    return 2;
  }
  std::vector<wvm::VmImage> images(n_vms);
  for (uint32_t i = 0; i < n_vms; ++i) {
    images[i].code = file.code;
    images[i].literals = file.literals;
    images[i].mem_size_words = 64;
  }
  wvm::PersistentRuntime rt;
  if (!rt.Init(images, err) || !rt.Launch(err)) {
    std::fprintf(stderr, "error: %s\n", err.c_str());
    return 1;
  }
  rt.BootAll();
  std::printf("demo: %u resident VMs from %s\n", n_vms, path);

  // Wait until every VM has received its ring message and stored mem[0].
  // Each stored aggregate is >= 496, so 0 means "not yet".
  bool ready = false;
  for (int waited = 0; waited < 10000; waited += 5) {
    ready = true;
    for (uint32_t i = 0; i < n_vms; ++i) {
      std::vector<uint32_t> m;
      if (!rt.ReadMem(i, 0, 1, m) || m.empty() || m[0] == 0) {
        ready = false;
        break;
      }
    }
    if (ready) break;
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  if (!ready) {
    std::printf("demo: FAIL (VMs did not complete the ring exchange)\n");
    rt.ShutdownAll();
    rt.Sync();
    return 1;
  }

  // Verify the ring: VM i holds the aggregate of VM (i-1) mod n, where
  // aggregate(vm) = 32*vm + 496 (the demo's 32-lane reduction).
  uint32_t bad = 0;
  uint32_t running = 0;
  for (uint32_t i = 0; i < n_vms; ++i) {
    std::vector<uint32_t> m;
    rt.ReadMem(i, 0, 1, m);
    const uint32_t prev = (i + n_vms - 1) % n_vms;
    const uint32_t expect = 32u * prev + 496u;
    if (m.empty() || m[0] != expect) ++bad;
    if (rt.Status(i) == wvm::kRunning) ++running;
  }
  std::printf("demo: ring exchange   %s (%u/%u correct)\n",
              bad == 0 ? "PASS" : "FAIL", n_vms - bad, n_vms);
  std::printf("demo: still RUNNING   %s (%u/%u)\n",
              running == n_vms ? "PASS" : "FAIL", running, n_vms);
  bool ok = bad == 0 && running == n_vms;

  // Live inspection: pick a VM mid-run and watch its instruction counter
  // advance (the control-plane counter updates at control points even while
  // the VM keeps running).
  const uint32_t probe = n_vms > 37 ? 37 : 0;
  const uint64_t i_first = rt.Instrs(probe);
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  const uint64_t i_second = rt.Instrs(probe);
  std::printf("demo: vm %u live       status=%s instrs %llu -> %llu (%s)\n",
              probe, wvm::StatusName(rt.Status(probe)),
              (unsigned long long)i_first, (unsigned long long)i_second,
              i_second > i_first ? "ticking" : "static");
  ok &= i_second > i_first && rt.Status(probe) == wvm::kRunning;

  rt.ShutdownAll();
  const cudaError_t e = rt.Sync();
  std::printf("demo: shutdown        %s\n",
              e == cudaSuccess ? "ok" : cudaGetErrorString(e));
  std::printf(ok && e == cudaSuccess ? "demo: PASS\n" : "demo: FAIL\n");
  return ok && e == cudaSuccess ? 0 : 1;
}

void Usage(const char* argv0) {
  std::printf("usage: %s <command>\n", argv0);
  std::printf("  run <file.wvm>  run a .wvm program to completion on one warp\n");
  std::printf("  serve <file.wvm> [--vms N] [--for S] [--compiled]\n");
  std::printf("                  boot N resident VMs, print `list` for S seconds\n");
  std::printf("  attach <file.wvm> [--vms N] [--compiled]\n");
  std::printf("                  boot N resident VMs, interactive console on stdin\n");
  std::printf("  demo <file.wvm> [--vms N]\n");
  std::printf("                  v0.1 capstone: N VMs compute + ring-message + stay live\n");
  std::printf("  slice1..slice5,slice7  self-test suites\n");
  std::printf("  hetero_tests           shared-program heterogeneous VM tests\n");
  std::printf("  supervisor_tests       dynamic VM lifecycle/recycling tests\n");
  std::printf("  supervise [--script FILE] [--interactive]\n");
  std::printf("                  long-lived heterogeneous CPU supervisor console\n");
  std::printf("  gfxa | gfxb            v0.1.1 graphics slices A/B self-tests\n");
  std::printf("  gfxsmoke <file.wvm>    headless framebuffer-copy smoke test\n");
  std::printf("  gfx_cap <file.wvm> [--vms N]\n");
  std::printf("                  v0.1.1 capstone: N VMs render distinct images\n");
  std::printf("  view <file.wvm> [--vm N | --vms N] [--compiled]\n");
  std::printf("                  show one VM or a tiled grid of N resident VMs\n");
  std::printf("  hetero_view <a.wvm> <b.wvm> [...]\n");
  std::printf("                  run different interpreted programs in one grid\n");
  std::printf("  hetero_smoke <a.wvm> <b.wvm> [...]\n");
  std::printf("                  headless heterogeneous graphics/lifecycle check\n");
  std::printf("  life_test <file.wvm> Program 01 packed-Life correctness test\n");
  std::printf("  life_bench <file.wvm> [--vms N] [--ms N] [--workers N]\n");
  std::printf("                  compare GPU/CPU WarpVM and native GPU/CPU\n");
  std::printf("  life_profile <file.wvm> [--ms N]\n");
  std::printf("                  one-VM opcode, phase, and matched-cost profile\n");
  std::printf("  life_census <file.wvm>\n");
  std::printf("                  exact steady-state WarpLife bytecode census\n");
  std::printf("  cpu_tests             v0.1.2 CPU interpreter self-tests\n");
  std::printf("  life_equiv <file.wvm> v0.1.2 CPU/GPU WarpLife equivalence\n");
  std::printf("  life_native_cpu_equiv <file.wvm>\n");
  std::printf("                  full-world WarpVM/native CPU equivalence\n");
  std::printf("  compiled_tests        v0.1.3 minimal PTX backend tests\n");
  std::printf("  compiled_run <file.wvm>\n");
  std::printf("                  exact CPU/compiled equivalence through HALT\n");
  std::printf("  compiled_resident <file.wvm> [--vms N] [--for S]\n");
  std::printf("                  continuously resident compiled VMs\n");
  std::printf("  compiled_fb_tests     resident compiled framebuffer regressions\n");
  std::printf("  resident_bench <file.wvm> [--vms N] [--ms N]\n");
  std::printf("                  interpreted/compiled resident frame benchmark\n");
  std::printf("  warpc_bench <sequential.wvm> <warp.wvm>\n");
  std::printf("                  v0.1.5 sequential/warp-native data benchmark\n");
  std::printf("  compiled_life <file.wvm>\n");
  std::printf("                  compiled WarpLife checkpoints and mode transitions\n");
  std::printf("  compiled_life_bench <file.wvm>\n");
  std::printf("                  resident compiled WarpLife benchmark matrix\n");
  std::printf("  compiled_life_profile <file.wvm>\n");
  std::printf("                  compiled phase and safe-point attribution\n");
  std::printf("  emit_ptx <file.wvm> -o <file.ptx>\n");
  std::printf("                  translate the supported bytecode subset to PTX\n");
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    Usage(argv[0]);
    return 2;
  }
  const char* cmd = argv[1];
  // The logical CPU interpreter is deliberately usable without a CUDA
  // device. Cross-engine commands continue through normal GPU setup below.
  if (std::strcmp(cmd, "cpu_tests") == 0)
    return wvm::RunCpuInterpreterTests();
  if (std::strcmp(cmd, "life_census") == 0) {
    if (argc < 3) {
      std::fprintf(stderr, "error: life_census requires a .wvm file\n");
      return 2;
    }
    return wvm::RunLifeCensus(argv[2]);
  }
  if (std::strcmp(cmd, "emit_ptx") == 0) {
    if (argc != 5 || std::strcmp(argv[3], "-o") != 0) {
      std::fprintf(stderr, "error: emit_ptx requires <file.wvm> -o <file.ptx>\n");
      return 2;
    }
    return EmitCompiledPtx(argv[2], argv[4]);
  }
  if (std::strcmp(cmd, "compiled_life") == 0) {
    if (argc != 3) {
      std::fprintf(stderr, "error: compiled_life requires a .wvm file\n");
      return 2;
    }
    return RunCompiledLife(argv[2]);
  }
  if (std::strcmp(cmd, "compiled_run") == 0) {
    if (argc != 3) {
      std::fprintf(stderr, "error: compiled_run requires a .wvm file\n");
      return 2;
    }
    return RunCompiledHaltEquivalence(argv[2]);
  }
  if (std::strcmp(cmd, "compiled_resident") == 0) {
    if (argc < 3) {
      std::fprintf(stderr, "error: compiled_resident requires a .wvm file\n");
      return 2;
    }
    uint32_t num_vms = 64;
    int seconds = 3;
    for (int i = 3; i < argc; ++i) {
      if (std::strcmp(argv[i], "--vms") == 0 && i + 1 < argc)
        num_vms = static_cast<uint32_t>(std::atoi(argv[++i]));
      else if (std::strcmp(argv[i], "--for") == 0 && i + 1 < argc)
        seconds = std::max(1, std::atoi(argv[++i]));
      else {
        std::fprintf(stderr, "error: unknown compiled_resident option '%s'\n",
                     argv[i]);
        return 2;
      }
    }
    return RunCompiledResident(argv[2], num_vms, seconds);
  }
  if (std::strcmp(cmd, "resident_bench") == 0) {
    if (argc < 3) {
      std::fprintf(stderr, "error: resident_bench requires a .wvm file\n");
      return 2;
    }
    uint32_t num_vms = 64;
    int duration_ms = 2000;
    for (int i = 3; i < argc; ++i) {
      if (std::strcmp(argv[i], "--vms") == 0 && i + 1 < argc)
        num_vms = static_cast<uint32_t>(std::atoi(argv[++i]));
      else if (std::strcmp(argv[i], "--ms") == 0 && i + 1 < argc)
        duration_ms = std::max(1, std::atoi(argv[++i]));
      else {
        std::fprintf(stderr, "error: unknown resident_bench option '%s'\n",
                     argv[i]);
        return 2;
      }
    }
    if (num_vms == 0 || num_vms > wvm::kMaxVms) {
      std::fprintf(stderr, "error: --vms must be 1..%u\n", wvm::kMaxVms);
      return 2;
    }
    return RunResidentBenchmark(argv[2], num_vms, duration_ms);
  }
  if (std::strcmp(cmd, "warpc_bench") == 0) {
    if (argc != 4) {
      std::fprintf(stderr,
                   "error: warpc_bench requires sequential and warp .wvm files\n");
      return 2;
    }
    return RunWarpCDataBenchmark(argv[2], argv[3]);
  }
  if (std::strcmp(cmd, "compiled_life_bench") == 0) {
    if (argc != 3) {
      std::fprintf(stderr,
                   "error: compiled_life_bench requires a .wvm file\n");
      return 2;
    }
    return RunCompiledLifeBenchmark(argv[2]);
  }
  if (std::strcmp(cmd, "compiled_life_profile") == 0) {
    if (argc != 3) {
      std::fprintf(stderr,
                   "error: compiled_life_profile requires a .wvm file\n");
      return 2;
    }
    return RunCompiledLifeProfile(argv[2]);
  }
  // The compiled backend uses the CUDA driver API directly, so its semantic
  // tests do not depend on the separately versioned CUDA runtime.
  if (std::strcmp(cmd, "compiled_tests") == 0) return RunCompiledSlice1();

  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  std::printf("device: %s  SMs=%d  cc=%d.%d\n", prop.name,
              prop.multiProcessorCount, prop.major, prop.minor);

  if (std::strcmp(cmd, "compiled_fb_tests") == 0)
    return RunCompiledResidentFramebufferTests();
  if (std::strcmp(cmd, "slice1") == 0) return RunSlice1();
  if (std::strcmp(cmd, "slice2") == 0) return RunSlice2();
  if (std::strcmp(cmd, "slice3") == 0) return RunSlice3();
  if (std::strcmp(cmd, "slice4") == 0) return RunSlice4();
  if (std::strcmp(cmd, "slice5") == 0) return RunSlice5();
  if (std::strcmp(cmd, "slice7") == 0) return RunSlice7();
  if (std::strcmp(cmd, "hetero_tests") == 0)
    return RunHeterogeneousProgramTests();
  if (std::strcmp(cmd, "supervisor_tests") == 0)
    return RunSupervisorLifecycleTests();
  if (std::strcmp(cmd, "supervise") == 0) {
    const char* script_path = nullptr;
    bool interactive = false;
    for (int i = 2; i < argc; ++i) {
      if (std::strcmp(argv[i], "--script") == 0 && i + 1 < argc)
        script_path = argv[++i];
      else if (std::strcmp(argv[i], "--interactive") == 0)
        interactive = true;
      else {
        std::fprintf(stderr, "error: unknown supervise option '%s'\n",
                     argv[i]);
        return 2;
      }
    }
    return wvm::RunSupervisorCli(script_path, interactive);
  }
  if (std::strcmp(cmd, "gfxa") == 0) return RunSliceGfxA();
  if (std::strcmp(cmd, "gfxb") == 0) return RunSliceGfxB();
  if (std::strcmp(cmd, "gfxsmoke") == 0) {
    if (argc < 3) {
      std::fprintf(stderr, "error: gfxsmoke requires a .wvm file\n");
      return 2;
    }
    return RunGfxSmoke(argv[2]);
  }
  if (std::strcmp(cmd, "gfx_cap") == 0) {
    if (argc < 3) {
      std::fprintf(stderr, "error: gfx_cap requires a .wvm file\n");
      return 2;
    }
    uint32_t n_vms = 64;
    for (int i = 3; i < argc; ++i) {
      if (std::strcmp(argv[i], "--vms") == 0 && i + 1 < argc)
        n_vms = static_cast<uint32_t>(std::atoi(argv[++i]));
    }
    return RunGfxCap(argv[2], n_vms);
  }
  if (std::strcmp(cmd, "view") == 0) {
    if (argc < 3) {
      std::fprintf(stderr, "error: view requires a .wvm file\n");
      return 2;
    }
    uint32_t vm_index = 0;
    uint32_t n_vms = 0;
    bool select_vm = false;
    bool select_grid = false;
    bool compiled = false;
    for (int i = 3; i < argc; ++i) {
      if (std::strcmp(argv[i], "--compiled") == 0) {
        compiled = true;
        continue;
      }
      const bool is_vm = std::strcmp(argv[i], "--vm") == 0;
      const bool is_vms = std::strcmp(argv[i], "--vms") == 0;
      if ((is_vm || is_vms) && i + 1 >= argc) {
        std::fprintf(stderr, "error: %s requires a value\n", argv[i]);
        return 2;
      }
      if ((is_vm || is_vms) && i + 1 < argc) {
        char* end = nullptr;
        const unsigned long value = std::strtoul(argv[++i], &end, 10);
        if (end == argv[i] || *end != '\0' || value > wvm::kMaxVms ||
            (is_vms && value == 0) || (is_vm && value >= wvm::kMaxVms)) {
          std::fprintf(stderr, "error: %s must be %s..%u\n",
                       is_vm ? "--vm" : "--vms", is_vm ? "0" : "1",
                       is_vm ? wvm::kMaxVms - 1 : wvm::kMaxVms);
          return 2;
        }
        if (is_vm) {
          vm_index = static_cast<uint32_t>(value);
          select_vm = true;
        } else {
          n_vms = static_cast<uint32_t>(value);
          select_grid = true;
        }
      } else {
        std::fprintf(stderr, "error: unknown view option '%s'\n", argv[i]);
        return 2;
      }
    }
    if (select_vm && select_grid) {
      std::fprintf(stderr, "error: --vm and --vms are mutually exclusive\n");
      return 2;
    }
    if (select_grid) return wvm::ViewVmGrid(argv[2], n_vms, compiled);
    return wvm::ViewSingleVm(argv[2], vm_index, compiled);
  }
  if (std::strcmp(cmd, "hetero_view") == 0) {
    if (argc < 3) {
      std::fprintf(stderr,
                   "error: hetero_view requires at least one .wvm file\n");
      return 2;
    }
    std::vector<std::string> paths;
    paths.reserve(static_cast<size_t>(argc - 2));
    for (int i = 2; i < argc; ++i) paths.emplace_back(argv[i]);
    return wvm::ViewHeterogeneousGrid(paths);
  }
  if (std::strcmp(cmd, "hetero_smoke") == 0) {
    if (argc < 3) {
      std::fprintf(stderr,
                   "error: hetero_smoke requires at least one .wvm file\n");
      return 2;
    }
    std::vector<std::string> paths;
    paths.reserve(static_cast<size_t>(argc - 2));
    for (int i = 2; i < argc; ++i) paths.emplace_back(argv[i]);
    return CmdHeterogeneousSmoke(paths);
  }
  if (std::strcmp(cmd, "life_test") == 0) {
    if (argc < 3) {
      std::fprintf(stderr, "error: life_test requires a .wvm file\n");
      return 2;
    }
    return RunLifeTest(argv[2]);
  }
  if (std::strcmp(cmd, "life_equiv") == 0) {
    if (argc < 3) {
      std::fprintf(stderr, "error: life_equiv requires a .wvm file\n");
      return 2;
    }
    return wvm::RunCpuGpuLifeEquivalence(argv[2]);
  }
  if (std::strcmp(cmd, "life_native_cpu_equiv") == 0) {
    if (argc < 3) {
      std::fprintf(stderr,
                   "error: life_native_cpu_equiv requires a .wvm file\n");
      return 2;
    }
    return wvm::RunNativeCpuLifeEquivalence(argv[2]);
  }
  if (std::strcmp(cmd, "life_profile") == 0) {
    if (argc < 3) {
      std::fprintf(stderr, "error: life_profile requires a .wvm file\n");
      return 2;
    }
    int duration_ms = 1000;
    for (int i = 3; i < argc; ++i) {
      if (std::strcmp(argv[i], "--ms") == 0 && i + 1 < argc) {
        duration_ms = std::atoi(argv[++i]);
        if (duration_ms < 200 || duration_ms > 10000) {
          std::fprintf(stderr, "error: --ms must be 200..10000\n");
          return 2;
        }
      } else {
        std::fprintf(stderr, "error: unknown life_profile option '%s'\n",
                     argv[i]);
        return 2;
      }
    }
    return wvm::RunLifeProfile(argv[2], duration_ms);
  }
  if (std::strcmp(cmd, "life_bench") == 0) {
    if (argc < 3) {
      std::fprintf(stderr, "error: life_bench requires a .wvm file\n");
      return 2;
    }
    std::vector<uint32_t> vm_counts{1, 8, 32, 64};
    int duration_ms = 2000;
    uint32_t cpu_workers = std::max(1u, std::thread::hardware_concurrency());
    for (int i = 3; i < argc; ++i) {
      if (std::strcmp(argv[i], "--vms") == 0 && i + 1 < argc) {
        const int value = std::atoi(argv[++i]);
        if (value < 1 || value > static_cast<int>(wvm::kMaxVms)) {
          std::fprintf(stderr, "error: --vms must be 1..%u\n", wvm::kMaxVms);
          return 2;
        }
        vm_counts.assign(1, static_cast<uint32_t>(value));
      } else if (std::strcmp(argv[i], "--ms") == 0 && i + 1 < argc) {
        duration_ms = std::atoi(argv[++i]);
        if (duration_ms < 100 || duration_ms > 60000) {
          std::fprintf(stderr, "error: --ms must be 100..60000\n");
          return 2;
        }
      } else if (std::strcmp(argv[i], "--workers") == 0 && i + 1 < argc) {
        const int value = std::atoi(argv[++i]);
        if (value < 1 || value > static_cast<int>(wvm::kMaxVms)) {
          std::fprintf(stderr, "error: --workers must be 1..%u\n",
                       wvm::kMaxVms);
          return 2;
        }
        cpu_workers = static_cast<uint32_t>(value);
      } else {
        std::fprintf(stderr, "error: unknown life_bench option '%s'\n",
                     argv[i]);
        return 2;
      }
    }
    return wvm::RunLifeBenchmark(argv[2], vm_counts, duration_ms,
                                 cpu_workers);
  }
  if (std::strcmp(cmd, "run") == 0) {
    if (argc < 3) {
      std::fprintf(stderr, "error: run requires a .wvm file\n");
      return 2;
    }
    return CmdRun(argv[2]);
  }
  if (std::strcmp(cmd, "serve") == 0) {
    if (argc < 3) {
      std::fprintf(stderr, "error: serve requires a .wvm file\n");
      return 2;
    }
    uint32_t n_vms = 8;
    int seconds = 3;
    bool compiled = false;
    for (int i = 3; i < argc; ++i) {
      if (std::strcmp(argv[i], "--vms") == 0 && i + 1 < argc) {
        n_vms = static_cast<uint32_t>(std::atoi(argv[++i]));
      } else if (std::strcmp(argv[i], "--for") == 0 && i + 1 < argc) {
        seconds = std::atoi(argv[++i]);
      } else if (std::strcmp(argv[i], "--compiled") == 0) {
        compiled = true;
      } else {
        std::fprintf(stderr, "error: unknown serve option '%s'\n", argv[i]);
        return 2;
      }
    }
    return CmdServe(argv[2], n_vms, seconds, compiled);
  }
  if (std::strcmp(cmd, "attach") == 0) {
    if (argc < 3) {
      std::fprintf(stderr, "error: attach requires a .wvm file\n");
      return 2;
    }
    uint32_t n_vms = 4;
    bool compiled = false;
    for (int i = 3; i < argc; ++i) {
      if (std::strcmp(argv[i], "--vms") == 0 && i + 1 < argc)
        n_vms = static_cast<uint32_t>(std::atoi(argv[++i]));
      else if (std::strcmp(argv[i], "--compiled") == 0)
        compiled = true;
      else {
        std::fprintf(stderr, "error: unknown attach option '%s'\n", argv[i]);
        return 2;
      }
    }
    return CmdAttach(argv[2], n_vms, compiled);
  }
  if (std::strcmp(cmd, "demo") == 0) {
    if (argc < 3) {
      std::fprintf(stderr, "error: demo requires a .wvm file (demo.wvm)\n");
      return 2;
    }
    uint32_t n_vms = 64;
    for (int i = 3; i < argc; ++i) {
      if (std::strcmp(argv[i], "--vms") == 0 && i + 1 < argc)
        n_vms = static_cast<uint32_t>(std::atoi(argv[++i]));
    }
    return CmdDemo(argv[2], n_vms);
  }
  if (std::strcmp(cmd, "help") == 0 || std::strcmp(cmd, "--help") == 0) {
    Usage(argv[0]);
    return 0;
  }
  std::fprintf(stderr, "unknown command: %s\n", cmd);
  Usage(argv[0]);
  return 2;
}
