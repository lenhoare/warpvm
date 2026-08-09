// WarpVM host entry point.
//
// Commands:
//   slice1          slice-1 warp arithmetic demo (direct CUDA)
//   slice2          slice-2 interpreter self-tests (embedded programs)
//   run <file.wvm>  load a .wvm program, run it on one warp, print state

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "gpu/vm_state.cuh"
#include "gpu/warpvm.cuh"
#include "host/wvm_file.h"

namespace wvm {
__global__ void Slice1Kernel(uint32_t* lane_out, uint32_t* sum_out);
__global__ void VmKernel(const uint32_t* code, uint32_t code_len,
                         const uint32_t* literals, uint32_t literals_len,
                         VmState* state);
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

bool ExecProgram(const std::vector<uint32_t>& code,
                 const std::vector<uint32_t>& literals, wvm::VmState& out) {
  uint32_t* d_code = nullptr;
  uint32_t* d_lit = nullptr;
  wvm::VmState* d_state = nullptr;

  CUDA_CHECK(cudaMalloc(&d_code, code.size() * sizeof(uint32_t)));
  CUDA_CHECK(cudaMemcpy(d_code, code.data(), code.size() * sizeof(uint32_t),
                        cudaMemcpyHostToDevice));
  if (!literals.empty()) {
    CUDA_CHECK(cudaMalloc(&d_lit, literals.size() * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(d_lit, literals.data(),
                          literals.size() * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));
  }
  CUDA_CHECK(cudaMalloc(&d_state, sizeof(wvm::VmState)));
  CUDA_CHECK(cudaMemset(d_state, 0, sizeof(wvm::VmState)));

  wvm::VmKernel<<<1, wvm::kLanes>>>(
      d_code, static_cast<uint32_t>(code.size()), d_lit,
      static_cast<uint32_t>(literals.size()), d_state);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(&out, d_state, sizeof(wvm::VmState),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_code));
  CUDA_CHECK(cudaFree(d_lit));
  CUDA_CHECK(cudaFree(d_state));
  return true;
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

  std::printf(ok ? "slice2: PASS\n" : "slice2: FAIL\n");
  return ok ? 0 : 1;
}

void Usage(const char* argv0) {
  std::printf("usage: %s <command>\n", argv0);
  std::printf("  run <file.wvm>  run a .wvm program on one warp\n");
  std::printf("  slice1          slice-1 warp arithmetic demo\n");
  std::printf("  slice2          slice-2 interpreter self-tests\n");
}

}  // namespace

int main(int argc, char** argv) {
  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  std::printf("device: %s  SMs=%d  cc=%d.%d\n", prop.name,
              prop.multiProcessorCount, prop.major, prop.minor);

  if (argc < 2) {
    Usage(argv[0]);
    return 2;
  }
  const char* cmd = argv[1];
  if (std::strcmp(cmd, "slice1") == 0) return RunSlice1();
  if (std::strcmp(cmd, "slice2") == 0) return RunSlice2();
  if (std::strcmp(cmd, "run") == 0) {
    if (argc < 3) {
      std::fprintf(stderr, "error: run requires a .wvm file\n");
      return 2;
    }
    return CmdRun(argv[2]);
  }
  if (std::strcmp(cmd, "help") == 0 || std::strcmp(cmd, "--help") == 0) {
    Usage(argv[0]);
    return 0;
  }
  std::fprintf(stderr, "unknown command: %s\n", cmd);
  Usage(argv[0]);
  return 2;
}
