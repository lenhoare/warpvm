// WarpVM host entry point.
//
// Commands:
//   slice1          slice-1 warp arithmetic demo (direct CUDA)
//   slice2          slice-2 interpreter self-tests (embedded programs)
//   run <file.wvm>  load a .wvm program, run it on one warp, print state

#include <algorithm>
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
__global__ void VmArrayKernel(const VmDesc* descs, VmState* states);
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

// Host-side description of one VM to execute: program, literals, and private
// RAM (zero-filled, optionally pre-seeded).
struct VmImage {
  std::vector<uint32_t> code;
  std::vector<uint32_t> literals;
  uint32_t mem_size_words = 0;
  std::vector<uint32_t> mem_init;
};

bool ExecVmArray(const std::vector<VmImage>& images,
                 std::vector<wvm::VmState>& states,
                 std::vector<std::vector<uint32_t>>& mem_out) {
  const size_t n = images.size();
  std::vector<uint32_t*> d_code(n, nullptr), d_lit(n, nullptr), d_mem(n,
                                                                      nullptr);
  std::vector<wvm::VmDesc> descs(n);

  for (size_t i = 0; i < n; ++i) {
    const VmImage& img = images[i];
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
    descs[i] = wvm::VmDesc{d_code[i],
                           static_cast<uint32_t>(img.code.size()),
                           d_lit[i],
                           static_cast<uint32_t>(img.literals.size()),
                           d_mem[i],
                           img.mem_size_words};
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
  }
  return true;
}

// Single-VM convenience wrapper used by `run` and the slice-2 self-tests.
bool ExecProgram(const std::vector<uint32_t>& code,
                 const std::vector<uint32_t>& literals, wvm::VmState& out) {
  VmImage img;
  img.code = code;
  img.literals = literals;
  img.mem_size_words = 16384;  // `run` default: 64 KB VM RAM
  std::vector<wvm::VmState> states;
  std::vector<std::vector<uint32_t>> mem;
  if (!ExecVmArray({img}, states, mem)) return false;
  out = states[0];
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
    std::vector<VmImage> images(N);
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
    std::vector<VmImage> images(N);
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

void Usage(const char* argv0) {
  std::printf("usage: %s <command>\n", argv0);
  std::printf("  run <file.wvm>  run a .wvm program on one warp\n");
  std::printf("  slice1          slice-1 warp arithmetic demo\n");
  std::printf("  slice2          slice-2 interpreter self-tests\n");
  std::printf("  slice3          slice-3 multi-VM independence tests\n");
  std::printf("  slice4          slice-4 warp-native/control-flow tests\n");
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
  if (std::strcmp(cmd, "slice3") == 0) return RunSlice3();
  if (std::strcmp(cmd, "slice4") == 0) return RunSlice4();
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
