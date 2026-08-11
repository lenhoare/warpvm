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
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "gpu/vm_state.cuh"
#include "gpu/warpvm.cuh"
#include "host/disasm.h"
#include "host/persistent.h"
#include "host/vm_image.h"
#include "host/wvm_file.h"

namespace wvm {
__global__ void Slice1Kernel(uint32_t* lane_out, uint32_t* sum_out);
__global__ void VmArrayKernel(const VmDesc* descs, VmState* states);
int ViewSingleVm(const char* path, uint32_t vm_index);  // host/view_sdl.cu
int ViewVmGrid(const char* path, uint32_t n_vms);       // host/view_sdl.cu
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
                                                                      nullptr);
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
  wvm::VmImage img;
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

void PrintVmTable(const wvm::PersistentRuntime& rt);

// `warpvm serve <file.wvm> [--vms N] [--for SECONDS]` — launch the persistent
// kernel with N copies of the program, boot them, print a `list` table once a
// second, then shut down. Bounded by design (display-GPU friendly).
int CmdServe(const char* path, uint32_t n_vms, int seconds) {
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
    images[i].mem_size_words = 16384;
  }

  wvm::PersistentRuntime rt;
  if (!rt.Init(images, err) || !rt.Launch(err)) {
    std::fprintf(stderr, "error: %s\n", err.c_str());
    return 1;
  }
  rt.BootAll();
  std::printf("serving %u VMs from %s for %ds\n", n_vms, path, seconds);

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
int CmdAttach(const char* path, uint32_t n_vms) {
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
    images[i].mem_size_words = 16384;
  }
  wvm::PersistentRuntime rt;
  if (!rt.Init(images, err) || !rt.Launch(err)) {
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
  std::printf("attached: %u VMs from %s\n", n_vms, path);
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
  std::printf("  %-4s %-9s %-8s %-6s %s\n", "vm", "status", "fault", "pc",
              "instrs");
  for (uint32_t i = 0; i < rt.num_vms(); ++i) {
    std::printf("  %-4u %-9s %-8s %-6u %llu\n", i,
                wvm::StatusName(rt.Status(i)), wvm::FaultName(rt.Fault(i)),
                rt.Pc(i), (unsigned long long)rt.Instrs(i));
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

int RunSlice7() {
  using wvm::enc_i;
  using wvm::enc_r;
  bool ok = true;

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
    image.mem_size_words = 16384;
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
  std::printf("  serve <file.wvm> [--vms N] [--for S]\n");
  std::printf("                  boot N resident VMs, print `list` for S seconds\n");
  std::printf("  attach <file.wvm> [--vms N]\n");
  std::printf("                  boot N resident VMs, interactive console on stdin\n");
  std::printf("  demo <file.wvm> [--vms N]\n");
  std::printf("                  v0.1 capstone: N VMs compute + ring-message + stay live\n");
  std::printf("  slice1..slice5,slice7  self-test suites\n");
  std::printf("  gfxa | gfxb            v0.1.1 graphics slices A/B self-tests\n");
  std::printf("  gfxsmoke <file.wvm>    headless framebuffer-copy smoke test\n");
  std::printf("  gfx_cap <file.wvm> [--vms N]\n");
  std::printf("                  v0.1.1 capstone: N VMs render distinct images\n");
  std::printf("  view <file.wvm> [--vm N | --vms N]\n");
  std::printf("                  show one VM or a tiled grid of N resident VMs\n");
  std::printf("  life_test <file.wvm> Program 01 packed-Life correctness test\n");
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
  if (std::strcmp(cmd, "slice5") == 0) return RunSlice5();
  if (std::strcmp(cmd, "slice7") == 0) return RunSlice7();
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
    for (int i = 3; i < argc; ++i) {
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
    if (select_grid) return wvm::ViewVmGrid(argv[2], n_vms);
    return wvm::ViewSingleVm(argv[2], vm_index);
  }
  if (std::strcmp(cmd, "life_test") == 0) {
    if (argc < 3) {
      std::fprintf(stderr, "error: life_test requires a .wvm file\n");
      return 2;
    }
    return RunLifeTest(argv[2]);
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
    for (int i = 3; i < argc; ++i) {
      if (std::strcmp(argv[i], "--vms") == 0 && i + 1 < argc) {
        n_vms = static_cast<uint32_t>(std::atoi(argv[++i]));
      } else if (std::strcmp(argv[i], "--for") == 0 && i + 1 < argc) {
        seconds = std::atoi(argv[++i]);
      } else {
        std::fprintf(stderr, "error: unknown serve option '%s'\n", argv[i]);
        return 2;
      }
    }
    return CmdServe(argv[2], n_vms, seconds);
  }
  if (std::strcmp(cmd, "attach") == 0) {
    if (argc < 3) {
      std::fprintf(stderr, "error: attach requires a .wvm file\n");
      return 2;
    }
    uint32_t n_vms = 4;
    for (int i = 3; i < argc; ++i) {
      if (std::strcmp(argv[i], "--vms") == 0 && i + 1 < argc)
        n_vms = static_cast<uint32_t>(std::atoi(argv[++i]));
    }
    return CmdAttach(argv[2], n_vms);
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
