// Host-side runtime for the persistent kernel: owns the mapped control
// plane, per-VM device state, and the resident launch. Provides the host
// commands (boot / pause / resume / reset / shutdown), status reads for
// `warpvm list`, and log snapshots.
//
// Include this from a CUDA translation unit (it launches the kernel).
#pragma once

#include <chrono>
#include <cstdint>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include <cuda_runtime.h>

#include "gpu/control.cuh"
#include "gpu/vm_state.cuh"
#include "host/vm_image.h"

namespace wvm {

__global__ void PersistentKernel(const VmDesc* descs, VmState* states,
                                 Control* ctrl, uint32_t num_vms);

struct LogSnapshot {
  uint32_t head = 0;
  std::vector<LogEntry> entries;
};

class PersistentRuntime {
 public:
  ~PersistentRuntime() { Free(); }

  bool Init(const std::vector<VmImage>& images, std::string& err) {
    num_vms_ = static_cast<uint32_t>(images.size());
    if (num_vms_ == 0 || num_vms_ > kMaxVms) {
      err = "vm count must be 1.." + std::to_string(kMaxVms);
      return false;
    }
    h_images_ = images;  // retained for inspection / disassembly

    if (cudaHostAlloc(&h_ctrl_, sizeof(Control), cudaHostAllocMapped) !=
        cudaSuccess) {
      err = "cudaHostAlloc failed";
      return false;
    }
    std::memset(h_ctrl_, 0, sizeof(Control));
    if (cudaHostGetDevicePointer(reinterpret_cast<void**>(&d_ctrl_), h_ctrl_,
                                 0) != cudaSuccess) {
      err = "cudaHostGetDevicePointer failed";
      return false;
    }

    d_code_.assign(num_vms_, nullptr);
    d_lit_.assign(num_vms_, nullptr);
    d_mem_.assign(num_vms_, nullptr);
    std::vector<VmDesc> descs(num_vms_);

    for (uint32_t i = 0; i < num_vms_; ++i) {
      const VmImage& img = images[i];
      if (cudaMalloc(&d_code_[i], img.code.size() * sizeof(uint32_t)) !=
          cudaSuccess) {
        err = "cudaMalloc code failed";
        return false;
      }
      cudaMemcpy(d_code_[i], img.code.data(),
                 img.code.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
      if (!img.literals.empty()) {
        cudaMalloc(&d_lit_[i], img.literals.size() * sizeof(uint32_t));
        cudaMemcpy(d_lit_[i], img.literals.data(),
                   img.literals.size() * sizeof(uint32_t),
                   cudaMemcpyHostToDevice);
      }
      if (img.mem_size_words > 0) {
        cudaMalloc(&d_mem_[i], img.mem_size_words * sizeof(uint32_t));
        cudaMemset(d_mem_[i], 0, img.mem_size_words * sizeof(uint32_t));
        const size_t seed =
            std::min<size_t>(img.mem_init.size(), img.mem_size_words);
        if (seed > 0)
          cudaMemcpy(d_mem_[i], img.mem_init.data(), seed * sizeof(uint32_t),
                     cudaMemcpyHostToDevice);
      }
      descs[i] = VmDesc{d_code_[i],
                        static_cast<uint32_t>(img.code.size()),
                        d_lit_[i],
                        static_cast<uint32_t>(img.literals.size()),
                        d_mem_[i],
                        img.mem_size_words};
    }

    if (cudaMalloc(reinterpret_cast<void**>(&d_descs_),
                   num_vms_ * sizeof(VmDesc)) != cudaSuccess) {
      err = "cudaMalloc descs failed";
      return false;
    }
    cudaMemcpy(d_descs_, descs.data(), num_vms_ * sizeof(VmDesc),
               cudaMemcpyHostToDevice);
    if (cudaMalloc(reinterpret_cast<void**>(&d_states_),
                   num_vms_ * sizeof(VmState)) != cudaSuccess) {
      err = "cudaMalloc states failed";
      return false;
    }
    cudaMemset(d_states_, 0, num_vms_ * sizeof(VmState));
    return true;
  }

  bool Launch(std::string& err) {
    // Launch on a non-blocking stream: the legacy default stream would make
    // cudaMemcpy / host reads implicitly wait on the resident kernel and
    // deadlock. Non-blocking streams are exempt from that synchronisation.
    if (stream_ == nullptr &&
        cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking) !=
            cudaSuccess) {
      err = "cudaStreamCreate failed";
      return false;
    }
    const int block = 256;
    const int grid = static_cast<int>((num_vms_ * kLanes + block - 1) / block);
    PersistentKernel<<<grid, block, 0, stream_>>>(d_descs_, d_states_, d_ctrl_,
                                                  num_vms_);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
      err = std::string("kernel launch failed: ") + cudaGetErrorString(e);
      return false;
    }
    launched_ = true;
    return true;
  }

  // ---- host commands -----------------------------------------------------
  void SendCmd(uint32_t vm, VmCmd cmd) { h_ctrl_->cmd[vm] = cmd; }
  void BootAll() {
    for (uint32_t i = 0; i < num_vms_; ++i) SendCmd(i, kCmdRun);
  }
  void ShutdownAll() { h_ctrl_->shutdown = 1u; }

  // Pause a VM and wait for it to reach the control point.
  bool Pause(uint32_t vm, int timeout_ms = 2000) {
    SendCmd(vm, kCmdPause);
    return WaitStatus(vm, kPaused, timeout_ms);
  }
  // Single-step a paused VM: retire one instruction and re-pause. Waits on the
  // per-VM seq counter (status stays PAUSED, so status polling can't detect
  // completion). Returns the post-step status (PAUSED, HALTED or FAULTED).
  uint32_t Step(uint32_t vm, int timeout_ms = 2000) {
    if (vm >= num_vms_) return kFaulted;
    const uint32_t seq0 = h_ctrl_->seq[vm];
    SendCmd(vm, kCmdStep);
    for (int waited = 0; waited < timeout_ms; waited += 1) {
      if (h_ctrl_->seq[vm] != seq0) return Status(vm);
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return Status(vm);
  }

  // ---- inspection while resident (no sync needed) -------------------------
  // Copy a VM's spilled register/state block to the host.
  bool ReadState(uint32_t vm, VmState& out) const {
    if (vm >= num_vms_) return false;
    return cudaMemcpy(&out, &d_states_[vm], sizeof(VmState),
                      cudaMemcpyDeviceToHost) == cudaSuccess;
  }
  // Copy `count` words of a VM's RAM starting at `addr`.
  bool ReadMem(uint32_t vm, uint32_t addr, uint32_t count,
               std::vector<uint32_t>& out) const {
    if (vm >= num_vms_) return false;
    const uint32_t size = h_images_[vm].mem_size_words;
    if (d_mem_[vm] == nullptr || addr >= size) return false;
    if (count > size - addr) count = size - addr;
    out.resize(count);
    if (count == 0) return true;
    return cudaMemcpy(out.data(), d_mem_[vm] + addr,
                      count * sizeof(uint32_t),
                      cudaMemcpyDeviceToHost) == cudaSuccess;
  }
  uint32_t MemSize(uint32_t vm) const {
    return vm < num_vms_ ? h_images_[vm].mem_size_words : 0;
  }
  // The program a VM was loaded with (for disassembly).
  const std::vector<uint32_t>& Code(uint32_t vm) const {
    return h_images_[vm].code;
  }
  const std::vector<uint32_t>& Literals(uint32_t vm) const {
    return h_images_[vm].literals;
  }

  // ---- status reads (mapped memory, no sync needed) -----------------------
  uint32_t num_vms() const { return num_vms_; }
  uint32_t Status(uint32_t vm) const { return h_ctrl_->status[vm]; }
  uint32_t Fault(uint32_t vm) const { return h_ctrl_->fault[vm]; }
  uint32_t Pc(uint32_t vm) const { return h_ctrl_->pc[vm]; }
  uint64_t Instrs(uint32_t vm) const { return h_ctrl_->instrs[vm]; }

  bool WaitStatus(uint32_t vm, uint32_t want, int timeout_ms) {
    for (int waited = 0; waited < timeout_ms; waited += 1) {
      if (Status(vm) == want) return true;
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return Status(vm) == want;
  }

  LogSnapshot ReadLog() {
    LogSnapshot s;
    s.head = h_ctrl_->log_head;
    const uint32_t n = s.head < kLogCapacity ? s.head : kLogCapacity;
    s.entries.reserve(n);
    for (uint32_t i = 0; i < n; ++i) {
      const uint32_t idx = (s.head - n + i) % kLogCapacity;
      s.entries.push_back(h_ctrl_->log[idx]);
    }
    return s;
  }

  // Block until the resident kernel has exited (call after ShutdownAll).
  cudaError_t Sync() {
    return stream_ ? cudaStreamSynchronize(stream_) : cudaDeviceSynchronize();
  }

  void Free() {
    if (d_descs_) cudaFree(d_descs_), d_descs_ = nullptr;
    if (d_states_) cudaFree(d_states_), d_states_ = nullptr;
    for (auto p : d_code_) cudaFree(p);
    for (auto p : d_lit_) cudaFree(p);
    for (auto p : d_mem_) cudaFree(p);
    d_code_.clear();
    d_lit_.clear();
    d_mem_.clear();
    if (h_ctrl_) cudaFreeHost(h_ctrl_), h_ctrl_ = nullptr;
    d_ctrl_ = nullptr;
    if (stream_) cudaStreamDestroy(stream_), stream_ = nullptr;
  }

 private:
  Control* h_ctrl_ = nullptr;
  Control* d_ctrl_ = nullptr;
  VmDesc* d_descs_ = nullptr;
  VmState* d_states_ = nullptr;
  std::vector<uint32_t*> d_code_, d_lit_, d_mem_;
  std::vector<VmImage> h_images_;
  cudaStream_t stream_ = nullptr;
  uint32_t num_vms_ = 0;
  bool launched_ = false;
};

}  // namespace wvm
