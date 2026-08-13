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
#include "host/vm_identity.h"
#include "host/vm_image.h"

namespace wvm {

__global__ void PersistentKernel(const VmDesc* descs, VmState* states,
                                 Control* ctrl, uint32_t num_vms,
                                 Mailbox* mailboxes);
__global__ void PersistentScalarRegsKernel(const VmDesc*, VmState*, Control*,
                                           uint32_t, Mailbox*);
__global__ void PersistentDenseDispatchKernel(const VmDesc*, VmState*,
                                              Control*, uint32_t, Mailbox*);
__global__ void PersistentHotDispatchKernel(const VmDesc*, VmState*, Control*,
                                            uint32_t, Mailbox*);
__global__ void PersistentHot4DispatchKernel(const VmDesc*, VmState*, Control*,
                                             uint32_t, Mailbox*);
__global__ void PersistentCycleProfileKernel(const VmDesc*, VmState*, Control*,
                                             uint32_t, Mailbox*);
__global__ void PersistentHotCycleProfileKernel(const VmDesc*, VmState*,
                                                Control*, uint32_t, Mailbox*);
__global__ void PersistentHot4CycleProfileKernel(const VmDesc*, VmState*,
                                                 Control*, uint32_t,
                                                 Mailbox*);
__global__ void PersistentScalarDenseKernel(const VmDesc*, VmState*, Control*,
                                            uint32_t, Mailbox*);
__global__ void PersistentSharedRegsKernel(const VmDesc*, VmState*, Control*,
                                           uint32_t, Mailbox*);
__global__ void PersistentSharedRegsThreeBlockKernel(const VmDesc*, VmState*,
                                                     Control*, uint32_t,
                                                     Mailbox*);
__global__ void PersistentSharedDenseThreeBlockKernel(const VmDesc*, VmState*,
                                                      Control*, uint32_t,
                                                      Mailbox*);
__global__ void PersistentNoFaultVoteKernel(const VmDesc*, VmState*, Control*,
                                            uint32_t, Mailbox*);
__global__ void PersistentYieldPollKernel(const VmDesc*, VmState*, Control*,
                                          uint32_t, Mailbox*);
__global__ void PersistentMinimalProfileKernel(const VmDesc*, VmState*,
                                               Control*, uint32_t, Mailbox*);

enum class PersistentKernelMode {
  kNormal,
  kScalarRegs,
  kDenseDispatch,
  kHotDispatch,
  kHot4Dispatch,
  kCycleProfile,
  kHotCycleProfile,
  kHot4CycleProfile,
  kScalarRegsDenseDispatch,
  kSharedRegs,
  kSharedRegsThreeBlock,
  kSharedRegsDenseThreeBlock,
  kNoFaultVotes,
  kYieldOnlyPolling,
  kNoFaultVotesYieldOnlyPolling,
};

struct LogSnapshot {
  uint32_t head = 0;
  std::vector<LogEntry> entries;
};

class PersistentRuntime {
 public:
  ~PersistentRuntime() { Free(); }

  bool Init(const std::vector<VmImage>& images, std::string& err) {
    std::vector<LogicalVmId> vm_ids(images.size());
    for (uint32_t slot = 0; slot < images.size(); ++slot)
      vm_ids[slot].value = slot;
    return Init(images, vm_ids, err);
  }

  // Initialize resident slots with explicit stable architectural identities.
  // Existing callers use the overload above and retain the historical 0..N-1
  // identity assignment. The supervisor uses this form when logical identity
  // and slot placement differ.
  bool Init(const std::vector<VmImage>& images,
            const std::vector<LogicalVmId>& vm_ids, std::string& err) {
    num_vms_ = static_cast<uint32_t>(images.size());
    if (num_vms_ == 0 || num_vms_ > kMaxVms) {
      err = "vm count must be 1.." + std::to_string(kMaxVms);
      return false;
    }
    if (vm_ids.size() != images.size()) {
      err = "logical VM ID count must match resident slot count";
      return false;
    }
    h_vm_ids_.resize(num_vms_);
    h_vm_routes_.assign(kVmIdCount, kInvalidVmSlot);
    for (VmSlot slot = 0; slot < num_vms_; ++slot) {
      const VmId id = vm_ids[slot].value;
      if (id >= kVmIdCount) {
        err = "logical VM ID must fit the 16-bit message address space";
        return false;
      }
      if (h_vm_routes_[id] != kInvalidVmSlot) {
        err = "logical VM IDs must be unique";
        return false;
      }
      h_vm_ids_[slot] = id;
      h_vm_routes_[id] = slot;
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
    if (cudaMalloc(reinterpret_cast<void**>(&d_vm_routes_),
                   kVmIdCount * sizeof(VmSlot)) != cudaSuccess) {
      err = "cudaMalloc VM route directory failed";
      return false;
    }
    if (cudaMemcpy(d_vm_routes_, h_vm_routes_.data(),
                   kVmIdCount * sizeof(VmSlot),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
      err = "VM route directory upload failed";
      return false;
    }
    // One flat framebuffer pool; VM i owns words [i*kVideoWords, (i+1)*...).
    if (cudaMalloc(reinterpret_cast<void**>(&d_framebuffers_),
                   static_cast<size_t>(num_vms_) * kVideoWords *
                       sizeof(uint32_t)) != cudaSuccess) {
      err = "cudaMalloc framebuffers failed";
      return false;
    }
    cudaMemset(d_framebuffers_, 0,
               static_cast<size_t>(num_vms_) * kVideoWords * sizeof(uint32_t));
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
                        img.mem_size_words,
                        d_framebuffers_ + static_cast<size_t>(i) * kVideoWords,
                        d_vm_routes_};
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
    std::vector<VmState> states(num_vms_);
    for (VmSlot slot = 0; slot < num_vms_; ++slot) {
      states[slot].vm_id = h_vm_ids_[slot];
      states[slot].status = kIdle;
      states[slot].rng_state =
          h_vm_ids_[slot] * 0x9E3779B9u + 0x1234567u;
    }
    if (cudaMemcpy(d_states_, states.data(), num_vms_ * sizeof(VmState),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
      err = "initial state upload failed";
      return false;
    }
    if (cudaMalloc(reinterpret_cast<void**>(&d_mailboxes_),
                   num_vms_ * sizeof(Mailbox)) != cudaSuccess) {
      err = "cudaMalloc mailboxes failed";
      return false;
    }
    std::vector<Mailbox> mailboxes(num_vms_);
    for (Mailbox& mailbox : mailboxes) {
      std::memset(&mailbox, 0, sizeof(mailbox));
      for (uint32_t slot = 0; slot < kMailboxSlots; ++slot)
        mailbox.slots[slot].sequence = slot;
    }
    if (cudaMemcpy(d_mailboxes_, mailboxes.data(),
                   num_vms_ * sizeof(Mailbox), cudaMemcpyHostToDevice) !=
        cudaSuccess) {
      err = "mailbox initialization upload failed";
      return false;
    }
    return true;
  }

  bool Launch(std::string& err,
              PersistentKernelMode mode = PersistentKernelMode::kNormal) {
    // Launch on a non-blocking stream: the legacy default stream would make
    // cudaMemcpy / host reads implicitly wait on the resident kernel and
    // deadlock. Non-blocking streams are exempt from that synchronisation.
    if (stream_ == nullptr &&
        cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking) !=
            cudaSuccess) {
      err = "cudaStreamCreate failed";
      return false;
    }
    const int block = kPersistentBlockThreads;
    const int grid = static_cast<int>((num_vms_ * kLanes + block - 1) / block);
    switch (mode) {
      case PersistentKernelMode::kNormal:
        PersistentKernel<<<grid, block, 0, stream_>>>(
            d_descs_, d_states_, d_ctrl_, num_vms_, d_mailboxes_);
        break;
      case PersistentKernelMode::kScalarRegs:
        PersistentScalarRegsKernel<<<grid, block, 0, stream_>>>(
            d_descs_, d_states_, d_ctrl_, num_vms_, d_mailboxes_);
        break;
      case PersistentKernelMode::kDenseDispatch:
        PersistentDenseDispatchKernel<<<grid, block, 0, stream_>>>(
            d_descs_, d_states_, d_ctrl_, num_vms_, d_mailboxes_);
        break;
      case PersistentKernelMode::kHotDispatch:
        PersistentHotDispatchKernel<<<grid, block, 0, stream_>>>(
            d_descs_, d_states_, d_ctrl_, num_vms_, d_mailboxes_);
        break;
      case PersistentKernelMode::kHot4Dispatch:
        PersistentHot4DispatchKernel<<<grid, block, 0, stream_>>>(
            d_descs_, d_states_, d_ctrl_, num_vms_, d_mailboxes_);
        break;
      case PersistentKernelMode::kCycleProfile:
        PersistentCycleProfileKernel<<<grid, block, 0, stream_>>>(
            d_descs_, d_states_, d_ctrl_, num_vms_, d_mailboxes_);
        break;
      case PersistentKernelMode::kHotCycleProfile:
        PersistentHotCycleProfileKernel<<<grid, block, 0, stream_>>>(
            d_descs_, d_states_, d_ctrl_, num_vms_, d_mailboxes_);
        break;
      case PersistentKernelMode::kHot4CycleProfile:
        PersistentHot4CycleProfileKernel<<<grid, block, 0, stream_>>>(
            d_descs_, d_states_, d_ctrl_, num_vms_, d_mailboxes_);
        break;
      case PersistentKernelMode::kScalarRegsDenseDispatch:
        PersistentScalarDenseKernel<<<grid, block, 0, stream_>>>(
            d_descs_, d_states_, d_ctrl_, num_vms_, d_mailboxes_);
        break;
      case PersistentKernelMode::kSharedRegs:
        PersistentSharedRegsKernel<<<grid, block, 0, stream_>>>(
            d_descs_, d_states_, d_ctrl_, num_vms_, d_mailboxes_);
        break;
      case PersistentKernelMode::kSharedRegsThreeBlock:
        PersistentSharedRegsThreeBlockKernel<<<grid, block, 0, stream_>>>(
            d_descs_, d_states_, d_ctrl_, num_vms_, d_mailboxes_);
        break;
      case PersistentKernelMode::kSharedRegsDenseThreeBlock:
        PersistentSharedDenseThreeBlockKernel<<<grid, block, 0, stream_>>>(
            d_descs_, d_states_, d_ctrl_, num_vms_, d_mailboxes_);
        break;
      case PersistentKernelMode::kNoFaultVotes:
        PersistentNoFaultVoteKernel<<<grid, block, 0, stream_>>>(
            d_descs_, d_states_, d_ctrl_, num_vms_, d_mailboxes_);
        break;
      case PersistentKernelMode::kYieldOnlyPolling:
        PersistentYieldPollKernel<<<grid, block, 0, stream_>>>(
            d_descs_, d_states_, d_ctrl_, num_vms_, d_mailboxes_);
        break;
      case PersistentKernelMode::kNoFaultVotesYieldOnlyPolling:
        PersistentMinimalProfileKernel<<<grid, block, 0, stream_>>>(
            d_descs_, d_states_, d_ctrl_, num_vms_, d_mailboxes_);
        break;
    }
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
      err = std::string("kernel launch failed: ") + cudaGetErrorString(e);
      return false;
    }
    launched_ = true;
    return true;
  }

  bool EnsureStream(std::string& err) {
    if (stream_ != nullptr) return true;
    if (cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking) !=
        cudaSuccess) {
      err = "cudaStreamCreate failed";
      return false;
    }
    return true;
  }

  void* DeviceDescs() const { return d_descs_; }
  void* DeviceStates() const { return d_states_; }
  void* DeviceControl() const { return d_ctrl_; }
  void* DeviceMailboxes() const { return d_mailboxes_; }
  void* DeviceVmRoutes() const { return d_vm_routes_; }
  cudaStream_t Stream() const { return stream_; }

  // ---- host commands -----------------------------------------------------
  void SendCmd(uint32_t vm, VmCmd cmd) { h_ctrl_->cmd[vm] = cmd; }
  uint64_t ProfileFrameCycles(uint32_t vm) const {
    return h_ctrl_->profile_frame_cycles[vm];
  }
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
  // Copy a VM's entire framebuffer (kVideoWords words) to the host.
  bool ReadFramebuffer(uint32_t vm, std::vector<uint32_t>& out) const {
    if (vm >= num_vms_ || d_framebuffers_ == nullptr) return false;
    out.resize(kVideoWords);
    return cudaMemcpy(out.data(),
                      d_framebuffers_ + static_cast<size_t>(vm) * kVideoWords,
                      kVideoWords * sizeof(uint32_t),
                      cudaMemcpyDeviceToHost) == cudaSuccess;
  }
  // Copy a contiguous range of VM framebuffers with one device transfer.
  // This is the efficient presentation path for the tiled viewer: the
  // physical pool remains a runtime detail and VM logical addressing is
  // unchanged.
  bool ReadFramebuffers(uint32_t first_vm, uint32_t count,
                        std::vector<uint32_t>& out) const {
    if (d_framebuffers_ == nullptr || first_vm >= num_vms_ || count == 0 ||
        count > num_vms_ - first_vm)
      return false;
    const size_t words = static_cast<size_t>(count) * kVideoWords;
    out.resize(words);
    return cudaMemcpy(
               out.data(),
               d_framebuffers_ + static_cast<size_t>(first_vm) * kVideoWords,
               words * sizeof(uint32_t), cudaMemcpyDeviceToHost) ==
           cudaSuccess;
  }

  // ---- status reads (mapped memory, no sync needed) -----------------------
  uint32_t num_vms() const { return num_vms_; }
  VmId VmIdAtSlot(VmSlot slot) const {
    return slot < h_vm_ids_.size() ? h_vm_ids_[slot] : kInvalidVmId;
  }
  VmSlot SlotForVmId(VmId id) const {
    return id < h_vm_routes_.size() ? h_vm_routes_[id] : kInvalidVmSlot;
  }
  uint32_t Status(uint32_t vm) const { return h_ctrl_->status[vm]; }
  uint32_t Fault(uint32_t vm) const { return h_ctrl_->fault[vm]; }
  uint32_t Pc(uint32_t vm) const { return h_ctrl_->pc[vm]; }
  uint64_t Instrs(uint32_t vm) const { return h_ctrl_->instrs[vm]; }
  uint32_t FrameSeq(uint32_t vm) const { return h_ctrl_->frame_seq[vm]; }

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
    if (d_mailboxes_) cudaFree(d_mailboxes_), d_mailboxes_ = nullptr;
    if (d_vm_routes_) cudaFree(d_vm_routes_), d_vm_routes_ = nullptr;
    if (d_framebuffers_) cudaFree(d_framebuffers_), d_framebuffers_ = nullptr;
    for (auto p : d_code_) cudaFree(p);
    for (auto p : d_lit_) cudaFree(p);
    for (auto p : d_mem_) cudaFree(p);
    d_code_.clear();
    d_lit_.clear();
    d_mem_.clear();
    h_vm_ids_.clear();
    h_vm_routes_.clear();
    if (h_ctrl_) cudaFreeHost(h_ctrl_), h_ctrl_ = nullptr;
    d_ctrl_ = nullptr;
    if (stream_) cudaStreamDestroy(stream_), stream_ = nullptr;
  }

 private:
  Control* h_ctrl_ = nullptr;
  Control* d_ctrl_ = nullptr;
  VmDesc* d_descs_ = nullptr;
  VmState* d_states_ = nullptr;
  Mailbox* d_mailboxes_ = nullptr;
  VmSlot* d_vm_routes_ = nullptr;
  uint32_t* d_framebuffers_ = nullptr;  // flat pool: num_vms_ * kVideoWords
  std::vector<uint32_t*> d_code_, d_lit_, d_mem_;
  std::vector<VmImage> h_images_;
  std::vector<VmId> h_vm_ids_;
  std::vector<VmSlot> h_vm_routes_;
  cudaStream_t stream_ = nullptr;
  uint32_t num_vms_ = 0;
  bool launched_ = false;
};

}  // namespace wvm
