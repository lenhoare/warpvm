// Host-side runtime for the persistent kernel: owns the mapped control
// plane, per-VM device state, and the resident launch. Provides the host
// commands (boot / pause / resume / reset / shutdown), status reads for
// `warpvm list`, and log snapshots.
//
// Include this from a CUDA translation unit (it launches the kernel).
#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>

#include "gpu/control.cuh"
#include "gpu/vm_state.cuh"
#include "host/program_registry.h"
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
    if (vm_ids.size() != images.size()) {
      err = "logical VM ID count must match resident slot count";
      return false;
    }
    ProgramRegistry registry;
    std::vector<VmBinding> bindings(images.size());
    for (VmSlot slot = 0; slot < images.size(); ++slot) {
      LoadedProgramId program_id;
      const WvmFile candidate{images[slot].code, images[slot].literals};
      bool found = false;
      for (const ProgramInfo& program : registry.programs()) {
        if (program.image.code == candidate.code &&
            program.image.literals == candidate.literals) {
          program_id = program.id;
          found = true;
          break;
        }
      }
      if (!found &&
          !registry.Add("legacy-program-" + std::to_string(registry.size()),
                        "", candidate, program_id, err))
        return false;
      bindings[slot].vm_id = vm_ids[slot];
      bindings[slot].program_id = program_id;
      bindings[slot].mem_size_words = images[slot].mem_size_words;
      bindings[slot].mem_init = images[slot].mem_init;
    }
    return Init(registry, bindings, err);
  }

  // Static heterogeneous population. This is a convenience wrapper over the
  // fixed-capacity runtime used by the supervisor.
  bool Init(const ProgramRegistry& registry,
            const std::vector<VmBinding>& bindings, std::string& err) {
    if (!InitCapacity(registry, static_cast<uint32_t>(bindings.size()), err))
      return false;
    for (VmSlot slot = 0; slot < bindings.size(); ++slot) {
      if (!BindSlot(registry, slot, bindings[slot], err)) return false;
    }
    return true;
  }

  // Allocate a resident population with no VM instances bound. RAM,
  // framebuffer, descriptor, state, mailbox, and control storage all exist
  // for every slot before launch, so later create/delete operations do not
  // need to grow the CUDA kernel or allocate memory behind a resident kernel.
  bool InitCapacity(const ProgramRegistry& registry, uint32_t capacity,
                    std::string& err) {
    num_vms_ = capacity;
    if (num_vms_ == 0 || num_vms_ > kMaxVms) {
      err = "vm count must be 1.." + std::to_string(kMaxVms);
      return false;
    }
    h_vm_ids_.assign(num_vms_, kInvalidVmId);
    h_vm_routes_.assign(kVmIdCount, kInvalidVmSlot);
    h_images_.resize(num_vms_);
    h_program_ids_.assign(num_vms_, kInvalidVmId);
    h_program_names_.resize(num_vms_);

    if (cudaMalloc(reinterpret_cast<void**>(&d_slot_program_ids_),
                   num_vms_ * sizeof(ProgramId)) != cudaSuccess) {
      err = "cudaMalloc slot program IDs failed";
      return false;
    }
    if (cudaMemcpy(d_slot_program_ids_, h_program_ids_.data(),
                   num_vms_ * sizeof(ProgramId),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
      err = "slot program ID initialization failed";
      return false;
    }

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
    if (cudaMemset(d_framebuffers_, 0,
                   static_cast<size_t>(num_vms_) * kVideoWords *
                       sizeof(uint32_t)) != cudaSuccess) {
      err = "framebuffer initialization failed";
      return false;
    }

    for (const ProgramInfo& program : registry.programs()) {
      DeviceProgram device_program;
      device_program.id = program.id.value;
      device_program.code_len = static_cast<uint32_t>(program.image.code.size());
      device_program.literals_len =
          static_cast<uint32_t>(program.image.literals.size());
      if (cudaMalloc(&device_program.code,
                     program.image.code.size() * sizeof(uint32_t)) !=
          cudaSuccess) {
        err = "cudaMalloc shared program code failed";
        return false;
      }
      if (cudaMemcpy(device_program.code, program.image.code.data(),
                     program.image.code.size() * sizeof(uint32_t),
                     cudaMemcpyHostToDevice) != cudaSuccess) {
        err = "shared program code upload failed";
        return false;
      }
      if (!program.image.literals.empty()) {
        if (cudaMalloc(&device_program.literals,
                       program.image.literals.size() * sizeof(uint32_t)) !=
            cudaSuccess) {
          err = "cudaMalloc shared program literals failed";
          return false;
        }
        if (cudaMemcpy(device_program.literals,
                       program.image.literals.data(),
                       program.image.literals.size() * sizeof(uint32_t),
                       cudaMemcpyHostToDevice) != cudaSuccess) {
          err = "shared program literal upload failed";
          return false;
        }
      }
      d_programs_.push_back(device_program);
    }

    for (VmSlot slot = 0; slot < num_vms_; ++slot) {
      if (cudaMalloc(&d_mem_[slot], kRamSizeWords * sizeof(uint32_t)) !=
          cudaSuccess) {
        err = "cudaMalloc resident RAM slot failed";
        return false;
      }
      if (cudaMemset(d_mem_[slot], 0,
                     kRamSizeWords * sizeof(uint32_t)) != cudaSuccess) {
        err = "resident RAM initialization failed";
        return false;
      }
    }

    std::vector<VmDesc> descs(num_vms_);
    for (VmSlot slot = 0; slot < num_vms_; ++slot)
      descs[slot] = VmDesc{nullptr, 0, nullptr, 0, d_mem_[slot], 0,
                           d_framebuffers_ +
                               static_cast<size_t>(slot) * kVideoWords,
                           d_vm_routes_};
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
      states[slot].vm_id = kInvalidVmId;
      states[slot].status = kIdle;
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
    for (VmSlot owner_slot = 0; owner_slot < num_vms_; ++owner_slot) {
      Mailbox& mailbox = mailboxes[owner_slot];
      std::memset(&mailbox, 0, sizeof(mailbox));
      mailbox.owner_vm_id = kInvalidVmId;
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

  // Bind one EMPTY slot. Program storage must already be present in the
  // registry supplied to InitCapacity; route publication occurs last.
  bool BindSlot(const ProgramRegistry& registry, VmSlot slot,
                const VmBinding& binding, std::string& err) {
    if (slot >= num_vms_) {
      err = "resident slot is outside configured capacity";
      return false;
    }
    if (h_vm_ids_[slot] != kInvalidVmId) {
      err = "resident slot is already occupied";
      return false;
    }
    if (binding.engine != ExecutionEngine::kInterpreted &&
        binding.engine != ExecutionEngine::kCompiled) {
      err = "unknown resident execution engine";
      return false;
    }
    const ProgramInfo* program = registry.Find(binding.program_id);
    const DeviceProgram* device_program = FindDeviceProgram(binding.program_id);
    if (program == nullptr || device_program == nullptr) {
      err = "resident slot references a program not loaded at launch";
      return false;
    }
    const VmId vm_id = binding.vm_id.value;
    if (vm_id >= kVmIdCount) {
      err = "logical VM ID must fit the 16-bit message address space";
      return false;
    }
    if (h_vm_routes_[vm_id] != kInvalidVmSlot) {
      err = "logical VM ID is already resident";
      return false;
    }
    if (binding.mem_size_words > kRamSizeWords) {
      err = "VM RAM request exceeds fixed resident-slot capacity";
      return false;
    }

    if (cudaMemset(d_mem_[slot], 0,
                   kRamSizeWords * sizeof(uint32_t)) != cudaSuccess) {
      err = "VM RAM reset failed";
      return false;
    }
    const size_t seed =
        std::min<size_t>(binding.mem_init.size(), binding.mem_size_words);
    if (seed != 0 &&
        cudaMemcpy(d_mem_[slot], binding.mem_init.data(),
                   seed * sizeof(uint32_t), cudaMemcpyHostToDevice) !=
            cudaSuccess) {
      err = "VM initial RAM upload failed";
      return false;
    }
    std::vector<uint32_t> black(kVideoWords, kVideoResetColor);
    if (cudaMemcpy(d_framebuffers_ + static_cast<size_t>(slot) * kVideoWords,
                   black.data(), kVideoWords * sizeof(uint32_t),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
      err = "VM framebuffer reset failed";
      return false;
    }

    const VmDesc desc{device_program->code,
                      device_program->code_len,
                      device_program->literals,
                      device_program->literals_len,
                      d_mem_[slot],
                      binding.mem_size_words,
                      d_framebuffers_ + static_cast<size_t>(slot) * kVideoWords,
                      d_vm_routes_};
    VmState state{};
    state.vm_id = vm_id;
    state.status = kIdle;
    state.rng_state = vm_id * 0x9E3779B9u + 0x1234567u;
    if (cudaMemcpy(&d_descs_[slot], &desc, sizeof(desc),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(&d_states_[slot], &state, sizeof(state),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
      err = "VM descriptor/state binding failed";
      return false;
    }
    if (!launched_) {
      Mailbox mailbox{};
      mailbox.owner_vm_id = vm_id;
      for (uint32_t entry = 0; entry < kMailboxSlots; ++entry)
        mailbox.slots[entry].sequence = entry;
      if (cudaMemcpy(&d_mailboxes_[slot], &mailbox, sizeof(mailbox),
                     cudaMemcpyHostToDevice) != cudaSuccess) {
        err = "VM mailbox binding failed";
        return false;
      }
    } else if (cudaMemcpy(
                   const_cast<uint32_t*>(&d_mailboxes_[slot].owner_vm_id),
                   &vm_id, sizeof(vm_id), cudaMemcpyHostToDevice) !=
               cudaSuccess) {
      err = "VM mailbox owner publication failed";
      return false;
    }

    h_vm_ids_[slot] = vm_id;
    h_program_ids_[slot] = binding.program_id.value;
    h_program_names_[slot] = program->name;
    h_images_[slot].code = program->image.code;
    h_images_[slot].literals = program->image.literals;
    h_images_[slot].mem_size_words = binding.mem_size_words;
    h_images_[slot].mem_init = binding.mem_init;
    h_vm_routes_[vm_id] = slot;
    if (cudaMemcpy(&d_vm_routes_[vm_id], &slot, sizeof(slot),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
      h_vm_routes_[vm_id] = kInvalidVmSlot;
      err = "VM route publication failed";
      return false;
    }
    // Publish the body selector last. A compiled resident warp may observe it
    // immediately and enter the selected body, so every descriptor/state,
    // mailbox and route dependency must already be valid.
    const ProgramId program_id = binding.program_id.value;
    if (cudaMemcpy(&d_slot_program_ids_[slot], &program_id,
                   sizeof(program_id), cudaMemcpyHostToDevice) !=
        cudaSuccess) {
      h_vm_routes_[vm_id] = kInvalidVmSlot;
      const VmSlot invalid_route = kInvalidVmSlot;
      cudaMemcpy(&d_vm_routes_[vm_id], &invalid_route, sizeof(invalid_route),
                 cudaMemcpyHostToDevice);
      err = "slot program ID publication failed";
      return false;
    }
    h_ctrl_->cmd[slot] = kCmdNone;
    h_ctrl_->status[slot] = kIdle;
    h_ctrl_->fault[slot] = kFaultOk;
    h_ctrl_->pc[slot] = 0;
    h_ctrl_->instrs[slot] = 0;
    h_ctrl_->frame_seq[slot] = 0;
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
  void* DeviceSlotProgramIds() const { return d_slot_program_ids_; }
  cudaStream_t Stream() const { return stream_; }
  void MarkExternallyLaunched() { launched_ = true; }

  // ---- host commands -----------------------------------------------------
  void SendCmd(uint32_t vm, VmCmd cmd) { h_ctrl_->cmd[vm] = cmd; }
  uint64_t ProfileFrameCycles(uint32_t vm) const {
    return h_ctrl_->profile_frame_cycles[vm];
  }
  void BootAll() {
    for (uint32_t i = 0; i < num_vms_; ++i)
      if (h_vm_ids_[i] != kInvalidVmId) SendCmd(i, kCmdRun);
  }
  void ShutdownAll() { h_ctrl_->shutdown = 1u; }

  bool StartSlot(VmSlot slot, std::string& err) {
    if (!RequireOccupied(slot, err)) return false;
    if (Status(slot) != kIdle) {
      err = "VM start requires READY state";
      return false;
    }
    SendCmd(slot, kCmdRun);
    return true;
  }

  bool StopSlot(VmSlot slot, int timeout_ms, std::string& err) {
    if (!RequireOccupied(slot, err)) return false;
    if (Status(slot) == kPaused) return true;
    if (Status(slot) != kRunning) {
      err = "VM stop requires RUNNING state";
      return false;
    }
    if (!Pause(slot, timeout_ms)) {
      err = "timed out waiting for VM to stop";
      return false;
    }
    return true;
  }

  bool ResumeSlot(VmSlot slot, std::string& err, int timeout_ms = 2000) {
    if (!RequireOccupied(slot, err)) return false;
    if (Status(slot) != kPaused) {
      err = "VM resume requires STOPPED state";
      return false;
    }
    SendCmd(slot, kCmdRun);
    for (int waited = 0; waited < timeout_ms; ++waited) {
      if (Status(slot) != kPaused) return true;
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    if (Status(slot) == kPaused) {
      err = "timed out waiting for VM to acknowledge resume";
      return false;
    }
    return true;
  }

  // Remove a binding while keeping the resident CUDA warp available for a
  // future VM. The target warp withdraws its route and drains pinned SENDs;
  // other resident VMs continue executing throughout.
  bool UnbindSlot(VmSlot slot, int timeout_ms, std::string& err) {
    if (!RequireOccupied(slot, err)) return false;
    const VmId old_vm_id = h_vm_ids_[slot];
    if (launched_) {
      if (Status(slot) == kRunning && !StopSlot(slot, timeout_ms, err))
        return false;
      const uint32_t seq0 = h_ctrl_->seq[slot];
      SendCmd(slot, kCmdDeactivate);
      if (!WaitSeq(slot, seq0, timeout_ms)) {
        err = "timed out draining VM mailbox during deletion";
        return false;
      }
    } else {
      h_vm_routes_[old_vm_id] = kInvalidVmSlot;
      if (cudaMemcpy(&d_vm_routes_[old_vm_id],
                     &h_vm_routes_[old_vm_id], sizeof(VmSlot),
                     cudaMemcpyHostToDevice) != cudaSuccess) {
        err = "VM route withdrawal failed";
        return false;
      }
      Mailbox mailbox{};
      mailbox.owner_vm_id = kInvalidVmId;
      for (uint32_t entry = 0; entry < kMailboxSlots; ++entry)
        mailbox.slots[entry].sequence = entry;
      if (cudaMemcpy(&d_mailboxes_[slot], &mailbox, sizeof(mailbox),
                     cudaMemcpyHostToDevice) != cudaSuccess) {
        err = "VM mailbox retirement failed";
        return false;
      }
    }
    h_vm_routes_[old_vm_id] = kInvalidVmSlot;

    // The compiled population wrapper withdraws this selector before its
    // acknowledgement so it cannot re-enter the retired body. Repeat the
    // store here for the interpreter path and as a host-side invariant.
    const ProgramId invalid_program = kInvalidVmId;
    if (cudaMemcpy(&d_slot_program_ids_[slot], &invalid_program,
                   sizeof(invalid_program), cudaMemcpyHostToDevice) !=
        cudaSuccess) {
      err = "slot program ID withdrawal failed";
      return false;
    }

    // Deactivation drains and invalidates a live mailbox but deliberately
    // does not rewrite the ring while senders may still be pinned to it.
    // Once acknowledgement arrives the host owns the slot and can install a
    // pristine ring for the next logical VM.
    Mailbox retired_mailbox{};
    retired_mailbox.owner_vm_id = kInvalidVmId;
    for (uint32_t entry = 0; entry < kMailboxSlots; ++entry)
      retired_mailbox.slots[entry].sequence = entry;
    if (cudaMemcpy(&d_mailboxes_[slot], &retired_mailbox,
                   sizeof(retired_mailbox), cudaMemcpyHostToDevice) !=
        cudaSuccess) {
      err = "VM mailbox retirement failed";
      return false;
    }

    const VmDesc empty_desc{
        nullptr, 0, nullptr, 0, d_mem_[slot], 0,
        d_framebuffers_ + static_cast<size_t>(slot) * kVideoWords,
        d_vm_routes_};
    VmState empty_state{};
    empty_state.vm_id = kInvalidVmId;
    empty_state.status = kIdle;
    if (cudaMemcpy(&d_descs_[slot], &empty_desc, sizeof(empty_desc),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(&d_states_[slot], &empty_state, sizeof(empty_state),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemset(d_mem_[slot], 0,
                   kRamSizeWords * sizeof(uint32_t)) != cudaSuccess ||
        cudaMemset(d_framebuffers_ + static_cast<size_t>(slot) * kVideoWords,
                   0, kVideoWords * sizeof(uint32_t)) != cudaSuccess) {
      err = "resident slot cleanup failed";
      return false;
    }
    h_vm_ids_[slot] = kInvalidVmId;
    h_program_ids_[slot] = kInvalidVmId;
    h_program_names_[slot].clear();
    h_images_[slot] = VmImage{};
    h_ctrl_->cmd[slot] = kCmdNone;
    h_ctrl_->status[slot] = kIdle;
    h_ctrl_->fault[slot] = kFaultOk;
    h_ctrl_->pc[slot] = 0;
    h_ctrl_->instrs[slot] = 0;
    h_ctrl_->frame_seq[slot] = 0;
    return true;
  }

  bool ResetSlot(const ProgramRegistry& registry, VmSlot slot,
                 int timeout_ms, std::string& err) {
    if (!RequireOccupied(slot, err)) return false;
    VmBinding binding;
    binding.vm_id = LogicalVmId{h_vm_ids_[slot]};
    binding.program_id = LoadedProgramId{h_program_ids_[slot]};
    binding.mem_size_words = h_images_[slot].mem_size_words;
    binding.mem_init = h_images_[slot].mem_init;
    return UnbindSlot(slot, timeout_ms, err) &&
           BindSlot(registry, slot, binding, err);
  }

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
  // Diagnostic snapshot used to audit shared program allocations. Device
  // pointers are opaque on the host but may be compared for identity.
  bool ReadDescriptor(VmSlot slot, VmDesc& out) const {
    if (slot >= num_vms_) return false;
    return cudaMemcpy(&out, &d_descs_[slot], sizeof(VmDesc),
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
  // Diagnostic mailbox snapshot. A RUNNING VM may change the ring while the
  // copy is in flight; stopped VMs provide a stable inspection point.
  bool ReadMailbox(VmSlot slot, Mailbox& out) const {
    if (slot >= num_vms_ || d_mailboxes_ == nullptr) return false;
    return cudaMemcpy(&out, &d_mailboxes_[slot], sizeof(Mailbox),
                      cudaMemcpyDeviceToHost) == cudaSuccess;
  }

  // ---- status reads (mapped memory, no sync needed) -----------------------
  uint32_t num_vms() const { return num_vms_; }
  VmId VmIdAtSlot(VmSlot slot) const {
    return slot < h_vm_ids_.size() ? h_vm_ids_[slot] : kInvalidVmId;
  }
  VmSlot SlotForVmId(VmId id) const {
    return id < h_vm_routes_.size() ? h_vm_routes_[id] : kInvalidVmSlot;
  }
  ProgramId ProgramIdAtSlot(VmSlot slot) const {
    return slot < h_program_ids_.size() ? h_program_ids_[slot]
                                        : static_cast<ProgramId>(-1);
  }
  const std::string& ProgramNameAtSlot(VmSlot slot) const {
    static const std::string empty;
    return slot < h_program_names_.size() ? h_program_names_[slot] : empty;
  }
  size_t device_program_count() const { return d_programs_.size(); }
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

  bool WaitSeq(uint32_t vm, uint32_t previous, int timeout_ms) {
    for (int waited = 0; waited < timeout_ms; waited += 1) {
      if (h_ctrl_->seq[vm] != previous) return true;
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return h_ctrl_->seq[vm] != previous;
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
    if (d_slot_program_ids_)
      cudaFree(d_slot_program_ids_), d_slot_program_ids_ = nullptr;
    if (d_framebuffers_) cudaFree(d_framebuffers_), d_framebuffers_ = nullptr;
    for (const DeviceProgram& program : d_programs_) {
      if (program.code) cudaFree(program.code);
      if (program.literals) cudaFree(program.literals);
    }
    for (auto p : d_mem_) cudaFree(p);
    d_programs_.clear();
    d_mem_.clear();
    h_vm_ids_.clear();
    h_vm_routes_.clear();
    h_program_ids_.clear();
    h_program_names_.clear();
    if (h_ctrl_) cudaFreeHost(h_ctrl_), h_ctrl_ = nullptr;
    d_ctrl_ = nullptr;
    if (stream_) cudaStreamDestroy(stream_), stream_ = nullptr;
    num_vms_ = 0;
    launched_ = false;
  }

 private:
  struct DeviceProgram {
    ProgramId id = 0;
    uint32_t* code = nullptr;
    uint32_t* literals = nullptr;
    uint32_t code_len = 0;
    uint32_t literals_len = 0;
  };

  const DeviceProgram* FindDeviceProgram(LoadedProgramId id) const {
    for (const DeviceProgram& program : d_programs_)
      if (program.id == id.value) return &program;
    return nullptr;
  }

  bool RequireOccupied(VmSlot slot, std::string& err) const {
    if (slot >= num_vms_) {
      err = "resident slot is outside configured capacity";
      return false;
    }
    if (h_vm_ids_[slot] == kInvalidVmId) {
      err = "resident slot is EMPTY";
      return false;
    }
    return true;
  }

  Control* h_ctrl_ = nullptr;
  Control* d_ctrl_ = nullptr;
  VmDesc* d_descs_ = nullptr;
  VmState* d_states_ = nullptr;
  Mailbox* d_mailboxes_ = nullptr;
  VmSlot* d_vm_routes_ = nullptr;
  ProgramId* d_slot_program_ids_ = nullptr;
  uint32_t* d_framebuffers_ = nullptr;  // flat pool: num_vms_ * kVideoWords
  std::vector<DeviceProgram> d_programs_;
  std::vector<uint32_t*> d_mem_;
  std::vector<VmImage> h_images_;
  std::vector<VmId> h_vm_ids_;
  std::vector<VmSlot> h_vm_routes_;
  std::vector<ProgramId> h_program_ids_;
  std::vector<std::string> h_program_names_;
  cudaStream_t stream_ = nullptr;
  uint32_t num_vms_ = 0;
  bool launched_ = false;
};

}  // namespace wvm
