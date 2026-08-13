// CPU-side owner of program objects, stable VM identities, resident slots,
// and the deliberately small WarpVM lifecycle.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "host/persistent.h"
#include "host/program_registry.h"
#include "host/ptx_compiler.h"
#include "host/vm_identity.h"

namespace wvm {

enum class VmLifecycle : uint32_t {
  kEmpty,
  kReady,
  kRunning,
  kStopped,
  kHalted,
  kFaulted,
};

inline const char* LifecycleName(VmLifecycle state) {
  switch (state) {
    case VmLifecycle::kEmpty: return "EMPTY";
    case VmLifecycle::kReady: return "READY";
    case VmLifecycle::kRunning: return "RUNNING";
    case VmLifecycle::kStopped: return "STOPPED";
    case VmLifecycle::kHalted: return "HALTED";
    case VmLifecycle::kFaulted: return "FAULTED";
  }
  return "?";
}

struct VmInstanceInfo {
  LogicalVmId vm_id;
  ResidentSlotId slot;
  LoadedProgramId program_id;
  ExecutionEngine engine = ExecutionEngine::kInterpreted;
  VmLifecycle lifecycle = VmLifecycle::kEmpty;
};

class Supervisor {
 public:
  ~Supervisor() { Shutdown(); }

  ProgramRegistry& programs() { return programs_; }
  const ProgramRegistry& programs() const { return programs_; }
  PersistentRuntime& runtime() { return runtime_; }
  const PersistentRuntime& runtime() const { return runtime_; }
  bool launched() const { return launched_; }
  uint32_t capacity() const {
    return static_cast<uint32_t>(instances_.size());
  }
  ExecutionEngine population_engine() const { return population_engine_; }
  size_t compiled_program_count() const {
    return compiled_population_ ? compiled_population_->program_count() : 0;
  }
  size_t compiled_ptx_bytes() const {
    return compiled_population_ ? compiled_population_->ptx().size() : 0;
  }
  double compiled_jit_milliseconds() const {
    return compiled_population_ ? compiled_population_->jit_milliseconds()
                                : 0.0;
  }

  bool ProgramLoad(const std::string& path, const std::string& name,
                   LoadedProgramId& result, std::string& err) {
    if (launched_) {
      err = "programs must currently be loaded before resident launch; "
            "adding device code requires a controlled relaunch";
      return false;
    }
    return programs_.Load(path, name, result, err);
  }

  bool ProgramAdd(const std::string& name, const WvmFile& image,
                  LoadedProgramId& result, std::string& err) {
    if (launched_) {
      err = "programs must currently be loaded before resident launch; "
            "adding device code requires a controlled relaunch";
      return false;
    }
    return programs_.Add(name, "", image, result, err);
  }

  bool ProgramUnload(LoadedProgramId id, std::string& err) {
    if (launched_ && population_engine_ == ExecutionEngine::kCompiled) {
      err = "cannot unload a program referenced by the active compiled "
            "population artifact; quiesce the population first";
      return false;
    }
    return programs_.Unload(id, err);
  }

  bool Launch(uint32_t capacity, std::string& err,
              ExecutionEngine engine = ExecutionEngine::kInterpreted) {
    if (launched_) {
      err = "supervisor population is already resident";
      return false;
    }
    if (programs_.size() == 0) {
      err = "load at least one program before resident launch";
      return false;
    }
    directory_.Reset(capacity);
    instances_.assign(capacity, VmInstanceInfo{});
    for (VmSlot slot = 0; slot < capacity; ++slot)
      instances_[slot].slot = ResidentSlotId{slot};
    if (!runtime_.InitCapacity(programs_, capacity, err)) {
      runtime_.Free();
      return false;
    }
    bool resident_started = false;
    if (engine == ExecutionEngine::kInterpreted) {
      resident_started = runtime_.Launch(err);
    } else {
      std::vector<CompiledPopulationProgram> programs;
      programs.reserve(programs_.size());
      for (const ProgramInfo& program : programs_.programs())
        programs.push_back({program.id.value, program.image});
      compiled_population_ =
          std::make_unique<PtxHeterogeneousResidentProgram>();
      resident_started =
          runtime_.EnsureStream(err) &&
          compiled_population_->Compile(programs, err) &&
          compiled_population_->Launch(
              reinterpret_cast<CUdeviceptr>(runtime_.DeviceStates()),
              capacity,
              reinterpret_cast<CUdeviceptr>(runtime_.DeviceDescs()),
              reinterpret_cast<CUdeviceptr>(runtime_.DeviceControl()),
              reinterpret_cast<CUdeviceptr>(runtime_.DeviceMailboxes()),
              reinterpret_cast<CUdeviceptr>(
                  runtime_.DeviceSlotProgramIds()),
              reinterpret_cast<CUstream>(runtime_.Stream()), err);
      if (resident_started) runtime_.MarkExternallyLaunched();
    }
    if (!resident_started) {
      compiled_population_.reset();
      runtime_.Free();
      return false;
    }
    population_engine_ = engine;
    launched_ = true;
    return true;
  }

  bool VmCreate(LoadedProgramId program_id, LogicalVmId& result,
                std::string& err, uint32_t mem_size_words = kRamSizeWords,
                const std::vector<uint32_t>& mem_init = {}) {
    if (!launched_) {
      err = "supervisor population is not resident";
      return false;
    }
    if (programs_.Find(program_id) == nullptr) {
      err = "program ID is not loaded";
      return false;
    }
    VmSlot free_slot = kInvalidVmSlot;
    for (VmSlot slot = 0; slot < instances_.size(); ++slot) {
      if (instances_[slot].lifecycle == VmLifecycle::kEmpty) {
        free_slot = slot;
        break;
      }
    }
    if (free_slot == kInvalidVmSlot) {
      err = "resident VM capacity is full";
      return false;
    }
    LogicalVmId vm_id;
    if (!directory_.Create(ResidentSlotId{free_slot}, vm_id, err))
      return false;
    VmBinding binding{vm_id, program_id, population_engine_,
                      mem_size_words, mem_init};
    if (!programs_.Retain(program_id, err)) return false;
    if (!runtime_.BindSlot(programs_, free_slot, binding, err)) {
      std::string ignored;
      programs_.Release(program_id, ignored);
      directory_.Retire(vm_id, ignored);
      return false;
    }
    instances_[free_slot] = VmInstanceInfo{
        vm_id, ResidentSlotId{free_slot}, program_id,
        population_engine_, VmLifecycle::kReady};
    result = vm_id;
    return true;
  }

  bool VmStart(LogicalVmId id, std::string& err) {
    VmInstanceInfo* vm = FindMutable(id, err);
    if (vm == nullptr) return false;
    Refresh(*vm);
    if (vm->lifecycle != VmLifecycle::kReady) {
      err = "vm_start requires READY state";
      return false;
    }
    if (!runtime_.StartSlot(vm->slot.value, err)) return false;
    vm->lifecycle = VmLifecycle::kRunning;
    return true;
  }

  bool VmStop(LogicalVmId id, std::string& err, int timeout_ms = 2000) {
    VmInstanceInfo* vm = FindMutable(id, err);
    if (vm == nullptr) return false;
    Refresh(*vm);
    if (vm->lifecycle != VmLifecycle::kRunning) {
      err = "vm_stop requires RUNNING state";
      return false;
    }
    if (!runtime_.StopSlot(vm->slot.value, timeout_ms, err)) return false;
    vm->lifecycle = VmLifecycle::kStopped;
    return true;
  }

  bool VmResume(LogicalVmId id, std::string& err) {
    VmInstanceInfo* vm = FindMutable(id, err);
    if (vm == nullptr) return false;
    Refresh(*vm);
    if (vm->lifecycle != VmLifecycle::kStopped) {
      err = "vm_resume requires STOPPED state";
      return false;
    }
    if (!runtime_.ResumeSlot(vm->slot.value, err)) return false;
    const uint32_t status = runtime_.Status(vm->slot.value);
    vm->lifecycle = status == kHalted
                        ? VmLifecycle::kHalted
                        : status == kFaulted ? VmLifecycle::kFaulted
                                             : VmLifecycle::kRunning;
    return true;
  }

  bool VmReset(LogicalVmId id, std::string& err, int timeout_ms = 2000) {
    VmInstanceInfo* vm = FindMutable(id, err);
    if (vm == nullptr) return false;
    Refresh(*vm);
    if (vm->lifecycle != VmLifecycle::kStopped &&
        vm->lifecycle != VmLifecycle::kHalted &&
        vm->lifecycle != VmLifecycle::kFaulted) {
      err = "vm_reset requires STOPPED, HALTED, or FAULTED state";
      return false;
    }
    if (!runtime_.ResetSlot(programs_, vm->slot.value, timeout_ms, err))
      return false;
    vm->lifecycle = VmLifecycle::kReady;
    return true;
  }

  bool VmSetEngine(LogicalVmId id, ExecutionEngine engine,
                   std::string& err) {
    VmInstanceInfo* vm = FindMutable(id, err);
    if (vm == nullptr) return false;
    Refresh(*vm);
    if (vm->lifecycle == VmLifecycle::kRunning) {
      err = "engine selection requires a non-RUNNING VM";
      return false;
    }
    if (engine != population_engine_) {
      err = "this resident epoch uses the " +
            std::string(population_engine_ == ExecutionEngine::kCompiled
                            ? "compiled"
                            : "interpreted") +
            " engine; simultaneous mixed engines are a later slice";
      return false;
    }
    vm->engine = engine;
    return true;
  }

  // Cold program replacement: preserve the logical computer/address, but
  // clear execution state, RAM, framebuffer, and mailbox and restart at the
  // new program's entry point in READY state.
  bool VmSetProgram(LogicalVmId id, LoadedProgramId new_program_id,
                    std::string& err, int timeout_ms = 2000) {
    VmInstanceInfo* vm = FindMutable(id, err);
    if (vm == nullptr) return false;
    Refresh(*vm);
    if (vm->lifecycle == VmLifecycle::kRunning) {
      err = "program replacement requires a non-RUNNING VM";
      return false;
    }
    if (programs_.Find(new_program_id) == nullptr) {
      err = "replacement program ID is not loaded";
      return false;
    }
    if (new_program_id.value == vm->program_id.value)
      return VmReset(id, err, timeout_ms);

    const VmSlot slot = vm->slot.value;
    const LoadedProgramId old_program_id = vm->program_id;
    const uint32_t mem_size_words = runtime_.MemSize(slot);
    if (!programs_.Retain(new_program_id, err)) return false;
    if (!runtime_.UnbindSlot(slot, timeout_ms, err)) {
      std::string ignored;
      programs_.Release(new_program_id, ignored);
      return false;
    }
    const VmBinding replacement{id, new_program_id, population_engine_,
                                mem_size_words, {}};
    if (!runtime_.BindSlot(programs_, slot, replacement, err)) {
      std::string ignored;
      programs_.Release(new_program_id, ignored);
      return false;
    }
    if (!programs_.Release(old_program_id, err)) return false;
    vm->program_id = new_program_id;
    vm->engine = population_engine_;
    vm->lifecycle = VmLifecycle::kReady;
    return true;
  }

  bool VmDelete(LogicalVmId id, std::string& err, int timeout_ms = 2000) {
    VmInstanceInfo* vm = FindMutable(id, err);
    if (vm == nullptr) return false;
    Refresh(*vm);
    const VmSlot slot = vm->slot.value;
    const LoadedProgramId program_id = vm->program_id;
    if (!runtime_.UnbindSlot(slot, timeout_ms, err)) return false;
    if (!directory_.Retire(id, err)) return false;
    if (!programs_.Release(program_id, err)) return false;
    instances_[slot] = VmInstanceInfo{};
    instances_[slot].slot = ResidentSlotId{slot};
    return true;
  }

  VmInstanceInfo* Find(LogicalVmId id) {
    std::string ignored;
    VmInstanceInfo* vm = FindMutable(id, ignored);
    if (vm != nullptr) Refresh(*vm);
    return vm;
  }

  const std::vector<VmInstanceInfo>& instances() {
    for (VmInstanceInfo& vm : instances_)
      if (vm.lifecycle != VmLifecycle::kEmpty) Refresh(vm);
    return instances_;
  }

  void Shutdown() {
    if (!launched_) return;
    runtime_.ShutdownAll();
    runtime_.Sync();
    compiled_population_.reset();
    runtime_.Free();
    for (VmInstanceInfo& vm : instances_) {
      if (vm.lifecycle == VmLifecycle::kEmpty) continue;
      std::string ignored;
      programs_.Release(vm.program_id, ignored);
    }
    instances_.clear();
    directory_.Reset(0);
    launched_ = false;
    population_engine_ = ExecutionEngine::kInterpreted;
  }

 private:
  VmInstanceInfo* FindMutable(LogicalVmId id, std::string& err) {
    const VmSlot slot = directory_.SlotFor(id);
    if (slot == kInvalidVmSlot || slot >= instances_.size() ||
        instances_[slot].lifecycle == VmLifecycle::kEmpty) {
      err = "logical VM ID is not active";
      return nullptr;
    }
    return &instances_[slot];
  }

  void Refresh(VmInstanceInfo& vm) {
    if (vm.lifecycle != VmLifecycle::kRunning) return;
    const uint32_t status = runtime_.Status(vm.slot.value);
    if (status == kPaused)
      vm.lifecycle = VmLifecycle::kStopped;
    else if (status == kHalted)
      vm.lifecycle = VmLifecycle::kHalted;
    else if (status == kFaulted)
      vm.lifecycle = VmLifecycle::kFaulted;
  }

  ProgramRegistry programs_;
  VmDirectory directory_;
  PersistentRuntime runtime_;
  std::unique_ptr<PtxHeterogeneousResidentProgram> compiled_population_;
  std::vector<VmInstanceInfo> instances_;
  ExecutionEngine population_engine_ = ExecutionEngine::kInterpreted;
  bool launched_ = false;
};

}  // namespace wvm
