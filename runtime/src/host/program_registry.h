// CPU-supervisor program registry and static VM population descriptions.
#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "gpu/warpvm.cuh"
#include "host/vm_identity.h"
#include "host/wvm_file.h"

namespace wvm {

enum class ExecutionEngine : uint32_t {
  kInterpreted = 0,
  kCompiled = 1,
};

struct ProgramInfo {
  LoadedProgramId id;
  std::string name;
  std::string path;
  WvmFile image;
  uint32_t reference_count = 0;
};

struct VmBinding {
  LogicalVmId vm_id;
  LoadedProgramId program_id;
  ExecutionEngine engine = ExecutionEngine::kInterpreted;
  uint32_t mem_size_words = kRamSizeWords;
  std::vector<uint32_t> mem_init;
};

class ProgramRegistry {
 public:
  bool Load(const std::string& path, const std::string& name,
            LoadedProgramId& result, std::string& err) {
    WvmFile image;
    if (!LoadWvm(path, image, err)) return false;
    return Add(name, path, image, result, err);
  }

  bool Add(const std::string& name, const std::string& path,
           const WvmFile& image, LoadedProgramId& result, std::string& err) {
    if (name.empty()) {
      err = "program name must not be empty";
      return false;
    }
    if (names_.find(name) != names_.end()) {
      err = "program name already loaded: " + name;
      return false;
    }
    if (image.code.empty() || image.code.size() > kMaxCodeWords) {
      err = "program code size is outside WarpVM limits";
      return false;
    }
    if (image.literals.size() > kMaxLiterals) {
      err = "program literal count is outside WarpVM limits";
      return false;
    }
    if (next_id_ == kInvalidVmId) {
      err = "program ID space exhausted";
      return false;
    }
    ProgramInfo program;
    program.id.value = next_id_++;
    program.name = name;
    program.path = path;
    program.image = image;
    names_.emplace(program.name, program.id.value);
    programs_.push_back(std::move(program));
    result = programs_.back().id;
    return true;
  }

  const ProgramInfo* Find(LoadedProgramId id) const {
    for (const ProgramInfo& program : programs_)
      if (program.id.value == id.value) return &program;
    return nullptr;
  }

  ProgramInfo* Find(LoadedProgramId id) {
    for (ProgramInfo& program : programs_)
      if (program.id.value == id.value) return &program;
    return nullptr;
  }

  const ProgramInfo* Find(const std::string& name) const {
    const auto found = names_.find(name);
    return found == names_.end() ? nullptr
                                : Find(LoadedProgramId{found->second});
  }

  bool Retain(LoadedProgramId id, std::string& err) {
    ProgramInfo* program = Find(id);
    if (program == nullptr) {
      err = "program ID is not loaded";
      return false;
    }
    ++program->reference_count;
    return true;
  }

  bool Release(LoadedProgramId id, std::string& err) {
    ProgramInfo* program = Find(id);
    if (program == nullptr) {
      err = "program ID is not loaded";
      return false;
    }
    if (program->reference_count == 0) {
      err = "program reference count is already zero";
      return false;
    }
    --program->reference_count;
    return true;
  }

  bool Unload(LoadedProgramId id, std::string& err) {
    for (auto it = programs_.begin(); it != programs_.end(); ++it) {
      if (it->id.value != id.value) continue;
      if (it->reference_count != 0) {
        err = "program is still referenced by " +
              std::to_string(it->reference_count) + " VM(s)";
        return false;
      }
      names_.erase(it->name);
      programs_.erase(it);
      return true;
    }
    err = "program ID is not loaded";
    return false;
  }

  const std::vector<ProgramInfo>& programs() const { return programs_; }
  size_t size() const { return programs_.size(); }

 private:
  std::vector<ProgramInfo> programs_;
  std::unordered_map<std::string, ProgramId> names_;
  ProgramId next_id_ = 0;
};

}  // namespace wvm
