// Whole-program PTX backends for WarpVM.
//
// The bytecode remains canonical. This object owns a CUDA-driver module JITed
// from generated PTX and can execute any number of independent VmState
// instances of that one program, one hardware warp per VM. The resident form
// is the live-machine engine; PtxCompiledProgram remains the bounded
// equivalence/benchmark harness.
#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <cuda.h>

#include "gpu/vm_state.cuh"
#include "host/wvm_file.h"

namespace wvm {

// Pure translation stage. This is intentionally independent of CUDA device
// availability so generated code and unsupported-opcode diagnostics can be
// inspected on build-only hosts.
bool TranslateWvmToPtx(const WvmFile& file, std::string& ptx,
                       std::string& err);
bool TranslateWvmToResidentPtx(const WvmFile& file, std::string& ptx,
                               std::string& err);

// Direct whole-program code running as continuously resident VM warps. The
// caller owns the canonical runtime pools; this object owns only the JITed
// program artifact and launches it asynchronously into the supplied stream.
class PtxResidentProgram {
 public:
  PtxResidentProgram() = default;
  ~PtxResidentProgram();
  PtxResidentProgram(const PtxResidentProgram&) = delete;
  PtxResidentProgram& operator=(const PtxResidentProgram&) = delete;

  bool Compile(const WvmFile& file, std::string& err);
  bool Launch(CUdeviceptr states, uint32_t num_vms, CUdeviceptr descs,
              CUdeviceptr control, CUdeviceptr mailboxes, CUstream stream,
              std::string& err) const;
  const std::string& ptx() const { return ptx_; }
  double jit_milliseconds() const { return jit_milliseconds_; }

 private:
  CUdevice device_ = 0;
  CUcontext context_ = nullptr;
  bool retained_primary_context_ = false;
  CUmodule module_ = nullptr;
  CUfunction function_ = nullptr;
  std::string ptx_;
  double jit_milliseconds_ = 0.0;
};

class PtxCompiledProgram {
 public:
  PtxCompiledProgram() = default;
  ~PtxCompiledProgram();

  PtxCompiledProgram(const PtxCompiledProgram&) = delete;
  PtxCompiledProgram& operator=(const PtxCompiledProgram&) = delete;

  // Unsupported bytecode is rejected with a source-PC error.
  bool Compile(const WvmFile& file, std::string& err);

  // Execute from canonical states supplied by the caller through a safe exit.
  bool Launch(std::vector<VmState>& states, std::string& err) const;

  // Execute with canonical private RAM and framebuffer storage. Buffers are
  // densely packed by VM; every VM in one batch has the same RAM size.
  bool Launch(std::vector<VmState>& states, std::vector<uint32_t>& memory,
              uint32_t memory_words, std::vector<uint32_t>& framebuffers,
              std::vector<uint32_t>& frame_seq, std::string& err) const;

  // Bounded test harness: run one native launch per HALT/YIELD checkpoint.
  // This is not the normal resident VM scheduler. Timing excludes allocation
  // and host/device copies but includes launches and device execution.
  bool LaunchCheckpoints(std::vector<VmState>& states,
                         std::vector<uint32_t>& memory,
                         uint32_t memory_words,
                         std::vector<uint32_t>& framebuffers,
                         std::vector<uint32_t>& frame_seq,
                         uint32_t checkpoints, double& elapsed_milliseconds,
                         std::string& err) const;

  const std::string& ptx() const { return ptx_; }
  double jit_milliseconds() const { return jit_milliseconds_; }

 private:
  CUdevice device_ = 0;
  CUcontext context_ = nullptr;
  bool retained_primary_context_ = false;
  CUstream stream_ = nullptr;
  CUmodule module_ = nullptr;
  CUfunction function_ = nullptr;
  mutable CUdeviceptr scratch_states_ = 0;
  mutable CUdeviceptr scratch_memory_ = 0;
  mutable CUdeviceptr scratch_framebuffers_ = 0;
  mutable CUdeviceptr scratch_frame_seq_ = 0;
  mutable CUdeviceptr scratch_mailboxes_ = 0;
  mutable size_t scratch_states_bytes_ = 0;
  mutable size_t scratch_memory_bytes_ = 0;
  mutable size_t scratch_framebuffers_bytes_ = 0;
  mutable size_t scratch_frame_seq_bytes_ = 0;
  mutable size_t scratch_mailboxes_bytes_ = 0;
  std::string ptx_;
  double jit_milliseconds_ = 0.0;
};

// Process-local program-identity cache. Hash collisions are resolved by an
// exact bytecode/literal comparison before an artifact is reused.
class PtxCompilationCache {
 public:
  bool GetOrCompile(const WvmFile& file,
                    std::shared_ptr<PtxCompiledProgram>& program,
                    bool& cache_hit, std::string& err);

  size_t size() const { return size_; }

 private:
  struct Entry {
    WvmFile file;
    std::shared_ptr<PtxCompiledProgram> program;
  };
  std::unordered_map<uint64_t, std::vector<Entry>> entries_;
  size_t size_ = 0;
};

}  // namespace wvm
