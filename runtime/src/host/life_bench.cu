// Program 01 application benchmark.
//
// Measures the real assembled WarpLife program through the persistent CUDA
// interpreter with presentation disabled, then runs a conventional native
// CUDA implementation of the same toroidal Life + ARGB framebuffer workload.
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "gpu/warpvm.cuh"
#include "host/cpu_interpreter.h"
#include "host/persistent.h"
#include "host/ptx_compiler.h"
#include "host/vm_image.h"
#include "host/wvm_file.h"

namespace wvm {
int RunCpuGpuLifeEquivalence(const char* path);
int RunNativeCpuLifeEquivalence(const char* path);
int RunCpuGpuLifeEquivalenceMode(const char* path,
                                 PersistentKernelMode mode,
                                 const char* label);

namespace {

constexpr uint32_t kLifeWidth = 128;
constexpr uint32_t kLifeHeight = 128;
constexpr uint32_t kLifeCells = kLifeWidth * kLifeHeight;
constexpr uint32_t kLifeWords = kLifeCells / 32;

struct WarpMeasurement {
  bool ok = false;
  double seconds = 0.0;
  double avg_generations_per_second = 0.0;
  double min_generations_per_second = 0.0;
  double max_generations_per_second = 0.0;
  double aggregate_cells_per_second = 0.0;
};

struct NativeMeasurement {
  bool ok = false;
  uint32_t generations = 0;
  double seconds = 0.0;
  double generations_per_second = 0.0;
  double aggregate_cells_per_second = 0.0;
};

uint32_t InitialPackedWord(uint32_t vm, uint32_t word) {
  if (vm == 0) {
    if (word == 257) return 0x80000000u;
    if (word == 258) return 0x00000003u;
    return 0;
  }
  if (vm == 1) {
    if (word == 0 || word == 508) return 0x00000001u;
    if (word == 3 || word == 511) return 0x80000000u;
    return 0;
  }

  uint32_t value = vm * 521u + word + 1u;
  value ^= value << 13;
  value ^= value >> 17;
  value ^= value << 5;
  return value;
}

std::vector<uint8_t> MakeInitialCells(uint32_t n_vms) {
  std::vector<uint8_t> cells(static_cast<size_t>(n_vms) * kLifeCells, 0);
  for (uint32_t vm = 0; vm < n_vms; ++vm) {
    for (uint32_t word = 0; word < kLifeWords; ++word) {
      const uint32_t packed = InitialPackedWord(vm, word);
      for (uint32_t bit = 0; bit < 32; ++bit) {
        cells[static_cast<size_t>(vm) * kLifeCells + word * 32 + bit] =
            static_cast<uint8_t>((packed >> bit) & 1u);
      }
    }
  }
  return cells;
}

__global__ void NativeLifeStep(const uint8_t* current, uint8_t* next,
                               uint32_t total_cells) {
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= total_cells) return;

  const uint32_t local = index & (kLifeCells - 1);
  const uint32_t vm_base = index - local;
  const uint32_t x = local & (kLifeWidth - 1);
  const uint32_t y = local >> 7;
  uint32_t neighbours = 0;

#pragma unroll
  for (int dy = -1; dy <= 1; ++dy) {
#pragma unroll
    for (int dx = -1; dx <= 1; ++dx) {
      if (dx == 0 && dy == 0) continue;
      const uint32_t nx = (x + dx) & (kLifeWidth - 1);
      const uint32_t ny = (y + dy) & (kLifeHeight - 1);
      neighbours += current[vm_base + ny * kLifeWidth + nx];
    }
  }
  const bool alive = current[index] != 0;
  next[index] = static_cast<uint8_t>(neighbours == 3 ||
                                     (alive && neighbours == 2));
}

__global__ void NativeLifeRender(const uint8_t* cells, uint32_t* framebuffer,
                                 uint32_t total_cells) {
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= total_cells) return;
  framebuffer[index] = cells[index] ? 0xFFFFFFFFu : 0xFF000000u;
}

bool WaitForWarmup(PersistentRuntime& rt, uint32_t n_vms,
                   uint32_t frames, int timeout_ms) {
  for (int waited = 0; waited < timeout_ms; ++waited) {
    bool ready = true;
    for (uint32_t vm = 0; vm < n_vms; ++vm) {
      if (rt.FrameSeq(vm) < frames || rt.Status(vm) != kRunning) {
        ready = false;
        break;
      }
    }
    if (ready) return true;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  return false;
}

WarpMeasurement MeasureWarpVm(const WvmFile& file, uint32_t n_vms,
                              int duration_ms, std::string& err,
                              PersistentKernelMode mode =
                                  PersistentKernelMode::kNormal) {
  WarpMeasurement result;
  std::vector<VmImage> images(n_vms);
  for (VmImage& image : images) {
    image.code = file.code;
    image.literals = file.literals;
    image.mem_size_words = 16384;
  }

  PersistentRuntime rt;
  if (!rt.Init(images, err) || !rt.Launch(err, mode)) return result;
  rt.BootAll();
  if (!WaitForWarmup(rt, n_vms, 2, 30000)) {
    err = "not all WarpLife VMs published two warm-up generations";
    rt.ShutdownAll();
    rt.Sync();
    return result;
  }

  std::vector<uint32_t> before(n_vms);
  for (uint32_t vm = 0; vm < n_vms; ++vm) before[vm] = rt.FrameSeq(vm);
  const auto start = std::chrono::steady_clock::now();
  std::this_thread::sleep_for(std::chrono::milliseconds(duration_ms));
  const auto stop = std::chrono::steady_clock::now();

  result.seconds = std::chrono::duration<double>(stop - start).count();
  uint64_t total_generations = 0;
  uint32_t min_generations = std::numeric_limits<uint32_t>::max();
  uint32_t max_generations = 0;
  bool running = true;
  for (uint32_t vm = 0; vm < n_vms; ++vm) {
    const uint32_t generations = rt.FrameSeq(vm) - before[vm];
    total_generations += generations;
    min_generations = std::min(min_generations, generations);
    max_generations = std::max(max_generations, generations);
    running &= rt.Status(vm) == kRunning;
  }

  rt.ShutdownAll();
  const cudaError_t shutdown = rt.Sync();
  if (!running || shutdown != cudaSuccess || total_generations == 0) {
    err = !running ? "a WarpLife VM stopped during measurement"
                   : (shutdown != cudaSuccess
                          ? std::string("persistent kernel shutdown failed: ") +
                                cudaGetErrorString(shutdown)
                          : "no WarpLife generations retired during measurement");
    return result;
  }

  result.avg_generations_per_second =
      static_cast<double>(total_generations) / n_vms / result.seconds;
  result.min_generations_per_second = min_generations / result.seconds;
  result.max_generations_per_second = max_generations / result.seconds;
  result.aggregate_cells_per_second =
      static_cast<double>(total_generations) * kLifeCells / result.seconds;
  result.ok = true;
  return result;
}

bool WarmCpuWarpVms(std::vector<CpuVm>& vms, uint32_t frames,
                    std::string& err) {
  for (CpuVm& vm : vms) {
    uint64_t quanta = 0;
    while (vm.frame_seq < frames && vm.status == kRunning &&
           quanta < 1000000) {
      vm.RunQuantum(kCpuVmQuantum);
      ++quanta;
    }
    if (vm.frame_seq < frames || vm.status != kRunning) {
      err = "CPU WarpVM failed during warm-up (vm " +
            std::to_string(vm.vm_id) + ", status " +
            std::to_string(vm.status) + ", fault " +
            std::to_string(vm.fault) + ")";
      return false;
    }
  }
  return true;
}

WarpMeasurement MeasureCpuWarpVm(const WvmFile& file, uint32_t n_vms,
                                 uint32_t requested_workers, int duration_ms,
                                 std::string& err) {
  WarpMeasurement result;
  std::vector<CpuVm> vms(n_vms);
  for (uint32_t vm = 0; vm < n_vms; ++vm)
    vms[vm].Init(vm, file, 16384);
  if (!WarmCpuWarpVms(vms, 2, err)) return result;

  const uint32_t workers = std::max(1u, std::min(requested_workers, n_vms));
  std::vector<uint32_t> before(n_vms);
  for (uint32_t vm = 0; vm < n_vms; ++vm) before[vm] = vms[vm].frame_seq;

  std::atomic<uint32_t> ready{0};
  std::atomic<bool> start{false};
  std::atomic<bool> stop{false};
  std::vector<std::thread> threads;
  threads.reserve(workers);
  for (uint32_t worker = 0; worker < workers; ++worker) {
    threads.emplace_back([&, worker] {
      ready.fetch_add(1, std::memory_order_release);
      while (!start.load(std::memory_order_acquire))
        std::this_thread::yield();
      while (!stop.load(std::memory_order_acquire)) {
        // Fixed static ownership and one fixed quantum per VM makes the
        // scheduler deterministic apart from the wall-clock stopping point.
        for (uint32_t vm = worker; vm < n_vms; vm += workers) {
          vms[vm].RunQuantum(kCpuVmQuantum);
          if (stop.load(std::memory_order_relaxed)) break;
        }
      }
    });
  }
  while (ready.load(std::memory_order_acquire) != workers)
    std::this_thread::yield();
  const auto begin = std::chrono::steady_clock::now();
  start.store(true, std::memory_order_release);
  std::this_thread::sleep_for(std::chrono::milliseconds(duration_ms));
  stop.store(true, std::memory_order_release);
  for (std::thread& thread : threads) thread.join();
  const auto end = std::chrono::steady_clock::now();

  result.seconds = std::chrono::duration<double>(end - begin).count();
  uint64_t total_generations = 0;
  uint32_t min_generations = std::numeric_limits<uint32_t>::max();
  uint32_t max_generations = 0;
  bool running = true;
  for (uint32_t vm = 0; vm < n_vms; ++vm) {
    const uint32_t generations = vms[vm].frame_seq - before[vm];
    total_generations += generations;
    min_generations = std::min(min_generations, generations);
    max_generations = std::max(max_generations, generations);
    running &= vms[vm].status == kRunning;
  }
  if (!running || total_generations == 0) {
    err = running ? "no CPU WarpVM generations retired during measurement"
                  : "a CPU WarpVM stopped during measurement";
    return result;
  }
  result.avg_generations_per_second =
      static_cast<double>(total_generations) / n_vms / result.seconds;
  result.min_generations_per_second = min_generations / result.seconds;
  result.max_generations_per_second = max_generations / result.seconds;
  result.aggregate_cells_per_second =
      static_cast<double>(total_generations) * kLifeCells / result.seconds;
  result.ok = true;
  return result;
}

bool NativeKnownPatternsMatch(const std::vector<uint8_t>& cells,
                              uint32_t n_vms, uint64_t generations) {
  auto alive = [&](uint32_t vm, uint32_t x, uint32_t y) {
    return cells[static_cast<size_t>(vm) * kLifeCells + y * kLifeWidth + x] !=
           0;
  };

  uint32_t alive0 = 0;
  for (uint32_t y = 0; y < kLifeHeight; ++y)
    for (uint32_t x = 0; x < kLifeWidth; ++x)
      alive0 += alive(0, x, y);
  if (alive0 != 3) return false;
  if ((generations & 1u) == 0) {
    if (!alive(0, 63, 64) || !alive(0, 64, 64) || !alive(0, 65, 64))
      return false;
  } else {
    if (!alive(0, 64, 63) || !alive(0, 64, 64) || !alive(0, 64, 65))
      return false;
  }
  if (n_vms == 1) return true;

  uint32_t alive1 = 0;
  for (uint32_t y = 0; y < kLifeHeight; ++y)
    for (uint32_t x = 0; x < kLifeWidth; ++x)
      alive1 += alive(1, x, y);
  return alive1 == 4 && alive(1, 0, 0) && alive(1, 127, 0) &&
         alive(1, 0, 127) && alive(1, 127, 127);
}

struct NativeCpuState {
  std::vector<uint8_t> a;
  std::vector<uint8_t> b;
  std::vector<uint32_t> framebuffer;
  std::vector<uint8_t> current_is_b;
  std::vector<uint64_t> generations;
};

NativeCpuState MakeNativeCpuState(uint32_t n_vms) {
  NativeCpuState state;
  state.a = MakeInitialCells(n_vms);
  state.b.resize(static_cast<size_t>(n_vms) * kLifeCells);
  state.framebuffer.resize(static_cast<size_t>(n_vms) * kLifeCells);
  state.current_is_b.resize(n_vms, 0);
  state.generations.resize(n_vms, 0);
  return state;
}

void NativeCpuStepVm(NativeCpuState& state, uint32_t vm) {
  const size_t base = static_cast<size_t>(vm) * kLifeCells;
  const uint8_t* current =
      (state.current_is_b[vm] ? state.b.data() : state.a.data()) + base;
  uint8_t* next =
      (state.current_is_b[vm] ? state.a.data() : state.b.data()) + base;
  uint32_t* framebuffer = state.framebuffer.data() + base;
  for (uint32_t y = 0; y < kLifeHeight; ++y) {
    const uint32_t above = ((y - 1) & (kLifeHeight - 1)) * kLifeWidth;
    const uint32_t row = y * kLifeWidth;
    const uint32_t below = ((y + 1) & (kLifeHeight - 1)) * kLifeWidth;
    for (uint32_t x = 0; x < kLifeWidth; ++x) {
      const uint32_t left = (x - 1) & (kLifeWidth - 1);
      const uint32_t right = (x + 1) & (kLifeWidth - 1);
      const uint32_t neighbours =
          current[above + left] + current[above + x] +
          current[above + right] + current[row + left] +
          current[row + right] + current[below + left] +
          current[below + x] + current[below + right];
      const bool alive = current[row + x] != 0;
      const uint8_t value =
          static_cast<uint8_t>(neighbours == 3 || (alive && neighbours == 2));
      next[row + x] = value;
      framebuffer[row + x] = value ? 0xFFFFFFFFu : 0xFF000000u;
    }
  }
  state.current_is_b[vm] ^= 1;
  ++state.generations[vm];
}

bool NativeCpuKnownPatternsMatch(const NativeCpuState& state,
                                 uint32_t n_vms) {
  auto current = [&](uint32_t vm) {
    const size_t base = static_cast<size_t>(vm) * kLifeCells;
    return (state.current_is_b[vm] ? state.b.data() : state.a.data()) + base;
  };
  const uint8_t* vm0 = current(0);
  uint32_t alive0 = 0;
  for (uint32_t cell = 0; cell < kLifeCells; ++cell) alive0 += vm0[cell];
  if (alive0 != 3) return false;
  if ((state.generations[0] & 1u) == 0) {
    if (!vm0[64 * kLifeWidth + 63] || !vm0[64 * kLifeWidth + 64] ||
        !vm0[64 * kLifeWidth + 65])
      return false;
  } else if (!vm0[63 * kLifeWidth + 64] ||
             !vm0[64 * kLifeWidth + 64] ||
             !vm0[65 * kLifeWidth + 64]) {
    return false;
  }
  if (n_vms == 1) return true;
  const uint8_t* vm1 = current(1);
  uint32_t alive1 = 0;
  for (uint32_t cell = 0; cell < kLifeCells; ++cell) alive1 += vm1[cell];
  return alive1 == 4 && vm1[0] && vm1[kLifeWidth - 1] &&
         vm1[(kLifeHeight - 1) * kLifeWidth] && vm1[kLifeCells - 1];
}

WarpMeasurement MeasureNativeCpu(uint32_t n_vms, uint32_t requested_workers,
                                 int duration_ms, std::string& err) {
  WarpMeasurement result;
  NativeCpuState state = MakeNativeCpuState(n_vms);
  for (uint32_t vm = 0; vm < n_vms; ++vm) {
    NativeCpuStepVm(state, vm);
    NativeCpuStepVm(state, vm);
  }

  const uint32_t workers = std::max(1u, std::min(requested_workers, n_vms));
  const std::vector<uint64_t> before = state.generations;
  std::atomic<uint32_t> ready{0};
  std::atomic<bool> start{false};
  std::atomic<bool> stop{false};
  std::vector<std::thread> threads;
  threads.reserve(workers);
  for (uint32_t worker = 0; worker < workers; ++worker) {
    threads.emplace_back([&, worker] {
      ready.fetch_add(1, std::memory_order_release);
      while (!start.load(std::memory_order_acquire))
        std::this_thread::yield();
      while (!stop.load(std::memory_order_acquire)) {
        for (uint32_t vm = worker; vm < n_vms; vm += workers) {
          NativeCpuStepVm(state, vm);
          if (stop.load(std::memory_order_relaxed)) break;
        }
      }
    });
  }
  while (ready.load(std::memory_order_acquire) != workers)
    std::this_thread::yield();
  const auto begin = std::chrono::steady_clock::now();
  start.store(true, std::memory_order_release);
  std::this_thread::sleep_for(std::chrono::milliseconds(duration_ms));
  stop.store(true, std::memory_order_release);
  for (std::thread& thread : threads) thread.join();
  const auto end = std::chrono::steady_clock::now();

  result.seconds = std::chrono::duration<double>(end - begin).count();
  uint64_t total_generations = 0;
  uint64_t min_generations = std::numeric_limits<uint64_t>::max();
  uint64_t max_generations = 0;
  for (uint32_t vm = 0; vm < n_vms; ++vm) {
    const uint64_t generations = state.generations[vm] - before[vm];
    total_generations += generations;
    min_generations = std::min(min_generations, generations);
    max_generations = std::max(max_generations, generations);
  }
  if (total_generations == 0 || !NativeCpuKnownPatternsMatch(state, n_vms)) {
    err = total_generations == 0
              ? "no native CPU generations retired during measurement"
              : "native CPU known-pattern validation failed";
    return result;
  }
  result.avg_generations_per_second =
      static_cast<double>(total_generations) / n_vms / result.seconds;
  result.min_generations_per_second = min_generations / result.seconds;
  result.max_generations_per_second = max_generations / result.seconds;
  result.aggregate_cells_per_second =
      static_cast<double>(total_generations) * kLifeCells / result.seconds;
  result.ok = true;
  return result;
}

bool CompiledKnownPatternsMatch(const std::vector<VmState>& states,
                                const std::vector<uint32_t>& memory,
                                uint32_t memory_words) {
  auto alive = [&](uint32_t vm, uint32_t x, uint32_t y) {
    const uint32_t base = states[vm].sregs[0];
    const uint32_t word = y * 4 + (x >> 5);
    const uint32_t packed =
        memory[static_cast<size_t>(vm) * memory_words + base + word];
    return ((packed >> (x & 31u)) & 1u) != 0;
  };
  uint32_t alive0 = 0;
  for (uint32_t y = 0; y < kLifeHeight; ++y)
    for (uint32_t x = 0; x < kLifeWidth; ++x)
      alive0 += alive(0, x, y);
  if (alive0 != 3) return false;
  // frame_seq is not part of VmState, but the current blinker orientation is
  // enough to accept either parity while still rejecting every wrong shape.
  const bool horizontal = alive(0, 63, 64) && alive(0, 64, 64) &&
                          alive(0, 65, 64);
  const bool vertical = alive(0, 64, 63) && alive(0, 64, 64) &&
                        alive(0, 64, 65);
  if (!horizontal && !vertical) return false;
  if (states.size() == 1) return true;
  uint32_t alive1 = 0;
  for (uint32_t y = 0; y < kLifeHeight; ++y)
    for (uint32_t x = 0; x < kLifeWidth; ++x)
      alive1 += alive(1, x, y);
  return alive1 == 4 && alive(1, 0, 0) && alive(1, 127, 0) &&
         alive(1, 0, 127) && alive(1, 127, 127);
}

WarpMeasurement MeasureCompiledWarpVm(PtxCompiledProgram& compiled,
                                       uint32_t n_vms, int duration_ms,
                                       std::string& err) {
  constexpr uint32_t kMemoryWords = 16384;
  WarpMeasurement result;
  std::vector<VmState> states(n_vms);
  std::vector<uint32_t> memory(static_cast<size_t>(n_vms) * kMemoryWords, 0);
  std::vector<uint32_t> framebuffers(
      static_cast<size_t>(n_vms) * kVideoWords, kVideoResetColor);
  std::vector<uint32_t> frame_seq(n_vms, 0);
  for (uint32_t vm = 0; vm < n_vms; ++vm) {
    states[vm].vm_id = vm;
    states[vm].status = kRunning;
    states[vm].rng_state = vm * 0x9E3779B9u + 0x1234567u;
  }
  if (!compiled.Launch(states, memory, kMemoryWords, framebuffers, frame_seq,
                       err))
    return result;

  constexpr uint32_t kCalibrationCheckpoints = 20;
  double calibration_ms = 0.0;
  if (!compiled.LaunchCheckpoints(
          states, memory, kMemoryWords, framebuffers, frame_seq,
          kCalibrationCheckpoints, calibration_ms, err) ||
      calibration_ms <= 0.0)
    return result;
  const double estimated = kCalibrationCheckpoints * duration_ms /
                           calibration_ms;
  const uint32_t checkpoints = static_cast<uint32_t>(
      std::clamp(estimated, 50.0, 10000.0));
  const std::vector<uint32_t> before = frame_seq;
  double elapsed_ms = 0.0;
  if (!compiled.LaunchCheckpoints(states, memory, kMemoryWords, framebuffers,
                                  frame_seq, checkpoints, elapsed_ms, err))
    return result;

  bool frames_ok = true;
  for (uint32_t vm = 0; vm < n_vms; ++vm)
    frames_ok &= frame_seq[vm] - before[vm] == checkpoints;
  if (!frames_ok || !CompiledKnownPatternsMatch(states, memory, kMemoryWords)) {
    err = !frames_ok ? "compiled frame count mismatch"
                     : "compiled known-pattern validation failed";
    return result;
  }
  result.seconds = elapsed_ms / 1000.0;
  result.avg_generations_per_second = checkpoints / result.seconds;
  result.min_generations_per_second = result.avg_generations_per_second;
  result.max_generations_per_second = result.avg_generations_per_second;
  result.aggregate_cells_per_second =
      static_cast<double>(checkpoints) * n_vms * kLifeCells / result.seconds;
  result.ok = true;
  return result;
}

NativeMeasurement MeasureNative(uint32_t n_vms, int duration_ms,
                                std::string& err) {
  NativeMeasurement result;
  const uint32_t total_cells = n_vms * kLifeCells;
  std::vector<uint8_t> initial = MakeInitialCells(n_vms);
  uint8_t* d_a = nullptr;
  uint8_t* d_b = nullptr;
  uint32_t* d_framebuffer = nullptr;
  cudaEvent_t start_event = nullptr;
  cudaEvent_t stop_event = nullptr;

  auto fail = [&](const char* operation, cudaError_t status) {
    err = std::string(operation) + ": " + cudaGetErrorString(status);
    return false;
  };
  if (cudaMalloc(&d_a, total_cells * sizeof(uint8_t)) != cudaSuccess ||
      cudaMalloc(&d_b, total_cells * sizeof(uint8_t)) != cudaSuccess ||
      cudaMalloc(&d_framebuffer, total_cells * sizeof(uint32_t)) !=
          cudaSuccess) {
    err = "native CUDA allocation failed";
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_framebuffer);
    return result;
  }
  if (cudaMemcpy(d_a, initial.data(), total_cells * sizeof(uint8_t),
                 cudaMemcpyHostToDevice) != cudaSuccess) {
    err = "native initial-state copy failed";
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_framebuffer);
    return result;
  }
  if (cudaEventCreate(&start_event) != cudaSuccess ||
      cudaEventCreate(&stop_event) != cudaSuccess) {
    err = "native CUDA event creation failed";
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_framebuffer);
    if (start_event) cudaEventDestroy(start_event);
    if (stop_event) cudaEventDestroy(stop_event);
    return result;
  }

  const int threads = 256;
  const int blocks = static_cast<int>((total_cells + threads - 1) / threads);
  uint8_t* current = d_a;
  uint8_t* next = d_b;
  uint64_t total_executed = 0;

  auto launch_generations = [&](uint32_t generations) {
    for (uint32_t generation = 0; generation < generations; ++generation) {
      NativeLifeStep<<<blocks, threads>>>(current, next, total_cells);
      NativeLifeRender<<<blocks, threads>>>(next, d_framebuffer, total_cells);
      std::swap(current, next);
    }
    total_executed += generations;
    return cudaGetLastError();
  };

  // Calibrate with enough work to make event timing stable, then choose a
  // generation count targeting the same wall-clock duration as WarpVM.
  constexpr uint32_t kCalibrationGenerations = 100;
  cudaEventRecord(start_event);
  cudaError_t status = launch_generations(kCalibrationGenerations);
  cudaEventRecord(stop_event);
  cudaEventSynchronize(stop_event);
  float calibration_ms = 0.0f;
  cudaEventElapsedTime(&calibration_ms, start_event, stop_event);
  if (status != cudaSuccess || calibration_ms <= 0.0f) {
    fail("native calibration", status);
  } else {
    const double estimated =
        kCalibrationGenerations * static_cast<double>(duration_ms) /
        calibration_ms;
    result.generations = static_cast<uint32_t>(
        std::clamp(estimated, 100.0, 1000000.0));

    cudaEventRecord(start_event);
    status = launch_generations(result.generations);
    cudaEventRecord(stop_event);
    cudaEventSynchronize(stop_event);
    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, start_event, stop_event);
    if (status == cudaSuccess && elapsed_ms > 0.0f) {
      result.seconds = elapsed_ms / 1000.0;
      result.generations_per_second = result.generations / result.seconds;
      result.aggregate_cells_per_second =
          static_cast<double>(result.generations) * total_cells /
          result.seconds;

      std::vector<uint8_t> final_cells(total_cells);
      status = cudaMemcpy(final_cells.data(), current,
                          total_cells * sizeof(uint8_t),
                          cudaMemcpyDeviceToHost);
      result.ok = status == cudaSuccess &&
                  NativeKnownPatternsMatch(final_cells, n_vms, total_executed);
      if (!result.ok)
        err = status == cudaSuccess ? "native known-pattern validation failed"
                                    : "native result copy failed";
    } else {
      fail("native timed workload", status);
    }
  }

  cudaEventDestroy(start_event);
  cudaEventDestroy(stop_event);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_framebuffer);
  return result;
}

uint32_t ResidentVmSlots(std::string& err) {
  int device = 0;
  cudaDeviceProp properties{};
  int blocks_per_sm = 0;
  cudaError_t status = cudaGetDevice(&device);
  if (status == cudaSuccess)
    status = cudaGetDeviceProperties(&properties, device);
  if (status == cudaSuccess) {
    status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm, PersistentKernel, kPersistentBlockThreads, 0);
  }
  if (status != cudaSuccess || blocks_per_sm < 1) {
    err = std::string("persistent-kernel occupancy query failed: ") +
          cudaGetErrorString(status);
    return 0;
  }
  const uint32_t vms_per_block = kPersistentBlockThreads / kLanes;
  const uint64_t capacity = static_cast<uint64_t>(blocks_per_sm) *
                            properties.multiProcessorCount * vms_per_block;
  return static_cast<uint32_t>(capacity);
}

struct PersistentKernelStats {
  int registers_per_thread = 0;
  size_t local_bytes_per_thread = 0;
  size_t shared_bytes_per_block = 0;
  int blocks_per_sm = 0;
  uint32_t resident_vm_slots = 0;
};

template <typename Kernel>
cudaError_t QueryPersistentKernel(Kernel kernel,
                                  PersistentKernelStats& stats) {
  cudaFuncAttributes attributes{};
  cudaError_t status = cudaFuncGetAttributes(&attributes, kernel);
  if (status == cudaSuccess)
    status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &stats.blocks_per_sm, kernel, kPersistentBlockThreads, 0);
  if (status != cudaSuccess) return status;
  stats.registers_per_thread = attributes.numRegs;
  stats.local_bytes_per_thread = attributes.localSizeBytes;
  stats.shared_bytes_per_block = attributes.sharedSizeBytes;
  int device = 0;
  cudaDeviceProp properties{};
  status = cudaGetDevice(&device);
  if (status == cudaSuccess)
    status = cudaGetDeviceProperties(&properties, device);
  if (status == cudaSuccess) {
    stats.resident_vm_slots = static_cast<uint32_t>(
        stats.blocks_per_sm * properties.multiProcessorCount *
        (kPersistentBlockThreads / kLanes));
  }
  return status;
}

bool PersistentStatsForMode(PersistentKernelMode mode,
                            PersistentKernelStats& stats,
                            std::string& err) {
  cudaError_t status = cudaErrorInvalidValue;
  switch (mode) {
    case PersistentKernelMode::kNormal:
      status = QueryPersistentKernel(PersistentKernel, stats);
      break;
    case PersistentKernelMode::kScalarRegs:
      status = QueryPersistentKernel(PersistentScalarRegsKernel, stats);
      break;
    case PersistentKernelMode::kDenseDispatch:
      status = QueryPersistentKernel(PersistentDenseDispatchKernel, stats);
      break;
    case PersistentKernelMode::kHotDispatch:
      status = QueryPersistentKernel(PersistentHotDispatchKernel, stats);
      break;
    case PersistentKernelMode::kHot4Dispatch:
      status = QueryPersistentKernel(PersistentHot4DispatchKernel, stats);
      break;
    case PersistentKernelMode::kScalarRegsDenseDispatch:
      status = QueryPersistentKernel(PersistentScalarDenseKernel, stats);
      break;
    case PersistentKernelMode::kSharedRegs:
      status = QueryPersistentKernel(PersistentSharedRegsKernel, stats);
      break;
    case PersistentKernelMode::kSharedRegsThreeBlock:
      status = QueryPersistentKernel(PersistentSharedRegsThreeBlockKernel,
                                     stats);
      break;
    case PersistentKernelMode::kSharedRegsDenseThreeBlock:
      status = QueryPersistentKernel(PersistentSharedDenseThreeBlockKernel,
                                     stats);
      break;
    default:
      break;
  }
  if (status != cudaSuccess) {
    err = std::string("kernel resource query failed: ") +
          cudaGetErrorString(status);
    return false;
  }
  return true;
}

const char* OpcodeName(uint32_t op) {
  switch (op) {
    case kNop: return "NOP";
    case kMov: return "MOV";
    case kAdd: return "ADD";
    case kMul: return "MUL";
    case kAnd: return "AND";
    case kXor: return "XOR";
    case kShl: return "SHL";
    case kShr: return "SHR";
    case kMovI: return "MOV_I";
    case kAddI: return "ADD_I";
    case kMulI: return "MUL_I";
    case kAndI: return "AND_I";
    case kShlI: return "SHL_I";
    case kShrI: return "SHR_I";
    case kLdw: return "LDW";
    case kSBcast: return "S_BCAST";
    case kCmpEq: return "CMP_EQ";
    case kCmpNe: return "CMP_NE";
    case kCmpLt: return "CMP_LT";
    case kCmpEqI: return "CMP_EQ_I";
    case kCmpNeI: return "CMP_NE_I";
    case kCmpLtI: return "CMP_LT_I";
    case kAndMask: return "ANDMASK";
    case kOrMask: return "ORMASK";
    case kBallot: return "BALLOT";
    case kLaneId: return "LANEID";
    case kReduceOr: return "REDUCE_OR";
    case kVmid: return "VMID";
    case kLoad: return "LOAD";
    case kStore: return "STORE";
    case kFlip: return "FLIP";
    case kSMov: return "S_MOV";
    case kSMovI: return "S_MOV_I";
    case kSAddI: return "S_ADD_I";
    case kSCmpLtI: return "S_CMP_LT_I";
    case kJmp: return "JMP";
    case kJmpIfAny: return "JMP_IF_ANY";
    case kCall: return "CALL";
    case kRet: return "RET";
    case kYield: return "YIELD";
    default: return "OTHER";
  }
}

uint32_t CpuActiveMask(const CpuVm& vm, uint32_t guard) {
  if (guard == 0) return kFullMask;
  const uint32_t mask = vm.preds[(guard - 1) & 3u];
  return guard >= 5 ? ~mask : mask;
}

struct LifeCensus {
  std::array<uint64_t, 128> opcodes{};
  std::array<uint64_t, 3> phases{};  // evolve, render, publication/control
  uint64_t ram_load_lanes = 0;
  uint64_t ram_load_unique_addresses = 0;
  uint64_t ram_store_lanes = 0;
  uint64_t framebuffer_store_lanes = 0;
  uint64_t control_polls = 0;
};

uint64_t CountUniqueActiveAddresses(const CpuVm& vm, uint32_t reg,
                                    uint32_t active, bool want_ram) {
  std::array<uint32_t, kLanes> seen{};
  uint32_t count = 0;
  for (uint32_t lane = 0; lane < kLanes; ++lane) {
    if (((active >> lane) & 1u) == 0) continue;
    const uint32_t address = vm.vregs[reg][lane];
    if ((address < vm.memory.size()) != want_ram) continue;
    bool duplicate = false;
    for (uint32_t index = 0; index < count; ++index)
      duplicate |= seen[index] == address;
    if (!duplicate) seen[count++] = address;
  }
  return count;
}

bool CollectLifeCensus(const WvmFile& file, LifeCensus& census,
                       std::string& err, uint32_t evolve_pc = 51,
                       uint32_t render_pc = 113,
                       uint32_t publish_pc = 137) {
  if (evolve_pc >= file.code.size() || render_pc >= file.code.size() ||
      publish_pc >= file.code.size()) {
    err = "invalid WarpLife phase boundaries for census";
    return false;
  }
  CpuVm cpu;
  cpu.Init(0, file, 16384);
  while (cpu.status == kRunning && cpu.frame_seq < 1) cpu.Step();
  if (cpu.status != kRunning) {
    err = "CPU WarpLife stopped before the census generation";
    return false;
  }

  uint32_t phase = 2;
  while (cpu.status == kRunning && cpu.frame_seq < 2) {
    const uint32_t pc = cpu.pc;
    if (pc == evolve_pc) phase = 0;
    else if (pc == render_pc) phase = 1;
    else if (pc == publish_pc) phase = 2;
    const uint32_t instruction = cpu.code[pc];
    const uint32_t op = (instruction >> kOpcodeShift) & kOpcodeMask;
    const uint32_t guard = (instruction >> kGuardShift) & kGuardMask;
    const uint32_t rd = (instruction >> kRdShift) & kRegFieldMask;
    const uint32_t rs1 = (instruction >> kRs1Shift) & kRegFieldMask;
    const uint32_t active = CpuActiveMask(cpu, guard);
    ++census.opcodes[op];
    ++census.phases[phase];

    if (op == kLoad) {
      for (uint32_t lane = 0; lane < kLanes; ++lane)
        if (((active >> lane) & 1u) &&
            cpu.vregs[rs1][lane] < cpu.memory.size())
          ++census.ram_load_lanes;
      census.ram_load_unique_addresses +=
          CountUniqueActiveAddresses(cpu, rs1, active, true);
    } else if (op == kStore) {
      for (uint32_t lane = 0; lane < kLanes; ++lane) {
        if (((active >> lane) & 1u) == 0) continue;
        const uint32_t address = cpu.vregs[rd][lane];
        if (address < cpu.memory.size()) ++census.ram_store_lanes;
        else if (address >= kVideoBaseWord && address < kVideoEndWord)
          ++census.framebuffer_store_lanes;
      }
    }

    cpu.Step();
    const bool backward =
        (op == kJmp || op == kJmpIfAny || op == kJmpIfAll) &&
        cpu.pc <= pc;
    if (backward || op == kYield) ++census.control_polls;
  }
  if (cpu.status != kRunning || cpu.frame_seq != 2) {
    err = "CPU WarpLife stopped during the census generation";
    return false;
  }
  return true;
}

struct InlinedLife {
  WvmFile file;
  uint32_t evolve_pc = 0;
  uint32_t render_pc = 0;
  uint32_t publish_pc = 0;
};

bool InlineLifeLoadCell(const WvmFile& source, InlinedLife& result,
                        std::string& err) {
  constexpr uint32_t kOriginalWords = 153;
  constexpr uint32_t kSubroutinePc = 143;
  constexpr uint32_t kSubroutineBodyWords = 9;  // excludes RET at 152
  if (source.code.size() != kOriginalWords ||
      ((source.code[kSubroutinePc] >> kOpcodeShift) & kOpcodeMask) != kAndI ||
      ((source.code[152] >> kOpcodeShift) & kOpcodeMask) != kRet) {
    err = "WarpLife load_cell layout does not match the inliner";
    return false;
  }

  std::array<uint32_t, kSubroutinePc + 1> relocated{};
  uint32_t output_pc = 0;
  uint32_t call_sites = 0;
  for (uint32_t pc = 0; pc < kSubroutinePc; ++pc) {
    relocated[pc] = output_pc;
    const uint32_t word = source.code[pc];
    const uint32_t op = (word >> kOpcodeShift) & kOpcodeMask;
    const uint32_t target = word & kLoMask;
    const bool inline_call = op == kCall && target == kSubroutinePc;
    output_pc += inline_call ? kSubroutineBodyWords : 1;
    call_sites += inline_call;
  }
  relocated[kSubroutinePc] = output_pc;
  if (call_sites != 9) {
    err = "WarpLife inliner expected nine load_cell call sites";
    return false;
  }

  result.file.literals = source.literals;
  result.file.code.reserve(output_pc);
  for (uint32_t pc = 0; pc < kSubroutinePc; ++pc) {
    uint32_t word = source.code[pc];
    const uint32_t op = (word >> kOpcodeShift) & kOpcodeMask;
    const uint32_t target = word & kLoMask;
    if (op == kCall && target == kSubroutinePc) {
      result.file.code.insert(result.file.code.end(),
                              source.code.begin() + kSubroutinePc,
                              source.code.begin() + kSubroutinePc +
                                  kSubroutineBodyWords);
      continue;
    }
    if (op == kJmp || op == kJmpIfAny || op == kJmpIfAll || op == kCall) {
      if (target >= kSubroutinePc) {
        err = "WarpLife inliner found an unsupported control target";
        return false;
      }
      word = (word & ~kLoMask) | (relocated[target] & kLoMask);
    }
    result.file.code.push_back(word);
  }
  result.evolve_pc = relocated[51];
  result.render_pc = relocated[113];
  result.publish_pc = relocated[137];
  if (result.file.code.size() != 215) {
    err = "WarpLife inliner produced an unexpected code size";
    return false;
  }
  return true;
}

bool CpuLifeProgramsMatch(const WvmFile& a, const WvmFile& b,
                          std::string& err) {
  constexpr std::array<uint32_t, 3> kVmIds{0, 2, 37};
  for (const uint32_t vm_id : kVmIds) {
    CpuVm left;
    CpuVm right;
    left.Init(vm_id, a, 16384);
    right.Init(vm_id, b, 16384);
    while (left.status == kRunning && left.frame_seq < 3) left.Step();
    while (right.status == kRunning && right.frame_seq < 3) right.Step();
    const uint32_t left_base = left.sregs[0];
    const uint32_t right_base = right.sregs[0];
    if (left.status != kRunning || right.status != kRunning ||
        left.frame_seq != 3 || right.frame_seq != 3 ||
        left_base + kLifeWords > left.memory.size() ||
        right_base + kLifeWords > right.memory.size() ||
        !std::equal(left.memory.begin() + left_base,
                    left.memory.begin() + left_base + kLifeWords,
                    right.memory.begin() + right_base)) {
      err = "inlined load_cell changed CPU world state for vm " +
            std::to_string(vm_id);
      return false;
    }
  }
  return true;
}

enum class MicroBody {
  kNop,
  kStepTrap,
  kAdd,
  kRamLoad,
  kRamStore,
  kReduceOr,
  kFbStore
};

WvmFile MakeMicroProgram(MicroBody body, uint32_t body_ops) {
  WvmFile file;
  file.literals = {kVideoBaseWord};
  // One-time setup. r0 is a two-address RAM pattern, r2 is the contiguous
  // framebuffer address, and r1 is an arbitrary value/source.
  file.code.push_back(enc_r(kLaneId, 0, 0, 0, 0));
  file.code.push_back(enc_i(kAndI, 0, 0, 0, 1));
  file.code.push_back(enc_i(kLdw, 0, 2, 0, 0));
  file.code.push_back(enc_r(kLaneId, 0, 3, 0, 0));
  file.code.push_back(enc_r(kAdd, 0, 2, 2, 3));
  file.code.push_back(enc_i(kMovI, 0, 1, 0, 1));
  const uint32_t frame_pc = static_cast<uint32_t>(file.code.size());
  file.code.push_back(enc_i(kSMovI, 0, 0, 0, 0));
  const uint32_t loop_pc = static_cast<uint32_t>(file.code.size());
  for (uint32_t index = 0; index < body_ops; ++index) {
    switch (body) {
      case MicroBody::kNop:
        file.code.push_back(enc_r(kNop, 0, 0, 0, 0));
        break;
      case MicroBody::kStepTrap:
        file.code.push_back(enc_r(kStepTrap, 0, 0, 0, 0));
        break;
      case MicroBody::kAdd:
        file.code.push_back(enc_r(kAdd, 0, 1, 1, 0));
        break;
      case MicroBody::kRamLoad:
        file.code.push_back(enc_r(kLoad, 0, 1, 0, 0));
        break;
      case MicroBody::kRamStore:
        file.code.push_back(enc_r(kStore, 0, 0, 1, 0));
        break;
      case MicroBody::kReduceOr:
        file.code.push_back(enc_r(kReduceOr, 0, 1, 1, 0));
        break;
      case MicroBody::kFbStore:
        file.code.push_back(enc_r(kStore, 0, 2, 1, 0));
        break;
    }
  }
  file.code.push_back(enc_i(kSAddI, 0, 0, 0, 1));
  file.code.push_back(enc_i(kSCmpLtI, 0, 0, 0, 1024));
  file.code.push_back(enc_i(kJmpIfAny, 1, 0, 0,
                            static_cast<int32_t>(loop_pc)));
  file.code.push_back(enc_r(kFlip, 0, 0, 0, 0));
  file.code.push_back(enc_r(kYield, 0, 0, 0, 0));
  file.code.push_back(
      enc_i(kJmp, 0, 0, 0, static_cast<int32_t>(frame_pc)));
  return file;
}

double MeasureGpuMillisecondsPerFrame(const WvmFile& file, int duration_ms,
                                      std::string& err,
                                      PersistentKernelMode mode) {
  VmImage image;
  image.code = file.code;
  image.literals = file.literals;
  image.mem_size_words = 16384;
  PersistentRuntime runtime;
  if (!runtime.Init({image}, err) || !runtime.Launch(err, mode)) return 0.0;
  runtime.BootAll();
  if (!WaitForWarmup(runtime, 1, 2, 30000)) {
    err = "profile program did not publish warm-up frames";
    runtime.ShutdownAll();
    runtime.Sync();
    return 0.0;
  }
  auto wait_above = [&](uint32_t sequence, int timeout_ms) {
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::milliseconds(timeout_ms);
    while (runtime.FrameSeq(0) <= sequence &&
           std::chrono::steady_clock::now() < deadline) {
      std::this_thread::yield();
    }
    return runtime.FrameSeq(0) > sequence;
  };

  // Calibrate across exact frame boundaries, then time an integer number of
  // complete frames close to the requested duration. This avoids the coarse
  // integer-frame quantisation of a fixed host sleep at ~25 frames/second.
  uint32_t sequence = runtime.FrameSeq(0);
  if (!wait_above(sequence, 30000)) {
    err = "profile frame-boundary calibration timed out";
    runtime.ShutdownAll();
    runtime.Sync();
    return 0.0;
  }
  sequence = runtime.FrameSeq(0);
  const auto calibration_start = std::chrono::steady_clock::now();
  const uint32_t calibration_frames = 3;
  while (runtime.FrameSeq(0) < sequence + calibration_frames)
    std::this_thread::yield();
  const auto calibration_end = std::chrono::steady_clock::now();
  const double calibration_ms =
      std::chrono::duration<double, std::milli>(calibration_end -
                                                calibration_start)
          .count() /
      calibration_frames;
  const uint32_t target_frames = static_cast<uint32_t>(std::clamp(
      std::ceil(duration_ms / calibration_ms), 5.0, 100000.0));

  sequence = runtime.FrameSeq(0);
  if (!wait_above(sequence, 30000)) {
    err = "profile measurement boundary timed out";
    runtime.ShutdownAll();
    runtime.Sync();
    return 0.0;
  }
  sequence = runtime.FrameSeq(0);
  const auto start = std::chrono::steady_clock::now();
  while (runtime.FrameSeq(0) < sequence + target_frames)
    std::this_thread::yield();
  const auto stop = std::chrono::steady_clock::now();
  runtime.ShutdownAll();
  if (runtime.Sync() != cudaSuccess || runtime.Status(0) != kIdle) {
    err = "profile persistent kernel shutdown failed";
    return 0.0;
  }
  return std::chrono::duration<double, std::milli>(stop - start).count() /
         target_frames;
}

double MedianGpuMillisecondsPerFrame(const WvmFile& file, int duration_ms,
                                     std::string& err,
                                     PersistentKernelMode mode =
                                         PersistentKernelMode::kNormal) {
  std::array<double, 3> samples{};
  for (double& sample : samples) {
    sample = MeasureGpuMillisecondsPerFrame(file, duration_ms, err, mode);
    if (sample == 0.0) return 0.0;
  }
  std::sort(samples.begin(), samples.end());
  return samples[1];
}

double MedianGpuCyclesPerFrame(const WvmFile& file, uint32_t num_vms,
                               std::string& err,
                               PersistentKernelMode mode) {
  VmImage image;
  image.code = file.code;
  image.literals = file.literals;
  image.mem_size_words = 16384;
  std::vector<VmImage> images(num_vms, image);
  PersistentRuntime runtime;
  if (!runtime.Init(images, err) || !runtime.Launch(err, mode)) return 0.0;
  runtime.BootAll();
  if (!WaitForWarmup(runtime, num_vms, 3, 30000)) {
    err = "cycle-profile program did not publish warm-up frames";
    runtime.ShutdownAll();
    runtime.Sync();
    return 0.0;
  }

  const uint32_t rounds = num_vms == 1 ? 31 : 7;
  std::vector<uint64_t> samples;
  samples.reserve(static_cast<size_t>(rounds) * num_vms);
  std::vector<uint32_t> sequences(num_vms);
  for (uint32_t round = 0; round < rounds; ++round) {
    for (uint32_t vm = 0; vm < num_vms; ++vm)
      sequences[vm] = runtime.FrameSeq(vm);
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::seconds(30);
    for (uint32_t vm = 0; vm < num_vms; ++vm) {
      while (runtime.FrameSeq(vm) <= sequences[vm] &&
             std::chrono::steady_clock::now() < deadline)
        std::this_thread::yield();
      if (runtime.FrameSeq(vm) <= sequences[vm]) {
        err = "cycle-profile frame wait timed out";
        runtime.ShutdownAll();
        runtime.Sync();
        return 0.0;
      }
      const uint64_t sample = runtime.ProfileFrameCycles(vm);
      if (sample == 0) {
        err = "cycle-profile kernel did not publish a cycle count";
        runtime.ShutdownAll();
        runtime.Sync();
        return 0.0;
      }
      samples.push_back(sample);
    }
  }
  runtime.ShutdownAll();
  if (runtime.Sync() != cudaSuccess) {
    err = "cycle-profile persistent kernel shutdown failed";
    return 0.0;
  }
  std::sort(samples.begin(), samples.end());
  return static_cast<double>(samples[samples.size() / 2]);
}

}  // namespace

int RunLifeBenchmark(const char* path, const std::vector<uint32_t>& vm_counts,
                     int duration_ms, uint32_t cpu_workers) {
  // Timing output is only meaningful after the same bytecode has passed the
  // deterministic cross-engine semantic gate.
  if (RunCpuGpuLifeEquivalence(path) != 0) {
    std::fprintf(stderr,
                 "error: refusing to benchmark non-equivalent CPU/GPU state\n");
    return 1;
  }
  if (RunNativeCpuLifeEquivalence(path) != 0) {
    std::fprintf(stderr,
                 "error: refusing to benchmark non-equivalent native CPU "
                 "state\n");
    return 1;
  }

  WvmFile file;
  std::string err;
  if (!LoadWvm(path, file, err)) {
    std::fprintf(stderr, "error: %s: %s\n", path, err.c_str());
    return 2;
  }
  PtxCompiledProgram compiled_program;
  if (!compiled_program.Compile(file, err)) {
    std::fprintf(stderr, "error: compiled WarpLife: %s\n", err.c_str());
    return 1;
  }

  std::vector<uint32_t> counts = vm_counts;
  std::string occupancy_error;
  const uint32_t resident_slots = ResidentVmSlots(occupancy_error);
  const uint32_t configured_capacity =
      resident_slots == 0 ? kMaxVms : std::min(resident_slots, kMaxVms);
  // The default suite has several entries. Extend it with the largest count
  // the runtime can configure while retaining an explicit single --vms.
  if (counts.size() > 1 &&
      std::find(counts.begin(), counts.end(), configured_capacity) ==
          counts.end()) {
    counts.push_back(configured_capacity);
  }

  std::printf("WarpLife benchmark\n");
  std::printf("  program: %s\n", path);
  std::printf("  viewer: disabled\n");
  std::printf("  sample: %d ms per implementation/count\n", duration_ms);
  std::printf("  CPU parallel workers requested: %u\n", cpu_workers);
  std::printf("  CPU WarpVM scheduling quantum: %u instructions\n",
              kCpuVmQuantum);
  std::printf("  workload: toroidal Life evolution + 128x128 ARGB render\n\n");
  std::printf("  compiled artifact: %zu PTX bytes, JIT %.3f ms\n\n",
              compiled_program.ptx().size(),
              compiled_program.jit_milliseconds());
  if (resident_slots > 0) {
    std::printf("  CUDA occupancy estimate: %u resident VM-warp slots\n",
                resident_slots);
    std::printf("  WarpVM configured limit: %u VMs\n\n", kMaxVms);
  } else {
    std::printf("  warning: %s\n\n", occupancy_error.c_str());
  }
  std::printf("Ratios below are left throughput / right throughput; >1 means "
              "the left side is faster.\n");

  bool pass = true;
  for (const uint32_t n_vms : counts) {
    err.clear();
    const WarpMeasurement warp = MeasureWarpVm(file, n_vms, duration_ms, err);
    if (!warp.ok) {
      std::fprintf(stderr, "error: %u-VM WarpVM benchmark: %s\n", n_vms,
                   err.c_str());
      pass = false;
      continue;
    }

    err.clear();
    const WarpMeasurement compiled = MeasureCompiledWarpVm(
        compiled_program, n_vms, duration_ms, err);
    if (!compiled.ok) {
      std::fprintf(stderr, "error: %u-VM compiled WarpVM benchmark: %s\n",
                   n_vms, err.c_str());
      pass = false;
      continue;
    }

    err.clear();
    const WarpMeasurement cpu_warp_1 =
        MeasureCpuWarpVm(file, n_vms, 1, duration_ms, err);
    if (!cpu_warp_1.ok) {
      std::fprintf(stderr, "error: %u-VM CPU WarpVM (1 worker): %s\n",
                   n_vms, err.c_str());
      pass = false;
      continue;
    }

    const uint32_t parallel_workers = std::min(cpu_workers, n_vms);
    err.clear();
    const WarpMeasurement cpu_warp_parallel =
        parallel_workers == 1
            ? cpu_warp_1
            : MeasureCpuWarpVm(file, n_vms, parallel_workers, duration_ms,
                               err);
    if (!cpu_warp_parallel.ok) {
      std::fprintf(stderr, "error: %u-VM CPU WarpVM (%u workers): %s\n",
                   n_vms, parallel_workers, err.c_str());
      pass = false;
      continue;
    }

    err.clear();
    const WarpMeasurement native_cpu_1 =
        MeasureNativeCpu(n_vms, 1, duration_ms, err);
    if (!native_cpu_1.ok) {
      std::fprintf(stderr, "error: %u-VM native CPU (1 worker): %s\n",
                   n_vms, err.c_str());
      pass = false;
      continue;
    }

    err.clear();
    const WarpMeasurement native_cpu_parallel =
        parallel_workers == 1
            ? native_cpu_1
            : MeasureNativeCpu(n_vms, parallel_workers, duration_ms, err);
    if (!native_cpu_parallel.ok) {
      std::fprintf(stderr, "error: %u-VM native CPU (%u workers): %s\n",
                   n_vms, parallel_workers, err.c_str());
      pass = false;
      continue;
    }

    err.clear();
    const NativeMeasurement native = MeasureNative(n_vms, duration_ms, err);
    if (!native.ok) {
      std::fprintf(stderr, "error: %u-VM native benchmark: %s\n", n_vms,
                   err.c_str());
      pass = false;
      continue;
    }

    auto print_vm = [&](const char* label, uint32_t workers,
                        const WarpMeasurement& measurement) {
      std::printf("  %-19s %7u  %7.3f  %9.2f (%8.2f..%8.2f)  %11.3f\n",
                  label, workers, measurement.seconds,
                  measurement.avg_generations_per_second,
                  measurement.min_generations_per_second,
                  measurement.max_generations_per_second,
                  measurement.aggregate_cells_per_second / 1.0e6);
    };
    std::printf("\n%u VMs\n", n_vms);
    std::printf("  implementation      workers  seconds   "
                "gen/s/VM (min..max)       Mcell/s\n");
    print_vm("GPU WarpVM", 0, warp);
    print_vm("compiled WarpVM", 0, compiled);
    print_vm("CPU WarpVM", 1, cpu_warp_1);
    if (parallel_workers > 1)
      print_vm("CPU WarpVM", parallel_workers, cpu_warp_parallel);
    print_vm("native CPU", 1, native_cpu_1);
    if (parallel_workers > 1)
      print_vm("native CPU", parallel_workers, native_cpu_parallel);
    std::printf("  %-19s %7s  %7.3f  %9.2f (%8.2f..%8.2f)  %11.3f\n",
                "native CUDA", "GPU", native.seconds,
                native.generations_per_second,
                native.generations_per_second, native.generations_per_second,
                native.aggregate_cells_per_second / 1.0e6);

    const double gpu_vs_cpu1 = warp.aggregate_cells_per_second /
                               cpu_warp_1.aggregate_cells_per_second;
    const double gpu_vs_cpup = warp.aggregate_cells_per_second /
                               cpu_warp_parallel.aggregate_cells_per_second;
    const double native_cpu_vs_warp_cpu =
        native_cpu_1.aggregate_cells_per_second /
        cpu_warp_1.aggregate_cells_per_second;
    const double native_cpu_p_vs_warp_cpu_p =
        native_cpu_parallel.aggregate_cells_per_second /
        cpu_warp_parallel.aggregate_cells_per_second;
    const double native_cuda_vs_warp_gpu =
        native.aggregate_cells_per_second / warp.aggregate_cells_per_second;
    const double compiled_vs_interpreted =
        compiled.aggregate_cells_per_second / warp.aggregate_cells_per_second;
    const double compiled_vs_native_cpu_1 =
        compiled.aggregate_cells_per_second /
        native_cpu_1.aggregate_cells_per_second;
    const double compiled_vs_native_cpu_parallel =
        compiled.aggregate_cells_per_second /
        native_cpu_parallel.aggregate_cells_per_second;
    const double native_cuda_vs_compiled =
        native.aggregate_cells_per_second /
        compiled.aggregate_cells_per_second;
    std::printf("  ratios: compiled WarpVM / GPU WarpVM = %.1fx\n",
                compiled_vs_interpreted);
    std::printf("          compiled WarpVM / native CPU(1) = %.3fx",
                compiled_vs_native_cpu_1);
    if (parallel_workers > 1)
      std::printf(", compiled WarpVM / native CPU(%u) = %.3fx",
                  parallel_workers, compiled_vs_native_cpu_parallel);
    std::printf("\n");
    std::printf("  ratios: GPU WarpVM / CPU WarpVM(1) = %.3fx", gpu_vs_cpu1);
    if (parallel_workers > 1)
      std::printf(", GPU WarpVM / CPU WarpVM(%u) = %.3fx",
                  parallel_workers, gpu_vs_cpup);
    std::printf("\n");
    std::printf("          native CPU(1) / CPU WarpVM(1) = %.1fx",
                native_cpu_vs_warp_cpu);
    if (parallel_workers > 1)
      std::printf(", native CPU(%u) / CPU WarpVM(%u) = %.1fx",
                  parallel_workers, parallel_workers,
                  native_cpu_p_vs_warp_cpu_p);
    std::printf("\n");
    std::printf("          native CUDA / GPU WarpVM = %.1fx\n",
                native_cuda_vs_warp_gpu);
    std::printf("          native CUDA / compiled WarpVM = %.1fx\n",
                native_cuda_vs_compiled);
  }
  std::printf("\nlife_benchmark: %s\n", pass ? "PASS" : "FAIL");
  return pass ? 0 : 1;
}

int RunLifeProfile(const char* path, int duration_ms) {
  if (RunCpuGpuLifeEquivalence(path) != 0) {
    std::fprintf(stderr,
                 "error: refusing to profile non-equivalent CPU/GPU state\n");
    return 1;
  }
  constexpr std::array<std::pair<PersistentKernelMode, const char*>, 8>
      kExperimentalModes{{
          {PersistentKernelMode::kScalarRegs, "scalar_regs"},
          {PersistentKernelMode::kDenseDispatch, "dense_dispatch"},
          {PersistentKernelMode::kHotDispatch, "hot_dispatch"},
          {PersistentKernelMode::kHot4Dispatch, "hot4_dispatch"},
          {PersistentKernelMode::kScalarRegsDenseDispatch, "scalar_dense"},
          {PersistentKernelMode::kSharedRegs, "shared_regs"},
          {PersistentKernelMode::kSharedRegsThreeBlock, "shared_regs_3block"},
          {PersistentKernelMode::kSharedRegsDenseThreeBlock,
           "shared_dense_3block"},
      }};
  for (const auto& [mode, label] : kExperimentalModes) {
    if (RunCpuGpuLifeEquivalenceMode(path, mode, label) != 0) {
      std::fprintf(stderr,
                   "error: refusing to profile non-equivalent %s kernel\n",
                   label);
      return 1;
    }
  }
  WvmFile file;
  std::string err;
  if (!LoadWvm(path, file, err)) {
    std::fprintf(stderr, "error: %s: %s\n", path, err.c_str());
    return 2;
  }

  const bool production_inlined = file.code.size() == 215;
  const uint32_t evolve_pc = 51;
  const uint32_t render_pc = production_inlined ? 185 : 113;
  const uint32_t publish_pc = production_inlined ? 209 : 137;
  const uint32_t framebuffer_store_pc = production_inlined ? 205 : 133;
  if ((!production_inlined && file.code.size() != 153) ||
      ((file.code[evolve_pc] >> kOpcodeShift) & kOpcodeMask) != kSMovI ||
      ((file.code[render_pc] >> kOpcodeShift) & kOpcodeMask) != kLdw ||
      ((file.code[framebuffer_store_pc] >> kOpcodeShift) & kOpcodeMask) !=
          kStore) {
    std::fprintf(stderr,
                 "error: WarpLife phase PCs do not match the profiled build\n");
    return 1;
  }

  LifeCensus census;
  if (!CollectLifeCensus(file, census, err, evolve_pc, render_pc,
                         publish_pc)) {
    std::fprintf(stderr, "error: %s\n", err.c_str());
    return 1;
  }
  InlinedLife inlined;
  LifeCensus inlined_census;
  if (production_inlined) {
    inlined.file = file;
    inlined.evolve_pc = evolve_pc;
    inlined.render_pc = render_pc;
    inlined.publish_pc = publish_pc;
    inlined_census = census;
  } else {
    if (!InlineLifeLoadCell(file, inlined, err) ||
        !CpuLifeProgramsMatch(file, inlined.file, err) ||
        !CollectLifeCensus(inlined.file, inlined_census, err,
                           inlined.evolve_pc, inlined.render_pc,
                           inlined.publish_pc)) {
      std::fprintf(stderr, "error: inlined load_cell profile: %s\n",
                   err.c_str());
      return 1;
    }
  }
  const uint64_t total = census.phases[0] + census.phases[1] +
                         census.phases[2];
  std::printf("\nWarpLife one-VM profile\n");
  std::printf("  dynamic census: one complete steady-state generation\n");
  std::printf("  retired bytecodes: %llu\n",
              static_cast<unsigned long long>(total));
  std::printf("  evolution:          %8llu  %5.1f%%\n",
              static_cast<unsigned long long>(census.phases[0]),
              100.0 * census.phases[0] / total);
  std::printf("  rendering:          %8llu  %5.1f%%\n",
              static_cast<unsigned long long>(census.phases[1]),
              100.0 * census.phases[1] / total);
  std::printf("  publish/control:    %8llu  %5.1f%%\n",
              static_cast<unsigned long long>(census.phases[2]),
              100.0 * census.phases[2] / total);
  std::printf("  per-op fault votes: %8llu  (one warp BALLOT per bytecode)\n",
              static_cast<unsigned long long>(total));
  std::printf("  host-control polls: %8llu  (backward transfers + YIELD)\n",
              static_cast<unsigned long long>(census.control_polls));
  std::printf("    evolve loop 511, render loop 511, generation JMP 1, "
              "YIELD 1 (RET is not a control point)\n");
  std::printf("  RAM load lanes:     %8llu, summed unique addresses: %llu\n",
              static_cast<unsigned long long>(census.ram_load_lanes),
              static_cast<unsigned long long>(
                  census.ram_load_unique_addresses));
  std::printf("  RAM store lanes:    %8llu\n",
              static_cast<unsigned long long>(census.ram_store_lanes));
  std::printf("  framebuffer stores: %8llu lane writes\n",
              static_cast<unsigned long long>(
                  census.framebuffer_store_lanes));

  std::vector<std::pair<uint64_t, uint32_t>> ranked;
  for (uint32_t op = 0; op < census.opcodes.size(); ++op)
    if (census.opcodes[op]) ranked.emplace_back(census.opcodes[op], op);
  std::sort(ranked.begin(), ranked.end(),
            [](const auto& a, const auto& b) { return a.first > b.first; });
  std::printf("\n  dynamic opcode mix\n");
  std::printf("    opcode             count   share\n");
  for (const auto& [count, op] : ranked) {
    std::printf("    %-14s %9llu  %5.1f%%\n", OpcodeName(op),
                static_cast<unsigned long long>(count),
                100.0 * count / total);
  }
  const uint64_t inlined_total = inlined_census.phases[0] +
                                 inlined_census.phases[1] +
                                 inlined_census.phases[2];
  std::printf("\n  load_cell inlining census\n");
  if (production_inlined) {
    std::printf("    production assembly:           load_cell is inlined\n");
    std::printf("    static code words:                   153 -> %zu\n",
                file.code.size());
    std::printf("    dynamic bytecodes/generation:      86030 -> %llu "
                "(%5.1f%% fewer)\n",
                static_cast<unsigned long long>(total),
                100.0 * (86030 - total) / 86030.0);
    std::printf("    current CALL / RET count:      %9llu / %llu\n",
                static_cast<unsigned long long>(census.opcodes[kCall]),
                static_cast<unsigned long long>(census.opcodes[kRet]));
  } else {
    std::printf("    static code words:             %9zu -> %zu\n",
                file.code.size(), inlined.file.code.size());
    std::printf("    dynamic bytecodes/generation:  %9llu -> %llu "
                "(%5.1f%% fewer)\n",
                static_cast<unsigned long long>(total),
                static_cast<unsigned long long>(inlined_total),
                100.0 * (total - inlined_total) / total);
    std::printf("    CALL / RET per generation:     %9llu / %llu -> 0 / 0\n",
                static_cast<unsigned long long>(census.opcodes[kCall]),
                static_cast<unsigned long long>(census.opcodes[kRet]));
    std::printf("    CPU worlds vm 0/2/37:          identical through generation 3\n");
  }
  WvmFile no_render = file;
  no_render.code[render_pc] = enc_i(kJmp, 0, 0, 0, publish_pc);
  WvmFile render_only = file;
  render_only.code[evolve_pc] = enc_i(kJmp, 0, 0, 0, render_pc);
  WvmFile no_framebuffer_write = file;
  no_framebuffer_write.code[framebuffer_store_pc] =
      enc_r(kNop, 0, 0, 0, 0);

  std::printf("\n  phase timing (median of 3, %d ms samples)\n", duration_ms);
  err.clear();
  const double full_ms =
      MedianGpuMillisecondsPerFrame(file, duration_ms, err);
  std::array<double, kExperimentalModes.size()> experimental_ms{};
  for (size_t index = 0; index < kExperimentalModes.size() && full_ms;
       ++index) {
    experimental_ms[index] = MedianGpuMillisecondsPerFrame(
        file, duration_ms, err, kExperimentalModes[index].first);
  }
  const bool experimental_timing_ok =
      std::all_of(experimental_ms.begin(), experimental_ms.end(),
                  [](double value) { return value != 0.0; });
  if (!full_ms || !experimental_timing_ok) {
    std::fprintf(stderr, "error: interpreter matrix: %s\n", err.c_str());
    return 1;
  }
  std::printf("    full WarpLife:                 %9.3f ms/generation\n",
              full_ms);
  std::printf("\n    interpreter experiment matrix (normal semantics):\n");
  std::printf("      %-24s %9.3f ms  %6.2fx baseline\n", "baseline",
              full_ms, 1.0);
  for (size_t index = 0; index < kExperimentalModes.size(); ++index) {
    std::printf("      %-24s %9.3f ms  %6.2fx baseline\n",
                kExperimentalModes[index].second, experimental_ms[index],
                full_ms / experimental_ms[index]);
  }
  // Counter sequential-order drift by alternating the production and hot
  // kernels. Each entry is one independently warmed, frame-aligned sample;
  // report the median of three for each mode.
  std::array<double, 3> alternating_baseline{};
  std::array<double, 3> alternating_hot{};
  for (size_t round = 0; round < alternating_baseline.size(); ++round) {
    double* first = round % 2 == 0 ? &alternating_baseline[round]
                                    : &alternating_hot[round];
    double* second = round % 2 == 0 ? &alternating_hot[round]
                                     : &alternating_baseline[round];
    const PersistentKernelMode first_mode =
        round % 2 == 0 ? PersistentKernelMode::kNormal
                       : PersistentKernelMode::kHotDispatch;
    const PersistentKernelMode second_mode =
        round % 2 == 0 ? PersistentKernelMode::kHotDispatch
                       : PersistentKernelMode::kNormal;
    *first = MeasureGpuMillisecondsPerFrame(file, duration_ms, err,
                                            first_mode);
    *second = *first ? MeasureGpuMillisecondsPerFrame(
                           file, duration_ms, err, second_mode)
                     : 0.0;
  }
  if (std::any_of(alternating_baseline.begin(), alternating_baseline.end(),
                  [](double value) { return value == 0.0; }) ||
      std::any_of(alternating_hot.begin(), alternating_hot.end(),
                  [](double value) { return value == 0.0; })) {
    std::fprintf(stderr, "error: alternating dispatch comparison: %s\n",
                 err.c_str());
    return 1;
  }
  std::sort(alternating_baseline.begin(), alternating_baseline.end());
  std::sort(alternating_hot.begin(), alternating_hot.end());
  const double alternating_baseline_ms = alternating_baseline[1];
  const double alternating_hot_ms = alternating_hot[1];
  std::printf("\n    alternating baseline/hot comparison:\n");
  std::printf("      baseline                    %9.3f ms\n",
              alternating_baseline_ms);
  std::printf("      hot_dispatch                %9.3f ms  %6.2fx baseline\n",
              alternating_hot_ms,
              alternating_baseline_ms / alternating_hot_ms);
  const double baseline_cycles = MedianGpuCyclesPerFrame(
      file, 1, err, PersistentKernelMode::kCycleProfile);
  const double hot_cycles = baseline_cycles
                                ? MedianGpuCyclesPerFrame(
                                      file, 1, err,
                                      PersistentKernelMode::kHotCycleProfile)
                                : 0.0;
  const double hot4_cycles = hot_cycles
                                 ? MedianGpuCyclesPerFrame(
                                       file, 1, err,
                                       PersistentKernelMode::kHot4CycleProfile)
                                 : 0.0;
  if (!hot4_cycles) {
    std::fprintf(stderr, "error: dispatch cycle comparison: %s\n",
                 err.c_str());
    return 1;
  }
  std::printf("\n    device-cycle comparison (median of 31 frames):\n");
  std::printf("      baseline                    %12.0f cycles/frame\n",
              baseline_cycles);
  std::printf("      hot_dispatch                %12.0f cycles/frame  "
              "%6.2fx baseline\n",
              hot_cycles, baseline_cycles / hot_cycles);
  std::printf("      hot4_dispatch               %12.0f cycles/frame  "
              "%6.2fx baseline\n",
              hot4_cycles, baseline_cycles / hot4_cycles);
  const double baseline_64_cycles = MedianGpuCyclesPerFrame(
      file, 64, err, PersistentKernelMode::kCycleProfile);
  const double hot_64_cycles = baseline_64_cycles
                                   ? MedianGpuCyclesPerFrame(
                                         file, 64, err,
                                         PersistentKernelMode::kHotCycleProfile)
                                   : 0.0;
  if (!hot_64_cycles) {
    std::fprintf(stderr, "error: 64-VM dispatch cycle comparison: %s\n",
                 err.c_str());
    return 1;
  }
  std::printf("      baseline, 64 VMs            %12.0f cycles/frame\n",
              baseline_64_cycles);
  std::printf("      hot_dispatch, 64 VMs        %12.0f cycles/frame  "
              "%6.2fx baseline\n",
              hot_64_cycles, baseline_64_cycles / hot_64_cycles);
  std::fflush(stdout);
  const double evolve_publish_ms =
      full_ms ? MedianGpuMillisecondsPerFrame(no_render, duration_ms, err) : 0;
  const double render_publish_ms =
      evolve_publish_ms
          ? MedianGpuMillisecondsPerFrame(render_only, duration_ms, err)
          : 0;
  const double no_fb_ms =
      render_publish_ms
          ? MedianGpuMillisecondsPerFrame(no_framebuffer_write, duration_ms,
                                          err)
          : 0;
  const double no_fault_vote_ms =
      no_fb_ms ? MedianGpuMillisecondsPerFrame(
                     file, duration_ms, err,
                     PersistentKernelMode::kNoFaultVotes)
               : 0;
  const double yield_poll_ms =
      no_fault_vote_ms
          ? MedianGpuMillisecondsPerFrame(
                file, duration_ms, err,
                PersistentKernelMode::kYieldOnlyPolling)
          : 0;
  const double minimal_ms =
      yield_poll_ms
          ? MedianGpuMillisecondsPerFrame(
                file, duration_ms, err,
                PersistentKernelMode::kNoFaultVotesYieldOnlyPolling)
          : 0;
  const double inlined_ms =
      (!production_inlined && minimal_ms)
          ? MedianGpuMillisecondsPerFrame(inlined.file, duration_ms, err)
          : 0;
  if (!minimal_ms || (!production_inlined && !inlined_ms)) {
    std::fprintf(stderr, "error: phase profile: %s\n", err.c_str());
    return 1;
  }
  const double evolve_ms = full_ms - render_publish_ms;
  const double render_ms = full_ms - evolve_publish_ms;
  const double phase_residual_ms =
      evolve_publish_ms + render_publish_ms - full_ms;
  const double framebuffer_memory_ms = full_ms - no_fb_ms;
  std::printf("\n    compiled kernel resources (%d-thread blocks):\n",
              kPersistentBlockThreads);
  constexpr std::array<std::pair<PersistentKernelMode, const char*>, 9>
      kMatrixModes{{
          {PersistentKernelMode::kNormal, "baseline"},
          {PersistentKernelMode::kScalarRegs, "scalar_regs"},
          {PersistentKernelMode::kDenseDispatch, "dense_dispatch"},
          {PersistentKernelMode::kHotDispatch, "hot_dispatch"},
          {PersistentKernelMode::kHot4Dispatch, "hot4_dispatch"},
          {PersistentKernelMode::kScalarRegsDenseDispatch, "scalar_dense"},
          {PersistentKernelMode::kSharedRegs, "shared_regs"},
          {PersistentKernelMode::kSharedRegsThreeBlock, "shared_regs_3block"},
          {PersistentKernelMode::kSharedRegsDenseThreeBlock,
           "shared_dense_3block"},
      }};
  for (const auto& [mode, label] : kMatrixModes) {
    PersistentKernelStats stats;
    if (!PersistentStatsForMode(mode, stats, err)) {
      std::fprintf(stderr, "error: %s\n", err.c_str());
      return 1;
    }
    std::printf("      %-24s regs %3d, local %4zu B/thread, "
                "shared %5zu B/block, %d blocks/SM, %u VM slots\n",
                label, stats.registers_per_thread,
                stats.local_bytes_per_thread, stats.shared_bytes_per_block,
                stats.blocks_per_sm,
                stats.resident_vm_slots);
  }

  std::printf("\n    64-VM throughput (%d ms samples):\n", duration_ms);
  for (const auto& [mode, label] : kMatrixModes) {
    err.clear();
    const WarpMeasurement measurement =
        MeasureWarpVm(file, 64, duration_ms, err, mode);
    if (!measurement.ok) {
      std::fprintf(stderr, "error: 64-VM %s: %s\n", label, err.c_str());
      return 1;
    }
    std::printf("      %-24s %8.2f gen/s/VM, %8.3f Mcell/s aggregate\n",
                label, measurement.avg_generations_per_second,
                measurement.aggregate_cells_per_second / 1.0e6);
  }

  std::printf("    evolution + publication:      %9.3f ms/generation\n",
              evolve_publish_ms);
  std::printf("    rendering + publication:      %9.3f ms/generation\n",
              render_publish_ms);
  std::printf("    full with FB STORE -> NOP:     %9.3f ms/generation\n",
              no_fb_ms);
  std::printf("    differential estimates: evolution %7.3f ms (%5.1f%%), "
              "render %7.3f ms (%5.1f%%)\n",
              evolve_ms, 100.0 * evolve_ms / full_ms, render_ms,
              100.0 * render_ms / full_ms);
  std::printf("    cross-run/additivity residual: %7.3f ms "
              "(not assigned to publication)\n",
              phase_residual_ms);
  std::printf("    FB memory-write estimate:      %9.3f ms (%5.1f%%)\n",
              framebuffer_memory_ms,
              100.0 * framebuffer_memory_ms / full_ms);
  std::printf("    benchmark-only safety ablations (normal semantics unchanged):\n");
  std::printf("      no per-op fault vote:        %9.3f ms, saving %7.3f ms (%5.1f%%)\n",
              no_fault_vote_ms, full_ms - no_fault_vote_ms,
              100.0 * (full_ms - no_fault_vote_ms) / full_ms);
  std::printf("      poll host only at YIELD:     %9.3f ms, saving %7.3f ms (%5.1f%%)\n",
              yield_poll_ms, full_ms - yield_poll_ms,
              100.0 * (full_ms - yield_poll_ms) / full_ms);
  std::printf("      both ablations:              %9.3f ms, saving %7.3f ms (%5.1f%%)\n",
              minimal_ms, full_ms - minimal_ms,
              100.0 * (full_ms - minimal_ms) / full_ms);
  if (production_inlined) {
    std::printf("    load_cell:                      inlined in production "
                "(no duplicate timing run)\n");
  } else {
    std::printf("    inlined load_cell (normal runtime): %6.3f ms, saving %7.3f ms "
                "(%5.1f%%), throughput %.2fx\n",
                inlined_ms, full_ms - inlined_ms,
                100.0 * (full_ms - inlined_ms) / full_ms,
                full_ms / inlined_ms);
  }

  auto micro = [&](MicroBody body, uint32_t body_ops) {
    err.clear();
    return MedianGpuMillisecondsPerFrame(MakeMicroProgram(body, body_ops),
                                         duration_ms, err);
  };
  const double nop0 = micro(MicroBody::kNop, 0);
  const double nop128 = nop0 ? micro(MicroBody::kNop, 128) : 0;
  const double step_trap128 =
      nop128 ? micro(MicroBody::kStepTrap, 128) : 0;
  const double nop32 = step_trap128 ? micro(MicroBody::kNop, 32) : 0;
  const double add32 = nop32 ? micro(MicroBody::kAdd, 32) : 0;
  const double load32 = add32 ? micro(MicroBody::kRamLoad, 32) : 0;
  const double store32 = load32 ? micro(MicroBody::kRamStore, 32) : 0;
  const double reduce32 = store32 ? micro(MicroBody::kReduceOr, 32) : 0;
  const double fb_store32 = reduce32 ? micro(MicroBody::kFbStore, 32) : 0;
  if (!fb_store32) {
    std::fprintf(stderr, "error: microprofile: %s\n", err.c_str());
    return 1;
  }

  constexpr double kBodyExecutions = 1024.0;
  const double incremental_nop_ns =
      (nop128 - nop0) * 1.0e6 / (128.0 * kBodyExecutions);
  const uint64_t nop0_instructions = 4 + 1024 * 3;
  const double estimated_poll_us =
      (nop0 - nop0_instructions * incremental_nop_ns / 1.0e6) * 1000.0 /
      1025.0;
  auto delta_ns = [&](double value) {
    return (value - nop32) * 1.0e6 / (32.0 * kBodyExecutions);
  };
  std::printf("\n  matched microprograms (1,024 backward polls + 1 YIELD poll/frame)\n");
  std::printf("    NOP body 0:                    %9.3f ms/frame\n", nop0);
  std::printf("    NOP body 32:                   %9.3f ms/frame\n", nop32);
  std::printf("    NOP body 128:                  %9.3f ms/frame\n", nop128);
  std::printf("    STEP_TRAP body 128:            %9.3f ms/frame\n",
              step_trap128);
  std::printf("    incremental NOP slot:          %9.1f ns/op\n",
              incremental_nop_ns);
  std::printf("    STEP_TRAP - NOP dispatch path: %9.1f ns/op\n",
              (step_trap128 - nop128) * 1.0e6 /
                  (128.0 * kBodyExecutions));
  std::printf("    inferred control poll:         %9.3f us/poll\n",
              estimated_poll_us);
  std::printf("    equal-count delta versus NOP: ADD %+.1f, RAM LOAD %+.1f, "
              "RAM STORE %+.1f, REDUCE_OR %+.1f, FB STORE %+.1f ns/op\n",
              delta_ns(add32), delta_ns(load32), delta_ns(store32),
              delta_ns(reduce32), delta_ns(fb_store32));
  const double fault_vote_ns =
      (full_ms - no_fault_vote_ms) * 1.0e6 / total;
  const double measured_poll_us =
      (full_ms - yield_poll_ms) * 1000.0 /
      (census.control_polls - 1);  // the YIELD poll remains in the ablation
  const double remaining_average_ns = minimal_ms * 1.0e6 / total;
  std::printf("\n  derived one-VM costs\n");
  std::printf("    per-op fault-vote contribution: %7.1f ns\n", fault_vote_ns);
  std::printf("    measured backward-poll contribution: %7.3f us/poll\n",
              measured_poll_us);
  std::printf("    NOP dispatch/execute floor after fault vote: ~%.1f ns/op\n",
              incremental_nop_ns - fault_vote_ns);
  std::printf("    workload average with both safety costs removed: %.1f ns/op\n",
              remaining_average_ns);
  std::printf("\nlife_profile: PASS\n");
  return 0;
}

int RunCpuGpuLifeEquivalenceMode(const char* path,
                                 PersistentKernelMode mode,
                                 const char* label) {
  WvmFile file;
  std::string err;
  if (!LoadWvm(path, file, err)) {
    std::fprintf(stderr, "error: %s: %s\n", path, err.c_str());
    return 2;
  }

  constexpr std::array<uint32_t, 4> kSelected{0, 1, 2, 37};
  constexpr uint32_t kGpuVms = 38;
  constexpr uint32_t kWorldWords = 512;
  std::vector<VmImage> images(kGpuVms);
  for (VmImage& image : images) {
    image.code = file.code;
    image.literals = file.literals;
    image.mem_size_words = 16384;
  }
  PersistentRuntime gpu;
  if (!gpu.Init(images, err) || !gpu.Launch(err, mode)) {
    std::fprintf(stderr, "error: %s\n", err.c_str());
    return 1;
  }
  gpu.BootAll();

  bool ready = false;
  for (int waited = 0; waited < 30000; ++waited) {
    ready = true;
    for (const uint32_t vm : kSelected)
      ready &= gpu.FrameSeq(vm) >= 3 && gpu.Status(vm) == kRunning;
    if (ready) break;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  struct GpuWorld {
    uint32_t frame_seq = 0;
    std::vector<uint32_t> words;
  };
  std::array<GpuWorld, kSelected.size()> expected;
  bool captured = ready;
  for (size_t index = 0; index < kSelected.size() && captured; ++index) {
    const uint32_t vm = kSelected[index];
    captured &= gpu.Pause(vm);
    VmState state{};
    captured &= gpu.ReadState(vm, state);
    expected[index].frame_seq = gpu.FrameSeq(vm);
    captured &= state.sregs[0] == 0 || state.sregs[0] == kWorldWords;
    captured &= gpu.ReadMem(vm, state.sregs[0], kWorldWords,
                            expected[index].words);
  }
  gpu.ShutdownAll();
  captured &= gpu.Sync() == cudaSuccess;

  auto checksum = [](const std::vector<uint32_t>& words) {
    uint64_t hash = 1469598103934665603ull;
    for (const uint32_t word : words) {
      hash ^= word;
      hash *= 1099511628211ull;
    }
    return hash;
  };

  bool equivalent = captured;
  for (size_t index = 0; index < kSelected.size() && captured; ++index) {
    const uint32_t vm_id = kSelected[index];
    CpuVm cpu;
    cpu.Init(vm_id, file, 16384);
    const uint32_t target = expected[index].frame_seq;
    const uint64_t budget = static_cast<uint64_t>(target + 2) * 200000u;
    while (cpu.status == kRunning && cpu.frame_seq < target &&
           cpu.instruction_counter < budget) {
      cpu.RunQuantum(kCpuVmQuantum);
    }
    const uint32_t base = cpu.sregs[0];
    std::vector<uint32_t> actual;
    if ((base == 0 || base == kWorldWords) &&
        cpu.memory.size() >= base + kWorldWords) {
      actual.assign(cpu.memory.begin() + base,
                    cpu.memory.begin() + base + kWorldWords);
    }
    const bool match = cpu.status == kRunning && cpu.frame_seq == target &&
                       actual == expected[index].words;
    equivalent &= match;
    std::printf(
        "%s_equivalence: vm=%u generation=%u cpu=%016llx gpu=%016llx %s\n",
        label, vm_id, target, static_cast<unsigned long long>(checksum(actual)),
        static_cast<unsigned long long>(checksum(expected[index].words)),
        match ? "PASS" : "FAIL");
  }
  std::printf("%s_equivalence: %s\n", label,
              equivalent ? "PASS" : "FAIL");
  return equivalent ? 0 : 1;
}

int RunCpuGpuLifeEquivalence(const char* path) {
  return RunCpuGpuLifeEquivalenceMode(path, PersistentKernelMode::kNormal,
                                      "cpu_gpu");
}

int RunNativeCpuLifeEquivalence(const char* path) {
  WvmFile file;
  std::string err;
  if (!LoadWvm(path, file, err)) {
    std::fprintf(stderr, "error: %s: %s\n", path, err.c_str());
    return 2;
  }

  constexpr std::array<uint32_t, 4> kSelected{0, 1, 2, 37};
  constexpr uint32_t kNativeVms = 38;
  constexpr uint32_t kTargetGeneration = 3;
  constexpr uint32_t kWorldWords = kLifeCells / 32;
  NativeCpuState native = MakeNativeCpuState(kNativeVms);
  for (uint32_t generation = 0; generation < kTargetGeneration; ++generation)
    for (uint32_t vm : kSelected) NativeCpuStepVm(native, vm);

  auto checksum = [](const std::vector<uint32_t>& words) {
    uint64_t hash = 1469598103934665603ull;
    for (uint32_t word : words) {
      hash ^= word;
      hash *= 1099511628211ull;
    }
    return hash;
  };

  bool equivalent = true;
  for (uint32_t vm_id : kSelected) {
    CpuVm interpreted;
    interpreted.Init(vm_id, file, 16384);
    const uint64_t budget =
        static_cast<uint64_t>(kTargetGeneration + 2) * 200000u;
    while (interpreted.status == kRunning &&
           interpreted.frame_seq < kTargetGeneration &&
           interpreted.instruction_counter < budget)
      interpreted.RunQuantum(kCpuVmQuantum);

    const size_t native_base = static_cast<size_t>(vm_id) * kLifeCells;
    const uint8_t* native_world =
        (native.current_is_b[vm_id] ? native.b.data() : native.a.data()) +
        native_base;
    std::vector<uint32_t> packed_native(kWorldWords, 0);
    for (uint32_t word = 0; word < kWorldWords; ++word)
      for (uint32_t bit = 0; bit < 32; ++bit)
        packed_native[word] |=
            static_cast<uint32_t>(native_world[word * 32 + bit] != 0) << bit;

    std::vector<uint32_t> packed_warpvm;
    const uint32_t world_base = interpreted.sregs[0];
    if ((world_base == 0 || world_base == kWorldWords) &&
        interpreted.memory.size() >= world_base + kWorldWords) {
      packed_warpvm.assign(interpreted.memory.begin() + world_base,
                           interpreted.memory.begin() + world_base +
                               kWorldWords);
    }
    const uint32_t* native_framebuffer =
        native.framebuffer.data() + native_base;
    const bool framebuffer_equal = std::equal(
        interpreted.framebuffer.begin(), interpreted.framebuffer.end(),
        native_framebuffer);
    const bool match = interpreted.status == kRunning &&
                       interpreted.frame_seq == kTargetGeneration &&
                       native.generations[vm_id] == kTargetGeneration &&
                       packed_warpvm == packed_native && framebuffer_equal;
    equivalent &= match;
    std::printf(
        "native_cpu_equivalence: vm=%u generation=%u warpvm=%016llx "
        "native=%016llx world=%s framebuffer=%s\n",
        vm_id, kTargetGeneration,
        static_cast<unsigned long long>(checksum(packed_warpvm)),
        static_cast<unsigned long long>(checksum(packed_native)),
        packed_warpvm == packed_native ? "PASS" : "FAIL",
        framebuffer_equal ? "PASS" : "FAIL");
  }
  std::printf("native_cpu_equivalence: %s\n",
              equivalent ? "PASS" : "FAIL");
  return equivalent ? 0 : 1;
}

}  // namespace wvm
