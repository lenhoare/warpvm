// WarpVM host entry point.
// Slice 1: one warp, lane-wise arithmetic + reduction, result to host.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "gpu/warpvm.cuh"

namespace {

#define CUDA_CHECK(expr)                                                   \
  do {                                                                     \
    cudaError_t err_ = (expr);                                             \
    if (err_ != cudaSuccess) {                                             \
      std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(err_));                              \
      std::exit(1);                                                        \
    }                                                                      \
  } while (0)

// Slice 1 demo: per-lane polynomial + warp sum. Written as straight CUDA;
// from slice 2 the same computation is expressed as bytecode executed by
// the interpreter.
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
  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  std::printf("device: %s  SMs=%d  cc=%d.%d\n", prop.name,
              prop.multiProcessorCount, prop.major, prop.minor);

  uint32_t* d_lane = nullptr;
  uint32_t* d_sum = nullptr;
  CUDA_CHECK(cudaMalloc(&d_lane, wvm::kLanes * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&d_sum, sizeof(uint32_t)));

  // Exactly one warp: block of 32 threads, VM 0.
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

  std::printf("lane[0..3] = %u %u %u %u ...\n", lane[0], lane[1], lane[2],
              lane[3]);
  std::printf("warp sum   = %u (expect %u)\n", sum, expect_sum);
  std::printf(ok ? "slice1: PASS\n" : "slice1: FAIL\n");
  return ok ? 0 : 1;
}

void Usage(const char* argv0) {
  std::printf("usage: %s <command>\n", argv0);
  std::printf("  slice1    run the slice-1 warp arithmetic demo\n");
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) return RunSlice1();

  const char* cmd = argv[1];
  if (std::strcmp(cmd, "slice1") == 0) return RunSlice1();
  if (std::strcmp(cmd, "help") == 0 || std::strcmp(cmd, "--help") == 0) {
    Usage(argv[0]);
    return 0;
  }
  std::fprintf(stderr, "unknown command: %s\n", cmd);
  Usage(argv[0]);
  return 2;
}
