// Host-side .wvm file loading (docs/isa.md §10).
#pragma once

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

#include "gpu/warpvm.cuh"

namespace wvm {

constexpr uint32_t kWvmMagic = 0x304D5657u;  // 'W','V','M','0'
constexpr uint32_t kWvmVersion = 1u;
constexpr size_t kWvmHeaderBytes = 32;

struct WvmFile {
  std::vector<uint32_t> code;
  std::vector<uint32_t> literals;
};

inline bool LoadWvm(const std::string& path, WvmFile& out, std::string& err) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    err = "cannot open " + path;
    return false;
  }
  const std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(f)),
                                   std::istreambuf_iterator<char>());
  if (bytes.size() < kWvmHeaderBytes) {
    err = "truncated header";
    return false;
  }
  auto rd32 = [&bytes](size_t off) -> uint32_t {
    return static_cast<uint32_t>(bytes[off]) |
           (static_cast<uint32_t>(bytes[off + 1]) << 8) |
           (static_cast<uint32_t>(bytes[off + 2]) << 16) |
           (static_cast<uint32_t>(bytes[off + 3]) << 24);
  };

  if (rd32(0) != kWvmMagic) {
    err = "bad magic (not a .wvm file)";
    return false;
  }
  if (rd32(4) != kWvmVersion) {
    err = "unsupported .wvm version";
    return false;
  }
  const uint32_t code_len = rd32(12);
  const uint32_t lit_len = rd32(16);
  if (code_len == 0 || code_len > kMaxCodeWords) {
    err = "bad code_len";
    return false;
  }
  if (lit_len > kMaxLiterals) {
    err = "bad literals_len";
    return false;
  }
  const size_t need =
      kWvmHeaderBytes + 4ull * (static_cast<size_t>(code_len) + lit_len);
  if (bytes.size() < need) {
    err = "truncated body";
    return false;
  }

  out.code.resize(code_len);
  out.literals.resize(lit_len);
  for (uint32_t i = 0; i < code_len; ++i)
    out.code[i] = rd32(kWvmHeaderBytes + 4ull * i);
  for (uint32_t i = 0; i < lit_len; ++i)
    out.literals[i] =
        rd32(kWvmHeaderBytes + 4ull * (static_cast<size_t>(code_len) + i));
  return true;
}

}  // namespace wvm
