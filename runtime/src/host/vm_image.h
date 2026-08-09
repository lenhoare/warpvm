// Host-side description of one VM image to load: program, literals, and
// private RAM (zero-filled, optionally pre-seeded).
#pragma once

#include <cstdint>
#include <vector>

namespace wvm {

struct VmImage {
  std::vector<uint32_t> code;
  std::vector<uint32_t> literals;
  uint32_t mem_size_words = 0;
  std::vector<uint32_t> mem_init;
};

}  // namespace wvm
