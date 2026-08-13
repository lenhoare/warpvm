// SDL2 framebuffer viewer (v0.1.1).
//
// Boots the persistent kernel with the given program, then opens either an
// enlarged single-VM view or a tiled view of every resident VM. VMs keep
// running while the host updates only tiles whose frame_seq has advanced.
// Close the window (or press Esc) to shut the kernel down.
#include <SDL2/SDL.h>

#include <algorithm>
#include <cstdio>
#include <string>
#include <unordered_map>
#include <vector>

#include "gpu/warpvm.cuh"
#include "host/persistent.h"
#include "host/ptx_compiler.h"
#include "host/supervisor.h"
#include "host/vm_image.h"
#include "host/wvm_file.h"

namespace wvm {
namespace {

struct ViewerLayout {
  uint32_t displayed_vms;
  uint32_t columns;
  uint32_t rows;
  uint32_t texture_width;
  uint32_t texture_height;
};

ViewerLayout MakeGridLayout(uint32_t n_vms) {
  uint32_t columns = 1;
  while (columns * columns < n_vms) ++columns;
  const uint32_t rows = (n_vms + columns - 1) / columns;
  return ViewerLayout{n_vms, columns, rows, columns * kVideoWidth,
                      rows * kVideoHeight};
}

ViewerLayout MakeSingleLayout() {
  return ViewerLayout{1, 1, 1, kVideoWidth, kVideoHeight};
}

SDL_Rect FitToWindow(SDL_Renderer* renderer, const ViewerLayout& layout) {
  int output_width = 0;
  int output_height = 0;
  SDL_GetRendererOutputSize(renderer, &output_width, &output_height);

  const double x_scale =
      static_cast<double>(output_width) / layout.texture_width;
  const double y_scale =
      static_cast<double>(output_height) / layout.texture_height;
  const double scale = std::min(x_scale, y_scale);
  const int width = static_cast<int>(layout.texture_width * scale);
  const int height = static_cast<int>(layout.texture_height * scale);
  return SDL_Rect{(output_width - width) / 2, (output_height - height) / 2,
                  width, height};
}

int PresentRuntime(PersistentRuntime& rt, const std::string& source,
                   uint32_t first_vm, const ViewerLayout& layout,
                   const char* mode,
                   const std::vector<VmId>* logical_ids = nullptr,
                   bool owns_runtime = true) {
  auto shutdown_owned_runtime = [&]() {
    if (!owns_runtime) return;
    rt.ShutdownAll();
    rt.Sync();
  };
  if (SDL_Init(SDL_INIT_VIDEO) != 0) {
    std::fprintf(stderr, "error: SDL_Init: %s\n", SDL_GetError());
    shutdown_owned_runtime();
    return 1;
  }

  const bool grid = layout.displayed_vms > 1;
  const int single_scale = 4;
  int window_width =
      grid ? static_cast<int>(layout.texture_width)
           : static_cast<int>(layout.texture_width) * single_scale;
  int window_height =
      grid ? static_cast<int>(layout.texture_height)
           : static_cast<int>(layout.texture_height) * single_scale;
  // Keep large grids wholly inside the current desktop. The texture remains
  // at architectural resolution and SDL performs nearest-neighbour scaling.
  SDL_Rect usable_bounds{};
  if (SDL_GetDisplayUsableBounds(0, &usable_bounds) == 0) {
    const int max_width = std::max(320, usable_bounds.w - 64);
    const int max_height = std::max(240, usable_bounds.h - 64);
    const double fit =
        std::min({1.0, static_cast<double>(max_width) / window_width,
                  static_cast<double>(max_height) / window_height});
    window_width = static_cast<int>(window_width * fit);
    window_height = static_cast<int>(window_height * fit);
  }
  char title[128];
  if (grid) {
    std::snprintf(title, sizeof(title), "WarpVM - %u VMs (%ux%u)",
                  layout.displayed_vms, layout.columns, layout.rows);
  } else {
    const VmId shown_id = logical_ids == nullptr ? first_vm
                                                  : logical_ids->front();
    std::snprintf(title, sizeof(title), "WarpVM - VM %u", shown_id);
  }

  SDL_Window* win = SDL_CreateWindow(
      title, SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, window_width,
      window_height, SDL_WINDOW_RESIZABLE);
  if (!win) {
    std::fprintf(stderr, "error: SDL_CreateWindow: %s\n", SDL_GetError());
    SDL_Quit();
    shutdown_owned_runtime();
    return 1;
  }
  SDL_Renderer* ren = SDL_CreateRenderer(win, -1, SDL_RENDERER_ACCELERATED);
  if (!ren) ren = SDL_CreateRenderer(win, -1, SDL_RENDERER_SOFTWARE);
  if (!ren) {
    std::fprintf(stderr, "error: SDL_CreateRenderer: %s\n", SDL_GetError());
    SDL_DestroyWindow(win);
    SDL_Quit();
    shutdown_owned_runtime();
    return 1;
  }
  SDL_Texture* tex = SDL_CreateTexture(
      ren, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STREAMING,
      static_cast<int>(layout.texture_width),
      static_cast<int>(layout.texture_height));
  if (!tex) {
    std::fprintf(stderr, "error: SDL_CreateTexture: %s\n", SDL_GetError());
    SDL_DestroyRenderer(ren);
    SDL_DestroyWindow(win);
    SDL_Quit();
    shutdown_owned_runtime();
    return 1;
  }
  SDL_SetTextureScaleMode(tex, SDL_ScaleModeNearest);
  SDL_SetRenderDrawColor(ren, 0, 0, 0, 255);

  // Streaming texture contents are initially undefined. Black also fills the
  // unused cells in a non-square final row until VMs publish their first frame.
  std::vector<uint32_t> atlas(
      static_cast<size_t>(layout.texture_width) * layout.texture_height,
      kVideoResetColor);
  if (SDL_UpdateTexture(tex, nullptr, atlas.data(),
                        layout.texture_width * sizeof(uint32_t)) != 0) {
    std::fprintf(stderr, "error: SDL_UpdateTexture: %s\n", SDL_GetError());
    SDL_DestroyTexture(tex);
    SDL_DestroyRenderer(ren);
    SDL_DestroyWindow(win);
    SDL_Quit();
    shutdown_owned_runtime();
    return 1;
  }

  if (grid) {
    std::printf("view%s: %u VMs from %s in a %ux%u grid "
                "(close window or Esc to exit)\n",
                mode, layout.displayed_vms, source.c_str(), layout.columns,
                layout.rows);
  } else {
    const VmId shown_id = logical_ids == nullptr ? first_vm
                                                  : logical_ids->front();
    std::printf("view%s: VM %u from %s (close window or Esc to exit)\n",
                mode, shown_id, source.c_str());
  }

  std::vector<uint32_t> framebuffers;
  std::vector<uint32_t> last_seq(layout.displayed_vms, 0);
  std::vector<uint32_t> current_seq(layout.displayed_vms, 0);
  std::vector<VmSlot> last_slots(layout.displayed_vms, kInvalidVmSlot);
  std::vector<VmSlot> current_slots(layout.displayed_vms, kInvalidVmSlot);
  bool running = true;
  while (running) {
    SDL_Event e;
    while (SDL_PollEvent(&e)) {
      if (e.type == SDL_QUIT) running = false;
      if (e.type == SDL_KEYDOWN && e.key.keysym.sym == SDLK_ESCAPE)
        running = false;
    }

    bool refresh = false;
    for (uint32_t tile = 0; tile < layout.displayed_vms; ++tile) {
      current_slots[tile] =
          logical_ids == nullptr
              ? first_vm + tile
              : rt.SlotForVmId((*logical_ids)[tile]);
      current_seq[tile] = current_slots[tile] == kInvalidVmSlot
                              ? 0
                              : rt.FrameSeq(current_slots[tile]);
      refresh |= current_seq[tile] != last_seq[tile] ||
                 current_slots[tile] != last_slots[tile];
    }

    const bool copied =
        !refresh ? false
                 : logical_ids == nullptr
                       ? rt.ReadFramebuffers(first_vm, layout.displayed_vms,
                                             framebuffers)
                       : rt.ReadFramebuffers(0, rt.num_vms(), framebuffers);
    if (refresh && copied) {
      if (grid) {
        // Device storage is VM-major; the SDL texture is a row-major atlas.
        // Compose changed tiles on the CPU, then upload the atlas once.
        for (uint32_t tile = 0; tile < layout.displayed_vms; ++tile) {
          if (current_seq[tile] == last_seq[tile] &&
              current_slots[tile] == last_slots[tile])
            continue;
          const uint32_t tile_x = (tile % layout.columns) * kVideoWidth;
          const uint32_t tile_y = (tile / layout.columns) * kVideoHeight;
          const bool present = current_slots[tile] != kInvalidVmSlot;
          const size_t source_slot = logical_ids == nullptr
                                         ? tile
                                         : current_slots[tile];
          const uint32_t* tile_source =
              present ? framebuffers.data() + source_slot * kVideoWords
                      : nullptr;
          for (uint32_t y = 0; y < kVideoHeight; ++y) {
            uint32_t* destination =
                atlas.data() + static_cast<size_t>(tile_y + y) *
                                   layout.texture_width +
                tile_x;
            if (tile_source != nullptr)
              std::copy_n(tile_source + static_cast<size_t>(y) * kVideoWidth,
                          kVideoWidth, destination);
            else
              std::fill_n(destination, kVideoWidth, kVideoResetColor);
          }
        }
        if (SDL_UpdateTexture(tex, nullptr, atlas.data(),
                              layout.texture_width * sizeof(uint32_t)) == 0) {
          last_seq = current_seq;
          last_slots = current_slots;
        }
      } else {
        const uint32_t* single_source =
            current_slots[0] == kInvalidVmSlot
                ? atlas.data()
                : framebuffers.data() +
                      (logical_ids == nullptr ? 0 : current_slots[0]) *
                          kVideoWords;
        if (SDL_UpdateTexture(tex, nullptr, single_source,
                              kVideoWidth * sizeof(uint32_t)) == 0) {
          last_seq = current_seq;
          last_slots = current_slots;
        }
      }
    }

    SDL_RenderClear(ren);
    const SDL_Rect destination = FitToWindow(ren, layout);
    SDL_RenderCopy(ren, tex, nullptr, &destination);
    SDL_RenderPresent(ren);
    SDL_Delay(16);  // ~60 fps cap; keeps the display GPU responsive
  }

  cudaError_t sync_result = cudaSuccess;
  if (owns_runtime) {
    rt.ShutdownAll();
    sync_result = rt.Sync();
  }
  SDL_DestroyTexture(tex);
  SDL_DestroyRenderer(ren);
  SDL_DestroyWindow(win);
  SDL_Quit();
  if (sync_result != cudaSuccess) {
    std::fprintf(stderr, "error: kernel shutdown: %s\n",
                 cudaGetErrorString(sync_result));
    return 1;
  }
  return 0;
}

int RunViewer(const char* path, uint32_t resident_vms, uint32_t first_vm,
              const ViewerLayout& layout, bool compiled) {
  WvmFile file;
  std::string err;
  if (!LoadWvm(path, file, err)) {
    std::fprintf(stderr, "error: %s: %s\n", path, err.c_str());
    return 2;
  }

  std::vector<VmImage> images(resident_vms);
  for (VmImage& image : images) {
    image.code = file.code;
    image.literals = file.literals;
    image.mem_size_words = kRamSizeWords;
  }
  PersistentRuntime rt;
  PtxResidentProgram resident_program;
  bool launched = rt.Init(images, err);
  if (launched && compiled) {
    launched = rt.EnsureStream(err) && resident_program.Compile(file, err) &&
               resident_program.Launch(
                   reinterpret_cast<CUdeviceptr>(rt.DeviceStates()),
                   resident_vms,
                   reinterpret_cast<CUdeviceptr>(rt.DeviceDescs()),
                   reinterpret_cast<CUdeviceptr>(rt.DeviceControl()),
                   reinterpret_cast<CUdeviceptr>(rt.DeviceMailboxes()),
                   reinterpret_cast<CUstream>(rt.Stream()), err);
  } else if (launched) {
    launched = rt.Launch(err);
  }
  if (!launched) {
    std::fprintf(stderr, "error: %s\n", err.c_str());
    return 1;
  }
  rt.BootAll();
  return PresentRuntime(rt, path, first_vm, layout,
                        compiled ? " (compiled resident)" : "");
}

std::string ProgramName(const std::string& path, size_t ordinal) {
  const size_t slash = path.find_last_of("/\\");
  std::string name =
      slash == std::string::npos ? path : path.substr(slash + 1);
  const size_t dot = name.rfind('.');
  if (dot != std::string::npos) name.resize(dot);
  if (name.empty()) name = "program";
  return name + "-" + std::to_string(ordinal);
}

}  // namespace

int ViewSingleVm(const char* path, uint32_t vm_index, bool compiled) {
  // Boot enough VMs that the selected logical VM exists.
  return RunViewer(path, vm_index + 1, vm_index, MakeSingleLayout(), compiled);
}

int ViewVmGrid(const char* path, uint32_t n_vms, bool compiled) {
  return RunViewer(path, n_vms, 0, MakeGridLayout(n_vms), compiled);
}

int ViewHeterogeneousGrid(const std::vector<std::string>& paths) {
  if (paths.empty() || paths.size() > kMaxVms) {
    std::fprintf(stderr, "error: hetero_view requires 1..%u programs\n",
                 kMaxVms);
    return 2;
  }

  ProgramRegistry registry;
  VmDirectory directory(static_cast<uint32_t>(paths.size()));
  std::unordered_map<std::string, LoadedProgramId> loaded_paths;
  std::vector<VmBinding> bindings(paths.size());
  std::string err;
  for (VmSlot slot = 0; slot < paths.size(); ++slot) {
    LoadedProgramId program_id;
    const auto loaded = loaded_paths.find(paths[slot]);
    if (loaded != loaded_paths.end()) {
      program_id = loaded->second;
    } else {
      if (!registry.Load(paths[slot], ProgramName(paths[slot], slot),
                         program_id, err)) {
        std::fprintf(stderr, "error: %s: %s\n", paths[slot].c_str(),
                     err.c_str());
        return 2;
      }
      loaded_paths.emplace(paths[slot], program_id);
    }
    LogicalVmId vm_id;
    if (!directory.Create(ResidentSlotId{slot}, vm_id, err) ||
        !registry.Retain(program_id, err)) {
      std::fprintf(stderr, "error: %s\n", err.c_str());
      return 1;
    }
    bindings[slot].vm_id = vm_id;
    bindings[slot].program_id = program_id;
  }

  PersistentRuntime rt;
  if (!rt.Init(registry, bindings, err) || !rt.Launch(err)) {
    std::fprintf(stderr, "error: %s\n", err.c_str());
    return 1;
  }
  rt.BootAll();
  std::printf("heterogeneous population: %zu VMs, %zu shared program "
              "image(s)\n",
              paths.size(), rt.device_program_count());
  for (VmSlot slot = 0; slot < paths.size(); ++slot)
    std::printf("  VM %u -> slot %u -> %s\n", rt.VmIdAtSlot(slot), slot,
                rt.ProgramNameAtSlot(slot).c_str());
  return PresentRuntime(rt, "heterogeneous registry", 0,
                        MakeGridLayout(static_cast<uint32_t>(paths.size())),
                        " (heterogeneous interpreted)");
}

int ViewSupervisorPopulation(Supervisor& supervisor,
                             const std::vector<VmId>& logical_ids) {
  if (!supervisor.launched() || logical_ids.empty() ||
      logical_ids.size() > kMaxVms) {
    std::fprintf(stderr, "error: supervisor view requires active VMs\n");
    return 2;
  }
  for (VmId id : logical_ids) {
    if (supervisor.Find(LogicalVmId{id}) == nullptr) {
      std::fprintf(stderr, "error: logical VM %u is not active\n", id);
      return 2;
    }
  }
  std::printf("supervisor view logical mapping:\n");
  for (VmId id : logical_ids) {
    const VmInstanceInfo* vm = supervisor.Find(LogicalVmId{id});
    const ProgramInfo* program = supervisor.programs().Find(vm->program_id);
    std::printf("  tile VM %u -> slot %u -> %s -> %s\n", id,
                vm->slot.value,
                program == nullptr ? "?" : program->name.c_str(),
                LifecycleName(vm->lifecycle));
  }
  return PresentRuntime(
      supervisor.runtime(), "CPU supervisor", 0,
      MakeGridLayout(static_cast<uint32_t>(logical_ids.size())),
      " (stable logical IDs)", &logical_ids, false);
}

}  // namespace wvm
