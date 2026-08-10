// SDL2 single-VM framebuffer viewer (v0.1.1, headless-tested copy path).
//
// Boots the persistent kernel with the given program, then opens a window
// showing one VM's 128x128 framebuffer enlarged with nearest-neighbour
// sampling. The VM keeps running; the texture refreshes whenever the VM
// publishes a new frame (frame_seq advances). Close the window (or press
// Esc) to shut the kernel down.
#include <SDL2/SDL.h>

#include <cstdio>
#include <string>
#include <vector>

#include "gpu/warpvm.cuh"
#include "host/persistent.h"
#include "host/vm_image.h"
#include "host/wvm_file.h"

namespace wvm {

int ViewSingleVm(const char* path, uint32_t vm_index) {
  WvmFile file;
  std::string err;
  if (!LoadWvm(path, file, err)) {
    std::fprintf(stderr, "error: %s: %s\n", path, err.c_str());
    return 2;
  }

  // Boot enough VMs that `vm_index` exists (others run too, harmlessly).
  const uint32_t n_vms = vm_index + 1;
  std::vector<VmImage> images(n_vms);
  for (uint32_t i = 0; i < n_vms; ++i) {
    images[i].code = file.code;
    images[i].literals = file.literals;
    images[i].mem_size_words = 8;
  }
  PersistentRuntime rt;
  if (!rt.Init(images, err) || !rt.Launch(err)) {
    std::fprintf(stderr, "error: %s\n", err.c_str());
    return 1;
  }
  rt.BootAll();

  if (SDL_Init(SDL_INIT_VIDEO) != 0) {
    std::fprintf(stderr, "error: SDL_Init: %s\n", SDL_GetError());
    rt.ShutdownAll();
    rt.Sync();
    return 1;
  }

  const int scale = 4;
  SDL_Window* win = SDL_CreateWindow(
      "WarpVM", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
      kVideoWidth * scale, kVideoHeight * scale, SDL_WINDOW_RESIZABLE);
  if (!win) {
    std::fprintf(stderr, "error: SDL_CreateWindow: %s\n", SDL_GetError());
    SDL_Quit();
    rt.ShutdownAll();
    rt.Sync();
    return 1;
  }
  SDL_Renderer* ren = SDL_CreateRenderer(win, -1, SDL_RENDERER_ACCELERATED);
  if (!ren) ren = SDL_CreateRenderer(win, -1, SDL_RENDERER_SOFTWARE);
  if (!ren) {
    std::fprintf(stderr, "error: SDL_CreateRenderer: %s\n", SDL_GetError());
    SDL_DestroyWindow(win);
    SDL_Quit();
    rt.ShutdownAll();
    rt.Sync();
    return 1;
  }
  SDL_Texture* tex =
      SDL_CreateTexture(ren, SDL_PIXELFORMAT_ARGB8888,
                        SDL_TEXTUREACCESS_STREAMING, kVideoWidth, kVideoHeight);
  if (!tex) {
    std::fprintf(stderr, "error: SDL_CreateTexture: %s\n", SDL_GetError());
    SDL_DestroyRenderer(ren);
    SDL_DestroyWindow(win);
    SDL_Quit();
    rt.ShutdownAll();
    rt.Sync();
    return 1;
  }
  SDL_SetTextureScaleMode(tex, SDL_ScaleModeNearest);

  std::printf("view: VM %u from %s (close window or Esc to exit)\n", vm_index,
              path);

  std::vector<uint32_t> fb;
  uint32_t last_seq = 0;
  bool running = true;
  while (running) {
    SDL_Event e;
    while (SDL_PollEvent(&e)) {
      if (e.type == SDL_QUIT) running = false;
      if (e.type == SDL_KEYDOWN && e.key.keysym.sym == SDLK_ESCAPE)
        running = false;
    }

    // Refresh the texture only when the VM published a new frame.
    const uint32_t seq = rt.FrameSeq(vm_index);
    if (seq != last_seq) {
      last_seq = seq;
      if (rt.ReadFramebuffer(vm_index, fb))
        SDL_UpdateTexture(tex, nullptr, fb.data(),
                          kVideoWidth * sizeof(uint32_t));
    }

    SDL_RenderClear(ren);
    SDL_RenderCopy(ren, tex, nullptr, nullptr);
    SDL_RenderPresent(ren);
    SDL_Delay(16);  // ~60 fps cap; keeps the display GPU responsive
  }

  rt.ShutdownAll();
  rt.Sync();
  SDL_DestroyTexture(tex);
  SDL_DestroyRenderer(ren);
  SDL_DestroyWindow(win);
  SDL_Quit();
  return 0;
}

}  // namespace wvm
