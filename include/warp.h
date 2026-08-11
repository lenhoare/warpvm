#ifndef WARP_H
#define WARP_H

/* Warp C v0.1.4 compiler-provided graphics interface.
 * Addresses and sizes are in 32-bit VM words. Pixels are ARGB8888 words. */
#define WARP_VIDEO_WIDTH 128
#define WARP_VIDEO_HEIGHT 128
#define WARP_VIDEO_WORDS 16384
#define WARP_VIDEO_BASE 0x00100000u

unsigned *warp_framebuffer(void);
void warp_flip(void);
int warp_lane_id(void);
unsigned warp_vm_id(void);
unsigned warp_argb(unsigned a, unsigned r, unsigned g, unsigned b);
void warp_set_pixel(int x, int y, unsigned colour);

#endif
