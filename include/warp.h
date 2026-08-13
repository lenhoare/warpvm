#ifndef WARP_H
#define WARP_H

/* Warp C v0.1.5 compiler-provided platform interface.
 * Addresses and sizes are in 32-bit VM words. Pixels are ARGB8888 words.
 * WARP is a predefined divergent int containing the current lane ID (0..31);
 * it is part of the language and intentionally has no declaration here. */
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

/* VM-wide mailbox operations. warp_try_recv returns nonzero on success and
 * writes payload plus (message_type << 16 | source_vm) metadata. */
void warp_send(unsigned destination, unsigned message_type, unsigned payload);
int warp_try_recv(unsigned *payload, unsigned *metadata);

/* Warp-wide collectives. Calls execute under uniform control; their values
 * may be lane-varying. shuffle_xor's mask must be a constant from 0 to 31. */
int warp_broadcast(int value, int lane);
int warp_shuffle(int value, int lane);
int warp_shuffle_xor(int value, int mask);
unsigned warp_ballot(int predicate);
int warp_any(int predicate);
int warp_all(int predicate);
int warp_reduce_add(int value);
unsigned warp_reduce_add_u(unsigned value);
int warp_reduce_min(int value);
int warp_reduce_max(int value);
unsigned warp_reduce_min_u(unsigned value);
unsigned warp_reduce_max_u(unsigned value);
unsigned warp_reduce_and(unsigned value);
unsigned warp_reduce_or(unsigned value);
unsigned warp_reduce_xor(unsigned value);

/* Type-generic integer compiler builtins. The result follows the usual
 * signed/unsigned arithmetic conversions of the two arguments. */
int min(int a, int b);
int max(int a, int b);

/* Compiler-provided Warp C source routines. Sizes and addresses are words. */
void warp_memcpy(unsigned *dst, unsigned *src, unsigned words);
void warp_memset(unsigned *dst, unsigned value, unsigned words);

#endif
