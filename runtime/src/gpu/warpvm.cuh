// WarpVM core constants and instruction encoding.
// Mirrors docs/isa.md — the Rust assembler carries its own copy of these
// values; keep the two in sync (round-trip tests police this).
#pragma once

#include <cstdint>

namespace wvm {

// ---- Machine geometry -------------------------------------------------
constexpr int kLanes = 32;        // one VM = one warp
constexpr int kVectorRegs = 16;   // r0-r15, one u32 per lane
constexpr int kScalarRegs = 8;    // s0-s7, uniform
constexpr int kPredRegs = 4;      // p0-p3, u32 lane masks
constexpr int kCallDepth = 8;
constexpr uint32_t kFullMask = 0xFFFFFFFFu;

constexpr uint32_t kMaxCodeWords = 4096;
constexpr uint32_t kMaxLiterals = 256;
constexpr uint32_t kMailboxSlots = 16;

// ---- Architectural display (docs/isa.md, v0.1.1) ----------------------
// Fixed memory-mapped framebuffer, word-addressed. Pixels are ordinary u32
// values 0xAARRGGBB accessed through LOAD/STORE.
constexpr uint32_t kVideoBaseWord = 0x00100000u;
constexpr uint32_t kVideoWidth = 128;
constexpr uint32_t kVideoHeight = 128;
constexpr uint32_t kVideoWords = kVideoWidth * kVideoHeight;  // 16384
constexpr uint32_t kVideoEndWord = kVideoBaseWord + kVideoWords;
constexpr uint32_t kVideoResetColor = 0xFF000000u;  // opaque black

// ---- Messaging (isa.md §4.7) -------------------------------------------
// Fixed-size 16-byte message. payload[1..2] are reserved (zero) in v0.1;
// SEND transmits payload[0] only.
struct Message {
  uint32_t header;      // src_vm in low 16 bits, msg_type in high 16 bits
  uint32_t payload[3];
};

// Per-VM inbound mailbox: a ring of kMailboxSlots messages. Producers claim a
// slot by atomically advancing head; the owner consumes from tail.
struct Mailbox {
  volatile uint32_t head;
  volatile uint32_t tail;
  Message slots[kMailboxSlots];
};

// ---- VM status (isa.md §6) ---------------------------------------------
enum Status : uint32_t {
  kIdle = 0,
  kRunning = 1,
  kPaused = 2,
  kHalted = 3,
  kFaulted = 4,
  kDebug = 5,
};

// ---- Fault codes (isa.md §5) -------------------------------------------
enum Fault : uint32_t {
  kFaultOk = 0,
  kFaultOpcode = 1,
  kFaultOperand = 2,
  kFaultJump = 3,
  kFaultMem = 4,
  kFaultStack = 5,
  kFaultMsg = 6,
  kFaultBudget = 7,
};

// ---- Instruction field layout (isa.md §2) -------------------------------
constexpr int kOpcodeShift = 25;
constexpr uint32_t kOpcodeMask = 0x7Fu;
constexpr int kGuardShift = 21;
constexpr uint32_t kGuardMask = 0xFu;
constexpr int kRdShift = 17;
constexpr int kRs1Shift = 13;
constexpr uint32_t kRegFieldMask = 0xFu;
constexpr uint32_t kLoMask = 0x1FFFu;
constexpr int kImmBits = 13;

constexpr uint32_t enc_r(uint32_t op, uint32_t guard, uint32_t rd,
                         uint32_t rs1, uint32_t rs2) {
  return (op << kOpcodeShift) | (guard << kGuardShift) | (rd << kRdShift) |
         (rs1 << kRs1Shift) | (rs2 & 0xFu);
}

constexpr uint32_t enc_i(uint32_t op, uint32_t guard, uint32_t rd,
                         uint32_t rs1, int32_t imm) {
  return (op << kOpcodeShift) | (guard << kGuardShift) | (rd << kRdShift) |
         (rs1 << kRs1Shift) | (static_cast<uint32_t>(imm) & kLoMask);
}

// ---- Opcodes (isa.md §4) ------------------------------------------------
enum Opcode : uint32_t {
  kNop = 0x00,

  kMov = 0x01, kAdd = 0x02, kSub = 0x03, kMul = 0x04, kDiv = 0x05,
  kMod = 0x06, kMin = 0x07, kMax = 0x08, kAnd = 0x09, kOr = 0x0A,
  kXor = 0x0B, kShl = 0x0C, kShr = 0x0D,

  kMovI = 0x10, kAddI = 0x11, kSubI = 0x12, kMulI = 0x13, kAndI = 0x14,
  kOrI = 0x15, kXorI = 0x16, kShlI = 0x17, kShrI = 0x18,
  kLdw = 0x19, kSBcast = 0x1A, kSGet = 0x1B,

  kAbs = 0x20, kNeg = 0x21, kNot = 0x22,

  kCmpEq = 0x28, kCmpNe = 0x29, kCmpLt = 0x2A, kCmpLe = 0x2B,
  kCmpGt = 0x2C, kCmpGe = 0x2D,
  kCmpEqI = 0x30, kCmpNeI = 0x31, kCmpLtI = 0x32, kCmpLeI = 0x33,
  kCmpGtI = 0x34, kCmpGeI = 0x35,

  kNotMask = 0x38, kAndMask = 0x39, kOrMask = 0x3A, kBallot = 0x3B,
  kAny = 0x3C, kAll = 0x3D,

  kLaneId = 0x40, kBroadcast = 0x41, kShuffle = 0x42, kShuffleXor = 0x43,
  kReduceAdd = 0x44, kReduceMin = 0x45, kReduceMax = 0x46,
  kReduceAnd = 0x47, kReduceOr = 0x48, kReduceXor = 0x49,
  kVmid = 0x4A, kClock = 0x4B, kRand = 0x4C,

  kLoad = 0x50, kStore = 0x51,

  kLog = 0x58, kLogI = 0x59, kFlip = 0x5A,

  kSend = 0x60, kTryRecv = 0x61,  // messaging (0x62-0x6F reserved)

  kJmp = 0x70, kJmpIfAny = 0x71, kJmpIfAll = 0x72, kCall = 0x73,
  kRet = 0x74, kHalt = 0x75, kYield = 0x76, kStepTrap = 0x77,

  kSMov = 0x78, kSMovI = 0x79, kSAdd = 0x7A, kSAddI = 0x7B, kSLdw = 0x7C,
  kSCmpLt = 0x7D, kSCmpLtI = 0x7E,
};

}  // namespace wvm
