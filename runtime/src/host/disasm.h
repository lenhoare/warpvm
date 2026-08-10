// Host-side single-word disassembler for the attach console.
// Mirrors tools/warpvm-asm/src/disasm.rs; falls back to `.word` for
// unrecognised / invalid encodings so output always re-assembles.
#pragma once

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "gpu/warpvm.cuh"

namespace wvm {

inline std::string GuardText(uint32_t guard) {
  if (guard == 0) return "";
  if (guard >= 1 && guard <= 4) return "@p" + std::to_string(guard - 1);
  if (guard >= 5 && guard <= 8) return "@!p" + std::to_string(guard - 5);
  return "";
}

inline int32_t DisasmSignExt13(uint32_t lo) {
  return (lo & 0x1000u) ? static_cast<int32_t>(lo | 0xFFFFE000u)
                        : static_cast<int32_t>(lo);
}

inline std::string DisasmWord(uint32_t word,
                              const std::vector<uint32_t>& literals) {
  const uint32_t op = (word >> kOpcodeShift) & kOpcodeMask;
  const uint32_t guard = (word >> kGuardShift) & kGuardMask;
  const uint32_t rd = (word >> kRdShift) & kRegFieldMask;
  const uint32_t rs1 = (word >> kRs1Shift) & kRegFieldMask;
  const uint32_t lo = word & kLoMask;
  const uint32_t rs2 = lo & 0xFu;
  const int32_t imm = DisasmSignExt13(lo);

  auto vreg = [](uint32_t n) { return "r" + std::to_string(n); };
  auto sreg = [](uint32_t n) { return "s" + std::to_string(n); };
  auto pred = [](uint32_t n) { return "p" + std::to_string(n); };
  auto sreg_ok = [](uint32_t n) { return n < 8; };
  auto pred_ok = [](uint32_t n) { return n < 4; };

  std::string body;
  switch (op) {
    case kNop: body = "NOP"; break;
    case kHalt: body = "HALT"; break;
    case kYield: body = "YIELD"; break;
    case kRet: body = "RET"; break;

    case kMov: body = "MOV " + vreg(rd) + ", " + vreg(rs1); break;
    case kMovI: body = "MOV_I " + vreg(rd) + ", #" + std::to_string(imm); break;

    case kAdd: case kSub: case kMul: case kDiv: case kMod:
    case kMin: case kMax: case kAnd: case kOr: case kXor:
    case kShl: case kShr: {
      const char* n = op == kAdd ? "ADD" : op == kSub ? "SUB" :
                      op == kMul ? "MUL" : op == kDiv ? "DIV" :
                      op == kMod ? "MOD" : op == kMin ? "MIN" :
                      op == kMax ? "MAX" : op == kAnd ? "AND" :
                      op == kOr ? "OR" : op == kXor ? "XOR" :
                      op == kShl ? "SHL" : "SHR";
      body = std::string(n) + " " + vreg(rd) + ", " + vreg(rs1) + ", " + vreg(rs2);
      break;
    }
    case kAddI: case kSubI: case kMulI: case kAndI: case kOrI:
    case kXorI: case kShlI: case kShrI: {
      const char* n = op == kAddI ? "ADD_I" : op == kSubI ? "SUB_I" :
                      op == kMulI ? "MUL_I" : op == kAndI ? "AND_I" :
                      op == kOrI ? "OR_I" : op == kXorI ? "XOR_I" :
                      op == kShlI ? "SHL_I" : "SHR_I";
      body = std::string(n) + " " + vreg(rd) + ", " + vreg(rs1) + ", #" + std::to_string(imm);
      break;
    }

    case kAbs: body = "ABS " + vreg(rd) + ", " + vreg(rs1); break;
    case kNeg: body = "NEG " + vreg(rd) + ", " + vreg(rs1); break;
    case kNot: body = "NOT " + vreg(rd) + ", " + vreg(rs1); break;

    case kCmpEq: case kCmpNe: case kCmpLt: case kCmpLe: case kCmpGt: case kCmpGe:
    case kCmpEqI: case kCmpNeI: case kCmpLtI: case kCmpLeI: case kCmpGtI: case kCmpGeI: {
      if (!pred_ok(rd)) return ".word 0x" + std::to_string(word);
      const char* n = (op == kCmpEq || op == kCmpEqI) ? "CMP_EQ" :
                      (op == kCmpNe || op == kCmpNeI) ? "CMP_NE" :
                      (op == kCmpLt || op == kCmpLtI) ? "CMP_LT" :
                      (op == kCmpLe || op == kCmpLeI) ? "CMP_LE" :
                      (op == kCmpGt || op == kCmpGtI) ? "CMP_GT" : "CMP_GE";
      const bool is_i = op >= kCmpEqI;
      body = std::string(n) + " " + pred(rd) + ", " + vreg(rs1) +
             (is_i ? ", #" + std::to_string(imm) : ", " + vreg(rs2));
      break;
    }

    case kNotMask:
      if (!pred_ok(rd) || !pred_ok(rs1)) return ".word 0x" + std::to_string(word);
      body = "NOTMASK " + pred(rd) + ", " + pred(rs1); break;
    case kAndMask: case kOrMask:
      if (!pred_ok(rd) || !pred_ok(rs1) || !pred_ok(rs2))
        return ".word 0x" + std::to_string(word);
      body = std::string(op == kAndMask ? "ANDMASK " : "ORMASK ") +
             pred(rd) + ", " + pred(rs1) + ", " + pred(rs2);
      break;
    case kBallot:
      if (!pred_ok(rd)) return ".word 0x" + std::to_string(word);
      body = "BALLOT " + pred(rd) + ", " + vreg(rs1); break;
    case kAny: case kAll:
      if (!pred_ok(rd) || !pred_ok(rs1)) return ".word 0x" + std::to_string(word);
      body = std::string(op == kAny ? "ANY " : "ALL ") + pred(rd) + ", " + pred(rs1);
      break;

    case kLaneId: body = "LANEID " + vreg(rd); break;
    case kVmid: body = "VMID " + vreg(rd); break;
    case kClock: body = "CLOCK " + vreg(rd); break;
    case kRand: body = "RAND " + vreg(rd); break;
    case kBroadcast:
      body = "BROADCAST " + vreg(rd) + ", " + vreg(rs1) + ", #" + std::to_string(lo); break;
    case kShuffleXor:
      body = "SHUFFLE_XOR " + vreg(rd) + ", " + vreg(rs1) + ", #" + std::to_string(lo); break;
    case kShuffle:
      body = "SHUFFLE " + vreg(rd) + ", " + vreg(rs1) + ", " + vreg(rs2); break;

    case kReduceAdd: case kReduceMin: case kReduceMax:
    case kReduceAnd: case kReduceOr: case kReduceXor: {
      const char* n = op == kReduceAdd ? "REDUCE_ADD" : op == kReduceMin ? "REDUCE_MIN" :
                      op == kReduceMax ? "REDUCE_MAX" : op == kReduceAnd ? "REDUCE_AND" :
                      op == kReduceOr ? "REDUCE_OR" : "REDUCE_XOR";
      body = std::string(n) + " " + vreg(rd) + ", " + vreg(rs1); break;
    }

    case kLoad: body = "LOAD " + vreg(rd) + ", " + vreg(rs1); break;
    case kStore: body = "STORE " + vreg(rd) + ", " + vreg(rs1); break;

    case kLdw: {
      std::string note;
      if (lo < literals.size()) note = " ; lit=" + std::to_string(literals[lo]);
      body = "LDW " + vreg(rd) + ", #" + std::to_string(lo) + note; break;
    }
    case kLog: body = "LOG " + vreg(rs1) + ", " + vreg(rs2); break;
    case kLogI: body = "LOG_I " + vreg(rs1) + ", #" + std::to_string(imm); break;
    case kFlip: body = "FLIP"; break;

    case kSMov:
      if (!sreg_ok(rd) || !sreg_ok(rs1)) return ".word 0x" + std::to_string(word);
      body = "S_MOV " + sreg(rd) + ", " + sreg(rs1); break;
    case kSMovI:
      if (!sreg_ok(rd)) return ".word 0x" + std::to_string(word);
      body = "S_MOV_I " + sreg(rd) + ", #" + std::to_string(imm); break;
    case kSAdd:
      if (!sreg_ok(rd) || !sreg_ok(rs1) || !sreg_ok(rs2)) return ".word 0x" + std::to_string(word);
      body = "S_ADD " + sreg(rd) + ", " + sreg(rs1) + ", " + sreg(rs2); break;
    case kSAddI:
      if (!sreg_ok(rd) || !sreg_ok(rs1)) return ".word 0x" + std::to_string(word);
      body = "S_ADD_I " + sreg(rd) + ", " + sreg(rs1) + ", #" + std::to_string(imm); break;
    case kSLdw:
      if (!sreg_ok(rd)) return ".word 0x" + std::to_string(word);
      body = "S_LDW " + sreg(rd) + ", #" + std::to_string(lo); break;
    case kSCmpLt:
      if (!pred_ok(rd) || !sreg_ok(rs1) || !sreg_ok(rs2)) return ".word 0x" + std::to_string(word);
      body = "S_CMP_LT " + pred(rd) + ", " + sreg(rs1) + ", " + sreg(rs2); break;
    case kSCmpLtI:
      if (!pred_ok(rd) || !sreg_ok(rs1)) return ".word 0x" + std::to_string(word);
      body = "S_CMP_LT_I " + pred(rd) + ", " + sreg(rs1) + ", #" + std::to_string(imm); break;
    case kSBcast:
      if (!sreg_ok(rs1)) return ".word 0x" + std::to_string(word);
      body = "S_BCAST " + vreg(rd) + ", " + sreg(rs1); break;
    case kSGet:
      if (!sreg_ok(rd)) return ".word 0x" + std::to_string(word);
      body = "S_GET " + sreg(rd) + ", " + vreg(rs1); break;

    case kJmp: body = "JMP #" + std::to_string(imm); break;
    case kCall: body = "CALL #" + std::to_string(imm); break;
    case kJmpIfAny: case kJmpIfAll: {
      if (guard < 1 || guard > 4) return ".word 0x" + std::to_string(word);
      body = std::string(op == kJmpIfAny ? "JMP_IF_ANY " : "JMP_IF_ALL ") +
             pred(guard - 1) + ", #" + std::to_string(imm);
      break;
    }

    default: {
      char buf[16];
      std::snprintf(buf, sizeof(buf), "0x%08x", word);
      return ".word " + std::string(buf);
    }
  }

  const std::string g = GuardText(guard);
  const bool suppress = (op == kJmpIfAny || op == kJmpIfAll);
  if (g.empty() || suppress) return body;
  return g + " " + body;
}

}  // namespace wvm
