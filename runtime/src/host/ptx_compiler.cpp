#include "host/ptx_compiler.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <string>

namespace wvm {
namespace {

std::string DriverError(CUresult result) {
  const char* name = nullptr;
  const char* description = nullptr;
  cuGetErrorName(result, &name);
  cuGetErrorString(result, &description);
  std::string out = name != nullptr ? name : "CUDA_ERROR_UNKNOWN";
  if (description != nullptr) out += ": " + std::string(description);
  return out;
}

uint32_t SignExt13(uint32_t value) {
  return (value & 0x1000u) != 0 ? value | 0xFFFFE000u : value;
}

std::string VReg(uint32_t index) { return "%v" + std::to_string(index); }

bool EmitPtx(const WvmFile& file, std::string& ptx, std::string& err) {
  if (file.code.empty()) {
    err = "cannot compile an empty program";
    return false;
  }

  std::ostringstream body;
  bool found_safe_exit = false;
  for (uint32_t pc = 0; pc < file.code.size(); ++pc) {
    const uint32_t instruction = file.code[pc];
    const uint32_t op =
        (instruction >> kOpcodeShift) & kOpcodeMask;
    const uint32_t guard =
        (instruction >> kGuardShift) & kGuardMask;
    const uint32_t rd =
        (instruction >> kRdShift) & kRegFieldMask;
    const uint32_t rs1 =
        (instruction >> kRs1Shift) & kRegFieldMask;
    const uint32_t lo = instruction & kLoMask;
    const uint32_t rs2 = lo & kRegFieldMask;
    const uint32_t imm = SignExt13(lo);

    auto reject = [&](const char* reason) {
      std::ostringstream message;
      message << "pc " << pc << ": " << reason << " (opcode 0x"
              << std::hex << std::setw(2) << std::setfill('0') << op << ')';
      err = message.str();
      return false;
    };
    if (guard > 8) return reject("invalid guard field");

    body << "L_pc_" << pc << ":\n"
         << "    // opcode 0x" << std::hex << op << std::dec << "\n"
         << "    add.u64 %ic, %ic, 1;\n";
    if (guard == 0) {
      body << "    mov.u32 %t9, 0xffffffff;\n";
    } else {
      const uint32_t pred = (guard - 1u) & 3u;
      if (guard < 5)
        body << "    mov.u32 %t9, %m" << pred << ";\n";
      else
        body << "    not.b32 %t9, %m" << pred << ";\n";
    }
    body << "    bfe.u32 %t8, %t9, %t3, 1;\n"
         << "    setp.ne.u32 %p2, %t8, 0;\n";
    const std::string predicated = "    @%p2 ";
    switch (op) {
      case kNop:
        break;
      case kMov:
        body << predicated << "mov.b32 " << VReg(rd) << ", "
             << VReg(rs1) << ";\n";
        break;
      case kAdd:
        body << predicated << "add.u32 " << VReg(rd) << ", " << VReg(rs1) << ", "
             << VReg(rs2) << ";\n";
        break;
      case kSub:
        body << predicated << "sub.u32 " << VReg(rd) << ", " << VReg(rs1) << ", "
             << VReg(rs2) << ";\n";
        break;
      case kMul:
        body << predicated << "mul.lo.u32 " << VReg(rd) << ", " << VReg(rs1) << ", "
             << VReg(rs2) << ";\n";
        break;
      case kDiv: case kMod:
        body << "    setp.ne.u32 %p3, " << VReg(rs2) << ", 0;\n"
             << "    and.pred %p3, %p3, %p2;\n"
             << "    mov.u32 %t8, 0;\n"
             << "    @%p3 " << (op == kDiv ? "div" : "rem") << ".u32 %t8, "
             << VReg(rs1) << ", " << VReg(rs2) << ";\n"
             << predicated << "mov.b32 " << VReg(rd) << ", %t8;\n";
        break;
      case kMin:
        body << predicated << "min.u32 " << VReg(rd) << ", " << VReg(rs1) << ", "
             << VReg(rs2) << ";\n";
        break;
      case kMax:
        body << predicated << "max.u32 " << VReg(rd) << ", " << VReg(rs1) << ", "
             << VReg(rs2) << ";\n";
        break;
      case kAnd:
        body << predicated << "and.b32 " << VReg(rd) << ", " << VReg(rs1) << ", "
             << VReg(rs2) << ";\n";
        break;
      case kOr:
        body << predicated << "or.b32 " << VReg(rd) << ", " << VReg(rs1) << ", "
             << VReg(rs2) << ";\n";
        break;
      case kXor:
        body << predicated << "xor.b32 " << VReg(rd) << ", " << VReg(rs1) << ", "
             << VReg(rs2) << ";\n";
        break;
      case kShl:
        body << "    and.b32 %t8, " << VReg(rs2) << ", 31;\n"
             << predicated << "shl.b32 " << VReg(rd) << ", " << VReg(rs1)
             << ", %t8;\n";
        break;
      case kShr:
        body << "    and.b32 %t8, " << VReg(rs2) << ", 31;\n"
             << predicated << "shr.u32 " << VReg(rd) << ", " << VReg(rs1)
             << ", %t8;\n";
        break;
      case kMovI:
        body << predicated << "mov.u32 " << VReg(rd) << ", " << imm << ";\n";
        break;
      case kAddI:
        body << predicated << "add.u32 " << VReg(rd) << ", " << VReg(rs1) << ", "
             << imm << ";\n";
        break;
      case kSubI:
        body << predicated << "sub.u32 " << VReg(rd) << ", " << VReg(rs1) << ", "
             << imm << ";\n";
        break;
      case kMulI:
        body << predicated << "mul.lo.u32 " << VReg(rd) << ", " << VReg(rs1) << ", "
             << imm << ";\n";
        break;
      case kAndI:
        body << predicated << "and.b32 " << VReg(rd) << ", " << VReg(rs1) << ", "
             << imm << ";\n";
        break;
      case kOrI:
        body << predicated << "or.b32 " << VReg(rd) << ", " << VReg(rs1) << ", "
             << imm << ";\n";
        break;
      case kXorI:
        body << predicated << "xor.b32 " << VReg(rd) << ", " << VReg(rs1) << ", "
             << imm << ";\n";
        break;
      case kShlI:
        body << predicated << "shl.b32 " << VReg(rd) << ", " << VReg(rs1) << ", "
             << (lo & 31u) << ";\n";
        break;
      case kShrI:
        body << predicated << "shr.u32 " << VReg(rd) << ", " << VReg(rs1) << ", "
             << (lo & 31u) << ";\n";
        break;
      case kLdw:
        if (lo >= file.literals.size())
          return reject("literal index is out of range");
        body << predicated << "mov.u32 " << VReg(rd) << ", "
             << file.literals[lo] << ";\n";
        break;
      case kNeg:
        body << predicated << "sub.u32 " << VReg(rd) << ", 0, " << VReg(rs1)
             << ";\n";
        break;
      case kAbs:
        body << predicated << "abs.s32 " << VReg(rd) << ", " << VReg(rs1)
             << ";\n";
        break;
      case kNot:
        body << predicated << "not.b32 " << VReg(rd) << ", " << VReg(rs1) << ";\n";
        break;
      case kCmpEq: case kCmpNe: case kCmpLt: case kCmpLe:
      case kCmpGt: case kCmpGe:
      case kCmpEqI: case kCmpNeI: case kCmpLtI: case kCmpLeI:
      case kCmpGtI: case kCmpGeI: {
        if (rd >= kPredRegs) return reject("invalid predicate destination");
        const bool immediate = op >= kCmpEqI;
        const char* relation = "eq";
        switch (op) {
          case kCmpEq: case kCmpEqI: relation = "eq"; break;
          case kCmpNe: case kCmpNeI: relation = "ne"; break;
          case kCmpLt: case kCmpLtI: relation = "lt"; break;
          case kCmpLe: case kCmpLeI: relation = "le"; break;
          case kCmpGt: case kCmpGtI: relation = "gt"; break;
          default: relation = "ge"; break;
        }
        body << "    setp." << relation << ".u32 %p3, " << VReg(rs1)
             << ", " << (immediate ? std::to_string(imm) : VReg(rs2))
             << ";\n"
             << "    and.pred %p3, %p3, %p2;\n"
             << "    vote.sync.ballot.b32 %t8, %p3, 0xffffffff;\n"
             << "    not.b32 %t4, %t9;\n"
             << "    and.b32 %m" << rd << ", %m" << rd << ", %t4;\n"
             << "    and.b32 %t8, %t8, %t9;\n"
             << "    or.b32 %m" << rd << ", %m" << rd << ", %t8;\n";
        break;
      }
      case kNotMask: {
        if (rd >= kPredRegs || rs1 >= kPredRegs)
          return reject("invalid predicate register");
        body << "    not.b32 %t8, %m" << rs1 << ";\n"
             << "    not.b32 %t4, %t9;\n"
             << "    and.b32 %m" << rd << ", %m" << rd << ", %t4;\n"
             << "    and.b32 %t8, %t8, %t9;\n"
             << "    or.b32 %m" << rd << ", %m" << rd << ", %t8;\n";
        break;
      }
      case kAndMask: case kOrMask: {
        if (rd >= kPredRegs || rs1 >= kPredRegs || rs2 >= kPredRegs)
          return reject("invalid predicate register");
        body << "    " << (op == kAndMask ? "and" : "or")
             << ".b32 %t8, %m" << rs1 << ", %m" << rs2 << ";\n"
             << "    not.b32 %t4, %t9;\n"
             << "    and.b32 %m" << rd << ", %m" << rd << ", %t4;\n"
             << "    and.b32 %t8, %t8, %t9;\n"
             << "    or.b32 %m" << rd << ", %m" << rd << ", %t8;\n";
        break;
      }
      case kBallot:
        if (rd >= kPredRegs) return reject("invalid ballot destination");
        body << "    setp.ne.u32 %p3, " << VReg(rs1) << ", 0;\n"
             << "    and.pred %p3, %p3, %p2;\n"
             << "    vote.sync.ballot.b32 %t8, %p3, 0xffffffff;\n"
             << "    not.b32 %t4, %t9;\n"
             << "    and.b32 %m" << rd << ", %m" << rd << ", %t4;\n"
             << "    and.b32 %t8, %t8, %t9;\n"
             << "    or.b32 %m" << rd << ", %m" << rd << ", %t8;\n";
        break;
      case kLaneId:
        body << predicated << "mov.u32 " << VReg(rd) << ", %t3;\n";
        break;
      case kShuffle:
        body << "    and.b32 %t8, " << VReg(rs2) << ", 31;\n"
             << "    shfl.sync.idx.b32 %t7, " << VReg(rs1)
             << ", %t8, 0x1f, 0xffffffff;\n"
             << predicated << "mov.b32 " << VReg(rd) << ", %t7;\n";
        break;
      case kShuffleXor:
        body << "    shfl.sync.bfly.b32 %t7, " << VReg(rs1) << ", "
             << (lo & 31u) << ", 0x1f, 0xffffffff;\n"
             << predicated << "mov.b32 " << VReg(rd) << ", %t7;\n";
        break;
      case kVmid:
        body << predicated << "mov.u32 " << VReg(rd) << ", %vid;\n";
        break;
      case kReduceOr:
        body << "    mov.b32 %t8, " << VReg(rs1) << ";\n";
        for (uint32_t delta : {16u, 8u, 4u, 2u, 1u}) {
          body << "    shfl.sync.bfly.b32 %t7, %t8, " << delta
               << ", 0x1f, 0xffffffff;\n"
               << "    or.b32 %t8, %t8, %t7;\n";
        }
        body << predicated << "mov.b32 " << VReg(rd) << ", %t8;\n";
        break;
      case kLoad: {
        body << "    setp.lt.u32 %p3, " << VReg(rs1) << ", %t5;\n"
             << "    and.pred %p3, %p3, %p2;\n"
             << "    @%p3 mul.wide.u32 %rd11, " << VReg(rs1) << ", 4;\n"
             << "    @%p3 add.u64 %rd11, %rd7, %rd11;\n"
             << "    @%p3 ld.global.u32 " << VReg(rd) << ", [%rd11];\n"
             << "    setp.ge.u32 %p4, " << VReg(rs1) << ", "
             << kVideoBaseWord << ";\n"
             << "    setp.lt.u32 %p5, " << VReg(rs1) << ", "
             << kVideoEndWord << ";\n"
             << "    and.pred %p4, %p4, %p5;\n"
             << "    setp.ne.u64 %p6, %rd8, 0;\n"
             << "    and.pred %p4, %p4, %p6;\n"
             << "    and.pred %p4, %p4, %p2;\n"
             << "    @%p4 sub.u32 %t8, " << VReg(rs1) << ", "
             << kVideoBaseWord << ";\n"
             << "    @%p4 mul.wide.u32 %rd11, %t8, 4;\n"
             << "    @%p4 add.u64 %rd11, %rd9, %rd11;\n"
             << "    @%p4 ld.global.u32 " << VReg(rd) << ", [%rd11];\n"
             << "    or.pred %p5, %p3, %p4;\n"
             << "    not.pred %p5, %p5;\n"
             << "    and.pred %p5, %p5, %p2;\n"
             << "    vote.sync.ballot.b32 %t8, %p5, 0xffffffff;\n"
             << "    setp.ne.u32 %p6, %t8, 0;\n"
             << "    @%p6 sub.u64 %ic, %ic, 1;\n"
             << "    @%p6 mov.u32 %t6, " << kFaulted << ";\n"
             << "    @%p6 mov.u32 %t7, " << pc << ";\n"
             << "    @%p6 mov.u32 %t4, " << kFaultMem << ";\n"
             << "    @%p6 bra L_save_fault;\n";
        break;
      }
      case kStore: {
        body << "    setp.lt.u32 %p3, " << VReg(rd) << ", %t5;\n"
             << "    and.pred %p3, %p3, %p2;\n"
             << "    @%p3 mul.wide.u32 %rd11, " << VReg(rd) << ", 4;\n"
             << "    @%p3 add.u64 %rd11, %rd7, %rd11;\n"
             << "    @%p3 st.global.u32 [%rd11], " << VReg(rs1) << ";\n"
             << "    setp.ge.u32 %p4, " << VReg(rd) << ", "
             << kVideoBaseWord << ";\n"
             << "    setp.lt.u32 %p5, " << VReg(rd) << ", "
             << kVideoEndWord << ";\n"
             << "    and.pred %p4, %p4, %p5;\n"
             << "    setp.ne.u64 %p6, %rd8, 0;\n"
             << "    and.pred %p4, %p4, %p6;\n"
             << "    and.pred %p4, %p4, %p2;\n"
             << "    @%p4 sub.u32 %t8, " << VReg(rd) << ", "
             << kVideoBaseWord << ";\n"
             << "    @%p4 mul.wide.u32 %rd11, %t8, 4;\n"
             << "    @%p4 add.u64 %rd11, %rd9, %rd11;\n"
             << "    @%p4 st.global.u32 [%rd11], " << VReg(rs1) << ";\n"
             // A WarpVM STORE retires before the following virtual
             // instruction for every lane. The warp barrier supplies the
             // cross-lane visibility that ordinary per-thread CUDA ordering
             // does not guarantee.
             << "    bar.warp.sync 0xffffffff;\n"
             << "    or.pred %p5, %p3, %p4;\n"
             << "    not.pred %p5, %p5;\n"
             << "    and.pred %p5, %p5, %p2;\n"
             << "    vote.sync.ballot.b32 %t8, %p5, 0xffffffff;\n"
             << "    setp.ne.u32 %p6, %t8, 0;\n"
             << "    @%p6 sub.u64 %ic, %ic, 1;\n"
             << "    @%p6 mov.u32 %t6, " << kFaulted << ";\n"
             << "    @%p6 mov.u32 %t7, " << pc << ";\n"
             << "    @%p6 mov.u32 %t4, " << kFaultMem << ";\n"
             << "    @%p6 bra L_save_fault;\n";
        break;
      }
      case kSMov:
        if (guard != 0 || rd >= kScalarRegs || rs1 >= kScalarRegs)
          return reject("invalid or guarded scalar move");
        body << "    mov.b32 %s" << rd << ", %s" << rs1 << ";\n";
        break;
      case kSMovI:
        if (guard != 0 || rd >= kScalarRegs)
          return reject("invalid or guarded scalar immediate move");
        body << "    mov.u32 %s" << rd << ", " << imm << ";\n";
        break;
      case kSAdd:
        if (guard != 0 || rd >= kScalarRegs || rs1 >= kScalarRegs ||
            rs2 >= kScalarRegs)
          return reject("invalid or guarded scalar add");
        body << "    add.u32 %s" << rd << ", %s" << rs1 << ", %s"
             << rs2 << ";\n";
        break;
      case kSAddI:
        if (guard != 0 || rd >= kScalarRegs || rs1 >= kScalarRegs)
          return reject("invalid or guarded scalar immediate add");
        body << "    add.u32 %s" << rd << ", %s" << rs1 << ", "
             << imm << ";\n";
        break;
      case kSLdw:
        if (guard != 0 || rd >= kScalarRegs || lo >= file.literals.size())
          return reject("invalid scalar literal load");
        body << "    mov.u32 %s" << rd << ", " << file.literals[lo]
             << ";\n";
        break;
      case kSCmpLt: case kSCmpLtI: {
        if (guard != 0 || rd >= kPredRegs || rs1 >= kScalarRegs ||
            (op == kSCmpLt && rs2 >= kScalarRegs))
          return reject("invalid or guarded scalar comparison");
        body << "    setp.lt.u32 %p3, %s" << rs1 << ", "
             << (op == kSCmpLtI ? std::to_string(imm)
                                : ("%s" + std::to_string(rs2)))
             << ";\n"
             << "    selp.u32 %m" << rd
             << ", 0xffffffff, 0, %p3;\n";
        break;
      }
      case kSBcast:
        if (rs1 >= kScalarRegs) return reject("invalid scalar broadcast source");
        body << predicated << "mov.b32 " << VReg(rd) << ", %s" << rs1
             << ";\n";
        break;
      case kFlip:
        if (guard != 0) return reject("FLIP cannot be guarded");
        body << "    setp.eq.u32 %p3, %t3, 0;\n"
             << "    setp.ne.u64 %p4, %rd10, 0;\n"
             << "    and.pred %p3, %p3, %p4;\n"
             << "    @%p3 mul.wide.u32 %rd11, %t2, 4;\n"
             << "    @%p3 add.u64 %rd11, %rd10, %rd11;\n"
             << "    @%p3 ld.global.u32 %t8, [%rd11];\n"
             << "    @%p3 add.u32 %t8, %t8, 1;\n"
             << "    @%p3 st.global.u32 [%rd11], %t8;\n";
        break;
      case kJmp:
        if (lo >= file.code.size()) return reject("jump target is out of range");
        body << "    bra.uni L_pc_" << lo << ";\n";
        break;
      case kJmpIfAny: case kJmpIfAll: {
        if (lo >= file.code.size()) return reject("jump target is out of range");
        if (guard == 0 || guard > 8)
          return reject("conditional branch requires p0..p3 or !p0..!p3");
        const char* relation = op == kJmpIfAny ? "ne" : "eq";
        const uint32_t value = op == kJmpIfAny ? 0u : kFullMask;
        body << "    setp." << relation << ".u32 %p1, %t9, "
             << value << ";\n"
             << "    @%p1 bra L_pc_" << lo << ";\n";
        break;
      }
      case kCall:
        if (guard != 0) return reject("CALL cannot be guarded");
        if (lo >= file.code.size()) return reject("call target is out of range");
        body << "    setp.ge.u32 %p3, %cd, " << kCallDepth << ";\n"
             << "    @%p3 sub.u64 %ic, %ic, 1;\n"
             << "    @%p3 mov.u32 %t6, " << kFaulted << ";\n"
             << "    @%p3 mov.u32 %t7, " << pc << ";\n"
             << "    @%p3 mov.u32 %t4, " << kFaultStack << ";\n"
             << "    @%p3 bra L_save_fault;\n";
        for (uint32_t depth = 0; depth < kCallDepth; ++depth) {
          body << "    setp.eq.u32 %p3, %cd, " << depth << ";\n"
               << "    @%p3 mov.u32 %cs" << depth << ", " << (pc + 1)
               << ";\n";
        }
        body << "    add.u32 %cd, %cd, 1;\n"
             << "    bra.uni L_pc_" << lo << ";\n";
        break;
      case kRet:
        if (guard != 0) return reject("RET cannot be guarded");
        body << "    setp.eq.u32 %p3, %cd, 0;\n"
             << "    @%p3 sub.u64 %ic, %ic, 1;\n"
             << "    @%p3 mov.u32 %t6, " << kFaulted << ";\n"
             << "    @%p3 mov.u32 %t7, " << pc << ";\n"
             << "    @%p3 mov.u32 %t4, " << kFaultStack << ";\n"
             << "    @%p3 bra L_save_fault;\n"
             << "    sub.u32 %cd, %cd, 1;\n"
             << "    mov.u32 %t8, 0;\n";
        for (uint32_t depth = 0; depth < kCallDepth; ++depth) {
          body << "    setp.eq.u32 %p3, %cd, " << depth << ";\n"
               << "    @%p3 mov.u32 %t8, %cs" << depth << ";\n";
        }
        body << "    mov.u32 %t7, %t8;\n"
             << "    bra.uni L_dispatch;\n";
        break;
      case kHalt:
        found_safe_exit = true;
        body << "    mov.u32 %t6, " << kHalted << ";\n"
             << "    mov.u32 %t7, " << pc << ";\n"
             << "    bra.uni L_save;\n";
        break;
      case kYield:
        found_safe_exit = true;
        body << "    mov.u32 %t6, " << kPaused << ";\n"
             << "    mov.u32 %t7, " << (pc + 1) << ";\n"
             << "    bra.uni L_save;\n";
        break;
      default:
        return reject("unsupported by the minimal PTX backend");
    }
  }
  if (!found_safe_exit) {
    err = "compiled programs require at least one HALT or YIELD safe point";
    return false;
  }

  std::ostringstream out;
  out << ".version 7.0\n"
      << ".target sm_70\n"
      << ".address_size 64\n\n"
      << ".visible .entry warpvm_compiled(\n"
      << "    .param .u64 states_param,\n"
      << "    .param .u32 num_vms_param,\n"
      << "    .param .u64 memory_param,\n"
      << "    .param .u32 memory_words_param,\n"
      << "    .param .u64 framebuffer_param,\n"
      << "    .param .u64 frame_seq_param\n"
      << ")\n"
      << ".maxntid 256, 1, 1\n"
      << "{\n"
      << "    .reg .pred %p<8>;\n"
      << "    .reg .b32 %t<10>;\n"
      << "    .reg .b32 %v<16>;\n"
      << "    .reg .b32 %m<4>;\n"
      << "    .reg .b32 %s<8>;\n"
      << "    .reg .b32 %cs<8>;\n"
      << "    .reg .b32 %cd;\n"
      << "    .reg .b32 %vid;\n"
      << "    .reg .b64 %rd<14>;\n"
      << "    .reg .b64 %ic;\n\n"
      << "    ld.param.u64 %rd0, [states_param];\n"
      << "    ld.param.u32 %t0, [num_vms_param];\n"
      << "    ld.param.u64 %rd6, [memory_param];\n"
      << "    ld.param.u32 %t5, [memory_words_param];\n"
      << "    ld.param.u64 %rd8, [framebuffer_param];\n"
      << "    ld.param.u64 %rd10, [frame_seq_param];\n"
      << "    mov.u32 %t1, %tid.x;\n"
      << "    mov.u32 %t4, %ctaid.x;\n"
      << "    mov.u32 %t2, %ntid.x;\n"
      << "    mad.lo.u32 %t1, %t4, %t2, %t1;\n"
      << "    shr.u32 %t2, %t1, 5;\n"
      << "    setp.ge.u32 %p0, %t2, %t0;\n"
      << "    @%p0 ret;\n"
      << "    and.b32 %t3, %t1, 31;\n"
      << "    mul.wide.u32 %rd1, %t2, " << sizeof(VmState) << ";\n"
      << "    add.u64 %rd2, %rd0, %rd1;\n"
      << "    ld.global.u32 %vid, [%rd2+" << offsetof(VmState, vm_id)
      << "];\n"
      << "    mul.wide.u32 %rd3, %t3, 4;\n\n";

  out << "    mul.lo.u32 %t6, %t2, %t5;\n"
      << "    mul.wide.u32 %rd5, %t6, 4;\n"
      << "    add.u64 %rd7, %rd6, %rd5;\n"
      << "    mul.wide.u32 %rd5, %t2, " << (kVideoWords * sizeof(uint32_t))
      << ";\n"
      << "    add.u64 %rd9, %rd8, %rd5;\n\n";

  for (uint32_t reg = 0; reg < kVectorRegs; ++reg) {
    const size_t offset = offsetof(VmState, vregs) +
                          reg * kLanes * sizeof(uint32_t);
    out << "    add.u64 %rd4, %rd2, " << offset << ";\n"
        << "    add.u64 %rd4, %rd4, %rd3;\n"
        << "    ld.global.u32 " << VReg(reg) << ", [%rd4];\n";
  }
  for (uint32_t pred = 0; pred < kPredRegs; ++pred) {
    const size_t offset = offsetof(VmState, preds) +
                          pred * sizeof(uint32_t);
    out << "    ld.global.u32 %m" << pred << ", [%rd2+" << offset
        << "];\n";
  }
  for (uint32_t scalar = 0; scalar < kScalarRegs; ++scalar) {
    const size_t offset = offsetof(VmState, sregs) +
                          scalar * sizeof(uint32_t);
    out << "    ld.global.u32 %s" << scalar << ", [%rd2+" << offset
        << "];\n";
  }
  for (uint32_t depth = 0; depth < kCallDepth; ++depth) {
    const size_t offset = offsetof(VmState, call_stack) + depth * sizeof(uint32_t);
    out << "    ld.global.u32 %cs" << depth << ", [%rd2+" << offset
        << "];\n";
  }
  out << "    ld.global.u32 %cd, [%rd2+" << offsetof(VmState, call_depth)
      << "];\n";
  out << "    ld.global.u64 %ic, [%rd2+"
      << offsetof(VmState, instruction_counter) << "];\n";
  out << "    ld.global.u32 %t7, [%rd2+" << offsetof(VmState, pc)
      << "];\n";
  out << "L_dispatch:\n";
  for (uint32_t pc = 0; pc < file.code.size(); ++pc) {
    out << "    setp.eq.u32 %p1, %t7, " << pc << ";\n"
        << "    @%p1 bra L_pc_" << pc << ";\n";
  }
  out << "    bra.uni L_fault_jump;\n\n"
      << body.str()
      << "\n    bra.uni L_fault_jump;\n\n"
      << "L_fault_jump:\n"
      << "    mov.u32 %t6, " << kFaulted << ";\n"
      << "    mov.u32 %t4, " << kFaultJump << ";\n"
      << "    bra.uni L_save_fault;\n\n"
      << "L_save:\n"
      << "    mov.u32 %t4, 0;\n"
      << "L_save_fault:\n";
  for (uint32_t reg = 0; reg < kVectorRegs; ++reg) {
    const size_t offset = offsetof(VmState, vregs) +
                          reg * kLanes * sizeof(uint32_t);
    out << "    add.u64 %rd4, %rd2, " << offset << ";\n"
        << "    add.u64 %rd4, %rd4, %rd3;\n"
        << "    st.global.u32 [%rd4], " << VReg(reg) << ";\n";
  }
  out << "\n    setp.eq.u32 %p1, %t3, 0;\n";
  for (uint32_t pred = 0; pred < kPredRegs; ++pred) {
    const size_t offset = offsetof(VmState, preds) +
                          pred * sizeof(uint32_t);
    out << "    @%p1 st.global.u32 [%rd2+" << offset << "], %m"
        << pred << ";\n";
  }
  for (uint32_t scalar = 0; scalar < kScalarRegs; ++scalar) {
    const size_t offset = offsetof(VmState, sregs) +
                          scalar * sizeof(uint32_t);
    out << "    @%p1 st.global.u32 [%rd2+" << offset << "], %s"
        << scalar << ";\n";
  }
  for (uint32_t depth = 0; depth < kCallDepth; ++depth) {
    const size_t offset = offsetof(VmState, call_stack) + depth * sizeof(uint32_t);
    out << "    @%p1 st.global.u32 [%rd2+" << offset << "], %cs" << depth
        << ";\n";
  }
  out << "    @%p1 st.global.u32 [%rd2+" << offsetof(VmState, call_depth)
      << "], %cd;\n";

  out << "    @%p1 st.global.u32 [%rd2+" << offsetof(VmState, vm_id)
      << "], %vid;\n"
      << "    @%p1 st.global.u32 [%rd2+" << offsetof(VmState, status)
      << "], %t6;\n"
      << "    @%p1 st.global.u32 [%rd2+" << offsetof(VmState, pc)
      << "], %t7;\n"
      << "    @%p1 st.global.u32 [%rd2+" << offsetof(VmState, fault_code)
      << "], %t4;\n"
      << "    @%p1 st.global.u64 [%rd2+"
      << offsetof(VmState, instruction_counter) << "], %ic;\n"
      << "    ret;\n"
      << "}\n";
  ptx = out.str();
  return true;
}

}  // namespace

bool TranslateWvmToPtx(const WvmFile& file, std::string& ptx,
                       std::string& err) {
  return EmitPtx(file, ptx, err);
}

PtxCompiledProgram::~PtxCompiledProgram() {
  if (context_ != nullptr) cuCtxSetCurrent(context_);
  if (scratch_frame_seq_ != 0) cuMemFree(scratch_frame_seq_);
  if (scratch_framebuffers_ != 0) cuMemFree(scratch_framebuffers_);
  if (scratch_memory_ != 0) cuMemFree(scratch_memory_);
  if (scratch_states_ != 0) cuMemFree(scratch_states_);
  if (module_ != nullptr) cuModuleUnload(module_);
  if (stream_ != nullptr) cuStreamDestroy(stream_);
  if (retained_primary_context_) cuDevicePrimaryCtxRelease(device_);
}

bool PtxCompiledProgram::Compile(const WvmFile& file, std::string& err) {
  if (module_ != nullptr) {
    cuModuleUnload(module_);
    module_ = nullptr;
    function_ = nullptr;
  }
  if (!TranslateWvmToPtx(file, ptx_, err)) return false;

  CUresult result = cuInit(0);
  if (result != CUDA_SUCCESS) {
    err = "CUDA driver initialization failed: " + DriverError(result);
    return false;
  }
  if (!retained_primary_context_) {
    result = cuDeviceGet(&device_, 0);
    if (result != CUDA_SUCCESS) {
      err = "CUDA device lookup failed: " + DriverError(result);
      return false;
    }
    result = cuDevicePrimaryCtxRetain(&context_, device_);
    if (result != CUDA_SUCCESS) {
      err = "CUDA primary-context creation failed: " + DriverError(result);
      context_ = nullptr;
      return false;
    }
    retained_primary_context_ = true;
  }
  result = cuCtxSetCurrent(context_);
  if (result != CUDA_SUCCESS) {
    err = "CUDA context activation failed: " + DriverError(result);
    return false;
  }
  if (stream_ == nullptr) {
    result = cuStreamCreate(&stream_, CU_STREAM_NON_BLOCKING);
    if (result != CUDA_SUCCESS) {
      err = "compiled stream creation failed: " + DriverError(result);
      return false;
    }
  }

  const auto start = std::chrono::steady_clock::now();
  char error_log[8192] = {};
  CUjit_option options[] = {
      CU_JIT_ERROR_LOG_BUFFER,
      CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
  };
  void* option_values[] = {
      error_log,
      reinterpret_cast<void*>(static_cast<uintptr_t>(sizeof(error_log))),
  };
  result = cuModuleLoadDataEx(&module_, ptx_.c_str(), 2, options,
                              option_values);
  const auto stop = std::chrono::steady_clock::now();
  jit_milliseconds_ =
      std::chrono::duration<double, std::milli>(stop - start).count();
  if (result != CUDA_SUCCESS) {
    err = "PTX JIT failed: " + DriverError(result);
    if (error_log[0] != '\0') err += "\n" + std::string(error_log);
    module_ = nullptr;
    return false;
  }
  result = cuModuleGetFunction(&function_, module_, "warpvm_compiled");
  if (result != CUDA_SUCCESS) {
    err = "compiled kernel lookup failed: " + DriverError(result);
    cuModuleUnload(module_);
    module_ = nullptr;
    function_ = nullptr;
    return false;
  }
  return true;
}

bool PtxCompiledProgram::Launch(std::vector<VmState>& states,
                                std::string& err) const {
  std::vector<uint32_t> memory;
  std::vector<uint32_t> framebuffers;
  std::vector<uint32_t> frame_seq;
  return Launch(states, memory, 0, framebuffers, frame_seq, err);
}

bool PtxCompiledProgram::Launch(std::vector<VmState>& states,
                                std::vector<uint32_t>& memory,
                                uint32_t memory_words,
                                std::vector<uint32_t>& framebuffers,
                                std::vector<uint32_t>& frame_seq,
                                std::string& err) const {
  double ignored_milliseconds = 0.0;
  return LaunchCheckpoints(states, memory, memory_words, framebuffers,
                           frame_seq, 1, ignored_milliseconds, err);
}

bool PtxCompiledProgram::LaunchCheckpoints(
    std::vector<VmState>& states, std::vector<uint32_t>& memory,
    uint32_t memory_words, std::vector<uint32_t>& framebuffers,
    std::vector<uint32_t>& frame_seq, uint32_t checkpoints,
    double& elapsed_milliseconds, std::string& err) const {
  if (function_ == nullptr) {
    err = "compiled program has no loaded kernel";
    return false;
  }
  if (states.empty()) {
    err = "compiled launch requires at least one VM state";
    return false;
  }
  if (checkpoints == 0) {
    err = "compiled checkpoint count must be non-zero";
    return false;
  }
  CUresult result = cuCtxSetCurrent(context_);
  if (result != CUDA_SUCCESS) {
    err = "CUDA context activation failed: " + DriverError(result);
    return false;
  }
  for (size_t vm = 0; vm < states.size(); ++vm) {
    const bool runnable = states[vm].status == kRunning ||
                          states[vm].status == kPaused;
    if (states[vm].fault_code != kFaultOk || !runnable) {
      err = "compiled launch requires runnable, non-faulted state for vm " +
            std::to_string(vm);
      return false;
    }
  }
  const size_t expected_memory = states.size() * memory_words;
  if (memory.size() != expected_memory) {
    err = "compiled memory buffer has " + std::to_string(memory.size()) +
          " words; expected " + std::to_string(expected_memory);
    return false;
  }
  const size_t expected_framebuffer = states.size() * kVideoWords;
  if ((!framebuffers.empty() && framebuffers.size() != expected_framebuffer) ||
      (!frame_seq.empty() && frame_seq.size() != states.size()) ||
      (framebuffers.empty() != frame_seq.empty())) {
    err = "compiled framebuffer storage must contain one framebuffer and "
          "frame counter per VM, or both be empty";
    return false;
  }

  const size_t bytes = states.size() * sizeof(VmState);
  auto reserve = [&](CUdeviceptr& storage, size_t& capacity, size_t needed,
                     const char* operation) {
    if (needed == 0 || capacity >= needed) return true;
    if (storage != 0) {
      const CUresult release = cuMemFree(storage);
      if (release != CUDA_SUCCESS) {
        err = std::string(operation) + " old-buffer release failed: " +
              DriverError(release);
        return false;
      }
      storage = 0;
      capacity = 0;
    }
    const CUresult allocation = cuMemAlloc(&storage, needed);
    if (allocation != CUDA_SUCCESS) {
      err = std::string(operation) + " allocation failed: " +
            DriverError(allocation);
      return false;
    }
    capacity = needed;
    return true;
  };
  const size_t memory_bytes = memory.size() * sizeof(uint32_t);
  const size_t framebuffer_bytes = framebuffers.size() * sizeof(uint32_t);
  const size_t frame_seq_bytes = frame_seq.size() * sizeof(uint32_t);
  if (!reserve(scratch_states_, scratch_states_bytes_, bytes,
               "compiled-state") ||
      !reserve(scratch_memory_, scratch_memory_bytes_, memory_bytes,
               "compiled-memory") ||
      !reserve(scratch_framebuffers_, scratch_framebuffers_bytes_,
               framebuffer_bytes, "compiled-framebuffer") ||
      !reserve(scratch_frame_seq_, scratch_frame_seq_bytes_, frame_seq_bytes,
               "compiled-frame-counter"))
    return false;
  CUdeviceptr device_states = scratch_states_;
  CUdeviceptr device_memory = memory.empty() ? 0 : scratch_memory_;
  CUdeviceptr device_framebuffers =
      framebuffers.empty() ? 0 : scratch_framebuffers_;
  CUdeviceptr device_frame_seq = frame_seq.empty() ? 0 : scratch_frame_seq_;
  auto fail = [&](const char* operation, CUresult failure) {
    err = std::string(operation) + ": " + DriverError(failure);
    return false;
  };
  result = cuMemcpyHtoD(device_states, states.data(), bytes);
  if (result != CUDA_SUCCESS) return fail("compiled-state upload failed", result);

  if (!memory.empty()) {
    result = cuMemcpyHtoD(device_memory, memory.data(), memory_bytes);
    if (result != CUDA_SUCCESS)
      return fail("compiled-memory upload failed", result);
  }
  if (!framebuffers.empty()) {
    result = cuMemcpyHtoD(device_framebuffers, framebuffers.data(),
                          framebuffer_bytes);
    if (result != CUDA_SUCCESS)
      return fail("compiled-framebuffer upload failed", result);
    result = cuMemcpyHtoD(device_frame_seq, frame_seq.data(), frame_seq_bytes);
    if (result != CUDA_SUCCESS)
      return fail("compiled-frame-counter upload failed", result);
  }

  uint32_t num_vms = static_cast<uint32_t>(states.size());
  void* parameters[] = {&device_states, &num_vms, &device_memory,
                        &memory_words, &device_framebuffers,
                        &device_frame_seq};
  constexpr uint32_t kBlockThreads = 256;
  const uint32_t grid =
      (num_vms * kLanes + kBlockThreads - 1) / kBlockThreads;
  const auto launch_start = std::chrono::steady_clock::now();
  for (uint32_t checkpoint = 0; checkpoint < checkpoints; ++checkpoint) {
    result = cuLaunchKernel(function_, grid, 1, 1, kBlockThreads, 1, 1, 0,
                            stream_, parameters, nullptr);
    if (result != CUDA_SUCCESS)
      return fail("compiled kernel launch failed", result);
  }
  result = cuStreamSynchronize(stream_);
  if (result != CUDA_SUCCESS) return fail("compiled kernel execution failed", result);
  const auto launch_stop = std::chrono::steady_clock::now();
  elapsed_milliseconds =
      std::chrono::duration<double, std::milli>(launch_stop - launch_start)
          .count();
  result = cuMemcpyDtoH(states.data(), device_states, bytes);
  if (result != CUDA_SUCCESS) return fail("compiled-state download failed", result);
  if (!memory.empty()) {
    result = cuMemcpyDtoH(memory.data(), device_memory,
                          memory.size() * sizeof(uint32_t));
    if (result != CUDA_SUCCESS)
      return fail("compiled-memory download failed", result);
  }
  if (!framebuffers.empty()) {
    result = cuMemcpyDtoH(framebuffers.data(), device_framebuffers,
                          framebuffers.size() * sizeof(uint32_t));
    if (result != CUDA_SUCCESS)
      return fail("compiled-framebuffer download failed", result);
    result = cuMemcpyDtoH(frame_seq.data(), device_frame_seq,
                          frame_seq.size() * sizeof(uint32_t));
    if (result != CUDA_SUCCESS)
      return fail("compiled-frame-counter download failed", result);
  }
  return true;
}

bool PtxCompilationCache::GetOrCompile(
    const WvmFile& file, std::shared_ptr<PtxCompiledProgram>& program,
    bool& cache_hit, std::string& err) {
  // FNV-1a over the canonical content plus versions that affect generated
  // code. The exact-content comparison below makes hash collisions harmless.
  uint64_t hash = 1469598103934665603ull;
  auto mix = [&](uint32_t word) {
    for (uint32_t byte = 0; byte < 4; ++byte) {
      hash ^= (word >> (byte * 8)) & 0xffu;
      hash *= 1099511628211ull;
    }
  };
  mix(kWvmVersion);
  mix(1);  // PTX backend ABI/compiler version.
  mix(static_cast<uint32_t>(file.code.size()));
  for (uint32_t word : file.code) mix(word);
  mix(static_cast<uint32_t>(file.literals.size()));
  for (uint32_t word : file.literals) mix(word);

  std::vector<Entry>& bucket = entries_[hash];
  for (const Entry& entry : bucket) {
    if (entry.file.code == file.code && entry.file.literals == file.literals) {
      program = entry.program;
      cache_hit = true;
      return true;
    }
  }

  auto compiled = std::make_shared<PtxCompiledProgram>();
  if (!compiled->Compile(file, err)) return false;
  bucket.push_back(Entry{file, compiled});
  ++size_;
  program = std::move(compiled);
  cache_hit = false;
  return true;
}

}  // namespace wvm
