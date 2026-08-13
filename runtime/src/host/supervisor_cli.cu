// Long-lived, line-oriented frontend for the CPU WarpVM supervisor.
#include "host/supervisor_cli.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "host/disasm.h"
#include "host/supervisor.h"

namespace wvm {

int ViewSupervisorPopulation(Supervisor& supervisor,
                             const std::vector<VmId>& logical_ids);

namespace {

std::vector<std::string> TokenizeCommand(const std::string& input) {
  const size_t comment = input.find('#');
  const std::string line = input.substr(0, comment);
  std::istringstream stream(line);
  std::vector<std::string> tokens;
  for (std::string token; stream >> token;) tokens.push_back(token);
  return tokens;
}

bool ParseU32(const std::string& text, uint32_t& value) {
  if (text.empty() || text[0] == '-') return false;
  char* end = nullptr;
  const unsigned long parsed = std::strtoul(text.c_str(), &end, 0);
  if (end == text.c_str() || *end != '\0' ||
      parsed > std::numeric_limits<uint32_t>::max())
    return false;
  value = static_cast<uint32_t>(parsed);
  return true;
}

const char* EngineName(ExecutionEngine engine) {
  return engine == ExecutionEngine::kCompiled ? "COMPILED" : "INTERPRETED";
}

void PrintSupervisorHelp() {
  std::printf("programs (load these before launch):\n");
  std::printf("  program load <name> <file.wvm>\n");
  std::printf("  program list | program unload <name-or-id>\n");
  std::printf("population:\n");
  std::printf("  launch <capacity> [interpreted|compiled]\n");
  std::printf("  vm create <program> [ram-words]\n");
  std::printf("  vm start|stop|resume|reset|delete <vm-id>\n");
  std::printf("  vm program <vm-id> <program>   cold program replacement\n");
  std::printf("  vm engine <vm-id> interpreted|compiled\n");
  std::printf("  list | status <vm-id> | wait <vm-id> <state> [timeout-ms]\n");
  std::printf("inspection (all IDs are stable logical VM IDs):\n");
  std::printf("  regs|sregs|preds|pc|frame <vm-id>\n");
  std::printf("  mem <vm-id> <addr> <count>\n");
  std::printf("  pixel <vm-id> <x> <y>\n");
  std::printf("  mailbox|messages <vm-id>      pending message snapshot\n");
  std::printf("  disasm <vm-id> [start] [count] | log [count]\n");
  std::printf("  view [vm-id ...]              logical-ID framebuffer grid\n");
  std::printf("  sleep <milliseconds> | help | quit\n");
}

class SupervisorCommandSession {
 public:
  bool quit() const { return quit_; }

  bool Execute(const std::string& line) {
    const std::vector<std::string> tok = TokenizeCommand(line);
    if (tok.empty()) return true;
    const std::string& command = tok[0];
    if (command == "quit" || command == "exit") {
      quit_ = true;
      return true;
    }
    if (command == "help") {
      PrintSupervisorHelp();
      return true;
    }
    if (command == "program") return ProgramCommand(tok);
    if (command == "launch") return LaunchCommand(tok);
    if (command == "vm") return VmCommand(tok);
    if (command == "list") return ListCommand();
    if (command == "status") return StatusCommand(tok);
    if (command == "wait") return WaitCommand(tok);
    if (command == "sleep") return SleepCommand(tok);
    if (command == "regs" || command == "sregs" || command == "preds" ||
        command == "pc" || command == "frame")
      return StateCommand(tok);
    if (command == "mem") return MemoryCommand(tok);
    if (command == "pixel") return PixelCommand(tok);
    if (command == "mailbox" || command == "messages")
      return MailboxCommand(tok);
    if (command == "disasm") return DisasmCommand(tok);
    if (command == "log") return LogCommand(tok);
    if (command == "view") return ViewCommand(tok);
    return Fail("unknown command '" + command + "' (try 'help')");
  }

 private:
  bool Fail(const std::string& message) {
    std::fprintf(stderr, "error: %s\n", message.c_str());
    return false;
  }

  bool NeedLaunched() {
    return supervisor_.launched() ||
           Fail("launch the resident capacity before using VM commands");
  }

  const ProgramInfo* ResolveProgram(const std::string& text) const {
    if (const ProgramInfo* named = supervisor_.programs().Find(text))
      return named;
    uint32_t id = 0;
    return ParseU32(text, id)
               ? supervisor_.programs().Find(LoadedProgramId{id})
               : nullptr;
  }

  VmInstanceInfo* ResolveVm(const std::string& text) {
    uint32_t id = 0;
    if (!ParseU32(text, id)) return nullptr;
    return supervisor_.Find(LogicalVmId{id});
  }

  bool ProgramCommand(const std::vector<std::string>& tok) {
    if (tok.size() < 2) return Fail("usage: program load|list|unload ...");
    if (tok[1] == "list") {
      if (tok.size() != 2) return Fail("usage: program list");
      std::printf("program_id name                 refs words literals path\n");
      for (const ProgramInfo& program : supervisor_.programs().programs()) {
        std::printf("%-10u %-20s %-4u %-5zu %-8zu %s\n", program.id.value,
                    program.name.c_str(), program.reference_count,
                    program.image.code.size(), program.image.literals.size(),
                    program.path.c_str());
      }
      return true;
    }
    if (tok[1] == "load") {
      if (tok.size() != 4)
        return Fail("usage: program load <name> <file.wvm>");
      LoadedProgramId id;
      std::string err;
      if (!supervisor_.ProgramLoad(tok[3], tok[2], id, err)) return Fail(err);
      std::printf("program loaded: id=%u name=%s path=%s\n", id.value,
                  tok[2].c_str(), tok[3].c_str());
      return true;
    }
    if (tok[1] == "unload") {
      if (tok.size() != 3)
        return Fail("usage: program unload <name-or-id>");
      const ProgramInfo* program = ResolveProgram(tok[2]);
      if (program == nullptr) return Fail("program is not loaded");
      const LoadedProgramId id = program->id;
      const std::string name = program->name;
      std::string err;
      if (!supervisor_.ProgramUnload(id, err)) return Fail(err);
      std::printf("program unloaded: id=%u name=%s\n", id.value,
                  name.c_str());
      return true;
    }
    return Fail("usage: program load|list|unload ...");
  }

  bool LaunchCommand(const std::vector<std::string>& tok) {
    uint32_t capacity = 0;
    if ((tok.size() != 2 && tok.size() != 3) ||
        !ParseU32(tok[1], capacity))
      return Fail("usage: launch <capacity> [interpreted|compiled]");
    ExecutionEngine engine = ExecutionEngine::kInterpreted;
    if (tok.size() == 3) {
      if (tok[2] == "compiled")
        engine = ExecutionEngine::kCompiled;
      else if (tok[2] != "interpreted")
        return Fail("engine must be interpreted or compiled");
    }
    std::string err;
    if (!supervisor_.Launch(capacity, err, engine)) return Fail(err);
    std::printf("population launched: capacity=%u programs=%zu engine=%s",
                capacity, supervisor_.programs().size(), EngineName(engine));
    if (engine == ExecutionEngine::kCompiled)
      std::printf(" compiled_bodies=%zu ptx_bytes=%zu jit_ms=%.3f",
                  supervisor_.compiled_program_count(),
                  supervisor_.compiled_ptx_bytes(),
                  supervisor_.compiled_jit_milliseconds());
    std::printf("\n");
    return true;
  }

  bool VmCommand(const std::vector<std::string>& tok) {
    if (!NeedLaunched()) return false;
    if (tok.size() < 2)
      return Fail("usage: vm create|start|stop|resume|reset|delete|program|engine");
    const std::string& operation = tok[1];
    std::string err;
    if (operation == "create") {
      if (tok.size() < 3 || tok.size() > 4)
        return Fail("usage: vm create <program> [ram-words]");
      const ProgramInfo* program = ResolveProgram(tok[2]);
      if (program == nullptr) return Fail("program is not loaded");
      uint32_t ram_words = kRamSizeWords;
      if (tok.size() == 4 && !ParseU32(tok[3], ram_words))
        return Fail("RAM word count must be an unsigned integer");
      LogicalVmId id;
      if (!supervisor_.VmCreate(program->id, id, err, ram_words))
        return Fail(err);
      const VmInstanceInfo* vm = supervisor_.Find(id);
      std::printf("vm created: vm_id=%u slot=%u program=%s state=READY\n",
                  id.value, vm->slot.value, program->name.c_str());
      return true;
    }
    if (operation == "program") {
      if (tok.size() != 4)
        return Fail("usage: vm program <vm-id> <program>");
      VmInstanceInfo* vm = ResolveVm(tok[2]);
      const ProgramInfo* program = ResolveProgram(tok[3]);
      if (vm == nullptr) return Fail("logical VM ID is not active");
      if (program == nullptr) return Fail("program is not loaded");
      const VmId vm_id = vm->vm_id.value;
      if (!supervisor_.VmSetProgram(vm->vm_id, program->id, err))
        return Fail(err);
      std::printf("vm rebound: vm_id=%u program=%s state=READY\n", vm_id,
                  program->name.c_str());
      return true;
    }
    if (operation == "engine") {
      if (tok.size() != 4)
        return Fail("usage: vm engine <vm-id> interpreted|compiled");
      VmInstanceInfo* vm = ResolveVm(tok[2]);
      if (vm == nullptr) return Fail("logical VM ID is not active");
      ExecutionEngine engine;
      if (tok[3] == "interpreted")
        engine = ExecutionEngine::kInterpreted;
      else if (tok[3] == "compiled")
        engine = ExecutionEngine::kCompiled;
      else
        return Fail("engine must be interpreted or compiled");
      if (!supervisor_.VmSetEngine(vm->vm_id, engine, err)) return Fail(err);
      std::printf("vm engine: vm_id=%u engine=%s\n", vm->vm_id.value,
                  EngineName(engine));
      return true;
    }
    if (tok.size() != 3)
      return Fail("usage: vm start|stop|resume|reset|delete <vm-id>");
    VmInstanceInfo* vm = ResolveVm(tok[2]);
    if (vm == nullptr) return Fail("logical VM ID is not active");
    const LogicalVmId id = vm->vm_id;
    bool changed = false;
    if (operation == "start")
      changed = supervisor_.VmStart(id, err);
    else if (operation == "stop")
      changed = supervisor_.VmStop(id, err);
    else if (operation == "resume")
      changed = supervisor_.VmResume(id, err);
    else if (operation == "reset")
      changed = supervisor_.VmReset(id, err);
    else if (operation == "delete")
      changed = supervisor_.VmDelete(id, err);
    else
      return Fail("unknown VM operation '" + operation + "'");
    if (!changed) return Fail(err);
    if (operation == "delete")
      std::printf("vm deleted: vm_id=%u state=EMPTY\n", id.value);
    else
      std::printf("vm %s: vm_id=%u state=%s\n", operation.c_str(), id.value,
                  LifecycleName(supervisor_.Find(id)->lifecycle));
    return true;
  }

  bool ListCommand() {
    if (!NeedLaunched()) return false;
    std::printf("vm_id slot program              engine      lifecycle device    fault pc instrs\n");
    for (const VmInstanceInfo& vm : supervisor_.instances()) {
      if (vm.lifecycle == VmLifecycle::kEmpty) {
        std::printf("-     %-4u -                    -           EMPTY     IDLE      OK    0  0\n",
                    vm.slot.value);
        continue;
      }
      const ProgramInfo* program = supervisor_.programs().Find(vm.program_id);
      const uint32_t slot = vm.slot.value;
      std::printf("%-5u %-4u %-20s %-11s %-9s %-9s %-5s %-2u %llu\n",
                  vm.vm_id.value, slot,
                  program == nullptr ? "?" : program->name.c_str(),
                  EngineName(vm.engine), LifecycleName(vm.lifecycle),
                  StatusName(supervisor_.runtime().Status(slot)),
                  FaultName(supervisor_.runtime().Fault(slot)),
                  supervisor_.runtime().Pc(slot),
                  static_cast<unsigned long long>(
                      supervisor_.runtime().Instrs(slot)));
    }
    return true;
  }

  bool StatusCommand(const std::vector<std::string>& tok) {
    if (!NeedLaunched()) return false;
    if (tok.size() != 2) return Fail("usage: status <vm-id>");
    VmInstanceInfo* vm = ResolveVm(tok[1]);
    if (vm == nullptr) return Fail("logical VM ID is not active");
    const ProgramInfo* program = supervisor_.programs().Find(vm->program_id);
    const VmSlot slot = vm->slot.value;
    std::printf("vm %u: slot=%u program=%s engine=%s lifecycle=%s "
                "device=%s fault=%s pc=%u instrs=%llu frame_seq=%u\n",
                vm->vm_id.value, slot,
                program == nullptr ? "?" : program->name.c_str(),
                EngineName(vm->engine), LifecycleName(vm->lifecycle),
                StatusName(supervisor_.runtime().Status(slot)),
                FaultName(supervisor_.runtime().Fault(slot)),
                supervisor_.runtime().Pc(slot),
                static_cast<unsigned long long>(
                    supervisor_.runtime().Instrs(slot)),
                supervisor_.runtime().FrameSeq(slot));
    return true;
  }

  bool WaitCommand(const std::vector<std::string>& tok) {
    if (!NeedLaunched()) return false;
    if (tok.size() < 3 || tok.size() > 4)
      return Fail("usage: wait <vm-id> <state> [timeout-ms]");
    uint32_t vm_id = 0, timeout = 5000;
    if (!ParseU32(tok[1], vm_id) ||
        (tok.size() == 4 && !ParseU32(tok[3], timeout)))
      return Fail("VM ID and timeout must be unsigned integers");
    std::string want = tok[2];
    std::transform(want.begin(), want.end(), want.begin(),
                   [](unsigned char c) { return std::toupper(c); });
    bool reached = false;
    for (uint32_t elapsed = 0; elapsed <= timeout; ++elapsed) {
      VmInstanceInfo* vm = supervisor_.Find(LogicalVmId{vm_id});
      if (want == "EMPTY") {
        reached = vm == nullptr;
      } else if (vm != nullptr) {
        const uint32_t device = supervisor_.runtime().Status(vm->slot.value);
        reached = (want == LifecycleName(vm->lifecycle));
        if (want == "RUNNING") reached &= device == kRunning;
        if (want == "STOPPED") reached &= device == kPaused;
        if (want == "HALTED") reached &= device == kHalted;
        if (want == "FAULTED") reached &= device == kFaulted;
      }
      if (reached) break;
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    if (!reached) return Fail("timed out waiting for VM state " + want);
    std::printf("wait complete: vm_id=%u state=%s\n", vm_id, want.c_str());
    return true;
  }

  bool SleepCommand(const std::vector<std::string>& tok) {
    uint32_t milliseconds = 0;
    if (tok.size() != 2 || !ParseU32(tok[1], milliseconds))
      return Fail("usage: sleep <milliseconds>");
    std::this_thread::sleep_for(std::chrono::milliseconds(milliseconds));
    return true;
  }

  bool StateCommand(const std::vector<std::string>& tok) {
    if (!NeedLaunched()) return false;
    if (tok.size() != 2) return Fail("usage: " + tok[0] + " <vm-id>");
    VmInstanceInfo* vm = ResolveVm(tok[1]);
    if (vm == nullptr) return Fail("logical VM ID is not active");
    const VmSlot slot = vm->slot.value;
    if (tok[0] == "frame") {
      std::printf("vm %u frame: %ux%u ARGB8888 seq=%u slot=%u\n",
                  vm->vm_id.value, kVideoWidth, kVideoHeight,
                  supervisor_.runtime().FrameSeq(slot), slot);
      return true;
    }
    if (tok[0] == "pc") {
      const uint32_t pc = supervisor_.runtime().Pc(slot);
      std::printf("vm %u pc=%u lifecycle=%s\n", vm->vm_id.value, pc,
                  LifecycleName(vm->lifecycle));
      const auto& code = supervisor_.runtime().Code(slot);
      if (pc < code.size())
        std::printf("  %u: %s\n", pc,
                    DisasmWord(code[pc], supervisor_.runtime().Literals(slot))
                        .c_str());
      return true;
    }
    VmState state{};
    if (!supervisor_.runtime().ReadState(slot, state))
      return Fail("VM state read failed");
    if (tok[0] == "regs") {
      for (int reg = 0; reg < kVectorRegs; ++reg) {
        const uint32_t* values = &state.vregs[reg * kLanes];
        bool uniform = true;
        for (int lane = 1; lane < kLanes; ++lane)
          uniform &= values[lane] == values[0];
        if (uniform)
          std::printf("r%-2d = %u (uniform)\n", reg, values[0]);
        else
          std::printf("r%-2d = [%u, %u, %u, %u, ...]\n", reg, values[0],
                      values[1], values[2], values[3]);
      }
    } else if (tok[0] == "sregs") {
      for (int reg = 0; reg < kScalarRegs; ++reg)
        std::printf("s%d = %u\n", reg, state.sregs[reg]);
    } else {
      for (int reg = 0; reg < kPredRegs; ++reg)
        std::printf("p%d = %08x\n", reg, state.preds[reg]);
    }
    return true;
  }

  bool MemoryCommand(const std::vector<std::string>& tok) {
    if (!NeedLaunched()) return false;
    uint32_t vm_id = 0, address = 0, count = 0;
    if (tok.size() != 4 || !ParseU32(tok[1], vm_id) ||
        !ParseU32(tok[2], address) || !ParseU32(tok[3], count))
      return Fail("usage: mem <vm-id> <addr> <count>");
    VmInstanceInfo* vm = supervisor_.Find(LogicalVmId{vm_id});
    if (vm == nullptr) return Fail("logical VM ID is not active");
    std::vector<uint32_t> words;
    if (!supervisor_.runtime().ReadMem(vm->slot.value, address, count, words))
      return Fail("memory read is outside the VM RAM range");
    for (size_t offset = 0; offset < words.size(); offset += 8) {
      std::printf("  [%5zu]:", address + offset);
      for (size_t i = offset; i < words.size() && i < offset + 8; ++i)
        std::printf(" %10u", words[i]);
      std::printf("\n");
    }
    return true;
  }

  bool PixelCommand(const std::vector<std::string>& tok) {
    if (!NeedLaunched()) return false;
    uint32_t vm_id = 0, x = 0, y = 0;
    if (tok.size() != 4 || !ParseU32(tok[1], vm_id) ||
        !ParseU32(tok[2], x) || !ParseU32(tok[3], y))
      return Fail("usage: pixel <vm-id> <x> <y>");
    if (x >= kVideoWidth || y >= kVideoHeight)
      return Fail("pixel coordinates are outside 128x128");
    VmInstanceInfo* vm = supervisor_.Find(LogicalVmId{vm_id});
    if (vm == nullptr) return Fail("logical VM ID is not active");
    std::vector<uint32_t> framebuffer;
    if (!supervisor_.runtime().ReadFramebuffer(vm->slot.value, framebuffer))
      return Fail("framebuffer read failed");
    std::printf("vm %u pixel(%u,%u) = 0x%08x\n", vm_id, x, y,
                framebuffer[y * kVideoWidth + x]);
    return true;
  }

  bool DisasmCommand(const std::vector<std::string>& tok) {
    if (!NeedLaunched()) return false;
    if (tok.size() < 2 || tok.size() > 4)
      return Fail("usage: disasm <vm-id> [start] [count]");
    VmInstanceInfo* vm = ResolveVm(tok[1]);
    if (vm == nullptr) return Fail("logical VM ID is not active");
    const VmSlot slot = vm->slot.value;
    const uint32_t pc = supervisor_.runtime().Pc(slot);
    uint32_t start = pc >= 2 ? pc - 2 : 0, count = 8;
    if ((tok.size() >= 3 && !ParseU32(tok[2], start)) ||
        (tok.size() == 4 && !ParseU32(tok[3], count)))
      return Fail("disassembly start/count must be unsigned integers");
    const auto& code = supervisor_.runtime().Code(slot);
    const auto& literals = supervisor_.runtime().Literals(slot);
    for (uint32_t at = start; at < code.size() && at < start + count; ++at)
      std::printf("%s %4u: %s\n", at == pc ? ">" : " ", at,
                  DisasmWord(code[at], literals).c_str());
    return true;
  }

  bool MailboxCommand(const std::vector<std::string>& tok) {
    if (!NeedLaunched()) return false;
    if (tok.size() != 2)
      return Fail("usage: mailbox|messages <vm-id>");
    VmInstanceInfo* vm = ResolveVm(tok[1]);
    if (vm == nullptr) return Fail("logical VM ID is not active");
    Mailbox mailbox{};
    if (!supervisor_.runtime().ReadMailbox(vm->slot.value, mailbox))
      return Fail("mailbox read failed");
    const uint32_t available = std::min<uint32_t>(
        mailbox.head - mailbox.tail, kMailboxSlots);
    std::printf("vm %u mailbox: owner=%u head=%u tail=%u pending=%u "
                "in_flight=%u\n",
                vm->vm_id.value, mailbox.owner_vm_id, mailbox.head,
                mailbox.tail, available, mailbox.in_flight_sends);
    for (uint32_t offset = 0; offset < available; ++offset) {
      const uint32_t position = mailbox.tail + offset;
      const MailboxSlot& slot = mailbox.slots[position % kMailboxSlots];
      if (slot.sequence != position + 1u) continue;
      std::printf("  src=%u type=%u payload=%u\n",
                  slot.message.header & 0xFFFFu,
                  slot.message.header >> 16, slot.message.payload[0]);
    }
    return true;
  }

  bool LogCommand(const std::vector<std::string>& tok) {
    if (!NeedLaunched()) return false;
    uint32_t count = 16;
    if (tok.size() > 2 || (tok.size() == 2 && !ParseU32(tok[1], count)))
      return Fail("usage: log [count]");
    const LogSnapshot snapshot = supervisor_.runtime().ReadLog();
    const size_t start = snapshot.entries.size() > count
                             ? snapshot.entries.size() - count
                             : 0;
    for (size_t i = start; i < snapshot.entries.size(); ++i) {
      const LogEntry& entry = snapshot.entries[i];
      std::printf("  vm=%u tag=%u value=%u\n", entry.vm_id, entry.tag,
                  entry.value);
    }
    std::printf("(%u total entries)\n", snapshot.head);
    return true;
  }

  bool ViewCommand(const std::vector<std::string>& tok) {
    if (!NeedLaunched()) return false;
    std::vector<VmId> ids;
    if (tok.size() == 1) {
      for (const VmInstanceInfo& vm : supervisor_.instances())
        if (vm.lifecycle != VmLifecycle::kEmpty) ids.push_back(vm.vm_id.value);
    } else {
      for (size_t i = 1; i < tok.size(); ++i) {
        uint32_t id = 0;
        if (!ParseU32(tok[i], id) ||
            supervisor_.Find(LogicalVmId{id}) == nullptr)
          return Fail("view contains an inactive logical VM ID");
        ids.push_back(id);
      }
    }
    if (ids.empty()) return Fail("there are no active VMs to display");
    return ViewSupervisorPopulation(supervisor_, ids) == 0;
  }

  Supervisor supervisor_;
  bool quit_ = false;
};

bool RunCommands(std::istream& input, SupervisorCommandSession& session,
                 const std::string& source, bool interactive) {
  std::string line;
  uint32_t line_number = 0;
  while (!session.quit()) {
    if (interactive) {
      std::printf("warpvm> ");
      std::fflush(stdout);
    }
    if (!std::getline(input, line)) break;
    ++line_number;
    if (session.Execute(line)) continue;
    if (!interactive) {
      std::fprintf(stderr, "%s:%u: command failed\n", source.c_str(),
                   line_number);
      return false;
    }
  }
  return true;
}

}  // namespace

int RunSupervisorCli(const char* script_path, bool continue_interactive) {
  SupervisorCommandSession session;
  std::printf("WarpVM CPU supervisor: type 'help' for commands\n");
  if (script_path != nullptr) {
    std::ifstream script(script_path);
    if (!script) {
      std::fprintf(stderr, "error: cannot open startup script: %s\n",
                   script_path);
      return 2;
    }
    if (!RunCommands(script, session, script_path, false)) return 1;
    if (session.quit() || !continue_interactive) return 0;
  }
  return RunCommands(std::cin, session, "stdin", true) ? 0 : 1;
}

}  // namespace wvm
