// Host-side logical VM identity and resident-slot directory.
//
// Logical IDs are stable architectural addresses. Slots are recyclable GPU
// storage/execution positions. Retiring a VM invalidates its route permanently
// for this supervisor epoch; creating another VM in the same slot allocates a
// fresh logical ID.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "gpu/warpvm.cuh"

namespace wvm {

struct LogicalVmId {
  VmId value = kInvalidVmId;
};

struct ResidentSlotId {
  VmSlot value = kInvalidVmSlot;
};

struct LoadedProgramId {
  ProgramId value = 0;
};

class VmDirectory {
 public:
  explicit VmDirectory(uint32_t capacity = 0) { Reset(capacity); }

  void Reset(uint32_t capacity) {
    routes_.assign(kVmIdCount, kInvalidVmSlot);
    slot_ids_.assign(capacity, kInvalidVmId);
    next_id_ = 0;
  }

  bool Create(ResidentSlotId slot, LogicalVmId& result, std::string& err) {
    if (slot.value >= slot_ids_.size()) {
      err = "resident slot is outside configured capacity";
      return false;
    }
    if (slot_ids_[slot.value] != kInvalidVmId) {
      err = "resident slot is already occupied";
      return false;
    }
    if (next_id_ >= kVmIdCount) {
      err = "logical VM address space exhausted for this supervisor epoch";
      return false;
    }
    const VmId id = next_id_++;
    routes_[id] = slot.value;
    slot_ids_[slot.value] = id;
    result.value = id;
    return true;
  }

  bool Retire(LogicalVmId id, std::string& err) {
    const VmSlot slot = SlotFor(id);
    if (slot == kInvalidVmSlot) {
      err = "logical VM ID is not active";
      return false;
    }
    routes_[id.value] = kInvalidVmSlot;
    slot_ids_[slot] = kInvalidVmId;
    return true;
  }

  VmSlot SlotFor(LogicalVmId id) const {
    return id.value < routes_.size() ? routes_[id.value] : kInvalidVmSlot;
  }

  VmId IdFor(ResidentSlotId slot) const {
    return slot.value < slot_ids_.size() ? slot_ids_[slot.value]
                                         : kInvalidVmId;
  }

  const std::vector<VmSlot>& routes() const { return routes_; }
  uint32_t capacity() const {
    return static_cast<uint32_t>(slot_ids_.size());
  }
  VmId next_id() const { return next_id_; }

 private:
  std::vector<VmSlot> routes_;
  std::vector<VmId> slot_ids_;
  VmId next_id_ = 0;
};

}  // namespace wvm
