#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#ifndef AUGMATCH_HAS_CUDA
#define AUGMATCH_HAS_CUDA 0
#endif

#if AUGMATCH_HAS_CUDA
#include <cuda_runtime_api.h>
#else
using cudaStream_t = void*;
#endif

namespace augmatch {

enum class SelectionKind : std::uint8_t {
  OneOf,
  OneOrOther,
  RandomApply,
  SomeOf,
  RandomOrder,
};

// A replay entry is the complete decision made by one control node. Entries are
// ordered by traversal, so nested controls do not need process-global ids.
struct StageSelectionRecord {
  SelectionKind kind = SelectionKind::OneOf;
  std::size_t option_count = 0;
  // Scalar selection retained for the original OneOf/OneOrOther records.
  std::size_t selected = 0;
  bool applied = false;
  bool with_replacement = false;
  // SomeOf stores child indices in application order. RandomOrder stores the
  // complete child permutation. Both vectors are explicit replay data rather
  // than implicit RNG state.
  std::vector<std::size_t> selected_indices;
  std::vector<std::size_t> order;
};

struct ReplayRecord {
  std::vector<StageSelectionRecord> selections;

  void clear() noexcept { selections.clear(); }
  bool empty() const noexcept { return selections.empty(); }
};

struct ExecutionContext {
  cudaStream_t stream = nullptr;
  std::uint64_t seed = 0;
  bool deterministic = true;
  // Exactly one pointer may be non-null. The record pointer appends decisions;
  // the replay pointer consumes and validates them. Neither pointer is owned.
  ReplayRecord* record = nullptr;
  const ReplayRecord* replay = nullptr;
  // Mutable only so stage callbacks can consume an explicitly supplied replay
  // without changing the public const callback contract. Composition resets it
  // at each top-level call.
  mutable std::size_t replay_index = 0;
};

}  // namespace augmatch
