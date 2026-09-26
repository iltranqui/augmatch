#pragma once

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

#include "augmatch/pipeline/composition.hpp"
#include "augmatch/core/status.hpp"

namespace augmatch {

using StageParameters = std::unordered_map<std::string, float>;

// CpuStage borrows its context. This wrapper lets factories provide an
// optional shared owner while retaining the existing CpuStage callback ABI.
struct RegisteredCpuStage {
  CpuStage stage{};
  std::shared_ptr<void> context{};
};

using CpuStageFactory =
    std::function<Result<RegisteredCpuStage>(const StageParameters& params)>;

// Names are process-wide and case-sensitive. Existing names, including the
// built-ins, cannot be replaced accidentally.
Status register_stage(const std::string& name, CpuStageFactory factory) noexcept;

// Resolve and validate one stage specification. Pipeline uses this once while
// loading; execution never performs name lookup or parameter parsing.
Result<RegisteredCpuStage> make_registered_cpu_stage(
    const std::string& name, const StageParameters& params) noexcept;

}  // namespace augmatch
