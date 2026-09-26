#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "augmatch/pipeline/composition.hpp"
#include "augmatch/core/status.hpp"

namespace augmatch {

// One named stage in a Pipeline's serialized form: the registered stage name,
// its float-valued parameters, and its application probability.
struct PipelineStageSpec {
  std::string name;
  std::unordered_map<std::string, float> params;
  float probability = 1.0f;
};

// Pipeline compiles a list of PipelineStageSpec entries into a CpuPipeline and
// runs it against a mutable image view. Loaded from file/string; a Pipeline is
// otherwise copyable and movable.
class Pipeline {
 public:
  static Result<Pipeline> load_from_file(const std::string& path) noexcept;
  static Result<Pipeline> load_from_string(const std::string& text) noexcept;

  Status run(MutableImageView image, ExecutionContext& execution) const noexcept;
  const std::vector<PipelineStageSpec>& stages() const noexcept { return stages_; }

 private:
  static Result<Pipeline> compile(std::vector<PipelineStageSpec> stages);

  std::vector<PipelineStageSpec> stages_;
  CpuPipeline compiled_;
  // Keeps every borrowed CpuStage::context alive across Pipeline copies/moves.
  std::vector<std::shared_ptr<void>> contexts_;
};

}  // namespace augmatch
