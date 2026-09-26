#include "augmatch/pipeline.hpp"

#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#include "augmatch/pipeline_registry.hpp"

namespace augmatch {
namespace {

struct ProbabilityContext {
  RegisteredCpuStage child;
  double probability = 1.0;
};

Result<Pipeline> pipeline_failure(StatusCode code, const char* message) {
  return Result<Pipeline>::failure({code, message});
}

std::string trim(const std::string& text) {
  const std::string whitespace = " \t\r\n";
  const std::size_t first = text.find_first_not_of(whitespace);
  if (first == std::string::npos) return {};
  const std::size_t last = text.find_last_not_of(whitespace);
  return text.substr(first, last - first + 1);
}

bool parse_float(const std::string& text, float* value) {
  if (value == nullptr || text.empty()) return false;
  // strtof stops at the first NUL in text.c_str(); without this check a value
  // like "1.5\0garbage" would parse as 1.5 and silently drop everything after
  // the embedded NUL instead of being rejected as malformed.
  if (text.find('\0') != std::string::npos) return false;
  errno = 0;
  char* end = nullptr;
  const float parsed = std::strtof(text.c_str(), &end);
  if (errno == ERANGE || end == text.c_str() || *end != '\0' || !std::isfinite(parsed))
    return false;
  *value = parsed;
  return true;
}

void probability_stage(MutableImageView image, const ExecutionContext& execution,
                       void* opaque) {
  const auto& context = *static_cast<const ProbabilityContext*>(opaque);
  CpuPipeline child;
  child.stages.push_back(context.child.stage);
  random_apply(image, child, context.probability, execution);
}

}  // namespace

Result<Pipeline> Pipeline::compile(std::vector<PipelineStageSpec> stages) {
  Pipeline pipeline;
  pipeline.stages_ = std::move(stages);
  pipeline.compiled_.stages.reserve(pipeline.stages_.size());
  pipeline.contexts_.reserve(pipeline.stages_.size());

  for (const PipelineStageSpec& spec : pipeline.stages_) {
    auto registered = make_registered_cpu_stage(spec.name, spec.params);
    if (!registered.ok()) return Result<Pipeline>::failure(registered.status());

    if (spec.probability < 1.0f) {
      auto context = std::make_shared<ProbabilityContext>();
      context->child = registered.value();
      context->probability = spec.probability;
      CpuStage wrapper;
      wrapper.callback = probability_stage;
      wrapper.context = context.get();
      pipeline.compiled_.stages.push_back(wrapper);
      pipeline.contexts_.push_back(std::move(context));
    } else {
      pipeline.compiled_.stages.push_back(registered.value().stage);
      pipeline.contexts_.push_back(registered.value().context);
    }
  }
  return Result<Pipeline>::success(std::move(pipeline));
}

Result<Pipeline> Pipeline::load_from_file(const std::string& path) noexcept {
  try {
    std::ifstream input(path);
    if (!input) return pipeline_failure(StatusCode::InvalidArgument,
                                        "cannot open pipeline configuration file");
    const std::string text((std::istreambuf_iterator<char>(input)),
                           std::istreambuf_iterator<char>());
    if (!input.good() && !input.eof())
      return pipeline_failure(StatusCode::ExecutionError,
                              "cannot read pipeline configuration file");
    return load_from_string(text);
  } catch (...) {
    return pipeline_failure(StatusCode::ExecutionError,
                            "failed to load pipeline configuration file");
  }
}

Result<Pipeline> Pipeline::load_from_string(const std::string& text) noexcept {
  try {
    std::vector<PipelineStageSpec> stages;
    PipelineStageSpec* current = nullptr;
    bool probability_set = false;
    std::istringstream input(text);
    std::string raw_line;

    while (std::getline(input, raw_line)) {
      const std::size_t comment = raw_line.find('#');
      const std::string line = trim(raw_line.substr(0, comment));
      if (line.empty()) continue;

      if (line.front() == '[') {
        if (line.size() < 3 || line.back() != ']')
          return pipeline_failure(StatusCode::InvalidArgument,
                                  "malformed pipeline section");
        const std::string name = trim(line.substr(1, line.size() - 2));
        if (name.empty())
          return pipeline_failure(StatusCode::InvalidArgument,
                                  "pipeline stage name is empty");
        stages.push_back(PipelineStageSpec{});
        stages.back().name = name;
        current = &stages.back();
        probability_set = false;
        continue;
      }

      if (current == nullptr)
        return pipeline_failure(StatusCode::InvalidArgument,
                                "pipeline parameter appears before a section");
      const std::size_t equals = line.find('=');
      if (equals == std::string::npos || line.find('=', equals + 1) != std::string::npos)
        return pipeline_failure(StatusCode::InvalidArgument,
                                "malformed pipeline parameter");
      const std::string key = trim(line.substr(0, equals));
      const std::string value_text = trim(line.substr(equals + 1));
      float value = 0.0f;
      if (key.empty() || !parse_float(value_text, &value))
        return pipeline_failure(StatusCode::InvalidArgument,
                                "pipeline parameter must be numeric");

      if (key == "probability") {
        if (probability_set)
          return pipeline_failure(StatusCode::InvalidArgument,
                                  "duplicate pipeline probability");
        if (value < 0.0f || value > 1.0f)
          return pipeline_failure(StatusCode::InvalidArgument,
                                  "pipeline probability must be in [0,1]");
        current->probability = value;
        probability_set = true;
      } else if (!current->params.emplace(key, value).second) {
        return pipeline_failure(StatusCode::InvalidArgument,
                                "duplicate pipeline parameter");
      }
    }

    return compile(std::move(stages));
  } catch (...) {
    return pipeline_failure(StatusCode::ExecutionError,
                            "failed to parse pipeline configuration");
  }
}

Status Pipeline::run(MutableImageView image, ExecutionContext& execution) const noexcept {
  if (!image.valid() || image.memory != MemorySpace::Host ||
      image.type != DataType::UInt8 || image.layout != Layout::HWC ||
      !image.contiguous())
    return {StatusCode::InvalidView,
            "Pipeline requires a contiguous host HWC uint8 image"};
  try {
    sequential(image, compiled_, execution);
    return Status::success();
  } catch (const std::invalid_argument&) {
    return {StatusCode::InvalidArgument, "pipeline stage rejected its input"};
  } catch (...) {
    return {StatusCode::ExecutionError, "pipeline execution failed"};
  }
}

}  // namespace augmatch
