#include "augmatch/composition.hpp"

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace {
struct StageState {
  int calls = 0;
  int order = 0;
  std::uint64_t seed = 0;
};

void add_stage(augmatch::MutableImageView image, const augmatch::ExecutionContext& execution,
               void* opaque) {
  auto& state = *static_cast<StageState*>(opaque);
  if (execution.seed != state.seed) throw std::runtime_error("stage received wrong context");
  ++state.calls;
  state.order = state.order * 10 + state.calls;
  auto* bytes = static_cast<std::uint8_t*>(image.data);
  for (int i = 0; i < image.width * image.height * image.channels; ++i)
    bytes[i] = static_cast<std::uint8_t>(bytes[i] + 1);
}

void multiply_stage(augmatch::MutableImageView image, const augmatch::ExecutionContext& execution,
                    void* opaque) {
  auto& state = *static_cast<StageState*>(opaque);
  if (execution.seed != state.seed) throw std::runtime_error("stage received wrong context");
  ++state.calls;
  state.order = state.order * 10 + state.calls;
  auto* bytes = static_cast<std::uint8_t*>(image.data);
  for (int i = 0; i < image.width * image.height * image.channels; ++i)
    bytes[i] = static_cast<std::uint8_t>(bytes[i] * 2);
}

bool equal(const std::vector<std::uint8_t>& lhs, const std::vector<std::uint8_t>& rhs) {
  return lhs == rhs;
}
}  // namespace

int main() {
  constexpr int width = 4;
  constexpr int height = 1;
  constexpr int channels = 1;
  const augmatch::ExecutionContext execution{nullptr, 1234, true};
  const auto view_for = [](std::vector<std::uint8_t>& bytes) {
    return augmatch::make_hwc_u8_view(bytes.data(), width, height, channels);
  };

  StageState compose_state{0, 0, execution.seed};
  std::vector<std::uint8_t> compose_image{1, 2, 3, 4};
  const auto compose_pipeline = augmatch::make_compose(
      std::vector<augmatch::CpuStage>{{add_stage, &compose_state},
                                      {multiply_stage, &compose_state}});
  augmatch::compose(view_for(compose_image), compose_pipeline, execution);
  if (!equal(compose_image, {4, 6, 8, 10}) || compose_state.calls != 2 || compose_state.order != 12)
    return 1;

  StageState sequential_state{0, 0, execution.seed};
  std::vector<std::uint8_t> sequential_image{5, 6, 7, 8};
  const auto sequential_pipeline = augmatch::make_sequential(
      std::vector<augmatch::CpuStage>{{multiply_stage, &sequential_state},
                                      {add_stage, &sequential_state}});
  augmatch::sequential(view_for(sequential_image), sequential_pipeline, execution);
  if (!equal(sequential_image, {11, 13, 15, 17}) || sequential_state.calls != 2 ||
      sequential_state.order != 12)
    return 2;

  std::vector<std::uint8_t> identity_image{9, 8, 7, 6};
  const auto identity_before = identity_image;
  augmatch::identity(view_for(identity_image), execution);
  augmatch::noop(view_for(identity_image), execution);
  augmatch::compose(view_for(identity_image), augmatch::CpuPipeline{}, execution);
  if (identity_image != identity_before) return 3;

  bool rejected_null = false;
  try {
    augmatch::compose(view_for(identity_image),
                      augmatch::make_compose(std::vector<augmatch::CpuStage>{{nullptr, nullptr}}),
                      execution);
  } catch (const std::invalid_argument&) {
    rejected_null = true;
  }
  if (!rejected_null) return 4;

  bool rejected_device = false;
  try {
    auto invalid = view_for(identity_image);
    invalid.memory = augmatch::MemorySpace::Device;
    augmatch::compose(invalid, compose_pipeline, execution);
  } catch (const std::invalid_argument&) {
    rejected_device = true;
  }
  if (!rejected_device) return 5;

  return 0;
}
