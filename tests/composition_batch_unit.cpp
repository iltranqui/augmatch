#include "augmatch/composition.hpp"

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace {
void add_one(augmatch::MutableImageView image, const augmatch::ExecutionContext&, void*) {
  auto* bytes = static_cast<std::uint8_t*>(image.data);
  for (int y = 0; y < image.height; ++y)
    for (int x = 0; x < image.width; ++x)
      for (int c = 0; c < image.channels; ++c)
        bytes[y * image.stride_y + x * image.stride_x + c * image.stride_c]++;
}
void add_ten(augmatch::MutableImageView image, const augmatch::ExecutionContext&, void*) {
  auto* bytes = static_cast<std::uint8_t*>(image.data);
  for (int y = 0; y < image.height; ++y)
    for (int x = 0; x < image.width; ++x)
      for (int c = 0; c < image.channels; ++c)
        bytes[y * image.stride_y + x * image.stride_x + c * image.stride_c] += 10;
}
augmatch::CpuPipeline one(augmatch::CpuStageCallback callback) {
  return augmatch::make_compose(std::vector<augmatch::CpuStage>{{callback, nullptr}});
}
}  // namespace

int main() {
  constexpr int width = 2;
  constexpr int channels = 3;
  const auto view = [](std::vector<std::uint8_t>& data) {
    return augmatch::make_hwc_u8_view(data.data(), width, 1, channels);
  };
  const std::vector<augmatch::CpuPipeline> choices{one(add_one), one(add_ten), one(add_one)};

  augmatch::ReplayRecord record;
  auto recorded = std::vector<std::uint8_t>{1, 2, 3, 4, 5, 6};
  augmatch::ExecutionContext recording{nullptr, 7, true, &record, nullptr};
  augmatch::some_of(view(recorded), choices, 2, recording);
  augmatch::random_order(view(recorded), choices, recording);
  if (record.selections.size() != 2 ||
      record.selections[0].kind != augmatch::SelectionKind::SomeOf ||
      record.selections[0].selected_indices.size() != 2 ||
      record.selections[1].kind != augmatch::SelectionKind::RandomOrder ||
      record.selections[1].order.size() != choices.size())
    return 1;

  auto replayed = std::vector<std::uint8_t>{1, 2, 3, 4, 5, 6};
  augmatch::ExecutionContext replaying{nullptr, 999, true, nullptr, &record};
  augmatch::some_of(view(replayed), choices, 2, replaying);
  augmatch::random_order(view(replayed), choices, replaying);
  if (replayed != recorded) return 2;

  // Property check: every seed records a valid unique selection and replays it.
  for (std::uint64_t seed = 0; seed < 64; ++seed) {
    augmatch::ReplayRecord property;
    augmatch::ExecutionContext save{nullptr, seed, true, &property, nullptr};
    auto a = std::vector<std::uint8_t>{1, 2, 3, 4, 5, 6};
    augmatch::some_of(view(a), choices, 2, save);
    augmatch::ExecutionContext load{nullptr, seed + 1, true, nullptr, &property};
    auto b = std::vector<std::uint8_t>{1, 2, 3, 4, 5, 6};
    augmatch::some_of(view(b), choices, 2, load);
    if (a != b || property.selections.size() != 1 ||
        property.selections[0].selected_indices[0] == property.selections[0].selected_indices[1])
      return 5;
  }

  // Only the middle channel is exposed to the callback; neighboring bytes stay unchanged.
  auto selected = std::vector<std::uint8_t>{1, 2, 3, 4, 5, 6};
  augmatch::selective_channel_transform(view(selected), one(add_ten), 1, 1);
  if (selected != std::vector<std::uint8_t>({1, 12, 3, 4, 15, 6})) return 3;
  auto stage_range = std::vector<std::uint8_t>{1, 2, 3, 4, 5, 6};
  const auto ranged = augmatch::make_compose(std::vector<augmatch::CpuStage>{{add_ten, nullptr,
                                                                               {2, 1}}});
  augmatch::compose(view(stage_range), ranged);
  if (stage_range != std::vector<std::uint8_t>({1, 2, 13, 4, 5, 16})) return 6;

  bool rejected_count = false;
  try {
    augmatch::some_of(view(selected), choices, choices.size() + 1);
  } catch (const std::invalid_argument&) {
    rejected_count = true;
  }
  if (!rejected_count) return 4;
  return 0;
}
