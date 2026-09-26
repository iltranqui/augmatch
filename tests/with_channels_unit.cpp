#include "augmatch/composition.hpp"

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace {
void add_ten(augmatch::MutableImageView image, const augmatch::ExecutionContext&, void*) {
  auto* bytes = static_cast<std::uint8_t*>(image.data);
  for (int y = 0; y < image.height; ++y)
    for (int x = 0; x < image.width; ++x)
      for (int c = 0; c < image.channels; ++c)
        bytes[y * image.stride_y + x * image.stride_x + c * image.stride_c] += 10;
}

augmatch::CpuPipeline pipeline() {
  return augmatch::make_compose(std::vector<augmatch::CpuStage>{{add_ten, nullptr}});
}

augmatch::MutableImageView view(std::vector<std::uint8_t>& bytes) {
  return augmatch::make_hwc_u8_view(bytes.data(), 2, 1, 4);
}
}  // namespace

int main() {
  // Explicit indices are applied independently to each contiguous HWC run.
  std::vector<std::uint8_t> selected{1, 2, 3, 4, 5, 6, 7, 8};
  augmatch::with_channels(view(selected), pipeline(), std::vector<int>{0, 2});
  if (selected != std::vector<std::uint8_t>({11, 2, 13, 4, 15, 6, 17, 8})) return 1;

  // The range overload and the existing SelectiveChannelTransform API agree.
  std::vector<std::uint8_t> ranged{1, 2, 3, 4, 5, 6, 7, 8};
  augmatch::with_channels(view(ranged), pipeline(), augmatch::ChannelRange{1, 2});
  if (ranged != std::vector<std::uint8_t>({1, 12, 13, 4, 5, 16, 17, 8})) return 2;
  std::vector<std::uint8_t> preserved{1, 2, 3, 4, 5, 6, 7, 8};
  augmatch::selective_channel_transform(view(preserved), pipeline(), 1, 1);
  if (preserved != std::vector<std::uint8_t>({1, 12, 3, 4, 5, 16, 7, 8})) return 3;

  // Explicit stage selections extend the native stage contract without copies.
  std::vector<std::uint8_t> stage_selected{1, 2, 3, 4, 5, 6, 7, 8};
  const auto stage = augmatch::make_compose(std::vector<augmatch::CpuStage>{
      {add_ten, nullptr, {}, augmatch::ChannelSelection{{1, 3}, {}}}});
  augmatch::compose(view(stage_selected), stage);
  if (stage_selected != std::vector<std::uint8_t>({1, 12, 3, 14, 5, 16, 7, 18})) return 4;

  // Explicit ranges compose with index lists and are normalized without repeats.
  std::vector<std::uint8_t> mixed{1, 2, 3, 4, 5, 6, 7, 8};
  augmatch::with_channels(
      view(mixed), pipeline(), augmatch::ChannelSelection{{1}, {{0, 1}, {3, 1}}});
  if (mixed != std::vector<std::uint8_t>({11, 12, 3, 14, 15, 16, 7, 18})) return 5;

  // Empty WithChannels is a validated no-op, and invalid indices are rejected.
  std::vector<std::uint8_t> empty{1, 2, 3, 4, 5, 6, 7, 8};
  augmatch::with_channels(view(empty), pipeline(), std::vector<int>{});
  if (empty != std::vector<std::uint8_t>({1, 2, 3, 4, 5, 6, 7, 8})) return 6;
  bool rejected = false;
  try {
    augmatch::with_channels(view(empty), pipeline(), std::vector<int>{4});
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  if (!rejected) return 7;

  // Property check across seeds: the selected channel set is invariant.
  for (std::uint64_t seed = 0; seed < 64; ++seed) {
    std::vector<std::uint8_t> property{0, 0, 0, 0, 0, 0, 0, 0};
    augmatch::with_channels(view(property), pipeline(), std::vector<int>{0, 2},
                            augmatch::ExecutionContext{nullptr, seed, true});
    if (property != std::vector<std::uint8_t>({10, 0, 10, 0, 10, 0, 10, 0})) return 8;
  }
  return 0;
}
