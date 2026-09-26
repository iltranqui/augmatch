#include "augmatch/core/convert.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

int main() {
#if AUGMATCH_HAS_CUDA
  return 77;
#else
  constexpr int width = 2;
  constexpr int height = 2;
  constexpr int channels = 3;
  const std::uint8_t input[] = {
      1, 2, 3, 4, 5, 6,
      7, 8, 9, 10, 11, 12};
  std::vector<float> output(width * height * channels, -1.0f);
  const auto view = augmatch::make_chw_f32_view(output.data(), width, height, channels);
  if (!view.valid() || !view.contiguous() || view.type != augmatch::DataType::Float32 ||
      view.layout != augmatch::Layout::CHW || view.memory != augmatch::MemorySpace::Host)
    return 1;

  const augmatch::ImageView input_view =
      augmatch::make_hwc_u8_view(input, width, height, channels);
  if (input_view.layout != augmatch::Layout::HWC || input_view.type != augmatch::DataType::UInt8)
    return 2;
  augmatch::to_tensor_v2(input_view, view, {});

  const float expected[] = {
      1.0f / 255.0f, 4.0f / 255.0f, 7.0f / 255.0f, 10.0f / 255.0f,
      2.0f / 255.0f, 5.0f / 255.0f, 8.0f / 255.0f, 11.0f / 255.0f,
      3.0f / 255.0f, 6.0f / 255.0f, 9.0f / 255.0f, 12.0f / 255.0f};
  for (std::size_t i = 0; i < output.size(); ++i)
    if (std::abs(output[i] - expected[i]) > 1e-7f) return 3;

  const augmatch::ToTensorV2Config normalized{width, height, channels, true,
                                               0.1f, 0.2f, 0.3f,
                                               0.5f, 0.25f, 0.2f};
  std::fill(output.begin(), output.end(), 0.0f);
  augmatch::to_tensor_v2_u8_f32(input, output.data(), normalized);
  if (std::abs(output[0] - ((1.0f / 255.0f - 0.1f) / 0.5f)) > 1e-7f ||
      std::abs(output[4] - ((2.0f / 255.0f - 0.2f) / 0.25f)) > 1e-7f ||
      std::abs(output[8] - ((3.0f / 255.0f - 0.3f) / 0.2f)) > 1e-7f)
    return 4;

  bool rejected = false;
  try {
    auto bad = view;
    bad.layout = augmatch::Layout::HWC;
    augmatch::to_tensor_v2(input_view, bad);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  if (!rejected) return 5;

  rejected = false;
  try {
    auto bad = normalized;
    bad.std0 = 0.0f;
    augmatch::to_tensor_v2_u8_f32(input, output.data(), bad);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  if (!rejected) return 6;
  return 0;
#endif
}
