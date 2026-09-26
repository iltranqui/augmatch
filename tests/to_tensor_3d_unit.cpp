#include "augmatch/convert.hpp"
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

int main() {
#if AUGMATCH_HAS_CUDA
  return 77;
#else
  const augmatch::ToTensor3DConfig config{2, 2, 2, 2, false};
  const std::uint8_t input[] = {
      1, 2, 3, 4, 5, 6, 7, 8,
      9, 10, 11, 12, 13, 14, 15, 16};
  std::vector<float> output(16, -1.0f);
  augmatch::to_tensor_3d_u8_f32(input, output.data(), config);
  const int expected_input[] = {
      1, 3, 5, 7, 9, 11, 13, 15,
      2, 4, 6, 8, 10, 12, 14, 16};
  for (std::size_t i = 0; i < output.size(); ++i)
    if (std::abs(output[i] - expected_input[i] / 255.0f) > 1e-7f) return 1;

  const augmatch::ToTensor3DConfig normalized{
      2, 2, 2, 2, true, 0.1f, 0.2f, 0.3f, 0.5f, 0.25f, 0.2f};
  augmatch::to_tensor_cdhw_f32(input, output.data(), normalized);
  if (std::abs(output[0] - ((1.0f / 255.0f - 0.1f) / 0.5f)) > 1e-7f ||
      std::abs(output[8] - ((2.0f / 255.0f - 0.2f) / 0.25f)) > 1e-7f)
    return 2;

  bool rejected = false;
  try {
    auto bad = config;
    bad.depth = 0;
    augmatch::to_tensor_3d(input, output.data(), bad);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  if (!rejected) return 3;
  rejected = false;
  try {
    auto bad = normalized;
    bad.std1 = 0.0f;
    augmatch::to_tensor_3d(input, output.data(), bad);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  if (!rejected) return 4;
  return 0;
#endif
}
