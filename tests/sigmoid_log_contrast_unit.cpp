#include "augmatch/tone.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

static std::uint8_t quantize(float value) {
  value = std::max(0.0f, std::min(255.0f, value));
  return static_cast<std::uint8_t>(std::floor(value + 0.5f));
}

int main() {
  const int width = 17, height = 3, channels = 4;
  const std::size_t count = static_cast<std::size_t>(width) * height * channels;
  std::vector<std::uint8_t> input(count), sigmoid(count), logarithm(count);
  for (std::size_t i = 0; i < count; ++i) input[i] = static_cast<std::uint8_t>((37 * i + 19) & 255);

  const augmatch::SigmoidContrastConfig sigmoid_config{width, height, channels, 0.43f, 7.25f};
  augmatch::sigmoid_contrast_u8(input.data(), sigmoid.data(), sigmoid_config);
  for (std::size_t i = 0; i < count; ++i) {
    const float x = static_cast<float>(input[i]) / 255.0f;
    const float y = 1.0f / (1.0f + std::exp(sigmoid_config.gain * (sigmoid_config.cutoff - x)));
    if (sigmoid[i] != quantize(y * 255.0f)) return 1;
  }

  const augmatch::LogContrastConfig log_config{width, height, channels, 1.7f, 2.5f};
  augmatch::log_contrast_u8(input.data(), logarithm.data(), log_config);
  const float scale = std::pow(log_config.base, log_config.gain) - 1.0f;
  const float denominator = std::log(log_config.base);
  for (std::size_t i = 0; i < count; ++i) {
    const float x = static_cast<float>(input[i]) / 255.0f;
    if (logarithm[i] != quantize(std::log1p(x * scale) / denominator * 255.0f)) return 2;
  }

  bool rejected = false;
  try { augmatch::sigmoid_contrast_u8(input.data(), sigmoid.data(), {width, height, channels, 1.1f, 1.0f}); }
  catch (const std::invalid_argument&) { rejected = true; }
  if (!rejected) return 3;
  rejected = false;
  try { augmatch::log_contrast_u8(input.data(), logarithm.data(), {width, height, channels, 1.0f, 1.0f}); }
  catch (const std::invalid_argument&) { rejected = true; }
  if (!rejected) return 4;
  return 0;
}
