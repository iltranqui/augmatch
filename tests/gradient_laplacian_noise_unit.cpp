#include "augmatch/augmatch.hpp"
#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <vector>

int main() {
  const int w = 19, h = 15, channels = 3;
  std::vector<std::uint8_t> input(w * h * channels), first(input.size()), repeat(input.size()), other(input.size());
  for (int y = 0; y < h; ++y) for (int x = 0; x < w; ++x) for (int ch = 0; ch < channels; ++ch)
    input[(y * w + x) * channels + ch] = static_cast<std::uint8_t>(x < w / 2 ? 64 + ch : 180 + ch);

  augmatch::GradientPlusLaplacianNoiseConfig config{w, h, channels, 7.0f, 1.5f, 2.0f, 0x12345678};
  augmatch::gradient_plus_laplacian_noise_u8(input.data(), first.data(), config);
  augmatch::gradient_plus_laplacian_noise_u8(input.data(), repeat.data(), config);
  assert(first == repeat);
  config.seed++;
  augmatch::gradient_plus_laplacian_noise_u8(input.data(), other.data(), config);
  assert(first != other);
  for (std::uint8_t v : first) assert(v <= 255);

  config.sigma = 0.0f;
  config.gradient_scale = config.laplacian_scale = 0.0f;
  augmatch::gradient_plus_laplacian_noise_u8(input.data(), other.data(), config);
  assert(input == other);

  config.sigma = -1.0f;
  bool rejected = false;
  try { augmatch::gradient_plus_laplacian_noise_u8(input.data(), other.data(), config); }
  catch (const std::invalid_argument&) { rejected = true; }
  assert(rejected);
}
