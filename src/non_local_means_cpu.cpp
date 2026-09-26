#include "augmatch/filter.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace augmatch {
namespace {
int reflect101_nlm(int value, int size) {
  if (size == 1) return 0;
  while (value < 0 || value >= size) value = value < 0 ? -value : 2 * size - value - 2;
  return value;
}

void validate_nlm(const std::uint8_t* in, const std::uint8_t* out,
                  const NonLocalMeansDenoisingConfig& c) {
  if (!in || !out || c.width <= 0 || c.height <= 0 || c.channels <= 0 ||
      c.patch_radius < 0 || c.patch_radius > 4 || c.search_radius < 0 ||
      c.search_radius > 8 || !std::isfinite(c.h) || c.h <= 0.0f) {
    throw std::invalid_argument("invalid non-local means denoising configuration");
  }
}
} // namespace

void non_local_means_denoising_u8(const std::uint8_t* in, std::uint8_t* out,
                                  const NonLocalMeansDenoisingConfig& c,
                                  cudaStream_t) {
  validate_nlm(in, out, c);
  const int patch_side = 2 * c.patch_radius + 1;
  const float inv_h2 = 1.0f / (c.h * c.h);
  for (int y = 0; y < c.height; ++y) {
    for (int x = 0; x < c.width; ++x) {
      std::vector<float> channel_sums(static_cast<std::size_t>(c.channels), 0.0f);
      float weight_sum = 0.0f;
      for (int sy = -c.search_radius; sy <= c.search_radius; ++sy) {
        for (int sx = -c.search_radius; sx <= c.search_radius; ++sx) {
          float squared_distance = 0.0f;
          for (int py = -c.patch_radius; py <= c.patch_radius; ++py) {
            for (int px = -c.patch_radius; px <= c.patch_radius; ++px) {
              const int ax = reflect101_nlm(x + px, c.width);
              const int ay = reflect101_nlm(y + py, c.height);
              const int bx = reflect101_nlm(x + sx + px, c.width);
              const int by = reflect101_nlm(y + sy + py, c.height);
              const std::size_t a = (static_cast<std::size_t>(ay) * c.width + ax) * c.channels;
              const std::size_t b = (static_cast<std::size_t>(by) * c.width + bx) * c.channels;
              for (int patch_ch = 0; patch_ch < c.channels; ++patch_ch) {
                const float difference = static_cast<float>(in[a + patch_ch]) - static_cast<float>(in[b + patch_ch]);
                squared_distance += difference * difference;
              }
            }
          }
          const float normalizer = static_cast<float>(patch_side * patch_side * c.channels);
          const float weight = std::exp(-(squared_distance / normalizer) * inv_h2);
          const int sample_x = reflect101_nlm(x + sx, c.width);
          const int sample_y = reflect101_nlm(y + sy, c.height);
          const std::size_t sample = (static_cast<std::size_t>(sample_y) * c.width + sample_x) * c.channels;
          for (int ch = 0; ch < c.channels; ++ch) channel_sums[ch] += weight * in[sample + ch];
          weight_sum += weight;
        }
      }
      for (int ch = 0; ch < c.channels; ++ch) {
        const float value = channel_sums[ch] / weight_sum;
        out[(static_cast<std::size_t>(y) * c.width + x) * c.channels + ch] =
            static_cast<std::uint8_t>(std::floor(std::max(0.0f, std::min(255.0f, value)) + 0.5f));
      }
    }
  }
}
} // namespace augmatch
