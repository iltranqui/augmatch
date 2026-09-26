#include "augmatch/color/dithering.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace augmatch {
namespace {
constexpr float kBayer4x4[16] = {
    0.0f, 8.0f, 2.0f, 10.0f,
    12.0f, 4.0f, 14.0f, 6.0f,
    3.0f, 11.0f, 1.0f, 9.0f,
    15.0f, 7.0f, 13.0f, 5.0f};

void validate(const DitheringConfig& c, const std::uint8_t* in, std::uint8_t* out) {
  if (!in || !out || c.width <= 0 || c.height <= 0 || c.channels <= 0)
    throw std::invalid_argument("invalid dithering dimensions or buffer");
  if (c.bit_depth < 1 || c.bit_depth > 8)
    throw std::invalid_argument("dithering bit depth must be in [1,8]");
  if (c.mode != DitheringMode::OrderedBayer4x4 &&
      c.mode != DitheringMode::ErrorDiffusionFloydSteinberg)
    throw std::invalid_argument("unsupported dithering mode");
}

int quantized_byte(int level, int levels) {
  return std::max(0, std::min(255, static_cast<int>(std::floor(
      static_cast<float>(level * 255) / static_cast<float>(levels) + 0.5f))));
}

void ordered(const std::uint8_t* in, std::uint8_t* out, const DitheringConfig& c, int levels) {
  for (int y = 0; y < c.height; ++y) {
    for (int x = 0; x < c.width; ++x) {
      const float threshold = (kBayer4x4[(y & 3) * 4 + (x & 3)] + 0.5f) / 16.0f;
      const std::size_t pixel = static_cast<std::size_t>(y) * c.width + x;
      for (int channel = 0; channel < c.channels; ++channel) {
        const std::size_t index = pixel * c.channels + channel;
        const float value = static_cast<float>(in[index]) / 255.0f;
        const int level = std::max(0, std::min(levels, static_cast<int>(std::floor(value * levels + threshold))));
        out[index] = static_cast<std::uint8_t>(quantized_byte(level, levels));
      }
    }
  }
}

void error_diffusion(const std::uint8_t* in, std::uint8_t* out, const DitheringConfig& c, int levels) {
  const std::size_t pixels = static_cast<std::size_t>(c.width) * c.height;
  std::vector<float> errors(pixels, 0.0f);
  for (int channel = 0; channel < c.channels; ++channel) {
    std::fill(errors.begin(), errors.end(), 0.0f);
    for (int y = 0; y < c.height; ++y) {
      for (int x = 0; x < c.width; ++x) {
        const std::size_t pixel = static_cast<std::size_t>(y) * c.width + x;
        const std::size_t index = pixel * c.channels + channel;
        const float working = std::max(0.0f, std::min(1.0f,
            static_cast<float>(in[index]) / 255.0f + errors[pixel]));
        const int level = std::max(0, std::min(levels,
            static_cast<int>(std::floor(working * levels + 0.5f))));
        const float quantized = static_cast<float>(level) / static_cast<float>(levels);
        out[index] = static_cast<std::uint8_t>(quantized_byte(level, levels));
        const float error = working - quantized;
        if (x + 1 < c.width) errors[pixel + 1] += error * (7.0f / 16.0f);
        if (y + 1 < c.height) {
          errors[pixel + c.width] += error * (5.0f / 16.0f);
          if (x > 0) errors[pixel + c.width - 1] += error * (3.0f / 16.0f);
          if (x + 1 < c.width) errors[pixel + c.width + 1] += error * (1.0f / 16.0f);
        }
      }
    }
  }
}
}  // namespace

void dithering_u8(const std::uint8_t* in, std::uint8_t* out, const DitheringConfig& c, cudaStream_t) {
  validate(c, in, out);
  const int levels = (1 << c.bit_depth) - 1;
  if (c.mode == DitheringMode::OrderedBayer4x4) ordered(in, out, c, levels);
  else error_diffusion(in, out, c, levels);
}

void dither_u8(const std::uint8_t* in, std::uint8_t* out, const DitheringConfig& c, cudaStream_t stream) {
  dithering_u8(in, out, c, stream);
}
}  // namespace augmatch
