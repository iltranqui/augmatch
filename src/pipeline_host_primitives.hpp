#pragma once

#include <cstdint>

// CUDA builds select device implementations for AUGMATCH's public transform
// symbols. The host Pipeline still needs the existing CPU implementations, so
// CMake compiles those sources a second time under this private namespace.
namespace augmatch_pipeline_host {

struct MultiplyBrightnessConfig {
  int width = 0, height = 0, channels = 0;
  float multiplier = 1.0f;
};
struct MultiplySaturationConfig {
  int width = 0, height = 0, channels = 0;
  float multiplier = 1.0f;
};
struct AddToHueConfig {
  int width = 0, height = 0, channels = 0;
  float amount = 0.0f;
};
struct ColorJitterConfig {
  int width = 0, height = 0, channels = 0;
  float brightness = 0.0f;
  float contrast = 1.0f;
  float saturation = 1.0f;
  float hue = 0.0f;
};
enum class BlurMode : int { Average = 0, Gaussian = 1, Median = 2 };
struct BlurConfig {
  int width = 0, height = 0, channels = 0, kernel_size = 3;
  float sigma = 1.0f;
  BlurMode mode = BlurMode::Average;
};

void multiply_brightness_u8(const std::uint8_t*, std::uint8_t*,
                            const MultiplyBrightnessConfig&, void* stream = nullptr);
void multiply_saturation_u8(const std::uint8_t*, std::uint8_t*,
                            const MultiplySaturationConfig&, void* stream = nullptr);
void add_to_hue_u8(const std::uint8_t*, std::uint8_t*, const AddToHueConfig&,
                   void* stream = nullptr);
void color_jitter_u8(const std::uint8_t*, std::uint8_t*, const ColorJitterConfig&,
                     void* stream = nullptr);
void blur_u8(const std::uint8_t*, std::uint8_t*, const BlurConfig&,
             void* stream = nullptr);
void horizontal_flip_u8(const std::uint8_t*, std::uint8_t*, int width, int height,
                        int channels, void* stream = nullptr);

}  // namespace augmatch_pipeline_host
