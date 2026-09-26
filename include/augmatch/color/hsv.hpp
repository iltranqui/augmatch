#pragma once
#include <cstdint>
#ifndef AUGMATCH_HAS_CUDA
#define AUGMATCH_HAS_CUDA 0
#endif
#if AUGMATCH_HAS_CUDA
#include <cuda_runtime_api.h>
#else
using cudaStream_t = void*;
#endif
namespace augmatch {
// Adds independent OpenCV-bin offsets to hue, saturation, and value.
struct HSVShiftConfig {
  int width = 0, height = 0, channels = 0;
  int hue_shift = 0, saturation_shift = 0, value_shift = 0;
};
void hsv_shift_u8(const std::uint8_t* input, std::uint8_t* output, const HSVShiftConfig& config, cudaStream_t stream = nullptr);

// The batch below uses OpenCV-style uint8 HSV bins: hue is in [0,180), while
// saturation and value are in [0,255]. Hue parameters are rounded to the nearest
// bin and wrapped modulo 180; saturation/value parameters are rounded half-up and
// clipped to [0,255]. RGB conversion uses the same half-up uint8 rounding on CPU
// and CUDA. RGB channels are transformed and channels >= 3 are copied unchanged.
struct MultiplyBrightnessConfig {
  int width = 0, height = 0, channels = 0;
  float multiplier = 1.0f;
};
struct AddToBrightnessConfig {
  int width = 0, height = 0, channels = 0;
  float amount = 0.0f;
};
// MultiplyAndAddToBrightness applies value = value * multiplier + addend in HSV bins.
// Both parameters are explicit, so no imgaug distribution or global RNG is used.
struct MultiplyAndAddToBrightnessConfig {
  int width = 0, height = 0, channels = 0;
  float multiplier = 1.0f, addend = 0.0f;
};
struct WithHueAndSaturationConfig {
  int width = 0, height = 0, channels = 0;
  float hue = 0.0f, saturation = 0.0f;
};
struct MultiplyHueAndSaturationConfig {
  int width = 0, height = 0, channels = 0;
  float hue_multiplier = 1.0f, saturation_multiplier = 1.0f;
};
struct MultiplyHueConfig {
  int width = 0, height = 0, channels = 0;
  float multiplier = 1.0f;
};
struct MultiplySaturationConfig {
  int width = 0, height = 0, channels = 0;
  float multiplier = 1.0f;
};
struct RemoveSaturationConfig {
  int width = 0, height = 0, channels = 0;
};
struct AddToHueAndSaturationConfig {
  int width = 0, height = 0, channels = 0;
  float hue_amount = 0.0f, saturation_amount = 0.0f;
};
struct AddToHueConfig {
  int width = 0, height = 0, channels = 0;
  float amount = 0.0f;
};
struct AddToSaturationConfig {
  int width = 0, height = 0, channels = 0;
  float amount = 0.0f;
};
void multiply_brightness_u8(const std::uint8_t* input, std::uint8_t* output, const MultiplyBrightnessConfig& config, cudaStream_t stream = nullptr);
void add_to_brightness_u8(const std::uint8_t* input, std::uint8_t* output, const AddToBrightnessConfig& config, cudaStream_t stream = nullptr);
void multiply_and_add_to_brightness_u8(const std::uint8_t* input, std::uint8_t* output, const MultiplyAndAddToBrightnessConfig& config, cudaStream_t stream = nullptr);
void with_hue_and_saturation_u8(const std::uint8_t* input, std::uint8_t* output, const WithHueAndSaturationConfig& config, cudaStream_t stream = nullptr);
void multiply_hue_and_saturation_u8(const std::uint8_t* input, std::uint8_t* output, const MultiplyHueAndSaturationConfig& config, cudaStream_t stream = nullptr);
void multiply_hue_u8(const std::uint8_t* input, std::uint8_t* output, const MultiplyHueConfig& config, cudaStream_t stream = nullptr);
void multiply_saturation_u8(const std::uint8_t* input, std::uint8_t* output, const MultiplySaturationConfig& config, cudaStream_t stream = nullptr);
void remove_saturation_u8(const std::uint8_t* input, std::uint8_t* output, const RemoveSaturationConfig& config, cudaStream_t stream = nullptr);
void add_to_hue_and_saturation_u8(const std::uint8_t* input, std::uint8_t* output, const AddToHueAndSaturationConfig& config, cudaStream_t stream = nullptr);
void add_to_hue_u8(const std::uint8_t* input, std::uint8_t* output, const AddToHueConfig& config, cudaStream_t stream = nullptr);
void add_to_saturation_u8(const std::uint8_t* input, std::uint8_t* output, const AddToSaturationConfig& config, cudaStream_t stream = nullptr);
}
