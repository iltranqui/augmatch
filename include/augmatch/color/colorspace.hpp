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
// RGB is interleaved sRGB uint8, HSV uses OpenCV bins (H=[0,180), S/V=[0,255]),
// and LAB uses uint8 CIELAB encoding (L*=0..100 mapped to 0..255, a*/b* +128).
enum class ColorSpace : int { RGB = 0, HSV = 1, LAB = 2 };
using Colorspace = ColorSpace;
// WithColorspace is the deterministic replacement for imgaug's child augmenter:
// convert RGB to the selected space, apply y=round_half_up(x*m+b) to each of
// its three channels, then convert back. RGB channels are transformed; channels
// >=3 (including alpha) are copied unchanged. LAB is an approximate D65 sRGB
// conversion and is not an ICC/color-management implementation.
struct WithColorspaceConfig {
  int width = 0, height = 0, channels = 0;
  ColorSpace colorspace = ColorSpace::RGB;
  float multiplier0 = 1.0f, multiplier1 = 1.0f, multiplier2 = 1.0f;
  float addend0 = 0.0f, addend1 = 0.0f, addend2 = 0.0f;
};
void with_colorspace_u8(const std::uint8_t* input, std::uint8_t* output, const WithColorspaceConfig& config, cudaStream_t stream = nullptr);

// WithBrightnessChannels applies the same explicit affine operation to selected
// channels in RGB or HSV. RGB accepts indices 0..2; HSV accepts only V (index 2)
// for a brightness operation. Unselected channels and channels >=3 are copied.
// Set channel0 to -1 for an explicit no-op; channel1/channel2 may also be -1.
struct WithBrightnessChannelsConfig {
  int width = 0, height = 0, channels = 0;
  ColorSpace colorspace = ColorSpace::HSV;
  int channel0 = 2, channel1 = -1, channel2 = -1;
  float multiplier = 1.0f, addend = 0.0f;
};
void with_brightness_channels_u8(const std::uint8_t* input, std::uint8_t* output, const WithBrightnessChannelsConfig& config, cudaStream_t stream = nullptr);
}
