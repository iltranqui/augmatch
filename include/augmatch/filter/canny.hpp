#pragma once
#include <cstdint>
#include "augmatch/filter/derivative.hpp"
#ifndef AUGMATCH_HAS_CUDA
#define AUGMATCH_HAS_CUDA 0
#endif
#if AUGMATCH_HAS_CUDA
#include <cuda_runtime_api.h>
#else
using cudaStream_t = void*;
#endif
namespace augmatch {

// Canny consumes interleaved HWC uint8 pixels. One luminance plane is built
// from channel 0 for grayscale, or BT.601 (0.299, 0.587, 0.114) for RGB;
// channels beyond RGB (including alpha) are ignored. The binary edge mask is
// replicated to every output channel. Clamp is replicate-at-border and
// Constant uses border_value for samples outside the image. The Sobel aperture
// is 3, 5, or 7; thresholds are gradient magnitudes in the matching Sobel
// scale, with low_threshold <= high_threshold. No derivative API is changed.
struct CannyConfig {
  int width = 0;
  int height = 0;
  int channels = 0;
  float low_threshold = 50.0f;
  float high_threshold = 150.0f;
  int aperture_size = 3;
  BorderPolicy border = BorderPolicy::Clamp;
  std::uint8_t border_value = 0;
};
// Applies deterministic Canny edge detection with hysteresis thresholding.
// Input/output are host pointers in CPU builds and device pointers in CUDA builds.
void canny_u8(const std::uint8_t* input, std::uint8_t* output, const CannyConfig& config, cudaStream_t stream = nullptr);

}
