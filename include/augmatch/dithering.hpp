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

// Dithering modes are deterministic and do not use random state. OrderedBayer4x4
// adds the fixed 4x4 Bayer threshold before quantization. ErrorDiffusionFloydSteinberg
// scans rows left-to-right and diffuses quantization error with weights 7/16, 3/16,
// 5/16, and 1/16 to right, next-left, next, and next-right pixels respectively.
enum class DitheringMode : int {
  OrderedBayer4x4 = 0,
  ErrorDiffusionFloydSteinberg = 1,
  // Short names are kept as source-compatible aliases for callers that do not
  // need to spell out the matrix or error-diffusion variant.
  Ordered = OrderedBayer4x4,
  Bayer4x4 = OrderedBayer4x4,
  ErrorDiffusion = ErrorDiffusionFloydSteinberg,
  FloydSteinberg = ErrorDiffusionFloydSteinberg
};
using DitherMode = DitheringMode;

// bit_depth (also available as bits) is the number of output bits per channel
// and is restricted to [1,8].
// The output levels are evenly spaced in [0,255]. Input and output are contiguous
// interleaved HWC uint8 buffers; every channel is dithered independently.
struct DitheringConfig {
  int width = 0;
  int height = 0;
  int channels = 0;
  union {
    int bit_depth = 8;
    int bits;
  };
  DitheringMode mode = DitheringMode::OrderedBayer4x4;
};
using DitherConfig = DitheringConfig;

void dithering_u8(const std::uint8_t*, std::uint8_t*, const DitheringConfig&, cudaStream_t stream = nullptr);
// Spelling alias for callers that use the verb form.
void dither_u8(const std::uint8_t*, std::uint8_t*, const DitheringConfig&, cudaStream_t stream = nullptr);

}  // namespace augmatch
