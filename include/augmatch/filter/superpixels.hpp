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

// Deterministic regular-grid superpixel averaging. The image is partitioned
// into non-overlapping cells [x0,x1) x [y0,y1), with x0 and y0 multiples of
// cell_width and cell_height. Partial cells at the right and bottom edges are
// included. Every output pixel in a cell receives the same per-channel value:
// integer_sum / pixel_count, rounded half up as (sum + count / 2) / count.
// Channels are independent, and the input and output buffers must not overlap.
struct SuperpixelConfig {
  int width = 0;
  int height = 0;
  int channels = 0;
  int cell_width = 16;
  int cell_height = 16;
};
void superpixels_u8(const std::uint8_t* input, std::uint8_t* output, const SuperpixelConfig& config, cudaStream_t stream = nullptr);

}
