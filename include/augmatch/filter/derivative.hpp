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

enum class BorderPolicy : int { Clamp = 0, Constant = 1 };
struct DerivativeConfig {
  int width = 0;
  int height = 0;
  int channels = 0;
  float scale = 1.0f;
  BorderPolicy border = BorderPolicy::Clamp;
  std::uint8_t border_value = 0;
};
struct LocalStatsConfig {
  int width = 0;
  int height = 0;
  int channels = 0;
  int radius = 1;
  float scale = 1.0f;
  BorderPolicy border = BorderPolicy::Clamp;
  std::uint8_t border_value = 0;
};
// Computes local variance over the (2*radius+1) square window per channel.
// Input/output are host pointers in CPU builds and device pointers in CUDA builds.
void local_variance_u8(const std::uint8_t* input, std::uint8_t* output, const LocalStatsConfig& config, cudaStream_t stream = nullptr);
// Computes local Shannon entropy over the (2*radius+1) square window per channel.
// Input/output are host pointers in CPU builds and device pointers in CUDA builds.
void local_entropy_u8(const std::uint8_t* input, std::uint8_t* output, const LocalStatsConfig& config, cudaStream_t stream = nullptr);
// Applies the horizontal 3x3 Sobel derivative.
// Input/output are host pointers in CPU builds and device pointers in CUDA builds.
void sobel_x_u8(const std::uint8_t* input, std::uint8_t* output, const DerivativeConfig& config, cudaStream_t stream = nullptr);
// Applies the vertical 3x3 Sobel derivative.
// Input/output are host pointers in CPU builds and device pointers in CUDA builds.
void sobel_y_u8(const std::uint8_t* input, std::uint8_t* output, const DerivativeConfig& config, cudaStream_t stream = nullptr);
// Applies the horizontal 3x3 Scharr derivative.
// Input/output are host pointers in CPU builds and device pointers in CUDA builds.
void scharr_x_u8(const std::uint8_t* input, std::uint8_t* output, const DerivativeConfig& config, cudaStream_t stream = nullptr);
// Applies the vertical 3x3 Scharr derivative.
// Input/output are host pointers in CPU builds and device pointers in CUDA builds.
void scharr_y_u8(const std::uint8_t* input, std::uint8_t* output, const DerivativeConfig& config, cudaStream_t stream = nullptr);
// Applies the 3x3 Laplacian second derivative.
// Input/output are host pointers in CPU builds and device pointers in CUDA builds.
void laplacian_u8(const std::uint8_t* input, std::uint8_t* output, const DerivativeConfig& config, cudaStream_t stream = nullptr);
// Computes the Sobel gradient magnitude.
// Input/output are host pointers in CPU builds and device pointers in CUDA builds.
void gradient_magnitude_u8(const std::uint8_t* input, std::uint8_t* output, const DerivativeConfig& config, cudaStream_t stream = nullptr);

}
