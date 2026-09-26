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

// Shared HWC uint8 dimensions plus brightness/contrast parameters for
// brightness_contrast_u8.
struct BrightnessContrastConfig {
  int width = 0;
  int height = 0;
  int channels = 0;
  float brightness = 0.0f;  // additive term, applied as brightness*255 before contrast scaling
  float contrast = 1.0f;    // multiplicative gain, must be >= 0
};
// Applies output = saturate(contrast*input + brightness*255) per byte. Input
// and output are host pointers in CPU builds and device pointers in CUDA
// builds.
void brightness_contrast_u8(const std::uint8_t* input, std::uint8_t* output, const BrightnessContrastConfig& config, cudaStream_t stream = nullptr);

// Shared HWC uint8 dimensions plus the gamma exponent for gamma_u8.
struct GammaConfig {
  int width = 0;
  int height = 0;
  int channels = 0;
  float gamma = 1.0f;  // exponent, must be > 0
};
// Applies output = saturate(255*(input/255)^gamma) per byte. Input and output
// are host pointers in CPU builds and device pointers in CUDA builds.
void gamma_u8(const std::uint8_t* input, std::uint8_t* output, const GammaConfig& config, cudaStream_t stream = nullptr);

// Shared HWC uint8 dimensions for grayscale_u8.
struct GrayscaleConfig {
  int width = 0;
  int height = 0;
  int channels = 0;
};
// Computes luma y = 0.299*R + 0.587*G + 0.114*B per pixel and writes it back
// to every channel. Input and output are host pointers in CPU builds and
// device pointers in CUDA builds.
void grayscale_u8(const std::uint8_t* input, std::uint8_t* output, const GrayscaleConfig& config, cudaStream_t stream = nullptr);
}
