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
struct BrightnessContrastConfig { int width=0,height=0,channels=0; float brightness=0.0f; float contrast=1.0f; };
void brightness_contrast_u8(const std::uint8_t*,std::uint8_t*,const BrightnessContrastConfig&,cudaStream_t stream=nullptr);
struct GammaConfig { int width=0,height=0,channels=0; float gamma=1.0f; };
void gamma_u8(const std::uint8_t*,std::uint8_t*,const GammaConfig&,cudaStream_t stream=nullptr);
struct GrayscaleConfig { int width=0,height=0,channels=0; };
void grayscale_u8(const std::uint8_t*,std::uint8_t*,const GrayscaleConfig&,cudaStream_t stream=nullptr);
}
