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
enum class BorderPolicy : int { Clamp=0, Constant=1 };
struct DerivativeConfig { int width=0,height=0,channels=0; float scale=1.0f; BorderPolicy border=BorderPolicy::Clamp; std::uint8_t border_value=0; };
struct LocalStatsConfig { int width=0,height=0,channels=0,radius=1; float scale=1.0f; BorderPolicy border=BorderPolicy::Clamp; std::uint8_t border_value=0; }; 
void local_variance_u8(const std::uint8_t*,std::uint8_t*,const LocalStatsConfig&,cudaStream_t stream=nullptr);
void local_entropy_u8(const std::uint8_t*,std::uint8_t*,const LocalStatsConfig&,cudaStream_t stream=nullptr);
void sobel_x_u8(const std::uint8_t*,std::uint8_t*,const DerivativeConfig&,cudaStream_t stream=nullptr);
void sobel_y_u8(const std::uint8_t*,std::uint8_t*,const DerivativeConfig&,cudaStream_t stream=nullptr);
void scharr_x_u8(const std::uint8_t*,std::uint8_t*,const DerivativeConfig&,cudaStream_t stream=nullptr);
void scharr_y_u8(const std::uint8_t*,std::uint8_t*,const DerivativeConfig&,cudaStream_t stream=nullptr);
void laplacian_u8(const std::uint8_t*,std::uint8_t*,const DerivativeConfig&,cudaStream_t stream=nullptr);
void gradient_magnitude_u8(const std::uint8_t*,std::uint8_t*,const DerivativeConfig&,cudaStream_t stream=nullptr);
}
