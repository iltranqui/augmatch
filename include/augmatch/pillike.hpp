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
// Deterministic Pillow ImageEnhance contracts. factor is finite and may be
// outside [0,1]. RGB channels use Pillow's integer 299/587/114 luma;
// channels after RGB (including alpha) are copied unchanged.
struct EnhanceColorConfig { int width=0,height=0,channels=0; float factor=1.0f; };
struct EnhanceContrastConfig { int width=0,height=0,channels=0; float factor=1.0f; };
struct EnhanceBrightnessConfig { int width=0,height=0,channels=0; float factor=1.0f; };
struct EnhanceSharpnessConfig { int width=0,height=0,channels=0; float factor=1.0f; };
void enhance_color_u8(const std::uint8_t*,std::uint8_t*,const EnhanceColorConfig&,cudaStream_t stream=nullptr);
void enhance_contrast_u8(const std::uint8_t*,std::uint8_t*,const EnhanceContrastConfig&,cudaStream_t stream=nullptr);
void enhance_brightness_u8(const std::uint8_t*,std::uint8_t*,const EnhanceBrightnessConfig&,cudaStream_t stream=nullptr);
void enhance_sharpness_u8(const std::uint8_t*,std::uint8_t*,const EnhanceSharpnessConfig&,cudaStream_t stream=nullptr);
// Exact Pillow built-in filters. Borders are copied unchanged (Pillow's
// convolution kernel is applied only where its complete support is present).
// Kernel division uses nearest integer (half-up for BLUR/SMOOTH and a fixed
// ties-to-even rule for SMOOTH_MORE), while enhancement blends truncate toward
// zero before clipping. This makes CPU and CUDA results deterministic; Pillow
// versions may differ by one at exact half ties.
// FilterBlur uses the 5x5 ring kernel/scale 16; FilterSmooth uses the 3x3
// kernel/scale 13; FilterSmoothMore uses Pillow's 5x5 kernel/scale 100.
// The remaining filters use Pillow's fixed 3x3 kernels and integer scales:
// EDGE_ENHANCE (10 center, scale 2), EDGE_ENHANCE_MORE (9 center, scale 1),
// FIND_EDGES (8 center, scale 1), CONTOUR (8 center, scale 1, offset 255),
// EMBOSS (diagonal -1/+1, scale 1, offset 128), SHARPEN (32 center/-2,
// scale 16), and DETAIL (cross -1/center 10, scale 6). All kernels are
// applied only where their complete support is present; border pixels are
// copied unchanged. Every interleaved channel, including alpha, is filtered,
// matching Pillow ImageFilter (unlike ImageEnhance APIs above, which preserve
// channels after RGB). Convolution sums use explicit integer nearest-value
// rounding (half-up for these fixed kernels; SMOOTH_MORE retains its historical
// ties-to-even rule) and then add the filter offset before clipping to [0,255].
struct FilterBlurConfig { int width=0,height=0,channels=0; };
struct FilterSmoothConfig { int width=0,height=0,channels=0; };
struct FilterSmoothMoreConfig { int width=0,height=0,channels=0; };
struct FilterEdgeEnhanceConfig { int width=0,height=0,channels=0; };
struct FilterEdgeEnhanceMoreConfig { int width=0,height=0,channels=0; };
struct FilterFindEdgesConfig { int width=0,height=0,channels=0; };
struct FilterContourConfig { int width=0,height=0,channels=0; };
struct FilterEmbossConfig { int width=0,height=0,channels=0; };
struct FilterSharpenConfig { int width=0,height=0,channels=0; };
struct FilterDetailConfig { int width=0,height=0,channels=0; };
void filter_blur_u8(const std::uint8_t*,std::uint8_t*,const FilterBlurConfig&,cudaStream_t stream=nullptr);
void filter_smooth_u8(const std::uint8_t*,std::uint8_t*,const FilterSmoothConfig&,cudaStream_t stream=nullptr);
void filter_smooth_more_u8(const std::uint8_t*,std::uint8_t*,const FilterSmoothMoreConfig&,cudaStream_t stream=nullptr);
void filter_edge_enhance_u8(const std::uint8_t*,std::uint8_t*,const FilterEdgeEnhanceConfig&,cudaStream_t stream=nullptr);
void filter_edge_enhance_more_u8(const std::uint8_t*,std::uint8_t*,const FilterEdgeEnhanceMoreConfig&,cudaStream_t stream=nullptr);
void filter_find_edges_u8(const std::uint8_t*,std::uint8_t*,const FilterFindEdgesConfig&,cudaStream_t stream=nullptr);
void filter_contour_u8(const std::uint8_t*,std::uint8_t*,const FilterContourConfig&,cudaStream_t stream=nullptr);
void filter_emboss_u8(const std::uint8_t*,std::uint8_t*,const FilterEmbossConfig&,cudaStream_t stream=nullptr);
void filter_sharpen_u8(const std::uint8_t*,std::uint8_t*,const FilterSharpenConfig&,cudaStream_t stream=nullptr);
void filter_detail_u8(const std::uint8_t*,std::uint8_t*,const FilterDetailConfig&,cudaStream_t stream=nullptr);
}
