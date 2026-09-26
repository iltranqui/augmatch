#pragma once
#include <cstddef>
#include <cstdint>
#ifndef AUGMATCH_HAS_CUDA
#define AUGMATCH_HAS_CUDA 0
#endif
#if AUGMATCH_HAS_CUDA
#include <cuda_runtime_api.h>
#else
using cudaStream_t = void*;
#endif
#include "augmatch/transforms.hpp"
namespace augmatch {
// Blend contracts: all image buffers are contiguous HWC uint8 with width*height*channels
// bytes, source and overlay are borrowed read-only and output is caller-owned. CPU pointers
// are host pointers; CUDA pointers are device pointers and remain valid until stream completion.
// Pixels are rounded half-up after out=(1-a)*source+a*overlay, with a clamped to [0,1].
struct BlendAlphaConfig { int width=0,height=0,channels=0; float alpha=0.5f; };
void blend_alpha_u8(const std::uint8_t*,const std::uint8_t*,std::uint8_t*,const BlendAlphaConfig&,cudaStream_t stream=nullptr);
// Mask is one uint8 value per HWC pixel. mask_scale converts it to alpha (the default is 1/255).
struct BlendAlphaMaskConfig { int width=0,height=0,channels=0; float mask_scale=1.0f/255.0f; const std::uint8_t* mask=nullptr; };
void blend_alpha_mask_u8(const std::uint8_t*,const std::uint8_t*,std::uint8_t*,const BlendAlphaMaskConfig&,cudaStream_t stream=nullptr);
void blend_alpha_mask_u8(const std::uint8_t*,const std::uint8_t*,const std::uint8_t*,std::uint8_t*,const BlendAlphaMaskConfig&,cudaStream_t stream=nullptr);
// Elementwise alpha has one uint8 alpha per pixel; a null mask generates a deterministic
// SplitMix64 field in [min_alpha,max_alpha].
struct BlendAlphaElementwiseConfig { int width=0,height=0,channels=0; float min_alpha=0.0f,max_alpha=1.0f; const std::uint8_t* mask=nullptr; float mask_scale=1.0f/255.0f; std::uint64_t seed=0; };
void blend_alpha_elementwise_u8(const std::uint8_t*,const std::uint8_t*,std::uint8_t*,const BlendAlphaElementwiseConfig&,cudaStream_t stream=nullptr);
void blend_alpha_elementwise_u8(const std::uint8_t*,const std::uint8_t*,const std::uint8_t*,std::uint8_t*,const BlendAlphaElementwiseConfig&,cudaStream_t stream=nullptr);
// Noise fields are deterministic value-field approximations. Octaves are integer and frequency
// is in cycles per image; they intentionally document a portable replacement for imgaug noise.
struct BlendAlphaSimplexNoiseConfig { int width=0,height=0,channels=0; float min_alpha=0,max_alpha=1; float frequency=4; int octaves=3; std::uint64_t seed=0; };
void blend_alpha_simplex_noise_u8(const std::uint8_t*,const std::uint8_t*,std::uint8_t*,const BlendAlphaSimplexNoiseConfig&,cudaStream_t stream=nullptr);
struct BlendAlphaFrequencyNoiseConfig { int width=0,height=0,channels=0; float min_alpha=0,max_alpha=1; float frequency=8; float amplitude=1; std::uint64_t seed=0; };
void blend_alpha_frequency_noise_u8(const std::uint8_t*,const std::uint8_t*,std::uint8_t*,const BlendAlphaFrequencyNoiseConfig&,cudaStream_t stream=nullptr);
// SomeColors selects pixels whose channel values are within tolerance of any borrowed RGB
// record. colors has color_count*channels entries and is host/device memory per API backend.
struct BlendAlphaSomeColorsConfig { int width=0,height=0,channels=0; float alpha=1; float tolerance=0; const std::uint8_t* colors=nullptr; int color_count=0; };
void blend_alpha_some_colors_u8(const std::uint8_t*,const std::uint8_t*,std::uint8_t*,const BlendAlphaSomeColorsConfig&,cudaStream_t stream=nullptr);
struct BlendAlphaGradientConfig { int width=0,height=0,channels=0; float alpha_start=0,alpha_end=1; };
void blend_alpha_horizontal_linear_gradient_u8(const std::uint8_t*,const std::uint8_t*,std::uint8_t*,const BlendAlphaGradientConfig&,cudaStream_t stream=nullptr);
void blend_alpha_vertical_linear_gradient_u8(const std::uint8_t*,const std::uint8_t*,std::uint8_t*,const BlendAlphaGradientConfig&,cudaStream_t stream=nullptr);
// Grid and checkerboard masks use pixel coordinates and are deterministic (no RNG).
struct BlendAlphaRegularGridConfig { int width=0,height=0,channels=0; int cell_width=8,cell_height=8; float alpha=1; bool offset=false; };
void blend_alpha_regular_grid_u8(const std::uint8_t*,const std::uint8_t*,std::uint8_t*,const BlendAlphaRegularGridConfig&,cudaStream_t stream=nullptr);
struct BlendAlphaCheckerboardConfig { int width=0,height=0,channels=0; int cell_width=8,cell_height=8; float alpha=1; bool invert=false; };
void blend_alpha_checkerboard_u8(const std::uint8_t*,const std::uint8_t*,std::uint8_t*,const BlendAlphaCheckerboardConfig&,cudaStream_t stream=nullptr);
// Segmentation metadata is caller-owned. class_ids is a host/device array of class labels;
// the HxW uint8 seg_map is the target metadata and is never modified.
struct BlendAlphaSegMapClassIdsConfig { int width=0,height=0,channels=0; float alpha=1; const std::uint8_t* seg_map=nullptr; const std::int32_t* class_ids=nullptr; int class_count=0; };
void blend_alpha_seg_map_class_ids_u8(const std::uint8_t*,const std::uint8_t*,std::uint8_t*,const BlendAlphaSegMapClassIdsConfig&,cudaStream_t stream=nullptr);
// Bounding boxes use continuous half-open x1,y1,x2,y2 image-edge coordinates. The records
// are borrowed and ownership remains with the caller; only pixels whose centres are covered
// by at least one valid record are blended.
struct BlendAlphaBoundingBoxesConfig { int width=0,height=0,channels=0; float alpha=1; const BoxXYXY* boxes=nullptr; int box_count=0; };
void blend_alpha_bounding_boxes_u8(const std::uint8_t*,const std::uint8_t*,std::uint8_t*,const BlendAlphaBoundingBoxesConfig&,cudaStream_t stream=nullptr);
}
