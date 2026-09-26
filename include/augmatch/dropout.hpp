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
struct PixelDropoutConfig { int width=0,height=0,channels=0; float probability=0.0f; std::uint8_t replacement=0; std::uint64_t seed=0; };
void pixel_dropout_u8(const std::uint8_t*,std::uint8_t*,const PixelDropoutConfig&,cudaStream_t stream=nullptr);
struct ChannelDropoutConfig { int width=0,height=0,channels=0; float probability=0.0f; std::uint8_t replacement=0; std::uint64_t seed=0; };
void channel_dropout_u8(const std::uint8_t*,std::uint8_t*,const ChannelDropoutConfig&,cudaStream_t stream=nullptr);
struct GridDropoutConfig { int width=0,height=0,channels=0; int cell_width=2,cell_height=2; float ratio=0.5f; std::uint8_t fill=0; };
void grid_dropout_u8(const std::uint8_t*,std::uint8_t*,const GridDropoutConfig&,cudaStream_t stream=nullptr);
struct CoarseDropoutConfig { int width=0,height=0,channels=0; int holes=1,hole_width=1,hole_height=1; std::uint8_t fill=0; std::uint64_t seed=0; };
void coarse_dropout_u8(const std::uint8_t*,std::uint8_t*,const CoarseDropoutConfig&,cudaStream_t stream=nullptr);
// MaskDropout consumes one uint8 mask byte per HWC pixel. Zero keeps the pixel;
// every nonzero value drops it. The fill byte replaces every channel of dropped
// pixels, and no random state is used, so identical inputs and masks are exact.
struct MaskDropoutConfig { int width=0,height=0,channels=0; std::uint8_t fill=0; };
void mask_dropout_u8(const std::uint8_t* in,const std::uint8_t* mask,std::uint8_t* out,const MaskDropoutConfig&,cudaStream_t stream=nullptr);

// A half-open interval [begin,end) selects rows or columns. The interval arrays
// are borrowed: they reside in host memory for the CPU API and device memory for
// the CUDA API. A pixel is masked when its row is in any row interval OR its
// column is in any column interval. Masked pixels have every channel replaced by
// fill; unmasked pixels are copied unchanged. Empty interval lists are valid.
struct XYMaskInterval { int begin=0,end=0; };
struct XYMaskingConfig {
  int width=0,height=0,channels=0;
  const XYMaskInterval* row_intervals=nullptr;
  int row_interval_count=0;
  const XYMaskInterval* column_intervals=nullptr;
  int column_interval_count=0;
  std::uint8_t fill=0;
};
void xy_masking_u8(const std::uint8_t* in,std::uint8_t* out,const XYMaskingConfig&,cudaStream_t stream=nullptr);

// Cutout applies an ordered list of integer half-open rectangles [x0,x1) x
// [y0,y1). Rectangle and configuration arrays are borrowed: CPU callers own
// host memory and CUDA callers own device memory for the duration of the call.
// Rectangles must be inside the image; an empty list is an exact identity.
// Every covered channel receives the scalar fill value. No random state is
// consumed; seed is reserved for compatibility and is ignored for explicit
// rectangles.
struct CutoutRectangle { int x0=0,y0=0,x1=0,y1=0; };
using CutoutRect = CutoutRectangle;
struct CutoutConfig {
  int width=0,height=0,channels=0;
  const CutoutRectangle* rectangles=nullptr;
  int rectangle_count=0;
  std::uint8_t fill=0;
  std::uint64_t seed=0;
};
void cutout_u8(const std::uint8_t* in,std::uint8_t* out,const CutoutConfig&,cudaStream_t stream=nullptr);

// TotalDropout either copies the image or replaces every channel with fill.
// With an explicit one-byte mask, mask[0]!=0 selects replacement and
// probability/seed are ignored. The mask is borrowed host memory for CPU and
// borrowed device memory for CUDA. Without a mask, one deterministic
// SplitMix64 draw keyed by seed selects replacement when it is below
// probability; probability must be in [0,1].
struct TotalDropoutConfig {
  int width=0,height=0,channels=0;
  float probability=1.0f;
  std::uint8_t fill=0;
  std::uint64_t seed=0;
  const std::uint8_t* mask=nullptr;
};
void total_dropout_u8(const std::uint8_t* in,std::uint8_t* out,const TotalDropoutConfig&,cudaStream_t stream=nullptr);
}
