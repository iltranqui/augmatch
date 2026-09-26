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
namespace augmatch {

// Geometry uses top-left image coordinates. Pixel samples are at integer
// centres (0,0)..(width-1,height-1), while box coordinates are continuous
// half-open image edges [x1,x2) x [y1,y2). Crop is applied before flips.
// Interleaved HWC uint8 buffers are device pointers in CUDA builds and host
// pointers in the CPU fallback. Crop is applied before flips.
struct Geometry {
  int input_width;
  int input_height;
  int channels;
  int crop_x;    // top-left crop origin, x
  int crop_y;    // top-left crop origin, y
  int output_width;   // crop extent width
  int output_height;  // crop extent height
  bool flip_horizontal;
  bool flip_vertical;
};
void transform_u8(const std::uint8_t* input, std::uint8_t* output, const Geometry& geometry, cudaStream_t stream=nullptr);
void horizontal_flip_u8(const std::uint8_t* input, std::uint8_t* output, int width, int height, int channels, cudaStream_t stream=nullptr);
void vertical_flip_u8(const std::uint8_t* input, std::uint8_t* output, int width, int height, int channels, cudaStream_t stream=nullptr);
void flip_u8(const std::uint8_t* input, std::uint8_t* output, int width, int height, int channels, int axis, cudaStream_t stream=nullptr);
enum class PoolMode : int { Average=0, Maximum=1, Minimum=2, Median=3 };
struct Pooling {
  int width, height, channels, kernel_width, kernel_height;
  PoolMode mode;
  bool keep_size;
};
// HWC uint8 image pooling. This first port supports dimensions divisible by
// the kernel; keep_size restores the input dimensions with nearest sampling.
void pool_u8(const std::uint8_t* input, std::uint8_t* output, const Pooling& pooling, cudaStream_t stream=nullptr);
struct BoxXYXY { float x1,y1,x2,y2; };
struct PointXY { float x,y; };

// Masks are labels by default. Linear is exposed for callers whose mask is a
// continuous field; this Geometry contract has no scale, so both policies
// sample the same source pixel (future resampling must honour the policy).
enum class MaskInterpolation : int { Nearest=0, Linear=1 };
void transform_mask_u8(const std::uint8_t* input, std::uint8_t* output, const Geometry& geometry,
                       MaskInterpolation interpolation=MaskInterpolation::Nearest,
                       cudaStream_t stream=nullptr);

// Applies the Geometry crop/flip transform to a single point/box.
PointXY transform_point(PointXY point, const Geometry& geometry);
BoxXYXY transform_box(BoxXYXY box, const Geometry& geometry);
// Metadata helpers are host-side in both CPU and CUDA builds. Arrays must be
// host-owned; no implicit host/device copies are performed.
void transform_points(const PointXY* input, PointXY* output, std::size_t count, const Geometry& geometry);
void transform_boxes(const BoxXYXY* input, BoxXYXY* output, std::size_t count, const Geometry& geometry);

float box_area(BoxXYXY box) noexcept;
bool box_is_valid(BoxXYXY box) noexcept;
BoxXYXY clip_box(BoxXYXY box, int width, int height) noexcept;

struct BoxTransformOptions {
  bool clip=true;
  float min_visibility=0.0f; // retained area / original area, in [0,1]
  float min_area=0.0f;
};
// Transforms, optionally clips, and compacts valid boxes. The returned count
// is the number written to output; labels can be compacted using the same
// source-index policy by applying this function to a parallel array.
std::size_t transform_boxes_filtered(const BoxXYXY* input, BoxXYXY* output, std::size_t count,
                                     const Geometry& geometry, const BoxTransformOptions& options={},
                                     std::size_t* output_indices=nullptr);
// Filters already-transformed boxes in-place order without changing them.
std::size_t filter_boxes(const BoxXYXY* input, BoxXYXY* output, std::size_t count,
                         float min_area=0.0f);
}
