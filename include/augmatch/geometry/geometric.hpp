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
enum class Interpolation : int { Nearest = 0, Linear = 1, Area = 2 };

// Resizes an HWC uint8 image from the input dimensions to the output
// dimensions using the configured interpolation.
struct ResizeConfig {
  int input_width = 0;
  int input_height = 0;
  int output_width = 0;
  int output_height = 0;
  int channels = 0;
  Interpolation interpolation = Interpolation::Linear;
};
// Host pointers in CPU builds, device pointers in CUDA builds.
void resize_u8(const std::uint8_t* input, std::uint8_t* output, const ResizeConfig& config, cudaStream_t stream = nullptr);

// Resizes an HWC uint8 image by the explicit scale factor using the
// configured interpolation.
struct ScaleConfig {
  int input_width = 0;
  int input_height = 0;
  int output_width = 0;
  int output_height = 0;
  int channels = 0;
  float scale = 1.0f;
  Interpolation interpolation = Interpolation::Linear;
};
void random_scale_u8(const std::uint8_t* input, std::uint8_t* output, const ScaleConfig& config, cudaStream_t stream = nullptr);

// Downscale performs a deterministic low-resolution round trip. The reduced
// dimensions are floor(width * scale) and floor(height * scale), clamped to one;
// the first resize uses interpolation and the second (upsampling) resize uses
// upsample_interpolation. No codec or compression library is involved.
struct DownscaleConfig {
  int width = 0, height = 0, channels = 0;
  float scale = 0.5f;
  Interpolation interpolation = Interpolation::Linear;
  Interpolation upsample_interpolation = Interpolation::Linear;
};
void downscale_u8(const std::uint8_t* input, std::uint8_t* output, const DownscaleConfig& config, cudaStream_t stream = nullptr);

// Scales the image horizontally about a fixed anchor, filling exposed
// columns with the configured fill value.
struct AxisScaleConfig {
  int width = 0;
  int height = 0;
  int channels = 0;
  float scale = 1.0f;
  Interpolation interpolation = Interpolation::Linear;
  std::uint8_t fill = 0;
};
void scale_x_u8(const std::uint8_t* input, std::uint8_t* output, const AxisScaleConfig& config, cudaStream_t stream = nullptr);
// Scales the image vertically; see scale_x_u8 for the shared contract.
void scale_y_u8(const std::uint8_t* input, std::uint8_t* output, const AxisScaleConfig& config, cudaStream_t stream = nullptr);

// Translates the image along one axis by the configured offset, filling
// exposed pixels with the configured fill value.
struct AxisTranslateConfig {
  int width = 0;
  int height = 0;
  int channels = 0;
  float offset = 0.0f;
  Interpolation interpolation = Interpolation::Linear;
  std::uint8_t fill = 0;
};
void translate_x_u8(const std::uint8_t* input, std::uint8_t* output, const AxisTranslateConfig& config, cudaStream_t stream = nullptr);
// Translates the image vertically; see translate_x_u8 for the shared contract.
void translate_y_u8(const std::uint8_t* input, std::uint8_t* output, const AxisTranslateConfig& config, cudaStream_t stream = nullptr);

// Shears the image along one axis by the configured shear factor, filling
// exposed pixels with the configured fill value.
struct AxisShearConfig {
  int width = 0;
  int height = 0;
  int channels = 0;
  float shear = 0.0f;
  Interpolation interpolation = Interpolation::Linear;
  std::uint8_t fill = 0;
};
void shear_x_u8(const std::uint8_t* input, std::uint8_t* output, const AxisShearConfig& config, cudaStream_t stream = nullptr);
// Shears the image vertically; see shear_x_u8 for the shared contract.
void shear_y_u8(const std::uint8_t* input, std::uint8_t* output, const AxisShearConfig& config, cudaStream_t stream = nullptr);

// Transposes an HWC uint8 image (swaps rows and columns); the output
// dimensions are the input dimensions with width and height swapped.
struct TransposeConfig {
  int input_width = 0;
  int input_height = 0;
  int channels = 0;
};
void transpose_u8(const std::uint8_t* input, std::uint8_t* output, const TransposeConfig& config, cudaStream_t stream = nullptr);

// Rotates the image by k * 90 degrees.
struct Rotate90Config {
  int width = 0, height = 0, channels = 0, k = 0;
};
void rotate90_u8(const std::uint8_t* input, std::uint8_t* output, const Rotate90Config& config, cudaStream_t stream = nullptr);

// Resizes so the longer of width/height equals max_size, preserving aspect ratio.
struct MaxSizeConfig {
  int input_width = 0;
  int input_height = 0;
  int output_width = 0;
  int output_height = 0;
  int max_size = 0;
  int channels = 0;
  Interpolation interpolation = Interpolation::Area;
};
void longest_max_size_u8(const std::uint8_t* input, std::uint8_t* output, const MaxSizeConfig& config, cudaStream_t stream = nullptr);
// Resizes so the shorter of width/height equals max_size, preserving aspect ratio.
void smallest_max_size_u8(const std::uint8_t* input, std::uint8_t* output, const MaxSizeConfig& config, cudaStream_t stream = nullptr);

// Applies an explicit 2x3 affine matrix [m00 m01 m02; m10 m11 m12] mapping
// output coordinates to input coordinates, filling exposed pixels with fill.
struct AffineConfig {
  int input_width = 0;
  int input_height = 0;
  int output_width = 0;
  int output_height = 0;
  int channels = 0;
  float m00 = 1, m01 = 0, m02 = 0, m10 = 0, m11 = 1, m12 = 0;
  Interpolation interpolation = Interpolation::Linear;
  std::uint8_t fill = 0;
};
void affine_u8(const std::uint8_t* input, std::uint8_t* output, const AffineConfig& config, cudaStream_t stream = nullptr);

// Focus breathing is a deterministic focus-dependent radial magnification map.
// focus_position is an explicit frame parameter; zero is an exact identity.
// At normalized radius r, the magnification is
// scale = 1 + focus_position * (breathing_strength + radial_strength*r*r),
// and each output pixel samples the inverse radial map about (center_x,center_y).
// Coordinates outside the source use fill, and uint8 samples are saturating
// half-up rounded. This is an affine/geometric approximation, not a calibrated
// lens model; interpolation must be Nearest or Linear (not Area).
struct FocusBreathingConfig {
  int width = 0, height = 0, channels = 0;
  float focus_position = 0.0f, breathing_strength = 0.0f, radial_strength = 0.0f;
  float center_x = 0.5f, center_y = 0.5f;
  Interpolation interpolation = Interpolation::Linear;
  std::uint8_t fill = 0;
};
void focus_breathing_u8(const std::uint8_t* input, std::uint8_t* output, const FocusBreathingConfig& config, cudaStream_t stream = nullptr);

// Applies an explicit 3x3 homography [h00 h01 h02; h10 h11 h12; h20 h21 h22]
// mapping output coordinates to input coordinates, filling exposed pixels with fill.
struct PerspectiveConfig {
  int width = 0;
  int height = 0;
  int channels = 0;
  float h00 = 1, h01 = 0, h02 = 0, h10 = 0, h11 = 1, h12 = 0, h20 = 0, h21 = 0, h22 = 1;
  Interpolation interpolation = Interpolation::Linear;
  std::uint8_t fill = 0;
};
void perspective_u8(const std::uint8_t* input, std::uint8_t* output, const PerspectiveConfig& config, cudaStream_t stream = nullptr);
// Explicit homography contract for imgaug-style perspective transforms.
void perspective_transform_u8(const std::uint8_t* input, std::uint8_t* output, const PerspectiveConfig& config, cudaStream_t stream = nullptr);

// Applies OpenCV-style radial (k1,k2) and tangential (p1,p2) lens distortion
// coefficients, filling exposed pixels with the configured fill value.
struct OpticalDistortionConfig {
  int width = 0;
  int height = 0;
  int channels = 0;
  float k1 = 0, k2 = 0, p1 = 0, p2 = 0;
  Interpolation interpolation = Interpolation::Linear;
  std::uint8_t fill = 0;
};
void optical_distortion_u8(const std::uint8_t* input, std::uint8_t* output, const OpticalDistortionConfig& config, cudaStream_t stream = nullptr);

// Thin-prism distortion uses the OpenCV thin-prism terms s1,s2,t1,t2
// against normalized radius powers r2 and r4. It is a deterministic HWC
// remap approximation with explicit interpolation and constant fill.
struct ThinPrismDistortionConfig {
  int width = 0;
  int height = 0;
  int channels = 0;
  float s1 = 0, s2 = 0, t1 = 0, t2 = 0;
  Interpolation interpolation = Interpolation::Linear;
  std::uint8_t fill = 0;
};
void thin_prism_distortion_u8(const std::uint8_t* input, std::uint8_t* output, const ThinPrismDistortionConfig& config, cudaStream_t stream = nullptr);

// Row shifts are borrowed H-element source displacements. Each output row uses
// (x+row_shift_x[y], y+row_shift_y[y]); CPU arrays are host-owned and CUDA
// arrays are device-owned until the supplied stream completes.
struct RollingShutterGeometricDistortionConfig {
  int width = 0;
  int height = 0;
  int channels = 0;
  const float* row_shift_x = nullptr;
  const float* row_shift_y = nullptr;
  Interpolation interpolation = Interpolation::Linear;
  std::uint8_t fill = 0;
};
void rolling_shutter_geometric_distortion_u8(const std::uint8_t* input, std::uint8_t* output, const RollingShutterGeometricDistortionConfig& config, cudaStream_t stream = nullptr);

// Explicit grid contract: displacement_x/y are row-major grid_width*grid_height
// vertex displacements in source pixels. Vertices span the image rectangle,
// and each output pixel samples input at (x + dx, y + dy). CPU callers provide
// host arrays; CUDA callers provide device arrays in the same configuration.
struct GridDistortionConfig {
  int width = 0, height = 0, channels = 0, grid_width = 0, grid_height = 0;
  const float* displacement_x = nullptr;
  const float* displacement_y = nullptr;
  Interpolation interpolation = Interpolation::Linear;
  std::uint8_t fill = 0;
};
void grid_distortion_u8(const std::uint8_t* input, std::uint8_t* output, const GridDistortionConfig& config, cudaStream_t stream = nullptr);

// Explicit piecewise-affine contract: displacement_x/y are row-major vertex
// displacements in source pixels. Vertices are uniformly spaced over the image
// rectangle. Each cell is split from its top-left to bottom-right vertex;
// points on the diagonal use the top/right/bottom triangle. Output (x,y)
// samples source (x+dx,y+dy). CPU callers provide host arrays;
// CUDA callers provide device arrays, matching GridDistortion ownership.
struct PiecewiseAffineConfig {
  int width = 0, height = 0, channels = 0, grid_width = 0, grid_height = 0;
  const float* displacement_x = nullptr;
  const float* displacement_y = nullptr;
  Interpolation interpolation = Interpolation::Linear;
  std::uint8_t fill = 0;
};
void piecewise_affine_u8(const std::uint8_t* input, std::uint8_t* output, const PiecewiseAffineConfig& config, cudaStream_t stream = nullptr);

// Explicit ElasticTransform contract: displacement_x/y are row-major per-pixel
// source displacements with width*height entries. Output (x,y) samples source
// (x + displacement_x[y*width+x], y + displacement_y[y*width+x]). CPU callers
// provide host arrays; CUDA callers provide device arrays. The arrays are
// borrowed until the CPU call returns or the CUDA stream completes, and are
// never copied by the library.
struct ElasticTransformConfig {
  int width = 0;
  int height = 0;
  int channels = 0;
  const float* displacement_x = nullptr;
  const float* displacement_y = nullptr;
  Interpolation interpolation = Interpolation::Linear;
  std::uint8_t fill = 0;
};
void elastic_transform_u8(const std::uint8_t* input, std::uint8_t* output, const ElasticTransformConfig& config, cudaStream_t stream = nullptr);

// RandomGridShuffle partitions the image into grid_rows x grid_cols cells and
// copies each source cell into one destination cell. Cell indices are row-major.
// cell_permutation[destination] is the source cell index and must be a bijection
// over [0, grid_rows * grid_cols). On CPU it is a borrowed host pointer until
// the call returns; on CUDA it is a borrowed device pointer until stream
// completion. If it is null, the library uses the deterministic seed contract:
// for i = cell_count - 1 down to 1, j = splitmix64(seed + i) % (i + 1), then
// swaps permutation[i] and permutation[j], starting from the identity order.
// No global or implicit RNG state is used. Grid boundaries are floor(i*size/
// grid_count); within each destination cell, source coordinates are scaled by
// the corresponding source-cell dimensions, preserving all pixels when sizes
// are not divisible by the grid dimensions.
struct RandomGridShuffleConfig {
  int width = 0, height = 0, channels = 0, grid_rows = 0, grid_cols = 0;
  const std::uint32_t* cell_permutation = nullptr;
  std::uint64_t seed = 0;
};
// Fills a caller-owned host permutation using the seed contract above.
void make_random_grid_shuffle_permutation(std::uint32_t* permutation, int grid_rows, int grid_cols, std::uint64_t seed);
void random_grid_shuffle_u8(const std::uint8_t* input, std::uint8_t* output, const RandomGridShuffleConfig& config, cudaStream_t stream = nullptr);

// Rotates the image by angle_degrees about its center, filling exposed
// pixels with the configured fill value; output keeps the input dimensions.
struct RotateConfig {
  int width = 0;
  int height = 0;
  int channels = 0;
  float angle_degrees = 0;
  Interpolation interpolation = Interpolation::Linear;
  std::uint8_t fill = 0;
};
void rotate_u8(const std::uint8_t* input, std::uint8_t* output, const RotateConfig& config, cudaStream_t stream = nullptr);

// Rotates the image by angle_degrees about its center like rotate_u8, but
// scales the content so the entire rotated image remains visible ("safe" rotate).
struct SafeRotateConfig {
  int width = 0;
  int height = 0;
  int channels = 0;
  float angle_degrees = 0;
  Interpolation interpolation = Interpolation::Linear;
  std::uint8_t fill = 0;
};
void safe_rotate_u8(const std::uint8_t* input, std::uint8_t* output, const SafeRotateConfig& config, cudaStream_t stream = nullptr);

// Combines a translation (shift_x,shift_y), a uniform scale, and a rotation
// about the center in that order, filling exposed pixels with fill.
struct ShiftScaleRotateConfig {
  int width = 0;
  int height = 0;
  int channels = 0;
  float shift_x = 0, shift_y = 0, scale = 1, angle_degrees = 0;
  Interpolation interpolation = Interpolation::Linear;
  std::uint8_t fill = 0;
};
void shift_scale_rotate_u8(const std::uint8_t* input, std::uint8_t* output, const ShiftScaleRotateConfig& config, cudaStream_t stream = nullptr);
}
