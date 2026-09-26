#pragma once
#include <cstdint>
#include <cstddef>
#include "augmatch/geometry/geometric.hpp"
#include "augmatch/core/image.hpp"
#include "augmatch/annotations/transforms.hpp"
#ifndef AUGMATCH_HAS_CUDA
#define AUGMATCH_HAS_CUDA 0
#endif
#if AUGMATCH_HAS_CUDA
#include <cuda_runtime_api.h>
#else
using cudaStream_t = void*;
#endif
namespace augmatch {
// Crops a fixed-size rectangle centered on the input image.
struct CenterCropConfig {
  int input_width = 0;
  int input_height = 0;
  int output_width = 0;
  int output_height = 0;
  int channels = 0;
};
void center_crop_u8(const std::uint8_t* input, std::uint8_t* output, const CenterCropConfig& config, cudaStream_t stream = nullptr);

// Crops a fixed-size rectangle with an explicit top-left offset (left,top).
struct RandomCropConfig {
  int input_width = 0, input_height = 0, output_width = 0, output_height = 0, channels = 0, left = 0, top = 0;
};
void random_crop_u8(const std::uint8_t* input, std::uint8_t* output, const RandomCropConfig& config, cudaStream_t stream = nullptr);

// RandomResizedCrop and RandomSizedCrop use the same explicit geometry contract:
// crop_x/crop_y are the source top-left, crop_width/crop_height are the source
// rectangle, and output_width/output_height are the resized result dimensions.
// Set crop_x or crop_y to -1 to generate that offset from seed. Set both crop
// dimensions to zero to generate a non-empty rectangle from seed; otherwise
// dimensions must be explicit. Generated dimensions and offsets use SplitMix64
// seed + indices 0..3 (width, height, x, y), modulo the available range.
// The output is exactly resize_u8(rectangle, output dimensions, interpolation),
// so Nearest, Linear, and Area follow the existing Resize contract.
struct RandomResizedCropConfig {
  int input_width = 0, input_height = 0, output_width = 0, output_height = 0, channels = 0;
  int crop_x = 0, crop_y = 0, crop_width = 0, crop_height = 0;
  Interpolation interpolation = Interpolation::Linear;
  std::uint64_t seed = 0;
};
using RandomSizedCropConfig = RandomResizedCropConfig;
// A resolved integer crop rectangle (top-left plus extent).
struct RandomResizedCropRectangle {
  int x = 0, y = 0, width = 0, height = 0;
};
// Resolves the source rectangle that random_resized_crop_u8 will use.
RandomResizedCropRectangle make_random_resized_crop_rectangle(const RandomResizedCropConfig& config);
void random_resized_crop_u8(const std::uint8_t* input, std::uint8_t* output, const RandomResizedCropConfig& config, cudaStream_t stream = nullptr);
void random_sized_crop_u8(const std::uint8_t* input, std::uint8_t* output, const RandomSizedCropConfig& config, cudaStream_t stream = nullptr);

// RandomCropFromBorders removes an independently selected integer number of
// pixels from each edge. Each crop_* value is a maximum fraction of the
// corresponding input dimension; its pixel maximum is floor(fraction * size).
// An offset of -1 requests deterministic generation from seed, while a
// nonnegative offset is explicit and takes precedence over seed. Generated
// offsets use splitmix64(seed + side_index) modulo (maximum + 1), with side
// order left, right, top, bottom. No global RNG state is used. The maximum
// possible crop must leave at least one pixel in each dimension.
struct RandomCropFromBordersConfig {
  int input_width = 0, input_height = 0, channels = 0;
  float crop_left = 0.0f, crop_right = 0.0f, crop_top = 0.0f, crop_bottom = 0.0f;
  int left_offset = -1, right_offset = -1, top_offset = -1, bottom_offset = -1;
  std::uint64_t seed = 0;
};
// The resolved pixel offsets removed from each edge.
struct RandomCropFromBordersOffsets {
  int left = 0, right = 0, top = 0, bottom = 0;
};
// Resolves the per-edge pixel offsets that random_crop_from_borders_u8 will use.
RandomCropFromBordersOffsets make_random_crop_from_borders_offsets(const RandomCropFromBordersConfig& config);
void random_crop_from_borders_u8(const std::uint8_t* input, std::uint8_t* output, const RandomCropFromBordersConfig& config, cudaStream_t stream = nullptr);

// RandomCropNearBBox expands an explicit Pascal VOC box by independently
// selected margins. Box coordinates are continuous image-edge coordinates in
// x1,y1,x2,y2 order; pixel crops use floor(x1,y1) and ceil(x2,y2), then clip
// the resulting half-open rectangle to the image. max_part_shift_* are the
// maximum margin as a fraction of the corresponding box extent. An offset of
// -1 requests deterministic SplitMix64 generation from seed (side order
// left, right, top, bottom); nonnegative margins are explicit pixel offsets.
struct RandomCropNearBBoxConfig {
  int input_width = 0, input_height = 0, channels = 0;
  BoxXYXY bbox{};
  float max_part_shift_x = 0.0f, max_part_shift_y = 0.0f;
  int left_offset = -1, right_offset = -1, top_offset = -1, bottom_offset = -1;
  std::uint64_t seed = 0;
};
// The resolved per-side margins added to the box before clipping.
struct RandomCropNearBBoxOffsets {
  int left = 0, right = 0, top = 0, bottom = 0;
};
// A resolved integer crop rectangle (top-left plus extent).
struct RandomCropNearBBoxRectangle {
  int x = 0, y = 0, width = 0, height = 0;
};
// Resolves the per-side margins that random_crop_near_bbox_u8 will use.
RandomCropNearBBoxOffsets make_random_crop_near_bbox_offsets(const RandomCropNearBBoxConfig& config);
// Resolves the clipped crop rectangle that random_crop_near_bbox_u8 will use.
RandomCropNearBBoxRectangle make_random_crop_near_bbox_rectangle(const RandomCropNearBBoxConfig& config);
void random_crop_near_bbox_u8(const std::uint8_t* input, std::uint8_t* output, const RandomCropNearBBoxConfig& config, cudaStream_t stream = nullptr);

// Target-aware crops use a borrowed host array of continuous Pascal-VOC boxes.
// Factories select one integer half-open image rectangle and return metadata;
// image functions copy only pixels (or resize pixels) and never mutate boxes.
// Callers own the box array and must transform annotations using the returned
// source rectangle and, for RandomSizedBBoxSafeCrop, its scale.
struct BBoxSafeRandomCropConfig {
  int input_width = 0, input_height = 0, channels = 0;
  const BoxXYXY* boxes = nullptr;
  std::size_t box_count = 0;
  int min_crop_width = 1, min_crop_height = 1;
  int max_crop_width = 0, max_crop_height = 0;  // zero means the input extent
  float erosion_rate = 0.0f;  // contracts the required envelope toward its center
  std::uint64_t seed = 0;
};
// The resolved crop rectangle; used_fallback is set when no box constrained it.
struct BBoxSafeRandomCropRectangle {
  int x = 0, y = 0, width = 0, height = 0;
  bool used_fallback = false;
};
BBoxSafeRandomCropRectangle make_bbox_safe_random_crop_rectangle(const BBoxSafeRandomCropConfig& config);
void bbox_safe_random_crop_u8(const std::uint8_t* input, std::uint8_t* output, const BBoxSafeRandomCropConfig& config, cudaStream_t stream = nullptr);

// Like BBoxSafeRandomCrop, but resizes the selected rectangle to
// output_width/output_height, reporting the applied scale.
struct RandomSizedBBoxSafeCropConfig {
  int input_width = 0, input_height = 0, output_width = 0, output_height = 0, channels = 0;
  const BoxXYXY* boxes = nullptr;
  std::size_t box_count = 0;
  float erosion_rate = 0.0f;
  Interpolation interpolation = Interpolation::Linear;
  std::uint64_t seed = 0;
};
// The resolved source rectangle plus the scale applied when resizing it to
// output_width/output_height.
struct RandomSizedBBoxSafeCropRectangle {
  int x = 0, y = 0, width = 0, height = 0, output_width = 0, output_height = 0;
  float scale_x = 1.0f, scale_y = 1.0f;
  bool used_fallback = false;
};
RandomSizedBBoxSafeCropRectangle make_random_sized_bbox_safe_crop_rectangle(const RandomSizedBBoxSafeCropConfig& config);
void random_sized_bbox_safe_crop_u8(const std::uint8_t* input, std::uint8_t* output, const RandomSizedBBoxSafeCropConfig& config, cudaStream_t stream = nullptr);

// AtLeastOneBBoxRandomCrop intersects one deterministically selected box with
// a fixed-size crop rectangle, guaranteeing that box is covered when possible.
struct AtLeastOneBBoxRandomCropConfig {
  int input_width = 0, input_height = 0, crop_width = 0, crop_height = 0, channels = 0;
  const BoxXYXY* boxes = nullptr;
  std::size_t box_count = 0;
  float erosion_factor = 0.0f;  // contracts the selected box before intersection
  int fallback_x = -1, fallback_y = -1;
  std::uint64_t seed = 0;
};
// The resolved crop rectangle, the selected box index, and whether the
// explicit/seeded fallback was used instead of a box.
struct AtLeastOneBBoxRandomCropRectangle {
  int x = 0, y = 0, width = 0, height = 0;
  std::size_t selected_box = static_cast<std::size_t>(-1);
  bool used_fallback = false;
};
AtLeastOneBBoxRandomCropRectangle make_at_least_one_bbox_random_crop_rectangle(const AtLeastOneBBoxRandomCropConfig& config);
void at_least_one_bbox_random_crop_u8(const std::uint8_t* input, std::uint8_t* output, const AtLeastOneBBoxRandomCropConfig& config, cudaStream_t stream = nullptr);

// CropNonEmptyMaskIfExists selects a fixed crop containing one nonzero mask
// pixel. Candidate pixels are visited row-major; the selected candidate and
// valid top-left positions are chosen with SplitMix64(seed + 0), +1, and +2.
// This is deterministic and does not use global RNG state. If mask is absent
// or empty, fallback_x/fallback_y are used when nonnegative, otherwise the
// same seed contract selects an ordinary random crop. mask_threshold is an
// inclusive background threshold: values greater than it are targets.
// The native view overload copies image and mask without interpolation (the
// mask therefore has nearest/label semantics), and all views/buffers are
// borrowed and remain owned by the caller.
struct CropNonEmptyMaskIfExistsConfig {
  int input_width = 0, input_height = 0, channels = 0, mask_channels = 1;
  int crop_width = 0, crop_height = 0;
  int fallback_x = -1, fallback_y = -1;
  std::uint8_t mask_threshold = 0;
  std::uint64_t seed = 0;
};
// The resolved crop rectangle; used_mask is set when a nonzero mask pixel
// selected the crop instead of the fallback/seed path.
struct CropNonEmptyMaskIfExistsRectangle {
  int x = 0, y = 0, width = 0, height = 0;
  bool used_mask = false;
};
CropNonEmptyMaskIfExistsRectangle make_crop_non_empty_mask_if_exists_rectangle(const ImageMaskView& view, const CropNonEmptyMaskIfExistsConfig& config);
void crop_non_empty_mask_if_exists(const ImageMaskView& view, const MutableImageMaskView& output, const CropNonEmptyMaskIfExistsConfig& config, cudaStream_t stream = nullptr);
void crop_non_empty_mask_if_exists_u8(const std::uint8_t* image, const std::uint8_t* mask, std::uint8_t* output_image, std::uint8_t* output_mask, const CropNonEmptyMaskIfExistsConfig& config, cudaStream_t stream = nullptr);

enum class PadBorder : int { Constant = 0, Replicate = 1, Reflect101 = 2 };
// Pads or crops an HWC image to explicit output dimensions using the
// configured border policy and constant fill value.
struct PadConfig {
  int input_width = 0, input_height = 0, output_width = 0, output_height = 0, channels = 0;
  std::uint8_t value = 0;
  PadBorder border = PadBorder::Constant;
};
// Size-catalog geometry uses integer dimensions and an explicit anchor.  A
// top-left operation keeps the origin fixed; Center splits odd differences by
// putting floor((target-source)/2) pixels before the source.  Padding uses the
// selected border/fill policy, while cropping is a half-open source rectangle.
enum class SizeAnchor : int { TopLeft = 0, Center = 1 };
enum class SizeRounding : int { Floor = 0, Ceil = 1, Nearest = 2 };
// Pads or crops so each dimension becomes a multiple of multiple_width/multiple_height.
struct MultiplesOfConfig {
  int input_width = 0, input_height = 0, channels = 0;
  int multiple_width = 1, multiple_height = 1;
  std::uint8_t value = 0;
  PadBorder border = PadBorder::Constant;
  SizeAnchor anchor = SizeAnchor::TopLeft;
};
// Pads or crops so each dimension becomes a power of base.
struct PowersOfConfig {
  int input_width = 0, input_height = 0, channels = 0;
  int base = 2;
  std::uint8_t value = 0;
  PadBorder border = PadBorder::Constant;
  SizeAnchor anchor = SizeAnchor::TopLeft;
};
// Pads or crops so width/height matches the configured aspect ratio.
struct AspectRatioConfig {
  int input_width = 0, input_height = 0, channels = 0;
  float aspect_ratio = 1.0f;
  SizeRounding rounding = SizeRounding::Nearest;
  std::uint8_t value = 0;
  PadBorder border = PadBorder::Constant;
  SizeAnchor anchor = SizeAnchor::TopLeft;
};
// A resolved target size for a size-catalog pad/crop transform.
struct SizeTransformDimensions {
  int width = 0, height = 0;
};
using PadToMultiplesOfConfig = MultiplesOfConfig;
using CropToMultiplesOfConfig = MultiplesOfConfig;
using PadToPowersOfConfig = PowersOfConfig;
using CropToPowersOfConfig = PowersOfConfig;
using PadToAspectRatioConfig = AspectRatioConfig;
using CropToAspectRatioConfig = AspectRatioConfig;
SizeTransformDimensions make_pad_to_multiples_of_dimensions(const MultiplesOfConfig& config);
SizeTransformDimensions make_crop_to_multiples_of_dimensions(const MultiplesOfConfig& config);
SizeTransformDimensions make_pad_to_powers_of_dimensions(const PowersOfConfig& config);
SizeTransformDimensions make_crop_to_powers_of_dimensions(const PowersOfConfig& config);
SizeTransformDimensions make_pad_to_aspect_ratio_dimensions(const AspectRatioConfig& config);
SizeTransformDimensions make_crop_to_aspect_ratio_dimensions(const AspectRatioConfig& config);
void pad_to_multiples_of_u8(const std::uint8_t* input, std::uint8_t* output, const MultiplesOfConfig& config, cudaStream_t stream = nullptr);
void crop_to_multiples_of_u8(const std::uint8_t* input, std::uint8_t* output, const MultiplesOfConfig& config, cudaStream_t stream = nullptr);
void pad_to_powers_of_u8(const std::uint8_t* input, std::uint8_t* output, const PowersOfConfig& config, cudaStream_t stream = nullptr);
void crop_to_powers_of_u8(const std::uint8_t* input, std::uint8_t* output, const PowersOfConfig& config, cudaStream_t stream = nullptr);
void pad_to_aspect_ratio_u8(const std::uint8_t* input, std::uint8_t* output, const AspectRatioConfig& config, cudaStream_t stream = nullptr);
void crop_to_aspect_ratio_u8(const std::uint8_t* input, std::uint8_t* output, const AspectRatioConfig& config, cudaStream_t stream = nullptr);
// Center-anchored variants of the pad/crop-to-multiples-of transforms above.
void center_pad_to_multiples_of_u8(const std::uint8_t* input, std::uint8_t* output, MultiplesOfConfig config, cudaStream_t stream = nullptr);
void center_crop_to_multiples_of_u8(const std::uint8_t* input, std::uint8_t* output, MultiplesOfConfig config, cudaStream_t stream = nullptr);
// Center-anchored variants of the pad/crop-to-powers-of transforms above.
void center_pad_to_powers_of_u8(const std::uint8_t* input, std::uint8_t* output, PowersOfConfig config, cudaStream_t stream = nullptr);
void center_crop_to_powers_of_u8(const std::uint8_t* input, std::uint8_t* output, PowersOfConfig config, cudaStream_t stream = nullptr);
// Center-anchored variants of the pad/crop-to-aspect-ratio transforms above.
void center_pad_to_aspect_ratio_u8(const std::uint8_t* input, std::uint8_t* output, AspectRatioConfig config, cudaStream_t stream = nullptr);
void center_crop_to_aspect_ratio_u8(const std::uint8_t* input, std::uint8_t* output, AspectRatioConfig config, cudaStream_t stream = nullptr);

// KeepSizeByResize downsizes to an intermediate resolution and back, keeping
// the original input dimensions for the final output.
struct KeepSizeByResizeConfig {
  int input_width = 0, input_height = 0, channels = 0;
  int intermediate_width = 0, intermediate_height = 0;
  Interpolation interpolation = Interpolation::Linear;
};
void keep_size_by_resize_u8(const std::uint8_t* input, std::uint8_t* output, const KeepSizeByResizeConfig& config, cudaStream_t stream = nullptr);
// Convenience overload: ResizeConfig's output dimensions are the intermediate
// shape; the final output always has ResizeConfig's input shape.
void keep_size_by_resize_u8(const std::uint8_t* input, std::uint8_t* output, const ResizeConfig& config, cudaStream_t stream = nullptr);

void pad_to_size_u8(const std::uint8_t* input, std::uint8_t* output, const PadConfig& config, cudaStream_t stream = nullptr);
void square_symmetric_pad_u8(const std::uint8_t* input, std::uint8_t* output, const PadConfig& config, cudaStream_t stream = nullptr);
void pad_to_fixed_size_u8(const std::uint8_t* input, std::uint8_t* output, const PadConfig& config, cudaStream_t stream = nullptr);
void center_pad_to_fixed_size_u8(const std::uint8_t* input, std::uint8_t* output, const PadConfig& config, cudaStream_t stream = nullptr);
void pad_to_square_u8(const std::uint8_t* input, std::uint8_t* output, const PadConfig& config, cudaStream_t stream = nullptr);
void center_pad_to_square_u8(const std::uint8_t* input, std::uint8_t* output, const PadConfig& config, cudaStream_t stream = nullptr);
void crop_to_fixed_size_u8(const std::uint8_t* input, std::uint8_t* output, const RandomCropConfig& config, cudaStream_t stream = nullptr);
void center_crop_to_fixed_size_u8(const std::uint8_t* input, std::uint8_t* output, const CenterCropConfig& config, cudaStream_t stream = nullptr);
void crop_to_square_u8(const std::uint8_t* input, std::uint8_t* output, const CenterCropConfig& config, cudaStream_t stream = nullptr);
void center_crop_to_square_u8(const std::uint8_t* input, std::uint8_t* output, const CenterCropConfig& config, cudaStream_t stream = nullptr);

// Crops and/or pads each edge independently: positive top/right/bottom/left
// values crop that many pixels, negative values pad that many pixels using value.
struct CropPadConfig {
  int input_width = 0, input_height = 0, channels = 0;
  int top = 0, right = 0, bottom = 0, left = 0;
  std::uint8_t value = 0;
};
void crop_and_pad_u8(const std::uint8_t* input, std::uint8_t* output, const CropPadConfig& config, cudaStream_t stream = nullptr);
}
