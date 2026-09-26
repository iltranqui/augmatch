#pragma once
#include <cstddef>
#include <cstdint>
#include <limits>
#ifndef AUGMATCH_HAS_CUDA
#define AUGMATCH_HAS_CUDA 0
#endif
#if AUGMATCH_HAS_CUDA
#include <cuda_runtime_api.h>
#else
using cudaStream_t = void*;
#endif
#include "augmatch/annotations/transforms.hpp"

namespace augmatch {
// A borrowed, contiguous HWC uint8 image. The descriptor is explicit so a
// multi-image operation cannot accidentally infer shape from a raw pointer.
struct ImageSourceU8 {
  const std::uint8_t* data = nullptr;
  int width = 0;
  int height = 0;
  int channels = 0;
};
struct ImageBatchU8 {
  const ImageSourceU8* sources = nullptr;
  std::size_t count = 0;
};
// Metadata is never allocated, retained, or resized by this library. Callers
// provide capacity and own the output arrays; count is written on success.
// Image-only Mosaic, TemplateTransform, and OverlayElements leave caller target
// records untouched; callers must transform/merge those records explicitly.
struct MixingTargets {
  BoxXYXY* boxes = nullptr;
  std::int32_t* labels = nullptr;
  std::size_t capacity = 0;
  std::size_t count = 0;
};
struct ReadOnlyMixingTargets {
  const BoxXYXY* boxes = nullptr;
  const std::int32_t* labels = nullptr;
  std::size_t count = 0;
};

// MixUp blends first and second with weight (or a seed-derived weight when
// sample_weight is set); weight is the second image's contribution.
struct MixUpConfig {
  int width = 0, height = 0, channels = 0;
  float weight = 0.5f;  // weight of the second image
  std::uint64_t seed = 0;
  bool sample_weight = false;  // deterministic U[0,1] from seed when true
};
void mixup_u8(const ImageSourceU8& first, const ImageSourceU8& second,
              std::uint8_t* output, const MixUpConfig& config,
              cudaStream_t stream = nullptr);
// Concatenates first and second's target metadata into output, weighting
// second's contribution by second_weight.
void mixup_targets(const ReadOnlyMixingTargets& first, const ReadOnlyMixingTargets& second,
                  float second_weight, MixingTargets& output);

// CutMix pastes patch into base within box (or a deterministic centred square
// sized by box_fraction when sample_box is set).
struct CutMixConfig {
  int width = 0, height = 0, channels = 0;
  BoxXYXY box{};  // explicit half-open output coordinates
  std::uint64_t seed = 0;
  bool sample_box = false;  // deterministic centred square when true
  float box_fraction = 0.5f;
};
void cutmix_u8(const ImageSourceU8& base, const ImageSourceU8& patch,
               std::uint8_t* output, const CutMixConfig& config,
               cudaStream_t stream = nullptr);
// Clips patch's target metadata to the selected box and merges it with base's
// metadata into output.
void cutmix_targets(const ReadOnlyMixingTargets& base, const ReadOnlyMixingTargets& patch,
                    const CutMixConfig& config, MixingTargets& output);

// Mosaic maps four source images into output quadrants split at
// (center_x,center_y), using nearest-neighbour scaling; -1 samples the split deterministically.
struct MosaicConfig {
  int width = 0, height = 0, channels = 0;
  int center_x = -1, center_y = -1;  // output split; -1 samples deterministically
  std::uint64_t seed = 0;
};
void mosaic_u8(const ImageBatchU8& sources, std::uint8_t* output,
              const MosaicConfig& config, cudaStream_t stream = nullptr);

// TemplateTransform blends image with a weighted combination of templates,
// normalizing image_weight and template_weights.
struct TemplateTransformConfig {
  int width = 0, height = 0, channels = 0;
  float image_weight = 1.0f;
  const float* template_weights = nullptr;  // borrowed count entries
  std::size_t template_count = 0;
};
void template_transform_u8(const ImageSourceU8& image, const ImageBatchU8& templates,
                           std::uint8_t* output, const TemplateTransformConfig& config,
                           cudaStream_t stream = nullptr);

// One element to composite: image is pasted at (x,y) (clipped to output),
// blended by alpha and the optional per-pixel mask.
struct OverlayElementU8 {
  ImageSourceU8 image{};
  int x = 0, y = 0;  // destination top-left, clipped to output
  float alpha = 1.0f;
  const std::uint8_t* mask = nullptr;  // optional HxW alpha bytes, borrowed
};
struct OverlayElementsConfig {
  int width = 0, height = 0, channels = 0;
  const OverlayElementU8* elements = nullptr;  // borrowed descriptor array
  std::size_t element_count = 0;
};
void overlay_elements_u8(const ImageSourceU8& base, std::uint8_t* output,
                         const OverlayElementsConfig& config,
                         cudaStream_t stream = nullptr);
}
