#pragma once

#include <cstddef>
#include <cstdint>
#include "augmatch/core/image.hpp"
#include "augmatch/core/status.hpp"

#ifndef AUGMATCH_HAS_CUDA
#define AUGMATCH_HAS_CUDA 0
#endif
#if AUGMATCH_HAS_CUDA
#include <cuda_runtime_api.h>
#else
using cudaStream_t = void*;
#endif

namespace augmatch {

// Border behavior is a separate policy because view-based operations may be
// used by neighborhood noise stages. The additive operation below is pointwise;
// resolve_noise_coordinate is provided for host-side neighborhood metadata.
// How a neighborhood sample outside the image extent is resolved.
enum class NoiseBorderPolicy : std::uint8_t { Clamp, Reflect101, Constant };
// Whether output values are clamped to [clip_min, clip_max].
enum class NoiseClipPolicy : std::uint8_t { None, Clamp };
// How a normalized float result is converted back to an integer sample.
enum class NoiseQuantizationPolicy : std::uint8_t {
  Preserve,
  RoundHalfUp,
  Floor,
  Ceil,
  Truncate,
};

// Per-channel gain, bias, and noise-stddev scale, plus the Bayer/CFA plane
// that channel belongs to (or -1 when the channel index is used directly).
struct NoiseChannelMetadata {
  float gain = 1.0f;
  float bias = 0.0f;
  float stddev_scale = 1.0f;
  int plane = -1;
};

// Per-plane gain, bias, and noise-stddev scale, indexed by plane number.
struct NoisePlaneMetadata {
  float gain = 1.0f;
  float bias = 0.0f;
  float stddev_scale = 1.0f;
};

// Metadata is borrowed. channel_planes, when present, has channels entries
// and maps each image channel to a plane record. Otherwise channel index is
// used as the plane index. Metadata resides in metadata_memory and must remain
// valid through a CUDA stream operation.
struct NoiseMetadataView {
  const NoiseChannelMetadata* channels = nullptr;
  std::size_t channel_count = 0;
  const NoisePlaneMetadata* planes = nullptr;
  std::size_t plane_count = 0;
  const int* channel_planes = nullptr;
  MemorySpace metadata_memory = MemorySpace::Host;
};

using NoiseStatus = Status;
using NoiseStatusCode = StatusCode;

// Additive-noise parameters shared by additive_noise_view: a seeded Gaussian
// draw with the given mean/stddev, clipping, quantization, and border policy.
struct NoiseViewConfig {
  std::uint64_t seed = 0;
  float mean = 0.0f;
  float stddev = 0.0f;
  NoiseClipPolicy clip = NoiseClipPolicy::Clamp;
  float clip_min = 0.0f;
  float clip_max = 1.0f;
  NoiseQuantizationPolicy quantization = NoiseQuantizationPolicy::Preserve;
  NoiseBorderPolicy border = NoiseBorderPolicy::Clamp;
  float border_value = 0.0f;
};

// A small counter-based generator. It has no mutable state, so traversal order,
// thread scheduling, and repeated calls cannot change a sample's random value.
struct CounterRng {
  std::uint64_t seed = 0;
  std::uint64_t stream = 0;
  std::uint64_t counter(std::uint64_t x, std::uint64_t y,
                        std::uint64_t channel, std::uint64_t plane = 0) const noexcept;
  float uniform(std::uint64_t x, std::uint64_t y,
                std::uint64_t channel, std::uint64_t plane = 0) const noexcept;
  float normal(std::uint64_t x, std::uint64_t y,
               std::uint64_t channel, std::uint64_t plane = 0) const noexcept;
};

// Resolve one integer coordinate according to the declared border policy.
// For Constant, *is_constant is set and the return value is -1.
int resolve_noise_coordinate(int coordinate, int extent, NoiseBorderPolicy policy,
                             bool* is_constant = nullptr) noexcept;

// Validates view shapes/dtypes, the noise config, and optional metadata
// counts before an additive_noise_view call; returns a non-OK Status on the
// first violation found.
Status validate_noise_view(const ImageView& input, const MutableImageView& output,
                          const NoiseViewConfig& config,
                          const NoiseMetadataView& metadata = {}) noexcept;

// Applies deterministic additive noise to any supported host view (uint8,
// uint16, or float32; HWC or CHW), including strided views. Integer inputs are
// normalized by their representable maximum and integer outputs use the
// selected quantizer followed by saturating conversion. Device views are
// implemented by the CUDA backend; host metadata helpers return Unsupported
// rather than dereferencing device memory in a CPU build.
Status additive_noise_view(const ImageView& input, const MutableImageView& output,
                          const NoiseViewConfig& config,
                          const NoiseMetadataView& metadata = {},
                          cudaStream_t stream = nullptr) noexcept;

inline Status apply_noise_view(const ImageView& input, const MutableImageView& output,
                               const NoiseViewConfig& config,
                               const NoiseMetadataView& metadata = {},
                               cudaStream_t stream = nullptr) noexcept {
  return additive_noise_view(input, output, config, metadata, stream);
}

}  // namespace augmatch
