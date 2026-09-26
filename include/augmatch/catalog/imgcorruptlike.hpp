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

// These APIs are deterministic native approximations of imgaug's
// augmenters.imgcorruptlike operations.  Inputs and outputs are contiguous
// interleaved HWC uint8 buffers.  CPU pointers and optional fields are host
// memory; CUDA pointers and fields are device memory and remain valid until
// the supplied stream completes.  No operation keeps random state.
struct SpeckleNoiseConfig {
  int width = 0;
  int height = 0;
  int channels = 0;
  float mean = 0.0f;
  float stddev = 0.1f;
  std::uint64_t seed = 0;
};
void speckle_noise_u8(const std::uint8_t* input, std::uint8_t* output, const SpeckleNoiseConfig& config, cudaStream_t stream = nullptr);

// Fog uses a deterministic per-pixel density field generated from seed.  The
// field is a borrowed HxW array when supplied, with values in [0,1].
struct FogConfig {
  int width = 0;
  int height = 0;
  int channels = 0;
  float density = 0.5f;
  float opacity = 1.0f;
  const float* field = nullptr;
  std::uint64_t seed = 0;
};
void fog_u8(const std::uint8_t* input, std::uint8_t* output, const FogConfig& config, cudaStream_t stream = nullptr);

// Frost is a cold white/blue veil driven by the same explicit field contract.
struct FrostConfig {
  int width = 0;
  int height = 0;
  int channels = 0;
  float intensity = 0.5f;
  float opacity = 1.0f;
  const float* field = nullptr;
  std::uint64_t seed = 0;
};
void frost_u8(const std::uint8_t* input, std::uint8_t* output, const FrostConfig& config, cudaStream_t stream = nullptr);

// Snow is sparse white accumulation.  density controls the occupied fraction;
// opacity controls the blend on occupied pixels.
struct SnowConfig {
  int width = 0;
  int height = 0;
  int channels = 0;
  float density = 0.5f;
  float opacity = 1.0f;
  const float* field = nullptr;
  std::uint64_t seed = 0;
};
void snow_u8(const std::uint8_t* input, std::uint8_t* output, const SnowConfig& config, cudaStream_t stream = nullptr);

// Contrast scales samples around the midpoint 127.5.  Brightness is a
// multiplicative exposure factor, matching the imgcorruptlike approximation.
struct ContrastConfig {
  int width = 0;
  int height = 0;
  int channels = 0;
  float factor = 1.0f;
};
void contrast_u8(const std::uint8_t* input, std::uint8_t* output, const ContrastConfig& config, cudaStream_t stream = nullptr);
struct BrightnessConfig {
  int width = 0;
  int height = 0;
  int channels = 0;
  float factor = 1.0f;
};
void brightness_u8(const std::uint8_t* input, std::uint8_t* output, const BrightnessConfig& config, cudaStream_t stream = nullptr);

// Saturate scales HSV saturation while retaining hue and value.  Channels
// beyond RGB are copied unchanged, as they are non-color data.
struct SaturateConfig {
  int width = 0;
  int height = 0;
  int channels = 0;
  float factor = 1.0f;
};
void saturate_u8(const std::uint8_t* input, std::uint8_t* output, const SaturateConfig& config, cudaStream_t stream = nullptr);

// Pixelate replaces each integer block with the nearest source sample at the
// block centre.  block_size==1 is the identity and no interpolation occurs.
struct PixelateConfig {
  int width = 0;
  int height = 0;
  int channels = 0;
  int block_size = 2;
};
void pixelate_u8(const std::uint8_t* input, std::uint8_t* output, const PixelateConfig& config, cudaStream_t stream = nullptr);

// Explicit catalog spellings are retained as aliases for source discoverability.
using ImgCorruptLikeSpeckleNoiseConfig = SpeckleNoiseConfig;
using ImgCorruptLikeFogConfig = FogConfig;
using ImgCorruptLikeFrostConfig = FrostConfig;
using ImgCorruptLikeSnowConfig = SnowConfig;
using ImgCorruptLikeContrastConfig = ContrastConfig;
using ImgCorruptLikeBrightnessConfig = BrightnessConfig;
using ImgCorruptLikeSaturateConfig = SaturateConfig;
using ImgCorruptLikePixelateConfig = PixelateConfig;
inline void imgcorruptlike_speckle_noise_u8(const std::uint8_t* i, std::uint8_t* o, const SpeckleNoiseConfig& c, cudaStream_t s = nullptr) { speckle_noise_u8(i, o, c, s); }
inline void imgcorruptlike_fog_u8(const std::uint8_t* i, std::uint8_t* o, const FogConfig& c, cudaStream_t s = nullptr) { fog_u8(i, o, c, s); }
inline void imgcorruptlike_frost_u8(const std::uint8_t* i, std::uint8_t* o, const FrostConfig& c, cudaStream_t s = nullptr) { frost_u8(i, o, c, s); }
inline void imgcorruptlike_snow_u8(const std::uint8_t* i, std::uint8_t* o, const SnowConfig& c, cudaStream_t s = nullptr) { snow_u8(i, o, c, s); }
inline void imgcorruptlike_contrast_u8(const std::uint8_t* i, std::uint8_t* o, const ContrastConfig& c, cudaStream_t s = nullptr) { contrast_u8(i, o, c, s); }
inline void imgcorruptlike_brightness_u8(const std::uint8_t* i, std::uint8_t* o, const BrightnessConfig& c, cudaStream_t s = nullptr) { brightness_u8(i, o, c, s); }
inline void imgcorruptlike_saturate_u8(const std::uint8_t* i, std::uint8_t* o, const SaturateConfig& c, cudaStream_t s = nullptr) { saturate_u8(i, o, c, s); }
inline void imgcorruptlike_pixelate_u8(const std::uint8_t* i, std::uint8_t* o, const PixelateConfig& c, cudaStream_t s = nullptr) { pixelate_u8(i, o, c, s); }

} // namespace augmatch
