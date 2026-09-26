#pragma once
#include <cstdint>
#include "augmatch/sensor/iso_profile.hpp"
#ifndef AUGMATCH_HAS_CUDA
#define AUGMATCH_HAS_CUDA 0
#endif
#if AUGMATCH_HAS_CUDA
#include <cuda_runtime_api.h>
#else
using cudaStream_t = void*;
#endif
namespace augmatch {

// Shared HWC uint8 dimensions plus a single multiplicative gain, used by
// exposure_scale_u8, analog_gain_u8, and digital_gain_u8.
struct SignalGainConfig {
  int width = 0;
  int height = 0;
  int channels = 0;
  float gain = 1;  // multiplicative gain, must be >= 0
};
// Applies output = saturate(input*gain) per byte. Input and output are host
// pointers in CPU builds and device pointers in CUDA builds.
void exposure_scale_u8(const std::uint8_t* input, std::uint8_t* output, const SignalGainConfig& config, cudaStream_t stream = nullptr);
// Applies output = saturate(input*gain) per byte. Input and output are host
// pointers in CPU builds and device pointers in CUDA builds.
void analog_gain_u8(const std::uint8_t* input, std::uint8_t* output, const SignalGainConfig& config, cudaStream_t stream = nullptr);
// Applies output = saturate(input*gain) per byte. Input and output are host
// pointers in CPU builds and device pointers in CUDA builds.
void digital_gain_u8(const std::uint8_t* input, std::uint8_t* output, const SignalGainConfig& config, cudaStream_t stream = nullptr);

// Shared HWC uint8 dimensions plus an integer level parameter, whose meaning
// depends on the function it configures (saturation ceiling, black-level
// offset, ADC quantization levels, or retained bit depth).
struct SignalLevelConfig {
  int width = 0;
  int height = 0;
  int channels = 0;
  int level = 0;
};
// Clamps each byte to at most level (level must be in [0,255]). Input and
// output are host pointers in CPU builds and device pointers in CUDA builds.
void pixel_saturation_u8(const std::uint8_t* input, std::uint8_t* output, const SignalLevelConfig& config, cudaStream_t stream = nullptr);
// Adds level to each byte and clamps to [0,255]. Input and output are host
// pointers in CPU builds and device pointers in CUDA builds.
void black_level_u8(const std::uint8_t* input, std::uint8_t* output, const SignalLevelConfig& config, cudaStream_t stream = nullptr);
// Quantizes each byte to level distinct steps (level must be in [2,256]).
// Input and output are host pointers in CPU builds and device pointers in
// CUDA builds.
void adc_quantize_u8(const std::uint8_t* input, std::uint8_t* output, const SignalLevelConfig& config, cudaStream_t stream = nullptr);
// Masks each byte down to level retained high bits (level must be in [1,8]).
// Input and output are host pointers in CPU builds and device pointers in
// CUDA builds.
void bit_reduce_u8(const std::uint8_t* input, std::uint8_t* output, const SignalLevelConfig& config, cudaStream_t stream = nullptr);

// Shared HWC uint8 dimensions plus a blend coefficient in [0,1], used by
// pixel_cross_talk_u8 and charge_leakage_u8.
struct SpatialArtifactConfig {
  int width = 0;
  int height = 0;
  int channels = 0;
  float coefficient = 0;  // blend weight toward the neighboring sample(s), in [0,1]
};
// Blends each pixel with the mean of its 4-connected neighbors:
// output = (1-coefficient)*pixel + coefficient*mean(neighbors). Input and
// output are host pointers in CPU builds and device pointers in CUDA builds.
void pixel_cross_talk_u8(const std::uint8_t* input, std::uint8_t* output, const SpatialArtifactConfig& config, cudaStream_t stream = nullptr);
// Blends each pixel with its left neighbor:
// output = (1-coefficient)*pixel + coefficient*left. Input and output are
// host pointers in CPU builds and device pointers in CUDA builds.
void charge_leakage_u8(const std::uint8_t* input, std::uint8_t* output, const SpatialArtifactConfig& config, cudaStream_t stream = nullptr);

// Shared HWC uint8 dimensions plus a per-channel gain (channel index >= 2
// reuses gain2), used by channel_gain_u8.
struct ChannelGainConfig {
  int width = 0;
  int height = 0;
  int channels = 0;
  float gain0 = 1;
  float gain1 = 1;
  float gain2 = 1;
};
// Applies output = saturate(input*gain[channel]) per byte, selecting gain0,
// gain1, or gain2 by channel index. Input and output are host pointers in
// CPU builds and device pointers in CUDA builds.
void channel_gain_u8(const std::uint8_t* input, std::uint8_t* output, const ChannelGainConfig& config, cudaStream_t stream = nullptr);
}
