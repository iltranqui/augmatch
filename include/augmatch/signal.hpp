#pragma once
#include <cstdint>
#include "augmatch/iso_profile.hpp"
#ifndef AUGMATCH_HAS_CUDA
#define AUGMATCH_HAS_CUDA 0
#endif
#if AUGMATCH_HAS_CUDA
#include <cuda_runtime_api.h>
#else
using cudaStream_t = void*;
#endif
namespace augmatch {
struct SignalGainConfig { int width=0,height=0,channels=0; float gain=1; };
void exposure_scale_u8(const std::uint8_t*,std::uint8_t*,const SignalGainConfig&,cudaStream_t stream=nullptr);
void analog_gain_u8(const std::uint8_t*,std::uint8_t*,const SignalGainConfig&,cudaStream_t stream=nullptr);
void digital_gain_u8(const std::uint8_t*,std::uint8_t*,const SignalGainConfig&,cudaStream_t stream=nullptr);
struct SignalLevelConfig { int width=0,height=0,channels=0; int level=0; };
void pixel_saturation_u8(const std::uint8_t*,std::uint8_t*,const SignalLevelConfig&,cudaStream_t stream=nullptr);
void black_level_u8(const std::uint8_t*,std::uint8_t*,const SignalLevelConfig&,cudaStream_t stream=nullptr);
void adc_quantize_u8(const std::uint8_t*,std::uint8_t*,const SignalLevelConfig&,cudaStream_t stream=nullptr);
void bit_reduce_u8(const std::uint8_t*,std::uint8_t*,const SignalLevelConfig&,cudaStream_t stream=nullptr);
struct SpatialArtifactConfig { int width=0,height=0,channels=0; float coefficient=0; };
void pixel_cross_talk_u8(const std::uint8_t*,std::uint8_t*,const SpatialArtifactConfig&,cudaStream_t stream=nullptr);
void charge_leakage_u8(const std::uint8_t*,std::uint8_t*,const SpatialArtifactConfig&,cudaStream_t stream=nullptr);
struct ChannelGainConfig { int width=0,height=0,channels=0; float gain0=1,gain1=1,gain2=1; };
void channel_gain_u8(const std::uint8_t*,std::uint8_t*,const ChannelGainConfig&,cudaStream_t stream=nullptr);
}
