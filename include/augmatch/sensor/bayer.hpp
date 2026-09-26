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
enum class BayerPattern : int { RGGB = 0, BGGR = 1, GRBG = 2, GBRG = 3 };
struct BayerConfig {
  int width = 0;
  int height = 0;
  BayerPattern pattern = BayerPattern::RGGB;
};

// Raw Bayer plane ownership is explicit and stable: plane 0=R, 1=Gr (the
// green sample on the first row), 2=Gb (the green sample on the second row),
// and 3=B. The map is indexed by tile position [even-row/even-column,
// even-row/odd-column, odd-row/even-column, odd-row/odd-column]. It is copied
// into each configuration, so no metadata is retained by an operation.
struct CfaPlaneMap { std::uint8_t plane[4] = {0, 1, 2, 3}; };
// Returns the canonical CfaPlaneMap for a Bayer pattern.
CfaPlaneMap bayer_plane_map(BayerPattern pattern);
// Returns the raw plane index (0-3) at pixel (x, y) according to map.
int bayer_plane_at(int y, int x, const CfaPlaneMap& map);

enum class CfaMissingReplacement : int { Zero = 0, Fill = 1, Nearest = 2 };

// The following effects consume and produce a contiguous HxW float32 raw
// plane. Configurations own scalar parameters; pointer fields are borrowed
// host memory for CPU calls and device memory for CUDA calls until the stream
// operation completes. All outputs are clipped to [clip_min, clip_max].
struct CfaChannelResponseVariationConfig {
  int width = 0;
  int height = 0;
  CfaPlaneMap plane_map{};
  float response[4] = {1.0f, 1.0f, 1.0f, 1.0f};
  float variation_stddev = 0.0f;
  float clip_min = 0.0f;
  float clip_max = 1.0f;
  std::uint64_t seed = 0;
};
void cfa_channel_response_variation_f32(const float* input, float* output, const CfaChannelResponseVariationConfig& config, cudaStream_t stream = nullptr);

// leakage[p*4+q] is the fraction of the nearest raw sample belonging to
// source plane q contributing to target plane p. Nearest samples are searched
// in a deterministic radius-2 square; missing planes fall back to the center.
struct CfaLeakageConfig {
  int width = 0;
  int height = 0;
  CfaPlaneMap plane_map{};
  float leakage[16] = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
                       0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f};
  float clip_min = 0.0f;
  float clip_max = 1.0f;
};
void cfa_leakage_f32(const float* input, float* output, const CfaLeakageConfig& config, cudaStream_t stream = nullptr);

struct CfaMisregistrationConfig {
  int width = 0;
  int height = 0;
  CfaPlaneMap plane_map{};
  int dx[4] = {0, 0, 0, 0}, dy[4] = {0, 0, 0, 0};
  float clip_min = 0.0f;
  float clip_max = 1.0f;
};
void cfa_misregistration_f32(const float* input, float* output, const CfaMisregistrationConfig& config, cudaStream_t stream = nullptr);

struct CfaMissingSamplesConfig {
  int width = 0;
  int height = 0;
  CfaPlaneMap plane_map{};
  const std::uint8_t* missing_mask = nullptr;  // borrowed HxW mask; nonzero means missing
  float missing_probability = 0.0f;
  CfaMissingReplacement replacement = CfaMissingReplacement::Zero;
  float replacement_value = 0.0f;
  float clip_min = 0.0f;
  float clip_max = 1.0f;
  std::uint64_t seed = 0;
};
void cfa_missing_samples_f32(const float* input, float* output, const CfaMissingSamplesConfig& config, cudaStream_t stream = nullptr);

struct BayerPlaneNoiseConfig {
  int width = 0;
  int height = 0;
  CfaPlaneMap plane_map{};
  float stddev[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float clip_min = 0.0f;
  float clip_max = 1.0f;
  std::uint64_t seed = 0;
};
void bayer_plane_noise_f32(const float* input, float* output, const BayerPlaneNoiseConfig& config, cudaStream_t stream = nullptr);

struct BayerPlaneGainConfig {
  int width = 0;
  int height = 0;
  CfaPlaneMap plane_map{};
  float gain[4] = {1.0f, 1.0f, 1.0f, 1.0f};
  float clip_min = 0.0f;
  float clip_max = 1.0f;
};
void bayer_plane_gain_f32(const float* input, float* output, const BayerPlaneGainConfig& config, cudaStream_t stream = nullptr);

// Quad-Bayer expands each Bayer cell to a 2x2 block. Input is HWC RGB and
// output is one borrowed/owned raw plane; the raw plane has no channel field.
enum class QuadBayerPattern : int { RGGB = 0, BGGR = 1, GRBG = 2, GBRG = 3 };
struct QuadBayerConfig {
  int width = 0;
  int height = 0;
  QuadBayerPattern pattern = QuadBayerPattern::RGGB;
};
// Returns the input RGB channel index sampled at pixel (x, y) for pattern.
int quad_bayer_channel_at(int y, int x, QuadBayerPattern pattern);
void quad_bayer_sample_u8(const std::uint8_t* input, std::uint8_t* output, const QuadBayerConfig& config, cudaStream_t stream = nullptr);

// RGBW uses channel indices R=0, G=1, B=2, W=3. RGWB is the canonical
// 2x2 tile [R G; W B]; the other values are explicit tile variants.
enum class RGBWPattern : int { RGWB = 0, RWBG = 1, GBWR = 2, WGRB = 3 };
struct RGBWConfig {
  int width = 0;
  int height = 0;
  RGBWPattern pattern = RGBWPattern::RGWB;
};
// Returns the input RGBW channel index sampled at pixel (x, y) for pattern.
int rgbw_channel_at(int y, int x, RGBWPattern pattern);
void rgbw_sample_u8(const std::uint8_t* input, std::uint8_t* output, const RGBWConfig& config, cudaStream_t stream = nullptr);

// mask is a repeating mask_width x mask_height tile of channel indices. It is
// host-owned for the CPU backend and device-owned for CUDA until stream work
// completes. Input is HWC with `channels` planes and output is one raw plane.
struct CustomCfaConfig {
  int width = 0;
  int height = 0;
  int channels = 0;
  int mask_width = 0;
  int mask_height = 0;
  const std::uint8_t* mask = nullptr;
};
using CfaMaskConfig = CustomCfaConfig;
// Returns the input channel index sampled at pixel (x, y) from config's mask.
int custom_cfa_channel_at(int y, int x, const CustomCfaConfig& config);
void custom_cfa_sample_u8(const std::uint8_t* input, std::uint8_t* output, const CustomCfaConfig& config, cudaStream_t stream = nullptr);

void bayer_sample_u8(const std::uint8_t* input, std::uint8_t* output, const BayerConfig& config, cudaStream_t stream = nullptr);
void bayer_demosaic_nearest_u8(const std::uint8_t* input, std::uint8_t* output, const BayerConfig& config, cudaStream_t stream = nullptr);
void bayer_demosaic_bilinear_u8(const std::uint8_t* input, std::uint8_t* output, const BayerConfig& config, cudaStream_t stream = nullptr);
// Malvar-He-Cutler uses the published 5x5 integer-weight filters. Samples at
// the image boundary use clamp-to-edge coordinates; floating-point results are
// clipped to [0,255] and rounded to nearest uint8. The BayerPattern in config
// explicitly selects RGGB, BGGR, GRBG, or GBRG raw-plane ownership.
void bayer_demosaic_malvar_he_cutler_u8(const std::uint8_t* input, std::uint8_t* output, const BayerConfig& config, cudaStream_t stream = nullptr);
// Edge-aware interpolation compares clamped horizontal/vertical (or diagonal)
// raw gradients. The lower-gradient direction wins; ties average both.
void bayer_demosaic_edge_aware_u8(const std::uint8_t* input, std::uint8_t* output, const BayerConfig& config, cudaStream_t stream = nullptr);
// Explicit raw-plane convenience entry points for each canonical Bayer tile.
void bayer_demosaic_malvar_he_cutler_rggb_u8(const std::uint8_t* input, std::uint8_t* output, int width, int height, cudaStream_t stream = nullptr);
void bayer_demosaic_malvar_he_cutler_bggr_u8(const std::uint8_t* input, std::uint8_t* output, int width, int height, cudaStream_t stream = nullptr);
void bayer_demosaic_malvar_he_cutler_grbg_u8(const std::uint8_t* input, std::uint8_t* output, int width, int height, cudaStream_t stream = nullptr);
void bayer_demosaic_malvar_he_cutler_gbrg_u8(const std::uint8_t* input, std::uint8_t* output, int width, int height, cudaStream_t stream = nullptr);
void bayer_demosaic_edge_aware_rggb_u8(const std::uint8_t* input, std::uint8_t* output, int width, int height, cudaStream_t stream = nullptr);
void bayer_demosaic_edge_aware_bggr_u8(const std::uint8_t* input, std::uint8_t* output, int width, int height, cudaStream_t stream = nullptr);
void bayer_demosaic_edge_aware_grbg_u8(const std::uint8_t* input, std::uint8_t* output, int width, int height, cudaStream_t stream = nullptr);
void bayer_demosaic_edge_aware_gbrg_u8(const std::uint8_t* input, std::uint8_t* output, int width, int height, cudaStream_t stream = nullptr);

// Deterministic demosaicing-artifact injections. Each operation consumes a
// contiguous HxW uint8 Bayer plane and produces interleaved HxWx3 RGB8. The
// edge-aware demosaicer is the baseline; strength zero is therefore the
// baseline result. Parameters are borrowed by value, no state is retained,
// and outputs are clipped to clip_min..clip_max (normally 0..255).
struct DirectionalDemosaicingArtifactsConfig {
  int width = 0;
  int height = 0;
  BayerPattern pattern = BayerPattern::RGGB;
  int direction = 0;  // 0=gradient-selected, 1=horizontal, 2=vertical
  float strength = 0.0f;
  int clip_min = 0, clip_max = 255;
};
struct FalseColorZipperArtifactsConfig {
  int width = 0;
  int height = 0;
  BayerPattern pattern = BayerPattern::RGGB;
  float strength = 0.0f;
  int clip_min = 0, clip_max = 255;
};
struct DemosaicingAliasingConfig {
  int width = 0;
  int height = 0;
  BayerPattern pattern = BayerPattern::RGGB;
  int period = 2;
  float phase = 0.0f, strength = 0.0f;
  int clip_min = 0, clip_max = 255;
};
struct DemosaicingRingingConfig {
  int width = 0;
  int height = 0;
  BayerPattern pattern = BayerPattern::RGGB;
  float strength = 0.0f;
  int clip_min = 0, clip_max = 255;
};
struct DemosaicingNoiseAmplificationConfig {
  int width = 0;
  int height = 0;
  BayerPattern pattern = BayerPattern::RGGB;
  float noise_stddev = 0.0f, amplification = 1.0f;
  std::uint64_t seed = 0;
  int clip_min = 0, clip_max = 255;
};
using DirectionalDemosaicArtifactsConfig = DirectionalDemosaicingArtifactsConfig;
using FalseColorZipperConfig = FalseColorZipperArtifactsConfig;
using DemosaicingAliasingArtifactsConfig = DemosaicingAliasingConfig;
using DemosaicingRingingArtifactsConfig = DemosaicingRingingConfig;
using DemosaicingNoiseAmplificationArtifactsConfig = DemosaicingNoiseAmplificationConfig;
void bayer_directional_demosaicing_artifacts_u8(const std::uint8_t* input, std::uint8_t* output, const DirectionalDemosaicingArtifactsConfig& config, cudaStream_t stream = nullptr);
void bayer_false_color_zipper_artifacts_u8(const std::uint8_t* input, std::uint8_t* output, const FalseColorZipperArtifactsConfig& config, cudaStream_t stream = nullptr);
void bayer_demosaicing_aliasing_u8(const std::uint8_t* input, std::uint8_t* output, const DemosaicingAliasingConfig& config, cudaStream_t stream = nullptr);
void bayer_demosaicing_ringing_u8(const std::uint8_t* input, std::uint8_t* output, const DemosaicingRingingConfig& config, cudaStream_t stream = nullptr);
void bayer_demosaicing_noise_amplification_u8(const std::uint8_t* input, std::uint8_t* output, const DemosaicingNoiseAmplificationConfig& config, cudaStream_t stream = nullptr);
// Short aliases retain the same contracts for callers naming the catalog item.
inline void directional_demosaicing_artifacts_u8(const std::uint8_t* i, std::uint8_t* o, const DirectionalDemosaicingArtifactsConfig& c, cudaStream_t s = nullptr){bayer_directional_demosaicing_artifacts_u8(i, o, c, s);}
inline void false_color_zipper_artifacts_u8(const std::uint8_t* i, std::uint8_t* o, const FalseColorZipperArtifactsConfig& c, cudaStream_t s = nullptr){bayer_false_color_zipper_artifacts_u8(i, o, c, s);}
inline void demosaicing_aliasing_u8(const std::uint8_t* i, std::uint8_t* o, const DemosaicingAliasingConfig& c, cudaStream_t s = nullptr){bayer_demosaicing_aliasing_u8(i, o, c, s);}
inline void bayer_demosaicing_aliasing_artifacts_u8(const std::uint8_t* i, std::uint8_t* o, const DemosaicingAliasingConfig& c, cudaStream_t s = nullptr){bayer_demosaicing_aliasing_u8(i, o, c, s);}
inline void demosaicing_ringing_u8(const std::uint8_t* i, std::uint8_t* o, const DemosaicingRingingConfig& c, cudaStream_t s = nullptr){bayer_demosaicing_ringing_u8(i, o, c, s);}
inline void bayer_demosaicing_ringing_artifacts_u8(const std::uint8_t* i, std::uint8_t* o, const DemosaicingRingingConfig& c, cudaStream_t s = nullptr){bayer_demosaicing_ringing_u8(i, o, c, s);}
inline void demosaicing_noise_amplification_u8(const std::uint8_t* i, std::uint8_t* o, const DemosaicingNoiseAmplificationConfig& c, cudaStream_t s = nullptr){bayer_demosaicing_noise_amplification_u8(i, o, c, s);}
inline void bayer_demosaicing_noise_amplification_artifacts_u8(const std::uint8_t* i, std::uint8_t* o, const DemosaicingNoiseAmplificationConfig& c, cudaStream_t s = nullptr){bayer_demosaicing_noise_amplification_u8(i, o, c, s);}
}
