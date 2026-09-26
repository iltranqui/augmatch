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
// ISP artifact contracts use contiguous interleaved HWC uint8 samples. Borders
// are clamped, calculations use float, and every result is saturating rounded
// half-up to [0,255]. These are deterministic native approximations, not
// camera-specific ISP implementations. Pointers are host memory in CPU builds
// and device memory in CUDA builds.

// Adds a signed 3x3 eight-neighbour high-pass residual, scaled by amount.
struct EdgeOversharpeningConfig {
  int width = 0;
  int height = 0;
  int channels = 0;
  float amount = 0.5f;
};
void edge_oversharpening_u8(const std::uint8_t* input, std::uint8_t* output, const EdgeOversharpeningConfig& config, cudaStream_t stream = nullptr);

// Adds a Gaussian-blur high-pass residual (unsharp-mask style), scaled by amount.
struct UnsharpMaskHalosConfig {
  int width = 0;
  int height = 0;
  int channels = 0;
  int radius = 1;
  float sigma = 1.0f;
  float amount = 0.5f;
};
void unsharp_mask_halos_u8(const std::uint8_t* input, std::uint8_t* output, const UnsharpMaskHalosConfig& config, cudaStream_t stream = nullptr);

// Adds a four-neighbour Laplacian residual, scaled by amount.
struct LaplacianHalosConfig {
  int width = 0;
  int height = 0;
  int channels = 0;
  float amount = 0.25f;
};
void laplacian_halos_u8(const std::uint8_t* input, std::uint8_t* output, const LaplacianHalosConfig& config, cudaStream_t stream = nullptr);

// Residual injection APIs accept an optional borrowed signed float residual map.
// A map has HxW entries when map_channels==1 (broadcast to every channel) or
// HxWxC entries when map_channels==channels. CPU maps are host-owned; CUDA maps
// are device-owned and must remain valid until the supplied stream completes.
// A null map derives the named residual from the input with clamped borders.
// Results use saturating half-up uint8 clipping to [0,255].
struct LaplacianResidualInjectionConfig {
  int width = 0;
  int height = 0;
  int channels = 0;
  const float* residual_map = nullptr;
  int map_channels = 1;
  float amount = 0.25f;
};
void laplacian_residual_injection_u8(const std::uint8_t* input, std::uint8_t* output, const LaplacianResidualInjectionConfig& config, cudaStream_t stream = nullptr);

// Sobel direction is 0 for X and 1 for Y. The residual map contract matches
// LaplacianResidualInjectionConfig and overrides the derived Sobel residual.
struct SobelResidualInjectionConfig {
  int width = 0;
  int height = 0;
  int channels = 0;
  const float* residual_map = nullptr;
  int map_channels = 1;
  float amount = 0.25f;
  int direction = 0;
};
void sobel_residual_injection_u8(const std::uint8_t* input, std::uint8_t* output, const SobelResidualInjectionConfig& config, cudaStream_t stream = nullptr);

// High-pass residual is the signed 3x3 eight-neighbour high-pass used by
// edge_oversharpening_u8 when residual_map is null.
struct HighPassResidualInjectionConfig {
  int width = 0;
  int height = 0;
  int channels = 0;
  const float* residual_map = nullptr;
  int map_channels = 1;
  float amount = 0.25f;
};
void high_pass_residual_injection_u8(const std::uint8_t* input, std::uint8_t* output, const HighPassResidualInjectionConfig& config, cudaStream_t stream = nullptr);
// Spelling retained for callers that use "highpass" as one word.
using HighpassResidualInjectionConfig = HighPassResidualInjectionConfig;
inline void highpass_residual_injection_u8(const std::uint8_t* i, std::uint8_t* o, const HighpassResidualInjectionConfig& c, cudaStream_t s = nullptr) { high_pass_residual_injection_u8(i, o, c, s); }

// A checkerboard sign on the four-neighbour Laplacian is a bounded ringing
// surrogate. The residual is enabled only when the centred gradient exceeds
// gradient_threshold and is clipped to +/-residual_bound before amount.
struct RingingNearStrongEdgesConfig {
  int width = 0;
  int height = 0;
  int channels = 0;
  float amount = 0.5f;
  float gradient_threshold = 16.0f;
  float residual_bound = 64.0f;
};
void ringing_near_strong_edges_u8(const std::uint8_t* input, std::uint8_t* output, const RingingNearStrongEdgesConfig& config, cudaStream_t stream = nullptr);

// Local contrast uses residual=(p-mean)*sigma/(sigma+epsilon), making flat
// windows stable while retaining an explicit local variance dependency.
struct LocalContrastEnhancementArtifactsConfig {
  int width = 0;
  int height = 0;
  int channels = 0;
  int radius = 1;
  float amount = 0.5f;
  float epsilon = 1.0f;
};
void local_contrast_enhancement_artifacts_u8(const std::uint8_t* input, std::uint8_t* output, const LocalContrastEnhancementArtifactsConfig& config, cudaStream_t stream = nullptr);

// Gaussian high-pass tone-mapping halo, with an explicit tone_strength gain.
struct HaloingFromToneMappingConfig {
  int width = 0;
  int height = 0;
  int channels = 0;
  int radius = 3;
  float sigma = 2.0f;
  float amount = 0.5f;
  float tone_strength = 1.0f;
};
void haloing_from_tone_mapping_u8(const std::uint8_t* input, std::uint8_t* output, const HaloingFromToneMappingConfig& config, cudaStream_t stream = nullptr);

// Gaussian high-pass sharpening residual with an explicit noise_gain amplifying it.
struct LocalSharpeningNoiseAmplificationConfig {
  int width = 0;
  int height = 0;
  int channels = 0;
  int radius = 1;
  float amount = 0.5f;
  float noise_gain = 1.0f;
};
void local_sharpening_noise_amplification_u8(const std::uint8_t* input, std::uint8_t* output, const LocalSharpeningNoiseAmplificationConfig& config, cudaStream_t stream = nullptr);

// Blends toward a deterministic box low-pass result, attenuating high frequencies.
struct HighFrequencyAttenuationConfig {
  int width = 0;
  int height = 0;
  int channels = 0;
  int radius = 1;
  float amount = 1.0f;
};
void high_frequency_attenuation_u8(const std::uint8_t* input, std::uint8_t* output, const HighFrequencyAttenuationConfig& config, cudaStream_t stream = nullptr);

// Blends toward a deterministic box or Gaussian low-pass result (gaussian selects the kernel).
struct DetailSmearingConfig {
  int width = 0;
  int height = 0;
  int channels = 0;
  int radius = 2;
  float sigma = 1.5f;
  float amount = 1.0f;
  bool gaussian = false;
};
void detail_smearing_u8(const std::uint8_t* input, std::uint8_t* output, const DetailSmearingConfig& config, cudaStream_t stream = nullptr);

// A bounded signed Gaussian high-boost residual with independent positive and
// negative limits (max_overshoot / max_undershoot).
struct OvershootAndUndershootConfig {
  int width = 0;
  int height = 0;
  int channels = 0;
  int radius = 2;
  float sigma = 1.5f;
  float amount = 1.0f;
  float max_overshoot = 32.0f;
  float max_undershoot = 32.0f;
};
void overshoot_and_undershoot_u8(const std::uint8_t* input, std::uint8_t* output, const OvershootAndUndershootConfig& config, cudaStream_t stream = nullptr);

// Additional ISP approximations use deterministic coordinate-keyed Gaussian draws.
// All outputs are saturating half-up uint8; borders clamp and clipping can hide
// large residuals. These are deliberately vendor-neutral surrogates.
// Seeded HWC uint8 Gaussian surrogate. Per-sample sigma is the configured
// sigma multiplied by 1 + gradient_scale*|centred gradient|/510 +
// laplacian_scale*|four-neighbour Laplacian|/1020. Borders clamp to edge;
// each sample's normal draw is keyed by seed and linear HWC index. This is a
// deterministic texture approximation, not a camera-vendor ISP noise model.
struct GradientPlusLaplacianNoiseConfig {
  int width = 0;
  int height = 0;
  int channels = 0;
  float sigma = 2.0f;
  float gradient_scale = 1.0f;
  float laplacian_scale = 1.0f;
  std::uint64_t seed = 0;
};
void gradient_plus_laplacian_noise_u8(const std::uint8_t* input, std::uint8_t* output, const GradientPlusLaplacianNoiseConfig& config, cudaStream_t stream = nullptr);

// Subtracts the mean local block-boundary gradient over block_size/radius,
// smoothing block edges with amount as the blend strength.
struct DeblockingHalosConfig {
  int width = 0;
  int height = 0;
  int channels = 0;
  int block_size = 8;
  int radius = 1;
  float amount = 0.5f;
};
void deblocking_halos_u8(const std::uint8_t* input, std::uint8_t* output, const DeblockingHalosConfig& config, cudaStream_t stream = nullptr);

// Injects a checkerboard-sign red/blue offset at green-channel edges whose
// gradient exceeds gradient_threshold, scaled by strength; requires 3+ channels.
struct DemosaicingEdgeArtifactsConfig {
  int width = 0;
  int height = 0;
  int channels = 0;
  float strength = 0.5f;
  float gradient_threshold = 8.0f;
};
void demosaicing_edge_artifacts_u8(const std::uint8_t* input, std::uint8_t* output, const DemosaicingEdgeArtifactsConfig& config, cudaStream_t stream = nullptr);

// Rounds each sample to the nearest multiple of step, where step grows with
// the local gradient magnitude scaled by edge_scale.
struct EdgeDependentQuantizationConfig {
  int width = 0;
  int height = 0;
  int channels = 0;
  int step = 8;
  float edge_scale = 1.0f;
};
void edge_dependent_quantization_u8(const std::uint8_t* input, std::uint8_t* output, const EdgeDependentQuantizationConfig& config, cudaStream_t stream = nullptr);

// Blends each sample toward its block_size-aligned block corner, weighted by
// strength and boosted near edges by edge_scale, plus coordinate-keyed noise.
struct EdgeDependentCompressionErrorConfig {
  int width = 0;
  int height = 0;
  int channels = 0;
  int block_size = 8;
  float strength = 1.0f;
  float edge_scale = 1.0f;
  std::uint64_t seed = 0;
};
void edge_dependent_compression_error_u8(const std::uint8_t* input, std::uint8_t* output, const EdgeDependentCompressionErrorConfig& config, cudaStream_t stream = nullptr);

// Subtracts the local horizontal/vertical gradient (scaled by amount) once its
// magnitude reaches gradient_threshold, reversing the edge direction.
struct GradientReversalConfig {
  int width = 0;
  int height = 0;
  int channels = 0;
  float amount = 0.5f;
  float gradient_threshold = 8.0f;
};
void gradient_reversal_u8(const std::uint8_t* input, std::uint8_t* output, const GradientReversalConfig& config, cudaStream_t stream = nullptr);

// A checkerboard-sign four-neighbour Laplacian residual, gated to edges whose
// gradient exceeds gradient_threshold and near saturated (<=1 or >=254) pixels,
// clipped to +/-residual_bound before amount.
struct ClippedEdgeRingingConfig {
  int width = 0;
  int height = 0;
  int channels = 0;
  float amount = 0.5f;
  float gradient_threshold = 8.0f;
  float residual_bound = 16.0f;
};
void clipped_edge_ringing_u8(const std::uint8_t* input, std::uint8_t* output, const ClippedEdgeRingingConfig& config, cudaStream_t stream = nullptr);
} // namespace augmatch
