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
// camera-specific ISP implementations.
struct EdgeOversharpeningConfig { int width=0,height=0,channels=0; float amount=0.5f; };
void edge_oversharpening_u8(const std::uint8_t*,std::uint8_t*,const EdgeOversharpeningConfig&,cudaStream_t stream=nullptr);

struct UnsharpMaskHalosConfig { int width=0,height=0,channels=0,radius=1; float sigma=1.0f,amount=0.5f; };
void unsharp_mask_halos_u8(const std::uint8_t*,std::uint8_t*,const UnsharpMaskHalosConfig&,cudaStream_t stream=nullptr);

struct LaplacianHalosConfig { int width=0,height=0,channels=0; float amount=0.25f; };
void laplacian_halos_u8(const std::uint8_t*,std::uint8_t*,const LaplacianHalosConfig&,cudaStream_t stream=nullptr);

// Residual injection APIs accept an optional borrowed signed float residual map.
// A map has HxW entries when map_channels==1 (broadcast to every channel) or
// HxWxC entries when map_channels==channels. CPU maps are host-owned; CUDA maps
// are device-owned and must remain valid until the supplied stream completes.
// A null map derives the named residual from the input with clamped borders.
// Results use saturating half-up uint8 clipping to [0,255].
struct LaplacianResidualInjectionConfig {
  int width=0,height=0,channels=0; const float* residual_map=nullptr;
  int map_channels=1; float amount=0.25f;
};
void laplacian_residual_injection_u8(const std::uint8_t*,std::uint8_t*,const LaplacianResidualInjectionConfig&,cudaStream_t stream=nullptr);

// Sobel direction is 0 for X and 1 for Y. The residual map contract matches
// LaplacianResidualInjectionConfig and overrides the derived Sobel residual.
struct SobelResidualInjectionConfig {
  int width=0,height=0,channels=0; const float* residual_map=nullptr;
  int map_channels=1; float amount=0.25f; int direction=0;
};
void sobel_residual_injection_u8(const std::uint8_t*,std::uint8_t*,const SobelResidualInjectionConfig&,cudaStream_t stream=nullptr);

// High-pass residual is the signed 3x3 eight-neighbour high-pass used by
// edge_oversharpening_u8 when residual_map is null.
struct HighPassResidualInjectionConfig {
  int width=0,height=0,channels=0; const float* residual_map=nullptr;
  int map_channels=1; float amount=0.25f;
};
void high_pass_residual_injection_u8(const std::uint8_t*,std::uint8_t*,const HighPassResidualInjectionConfig&,cudaStream_t stream=nullptr);
// Spelling retained for callers that use "highpass" as one word.
using HighpassResidualInjectionConfig = HighPassResidualInjectionConfig;
inline void highpass_residual_injection_u8(const std::uint8_t* i,std::uint8_t* o,const HighpassResidualInjectionConfig& c,cudaStream_t s=nullptr) { high_pass_residual_injection_u8(i,o,c,s); }

// A checkerboard sign on the four-neighbour Laplacian is a bounded ringing
// surrogate. The residual is enabled only when the centred gradient exceeds
// gradient_threshold and is clipped to +/-residual_bound before amount.
struct RingingNearStrongEdgesConfig { int width=0,height=0,channels=0; float amount=0.5f,gradient_threshold=16.0f,residual_bound=64.0f; };
void ringing_near_strong_edges_u8(const std::uint8_t*,std::uint8_t*,const RingingNearStrongEdgesConfig&,cudaStream_t stream=nullptr);

// Local contrast uses residual=(p-mean)*sigma/(sigma+epsilon), making flat
// windows stable while retaining an explicit local variance dependency.
struct LocalContrastEnhancementArtifactsConfig { int width=0,height=0,channels=0,radius=1; float amount=0.5f,epsilon=1.0f; };
void local_contrast_enhancement_artifacts_u8(const std::uint8_t*,std::uint8_t*,const LocalContrastEnhancementArtifactsConfig&,cudaStream_t stream=nullptr);

struct HaloingFromToneMappingConfig { int width=0,height=0,channels=0,radius=3; float sigma=2.0f,amount=0.5f,tone_strength=1.0f; };
void haloing_from_tone_mapping_u8(const std::uint8_t*,std::uint8_t*,const HaloingFromToneMappingConfig&,cudaStream_t stream=nullptr);

struct LocalSharpeningNoiseAmplificationConfig { int width=0,height=0,channels=0,radius=1; float amount=0.5f,noise_gain=1.0f; };
void local_sharpening_noise_amplification_u8(const std::uint8_t*,std::uint8_t*,const LocalSharpeningNoiseAmplificationConfig&,cudaStream_t stream=nullptr);

struct HighFrequencyAttenuationConfig { int width=0,height=0,channels=0,radius=1; float amount=1.0f; };
void high_frequency_attenuation_u8(const std::uint8_t*,std::uint8_t*,const HighFrequencyAttenuationConfig&,cudaStream_t stream=nullptr);

struct DetailSmearingConfig { int width=0,height=0,channels=0,radius=2; float sigma=1.5f,amount=1.0f; bool gaussian=false; };
void detail_smearing_u8(const std::uint8_t*,std::uint8_t*,const DetailSmearingConfig&,cudaStream_t stream=nullptr);

struct OvershootAndUndershootConfig { int width=0,height=0,channels=0,radius=2; float sigma=1.5f,amount=1.0f,max_overshoot=32.0f,max_undershoot=32.0f; };
void overshoot_and_undershoot_u8(const std::uint8_t*,std::uint8_t*,const OvershootAndUndershootConfig&,cudaStream_t stream=nullptr);

// Additional ISP approximations use deterministic coordinate-keyed Gaussian draws.
// All outputs are saturating half-up uint8; borders clamp and clipping can hide
// large residuals. These are deliberately vendor-neutral surrogates.
// Seeded HWC uint8 Gaussian surrogate. Per-sample sigma is the configured
// sigma multiplied by 1 + gradient_scale*|centred gradient|/510 +
// laplacian_scale*|four-neighbour Laplacian|/1020. Borders clamp to edge;
// each sample's normal draw is keyed by seed and linear HWC index. This is a
// deterministic texture approximation, not a camera-vendor ISP noise model.
struct GradientPlusLaplacianNoiseConfig { int width=0,height=0,channels=0; float sigma=2.0f,gradient_scale=1.0f,laplacian_scale=1.0f; std::uint64_t seed=0; };
void gradient_plus_laplacian_noise_u8(const std::uint8_t*,std::uint8_t*,const GradientPlusLaplacianNoiseConfig&,cudaStream_t stream=nullptr);
struct DeblockingHalosConfig { int width=0,height=0,channels=0,block_size=8,radius=1; float amount=0.5f; };
void deblocking_halos_u8(const std::uint8_t*,std::uint8_t*,const DeblockingHalosConfig&,cudaStream_t stream=nullptr);
struct DemosaicingEdgeArtifactsConfig { int width=0,height=0,channels=0; float strength=0.5f,gradient_threshold=8.0f; };
void demosaicing_edge_artifacts_u8(const std::uint8_t*,std::uint8_t*,const DemosaicingEdgeArtifactsConfig&,cudaStream_t stream=nullptr);
struct EdgeDependentQuantizationConfig { int width=0,height=0,channels=0; int step=8; float edge_scale=1.0f; };
void edge_dependent_quantization_u8(const std::uint8_t*,std::uint8_t*,const EdgeDependentQuantizationConfig&,cudaStream_t stream=nullptr);
struct EdgeDependentCompressionErrorConfig { int width=0,height=0,channels=0,block_size=8; float strength=1.0f,edge_scale=1.0f; std::uint64_t seed=0; };
void edge_dependent_compression_error_u8(const std::uint8_t*,std::uint8_t*,const EdgeDependentCompressionErrorConfig&,cudaStream_t stream=nullptr);
struct GradientReversalConfig { int width=0,height=0,channels=0; float amount=0.5f,gradient_threshold=8.0f; };
void gradient_reversal_u8(const std::uint8_t*,std::uint8_t*,const GradientReversalConfig&,cudaStream_t stream=nullptr);
struct ClippedEdgeRingingConfig { int width=0,height=0,channels=0; float amount=0.5f,gradient_threshold=8.0f,residual_bound=16.0f; };
void clipped_edge_ringing_u8(const std::uint8_t*,std::uint8_t*,const ClippedEdgeRingingConfig&,cudaStream_t stream=nullptr);
} // namespace augmatch
