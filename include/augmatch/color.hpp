#pragma once
#include <cstdint>
#include "augmatch/geometric.hpp"
#include "augmatch/dithering.hpp"
#ifndef AUGMATCH_HAS_CUDA
#define AUGMATCH_HAS_CUDA 0
#endif
#if AUGMATCH_HAS_CUDA
#include <cuda_runtime_api.h>
#else
using cudaStream_t = void*;
#endif
namespace augmatch {
struct RGBShiftConfig { int width=0,height=0,channels=0; int shift0=0,shift1=0,shift2=0; };
void rgb_shift_u8(const std::uint8_t*,std::uint8_t*,const RGBShiftConfig&,cudaStream_t stream=nullptr);
struct ChannelShuffleConfig { int width=0,height=0,channels=0; int order0=0,order1=1,order2=2; };
void channel_shuffle_u8(const std::uint8_t*,std::uint8_t*,const ChannelShuffleConfig&,cudaStream_t stream=nullptr);
// Deterministic ColorJitter applies one fixed RGB/HSV pipeline: clip(round(contrast *
// input + brightness * 255)), convert RGB to OpenCV-style HSV, multiply saturation,
// add hue in OpenCV half-degree units modulo 180, then convert back to RGB. The
// brightness and hue values are additive, while contrast and saturation are factors.
// The operation requires at least three HWC uint8 channels and copies channels >= 3.
struct ColorJitterConfig {
  int width=0,height=0,channels=0;
  float brightness=0.0f;
  float contrast=1.0f;
  float saturation=1.0f;
  float hue=0.0f;
};
void color_jitter_u8(const std::uint8_t*,std::uint8_t*,const ColorJitterConfig&,cudaStream_t stream=nullptr);
// RandomColorJitter samples one parameter tuple per invocation, not one tuple per
// pixel. Each parameter is generated from SplitMix64(seed + parameter_index),
// using indices brightness=0, contrast=1, saturation=2, hue=3. The top 24 bits
// are divided by 2^24 and linearly mapped to the corresponding [min,max)
// interval (degenerate intervals remain exact). This makes seed and ranges the
// complete random state.
struct RandomColorJitterConfig {
  int width=0,height=0,channels=0;
  float brightness_min=0.0f,brightness_max=0.0f;
  float contrast_min=1.0f,contrast_max=1.0f;
  float saturation_min=1.0f,saturation_max=1.0f;
  float hue_min=0.0f,hue_max=0.0f;
  std::uint64_t seed=0;
};
// Generates the explicit ColorJitterConfig represented by seed and ranges.
// The returned dimensions and channels are copied from the random config.
ColorJitterConfig make_random_color_jitter_config(const RandomColorJitterConfig&);
void random_color_jitter_u8(const std::uint8_t*,std::uint8_t*,const RandomColorJitterConfig&,cudaStream_t stream=nullptr);
// PlanckianJitter applies deterministic channel gains derived from the Tanner-Helland
// blackbody RGB approximation. temperature_kelvin is in [1000,40000] K and gains
// are normalized by the same approximation at the fixed D65-like reference 6500 K.
// The result is clip(floor(input * gain + 0.5), 0, 255); channels >= 3 are copied.
struct PlanckianJitterConfig { int width=0,height=0,channels=0; float temperature_kelvin=6500.0f; };
void planckian_jitter_u8(const std::uint8_t*,std::uint8_t*,const PlanckianJitterConfig&,cudaStream_t stream=nullptr);
// ChangeColorTemperature uses the same deterministic Tanner-Helland approximation
// and normalized 6500 K RGB gains as PlanckianJitter. temperature_kelvin is in
// [1000,40000] K; RGB uses clip(floor(input * gain + 0.5)), and channels >= 3
// are copied. This is a color-temperature look, not spectral adaptation or ICC
// color management.
struct ChangeColorTemperatureConfig { int width=0,height=0,channels=0; float temperature_kelvin=6500.0f; };
void change_color_temperature_u8(const std::uint8_t*,std::uint8_t*,const ChangeColorTemperatureConfig&,cudaStream_t stream=nullptr);
// Per-channel gain noise is a deterministic multiplicative Gaussian surrogate.
struct PerChannelGainNoiseConfig { int width=0,height=0,channels=0; float gains[3]={1,1,1}; float stddev=0.0f; std::uint64_t seed=0; };
void per_channel_gain_noise_u8(const std::uint8_t*,std::uint8_t*,const PerChannelGainNoiseConfig&,cudaStream_t stream=nullptr);
// Applies deterministic white-balance error with Tanner-Helland's empirical RGB
// blackbody-temperature approximation (not physical spectral/ICC adaptation).
// Gains are normalized to 6500 K; strength linearly interpolates from neutral
// gains (0) to target gains (1). Requires three leading interleaved RGB channels;
// trailing channels are copied unchanged. Temperature: [1000,40000] K; strength: [0,1].
struct ColorTemperatureErrorConfig { int width=0,height=0,channels=0; float temperature_kelvin=6500.0f,strength=1.0f; };
void color_temperature_error_u8(const std::uint8_t*,std::uint8_t*,const ColorTemperatureErrorConfig&,cudaStream_t stream=nullptr);
// ChromaticAberration applies independent normalized radial maps to RGB channels.
// For channel k, normalized coordinates use cx=(width-1)/2, cy=(height-1)/2,
// fx=max(1,width/2), fy=max(1,height/2), and source coordinates are
// (cx + xn * (1 + radial_k * r2) * fx,
//  cy + yn * (1 + radial_k * r2) * fy), where r2=xn*xn+yn*yn.
// The source is sampled with the selected interpolation and constant fill, then
// multiplied by gain_k and rounded/clipped. Channels >= 3 are copied unchanged.
// This is a deterministic lateral-dispersion approximation; gains approximate
// longitudinal/photometric effects and are not a spectral lens model.
struct ChromaticAberrationConfig {
  int width=0,height=0,channels=0;
  float radial0=0.0f,radial1=0.0f,radial2=0.0f;
  float gain0=1.0f,gain1=1.0f,gain2=1.0f;
  Interpolation interpolation=Interpolation::Linear;
  std::uint8_t fill=0;
};
void chromatic_aberration_u8(const std::uint8_t*,std::uint8_t*,const ChromaticAberrationConfig&,cudaStream_t stream=nullptr);
// Optical catalog aliases intentionally expose independent contracts while
// sharing the deterministic radial remap implementation above.
struct OpticalChromaticAberrationConfig {
  int width=0,height=0,channels=0;
  float radial0=0.0f,radial1=0.0f,radial2=0.0f;
  float gain0=1.0f,gain1=1.0f,gain2=1.0f;
  Interpolation interpolation=Interpolation::Linear;
  std::uint8_t fill=0;
};
void optical_chromatic_aberration_u8(const std::uint8_t*,std::uint8_t*,const OpticalChromaticAberrationConfig&,cudaStream_t stream=nullptr);
struct LateralChromaticAberrationConfig {
  int width=0,height=0,channels=0;
  float radial0=0.0f,radial1=0.0f,radial2=0.0f;
  Interpolation interpolation=Interpolation::Linear;
  std::uint8_t fill=0;
};
void lateral_chromatic_aberration_u8(const std::uint8_t*,std::uint8_t*,const LateralChromaticAberrationConfig&,cudaStream_t stream=nullptr);
struct LongitudinalChromaticAberrationConfig {
  int width=0,height=0,channels=0;
  int radius0=0,radius1=0,radius2=0;
  float sigma=1.0f;
};
void longitudinal_chromatic_aberration_u8(const std::uint8_t*,std::uint8_t*,const LongitudinalChromaticAberrationConfig&,cudaStream_t stream=nullptr);
// FancyPCA applies an explicit, deterministic PCA color offset. The basis is a
// row-major 3x3 matrix whose columns are the color eigenvectors. For channel c,
// delta[c] = sum_k basis[c*3+k] * eigenvalues[k] * perturbation[k] * alpha.
// eigenvalues and perturbation are expressed in uint8 intensity units; alpha is
// a fixed scalar (default 1), replacing stochastic PCA sampling. The basis must
// be finite and orthonormal, eigenvalues must be finite and nonnegative, and all
// RGB results use floor(value + 0.5) followed by clipping. Channels >= 3 copy.
struct FancyPCAConfig {
  int width=0,height=0,channels=0;
  float basis[9]={1.0f,0.0f,0.0f,0.0f,1.0f,0.0f,0.0f,0.0f,1.0f};
  float eigenvalues[3]={0.0f,0.0f,0.0f};
  float perturbation[3]={0.0f,0.0f,0.0f};
  float alpha=1.0f;
};
void fancy_pca_u8(const std::uint8_t*,std::uint8_t*,const FancyPCAConfig&,cudaStream_t stream=nullptr);
// PlasmaContrast applies a deterministic spatial contrast modulation.  For each
// pixel, field values are normalized to [0,1] and the local factor is
// contrast * (1 + plasma_strength * (2 * field - 1)); strength is restricted
// to [0,1], so the factor never inverts contrast.  The result is rounded after
// scaling around the uint8 midpoint 127.5 and clipped to [0,255].  A non-null
// field must contain width*height host floats for the CPU API or device floats
// for the CUDA API; it is preferred when exact external reproducibility is
// required.  When field is null, the same seed-driven four-scale hash field is
// generated by CPU and CUDA.  This is a deterministic plasma/noise
// approximation, not the implementation-specific stochastic plasma sampler
// used by external augmentation libraries.
struct PlasmaContrastConfig {
  int width=0,height=0,channels=0;
  float contrast=1.0f;
  float plasma_strength=0.0f;
  const float* field=nullptr;
  std::uint64_t seed=0;
};
// Fills a caller-owned width*height field using the seed contract above.
void make_plasma_field(float* field,int width,int height,std::uint64_t seed);
void plasma_contrast_u8(const std::uint8_t*,std::uint8_t*,const PlasmaContrastConfig&,cudaStream_t stream=nullptr);
// PlasmaBrightnessContrast extends PlasmaContrast with the same deterministic
// field/seed and local contrast factor, then adds brightness * 255 to every
// channel.  The result is round-to-nearest with floor(value + 0.5) and clipped
// to [0,255].  brightness is an additive normalized offset, contrast is a
// nonnegative midpoint scaling factor, and plasma_strength is in [0,1].  A
// non-null width*height field is borrowed from host memory by the CPU API and
// device memory by the CUDA API; a null field uses the shared four-scale hash
// generated from seed.  Channels are transformed independently and no channel
// is copied specially, matching PlasmaContrast's interleaved HWC contract.
struct PlasmaBrightnessContrastConfig {
  int width=0,height=0,channels=0;
  float brightness=0.0f;
  float contrast=1.0f;
  float plasma_strength=0.0f;
  const float* field=nullptr;
  std::uint64_t seed=0;
};
void plasma_brightness_contrast_u8(const std::uint8_t*,std::uint8_t*,const PlasmaBrightnessContrastConfig&,cudaStream_t stream=nullptr);
// PlasmaShadow darkens pixels where the deterministic plasma field is below threshold.
// For field < threshold, shadow = (threshold - field) / threshold and the local
// attenuation factor is 1 - strength * shadow; field >= threshold is unchanged.
// threshold and strength are finite values in [0,1]. A zero threshold disables
// shadows. The explicit field contains width*height row-major floats in [0,1],
// or a null field selects the shared four-scale hash generated from seed. Values
// are multiplied by the factor, rounded with floor(value + 0.5), and clipped.
struct PlasmaShadowConfig {
  int width=0,height=0,channels=0;
  float threshold=0.5f;
  float strength=1.0f;
  const float* field=nullptr;
  std::uint64_t seed=0;
};
void plasma_shadow_u8(const std::uint8_t*,std::uint8_t*,const PlasmaShadowConfig&,cudaStream_t stream=nullptr);
struct ToRGBConfig { int width=0,height=0,input_channels=1; };
void to_rgb_u8(const std::uint8_t*,std::uint8_t*,const ToRGBConfig&,cudaStream_t stream=nullptr);
// UniformColorQuantization divides the uint8 range [0,256) into `levels`
// equal-width bins. Each quantized value is floor(v / q) * q + q / 2,
// q=256/levels, then rounded half-up and clipped to [0,255]. `levels` is
// explicit and must be in [2,256]. A four-channel image keeps channel 3 as
// an unmodified alpha channel; all other channels are quantized.
struct UniformColorQuantizationConfig { int width=0,height=0,channels=0; int levels=2; };
void uniform_color_quantization_u8(const std::uint8_t*,std::uint8_t*,const UniformColorQuantizationConfig&,cudaStream_t stream=nullptr);
// UniformColorQuantizationToNBits is the lower-bound variant with levels=2^bits:
// output = floor(v / (256/2^bits)) * (256/2^bits). It therefore clears the
// 8-bits rightmost bits for uint8 input, rounds no intermediate value, and
// clips the final result to [0,255]. `bits` must be in [1,8].
struct UniformColorQuantizationToNBitsConfig { int width=0,height=0,channels=0; int bits=8; };
void uniform_color_quantization_to_n_bits_u8(const std::uint8_t*,std::uint8_t*,const UniformColorQuantizationToNBitsConfig&,cudaStream_t stream=nullptr);

// Color/photometric contracts use interleaved HWC uint8 RGB(A) data. RGB is
// interpreted as full-range linear-light surrogate [0,1] (no transfer-function
// conversion); channels >= 3 are copied. Matrices are row-major and map input
// RGB to output RGB. All results use half-up rounding and saturating clipping.
struct ColorMatrixPerturbationConfig {
  int width=0,height=0,channels=0;
  float matrix[9]={1,0,0,0,1,0,0,0,1};
  float offset[3]={0,0,0};
  float clip_min=0.0f,clip_max=1.0f;
};
void color_matrix_perturbation_u8(const std::uint8_t*,std::uint8_t*,const ColorMatrixPerturbationConfig&,cudaStream_t stream=nullptr);
// Profile variation applies RGB' = gains * (matrix * RGB + offset) + Gaussian noise.
// noise_stddev is normalized RGB units and seed is a coordinate-keyed state.
struct CameraColorProfileVariationConfig {
  int width=0,height=0,channels=0;
  float matrix[9]={1,0,0,0,1,0,0,0,1};
  float offset[3]={0,0,0};
  float clip_min=0.0f,clip_max=1.0f;
  float gains[3]={1,1,1};
  float noise_stddev=0.0f;
  std::uint64_t seed=0;
};
void camera_color_profile_variation_u8(const std::uint8_t*,std::uint8_t*,const CameraColorProfileVariationConfig&,cudaStream_t stream=nullptr);
// Cross-talk is a deterministic row-major 3x3 RGB mixing matrix, with no offset.
struct RGBChannelCrossTalkConfig {
  int width=0,height=0,channels=0;
  float matrix[9]={1,0,0,0,1,0,0,0,1};
  float offset[3]={0,0,0};
  float clip_min=0.0f,clip_max=1.0f;
};
void rgb_channel_cross_talk_u8(const std::uint8_t*,std::uint8_t*,const RGBChannelCrossTalkConfig&,cudaStream_t stream=nullptr);
// Spectral variation perturbs each response-matrix coefficient by an independent
// seeded N(0,response_stddev) value before applying it. response is row-major.
struct SensorSpectralResponseVariationConfig {
  int width=0,height=0,channels=0;
  float response[9]={1,0,0,0,1,0,0,0,1};
  float offset[3]={0,0,0};
  float clip_min=0.0f,clip_max=1.0f;
  float response_stddev=0.0f;
  std::uint64_t seed=0;
};
void sensor_spectral_response_variation_u8(const std::uint8_t*,std::uint8_t*,const SensorSpectralResponseVariationConfig&,cudaStream_t stream=nullptr);
// Color clipping applies independent normalized lower/upper bounds to RGB.
struct ColorClippingConfig {
  int width=0,height=0,channels=0;
  float clip_min[3]={0,0,0},clip_max[3]={1,1,1};
};
void color_clipping_u8(const std::uint8_t*,std::uint8_t*,const ColorClippingConfig&,cudaStream_t stream=nullptr);
// White-balance clipping applies gains before the common normalized clip range.
struct WhiteBalanceClippingConfig {
  int width=0,height=0,channels=0;
  float gains[3]={1,1,1};
  float clip_min=0.0f,clip_max=1.0f;
};
void white_balance_clipping_u8(const std::uint8_t*,std::uint8_t*,const WhiteBalanceClippingConfig&,cudaStream_t stream=nullptr);
// Chroma/luma use BT.601 full-range Y=0.299R+0.587G+0.114B, Cb=B-Y,
// Cr=R-Y. Chroma noise perturbs Cb and Cr independently in normalized units.
struct ChromaNoiseConfig { int width=0,height=0,channels=0; float stddev=0.0f; std::uint64_t seed=0; };
void chroma_noise_u8(const std::uint8_t*,std::uint8_t*,const ChromaNoiseConfig&,cudaStream_t stream=nullptr);
struct LumaNoiseConfig { int width=0,height=0,channels=0; float stddev=0.0f; std::uint64_t seed=0; };
void luma_noise_u8(const std::uint8_t*,std::uint8_t*,const LumaNoiseConfig&,cudaStream_t stream=nullptr);
// Correlated noise uses the same Y/Cb/Cr basis. luma_chroma_correlation is the
// correlation for both Y-Cb and Y-Cr; the two chroma samples are independent.
struct CorrelatedLumaChromaNoiseConfig { int width=0,height=0,channels=0; float luma_stddev=0.0f,chroma_stddev=0.0f,luma_chroma_correlation=0.0f; std::uint64_t seed=0; };
void correlated_luma_chroma_noise_u8(const std::uint8_t*,std::uint8_t*,const CorrelatedLumaChromaNoiseConfig&,cudaStream_t stream=nullptr);

// ChromaSubsamplingArtifacts models YCbCr chroma-plane reduction without JPEG
// DCT or quantization. RGB uses the existing BT.601 full-range surrogate
// Y=0.299R+0.587G+0.114B, Cb=B-Y, Cr=R-Y. Y444 keeps one chroma sample per
// pixel; Y422 averages each horizontal pair; Y420 averages each 2x2 block.
// The block average is replicated to its source pixels, Y is retained, and RGB
// is reconstructed with saturating half-up rounding. CPU buffers are host-owned;
// CUDA buffers are device-owned until the supplied stream completes.
enum class ChromaSubsampling : int { Y444=0, Y422=1, Y420=2, S444=Y444, S422=Y422, S420=Y420 };
struct ChromaSubsamplingArtifactsConfig { int width=0,height=0,channels=0; ChromaSubsampling subsampling=ChromaSubsampling::Y420; };
void chroma_subsampling_artifacts_u8(const std::uint8_t*,std::uint8_t*,const ChromaSubsamplingArtifactsConfig&,cudaStream_t stream=nullptr);

// LocalToneMappingNoise applies a deterministic local luminance contraction and
// additive noise in BT.601 YCbCr. For each pixel, m is the arithmetic mean of Y
// over the clamped square neighborhood radius, and Y' = Y + tone_strength*(m-Y)
// + noise_stddev*N(0,1). tone_strength is in [0,1], noise_stddev is normalized
// Y units and seed is a coordinate-keyed SplitMix64 state. Cb/Cr are preserved;
// reconstructed RGB and channels >= 3 are clipped/rounded and alpha is copied.
// CPU buffers are host-owned; CUDA buffers are device-owned until stream end.
struct LocalToneMappingNoiseConfig { int width=0,height=0,channels=0,radius=3; float tone_strength=0.5f,noise_stddev=0.0f; std::uint64_t seed=0; };
void local_tone_mapping_noise_u8(const std::uint8_t*,std::uint8_t*,const LocalToneMappingNoiseConfig&,cudaStream_t stream=nullptr);
}
