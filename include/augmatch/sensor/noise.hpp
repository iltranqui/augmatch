#pragma once
#include <cstdint>
#include <cstddef>
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

// SensorNoiseConfig configures sensor_noise_f32, the combined forward sensor
// model: shot noise, read/amplifier/reset/row/column/correlated noise, dark
// current, PRNU, FPN, and hot/dead/stuck defects, quantized by an ADC stage.
struct SensorNoiseConfig {
  int width = 0;
  int height = 0;
  int channels = 0;
  float exposure = 1.0f;
  float electrons_per_unit = 1000.0f;
  float read_noise_electrons = 3.0f;  // read noise, electrons RMS
  float amplifier_noise_electrons = 0.0f;
  float reset_noise_electrons = 0.0f;
  float dark_frame_offset_electrons = 0.0f;
  float row_noise_electrons = 0.0f;
  float column_noise_electrons = 0.0f;
  float correlated_read_noise_electrons = 0.0f;
  float prnu_stddev = 0.0f;
  float dark_current_electrons = 0.0f;
  float hot_pixel_probability = 0.0f;
  float dead_pixel_probability = 0.0f;
  float hot_pixel_electrons = 0.0f;
  float stuck_pixel_probability = 0.0f;
  float stuck_pixel_value = 0.0f;
  float fpn_stddev = 0.0f;
  float temperature_celsius = 25.0f;
  float reference_temperature_celsius = 25.0f;
  float dark_current_temp_coefficient = 0.0f;
  int adc_levels = 256;
  float black_level = 0.0f;
  float white_level = 1.0f;
  std::uint64_t seed = 0;
};
// Applies the combined sensor forward model to normalized HWC float32 data,
// producing quantized output in [black_level, white_level].
void sensor_noise_f32(const float* input, float* output, const SensorNoiseConfig& config, cudaStream_t stream = nullptr);

// PixelResponseNonUniformity applies one gain to each HWC sample. If gain_map
// is null, a deterministic map is generated from seed; otherwise it borrows
// width*height*channels float gains in the execution memory space.
struct PixelResponseNonUniformityConfig {
  int width = 0;
  int height = 0;
  int channels = 0;
  float stddev = 0.0f;
  const float* gain_map = nullptr;
  std::uint64_t seed = 0;
};
void pixel_response_non_uniformity_f32(const float* input, float* output, const PixelResponseNonUniformityConfig& config, cudaStream_t stream = nullptr);

// FixedPatternOffsetNoise adds one offset to each HWC sample. If offset_map is
// null, a deterministic map is generated from seed; otherwise it borrows
// width*height*channels float offsets in the execution memory space.
struct FixedPatternOffsetNoiseConfig {
  int width = 0;
  int height = 0;
  int channels = 0;
  float stddev = 0.0f;
  const float* offset_map = nullptr;
  std::uint64_t seed = 0;
};
void fixed_pattern_offset_noise_f32(const float* input, float* output, const FixedPatternOffsetNoiseConfig& config, cudaStream_t stream = nullptr);

// ISONoise applies an explicit ISO/analog-gain scale to shared Gaussian luma
// noise and luma-preserving per-channel chroma noise. Inputs and outputs are
// interleaved HWC uint8 samples in [0,255].
struct ISONoiseConfig {
  int width = 0;
  int height = 0;
  int channels = 0;
  float iso = 100.0f;
  float base_iso = 100.0f;
  float analog_gain = 1.0f;
  float digital_gain = 1.0f;
  float gaussian_stddev = 0.01f;
  float chroma_stddev = 0.01f;
  std::uint64_t seed = 0;
};
void iso_noise_u8(const std::uint8_t* input, std::uint8_t* output, const ISONoiseConfig& config, cudaStream_t stream = nullptr);

// ShotNoise samples Poisson photons from each normalized input sample. Gain is
// the output signal gain, while scale is the photon count per unit signal; the
// output is clipped and rounded back to interleaved HWC uint8 data.
struct ShotNoiseConfig {
  int width = 0;
  int height = 0;
  int channels = 0;
  float gain = 1.0f;
  float scale = 100.0f;
  std::uint64_t seed = 0;
};
void shot_noise_u8(const std::uint8_t* input, std::uint8_t* output, const ShotNoiseConfig& config, cudaStream_t stream = nullptr);

// EdgeDependentNoise scales Gaussian noise by the local gradient and Laplacian
// magnitude, so noise increases near edges and texture.
struct EdgeDependentNoiseConfig {
  int width = 0;
  int height = 0;
  int channels = 0;
  float base_stddev = 0.0f;
  float gradient_scale = 0.0f;
  float laplacian_scale = 0.0f;
  std::uint64_t seed = 0;
};
void edge_dependent_noise_f32(const float* input, float* output, const EdgeDependentNoiseConfig& config, cudaStream_t stream = nullptr);

// RowColumnCorrelatedNoise adds one shared offset per row and per column. Maps
// are borrowed: row_map has height*channels values and column_map has
// width*channels values, both in normalized signal units. A null map is
// generated deterministically from seed. CPU maps are host memory; CUDA maps
// are device memory and must outlive the supplied stream operation. Output is
// clipped to [clip_min, clip_max].
struct RowColumnCorrelatedNoiseConfig {
  int width = 0;
  int height = 0;
  int channels = 0;
  float row_stddev = 0.0f;
  float column_stddev = 0.0f;
  const float* row_map = nullptr;
  const float* column_map = nullptr;
  float clip_min = 0.0f;
  float clip_max = 1.0f;
  std::uint64_t seed = 0;
};
void row_column_correlated_noise_f32(const float* input, float* output, const RowColumnCorrelatedNoiseConfig& config, cudaStream_t stream = nullptr);

// Clustered defects are circular records in pixel coordinates. A channel of
// -1 applies to every channel. Records are borrowed and contain
// width/height-independent coordinates. With defects==nullptr, cluster_count
// deterministic clusters are generated from seed with cluster_radius and the
// three replacement values. Values are normalized and clipped to [0,1].
enum class ClusteredDefectKind : int { Hot = 0, Dead = 1, Stuck = 2 };
struct ClusteredDefect {
  int center_x = 0;
  int center_y = 0;
  int radius = 0;
  ClusteredDefectKind kind = ClusteredDefectKind::Dead;
  float value = 0.0f;
  int channel = -1;
};
struct ClusteredDefectivePixelsConfig {
  int width = 0;
  int height = 0;
  int channels = 0;
  const ClusteredDefect* defects = nullptr;
  std::size_t defect_count = 0;
  int cluster_count = 0;
  int cluster_radius = 0;
  float hot_value = 1.0f;
  float stuck_value = 0.0f;
  std::uint64_t seed = 0;
};
void clustered_defective_pixels_f32(const float* input, float* output, const ClusteredDefectivePixelsConfig& config, cudaStream_t stream = nullptr);

// ADC differential non-linearity uses a channel-major levels-entry LUT of
// positive code-width multipliers.  The first and last code use half-width
// endpoint bins, so an all-ones LUT is exactly the existing nearest-code ADC
// quantizer.  With lut == nullptr, each multiplier is 1 + stddev*N(0,1),
// generated from (seed, channel, code).  LUT values and stddev are dimensionless.
// Input and output are contiguous HWC float32 samples normalized to [0,1].
// The LUT is borrowed host memory for CPU calls and borrowed device memory for
// CUDA calls; it must remain valid until the supplied stream operation finishes.
struct AdcDifferentialNonLinearityConfig {
  int width = 0;
  int height = 0;
  int channels = 0;
  int levels = 256;
  const float* lut = nullptr;
  float stddev = 0.0f;
  std::uint64_t seed = 0;
};
void adc_differential_non_linearity_f32(const float* input, float* output, const AdcDifferentialNonLinearityConfig& config, cudaStream_t stream = nullptr);
// Spelling aliases retain discoverability for callers that omit word separators.
using ADCDifferentialNonLinearityConfig = AdcDifferentialNonLinearityConfig;
inline void adc_differential_nonlinearity_f32(const float* i, float* o, const AdcDifferentialNonLinearityConfig& c, cudaStream_t s = nullptr) { adc_differential_non_linearity_f32(i, o, c, s); }

// ADC integral non-linearity uses a channel-major levels-entry LUT of output
// code offsets in ADC LSBs.  The LUT is applied after nearest-code
// quantization, then clipped to [0, levels-1] before normalization.  With lut
// == nullptr, offsets are stddev*N(0,1) LSBs from (seed, channel, code).
// Input/output, clipping, and LUT ownership follow the DNL contract above.
struct AdcIntegralNonLinearityConfig {
  int width = 0;
  int height = 0;
  int channels = 0;
  int levels = 256;
  const float* lut = nullptr;
  float stddev = 0.0f;
  std::uint64_t seed = 0;
};
void adc_integral_non_linearity_f32(const float* input, float* output, const AdcIntegralNonLinearityConfig& config, cudaStream_t stream = nullptr);
using ADCIntegralNonLinearityConfig = AdcIntegralNonLinearityConfig;
inline void adc_integral_nonlinearity_f32(const float* i, float* o, const AdcIntegralNonLinearityConfig& c, cudaStream_t s = nullptr) { adc_integral_non_linearity_f32(i, o, c, s); }

// Sensor well-capacity variation uses a channel-major width*height*channels
// map of positive capacity ratios relative to the nominal full well.  A null
// map generates 1 + stddev*N(0,1) ratios from (seed, x, y, channel).
// Inputs are normalized signal/full-well units; each sample is clipped to
// [0, min(1, capacity_ratio)] and output remains normalized [0,1].  The map is
// borrowed host memory for CPU and borrowed device memory for CUDA.
struct SensorWellCapacityVariationConfig {
  int width = 0;
  int height = 0;
  int channels = 0;
  const float* capacity_map = nullptr;
  float stddev = 0.0f;
  std::uint64_t seed = 0;
};
void sensor_well_capacity_variation_f32(const float* input, float* output, const SensorWellCapacityVariationConfig& config, cudaStream_t stream = nullptr);
using WellCapacityVariationConfig = SensorWellCapacityVariationConfig;

// BrightPixel records are borrowed pixel-coordinate sources for vertical smear.
// A channel of -1 applies the source to every channel. Values are normalized
// source intensities and records remain valid for the duration of the call (or
// until the CUDA stream completes).
struct BrightPixel {
  int x = 0;
  int y = 0;
  float value = 1.0f;
  int channel = -1;
};
struct BloomingVerticalSmearConfig {
  int width = 0;
  int height = 0;
  int channels = 0;
  const BrightPixel* bright_pixels = nullptr;
  std::size_t bright_pixel_count = 0;
  // Optional width*channels map. Its value is an additional source amplitude
  // at row zero for that column/channel and is decayed down the column.
  const float* column_smear_map = nullptr;
  float bright_threshold = 0.8f;
  float smear_strength = 1.0f;
  float smear_decay = 0.9f;
  float clip_min = 0.0f;
  float clip_max = 1.0f;
  std::uint64_t seed = 0;  // reserved for API stability; explicit operation is deterministic.
};
void blooming_vertical_smear_f32(const float* input, float* output, const BloomingVerticalSmearConfig& config, cudaStream_t stream = nullptr);
using BloomingAndVerticalSmearConfig = BloomingVerticalSmearConfig;
using BloomingConfig = BloomingVerticalSmearConfig;
using BrightSource = BrightPixel;
inline void blooming_and_vertical_smear_f32(const float* i, float* o, const BloomingVerticalSmearConfig& c, cudaStream_t s = nullptr) { blooming_vertical_smear_f32(i, o, c, s); }
inline void blooming_smear_f32(const float* i, float* o, const BloomingVerticalSmearConfig& c, cudaStream_t s = nullptr) { blooming_vertical_smear_f32(i, o, c, s); }

// The mask is one borrowed uint8 value per pixel (row-major HxW); nonzero is
// opaque. Unmasked samples are copied exactly. For an opaque sample,
// blur_radius==0 writes fill; otherwise it averages unmasked samples in the
// clipped square neighborhood and writes fill only when that neighborhood has
// no unmasked sample. CPU masks are host memory; CUDA masks are device memory
// and must outlive the supplied stream operation.
struct SensorDustOpaqueMaskConfig {
  int width = 0;
  int height = 0;
  int channels = 0;
  const std::uint8_t* mask = nullptr;
  int blur_radius = 0;
  float fill = 0.0f;
};
void sensor_dust_opaque_mask_f32(const float* input, float* output, const SensorDustOpaqueMaskConfig& config, cudaStream_t stream = nullptr);
using SensorDustAndOpaquePixelMaskConfig = SensorDustOpaqueMaskConfig;
using SensorDustConfig = SensorDustOpaqueMaskConfig;
using OpaquePixelMaskConfig = SensorDustOpaqueMaskConfig;
inline void sensor_dust_and_opaque_pixel_mask_f32(const float* i, float* o, const SensorDustOpaqueMaskConfig& c, cudaStream_t s = nullptr) { sensor_dust_opaque_mask_f32(i, o, c, s); }
inline void sensor_dust_opaque_pixel_mask_f32(const float* i, float* o, const SensorDustOpaqueMaskConfig& c, cudaStream_t s = nullptr) { sensor_dust_opaque_mask_f32(i, o, c, s); }

// Optical gain maps use normalized HWC float32 data. Coordinates are normalized
// by the image extent: qx=(x/(width-1)-center_x)/radius_x and likewise for qy;
// r2=qx*qx+qy*qy. The radial polynomial is c0+c1*r2+c2*r2^2+c3*r2^3.
// A null map selects that polynomial; a supplied map is borrowed and contains
// map_channels==1 (HxW, broadcast) or map_channels==channels (HWC) gains.
// CPU maps are host memory; CUDA maps are device memory valid through stream.
struct LensVignettingConfig {
  int width = 0;
  int height = 0;
  int channels = 0;
  float center_x = 0.5f;
  float center_y = 0.5f;
  float radius_x = 0.5f;
  float radius_y = 0.5f;
  float c0 = 1.0f;
  float c1 = 0.0f;
  float c2 = 0.0f;
  float c3 = 0.0f;
  const float* map = nullptr;
  int map_channels = 1;
  float clip_min = 0.0f;
  float clip_max = 1.0f;
};
void lens_vignetting_f32(const float* input, float* output, const LensVignettingConfig& config, cudaStream_t stream = nullptr);

// Color-dependent vignetting uses one channel-major four-coefficient row per
// channel (channels*4 entries) when coefficients is non-null. Otherwise the
// shared LensVignettingConfig polynomial is used. The optional map follows the
// LensVignettingConfig ownership and shape contract.
struct ColorDependentVignettingConfig : LensVignettingConfig {
  const float* coefficients = nullptr;
};
void color_dependent_vignetting_f32(const float* input, float* output, const ColorDependentVignettingConfig& config, cudaStream_t stream = nullptr);
using ColorVignettingConfig = ColorDependentVignettingConfig;
inline void color_vignetting_f32(const float* i, float* o, const ColorDependentVignettingConfig& c, cudaStream_t s = nullptr) { color_dependent_vignetting_f32(i, o, c, s); }

// Optical falloff is the same normalized radial polynomial contract, exposed as
// a separate API because callers may calibrate it independently from vignetting.
using OpticalFalloffConfig = LensVignettingConfig;
void optical_falloff_f32(const float* input, float* output, const OpticalFalloffConfig& config, cudaStream_t stream = nullptr);

// Lens shading applies a borrowed calibrated gain map (HxW when map_channels=1
// or HWC when map_channels=channels). A null map uses gain. Output is clipped.
struct LensShadingConfig {
  int width = 0;
  int height = 0;
  int channels = 0;
  const float* map = nullptr;
  int map_channels = 1;
  float gain = 1.0f;
  float clip_min = 0.0f;
  float clip_max = 1.0f;
};
void lens_shading_f32(const float* input, float* output, const LensShadingConfig& config, cudaStream_t stream = nullptr);

// Uneven illumination reuses the same borrowed scalar/HWC map layout. The map
// is a multiplicative gain around base_gain, followed by additive offset.
struct UnevenIlluminationConfig {
  int width = 0;
  int height = 0;
  int channels = 0;
  const float* map = nullptr;
  int map_channels = 1;
  float base_gain = 1.0f;
  float offset = 0.0f;
  float clip_min = 0.0f;
  float clip_max = 1.0f;
};
void uneven_illumination_f32(const float* input, float* output, const UnevenIlluminationConfig& config, cudaStream_t stream = nullptr);

// Sensor/lens dust shadows use a borrowed HxW opacity map in [0,1]. Shadow
// strength scales opacity, so zero strength or a null map is identity.
struct SensorLensDustShadowsConfig {
  int width = 0;
  int height = 0;
  int channels = 0;
  const float* shadow_map = nullptr;
  float strength = 1.0f;
  float clip_min = 0.0f;
  float clip_max = 1.0f;
};
void sensor_lens_dust_shadows_f32(const float* input, float* output, const SensorLensDustShadowsConfig& config, cudaStream_t stream = nullptr);
using SensorLensDustShadowConfig = SensorLensDustShadowsConfig;

// Optical flare is a deterministic normalized-HWC approximation. Sources and
// halos are borrowed pixel-coordinate records (host memory for CPU, device
// memory for CUDA) and remain valid through the call/stream. Source discs use
// hard coverage; halos use a linear radial falloff to zero at radius. A halo
// with source_index >= 0 is centred on that source, otherwise its x/y fields
// are used. The optional HxW or HWC map is a nonnegative multiplicative local
// gain. Each contribution is white alpha blending, clipped to the configured
// range; no random state is used.
struct FlareSource {
  float x = 0.0f;
  float y = 0.0f;
  float radius = 1.0f;
  float intensity = 1.0f;
  int channel = -1;
};
struct FlareHalo {
  float x = 0.0f;
  float y = 0.0f;
  float radius = 1.0f;
  float intensity = 1.0f;
  int source_index = -1;
};
struct FlareConfig {
  int width = 0;
  int height = 0;
  int channels = 0;
  const FlareSource* sources = nullptr;
  std::size_t source_count = 0;
  const FlareHalo* halos = nullptr;
  std::size_t halo_count = 0;
  const float* map = nullptr;
  int map_channels = 1;
  float opacity = 1.0f;
  float clip_min = 0.0f;
  float clip_max = 1.0f;
};
void flare_f32(const float* input, float* output, const FlareConfig& config, cudaStream_t stream = nullptr);
using OpticalFlareConfig = FlareConfig;
inline void optical_flare_f32(const float* i, float* o, const FlareConfig& c, cudaStream_t s = nullptr) { flare_f32(i, o, c, s); }

// Ghosting overlays shifted copies of the input. A ghost's dx/dy is the
// destination displacement (the source sample is x-dx,y-dy); scale is a
// source-signal multiplier and alpha controls weather-style blending. Ghost
// records and the optional HxW/HWC map are borrowed in the execution memory
// space. Nearest clamped sampling is intentional and deterministic.
struct Ghost {
  float dx = 0.0f;
  float dy = 0.0f;
  float scale = 1.0f;
  float alpha = 0.0f;
};
struct GhostingConfig {
  int width = 0;
  int height = 0;
  int channels = 0;
  const Ghost* ghosts = nullptr;
  std::size_t ghost_count = 0;
  const float* map = nullptr;
  int map_channels = 1;
  float opacity = 1.0f;
  float clip_min = 0.0f;
  float clip_max = 1.0f;
};
void ghosting_f32(const float* input, float* output, const GhostingConfig& config, cudaStream_t stream = nullptr);
using OpticalGhostingConfig = GhostingConfig;

// Veiling glare adds a per-pixel white veil proportional to the local input
// luminance. This local source-luminance model is a deterministic surrogate
// for lens scattering, not a physical PSF. The optional HxW/HWC map is a
// borrowed nonnegative gain map in the execution memory space. Output clips.
struct VeilingGlareConfig {
  int width = 0;
  int height = 0;
  int channels = 0;
  const float* map = nullptr;
  int map_channels = 1;
  float strength = 0.0f;
  float clip_min = 0.0f;
  float clip_max = 1.0f;
};
void veiling_glare_f32(const float* input, float* output, const VeilingGlareConfig& config, cudaStream_t stream = nullptr);
using OpticalVeilingGlareConfig = VeilingGlareConfig;

// Bloom is a deterministic square-box bright-source surrogate. Pixels above
// threshold contribute their excess to a radius neighbourhood; the optional
// HxW/HWC map scales the local contribution. Map memory is borrowed and has
// CPU-host/CUDA-device ownership, and all output is clipped.
struct BloomConfig {
  int width = 0;
  int height = 0;
  int channels = 0;
  int radius = 0;
  float threshold = 1.0f;
  float strength = 0.0f;
  const float* map = nullptr;
  int map_channels = 1;
  float clip_min = 0.0f;
  float clip_max = 1.0f;
};
void bloom_f32(const float* input, float* output, const BloomConfig& config, cudaStream_t stream = nullptr);
using OpticalBloomConfig = BloomConfig;
inline void optical_bloom_f32(const float* i, float* o, const BloomConfig& c, cudaStream_t s = nullptr) { bloom_f32(i, o, c, s); }

// Temporal APIs consume one contiguous frame batch in T-H-W-C order:
// index(t,y,x,ch)=(((t*height+y)*width+x)*channels+ch).  CPU pointers are
// host-owned; CUDA pointers (including base_map) are device-owned and must
// remain valid until stream completion.  No API retains state.  The AR(1)
// contract is state_t=rho*state_(t-1)+sqrt(1-rho^2)*N(seed,t,y,x,ch), with
// state_-1=0; the same counter key and traversal are used by CPU and CUDA.
struct TemporalBatchConfig {
  int frames = 0;
  int height = 0;
  int width = 0;
  int channels = 0;
  float temporal_correlation = 0.0f;
  float clip_min = 0.0f;
  float clip_max = 1.0f;
  std::uint64_t seed = 0;
};
struct TemporalGaussianNoiseConfig : TemporalBatchConfig {
  float stddev = 0.0f;
};
void temporal_gaussian_noise_f32(const float* input, float* output, const TemporalGaussianNoiseConfig& config, cudaStream_t stream = nullptr);

struct TemporallyCorrelatedShotNoiseConfig : TemporalBatchConfig {
  float photons_per_unit = 100.0f;
};
void temporally_correlated_shot_noise_f32(const float* input, float* output, const TemporallyCorrelatedShotNoiseConfig& config, cudaStream_t stream = nullptr);
inline void temporal_shot_noise_f32(const float* i, float* o, const TemporallyCorrelatedShotNoiseConfig& c, cudaStream_t s = nullptr) { temporally_correlated_shot_noise_f32(i, o, c, s); }

struct TemporalReadNoiseCorrelationConfig : TemporalBatchConfig {
  float stddev = 0.0f;
};
void temporal_read_noise_correlation_f32(const float* input, float* output, const TemporalReadNoiseCorrelationConfig& config, cudaStream_t stream = nullptr);
inline void temporal_read_noise_f32(const float* i, float* o, const TemporalReadNoiseCorrelationConfig& c, cudaStream_t s = nullptr) { temporal_read_noise_correlation_f32(i, o, c, s); }

// Flicker is a frame-global multiplicative illumination factor.  Its
// normalized state has unit standard deviation before amplitude scaling.
struct FlickerConfig : TemporalBatchConfig {
  float stddev = 0.0f;
};
void flicker_f32(const float* input, float* output, const FlickerConfig& config, cudaStream_t stream = nullptr);
struct ExposureFlickerConfig : TemporalBatchConfig {
  float stddev = 0.0f;
};
using TemporalFlickerConfig = FlickerConfig;
void exposure_flicker_f32(const float* input, float* output, const ExposureFlickerConfig& config, cudaStream_t stream = nullptr);
struct WhiteBalanceFlickerConfig : TemporalBatchConfig {
  float red_stddev = 0.0f;
  float green_stddev = 0.0f;
  float blue_stddev = 0.0f;
};
void white_balance_flicker_f32(const float* input, float* output, const WhiteBalanceFlickerConfig& config, cudaStream_t stream = nullptr);
struct GainFlickerConfig : TemporalBatchConfig {
  float stddev = 0.0f;
};
void gain_flicker_f32(const float* input, float* output, const GainFlickerConfig& config, cudaStream_t stream = nullptr);
using GainNoiseFlickerConfig = GainFlickerConfig;

// Fixed-pattern drift starts from an optional borrowed HWC base_map (or a
// deterministic seeded map with base_stddev) and evolves each pixel/channel
// through the same AR(1) state.  base_map is not indexed by frame.
struct FixedPatternNoiseDriftConfig : TemporalBatchConfig {
  const float* base_map = nullptr;
  float base_stddev = 0.0f;
  float drift_stddev = 0.0f;
};
void fixed_pattern_noise_drift_f32(const float* input, float* output, const FixedPatternNoiseDriftConfig& config, cudaStream_t stream = nullptr);
// Row noise is an H-wise offset. A deterministic phase state flips sign with
// phase_change_probability, while temporal_correlation smooths its amplitude.
struct RowNoisePhaseChangesConfig : TemporalBatchConfig {
  float row_stddev = 0.0f;
  float phase_change_probability = 0.1f;
};
void row_noise_phase_changes_f32(const float* input, float* output, const RowNoisePhaseChangesConfig& config, cudaStream_t stream = nullptr);
using TemporalFPNDriftConfig = FixedPatternNoiseDriftConfig;
inline void temporal_fpn_drift_f32(const float* i, float* o, const FixedPatternNoiseDriftConfig& c, cudaStream_t s = nullptr) { fixed_pattern_noise_drift_f32(i, o, c, s); }

// Video/temporal artifact APIs use the same borrowed contiguous T-H-W-C
// buffers and clipping contract as the temporal noise APIs above. Optional
// masks, index arrays, and transform arrays are host-owned for CPU calls and
// device-owned for CUDA calls; no pointer is retained after the call (or the
// supplied CUDA stream completes). Masks are row-major H-W-C or T entries as
// documented by each config. Explicit arrays take precedence over seeded
// generation, making replay and property tests independent of RNG state.
struct DeadPixelPersistenceConfig : TemporalBatchConfig {
  const std::uint8_t* mask = nullptr;  // H*W*C, nonzero means dead.
  float probability = 0.0f, dead_value = 0.0f;
};
void dead_pixel_persistence_f32(const float* input, float* output, const DeadPixelPersistenceConfig& config, cudaStream_t stream = nullptr);

struct HotPixelPersistenceConfig : TemporalBatchConfig {
  const std::uint8_t* mask = nullptr;  // H*W*C, nonzero means hot.
  float probability = 0.0f, hot_value = 1.0f;
};
void hot_pixel_persistence_f32(const float* input, float* output, const HotPixelPersistenceConfig& config, cudaStream_t stream = nullptr);

struct FrameDropConfig : TemporalBatchConfig {
  const std::uint8_t* drop_mask = nullptr;  // T entries, nonzero means dropped.
  const int* drop_indices = nullptr;
  std::size_t drop_count = 0;
  float probability = 0.0f, fill_value = 0.0f;
};
void frame_drops_f32(const float* input, float* output, const FrameDropConfig& config, cudaStream_t stream = nullptr);
inline void frame_drop_f32(const float* i, float* o, const FrameDropConfig& c, cudaStream_t s = nullptr) { frame_drops_f32(i, o, c, s); }

struct DuplicateFrameConfig : TemporalBatchConfig {
  const int* source_indices = nullptr;  // T source frame indices; -1 means current.
  float probability = 0.0f;
};
void duplicate_frames_f32(const float* input, float* output, const DuplicateFrameConfig& config, cudaStream_t stream = nullptr);
inline void duplicate_frame_f32(const float* i, float* o, const DuplicateFrameConfig& c, cudaStream_t s = nullptr) { duplicate_frames_f32(i, o, c, s); }

struct FrameBlendingConfig : TemporalBatchConfig {
  const float* weights = nullptr;  // T previous-frame weights, when non-null.
  const int* source_indices = nullptr;  // T source indices; -1 selects t-1.
  float weight = 0.5f;
};
void frame_blending_f32(const float* input, float* output, const FrameBlendingConfig& config, cudaStream_t stream = nullptr);

struct TemporalGhostTransform {
  int source_frame_offset = -1;
  int dx = 0;
  int dy = 0;
  float alpha = 0.0f;
};
struct TemporalGhostingConfig : TemporalBatchConfig {
  const TemporalGhostTransform* transforms = nullptr;  // T records, one per output frame.
  int source_frame_offset = -1;
  int dx = 0;
  int dy = 0;
  float alpha = 0.0f;
};
void temporal_ghosting_f32(const float* input, float* output, const TemporalGhostingConfig& config, cudaStream_t stream = nullptr);

struct MotionCompensationTransform {
  int source_frame_offset = -1;
  int dx = 0;
  int dy = 0;
  float weight = 1.0f;
};
struct MotionCompensationErrorConfig : TemporalBatchConfig {
  const MotionCompensationTransform* transforms = nullptr;  // T records.
  int source_frame_offset = -1;
  int dx = 0;
  int dy = 0;
  float weight = 1.0f;
};
void motion_compensation_errors_f32(const float* input, float* output, const MotionCompensationErrorConfig& config, cudaStream_t stream = nullptr);
inline void motion_compensation_error_f32(const float* i, float* o, const MotionCompensationErrorConfig& c, cudaStream_t s = nullptr) { motion_compensation_errors_f32(i, o, c, s); }

struct RollingShutterTransform {
  int source_frame_offset = 0;
  int dx = 0;
  int dy = 0;
};
struct VideoRollingShutterConfig : TemporalBatchConfig {
  const RollingShutterTransform* transforms = nullptr;  // T*H records, row-major.
  float motion_dx = 0.0f, motion_dy = 0.0f, readout_fraction = 0.0f;
};
void video_sensor_rolling_shutter_f32(const float* input, float* output, const VideoRollingShutterConfig& config, cudaStream_t stream = nullptr);
inline void rolling_shutter_video_f32(const float* i, float* o, const VideoRollingShutterConfig& c, cudaStream_t s = nullptr) { video_sensor_rolling_shutter_f32(i, o, c, s); }
inline void video_sensor_rolling_shutter(const float* i, float* o, const VideoRollingShutterConfig& c, cudaStream_t s = nullptr) { video_sensor_rolling_shutter_f32(i, o, c, s); }
inline void drop_frames_f32(const float* i, float* o, const FrameDropConfig& c, cudaStream_t s = nullptr) { frame_drops_f32(i, o, c, s); }
inline void duplicate_frame_sequence_f32(const float* i, float* o, const DuplicateFrameConfig& c, cudaStream_t s = nullptr) { duplicate_frames_f32(i, o, c, s); }

// Random telegraph signal noise is an H-W-C sensor operation. The optional
// borrowed state_map contains HWC uint8 state values (zero=low, nonzero=high).
// Without a map, each logical sample independently toggles initial_state with
// probability transition_probability using a coordinate-keyed RNG. The seed
// contract is traversal-independent and identical on CPU/CUDA; no state is
// retained between calls. This independent single-exposure model is a native
// approximation to temporally correlated defect-state telegraph noise.
struct RandomTelegraphSignalNoiseConfig {
  int width = 0;
  int height = 0;
  int channels = 0;
  const std::uint8_t* state_map = nullptr;
  float low_offset = 0.0f;
  float high_offset = 0.0f;
  float transition_probability = 0.0f;
  std::uint8_t initial_state = 0;
  float clip_min = 0.0f;
  float clip_max = 1.0f;
  std::uint64_t seed = 0;
};
void random_telegraph_signal_noise_f32(const float* input, float* output, const RandomTelegraphSignalNoiseConfig& config, cudaStream_t stream = nullptr);
using RandomTelegraphConfig = RandomTelegraphSignalNoiseConfig;
inline void random_telegraph_noise_f32(const float* i, float* o, const RandomTelegraphSignalNoiseConfig& c, cudaStream_t s = nullptr) { random_telegraph_signal_noise_f32(i, o, c, s); }
inline void random_telegraph_signal_f32(const float* i, float* o, const RandomTelegraphSignalNoiseConfig& c, cudaStream_t s = nullptr) { random_telegraph_signal_noise_f32(i, o, c, s); }

// Codec-independent video surrogates consume contiguous T-H-W-C float32 data.
// They do not parse or emit AV1/H.264/H.265 bitstreams.  An inter-frame
// record is an optional per-frame quantization step; null records use the
// configured step and seeded coordinate noise.
struct InterFrameCompressionRecord {
  float quantization_step = 0.0f;
  float strength = 1.0f;
};
struct InterFrameCompressionNoiseConfig : TemporalBatchConfig {
  const InterFrameCompressionRecord* records = nullptr;  // T records, optional.
  float quantization_step = 0.05f, strength = 1.0f;
};
void inter_frame_compression_noise_f32(const float* input, float* output, const InterFrameCompressionNoiseConfig& config, cudaStream_t stream = nullptr);
using InterFrameCompressionConfig = InterFrameCompressionNoiseConfig;
inline void inter_frame_compression_f32(const float* i, float* o, const InterFrameCompressionNoiseConfig& c, cudaStream_t s = nullptr) { inter_frame_compression_noise_f32(i, o, c, s); }

// GOP artifacts use explicit keyframe records when supplied.  A record's
// keyframe flag selects intra-frame quantization; other frames use the
// previous-frame residual surrogate.  gop_size is used only when records are
// null, and is not a codec parser or reference-picture implementation.
struct GopFrameRecord {
  std::uint8_t keyframe = 0;
  float strength = 1.0f;
};
struct GopKeyframeArtifactsConfig : TemporalBatchConfig {
  const GopFrameRecord* records = nullptr;  // T records, optional.
  const std::uint8_t* keyframe_mask = nullptr;  // T entries, optional.
  int gop_size = 12;
  float keyframe_strength = 0.15f, interframe_strength = 0.10f;
};
void gop_keyframe_artifacts_f32(const float* input, float* output, const GopKeyframeArtifactsConfig& config, cudaStream_t stream = nullptr);
using GOPKeyframeArtifactsConfig = GopKeyframeArtifactsConfig;
inline void gop_keyframe_f32(const float* i, float* o, const GopKeyframeArtifactsConfig& c, cudaStream_t s = nullptr) { gop_keyframe_artifacts_f32(i, o, c, s); }

// Block motion-estimation artifacts use explicit T*block_rows*block_cols
// MotionVectorRecord entries when supplied.  Vectors are integer pixel shifts
// applied to the previous frame; a seeded vector field is used otherwise.
struct MotionVectorRecord {
  int dx = 0, dy = 0;
  float strength = 1.0f;
};
struct BlockMotionEstimationArtifactsConfig : TemporalBatchConfig {
  const MotionVectorRecord* vectors = nullptr;
  int block_width = 8, block_height = 8;
  float strength = 1.0f;
};
void block_motion_estimation_artifacts_f32(const float* input, float* output, const BlockMotionEstimationArtifactsConfig& config, cudaStream_t stream = nullptr);
using BlockMotionArtifactsConfig = BlockMotionEstimationArtifactsConfig;
inline void block_motion_estimation_f32(const float* i, float* o, const BlockMotionEstimationArtifactsConfig& c, cudaStream_t s = nullptr) { block_motion_estimation_artifacts_f32(i, o, c, s); }

// Water droplets are H-W-C float32 lens records.  Explicit droplets are
// borrowed and processed in order; null records are generated from seed and
// can be materialized with make_water_droplets.  The model is a documented
// opacity-weighted local-box blur, not a refractive fluid simulation.
struct WaterDroplet {
  float x = 0.0f, y = 0.0f, radius = 1.0f, opacity = 1.0f;
};
struct WaterDropletsOnLensConfig {
  int width = 0;
  int height = 0;
  int channels = 0;
  const WaterDroplet* droplets = nullptr;
  int droplet_count = 0;
  float opacity = 0.5f;
  float radius_min = 2.0f;
  float radius_max = 8.0f;
  int blur_radius = 2;
  float clip_min = 0.0f;
  float clip_max = 1.0f;
  std::uint64_t seed = 0;
};
void make_water_droplets(WaterDroplet* output, const WaterDropletsOnLensConfig& config);
void water_droplets_on_lens_f32(const float* input, float* output, const WaterDropletsOnLensConfig& config, cudaStream_t stream = nullptr);
using WaterDropletConfig = WaterDropletsOnLensConfig;
inline void water_droplets_lens_f32(const float* i, float* o, const WaterDropletsOnLensConfig& c, cudaStream_t s = nullptr) { water_droplets_on_lens_f32(i, o, c, s); }
inline void water_droplet_lens_f32(const float* i, float* o, const WaterDropletsOnLensConfig& c, cudaStream_t s = nullptr) { water_droplets_on_lens_f32(i, o, c, s); }

using DeadPixelPersistence = DeadPixelPersistenceConfig;
using HotPixelPersistence = HotPixelPersistenceConfig;
using FrameDropsConfig = FrameDropConfig;
using DuplicateFramesConfig = DuplicateFrameConfig;
using FrameBlending = FrameBlendingConfig;
using TemporalGhosting = TemporalGhostingConfig;
using MotionCompensationErrorsConfig = MotionCompensationErrorConfig;
using VideoSensorRollingShutterConfig = VideoRollingShutterConfig;

using DustShadowConfig = SensorLensDustShadowsConfig;
inline void sensor_lens_dust_shadow_f32(const float* i, float* o, const SensorLensDustShadowsConfig& c, cudaStream_t s = nullptr) { sensor_lens_dust_shadows_f32(i, o, c, s); }
inline void sensor_dust_shadows_f32(const float* i, float* o, const SensorLensDustShadowsConfig& c, cudaStream_t s = nullptr) { sensor_lens_dust_shadows_f32(i, o, c, s); }
}
