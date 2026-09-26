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
struct ToneConfig { int width=0,height=0,channels=0; };

// CLAHE uses an explicit OpenCV-compatible contract: when either dimension is
// not divisible by its tile count, the image is padded on the right and bottom
// with REFLECT_101 (including a full grid-width border for a zero remainder)
// to make tiles_x by tiles_y equal-sized tiles; each channel is processed
// independently; clip_limit is normalized as
// OpenCV's floating-point limit and must be finite and positive. Output and
// interpolated LUT values use nearest-integer rounding (lrint, ties-to-even)
// and are clipped to [0,255]. CPU pointers are host memory; CUDA pointers are
// device memory. Channels after RGB are not discarded: every supplied channel
// receives the same per-channel CLAHE treatment.
struct CLAHEConfig {
  int width=0,height=0,channels=0;
  float clip_limit=40.0f;
  int tiles_x=8,tiles_y=8;
};
void clahe_u8(const std::uint8_t*,std::uint8_t*,const CLAHEConfig&,cudaStream_t stream=nullptr);
// AllChannelsCLAHE is the explicit imgaug alias policy: every supplied HWC
// channel, including channels after RGB (such as alpha), is processed
// independently. It uses the native CLAHE implementation without color-space
// conversion, channel dropping, or a host/device fallback.
struct AllChannelsCLAHEConfig {
  int width=0,height=0,channels=0;
  float clip_limit=40.0f;
  int tiles_x=8,tiles_y=8;
};
void all_channels_clahe_u8(const std::uint8_t*,std::uint8_t*,const AllChannelsCLAHEConfig&,cudaStream_t stream=nullptr);
void sepia_u8(const std::uint8_t*,std::uint8_t*,const ToneConfig&,cudaStream_t stream=nullptr);
// RandomToneCurve applies a channel-major LUT: lut[channel * 256 + input]
// selects the output byte for each input byte. The LUT is borrowed and lives
// in host memory for the CPU API or device memory for the CUDA API. When lut
// is null, the API generates the same monotone 256-entry curves from seed;
// therefore callers supplying a LUT need no seed or random state.
struct RandomToneCurveConfig {
  int width=0,height=0,channels=0;
  const std::uint8_t* lut=nullptr;
  std::uint64_t seed=0;
};
// Fills channels consecutive 256-entry channel-major LUTs using the same
// deterministic curve contract used when RandomToneCurveConfig::lut is null.
void make_random_tone_curve_lut(std::uint8_t* lut,int channels,std::uint64_t seed);
void random_tone_curve_u8(const std::uint8_t*,std::uint8_t*,const RandomToneCurveConfig&,cudaStream_t stream=nullptr);
void autocontrast_u8(const std::uint8_t*,std::uint8_t*,const ToneConfig&,cudaStream_t stream=nullptr);
void equalize_u8(const std::uint8_t*,std::uint8_t*,const ToneConfig&,cudaStream_t stream=nullptr);
// AllChannelsHistogramEqualization independently equalizes every supplied
// HWC channel, including alpha and channels after RGB. This explicit alias
// reuses equalize_u8's native per-channel histogram kernel and has no fallback.
struct AllChannelsHistogramEqualizationConfig {
  int width=0,height=0,channels=0;
};
void all_channels_histogram_equalization_u8(const std::uint8_t*,std::uint8_t*,const AllChannelsHistogramEqualizationConfig&,cudaStream_t stream=nullptr);
// SigmoidContrast applies the normalized transfer f(x) = 1 / (1 +
// exp(gain * (cutoff - x))) to every HWC channel, where x is input / 255.
// cutoff is finite in [0,1] and gain is finite and nonnegative. The result
// is multiplied by 255, rounded half-up with floor(value + 0.5f), and clipped
// to [0,255]. CPU pointers refer to host memory; CUDA pointers refer to device
// memory. The explicit parameters replace imgaug's random parameter draws.
struct SigmoidContrastConfig {
  int width=0,height=0,channels=0;
  float cutoff=0.5f;
  float gain=10.0f;
};
void sigmoid_contrast_u8(const std::uint8_t*,std::uint8_t*,const SigmoidContrastConfig&,cudaStream_t stream=nullptr);
// LogContrast applies f(x) = log(1 + x * (base^gain - 1)) / log(base) to
// normalized x=input/255. base must be finite and greater than one; gain is
// finite and nonnegative. This is imgaug's base-2 transfer when base=2. The
// output is multiplied by 255, rounded half-up with floor(value + 0.5f), and
// clipped to [0,255]. CPU pointers are host memory and CUDA pointers are
// device memory; parameters are explicit and deterministic.
struct LogContrastConfig {
  int width=0,height=0,channels=0;
  float gain=1.0f;
  float base=2.0f;
};
void log_contrast_u8(const std::uint8_t*,std::uint8_t*,const LogContrastConfig&,cudaStream_t stream=nullptr);

// Color/photometric variation contracts use contiguous interleaved HWC uint8
// data. CPU buffers and LUTs are host-owned; CUDA buffers and LUTs are
// device-owned until the supplied stream completes. Every transfer is
// evaluated on x=input/255, rounded half-up, and clipped to [0,255].
struct GammaVariationConfig { int width=0,height=0,channels=0; float gamma=1.0f; };
void gamma_variation_u8(const std::uint8_t*,std::uint8_t*,const GammaVariationConfig&,cudaStream_t stream=nullptr);

// ToneCurveVariationConfig::lut contains exactly channels*256 bytes in
// channel-major order: lut[channel*256 + input]. The LUT is borrowed and is
// required to be non-null; callers own validation of monotonicity.
struct ToneCurveVariationConfig { int width=0,height=0,channels=0; const std::uint8_t* lut=nullptr; };
void tone_curve_variation_u8(const std::uint8_t*,std::uint8_t*,const ToneCurveVariationConfig&,cudaStream_t stream=nullptr);

// S-curve uses y=x + amount*4*x*(1-x)*(2*x-1), with finite amount in [-1,1].
struct SCurveContrastVariationConfig { int width=0,height=0,channels=0; float amount=0.0f; };
void s_curve_contrast_variation_u8(const std::uint8_t*,std::uint8_t*,const SCurveContrastVariationConfig&,cudaStream_t stream=nullptr);

// Highlight roll-off leaves x below threshold unchanged and maps highlights to
// threshold + d/(1 + strength*d/(1-threshold)); threshold is [0,1] and
// strength is nonnegative. Shadow lift/crush use the same threshold window.
struct HighlightRolloffVariationConfig { int width=0,height=0,channels=0; float threshold=0.8f,strength=1.0f; };
void highlight_rolloff_variation_u8(const std::uint8_t*,std::uint8_t*,const HighlightRolloffVariationConfig&,cudaStream_t stream=nullptr);
struct ShadowLiftConfig { int width=0,height=0,channels=0; float amount=0.0f,threshold=0.25f; };
void shadow_lift_u8(const std::uint8_t*,std::uint8_t*,const ShadowLiftConfig&,cudaStream_t stream=nullptr);
struct ShadowCrushConfig { int width=0,height=0,channels=0; float amount=0.0f,threshold=0.25f; };
void shadow_crush_u8(const std::uint8_t*,std::uint8_t*,const ShadowCrushConfig&,cudaStream_t stream=nullptr);

// Posterization and low-bit-depth banding are explicit aliases over floor
// quantization to 2^bits levels (bits is in [1,8]); no dither is introduced.
struct PosterizationConfig { int width=0,height=0,channels=0,bits=8; };
void posterization_u8(const std::uint8_t*,std::uint8_t*,const PosterizationConfig&,cudaStream_t stream=nullptr);
struct LowBitDepthBandingConfig { int width=0,height=0,channels=0,bits=8; };
void low_bit_depth_banding_u8(const std::uint8_t*,std::uint8_t*,const LowBitDepthBandingConfig&,cudaStream_t stream=nullptr);
// Compatibility spellings retain the same contracts without duplicate kernels.
using ToneCurveConfig = ToneCurveVariationConfig;
using SCurveConfig = SCurveContrastVariationConfig;
using HighlightRollOffVariationConfig = HighlightRolloffVariationConfig;
using LowBitDepthToneMappingBandingConfig = LowBitDepthBandingConfig;
inline void tone_curve_u8(const std::uint8_t* i,std::uint8_t* o,const ToneCurveConfig& c,cudaStream_t s=nullptr){tone_curve_variation_u8(i,o,c,s);}
inline void s_curve_contrast_u8(const std::uint8_t* i,std::uint8_t* o,const SCurveConfig& c,cudaStream_t s=nullptr){s_curve_contrast_variation_u8(i,o,c,s);}
inline void highlight_roll_off_variation_u8(const std::uint8_t* i,std::uint8_t* o,const HighlightRollOffVariationConfig& c,cudaStream_t s=nullptr){highlight_rolloff_variation_u8(i,o,c,s);}
inline void low_bit_depth_tone_mapping_banding_u8(const std::uint8_t* i,std::uint8_t* o,const LowBitDepthToneMappingBandingConfig& c,cudaStream_t s=nullptr){low_bit_depth_banding_u8(i,o,c,s);}
}
