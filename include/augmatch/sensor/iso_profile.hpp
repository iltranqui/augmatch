#pragma once
#include <cstddef>
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

// ISO profile values use normalized signal units: gain and all scale fields are
// dimensionless ratios, while black_level and saturation_level are [0,1]
// fractions of the ADC range. Knots must be sorted by increasing ISO.
struct IsoNoiseProfilePoint {
  float iso = 100.0f;
  float gain = 1.0f;
  float shot_noise_scale = 1.0f;
  float read_noise_scale = 1.0f;
  float fpn_scale = 1.0f;
  float black_level = 0.0f;
  float saturation_level = 1.0f;
};
// A sorted-by-ISO knot table; points is borrowed host memory for CPU calls
// and device memory for CUDA calls.
struct IsoNoiseProfile {
  const IsoNoiseProfilePoint* points = nullptr;
  std::size_t point_count = 0;
};

enum class IsoNoiseProfileParameter : int {
  Gain = 0, ShotNoiseScale = 1, ReadNoiseScale = 2, FPNScale = 3,
  BlackLevel = 4, SaturationLevel = 5
};

#if defined(__CUDACC__)
#define AUGMATCH_ISO_HOST_DEVICE __host__ __device__
#else
#define AUGMATCH_ISO_HOST_DEVICE
#endif

// Piecewise-linear interpolation clamps outside the first/last knot. The
// function is usable in host code and in CUDA kernels with device-resident knots.
AUGMATCH_ISO_HOST_DEVICE inline float iso_profile_value(
    float iso, const IsoNoiseProfile& profile, IsoNoiseProfileParameter parameter) {
  if (profile.points == nullptr || profile.point_count == 0) return 0.0f;
  if (profile.point_count == 1) {
    const IsoNoiseProfilePoint& p = profile.points[0];
    switch (parameter) {
      case IsoNoiseProfileParameter::Gain: return p.gain;
      case IsoNoiseProfileParameter::ShotNoiseScale: return p.shot_noise_scale;
      case IsoNoiseProfileParameter::ReadNoiseScale: return p.read_noise_scale;
      case IsoNoiseProfileParameter::FPNScale: return p.fpn_scale;
      case IsoNoiseProfileParameter::BlackLevel: return p.black_level;
      case IsoNoiseProfileParameter::SaturationLevel: return p.saturation_level;
    }
  }
  std::size_t hi = 0;
  while (hi < profile.point_count && profile.points[hi].iso < iso) ++hi;
  if (hi == 0) hi = 1;
  if (hi >= profile.point_count) hi = profile.point_count - 1;
  const IsoNoiseProfilePoint& a = profile.points[hi - 1];
  const IsoNoiseProfilePoint& b = profile.points[hi];
  const float denominator = b.iso - a.iso;
  const float raw_t = denominator > 0.0f ? (iso - a.iso) / denominator : 0.0f;
  const float t = raw_t < 0.0f ? 0.0f : (raw_t > 1.0f ? 1.0f : raw_t);
  float av = 0.0f, bv = 0.0f;
  switch (parameter) {
    case IsoNoiseProfileParameter::Gain: av = a.gain; bv = b.gain; break;
    case IsoNoiseProfileParameter::ShotNoiseScale: av = a.shot_noise_scale; bv = b.shot_noise_scale; break;
    case IsoNoiseProfileParameter::ReadNoiseScale: av = a.read_noise_scale; bv = b.read_noise_scale; break;
    case IsoNoiseProfileParameter::FPNScale: av = a.fpn_scale; bv = b.fpn_scale; break;
    case IsoNoiseProfileParameter::BlackLevel: av = a.black_level; bv = b.black_level; break;
    case IsoNoiseProfileParameter::SaturationLevel: av = a.saturation_level; bv = b.saturation_level; break;
  }
  return av + (bv - av) * t;
}
// Convenience wrappers around iso_profile_value for each profile parameter.
AUGMATCH_ISO_HOST_DEVICE inline float iso_to_gain(float iso, const IsoNoiseProfile& p) { return iso_profile_value(iso, p, IsoNoiseProfileParameter::Gain); }
AUGMATCH_ISO_HOST_DEVICE inline float iso_shot_noise_scale(float iso, const IsoNoiseProfile& p) { return iso_profile_value(iso, p, IsoNoiseProfileParameter::ShotNoiseScale); }
AUGMATCH_ISO_HOST_DEVICE inline float iso_read_noise_scale(float iso, const IsoNoiseProfile& p) { return iso_profile_value(iso, p, IsoNoiseProfileParameter::ReadNoiseScale); }
AUGMATCH_ISO_HOST_DEVICE inline float iso_fpn_scale(float iso, const IsoNoiseProfile& p) { return iso_profile_value(iso, p, IsoNoiseProfileParameter::FPNScale); }
AUGMATCH_ISO_HOST_DEVICE inline float iso_black_level(float iso, const IsoNoiseProfile& p) { return iso_profile_value(iso, p, IsoNoiseProfileParameter::BlackLevel); }
AUGMATCH_ISO_HOST_DEVICE inline float iso_saturation_level(float iso, const IsoNoiseProfile& p) { return iso_profile_value(iso, p, IsoNoiseProfileParameter::SaturationLevel); }

// Application parameters use HWC data. read_noise_stddev and fpn_stddev are
// normalized signal standard deviations before their ISO scale is applied;
// shot_scale is photons (or electrons) per unit normalized signal. Profile
// points are borrowed host memory for CPU and device memory for CUDA.
struct IsoNoiseApplicationConfig {
  int width = 0;
  int height = 0;
  int channels = 0;
  float iso = 100.0f;
  IsoNoiseProfile profile{};
  float shot_scale = 100.0f;
  float read_noise_stddev = 0.0f;
  float fpn_stddev = 0.0f;
  float clip_min = 0.0f;
  float clip_max = 1.0f;
  std::uint64_t seed = 0;
};

// Samples one seeded Poisson shot-noise draw per channel, scaled by the
// profile's ISO-dependent shot_noise_scale. Input/output are interleaved
// HWC uint8 samples.
void iso_shot_noise_u8(const std::uint8_t* input, std::uint8_t* output, const IsoNoiseApplicationConfig& config, cudaStream_t stream = nullptr);
// Adds seeded Gaussian read noise scaled by the profile's read_noise_scale.
// Input/output are normalized HWC float32 samples.
void iso_read_noise_f32(const float* input, float* output, const IsoNoiseApplicationConfig& config, cudaStream_t stream = nullptr);
// Adds seeded fixed-pattern noise scaled by the profile's fpn_scale. Input
// and output are normalized HWC float32 samples.
void iso_fpn_f32(const float* input, float* output, const IsoNoiseApplicationConfig& config, cudaStream_t stream = nullptr);
// Applies the profile's ISO-dependent black-level floor. Input/output are
// interleaved HWC uint8 samples.
void iso_black_level_u8(const std::uint8_t* input, std::uint8_t* output, const IsoNoiseApplicationConfig& config, cudaStream_t stream = nullptr);
// Applies the profile's ISO-dependent saturation ceiling. Input/output are
// interleaved HWC uint8 samples.
void iso_saturation_level_u8(const std::uint8_t* input, std::uint8_t* output, const IsoNoiseApplicationConfig& config, cudaStream_t stream = nullptr);
// Applies both the black-level floor and the saturation ceiling in one pass.
// Input/output are interleaved HWC uint8 samples.
void iso_signal_levels_u8(const std::uint8_t* input, std::uint8_t* output, const IsoNoiseApplicationConfig& config, cudaStream_t stream = nullptr);

// ExposureTimeDarkCurrentConfig models dark-current charge in physical electron
// units. Input/output samples are normalized HWC float32 signal values. The
// profile gain converts the accumulated dark electrons to the normalized ADC
// domain; electrons_per_unit is the nominal full-scale electron count. A
// Poisson draw is made per sample, so exposure_seconds is a physical duration
// and dark_current_electrons_per_second is a sensor rate at the profile ISO.
// Profile knots are borrowed host memory for CPU and device memory for CUDA.
struct ExposureTimeDarkCurrentConfig {
  int width = 0;
  int height = 0;
  int channels = 0;
  float iso = 100.0f;
  IsoNoiseProfile profile{};
  float dark_current_electrons_per_second = 0.0f;
  float exposure_seconds = 1.0f;
  float electrons_per_unit = 1000.0f;
  float clip_min = 0.0f;
  float clip_max = 1.0f;
  std::uint64_t seed = 0;
};
using ExposureDarkCurrentConfig = ExposureTimeDarkCurrentConfig;
using IsoExposureTimeDarkCurrentConfig = ExposureTimeDarkCurrentConfig;
using IsoDarkCurrentConfig = ExposureTimeDarkCurrentConfig;
// Samples Poisson dark-current charge over exposure_seconds and adds it to
// normalized HWC float32 input.
void exposure_time_dark_current_f32(const float* input, float* output, const ExposureTimeDarkCurrentConfig& config, cudaStream_t stream = nullptr);
inline void exposure_dark_current_f32(const float* i, float* o, const ExposureTimeDarkCurrentConfig& c, cudaStream_t s = nullptr) { exposure_time_dark_current_f32(i, o, c, s); }
inline void iso_exposure_time_dark_current_f32(const float* i, float* o, const ExposureTimeDarkCurrentConfig& c, cudaStream_t s = nullptr) { exposure_time_dark_current_f32(i, o, c, s); }
inline void iso_dark_current_f32(const float* i, float* o, const ExposureTimeDarkCurrentConfig& c, cudaStream_t s = nullptr) { exposure_time_dark_current_f32(i, o, c, s); }

// TemperatureNoiseScalingConfig applies temperature-scaled Gaussian sensor
// noise in the electron domain. temperature_noise_coefficient_per_celsius is
// the natural-log slope: scale=exp(coefficient*(temperature-reference)). The
// ISO profile gain and read-noise scale are applied before conversion through
// electrons_per_unit. Samples and clipping remain normalized float32 values.
struct TemperatureNoiseScalingConfig {
  int width = 0;
  int height = 0;
  int channels = 0;
  float iso = 100.0f;
  IsoNoiseProfile profile{};
  float noise_stddev_electrons = 0.0f;
  float temperature_celsius = 25.0f;
  float reference_temperature_celsius = 25.0f;
  float temperature_noise_coefficient_per_celsius = 0.0f;
  float electrons_per_unit = 1000.0f;
  float clip_min = 0.0f;
  float clip_max = 1.0f;
  std::uint64_t seed = 0;
};
using TemperatureDependentNoiseScalingConfig = TemperatureNoiseScalingConfig;
using IsoTemperatureNoiseScalingConfig = TemperatureNoiseScalingConfig;
using TemperatureNoiseConfig = TemperatureNoiseScalingConfig;
// Adds temperature-scaled Gaussian sensor noise (see the coefficient formula
// above) to normalized HWC float32 input.
void temperature_noise_scaling_f32(const float* input, float* output, const TemperatureNoiseScalingConfig& config, cudaStream_t stream = nullptr);
inline void iso_temperature_noise_scaling_f32(const float* i, float* o, const TemperatureNoiseScalingConfig& c, cudaStream_t s = nullptr) { temperature_noise_scaling_f32(i, o, c, s); }
inline void temperature_dependent_noise_scaling_f32(const float* i, float* o, const TemperatureNoiseScalingConfig& c, cudaStream_t s = nullptr) { temperature_noise_scaling_f32(i, o, c, s); }
inline void temperature_dependent_noise_f32(const float* i, float* o, const TemperatureNoiseScalingConfig& c, cudaStream_t s = nullptr) { temperature_noise_scaling_f32(i, o, c, s); }

// DualConversionGainConfig models a two-conversion-gain readout.  Input and
// output are contiguous HWC float32 normalized signal units.  iso_to_gain(iso,
// profile) selects the operating point: below low_gain_threshold the low branch
// is used, above high_gain_threshold the high branch is used, and values between
// them are linearly blended.  Conversion gains are dimensionless signal ratios;
// the nominal low branch is normally 1.0.  All outputs are clipped to the
// explicit normalized clip interval.  Profile knots and optional maps are
// borrowed host memory for CPU and device memory for CUDA.
struct DualConversionGainConfig {
  int width = 0;
  int height = 0;
  int channels = 0;
  float iso = 100.0f;
  IsoNoiseProfile profile{};
  float low_gain_threshold = 1.0f;
  float high_gain_threshold = 4.0f;
  float low_conversion_gain = 1.0f;
  float high_conversion_gain = 1.0f;
  float low_read_noise_electrons = 0.0f;
  float high_read_noise_electrons = 0.0f;
  float electrons_per_unit = 1000.0f;
  float clip_min = 0.0f;
  float clip_max = 1.0f;
  std::uint64_t seed = 0;
};
// Blends between a low and high conversion-gain branch based on iso_to_gain,
// applying the branch's read noise; see the threshold contract above.
void dual_conversion_gain_f32(const float* input, float* output, const DualConversionGainConfig& config, cudaStream_t stream = nullptr);
using DualConversionGainSensorConfig = DualConversionGainConfig;
inline void dual_conversion_gain_sensor_f32(const float* i, float* o, const DualConversionGainConfig& c, cudaStream_t s = nullptr) { dual_conversion_gain_f32(i, o, c, s); }
inline void dual_conversion_gain_sensor_model_f32(const float* i, float* o, const DualConversionGainConfig& c, cudaStream_t s = nullptr) { dual_conversion_gain_f32(i, o, c, s); }

// GainSwitchTransitionConfig describes a stateless frame transition.  The
// hysteresis value widens the threshold interval by that many gain units on
// each side; transition_width controls the smooth blend around each threshold.
// initial_high_gain selects the side held by a caller's previous frame state.
// transition_strength adds a bounded mid-transition overshoot in normalized
// signal units.  This explicit state-in/state-out contract lets callers retain
// hysteresis across frames without hidden global state.
struct GainSwitchTransitionConfig {
  int width = 0;
  int height = 0;
  int channels = 0;
  float iso = 100.0f;
  IsoNoiseProfile profile{};
  float low_gain_threshold = 1.0f;
  float high_gain_threshold = 4.0f;
  float low_signal_gain = 1.0f;
  float high_signal_gain = 1.0f;
  float hysteresis = 0.0f;
  float transition_width = 0.0f;
  float transition_strength = 0.0f;
  bool initial_high_gain = false;
  float clip_min = 0.0f;
  float clip_max = 1.0f;
  std::uint64_t seed = 0;
};
// Applies a stateless gain-switch transition with hysteresis and a smooth
// blend around each threshold; see the field contract above.
void gain_switch_transition_f32(const float* input, float* output, const GainSwitchTransitionConfig& config, cudaStream_t stream = nullptr);
using GainSwitchArtifactsConfig = GainSwitchTransitionConfig;
inline void gain_switch_artifacts_f32(const float* i, float* o, const GainSwitchTransitionConfig& c, cudaStream_t s = nullptr) { gain_switch_transition_f32(i, o, c, s); }
inline void gain_switch_transition_artifacts_f32(const float* i, float* o, const GainSwitchTransitionConfig& c, cudaStream_t s = nullptr) { gain_switch_transition_f32(i, o, c, s); }

// Camera profile LUT entries extend the ISO profile with physical-domain
// baseline parameters. Values are borrowed, sorted by increasing ISO, and
// interpolated linearly with endpoint clamping. Gain and scale fields are
// dimensionless ratios; shot_scale is electrons per normalized signal unit;
// read_noise_stddev and fpn_stddev are normalized signal standard deviations.
struct CameraProfileLookupPoint {
  float iso = 100.0f;
  float gain = 1.0f, shot_noise_scale = 1.0f, read_noise_scale = 1.0f, fpn_scale = 1.0f;
  float black_level = 0.0f, saturation_level = 1.0f;
  float shot_scale = 100.0f, read_noise_stddev = 0.0f, fpn_stddev = 0.0f;
};
// A sorted-by-ISO camera profile LUT; points is borrowed host memory for CPU
// calls and device memory for CUDA calls.
struct CameraProfileLookupTable {
  const CameraProfileLookupPoint* points = nullptr;
  std::size_t point_count = 0;
};
enum class CameraProfileLookupParameter : int {
  Gain = 0, ShotNoiseScale = 1, ReadNoiseScale = 2, FPNScale = 3,
  BlackLevel = 4, SaturationLevel = 5, ShotScale = 6,
  ReadNoiseStddev = 7, FPNStddev = 8
};
#if defined(__CUDACC__)
#define AUGMATCH_CAMERA_HOST_DEVICE __host__ __device__
#else
#define AUGMATCH_CAMERA_HOST_DEVICE
#endif
AUGMATCH_CAMERA_HOST_DEVICE inline float camera_profile_parameter_value(const CameraProfileLookupPoint& p, CameraProfileLookupParameter parameter) {
  switch(parameter) {
    case CameraProfileLookupParameter::Gain: return p.gain;
    case CameraProfileLookupParameter::ShotNoiseScale: return p.shot_noise_scale;
    case CameraProfileLookupParameter::ReadNoiseScale: return p.read_noise_scale;
    case CameraProfileLookupParameter::FPNScale: return p.fpn_scale;
    case CameraProfileLookupParameter::BlackLevel: return p.black_level;
    case CameraProfileLookupParameter::SaturationLevel: return p.saturation_level;
    case CameraProfileLookupParameter::ShotScale: return p.shot_scale;
    case CameraProfileLookupParameter::ReadNoiseStddev: return p.read_noise_stddev;
    case CameraProfileLookupParameter::FPNStddev: return p.fpn_stddev;
  }
  return 0.0f;
}
AUGMATCH_CAMERA_HOST_DEVICE inline float camera_profile_lookup_value(float iso, const CameraProfileLookupTable& table, CameraProfileLookupParameter parameter) {
  if (!table.points || table.point_count==0) return 0.0f;
  if (table.point_count==1) return camera_profile_parameter_value(table.points[0],parameter);
  std::size_t hi=0; while(hi<table.point_count && table.points[hi].iso<iso) ++hi;
  if(hi==0) hi=1; if(hi>=table.point_count) hi=table.point_count-1;
  const auto& a=table.points[hi-1]; const auto& b=table.points[hi];
  const float d=b.iso-a.iso; const float raw=d>0.0f?(iso-a.iso)/d:0.0f;
  const float t=raw<0.0f?0.0f:(raw>1.0f?1.0f:raw);
  return camera_profile_parameter_value(a,parameter)+(camera_profile_parameter_value(b,parameter)-camera_profile_parameter_value(a,parameter))*t;
}
// Convenience wrappers around camera_profile_lookup_value for each parameter.
AUGMATCH_CAMERA_HOST_DEVICE inline float camera_profile_gain(float iso,const CameraProfileLookupTable& p){return camera_profile_lookup_value(iso,p,CameraProfileLookupParameter::Gain);}
AUGMATCH_CAMERA_HOST_DEVICE inline float camera_profile_shot_scale(float iso,const CameraProfileLookupTable& p){return camera_profile_lookup_value(iso,p,CameraProfileLookupParameter::ShotScale);}
AUGMATCH_CAMERA_HOST_DEVICE inline float camera_profile_read_noise_stddev(float iso,const CameraProfileLookupTable& p){return camera_profile_lookup_value(iso,p,CameraProfileLookupParameter::ReadNoiseStddev);}
AUGMATCH_CAMERA_HOST_DEVICE inline float camera_profile_fpn_stddev(float iso,const CameraProfileLookupTable& p){return camera_profile_lookup_value(iso,p,CameraProfileLookupParameter::FPNStddev);}
AUGMATCH_CAMERA_HOST_DEVICE inline float camera_profile_black_level(float iso,const CameraProfileLookupTable& p){return camera_profile_lookup_value(iso,p,CameraProfileLookupParameter::BlackLevel);}
AUGMATCH_CAMERA_HOST_DEVICE inline float camera_profile_saturation_level(float iso,const CameraProfileLookupTable& p){return camera_profile_lookup_value(iso,p,CameraProfileLookupParameter::SaturationLevel);}
struct CameraProfileLookupConfig {
  int width = 0;
  int height = 0;
  int channels = 0;
  float iso = 100.0f;
  CameraProfileLookupTable profile{};
  float clip_min = 0.0f;
  float clip_max = 1.0f;
  std::uint64_t seed = 0;
};
// Interpolates and applies camera_profile_lookup_value's noise parameters to
// normalized HWC float32 input.
void camera_profile_lookup_f32(const float* input, float* output, const CameraProfileLookupConfig& config, cudaStream_t stream = nullptr);
using CameraProfileLUTPoint = CameraProfileLookupPoint;
using CameraProfileLUT = CameraProfileLookupTable;
using CameraProfileLUTConfig = CameraProfileLookupConfig;
inline void camera_profile_lut_f32(const float* i, float* o, const CameraProfileLookupConfig& c, cudaStream_t s = nullptr){camera_profile_lookup_f32(i,o,c,s);}
inline void camera_profile_lookup_table_f32(const float* i, float* o, const CameraProfileLookupConfig& c, cudaStream_t s = nullptr){camera_profile_lookup_f32(i,o,c,s);}

// Calibration-derived values are normalized HWC quantities. Scalar fields are
// used when their map is null. Maps contain width*height*channels entries and
// are borrowed in the execution memory space (host for CPU, device for CUDA)
// until the supplied stream completes. read_noise_map is a standard deviation,
// fpn_map is an additive offset, and gain_map is a dimensionless multiplier.
struct CalibrationDerivedNoise {
  float shot_noise_scale = 0.0f, read_noise_stddev = 0.0f, fpn_stddev = 0.0f;
  const float* read_noise_map = nullptr;
  const float* fpn_map = nullptr;
  const float* gain_map = nullptr;
  std::size_t map_count = 0;
};
struct CalibrationFrameNoiseConfig {
  int width = 0;
  int height = 0;
  int channels = 0;
  CalibrationDerivedNoise calibration{};
  float clip_min = 0.0f;
  float clip_max = 1.0f;
  std::uint64_t seed = 0;
};
// Applies per-pixel calibration-derived shot/read/FPN/gain noise (scalar
// fields, or the borrowed maps when non-null) to normalized HWC float32 input.
void calibration_frame_noise_f32(const float* input, float* output, const CalibrationFrameNoiseConfig& config, cudaStream_t stream = nullptr);
using CalibrationNoiseParameters = CalibrationDerivedNoise;
using CalibrationFrameNoiseParameters = CalibrationDerivedNoise;
using CalibrationDerivedNoiseConfig = CalibrationFrameNoiseConfig;
using CalibrationFrameDrivenNoiseConfig = CalibrationFrameNoiseConfig;
using CameraProfile = CameraProfileLookupTable;
inline void calibration_noise_f32(const float* i, float* o, const CalibrationFrameNoiseConfig& c, cudaStream_t s = nullptr){calibration_frame_noise_f32(i,o,c,s);}
inline void calibration_frame_driven_noise_f32(const float* i, float* o, const CalibrationFrameNoiseConfig& c, cudaStream_t s = nullptr){calibration_frame_noise_f32(i,o,c,s);}

#undef AUGMATCH_CAMERA_HOST_DEVICE

using ISOProfilePoint = IsoNoiseProfilePoint;
using ISOProfile = IsoNoiseProfile;
using ISOProfileConfig = IsoNoiseApplicationConfig;
inline float iso_gain_from_iso(float iso, const IsoNoiseProfile& p) { return iso_to_gain(iso, p); }
inline float iso_to_analog_gain(float iso, const IsoNoiseProfile& p) { return iso_to_gain(iso, p); }
inline float iso_dependent_shot_noise_scale(float iso, const IsoNoiseProfile& p) { return iso_shot_noise_scale(iso, p); }
inline float iso_dependent_read_noise_scale(float iso, const IsoNoiseProfile& p) { return iso_read_noise_scale(iso, p); }
inline float iso_dependent_fpn_scale(float iso, const IsoNoiseProfile& p) { return iso_fpn_scale(iso, p); }
inline float iso_dependent_black_level(float iso, const IsoNoiseProfile& p) { return iso_black_level(iso, p); }
inline float iso_dependent_saturation_level(float iso, const IsoNoiseProfile& p) { return iso_saturation_level(iso, p); }
inline void iso_dependent_shot_noise_u8(const std::uint8_t* i, std::uint8_t* o, const IsoNoiseApplicationConfig& c, cudaStream_t s = nullptr) { iso_shot_noise_u8(i, o, c, s); }
inline void iso_dependent_read_noise_f32(const float* i, float* o, const IsoNoiseApplicationConfig& c, cudaStream_t s = nullptr) { iso_read_noise_f32(i, o, c, s); }
inline void iso_dependent_fpn_f32(const float* i, float* o, const IsoNoiseApplicationConfig& c, cudaStream_t s = nullptr) { iso_fpn_f32(i, o, c, s); }
inline void iso_dependent_black_level_u8(const std::uint8_t* i, std::uint8_t* o, const IsoNoiseApplicationConfig& c, cudaStream_t s = nullptr) { iso_black_level_u8(i, o, c, s); }
inline void iso_dependent_saturation_level_u8(const std::uint8_t* i, std::uint8_t* o, const IsoNoiseApplicationConfig& c, cudaStream_t s = nullptr) { iso_saturation_level_u8(i, o, c, s); }

#undef AUGMATCH_ISO_HOST_DEVICE
}
