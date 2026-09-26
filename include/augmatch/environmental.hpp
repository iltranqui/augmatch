#pragma once

#include <cstddef>
#include <cstdint>
#include "augmatch/noise.hpp"
#include "augmatch/signal.hpp"
#include "augmatch/weather.hpp"

namespace augmatch {

// Environmental veil operations use contiguous HWC float32 samples. `map` is
// a borrowed HxW mask when map_channels==1, or HxWxC when map_channels==C;
// CPU callers own host maps and CUDA callers own device maps through stream
// completion. A null map means a uniform mask of one. Values are clamped to
// [0,1], and output is clipped to [clip_min,clip_max].
struct EnvironmentalVeilConfig {
  int width=0, height=0, channels=0;
  const float* map=nullptr;
  int map_channels=1;
  float strength=0.0f;
  float airlight=1.0f;
  float clip_min=0.0f, clip_max=1.0f;
};
void environmental_veil_f32(const float*, float*, const EnvironmentalVeilConfig&, cudaStream_t stream=nullptr);
void atmospheric_haze_f32(const float*, float*, const EnvironmentalVeilConfig&, cudaStream_t stream=nullptr);
void smoke_veil_f32(const float*, float*, const EnvironmentalVeilConfig&, cudaStream_t stream=nullptr);
void window_glare_f32(const float*, float*, const EnvironmentalVeilConfig&, cudaStream_t stream=nullptr);
void backlight_washout_f32(const float*, float*, const EnvironmentalVeilConfig&, cudaStream_t stream=nullptr);

// Gain operations are deterministic multiplicative approximations. Gain must
// be finite and nonnegative; clipping happens after multiplication.
struct AcquisitionGainConfig {
  int width=0, height=0, channels=0;
  float gain=1.0f;
  float clip_min=0.0f, clip_max=1.0f;
};
void acquisition_gain_f32(const float*, float*, const AcquisitionGainConfig&, cudaStream_t stream=nullptr);
void low_light_amplification_f32(const float*, float*, const AcquisitionGainConfig&, cudaStream_t stream=nullptr);
void underexposure_f32(const float*, float*, const AcquisitionGainConfig&, cudaStream_t stream=nullptr);
void overexposure_f32(const float*, float*, const AcquisitionGainConfig&, cudaStream_t stream=nullptr);

// Dirty-lens blur is a bounded square-box PSF blended by a borrowed opacity map.
// The map is HxW when map_channels==1 or HxWxC when map_channels==channels;
// null means opacity one. CPU maps are host-owned and CUDA maps are device-owned
// through stream completion. Radius is limited to 16, strength and opacity are
// clamped to [0,1], and output is clipped. This is a deterministic dirt surrogate,
// not a particulate lens renderer.
struct DirtyLensBlurConfig {
  int width=0,height=0,channels=0;
  const float* map=nullptr; int map_channels=1;
  int radius=1; float strength=1.0f;
  float clip_min=0.0f,clip_max=1.0f;
};
void dirty_lens_blur_f32(const float*,float*,const DirtyLensBlurConfig&,cudaStream_t stream=nullptr);

// Sensor temperature drift consumes a contiguous T-H-W-C float32 batch. The
// temperature follows a linear start-to-end ramp over frames. Dark current is
// approximated by D*exposure*exp(coefficient*(temperature-reference))/electrons
// and added to every sample; optional sensitivity_map is borrowed HWC and scales
// that offset (1 or channels map layout). Values are clipped and no state is
// retained. This intentionally reuses the temporal batch layout without adding
// stochastic sensor physics.
struct SensorTemperatureDriftConfig : TemporalBatchConfig {
  float start_temperature_celsius=25.0f,end_temperature_celsius=25.0f;
  float reference_temperature_celsius=25.0f;
  float dark_current_electrons_per_second=0.0f;
  float exposure_seconds=1.0f,electrons_per_unit=1000.0f;
  float temperature_coefficient_per_celsius=0.0f;
  const float* sensitivity_map=nullptr; int map_channels=1;
};
void sensor_temperature_drift_f32(const float*,float*,const SensorTemperatureDriftConfig&,cudaStream_t stream=nullptr);

// Electromagnetic interference is a deterministic sinusoidal additive field in
// normalized HWC float32. Optional HxW/HxWxC map values are nonnegative local
// amplitude multipliers. frequency_x/y are cycles per image, phase is radians;
// output is clipped. It is a reproducible EMI approximation, not an RF model.
struct ElectromagneticInterferenceConfig {
  int width=0,height=0,channels=0;
  const float* map=nullptr; int map_channels=1;
  float amplitude=0.0f,frequency_x=0.0f,frequency_y=0.0f,phase=0.0f;
  float clip_min=0.0f,clip_max=1.0f;
};
void electromagnetic_interference_f32(const float*,float*,const ElectromagneticInterferenceConfig&,cudaStream_t stream=nullptr);

// Power-supply banding consumes T-H-W-C data and applies a row-periodic
// multiplicative ripple: 1 + amplitude*sin(2*pi*(temporal_frequency*t/frame_rate
// + row_frequency*y/height)+phase). An optional row map has H or HxC entries
// and scales the amplitude. Output is clipped and deterministic.
struct PowerSupplyBandingConfig : TemporalBatchConfig {
  float amplitude=0.0f,temporal_frequency_hz=0.0f,frame_rate_hz=1.0f;
  float row_frequency=0.0f,phase=0.0f;
  const float* row_map=nullptr; int row_map_channels=1;
};
void power_supply_banding_f32(const float*,float*,const PowerSupplyBandingConfig&,cudaStream_t stream=nullptr);

// Fluorescent-light flicker is a frame-global sinusoidal exposure factor. The
// phase is sampled at t/frame_rate_hz, so the contract is independent of image
// dimensions. This deterministic model reuses the temporal batch/clipping API
// but does not claim ballast waveform or mains-phase fidelity.
struct FluorescentLightFlickerConfig : TemporalBatchConfig {
  float amplitude=0.0f,flicker_frequency_hz=0.0f,frame_rate_hz=1.0f,phase=0.0f;
};
void fluorescent_light_flicker_f32(const float*,float*,const FluorescentLightFlickerConfig&,cudaStream_t stream=nullptr);

// LED rolling-band artifacts use a row phase plus a frame phase over T-H-W-C:
// 1 + amplitude*sin(2*pi*(frequency*t/frame_rate + row_cycles*y/height)+phase).
// This is a clipped rolling-shutter modulation surrogate; it does not simulate
// LED driver PWM, exposure integration, or per-row sensor timing.
struct LedRollingBandArtifactsConfig : TemporalBatchConfig {
  float amplitude=0.0f,modulation_frequency_hz=0.0f,frame_rate_hz=1.0f;
  float row_cycles=0.0f,phase=0.0f;
};
void led_rolling_band_artifacts_f32(const float*,float*,const LedRollingBandArtifactsConfig&,cudaStream_t stream=nullptr);
using TemperatureDriftConfig = SensorTemperatureDriftConfig;
using EMIConfig = ElectromagneticInterferenceConfig;
using PowerBandingConfig = PowerSupplyBandingConfig;
using FluorescentFlickerConfig = FluorescentLightFlickerConfig;
using LedRollingBandConfig = LedRollingBandArtifactsConfig;
inline void temperature_drift_f32(const float* i,float* o,const SensorTemperatureDriftConfig& c,cudaStream_t s=nullptr) { sensor_temperature_drift_f32(i,o,c,s); }
inline void emi_f32(const float* i,float* o,const ElectromagneticInterferenceConfig& c,cudaStream_t s=nullptr) { electromagnetic_interference_f32(i,o,c,s); }
inline void power_banding_f32(const float* i,float* o,const PowerSupplyBandingConfig& c,cudaStream_t s=nullptr) { power_supply_banding_f32(i,o,c,s); }
inline void fluorescent_flicker_f32(const float* i,float* o,const FluorescentLightFlickerConfig& c,cudaStream_t s=nullptr) { fluorescent_light_flicker_f32(i,o,c,s); }
inline void led_rolling_band_f32(const float* i,float* o,const LedRollingBandArtifactsConfig& c,cudaStream_t s=nullptr) { led_rolling_band_artifacts_f32(i,o,c,s); }

// uint8 aliases reuse the established signal gain/clipping contract.
using AcquisitionGainU8Config = SignalGainConfig;
inline void low_light_amplification_u8(const std::uint8_t* i,std::uint8_t* o,const AcquisitionGainU8Config& c,cudaStream_t s=nullptr) { analog_gain_u8(i,o,c,s); }
inline void underexposure_u8(const std::uint8_t* i,std::uint8_t* o,const AcquisitionGainU8Config& c,cudaStream_t s=nullptr) { exposure_scale_u8(i,o,c,s); }
inline void overexposure_u8(const std::uint8_t* i,std::uint8_t* o,const AcquisitionGainU8Config& c,cudaStream_t s=nullptr) { exposure_scale_u8(i,o,c,s); }

// Reflections reuse the deterministic nearest/clamped ghost model. Records
// and optional maps have the same borrowed-memory ownership as GhostingConfig.
using ReflectionsConfig = GhostingConfig;
inline void reflections_f32(const float* i,float* o,const ReflectionsConfig& c,cudaStream_t s=nullptr) { ghosting_f32(i,o,c,s); }
inline void reflection_f32(const float* i,float* o,const ReflectionsConfig& c,cudaStream_t s=nullptr) { ghosting_f32(i,o,c,s); }

// Catalog aliases retain the explicit seeded weather contracts.
using FogVeilConfig = RandomFogConfig;
using RainStreaksConfig = RandomRainConfig;
using SnowOcclusionConfig = RandomSnowConfig;
using DustParticlesConfig = RandomGravelConfig;
using HeadlightFlareConfig = FlareConfig;
inline void fog_veil_u8(const std::uint8_t* i,std::uint8_t* o,const FogVeilConfig& c,cudaStream_t s=nullptr) { random_fog_u8(i,o,c,s); }
inline void rain_streaks_u8(const std::uint8_t* i,std::uint8_t* o,const RainStreaksConfig& c,cudaStream_t s=nullptr) { random_rain_u8(i,o,c,s); }
inline void snow_occlusion_u8(const std::uint8_t* i,std::uint8_t* o,const SnowOcclusionConfig& c,cudaStream_t s=nullptr) { random_snow_u8(i,o,c,s); }
inline void dust_particles_u8(const std::uint8_t* i,std::uint8_t* o,const DustParticlesConfig& c,cudaStream_t s=nullptr) { random_gravel_u8(i,o,c,s); }
inline void headlight_flare_f32(const float* i,float* o,const HeadlightFlareConfig& c,cudaStream_t s=nullptr) { flare_f32(i,o,c,s); }

}  // namespace augmatch
