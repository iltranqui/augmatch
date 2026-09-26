# Sensor noise, ISO profiles, and Bayer/CFA

Headers: `include/augmatch/sensor/`. Back to the [docs index](../README.md).

## Contents

- [Native sensor API contract](#native-sensor-api-contract)
- [PRNU and fixed-pattern offset contracts](#prnu-and-fixed-pattern-offset-contracts)
- [Row-column correlated sensor noise](#row-column-correlated-sensor-noise)
- [Clustered defective pixels](#clustered-defective-pixels)
- [Blooming and vertical smear](#blooming-and-vertical-smear)
- [Sensor dust and opaque-pixel masks](#sensor-dust-and-opaque-pixel-masks)
- [Bayer and CFA sampling contracts](#bayer-and-cfa-sampling-contracts)
- [CFA raw-plane effect contracts](#cfa-raw-plane-effect-contracts)
- [ISONoise contract](#isonoise-contract)
- [ShotNoise contract](#shotnoise-contract)
- [ISO and gain-dependent profile contract](#iso-and-gain-dependent-profile-contract)

## Native sensor API contract

`augmatch/sensor/native_api.hpp` is the C++17-first API for the remaining
NOISE_CATALOG implementation rows. It provides strided HWC/CHW uint8, uint16,
and float32 views, `RawBayerView`, borrowed channel/plane metadata, and owning
`CameraProfileConfig` LUTs. `CounterRng`, clipping/quantization policies, and
border policies are deterministic and explicit. `sensor::fused_pipeline` uses a
caller-owned `Workspace` (zero bytes for the current operation) and returns
`Status`; `additive_noise_batch` accepts borrowed view arrays. CUDA calls enqueue
on the supplied stream, while `additive_noise_async` is intentionally host-only
and requires callers to retain captured buffers until its future completes.
Profiles serialize as versioned `AUGMATCH_CAMERA_PROFILE_V1` text. The CMake
install exports `augmatch::augmatch` and `augmatchConfig.cmake`. Build
`native_api_example_cpp17` and `native_api_example_cpp23` for complete examples;
`tests/native_api_unit.cpp` covers the CPU reference contract.

## PRNU and fixed-pattern offset contracts

`pixel_response_non_uniformity_f32` and `fixed_pattern_offset_noise_f32` accept
contiguous interleaved HWC float32 data. PRNU computes `out[i] = in[i] * g[i]`,
where `g[i]` is either the borrowed `gain_map[i]` or
`1 + stddev * Z[i]`. FPN computes `out[i] = in[i] + o[i]`, where `o[i]` is
either the borrowed `offset_map[i]` or `stddev * Z[i]`. The seeded maps use the
same coordinate/channel counter hash on CPU and CUDA and are constant across
frames for a fixed seed. Null maps therefore model fixed sensor calibration,
not frame-varying noise. These float APIs do not clip or quantize their output;
callers can compose them with `sensor_noise_f32` or ADC stages afterward.

Map pointers are borrowed. CPU calls require host memory; CUDA calls require
device memory and retain the pointer until the supplied stream has completed.
The CLI forms are `prnu IN.f32 OUT.f32 W H C STDDEV SEED [MAP.f32]` and
`fpn IN.f32 OUT.f32 W H C STDDEV SEED [MAP.f32]`; map fixtures contain
`W*H*C` little-endian float32 values. Full operation-name aliases are accepted.

## Row-column correlated sensor noise

`row_column_correlated_noise_f32` adds one offset shared by every pixel in a row and one offset shared by every pixel in a column, inducing fixed spatial correlation while retaining channel-specific maps. A supplied `row_map` contains exactly `height*channels` normalized offsets and a supplied `column_map` contains `width*channels`; both pointers are borrowed. Null maps are generated deterministically from `seed`. CPU maps must be host allocations; CUDA maps must be device allocations and remain valid until the stream completes. The output is clipped to `[clip_min,clip_max]` (the default is `[0,1]`). The CLI is `row_column_correlated_noise IN.f32 OUT.f32 W H C ROW_STDDEV COLUMN_STDDEV SEED [ROW_MAP.f32 COLUMN_MAP.f32]`; `row_column_noise` is an alias.

## Clustered defective pixels

`clustered_defective_pixels_f32` applies circular dead, hot, and stuck-pixel defects to normalized HWC float data. Explicit `ClusteredDefect` records are borrowed and use pixel-coordinate centers, inclusive radii, a channel (`-1` means all channels), and a normalized value. Hot values are additive; stuck values replace the signal; dead pixels become zero. Results are clipped to `[0,1]`. If no records are supplied, `cluster_count` deterministic clusters are generated from `seed` with `cluster_radius`; generated clusters choose their type deterministically and use `hot_value` or `stuck_value`. CUDA record arrays must be device memory and remain valid until the stream completes, while CPU records must be host memory. The CLI is `clustered_defective_pixels IN.f32 OUT.f32 W H C CLUSTER_COUNT RADIUS HOT_VALUE STUCK_VALUE SEED [RECORDS.bin]`; `clustered_defects` is an alias. A record fixture is a packed array of `ClusteredDefect` values.

## Blooming and vertical smear

`blooming_vertical_smear_f32` accepts normalized HWC float32 data. Explicit borrowed `BrightPixel` records `(x,y,value,channel)` are deterministic sources; `channel=-1` applies to all channels. Each source adds `max(value-threshold,0) * strength * decay^(dy)` down its column, and output is clipped. With no records, input samples above the threshold become sources. An optional borrowed `column_smear_map` has `width*channels` row-zero amplitudes and is decayed down each column. CPU records/maps are host memory; CUDA records/maps are device memory and remain valid through the stream. The CLI is `blooming_vertical_smear IN.f32 OUT.f32 W H C THRESHOLD STRENGTH DECAY SEED [BRIGHT_PIXELS.bin [COLUMN_MAP.f32]]`.

## Sensor dust and opaque-pixel masks

`sensor_dust_opaque_mask_f32` uses a borrowed row-major HxW uint8 mask on normalized HWC float data. Zero copies exactly. Nonzero pixels receive `fill` when `blur_radius` is zero, or the mean of unmasked samples in the clipped square neighborhood when the radius is positive; no available neighbor falls back to `fill`. Masked results are clipped to `[0,1]`; unmasked values are preserved. A null mask is identity. CPU masks are host memory and CUDA masks are device memory valid through the stream. The CLI is `sensor_dust_opaque_mask IN.f32 OUT.f32 W H C BLUR_RADIUS FILL MASK.raw`.

## Bayer and CFA sampling contracts

`quad_bayer_sample_u8` samples HWC RGB8 input with an explicit 4x4 expanded Bayer pattern (`QuadBayerPattern`) and writes one raw HxW plane. `rgbw_sample_u8` samples HWC RGBW8 input with channel ownership R=0, G=1, B=2, W=3 and an explicit 2x2 `RGBWPattern`; it also writes one raw plane. These functions only sample and never demosaic. `custom_cfa_sample_u8` takes HWC input plus a borrowed repeating byte mask. Each mask byte is an input channel index, and values outside `[0, channels)` are invalid. CPU masks are host-owned; CUDA masks are device-owned until stream completion. The CLI forms are `quad_bayer IN.raw OUT.raw W H PATTERN`, `rgbw IN.raw OUT.raw W H PATTERN`, and `custom_cfa IN.raw OUT.raw W H CHANNELS MASK_W MASK_H MASK.raw`. The existing Bayer API and all three new CFA APIs are included by `augmatch/augmatch.hpp`.

## CFA raw-plane effect contracts

The six CFA effect APIs consume and produce contiguous HxW float32 raw planes. `CfaPlaneMap` explicitly owns the repeating 2x2 tile and uses plane ownership 0=R, 1=Gr, 2=Gb, 3=B; call `bayer_plane_map` for canonical RGGB/BGGR/GRBG/GBRG maps. Maps are copied by value. A `missing_mask` is borrowed row-major HxW memory: host-owned on CPU and device-owned until CUDA stream completion. Response variation, leakage, integer misregistration, missing samples, plane noise, and plane gain all validate parameters and clip to the configured range. Leakage uses a row-major target/source 4x4 matrix with deterministic radius-two nearest-plane lookup; missing samples support zero, fill, and nearest same-plane replacement. Seeded operations use coordinate-stable counter keys, so repeated calls with the same seed are deterministic. Raw planes have no retained channel metadata. CLI forms are `cfa_channel_response IN.f32 OUT.f32 W H R GR GB B`, `cfa_leakage IN.f32 OUT.f32 W H MATRIX16`, `cfa_misregistration IN.f32 OUT.f32 W H DX4 DY4`, `cfa_missing_samples IN.f32 OUT.f32 W H PROBABILITY REPLACEMENT FILL SEED`, `bayer_plane_noise IN.f32 OUT.f32 W H SIGMA4`, and `bayer_plane_gain IN.f32 OUT.f32 W H GAIN4`.

## ISONoise contract

Color/photometric APIs (`color_matrix_perturbation_u8`, `camera_color_profile_variation_u8`, `rgb_channel_cross_talk_u8`, `sensor_spectral_response_variation_u8`, `color_clipping_u8`, `white_balance_clipping_u8`, `chroma_noise_u8`, `luma_noise_u8`, and `correlated_luma_chroma_noise_u8`) accept interleaved HWC uint8 RGB(A). They use normalized full-range RGB with BT.601 Y/Cb/Cr for luma/chroma operations, preserve channels >= 3, and saturate half-up to uint8. Matrix coefficients are row-major; profile and spectral noise are seeded normalized Gaussian perturbations. CPU pointers are host memory and CUDA pointers are device memory valid through the stream.

`iso_noise_u8` accepts interleaved HWC `uint8` data and applies the explicit gain `G = (iso / base_iso) * analog_gain`. The signal is first multiplied by `digital_gain`. For each pixel, the seeded SplitMix64 counter/hash RNG derives one shared standard-normal luma sample and one sample per channel. Gaussian luma noise is `gaussian_stddev * G * Z_luma`. For RGB data, chroma noise is luma-preserving: `chroma_stddev * G * (Z_channel - 0.2126 Z_R - 0.7152 Z_G - 0.0722 Z_B)`. Chroma noise is disabled for fewer than three channels. The result is clipped to `[0,1]` and rounded to `uint8`; no global RNG state is used. CPU and CUDA use the same coordinate/channel hash and the CLI accepts `iso_noise IN.raw OUT.raw W H C ISO BASE_ISO ANALOG_GAIN DIGITAL_GAIN GAUSSIAN_STDDEV CHROMA_STDDEV SEED` (the `isonoise` alias is also accepted).

## ShotNoise contract

`shot_noise_u8` accepts interleaved HWC `uint8` data and samples one seeded Poisson
count per channel. For normalized input `v = I/255`, the photon rate is
`lambda = v * gain * scale`; the sampled signal is `Poisson(lambda) / scale`,
so `gain` controls the mean output signal and `scale` controls photon count and
shot-noise variance. The result is clipped to `[0,1]` and rounded to `uint8`.
The CPU and CUDA paths use the same coordinate/channel SplitMix64 counter key,
Poisson inversion below 64 photons, and bounded normal approximation above it.
There is no global RNG state. `gain` must be finite and nonnegative; `scale`
must be finite and positive. The CLI form is
`shot_noise IN.raw OUT.raw W H C GAIN SCALE SEED` (the `shotnoise` alias is
also accepted).

## ISO and gain-dependent profile contract

`IsoNoiseProfile` stores sorted ISO knots and evaluates gain, shot/read/FPN
scales, black level, and saturation level with clamped piecewise-linear
interpolation. Scale and gain values are dimensionless; levels are normalized
ADC fractions. The profile points are borrowed host memory for CPU calls and
device memory for CUDA calls. `IsoNoiseApplicationConfig` applies one profile
parameter to HWC data with deterministic seeded sampling. `shot_scale` is the
photon/electron count per normalized signal unit, read and FPN deviations are
normalized units, and float outputs clip to `clip_min..clip_max`. Level APIs
convert normalized levels to uint8 codes with nearest rounding; the combined
level API applies black floor then saturation ceiling.

The CLI accepts a little-endian float32 knot file (seven values per knot in
`IsoNoiseProfilePoint` order):

```sh
augmatch_cli iso_profile IN OUT W H C ISO KNOTS_F32 COUNT SHOT_SCALE READ_STD FPN_STD SEED OP
```

`OP` is `shot`, `read`, `fpn`, `black`, `saturation`, or `levels`. The CLI
expects uint8 input/output for `shot`, `black`, `saturation`, and `levels`, and
float32 normalized input/output for `read` and `fpn`.

Physical ISO/profile operations use `ExposureTimeDarkCurrentConfig` and
`TemperatureNoiseScalingConfig`. Their HWC float32 samples are normalized,
while `dark_current_electrons_per_second`, `noise_stddev_electrons`, and
`electrons_per_unit` are electron-domain scalars. Dark current is Poisson
charge over the requested exposure and is scaled by profile ISO gain;
temperature noise uses `exp(coefficient_per_celsius * (temperature-reference))`
and the profile read-noise scale. Both clip normalized output and use a
counter-based seed, with borrowed host/device profile storage matching the
other ISO APIs. Their CLI adapters are:

```sh
augmatch_cli iso_dark_current IN OUT W H C ISO KNOTS_F32 COUNT RATE_EPS EXPOSURE_SECONDS ELECTRONS_PER_UNIT SEED
augmatch_cli iso_temperature_noise IN OUT W H C ISO KNOTS_F32 COUNT NOISE_E TEMP_C REF_C COEFF_PER_C ELECTRONS_PER_UNIT SEED
```

Dual conversion gain and gain-switch transitions reuse the interpolated profile
analog gain. `low_gain_threshold` and `high_gain_threshold` are dimensionless
and explicit. Conversion and signal gains are dimensionless ratios; read-noise
and `electrons_per_unit` are electrons. Normalized float32 HWC outputs are
clipped, and borrowed profile knots follow the CPU host/CUDA device ownership
contract. `GainSwitchTransitionConfig::hysteresis` is a per-side gain deadband,
`transition_width` is the smooth blend width, and `initial_high_gain` supplies
caller-owned previous-frame state; implementations retain no hidden state.
Both operations are deterministic for a fixed seed:

```sh
augmatch_cli dual_conversion_gain IN OUT W H C ISO KNOTS_F32 COUNT LOW_T HIGH_T LOW_GAIN HIGH_GAIN LOW_READ_E HIGH_READ_E ELECTRONS_PER_UNIT SEED
augmatch_cli gain_switch_transition IN OUT W H C ISO KNOTS_F32 COUNT LOW_T HIGH_T LOW_SIGNAL HIGH_SIGNAL HYSTERESIS TRANSITION_WIDTH STRENGTH INITIAL_HIGH SEED
```
