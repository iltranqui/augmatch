# Camera, Sensor, Optical, ISP, and Compression Noise Catalog

This is the implementation checklist for realistic camera-domain corruption in Augmatch. These operations are separate from generic Albumentations/imgaug noise transforms.

## 13. Native NOISE_CATALOG API contract

- [x] Image-view dtype and layout support
- [x] Per-channel and per-plane metadata
- [x] Counter-based deterministic RNG
- [x] Clipping and quantization policies
- [x] Configurable border policies
- [x] Explicit status results
- [x] CUDA stream-aware view execution

## 1. Sensor signal model

Use the common forward model:

```text
I_raw = ADC( gain * Poisson(I / gain)
           + Gaussian(0, sigma_read^2)
           + N_row
           + N_column
           + N_FPN
           + N_dark )
```

Each operation must support deterministic seeds, CPU execution, CUDA execution, clipping, dtype conversion, and configurable channel behavior.

### View, metadata, RNG, and status contract

`noise_view.hpp` adds `make_hwc_view<T>`/`make_chw_view<T>` support for
strided `uint8`, `uint16`, and `float32` image views. `additive_noise_view`
accepts host views on either layout and returns an explicit `Status`; invalid
view, metadata, parameter, unsupported-memory, and CUDA-launch failures are
represented by `StatusCode` values. The CUDA implementation accepts device
views and borrowed device metadata and enqueues work on the caller's
`cudaStream_t` without synchronizing. The CPU implementation deliberately
returns `Unsupported` for device views.

`NoiseChannelMetadata` and `NoisePlaneMetadata` carry per-channel and
per-plane gain, bias, and standard-deviation scale. A borrowed channel-to-plane
map makes Bayer-plane ownership explicit. `CounterRng` is stateless and keys
each draw by seed, logical x/y coordinate, channel, plane, and stream; repeated
calls are deterministic and independent of traversal order. `NoiseClipPolicy`,
`NoiseQuantizationPolicy`, and `NoiseBorderPolicy` make clipping, integer
conversion, and neighborhood border behavior explicit. `resolve_noise_coordinate`
is a host metadata helper; no unsupported catalog transform is marked complete
by this infrastructure alone.

PRNU and FPN are available as `pixel_response_non_uniformity_f32` and
`fixed_pattern_offset_noise_f32`. Both operate on contiguous interleaved HWC
float32 samples without implicit clipping: PRNU multiplies by a per-sample gain
and FPN adds a per-sample offset. A null map deterministically generates an
HWC map from `seed` and `stddev`; a supplied map is borrowed and must contain
`width*height*channels` float32 values. CPU callers own host maps, while CUDA
callers own device maps for the duration of the asynchronous stream operation.
The CLI accepts `prnu`/`fpn` with optional little-endian float32 map fixtures.

ADC DNL and INL operate on normalized contiguous HWC float32 data and reuse
nearest-code quantization with `levels` codes. `AdcDifferentialNonLinearityConfig`
uses a channel-major `channels*levels` LUT of positive code-width multipliers;
endpoint bins have half width, so an all-ones LUT is the ordinary ADC
quantizer. A null LUT uses `1 + stddev*N(0,1)` per `(seed, channel, code)`.
`AdcIntegralNonLinearityConfig` uses a channel-major LUT of code offsets in ADC
LSBs, or seeded `stddev*N(0,1)` LSB offsets. Both outputs are clipped to
`[0,1]`. `SensorWellCapacityVariationConfig` uses a per-sample HWC capacity
ratio map relative to nominal full well, or seeded `1 + stddev*N(0,1)` ratios;
its normalized output is clipped to `[0,min(1,capacity_ratio)]`. LUT/map values
are borrowed: CPU callers provide host memory and CUDA callers provide device
memory valid through the supplied stream operation. The CLI commands `adc_dnl`, `adc_inl`, and `well_capacity` read/write
little-endian float32 fixtures. Their forms are
`adc_dnl IN OUT W H C LEVELS STDDEV SEED [LUT_F32]`,
`adc_inl IN OUT W H C LEVELS STDDEV SEED [LUT_F32]`, and
`well_capacity IN OUT W H C STDDEV SEED [CAPACITY_MAP_F32]`. The optional
DNL/INL LUTs have `C*LEVELS` entries; the optional well map has `W*H*C`
entries. Aliases with the full catalog names are also accepted.

`random_telegraph_signal_noise_f32` is a contiguous HWC float32 operation.
`RandomTelegraphSignalNoiseConfig` applies a low or high offset selected by an
explicit borrowed HWC uint8 state map (zero=low, nonzero=high), or by an
independent per-sample toggle of `initial_state` with coordinate-keyed
`transition_probability` and `seed`. The coordinate RNG makes results
traversal-independent and consistent between CPU and CUDA; no state is
retained between calls. This single-exposure independent-state model is a
native approximation and does not model temporal dwell-time correlation or
state evolution across exposures. CPU maps are host-owned and CUDA maps are
device-owned through the stream operation. Results are clipped to the
configured range. The CLI form is
`random_telegraph_signal_noise IN OUT W H C LOW_OFFSET HIGH_OFFSET PROBABILITY INITIAL_STATE SEED`.

- [~] Exposure scaling.
- [~] Analog gain.
- [~] Digital gain.
- [~] Signal-dependent Poisson shot noise.
- [~] Gaussian read noise.
- [~] Correlated read noise.
- [~] Amplifier noise.
- [~] Reset noise / kTC noise.
- [~] Dark current noise.
- [~] Dark-frame offset.
- [~] Temperature-dependent dark current.
- [x] Pixel-response non-uniformity (PRNU).
- [x] Fixed-pattern offset noise (FPN).
- [~] Row-wise banding noise.
- [~] Column-wise banding noise.
- [~] Row-column correlated noise.
- [x] Random telegraph signal noise.
- [~] Hot pixels.
- [~] Dead pixels.
- [~] Stuck pixels.
- [~] Clustered defective pixels.
- [~] Pixel saturation.
- [~] Black-level offset.
- [~] White-level variation.
- [~] ADC quantization.
- [x] ADC differential non-linearity.
- [x] ADC integral non-linearity.
- [~] ADC clipping.
- [~] Bit-depth reduction.
- [~] Bit truncation.
- [x] Sensor well-capacity variation.
- [~] Pixel cross-talk.
- [~] Charge leakage.
- [x] Blooming and vertical smear.
- [x] Sensor dust and opaque-pixel masks.

### Blooming and vertical smear

`blooming_vertical_smear_f32` accepts normalized HWC float32 data and applies deterministic downward column smear. Explicit borrowed `BrightPixel` records provide `(x,y,value,channel)` sources; `channel=-1` applies to every channel. A source contributes `max(value-bright_threshold,0) * smear_strength * smear_decay^(output_y-source_y)` to pixels below it in the same column, then the result is clipped to `[clip_min,clip_max]`. With no records, every input sample above `bright_threshold` is used as a source. An optional borrowed `column_smear_map` contains `width*channels` nonnegative row-zero source amplitudes and is decayed down its column. Records and maps are host memory for CPU calls and device memory for CUDA calls; CUDA storage must remain valid until the supplied stream completes. The operation has no random draws; `seed` is reserved for API stability. The CLI form is `blooming_vertical_smear IN.f32 OUT.f32 W H C THRESHOLD STRENGTH DECAY SEED [BRIGHT_PIXELS.bin [COLUMN_MAP.f32]]`; aliases `blooming_and_vertical_smear` and `blooming_smear` are accepted.

### Sensor dust and opaque-pixel masks

`sensor_dust_opaque_mask_f32` accepts normalized HWC float32 data and a borrowed row-major `width*height` uint8 mask. Zero mask bytes copy every channel exactly; nonzero bytes are opaque. With `blur_radius=0`, an opaque pixel receives `fill`. With a positive radius, each opaque channel receives the mean of unmasked input samples in the clipped square neighborhood; if no unmasked sample is available, it receives `fill`. The masked result is clipped to `[0,1]`, while unmasked samples are not changed. CPU masks are host memory, CUDA masks are device memory and must remain valid until the supplied stream completes. A null mask is an identity operation. The CLI form is `sensor_dust_opaque_mask IN.f32 OUT.f32 W H C BLUR_RADIUS FILL MASK.raw`; aliases `sensor_dust_and_opaque_pixel_mask` and `sensor_dust` are accepted.

## 2. ISO and gain-dependent noise

Noise parameters must vary with ISO, analog gain, exposure, temperature, and sensor profile.

- [x] ISO-to-gain mapping.
- [x] ISO-dependent shot-noise scale.
- [x] ISO-dependent read-noise scale.
- [x] ISO-dependent FPN scale.
- [x] ISO-dependent black level.
- [x] ISO-dependent saturation level.

### ISO and gain profile contract

`IsoNoiseProfile` is an explicit, borrowed array of `IsoNoiseProfilePoint` knots. Knots are sorted by strictly increasing ISO and are evaluated with piecewise-linear interpolation; values below and above the table clamp to the first and last knot. `gain`, `shot_noise_scale`, `read_noise_scale`, and `fpn_scale` are dimensionless ratios. `black_level` and `saturation_level` are normalized ADC fractions in `[0,1]`, with black no greater than saturation. `iso_to_gain`, `iso_shot_noise_scale`, `iso_read_noise_scale`, `iso_fpn_scale`, `iso_black_level`, and `iso_saturation_level` are host/device scalar helpers.

`IsoNoiseApplicationConfig` uses HWC data. `shot_scale` is photons (or electrons) per normalized signal unit; it is multiplied by the profile shot scale and the profile gain is applied to the expected signal. Read and FPN standard deviations are normalized signal units before their profile multipliers. The read/FPN operations clip to `clip_min..clip_max`. The black and saturation operations clamp uint8 samples to the profile's normalized levels (rounded to ADC codes); `iso_signal_levels_u8` applies both in that order. Profile knots are borrowed host memory for CPU calls and borrowed device memory for CUDA calls, and must remain valid until the supplied stream completes. The operations are deterministic for a given seed and clip outputs rather than wrapping.

`ExposureTimeDarkCurrentConfig` and `TemperatureNoiseScalingConfig` make the
physical-domain assumptions explicit. Both consume normalized contiguous HWC
float32 samples and require a borrowed `IsoNoiseProfile`;
`electrons_per_unit` is the nominal electron count represented by one
normalized full-scale unit. The dark-current rate is in electrons/second and
is multiplied by `exposure_seconds` and the interpolated ISO gain before a
counter-based Poisson draw. The draw is divided by `electrons_per_unit`, added
to the input, and clipped to `clip_min..clip_max`. Thus zero exposure is an
identity operation, and equal seeds produce equal CPU/CUDA samples.
`exposure_time_dark_current_f32` (and aliases `exposure_dark_current_f32` and
`iso_exposure_time_dark_current_f32`) exposes this operation.

Temperature noise uses `noise_stddev_electrons` before ISO gain and the profile
read-noise scale. Its dimensionless thermal factor is
`exp(temperature_noise_coefficient_per_celsius *
(temperature_celsius-reference_temperature_celsius))`; the result is converted
through `electrons_per_unit`, added as deterministic seeded Gaussian noise, and
clipped. `temperature_noise_scaling_f32` and
`iso_temperature_noise_scaling_f32` provide CPU and CUDA application APIs.
The CLI forms are `iso_dark_current IN OUT W H C ISO KNOTS_F32 COUNT RATE_EPS
EXPOSURE_SECONDS ELECTRONS_PER_UNIT SEED` and `iso_temperature_noise IN OUT W H
C ISO KNOTS_F32 COUNT NOISE_E TEMP_C REF_C COEFF_PER_C ELECTRONS_PER_UNIT SEED`.
Profile knots remain the existing seven little-endian float32 fields, so prior
ISO profile fixtures and APIs remain valid.
- [x] Exposure-time-dependent dark current.
- [x] Temperature-dependent noise scaling.
- [~] Per-channel gain variation.
- [x] Dual-conversion-gain sensor model.
- [x] Gain-switch transition artifacts.
- [x] Camera-profile lookup-table parameters.
- [x] Calibration-frame-driven noise parameters.

`DualConversionGainConfig` models a low/high conversion-gain readout on normalized contiguous HWC float32 samples. `low_gain_threshold` and `high_gain_threshold` are explicit dimensionless analog-gain thresholds and must satisfy `0 <= low <= high`; the interpolated `IsoNoiseProfile::gain` selects the low branch, high branch, or linear blend. Conversion gains are dimensionless signal ratios, while read-noise fields and `electrons_per_unit` are electrons. The read-noise draw is converted to normalized units by division by `electrons_per_unit`; output is clipped to `clip_min..clip_max`. Profile knots are borrowed host memory for CPU and device memory for CUDA. `dual_conversion_gain_f32` is deterministic for a fixed seed and clips rather than wraps.

`CameraProfileLookupTable` is a borrowed, strictly ISO-sorted LUT of `CameraProfileLookupPoint` records. Its fields are the existing dimensionless gain/noise ratios plus `shot_scale` (electrons per normalized signal unit), `read_noise_stddev`, and `fpn_stddev` (normalized units). `camera_profile_lookup_value` and the named helpers use clamped piecewise-linear interpolation. `camera_profile_lookup_f32` applies the interpolated gain, Poisson shot noise, Gaussian read noise, and FPN noise to normalized HWC float32 data and clips to the explicit interval. CPU LUT memory is host-owned; CUDA LUT memory is device-owned and remains valid through the supplied stream. The CLI form is `camera_profile_lookup IN OUT W H C ISO LUT_F32 COUNT SEED`; each LUT record is ten little-endian float32 values in struct order. `camera_profile_lut` is an accepted alias.

`CalibrationDerivedNoise` is the scalar/map contract for calibration-frame estimates. Scalar `shot_noise_scale` is electrons per normalized unit, scalar `read_noise_stddev` is a normalized Gaussian standard deviation, and scalar `fpn_stddev` is a normalized deterministic-pattern standard deviation. Optional borrowed HWC maps override the corresponding scalar: `read_noise_map` contains standard deviations, `fpn_map` additive offsets, and `gain_map` dimensionless multiplicative gains. `map_count` must equal `width*height*channels` whenever any map is present. `calibration_frame_noise_f32` and aliases apply these values deterministically with seeded Poisson/Gaussian draws and clipping on both CPU and CUDA. CPU maps are host-owned; CUDA maps are device-owned until stream completion. The CLI form is `calibration_noise IN OUT W H C READ_STD FPN_STD SHOT_SCALE SEED [MAP_F32]`; an optional map fixture stores read, FPN, and gain maps consecutively as little-endian float32 values. `calibration_frame_noise` is an accepted alias.

`GainSwitchTransitionConfig` applies the same explicit thresholds to a frame transition. `hysteresis` widens the low/high decision interval by that many gain units on each side, and `transition_width` selects a smoothstep blend width in gain units. `initial_high_gain` is the caller-owned previous-frame state used inside the hysteresis band; no hidden state is retained. `transition_strength` adds a bounded `strength*w*(1-w)` normalized transition artifact before clipping. `gain_switch_transition_f32` and alias `gain_switch_artifacts_f32` expose CPU and CUDA implementations. CLI forms are `dual_conversion_gain IN OUT W H C ISO KNOTS_F32 COUNT LOW_T HIGH_T LOW_GAIN HIGH_GAIN LOW_READ_E HIGH_READ_E ELECTRONS_PER_UNIT SEED` and `gain_switch_transition IN OUT W H C ISO KNOTS_F32 COUNT LOW_T HIGH_T LOW_SIGNAL HIGH_SIGNAL HYSTERESIS TRANSITION_WIDTH STRENGTH INITIAL_HIGH SEED`.

## 3. Bayer and color-filter-array effects

These operations should work on mosaiced sensor data before demosaicing.

- [~] RGGB Bayer CFA sampling.
- [~] BGGR Bayer CFA sampling.
- [~] GRBG Bayer CFA sampling.
- [~] GBRG Bayer CFA sampling.
- [x] Quad-Bayer CFA sampling.
- [x] RGBW CFA sampling.
- [x] Custom CFA masks.
- [x] CFA channel response variation.
- [x] CFA leakage / spectral cross-talk.
- [x] CFA misregistration.
- [x] CFA missing or defective samples.
- [x] Bayer-plane-specific noise.
- [x] Bayer-plane-specific gain.
- [~] Bilinear demosaicing.
- [~] Nearest-neighbor demosaicing.
- [x] Malvar-He-Cutler demosaicing.
- [x] Edge-aware demosaicing.
- [~] Directional demosaicing artifacts.
- [~] False-color zipper artifacts.
- [~] Demosaicing aliasing.
- [~] Demosaicing ringing.
- [~] Demosaicing noise amplification.

### Implemented CFA sampling contract

`quad_bayer_sample_u8` accepts an HWC RGB8 image and a 4x4 Quad-Bayer tile selected by `QuadBayerPattern`; each logical Bayer sample occupies a 2x2 block. `rgbw_sample_u8` accepts HWC RGBW8 data with channel ownership R=0, G=1, B=2, W=3 and the explicit 2x2 `RGBWPattern` tile. Both operations produce one raw 8-bit plane and do not demosaic or retain channel metadata.

`custom_cfa_sample_u8` accepts HWC input, a borrowed repeating `mask_width` x `mask_height` byte mask, and an explicit input channel count. Each mask byte owns the corresponding input plane index; values outside `[0, channels)` are rejected by the CPU API. CPU masks are host-owned, while CUDA masks must remain device-resident until the supplied stream completes. The CLI forms are `quad_bayer IN OUT W H PATTERN`, `rgbw IN OUT W H PATTERN`, and `custom_cfa IN OUT W H CHANNELS MASK_W MASK_H MASK_RAW`.

### Implemented CFA effect contract

The six CFA effects use a contiguous `width*height` float32 raw plane and do not retain pattern metadata. `CfaPlaneMap` is an explicit four-entry permutation indexed by the 2x2 tile positions `[even/even, even/odd, odd/even, odd/odd]`; canonical Bayer maps are returned by `bayer_plane_map`. Plane ownership is always 0=R, 1=Gr, 2=Gb, 3=B. Pointer maps such as `missing_mask` are borrowed host memory for CPU and borrowed device memory for CUDA until the supplied stream completes.

`cfa_channel_response_variation_f32` applies one response factor per plane; deterministic seeded variation is sampled once per plane. `cfa_leakage_f32` uses a row-major 4x4 `leakage[target][source]` matrix and the nearest source sample within a radius-two Manhattan search, with deterministic center fallback. `cfa_misregistration_f32` shifts each target plane by its explicit integer `(dx,dy)` and resolves the shifted sample from that plane. `cfa_missing_samples_f32` treats a nonzero borrowed HxW mask as missing (or uses a seeded probability) and replaces it with zero, fill, or the nearest same-plane sample. `bayer_plane_noise_f32` adds seeded Gaussian noise from four plane standard deviations, while `bayer_plane_gain_f32` applies four plane gains. Every operation clips to its configured range, validates finite parameters, and leaves the input untouched.

The CPU and CUDA APIs share these maps, ordering, clipping, search, and counter-based seed contracts. The raw-float CLI forms are `cfa_channel_response IN OUT W H R GR GB B`, `cfa_leakage IN OUT W H MATRIX16`, `cfa_misregistration IN OUT W H DX4 DY4`, `cfa_missing_samples IN OUT W H PROBABILITY REPLACEMENT FILL SEED`, `bayer_plane_noise IN OUT W H SIGMA4`, and `bayer_plane_gain IN OUT W H GAIN4` (the four values are positional arguments, not files).

### Implemented demosaicing contracts

`bayer_demosaic_malvar_he_cutler_u8` consumes a contiguous HxW uint8 raw plane and produces interleaved RGB8. It uses the published Malvar-He-Cutler 5x5 filters, with coefficients divided by 8. RGGB, BGGR, GRBG, and GBRG are selected by `BayerConfig::pattern`; explicit raw-plane convenience functions are also provided for every pattern and algorithm. Filter coordinates are clamped to the image edge, and results are clipped to [0,255] before round-to-nearest conversion. Known CFA samples are copied exactly.

`bayer_demosaic_edge_aware_u8` uses clamped horizontal and vertical gradients at green sites and clamped diagonal gradients at red/blue sites. The lower-gradient direction is selected, while equal gradients average both directions. A target-color sample is searched at distance one and then two along each direction; this makes the border and tie policy deterministic. CPU and CUDA use the same arithmetic and policy. The CLI forms are `demosaic_malvar IN.raw OUT.raw W H PATTERN` and `demosaic_edge_aware IN.raw OUT.raw W H PATTERN`; aliases `demosaic_malvar_he_cutler` and `demosaic_edgeaware` are accepted.

### Demosaicing artifact injection contracts

The five artifact APIs consume a contiguous HxW uint8 Bayer plane and produce interleaved HxWx3 RGB8. They use edge-aware demosaicing as the zero-strength baseline, select the Bayer tile through `BayerPattern`, retain no state, and clip to the explicit integer `clip_min..clip_max` interval (normally 0..255). CPU pointer arguments are host-owned; CUDA pointers are device-owned until the supplied stream completes. These are deterministic approximation contracts rather than claims of camera-specific ISP parity.

`bayer_directional_demosaicing_artifacts_u8` adds `strength * (selected_direction - other_direction)` at interpolated green sites. `direction` is 0 for gradient-selected, 1 for horizontal, and 2 for vertical interpolation; known CFA samples are preserved. `bayer_false_color_zipper_artifacts_u8` adds alternating red/blue chroma proportional to the horizontal raw gradient, reproducing a deterministic zipper surrogate. `bayer_demosaicing_aliasing_u8` adds an alternating red/blue sinusoid with explicit integer `period` and `phase`; `period` must be positive. `bayer_demosaicing_ringing_u8` adds a signed four-neighbor raw Laplacian multiplied by `strength` to interpolated channels. `bayer_demosaicing_noise_amplification_u8` adds seeded counter-based Gaussian noise with standard deviation `noise_stddev * (1 + amplification * gradient / 510)`; repeated calls with the same seed are exact. Strengths and standard deviations are nonnegative, and all clipping is saturating before half-up uint8 rounding.

CLI forms are `demosaic_directional IN.raw OUT.raw W H PATTERN DIRECTION STRENGTH`, `demosaic_zipper IN.raw OUT.raw W H PATTERN STRENGTH`, `demosaic_aliasing IN.raw OUT.raw W H PATTERN PERIOD PHASE STRENGTH`, `demosaic_ringing IN.raw OUT.raw W H PATTERN STRENGTH`, and `demosaic_noise_amplification IN.raw OUT.raw W H PATTERN NOISE_STDDEV AMPLIFICATION SEED`. The manifest marks these entries partial because the formulas are explicit native approximations, not upstream reference implementations.

## 4. Optical artifacts

`depth_dependent_defocus_u8` consumes a borrowed HxW float32 depth field. It uses
`round(abs(depth-focus_depth)*blur_scale)` as a disk radius, clamps that radius to
`max_radius`, averages the disk with REFLECT_101 borders, and clips to uint8.
This is a deterministic thin-lens approximation; depth units and blur scale are
caller-defined and no occlusion or aperture shape is inferred.

`camera_shake_blur_u8` consumes borrowed sample-count translation arrays in pixel
units. Each listed offset is sampled with nearest-neighbor REFLECT_101 addressing
and all samples receive equal weight. `linear_directional_blur_u8` is the same
explicit equal-weight model over a centered segment defined by length, angle,
and sample count. Both APIs preserve shape and use saturating half-up rounding.

`rotational_motion_blur_u8` samples a centered angular interval around an explicit
pixel center. It uses nearest-neighbor rotated coordinates and REFLECT_101 border
clipping, so it is a deterministic rotational PSF surrogate rather than a
continuous exposure integral.

`rolling_shutter_motion_blur_u8` consumes borrowed H-element row displacement
fields. Every row averages samples from -0.5 through +0.5 of its `(dx,dy)` total
readout displacement. Fields are host-owned for CPU and device-owned until the
CUDA stream completes. The CLI forms are:
`depth_defocus IN OUT W H C DEPTH_F32 FOCUS SCALE MAX_RADIUS`,
`camera_shake IN OUT W H C OFFSETS_X_F32 OFFSETS_Y_F32 SAMPLES`,
`linear_directional IN OUT W H C LENGTH ANGLE_DEGREES SAMPLES`,
`rotational_motion IN OUT W H C CENTER_X CENTER_Y ANGLE_DEGREES SAMPLES`, and
`rolling_shutter IN OUT W H C ROW_DX_F32 ROW_DY_F32 SAMPLES`.

`focus_breathing_u8` models a focus-dependent change in magnification with an
explicit frame parameter. For normalized radius `r`, it uses
`scale = 1 + focus_position * (breathing_strength + radial_strength*r^2)` and
samples the inverse radial map about `(center_x,center_y)`. `focus_position=0`
is an exact copy. The API uses the existing geometric interpolation contract
(Nearest or Linear), applies `fill` outside the source, and saturating half-up
rounding. This deterministic radial/scale field is an affine/geometric lens
approximation; it does not estimate focus from scene depth, model breathing
hysteresis, or simulate a calibrated lens. Nonpositive local scales are clipped
to `fill`. The CLI form is
`focus_breathing IN.raw OUT.raw W H C FOCUS_POSITION BREATHING_STRENGTH RADIAL_STRENGTH CENTER_X CENTER_Y [INTERPOLATION FILL]`.

- [x] Defocus blur.
- [x] Depth-dependent defocus blur.
- [x] Motion blur.
- [x] Camera-shake blur.
- [x] Linear directional blur.
- [x] Rotational motion blur.
- [x] Zoom blur.
- [x] Rolling-shutter motion blur.
- [x] Chromatic aberration.
- [x] Lateral chromatic aberration.
- [x] Longitudinal chromatic aberration.
- [~] Radial lens distortion.
- [~] Tangential lens distortion.
- [x] Thin-prism distortion.
- [x] Lens vignetting.
- [x] Color-dependent vignetting.
- [x] Optical falloff.
- [x] Lens shading.
- [x] Uneven illumination.
- [x] Sensor/lens dust shadows.
- [x] Flare.
- [x] Ghosting.
- [x] Veiling glare.
- [x] Bloom.
- [x] Diffraction blur.
- [x] Bokeh blur.
- [x] Cat-eye bokeh.
- [x] Aperture-shape blur.
- [x] Focus breathing.
- [x] Rolling-shutter geometric distortion.

### Optical catalog blur, chromatic, and rolling-shutter contracts

The eight optical APIs use distinct `Optical*Config` or artifact-specific configuration types even where they reuse established filter, color, geometric, and row-wise sampling kernels. All still-image APIs consume contiguous HWC `uint8` buffers and preserve channels after RGB. CPU pointers and borrowed maps are host-owned; CUDA pointers and maps are device-owned until the supplied stream completes. Blur outputs use `REFLECT_101` clipping and saturating half-up rounding. Geometric and chromatic remaps use constant `fill`, with bilinear interpolation in the CLI. These are deterministic finite-kernel approximations, not calibrated lens simulations or wavelength-resolved optical propagation.

`optical_defocus`, `optical_motion_blur`, and `optical_zoom_blur` expose catalog names independently from existing augmenter filters. Their CLI forms are `optical_defocus IN.raw OUT.raw W H C RADIUS ALIAS_BLUR`, `optical_motion_blur IN.raw OUT.raw W H C KERNEL_SIZE ANGLE DIRECTION`, and `optical_zoom_blur IN.raw OUT.raw W H C MIN_FACTOR MAX_FACTOR STEPS`. Zoom factors include both endpoints and use bilinear centre-preserving samples.

`optical_chromatic_aberration` applies independent normalized radial RGB maps and gains. `lateral_chromatic_aberration` is the gain-free radial-dispersion variant. `longitudinal_chromatic_aberration` applies independent finite Gaussian supports per RGB channel, approximating wavelength-dependent focal spread. Their CLI forms are `optical_chromatic_aberration IN.raw OUT.raw W H C R0 R1 R2 G0 G1 G2 FILL`, `lateral_chromatic_aberration IN.raw OUT.raw W H C R0 R1 R2`, and `longitudinal_chromatic_aberration IN.raw OUT.raw W H C RADIUS_R RADIUS_G RADIUS_B SIGMA`. These models deliberately approximate spectral effects and clip out-of-bounds samples rather than model glass, wavefront, or sensor spectral responses.

`thin_prism_distortion` uses normalized OpenCV-style thin-prism terms `x'=x+s1*r2+s2*r4` and `y'=y+t1*r2+t2*r4`; `fill` is used after clipping. `rolling_shutter_geometric_distortion` borrows H-element row x/y shifts and samples each row independently, making its image contract H-W-C rather than a T-H-W-C video batch. Their CLI forms are `thin_prism_distortion IN.raw OUT.raw W H C S1 S2 T1 T2 FILL` and `rolling_shutter_geometric_distortion IN.raw OUT.raw W H C ROW_X.f32 ROW_Y.f32 FILL`.

### Discrete aperture PSF contracts

`diffraction_blur_u8`, `bokeh_blur_u8`, `cat_eye_bokeh_u8`, and `aperture_shape_blur_u8` consume contiguous HWC uint8 buffers. Each operation evaluates a finite, normalized per-output PSF with `REFLECT_101` borders and saturating half-up uint8 rounding. A zero radius is an exact copy; supported radius is 0..32. The CPU pointer is host-owned and the CUDA pointer is device-owned for the supplied stream.

Diffraction blur uses the deterministic Airy-inspired surrogate `w(r)=sinc(pi*r/(R*S))^2` inside a circular support, where `R=radius` and `S` is the bounded physical-parameter scale; wavelength, aperture diameter, and focal length scale the dimensionless lobe by `clamp((wavelength_nm/550)*(focal_length_mm/50)*(2/aperture_diameter_mm),0.25,4)`, without claiming pixel-calibrated diffraction. Bokeh blur uses a uniform disk and optional linear radial apodization (`edge_softness` in [0,1]). Cat-eye bokeh uses an elliptical disk whose horizontal semi-axis shrinks toward the explicit normalized image boundary according to `cat_eye_strength`; it is not a ray-traced pupil model. Aperture-shape blur uses a regular polygon with 3..32 blades, explicit rotation, and a linear polygon-to-circle `roundness` blend. Discrete sampling and clipping can hide energy outside the finite support and are intentional approximation limits.

The CLI forms are `diffraction_blur IN.raw OUT.raw W H C RADIUS WAVELENGTH_NM APERTURE_DIAMETER_MM FOCAL_LENGTH_MM`, `bokeh_blur IN.raw OUT.raw W H C RADIUS EDGE_SOFTNESS`, `cat_eye_bokeh IN.raw OUT.raw W H C RADIUS CAT_EYE_STRENGTH CENTER_X CENTER_Y`, and `aperture_shape_blur IN.raw OUT.raw W H C RADIUS BLADES ROTATION_DEGREES ROUNDNESS`.

### Radial optical gain contracts

The implemented optical gain APIs operate on contiguous normalized HWC float32 data and clip every output to `clip_min..clip_max` (default `[0,1]`). `lens_vignetting_f32` and `optical_falloff_f32` evaluate `g(r)=c0+c1*r2+c2*r2^2+c3*r2^3`, where `qx=(x/(W-1)-center_x)/radius_x`, `qy=(y/(H-1)-center_y)/radius_y`, and `r2=qx^2+qy^2`; defaults are identity. A borrowed optional map multiplies this gain and is either HxW (`map_channels=1`, broadcast) or HxWxC (`map_channels=C`, channel-specific). `color_dependent_vignetting_f32` accepts a borrowed channel-major `C*4` polynomial table in place of the shared coefficients, and otherwise follows the same radial/map contract. These are deterministic polynomial approximations, not a physical pupil or spectral model.

`lens_shading_f32` applies `input * gain * map`, while `uneven_illumination_f32` applies `input * (base_gain * map) + offset`. Both reuse the scalar HxW or HWC gain-map layout; a null map is identity. `sensor_lens_dust_shadows_f32` consumes a borrowed HxW opacity map, clamps each opacity to `[0,1]`, and applies `input * (1-strength*opacity)`. A null dust map is identity. CPU map pointers are host-owned; CUDA pointers are device-owned and must remain valid through the supplied stream. Inputs and outputs may not alias unless the caller has verified the backend's in-place behavior.

The CLI forms are `lens_vignetting IN.f32 OUT.f32 W H C C1 C2 C3 [MAP.f32]`, `color_dependent_vignetting ...`, and `optical_falloff ...` (the three radial coefficients are `c1,c2,c3` with `c0=1`); `lens_shading IN.f32 OUT.f32 W H C GAIN [MAP.f32]`; `uneven_illumination IN.f32 OUT.f32 W H C BASE_GAIN OFFSET [MAP.f32]`; and `sensor_lens_dust_shadows IN.f32 OUT.f32 W H C STRENGTH [OPACITY_MAP.f32]`. Operation aliases are `color_vignetting` and `dust_shadows`. Maps are little-endian float32 fixtures. Clipping is intentional and can hide over-correction at saturated values.

### Flare, ghosting, veiling glare, and bloom

These four APIs use normalized HWC float32 buffers and explicit borrowed records or maps. `flare_f32` takes `FlareSource` discs and `FlareHalo` records in image-pixel coordinates. A halo with `source_index` set follows that source; otherwise its own center is used. Source discs have hard coverage, halos have linear falloff, and both use deterministic white weather-style alpha blending. `ghosting_f32` takes borrowed `Ghost` records `(dx,dy,scale,alpha)`; displacement is destination minus source and nearest clamped sampling overlays shifted input copies in record order. These records are host-owned for CPU and device-owned for CUDA until stream completion.

`veiling_glare_f32` is a local source-luminance approximation: each pixel adds a white amount `strength*max(channel input)*map`. It is deliberately not a physical scattering point-spread function. `bloom_f32` averages threshold excess in a clipped square neighbourhood of integer `radius`; it is a deterministic box-kernel surrogate, not a Gaussian or diffraction model. All maps are borrowed HxW (`map_channels=1`) or HxWxC (`map_channels=C`) float32 gains in the execution memory space; CUDA maps must outlive the stream operation. Missing maps are identity. Alpha, gain, source records, and map values are validated; outputs clip to `clip_min..clip_max` (default `[0,1]`).

CLI record files are native packed `FlareSource`/`FlareHalo`/`Ghost` arrays: `flare IN OUT W H C OPACITY SOURCES.bin HALOS.bin [MAP.f32]`, `ghosting IN OUT W H C OPACITY GHOSTS.bin [MAP.f32]`, `veiling_glare IN OUT W H C STRENGTH [MAP.f32]`, and `bloom IN OUT W H C THRESHOLD STRENGTH RADIUS [MAP.f32]`. These approximations reuse the weather blending rule but are not intended as calibrated lens simulation.

## 5. ISP and derivative-based artifacts

These are not physical sensor noise. They model image-signal-processor behavior and should be exposed under `isp`.

- [~] Gradient-dependent Gaussian noise.
- [~] Laplacian-dependent Gaussian noise.
- [x] Gradient-plus-Laplacian noise.
- [~] Edge oversharpening.
- [~] Unsharp-mask halos.
- [~] Laplacian halos.
- [~] Ringing near strong edges.
- [x] Deblocking halos.
- [x] Demosaicing edge artifacts.
- [~] Local contrast enhancement artifacts.
- [~] Haloing from tone mapping.
- [~] Local sharpening noise amplification.
- [x] Edge-dependent quantization.
- [x] Edge-dependent compression error.
- [x] Gradient reversal.
- [x] Clipped-edge ringing.
- [~] High-frequency attenuation.
- [~] Detail smearing.
- [~] Overshoot and undershoot.
- [x] Laplacian residual injection.
- [x] Sobel residual injection.
- [x] High-pass residual injection.

### Implemented ISP artifact contract

The thirteen checked artifacts expose distinct CPU and CUDA APIs in `isp_artifacts.hpp` and consume contiguous interleaved HWC uint8 data. All borders clamp to the nearest pixel, arithmetic is float, and output is saturating half-up rounded to `[0,255]`. These formulas are deterministic native approximations; they do not claim parity with a camera vendor ISP, and clipping can hide large residuals.

`edge_oversharpening_u8` adds `amount*(8p-sum8)` from a signed 3x3 high-pass. `unsharp_mask_halos_u8` adds `amount*(p-Gaussian(p))` with integer `radius` and `sigma`. `laplacian_halos_u8` adds `amount*(4p-left-right-up-down)`. `ringing_near_strong_edges_u8` gates a checkerboard-sign four-neighbour Laplacian on `abs(right-left)+abs(down-up) >= gradient_threshold` and clips the residual to `+/-residual_bound` before applying `amount`.

`local_contrast_enhancement_artifacts_u8` adds `amount*(p-mean)*sigma/(sigma+epsilon)` over a square window. `haloing_from_tone_mapping_u8` is a global Gaussian high-pass surrogate scaled by `amount*tone_strength`. `local_sharpening_noise_amplification_u8` applies a radius-window Gaussian high-pass scaled by `amount*noise_gain`. `high_frequency_attenuation_u8` blends the input with a deterministic box low-pass by `amount` in `[0,1]`. `detail_smearing_u8` blends with a box or Gaussian blur (`gaussian` is an explicit boolean). `overshoot_and_undershoot_u8` applies a signed Gaussian high-boost residual and independently bounds positive and negative residuals by `max_overshoot` and `max_undershoot` before final clipping.

The remaining checked ISP APIs are deterministic native approximations. `gradient_plus_laplacian_noise_u8` scales seeded Gaussian noise by the normalized magnitude of the centred right-minus-left/down-minus-up gradient and the absolute four-neighbour Laplacian. `deblocking_halos_u8` attenuates discontinuities within an explicit block boundary radius. `demosaicing_edge_artifacts_u8` injects alternating red/blue zipper chroma on strong green-channel edges. `edge_dependent_quantization_u8` increases the scalar quantization step with local gradient. `edge_dependent_compression_error_u8` uses block-anchor residual plus seeded edge-weighted error as a lightweight JPEG-block surrogate; it does not invoke an external codec. `gradient_reversal_u8` subtracts the signed local gradient above threshold. `clipped_edge_ringing_u8` applies bounded checkerboard Laplacian ringing only at near-black or near-white edge samples. All use saturating uint8 clipping, so clipped residuals are intentional.

The CLI forms are `gradient_plus_laplacian_noise IN OUT W H C SIGMA GRADIENT_SCALE LAPLACIAN_SCALE SEED`, `deblocking_halos IN OUT W H C BLOCK_SIZE RADIUS AMOUNT`, `demosaicing_edge_artifacts IN OUT W H C STRENGTH GRADIENT_THRESHOLD`, `edge_dependent_quantization IN OUT W H C STEP EDGE_SCALE`, `edge_dependent_compression_error IN OUT W H C BLOCK_SIZE STRENGTH EDGE_SCALE SEED`, `gradient_reversal IN OUT W H C AMOUNT GRADIENT_THRESHOLD`, and `clipped_edge_ringing IN OUT W H C AMOUNT GRADIENT_THRESHOLD RESIDUAL_BOUND`.

`per_channel_gain_noise_u8` applies independent coordinate-keyed Gaussian multiplicative gain perturbations to RGB, while `color_temperature_error_u8` interpolates Tanner-Helland blackbody gains from D65 by `strength`. Both copy channels after RGB and clip by uint8 saturation. `row_noise_phase_changes_f32` applies one correlated row offset to every pixel in a row and frame, with a seeded phase sign change probability. Its input/output is T-H-W-C float32 and it uses the temporal clipping contract. The color CLI forms are `per_channel_gain_noise IN OUT W H C GAIN_R GAIN_G GAIN_B STDDEV SEED` and `color_temperature_error IN OUT W H C TEMPERATURE_KELVIN STRENGTH`; the temporal form is `row_noise_phase_changes IN.f32 OUT.f32 T H W C ROW_STDDEV PHASE_PROBABILITY RHO SEED`.

`laplacian_residual_injection_u8` adds `amount * residual` where the default residual is the signed four-neighbour Laplacian. `sobel_residual_injection_u8` uses the clamped 3x3 Sobel X or Y derivative selected by `direction` (0 or 1). `high_pass_residual_injection_u8` uses the signed eight-neighbour 3x3 high-pass. Each API accepts an optional borrowed signed float32 `residual_map`: HxW (`map_channels=1`) broadcasts across channels, while HxWxC (`map_channels=channels`) selects a channel-specific map; a null map derives the named residual. CPU maps are host-owned and CUDA maps are device-owned until the supplied stream completes. Inputs and outputs remain caller-owned. Every result is saturating half-up clipped to `[0,255]`, so clipping can hide an over-large residual.

The CLI forms are `edge_oversharpening IN OUT W H C AMOUNT`, `unsharp_mask_halos IN OUT W H C RADIUS SIGMA AMOUNT`, `laplacian_halos IN OUT W H C AMOUNT`, `ringing_near_strong_edges IN OUT W H C AMOUNT GRADIENT_THRESHOLD RESIDUAL_BOUND`, `local_contrast_enhancement_artifacts IN OUT W H C RADIUS AMOUNT EPSILON`, `haloing_from_tone_mapping IN OUT W H C RADIUS SIGMA AMOUNT TONE_STRENGTH`, `local_sharpening_noise_amplification IN OUT W H C RADIUS AMOUNT NOISE_GAIN`, `high_frequency_attenuation IN OUT W H C RADIUS AMOUNT`, `detail_smearing IN OUT W H C RADIUS SIGMA AMOUNT GAUSSIAN(0|1)`, and `overshoot_and_undershoot IN OUT W H C RADIUS SIGMA AMOUNT MAX_OVERSHOOT MAX_UNDERSHOOT`. Residual injection forms are `laplacian_residual_injection IN OUT W H C AMOUNT [MAP.f32]`, `sobel_residual_injection IN OUT W H C AMOUNT DIRECTION [MAP.f32]`, and `high_pass_residual_injection IN OUT W H C AMOUNT [MAP.f32]`; supplied maps are little-endian float32 fixtures and the CLI uses HxWxC maps.

Required primitives:

- [~] Sobel X derivative.
- [~] Sobel Y derivative.
- [~] Scharr X derivative.
- [~] Scharr Y derivative.
- [~] Laplacian derivative.
- [~] Gradient magnitude.
- [~] Local variance map.
- [~] Local entropy map.
- [~] Configurable derivative border policy.

## 6. Color and photometric pipeline effects

- [~] White-balance gain error.
- [x] Per-channel gain noise.
- [x] Color-temperature error.
- [x] Color-matrix perturbation.
- [x] Camera color-profile variation.
- [x] RGB channel cross-talk.
- [x] Sensor spectral-response variation.
- [x] Gamma variation.
- [x] Tone-curve variation.
- [x] S-curve contrast variation.
- [x] Highlight roll-off variation.
- [x] Shadow lift.
- [x] Shadow crush.
- [x] Local tone-mapping noise.
- [x] Chroma noise.
- [x] Luma noise.
- [x] Correlated luma-chroma noise.
- [x] Chroma subsampling artifacts.
- [x] Color clipping.
- [x] White-balance clipping.
- [x] Posterization.
- [x] Banding from low-bit-depth tone mapping.

### Color and photometric variation contract

The color/photometric additions use contiguous interleaved HWC uint8 RGB(A) data and explicit deterministic parameters. RGB is treated as full-range normalized linear-light surrogate data (no transfer-function or ICC conversion); channels >= 3 are copied. Matrices are row-major target-row/source-column maps. Color-matrix perturbation applies matrix*RGB plus normalized offset. Camera-profile variation applies that matrix, per-channel gains, and seeded Gaussian noise. RGB cross-talk is matrix-only. Spectral-response variation uses a seeded per-coefficient Gaussian perturbation before matrix application. Results use half-up rounding and saturating clipping.

Color clipping uses independent normalized RGB bounds. White-balance clipping applies nonnegative RGB gains before a common normalized clip interval. Chroma and luma noise use BT.601 full-range `Y=0.299R+0.587G+0.114B`, `Cb=B-Y`, `Cr=R-Y`; chroma noise perturbs Cb/Cr and luma noise perturbs Y. Correlated luma-chroma noise shares one standard-normal luma variate with each chroma variate at the configured correlation. Noise standard deviations are normalized signal units and seeds are coordinate-keyed SplitMix64 states. CPU pointers are host-owned and CUDA pointers are device-owned until stream completion. These are deterministic pipeline surrogates, not spectral or color-management models.

The CLI forms are `color_matrix_perturbation IN OUT W H C M00 M01 M02 M10 M11 M12 M20 M21 M22 [O0 O1 O2]`, `rgb_channel_cross_talk IN OUT W H C M00 M01 M02 M10 M11 M12 M20 M21 M22`, `camera_color_profile_variation IN OUT W H C GAIN_R GAIN_G GAIN_B NOISE_STDDEV`, `sensor_spectral_response_variation IN OUT W H C RESPONSE_STDDEV SEED`, `color_clipping IN OUT W H C MIN MAX`, `white_balance_clipping IN OUT W H C GAIN_R GAIN_G GAIN_B MIN MAX`, `chroma_noise IN OUT W H C STDDEV SEED`, `luma_noise IN OUT W H C STDDEV SEED`, and `correlated_luma_chroma_noise IN OUT W H C LUMA_STDDEV CHROMA_STDDEV CORRELATION SEED`. 

`chroma_subsampling_artifacts_u8` converts RGB to the documented BT.601 full-range Y/Cb/Cr surrogate, averages Cb/Cr over horizontal 2-pixel blocks for 4:2:2 or clipped 2x2 blocks for 4:2:0, replicates each average, and reconstructs RGB while retaining Y. 4:4:4 is an exact copy. The explicit enum is `Y444`, `Y422`, or `Y420`; CPU image pointers are host-owned and CUDA pointers are device-owned. The CLI is `chroma_subsampling_artifacts IN.raw OUT.raw W H C 444|422|420`. This is a deterministic chroma-plane artifact model, not JPEG DCT or quantization.

`local_tone_mapping_noise_u8` computes a clamped square-neighborhood arithmetic mean of BT.601 Y with radius in `[0,32]`, then applies `Y' = Y + tone_strength*(mean-Y) + noise_stddev*N(0,1)`. `tone_strength` is in `[0,1]`, noise standard deviation is normalized Y units, and the SplitMix64 state is keyed by pixel index plus `seed`; Cb/Cr are preserved. Reconstructed RGB is clipped and half-up rounded; channels after RGB are copied. CPU buffers are host-owned and CUDA buffers are device-owned. The CLI is `local_tone_mapping_noise IN.raw OUT.raw W H C RADIUS TONE_STRENGTH NOISE_STDDEV SEED`.

The eight checked operations use contiguous interleaved HWC uint8 data and explicit deterministic parameters. `gamma_variation_u8` applies `y=x^gamma`; `tone_curve_variation_u8` applies a borrowed channel-major `channels*256` LUT. S-curve contrast uses `y=x+amount*4*x*(1-x)*(2*x-1)`, with amount in [-1,1]. Highlight roll-off leaves values below threshold unchanged and maps the remainder with `threshold+d/(1+strength*d/(1-threshold))`. Shadow lift and crush apply a linear shadow-window weight to lift toward white or attenuate toward black. All normalized values are clipped and half-up rounded to uint8; clipping can hide out-of-range curve values. CPU pointers are host-owned and CUDA pointers are device-owned until stream completion. These formulas are deterministic tone surrogates, not camera-profile calibration.

Posterization and low-bit-depth banding reuse floor quantization to `2^bits` levels (`bits` in [1,8]) without dither. Their outputs are therefore intentionally stepped and can exhibit visible bands. The CLI forms are `tone gamma_variation IN OUT W H C GAMMA`, `tone tone_curve_variation IN OUT W H C LUT.raw`, `tone s_curve_contrast_variation IN OUT W H C AMOUNT`, `tone highlight_rolloff_variation IN OUT W H C THRESHOLD STRENGTH`, `tone shadow_lift IN OUT W H C AMOUNT THRESHOLD`, `tone shadow_crush IN OUT W H C AMOUNT THRESHOLD`, `tone posterization IN OUT W H C BITS`, and `tone low_bit_depth_banding IN OUT W H C BITS`.

## 7. Video and temporal artifacts

- [x] Temporal Gaussian noise.
- [x] Temporally correlated shot noise.
- [x] Temporal read-noise correlation.
- [x] Flicker.
- [x] Exposure flicker.
- [x] White-balance flicker.
- [x] Gain flicker.
- [x] Row-noise phase changes.
- [x] Fixed-pattern noise drift.

### Implemented temporal batch contract

The temporal APIs operate on one borrowed contiguous float32 batch with layout
`T×H×W×C` and index `(((t*H+y)*W+x)*C+channel)`. They write a separate output
batch and retain no state. CPU pointers are host-owned. CUDA input, output, and
optional `base_map` pointers are device-owned and remain valid until the supplied
stream completes. `t=0` is the first frame; all temporal indexing is zero-based.

`temporal_gaussian_noise_f32`, `temporal_read_noise_correlation_f32`, and
`temporally_correlated_shot_noise_f32` use a counter-based SplitMix-style seed
key. Their temporal state is the explicit AR(1) recurrence
`q_t=rho*q_(t-1)+sqrt(1-rho^2)*N(seed,t,y,x,c)`, with `q_-1=0` and
`rho` in `[-1,1]`. A fixed seed and configuration is therefore bitwise
repeatable within each backend and CPU/CUDA use the same coordinate contract.
Shot noise is a centered signal-dependent normal approximation with
`photons_per_unit` variance scaling, which keeps the CUDA path bounded and
preserves the requested temporal correlation.

`flicker_f32`, `exposure_flicker_f32`, and `gain_flicker_f32` apply one
frame-global multiplicative factor. `white_balance_flicker_f32` applies an
independent channel factor (RGB channels 0,1,2; additional channels use blue's
factor). `fixed_pattern_noise_drift_f32` adds one HWC base pattern followed by
AR(1) drift; a supplied `base_map` is borrowed, while a null map is generated
from `base_stddev` and the seed. Every output is saturating-clipped to the
configured interval. CLI forms use float32 batch files and the same dimensions,
seed, and correlation arguments as the C++ APIs.

### Implemented video artifact batch contract

The eight video operations use the same borrowed contiguous `T×H×W×C`
float32 layout and explicit `clip_min..clip_max` interval. CPU buffers and
fixture arrays are host-owned; CUDA buffers and optional masks, indices, and
transform arrays are device-owned until the supplied stream completes. No
operation retains a pointer or hidden temporal state. A separate output batch
is required. Persistence masks are static HWC bytes, drop masks are T bytes,
source-index arrays are T signed integers, blend weights are T float32 values,
and transform arrays are T (or T×H for rolling shutter) native records.
Explicit arrays override seeded generation and are therefore suitable for
replay.

`dead_pixel_persistence_f32` and `hot_pixel_persistence_f32` apply one static
HWC mask to every frame, or generate a deterministic mask from `(seed,y,x,c)`;
selected samples are replaced by the configured value. `frame_drops_f32`
replaces a selected frame with the preceding input frame (or `fill_value` for
the first frame). `duplicate_frames_f32` copies the explicit source index for
each output frame, or deterministically repeats the preceding frame. Both
operations clamp source indices and never wrap around.

`frame_blending_f32` mixes each frame with an explicit source frame using a
scalar or T-entry weight. `temporal_ghosting_f32` overlays a translated source
frame with a per-frame `(source_frame_offset,dx,dy,alpha)` transform.
`motion_compensation_errors_f32` uses the analogous transform and weight to
blend a wrongly compensated source frame into the current frame. Coordinates
are nearest-neighbour and clamp at image edges. `inter_frame_compression_noise_f32`
quantizes each previous-frame residual with a scalar or T-entry
`InterFrameCompressionRecord`, then applies deterministic seeded residual
perturbation. `gop_keyframe_artifacts_f32` uses explicit `GopFrameRecord`
records, a T-entry keyframe mask, or `gop_size`; keyframes receive an intra
quantization surrogate and other frames use the residual surrogate.
`block_motion_estimation_artifacts_f32` uses explicit T-by-block
`MotionVectorRecord` records, or a seeded integer vector field, and blends the
clamped previous-frame sample selected by each block. These are codec-
independent surrogates: they do not parse or emit AV1, H.264, or H.265.
`video_sensor_rolling_shutter_f32`
uses one explicit row transform per frame and row; without rows it generates a
deterministic linear readout displacement from `motion_dx`, `motion_dy`, and
`readout_fraction`. All values are saturating-clipped. CLI forms are
`dead_pixel_persistence IN OUT T H W C VALUE PROB SEED [MASK.raw]`,
`hot_pixel_persistence ...`, `frame_drops IN OUT T H W C FILL PROB SEED [MASK.raw]`,
`duplicate_frames IN OUT T H W C PROB SEED [INDICES.i32]`,
`frame_blending IN OUT T H W C WEIGHT [WEIGHTS.f32]`,
`temporal_ghosting IN OUT T H W C OFFSET DX DY ALPHA [TRANSFORMS.bin]`,
`motion_compensation_errors IN OUT T H W C OFFSET DX DY WEIGHT [TRANSFORMS.bin]`,
`inter_frame_compression_noise IN OUT T H W C STEP STRENGTH SEED`,
`gop_keyframe_artifacts IN OUT T H W C GOP KEYFRAME_STRENGTH INTERFRAME_STRENGTH SEED`,
`block_motion_estimation_artifacts IN OUT T H W C BLOCK_W BLOCK_H STRENGTH SEED`,
and `video_sensor_rolling_shutter IN OUT T H W C DX DY READOUT [ROWS.bin]`.
- [x] Dead-pixel persistence.
- [x] Hot-pixel persistence.
- [x] Frame drops.
- [x] Duplicate frames.
- [x] Frame blending.
- [x] Temporal ghosting.
- [x] Motion-compensation errors.
- [x] Video sensor rolling shutter.
- [~] Inter-frame compression noise (codec-independent surrogate).
- [~] GOP/keyframe artifacts (codec-independent surrogate).
- [~] Block motion-estimation artifacts (codec-independent surrogate).

## 8. Compression and transport artifacts

The JPEG entries use libjpeg/libjpeg-turbo for still-image encode/decode. CPU pointers are host-owned. CUDA pointers are device-owned and use a synchronous host-codec fallback with stream-ordered transfers; no GPU-native JPEG codec is claimed. Quality is `[1,100]`, quantization multipliers are positive, and sampling is `Y444`, `Y422`, or `Y420`. Ringing, blocking, and mosquito noise are deterministic clipped post-decode artifact models. Restart-marker damage emits restart markers and deterministically corrupts entropy bytes at selected restart boundaries; progressive decoding encodes a progressive JPEG and models an early scan with a clipped low-frequency preview. WebP uses libwebp when its headers and native library are found, with the same documented CUDA host-codec fallback; builds without libwebp expose a clear runtime/CLI fallback error. AV1, H.264/H.265, and packet/video codecs remain unsupported.

The codec-independent transport entries below operate on raw bytes, unpacked planes, or interleaved HWC frames. They do not parse or generate AV1, H.264, or H.265 bitstreams. `transport.hpp` defines host-owned CPU and device-owned CUDA pointers; all borrowed masks remain valid through the call/stream. Byte operations preserve shape and use explicit byte counts. Frame operations use a final partial block at image edges. Chroma misalignment is a per-plane integer shift with clamped edge sampling. Raw Bayer transport corruption works on unpacked uint16 samples and clips to the configured bit depth. CUDA provides one kernel per operation; these kernels model transport damage, not codec reconstruction.

- [~] JPEG compression.
- [~] JPEG quality variation.
- [~] JPEG quantization-table variation.
- [~] JPEG chroma subsampling 4:4:4.
- [~] JPEG chroma subsampling 4:2:2.
- [~] JPEG chroma subsampling 4:2:0.
- [~] JPEG ringing.
- [~] JPEG blocking.
- [~] JPEG mosquito noise.
- [x] JPEG restart-marker damage.
- [x] JPEG progressive decoding artifacts.
- [~] WebP lossy compression (conditional on libwebp).
- [~] WebP lossless compression (conditional on libwebp).
- [~] AV1 intra-frame artifacts (conditional FFmpeg single-frame encode/decode; does not modify arbitrary AV1 bitstreams).
- [~] H.264 compression (conditional FFmpeg single-frame intra encode/decode).
- [~] H.265/HEVC compression (conditional FFmpeg single-frame intra encode/decode).
- [~] AV1 video compression (single-frame packet only; temporal encoding remains unsupported).
- [x] Block quantization.
- [x] Deblocking-filter mismatch.
- [x] Chroma-plane misalignment.
- [x] Packet-loss macroblocks.
- [x] Truncated-frame corruption.
- [x] Bit flips.
- [x] Raw Bayer transport corruption.

### Codec blockers (explicitly unsupported)

The four codec entries have conditional native APIs in `video_codec.hpp`, backed by libavcodec/libavutil/libswscale when those development libraries are detected. They encode and decode library-produced single intra frames; AV1 intra artifacts mean this encode/decode round trip, not mutation or exact parsing of arbitrary AV1 streams. They do not implement multi-frame temporal prediction, reference-picture behavior, container parsing, or parity with a specific external codec configuration, so manifest status remains conservative (`partial`). Without the development dependencies a clear unavailable implementation is built. CUDA calls use synchronous device-to-host copies, a host codec call, and host-to-device copies; this is not GPU-native encoding. Codec-independent transport operations remain distinct and are not counted as codec support.

## 9. Environmental and acquisition artifacts

The environmental APIs use explicit, deterministic approximations. `environmental.hpp`
provides float32 HWC veil, gain, dirty-lens, and EMI operations plus T-H-W-C
sensor-temperature, power-band, fluorescent-flicker, and LED rolling-band
operations. Dirty-lens and EMI maps are borrowed HxW when `map_channels==1`
or HxWxC when it equals the image channel count; CPU maps are host-owned and
CUDA maps are device-owned until stream completion. Dirty-lens blur is a
square-box PSF blended by map opacity. EMI is a sinusoidal additive field in
normalized units. Temperature drift is a linear frame temperature ramp whose
dark-current offset uses `D*exposure*exp(coefficient*(T-reference))/electrons`
with an optional HWC sensitivity map. Power banding and LED artifacts are
sinusoidal row/time multiplicative fields; fluorescent flicker is a frame-global
sinusoidal multiplier. These are deterministic acquisition surrogates, not
physical RF, lens-particle, ballast, PWM, or exposure-integration simulations.
All results clip to configured ranges. The CLI forms are
`dirty_lens_blur IN OUT W H C RADIUS STRENGTH [MAP.f32]`,
`electromagnetic_interference IN OUT W H C AMPLITUDE FREQ_X FREQ_Y PHASE [MAP.f32]`,
`sensor_temperature_drift IN OUT T H W C START_C END_C REF_C DARK_ELECTRONS_PER_SECOND ELECTRONS EXPOSURE COEFF_PER_C`,
`power_supply_banding IN OUT T H W C AMPLITUDE FREQUENCY_HZ FRAME_RATE_HZ ROW_CYCLES PHASE`,
`fluorescent_light_flicker IN OUT T H W C AMPLITUDE FREQUENCY_HZ FRAME_RATE_HZ PHASE`, and
`led_rolling_band_artifacts IN OUT T H W C AMPLITUDE FREQUENCY_HZ FRAME_RATE_HZ ROW_CYCLES PHASE`.

`water_droplets_on_lens_f32` consumes normalized float32 HWC data. Explicit
`WaterDroplet` records are processed in order; a null record list generates
records from `seed` and the radius range, and `make_water_droplets` materializes
that list for replay. Each covered pixel is blended toward a local box blur by
`opacity`; this is a deterministic lens-occlusion surrogate, not a refractive
fluid simulation. The CLI form is `water_droplets_on_lens IN OUT W H C COUNT
OPACITY RADIUS_MIN RADIUS_MAX BLUR_RADIUS SEED`.

The existing weather records provide the uint8 variants: `FogVeilConfig` aliases
`RandomFogConfig`, `RainStreaksConfig` aliases `RandomRainConfig`,
`SnowOcclusionConfig` aliases `RandomSnowConfig`, and `DustParticlesConfig`
aliases `RandomGravelConfig`. Explicit field/streak/flake/particle arrays are
borrowed in the execution memory space; null arrays use the documented
SplitMix64 seed generation. Streaks use pixel-centre capsule coverage, masks
use row-major HxW ownership, and overlaps are sequential white alpha blends.
Headlight flare and reflections reuse the explicit `FlareConfig` and
`GhostingConfig` records. The CLI float forms are
`atmospheric_haze|smoke_veil|window_glare|backlight_washout IN OUT W H C STRENGTH [MASK.f32]`
and `low_light_amplification|underexposure|overexposure IN OUT W H C GAIN`.

- [x] Fog veil.
- [x] Atmospheric haze.
- [x] Rain streaks.
- [~] Water droplets on lens (deterministic local-blur surrogate).
- [x] Snow occlusion.
- [x] Dust particles.
- [x] Smoke veil.
- [x] Low-light amplification.
- [x] Underexposure.
- [x] Overexposure.
- [x] Headlight flare.
- [x] Backlight washout.
- [x] Reflections.
- [x] Window glare.
- [x] Dirty-lens blur.
- [x] Sensor temperature drift.
- [x] Electromagnetic interference.
- [x] Power-supply banding.
- [x] Fluorescent-light flicker.
- [x] LED rolling-band artifacts.

## 10. API and implementation tasks

The native contract is implemented in `include/augmatch/sensor/native_api.hpp` and is
included by `augmatch/augmatch.hpp`. `ImageView` and `MutableImageView` accept
strided HWC or CHW `uint8`, `uint16`, and `float32` data. `RawBayerView` adds a
strided, single-plane representation with an explicit CFA pattern. Metadata is
borrowed through `NoiseMetadataView`; channel records can map to independent
plane records. `CameraProfileConfig` owns serializable LUT points while its
`view()` exposes the existing device-compatible lookup type.

`CounterRng` is coordinate-keyed and stateless. The CPU reference is
`native_api_cpu.cpp`; the CUDA translation unit reuses the stream-ordered
`noise_view.cu` kernel. Device operations are asynchronous on the caller's
stream. `Workspace` makes temporary storage an explicit caller-owned resource
(the current fused operation needs zero bytes). `sensor::fused_pipeline` combines
profile/analog gain metadata with additive sensor noise. `NoiseClipPolicy`,
`NoiseQuantizationPolicy`, and `NoiseBorderPolicy` are explicit policies.
`additive_noise_batch` provides ordered batch execution and
`additive_noise_async` is deliberately host-only: it captures non-owning views,
so callers retain storage until the future completes. Every operation returns a
`Status`/`StatusCode` rather than throwing.

Profile text uses the versioned `AUGMATCH_CAMERA_PROFILE_V1` format. CMake
installs headers, the static target, and relocatable `augmatchConfig.cmake`
export files. `examples/native_api_cpp17.cpp` and
`examples/native_api_cpp23.cpp` compile as separate targets. The corresponding
unit test is `tests/native_api_unit.cpp`; CUDA builds retain the same API and
stream contract.

- [x] Add sensor/, optics/, isp/, compression/, and video/ transform namespaces.
- [x] Add float and integer image-view support.
- [x] Add raw Bayer image representation.
- [x] Add per-channel and per-plane metadata.
- [x] Add camera-profile configuration objects.
- [x] Add deterministic counter-based RNG.
- [x] Add CPU reference implementations.
- [x] Add CUDA kernels.
- [x] Add CUDA stream support.
- [x] Add scratch/workspace allocation.
- [x] Add fused sensor-pipeline kernels.
- [x] Add clipping and quantization policies.
- [x] Add configurable border policies.
- [x] Add batch execution.
- [~] Add asynchronous execution APIs (host-only).
- [x] Add explicit error/status results.
- [x] Add serialization for camera profiles.
- [x] Add CMake install/export targets.
- [x] Add C++17 examples.
- [x] Add C++23 examples.

## 11. Validation tasks

Validation rows 591--610 are driven by `tests/noise_catalog_validation.cpp`,
`tests/noise_catalog_cuda_validation.cu`, and
`tests/noise_catalog_benchmarks.py`. The CPU suite uses only native APIs and
checks measured samples; it does not embed fabricated measurements. CUDA tests
return CTest's skip code 77 when a compiler or runtime device is unavailable.
The benchmark script records measured transform and copy times for all three
requested resolutions. Camera-domain regression is runtime-gated: fixtures are
enumerated from the repository `data/` directory and no profile is synthesized
when that directory has no camera-profile records. See `NOISE_VALIDATION.md`
for tolerances, ownership, and blockers.

- [~] Verify Poisson variance approximately equals the mean.
- [~] Verify read-noise variance is independent of signal.
- [x] Verify PRNU scales with signal.
- [~] Verify row noise is constant across each row.
- [~] Verify column noise is constant across each column.
- [~] Verify hot/dead pixel probabilities.
- [x] Verify ADC quantization levels.
- [x] Verify ADC non-linearity vectors, seeded determinism, statistics, and bounds.
- [~] Verify clipping and black-level behavior.
- [~] Verify Bayer sampling and reconstruction.
- [~] Verify derivative maps against analytical images.
- [~] Verify CPU/CUDA statistical equivalence.
- [x] Verify fixed-seed determinism.
- [x] Verify different seeds produce independent samples.
- [~] Add golden vectors for every deterministic transform.
- [~] Add distributional tests for stochastic transforms.
- [x] Add property tests for bounds and dtype preservation.
- [~] Add performance benchmarks for 512x512, 1080p, and 4K images.
- [~] Add memory-transfer benchmarks.
- [~] Add profile-based camera-domain regression datasets.

## 12. Documentation tasks

- [x] Document physical sensor noise versus ISP artifact noise.
- [x] Document parameter units and calibration procedures.
- [x] Document normalized versus electron-domain inputs.
- [x] Document RNG and reproducibility guarantees.
- [x] Document CPU/CUDA numerical tolerances.
- [x] Document unsupported external-codec dependencies.
- [x] Add examples for raw Bayer pipelines.
- [x] Add examples for RGB camera pipelines.
- [x] Add examples for video pipelines.
- [x] Add a transform support matrix.
- [x] Add references to sensor-noise and camera-pipeline literature.

### Physical meaning and stage boundary

Sensor operations model signal formation before demosaicing: photon shot noise, read/reset noise, dark current, PRNU, fixed-pattern offsets, hot/dead sites, and ADC effects. Their parameters describe sensor quantities or explicitly normalized approximations. ISP operations model downstream processing or visible defects: demosaicing zipper/ringing, sharpening halos, denoising residue, quantization, and compression/transport damage. An ISP surrogate is not a physical sensor-noise model and should not be used to estimate sensor parameters. The catalog operation's name and input domain determine its scope; combining stages is the caller's responsibility.

### Units, calibration, and signal domains

Unless a specific API says otherwise, dimensions and strides are pixels/elements, image values are in the declared dtype's numerical units, and standard deviations use the same units as the signal being perturbed. Float APIs documented as normalized take values in `[0,1]`; a standard deviation of `0.01` then means one percent of full scale. Integer image-domain noise uses code-value units (for example uint8 codes in `[0,255]`), not volts or electrons. ADC DNL/INL amplitudes expressed in LSB use one output code as one LSB. Temperatures are degrees Celsius when an API explicitly takes Celsius; exposure time is seconds and gain ratios are dimensionless where those physical parameters are exposed. Consult the individual API contract before converting.

For physical calibration, acquire dark frames over exposure and temperature to estimate bias, read variance, dark-current slope, and fixed spatial patterns. Acquire uniform flat fields at multiple signal levels: subtract bias, use temporal variance against mean signal to estimate conversion gain (electrons/ADU), then estimate PRNU from the normalized spatial mean. Estimate row/column covariance from repeated flats and darks. Fit saturation/full-well and ADC response only from unclipped exposures and measured code histograms. Record black level, white level, bit depth, temperature, exposure, analog/digital gain, and CFA metadata with every calibration set. This procedure is a guide, not an automatic calibration routine; seeded synthetic maps are not measured camera profiles.

Electron-domain data is not interchangeable with normalized float data. For an electron count `e-`, a simple photoelectron model draws `Poisson(e-)`; if conversion gain is `g` electrons/ADU, convert a mean electron signal to ADU by dividing by `g`, and scale read-noise sigma consistently before adding it. In normalized APIs, scale values by the declared white/full-scale code and convert all additive parameters by that same scale. Do not pass electron counts to a normalized `[0,1]` API or interpret code-value sigma as electrons. Black-level subtraction, clipping, and quantization order must be explicit because they change distributions.

### RNG, reproducibility, and backend tolerances

Seeded routines use deterministic seed-derived draws; the native view contract's `CounterRng` additionally keys samples by logical coordinates, channel, plane, and stream so traversal order does not affect a pixel's draw. Repeatability is guaranteed for the same API, seed, parameters, input, and backend/toolchain contract. It is not a promise that every historical API or arbitrary CPU and GPU math library produces bitwise-identical floating-point samples. CPU/CUDA implementations that use transcendental operations can differ by floating-point rounding and distribution-transform details. Compare integer outputs exactly only where tests specify exact parity; otherwise compare unclipped floating outputs with an absolute/relative tolerance appropriate to scale and separately test distribution mean, variance, bounds, and spatial correlation. The unit/property/statistics and `*_cuda_validation.cu` tests are the concrete per-operation contract. CUDA compilation does not certify runtime support: this checkout's documented runtime/toolchain may reject PTX at launch, and CUDA validation can therefore be unavailable even when compilation succeeds. The CPU implementation remains the reference path; CPU APIs reject device-owned views rather than silently copying them.

### Pipeline examples

A runnable C++17 RGB sensor-noise smoke example is `examples/noise_camera_workflows.cpp`; it uses an HWC uint8 RGB view, a fixed seed, and verifies identical repeated outputs. Configure/build the `noise_camera_workflows` target and run CTest's `noise_camera_workflows_smoke` test.

A raw Bayer workflow starts with unpacked, single-plane samples and explicit CFA pattern/bit depth. Apply dark/offset and shot/read effects in the raw domain, clip/quantize according to sensor ADC policy, then demosaic using the matching CFA pattern; apply RGB/ISP effects only after demosaicing. `RawBayerView` carries the raw plane metadata. The current Bayer examples and implementations are deterministic native operations, not a vendor ISP emulator, and the documented uint8 Bayer artifact functions are not a replacement for a calibrated high-bit-depth raw pipeline. Do not treat packed RAW formats as unpacked uint16 planes.

For video, represent contiguous temporal data as T-H-W-C and apply time-dependent sensor drift/flicker or frame-correlated artifacts before encoding, preserving frame order and one deterministic seed/config per sequence. `temporal_noise_unit` and `video_temporal_unit` exercise CPU APIs; associated CUDA validation requires a usable CUDA runtime. Existing codec-independent temporal/transport effects are explicit surrogates, not bitstream decoding. AV1, H.264, and H.265/HEVC codec artifacts remain blocked on external parsers/decoders, reference pictures, and codec-specific reconstruction. Enabling a documented transport primitive does not imply codec support.

### Support matrix

| Domain | Supported surface and evidence | Limits |
|---|---|---|
| Sensor / signal | Shot/read, dark, PRNU/FPN, row/column, hot/dead, ADC and signal transforms; CPU and CUDA sources listed per row in `IMPLEMENTATION_MANIFEST.tsv`. | Per-operation dtype, layout, units, and availability vary; use each API contract. |
| Raw Bayer / CFA | Explicit CFA metadata, uint8 Bayer sampling/demosaicing and artifact surrogates; see Bayer rows and `tests/demosaic_artifacts_unit.cpp`. | No claim of complete calibrated high-bit-depth or vendor-specific RAW pipeline. |
| RGB / ISP | HWC signal and ISP artifact APIs, float32 normalized maps, integer image operations; see `isp_artifacts.hpp`, `signal.hpp`, and manifest. | Deterministic formulas are native surrogates, not vendor-specific pipeline replicas. |
| Temporal / video | T-H-W-C temporal noise and codec-independent transport artifacts; see temporal/video unit and property tests. | Full AV1/H.264/H.265 encode/decode artifacts unsupported; CUDA runtime depends on installed compiler, driver, and device compatibility. |
| External still codecs | JPEG and conditional WebP support as marked in the manifest. | WebP availability depends on libwebp; CUDA may use synchronous host-codec fallback; no video codec dependency is bundled. |

The manifest reports operation-level `implemented`, `partial`, or `todo` status and evidence paths. Documentation status does not upgrade a partial operation to complete, nor does a compile-only CUDA result certify CUDA runtime behavior.

### References

1. J. R. Janesick, *Photon Transfer: DN → λ*, SPIE Press, 2007. (Photon-transfer calibration, conversion gain, read noise, and full well.)
2. E. R. Fossum, “CMOS Image Sensors: Electronic Camera-on-a-Chip,” *IEEE Transactions on Electron Devices*, 44(10), 1997, pp. 1689–1698. doi:10.1109/16.628824. (CMOS pixel and sensor noise foundations.)
3. G. C. Holst, *CMOS/CCD Sensors and Camera Systems*, 2nd ed., JCD Publishing, 2007. (Sensor response and camera measurement.)
4. R. Ramanath, W. E. Snyder, Y. Yoo, and M. S. Drew, “Color Image Processing Pipeline,” *IEEE Signal Processing Magazine*, 22(1), 2005, pp. 34–43. doi:10.1109/MSP.2005.1407713. (Color-camera pipeline stages.)
5. B. E. Bayer, “Color imaging array,” U.S. Patent 3,971,065, 1976. (Color-filter-array sampling.)
