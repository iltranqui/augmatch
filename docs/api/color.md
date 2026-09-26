# Color, tone, and arithmetic

Headers: `include/augmatch/color/`. Back to the [docs index](../README.md).

## Contents

- [Elementwise arithmetic contract](#elementwise-arithmetic-contract)
- [ReplaceElementwise and ImpulseNoise contract](#replaceelementwise-and-impulsenoise-contract)
- [Salt, Pepper, and coarse impulse contracts](#salt-pepper-and-coarse-impulse-contracts)
- [RandomToneCurve contract](#randomtonecurve-contract)
- [SigmoidContrast and LogContrast contract](#sigmoidcontrast-and-logcontrast-contract)
- [Color and photometric tone variations](#color-and-photometric-tone-variations)
- [Chroma subsampling and local tone-mapping noise](#chroma-subsampling-and-local-tone-mapping-noise)
- [ColorJitter contract](#colorjitter-contract)
- [Deterministic imgaug colorspace meta contracts](#deterministic-imgaug-colorspace-meta-contracts)
- [Deterministic imgaug HSV color batch contract](#deterministic-imgaug-hsv-color-batch-contract)
- [RandomColorJitter contract](#randomcolorjitter-contract)
- [PlanckianJitter contract](#planckianjitter-contract)
- [ChangeColorTemperature contract](#changecolortemperature-contract)
- [Uniform color quantization contract](#uniform-color-quantization-contract)
- [FancyPCA contract](#fancypca-contract)
- [PlasmaContrast contract](#plasmacontrast-contract)
- [PlasmaBrightnessContrast contract](#plasmabrightnesscontrast-contract)
- [CLAHE contract](#clahe-contract)
- [All-channel contrast aliases](#all-channel-contrast-aliases)
- [Dithering contract](#dithering-contract)

## Elementwise arithmetic contract

`add_elementwise_u8` and `multiply_elementwise_u8` apply one parameter to every
interleaved HWC `uint8` element. `AddElementwiseConfig::values` and
`MultiplyElementwiseConfig::values` are borrowed arrays with exactly
`width*height*channels` float entries in row-major HWC order. CPU callers own
host arrays; CUDA callers own device arrays and must keep them alive until the
supplied stream has completed. A null array selects deterministic generation
from `seed`, with each value in the half-open `[min_value,max_value)` range;
`MultiplyElementwise` additionally requires nonnegative factors. Additive
results and products are rounded to nearest and clipped to `[0,255]`. The
`make_*_elementwise_values` helpers materialize the same host-side sequence for
logging or exact replay. The CLI accepts either an explicit float32 file or a
seeded range:

```sh
augmatch_cli arithmetic add_elementwise IN.raw OUT.raw W H C VALUES.f32
augmatch_cli arithmetic multiply_elementwise IN.raw OUT.raw W H C MIN MAX SEED
augmatch_cli arithmetic add_elementwise IN.raw OUT.raw W H C MIN MAX SEED VALUES.f32
```

## ReplaceElementwise and ImpulseNoise contract

`replace_elementwise_u8` and `impulse_noise_u8` operate on contiguous HWC
`uint8` data. For exact replay, `mask` and `values` are borrowed one-byte
arrays with one entry per pixel channel; a nonzero mask selects the matching
value and the caller owns both arrays. CPU arrays are host memory, while CUDA
arrays are device memory and must remain valid until the supplied stream has
completed. In seeded mode, a null mask uses deterministic Bernoulli sampling
from `probability` and `seed`, without global RNG state. ReplaceElementwise
uses its scalar `replacement_value`; ImpulseNoise uses `salt_probability` to
select 255 or 0. Explicit values override that scalar/impulse choice. The
`make_*_mask` helpers materialize the same deterministic mask for logging or
exact replay. The CLI forms are:

```sh
augmatch_cli arithmetic replace_elementwise IN.raw OUT.raw W H C MASK.raw VALUES.raw
augmatch_cli arithmetic replace_elementwise IN.raw OUT.raw W H C PROBABILITY REPLACEMENT SEED
augmatch_cli arithmetic impulse_noise IN.raw OUT.raw W H C MASK.raw VALUES.raw
augmatch_cli arithmetic impulse_noise IN.raw OUT.raw W H C PROBABILITY SALT_PROBABILITY SEED
```

These are deterministic native contracts rather than claims that imgaug's
implicit parameter distributions or global random state are reproduced.

| imgaug transform | C++ API / behavior |
| --- | --- |

## Salt, Pepper, and coarse impulse contracts

`Salt`, `Pepper`, `SaltAndPepper`, and their `Coarse*` variants operate on
contiguous HWC `uint8` pixels. The shared `SaltPepperConfig` supports either a
borrowed one-byte-per-pixel mask or an ordered list of half-open integer
rectangles `[x0,x1) x [y0,y1)`; explicit selections ignore probability and seed.
For Salt/Pepper every nonzero mask byte selects 255/0. For SaltAndPepper mask
byte 1 selects salt and byte 2 selects pepper. Rectangle selections use the
fixed `salt_probability >= 0.5` choice, so explicit calls are exact and do not
consume random state. CPU arrays are host-owned; CUDA arrays are device-owned
and remain valid until the stream completes.

Without explicit selections, `probability` is a Bernoulli rate sampled from a
SplitMix64-compatible counter keyed by `seed`. Fine variants sample one draw
per pixel. Coarse variants sample one draw per `block_width` by `block_height`
block and broadcast the result to all covered channels. `SaltAndPepper` uses a
second deterministic draw and `salt_probability` for the salt/pepper choice.
These are deterministic native semantics; they do not claim to reproduce
imgaug's implicit parameter distributions or global RNG state. CLI forms are:

```sh
augmatch_cli arithmetic salt IN.raw OUT.raw W H C PROBABILITY SEED
augmatch_cli arithmetic salt IN.raw OUT.raw W H C MASK.raw
augmatch_cli arithmetic coarse_salt IN.raw OUT.raw W H C PROBABILITY SEED BLOCK_W BLOCK_H
augmatch_cli arithmetic coarse_salt IN.raw OUT.raw W H C RECTANGLES
augmatch_cli arithmetic salt_and_pepper IN.raw OUT.raw W H C PROBABILITY SALT_PROBABILITY SEED
```

Rectangle text is comma-separated `x0:y0:x1:y1` records. The API names are
`salt_u8`, `pepper_u8`, `salt_and_pepper_u8`, `coarse_salt_u8`,
`coarse_pepper_u8`, and `coarse_salt_and_pepper_u8`.

| `AveragePooling` | `pool_u8(..., PoolMode::Average, ...)` |
| `MaxPooling` | `pool_u8(..., PoolMode::Maximum, ...)` |
| `MinPooling` | `pool_u8(..., PoolMode::Minimum, ...)` |
| `MedianPooling` | `pool_u8(..., PoolMode::Median, ...)` |
| `WithChannels` | `with_channels(..., std::vector<int>{...})`; native CPU/CUDA selective channel views |
| `ReplaceElementwise` | `replace_elementwise_u8`; explicit borrowed mask/value bytes or seeded Bernoulli scalar replacement |
| `ImpulseNoise` | `impulse_noise_u8`; explicit borrowed mask/value bytes or seeded salt/pepper impulses |

Pooling currently accepts HWC `uint8` images whose width and height are divisible by the kernel dimensions. With `keep_size=true`, the reduced result is expanded by nearest-neighbor sampling, matching imgaug's keep-size path for the divisible test cases. The Python suite compares all four operations with `imgaug==0.4.0`; install the test requirements before configuring CMake to enable it.

The catalog's imgaug-only transform set has native source, CUDA or documented host-only scope, and test evidence in `IMPLEMENTATION_MANIFEST.tsv`. Some entries intentionally use explicit deterministic contracts instead of imgaug's implicit parameter distributions or global RNG, and remain `partial` where exact upstream parity or a dependency is unavailable. The pinned imgaug pooling reference is currently blocked by its import failure with NumPy 2.x; this is recorded as a blocker rather than a pass.

## RandomToneCurve contract

`random_tone_curve_u8` applies a borrowed, explicit channel-major LUT. The table contains exactly `channels * 256` bytes, with entry `lut[channel * 256 + input]` selecting the output byte for that channel and input value. The CPU API expects host LUT storage; the CUDA API expects device LUT storage. A null LUT selects the deterministic built-in curve generated from `seed`, with fixed monotone control points at inputs 0, 64, 128, 192, and 255. For zero-based channel `c` and interior control point `j`, the control value is `max(previous, min(255, 64*j + (SplitMix64(seed + 4*c + j) mod 65) - 32))`; each input is integer linearly interpolated between adjacent controls, with half values rounded up. Supplying a LUT ignores `seed`, so this path is seed-free and exact across repeated calls. The CLI form is `tone random_tone_curve IN.raw OUT.raw W H C LUT.raw [SEED]`; use `-` instead of `LUT.raw` to use the generated curve and provide an optional seed.

## SigmoidContrast and LogContrast contract

`sigmoid_contrast_u8` and `log_contrast_u8` implement deterministic imgaug-style
contrast transfer functions for every channel of contiguous HWC `uint8` data.
SigmoidContrast first normalizes `x=input/255` and computes
`y=1/(1+exp(gain*(cutoff-x)))`; `cutoff` is finite in `[0,1]` and `gain` is
finite and nonnegative. LogContrast computes
`y=log(1+x*(base^gain-1))/log(base)`; `base` is finite and greater than one
and `gain` is finite and nonnegative. `base=2` is the imgaug transfer. Both
outputs compute `clip(floor(255*y+0.5),0,255)`, so rounding is half-up rather
than the host's locale-dependent integer conversion. CPU parameters and image
buffers are host-owned; CUDA image buffers are device-owned and the operation
is enqueued on the supplied stream. Existing `ToneConfig` and pixel
brightness/contrast APIs are unchanged. The explicit parameters replace
imgaug's random parameter sampling. The CLI forms are:

```sh
augmatch_cli tone sigmoid_contrast IN.raw OUT.raw W H C CUTOFF GAIN
augmatch_cli tone log_contrast IN.raw OUT.raw W H C GAIN BASE
```

The exact parity test uses the same float32 transfer order before uint8
quantization. Values outside the normalized transfer's natural range are
clipped; extreme gains can therefore saturate large portions of an image.

## Color and photometric tone variations

The tone variation APIs operate on contiguous interleaved HWC `uint8` data with explicit deterministic configuration. `gamma_variation_u8` evaluates `x^gamma`; `tone_curve_variation_u8` consumes a borrowed channel-major LUT with exactly `channels*256` entries. `s_curve_contrast_variation_u8` uses the bounded cubic S-curve `x+amount*4*x*(1-x)*(2*x-1)`. `highlight_rolloff_variation_u8` uses a thresholded rational shoulder, while `shadow_lift_u8` and `shadow_crush_u8` apply a threshold-window lift or attenuation. Normalized results are half-up rounded and clipped to `[0,255]`; this clipping can conceal an extreme curve. CPU pointers are host-owned and CUDA pointers are device-owned until stream completion. These are deterministic approximations, not camera-profile calibration.

`posterization_u8` and `low_bit_depth_banding_u8` use the existing floor quantizer at `2^bits` levels, with `bits` in `[1,8]` and no dither. Thus visible steps are intentional. CLI forms are `tone gamma_variation IN.raw OUT.raw W H C GAMMA`, `tone tone_curve_variation IN.raw OUT.raw W H C LUT.raw`, `tone s_curve_contrast_variation IN.raw OUT.raw W H C AMOUNT`, `tone highlight_rolloff_variation IN.raw OUT.raw W H C THRESHOLD STRENGTH`, `tone shadow_lift IN.raw OUT.raw W H C AMOUNT THRESHOLD`, `tone shadow_crush IN.raw OUT.raw W H C AMOUNT THRESHOLD`, `tone posterization IN.raw OUT.raw W H C BITS`, and `tone low_bit_depth_banding IN.raw OUT.raw W H C BITS`.

## Chroma subsampling and local tone-mapping noise

`chroma_subsampling_artifacts_u8` uses the existing full-range BT.601 surrogate (`Y=.299R+.587G+.114B`, `Cb=B-Y`, `Cr=R-Y`), not studio-range offsets, transfer-function conversion, or ICC color management. `Y444` copies exactly; `Y422` averages each horizontal chroma pair; `Y420` averages each clipped 2x2 chroma block and replicates it. Luma is retained, RGB is reconstructed with saturating half-up rounding, and alpha/channels after RGB are copied. The enum is shared explicitly as `ChromaSubsampling::{Y444,Y422,Y420}`. CPU data is host-owned and CUDA data is device-owned. Use `chroma_subsampling_artifacts IN.raw OUT.raw W H C 444|422|420`.

`local_tone_mapping_noise_u8` averages BT.601 Y in a clamped square neighborhood and applies `Y' = Y + tone_strength*(mean-Y) + noise_stddev*N(0,1)`. Radius is `[0,32]`, tone strength is `[0,1]`, noise is normalized Y units, and the coordinate-keyed seed is deterministic across CPU/CUDA. Cb/Cr and channels after RGB are preserved. Use `local_tone_mapping_noise IN.raw OUT.raw W H C RADIUS TONE_STRENGTH NOISE_STDDEV SEED`.

## ColorJitter contract

`color_jitter_u8` is a deterministic combined color operation for HWC RGB/RGBA
`uint8` data. It applies the fixed pipeline `clip(round(contrast * input +
brightness * 255))`, RGB-to-OpenCV-HSV conversion, saturation multiplication,
hue addition modulo 180, and HSV-to-RGB conversion. `ColorJitterConfig::brightness`
is an additive normalized value, `contrast` and `saturation` are nonnegative
multipliers, and `hue` is an OpenCV HSV half-degree-bin offset. No random
sampling or operation reordering is performed; channels after RGB are copied
unchanged. CPU and CUDA use native per-pixel kernels; comparison with OpenCV can
differ by up to five uint8 levels at HSV rounding boundaries. Random ranges are
not part of `ColorJitterConfig`: use `RandomColorJitterConfig` (ranges plus seed)
to represent them. The API is intentionally limited to contiguous HWC `uint8`
buffers and does not implement torchvision's arbitrary CHW/float views. The CLI
form is `color color_jitter IN.raw OUT.raw W H C BRIGHTNESS CONTRAST SATURATION HUE`.

## Deterministic imgaug colorspace meta contracts

`with_colorspace_u8` and `with_brightness_channels_u8` are explicit deterministic
contracts for the imgaug `WithColorspace` and `WithBrightnessChannels` meta
augmenters. Both accept contiguous interleaved HWC `uint8` RGB/RGBA buffers;
CPU pointers are host-owned and CUDA pointers are device-owned, and CUDA work is
enqueued on the supplied stream. `ColorSpace::RGB` uses sRGB component bins,
`ColorSpace::HSV` uses OpenCV bins (H in `[0,180)`, S/V in `[0,255]`), and
`ColorSpace::LAB` uses uint8 CIELAB encoding (L mapped from 0..100 and a/b
biased by 128). `WithColorspace` applies explicit `round_half_up(x * multiplier
+ addend)` values to all three channels in the selected space before converting
back to RGB; RGB channels are emitted and channels after RGB (including alpha)
are copied. LAB conversion is a deterministic D65 sRGB approximation, not ICC
color management or a spectral model. `WithBrightnessChannels` applies one
explicit affine pair to selected RGB channel indices, or to HSV V (index 2);
HSV H/S selections are rejected to keep brightness semantics unambiguous.
Negative multipliers and non-finite parameters are rejected. CLI forms are
`color with_colorspace IN.raw OUT.raw W H C SPACE M0 M1 M2 A0 A1 A2` and
`color with_brightness_channels IN.raw OUT.raw W H C SPACE CH0 CH1 CH2 MULT ADD`.

## Deterministic imgaug HSV color batch contract

The HSV batch implements `MultiplyAndAddToBrightness`, `MultiplyBrightness`,
`AddToBrightness`, `WithHueAndSaturation`, `MultiplyHueAndSaturation`,
`MultiplyHue`, `MultiplySaturation`, `RemoveSaturation`,
`AddToHueAndSaturation`, `AddToHue`, and `AddToSaturation` through the
corresponding `*_u8` APIs. Each operation accepts an explicit parameter config;
no imgaug distribution or global RNG is used. `MultiplyAndAddToBrightness`
computes `clip(round(value * multiplier + addend))`. The CLI forms are
`color OP IN.raw OUT.raw W H C PARAM...`, where combined operations, including
multiply-and-add brightness, take two parameters and `remove_saturation` takes
none.

These functions use OpenCV-style uint8 HSV: hue has 180 half-degree bins
`[0,180)`, while saturation and value have 256 bins `[0,255]`. Hue parameters
are rounded half-up to integer bins and wrapped modulo 180, including negative
additions. Saturation and value results are rounded half-up and clipped. RGB
conversion uses the same half-up rounding; alpha and channels after RGB are
copied unchanged. `MultiplyHue` multiplies hue bins before wrapping, whereas
`AddToHue` adds bins. Brightness means the HSV value channel, and saturation
operations act on HSV saturation. CPU and CUDA use the same per-pixel contract.

## RandomColorJitter contract

`random_color_jitter_u8` generates one explicit `ColorJitterConfig` per image and
then composes the native `color_jitter_u8` kernel. `RandomColorJitterConfig`
contains bounded ranges for brightness, contrast, saturation, and hue plus a
64-bit seed; it is the complete random state and no global RNG is consulted. For
parameter index 0, 1, 2, or 3 respectively, the sampled unit is
`(SplitMix64(seed + index) >> 40) / 2^24`, mapped linearly to that parameter's
range. CPU and CUDA therefore receive identical parameters and preserve all
channels after RGB. The CLI form is
`color random_color_jitter IN.raw OUT.raw W H C BRIGHTNESS_MIN BRIGHTNESS_MAX CONTRAST_MIN CONTRAST_MAX SATURATION_MIN SATURATION_MAX HUE_MIN HUE_MAX SEED`.
Degenerate ranges are a fixed ColorJitter configuration.

## PlanckianJitter contract

`planckian_jitter_u8` is a deterministic RGB channel-gain approximation for
HWC `uint8` images. For temperature `T` in `[1000,40000]` K, it evaluates the
Tanner-Helland blackbody RGB approximation: below 6600 K, red is fixed at 255
and green/blue use logarithms; above 6600 K, red/green use power laws and blue
is fixed at 255. Each channel is clipped to `[0,255]` and divided by its value
at the fixed 6500 K reference, then every input channel is emitted as
`floor(input * gain + 0.5)` and clipped to uint8. Channels after RGB are copied
unchanged. This is a color-temperature look, not a spectral blackbody,
chromatic-adaptation, camera-white-balance, or ICC-color-management model; it
cannot reproduce those effects or upstream random temperature sampling. CPU
and CUDA use the same per-pixel gain contract, and the operation has no random
state. The CLI form is `color planckian_jitter IN.raw OUT.raw W H C TEMPERATURE_K`;
`color planckian` is accepted as an alias.

## ChangeColorTemperature contract

`change_color_temperature_u8` is the deterministic imgaug-style color-temperature
operation. It uses the Tanner-Helland RGB approximation and the same normalized
6500 K reference gains as `planckian_jitter_u8`, with `temperature_kelvin` in
`[1000,40000]` K. RGB output is `clip(floor(input * gain + 0.5), 0, 255)` and
channels after RGB are copied unchanged. The CPU and CUDA APIs have identical
validation and rounding behavior; `color change_temperature` and the top-level
`change_color_temperature` CLI spelling are accepted aliases. This approximation
is a color-temperature look, not spectral blackbody rendering, chromatic
adaptation, camera white balance, ICC color management, or imgaug's random
parameter sampling. The CLI form is `color change_color_temperature IN.raw
OUT.raw W H C TEMPERATURE_K`.

## Uniform color quantization contract

`uniform_color_quantization_u8` quantizes each uint8 component into an explicit
number of equal-width levels. For `N` levels, `q=256/N` and the output is
`clip(floor(floor(v/q)*q + q/2 + 0.5), 0, 255)`, i.e. half-up rounding after
mapping to the bin center. `levels` must be in `[2,256]`; 256 is identity.
`uniform_color_quantization_to_n_bits_u8` uses `N=2^bits` with the lower bin
boundary rather than the center, exactly clearing the `8-bits` least-significant
bits for uint8 input. `bits` must be in `[1,8]`. A four-channel HWC image keeps
channel 3 (alpha) unchanged; other channels are quantized. CPU pointers are
host-owned and CUDA pointers are device-owned, and neither API synchronizes a
CUDA stream. The CLI forms are:

```text
augmatch_cli color uniform_color_quantization IN.raw OUT.raw W H C LEVELS
augmatch_cli color uniform_color_quantization_to_n_bits IN.raw OUT.raw W H C BITS
```

`quantize_uniform` and `quantize_uniform_to_n_bits` are CLI aliases. These
operations do not alter the existing arithmetic `posterize` or tone APIs.

## FancyPCA contract

`fancy_pca_u8` applies a deterministic PCA-style RGB offset to contiguous HWC
`uint8` data. `FancyPCAConfig::basis` is a row-major 3x3 orthonormal matrix
whose columns are the supplied color eigenvectors. For channel `c`, the offset is
`delta[c] = sum_k basis[c*3+k] * eigenvalues[k] * perturbation[k] * alpha`.
Eigenvalues and perturbations are in uint8 intensity units; results are rounded
with `floor(value + 0.5)` and clipped to `[0,255]`. `alpha` is an optional fixed
scalar (default `1`), and the explicit perturbation vector replaces the stochastic
PCA samples used by augmentation libraries. Channels after RGB are copied
unchanged. The CPU and CUDA APIs accept the same embedded basis and parameters,
with no random state. The CLI form is `color fancy_pca IN.raw OUT.raw W H C
B00 B01 B02 B10 B11 B12 B20 B21 B22 E0 E1 E2 P0 P1 P2 [ALPHA]`; the top-level
`fancy_pca` spelling is also accepted.

## PlasmaContrast contract

`plasma_contrast_u8` applies a deterministic spatial contrast field to contiguous HWC `uint8` data. For each pixel, the local factor is `contrast * (1 + plasma_strength * (2 * field - 1))`, where `contrast >= 0`, `plasma_strength` is in `[0,1]`, and `field` is in `[0,1]`. Values are scaled around 127.5, rounded with `floor(value + 0.5)`, and clipped to uint8. The explicit field contains exactly `width*height` floats in row-major order and is the preferred CPU/host or CUDA/device input when an external reference must be reproduced. With a null field, CPU and CUDA derive the same four-scale integer-hash field from `seed`. This is a deterministic plasma/noise approximation, not a claim of parity with any implementation-specific stochastic plasma sampler. Channels are modulated independently by the same spatial field. The CLI form is `color plasma_contrast IN.raw OUT.raw W H C CONTRAST STRENGTH SEED [FIELD.f32]`.

## PlasmaBrightnessContrast contract

`plasma_brightness_contrast_u8` reuses the deterministic PlasmaContrast field and
local factor, then applies an additive brightness offset. For each pixel,
`factor = contrast * (1 + plasma_strength * (2 * field - 1))` and each channel is
computed as `round(127.5 + (input - 127.5) * factor + brightness * 255)`, clipped
to `[0,255]`. `contrast` is nonnegative, `plasma_strength` is in `[0,1]`, and
`brightness` is a finite normalized offset. The explicit field contains exactly
`width*height` finite floats in `[0,1]` in row-major order; a null field uses the
same four-scale hash field generated from `seed` on CPU and CUDA. The CLI form is
`color plasma_brightness_contrast IN.raw OUT.raw W H C BRIGHTNESS CONTRAST STRENGTH SEED [FIELD.f32]`.
A zero brightness call is exactly the native PlasmaContrast operation.

## CLAHE contract

`clahe_u8` implements 8-bit contrast-limited adaptive histogram equalization with
an explicit `tiles_x` by `tiles_y` grid and positive finite `clip_limit`. If either dimension is not divisible by its tile count, OpenCV's border path
extends both the right edge by `tiles_x - width % tiles_x` and the bottom edge by
`tiles_y - height % tiles_y` (including a full grid-width extension for a
zero remainder). Otherwise no extension is made. The resulting equal tile width
and height are used, and only the original image rectangle is emitted; extension
pixels use OpenCV `REFLECT_101`.
Each supplied HWC channel is processed independently, including channels after
RGB (no implicit colorspace conversion or alpha dropping). Histogram bins are
clipped using `floor(clip_limit * tile_area / 256)`, with a minimum of one, and
excess counts are redistributed in ascending bins using OpenCV's deterministic
batch and residual-step rule. Per-tile LUTs and bilinear tile interpolation use
nearest-integer `lrint` rounding (ties-to-even) and clip to `[0,255]`. CPU pointers
refer to host memory. CUDA pointers refer to device memory; the CUDA adapter
uses the same algorithm and synchronizes its transfer. Existing `equalize_u8`
and other tone APIs are unchanged. The CLI form is
`tone clahe IN.raw OUT.raw W H C CLIP_LIMIT TILES_X TILES_Y` (the top-level
`clahe` spelling is also accepted). The parity test compares each channel with
`cv2.createCLAHE`.

## All-channel contrast aliases

`all_channels_clahe_u8` and `all_channels_histogram_equalization_u8` are distinct
CPU/CUDA APIs with explicit alias configs. They process every channel in the
interleaved HWC input independently, including channels after RGB such as alpha;
there is no implicit colorspace conversion or channel dropping. The CLAHE alias
uses `AllChannelsCLAHEConfig` and reuses the native `clahe_u8` kernel. The
histogram alias uses `AllChannelsHistogramEqualizationConfig` and reuses the
native `equalize_u8` kernel. Both APIs are deterministic and have no host/device
fallback. Their CLI forms are `tone all_channels_clahe IN.raw OUT.raw W H C
CLIP_LIMIT TILES_X TILES_Y` and `tone all_channels_histogram_equalization IN.raw
OUT.raw W H C`. The dedicated parity test compares all channels with OpenCV.

## Dithering contract

`dithering_u8` deterministically reduces every HWC `uint8` channel to the requested
`DitheringConfig::bit_depth` in `[1,8]`. Output levels are evenly spaced,
`round(level * 255 / (2^bit_depth-1))`; bit depth 8 is an identity.
`DitheringMode::OrderedBayer4x4` uses the fixed matrix
`[[0,8,2,10],[12,4,14,6],[3,11,1,9],[15,7,13,5]]` and adds
`(matrix[y mod 4][x mod 4] + 0.5) / 16` before flooring the scaled level.
`DitheringMode::ErrorDiffusionFloydSteinberg` scans left-to-right, top-to-bottom
and sends error to right, next-left, next, and next-right pixels with weights
`7/16, 3/16, 5/16, 1/16`; channels are independent. Both modes are seed-free
and deterministic. CPU pointers refer to host memory and CUDA pointers to device
memory, with the same output contract. The CLI form is
`color dithering IN.raw OUT.raw W H C BIT_DEPTH MODE`, where `MODE` is `ordered`
(or `bayer`) or `error_diffusion` (or `floyd_steinberg`); top-level `dithering`
is also accepted. This operation is separate from existing `bit_reduce_u8`,
`adc_quantize_u8`, and tone APIs, which retain their prior semantics.
