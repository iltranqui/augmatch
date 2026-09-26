# Filters, blur, edges, and blending

Headers: `include/augmatch/filter/`. Back to the [docs index](../README.md).

## Contents

- [Segmentation Voronoi contract](#segmentation-voronoi-contract)
- [Blend augmentation contract](#blend-augmentation-contract)
- [Convolutional filters contract](#convolutional-filters-contract)
- [Canny edge contract](#canny-edge-contract)
- [AdvancedBlur contract](#advancedblur-contract)
- [BilateralBlur contract](#bilateralblur-contract)
- [MeanShiftBlur contract](#meanshiftblur-contract)
- [GlassBlur contract](#glassblur-contract)
- [RingingOvershoot contract](#ringingovershoot-contract)
- [ZoomBlur contract](#zoomblur-contract)
- [NonLocalMeansDenoising contract](#nonlocalmeansdenoising-contract)

## Segmentation Voronoi contract

`include/augmatch/filter/voronoi.hpp` provides four deterministic CPU/CUDA Voronoi
partition APIs over contiguous HxW `uint8` target masks. `Voronoi` uses a seed
and point count (or explicit points), `UniformVoronoi` uses seeded stratified
jitter, `RegularGridVoronoi` uses explicit grid counts, and
`RelativeRegularGridVoronoi` uses fractional cell widths. Every target-map
operation copies the winning site's original discrete label; it does not
interpolate or invent target values. The `*_labels_u32` variants return site
indices. Ties choose the lowest site index. These are deterministic nearest-site
approximations, not a replacement for a continuous Voronoi geometry library.
The CLI accepts `voronoi`, `uniform_voronoi`, `regular_grid_voronoi`, and
`relative_regular_grid_voronoi` raw HxW masks.

## Blend augmentation contract

`include/augmatch/filter/blend.hpp` implements all `augmenters.blend` entries with paired
CPU/CUDA APIs. Source and overlay are contiguous HWC `uint8` buffers; masks and
segmentation maps are borrowed HxW `uint8` metadata, and boxes/class IDs/color
records remain caller-owned. Alpha is clamped and half-up rounded. Seeded
masks and noise are deterministic and do not use global RNG state. Noise fields
are documented portable approximations. See `BLEND_CATALOG.md` for ownership,
CLI, and approximation details.

## Convolutional filters contract

`emboss_u8`, `edge_detect_u8`, and `directed_edge_detect_u8` implement the
imgaug 0.4.0 3x3 convolutional filters for contiguous HWC `uint8` images.
The deterministic configs expose `alpha` explicitly; Emboss also exposes
`strength`, and DirectedEdgeDetect exposes `direction` in normalized turns
(`direction * 360` degrees clockwise from the top, truncated to integer
 degrees). Emboss uses
`[[-1-s,-s,0],[-s,1,s],[0,s,1+s]]`; EdgeDetect uses
`[[0,1,0],[1,-4,1],[0,1,0]]`. DirectedEdgeDetect constructs imgaug's
angle-similarity stencil from the normalized direction.

All channels use OpenCV `filter2D` semantics: the border is
`BORDER_REFLECT_101`, and output is saturated to `[0,255]` with OpenCV-compatible
half-up nearest-integer conversion (rather than truncation). Alpha blends the filtered response with the
source before that conversion. CPU pointers refer to host memory and CUDA
pointers refer to device memory. CLI forms are:

```sh
augmatch_cli emboss IN.raw OUT.raw WIDTH HEIGHT CHANNELS ALPHA STRENGTH
augmatch_cli edge_detect IN.raw OUT.raw WIDTH HEIGHT CHANNELS ALPHA
augmatch_cli directed_edge_detect IN.raw OUT.raw WIDTH HEIGHT CHANNELS ALPHA DIRECTION
```

`tests/imgaug_convolutional_parity.py` compares all three operations exactly
against the corresponding OpenCV reference.

## Canny edge contract

`canny_u8` implements deterministic Canny with explicit low/high gradient
thresholds, odd Sobel aperture sizes 3, 5, or 7, and `BorderPolicy::Clamp`
(replicated samples) or `BorderPolicy::Constant` (the configured
`border_value`). Input is contiguous interleaved HWC `uint8`: one channel is
used directly, while RGB and larger images use BT.601 luminance from channels
0--2; channels after RGB, including alpha, are ignored. The binary edge mask
(0 or 255) is replicated to every output channel. This preserves image shape
without changing the existing derivative APIs. CPU pointers are host pointers;
CUDA pointers are device pointers. The CLI is:

```sh
augmatch_cli canny IN.raw OUT.raw WIDTH HEIGHT CHANNELS LOW HIGH APERTURE BORDER BORDER_VALUE
```

`BORDER` is `0` for Clamp or `1` for Constant. Canny parity and deterministic
channel/property checks are in `tests/canny_parity.py`; `tests/canny_unit.cpp`
checks validation and repeatability.

## AdvancedBlur contract

`advanced_blur_u8` is a deterministic simplification of parameterized augmentation libraries: it does not sample sigma, rotation, beta, or per-kernel noise. `AdvancedBlurConfig` supplies `kernel_size` (odd, at most 15), `sigma_x`, `sigma_y`, and `angle_degrees`. For integer offset `(dx,dy)`, rotate to `x'=cos(theta)dx+sin(theta)dy` and `y'=-sin(theta)dx+cos(theta)dy`; the kernel weight is `exp(-0.5*((x'/sigma_x)^2+(y'/sigma_y)^2))`, normalized by the sum over the finite square kernel. Each channel is filtered independently with `REFLECT_101` borders and rounded to the nearest clipped uint8 value (nonnegative half values round up). The CLI form is `advanced_blur IN.raw OUT.raw W H C KERNEL_SIZE SIGMA_X SIGMA_Y ANGLE_DEGREES`.

## BilateralBlur contract

`bilateral_blur_u8` is a deterministic imgaug-style bilateral filter. `BilateralBlurConfig` supplies an integer `radius` in `[1,15]`, positive `sigma_space`, and positive `sigma_color`; there is no implicit random parameter sampling. For each offset `(dx,dy)` in the finite square, the weight is `exp(-(dx^2+dy^2)/(2*sigma_space^2) - ||I(x+dx,y+dy)-I(x,y)||^2/(2*sigma_color^2))`, where the color norm spans all interleaved channels. Samples outside the image use `REFLECT_101`. Each channel receives the normalized weighted sum independently, clipped to `[0,255]`, with nonnegative half values rounded up. CPU and CUDA use the same radius, border, color-distance, and rounding contract. The CLI form is `bilateral_blur IN.raw OUT.raw W H C RADIUS SIGMA_SPACE SIGMA_COLOR`.

## MeanShiftBlur contract

`mean_shift_blur_u8` is a deterministic fixed-iteration spatial/color mean-shift filter. `MeanShiftBlurConfig` supplies an integer `spatial_radius` in `[1,15]`, positive `color_radius`, and `iterations` in `[1,16]`; no random parameters are sampled. For each output pixel, every iteration examines the finite square window around the original pixel, using `REFLECT_101` for samples outside the image. A sample participates when the complete interleaved pixel's squared Euclidean color distance from the current color is at most `color_radius^2`. The current color is replaced by the arithmetic mean of participating samples; if none participate, it remains unchanged. After the fixed iterations, values are clipped to `[0,255]` and rounded with nonnegative half-up uint8 rounding. CPU and CUDA use the same neighborhood, color-distance, border, iteration, and rounding contract. The CLI form is `mean_shift_blur IN.raw OUT.raw W H C SPATIAL_RADIUS COLOR_RADIUS ITERATIONS`.

## GlassBlur contract

`glass_blur_u8` first applies a normalized Gaussian kernel with `radius=ceil(3*sigma)` (limited to 15) and `REFLECT_101` borders, then performs `iterations` swap passes over the interior rectangle `[max_delta, height-max_delta) x [max_delta, width-max_delta)`, and applies the same Gaussian blur again. A `GlassBlurSwap` entry `(dx,dy)` swaps all channels at the current pixel with `(x+dx,y+dy)`. The borrowed sequence is ordered by iteration, then y, then x; every offset must lie in `[-max_delta,max_delta]`. If it is null, `make_glass_blur_swap_sequence` uses SplitMix64 at `seed+2*k` and `seed+2*k+1` to select each offset, with no global RNG state. The CUDA sequence pointer is device memory. The CLI form is `glass_blur IN.raw OUT.raw W H C SIGMA MAX_DELTA ITERATIONS SEED [SWAPS.i8]`, where the optional file contains two signed bytes `(dx,dy)` per sequence entry.

## RingingOvershoot contract

`ringing_overshoot_u8` is a deterministic Gaussian high-boost filter. For each channel it computes the normalized finite-square Gaussian blur `B` with `w(dx,dy)=exp(-(dx^2+dy^2)/(2*sigma^2))` and `REFLECT_101` borders, then emits `clip(round(I + amount*(I-B)), 0, 255)`. The odd `kernel_size` is limited to 15, `sigma` must be positive, and `amount` is finite in `[0,4]`. The residual deliberately creates bright and dark edge overshoot (ringing) without random sampling; repeated calls with the same input and configuration are identical. The CLI form is `ringing_overshoot IN.raw OUT.raw W H C KERNEL_SIZE SIGMA AMOUNT`.

## ZoomBlur contract

`zoom_blur_u8` averages `steps+1` equally weighted, center-preserving bilinear samples. Its factors are the explicit deterministic sequence `min_factor + i*(max_factor-min_factor)/steps`, for `i=0..steps`; coordinates outside the image use `REFLECT_101`. `min_factor` must be at least 1, and `steps` is limited to 256. This explicit interval and step count replaces the randomized factor range and implicit sampling used by Albumentations/imgaug, so repeated calls are identical without a seed. The CLI form is `zoom_blur IN.raw OUT.raw W H C MIN_FACTOR MAX_FACTOR STEPS`.

## NonLocalMeansDenoising contract

`non_local_means_denoising_u8` uses a deterministic, bounded square search. `NonLocalMeansDenoisingConfig::patch_radius` is the complete patch radius and is limited to `[0,4]`; `search_radius` is the candidate radius and is limited to `[0,8]`. Every candidate in the inclusive square window is evaluated, including the center candidate. Its weight is `exp(-d/(h*h))`, where `d` is the mean squared difference over the complete patch and all channels, and `h` must be finite and positive. Patches and candidate pixels use `REFLECT_101` borders. The normalized weighted result is independently rounded to nearest uint8 per channel, with nonnegative half values rounding up. No random state or implicit search expansion is used, so CPU and CUDA calls are deterministic for the same input and configuration. The CLI form is `non_local_means IN.raw OUT.raw W H C PATCH_RADIUS SEARCH_RADIUS H` (the alias `non_local_means_denoising` is also accepted).
