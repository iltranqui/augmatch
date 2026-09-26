# Geometry, size, dropout, and mixing

Headers: `include/augmatch/geometry/`. Back to the [docs index](../README.md).

## Contents

- [Mixing and multi-image transforms](#mixing-and-multi-image-transforms)
- [Downscale contract](#downscale-contract)
- [MaskDropout contract](#maskdropout-contract)
- [XYMasking contract](#xymasking-contract)
- [Cutout and TotalDropout contracts](#cutout-and-totaldropout-contracts)
- [Crop, flip, and random-crop contract](#crop-flip-and-random-crop-contract)

## Mixing and multi-image transforms

`include/augmatch/geometry/mixing.hpp` defines explicit borrowed `ImageSourceU8` and
`ImageBatchU8` records for contiguous HWC `uint8` inputs. `mixup_u8`,
`cutmix_u8`, `mosaic_u8`, `template_transform_u8`, and `overlay_elements_u8`
write caller-owned output buffers. CPU pointers are host pointers; CUDA
pointers and descriptor arrays are device pointers and remain valid until the
stream completes. `MixUp` clamps an explicit weight (or derives one from a
seed), `CutMix` uses a half-open output box, `Mosaic` maps four sources into
quadrants with nearest-neighbour scaling, and `TemplateTransform` normalizes
its supplied image/template weights. These are deterministic native
approximations, not hidden global-RNG or framework-specific implementations.

Target boxes and labels use `ReadOnlyMixingTargets` and `MixingTargets`.
Metadata arrays are borrowed and caller-owned; the caller supplies output
capacity and the operation writes `count`. MixUp concatenates records, while
CutMix clips patch records to the selected box. No operation allocates or
retains target metadata. Image-only Mosaic, TemplateTransform, and
OverlayElements leave caller target records unchanged; callers explicitly own
any corresponding target transforms.

Raw CLI examples (HWC bytes) are:

```sh
augmatch_cli mixup A.raw B.raw OUT.raw W H C 0.5 7
augmatch_cli cutmix A.raw B.raw OUT.raw W H C 8 8 24 24
augmatch_cli mosaic A.raw B.raw C.raw D.raw OUT.raw W H C CX CY
augmatch_cli template_transform A.raw T.raw OUT.raw W H C 1 1
augmatch_cli overlay_elements BASE.raw ELEMENT.raw OUT.raw W H C 4 4 0.75
```

CUDA builds expose the same functions and stream argument. CUDA batch and
overlay descriptor arrays must themselves be device-resident. Mosaic and
template scaling intentionally use nearest-neighbour sampling; alpha and
weight rounding uses half-up conversion to `[0,255]`.

## Downscale contract

`downscale_u8` is a deterministic low-resolution round trip, not a codec-dependent compression operation. Its scale is a finite value in `(0,1]` and defaults to the fixed value `0.5`; reduced dimensions are `max(1, floor(width * scale))` and `max(1, floor(height * scale))`. It first calls the native `resize_u8` kernel with the configured interpolation and then calls that same native kernel to upsample back to the original dimensions with `upsample_interpolation`. The default for both stages is linear interpolation. No JPEG, WebP, or other compression library is used, and no random state is sampled. The CPU and CUDA APIs use the same contract. The CLI form is `downscale IN.raw OUT.raw W H C [SCALE [INTERPOLATION [UPSAMPLE_INTERPOLATION]]]`; interpolation values are `0=nearest`, `1=linear`, and `2=area`.

## MaskDropout contract

`mask_dropout_u8` takes an HWC `uint8` image and a separate one-byte-per-pixel HxW mask. A zero mask byte keeps all channels unchanged; every nonzero byte drops the pixel and writes the scalar `MaskDropoutConfig::fill` to every channel. The mask is read in row-major order, is never modified, and the operation has no random state, so CPU and CUDA outputs are exact for identical inputs, masks, and configuration. The mask may use either `1` or conventional `255` foreground values. The CLI form is `dropout mask IN.raw OUT.raw MASK.raw W H C FILL`.

## XYMasking contract

`xy_masking_u8` uses explicit half-open row and column intervals. A pixel is
masked when its row belongs to any configured row interval or its column belongs
to any configured column interval. The interval arrays are borrowed and ordered
as supplied; empty arrays are valid. Every channel of a masked HWC pixel is
replaced by `fill`, while all unmasked pixels are copied unchanged. The CPU API
expects interval arrays in host memory; the CUDA API expects them in device
memory. No random state is used, so repeated calls are exact. The CLI accepts
`xy_masking IN.raw OUT.raw W H C FILL ROWS COLUMNS`, where `ROWS` and `COLUMNS`
are comma-separated half-open intervals such as `1:3,7:9`, or `-` for none. The
same operation is available as `dropout xy_masking ...`.

## Cutout and TotalDropout contracts

`cutout_u8` applies an ordered, borrowed list of integer half-open rectangles
`[x0,x1) x [y0,y1)` to interleaved HWC `uint8` data. Rectangles must lie inside
the image; empty lists and zero-area rectangles are identities. Every channel in
a covered pixel receives the scalar `CutoutConfig::fill`, while uncovered pixels
are copied unchanged. CPU callers own host rectangle storage and CUDA callers own
device rectangle storage until the call returns. The explicit rectangle path
consumes no random state; `seed` is reserved and ignored. The CLI form is
`cutout IN.raw OUT.raw W H C FILL X0:Y0:X1:Y1,...`; `-` selects no rectangles,
and `dropout cutout` is an equivalent spelling.

`total_dropout_u8` replaces the complete image with scalar `fill` or copies it.
When `TotalDropoutConfig::mask` is non-null, it points to one borrowed byte
(host memory on CPU, device memory on CUDA); zero keeps the image and nonzero
drops it, regardless of probability or seed. With a null mask, one SplitMix64
draw keyed by `seed` drops when it is below `probability` in `[0,1]`. Thus the
seed is local to the call and repeated configurations are exact; there is no
global RNG state. The CLI supports `dropout total IN.raw OUT.raw W H C
PROBABILITY FILL SEED`, or the explicit decision form `... FILL MASK
PROBABILITY SEED` where `MASK` is `0` or nonzero. Existing pixel, channel, grid,
coarse, mask, and XY dropout APIs retain their prior contracts.

## Crop, flip, and random-crop contract

Crop coordinates are top-left `(crop_x,crop_y)` with extent `(output_width,output_height)`. Operations run in crop, horizontal flip, vertical flip order. `transform_u8` and `pool_u8` accept device pointers and a CUDA stream in CUDA builds, and host pointers in CPU builds. `RandomCropFromBorders` accepts four maximum fractions in left, right, top, bottom order. Each limit is `floor(fraction * input_dimension)`. Config offsets default to `-1`, which selects `splitmix64(seed + side_index) % (limit + 1)` with side indices left=0, right=1, top=2, bottom=3. Nonnegative offsets are explicit and override the seed; they must not exceed their limits. The maximum allowed crop leaves at least one pixel per dimension, and no global RNG state is used. Its CLI form is `crop_from_borders IN.raw OUT.raw W H C LEFT_F RIGHT_F TOP_F BOTTOM_F SEED [LEFT RIGHT TOP BOTTOM]`. `RandomResizedCrop` and `RandomSizedCrop` share an explicit source rectangle and resize contract: `OP IN.raw OUT.raw W H C OUT_W OUT_H CROP_X CROP_Y CROP_W CROP_H INTERPOLATION [SEED]`, with interpolation `0=Nearest`, `1=Linear`, or `2=Area`. Set both crop dimensions to zero and an offset to `-1` to generate a rectangle using SplitMix64 seed offsets width=0, height=1, x=2, y=3; explicit coordinates are deterministic and identical on CPU/CUDA. This generation contract intentionally differs from hidden Albumentations sampling/rejection loops, while resize uses the existing `Resize` implementation exactly. `RandomCropNearBBox` takes `W H C X1 Y1 X2 Y2 MAX_SHIFT_X MAX_SHIFT_Y SEED [LEFT RIGHT TOP BOTTOM]` in its CLI form. The box is continuous Pascal VOC `(x1,y1,x2,y2)` image-edge coordinates; the crop support is the half-open pixel rectangle from floor/ceil box edges plus deterministic or explicit integer margins, clipped to the image. Its API returns this rectangle for output allocation and changes image pixels only. Box and other annotation transformation is explicit metadata owned by the caller, rather than hidden Python targets. Target-aware crops take a borrowed host `BoxXYXY` array: `BBoxSafeRandomCrop` contains the all-box envelope, `RandomSizedBBoxSafeCrop` applies that source selection then resizes, and `AtLeastOneBBoxRandomCrop` intersects one deterministic selected box. Empty arrays use their documented explicit or seeded fallback. Factories return source geometry (and scale for the resized form); callers own annotation transformation. The target-aware CLI adapters use a text box file with one `x1 y1 x2 y2` record per line. `RandomGridShuffle` uses row-major destination-to-source cell permutations; callers may provide a borrowed host/device permutation or leave it null for the documented deterministic SplitMix64 seed shuffle. Its CLI form is `random_grid_shuffle IN.raw OUT.raw W H C GRID_ROWS GRID_COLS SEED [PERMUTATION.u32]`, where the optional permutation file contains little-endian `uint32` source cell indices. The CLI is a parity-test adapter that copies raw images to and from the GPU.
