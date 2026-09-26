# augmatch

**TL;DR:** a native C++17/CUDA image-augmentation library — drop-in transform
catalog with Albumentations/imgaug parity, plus a config-driven `Pipeline` API
(`augmentation.cfg`) so you can embed it as a git submodule in another
C++/CUDA project and run a whole augmentation sequence with one call.

A C++17/CUDA image-augmentation library: native CPU and CUDA implementations
of image-augmentation transforms (geometric, color/HSV, noise, blur, weather,
compression artifacts, sensor/ISP simulation, and more — see
`IMPLEMENTATION_MANIFEST.tsv` for the full catalog), with Python parity checks
against Albumentations/imgaug where an exact reference exists.

## Features

- Native CPU + CUDA implementations behind one set of headers, selected at
  build time (`AUGMATCH_ENABLE_CUDA`, default `ON`; falls back to a CPU-only
  build automatically when `nvcc` isn't found).
- A large transform catalog (crop/flip/resize, HSV and color adjustments,
  blur/noise/weather/compression artifacts, sensor/ISP noise simulation,
  bounding-box/keypoint metadata ops, and more) — see the section headings
  below and `IMPLEMENTATION_MANIFEST.tsv` for the itemized status of each.
- `augmatch::Pipeline` (see [`PIPELINE.md`](PIPELINE.md)): load an ordered
  list of augmentation stages from a small INI-style `augmentation.cfg` file
  and run it with one call — the config lists transforms and parameters, no
  recompilation needed to change the augmentation sequence.
- Builds as a static library with an installable CMake package
  (`augmatch::augmatch`), so it can be vendored as a git submodule and pulled
  in with `add_subdirectory` from another C++/CUDA project — see
  [`PIPELINE.md`](PIPELINE.md#vendoring-as-a-git-submodule) for the exact
  CMake snippet, including the options (`AUGMATCH_BUILD_TESTS`,
  `AUGMATCH_BUILD_EXAMPLES`, `AUGMATCH_ENABLE_WEBP/FFMPEG/JPEG`) that keep a
  submodule build lean.
- Public headers are C++17 *and* C++23 compatible (see the paired
  `examples/*_cpp17.cpp` / `*_cpp23.cpp` examples and their CTest smoke
  tests).

## Quickstart

```sh
git clone https://github.com/iltranqui/augmatch.git
cd augmatch
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j 4
ctest --test-dir build -C Release --output-on-failure
```

See [Build](#build) below for the CPU-only variant and Python test
dependencies, and [`PIPELINE.md`](PIPELINE.md) for the `augmatch::Pipeline` /
`augmentation.cfg` API.

## Documentation map

This README doubles as a deep-dive API reference, organized by catalog area
(the `##`/`###` headings below). Related documents:

- [`PIPELINE.md`](PIPELINE.md) — the config-driven `Pipeline` API and how to
  vendor this library as a git submodule.
- [`AUGMENTATION_CATALOG.md`](AUGMENTATION_CATALOG.md), [`NOISE_CATALOG.md`](NOISE_CATALOG.md),
  [`BLEND_CATALOG.md`](BLEND_CATALOG.md), [`SIZE_CATALOG.md`](SIZE_CATALOG.md),
  [`IMGCORRUPTLIKE_CATALOG.md`](IMGCORRUPTLIKE_CATALOG.md) — per-area transform
  catalogs and their implementation checklists.
- [`IMPLEMENTATION_MANIFEST.tsv`](IMPLEMENTATION_MANIFEST.tsv),
  [`COMPLETION_AUDIT.md`](COMPLETION_AUDIT.md), [`PARITY_AUDIT.md`](PARITY_AUDIT.md) —
  the generated status of every catalog item and the latest parity-test
  evidence.
- [`API_DESIGN.md`](API_DESIGN.md), [`SUPPORT_CONTRACT.md`](SUPPORT_CONTRACT.md),
  [`COMPATIBILITY.md`](COMPATIBILITY.md) — API conventions, platform/backend
  support guarantees.

## Additional imgaug catalog APIs

`remaining_catalog.hpp` supplies deterministic HWC `uint8` APIs for Laplace and
Poisson noise, Cartoon, RandAugment, RGB/HSV colorspace changes, seeded RGB
k-means quantization, borrowed-kernel Convolve, polar warping, Jigsaw, and FDA.
The implementations are explicit host-only contracts even in CUDA builds;
callers must supply host pointers, and no implicit device-memory dereference is provided.
FDA/FourierDomainAdaptation is a documented low-frequency color-statistics
approximation (not a full FFT amplitude swap). LAB conversion is an explicit
copy approximation. Debug output is synchronous PGM/PPM; callbacks and
framework-managed file sinks are unsupported.

The raw CLI form is:

```sh
augmatch_cli remaining additive_laplace IN.raw OUT.raw W H C SCALE SEED
augmatch_cli remaining additive_poisson IN.raw OUT.raw W H C SCALE SEED
augmatch_cli remaining kmeans IN.raw OUT.raw W H C CLUSTERS ITERATIONS SEED
augmatch_cli remaining convolve IN.raw OUT.raw W H C
```
 It implements crop, horizontal/vertical flips, nearest/linear resize, physical sensor noise, derivative-dependent ISP noise, Pascal VOC `xyxy` boxes, `xy` keypoints, and imgaug's average, max, min, and median pooling augmenters.

## Mixing and multi-image transforms

`include/augmatch/mixing.hpp` defines explicit borrowed `ImageSourceU8` and
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

## Build

Requires CMake 3.24+ and a C++17 compiler. CMake enables the CUDA backend when `nvcc` is available; otherwise it builds a CPU fallback so geometry and parity tests can still run. The same project targets Linux and Windows (use the Visual Studio generator on Windows).

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
python3 tools_generate_manifest.py --check
python3 tools_audit_manifest.py

# CPU-only build, useful on systems without a compatible CUDA runtime:
cmake -S . -B build-cpu -DAUGMATCH_ENABLE_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
```

Install Python test dependencies with `pip install -r requirements-test.txt`.

### imgcorruptlike compatibility contract

`imgcorruptlike.hpp` provides deterministic CPU/CUDA approximations for
SpeckleNoise, Fog, Frost, Snow, Contrast, Brightness, Saturate, and Pixelate.
They use explicit configs and coordinate-keyed seeds rather than imgaug or
Albumentations global random state. See `IMGCORRUPTLIKE_CATALOG.md` for the
formulae, borrowed field ownership, and raw-byte CLI forms.

### ToTensorV2 native tensor-buffer contract

`to_tensor_v2_u8_f32` and `to_tensor_v2` convert contiguous HWC `uint8` pixels to a caller-owned contiguous CHW `float32` buffer. The output index is `output[c * height * width + y * width + x]`; without normalization it is `input[(y * width + x) * channels + c] / 255`. With `normalize=true`, the value is `(input / 255 - mean[min(c,2)]) / std[min(c,2)]`; channels after the third reuse the third parameter. `TensorViewF32` and `MutableTensorViewF32` explicitly describe CHW layout, float32 dtype, element strides, and host/device memory. The conversion does not allocate or copy ownership and is a native tensor-buffer operation, not a dependency on PyTorch.

CPU and CUDA use the same API and binary layout. The CUDA view API requires device views and the CPU view API requires host views; callers synchronize a supplied CUDA stream when completion is needed. The CLI path writes little-endian float32 CHW data:

```sh
augmatch_cli to_tensor_v2 IN.raw OUT.f32 WIDTH HEIGHT CHANNELS [NORMALIZE]
augmatch_cli to_tensor_v2 IN.raw OUT.f32 WIDTH HEIGHT CHANNELS 1 MEAN0 MEAN1 MEAN2 STD0 STD1 STD2
```

### ToTensor3D native tensor-buffer contract

`to_tensor_3d_u8_f32` converts contiguous DHWC `uint8` voxels to contiguous
CDHW `float32` data. The source index is `(((d * H + y) * W + x) * C + c)`;
the output index is `(((c * D + d) * H + y) * W + x)`. Values are divided by
255, then optionally normalized with the same three channel mean/std pairs as
ToTensorV2 (channels after the third reuse the third pair). CPU pointers refer
to host memory and CUDA pointers refer to device memory; the stream argument is
asynchronous on CUDA. The CLI writes little-endian float32 CDHW data:

```sh
augmatch_cli to_tensor_3d IN.raw OUT.f32 WIDTH HEIGHT DEPTH CHANNELS [NORMALIZE]
augmatch_cli to_tensor_3d IN.raw OUT.f32 WIDTH HEIGHT DEPTH CHANNELS 1 MEAN0 MEAN1 MEAN2 STD0 STD1 STD2
```

## Segmentation Voronoi contract

`include/augmatch/voronoi.hpp` provides four deterministic CPU/CUDA Voronoi
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

## Native sensor API contract

`augmatch/native_api.hpp` is the C++17-first API for the remaining
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

## Remaining video and lens artifact contract

`random_telegraph_signal_noise_f32` is a deterministic HWC float32 sensor
operation with an optional borrowed uint8 state map or coordinate-keyed seed.
The video APIs consume contiguous T-H-W-C float32 batches:
`inter_frame_compression_noise_f32`, `gop_keyframe_artifacts_f32`, and
`block_motion_estimation_artifacts_f32` accept scalar parameters or explicit
per-frame/keyframe/block records and use codec-independent residual/vector
surrogates. They do not parse or emit AV1, H.264, or H.265. `water_droplets_on_lens_f32`
consumes HWC float32 data and explicit or seeded `WaterDroplet` records; its
opacity-weighted local-box blur is intentionally not a refractive fluid model.
CLI forms and record ownership are documented in `NOISE_CATALOG.md`.

## Environmental and acquisition contract

`augmatch/environmental.hpp` adds deterministic float32 HWC veil, gain, dirty-lens, and electromagnetic-interference APIs, plus T-H-W-C sensor-temperature drift, power-supply banding, fluorescent flicker, and LED rolling-band APIs. HWC maps are borrowed HxW or HxWxC arrays in the execution memory space; temporal batches are contiguous T-H-W-C. CPU and CUDA maps follow the existing host/device ownership contract, and all outputs clip to the configured range. Dirty-lens is an opacity-weighted square-box blur; EMI is a sinusoidal additive field; temperature drift is exponential dark-current scaling across an explicit temperature ramp; the lighting artifacts are sinusoidal acquisition approximations rather than physical lens, RF, ballast, PWM, or exposure simulations. CLI forms and parameters are documented in `NOISE_CATALOG.md`. Fog, rain, snow, and dust use the explicit seeded weather records in `weather.hpp`; headlight flare and reflections reuse the explicit optical source/ghost records.

## Blend augmentation contract

`include/augmatch/blend.hpp` implements all `augmenters.blend` entries with paired
CPU/CUDA APIs. Source and overlay are contiguous HWC `uint8` buffers; masks and
segmentation maps are borrowed HxW `uint8` metadata, and boxes/class IDs/color
records remain caller-owned. Alpha is clamped and half-up rounded. Seeded
masks and noise are deterministic and do not use global RNG state. Noise fields
are documented portable approximations. See `BLEND_CATALOG.md` for ownership,
CLI, and approximation details.

## Composition and control contract

`composition.hpp` provides deterministic in-place `compose` and `sequential`
pipelines plus validation-preserving `identity` and `noop` operations. CPU
stages are borrowed synchronous callbacks receiving a mutable image view, a
read-only `ExecutionContext`, and caller-owned `void*` context; callbacks must
not retain either pointer. Stage order is preserved and the context seed is
never changed by the pipeline.

CUDA does not accept CPU callbacks for device images. `DeviceStage` is the
explicit boundary: its host launch function enqueues device work on the supplied
stream, and its context remains caller-owned until the call returns. A null
launch is an identity/no-op stage. Pipelines do not synchronize CUDA streams;
callers synchronize when completion is required.

`one_of`, `one_or_other`, `random_apply`, `some_of`, and `random_order` operate
on vectors or pipelines of `CpuStage`/`DeviceStage` representations. Each
decision uses `SplitMix64(seed + traversal_index)` and never consults a global
RNG. `SomeOf` records selected child indices and `RandomOrder` records a full
permutation. Set `ExecutionContext::record` to append explicit
`StageSelectionRecord` entries; set `ExecutionContext::replay` to validate and
reuse those entries. `replay_compose` is the explicit top-level replay boundary
and resets traversal at the start of a call. `ChannelRange`,
`ChannelSelection`, and `selective_channel_transform` pass borrowed channel
subviews to CPU callbacks or CUDA launch callbacks, retaining original strides
and avoiding copies. `with_channels` applies a child pipeline to explicit
imgaug-style channel indices or ranges. Discontiguous HWC selections invoke a
stage once per contiguous run. CPU and CUDA controls share the record format,
while device stages remain host launch callbacks.

## Implemented imgaug transforms

### Convolutional filters contract

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

### Canny edge contract

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

### Elementwise arithmetic contract

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

### ReplaceElementwise and ImpulseNoise contract

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
### Salt, Pepper, and coarse impulse contracts

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

### JPEG ImageCompression/JpegCompression contract

`jpeg_compression_u8` and its `image_compression_u8` alias perform a real JPEG encode/decode round trip through libjpeg-turbo (or a compatible libjpeg implementation discovered by CMake). The input is contiguous HWC `uint8` with one grayscale or three RGB channels. `JpegCompressionConfig` makes quality `[1,100]` and chroma sampling explicit: `Y444`, `Y422`, or `Y420`; the output has the same dimensions and channel count. `jpeg_encode_u8` exposes the host-owned encoded byte stream, and `jpeg_decode_u8` decodes it into a caller-owned buffer. JPEG is lossy, so exact identity is not expected.

The CPU implementation is native libjpeg-turbo. The CUDA `.cu` implementation does **not** claim GPU-native JPEG: it synchronously copies device pixels to host memory, invokes the host codec, and copies decoded pixels back to the device. This fallback is intentional because the project has no CUDA JPEG codec dependency. If libjpeg-turbo/libjpeg is not found, CMake disables these APIs and reports the dependency warning. The CLI form is `jpeg_compression IN.raw OUT.raw W H C QUALITY SUBSAMPLING`, where subsampling is `444`, `422`, or `420`; `image_compression` is an alias.

`jpeg_quality_variation_u8` exposes the same codec with an explicit quality value, and `jpeg_chroma_subsampling_u8` exposes the three libjpeg sampling modes directly. `jpeg_quantization_table_variation_u8` accepts independent positive luminance and chrominance table multipliers (one is neutral); its CLI is `jpeg_quantization_table_variation IN.raw OUT.raw W H C QUALITY LUMA_SCALE CHROMA_SCALE SUBSAMPLING`. JPEG ringing, blocking, and mosquito-noise entries first perform the same libjpeg round trip and then apply deterministic, clipped post-decode artifact models. Their shared CLI form is `jpeg_{ringing|blocking|mosquito_noise} IN.raw OUT.raw W H C QUALITY SUBSAMPLING STRENGTH`.

`jpeg_restart_marker_damage_u8` emits a configured restart interval and deterministically corrupts entropy bytes at selected restart boundaries before decoding. `jpeg_progressive_decoding_u8` uses a progressive JPEG stream and exposes a deterministic early-scan preview model. Their CLI forms are `jpeg_restart_marker_damage IN.raw OUT.raw W H C QUALITY SUBSAMPLING STRENGTH RESTART_INTERVAL` and `jpeg_progressive_decoding IN.raw OUT.raw W H C QUALITY SUBSAMPLING STRENGTH SCAN`. All CPU pointers are host-owned. CUDA pointers are device-owned, but every JPEG entry uses the documented synchronous host-codec fallback and honors the supplied stream for transfers.

### Codec blockers and WebP

`webp_lossy_compression_u8` and `webp_lossless_compression_u8` are explicit libwebp-backed catalog entry points; their aliases are `webp_lossy_u8` and `webp_lossless_u8`. The CLI forms are `webp_lossy_compression IN.raw OUT.raw W H C QUALITY` and `webp_lossless_compression IN.raw OUT.raw W H C QUALITY`. CMake conditionally enables libwebp when both public headers and the native library are found. Without it, the public symbols remain available but fail clearly with a dependency error rather than silently substituting another codec. CUDA uses the documented synchronous host-codec fallback.

`video_codec.hpp` provides conditional native FFmpeg single-frame encode/decode for AV1, H.264, and HEVC, plus an AV1 intra encode/decode artifact entry point. Its CLI round-trip is `augmatch_cli video_codec IN.raw OUT.raw W H C h264|hevc|av1 QUALITY`; input/output are interleaved uint8 grayscale or RGB, dimensions must be positive and even, and quality is in `[1,100]`. CMake enables the API only when libavcodec, libavutil, and libswscale development files are found; otherwise calls fail clearly as unavailable. This is not an arbitrary bitstream parser or mutator and does not implement temporal prediction, reference-frame semantics, container support, or external-codec parity, so all four catalog entries remain partial. CUDA buffers use a documented synchronous host-codec fallback rather than GPU-native encoding. `video_codec_unit` attempts actual encoder/decoder round trips for each codec and reports unavailable individual encoders without turning transport surrogates into codec behavior.

### Codec-independent transport corruption

`transport.hpp` adds raw transport operations without an external video codec. `truncated_frame_u8` and `bit_flips_u8` operate on explicit byte counts; `block_quantization_u8`, `deblocking_filter_mismatch_u8`, and `packet_loss_macroblocks_u8` operate on interleaved HWC frames; `chroma_plane_misalignment_u8` operates on one row-major plane (invoke it separately for U and V); and `raw_bayer_transport_corruption_u16` operates on unpacked Bayer samples with explicit bit depth. CPU pointers are host-owned and CUDA pointers are device-owned; optional masks are borrowed in the execution memory space and stream-safe. Every operation has a CUDA kernel where meaningful. Example CLI forms are `truncated_frame IN OUT BYTES PRESERVE FILL`, `bit_flips IN OUT BYTES PROBABILITY MASK SEED`, `block_quantization IN OUT W H C BLOCK QUANT_STEP`, and `chroma_plane_misalignment IN OUT W H STRIDE DX DY FILL`. AV1, H.264, and H.265 bitstream parsing, reference-picture reconstruction, and codec-specific damage remain unsupported blockers.

### WebP WebPCompression contract

`webp_compression_u8` performs a real native libwebp encode/decode round trip for contiguous HWC `uint8` images with one grayscale or three RGB channels. `WebPCompressionConfig` makes quality `[0,100]` and the `lossless` mode explicit. Lossy mode uses libwebp's quality scale; lossless mode selects libwebp lossless encoding. The output preserves dimensions and channel count, while lossy output is not expected to be identical. `webp_encode_u8` exposes the host-owned WebP byte stream and `webp_decode_u8` decodes it into a caller-owned buffer.

The CPU implementation is native libwebp. The CUDA `.cu` implementation does **not** claim GPU-native WebP: it synchronously copies device pixels to host memory, invokes libwebp, and copies decoded pixels back to the device. CMake enables these APIs and their unit/parity tests only when both libwebp headers and the native library are found; otherwise it reports a warning and omits the codec source. The CLI form is `webp_compression IN.raw OUT.raw W H C QUALITY LOSSLESS`, where `LOSSLESS` is `0` or `1`; `webp` is an alias. This entry covers WebP image compression only and makes no AV1 or video-codec claim.

### Downscale contract

`downscale_u8` is a deterministic low-resolution round trip, not a codec-dependent compression operation. Its scale is a finite value in `(0,1]` and defaults to the fixed value `0.5`; reduced dimensions are `max(1, floor(width * scale))` and `max(1, floor(height * scale))`. It first calls the native `resize_u8` kernel with the configured interpolation and then calls that same native kernel to upsample back to the original dimensions with `upsample_interpolation`. The default for both stages is linear interpolation. No JPEG, WebP, or other compression library is used, and no random state is sampled. The CPU and CUDA APIs use the same contract. The CLI form is `downscale IN.raw OUT.raw W H C [SCALE [INTERPOLATION [UPSAMPLE_INTERPOLATION]]]`; interpolation values are `0=nearest`, `1=linear`, and `2=area`.

### AdvancedBlur contract

`advanced_blur_u8` is a deterministic simplification of parameterized augmentation libraries: it does not sample sigma, rotation, beta, or per-kernel noise. `AdvancedBlurConfig` supplies `kernel_size` (odd, at most 15), `sigma_x`, `sigma_y`, and `angle_degrees`. For integer offset `(dx,dy)`, rotate to `x'=cos(theta)dx+sin(theta)dy` and `y'=-sin(theta)dx+cos(theta)dy`; the kernel weight is `exp(-0.5*((x'/sigma_x)^2+(y'/sigma_y)^2))`, normalized by the sum over the finite square kernel. Each channel is filtered independently with `REFLECT_101` borders and rounded to the nearest clipped uint8 value (nonnegative half values round up). The CLI form is `advanced_blur IN.raw OUT.raw W H C KERNEL_SIZE SIGMA_X SIGMA_Y ANGLE_DEGREES`.

### BilateralBlur contract

`bilateral_blur_u8` is a deterministic imgaug-style bilateral filter. `BilateralBlurConfig` supplies an integer `radius` in `[1,15]`, positive `sigma_space`, and positive `sigma_color`; there is no implicit random parameter sampling. For each offset `(dx,dy)` in the finite square, the weight is `exp(-(dx^2+dy^2)/(2*sigma_space^2) - ||I(x+dx,y+dy)-I(x,y)||^2/(2*sigma_color^2))`, where the color norm spans all interleaved channels. Samples outside the image use `REFLECT_101`. Each channel receives the normalized weighted sum independently, clipped to `[0,255]`, with nonnegative half values rounded up. CPU and CUDA use the same radius, border, color-distance, and rounding contract. The CLI form is `bilateral_blur IN.raw OUT.raw W H C RADIUS SIGMA_SPACE SIGMA_COLOR`.

### MeanShiftBlur contract

`mean_shift_blur_u8` is a deterministic fixed-iteration spatial/color mean-shift filter. `MeanShiftBlurConfig` supplies an integer `spatial_radius` in `[1,15]`, positive `color_radius`, and `iterations` in `[1,16]`; no random parameters are sampled. For each output pixel, every iteration examines the finite square window around the original pixel, using `REFLECT_101` for samples outside the image. A sample participates when the complete interleaved pixel's squared Euclidean color distance from the current color is at most `color_radius^2`. The current color is replaced by the arithmetic mean of participating samples; if none participate, it remains unchanged. After the fixed iterations, values are clipped to `[0,255]` and rounded with nonnegative half-up uint8 rounding. CPU and CUDA use the same neighborhood, color-distance, border, iteration, and rounding contract. The CLI form is `mean_shift_blur IN.raw OUT.raw W H C SPATIAL_RADIUS COLOR_RADIUS ITERATIONS`.

### GlassBlur contract

`glass_blur_u8` first applies a normalized Gaussian kernel with `radius=ceil(3*sigma)` (limited to 15) and `REFLECT_101` borders, then performs `iterations` swap passes over the interior rectangle `[max_delta, height-max_delta) x [max_delta, width-max_delta)`, and applies the same Gaussian blur again. A `GlassBlurSwap` entry `(dx,dy)` swaps all channels at the current pixel with `(x+dx,y+dy)`. The borrowed sequence is ordered by iteration, then y, then x; every offset must lie in `[-max_delta,max_delta]`. If it is null, `make_glass_blur_swap_sequence` uses SplitMix64 at `seed+2*k` and `seed+2*k+1` to select each offset, with no global RNG state. The CUDA sequence pointer is device memory. The CLI form is `glass_blur IN.raw OUT.raw W H C SIGMA MAX_DELTA ITERATIONS SEED [SWAPS.i8]`, where the optional file contains two signed bytes `(dx,dy)` per sequence entry.

### RingingOvershoot contract

`ringing_overshoot_u8` is a deterministic Gaussian high-boost filter. For each channel it computes the normalized finite-square Gaussian blur `B` with `w(dx,dy)=exp(-(dx^2+dy^2)/(2*sigma^2))` and `REFLECT_101` borders, then emits `clip(round(I + amount*(I-B)), 0, 255)`. The odd `kernel_size` is limited to 15, `sigma` must be positive, and `amount` is finite in `[0,4]`. The residual deliberately creates bright and dark edge overshoot (ringing) without random sampling; repeated calls with the same input and configuration are identical. The CLI form is `ringing_overshoot IN.raw OUT.raw W H C KERNEL_SIZE SIGMA AMOUNT`.

### ZoomBlur contract

`zoom_blur_u8` averages `steps+1` equally weighted, center-preserving bilinear samples. Its factors are the explicit deterministic sequence `min_factor + i*(max_factor-min_factor)/steps`, for `i=0..steps`; coordinates outside the image use `REFLECT_101`. `min_factor` must be at least 1, and `steps` is limited to 256. This explicit interval and step count replaces the randomized factor range and implicit sampling used by Albumentations/imgaug, so repeated calls are identical without a seed. The CLI form is `zoom_blur IN.raw OUT.raw W H C MIN_FACTOR MAX_FACTOR STEPS`.

### Optical catalog artifacts

The NOISE_CATALOG optical entries expose distinct `optical_defocus_u8`,
`optical_motion_blur_u8`, `optical_zoom_blur_u8`,
`optical_chromatic_aberration_u8`, `lateral_chromatic_aberration_u8`,
`longitudinal_chromatic_aberration_u8`, `thin_prism_distortion_u8`, and
`rolling_shutter_geometric_distortion_u8` APIs. They preserve contiguous HWC
`uint8` buffers, use explicit clipping/fill and interpolation, and keep borrowed
H-element row maps in host memory for CPU or device memory for CUDA. The
rolling-shutter geometric operation is an H-W-C still-image remap, not the
existing T-H-W-C video rolling-shutter operation. Thin-prism and chromatic
operations are normalized polynomial approximations; longitudinal aberration
uses independent finite RGB Gaussian supports. The CLI forms and approximation
limits are documented in `NOISE_CATALOG.md`.

### Optical motion artifact contracts

The optical motion APIs use contiguous HWC uint8 buffers, REFLECT_101 borders,
nearest coordinate sampling, equal sample weights, and saturating half-up
rounding. `depth_dependent_defocus_u8` borrows an HxW float32 depth field and
uses `round(abs(depth-focus_depth)*blur_scale)` as a clamped disk radius.
`camera_shake_blur_u8` borrows a deterministic sequence of pixel translations;
`linear_directional_blur_u8` samples a centered segment; and
`rotational_motion_blur_u8` samples a centered angular interval around an
explicit pixel center. `rolling_shutter_motion_blur_u8` borrows H-element row
translation fields and samples each row over its -0.5..+0.5 readout interval.
CPU fields are host-owned; CUDA fields are device-owned until stream completion.
These are documented deterministic surrogates, not continuous optical exposure
integrals. CLI forms are listed in `NOISE_CATALOG.md`.

### Focus breathing contract

`focus_breathing_u8` applies the explicit frame-dependent radial magnification
`scale=1+focus_position*(breathing_strength+radial_strength*r^2)` about the
normalized centre. It inverse-maps each output pixel using Nearest or Linear
sampling and fills/clips out-of-range values with the geometric API contract.
Zero focus position is an exact identity. This is a deterministic radial scale
approximation, not a calibrated lens or depth renderer; local nonpositive scales
are filled. The CLI form is documented in `NOISE_CATALOG.md`.

### RandomToneCurve contract

`random_tone_curve_u8` applies a borrowed, explicit channel-major LUT. The table contains exactly `channels * 256` bytes, with entry `lut[channel * 256 + input]` selecting the output byte for that channel and input value. The CPU API expects host LUT storage; the CUDA API expects device LUT storage. A null LUT selects the deterministic built-in curve generated from `seed`, with fixed monotone control points at inputs 0, 64, 128, 192, and 255. For zero-based channel `c` and interior control point `j`, the control value is `max(previous, min(255, 64*j + (SplitMix64(seed + 4*c + j) mod 65) - 32))`; each input is integer linearly interpolated between adjacent controls, with half values rounded up. Supplying a LUT ignores `seed`, so this path is seed-free and exact across repeated calls. The CLI form is `tone random_tone_curve IN.raw OUT.raw W H C LUT.raw [SEED]`; use `-` instead of `LUT.raw` to use the generated curve and provide an optional seed.

### SigmoidContrast and LogContrast contract

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

### Color and photometric tone variations

The tone variation APIs operate on contiguous interleaved HWC `uint8` data with explicit deterministic configuration. `gamma_variation_u8` evaluates `x^gamma`; `tone_curve_variation_u8` consumes a borrowed channel-major LUT with exactly `channels*256` entries. `s_curve_contrast_variation_u8` uses the bounded cubic S-curve `x+amount*4*x*(1-x)*(2*x-1)`. `highlight_rolloff_variation_u8` uses a thresholded rational shoulder, while `shadow_lift_u8` and `shadow_crush_u8` apply a threshold-window lift or attenuation. Normalized results are half-up rounded and clipped to `[0,255]`; this clipping can conceal an extreme curve. CPU pointers are host-owned and CUDA pointers are device-owned until stream completion. These are deterministic approximations, not camera-profile calibration.

`posterization_u8` and `low_bit_depth_banding_u8` use the existing floor quantizer at `2^bits` levels, with `bits` in `[1,8]` and no dither. Thus visible steps are intentional. CLI forms are `tone gamma_variation IN.raw OUT.raw W H C GAMMA`, `tone tone_curve_variation IN.raw OUT.raw W H C LUT.raw`, `tone s_curve_contrast_variation IN.raw OUT.raw W H C AMOUNT`, `tone highlight_rolloff_variation IN.raw OUT.raw W H C THRESHOLD STRENGTH`, `tone shadow_lift IN.raw OUT.raw W H C AMOUNT THRESHOLD`, `tone shadow_crush IN.raw OUT.raw W H C AMOUNT THRESHOLD`, `tone posterization IN.raw OUT.raw W H C BITS`, and `tone low_bit_depth_banding IN.raw OUT.raw W H C BITS`.

### Chroma subsampling and local tone-mapping noise

`chroma_subsampling_artifacts_u8` uses the existing full-range BT.601 surrogate (`Y=.299R+.587G+.114B`, `Cb=B-Y`, `Cr=R-Y`), not studio-range offsets, transfer-function conversion, or ICC color management. `Y444` copies exactly; `Y422` averages each horizontal chroma pair; `Y420` averages each clipped 2x2 chroma block and replicates it. Luma is retained, RGB is reconstructed with saturating half-up rounding, and alpha/channels after RGB are copied. The enum is shared explicitly as `ChromaSubsampling::{Y444,Y422,Y420}`. CPU data is host-owned and CUDA data is device-owned. Use `chroma_subsampling_artifacts IN.raw OUT.raw W H C 444|422|420`.

`local_tone_mapping_noise_u8` averages BT.601 Y in a clamped square neighborhood and applies `Y' = Y + tone_strength*(mean-Y) + noise_stddev*N(0,1)`. Radius is `[0,32]`, tone strength is `[0,1]`, noise is normalized Y units, and the coordinate-keyed seed is deterministic across CPU/CUDA. Cb/Cr and channels after RGB are preserved. Use `local_tone_mapping_noise IN.raw OUT.raw W H C RADIUS TONE_STRENGTH NOISE_STDDEV SEED`.

### ColorJitter contract

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

### Deterministic imgaug colorspace meta contracts

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

### Deterministic imgaug HSV color batch contract

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

### RandomColorJitter contract

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

### PlanckianJitter contract

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

### ChangeColorTemperature contract

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

### Uniform color quantization contract

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

### ChromaticAberration contract

`chromatic_aberration_u8` applies deterministic per-channel radial remaps to the first three channels of contiguous HWC `uint8` data. For channel `k`, normalized coordinates use `cx=(width-1)/2`, `cy=(height-1)/2`, `fx=max(1,width/2)`, and `fy=max(1,height/2)`. The source map is `x'=cx+xn*(1+radial_k*r2)*fx`, `y'=cy+yn*(1+radial_k*r2)*fy`, where `r2=xn*xn+yn*yn`. Each map uses bilinear interpolation with constant `fill` outside the image, then emits `clip(floor(gain_k*sample+0.5),0,255)`. Channels after RGB are copied unchanged. The CLI form is `color chromatic_aberration IN.raw OUT.raw W H C RADIAL0 RADIAL1 RADIAL2 GAIN0 GAIN1 GAIN2 FILL`; the top-level `chromatic_aberration` spelling is also accepted.

This is a deterministic lateral-dispersion and photometric-gain approximation, not a spectral or wavelength-dependent lens model. It does not model longitudinal focus shifts, wavelength-specific point-spread functions, vignetting, or upstream random parameter sampling. CPU output is compared with fixed OpenCV `remap` maps; CUDA uses the same formula and interpolation contract, with small floating-point rounding differences possible.

### FancyPCA contract

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

### PlasmaContrast contract

`plasma_contrast_u8` applies a deterministic spatial contrast field to contiguous HWC `uint8` data. For each pixel, the local factor is `contrast * (1 + plasma_strength * (2 * field - 1))`, where `contrast >= 0`, `plasma_strength` is in `[0,1]`, and `field` is in `[0,1]`. Values are scaled around 127.5, rounded with `floor(value + 0.5)`, and clipped to uint8. The explicit field contains exactly `width*height` floats in row-major order and is the preferred CPU/host or CUDA/device input when an external reference must be reproduced. With a null field, CPU and CUDA derive the same four-scale integer-hash field from `seed`. This is a deterministic plasma/noise approximation, not a claim of parity with any implementation-specific stochastic plasma sampler. Channels are modulated independently by the same spatial field. The CLI form is `color plasma_contrast IN.raw OUT.raw W H C CONTRAST STRENGTH SEED [FIELD.f32]`.

### PlasmaBrightnessContrast contract

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

### PlasmaShadow contract

`plasma_shadow_u8` applies deterministic spatial shadows to contiguous HWC `uint8`
data. For a field value below `threshold`, the normalized shadow amount is
`(threshold - field) / threshold`; values at or above the threshold are unchanged.
Each channel is multiplied by `1 - strength * shadow`, rounded with
`floor(value + 0.5)`, and clipped to `[0,255]`. `threshold` and `strength` are
finite values in `[0,1]`; a zero threshold disables shadows. The explicit field
contains exactly `width*height` finite floats in `[0,1]` in row-major order, while
a null field uses the shared four-scale hash generated from `seed` on CPU and
CUDA. The CLI form is `color plasma_shadow IN.raw OUT.raw W H C THRESHOLD
STRENGTH SEED [FIELD.f32]`.

### RandomRain contract

`random_rain_u8` accepts contiguous HWC `uint8` data and overlays white rain on
pixel-centre samples. A `RainStreak` is an explicit `(x0,y0,x1,y1,width,alpha)`
segment in image pixels. Supplying `RandomRainConfig::streaks` makes placement
fully explicit; a null list generates the requested count from
`SplitMix64(seed + 6*index + offset)` and the configured length, angle, and width
ranges. `make_random_rain_streaks` materializes that generated list for logging
or exact CPU/CUDA replay, so stochastic-looking placement is never hidden.
For each covered pixel and each channel, segments are composited in list order as
`round((1-a)*current + a*255)`, where `a=clamp(config.alpha*streak.alpha,0,1)`;
overlaps are sequential and alpha zero is identity. CPU streak lists are host
memory and CUDA streak lists are device memory. The CLI form is
`weather random_rain IN.raw OUT.raw W H C COUNT ALPHA SEED [LENGTH_MIN LENGTH_MAX ANGLE_MIN ANGLE_MAX WIDTH [STREAKS.bin]]`;
the direct `random_rain` spelling is also accepted. A binary streak list contains
`COUNT` little-endian records of six float32 values.

### RandomSnow contract

`random_snow_u8` accepts contiguous HWC `uint8` data and overlays white circular
snowflakes on pixel-centre samples. A `Snowflake` is an explicit
`(x,y,radius,alpha)` record in image pixels. Supplying
`RandomSnowConfig::snowflakes` makes placement fully explicit; a null list
generates the requested count from `SplitMix64(seed + 4*index + offset)` with
uniform position and radius in the configured range. `make_random_snowflakes`
materializes generated records for logging or exact CPU/CUDA replay. For each
covered pixel and channel, records are composited in list order as
`round((1-a)*current + a*255)`, where `a=clamp(config.alpha*snowflake.alpha,0,1)`;
overlaps are sequential, alpha zero is identity, and all channels receive the
same blend. CPU records are host memory and CUDA records are device memory. The
CLI form is `weather random_snow IN.raw OUT.raw W H C COUNT ALPHA SEED [RADIUS_MIN RADIUS_MAX [SNOWFLAKES.bin]]`; the direct `random_snow` spelling is also accepted. A binary snowflake list contains `COUNT` little-endian records of four float32 values.

### SnowStamp contract

`snow_stamp_u8` overlays one explicit stamp image and mask at an ordered list of
integer top-left placements. The stamp image is row-major HWC `uint8` with one
channel (broadcast) or exactly the output channel count; a null image uses the
scalar `fill`. The optional row-major one-channel mask contains opacity bytes
(0..255), with a null mask treated as fully opaque. Each placement is a binary
12-byte little-endian record `(int32 x, int32 y, float32 alpha)`. For each
covered stamp pixel and output channel, in placement order, the result is
`round((1-a)*current+a*fill)`, where
`a=clamp(global_alpha*placement.alpha*mask/255,0,1)`; image values supply
`fill` when present. Stamps are clipped at image boundaries, overlaps are
sequential, and there is no random state or interpolation. CPU arrays are host
memory and CUDA arrays are device memory. The CLI form is
`weather snow_stamp IN.raw OUT.raw W H C STAMP_W STAMP_H COUNT ALPHA FILL IMAGE.raw MASK.raw PLACEMENTS.bin`;
use `-` for IMAGE or MASK to select the null behavior. The direct `snow_stamp`
spelling is also accepted.

### RandomGravel contract

`random_gravel_u8` accepts contiguous HWC `uint8` data and overlays filled gravel
particles at pixel-centre samples. A `GravelParticle` is an explicit
`(x,y,radius,alpha,fill)` record; `fill` is one scalar uint8 value applied to every
channel. Supplying `RandomGravelConfig::particles` makes placement, radius, alpha,
and fill fully explicit. A null list generates the requested count from
`SplitMix64(seed + 4*index + offset)` with uniform position and configured radius
range, using `config.fill` and local alpha one. `make_random_gravel_particles`
materializes generated records for logging or exact CPU/CUDA replay. For each
covered pixel and channel, records are composited in list order as
`round((1-a)*current + a*fill)`, where `a=clamp(config.alpha*particle.alpha,0,1)`;
overlaps are sequential, values are clipped to `[0,255]`, and zero alpha or an
empty list is identity. CPU particle lists are host memory and CUDA lists are
device memory. The CLI form is
`weather random_gravel IN.raw OUT.raw W H C COUNT ALPHA SEED [RADIUS_MIN RADIUS_MAX FILL [PARTICLES.bin]]`;
the direct spelling is also accepted. A binary particle list contains `COUNT`
records of four little-endian float32 values followed by uint8 fill and three
padding bytes.

### Spatter contract

`random_spatter_u8` (also exposed as `spatter_u8`) accepts contiguous HWC `uint8`
data and overlays colored filled discs at pixel-centre samples. A
`SpatterDroplet` is an explicit `(x,y,radius,red,green,blue,alpha)` record;
its RGB color is blended into the first three channels and channels beyond RGB
are copied unchanged. Supplying `RandomSpatterConfig::droplets` makes
placement, radius, color, and local alpha fully explicit. A null list generates
the requested count from `SplitMix64(seed + 4*index + offset)` with uniform
position and configured radius range, using the configured RGB color and local
alpha one. `make_random_spatter_droplets` materializes generated records for
logging or exact CPU/CUDA replay. For covered RGB channels, records are
composited in list order as `round((1-a)*current+a*color)`, where
`a=clamp(config.alpha*droplet.alpha,0,1)`; results are clipped to `[0,255]` and
alpha zero or an empty list is identity. CPU records are host memory and CUDA
records are device memory. The CLI form is
`weather spatter IN.raw OUT.raw W H C COUNT ALPHA SEED [RADIUS_MIN RADIUS_MAX RED GREEN BLUE [DROPLETS.bin]]`;
`random_spatter` and the direct `spatter` spelling are also accepted. A binary
droplet list contains `COUNT` little-endian records of three float32 values,
three color bytes, one padding byte, and one float32 alpha (20 bytes).

### RandomSunFlare contract

`random_sun_flare_u8` accepts contiguous HWC `uint8` data and overlays a white
source disc followed by radial ray capsules. `RandomSunFlareConfig::source` is
an explicit `(x,y,radius,alpha)` source in image pixels. A non-null `rays` list
contains explicit `(angle,length,width,alpha)` records, with angle in degrees
from +x; a null list generates `ray_count` records from
`SplitMix64(seed + 4*index + offset)` and the configured ranges. The generated
records can be materialized with `make_random_sun_flare_rays`. Pixel centres
use hard disc/capsule tests. The source is blended first, then rays in list
order, using `round((1-a)*current + a*255)` with
`a=clamp(opacity*local_alpha,0,1)` for every channel. CPU ray lists are host
memory and CUDA ray lists are device memory. The CLI form is
`weather random_sun_flare IN.raw OUT.raw W H C COUNT OPACITY SEED [SOURCE_X SOURCE_Y SOURCE_RADIUS SOURCE_ALPHA [LENGTH_MIN LENGTH_MAX ANGLE_MIN ANGLE_MAX WIDTH_MIN WIDTH_MAX [RAYS.bin]]]`; the direct spelling is also accepted. A binary ray list contains `COUNT` little-endian records of four float32 values.

### RandomShadow contract

`random_shadow_u8` applies deterministic explicit shadow masks to contiguous HWC
`uint8` data. A `ShadowPolygon` contains a borrowed point array and local alpha;
a `ShadowRectangle` contains two image-pixel corners and local alpha. Pixel
centres `(x+0.5,y+0.5)` are tested with an even-odd polygon rule (boundary
included) and inclusive normalized rectangle bounds. Polygon records are
composited first, followed by rectangles in list order. For every covered
channel, the result is `round((1-a)*current + a*fill)`, where
`a=clamp(opacity*record.alpha,0,1)` and the scalar `fill` is applied to every
channel. Values are clipped to `[0,255]`; empty lists and zero opacity are
identity. The seed is retained for API symmetry but does not affect explicit
masks. CPU mask and point arrays are host memory; CUDA arrays and point arrays
are device memory. The CLI form is
`weather random_shadow IN.raw OUT.raw W H C OPACITY FILL POLYGONS.bin RECTANGLES.bin`;
the direct spelling is also accepted. Use `-` for either empty file. A polygon
file contains repeated little-endian records `<uint32 point_count, float alpha>`
followed by `point_count` `<float x,float y>` pairs. A rectangle file contains
repeated little-endian `<float x0,y0,x1,y1,alpha>` records.

### RandomFog contract

`random_fog_u8` accepts contiguous HWC `uint8` data and applies one white,
depth-independent veil per pixel. A non-null `RandomFogConfig::field` contains
exactly `width*height` row-major float32 values in `[0,1]`; each value is the
local fog density. A null field uses `SplitMix64(seed + pixel)` on both CPU and
CUDA, and `make_random_fog_field` materializes that deterministic field. For
every channel, the result is
`round((1-a)*input + a*255)`, with
`a=clamp(config.opacity*config.density*field[pixel],0,1)`, followed by clipping
to `[0,255]`. Density and opacity are finite values in `[0,1]`; zero density,
zero opacity, and an all-zero field are identity. The same veil is applied to
all channels and no depth input is accepted. CPU fields are host memory and
CUDA fields are device memory. The CLI form is
`weather random_fog IN.raw OUT.raw W H C DENSITY OPACITY SEED [FIELD.f32]`; the
direct `random_fog` spelling is also accepted. A field file contains
`width*height` little-endian float32 values.

### Catalog weather contracts

The eight `augmenters.weather` entries are available through
`augmatch/weather.hpp` and the umbrella header. `FastSnowyLandscape` measures
mean RGB brightness and blends bright pixels toward white above `snow_point`;
this is a deterministic brightness-threshold approximation, not a semantic
landscape model. `Clouds` uses a deterministic four-counter SplitMix64 field
(or a caller-supplied HxW field) and white-composites it. `Fog` is the existing
seeded depth-independent white veil under the catalog spelling.

`Snowflakes` and `Rain` expose the explicit `Snowflake` and `RainStreak`
records and their seeded materializers under catalog aliases. Records are
processed in list order, overlaps are sequential, and zero alpha or an empty
list is identity. `CloudLayer`, `SnowflakesLayer`, and `RainLayer` each accept
a borrowed row-major HxW float32 map in `[0,1]`; the map is never retained or
resized and CUDA maps must be device allocations. Their target is a borrowed
contiguous HWC uint8 image, and every channel receives
`round((1-a)*input+a*255)` with `a=clamp(opacity*map,0,1)`.

CLI examples are `weather fast_snowy_landscape IN.raw OUT.raw W H C
SNOW_POINT ALPHA`, `weather clouds IN.raw OUT.raw W H C ALPHA DENSITY SEED`,
`weather fog IN.raw OUT.raw W H C DENSITY OPACITY SEED`, and
`weather {cloud_layer|snowflakes_layer|rain_layer} IN.raw OUT.raw W H C
OPACITY LAYER.f32`. `weather snowflakes` and `weather rain` use the same
record formats and seeded arguments as `random_snow` and `random_rain`.
These approximations intentionally operate only on uint8 HWC images.

### PRNU and fixed-pattern offset contracts

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

### Row-column correlated sensor noise

`row_column_correlated_noise_f32` adds one offset shared by every pixel in a row and one offset shared by every pixel in a column, inducing fixed spatial correlation while retaining channel-specific maps. A supplied `row_map` contains exactly `height*channels` normalized offsets and a supplied `column_map` contains `width*channels`; both pointers are borrowed. Null maps are generated deterministically from `seed`. CPU maps must be host allocations; CUDA maps must be device allocations and remain valid until the stream completes. The output is clipped to `[clip_min,clip_max]` (the default is `[0,1]`). The CLI is `row_column_correlated_noise IN.f32 OUT.f32 W H C ROW_STDDEV COLUMN_STDDEV SEED [ROW_MAP.f32 COLUMN_MAP.f32]`; `row_column_noise` is an alias.

### Clustered defective pixels

`clustered_defective_pixels_f32` applies circular dead, hot, and stuck-pixel defects to normalized HWC float data. Explicit `ClusteredDefect` records are borrowed and use pixel-coordinate centers, inclusive radii, a channel (`-1` means all channels), and a normalized value. Hot values are additive; stuck values replace the signal; dead pixels become zero. Results are clipped to `[0,1]`. If no records are supplied, `cluster_count` deterministic clusters are generated from `seed` with `cluster_radius`; generated clusters choose their type deterministically and use `hot_value` or `stuck_value`. CUDA record arrays must be device memory and remain valid until the stream completes, while CPU records must be host memory. The CLI is `clustered_defective_pixels IN.f32 OUT.f32 W H C CLUSTER_COUNT RADIUS HOT_VALUE STUCK_VALUE SEED [RECORDS.bin]`; `clustered_defects` is an alias. A record fixture is a packed array of `ClusteredDefect` values.

### Blooming and vertical smear

`blooming_vertical_smear_f32` accepts normalized HWC float32 data. Explicit borrowed `BrightPixel` records `(x,y,value,channel)` are deterministic sources; `channel=-1` applies to all channels. Each source adds `max(value-threshold,0) * strength * decay^(dy)` down its column, and output is clipped. With no records, input samples above the threshold become sources. An optional borrowed `column_smear_map` has `width*channels` row-zero amplitudes and is decayed down each column. CPU records/maps are host memory; CUDA records/maps are device memory and remain valid through the stream. The CLI is `blooming_vertical_smear IN.f32 OUT.f32 W H C THRESHOLD STRENGTH DECAY SEED [BRIGHT_PIXELS.bin [COLUMN_MAP.f32]]`.

### Sensor dust and opaque-pixel masks

`sensor_dust_opaque_mask_f32` uses a borrowed row-major HxW uint8 mask on normalized HWC float data. Zero copies exactly. Nonzero pixels receive `fill` when `blur_radius` is zero, or the mean of unmasked samples in the clipped square neighborhood when the radius is positive; no available neighbor falls back to `fill`. Masked results are clipped to `[0,1]`; unmasked values are preserved. A null mask is identity. CPU masks are host memory and CUDA masks are device memory valid through the stream. The CLI is `sensor_dust_opaque_mask IN.f32 OUT.f32 W H C BLUR_RADIUS FILL MASK.raw`.

### Discrete aperture PSF contracts

The four uint8 aperture APIs in `filter.hpp` use finite, normalized HWC PSFs, `REFLECT_101` borders, and saturating half-up rounding. Radius is an integer in `[0,32]`; zero copies exactly. Diffraction uses the Airy-inspired `sinc^2` disk surrogate; wavelength/aperture/focal metadata scale its lobe with a bounded dimensionless ratio. Bokeh uses a disk with optional linear edge softness. Cat-eye bokeh shrinks the horizontal ellipse toward the explicit normalized image boundary. Aperture-shape blur rasterizes a regular 3--32 blade polygon, with explicit rotation and polygon-to-circle roundness. These are deterministic discrete approximations, not calibrated wave-optics or ray-traced lens models; finite support and uint8 clipping are intentional. CPU buffers are host-owned and CUDA buffers are device-owned until the stream completes. CLI forms are documented in `NOISE_CATALOG.md`.

### Optical artifact contracts

The radial optical APIs (`lens_vignetting_f32`, `color_dependent_vignetting_f32`, and `optical_falloff_f32`) use normalized coordinates and a cubic polynomial in squared radius. Optional scalar HxW or channel-specific HWC gain maps are borrowed, not copied. `lens_shading_f32` and `uneven_illumination_f32` reuse that map layout; `sensor_lens_dust_shadows_f32` uses an HxW opacity map. CPU maps are host memory and CUDA maps are device memory valid through the stream. All six APIs clip normalized float output to the configured interval. The models are deterministic calibration/augmentation approximations, not physical pupil, spectral, or dust-transport simulations. CLI forms and map fixture sizes are documented in `NOISE_CATALOG.md`.

### ISP artifact contracts

The thirteen `isp_artifacts.hpp` operations use contiguous HWC uint8 buffers with clamped borders, float arithmetic, and saturating half-up rounding to `[0,255]`. CPU pointers are host-owned; CUDA pointers are device-owned for the supplied stream. These deterministic formulas are native ISP surrogates, not vendor-specific reference pipelines. The APIs and CLI forms are documented in `NOISE_CATALOG.md`.

`edge_oversharpening_u8`, `unsharp_mask_halos_u8`, and `laplacian_halos_u8` add signed 3x3, Gaussian, and four-neighbour Laplacian residuals. `ringing_near_strong_edges_u8` gates a checkerboard-sign Laplacian on a centred gradient and applies an explicit residual bound. Local contrast uses a mean/variance residual; tone-mapping haloing and local sharpening use Gaussian high-pass residuals, with explicit tone/noise gains. High-frequency attenuation and detail smearing blend deterministic box/Gaussian low-pass results. Overshoot/undershoot applies a bounded signed Gaussian high-boost residual with independent positive and negative limits. Clipping is intentional and may hide over-correction at saturated pixels. The Laplacian, Sobel, and high-pass residual injection APIs accept borrowed signed float32 HxW or HxWxC maps (`map_channels=1` or `channels`); CPU maps are host-owned and CUDA maps remain device-owned through the supplied stream. Null maps derive the named residual, and all results clip to uint8 `[0,255]`.

### Bayer and CFA sampling contracts

`quad_bayer_sample_u8` samples HWC RGB8 input with an explicit 4x4 expanded Bayer pattern (`QuadBayerPattern`) and writes one raw HxW plane. `rgbw_sample_u8` samples HWC RGBW8 input with channel ownership R=0, G=1, B=2, W=3 and an explicit 2x2 `RGBWPattern`; it also writes one raw plane. These functions only sample and never demosaic. `custom_cfa_sample_u8` takes HWC input plus a borrowed repeating byte mask. Each mask byte is an input channel index, and values outside `[0, channels)` are invalid. CPU masks are host-owned; CUDA masks are device-owned until stream completion. The CLI forms are `quad_bayer IN.raw OUT.raw W H PATTERN`, `rgbw IN.raw OUT.raw W H PATTERN`, and `custom_cfa IN.raw OUT.raw W H CHANNELS MASK_W MASK_H MASK.raw`. The existing Bayer API and all three new CFA APIs are included by `augmatch/augmatch.hpp`.

### Advanced Bayer demosaicing contracts

`bayer_demosaic_malvar_he_cutler_u8` implements the Malvar-He-Cutler 5x5 filters. The green-at-red/blue filter has weights `[[0,0,-1,0,0],[0,0,2,0,0],[-1,2,4,2,-1],[0,0,2,0,0],[0,0,-1,0,0]]/8`; the same-color-at-green filter uses center weight 5 and the diagonal filter uses center weight 6, with the published 0.5, -1, 4 and -1.5, 2 coefficients. All filters use clamp-to-edge coordinates, then clip and round to uint8. `bayer_demosaic_edge_aware_u8` compares clamped horizontal/vertical raw gradients at green sites and the two diagonal gradients at red/blue sites. It selects the lower-gradient direction, averages both on a tie, and searches distances one then two for the target-color pair. Both APIs consume a contiguous HxW raw plane and explicitly select `RGGB`, `BGGR`, `GRBG`, or `GBRG` through `BayerConfig`; the eight `*_rggb_u8`, `*_bggr_u8`, `*_grbg_u8`, and `*_gbrg_u8` convenience APIs select a tile directly. CLI forms are `demosaic_malvar IN.raw OUT.raw W H PATTERN` and `demosaic_edge_aware IN.raw OUT.raw W H PATTERN`.

### Demosaicing artifact contracts

`bayer_directional_demosaicing_artifacts_u8`, `bayer_false_color_zipper_artifacts_u8`, `bayer_demosaicing_aliasing_u8`, `bayer_demosaicing_ringing_u8`, and `bayer_demosaicing_noise_amplification_u8` consume HxW raw Bayer `uint8` and produce HxWx3 RGB8. They use edge-aware demosaicing at zero strength, preserve known CFA samples where applicable, and saturate to `clip_min..clip_max` before uint8 rounding. Directional injection selects gradient, horizontal, or vertical interpolation; zipper injection alternates red/blue by the horizontal raw gradient; aliasing injects an explicit period/phase sinusoid; ringing uses a signed four-neighbor Laplacian; noise amplification uses coordinate-keyed Gaussian noise scaled by local gradient. CPU uses host buffers and CUDA uses device buffers valid through the stream. The formulas are deterministic native approximations, not upstream camera/ISP parity. CLI forms are `demosaic_directional IN.raw OUT.raw W H PATTERN DIRECTION STRENGTH`, `demosaic_zipper IN.raw OUT.raw W H PATTERN STRENGTH`, `demosaic_aliasing IN.raw OUT.raw W H PATTERN PERIOD PHASE STRENGTH`, `demosaic_ringing IN.raw OUT.raw W H PATTERN STRENGTH`, and `demosaic_noise_amplification IN.raw OUT.raw W H PATTERN NOISE_STDDEV AMPLIFICATION SEED`.

### CFA raw-plane effect contracts

The six CFA effect APIs consume and produce contiguous HxW float32 raw planes. `CfaPlaneMap` explicitly owns the repeating 2x2 tile and uses plane ownership 0=R, 1=Gr, 2=Gb, 3=B; call `bayer_plane_map` for canonical RGGB/BGGR/GRBG/GBRG maps. Maps are copied by value. A `missing_mask` is borrowed row-major HxW memory: host-owned on CPU and device-owned until CUDA stream completion. Response variation, leakage, integer misregistration, missing samples, plane noise, and plane gain all validate parameters and clip to the configured range. Leakage uses a row-major target/source 4x4 matrix with deterministic radius-two nearest-plane lookup; missing samples support zero, fill, and nearest same-plane replacement. Seeded operations use coordinate-stable counter keys, so repeated calls with the same seed are deterministic. Raw planes have no retained channel metadata. CLI forms are `cfa_channel_response IN.f32 OUT.f32 W H R GR GB B`, `cfa_leakage IN.f32 OUT.f32 W H MATRIX16`, `cfa_misregistration IN.f32 OUT.f32 W H DX4 DY4`, `cfa_missing_samples IN.f32 OUT.f32 W H PROBABILITY REPLACEMENT FILL SEED`, `bayer_plane_noise IN.f32 OUT.f32 W H SIGMA4`, and `bayer_plane_gain IN.f32 OUT.f32 W H GAIN4`.

### ISONoise contract

Color/photometric APIs (`color_matrix_perturbation_u8`, `camera_color_profile_variation_u8`, `rgb_channel_cross_talk_u8`, `sensor_spectral_response_variation_u8`, `color_clipping_u8`, `white_balance_clipping_u8`, `chroma_noise_u8`, `luma_noise_u8`, and `correlated_luma_chroma_noise_u8`) accept interleaved HWC uint8 RGB(A). They use normalized full-range RGB with BT.601 Y/Cb/Cr for luma/chroma operations, preserve channels >= 3, and saturate half-up to uint8. Matrix coefficients are row-major; profile and spectral noise are seeded normalized Gaussian perturbations. CPU pointers are host memory and CUDA pointers are device memory valid through the stream.

`iso_noise_u8` accepts interleaved HWC `uint8` data and applies the explicit gain `G = (iso / base_iso) * analog_gain`. The signal is first multiplied by `digital_gain`. For each pixel, the seeded SplitMix64 counter/hash RNG derives one shared standard-normal luma sample and one sample per channel. Gaussian luma noise is `gaussian_stddev * G * Z_luma`. For RGB data, chroma noise is luma-preserving: `chroma_stddev * G * (Z_channel - 0.2126 Z_R - 0.7152 Z_G - 0.0722 Z_B)`. Chroma noise is disabled for fewer than three channels. The result is clipped to `[0,1]` and rounded to `uint8`; no global RNG state is used. CPU and CUDA use the same coordinate/channel hash and the CLI accepts `iso_noise IN.raw OUT.raw W H C ISO BASE_ISO ANALOG_GAIN DIGITAL_GAIN GAUSSIAN_STDDEV CHROMA_STDDEV SEED` (the `isonoise` alias is also accepted).

### ShotNoise contract

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

### MaskDropout contract

`mask_dropout_u8` takes an HWC `uint8` image and a separate one-byte-per-pixel HxW mask. A zero mask byte keeps all channels unchanged; every nonzero byte drops the pixel and writes the scalar `MaskDropoutConfig::fill` to every channel. The mask is read in row-major order, is never modified, and the operation has no random state, so CPU and CUDA outputs are exact for identical inputs, masks, and configuration. The mask may use either `1` or conventional `255` foreground values. The CLI form is `dropout mask IN.raw OUT.raw MASK.raw W H C FILL`.

### XYMasking contract

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

### Cutout and TotalDropout contracts

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

### NonLocalMeansDenoising contract

`non_local_means_denoising_u8` uses a deterministic, bounded square search. `NonLocalMeansDenoisingConfig::patch_radius` is the complete patch radius and is limited to `[0,4]`; `search_radius` is the candidate radius and is limited to `[0,8]`. Every candidate in the inclusive square window is evaluated, including the center candidate. Its weight is `exp(-d/(h*h))`, where `d` is the mean squared difference over the complete patch and all channels, and `h` must be finite and positive. Patches and candidate pixels use `REFLECT_101` borders. The normalized weighted result is independently rounded to nearest uint8 per channel, with nonnegative half values rounding up. No random state or implicit search expansion is used, so CPU and CUDA calls are deterministic for the same input and configuration. The CLI form is `non_local_means IN.raw OUT.raw W H C PATCH_RADIUS SEARCH_RADIUS H` (the alias `non_local_means_denoising` is also accepted).

### CLAHE contract

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

### All-channel contrast aliases

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

### Dithering contract

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

## Target metadata formats

`target_metadata.hpp` adds host CPU behavior for additional image and mask target
view pairs and for common annotation formats. Additional targets borrow strided
`uint8` HWC/CHW views and apply one shared `Geometry`; masks use an explicit
nearest/linear policy (nearest by default). The library does not own or transfer
these buffers, including in CUDA builds.

`BoxXYXY` is absolute Pascal VOC half-open edge geometry. `COCOBox` stores
absolute `x,y,width,height`; `YOLOBox` stores normalized center `x,y,width,height`;
and `AlbumentationsBox` stores normalized `x1,y1,x2,y2`. Normalized values are
fractions of image dimensions and are validated in `[0,1]`. Conversion and
transform helpers clip output boxes to the output image. `KeypointXYV` uses
continuous pixel-center coordinates and visibility in `[0,1]`; its transform
marks an out-of-frame point invisible by default. These APIs are host metadata
operations rather than CUDA kernels.

## Geometry targets

`Geometry` uses top-left coordinates and applies crop, horizontal flip, then
vertical flip. Image samples are pixel centres at integer `(x,y)` indices.
`BoxXYXY` is continuous Pascal-VOC edge geometry `(x1,y1,x2,y2)` with a
half-open extent; flips therefore map edges with `width-x2,width-x1`, while
`PointXY` centres map with `width-1-x`. `transform_points` and
`transform_boxes` transform host-owned arrays without changing their order.

`transform_boxes_filtered` optionally clips transformed boxes to the output
image and compacts boxes that are invalid, smaller than `min_area`, or below
`min_visibility` (retained area divided by source area). `filter_boxes` filters
already-transformed boxes by area. Metadata helpers run on host memory in both
CPU and CUDA builds; they do not copy device metadata or labels. The optional
`output_indices` array from `transform_boxes_filtered` identifies retained
source records for compacting parallel labels.

`transform_mask_u8` exposes `MaskInterpolation::Nearest` and `Linear`. Masks
and segmentation labels should use nearest. The current `Geometry` contract
only crops and flips, so both policies sample exact pixel centres and produce
the same bytes; the explicit policy prevents accidental linear interpolation
when a future geometry adds scale. Image and mask buffers are host pointers in
CPU builds and device pointers in CUDA builds, with the supplied CUDA stream.

## Contract

Crop coordinates are top-left `(crop_x,crop_y)` with extent `(output_width,output_height)`. Operations run in crop, horizontal flip, vertical flip order. `transform_u8` and `pool_u8` accept device pointers and a CUDA stream in CUDA builds, and host pointers in CPU builds. `RandomCropFromBorders` accepts four maximum fractions in left, right, top, bottom order. Each limit is `floor(fraction * input_dimension)`. Config offsets default to `-1`, which selects `splitmix64(seed + side_index) % (limit + 1)` with side indices left=0, right=1, top=2, bottom=3. Nonnegative offsets are explicit and override the seed; they must not exceed their limits. The maximum allowed crop leaves at least one pixel per dimension, and no global RNG state is used. Its CLI form is `crop_from_borders IN.raw OUT.raw W H C LEFT_F RIGHT_F TOP_F BOTTOM_F SEED [LEFT RIGHT TOP BOTTOM]`. `RandomResizedCrop` and `RandomSizedCrop` share an explicit source rectangle and resize contract: `OP IN.raw OUT.raw W H C OUT_W OUT_H CROP_X CROP_Y CROP_W CROP_H INTERPOLATION [SEED]`, with interpolation `0=Nearest`, `1=Linear`, or `2=Area`. Set both crop dimensions to zero and an offset to `-1` to generate a rectangle using SplitMix64 seed offsets width=0, height=1, x=2, y=3; explicit coordinates are deterministic and identical on CPU/CUDA. This generation contract intentionally differs from hidden Albumentations sampling/rejection loops, while resize uses the existing `Resize` implementation exactly. `RandomCropNearBBox` takes `W H C X1 Y1 X2 Y2 MAX_SHIFT_X MAX_SHIFT_Y SEED [LEFT RIGHT TOP BOTTOM]` in its CLI form. The box is continuous Pascal VOC `(x1,y1,x2,y2)` image-edge coordinates; the crop support is the half-open pixel rectangle from floor/ceil box edges plus deterministic or explicit integer margins, clipped to the image. Its API returns this rectangle for output allocation and changes image pixels only. Box and other annotation transformation is explicit metadata owned by the caller, rather than hidden Python targets. Target-aware crops take a borrowed host `BoxXYXY` array: `BBoxSafeRandomCrop` contains the all-box envelope, `RandomSizedBBoxSafeCrop` applies that source selection then resizes, and `AtLeastOneBBoxRandomCrop` intersects one deterministic selected box. Empty arrays use their documented explicit or seeded fallback. Factories return source geometry (and scale for the resized form); callers own annotation transformation. The target-aware CLI adapters use a text box file with one `x1 y1 x2 y2` record per line. `RandomGridShuffle` uses row-major destination-to-source cell permutations; callers may provide a borrowed host/device permutation or leave it null for the documented deterministic SplitMix64 seed shuffle. Its CLI form is `random_grid_shuffle IN.raw OUT.raw W H C GRID_ROWS GRID_COLS SEED [PERMUTATION.u32]`, where the optional permutation file contains little-endian `uint32` source cell indices. The CLI is a parity-test adapter that copies raw images to and from the GPU.

### ISO and gain-dependent profile contract

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

## `augmenters.meta` target-aware operations

`meta.hpp` adds native replacements for Lambda, AssertLambda, AssertShape,
RemoveCBAsByOutOfImageFraction, and ClipCBAsToImagePlanes. `LambdaCallback`
and `AssertLambdaPredicate` are plain synchronous host function pointers, not
`std::function` objects: callback/predicate and `void*` context are borrowed,
owned by the caller, and must not be retained. A callback may mutate the image
and caller-owned box/keypoint arrays; a predicate receives const views. Null
callbacks, invalid views, and null arrays with nonzero counts throw
`std::invalid_argument`. Device image views are rejected explicitly. This is
intentional: there is no safe ABI for arbitrary host callbacks or predicates in
a CUDA kernel. Device pipelines must use `DeviceStage` launch functions.

`AssertShapeConfig` uses `-1` as a wildcard dimension and can independently
check datatype and layout. It never changes image bytes. Target metadata is
represented by borrowed `MutableTargetAnnotations`/`TargetAnnotations`; the
library never allocates, frees, or resizes those arrays.

`remove_cbas_by_out_of_image_fraction` compacts boxes and keypoints in place
and returns both retained counts. A box's out-of-image fraction is one minus
its clipped-area/original-area ratio; malformed boxes count as fully outside.
A keypoint is inside when its finite pixel centre is in `[0,width) x
[0,height)`, otherwise it has fraction one. Optional source-index arrays let a
caller compact labels in lockstep. The threshold is finite and in `[0,1]`.
`clip_cbas_to_image_planes` clips box edges to `[0,width] x [0,height]` and
keypoint centres to `[0,width-1] x [0,height-1]`; non-finite keypoints are an
error. Metadata operations are host-only in CPU and CUDA builds.

The CLI metadata forms are:

```text
augmatch_cli meta_clip BOXES_IN BOXES_OUT KEYPOINTS_IN KEYPOINTS_OUT WIDTH HEIGHT
augmatch_cli meta_remove BOXES_IN BOXES_OUT KEYPOINTS_IN KEYPOINTS_OUT WIDTH HEIGHT MAX_FRACTION
```

The files contain whitespace-separated `x1 y1 x2 y2` boxes and `x y`
keypoints. `examples/meta_example.cpp` demonstrates callback ownership and
annotation clipping.

## License

MIT — see [`LICENSE`](LICENSE).
