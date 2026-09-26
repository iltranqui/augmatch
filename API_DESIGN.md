# Public API Design

The native API is C++17-first and remains source-compatible with C++23.

## Image views

`include/augmatch/image.hpp` defines non-owning `ImageView` and `MutableImageView` objects. Every view explicitly describes dimensions, strides, datatype, layout, and memory space. Transform implementations must not assume contiguous HWC data unless the operation documents that restriction.

Supported initial datatypes are `uint8`, `uint16`, and `float32`. The implementation may add datatypes without changing the view ABI.

`make_hwc_view<T>` and `make_chw_view<T>` provide explicit non-owning
factories for `uint8_t`, `uint16_t`, and `float` views. Strides remain byte
strides, and CHW views retain the same `(x,y,c)` stride fields. The
view-oriented noise API in `noise_view.hpp` accepts these layouts, including
strided host views, and returns `Status` instead of throwing. Device views are
launched asynchronously on the supplied CUDA stream; CPU builds return
`StatusCode::Unsupported` for device views rather than dereferencing them.

`TensorViewF32` and `MutableTensorViewF32` in `convert.hpp` are the explicit
ToTensorV2 contract: contiguous CHW float32 storage with strides measured in
float elements. `to_tensor_v2` accepts contiguous HWC uint8 input and performs
no ownership transfer or hidden copy. Optional normalization is applied after
conversion to `[0,1]`. This is a native tensor-buffer conversion and does not
include or depend on PyTorch.

`ToTensor3DConfig` and `to_tensor_3d_u8_f32` define the corresponding 3D
contract without overloading the 2D image view: the input is contiguous DHWC
uint8 and the output is contiguous CDHW float32. For input coordinates
`(d,h,w,c)`, the source offset is `(((d * H + h) * W + w) * C + c)` and the
output offset is `(((c * D + d) * H + h) * W + w)`. Optional normalization is
applied after division by 255, with channels after the third reusing the third
mean and standard deviation. CPU pointers are host pointers; CUDA pointers are
device pointers, and a caller synchronizes the supplied stream when needed.

## Temporal noise batches

Temporal noise APIs in `noise.hpp` accept contiguous float32 frame batches in
`T×H×W×C` order, with offset `(((t*H+y)*W+x)*C+channel)`. `t=0` is the first
frame and all dimensions are explicit in each config. Inputs and outputs are
non-owning; CPU pointers are host memory and CUDA pointers are device memory
valid through the supplied stream. The operations retain no temporal state.

Correlation is an explicit AR(1) coefficient in `[-1,1]`, seeded by logical
coordinates rather than launch order. The recurrence starts at zero for the
first frame, so repeated calls with the same seed/configuration are
reproducible and changing batch shape cannot silently change spatial indexing.

## Execution

`include/augmatch/execution.hpp` defines `ExecutionContext`, which carries the CUDA stream, seed, and deterministic-execution policy. `ReplayRecord` is an ordered, borrowed list of `StageSelectionRecord` decisions. A recording context appends to `record`; a replaying context validates and consumes `replay`. The traversal index is reset by each top-level composition call, so no global RNG or hidden state is involved.

## Composition and control

`include/augmatch/composition.hpp` adds deterministic in-place `compose` and
`sequential` pipelines. `CpuStage` is a synchronous host callback with the
signature `void(MutableImageView, const ExecutionContext&, void*)`. Stage order
is preserved, the execution context is read-only, and no global RNG state is
used. The callback pointer and its `void*` context are borrowed; the caller
owns them and must keep them alive until the pipeline returns. The callback
must not retain either pointer.

`identity` and `noop` validate the view and leave all bytes and execution state
unchanged. An empty pipeline has the same identity behavior. Compose and
Sequential currently have identical order-preserving semantics, with separate
constructors so later control policies can be added without changing the API.

A host callback is not a safe CUDA stage contract. CUDA pipelines therefore use
`DeviceStage`: its optional `DeviceStageLaunch` is a host function that must
enqueue device work on the supplied `ExecutionContext::stream`; it must not
dereference device image memory on the host. Its context is also borrowed and
must remain valid through the call. A null device launch is the explicit
Identity/Noop representation. CUDA functions do not synchronize the stream;
callers synchronize when completion is required.

`one_of`, `one_or_other`, `random_apply`, `some_of`, and `random_order` are
explicit control operations for CPU or CUDA child pipelines. Their
backend-neutral records store the chosen index, selected index list, order
permutation, or applied bit. `replay_compose` is the named replay boundary and
uses the same record format on both backends. `ChannelRange` gives each CPU or
CUDA stage an optional borrowed channel interval. `ChannelSelection` extends
that contract with explicit indices and ranges; callbacks receive one view per
contiguous selected run, retaining original strides and avoiding copies.
`selective_channel_transform` applies a pipeline to selected ranges or indices,
and `with_channels` is the imgaug meta-operation using the same contract.

## Deterministic Voronoi target maps

`VoronoiConfig`, `UniformVoronoiConfig`, `RegularGridVoronoiConfig`, and
`RelativeRegularGridVoronoiConfig` operate on contiguous HxW discrete target
masks. Voronoi and UniformVoronoi use `seed` and `point_count`; Voronoi may
instead borrow an explicit ordered `VoronoiPoint` array. UniformVoronoi places
one seeded-jittered point per deterministic stratum. RegularGridVoronoi uses
`grid_width` by `grid_height` cell counts. RelativeRegularGridVoronoi treats
its two fractions as cell widths relative to the image and uses
`ceil(1/fraction)` cells per axis. Nearest-site ties select the lowest site
index. A target-map call copies the original mask byte at the winning site's
pixel, preserving discrete labels without interpolation; it never invents a
label. The `*_labels_u32` calls expose the site index map instead. CPU buffers
are host-owned and CUDA buffers are device-owned for the stream lifetime.

## Compatibility policy

The existing raw-pointer functions in `transforms.hpp` remain available as compatibility wrappers. New APIs will use views and contexts. This permits existing callers to migrate incrementally.

## Deterministic impulse arithmetic

`SaltConfig`, `PepperConfig`, `SaltAndPepperConfig`, and their `Coarse*`
variants operate on contiguous HWC `uint8` storage. A borrowed mask contains
one byte per pixel, and a borrowed rectangle list contains integer half-open
`[x0,x1) x [y0,y1)` records. Explicit selection is authoritative and consumes
no random state. Salt/Pepper treat nonzero mask bytes as selected; the combined
operation interprets 1 as salt and 2 as pepper. Seeded mode uses deterministic
counter-based Bernoulli draws from `probability`; coarse mode shares one draw
across each configured block. This is a native deterministic contract and does
not reproduce imgaug's implicit distributions or global RNG state. CPU buffers
are host-owned and CUDA buffers/records are device-owned for the stream
lifetime.

## Deterministic colorspace meta aliases

`ColorSpace`, `WithColorspaceConfig`, and `WithBrightnessChannelsConfig` define
explicit replacements for imgaug's child-based `WithColorspace` and
`WithBrightnessChannels` APIs. Inputs are contiguous interleaved HWC uint8
RGB/RGBA; CPU pointers are host-owned and CUDA pointers are device-owned.
RGB uses sRGB bins, HSV uses OpenCV uint8 bins, and LAB uses a deterministic
D65 sRGB CIELAB approximation. `with_colorspace_u8` applies three explicit
multiplier/addend pairs in the selected domain and converts back to RGB;
channels after RGB are copied. `with_brightness_channels_u8` applies one
multiplier/addend to explicit RGB indices or HSV V (index 2), rejecting HSV H/S
selection. Half-up rounding and clipping are shared by CPU and CUDA. LAB is not
ICC color management, and neither alias samples imgaug child distributions.

## Uniform color quantization

`UniformColorQuantizationConfig::levels` is an explicit integer in `[2,256]`.
For each uint8 component, `q=256/levels`, the bin-center result is
`clip(round_half_up(floor(v/q)*q + q/2), 0, 255)`. `levels=256` is identity.
`UniformColorQuantizationToNBitsConfig::bits` is an explicit integer in `[1,8]`;
it uses the lower bin boundary with `levels=2^bits`, equivalently clears the
`8-bits` least-significant bits. A four-channel HWC image preserves channel 3
as alpha. CPU pointers are host-owned and CUDA pointers are device-owned; CUDA
launches are asynchronous on the supplied stream.

## All-channel contrast aliases

`AllChannelsCLAHEConfig`/`all_channels_clahe_u8` and
`AllChannelsHistogramEqualizationConfig`/`all_channels_histogram_equalization_u8`
are explicit imgaug alias APIs. They operate on contiguous interleaved HWC
uint8 buffers and process every supplied channel independently, including
channels after RGB (for example alpha), without colorspace conversion or
channel dropping. CPU pointers are host pointers and CUDA pointers are device
pointers. The aliases delegate to the native CLAHE and equalize implementations
respectively, so no host/device fallback is hidden behind either API.

## RandomCropNearBBox

`RandomCropNearBBoxConfig` takes an explicit `BoxXYXY` in Pascal VOC order
`(x1, y1, x2, y2)`. Coordinates are continuous image-edge coordinates, with
`x2 > x1` and `y2 > y1`; the box must lie inside the image. The crop uses
`floor(x1), floor(y1), ceil(x2), ceil(y2)` as its half-open pixel support, then
adds independently selected integer margins and clips the rectangle to the
image. `max_part_shift_x` and `max_part_shift_y` bound each horizontal and
vertical margin as fractions of the corresponding continuous box extent.
Offsets `left_offset`, `right_offset`, `top_offset`, and `bottom_offset` are
explicit pixel margins when nonnegative. The default `-1` generates a margin
with `splitmix64(seed + side_index) % (maximum + 1)`, in left/right/top/bottom
order. `make_random_crop_near_bbox_rectangle` returns the output geometry;
the caller allocates `width * height * channels` bytes before calling
`random_crop_near_bbox_u8`. CPU pointers are host pointers and CUDA pointers
are device pointers. The crop API transforms pixels only. Transforming the
source box or any other annotation is explicit metadata work by the caller;
there are no hidden Python targets or annotation containers.

## Target-aware bbox crops

`BBoxSafeRandomCropConfig`, `RandomSizedBBoxSafeCropConfig`, and
`AtLeastOneBBoxRandomCropConfig` accept a borrowed host array of explicit
Pascal-VOC `BoxXYXY` records. Coordinates are continuous image-edge values and
must be finite, non-empty, and inside the input image. The factories return
integer half-open source crop metadata before allocation. `BBoxSafeRandomCrop`
chooses dimensions between the configured minimum and maximum while containing
the union of all boxes; an empty array uses a seeded ordinary crop. Its
`erosion_rate` contracts that required envelope toward its center, deliberately
allowing controlled clipping when nonzero. `RandomSizedBBoxSafeCrop` uses the
same source selection and resizes it to its explicit output dimensions,
returning both source geometry and scale. `AtLeastOneBBoxRandomCrop` chooses a
box with `splitmix64(seed)` and chooses an origin whose fixed crop intersects
the selected (optionally contracted) box; empty arrays use explicit fallback
coordinates or seeded ordinary placement. The source selection uses seed
offsets in order and is identical on CPU and CUDA.

The raw image functions copy only image pixels (and resize pixels for
`RandomSizedBBoxSafeCrop`). CPU image pointers are host pointers; CUDA image
pointers and the CUDA stream are device execution inputs. Box arrays remain
borrowed host metadata and are never copied or owned by the library. Returned
crop metadata is the annotation boundary: callers own and explicitly transform
boxes, keypoints, polygons, masks, and labels using the source offset and,
when applicable, scale. No annotation container or hidden Python target is
mutated.

## Target-aware image/mask crop

`ImageMaskView` and `MutableImageMaskView` are non-owning native views for
`CropNonEmptyMaskIfExists`. The mask is optional for the fallback path, while a
present mask must match the source image dimensions. CPU views may be strided;
CUDA views are contiguous HWC `uint8` device views. The operation returns/copies
only the image and mask crop. Mask values are copied exactly (no linear
interpolation), and callers retain ownership of boxes, keypoints, polygons, and
other annotation metadata.

## Sensor calibration maps

`PixelResponseNonUniformityConfig` and `FixedPatternOffsetNoiseConfig` apply
borrowed contiguous HWC float32 gain and offset maps through
`pixel_response_non_uniformity_f32` and `fixed_pattern_offset_noise_f32`. A null
map selects deterministic per-sample generation from `seed` and `stddev`; no
mutable RNG state is retained. CPU map pointers are host-owned. CUDA map
pointers are device-owned and must remain valid until the supplied stream
finishes; the library performs no host-device map copy. These operations leave
float values unclipped so callers can place sensor calibration before the ADC
or clipping stage.

## Target metadata formats and additional targets

`target_metadata.hpp` provides host-only borrowed pairs for additional image and
mask targets. Each pair has an input and output `ImageView`; dimensions must match
one shared `Geometry`, views must be host `uint8`, and strided HWC or CHW layouts
are accepted. Image and mask targets follow the same crop and flip parameters as
the primary image. Mask interpolation is explicit, with nearest as the label-safe
default. The implementation does not allocate persistent storage or copy CUDA
metadata; callers own every view and buffer.

The metadata conversion APIs use these explicit contracts:

- Pascal VOC `BoxXYXY` is absolute continuous half-open `(x1,y1,x2,y2)` edge
  geometry.
- COCO `COCOBox` is absolute `(x,y,width,height)`.
- YOLO `YOLOBox` is normalized `(center_x,center_y,width,height)`.
- Albumentations `AlbumentationsBox` is normalized `(x1,y1,x2,y2)`.

Normalized coordinates are fractions of image width and height and are validated
in `[0,1]`; widths and heights must be positive. Conversion and transform helpers
validate records synchronously, clip transformed boxes to the output image, and
preserve order. `KeypointXYV.visibility` is a finite float in `[0,1]`. Geometry
preserves visibility for in-frame points and, by default, sets an out-of-frame
point to zero; callers can disable that policy when their visibility labels have
a separate meaning. These host metadata APIs are available in CUDA builds but
are not CUDA kernels and never move annotation arrays to or from device memory.

## Error policy

Invalid metadata, dimensions, strides, and parameter combinations are reported synchronously with `std::invalid_argument` by the existing raw-pointer APIs. The view-oriented noise API additionally exposes `StatusCode` results (`InvalidView`, `InvalidMetadata`, `Unsupported`, and `ExecutionError`) so callers that cannot use exceptions can inspect failures directly. CUDA launch errors are reported at launch time; callers that require completion must synchronize the supplied stream.

`NoiseChannelMetadata` and `NoisePlaneMetadata` are borrowed records. They
provide independent gain, bias, and standard-deviation scales, with an
explicit channel-to-plane map for Bayer or other multi-plane data. The
`CounterRng` maps seed, logical coordinates, channel, plane, and stream to a
stateless sample; traversal and launch order do not affect results. Clipping,
quantization (`Preserve`, half-up, floor, ceil, or truncate), and border
policies (`Clamp`, `Reflect101`, or `Constant`) are explicit. Border resolution
is also available as a host metadata helper for neighborhood implementations.

## ABI policy

- No `std::span`, ranges, or other C++20-only types in public headers.
- No ownership is implied by a view.
- No hidden host-device copies in library transform functions.
- No implicit global RNG state.

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
