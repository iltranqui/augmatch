# Augmatch Compatibility Target

## Purpose

Augmatch provides a native C++ API and CPU/CUDA implementations with semantic parity for selected Albumentations and imgaug transforms. It does not attempt to reproduce Python syntax or Python object identity.

## Reference versions

The initial parity target is pinned to:

| Dependency | Version |
|---|---:|
| Albumentations | `2.0.8` |
| imgaug | `0.4.0` |
| NumPy | `1.26.4` |
| OpenCV Python | `4.11.0.86` |
| PyTorch parity helper | `2.2.2` |

The Python oracle must run in an isolated environment using these versions. Newer upstream releases require a new compatibility review and test-vector update.

## Compatibility levels

### Exact parity

The output must be byte-identical for integer images and equal within a documented tolerance for floating-point images. This applies to:

- Shape and dtype.
- Channel ordering.
- Border and interpolation behavior.
- Parameter sampling when the same seed and distribution are supported.
- Bounding-box, keypoint, mask, and segmentation-map coordinates.

### Semantic parity

The result must represent the same augmentation, but implementation-specific floating-point rounding, parallel reduction order, or random-number streams may differ.

### Declared approximation

The transform is available, but a dependency-specific operation such as PIL, SciPy, a codec, or a weather texture cannot be reproduced exactly. The API and documentation must state the approximation.

### Not implemented

The transform remains listed in the support manifest and must fail with an explicit unsupported-transform error rather than silently changing behavior.

## Required input contract

The first complete API must support:

- HWC and CHW layouts.
- `uint8` and `float32` images.
- Single images and batches through repeated image views.
- Images, masks, bounding boxes, keypoints, polygons, and segmentation maps.
- Host memory and CUDA device memory.
- Explicit interpolation, border, clipping, and visibility policies.

Additional dtypes and annotation types may be added without changing the core transform contract.

## RandomResizedCrop and RandomSizedCrop contract

Both transforms take an explicit source rectangle (`crop_x`, `crop_y`, `crop_width`, `crop_height`), output dimensions, and one `Interpolation` value. The implementation first applies `random_crop_u8` to that rectangle and then the existing `resize_u8` contract. Set both crop dimensions to zero and either offset to `-1` to request deterministic SplitMix64 generation from `seed`; generated width, height, x, and y consume seed offsets 0, 1, 2, and 3. This deliberately avoids depending on hidden upstream RNG state or rejection sampling, so generated rectangles are deterministic but are not claimed to match Albumentations' private sampling sequence. Explicit rectangles provide exact CPU/CUDA parity.

## RandomCropNearBBox contract

`RandomCropNearBBox` accepts one explicit Pascal VOC `BoxXYXY` in continuous
image-edge coordinates `(x1,y1,x2,y2)`, not normalized or center-width-height
coordinates. The box must be non-empty and inside the image. Pixel support is
`[floor(x1), floor(y1), ceil(x2), ceil(y2))`; independent integer margins are
then subtracted from the left/top and added to the right/bottom, with the final
rectangle clipped to image bounds. `max_part_shift_x/y` are fractions of the
box width/height and determine the maximum margin. Nonnegative offset fields
are explicit margins; `-1` uses `splitmix64(seed + side)` modulo the maximum
plus one, with sides ordered left, right, top, bottom. The public rectangle
factory is the required allocation/metadata boundary. `random_crop_near_bbox_u8`
changes image pixels only. Box, keypoint, mask, and other annotation
transforms are explicit metadata operations and are not hidden Python targets;
callers must apply the returned crop metadata to each annotation they own.

## Target-aware bbox crop contract

The target-aware bbox APIs use borrowed host arrays of explicit Pascal-VOC
`BoxXYXY` records. `BBoxSafeRandomCrop` samples a source rectangle containing
all boxes (or a seeded fallback when the array is empty), while
`RandomSizedBBoxSafeCrop` applies the same selection before resizing to its
fixed output dimensions. `AtLeastOneBBoxRandomCrop` samples one requested box
and returns a fixed crop intersecting it; empty arrays use explicit fallback
coordinates or seeded placement. `erosion_rate`/`erosion_factor` contract the
required box support toward its center and are therefore an explicit,
controlled approximation of upstream erosion behavior. Factories return crop
metadata and are the required allocation boundary. CPU and CUDA image
functions change pixels only; callers retain ownership and must transform
all annotations explicitly using the returned offset and scale.

## CropNonEmptyMaskIfExists contract

`crop_non_empty_mask_if_exists` accepts a borrowed HWC `uint8` image/mask pair and
fixed crop dimensions. A target is any mask pixel greater than `mask_threshold`;
mask channels are reduced by logical OR. Targets are enumerated row-major, then a
target and a valid crop origin containing it are selected with
`splitmix64(seed + 0)`, `+1`, and `+2`. An absent or empty mask uses explicit
`fallback_x/fallback_y`, or a seeded ordinary crop when both are `-1`. The image
and mask outputs are cropped together without interpolation. Masks therefore
retain nearest-neighbor/label semantics, and the caller owns all buffers and any
other annotations such as boxes, keypoints, and polygons.

## Geometry target helpers

The native `Geometry` contract applies crop first, then horizontal and vertical
flips. Image indices identify pixel centres at integer coordinates. `PointXY`
uses these centre coordinates and maps a horizontal flip to `output_width - 1 - x`; `BoxXYXY` uses continuous half-open edge coordinates in Pascal VOC order
and maps edges to `output_width - x2` and `output_width - x1` (and similarly for
y). `transform_points` and `transform_boxes` operate on host-owned arrays;
CUDA builds do not copy annotation metadata between host and device.

`transform_boxes_filtered` provides explicit clipping, minimum retained-area
visibility, and minimum-area filtering. `filter_boxes` filters already
transformed boxes. The optional output-index array identifies retained
source records for compacting parallel labels. `transform_mask_u8` takes an
explicit nearest or linear policy. Because this Geometry currently has only
integer crop and flips, both policies sample exact pixel centres; nearest is
still required for masks and segmentation labels, while linear is intended for
continuous fields when resampling geometry is added.

## Randomness contract

- Every stochastic pipeline receives an explicit seed or RNG object.
- Composition controls additionally accept an explicit ordered `ReplayRecord`; replay does not resample or consult process-global state.
- A pipeline must be deterministic for a fixed seed, input, transform configuration, backend, and library version.
- CPU and CUDA are not required to use identical random bit streams initially.
- CPU and CUDA must use equivalent distributions and produce results within the transform's declared parity tolerance.
- Randomness must not depend on thread scheduling.

## Additional target and annotation-format contract

Additional image and mask targets are borrowed host `ImageView`/`MutableImageView`
pairs. They share the primary image `Geometry`, accept positive-stride `uint8`
HWC or CHW views, and never retain or transfer metadata. Masks expose an explicit
nearest/linear policy; nearest is the default for label masks. In CUDA builds
these metadata functions remain host-only because annotation arrays and arbitrary
host views are not copied by the library.

Pascal VOC boxes are absolute continuous half-open `BoxXYXY` edges. COCO boxes are
absolute `x,y,width,height`. YOLO boxes are normalized center `x,y,width,height`,
and Albumentations boxes are normalized `x1,y1,x2,y2`; normalized values and
widths/heights are validated in their documented ranges. `KeypointXYV` uses
continuous pixel-center coordinates and visibility in `[0,1]`; transformed points
outside the output image become visibility zero by default.

## Coordinate contract

- Image pixel indices identify pixel centers.
- Bounding boxes use continuous edge coordinates.
- Default box format is Pascal VOC `x_min, y_min, x_max, y_max`.
- Keypoints use continuous `x, y` center coordinates.
- Geometric transforms apply to all targets using one sampled parameter set.
- Clipping, visibility filtering, and removal policies are explicit configuration values.

## Initial implementation priority

1. Existing crop, horizontal flip, vertical flip, and pooling behavior.
2. Resize, pad, rotate, affine, perspective, and 90-degree rotation.
3. Brightness, contrast, gamma, grayscale, hue, saturation, and normalization.
4. Gaussian, average, median, bilateral, and motion blur.
5. Additive noise, multiplicative noise, dropout, cutout, salt-and-pepper, and invert.
6. Composition, probability, `OneOf`, `SomeOf`, and seeded execution.
7. Remaining Albumentations and imgaug-specific transforms.

## Excluded from initial exact-parity commitment

- Python callbacks and lambda transforms.
- Arbitrary user Python code.
- Exact PIL/SciPy implementation quirks where no equivalent native primitive exists.
- Codec-specific JPEG behavior unless the same codec and settings are used.
- Exact cross-backend random bit streams.

## Versioning policy

The compatibility manifest is part of the public contract. A transform change that alters reference output requires:

1. A manifest version increment.
2. Updated reference vectors.
3. A migration note.
4. Regression tests for the previous behavior where practical.
