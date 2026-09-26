# Annotations: boxes, keypoints, masks

Headers: `include/augmatch/annotations/`. Back to the [docs index](../README.md).

## Contents

- [Target metadata formats](#target-metadata-formats)
- [Geometry targets](#geometry-targets)

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
