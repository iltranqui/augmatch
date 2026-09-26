# Native size catalog

`include/augmatch/geometry/size.hpp` exposes the remaining `augmenters.size` operations for
contiguous interleaved uint8 images. CPU and CUDA functions share the same
integer geometry contract.

* `pad_to_*` chooses the smallest dimensions at least as large as the input;
  `crop_to_*` chooses the largest dimensions no larger than the input.
* Multiples use `multiple_width`/`multiple_height`; powers use `base` (the
  default is 2). Aspect-ratio operations use `aspect_ratio = width / height`.
* `SizeAnchor::TopLeft` preserves the image origin. `SizeAnchor::Center` places
  `floor((target-source)/2)` pixels before the source, making odd differences
  deterministic.
* Padding uses `value` for `Constant`, and the existing `PadBorder` replicate
  or reflect-101 policies. Cropping never reads outside the source.
* Aspect ratio dimensions use explicit `SizeRounding` (`Floor`, `Ceil`, or
  `Nearest`). All dimensions are at least one and invalid/overflowing inputs
  throw `std::invalid_argument`.
* `keep_size_by_resize_u8` performs a resize to the explicit intermediate
  dimensions and a resize back to the original dimensions. Its interpolation
  is applied in both passes and output storage is always input-sized.

The CLI accepts raw HWC uint8 input. For example:

```sh
augmatch_cli size pad_multiples in.raw out.raw 640 481 3 32 32 0 0 0
augmatch_cli size center_crop_aspect in.raw out.raw 640 481 3 1.7777778 2 0 0 1
augmatch_cli size keep_size_resize in.raw out.raw 640 481 3 320 240 1
```

The output dimensions for geometry operations are available through the
`make_*_dimensions` factories before allocating the output buffer.
