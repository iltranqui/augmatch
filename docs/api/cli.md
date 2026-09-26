# `augmatch_cli` command index

Every raw-byte CLI form documented in the API pages, grouped by page. Inputs are
interleaved HWC `uint8` `.raw` files unless the linked section says otherwise.
The CLI is a parity-test adapter; the C++ API is the primary interface.

Back to the [docs index](../README.md).

## [imgaug / imgcorruptlike catalog APIs](catalog.md)

```sh
augmatch_cli remaining additive_laplace IN.raw OUT.raw W H C SCALE SEED
augmatch_cli remaining additive_poisson IN.raw OUT.raw W H C SCALE SEED
augmatch_cli remaining kmeans IN.raw OUT.raw W H C CLUSTERS ITERATIONS SEED
augmatch_cli remaining convolve IN.raw OUT.raw W H C
```

## [Color, tone, and arithmetic](color.md)

```sh
augmatch_cli arithmetic add_elementwise IN.raw OUT.raw W H C VALUES.f32
augmatch_cli arithmetic multiply_elementwise IN.raw OUT.raw W H C MIN MAX SEED
augmatch_cli arithmetic add_elementwise IN.raw OUT.raw W H C MIN MAX SEED VALUES.f32
augmatch_cli arithmetic replace_elementwise IN.raw OUT.raw W H C MASK.raw VALUES.raw
augmatch_cli arithmetic replace_elementwise IN.raw OUT.raw W H C PROBABILITY REPLACEMENT SEED
augmatch_cli arithmetic impulse_noise IN.raw OUT.raw W H C MASK.raw VALUES.raw
augmatch_cli arithmetic impulse_noise IN.raw OUT.raw W H C PROBABILITY SALT_PROBABILITY SEED
augmatch_cli arithmetic salt IN.raw OUT.raw W H C PROBABILITY SEED
augmatch_cli arithmetic salt IN.raw OUT.raw W H C MASK.raw
augmatch_cli arithmetic coarse_salt IN.raw OUT.raw W H C PROBABILITY SEED BLOCK_W BLOCK_H
augmatch_cli arithmetic coarse_salt IN.raw OUT.raw W H C RECTANGLES
augmatch_cli arithmetic salt_and_pepper IN.raw OUT.raw W H C PROBABILITY SALT_PROBABILITY SEED
augmatch_cli tone sigmoid_contrast IN.raw OUT.raw W H C CUTOFF GAIN
augmatch_cli tone log_contrast IN.raw OUT.raw W H C GAIN BASE
augmatch_cli color uniform_color_quantization IN.raw OUT.raw W H C LEVELS
augmatch_cli color uniform_color_quantization_to_n_bits IN.raw OUT.raw W H C BITS
```

## [Core: views, conversion, tensors](core.md)

```sh
augmatch_cli to_tensor_v2 IN.raw OUT.f32 WIDTH HEIGHT CHANNELS [NORMALIZE]
augmatch_cli to_tensor_v2 IN.raw OUT.f32 WIDTH HEIGHT CHANNELS 1 MEAN0 MEAN1 MEAN2 STD0 STD1 STD2
augmatch_cli to_tensor_3d IN.raw OUT.f32 WIDTH HEIGHT DEPTH CHANNELS [NORMALIZE]
augmatch_cli to_tensor_3d IN.raw OUT.f32 WIDTH HEIGHT DEPTH CHANNELS 1 MEAN0 MEAN1 MEAN2 STD0 STD1 STD2
```

## [Filters, blur, edges, and blending](filter.md)

```sh
augmatch_cli emboss IN.raw OUT.raw WIDTH HEIGHT CHANNELS ALPHA STRENGTH
augmatch_cli edge_detect IN.raw OUT.raw WIDTH HEIGHT CHANNELS ALPHA
augmatch_cli directed_edge_detect IN.raw OUT.raw WIDTH HEIGHT CHANNELS ALPHA DIRECTION
augmatch_cli canny IN.raw OUT.raw WIDTH HEIGHT CHANNELS LOW HIGH APERTURE BORDER BORDER_VALUE
```

## [Geometry, size, dropout, and mixing](geometry.md)

```sh
augmatch_cli mixup A.raw B.raw OUT.raw W H C 0.5 7
augmatch_cli cutmix A.raw B.raw OUT.raw W H C 8 8 24 24
augmatch_cli mosaic A.raw B.raw C.raw D.raw OUT.raw W H C CX CY
augmatch_cli template_transform A.raw T.raw OUT.raw W H C 1 1
augmatch_cli overlay_elements BASE.raw ELEMENT.raw OUT.raw W H C 4 4 0.75
```

## [Pipeline and composition](pipeline.md)

```sh
augmatch_cli meta_clip BOXES_IN BOXES_OUT KEYPOINTS_IN KEYPOINTS_OUT WIDTH HEIGHT
augmatch_cli meta_remove BOXES_IN BOXES_OUT KEYPOINTS_IN KEYPOINTS_OUT WIDTH HEIGHT MAX_FRACTION
```

## [Sensor noise, ISO profiles, and Bayer/CFA](sensor.md)

```sh
augmatch_cli iso_profile IN OUT W H C ISO KNOTS_F32 COUNT SHOT_SCALE READ_STD FPN_STD SEED OP
augmatch_cli iso_dark_current IN OUT W H C ISO KNOTS_F32 COUNT RATE_EPS EXPOSURE_SECONDS ELECTRONS_PER_UNIT SEED
augmatch_cli iso_temperature_noise IN OUT W H C ISO KNOTS_F32 COUNT NOISE_E TEMP_C REF_C COEFF_PER_C ELECTRONS_PER_UNIT SEED
augmatch_cli dual_conversion_gain IN OUT W H C ISO KNOTS_F32 COUNT LOW_T HIGH_T LOW_GAIN HIGH_GAIN LOW_READ_E HIGH_READ_E ELECTRONS_PER_UNIT SEED
augmatch_cli gain_switch_transition IN OUT W H C ISO KNOTS_F32 COUNT LOW_T HIGH_T LOW_SIGNAL HIGH_SIGNAL HYSTERESIS TRANSITION_WIDTH STRENGTH INITIAL_HIGH SEED
```
