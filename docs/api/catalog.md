# imgaug / imgcorruptlike catalog APIs

Headers: `include/augmatch/catalog/`. Back to the [docs index](../README.md).

## Contents

- [Additional imgaug catalog APIs](#additional-imgaug-catalog-apis)
- [imgcorruptlike compatibility contract](#imgcorruptlike-compatibility-contract)

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

## imgcorruptlike compatibility contract

`imgcorruptlike.hpp` provides deterministic CPU/CUDA approximations for
SpeckleNoise, Fog, Frost, Snow, Contrast, Brightness, Saturate, and Pixelate.
They use explicit configs and coordinate-keyed seeds rather than imgaug or
Albumentations global random state. See `IMGCORRUPTLIKE_CATALOG.md` for the
formulae, borrowed field ownership, and raw-byte CLI forms.
