# Native `augmenters.imgcorruptlike` catalog

Augmatch exposes eight remaining imgaug names (and the corresponding
Albumentations/imgcorruptlike compatibility surface) as contiguous interleaved HWC
`uint8` CPU/CUDA operations. They are deterministic native approximations,
not byte-for-byte promises for imgaug's stochastic image assets or RNG.
Every config has explicit parameters and a seed where the operation needs a
field; no global RNG state is retained. CUDA pointers, including optional
fields, are borrowed device pointers valid through the supplied stream.

| Operation | Contract |
|---|---|
| `SpeckleNoise` | `input * (1 + mean + stddev*N(0,1))`, coordinate-keyed SplitMix64 Gaussian, clipped and half-up rounded. |
| `Fog` | White veil with `opacity*density*field`; field is explicit HxW or deterministic from seed. |
| `Frost` | Cold white/blue veil with the same explicit field ownership contract. |
| `Snow` | Sparse white accumulation where deterministic field values exceed `1-density`. |
| `Contrast` | Scales every supplied channel around 127.5 by `factor`. |
| `Brightness` | Multiplies every supplied channel by `factor`. |
| `Saturate` | Scales RGB HSV saturation; channels after RGB are copied unchanged. |
| `Pixelate` | Samples the nearest source value at each integer block centre; block size 1 is identity. |

The fog, frost, and snow contracts intentionally avoid hidden external frost
images and imgaug's random asset selection. This makes replay, CPU/CUDA
comparison, and CLI fixtures stable while documenting the approximation.

## CLI

All commands read and write raw HWC bytes. The short form and nested form are
both accepted:

```text
speckle_noise IN OUT W H C MEAN STDDEV SEED
fog           IN OUT W H C DENSITY OPACITY SEED
frost         IN OUT W H C INTENSITY OPACITY SEED
snow          IN OUT W H C DENSITY OPACITY SEED
contrast      IN OUT W H C FACTOR
brightness    IN OUT W H C FACTOR
saturate      IN OUT W H C FACTOR
pixelate      IN OUT W H C BLOCK_SIZE
imgcorruptlike OP ...   # equivalent nested spelling
```

Build with `cmake -S . -B build -DAUGMATCH_ENABLE_CUDA=OFF && cmake --build
build`; CUDA uses the same config and launches stream-aware native kernels.
