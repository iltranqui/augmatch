# Blend augmentation contract

All blend functions consume a contiguous HWC `uint8` source and overlay and write a
caller-owned HWC `uint8` output. Source and overlay may alias output. CPU pointers
refer to host memory; CUDA pointers refer to device memory and remain valid until
the supplied stream completes. Scalar and generated alpha values are clamped to
`[0,1]`, and results use half-up rounding to uint8.

`BlendAlphaMask` and `BlendAlphaElementwise` use a borrowed one-byte-per-pixel
H×W mask. The default conversion is `mask/255`; the mask is read-only and target
metadata ownership remains with the caller. Segmentation maps and class-id arrays
are also borrowed and never modified. Bounding-box records are borrowed
continuous half-open `BoxXYXY` image-edge coordinates; boxes must be inside the
image and target metadata remains caller-owned.

Noise fields use deterministic SplitMix64-derived value noise or a sinusoidal
frequency field. These are portable approximations of imgaug's implementation,
not claims of bit-for-bit parity with every version's implicit RNG/noise sampler.
Gradients, regular grids, and checkerboards use pixel coordinates and consume no
random state. `SomeColors` compares source pixels to explicit color records.

The CPU and CUDA APIs have the same structs and semantics. The CLI currently
exposes the scalar operation:

```text
augmatch_cli blend_alpha SOURCE.raw OVERLAY.raw OUT.raw WIDTH HEIGHT CHANNELS ALPHA
```

Composition APIs are unchanged; blend operations are ordinary explicit buffer
operations and can be wrapped in existing `CpuStage`/`DeviceStage` callbacks.
