# ISP artifacts and demosaicing

Headers: `include/augmatch/isp/`. Back to the [docs index](../README.md).

## Contents

- [ISP artifact contracts](#isp-artifact-contracts)
- [Advanced Bayer demosaicing contracts](#advanced-bayer-demosaicing-contracts)
- [Demosaicing artifact contracts](#demosaicing-artifact-contracts)

## ISP artifact contracts

The thirteen `isp_artifacts.hpp` operations use contiguous HWC uint8 buffers with clamped borders, float arithmetic, and saturating half-up rounding to `[0,255]`. CPU pointers are host-owned; CUDA pointers are device-owned for the supplied stream. These deterministic formulas are native ISP surrogates, not vendor-specific reference pipelines. The APIs and CLI forms are documented in `NOISE_CATALOG.md`.

`edge_oversharpening_u8`, `unsharp_mask_halos_u8`, and `laplacian_halos_u8` add signed 3x3, Gaussian, and four-neighbour Laplacian residuals. `ringing_near_strong_edges_u8` gates a checkerboard-sign Laplacian on a centred gradient and applies an explicit residual bound. Local contrast uses a mean/variance residual; tone-mapping haloing and local sharpening use Gaussian high-pass residuals, with explicit tone/noise gains. High-frequency attenuation and detail smearing blend deterministic box/Gaussian low-pass results. Overshoot/undershoot applies a bounded signed Gaussian high-boost residual with independent positive and negative limits. Clipping is intentional and may hide over-correction at saturated pixels. The Laplacian, Sobel, and high-pass residual injection APIs accept borrowed signed float32 HxW or HxWxC maps (`map_channels=1` or `channels`); CPU maps are host-owned and CUDA maps remain device-owned through the supplied stream. Null maps derive the named residual, and all results clip to uint8 `[0,255]`.

## Advanced Bayer demosaicing contracts

`bayer_demosaic_malvar_he_cutler_u8` implements the Malvar-He-Cutler 5x5 filters. The green-at-red/blue filter has weights `[[0,0,-1,0,0],[0,0,2,0,0],[-1,2,4,2,-1],[0,0,2,0,0],[0,0,-1,0,0]]/8`; the same-color-at-green filter uses center weight 5 and the diagonal filter uses center weight 6, with the published 0.5, -1, 4 and -1.5, 2 coefficients. All filters use clamp-to-edge coordinates, then clip and round to uint8. `bayer_demosaic_edge_aware_u8` compares clamped horizontal/vertical raw gradients at green sites and the two diagonal gradients at red/blue sites. It selects the lower-gradient direction, averages both on a tie, and searches distances one then two for the target-color pair. Both APIs consume a contiguous HxW raw plane and explicitly select `RGGB`, `BGGR`, `GRBG`, or `GBRG` through `BayerConfig`; the eight `*_rggb_u8`, `*_bggr_u8`, `*_grbg_u8`, and `*_gbrg_u8` convenience APIs select a tile directly. CLI forms are `demosaic_malvar IN.raw OUT.raw W H PATTERN` and `demosaic_edge_aware IN.raw OUT.raw W H PATTERN`.

## Demosaicing artifact contracts

`bayer_directional_demosaicing_artifacts_u8`, `bayer_false_color_zipper_artifacts_u8`, `bayer_demosaicing_aliasing_u8`, `bayer_demosaicing_ringing_u8`, and `bayer_demosaicing_noise_amplification_u8` consume HxW raw Bayer `uint8` and produce HxWx3 RGB8. They use edge-aware demosaicing at zero strength, preserve known CFA samples where applicable, and saturate to `clip_min..clip_max` before uint8 rounding. Directional injection selects gradient, horizontal, or vertical interpolation; zipper injection alternates red/blue by the horizontal raw gradient; aliasing injects an explicit period/phase sinusoid; ringing uses a signed four-neighbor Laplacian; noise amplification uses coordinate-keyed Gaussian noise scaled by local gradient. CPU uses host buffers and CUDA uses device buffers valid through the stream. The formulas are deterministic native approximations, not upstream camera/ISP parity. CLI forms are `demosaic_directional IN.raw OUT.raw W H PATTERN DIRECTION STRENGTH`, `demosaic_zipper IN.raw OUT.raw W H PATTERN STRENGTH`, `demosaic_aliasing IN.raw OUT.raw W H PATTERN PERIOD PHASE STRENGTH`, `demosaic_ringing IN.raw OUT.raw W H PATTERN STRENGTH`, and `demosaic_noise_amplification IN.raw OUT.raw W H PATTERN NOISE_STDDEV AMPLIFICATION SEED`.
