# Video and temporal artifacts

Headers: `include/augmatch/video/`. Back to the [docs index](../README.md).

## Contents

- [Remaining video and lens artifact contract](#remaining-video-and-lens-artifact-contract)

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
