# Optics and lens artifacts

Headers: `include/augmatch/optics/ (declarations live in filter/, geometry/, sensor/)`. Back to the [docs index](../README.md).

## Contents

- [Optical catalog artifacts](#optical-catalog-artifacts)
- [Optical motion artifact contracts](#optical-motion-artifact-contracts)
- [Focus breathing contract](#focus-breathing-contract)
- [ChromaticAberration contract](#chromaticaberration-contract)
- [Discrete aperture PSF contracts](#discrete-aperture-psf-contracts)
- [Optical artifact contracts](#optical-artifact-contracts)

## Optical catalog artifacts

The NOISE_CATALOG optical entries expose distinct `optical_defocus_u8`,
`optical_motion_blur_u8`, `optical_zoom_blur_u8`,
`optical_chromatic_aberration_u8`, `lateral_chromatic_aberration_u8`,
`longitudinal_chromatic_aberration_u8`, `thin_prism_distortion_u8`, and
`rolling_shutter_geometric_distortion_u8` APIs. They preserve contiguous HWC
`uint8` buffers, use explicit clipping/fill and interpolation, and keep borrowed
H-element row maps in host memory for CPU or device memory for CUDA. The
rolling-shutter geometric operation is an H-W-C still-image remap, not the
existing T-H-W-C video rolling-shutter operation. Thin-prism and chromatic
operations are normalized polynomial approximations; longitudinal aberration
uses independent finite RGB Gaussian supports. The CLI forms and approximation
limits are documented in `NOISE_CATALOG.md`.

## Optical motion artifact contracts

The optical motion APIs use contiguous HWC uint8 buffers, REFLECT_101 borders,
nearest coordinate sampling, equal sample weights, and saturating half-up
rounding. `depth_dependent_defocus_u8` borrows an HxW float32 depth field and
uses `round(abs(depth-focus_depth)*blur_scale)` as a clamped disk radius.
`camera_shake_blur_u8` borrows a deterministic sequence of pixel translations;
`linear_directional_blur_u8` samples a centered segment; and
`rotational_motion_blur_u8` samples a centered angular interval around an
explicit pixel center. `rolling_shutter_motion_blur_u8` borrows H-element row
translation fields and samples each row over its -0.5..+0.5 readout interval.
CPU fields are host-owned; CUDA fields are device-owned until stream completion.
These are documented deterministic surrogates, not continuous optical exposure
integrals. CLI forms are listed in `NOISE_CATALOG.md`.

## Focus breathing contract

`focus_breathing_u8` applies the explicit frame-dependent radial magnification
`scale=1+focus_position*(breathing_strength+radial_strength*r^2)` about the
normalized centre. It inverse-maps each output pixel using Nearest or Linear
sampling and fills/clips out-of-range values with the geometric API contract.
Zero focus position is an exact identity. This is a deterministic radial scale
approximation, not a calibrated lens or depth renderer; local nonpositive scales
are filled. The CLI form is documented in `NOISE_CATALOG.md`.

## ChromaticAberration contract

`chromatic_aberration_u8` applies deterministic per-channel radial remaps to the first three channels of contiguous HWC `uint8` data. For channel `k`, normalized coordinates use `cx=(width-1)/2`, `cy=(height-1)/2`, `fx=max(1,width/2)`, and `fy=max(1,height/2)`. The source map is `x'=cx+xn*(1+radial_k*r2)*fx`, `y'=cy+yn*(1+radial_k*r2)*fy`, where `r2=xn*xn+yn*yn`. Each map uses bilinear interpolation with constant `fill` outside the image, then emits `clip(floor(gain_k*sample+0.5),0,255)`. Channels after RGB are copied unchanged. The CLI form is `color chromatic_aberration IN.raw OUT.raw W H C RADIAL0 RADIAL1 RADIAL2 GAIN0 GAIN1 GAIN2 FILL`; the top-level `chromatic_aberration` spelling is also accepted.

This is a deterministic lateral-dispersion and photometric-gain approximation, not a spectral or wavelength-dependent lens model. It does not model longitudinal focus shifts, wavelength-specific point-spread functions, vignetting, or upstream random parameter sampling. CPU output is compared with fixed OpenCV `remap` maps; CUDA uses the same formula and interpolation contract, with small floating-point rounding differences possible.

## Discrete aperture PSF contracts

The four uint8 aperture APIs in `filter.hpp` use finite, normalized HWC PSFs, `REFLECT_101` borders, and saturating half-up rounding. Radius is an integer in `[0,32]`; zero copies exactly. Diffraction uses the Airy-inspired `sinc^2` disk surrogate; wavelength/aperture/focal metadata scale its lobe with a bounded dimensionless ratio. Bokeh uses a disk with optional linear edge softness. Cat-eye bokeh shrinks the horizontal ellipse toward the explicit normalized image boundary. Aperture-shape blur rasterizes a regular 3--32 blade polygon, with explicit rotation and polygon-to-circle roundness. These are deterministic discrete approximations, not calibrated wave-optics or ray-traced lens models; finite support and uint8 clipping are intentional. CPU buffers are host-owned and CUDA buffers are device-owned until the stream completes. CLI forms are documented in `NOISE_CATALOG.md`.

## Optical artifact contracts

The radial optical APIs (`lens_vignetting_f32`, `color_dependent_vignetting_f32`, and `optical_falloff_f32`) use normalized coordinates and a cubic polynomial in squared radius. Optional scalar HxW or channel-specific HWC gain maps are borrowed, not copied. `lens_shading_f32` and `uneven_illumination_f32` reuse that map layout; `sensor_lens_dust_shadows_f32` uses an HxW opacity map. CPU maps are host memory and CUDA maps are device memory valid through the stream. All six APIs clip normalized float output to the configured interval. The models are deterministic calibration/augmentation approximations, not physical pupil, spectral, or dust-transport simulations. CLI forms and map fixture sizes are documented in `NOISE_CATALOG.md`.
