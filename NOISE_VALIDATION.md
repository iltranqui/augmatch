# NOISE_CATALOG validation evidence

Rows 591--610 are covered by the dedicated CTest targets:

- `noise_catalog_validation` runs native CPU statistical, spatial, defect,
  clipping, Bayer, derivative, determinism, golden-vector, and bounds checks.
- `noise_catalog_benchmarks` invokes the CPU executable with `--benchmark` and
  records measured transform and `memcpy` timings at 512x512, 1920x1080, and
  3840x2160. It does not check in timing numbers.
- `noise_catalog_cuda_validation` compiles with the CUDA backend and, when a
  device is available, exercises device shot noise, moments, fixed-seed
  replay, and independent seeds. It returns 77 when CUDA runtime discovery or
  allocation is unavailable, so CTest reports a runtime-gated blocker rather
  than a fabricated result.

The Poisson check compares normalized sample moments to the configured
Poisson mean and variance. Read-noise independence disables exposure and
compares two signal levels. Explicit row and column maps are checked at every
pixel. Hot/dead probabilities use binomial tolerance, and all sensor outputs
are checked against the configured normalized bounds. CFA and derivative
vectors are explicit small golden vectors, while stochastic transforms use
fixed and changed seeds.

The CPU/CUDA check uses the same public `ShotNoiseConfig` and compares each
backend with the analytic moments, rather than claiming bitwise CPU/CUDA
identity. CUDA numerical tolerances are wider because output is quantized to
uint8. A CUDA compiler without a runnable device is an explicit blocker.

Profile-based camera-domain regression is intentionally data-driven. The
benchmark script inventories profile-like records under the repository
`data/` directory. If none are present, it prints
`PROFILE_DATASET=SKIP blocker=...`; it does not generate a profile or report a
measurement. Existing image files alone are not treated as camera calibration
profiles. A future profile fixture must use the versioned
`AUGMATCH_CAMERA_PROFILE_V1` format and be added under `data/` before this row
can become fully implemented.

Tests use the Python interpreter selected by CMake's `find_package(Python3)`.
No optional Python package or external reference implementation is required.
