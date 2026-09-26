# Cross-library augmentation support contract

This document closes `AUGMENTATION_CATALOG.md` rows 327--346.  The contract is
about the native API that is present in this tree; it does not change any
per-transform evidence or codec availability.  `implemented` means that a
callable API and a checked source/test/example path exist.  `partial` names the
missing type or reference as an explicit blocker; it is not a placeholder.

## Row-to-evidence matrix

| Row | Status | Current contract and evidence |
|---:|:---|:---|
| 327 | implemented | Parameters are typed structs/enums such as `NoiseViewConfig`, `Geometry`, and `BoxTransformOptions`; see `include/augmatch/noise_view.hpp`, `include/augmatch/transforms.hpp`, and `include/augmatch/target_metadata.hpp`. `tests/support_contract_unit.cpp` constructs and executes them. |
| 328 | implemented | `ExecutionContext::seed` and the stateless `CounterRng` in `include/augmatch/execution.hpp` and `include/augmatch/noise_view.hpp` define seeded execution. The CPU implementation is `src/noise_view_cpu.cpp`; the matching CUDA counter is in `src/noise_view.cu`. The unit checks repeatability and HWC/CHW coordinate independence. |
| 329 | implemented | Synchronous host reference: `src/noise_view_cpu.cpp`, `src/transforms_cpu.cpp`, and `src/geometry_metadata.cpp`. Checked by `tests/support_contract_unit.cpp` and the existing `tests/geometry_targets_unit.cpp`. |
| 330 | implemented | Device implementation of the view contract: `src/noise_view.cu` and the CUDA portions of `src/transforms.cu`. `tests/noise_view_cuda_validation.cu` checks a device allocation, asynchronous stream launch, and output. A machine without CUDA skips that runtime test rather than claiming a device result. |
| 331 | implemented | The public headers are C++17-compatible and the CMake target requires C++17 (`CMakeLists.txt`). `examples/native_api_cpp17.cpp` and the `native_api_example_cpp17_smoke` test are compile/run evidence. |
| 332 | implemented | `examples/native_api_cpp23.cpp` is compiled as C++23 by `CMakeLists.txt`; `native_api_example_cpp23_smoke` is the CPU smoke test. No C++23-only API is required. |
| 333 | implemented | `ImageView`/`MutableImageView` carry explicit `Layout::HWC` or `Layout::CHW`, byte strides, dtype, and memory space in `include/augmatch/image.hpp`. `make_hwc_view` and `make_chw_view` are exercised side-by-side in `tests/support_contract_unit.cpp`; strided view coverage is in `tests/noise_view_unit.cpp` and `tests/noise_view_properties.py`. |
| 334 | implemented | `ImageMaskView`, `MutableImageMaskView`, `transform_mask_u8`, and `target_metadata.hpp` provide label-preserving mask/segmentation targets. Nearest-label exactness is checked in `tests/support_contract_unit.cpp`; target-aware crop behavior is covered by `tests/crop_non_empty_mask_if_exists_parity.py`. |
| 335 | implemented | Pascal VOC `BoxXYXY`, COCO, YOLO, and normalized Albumentations representations and conversions are declared in `include/augmatch/target_metadata.hpp`; geometric filtering is in `include/augmatch/transforms.hpp`. `tests/geometry_targets_unit.cpp` and the support unit check conversion, clipping, visibility filtering, and order. |
| 336 | implemented | `KeypointXYV` and `transform_keypoints_with_visibility` are the explicit keypoint contract in `include/augmatch/target_metadata.hpp`. `tests/support_contract_unit.cpp` checks coordinate and visibility behavior; `tests/geometry_targets_unit.cpp` checks batch metadata behavior. |
| 337 | implemented | `PolygonView` represents ordered vertices plus validated ring offsets; `validate_polygon` and `transform_polygon` preserve ring/vertex order and apply crop/flips without clipping. API: `include/augmatch/target_metadata.hpp`; host reference: `src/target_metadata_cpu.cpp`; vectors and invalid-ring checks: `tests/annotation_geometry_unit.cpp`. Metadata pointers are host-owned in CPU and CUDA builds. |
| 338 | implemented | `LineStringView` defines an ordered open path with at least two finite, in-frame vertices; `validate_line_string` and `transform_line_string` preserve segment order and endpoints under crop/flips. API and CPU evidence are shared with row 337. Metadata pointers are host-owned in CPU and CUDA builds. |
| 339 | implemented | `HeatmapView`/`MutableHeatmapView` define strided host float32 continuous scalar/channel planes. `transform_heatmap` applies exact crop/flips without resampling and supports aliased storage using a temporary. API: `include/augmatch/target_metadata.hpp`; CPU reference and exact flip/involution fixtures: `src/target_metadata_cpu.cpp`, `tests/annotation_geometry_unit.cpp`. Heatmap views are explicitly host-only in CUDA builds; no device kernel or implicit transfer is claimed. |
| 340 | implemented | Exact native vectors are checked in `tests/support_contract_unit.cpp` (flip plus label map). Transform-specific external/reference vectors remain in the existing per-transform tests, including `tests/flip_parity.py`, `tests/affine_parity.py`, `tests/resize_parity.py`, and `tests/random_crop_parity.py`. Blocked references are documented rather than synthesized: no common upstream reference exists for the missing polygon, line-string, and heatmap contracts, and optional Python reference packages may be unavailable at test time. |
| 341 | implemented | `tests/support_contract_unit.cpp` checks seeded repeatability and sample mean/variance for a 16,384-sample normal field. Existing stochastic checks include `tests/noise_view_properties.py` and `tests/temporal_noise_statistics.py`; these test distributions/properties, not one fragile random vector. |
| 342 | implemented | `tests/support_contract_unit.cpp` checks the flip involution over finite points. `tests/geometry_targets_unit.cpp` checks scalar-vs-batch points/boxes and clipping/filtering; `tests/*_properties.py` add transform-specific property coverage. |
| 343 | implemented | The tolerance policy is source-visible in `tests/noise_view_cuda_validation.cu`: byte/label/geometry vectors are exact; float CPU/CUDA comparisons use absolute tolerance `1e-6` for the deterministic fixture and up to `2e-6` for seeded normal arithmetic because CPU libm and CUDA `sqrtf/logf/cosf` can round differently. No tolerance converts a missing API into parity. |
| 344 | implemented | Install/export is configured by `CMakeLists.txt` (`install(TARGETS ...)`, headers, `augmatchTargets`, and package config). `tests/package_consumer/run.cmake` installs, configures a `find_package(augmatch)` consumer, builds it, and runs it; CTest registers `install_package_consumer`. |
| 345 | implemented | Measured benchmark evidence is `tests/noise_catalog_validation.cpp --benchmark` and its checker `tests/noise_catalog_benchmarks.py`, registered as `noise_catalog_benchmarks`. It requires and reports 512x512, 1920x1080, and 3840x2160 measurements and rejects malformed/missing output. |
| 346 | implemented | This contract is the documentation index. Runnable examples are `examples/support_contract.cpp`, `examples/native_api_cpp17.cpp`, and `examples/native_api_cpp23.cpp`; their targets and smoke tests are declared in `CMakeLists.txt`. Public API rationale is in `API_DESIGN.md`. |

## Interchange and tolerance rules

A transform configuration is passed as a typed value with explicit dimensions,
seed, interpolation, clipping, and ownership fields.  A view never transfers
ownership.  CPU pointers are host pointers; CUDA pointers are device pointers
valid for the supplied stream.  Metadata arrays are explicit borrowed arrays.
Masks use nearest/label semantics unless a caller explicitly selects the
continuous `MaskInterpolation::Linear` policy.

The shared seeded contract is coordinate-keyed, not traversal-keyed.  Thus a
parallel CUDA launch and a serial CPU loop address the same random sample by
`(x,y,channel,plane)`.  Deterministic integer and geometry outputs are compared
byte-for-byte.  Float outputs use absolute error: `1e-6` for deterministic
fixtures and `2e-6` for seeded CPU/CUDA normal-field comparisons.  Statistical
checks use distribution tolerances and are not substitutes for exact vectors.

The missing polygon, line-string, and heatmap records are deliberate partial
rows.  They must not be inferred from a raw `PointXY*` or float image buffer,
and no unsupported codec row is changed by this document.
