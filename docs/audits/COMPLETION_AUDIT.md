# Completion Audit

This audit maps the requested deliverables to repository evidence. It deliberately separates implementation/test evidence from exact upstream parity, which remains conditional where a dependency or upstream semantic is unavailable.

| Requirement | Evidence | Result |
|---|---|---|
| Every augmentation catalog entry | `AUGMENTATION_CATALOG.md` contains 346 checklist rows; `IMPLEMENTATION_MANIFEST.tsv` contains the same 346 rows. | Pass; no `[ ]` rows. |
| Every noise/artifact catalog entry | `NOISE_CATALOG.md` contains 275 checklist rows; the manifest contains the same 275 rows. | Pass; no `[ ]` rows. |
| Per-entry CPU/source, CUDA/host-scope, and test evidence | `python3 tools_audit_manifest.py` checks catalog/manifest multiplicity, synchronized markers, and every semicolon-separated path. | Pass; 621 rows, zero missing artifacts. Documented `host-only` fields are intentional API/documentation boundaries. |
| Native C++17 API | `CMakeLists.txt` sets C++17; `native_api_example_cpp17` and package consumer smoke tests pass. | Pass. |
| C++23 compatibility | `CMakeLists.txt` builds `native_api_example_cpp23` with C++23; its smoke test passes. | Pass. |
| CPU implementations | CPU static build `AUGMATCH/build-three` and full CTest pass. | Pass: 149/150; one optional imgaug test skipped. |
| CUDA implementations/validation | CUDA build `AUGMATCH/build-cuda-final`, CUDA 13.3, `sm_89`; all registered CUDA validation tests pass. | Pass: 53/53. |
| Pinned Albumentations/imgaug verification | `requirements-test.txt` pins Albumentations 2.0.8, imgaug 0.4.0, OpenCV 4.11.0.86. The rebuilt pinned-environment parity/data suite passed 100/100 registered tests; the earlier manual sweep covered 97 scripts. | Albumentations evidence pass. imgaug pooling is blocked because imgaug 0.4.0 cannot import with NumPy 2.2.6. Available OpenCV is 5.0.0, not pinned 4.11.0.86. |
| Use available `data/` images where applicable | `data/` contains four JPG fixtures; `tests/data_smoke.py` and noise benchmark scripts reference the directory. | Pass for applicable smoke/data paths; no camera-profile regression dataset is available. |
| Umbrella header | `include/augmatch/augmatch.hpp` includes all 41 other public `.hpp` headers; package consumer includes it. | Pass. |
| Installable static library | `add_library(augmatch STATIC ...)`, install/export rules, and `install_package_consumer` test. | Pass; CMake emits `.a` on Linux and `.lib` with Visual Studio generators. |
| CMake/unit/property/parity integration | CMake registers the manifest audit, units, properties, parity scripts, package consumer, and CUDA validation targets. | Pass; full CPU and CUDA gates above. |

## Current partial boundaries

The 304 `partial` rows have concrete source and test paths. Their status records one or more of: a documented native approximation, a host-only metadata/API contract, a conditional codec dependency, a CUDA host-codec fallback, unavailable upstream reference semantics, or an external dependency blocker. AV1/H.264/H.265 temporal/reference-picture behavior is not claimed by the single-frame APIs. WebP remains conditional on libwebp. Polygon/line-string/heatmap metadata has no implicit CUDA transfer.

## Verification commands

```text
python3 AUGMATCH/tools_generate_manifest.py --check
python3 AUGMATCH/tools_audit_manifest.py
/usr/bin/ctest --test-dir AUGMATCH/build-three --output-on-failure
/usr/bin/ctest --test-dir AUGMATCH/build-cuda-final -R 'cuda_validation' --output-on-failure
compute-sanitizer --tool memcheck --error-exitcode=1 AUGMATCH/build-cuda-final/demosaic_artifacts_cuda_validation
```
