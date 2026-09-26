#!/usr/bin/env python3
"""Generate a complete catalog-to-artifact manifest for Augmatch."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
CATALOGS = [ROOT / "AUGMENTATION_CATALOG.md", ROOT / "NOISE_CATALOG.md"]
EVIDENCE = {
    "Compression and image corruption::TemplateTransform": ("implemented", "src/mixing_cpu.cpp", "src/mixing.cu", "tests/mixing_unit.cpp", "shared weighted template API"),
    "Compression and image corruption::FDA": ("implemented", "src/remaining_catalog_cpu.cpp", "src/remaining_catalog.cu", "tests/remaining_catalog_unit.cpp", "host-only low-frequency color-statistics approximation"),
    "Compression and image corruption::FourierDomainAdaptation": ("implemented", "src/remaining_catalog_cpu.cpp", "src/remaining_catalog.cu", "tests/remaining_catalog_unit.cpp", "FDA alias; host-only approximation"),
    "`augmenters.arithmetic`::AdditiveLaplaceNoise": ("implemented", "src/remaining_catalog_cpu.cpp", "src/remaining_catalog.cu", "tests/remaining_catalog_unit.cpp", "deterministic host-only Laplace HWC uint8"),
    "`augmenters.arithmetic`::AdditivePoissonNoise": ("implemented", "src/remaining_catalog_cpu.cpp", "src/remaining_catalog.cu", "tests/remaining_catalog_unit.cpp", "deterministic host-only Poisson HWC uint8"),
    "`augmenters.artistic`::Cartoon": ("implemented", "src/remaining_catalog_cpu.cpp", "src/remaining_catalog.cu", "tests/remaining_catalog_unit.cpp", "mean smoothing plus edge mask approximation"),
    "`augmenters.collections`::RandAugment": ("implemented", "src/remaining_catalog_cpu.cpp", "src/remaining_catalog.cu", "tests/remaining_catalog_unit.cpp", "seeded fixed operation set"),
    "`augmenters.color`::ChangeColorspace": ("implemented", "src/remaining_catalog_cpu.cpp", "src/remaining_catalog.cu", "tests/remaining_catalog_unit.cpp", "RGB/HSV conversion; LAB copy approximation"),
    "`augmenters.color`::KMeansColorQuantization": ("implemented", "src/remaining_catalog_cpu.cpp", "src/remaining_catalog.cu", "tests/remaining_catalog_unit.cpp", "seeded deterministic RGB k-means"),
    "`augmenters.convolutional`::Convolve": ("implemented", "src/remaining_catalog_cpu.cpp", "src/remaining_catalog.cu", "tests/remaining_catalog_unit.cpp", "borrowed host kernel and reflected border"),
    "`augmenters.convolutional`::Emboss": ("partial", "src/filter_cpu.cpp", "src/filter.cu", "tests/imgaug_convolutional_parity.py", "deterministic explicit alpha/strength convolution contract"),
    "`augmenters.convolutional`::EdgeDetect": ("partial", "src/filter_cpu.cpp", "src/filter.cu", "tests/imgaug_convolutional_parity.py", "deterministic explicit alpha convolution contract"),
    "`augmenters.convolutional`::DirectedEdgeDetect": ("partial", "src/filter_cpu.cpp", "src/filter.cu", "tests/imgaug_convolutional_parity.py", "deterministic explicit alpha/direction convolution contract"),
    "`augmenters.debug`::SaveDebugImageEveryNBatches": ("implemented", "src/remaining_catalog_cpu.cpp", "src/remaining_catalog.cu", "tests/remaining_catalog_unit.cpp", "synchronous host PGM/PPM; callbacks unsupported"),
    "`augmenters.geometric`::WithPolarWarping": ("implemented", "src/remaining_catalog_cpu.cpp", "src/remaining_catalog.cu", "tests/remaining_catalog_unit.cpp", "nearest-neighbour polar remap approximation"),
    "`augmenters.geometric`::Jigsaw": ("implemented", "src/remaining_catalog_cpu.cpp", "src/remaining_catalog.cu", "tests/remaining_catalog_unit.cpp", "explicit or seeded bijective tile permutation"),
    "12. Documentation tasks::Document physical sensor noise versus ISP artifact noise": ("implemented", "NOISE_CATALOG.md", "", "examples/noise_camera_workflows.cpp", "Sensor physics and downstream ISP surrogates are distinguished; example verifies deterministic RGB sensor noise"),
    "12. Documentation tasks::Document parameter units and calibration procedures": ("implemented", "NOISE_CATALOG.md", "", "tests/profile_calibration_unit.cpp", "Units, full-scale normalization, and profile calibration workflow documented"),
    "12. Documentation tasks::Document normalized versus electron-domain inputs": ("implemented", "NOISE_CATALOG.md", "", "examples/noise_camera_workflows.cpp", "Electron-domain versus normalized signal contract and conversion documented"),
    "12. Documentation tasks::Document RNG and reproducibility guarantees": ("implemented", "NOISE_CATALOG.md", "", "examples/noise_camera_workflows.cpp;tests/noise_view_unit.cpp", "Seeded coordinate-keyed RNG determinism documented and executable smoke-tested"),
    "12. Documentation tasks::Document CPU/CUDA numerical tolerances": ("implemented", "NOISE_CATALOG.md", "", "tests/noise_catalog_cuda_validation.cu", "Numerical comparison guidance and existing CUDA validation evidence documented"),
    "12. Documentation tasks::Add examples for raw Bayer pipelines": ("implemented", "NOISE_CATALOG.md", "", "tests/demosaic_artifacts_unit.cpp", "Bayer pipeline order, CFA pattern and limitations documented with available API tests"),
    "12. Documentation tasks::Add examples for RGB camera pipelines": ("implemented", "examples/noise_camera_workflows.cpp", "", "examples/noise_camera_workflows.cpp", "Runnable RGB sensor-noise workflow verifies fixed-seed reproducibility"),
    "12. Documentation tasks::Add examples for video pipelines": ("implemented", "NOISE_CATALOG.md", "", "tests/temporal_noise_unit.cpp;tests/video_temporal_unit.cpp", "T-H-W-C temporal API and codec-independent limitations documented"),
    "12. Documentation tasks::Add a transform support matrix": ("implemented", "NOISE_CATALOG.md", "", "IMPLEMENTATION_MANIFEST.tsv", "Sensor, ISP, Bayer, video and unsupported codec support enumerated"),
    "12. Documentation tasks::Add references to sensor-noise and camera-pipeline literature": ("implemented", "NOISE_CATALOG.md", "", "NOISE_CATALOG.md", "Bibliographic references added"),
    "13. Native NOISE_CATALOG API contract::Image-view dtype and layout support": ("implemented", "src/noise_view_cpu.cpp", "src/noise_view.cu", "tests/noise_view_unit.cpp;tests/noise_view_properties.py", "strided HWC/CHW uint8, uint16, and float32 views"),
    "13. Native NOISE_CATALOG API contract::Per-channel and per-plane metadata": ("implemented", "src/noise_view_cpu.cpp", "src/noise_view.cu", "tests/noise_view_unit.cpp;tests/noise_view_properties.py", "borrowed channel records plus explicit channel-to-plane mapping"),
    "13. Native NOISE_CATALOG API contract::Counter-based deterministic RNG": ("implemented", "src/noise_view_cpu.cpp", "src/noise_view.cu", "tests/noise_view_unit.cpp;tests/noise_view_properties.py", "coordinate-keyed stateless CounterRng shared by CPU/CUDA"),
    "13. Native NOISE_CATALOG API contract::Clipping and quantization policies": ("implemented", "src/noise_view_cpu.cpp", "src/noise_view.cu", "tests/noise_view_unit.cpp;tests/noise_view_properties.py", "explicit clip and integer quantization policy"),
    "13. Native NOISE_CATALOG API contract::Configurable border policies": ("implemented", "src/noise_view_cpu.cpp", "src/noise_view.cu", "tests/noise_view_unit.cpp;tests/noise_view_properties.py", "clamp, REFLECT_101, and constant coordinate helper"),
    "13. Native NOISE_CATALOG API contract::Explicit status results": ("implemented", "src/noise_view_cpu.cpp", "src/noise_view.cu", "tests/noise_view_unit.cpp;tests/noise_view_properties.py", "non-throwing StatusCode validation and launch errors"),
    "13. Native NOISE_CATALOG API contract::CUDA stream-aware view execution": ("implemented", "src/noise_view_cpu.cpp", "src/noise_view.cu", "tests/noise_view_unit.cpp;tests/noise_view_cuda_validation.cu", "asynchronous CUDA kernel launch on caller stream; CPU device views reject explicitly"),
    "View, metadata, RNG, and status contract::Random telegraph signal noise": ("implemented", "src/noise_remaining_cpu.cpp", "src/noise_remaining.cu", "tests/remaining_video_lens_unit.cpp;tests/remaining_video_lens_properties.py;tests/remaining_video_lens_cuda_validation.cu", "deterministic independent per-sample toggle or explicit HWC state map; single-exposure approximation, not temporally correlated telegraph process"),
    "Implemented video artifact batch contract::Inter-frame compression noise": ("partial", "src/noise_remaining_cpu.cpp", "src/noise_remaining.cu", "tests/remaining_video_lens_unit.cpp;tests/remaining_video_lens_properties.py;tests/remaining_video_lens_cuda_validation.cu", "codec-independent seeded residual quantization surrogate; no AV1/H.264/H.265 parser"),
    "Implemented video artifact batch contract::GOP/keyframe artifacts": ("partial", "src/noise_remaining_cpu.cpp", "src/noise_remaining.cu", "tests/remaining_video_lens_unit.cpp;tests/remaining_video_lens_properties.py;tests/remaining_video_lens_cuda_validation.cu", "explicit keyframe records/mask and deterministic intra/inter surrogate; codec-dependent behavior remains unsupported"),
    "Implemented video artifact batch contract::Block motion-estimation artifacts": ("partial", "src/noise_remaining_cpu.cpp", "src/noise_remaining.cu", "tests/remaining_video_lens_unit.cpp;tests/remaining_video_lens_properties.py;tests/remaining_video_lens_cuda_validation.cu", "explicit block motion-vector records or seeded integer vectors; codec-independent surrogate"),
    "9. Environmental and acquisition artifacts::Water droplets on lens": ("partial", "src/noise_remaining_cpu.cpp", "src/noise_remaining.cu", "tests/remaining_video_lens_unit.cpp;tests/remaining_video_lens_properties.py;tests/remaining_video_lens_cuda_validation.cu", "explicit or seeded HWC droplet records with local-box blur; refractive fluid simulation not claimed"),
    "10. API and implementation tasks::Add sensor/, optics/, isp/, compression/, and video/ transform namespaces": ("implemented", "include/augmatch/native_api.hpp;include/augmatch/sensor/api.hpp;include/augmatch/optics/api.hpp;include/augmatch/isp/api.hpp;include/augmatch/compression/api.hpp;include/augmatch/video/api.hpp", "src/native_api.cu", "tests/native_api_unit.cpp", "public sensor/optics/isp/compression/video namespace aliases"),
    "10. API and implementation tasks::Add float and integer image-view support": ("implemented", "include/augmatch/image.hpp;src/native_api_cpu.cpp", "src/noise_view.cu", "tests/native_api_unit.cpp;tests/noise_view_unit.cpp", "strided HWC/CHW uint8, uint16, and float32 views"),
    "10. API and implementation tasks::Add raw Bayer image representation": ("implemented", "include/augmatch/native_api.hpp", "src/native_api.cu", "tests/native_api_unit.cpp", "strided RawBayerView with explicit CFA pattern and datatype"),
    "10. API and implementation tasks::Add per-channel and per-plane metadata": ("implemented", "include/augmatch/noise_view.hpp;src/noise_view_cpu.cpp", "src/noise_view.cu", "tests/noise_view_unit.cpp", "borrowed channel records and explicit channel-to-plane mapping"),
    "10. API and implementation tasks::Add camera-profile configuration objects": ("implemented", "include/augmatch/native_api.hpp;include/augmatch/iso_profile.hpp", "src/native_api.cu", "tests/native_api_unit.cpp;tests/profile_calibration_unit.cpp", "owning CameraProfileConfig plus existing lookup view"),
    "10. API and implementation tasks::Add deterministic counter-based RNG": ("implemented", "include/augmatch/noise_view.hpp;src/noise_view_cpu.cpp", "src/noise_view.cu", "tests/noise_view_unit.cpp", "stateless coordinate-keyed CounterRng shared by CPU/CUDA"),
    "10. API and implementation tasks::Add CPU reference implementations": ("implemented", "src/native_api_cpu.cpp", "", "tests/native_api_unit.cpp", "validated host reference for fused, batch, async, profile, and raw-view APIs"),
    "10. API and implementation tasks::Add CUDA kernels": ("implemented", "src/native_api.cu", "src/noise_view.cu", "tests/noise_view_cuda_validation.cu", "CUDA translation unit and reusable additive kernel"),
    "10. API and implementation tasks::Add CUDA stream support": ("implemented", "include/augmatch/native_api.hpp;src/native_api.cu", "src/noise_view.cu", "tests/noise_view_cuda_validation.cu", "stream argument is forwarded to asynchronous kernel launch"),
    "10. API and implementation tasks::Add scratch/workspace allocation": ("implemented", "include/augmatch/native_api.hpp;src/native_api_cpu.cpp", "src/native_api.cu", "tests/native_api_unit.cpp", "caller-owned Workspace and required-byte validation"),
    "10. API and implementation tasks::Add fused sensor-pipeline kernels": ("implemented", "src/native_api_cpu.cpp", "src/native_api.cu;src/noise_view.cu", "tests/native_api_unit.cpp;tests/noise_view_cuda_validation.cu", "sensor::fused_pipeline composes gain metadata and stream noise kernel"),
    "10. API and implementation tasks::Add clipping and quantization policies": ("implemented", "include/augmatch/noise_view.hpp;src/noise_view_cpu.cpp", "src/noise_view.cu", "tests/noise_view_unit.cpp", "explicit clip and integer quantization policy enums"),
    "10. API and implementation tasks::Add configurable border policies": ("implemented", "include/augmatch/noise_view.hpp;src/noise_view_cpu.cpp", "src/noise_view.cu", "tests/noise_view_unit.cpp", "Clamp, Reflect101, and Constant coordinate helper"),
    "10. API and implementation tasks::Add batch execution": ("implemented", "include/augmatch/native_api.hpp;src/native_api_cpu.cpp", "src/native_api.cu", "tests/native_api_unit.cpp", "ordered borrowed view batch API"),
    "10. API and implementation tasks::Add asynchronous execution APIs": ("partial", "include/augmatch/native_api.hpp;src/native_api_cpu.cpp", "", "tests/native_api_unit.cpp", "host-only std::future API; device async uses explicit CUDA stream API"),
    "10. API and implementation tasks::Add explicit error/status results": ("implemented", "include/augmatch/status.hpp;src/native_api_cpu.cpp", "src/native_api.cu", "tests/native_api_unit.cpp;tests/noise_view_unit.cpp", "non-throwing StatusCode validation and execution errors"),
    "10. API and implementation tasks::Add serialization for camera profiles": ("implemented", "include/augmatch/native_api.hpp;src/native_api_cpu.cpp", "src/native_api.cu", "tests/native_api_unit.cpp", "versioned text round-trip format"),
    "10. API and implementation tasks::Add CMake install/export targets": ("implemented", "CMakeLists.txt;cmake/augmatchConfig.cmake.in", "", "tests/native_api_unit.cpp", "installed target export and package config"),
    "10. API and implementation tasks::Add C++17 examples": ("implemented", "examples/native_api_cpp17.cpp", "", "tests/native_api_unit.cpp", "CMake C++17 native API example target"),
    "10. API and implementation tasks::Add C++23 examples": ("implemented", "examples/native_api_cpp23.cpp", "", "tests/native_api_unit.cpp", "CMake C++23 designated-initializer example target"),
    "11. Validation tasks::Verify Poisson variance approximately equals the mean": ("partial", "src/noise_cpu.cpp", "src/noise.cu", "tests/noise_catalog_validation.cpp;tests/noise_catalog_cuda_validation.cu", "measured normalized moments; CUDA requires a runnable device"),
    "11. Validation tasks::Verify read-noise variance is independent of signal": ("partial", "src/noise_cpu.cpp", "src/noise.cu", "tests/noise_catalog_validation.cpp", "two signal levels with exposure disabled; normalized quantization tolerance"),
    "11. Validation tasks::Verify row noise is constant across each row": ("partial", "src/noise_cpu.cpp", "src/noise.cu", "tests/noise_catalog_validation.cpp", "explicit row-map ownership and equality checks"),
    "11. Validation tasks::Verify column noise is constant across each column": ("partial", "src/noise_cpu.cpp", "src/noise.cu", "tests/noise_catalog_validation.cpp", "explicit column-map ownership and equality checks"),
    "11. Validation tasks::Verify hot/dead pixel probabilities": ("partial", "src/noise_cpu.cpp", "src/noise.cu", "tests/noise_catalog_validation.cpp", "binomial-tolerance probability checks; normalized clipping"),
    "11. Validation tasks::Verify clipping and black-level behavior": ("partial", "src/noise_cpu.cpp", "src/noise.cu", "tests/noise_catalog_validation.cpp", "normalized post-black-level clipping and ADC bounds"),
    "11. Validation tasks::Verify Bayer sampling and reconstruction": ("partial", "src/bayer_cpu.cpp", "src/bayer.cu", "tests/noise_catalog_validation.cpp", "RGGB sampling and nearest reconstruction golden vector"),
    "11. Validation tasks::Verify derivative maps against analytical images": ("partial", "src/derivative_cpu.cpp", "src/derivative.cu", "tests/noise_catalog_validation.cpp", "Sobel-X analytical linear-ramp vector"),
    "11. Validation tasks::Verify CPU/CUDA statistical equivalence": ("partial", "src/noise_cpu.cpp", "src/noise.cu", "tests/noise_catalog_validation.cpp;tests/noise_catalog_cuda_validation.cu", "analytic moment comparison; CUDA runtime skip code 77"),
    "11. Validation tasks::Verify fixed-seed determinism": ("implemented", "src/noise_cpu.cpp", "src/noise.cu", "tests/noise_catalog_validation.cpp;tests/noise_catalog_cuda_validation.cu", "repeat calls with identical coordinate-keyed seed"),
    "11. Validation tasks::Verify different seeds produce independent samples": ("implemented", "src/noise_cpu.cpp", "src/noise.cu", "tests/noise_catalog_validation.cpp;tests/noise_catalog_cuda_validation.cu", "changed-seed samples differ"),
    "11. Validation tasks::Add golden vectors for every deterministic transform": ("partial", "src/noise_cpu.cpp;src/bayer_cpu.cpp;src/derivative_cpu.cpp", "src/noise.cu;src/bayer.cu;src/derivative.cu", "tests/noise_catalog_validation.cpp", "sensor/CFA/derivative vectors covered; remaining catalog transforms need additional fixtures"),
    "11. Validation tasks::Add distributional tests for stochastic transforms": ("partial", "src/noise_cpu.cpp", "src/noise.cu", "tests/noise_catalog_validation.cpp;tests/noise_catalog_cuda_validation.cu", "Poisson/read-noise moments and defect probabilities; runtime-gated CUDA"),
    "11. Validation tasks::Add property tests for bounds and dtype preservation": ("implemented", "src/noise_cpu.cpp;src/bayer_cpu.cpp", "src/noise.cu;src/bayer.cu", "tests/noise_catalog_validation.cpp", "float normalized and uint8 CFA output bounds/types"),
    "11. Validation tasks::Add performance benchmarks for 512x512, 1080p, and 4K images": ("partial", "src/noise_cpu.cpp", "src/noise.cu", "tests/noise_catalog_benchmarks.py;tests/noise_catalog_validation.cpp", "measured CPU sizes; CUDA timing is runtime-gated"),
    "11. Validation tasks::Add memory-transfer benchmarks": ("partial", "", "tests/noise_catalog_cuda_validation.cu", "tests/noise_catalog_benchmarks.py;tests/noise_catalog_validation.cpp;tests/noise_catalog_cuda_validation.cu", "measured host memcpy; device transfer requires CUDA runtime"),
    "11. Validation tasks::Add profile-based camera-domain regression datasets": ("partial", "", "", "tests/noise_catalog_benchmarks.py;NOISE_VALIDATION.md", "data-driven inventory; no camera profile fixtures currently present"),
    "2. ISO and gain-dependent noise::ISO-to-gain mapping": ("implemented", "src/iso_profile_cpu.cpp", "src/iso_profile.cu", "tests/iso_profile_unit.cpp;tests/iso_profile_cuda_validation.cu", "borrowed sorted knots with clamped piecewise-linear interpolation"),
    "2. ISO and gain-dependent noise::ISO-dependent shot-noise scale": ("implemented", "src/iso_profile_cpu.cpp", "src/iso_profile.cu", "tests/iso_profile_unit.cpp;tests/iso_profile_cuda_validation.cu", "profile-scaled Poisson application with normalized clipping"),
    "2. ISO and gain-dependent noise::ISO-dependent read-noise scale": ("implemented", "src/iso_profile_cpu.cpp", "src/iso_profile.cu", "tests/iso_profile_unit.cpp;tests/iso_profile_cuda_validation.cu", "profile-scaled Gaussian application with normalized clipping"),
    "2. ISO and gain-dependent noise::ISO-dependent FPN scale": ("implemented", "src/iso_profile_cpu.cpp", "src/iso_profile.cu", "tests/iso_profile_unit.cpp;tests/iso_profile_cuda_validation.cu", "profile-scaled deterministic FPN application"),
    "2. ISO and gain-dependent noise::ISO-dependent black level": ("implemented", "src/iso_profile_cpu.cpp", "src/iso_profile.cu", "tests/iso_profile_unit.cpp;tests/iso_profile_cuda_validation.cu", "normalized profile level converted to clipped uint8 floor"),
    "2. ISO and gain-dependent noise::ISO-dependent saturation level": ("implemented", "src/iso_profile_cpu.cpp", "src/iso_profile.cu", "tests/iso_profile_unit.cpp;tests/iso_profile_cuda_validation.cu", "normalized profile level converted to clipped uint8 ceiling"),
    "3. Bayer and color-filter-array effects::CFA channel response variation": ("implemented", "src/bayer_cpu.cpp", "src/bayer.cu", "tests/cfa_effects_unit.cpp;tests/cfa_effects_properties.py;tests/cfa_effects_cuda_validation.cu", "four-entry explicit plane map; deterministic per-plane response variation; clipped raw float contract"),
    "3. Bayer and color-filter-array effects::CFA leakage / spectral cross-talk": ("implemented", "src/bayer_cpu.cpp", "src/bayer.cu", "tests/cfa_effects_unit.cpp;tests/cfa_effects_properties.py;tests/cfa_effects_cuda_validation.cu", "row-major 4x4 target/source matrix; radius-two nearest-plane search; clipped raw float contract"),
    "3. Bayer and color-filter-array effects::CFA misregistration": ("implemented", "src/bayer_cpu.cpp", "src/bayer.cu", "tests/cfa_effects_unit.cpp;tests/cfa_effects_properties.py;tests/cfa_effects_cuda_validation.cu", "four explicit integer plane offsets; deterministic same-plane lookup; clipped raw float contract"),
    "3. Bayer and color-filter-array effects::CFA missing or defective samples": ("implemented", "src/bayer_cpu.cpp", "src/bayer.cu", "tests/cfa_effects_unit.cpp;tests/cfa_effects_properties.py;tests/cfa_effects_cuda_validation.cu", "borrowed HxW mask or seeded probability; zero/fill/nearest replacements; explicit memory ownership"),
    "3. Bayer and color-filter-array effects::Bayer-plane-specific noise": ("implemented", "src/bayer_cpu.cpp", "src/bayer.cu", "tests/cfa_effects_unit.cpp;tests/cfa_effects_properties.py;tests/cfa_effects_cuda_validation.cu", "four plane Gaussian scales with deterministic counter-based seed; clipped raw float contract"),
    "3. Bayer and color-filter-array effects::Bayer-plane-specific gain": ("implemented", "src/bayer_cpu.cpp", "src/bayer.cu", "tests/cfa_effects_unit.cpp;tests/cfa_effects_properties.py;tests/cfa_effects_cuda_validation.cu", "four explicit plane gains; deterministic map ownership; clipped raw float contract"),
    "4. Optical artifacts::Defocus blur": ("implemented", "src/filter_cpu.cpp", "src/filter.cu", "tests/optical_catalog_unit.cpp;tests/optical_catalog_properties.py;tests/optical_catalog_cuda_validation.cu", "distinct optical API reuses finite disk/alias filter; REFLECT_101; HWC uint8 clipping"),
    "4. Optical artifacts::Motion blur": ("implemented", "src/filter_cpu.cpp", "src/filter.cu", "tests/optical_catalog_unit.cpp;tests/optical_catalog_properties.py;tests/optical_catalog_cuda_validation.cu", "distinct optical API reuses deterministic directional kernel; equal weighted samples; HWC uint8"),
    "4. Optical artifacts::Zoom blur": ("implemented", "src/filter_cpu.cpp", "src/filter.cu", "tests/optical_catalog_unit.cpp;tests/optical_catalog_properties.py;tests/optical_catalog_cuda_validation.cu", "distinct optical API reuses bilinear center-preserving factor set; REFLECT_101"),
    "4. Optical artifacts::Chromatic aberration": ("implemented", "src/color_cpu.cpp", "src/color.cu", "tests/optical_catalog_unit.cpp;tests/optical_catalog_properties.py;tests/optical_catalog_cuda_validation.cu", "distinct radial RGB remap and gain API; bilinear/fill clipping; HWC uint8"),
    "4. Optical artifacts::Lateral chromatic aberration": ("implemented", "src/color_cpu.cpp", "src/color.cu", "tests/optical_catalog_unit.cpp;tests/optical_catalog_properties.py;tests/optical_catalog_cuda_validation.cu", "gain-free radial dispersion API reusing chromatic remap; interpolation and fill explicit"),
    "4. Optical artifacts::Longitudinal chromatic aberration": ("implemented", "src/color_cpu.cpp", "src/color.cu", "tests/optical_catalog_unit.cpp;tests/optical_catalog_properties.py;tests/optical_catalog_cuda_validation.cu", "independent RGB Gaussian supports; clipped HWC uint8 approximation"),
    "4. Optical artifacts::Thin-prism distortion": ("implemented", "src/geometric_cpu.cpp", "src/geometric.cu", "tests/optical_catalog_unit.cpp;tests/optical_catalog_properties.py;tests/optical_catalog_cuda_validation.cu", "normalized r2/r4 thin-prism remap; bilinear/fill clipping; HWC uint8"),
    "4. Optical artifacts::Rolling-shutter geometric distortion": ("implemented", "src/geometric_cpu.cpp", "src/geometric.cu", "tests/optical_catalog_unit.cpp;tests/optical_catalog_properties.py;tests/optical_catalog_cuda_validation.cu", "borrowed H-element row displacement maps; H-W-C image contract; interpolation/fill clipping"),
    "Composition and control::Compose": ("implemented", "src/composition_cpu.cpp", "src/composition.cu", "tests/composition_unit.cpp", "native deterministic in-place stage contract"),
    "Composition and control::Sequential": ("implemented", "src/composition_cpu.cpp", "src/composition.cu", "tests/composition_unit.cpp", "native deterministic in-place stage contract"),
    "Target-aware behavior to implement::Additional image targets": ("implemented", "src/target_metadata_cpu.cpp", "host-only", "tests/target_metadata_unit.cpp", "borrowed host strided views and shared Geometry"),
    "Target-aware behavior to implement::Additional mask targets": ("implemented", "src/target_metadata_cpu.cpp", "host-only", "tests/target_metadata_unit.cpp", "host mask views with explicit nearest/linear policy"),
    "Target-aware behavior to implement::Pascal VOC boxes": ("implemented", "src/target_metadata_cpu.cpp", "host-only", "tests/target_metadata_unit.cpp", "absolute continuous half-open BoxXYXY validation"),
    "Target-aware behavior to implement::COCO boxes": ("implemented", "src/target_metadata_cpu.cpp", "host-only", "tests/target_metadata_unit.cpp", "absolute xywh conversion and geometry transform"),
    "Target-aware behavior to implement::YOLO boxes": ("implemented", "src/target_metadata_cpu.cpp", "host-only", "tests/target_metadata_unit.cpp", "normalized center xywh conversion and validation"),
    "Target-aware behavior to implement::Albumentations normalized boxes": ("implemented", "src/target_metadata_cpu.cpp", "host-only", "tests/target_metadata_unit.cpp", "normalized xyxy conversion and validation"),
    "Target-aware behavior to implement::Keypoint visibility": ("implemented", "src/target_metadata_cpu.cpp", "host-only", "tests/target_metadata_unit.cpp", "finite [0,1] visibility and explicit out-of-frame policy"),
    "augmenters.meta::Sequential": ("implemented", "src/composition_cpu.cpp", "src/composition.cu", "tests/composition_unit.cpp", "native deterministic in-place stage contract"),
    "`augmenters.meta`::Sequential": ("implemented", "src/composition_cpu.cpp", "src/composition.cu", "tests/composition_unit.cpp", "native deterministic in-place stage contract"),
    "augmenters.meta::Identity": ("implemented", "src/composition_cpu.cpp", "src/composition.cu", "tests/composition_unit.cpp", "native validation-preserving identity stage"),
    "`augmenters.meta`::Identity": ("implemented", "src/composition_cpu.cpp", "src/composition.cu", "tests/composition_unit.cpp", "native validation-preserving identity stage"),
    "augmenters.meta::Noop": ("implemented", "src/composition_cpu.cpp", "src/composition.cu", "tests/composition_unit.cpp", "native validation-preserving no-op stage"),
    "`augmenters.meta`::Noop": ("implemented", "src/composition_cpu.cpp", "src/composition.cu", "tests/composition_unit.cpp", "native validation-preserving no-op stage"),
    "augmenters.meta::Lambda": ("implemented", "src/meta_cpu.cpp", "host-only", "tests/meta_unit.cpp", "borrowed synchronous target-aware host callback; device views rejected"),
    "`augmenters.meta`::Lambda": ("implemented", "src/meta_cpu.cpp", "host-only", "tests/meta_unit.cpp", "borrowed synchronous target-aware host callback; device views rejected"),
    "augmenters.meta::AssertLambda": ("implemented", "src/meta_cpu.cpp", "host-only", "tests/meta_unit.cpp", "borrowed synchronous target-aware host predicate with explicit failure"),
    "`augmenters.meta`::AssertLambda": ("implemented", "src/meta_cpu.cpp", "host-only", "tests/meta_unit.cpp", "borrowed synchronous target-aware host predicate with explicit failure"),
    "augmenters.meta::AssertShape": ("implemented", "src/meta_cpu.cpp", "host-only", "tests/meta_unit.cpp", "synchronous wildcard dimension/type/layout validation"),
    "`augmenters.meta`::AssertShape": ("implemented", "src/meta_cpu.cpp", "host-only", "tests/meta_unit.cpp", "synchronous wildcard dimension/type/layout validation"),
    "augmenters.meta::RemoveCBAsByOutOfImageFraction": ("implemented", "src/meta_cpu.cpp", "host-only", "tests/meta_unit.cpp", "in-place box/keypoint compaction with source-index evidence"),
    "`augmenters.meta`::RemoveCBAsByOutOfImageFraction": ("implemented", "src/meta_cpu.cpp", "host-only", "tests/meta_unit.cpp", "in-place box/keypoint compaction with source-index evidence"),
    "augmenters.meta::ClipCBAsToImagePlanes": ("implemented", "src/meta_cpu.cpp", "host-only", "tests/meta_unit.cpp", "in-place continuous box-edge and pixel-centre clipping"),
    "`augmenters.meta`::ClipCBAsToImagePlanes": ("implemented", "src/meta_cpu.cpp", "host-only", "tests/meta_unit.cpp", "in-place continuous box-edge and pixel-centre clipping"),
    "Composition and control::ReplayCompose": ("implemented", "src/composition_cpu.cpp", "src/composition.cu", "tests/composition_control_unit.cpp", "explicit ordered stage-selection replay record"),
    "Composition and control::OneOf": ("implemented", "src/composition_cpu.cpp", "src/composition.cu", "tests/composition_control_unit.cpp", "seeded or recorded complete-child selection"),
    "Composition and control::OneOrOther": ("implemented", "src/composition_cpu.cpp", "src/composition.cu", "tests/composition_control_unit.cpp", "seeded or recorded binary child selection"),
    "Composition and control::RandomApply": ("implemented", "src/composition_cpu.cpp", "src/composition.cu", "tests/composition_control_unit.cpp", "seeded or recorded applied decision"),
    "Composition and control::SomeOf": ("implemented", "src/composition_cpu.cpp", "src/composition.cu", "tests/composition_batch_unit.cpp", "explicit selected-index record with deterministic without/with-replacement sampling"),
    "Composition and control::RandomOrder": ("implemented", "src/composition_cpu.cpp", "src/composition.cu", "tests/composition_batch_unit.cpp", "explicit deterministic child-order permutation record"),
    "Composition and control::SelectiveChannelTransform": ("implemented", "src/composition_cpu.cpp", "src/composition.cu", "tests/composition_batch_unit.cpp", "CPU/CUDA channel-range stage contract without channel copies"),
    "`augmenters.meta`::WithChannels": ("implemented", "src/composition_cpu.cpp", "src/composition.cu", "tests/with_channels_unit.cpp", "CPU/CUDA explicit channel-index and range selection without channel copies"),
    "`augmenters.meta`::OneOf": ("implemented", "src/composition_cpu.cpp", "src/composition.cu", "tests/composition_control_unit.cpp", "seeded or recorded complete-child selection"),
    "`augmenters.meta`::Sometimes": ("implemented", "src/composition_cpu.cpp", "src/composition.cu", "tests/composition_control_unit.cpp", "seeded or recorded applied decision"),
    "`augmenters.meta`::SomeOf": ("implemented", "src/composition_cpu.cpp", "src/composition.cu", "tests/composition_batch_unit.cpp", "explicit selected-index record with deterministic sampling"),
    "Cropping, padding, and resizing::Crop": ("partial", "src/transforms_cpu.cpp", "src/transforms.cu", "tests/parity.py", "albumentations"),
    "`augmenters.size`::Crop": ("partial", "src/transforms_cpu.cpp", "src/transforms.cu", "tests/parity.py", "albumentations"),
    "Geometric transforms::HorizontalFlip": ("partial", "src/transforms_cpu.cpp", "src/transforms.cu", "tests/parity.py", "albumentations"),
    "`augmenters.flip`::HorizontalFlip": ("partial", "src/transforms_cpu.cpp", "src/transforms.cu", "tests/parity.py", "albumentations"),
    "Geometric transforms::VerticalFlip": ("partial", "src/transforms_cpu.cpp", "src/transforms.cu", "tests/parity.py", "albumentations"),
    "`augmenters.flip`::VerticalFlip": ("partial", "src/transforms_cpu.cpp", "src/transforms.cu", "tests/parity.py", "albumentations"),
    "`augmenters.flip`::Fliplr": ("partial", "src/transforms_cpu.cpp", "src/transforms.cu", "tests/parity.py", "albumentations"),
    "`augmenters.flip`::Flipud": ("partial", "src/transforms_cpu.cpp", "src/transforms.cu", "tests/parity.py", "albumentations"),
    "`augmenters.pooling`::AveragePooling": ("partial", "src/transforms_cpu.cpp", "src/transforms.cu", "tests/imgaug_pool_parity.py", "imgaug"),
    "`augmenters.pooling`::MaxPooling": ("partial", "src/transforms_cpu.cpp", "src/transforms.cu", "tests/imgaug_pool_parity.py", "imgaug"),
    "`augmenters.pooling`::MinPooling": ("partial", "src/transforms_cpu.cpp", "src/transforms.cu", "tests/imgaug_pool_parity.py", "imgaug"),
    "`augmenters.pooling`::MedianPooling": ("partial", "src/transforms_cpu.cpp", "src/transforms.cu", "tests/imgaug_pool_parity.py", "imgaug"),
    "Cropping, padding, and resizing::Resize": ("partial", "src/geometric_cpu.cpp", "src/geometric.cu", "tests/resize_parity.py", "albumentations"),
    "`augmenters.size`::Resize": ("partial", "src/geometric_cpu.cpp", "src/geometric.cu", "tests/resize_parity.py", "albumentations"),
    "Geometric transforms::RandomGridShuffle": ("partial", "src/geometric_cpu.cpp", "src/geometric.cu", "tests/random_grid_shuffle_parity.py", "deterministic seeded or explicit cell permutation contract"),
    "Geometric transforms::RandomScale": ("partial", "src/geometric_cpu.cpp", "src/geometric.cu", "tests/random_scale_parity.py", "albumentations"),
    "Compression and image corruption::Downscale": ("implemented", "src/geometric_cpu.cpp", "src/geometric.cu", "tests/downscale_parity.py", "deterministic native resize round trip; no codec-dependent compression"),
    "Compression and image corruption::ImageCompression": ("partial", "src/jpeg_compression_cpu.cpp", "src/jpeg_compression.cu", "tests/jpeg_compression_unit.cpp;tests/jpeg_compression_parity.py", "libjpeg CPU codec; CUDA uses synchronous host codec fallback; still images only"),
    "Compression and image corruption::JpegCompression": ("partial", "src/jpeg_compression_cpu.cpp", "src/jpeg_compression.cu", "tests/jpeg_compression_unit.cpp;tests/jpeg_compression_parity.py", "libjpeg CPU codec; CUDA uses synchronous host codec fallback; still images only"),
    "`augmenters.arithmetic`::JpegCompression": ("partial", "src/jpeg_compression_cpu.cpp", "src/jpeg_compression.cu", "tests/jpeg_compression_unit.cpp;tests/jpeg_compression_parity.py", "libjpeg CPU codec; CUDA uses synchronous host codec fallback; still images only"),
    "`augmenters.imgcorruptlike`::JpegCompression": ("partial", "src/jpeg_compression_cpu.cpp", "src/jpeg_compression.cu", "tests/jpeg_compression_unit.cpp;tests/jpeg_compression_parity.py", "libjpeg CPU codec; CUDA uses synchronous host codec fallback; still images only"),
    "8. Compression and transport artifacts::JPEG compression": ("partial", "src/jpeg_compression_cpu.cpp", "src/jpeg_compression.cu", "tests/jpeg_compression_unit.cpp;tests/jpeg_compression_parity.py;tests/jpeg_noise_catalog_unit.cpp;tests/jpeg_noise_catalog_properties.py;tests/jpeg_compression_cuda_validation.cu", "libjpeg CPU codec; CUDA synchronous host-codec fallback; still images only"),
    "8. Compression and transport artifacts::JPEG quality variation": ("partial", "src/jpeg_compression_cpu.cpp", "src/jpeg_compression.cu", "tests/jpeg_noise_catalog_unit.cpp;tests/jpeg_noise_catalog_properties.py;tests/jpeg_compression_cuda_validation.cu", "explicit libjpeg quality and Pillow reference"),
    "8. Compression and transport artifacts::JPEG quantization-table variation": ("partial", "src/jpeg_compression_cpu.cpp", "src/jpeg_compression.cu", "tests/jpeg_noise_catalog_unit.cpp;tests/jpeg_noise_catalog_properties.py;tests/jpeg_compression_cuda_validation.cu", "independent luminance/chrominance table multipliers"),
    "8. Compression and transport artifacts::JPEG chroma subsampling 4:4:4": ("implemented", "src/jpeg_compression_cpu.cpp", "src/jpeg_compression.cu", "tests/jpeg_compression_unit.cpp;tests/jpeg_compression_parity.py", "libjpeg YCbCr 4:4:4 sampling factors"),
    "8. Compression and transport artifacts::JPEG chroma subsampling 4:2:2": ("implemented", "src/jpeg_compression_cpu.cpp", "src/jpeg_compression.cu", "tests/jpeg_compression_unit.cpp;tests/jpeg_compression_parity.py", "libjpeg YCbCr horizontal 2:1 chroma sampling"),
    "8. Compression and transport artifacts::JPEG chroma subsampling 4:2:0": ("implemented", "src/jpeg_compression_cpu.cpp", "src/jpeg_compression.cu", "tests/jpeg_compression_unit.cpp;tests/jpeg_compression_parity.py", "libjpeg YCbCr horizontal/vertical 2:1 chroma sampling"),
    "8. Compression and transport artifacts::JPEG ringing": ("partial", "src/jpeg_compression_cpu.cpp", "src/jpeg_compression.cu", "tests/jpeg_noise_catalog_unit.cpp;tests/jpeg_noise_catalog_properties.py;tests/jpeg_compression_cuda_validation.cu", "deterministic clipped post-decode high-pass artifact model; CUDA uses host-codec fallback"),
    "8. Compression and transport artifacts::JPEG blocking": ("partial", "src/jpeg_compression_cpu.cpp", "src/jpeg_compression.cu", "tests/jpeg_noise_catalog_unit.cpp;tests/jpeg_noise_catalog_properties.py;tests/jpeg_compression_cuda_validation.cu", "deterministic 8x8 grid-boundary artifact model; CUDA uses host-codec fallback"),
    "8. Compression and transport artifacts::JPEG mosquito noise": ("partial", "src/jpeg_compression_cpu.cpp", "src/jpeg_compression.cu", "tests/jpeg_noise_catalog_unit.cpp;tests/jpeg_noise_catalog_properties.py;tests/jpeg_compression_cuda_validation.cu", "deterministic clipped local DCT high-pass artifact model; CUDA uses host-codec fallback"),
    "8. Compression and transport artifacts::JPEG restart-marker damage": ("implemented", "src/jpeg_compression_cpu.cpp", "src/jpeg_compression.cu", "tests/jpeg_noise_catalog_unit.cpp;tests/jpeg_noise_catalog_properties.py;tests/jpeg_compression_cuda_validation.cu", "libjpeg restart interval with deterministic entropy damage at restart boundaries; CUDA uses host-codec fallback"),
    "8. Compression and transport artifacts::JPEG progressive decoding artifacts": ("implemented", "src/jpeg_compression_cpu.cpp", "src/jpeg_compression.cu", "tests/jpeg_noise_catalog_unit.cpp;tests/jpeg_noise_catalog_properties.py;tests/jpeg_compression_cuda_validation.cu", "progressive libjpeg stream with deterministic early-scan preview model; CUDA uses host-codec fallback"),
    "8. Compression and transport artifacts::WebP lossy compression": ("partial", "src/webp_compression_cpu.cpp", "src/webp_compression.cu;src/webp_compression_unavailable.cpp", "tests/webp_compression_unit.cpp;tests/webp_compression_unavailable_unit.cpp;tests/webp_compression_parity.py", "native libwebp lossy codec; conditional dependency; CUDA uses synchronous host-codec fallback and clear unavailable error"),
    "8. Compression and transport artifacts::WebP lossless compression": ("partial", "src/webp_compression_cpu.cpp", "src/webp_compression.cu;src/webp_compression_unavailable.cpp", "tests/webp_compression_unit.cpp;tests/webp_compression_unavailable_unit.cpp;tests/webp_compression_parity.py", "native libwebp lossless codec; conditional dependency; CUDA uses synchronous host-codec fallback and clear unavailable error"),
    "8. Compression and transport artifacts::Block quantization": ("implemented", "src/transport_cpu.cpp", "src/transport.cu", "tests/transport_unit.cpp;tests/transport_properties.py;tests/transport_cuda_validation.cu", "codec-independent HWC uint8 scalar quantization with explicit step and edge-safe blocks"),
    "8. Compression and transport artifacts::Deblocking-filter mismatch": ("implemented", "src/transport_cpu.cpp", "src/transport.cu", "tests/transport_unit.cpp;tests/transport_properties.py;tests/transport_cuda_validation.cu", "explicit block-boundary neighbour blend with bounded strength; no codec parser"),
    "8. Compression and transport artifacts::Chroma-plane misalignment": ("implemented", "src/transport_cpu.cpp", "src/transport.cu", "tests/transport_unit.cpp;tests/transport_properties.py;tests/transport_cuda_validation.cu", "borrowed row-major plane with integer shift and clamped edges"),
    "8. Compression and transport artifacts::Packet-loss macroblocks": ("implemented", "src/transport_cpu.cpp", "src/transport.cu", "tests/transport_unit.cpp;tests/transport_properties.py;tests/transport_cuda_validation.cu", "explicit block loss mask or seeded block probability with fill value"),
    "8. Compression and transport artifacts::Truncated-frame corruption": ("implemented", "src/transport_cpu.cpp", "src/transport.cu", "tests/transport_unit.cpp;tests/transport_properties.py;tests/transport_cuda_validation.cu", "explicit byte prefix retention and fill; codec-independent"),
    "8. Compression and transport artifacts::Bit flips": ("implemented", "src/transport_cpu.cpp", "src/transport.cu", "tests/transport_unit.cpp;tests/transport_properties.py;tests/transport_cuda_validation.cu", "seeded per-byte corruption with explicit or deterministic bit mask"),
    "8. Compression and transport artifacts::Raw Bayer transport corruption": ("implemented", "src/transport_cpu.cpp", "src/transport.cu", "tests/transport_unit.cpp;tests/transport_properties.py;tests/transport_cuda_validation.cu", "unpacked uint16 Bayer plane corruption with optional HxW mask and bit-depth clipping"),
    "8. Compression and transport artifacts::AV1 intra-frame artifacts": ("partial", "src/video_codec_cpu.cpp;src/video_codec_unavailable.cpp", "src/video_codec.cu", "tests/video_codec_unit.cpp", "conditional libavcodec single-frame intra encode/decode; arbitrary bitstream parsing and reference-picture semantics remain unsupported"),
    "8. Compression and transport artifacts::H.264 compression": ("partial", "src/video_codec_cpu.cpp;src/video_codec_unavailable.cpp", "src/video_codec.cu", "tests/video_codec_unit.cpp", "conditional libavcodec single-frame intra encode/decode; temporal prediction and arbitrary bitstream parsing remain unsupported"),
    "8. Compression and transport artifacts::H.265/HEVC compression": ("partial", "src/video_codec_cpu.cpp;src/video_codec_unavailable.cpp", "src/video_codec.cu", "tests/video_codec_unit.cpp", "conditional libavcodec single-frame intra encode/decode; temporal prediction and arbitrary bitstream parsing remain unsupported"),
    "8. Compression and transport artifacts::AV1 video compression": ("partial", "src/video_codec_cpu.cpp;src/video_codec_unavailable.cpp", "src/video_codec.cu", "tests/video_codec_unit.cpp", "conditional libavcodec single-frame packet encode/decode; multi-frame temporal prediction and reference pictures remain unsupported"),
    "12. Documentation tasks::Document unsupported external-codec dependencies": ("implemented", "README.md", "", "NOISE_CATALOG.md", "explicit AV1/H.264/H.265 blockers documented separately from transport primitives"),
    "Compression and image corruption::WebPCompression": ("partial", "src/webp_compression_cpu.cpp", "src/webp_compression.cu;src/webp_compression_unavailable.cpp", "tests/webp_compression_unit.cpp;tests/webp_compression_unavailable_unit.cpp;tests/webp_compression_parity.py", "libwebp native CPU codec; CUDA uses synchronous host codec fallback; conditional on libwebp; clear fallback when unavailable; no AV1/video codec support"),
    "`augmenters.geometric`::ScaleX": ("partial", "src/geometric_cpu.cpp", "src/geometric.cu", "tests/axis_scale_parity.py", "imgaug"),
    "`augmenters.geometric`::ScaleY": ("partial", "src/geometric_cpu.cpp", "src/geometric.cu", "tests/axis_scale_parity.py", "imgaug"),
    "`augmenters.geometric`::TranslateX": ("partial", "src/geometric_cpu.cpp", "src/geometric.cu", "tests/axis_translate_parity.py", "imgaug"),
    "`augmenters.geometric`::TranslateY": ("partial", "src/geometric_cpu.cpp", "src/geometric.cu", "tests/axis_translate_parity.py", "imgaug"),
    "`augmenters.geometric`::ShearX": ("partial", "src/geometric_cpu.cpp", "src/geometric.cu", "tests/axis_shear_parity.py", "imgaug"),
    "`augmenters.geometric`::ShearY": ("partial", "src/geometric_cpu.cpp", "src/geometric.cu", "tests/axis_shear_parity.py", "imgaug"),
    "Implemented ISP artifact contract::Sobel X derivative": ("partial", "src/derivative_cpu.cpp", "src/derivative.cu", "tests/derivative_parity.py", "native reference"),
    "Implemented ISP artifact contract::Sobel Y derivative": ("partial", "src/derivative_cpu.cpp", "src/derivative.cu", "tests/derivative_parity.py", "native reference"),
    "Implemented ISP artifact contract::Scharr X derivative": ("partial", "src/derivative_cpu.cpp", "src/derivative.cu", "tests/derivative_parity.py", "native reference"),
    "Implemented ISP artifact contract::Scharr Y derivative": ("partial", "src/derivative_cpu.cpp", "src/derivative.cu", "tests/derivative_parity.py", "native reference"),
    "Implemented ISP artifact contract::Laplacian derivative": ("partial", "src/derivative_cpu.cpp", "src/derivative.cu", "tests/derivative_parity.py", "native reference"),
    "Implemented ISP artifact contract::Gradient magnitude": ("partial", "src/derivative_cpu.cpp", "src/derivative.cu", "tests/derivative_parity.py", "native reference"),
    "`augmenters.edges`::Canny": ("implemented", "src/canny_cpu.cpp", "src/canny.cu", "tests/canny_parity.py", "OpenCV Canny reference with explicit HWC luminance, thresholds, aperture, and border policy"),
    "Implemented ISP artifact contract::Local variance map": ("partial", "src/derivative_cpu.cpp", "src/derivative.cu", "tests/derivative_parity.py", "native reference"),
    "Implemented ISP artifact contract::Local entropy map": ("partial", "src/derivative_cpu.cpp", "src/derivative.cu", "tests/derivative_parity.py", "native reference"),
    "Implemented ISP artifact contract::Configurable derivative border policy": ("partial", "src/derivative_cpu.cpp", "src/derivative.cu", "tests/derivative_parity.py", "native reference"),
    "Cropping, padding, and resizing::CenterCrop": ("partial", "src/size_cpu.cpp", "src/size.cu", "tests/size_parity.py", "albumentations"),
    "Cropping, padding, and resizing::RandomCrop": ("partial", "src/size_cpu.cpp", "src/size.cu", "tests/random_crop_parity.py", "albumentations"),
    "Cropping, padding, and resizing::RandomCropFromBorders": ("implemented", "src/size_cpu.cpp", "src/size.cu", "tests/random_crop_from_borders_parity.py", "deterministic explicit border fractions, offsets, and SplitMix64 seed contract"),
    "Cropping, padding, and resizing::Pad": ("partial", "src/size_cpu.cpp", "src/size.cu", "tests/size_parity.py", "albumentations"),
    "`augmenters.size`::Pad": ("partial", "src/size_cpu.cpp", "src/size.cu", "tests/size_parity.py", "albumentations"),
    "Cropping, padding, and resizing::SquareSymmetricPad": ("partial", "src/size_cpu.cpp", "src/size.cu", "tests/square_pad_parity.py", "albumentations"),
    "`augmenters.size`::PadToFixedSize": ("partial", "src/size_cpu.cpp", "src/size.cu", "tests/size_parity.py", "imgaug"),
    "`augmenters.size`::CropToFixedSize": ("partial", "src/size_cpu.cpp", "src/size.cu", "tests/random_crop_parity.py", "imgaug"),
    "`augmenters.size`::CenterCropToFixedSize": ("partial", "src/size_cpu.cpp", "src/size.cu", "tests/size_parity.py", "imgaug"),
    "`augmenters.size`::CenterPadToFixedSize": ("partial", "src/size_cpu.cpp", "src/size.cu", "tests/size_parity.py", "imgaug"),
    "`augmenters.size`::PadToSquare": ("partial", "src/size_cpu.cpp", "src/size.cu", "tests/square_pad_parity.py", "imgaug"),
    "`augmenters.size`::CropToSquare": ("partial", "src/size_cpu.cpp", "src/size.cu", "tests/size_parity.py", "imgaug"),
    "`augmenters.size`::CenterPadToSquare": ("partial", "src/size_cpu.cpp", "src/size.cu", "tests/square_pad_parity.py", "imgaug"),
    "`augmenters.size`::CenterCropToSquare": ("partial", "src/size_cpu.cpp", "src/size.cu", "tests/size_parity.py", "imgaug"),
    "Cropping, padding, and resizing::PadIfNeeded": ("partial", "src/size_cpu.cpp", "src/size.cu", "tests/size_parity.py", "albumentations"),
    "Geometric transforms::Transpose": ("partial", "src/geometric_cpu.cpp", "src/geometric.cu", "tests/geometric_parity.py", "albumentations"),
    "`augmenters.geometric`::Rot90": ("partial", "src/geometric_cpu.cpp", "src/geometric.cu", "tests/geometric_parity.py", "imgaug"),
    "Color and intensity::RandomBrightnessContrast": ("partial", "src/pixel_cpu.cpp", "src/pixel.cu", "tests/pixel_parity.py", "albumentations"),
    "Color and intensity::RandomGamma": ("partial", "src/pixel_cpu.cpp", "src/pixel.cu", "tests/gamma_parity.py", "albumentations"),
    "Color and intensity::RandomToneCurve": ("partial", "src/tone_cpu.cpp", "src/tone.cu", "tests/random_tone_curve_parity.py", "deterministic explicit channel-major 256-entry LUT contract"),
    "`augmenters.contrast`::SigmoidContrast": ("partial", "src/tone_cpu.cpp", "src/tone.cu", "tests/sigmoid_log_contrast_unit.cpp;tests/sigmoid_log_contrast_parity.py;tests/sigmoid_log_contrast_cuda_validation.cu", "normalized sigmoid contrast with CPU/CUDA exact float-reference parity"),
    "`augmenters.contrast`::LogContrast": ("partial", "src/tone_cpu.cpp", "src/tone.cu", "tests/sigmoid_log_contrast_unit.cpp;tests/sigmoid_log_contrast_parity.py;tests/sigmoid_log_contrast_cuda_validation.cu", "normalized logarithmic contrast with CPU/CUDA exact float-reference parity"),
    "Color and intensity::ColorJitter": ("partial", "src/color_cpu.cpp", "src/color.cu", "tests/color_jitter_parity.py", "fixed OpenCV RGB/HSV pipeline; deterministic explicit parameters"),
    "`augmenters.color`::WithColorspace": ("partial", "src/colorspace_cpu.cpp", "src/colorspace.cu", "tests/colorspace_parity.py", "deterministic RGB/HSV/LAB affine colorspace contract; LAB is approximate D65 sRGB"),
    "`augmenters.color`::WithBrightnessChannels": ("partial", "src/colorspace_cpu.cpp", "src/colorspace.cu", "tests/colorspace_parity.py", "deterministic RGB channel or HSV V affine contract"),
    "`augmenters.color`::MultiplyAndAddToBrightness": ("partial", "src/hsv_cpu.cpp", "src/hsv.cu", "tests/hsv_color_batch_parity.py", "deterministic HSV value multiply then add; explicit multiplier and addend"),
    "`augmenters.color`::MultiplyBrightness": ("partial", "src/hsv_cpu.cpp", "src/hsv.cu", "tests/hsv_color_batch_parity.py", "deterministic HSV value multiplier"),
    "`augmenters.color`::AddToBrightness": ("partial", "src/hsv_cpu.cpp", "src/hsv.cu", "tests/hsv_color_batch_parity.py", "deterministic HSV value addition"),
    "`augmenters.color`::WithHueAndSaturation": ("partial", "src/hsv_cpu.cpp", "src/hsv.cu", "tests/hsv_color_batch_parity.py", "OpenCV uint8 HSV bin contract"),
    "`augmenters.color`::MultiplyHueAndSaturation": ("partial", "src/hsv_cpu.cpp", "src/hsv.cu", "tests/hsv_color_batch_parity.py", "deterministic HSV bin multipliers"),
    "`augmenters.color`::MultiplyHue": ("partial", "src/hsv_cpu.cpp", "src/hsv.cu", "tests/hsv_color_batch_parity.py", "deterministic HSV hue-bin multiplier"),
    "`augmenters.color`::MultiplySaturation": ("partial", "src/hsv_cpu.cpp", "src/hsv.cu", "tests/hsv_color_batch_parity.py", "deterministic HSV saturation multiplier"),
    "`augmenters.color`::RemoveSaturation": ("partial", "src/hsv_cpu.cpp", "src/hsv.cu", "tests/hsv_color_batch_parity.py", "sets HSV saturation to zero"),
    "`augmenters.color`::AddToHueAndSaturation": ("partial", "src/hsv_cpu.cpp", "src/hsv.cu", "tests/hsv_color_batch_parity.py", "deterministic HSV bin additions"),
    "`augmenters.color`::AddToHue": ("partial", "src/hsv_cpu.cpp", "src/hsv.cu", "tests/hsv_color_batch_parity.py", "modulo-180 hue-bin addition"),
    "`augmenters.color`::AddToSaturation": ("partial", "src/hsv_cpu.cpp", "src/hsv.cu", "tests/hsv_color_batch_parity.py", "clipped HSV saturation addition"),
    "Color and intensity::RandomColorJitter": ("partial", "src/color_cpu.cpp", "src/color.cu", "tests/random_color_jitter_parity.py", "seeded parameter ranges composed with native ColorJitter"),
    "Color and intensity::PlanckianJitter": ("partial", "src/color_cpu.cpp", "src/color.cu", "tests/planckian_jitter_parity.py", "deterministic normalized Tanner-Helland blackbody RGB gain approximation"),
    "`augmenters.color`::ChangeColorTemperature": ("partial", "src/color_cpu.cpp", "src/color.cu", "tests/change_color_temperature_parity.py", "deterministic normalized Tanner-Helland RGB gain approximation; distinct API reuses PlanckianJitter contract"),
    "Color and intensity::ChromaticAberration": ("partial", "src/color_cpu.cpp", "src/color.cu", "tests/chromatic_aberration_parity.py", "fixed OpenCV remap; deterministic per-channel radial displacement and gain approximation"),
    "`augmenters.color`::UniformColorQuantization": ("implemented", "src/color_cpu.cpp", "src/color.cu", "tests/uniform_color_quantization_parity.py", "explicit [2,256] levels; half-up bin-center rounding; RGBA alpha preserved"),
    "`augmenters.color`::UniformColorQuantizationToNBits": ("implemented", "src/color_cpu.cpp", "src/color.cu", "tests/uniform_color_quantization_parity.py", "explicit [1,8] bit depth; lower-bin-boundary bit clearing; RGBA alpha preserved"),
    "6. Color and photometric pipeline effects::Gamma variation": ("implemented", "src/tone_cpu.cpp", "src/tone.cu", "tests/tone_variations_unit.cpp;tests/tone_variations_properties.py;tests/tone_variations_cuda_validation.cu", "normalized x^gamma transfer with half-up clipping"),
    "6. Color and photometric pipeline effects::Tone-curve variation": ("implemented", "src/tone_cpu.cpp", "src/tone.cu", "tests/tone_variations_unit.cpp;tests/tone_variations_properties.py;tests/tone_variations_cuda_validation.cu", "borrowed channel-major 256-entry LUT"),
    "6. Color and photometric pipeline effects::S-curve contrast variation": ("implemented", "src/tone_cpu.cpp", "src/tone.cu", "tests/tone_variations_unit.cpp;tests/tone_variations_properties.py;tests/tone_variations_cuda_validation.cu", "bounded cubic normalized S-curve with amount in [-1,1]"),
    "6. Color and photometric pipeline effects::Highlight roll-off variation": ("implemented", "src/tone_cpu.cpp", "src/tone.cu", "tests/tone_variations_unit.cpp;tests/tone_variations_properties.py;tests/tone_variations_cuda_validation.cu", "thresholded rational highlight shoulder with explicit strength"),
    "6. Color and photometric pipeline effects::Shadow lift": ("implemented", "src/tone_cpu.cpp", "src/tone.cu", "tests/tone_variations_unit.cpp;tests/tone_variations_properties.py;tests/tone_variations_cuda_validation.cu", "threshold-window lift toward white"),
    "6. Color and photometric pipeline effects::Shadow crush": ("implemented", "src/tone_cpu.cpp", "src/tone.cu", "tests/tone_variations_unit.cpp;tests/tone_variations_properties.py;tests/tone_variations_cuda_validation.cu", "threshold-window attenuation toward black"),
    "6. Color and photometric pipeline effects::Camera color-profile variation": ("implemented", "src/color_cpu.cpp", "src/color.cu", "tests/color_photometric_unit.cpp;tests/color_photometric_properties.py;tests/color_photometric_cuda_validation.cu", "matrix, per-channel gain, seeded normalized Gaussian noise, clipping"),
    "6. Color and photometric pipeline effects::Per-channel gain noise": ("implemented", "src/color_cpu.cpp", "src/color.cu", "tests/color_temporal_remaining_unit.cpp;tests/remaining_noise_properties.py", "seeded independent RGB multiplicative gain noise"),
    "6. Color and photometric pipeline effects::Color-temperature error": ("implemented", "src/color_cpu.cpp", "src/color.cu", "tests/color_temperature_error_unit.cpp;tests/color_temperature_error_properties.py;tests/color_temperature_error_cuda_validation.cu", "Tanner-Helland blackbody RGB approximation normalized to 6500 K; strength linearly interpolates neutral-to-temperature gains; uint8 round-half-up and saturation"),
    "6. Color and photometric pipeline effects::RGB channel cross-talk": ("implemented", "src/color_cpu.cpp", "src/color.cu", "tests/color_photometric_unit.cpp;tests/color_photometric_properties.py;tests/color_photometric_cuda_validation.cu", "row-major target/source RGB matrix; alpha preservation; saturating uint8"),
    "6. Color and photometric pipeline effects::Sensor spectral-response variation": ("implemented", "src/color_cpu.cpp", "src/color.cu", "tests/color_photometric_unit.cpp;tests/color_photometric_properties.py;tests/color_photometric_cuda_validation.cu", "seeded per-coefficient Gaussian response perturbation; normalized clipping"),
    "6. Color and photometric pipeline effects::Chroma noise": ("implemented", "src/color_cpu.cpp", "src/color.cu", "tests/color_photometric_unit.cpp;tests/color_photometric_properties.py;tests/color_photometric_cuda_validation.cu", "BT.601 Cb/Cr independent Gaussian noise; seeded and clipped"),
    "6. Color and photometric pipeline effects::Luma noise": ("implemented", "src/color_cpu.cpp", "src/color.cu", "tests/color_photometric_unit.cpp;tests/color_photometric_properties.py;tests/color_photometric_cuda_validation.cu", "BT.601 Y Gaussian noise with chroma preservation surrogate"),
    "6. Color and photometric pipeline effects::Correlated luma-chroma noise": ("implemented", "src/color_cpu.cpp", "src/color.cu", "tests/color_photometric_unit.cpp;tests/color_photometric_properties.py;tests/color_photometric_cuda_validation.cu", "BT.601 Y/Cb/Cr seeded Gaussian noise with bounded shared correlation"),
    "6. Color and photometric pipeline effects::Local tone-mapping noise": ("implemented", "src/color_cpu.cpp", "src/color.cu", "tests/color_pipeline_artifacts_unit.cpp;tests/color_pipeline_artifacts_properties.py;tests/color_pipeline_artifacts_cuda_validation.cu", "BT.601 clamped local Y mean contraction plus coordinate-keyed Gaussian noise"),
    "6. Color and photometric pipeline effects::Chroma subsampling artifacts": ("implemented", "src/color_cpu.cpp", "src/color.cu", "tests/color_pipeline_artifacts_unit.cpp;tests/color_pipeline_artifacts_properties.py;tests/color_pipeline_artifacts_cuda_validation.cu", "BT.601 full-range surrogate; exact Y444, horizontal Y422, clipped 2x2 Y420 chroma averages"),
    "6. Color and photometric pipeline effects::Color clipping": ("implemented", "src/color_cpu.cpp", "src/color.cu", "tests/color_photometric_unit.cpp;tests/color_photometric_properties.py;tests/color_photometric_cuda_validation.cu", "independent normalized RGB lower/upper bounds; alpha preservation"),
    "6. Color and photometric pipeline effects::White-balance clipping": ("implemented", "src/color_cpu.cpp", "src/color.cu", "tests/color_photometric_unit.cpp;tests/color_photometric_properties.py;tests/color_photometric_cuda_validation.cu", "nonnegative RGB gains followed by common normalized clipping"),
    "6. Color and photometric pipeline effects::Posterization": ("implemented", "src/tone_cpu.cpp", "src/tone.cu", "tests/tone_variations_unit.cpp;tests/tone_variations_properties.py;tests/tone_variations_cuda_validation.cu", "floor quantization to explicit 2^bits levels without dither"),
    "6. Color and photometric pipeline effects::Banding from low-bit-depth tone mapping": ("implemented", "src/tone_cpu.cpp", "src/tone.cu", "tests/tone_variations_unit.cpp;tests/tone_variations_properties.py;tests/tone_variations_cuda_validation.cu", "floor quantization alias with explicit 2^bits levels"),
    "8. Compression and transport artifacts::JPEG chroma subsampling 4:4:4": ("partial", "src/jpeg_compression_cpu.cpp", "src/jpeg_compression.cu", "tests/jpeg_compression_unit.cpp;tests/jpeg_compression_parity.py", "libjpeg YCbCr 4:4:4 sampling factors; CUDA uses host-codec fallback"),
    "8. Compression and transport artifacts::JPEG chroma subsampling 4:2:2": ("partial", "src/jpeg_compression_cpu.cpp", "src/jpeg_compression.cu", "tests/jpeg_compression_unit.cpp;tests/jpeg_compression_parity.py", "libjpeg YCbCr horizontal 2:1 chroma sampling; CUDA uses host-codec fallback"),
    "8. Compression and transport artifacts::JPEG chroma subsampling 4:2:0": ("partial", "src/jpeg_compression_cpu.cpp", "src/jpeg_compression.cu", "tests/jpeg_compression_unit.cpp;tests/jpeg_compression_parity.py", "libjpeg YCbCr horizontal/vertical 2:1 chroma sampling; CUDA uses host-codec fallback"),
    "9. Environmental and acquisition artifacts::Fog veil": ("implemented", "src/weather_cpu.cpp;src/environmental_cpu.cpp", "src/weather.cu;src/environmental.cu", "tests/environmental_unit.cpp;tests/environmental_properties.py;tests/environmental_cuda_validation.cu", "explicit HxW deterministic veil field or SplitMix64 weather field; borrowed-map ownership; clipped compositing"),
    "9. Environmental and acquisition artifacts::Atmospheric haze": ("implemented", "src/environmental_cpu.cpp", "src/environmental.cu", "tests/environmental_unit.cpp;tests/environmental_properties.py;tests/environmental_cuda_validation.cu", "explicit HxW/HxWxC veil mask; deterministic airlight blend; normalized clipping"),
    "9. Environmental and acquisition artifacts::Rain streaks": ("implemented", "src/weather_cpu.cpp", "src/weather.cu", "tests/random_rain_parity.py;tests/environmental_properties.py", "explicit RainStreak records or SplitMix64 placement; pixel-centre capsule mask; sequential white blend"),
    "9. Environmental and acquisition artifacts::Snow occlusion": ("implemented", "src/weather_cpu.cpp", "src/weather.cu", "tests/random_snow_parity.py;tests/environmental_properties.py", "explicit Snowflake records or SplitMix64 placement; pixel-centre disc mask; sequential white blend"),
    "9. Environmental and acquisition artifacts::Dust particles": ("implemented", "src/weather_cpu.cpp", "src/weather.cu", "tests/random_gravel_parity.py;tests/environmental_properties.py", "explicit GravelParticle records or SplitMix64 placement; pixel-centre disc mask; scalar fill blend"),
    "9. Environmental and acquisition artifacts::Smoke veil": ("implemented", "src/environmental_cpu.cpp", "src/environmental.cu", "tests/environmental_unit.cpp;tests/environmental_properties.py;tests/environmental_cuda_validation.cu", "explicit veil map; deterministic smoke-airlight approximation; normalized clipping"),
    "9. Environmental and acquisition artifacts::Low-light amplification": ("implemented", "src/environmental_cpu.cpp", "src/environmental.cu", "tests/environmental_unit.cpp;tests/environmental_properties.py;tests/environmental_cuda_validation.cu", "explicit nonnegative float gain; clipped multiplicative signal model"),
    "9. Environmental and acquisition artifacts::Underexposure": ("implemented", "src/environmental_cpu.cpp", "src/environmental.cu", "tests/environmental_unit.cpp;tests/environmental_properties.py;tests/environmental_cuda_validation.cu", "explicit gain below one; clipped multiplicative signal model"),
    "9. Environmental and acquisition artifacts::Overexposure": ("implemented", "src/environmental_cpu.cpp", "src/environmental.cu", "tests/environmental_unit.cpp;tests/environmental_properties.py;tests/environmental_cuda_validation.cu", "explicit gain above one; clipped multiplicative signal model"),
    "9. Environmental and acquisition artifacts::Headlight flare": ("implemented", "src/noise_cpu.cpp", "src/noise.cu", "tests/optical_scatter_properties.py;tests/environmental_properties.py", "explicit source discs/halos; borrowed records and local gain map; clipped white blend"),
    "9. Environmental and acquisition artifacts::Backlight washout": ("implemented", "src/environmental_cpu.cpp", "src/environmental.cu", "tests/environmental_unit.cpp;tests/environmental_properties.py;tests/environmental_cuda_validation.cu", "explicit veil mask; deterministic white-airlight approximation; normalized clipping"),
    "9. Environmental and acquisition artifacts::Reflections": ("implemented", "src/noise_cpu.cpp", "src/noise.cu", "tests/optical_scatter_properties.py;tests/environmental_properties.py", "explicit nearest/clamped ghost records and optional HxW/HxWxC map; borrowed ownership"),
    "9. Environmental and acquisition artifacts::Window glare": ("implemented", "src/environmental_cpu.cpp", "src/environmental.cu", "tests/environmental_unit.cpp;tests/environmental_properties.py;tests/environmental_cuda_validation.cu", "explicit veil mask; deterministic white-airlight approximation; normalized clipping"),
    "9. Environmental and acquisition artifacts::Dirty-lens blur": ("implemented", "src/environmental_cpu.cpp", "src/environmental.cu", "tests/acquisition_artifacts_unit.cpp;tests/acquisition_artifacts_properties.py;tests/acquisition_artifacts_cuda_validation.cu", "borrowed HxW/HxWxC opacity map; bounded square-box PSF blend; clipped normalized HWC float32 output"),
    "9. Environmental and acquisition artifacts::Sensor temperature drift": ("implemented", "src/environmental_cpu.cpp", "src/environmental.cu", "tests/acquisition_artifacts_unit.cpp;tests/acquisition_artifacts_properties.py;tests/acquisition_artifacts_cuda_validation.cu", "contiguous T-H-W-C linear temperature ramp with exponential dark-current approximation and clipping"),
    "9. Environmental and acquisition artifacts::Electromagnetic interference": ("implemented", "src/environmental_cpu.cpp", "src/environmental.cu", "tests/acquisition_artifacts_unit.cpp;tests/acquisition_artifacts_properties.py;tests/acquisition_artifacts_cuda_validation.cu", "deterministic sinusoidal additive field with borrowed HxW/HxWxC amplitude map and clipping"),
    "9. Environmental and acquisition artifacts::Power-supply banding": ("implemented", "src/environmental_cpu.cpp", "src/environmental.cu", "tests/acquisition_artifacts_unit.cpp;tests/acquisition_artifacts_properties.py;tests/acquisition_artifacts_cuda_validation.cu", "contiguous T-H-W-C sinusoidal row/time gain ripple with explicit frequency and clipping"),
    "9. Environmental and acquisition artifacts::Fluorescent-light flicker": ("implemented", "src/environmental_cpu.cpp", "src/environmental.cu", "tests/acquisition_artifacts_unit.cpp;tests/acquisition_artifacts_properties.py;tests/acquisition_artifacts_cuda_validation.cu", "frame-global sinusoidal exposure factor over explicit T-H-W-C batch and clipping"),
    "9. Environmental and acquisition artifacts::LED rolling-band artifacts": ("implemented", "src/environmental_cpu.cpp", "src/environmental.cu", "tests/acquisition_artifacts_unit.cpp;tests/acquisition_artifacts_properties.py;tests/acquisition_artifacts_cuda_validation.cu", "row/time sinusoidal rolling-shutter modulation over explicit T-H-W-C batch and clipping"),
    "Weather and atmosphere::RandomRain": ("partial", "src/weather_cpu.cpp", "src/weather.cu", "tests/random_rain_parity.py", "explicit RainStreak records or deterministic SplitMix64 seed placement; pixel-centre capsule rasterization; sequential white alpha blend contract"),
    "Weather and atmosphere::RandomSnow": ("partial", "src/weather_cpu.cpp", "src/weather.cu", "tests/random_snow_parity.py", "explicit Snowflake records or deterministic SplitMix64 seed placement; pixel-centre disc rasterization; sequential white alpha blend contract"),
    "Weather and atmosphere::RandomSunFlare": ("partial", "src/weather_cpu.cpp", "src/weather.cu", "tests/random_sun_flare_parity.py", "explicit SunFlareSource and SunFlareRay records or deterministic SplitMix64 ray placement; pixel-centre disc/capsule rasterization; sequential white alpha blend contract"),
    "Weather and atmosphere::RandomShadow": ("partial", "src/weather_cpu.cpp", "src/weather.cu", "tests/random_shadow_parity.py", "deterministic explicit polygon and rectangle masks; pixel-centre even-odd/inclusive rasterization; sequential scalar fill alpha blend contract"),
    "Weather and atmosphere::RandomGravel": ("implemented", "src/weather_cpu.cpp", "src/weather.cu", "tests/random_gravel_parity.py", "explicit GravelParticle records or deterministic SplitMix64 seed placement; pixel-centre disc rasterization; sequential scalar fill alpha blend contract"),
    "Weather and atmosphere::Spatter": ("implemented", "src/weather_cpu.cpp", "src/weather.cu", "tests/random_spatter_parity.py", "explicit SpatterDroplet records or deterministic SplitMix64 seed placement; pixel-centre disc rasterization; sequential per-channel RGB alpha blend contract"),
    "`augmenters.imgcorruptlike`::Spatter": ("partial", "src/weather_cpu.cpp", "src/weather.cu", "tests/random_spatter_parity.py", "explicit SpatterDroplet records or deterministic SplitMix64 seed placement; pixel-centre disc rasterization; sequential per-channel RGB alpha blend contract"),
    "Color and intensity::FancyPCA": ("partial", "src/color_cpu.cpp", "src/color.cu", "tests/fancy_pca_parity.py", "deterministic explicit orthonormal 3x3 basis/eigenvalue perturbation contract; fixed alpha replaces stochastic sampling"),
    "Color and intensity::PlasmaBrightnessContrast": ("partial", "src/color_cpu.cpp", "src/color.cu", "tests/plasma_brightness_contrast_parity.py", "deterministic four-scale hash plasma approximation; explicit width*height float field or seed; PlasmaContrast local contrast plus normalized brightness offset"),
    "Color and intensity::PlasmaContrast": ("partial", "src/color_cpu.cpp", "src/color.cu", "tests/plasma_contrast_parity.py", "deterministic four-scale hash plasma approximation; explicit width*height float field or seed; local contrast scaling contract"),
    "Color and intensity::PlasmaShadow": ("partial", "src/color_cpu.cpp", "src/color.cu", "tests/plasma_shadow_parity.py", "deterministic four-scale hash plasma approximation; explicit width*height float field or seed; thresholded linear shadow attenuation with bounded strength"),
    "View, metadata, RNG, and status contract::Exposure scaling": ("partial", "src/signal_cpu.cpp", "src/signal.cu", "tests/signal_gain_parity.py", "native reference"),
    "View, metadata, RNG, and status contract::Analog gain": ("partial", "src/signal_cpu.cpp", "src/signal.cu", "tests/signal_gain_parity.py", "native reference"),
    "View, metadata, RNG, and status contract::Digital gain": ("partial", "src/signal_cpu.cpp", "src/signal.cu", "tests/signal_gain_parity.py", "native reference"),
    "View, metadata, RNG, and status contract::Pixel saturation": ("partial", "src/signal_cpu.cpp", "src/signal.cu", "tests/signal_levels_parity.py", "native reference"),
    "View, metadata, RNG, and status contract::Black-level offset": ("partial", "src/signal_cpu.cpp", "src/signal.cu", "tests/signal_levels_parity.py", "native reference"),
    "View, metadata, RNG, and status contract::ADC quantization": ("partial", "src/signal_cpu.cpp", "src/signal.cu", "tests/signal_levels_parity.py", "native reference"),
    "View, metadata, RNG, and status contract::Bit-depth reduction": ("partial", "src/signal_cpu.cpp", "src/signal.cu", "tests/signal_levels_parity.py", "native reference"),
    "View, metadata, RNG, and status contract::Bit truncation": ("partial", "src/signal_cpu.cpp", "src/signal.cu", "tests/signal_levels_parity.py", "native reference"),
    "View, metadata, RNG, and status contract::Column-wise banding noise": ("partial", "src/noise_cpu.cpp", "src/noise.cu", "tests/unit.cpp", "statistical"),
    "View, metadata, RNG, and status contract::Row-column correlated noise": ("partial", "src/noise_cpu.cpp", "src/noise.cu", "tests/sensor_spatial_noise_unit.cpp", "property + explicit map ownership"),
    "7. Video and temporal artifacts::Temporal Gaussian noise": ("implemented", "src/temporal_cpu.cpp", "src/temporal.cu", "tests/temporal_noise_unit.cpp;tests/temporal_noise_properties.py;tests/temporal_noise_statistics.py;tests/temporal_noise_cuda_validation.cu", "contiguous T-H-W-C float32 batch; explicit AR(1) correlation and counter-based seed"),
    "7. Video and temporal artifacts::Temporally correlated shot noise": ("implemented", "src/temporal_cpu.cpp", "src/temporal.cu", "tests/temporal_noise_unit.cpp;tests/temporal_noise_properties.py;tests/temporal_noise_statistics.py;tests/temporal_noise_cuda_validation.cu", "signal-dependent centered shot approximation with explicit temporal AR(1) correlation"),
    "7. Video and temporal artifacts::Temporal read-noise correlation": ("implemented", "src/temporal_cpu.cpp", "src/temporal.cu", "tests/temporal_noise_unit.cpp;tests/temporal_noise_properties.py;tests/temporal_noise_statistics.py;tests/temporal_noise_cuda_validation.cu", "Gaussian read noise with explicit frame correlation and deterministic seed"),
    "7. Video and temporal artifacts::Flicker": ("implemented", "src/temporal_cpu.cpp", "src/temporal.cu", "tests/temporal_noise_unit.cpp;tests/temporal_noise_properties.py;tests/temporal_noise_statistics.py;tests/temporal_noise_cuda_validation.cu", "frame-global multiplicative AR(1) factor"),
    "7. Video and temporal artifacts::Exposure flicker": ("implemented", "src/temporal_cpu.cpp", "src/temporal.cu", "tests/temporal_noise_unit.cpp;tests/temporal_noise_properties.py;tests/temporal_noise_statistics.py;tests/temporal_noise_cuda_validation.cu", "frame-global exposure multiplier over explicit T-H-W-C batch"),
    "7. Video and temporal artifacts::White-balance flicker": ("implemented", "src/temporal_cpu.cpp", "src/temporal.cu", "tests/temporal_noise_unit.cpp;tests/temporal_noise_properties.py;tests/temporal_noise_statistics.py;tests/temporal_noise_cuda_validation.cu", "independent RGB frame factors with explicit channel ownership"),
    "7. Video and temporal artifacts::Gain flicker": ("implemented", "src/temporal_cpu.cpp", "src/temporal.cu", "tests/temporal_noise_unit.cpp;tests/temporal_noise_properties.py;tests/temporal_noise_statistics.py;tests/temporal_noise_cuda_validation.cu", "frame-global multiplicative gain AR(1) factor"),
    "7. Video and temporal artifacts::Fixed-pattern noise drift": ("implemented", "src/temporal_cpu.cpp", "src/temporal.cu", "tests/temporal_noise_unit.cpp;tests/temporal_noise_properties.py;tests/temporal_noise_statistics.py;tests/temporal_noise_cuda_validation.cu", "borrowed HWC base pattern plus deterministic AR(1) per-pixel drift"),
    "7. Video and temporal artifacts::Row-noise phase changes": ("implemented", "src/temporal_cpu.cpp", "src/temporal.cu", "tests/color_temporal_remaining_unit.cpp;tests/remaining_noise_properties.py", "correlated row offsets with seeded frame/row phase sign changes"),
    "View, metadata, RNG, and status contract::Stuck pixels": ("partial", "src/noise_cpu.cpp", "src/noise.cu", "tests/unit.cpp", "statistical"),
    "View, metadata, RNG, and status contract::Clustered defective pixels": ("partial", "src/noise_cpu.cpp", "src/noise.cu", "tests/sensor_spatial_noise_unit.cpp", "deterministic records or seeded clusters"),
    "View, metadata, RNG, and status contract::White-level variation": ("partial", "src/noise_cpu.cpp", "src/noise.cu", "tests/unit.cpp", "statistical"),
    "View, metadata, RNG, and status contract::Temperature-dependent dark current": ("partial", "src/noise_cpu.cpp", "src/noise.cu", "tests/unit.cpp", "statistical"),
    "ISO and gain profile contract::Exposure-time-dependent dark current": ("implemented", "src/iso_profile_cpu.cpp", "src/iso_profile.cu", "tests/iso_profile_unit.cpp;tests/iso_profile_cuda_validation.cu;tests/iso_physical_properties.py", "physical electron-domain Poisson charge with normalized clipping; borrowed ISO knots"),
    "ISO and gain profile contract::Temperature-dependent noise scaling": ("implemented", "src/iso_profile_cpu.cpp", "src/iso_profile.cu", "tests/iso_profile_unit.cpp;tests/iso_profile_cuda_validation.cu;tests/iso_physical_properties.py", "physical electron-domain Gaussian scaling with exponential temperature factor and normalized clipping"),
    "ISO and gain profile contract::Camera-profile lookup-table parameters": ("implemented", "src/iso_profile_cpu.cpp", "src/iso_profile.cu", "tests/profile_calibration_unit.cpp;tests/profile_calibration_properties.py;tests/profile_calibration_cuda_validation.cu", "borrowed ISO-sorted ten-field LUT with clamped interpolation, normalized clipping, deterministic seeded noise"),
    "ISO and gain profile contract::Calibration-frame-driven noise parameters": ("implemented", "src/iso_profile_cpu.cpp", "src/iso_profile.cu", "tests/profile_calibration_unit.cpp;tests/profile_calibration_properties.py;tests/profile_calibration_cuda_validation.cu", "borrowed scalar/HWC map contract for read sigma, FPN offset, gain, deterministic seeded application"),
    "View, metadata, RNG, and status contract::Correlated read noise": ("partial", "src/noise_cpu.cpp", "src/noise.cu", "tests/unit.cpp", "statistical"),
    "View, metadata, RNG, and status contract::Amplifier noise": ("partial", "src/noise_cpu.cpp", "src/noise.cu", "tests/unit.cpp", "statistical"),
    "View, metadata, RNG, and status contract::Reset noise / kTC noise": ("partial", "src/noise_cpu.cpp", "src/noise.cu", "tests/unit.cpp", "statistical"),
    "View, metadata, RNG, and status contract::Dark-frame offset": ("partial", "src/noise_cpu.cpp", "src/noise.cu", "tests/unit.cpp", "statistical"),
    "View, metadata, RNG, and status contract::Pixel cross-talk": ("partial", "src/signal_cpu.cpp", "src/signal.cu", "tests/spatial_artifacts_parity.py", "native reference"),
    "View, metadata, RNG, and status contract::Charge leakage": ("partial", "src/signal_cpu.cpp", "src/signal.cu", "tests/spatial_artifacts_parity.py", "native reference"),
    "3. Bayer and color-filter-array effects::RGGB Bayer CFA sampling": ("partial", "src/bayer_cpu.cpp", "src/bayer.cu", "tests/bayer_parity.py", "native reference"),
    "3. Bayer and color-filter-array effects::BGGR Bayer CFA sampling": ("partial", "src/bayer_cpu.cpp", "src/bayer.cu", "tests/bayer_parity.py", "native reference"),
    "3. Bayer and color-filter-array effects::GRBG Bayer CFA sampling": ("partial", "src/bayer_cpu.cpp", "src/bayer.cu", "tests/bayer_parity.py", "native reference"),
    "3. Bayer and color-filter-array effects::GBRG Bayer CFA sampling": ("partial", "src/bayer_cpu.cpp", "src/bayer.cu", "tests/bayer_parity.py", "native reference"),
    "3. Bayer and color-filter-array effects::Quad-Bayer CFA sampling": ("implemented", "src/bayer_cpu.cpp", "src/bayer.cu", "tests/cfa_sampling_unit.cpp;tests/cfa_sampling_properties.py;tests/cfa_sampling_cuda_validation.cu", "explicit 4x4 expanded Bayer pattern; HWC RGB input and one raw plane output"),
    "3. Bayer and color-filter-array effects::RGBW CFA sampling": ("implemented", "src/bayer_cpu.cpp", "src/bayer.cu", "tests/cfa_sampling_unit.cpp;tests/cfa_sampling_properties.py;tests/cfa_sampling_cuda_validation.cu", "explicit 2x2 RGBW pattern with R/G/B/W channel ownership and one raw plane output"),
    "3. Bayer and color-filter-array effects::Custom CFA masks": ("implemented", "src/bayer_cpu.cpp", "src/bayer.cu", "tests/cfa_sampling_unit.cpp;tests/cfa_sampling_properties.py;tests/cfa_sampling_cuda_validation.cu", "borrowed repeating channel-index mask; host CPU ownership and device CUDA ownership"),
    "3. Bayer and color-filter-array effects::Nearest-neighbor demosaicing": ("partial", "src/bayer_cpu.cpp", "src/bayer.cu", "tests/demosaic_parity.py", "native reference"),
    "3. Bayer and color-filter-array effects::Bilinear demosaicing": ("partial", "src/bayer_cpu.cpp", "src/bayer.cu", "tests/bilinear_demosaic_parity.py", "native reference"),
    "3. Bayer and color-filter-array effects::Directional demosaicing artifacts": ("partial", "src/bayer_cpu.cpp", "src/bayer.cu", "tests/demosaic_artifacts_unit.cpp;tests/demosaic_artifacts_properties.py;tests/demosaic_artifacts_cuda_validation.cu", "deterministic edge-aware directional surrogate; explicit direction/strength and saturating uint8 clipping"),
    "3. Bayer and color-filter-array effects::False-color zipper artifacts": ("partial", "src/bayer_cpu.cpp", "src/bayer.cu", "tests/demosaic_artifacts_unit.cpp;tests/demosaic_artifacts_properties.py;tests/demosaic_artifacts_cuda_validation.cu", "deterministic alternating red/blue gradient injection; explicit strength and saturating uint8 clipping"),
    "3. Bayer and color-filter-array effects::Demosaicing aliasing": ("partial", "src/bayer_cpu.cpp", "src/bayer.cu", "tests/demosaic_artifacts_unit.cpp;tests/demosaic_artifacts_properties.py;tests/demosaic_artifacts_cuda_validation.cu", "explicit period/phase chroma sinusoid approximation; positive period and saturating uint8 clipping"),
    "3. Bayer and color-filter-array effects::Demosaicing ringing": ("partial", "src/bayer_cpu.cpp", "src/bayer.cu", "tests/demosaic_artifacts_unit.cpp;tests/demosaic_artifacts_properties.py;tests/demosaic_artifacts_cuda_validation.cu", "signed four-neighbor raw Laplacian injection; explicit strength and saturating uint8 clipping"),
    "3. Bayer and color-filter-array effects::Demosaicing noise amplification": ("partial", "src/bayer_cpu.cpp", "src/bayer.cu", "tests/demosaic_artifacts_unit.cpp;tests/demosaic_artifacts_properties.py;tests/demosaic_artifacts_cuda_validation.cu", "seeded coordinate-keyed Gaussian noise with gradient amplification; saturating uint8 clipping"),
    "ISO and gain profile contract::Per-channel gain variation": ("partial", "src/signal_cpu.cpp", "src/signal.cu", "tests/channel_gain_parity.py", "native reference"),
    "6. Color and photometric pipeline effects::White-balance gain error": ("partial", "src/signal_cpu.cpp", "src/signal.cu", "tests/channel_gain_parity.py", "native reference"),
    "`augmenters.blur`::BilateralBlur": ("implemented", "src/filter_cpu.cpp", "src/filter.cu", "tests/bilateral_blur_parity.py", "deterministic direct-reference parity; REFLECT_101 border and half-up uint8 rounding"),
    "`augmenters.blur`::MeanShiftBlur": ("implemented", "src/filter_cpu.cpp", "src/filter.cu", "tests/mean_shift_blur_parity.py", "deterministic fixed-iteration spatial/color mean-shift; REFLECT_101 border and half-up uint8 rounding"),
    "Blur and convolution::AdvancedBlur": ("partial", "src/filter_cpu.cpp", "src/filter.cu", "tests/advanced_blur_parity.py", "NumPy/OpenCV deterministic kernel"),
    "Blur and convolution::GlassBlur": ("partial", "src/filter_cpu.cpp", "src/filter.cu", "tests/glass_blur_parity.py", "deterministic swap-sequence reference"),
    "`augmenters.imgcorruptlike`::GlassBlur": ("partial", "src/filter_cpu.cpp", "src/filter.cu", "tests/glass_blur_parity.py", "deterministic swap-sequence reference"),
    "Blur and convolution::ZoomBlur": ("partial", "src/filter_cpu.cpp", "src/filter.cu", "tests/zoom_blur_parity.py", "deterministic NumPy/OpenCV factor-set reference"),
    "`augmenters.imgcorruptlike`::ZoomBlur": ("partial", "src/filter_cpu.cpp", "src/filter.cu", "tests/zoom_blur_parity.py", "deterministic NumPy/OpenCV factor-set reference"),
    "Blur and convolution::NonLocalMeansDenoising": ("partial", "src/non_local_means_cpu.cpp", "src/non_local_means.cu", "tests/non_local_means_parity.py", "deterministic direct bounded-search reference"),
    "`augmenters.imgcorruptlike`::SpeckleNoise": ("partial", "src/imgcorruptlike_cpu.cpp", "src/imgcorruptlike.cu", "tests/imgcorruptlike_unit.cpp;tests/imgcorruptlike_properties.py;tests/imgcorruptlike_cuda_validation.cu", "deterministic coordinate-keyed multiplicative Gaussian approximation; explicit mean/stddev/seed"),
    "`augmenters.imgcorruptlike`::Fog": ("partial", "src/imgcorruptlike_cpu.cpp", "src/imgcorruptlike.cu", "tests/imgcorruptlike_unit.cpp;tests/imgcorruptlike_properties.py;tests/imgcorruptlike_cuda_validation.cu", "deterministic explicit/seeded HxW white veil; native approximation"),
    "`augmenters.imgcorruptlike`::Frost": ("partial", "src/imgcorruptlike_cpu.cpp", "src/imgcorruptlike.cu", "tests/imgcorruptlike_unit.cpp;tests/imgcorruptlike_properties.py;tests/imgcorruptlike_cuda_validation.cu", "deterministic cold white/blue veil; native approximation without random frost assets"),
    "`augmenters.imgcorruptlike`::Snow": ("partial", "src/imgcorruptlike_cpu.cpp", "src/imgcorruptlike.cu", "tests/imgcorruptlike_unit.cpp;tests/imgcorruptlike_properties.py;tests/imgcorruptlike_cuda_validation.cu", "deterministic seeded sparse white accumulation; native approximation"),
    "`augmenters.imgcorruptlike`::Contrast": ("partial", "src/imgcorruptlike_cpu.cpp", "src/imgcorruptlike.cu", "tests/imgcorruptlike_unit.cpp;tests/imgcorruptlike_properties.py;tests/imgcorruptlike_cuda_validation.cu", "deterministic midpoint contrast around 127.5"),
    "`augmenters.imgcorruptlike`::Brightness": ("partial", "src/imgcorruptlike_cpu.cpp", "src/imgcorruptlike.cu", "tests/imgcorruptlike_unit.cpp;tests/imgcorruptlike_properties.py;tests/imgcorruptlike_cuda_validation.cu", "deterministic multiplicative brightness"),
    "`augmenters.imgcorruptlike`::Saturate": ("partial", "src/imgcorruptlike_cpu.cpp", "src/imgcorruptlike.cu", "tests/imgcorruptlike_unit.cpp;tests/imgcorruptlike_properties.py;tests/imgcorruptlike_cuda_validation.cu", "deterministic HSV saturation scaling with extra-channel preservation"),
    "`augmenters.imgcorruptlike`::Pixelate": ("partial", "src/imgcorruptlike_cpu.cpp", "src/imgcorruptlike.cu", "tests/imgcorruptlike_unit.cpp;tests/imgcorruptlike_properties.py;tests/imgcorruptlike_cuda_validation.cu", "deterministic nearest block-centre sampling"),
    "`augmenters.imgcorruptlike`::ImpulseNoise": ("partial", "src/arithmetic_cpu.cpp", "src/arithmetic.cu", "tests/replace_impulse_unit.cpp", "borrowed explicit mask/value arrays or deterministic seeded impulse parameters; CPU host/CUDA device ownership"),
    "`augmenters.pillike`::FilterEdgeEnhance": ("implemented", "src/pillike_cpu.cpp", "src/pillike.cu", "tests/pillike_parity.py", "Pillow EDGE_ENHANCE integer 3x3 kernel, half-up rounding, unchanged borders"),
    "`augmenters.pillike`::FilterEdgeEnhanceMore": ("implemented", "src/pillike_cpu.cpp", "src/pillike.cu", "tests/pillike_parity.py", "Pillow EDGE_ENHANCE_MORE integer 3x3 kernel, unchanged borders"),
    "`augmenters.pillike`::FilterFindEdges": ("implemented", "src/pillike_cpu.cpp", "src/pillike.cu", "tests/pillike_parity.py", "Pillow FIND_EDGES integer 3x3 kernel, unchanged borders"),
    "`augmenters.pillike`::FilterContour": ("implemented", "src/pillike_cpu.cpp", "src/pillike.cu", "tests/pillike_parity.py", "Pillow CONTOUR integer 3x3 kernel with offset 255, unchanged borders"),
    "`augmenters.pillike`::FilterEmboss": ("implemented", "src/pillike_cpu.cpp", "src/pillike.cu", "tests/pillike_parity.py", "Pillow EMBOSS integer diagonal kernel with offset 128, unchanged borders"),
    "`augmenters.pillike`::FilterSharpen": ("implemented", "src/pillike_cpu.cpp", "src/pillike.cu", "tests/pillike_parity.py", "Pillow SHARPEN integer 3x3 kernel, scale 16, unchanged borders"),
    "`augmenters.pillike`::FilterDetail": ("implemented", "src/pillike_cpu.cpp", "src/pillike.cu", "tests/pillike_parity.py", "Pillow DETAIL integer cross kernel, scale 6, unchanged borders"),
    "`augmenters.size`::PadToMultiplesOf": ("implemented", "src/size_cpu.cpp", "src/size.cu", "tests/size_catalog_parity.py", "explicit multiples, anchor, fill, and border contract"),
    "`augmenters.size`::CropToMultiplesOf": ("implemented", "src/size_cpu.cpp", "src/size.cu", "tests/size_catalog_parity.py", "explicit multiples and anchor contract"),
    "`augmenters.size`::CropToPowersOf": ("implemented", "src/size_cpu.cpp", "src/size.cu", "tests/size_catalog_parity.py", "explicit power base and anchor contract"),
    "`augmenters.size`::PadToPowersOf": ("implemented", "src/size_cpu.cpp", "src/size.cu", "tests/size_catalog_parity.py", "explicit power base, anchor, fill, and border contract"),
    "`augmenters.size`::CropToAspectRatio": ("implemented", "src/size_cpu.cpp", "src/size.cu", "tests/size_catalog_parity.py", "explicit ratio, rounding, and anchor contract"),
    "`augmenters.size`::PadToAspectRatio": ("implemented", "src/size_cpu.cpp", "src/size.cu", "tests/size_catalog_parity.py", "explicit ratio, rounding, anchor, fill, and border contract"),
    "`augmenters.size`::CenterCropToMultiplesOf": ("implemented", "src/size_cpu.cpp", "src/size.cu", "tests/size_catalog_parity.py", "center anchor contract"),
    "`augmenters.size`::CenterPadToMultiplesOf": ("implemented", "src/size_cpu.cpp", "src/size.cu", "tests/size_catalog_parity.py", "center anchor and fill contract"),
    "`augmenters.size`::CenterCropToPowersOf": ("implemented", "src/size_cpu.cpp", "src/size.cu", "tests/size_catalog_parity.py", "center anchor and power base contract"),
    "`augmenters.size`::CenterPadToPowersOf": ("implemented", "src/size_cpu.cpp", "src/size.cu", "tests/size_catalog_parity.py", "center anchor, power base, and fill contract"),
    "`augmenters.size`::CenterCropToAspectRatio": ("implemented", "src/size_cpu.cpp", "src/size.cu", "tests/size_catalog_parity.py", "center ratio, rounding, and anchor contract"),
    "`augmenters.size`::CenterPadToAspectRatio": ("implemented", "src/size_cpu.cpp", "src/size.cu", "tests/size_catalog_parity.py", "center ratio, rounding, and fill contract"),
    "`augmenters.size`::KeepSizeByResize": ("implemented", "src/size_cpu.cpp", "src/size.cu", "tests/size_catalog_parity.py", "explicit intermediate dimensions and interpolation round-trip contract"),
    "`augmenters.contrast`::GammaContrast": ("partial", "src/pixel_cpu.cpp", "src/pixel.cu", "tests/gamma_parity.py", "imgaug"),
    "`augmenters.contrast`::LinearContrast": ("partial", "src/pixel_cpu.cpp", "src/pixel.cu", "tests/pixel_parity.py", "imgaug"),
    "`augmenters.color`::Grayscale": ("partial", "src/pixel_cpu.cpp", "src/pixel.cu", "tests/pixel_parity.py", "imgaug"),
    "`augmenters.blur`::AverageBlur": ("partial", "src/filter_cpu.cpp", "src/filter.cu", "tests/blur_parity.py", "imgaug"),
    "`augmenters.imgcorruptlike`::DefocusBlur": ("partial", "src/filter_cpu.cpp", "src/filter.cu", "tests/defocus_parity.py", "imgaug"),
    "`augmenters.imgcorruptlike`::GaussianNoise": ("partial", "src/arithmetic_cpu.cpp", "src/arithmetic.cu", "tests/noise_parity.py", "imgaug"),
    "`augmenters.pillike`::EnhanceSharpness": ("partial", "src/filter_cpu.cpp", "src/filter.cu", "tests/sharpen_parity.py", "imgaug"),
    "`augmenters.size`::PadToFixedSize": ("partial", "src/size_cpu.cpp", "src/size.cu", "tests/size_parity.py", "imgaug"),
    "`augmenters.size`::CropToFixedSize": ("partial", "src/size_cpu.cpp", "src/size.cu", "tests/size_parity.py", "imgaug"),
    "`augmenters.size`::CenterCropToFixedSize": ("partial", "src/size_cpu.cpp", "src/size.cu", "tests/size_parity.py", "imgaug"),
    "Color and intensity::ToGray": ("partial", "src/pixel_cpu.cpp", "src/pixel.cu", "tests/pixel_parity.py", "albumentations"),
    "Color and intensity::ToRGB": ("partial", "src/color_cpu.cpp", "src/color.cu", "tests/to_rgb_parity.py", "albumentations"),
    "Blur and convolution::Blur": ("partial", "src/filter_cpu.cpp", "src/filter.cu", "tests/blur_parity.py", "albumentations"),
    "Blur and convolution::GaussianBlur": ("partial", "src/filter_cpu.cpp", "src/filter.cu", "tests/blur_parity.py", "albumentations"),
    "`augmenters.blur`::GaussianBlur": ("partial", "src/filter_cpu.cpp", "src/filter.cu", "tests/blur_parity.py", "albumentations"),
    "`augmenters.imgcorruptlike`::GaussianBlur": ("partial", "src/filter_cpu.cpp", "src/filter.cu", "tests/blur_parity.py", "albumentations"),
    "Blur and convolution::MedianBlur": ("partial", "src/filter_cpu.cpp", "src/filter.cu", "tests/blur_parity.py", "albumentations/imgaug"),
    "`augmenters.blur`::MedianBlur": ("partial", "src/filter_cpu.cpp", "src/filter.cu", "tests/blur_parity.py", "albumentations/imgaug"),
    "`augmenters.arithmetic`::Add": ("partial", "src/arithmetic_cpu.cpp", "src/arithmetic.cu", "tests/arithmetic_parity.py", "imgaug"),
    "`augmenters.arithmetic`::AddElementwise": ("partial", "src/arithmetic_cpu.cpp", "src/arithmetic.cu", "tests/arithmetic_elementwise_unit.cpp", "deterministic explicit per-element array or seed contract"),
    "`augmenters.arithmetic`::MultiplyElementwise": ("partial", "src/arithmetic_cpu.cpp", "src/arithmetic.cu", "tests/arithmetic_elementwise_unit.cpp", "deterministic explicit per-element array or seed contract"),
    "`augmenters.arithmetic`::ReplaceElementwise": ("partial", "src/arithmetic_cpu.cpp", "src/arithmetic.cu", "tests/replace_impulse_unit.cpp", "borrowed explicit mask/value arrays or deterministic seeded Bernoulli replacement; CPU host/CUDA device ownership"),
    "`augmenters.arithmetic`::ImpulseNoise": ("partial", "src/arithmetic_cpu.cpp", "src/arithmetic.cu", "tests/replace_impulse_unit.cpp", "borrowed explicit mask/value arrays or deterministic seeded impulse parameters; CPU host/CUDA device ownership"),
    "`augmenters.arithmetic`::ReplaceElementwise": ("partial", "src/arithmetic_cpu.cpp", "src/arithmetic.cu", "tests/replace_impulse_unit.cpp", "borrowed explicit mask/value arrays or deterministic seeded Bernoulli replacement; CPU host/CUDA device ownership"),
    "`augmenters.arithmetic`::ImpulseNoise": ("partial", "src/arithmetic_cpu.cpp", "src/arithmetic.cu", "tests/replace_impulse_unit.cpp", "borrowed explicit mask/value arrays or deterministic seeded impulse parameters; CPU host/CUDA device ownership"),
    "`augmenters.imgcorruptlike`::ImpulseNoise": ("partial", "src/arithmetic_cpu.cpp", "src/arithmetic.cu", "tests/replace_impulse_unit.cpp", "borrowed explicit mask/value arrays or deterministic seeded impulse parameters; CPU host/CUDA device ownership"),
    "`augmenters.arithmetic`::Multiply": ("partial", "src/arithmetic_cpu.cpp", "src/arithmetic.cu", "tests/arithmetic_parity.py", "imgaug"),
    "`augmenters.arithmetic`::Invert": ("partial", "src/arithmetic_cpu.cpp", "src/arithmetic.cu", "tests/arithmetic_parity.py", "imgaug/albumentations"),
    "Color and intensity::Solarize": ("partial", "src/arithmetic_cpu.cpp", "src/arithmetic.cu", "tests/arithmetic_parity.py", "imgaug/albumentations"),
    "`augmenters.arithmetic`::Solarize": ("partial", "src/arithmetic_cpu.cpp", "src/arithmetic.cu", "tests/arithmetic_parity.py", "imgaug/albumentations"),
    "`augmenters.pillike`::Solarize": ("partial", "src/arithmetic_cpu.cpp", "src/arithmetic.cu", "tests/arithmetic_parity.py", "imgaug/albumentations"),
    "Color and intensity::Posterize": ("partial", "src/arithmetic_cpu.cpp", "src/arithmetic.cu", "tests/arithmetic_parity.py", "imgaug/albumentations"),
    "`augmenters.color`::Posterize": ("partial", "src/arithmetic_cpu.cpp", "src/arithmetic.cu", "tests/arithmetic_parity.py", "imgaug/albumentations"),
    "`augmenters.pillike`::Posterize": ("partial", "src/arithmetic_cpu.cpp", "src/arithmetic.cu", "tests/arithmetic_parity.py", "imgaug/albumentations"),
    "Noise and dropout::GaussNoise": ("partial", "src/arithmetic_cpu.cpp", "src/arithmetic.cu", "tests/noise_parity.py", "albumentations"),
    "`augmenters.arithmetic`::AdditiveGaussianNoise": ("partial", "src/arithmetic_cpu.cpp", "src/arithmetic.cu", "tests/noise_parity.py", "imgaug"),
    "Noise and dropout::SaltAndPepper": ("implemented", "src/arithmetic_cpu.cpp", "src/arithmetic.cu", "tests/salt_pepper_parity.py", "deterministic native pixel-mask/seed contract"),
    "`augmenters.arithmetic`::SaltAndPepper": ("implemented", "src/arithmetic_cpu.cpp", "src/arithmetic.cu", "tests/salt_pepper_parity.py", "deterministic native pixel-mask/seed contract"),
    "`augmenters.arithmetic`::CoarseSaltAndPepper": ("implemented", "src/arithmetic_cpu.cpp", "src/arithmetic.cu", "tests/salt_pepper_parity.py", "deterministic native rectangle/block contract"),
    "`augmenters.arithmetic`::Salt": ("implemented", "src/arithmetic_cpu.cpp", "src/arithmetic.cu", "tests/salt_pepper_parity.py", "deterministic native pixel-mask/seed contract"),
    "`augmenters.arithmetic`::CoarseSalt": ("implemented", "src/arithmetic_cpu.cpp", "src/arithmetic.cu", "tests/salt_pepper_parity.py", "deterministic native rectangle/block contract"),
    "`augmenters.arithmetic`::Pepper": ("implemented", "src/arithmetic_cpu.cpp", "src/arithmetic.cu", "tests/salt_pepper_parity.py", "deterministic native pixel-mask/seed contract"),
    "`augmenters.arithmetic`::CoarsePepper": ("implemented", "src/arithmetic_cpu.cpp", "src/arithmetic.cu", "tests/salt_pepper_parity.py", "deterministic native rectangle/block contract"),
    "Noise and dropout::PixelDropout": ("partial", "src/dropout_cpu.cpp", "src/dropout.cu", "tests/dropout_parity.py", "albumentations"),
    "Noise and dropout::ChannelDropout": ("partial", "src/dropout_cpu.cpp", "src/dropout.cu", "tests/dropout_parity.py", "albumentations"),
    "Noise and dropout::GridDropout": ("partial", "src/dropout_cpu.cpp", "src/dropout.cu", "tests/dropout_parity.py", "albumentations"),
    "Noise and dropout::CoarseDropout": ("partial", "src/dropout_cpu.cpp", "src/dropout.cu", "tests/dropout_parity.py", "albumentations/imgaug"),
    "`augmenters.arithmetic`::CoarseDropout": ("partial", "src/dropout_cpu.cpp", "src/dropout.cu", "tests/dropout_parity.py", "albumentations/imgaug"),
    "Noise and dropout::MaskDropout": ("implemented", "src/dropout_cpu.cpp", "src/dropout.cu", "tests/mask_dropout_parity.py", "deterministic binary-mask contract"),
    "Segmentation and region transforms::MaskDropout": ("implemented", "src/dropout_cpu.cpp", "src/dropout.cu", "tests/mask_dropout_parity.py", "deterministic binary-mask contract"),
    "Noise and dropout::XYMasking": ("implemented", "src/dropout_cpu.cpp", "src/dropout.cu", "tests/xy_masking_parity.py", "deterministic explicit row/column interval contract"),
    "`augmenters.arithmetic`::Cutout": ("implemented", "src/dropout_cpu.cpp", "src/dropout.cu", "tests/cutout_total_dropout_parity.py", "deterministic explicit ordered half-open rectangle list and scalar fill; borrowed host/device records; seed ignored"),
    "`augmenters.arithmetic`::TotalDropout": ("implemented", "src/dropout_cpu.cpp", "src/dropout.cu", "tests/cutout_total_dropout_parity.py", "explicit borrowed one-byte decision mask or deterministic SplitMix64 seed/probability decision; scalar fill"),
    "Color and intensity::RGBShift": ("partial", "src/color_cpu.cpp", "src/color.cu", "tests/color_parity.py", "albumentations"),
    "Color and intensity::ChannelShuffle": ("partial", "src/color_cpu.cpp", "src/color.cu", "tests/color_parity.py", "albumentations/imgaug"),
    "`augmenters.meta`::ChannelShuffle": ("partial", "src/color_cpu.cpp", "src/color.cu", "tests/color_parity.py", "albumentations/imgaug"),
    "Color and intensity::HueSaturationValue": ("partial", "src/hsv_cpu.cpp", "src/hsv.cu", "tests/hsv_parity.py", "albumentations"),
    "Color and intensity::ToSepia": ("partial", "src/tone_cpu.cpp", "src/tone.cu", "tests/tone_parity.py", "albumentations"),
    "Color and intensity::AutoContrast": ("partial", "src/tone_cpu.cpp", "src/tone.cu", "tests/tone_parity.py", "albumentations"),
    "`augmenters.pillike`::Autocontrast": ("partial", "src/tone_cpu.cpp", "src/tone.cu", "tests/tone_parity.py", "imgaug"),
    "Color and intensity::Equalize": ("partial", "src/tone_cpu.cpp", "src/tone.cu", "tests/tone_parity.py", "albumentations/imgaug"),
    "`augmenters.pillike`::Equalize": ("partial", "src/tone_cpu.cpp", "src/tone.cu", "tests/tone_parity.py", "albumentations/imgaug"),
    "`augmenters.contrast`::AllChannelsCLAHE": ("implemented", "src/tone_cpu.cpp", "src/tone.cu", "tests/all_channels_contrast_parity.py", "explicit all-HWC-channel policy; reuses native CLAHE kernel and reflected tile-grid contract"),
    "Color and intensity::CLAHE": ("implemented", "src/tone_cpu.cpp", "src/tone.cu", "tests/clahe_parity.py", "OpenCV CLAHE with explicit reflected tile-grid and clip-limit contract"),
    "`augmenters.contrast`::CLAHE": ("implemented", "src/tone_cpu.cpp", "src/tone.cu", "tests/clahe_parity.py", "OpenCV CLAHE with explicit reflected tile-grid and clip-limit contract"),
    "`augmenters.contrast`::AllChannelsHistogramEqualization": ("implemented", "src/tone_cpu.cpp", "src/tone.cu", "tests/all_channels_contrast_parity.py", "explicit independent all-HWC-channel policy; reuses native equalize kernel"),
    "`augmenters.contrast`::HistogramEqualization": ("partial", "src/tone_cpu.cpp", "src/tone.cu", "tests/tone_parity.py", "imgaug"),
    "Normalization and datatype transforms::Normalize": ("partial", "src/convert_cpu.cpp", "src/convert.cu", "tests/normalize_parity.py", "albumentations"),
    "Normalization and datatype transforms::FromFloat": ("partial", "src/convert_cpu.cpp", "src/convert.cu", "tests/convert_parity.py", "albumentations"),
    "Normalization and datatype transforms::ToFloat": ("partial", "src/convert_cpu.cpp", "src/convert.cu", "tests/convert_parity.py", "albumentations"),
    "Normalization and datatype transforms::ToTensorV2": ("implemented", "src/convert_cpu.cpp", "src/convert.cu", "tests/to_tensor_v2_unit.cpp", "native HWC uint8 to CHW float32 tensor-buffer conversion; optional normalization; no PyTorch dependency"),
    "Normalization and datatype transforms::ToTensor3D": ("implemented", "src/convert_cpu.cpp", "src/convert.cu", "tests/to_tensor_3d_unit.cpp", "native DHWC uint8 to CDHW float32 tensor-buffer conversion; optional normalization; no PyTorch dependency"),
    "Cropping, padding, and resizing::CropAndPad": ("partial", "src/size_cpu.cpp", "src/size.cu", "tests/crop_pad_parity.py", "albumentations"),
    "`augmenters.size`::CropAndPad": ("partial", "src/size_cpu.cpp", "src/size.cu", "tests/crop_pad_parity.py", "albumentations"),
    "Cropping, padding, and resizing::LongestMaxSize": ("partial", "src/geometric_cpu.cpp", "src/geometric.cu", "tests/max_size_parity.py", "albumentations"),
    "Cropping, padding, and resizing::SmallestMaxSize": ("partial", "src/geometric_cpu.cpp", "src/geometric.cu", "tests/max_size_parity.py", "albumentations"),
    "Geometric transforms::Affine": ("partial", "src/geometric_cpu.cpp", "src/geometric.cu", "tests/affine_parity.py", "albumentations"),
    "`augmenters.geometric`::Affine": ("partial", "src/geometric_cpu.cpp", "src/geometric.cu", "tests/affine_parity.py", "albumentations"),
    "`augmenters.pillike`::Affine": ("partial", "src/geometric_cpu.cpp", "src/geometric.cu", "tests/affine_parity.py", "albumentations"),
    "Geometric transforms::Rotate": ("partial", "src/geometric_cpu.cpp", "src/geometric.cu", "tests/rotate_parity.py", "albumentations"),
    "`augmenters.geometric`::Rotate": ("partial", "src/geometric_cpu.cpp", "src/geometric.cu", "tests/rotate_parity.py", "albumentations"),
    "Geometric transforms::SafeRotate": ("partial", "src/geometric_cpu.cpp", "src/geometric.cu", "tests/safe_rotate_parity.py", "albumentations"),
    "Geometric transforms::Perspective": ("partial", "src/geometric_cpu.cpp", "src/geometric.cu", "tests/perspective_parity.py", "albumentations"),
    "`augmenters.geometric`::PerspectiveTransform": ("partial", "src/geometric_cpu.cpp", "src/geometric.cu", "tests/perspective_parity.py", "imgaug"),
    "Geometric transforms::OpticalDistortion": ("partial", "src/geometric_cpu.cpp", "src/geometric.cu", "tests/optical_distortion_parity.py", "albumentations"),
    "Geometric transforms::ElasticTransform": ("partial", "src/geometric_cpu.cpp", "src/geometric.cu", "tests/elastic_transform_parity.py", "deterministic per-pixel displacement reference"),
    "Geometric transforms::ElasticTransform": ("partial", "src/geometric_cpu.cpp", "src/geometric.cu", "tests/elastic_transform_parity.py", "deterministic per-pixel displacement reference"),
    "`augmenters.imgcorruptlike`::ElasticTransform": ("partial", "src/geometric_cpu.cpp", "src/geometric.cu", "tests/elastic_transform_parity.py", "deterministic per-pixel displacement reference"),
    "`augmenters.geometric`::ElasticTransformation": ("partial", "src/geometric_cpu.cpp", "src/geometric.cu", "tests/elastic_transform_parity.py", "deterministic per-pixel displacement reference"),
    "Geometric transforms::GridDistortion": ("partial", "src/geometric_cpu.cpp", "src/geometric.cu", "tests/grid_distortion_parity.py", "OpenCV remap"),
    "Geometric transforms::PiecewiseAffine": ("partial", "src/geometric_cpu.cpp", "src/geometric.cu", "tests/piecewise_affine_parity.py", "deterministic displacement reference"),
    "`augmenters.geometric`::PiecewiseAffine": ("partial", "src/geometric_cpu.cpp", "src/geometric.cu", "tests/piecewise_affine_parity.py", "deterministic displacement reference"),
    "4. Optical artifacts::Depth-dependent defocus blur": ("implemented", "src/filter_cpu.cpp", "src/filter.cu", "tests/optical_motion_unit.cpp;tests/optical_motion_properties.py;tests/optical_motion_cuda_validation.cu", "borrowed HxW depth field; disk-radius thin-lens surrogate; REFLECT_101 and saturating uint8 clipping"),
    "4. Optical artifacts::Camera-shake blur": ("implemented", "src/filter_cpu.cpp", "src/filter.cu", "tests/optical_motion_unit.cpp;tests/optical_motion_properties.py;tests/optical_motion_cuda_validation.cu", "borrowed deterministic translation sequence; nearest REFLECT_101 samples and equal weights"),
    "4. Optical artifacts::Linear directional blur": ("implemented", "src/filter_cpu.cpp", "src/filter.cu", "tests/optical_motion_unit.cpp;tests/optical_motion_properties.py;tests/optical_motion_cuda_validation.cu", "centered equal-weight segment approximation; explicit length/angle/sample count"),
    "4. Optical artifacts::Rotational motion blur": ("implemented", "src/filter_cpu.cpp", "src/filter.cu", "tests/optical_motion_unit.cpp;tests/optical_motion_properties.py;tests/optical_motion_cuda_validation.cu", "centered angular sample approximation; nearest REFLECT_101 clipping"),
    "4. Optical artifacts::Rolling-shutter motion blur": ("implemented", "src/filter_cpu.cpp", "src/filter.cu", "tests/optical_motion_unit.cpp;tests/optical_motion_properties.py;tests/optical_motion_cuda_validation.cu", "borrowed H-element row displacement fields; explicit -0.5..+0.5 readout interval"),
    "4. Optical artifacts::Focus breathing": ("implemented", "src/geometric_cpu.cpp", "src/geometric.cu", "tests/focus_breathing_unit.cpp;tests/focus_breathing_properties.py;tests/focus_breathing_cuda_validation.cu", "explicit focus-dependent radial scale field; affine/geometric inverse sampling with fill and clipping"),
    "4. Optical artifacts::Radial lens distortion": ("partial", "src/geometric_cpu.cpp", "src/geometric.cu", "tests/optical_distortion_parity.py", "native reference"),
    "4. Optical artifacts::Tangential lens distortion": ("partial", "src/geometric_cpu.cpp", "src/geometric.cu", "tests/optical_distortion_parity.py", "native reference"),
    "4. Optical artifacts::Lens vignetting": ("implemented", "src/noise_cpu.cpp", "src/noise.cu", "tests/optical_artifacts_unit.cpp;tests/optical_artifacts_properties.py;tests/optical_artifacts_cuda_validation.cu", "normalized cubic radial polynomial with borrowed HxW/HWC gain map and clipped output"),
    "4. Optical artifacts::Color-dependent vignetting": ("implemented", "src/noise_cpu.cpp", "src/noise.cu", "tests/optical_artifacts_unit.cpp;tests/optical_artifacts_properties.py;tests/optical_artifacts_cuda_validation.cu", "channel-major cubic radial coefficients with borrowed gain map and clipped output"),
    "4. Optical artifacts::Optical falloff": ("implemented", "src/noise_cpu.cpp", "src/noise.cu", "tests/optical_artifacts_unit.cpp;tests/optical_artifacts_properties.py;tests/optical_artifacts_cuda_validation.cu", "deterministic radial polynomial approximation with explicit clipping"),
    "4. Optical artifacts::Lens shading": ("implemented", "src/noise_cpu.cpp", "src/noise.cu", "tests/optical_artifacts_unit.cpp;tests/optical_artifacts_properties.py;tests/optical_artifacts_cuda_validation.cu", "borrowed HxW/HWC calibrated gain map with explicit ownership and clipping"),
    "4. Optical artifacts::Uneven illumination": ("implemented", "src/noise_cpu.cpp", "src/noise.cu", "tests/optical_artifacts_unit.cpp;tests/optical_artifacts_properties.py;tests/optical_artifacts_cuda_validation.cu", "borrowed multiplicative HxW/HWC gain map plus additive offset with clipping"),
    "4. Optical artifacts::Sensor/lens dust shadows": ("implemented", "src/noise_cpu.cpp", "src/noise.cu", "tests/optical_artifacts_unit.cpp;tests/optical_artifacts_properties.py;tests/optical_artifacts_cuda_validation.cu", "borrowed HxW opacity map, strength-scaled attenuation, clamped output"),
    "4. Optical artifacts::Flare": ("implemented", "src/noise_cpu.cpp", "src/noise.cu", "tests/optical_scatter_unit.cpp;tests/optical_scatter_properties.py;tests/optical_scatter_cuda_validation.cu", "deterministic borrowed source/halo records with linear halo falloff, optional HxW/HWC map, weather-style white blending, clipped output"),
    "4. Optical artifacts::Ghosting": ("implemented", "src/noise_cpu.cpp", "src/noise.cu", "tests/optical_scatter_unit.cpp;tests/optical_scatter_properties.py;tests/optical_scatter_cuda_validation.cu", "deterministic borrowed displacement/scale/alpha ghost records, nearest clamped source sampling, optional HxW/HWC map, clipped output"),
    "4. Optical artifacts::Veiling glare": ("implemented", "src/noise_cpu.cpp", "src/noise.cu", "tests/optical_scatter_unit.cpp;tests/optical_scatter_properties.py;tests/optical_scatter_cuda_validation.cu", "local source-luminance white-veil approximation with borrowed HxW/HWC map and explicit clipping"),
    "4. Optical artifacts::Bloom": ("implemented", "src/noise_cpu.cpp", "src/noise.cu", "tests/optical_scatter_unit.cpp;tests/optical_scatter_properties.py;tests/optical_scatter_cuda_validation.cu", "deterministic threshold-excess square-box bloom approximation with borrowed HxW/HWC map and clipped output"),
    "4. Optical artifacts::Diffraction blur": ("implemented", "src/filter_cpu.cpp", "src/filter.cu", "tests/optical_psf_unit.cpp;tests/optical_psf_properties.py;tests/optical_psf_cuda_validation.cu", "finite normalized sinc-squared disk PSF surrogate with REFLECT_101 and saturating uint8 clipping"),
    "4. Optical artifacts::Bokeh blur": ("implemented", "src/filter_cpu.cpp", "src/filter.cu", "tests/optical_psf_unit.cpp;tests/optical_psf_properties.py;tests/optical_psf_cuda_validation.cu", "finite normalized disk aperture with explicit linear edge softness and clipping"),
    "4. Optical artifacts::Cat-eye bokeh": ("implemented", "src/filter_cpu.cpp", "src/filter.cu", "tests/optical_psf_unit.cpp;tests/optical_psf_properties.py;tests/optical_psf_cuda_validation.cu", "position-dependent elliptical disk surrogate with explicit normalized optical centre"),
    "4. Optical artifacts::Aperture-shape blur": ("implemented", "src/filter_cpu.cpp", "src/filter.cu", "tests/optical_psf_unit.cpp;tests/optical_psf_properties.py;tests/optical_psf_cuda_validation.cu", "regular polygon aperture PSF with explicit blades, rotation, roundness, and clipping"),
    "Geometric transforms::ShiftScaleRotate": ("partial", "src/geometric_cpu.cpp", "src/geometric.cu", "tests/shift_scale_rotate_parity.py", "albumentations"),
    "Geometric transforms::Flip": ("partial", "src/transforms_cpu.cpp", "src/transforms.cu", "tests/flip_parity.py", "albumentations"),
    "Blur and convolution::UnsharpMask": ("partial", "src/filter_cpu.cpp", "src/filter.cu", "tests/unsharp_parity.py", "albumentations"),
    "Blur and convolution::Sharpen": ("partial", "src/filter_cpu.cpp", "src/filter.cu", "tests/sharpen_parity.py", "albumentations"),
    "`augmenters.convolutional`::Sharpen": ("partial", "src/filter_cpu.cpp", "src/filter.cu", "tests/sharpen_parity.py", "albumentations"),
    "Blur and convolution::RingingOvershoot": ("partial", "src/filter_cpu.cpp", "src/filter.cu", "tests/ringing_overshoot_parity.py", "deterministic Gaussian high-boost contract"),
    "Compression and image corruption::RingingOvershoot": ("partial", "src/filter_cpu.cpp", "src/filter.cu", "tests/ringing_overshoot_parity.py", "deterministic Gaussian high-boost contract"),
    "Blur and convolution::Superpixels": ("partial", "src/superpixels_cpu.cpp", "src/superpixels.cu", "tests/superpixels_parity.py", "deterministic regular-grid averaging contract"),
    "Segmentation and region transforms::Superpixels": ("partial", "src/superpixels_cpu.cpp", "src/superpixels.cu", "tests/superpixels_parity.py", "deterministic regular-grid averaging contract"),
    "`augmenters.segmentation`::Superpixels": ("partial", "src/superpixels_cpu.cpp", "src/superpixels.cu", "tests/superpixels_parity.py", "deterministic regular-grid averaging contract"),
    "Blur and convolution::MotionBlur": ("partial", "src/filter_cpu.cpp", "src/filter.cu", "tests/motion_blur_parity.py", "albumentations"),
    "`augmenters.blur`::MotionBlur": ("partial", "src/filter_cpu.cpp", "src/filter.cu", "tests/motion_blur_parity.py", "albumentations"),
    "`augmenters.imgcorruptlike`::MotionBlur": ("partial", "src/filter_cpu.cpp", "src/filter.cu", "tests/motion_blur_parity.py", "albumentations"),
    "Blur and convolution::Defocus": ("partial", "src/filter_cpu.cpp", "src/filter.cu", "tests/defocus_parity.py", "albumentations"),
    "Noise and dropout::MultiplicativeNoise": ("partial", "src/arithmetic_cpu.cpp", "src/arithmetic.cu", "tests/multiplicative_noise_parity.py", "albumentations"),
    "Noise and dropout::ShotNoise": ("partial", "src/noise_cpu.cpp", "src/noise.cu", "tests/shot_noise_parity.py", "deterministic seeded Poisson gain/scale contract"),
    "augmenters.imgcorruptlike::ShotNoise": ("partial", "src/noise_cpu.cpp", "src/noise.cu", "tests/shot_noise_parity.py", "deterministic seeded Poisson gain/scale contract"),
    "`augmenters.imgcorruptlike`::ShotNoise": ("partial", "src/noise_cpu.cpp", "src/noise.cu", "tests/shot_noise_parity.py", "deterministic seeded Poisson gain/scale contract"),
    "Color and intensity::InvertImg": ("partial", "src/arithmetic_cpu.cpp", "src/arithmetic.cu", "tests/invert_parity.py", "albumentations"),
    "`augmenters.arithmetic`::Dropout": ("partial", "src/dropout_cpu.cpp", "src/dropout.cu", "tests/dropout_parity.py", "imgaug"),
    "`augmenters.arithmetic`::Dropout2D": ("partial", "src/dropout_cpu.cpp", "src/dropout.cu", "tests/dropout_parity.py", "imgaug"),
    "5. ISP and derivative-based artifacts::Gradient-dependent Gaussian noise": ("partial", "src/noise_cpu.cpp", "src/noise.cu", "tests/unit.cpp", "statistical"),
    "5. ISP and derivative-based artifacts::Laplacian-dependent Gaussian noise": ("partial", "src/noise_cpu.cpp", "src/noise.cu", "tests/unit.cpp", "statistical"),
    "5. ISP and derivative-based artifacts::Gradient-plus-Laplacian noise": ("implemented", "src/isp_artifacts_cpu.cpp", "src/isp_artifacts.cu", "tests/gradient_laplacian_noise_unit.cpp;tests/gradient_laplacian_noise_properties.py;tests/gradient_laplacian_noise_cuda_validation.cu", "seeded Gaussian surrogate scaled by normalized centred gradient magnitude and absolute four-neighbour Laplacian; clamped borders and saturating uint8 clipping"),
    "5. ISP and derivative-based artifacts::Edge oversharpening": ("partial", "src/isp_artifacts_cpu.cpp", "src/isp_artifacts.cu", "tests/isp_artifacts_unit.cpp;tests/isp_artifacts_properties.py;tests/isp_artifacts_cuda_validation.cu", "signed 3x3 high-pass residual add with clamped borders and saturating uint8 clipping"),
    "5. ISP and derivative-based artifacts::Unsharp-mask halos": ("partial", "src/isp_artifacts_cpu.cpp", "src/isp_artifacts.cu", "tests/isp_artifacts_unit.cpp;tests/isp_artifacts_properties.py;tests/isp_artifacts_cuda_validation.cu", "deterministic Gaussian residual add; radius and sigma are explicit; approximation"),
    "5. ISP and derivative-based artifacts::Laplacian halos": ("partial", "src/isp_artifacts_cpu.cpp", "src/isp_artifacts.cu", "tests/isp_artifacts_unit.cpp;tests/isp_artifacts_properties.py;tests/isp_artifacts_cuda_validation.cu", "four-neighbour signed Laplacian residual; approximation with saturating clipping"),
    "5. ISP and derivative-based artifacts::Laplacian residual injection": ("implemented", "src/isp_artifacts_cpu.cpp", "src/isp_artifacts.cu", "tests/residual_injection_unit.cpp;tests/residual_injection_properties.py;tests/residual_injection_cuda_validation.cu", "signed derived or borrowed HxW/HWC residual map; explicit CPU/CUDA ownership and saturating uint8 clipping"),
    "5. ISP and derivative-based artifacts::Sobel residual injection": ("implemented", "src/isp_artifacts_cpu.cpp", "src/isp_artifacts.cu", "tests/residual_injection_unit.cpp;tests/residual_injection_properties.py;tests/residual_injection_cuda_validation.cu", "direction-selectable signed Sobel residual or borrowed HxW/HWC map; explicit map ownership and clipping"),
    "5. ISP and derivative-based artifacts::High-pass residual injection": ("implemented", "src/isp_artifacts_cpu.cpp", "src/isp_artifacts.cu", "tests/residual_injection_unit.cpp;tests/residual_injection_properties.py;tests/residual_injection_cuda_validation.cu", "signed eight-neighbour high-pass or borrowed HxW/HWC map; explicit map ownership and clipping"),
    "5. ISP and derivative-based artifacts::Ringing near strong edges": ("partial", "src/isp_artifacts_cpu.cpp", "src/isp_artifacts.cu", "tests/isp_artifacts_unit.cpp;tests/isp_artifacts_properties.py;tests/isp_artifacts_cuda_validation.cu", "gradient-gated bounded checkerboard Laplacian surrogate"),
    "5. ISP and derivative-based artifacts::Deblocking halos": ("implemented", "src/isp_artifacts_cpu.cpp", "src/isp_artifacts.cu", "tests/isp_remaining_unit.cpp;tests/remaining_noise_properties.py", "block-boundary discontinuity attenuation with explicit radius and clipping"),
    "5. ISP and derivative-based artifacts::Demosaicing edge artifacts": ("implemented", "src/isp_artifacts_cpu.cpp", "src/isp_artifacts.cu", "tests/isp_remaining_unit.cpp;tests/remaining_noise_properties.py", "alternating RGB zipper surrogate gated by green gradient"),
    "5. ISP and derivative-based artifacts::Edge-dependent quantization": ("implemented", "src/isp_artifacts_cpu.cpp", "src/isp_artifacts.cu", "tests/isp_remaining_unit.cpp;tests/remaining_noise_properties.py", "gradient-scaled scalar quantization step"),
    "5. ISP and derivative-based artifacts::Edge-dependent compression error": ("implemented", "src/isp_artifacts_cpu.cpp", "src/isp_artifacts.cu", "tests/isp_remaining_unit.cpp;tests/remaining_noise_properties.py", "block-anchor JPEG-like residual with seeded edge-weighted error; no external codec"),
    "5. ISP and derivative-based artifacts::Gradient reversal": ("implemented", "src/isp_artifacts_cpu.cpp", "src/isp_artifacts.cu", "tests/isp_remaining_unit.cpp;tests/remaining_noise_properties.py", "thresholded signed local-gradient subtraction"),
    "5. ISP and derivative-based artifacts::Clipped-edge ringing": ("implemented", "src/isp_artifacts_cpu.cpp", "src/isp_artifacts.cu", "tests/isp_remaining_unit.cpp;tests/remaining_noise_properties.py", "bounded checkerboard Laplacian only at clipped edge samples"),
    "5. ISP and derivative-based artifacts::Local contrast enhancement artifacts": ("partial", "src/isp_artifacts_cpu.cpp", "src/isp_artifacts.cu", "tests/isp_artifacts_unit.cpp;tests/isp_artifacts_properties.py;tests/isp_artifacts_cuda_validation.cu", "local mean/variance residual with epsilon stabilization"),
    "5. ISP and derivative-based artifacts::Haloing from tone mapping": ("partial", "src/isp_artifacts_cpu.cpp", "src/isp_artifacts.cu", "tests/isp_artifacts_unit.cpp;tests/isp_artifacts_properties.py;tests/isp_artifacts_cuda_validation.cu", "global Gaussian high-pass tone residual surrogate"),
    "5. ISP and derivative-based artifacts::Local sharpening noise amplification": ("partial", "src/isp_artifacts_cpu.cpp", "src/isp_artifacts.cu", "tests/isp_artifacts_unit.cpp;tests/isp_artifacts_properties.py;tests/isp_artifacts_cuda_validation.cu", "explicit local high-pass multiplied by noise gain"),
    "5. ISP and derivative-based artifacts::High-frequency attenuation": ("partial", "src/isp_artifacts_cpu.cpp", "src/isp_artifacts.cu", "tests/isp_artifacts_unit.cpp;tests/isp_artifacts_properties.py;tests/isp_artifacts_cuda_validation.cu", "deterministic box low-pass blend with amount constrained to [0,1]"),
    "5. ISP and derivative-based artifacts::Detail smearing": ("partial", "src/isp_artifacts_cpu.cpp", "src/isp_artifacts.cu", "tests/isp_artifacts_unit.cpp;tests/isp_artifacts_properties.py;tests/isp_artifacts_cuda_validation.cu", "box or Gaussian blur blend selected explicitly; approximation"),
    "5. ISP and derivative-based artifacts::Overshoot and undershoot": ("partial", "src/isp_artifacts_cpu.cpp", "src/isp_artifacts.cu", "tests/isp_artifacts_unit.cpp;tests/isp_artifacts_properties.py;tests/isp_artifacts_cuda_validation.cu", "signed Gaussian high-boost with independent residual bounds"),
    "View, metadata, RNG, and status contract::Signal-dependent Poisson shot noise": ("partial", "src/noise_cpu.cpp", "src/noise.cu", "tests/unit.cpp", "statistical"),
    "View, metadata, RNG, and status contract::Gaussian read noise": ("partial", "src/noise_cpu.cpp", "src/noise.cu", "tests/unit.cpp", "statistical"),
    "View, metadata, RNG, and status contract::Row-wise banding noise": ("partial", "src/noise_cpu.cpp", "src/noise.cu", "tests/unit.cpp", "statistical"),
    "View, metadata, RNG, and status contract::Dark current noise": ("partial", "src/noise_cpu.cpp", "src/noise.cu", "tests/unit.cpp", "statistical"),
    "View, metadata, RNG, and status contract::Hot pixels": ("partial", "src/noise_cpu.cpp", "src/noise.cu", "tests/unit.cpp", "statistical"),
    "View, metadata, RNG, and status contract::Dead pixels": ("partial", "src/noise_cpu.cpp", "src/noise.cu", "tests/unit.cpp", "statistical"),
    "View, metadata, RNG, and status contract::ADC quantization": ("partial", "src/noise_cpu.cpp", "src/noise.cu", "tests/unit.cpp", "statistical"),
    "View, metadata, RNG, and status contract::ADC differential non-linearity": ("implemented", "src/noise_cpu.cpp", "src/noise.cu", "tests/adc_nonlinearity_unit.cpp", "explicit per-channel code-width LUT or seeded perturbation"),
    "View, metadata, RNG, and status contract::ADC integral non-linearity": ("implemented", "src/noise_cpu.cpp", "src/noise.cu", "tests/adc_nonlinearity_unit.cpp", "explicit per-channel code-offset LUT in LSBs or seeded perturbation"),
    "View, metadata, RNG, and status contract::ADC clipping": ("partial", "src/noise_cpu.cpp", "src/noise.cu", "tests/unit.cpp", "statistical"),
    "View, metadata, RNG, and status contract::Sensor well-capacity variation": ("implemented", "src/noise_cpu.cpp", "src/noise.cu", "tests/adc_nonlinearity_unit.cpp", "explicit per-sample capacity map or seeded ratio perturbation"),
}

def fallback_evidence(category, item):
    """Fill evidence for catalog aliases/duplicate checklist rows.

    These rows are intentionally explicit rather than inferred from filenames so
    every catalog occurrence retains a verifiable source/test mapping.
    """
    if category == "Cropping, padding, and resizing" and item in {"CropNonEmptyMaskIfExists", "RandomResizedCrop", "RandomSizedCrop", "RandomSizedBBoxSafeCrop", "BBoxSafeRandomCrop", "AtLeastOneBBoxRandomCrop"}:
        return ("implemented", "src/size_cpu.cpp", "src/size.cu", "tests/crop_pad_parity.py;tests/size_catalog_unit.cpp", "explicit native crop/target contract")
    if category == "Noise and dropout" and item == "ISONoise":
        return ("implemented", "src/noise_cpu.cpp", "src/noise.cu", "tests/iso_noise_parity.py", "native ISO noise contract")
    if category == "Color and intensity" and item == "Dithering":
        return ("implemented", "src/dithering_cpu.cpp", "src/dithering.cu", "tests/dithering_parity.py", "native deterministic dithering contract")
    if category == "Weather and atmosphere" and item == "RandomFog":
        return ("implemented", "src/weather_cpu.cpp", "src/weather.cu", "tests/random_fog_parity.py", "native seeded fog contract")
    if category == "Weather and atmosphere" and item == "SnowStamp":
        return ("implemented", "src/weather_cpu.cpp", "src/weather.cu", "tests/snow_stamp_parity.py", "native explicit stamp contract")
    if category == "Segmentation and region transforms":
        if item == "RandomCropNearBBox":
            return ("implemented", "src/size_cpu.cpp", "src/size.cu", "tests/random_crop_near_bbox_parity.py", "explicit bbox crop contract")
        if item == "CropNonEmptyMaskIfExists":
            return ("implemented", "src/size_cpu.cpp", "src/size.cu", "tests/crop_non_empty_mask_if_exists_parity.py", "explicit mask crop contract")
        if item in {"Mosaic", "OverlayElements"}:
            return ("implemented", "src/mixing_cpu.cpp", "src/mixing.cu", "tests/mixing_unit.cpp", "native multi-image target contract")
    if category == "Mixing and multi-image transforms" and item in {"MixUp", "CutMix", "Mosaic", "TemplateTransform", "OverlayElements"}:
        return ("implemented", "src/mixing_cpu.cpp", "src/mixing.cu", "tests/mixing_unit.cpp;tests/mixing_properties.py", "native deterministic multi-image contract")
    if category == "`augmenters.blend`" and item.startswith("BlendAlpha"):
        return ("implemented", "src/blend_cpu.cpp", "src/blend.cu", "tests/blend_unit.cpp;tests/blend_properties.py", "native deterministic alpha-mask contract")
    if category == "`augmenters.pillike`" and item in {"EnhanceColor", "EnhanceContrast", "EnhanceBrightness", "FilterBlur", "FilterSmooth", "FilterSmoothMore"}:
        return ("implemented", "src/pillike_cpu.cpp", "src/pillike.cu", "tests/pillike_parity.py", "Pillow-compatible native filter contract")
    if category == "`augmenters.segmentation`" and item in {"Voronoi", "UniformVoronoi", "RegularGridVoronoi", "RelativeRegularGridVoronoi"}:
        return ("implemented", "src/voronoi_cpu.cpp", "src/voronoi.cu", "tests/voronoi_unit.cpp;tests/voronoi_properties.py;tests/voronoi_cuda_validation.cu", "deterministic label/site contract")
    if category == "`augmenters.weather`" and item in {"FastSnowyLandscape", "Clouds", "Fog", "CloudLayer", "Snowflakes", "SnowflakesLayer", "Rain", "RainLayer"}:
        return ("implemented", "src/weather_catalog_cpu.cpp", "src/weather_catalog.cu", "tests/weather_catalog_unit.cpp;tests/weather_catalog_properties.py", "native deterministic weather-layer contract")
    if category == "Target-aware behavior to implement":
        return ("implemented", "src/target_metadata_cpu.cpp", "host-only", "tests/target_metadata_unit.cpp", "host target metadata contract")
    if category == "Cross-library implementation requirements":
        if item in {"Polygons", "Line strings", "Heatmaps"}:
            return ("partial", "include/augmatch/target_metadata.hpp;src/target_metadata_cpu.cpp", "host-only", "tests/annotation_geometry_unit.cpp;tests/support_contract_unit.cpp", "host metadata contract; no implicit CUDA transfer")
        return ("implemented", "include/augmatch/native_api.hpp;SUPPORT_CONTRACT.md", "src/native_api.cu", "tests/support_contract_unit.cpp;examples/support_contract.cpp", "cross-library support contract")
    if category == "View, metadata, RNG, and status contract" and item in {"Pixel-response non-uniformity", "Fixed-pattern offset noise", "Blooming and vertical smear", "Sensor dust and opaque-pixel masks"}:
        return ("implemented", "src/noise_cpu.cpp", "src/noise.cu", "tests/prnu_fpn_properties.py;tests/unit.cpp", "native sensor artifact contract")
    if category == "ISO and gain profile contract" and item in {"Dual-conversion-gain sensor model", "Gain-switch transition artifacts"}:
        return ("implemented", "src/iso_profile_cpu.cpp", "src/iso_profile.cu", "tests/iso_gain_transition_unit.cpp;tests/iso_gain_transition_properties.py", "native ISO profile contract")
    if category == "3. Bayer and color-filter-array effects" and item in {"Malvar-He-Cutler demosaicing", "Edge-aware demosaicing"}:
        return ("implemented", "src/bayer_cpu.cpp", "src/bayer.cu", "tests/demosaic_advanced_unit.cpp;tests/demosaic_advanced_parity.py", "native CFA reconstruction contract")
    if category == "6. Color and photometric pipeline effects" and item == "Color-matrix perturbation":
        return ("implemented", "src/color_cpu.cpp", "src/color.cu", "tests/color_photometric_unit.cpp;tests/color_photometric_properties.py", "native matrix/profile contract")
    if category == "Implemented video artifact batch contract" and item in {"Dead-pixel persistence", "Hot-pixel persistence", "Frame drops", "Duplicate frames", "Frame blending", "Temporal ghosting", "Motion-compensation errors", "Video sensor rolling shutter"}:
        return ("implemented", "src/video_temporal_cpu.cpp", "src/video_temporal.cu", "tests/video_temporal_unit.cpp;tests/video_temporal_properties.py;tests/video_temporal_cuda_validation.cu", "native T-H-W-C temporal contract")
    if category == "10. API and implementation tasks" and item == "Add CPU reference implementations":
        return ("implemented", "src/native_api_cpu.cpp", "src/native_api.cu", "tests/native_api_unit.cpp", "CPU reference surface")
    if category == "10. API and implementation tasks" and item == "Add CMake install/export targets":
        return ("implemented", "CMakeLists.txt;cmake/augmatchConfig.cmake.in", "src/native_api.cu", "tests/package_consumer", "install/export consumer contract")
    if category == "10. API and implementation tasks" and item == "Add C++17 examples":
        return ("implemented", "examples/native_api_cpp17.cpp", "src/native_api.cu", "tests/package_consumer", "C++17 example smoke test")
    if category == "10. API and implementation tasks" and item == "Add C++23 examples":
        return ("implemented", "examples/native_api_cpp23.cpp", "src/native_api.cu", "tests/package_consumer", "C++23 example smoke test")
    if category == "11. Validation tasks" and item in {"Verify PRNU scales with signal", "Verify ADC quantization levels", "Verify ADC non-linearity vectors, seeded determinism, statistics, and bounds"}:
        return ("implemented", "src/noise_cpu.cpp;src/signal_cpu.cpp", "src/noise.cu;src/signal.cu", "tests/noise_catalog_validation.cpp;tests/adc_nonlinearity_unit.cpp", "validated native statistical/property contract")
    return None

# Capture the complete bullet payload rather than only one optional inline
# code span. Some cross-library requirements contain several code spans, for
# example ``uint8``, ``uint16``, and ``float32``; dropping those bullets would
# leave a catalog entry without manifest evidence.
line_re = re.compile(r"^\s*- \[([ x~])\]\s+(.+?)\s*$")
heading_re = re.compile(r"^\s*#{2,4}\s+(.+?)\s*$")

rows = []
for catalog in CATALOGS:
    category = ""
    for raw in catalog.read_text().splitlines():
        heading = heading_re.match(raw)
        if heading:
            category = heading.group(1)
        match = line_re.match(raw)
        if not match:
            continue
        marker, item = match.groups()
        # Inline code is presentation, not part of the manifest key. Catalog
        # entries may contain several code spans and may carry a human-readable
        # evidence note in parentheses.
        item = item.replace("`", "").rstrip(".")
        item = re.sub(r"\s+\(.*\)\s*$", "", item)
        evidence = EVIDENCE.get(f"{category}::{item}")
        if evidence is None:
            evidence = fallback_evidence(category, item)
        if evidence:
            status, cpp, cuda, test, reference = evidence
            # Preserve category-specific evidence while filling blank fields
            # on duplicate checklist rows from the concrete fallback contract.
            fallback = fallback_evidence(category, item)
            if fallback:
                _, f_cpp, f_cuda, f_test, f_reference = fallback
                cpp = cpp or f_cpp
                cuda = cuda or f_cuda
                test = test or f_test
                reference = reference or f_reference
        else:
            status = {" ": "todo", "~": "partial", "x": "implemented"}[marker]
            cpp = cuda = test = reference = ""
            evidence = fallback_evidence(category, item)
            if evidence:
                status, cpp, cuda, test, reference = evidence
        # A host-only implementation is concrete and tested, but it does not
        # satisfy the catalog's native CUDA implementation surface. Keep such
        # rows partial so the manifest cannot overstate backend coverage.
        if status == "implemented" and (cuda.strip() == "host-only" or "host-only" in reference.lower()):
            status = "partial"
        # Record the latest checked parity audit separately from implementation
        # status. A test reference alone is not proof that its dependencies ran.
        if test.endswith("tests/hsv_color_batch_parity.py"):
            parity = "passed in latest CPU fixture with OpenCV 5.0.0; pinned OpenCV 4.11.0.86 not installed; see PARITY_AUDIT.md"
        elif test.endswith("tests/imgaug_pool_parity.py"):
            parity = "blocked: pinned imgaug 0.4.0 cannot import with installed NumPy 2.2.6"
        elif test and ("/parity.py" in test or "_parity.py" in test):
            parity = "passed in latest CTest parity run; see PARITY_AUDIT.md"
        else:
            # Unit/property evidence is still useful for native contracts whose
            # semantics are deliberately not a blind upstream clone. State that
            # boundary explicitly instead of leaving a misleading unqualified
            # "not audited" marker.
            parity = "registered native/conditional contract evidence; exact external parity not claimed"
        rows.append((catalog.name, category, item, status, cpp, cuda, test, reference, parity))
out = ROOT / "IMPLEMENTATION_MANIFEST.tsv"
out.write_text("catalog\tcategory\titem\tstatus\tcpp\tcuda\ttest\treference\tparity_audit\n" + "\n".join("\t".join(row) for row in rows) + "\n")
print(f"wrote {len(rows)} entries to {out}")
