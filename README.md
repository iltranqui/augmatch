# augmatch

A native C++17/CUDA image-augmentation library. Its transform catalog follows
Albumentations/imgaug semantics, and a config-driven `Pipeline` runs a whole
augmentation recipe from an INI file. You can embed it as a git submodule in
any C++/CUDA project.

## Features

- **One API, two backends.** CPU and CUDA share the same headers. The backend is
  chosen at build time (`AUGMATCH_ENABLE_CUDA`, default `ON`). If `nvcc` is
  missing, the build falls back to CPU-only.
- **A large catalog.** Crop, flip, and resize; color/HSV; blur and filters; noise;
  weather; JPEG/WebP and transport artifacts; sensor and ISP simulation; Bayer/CFA;
  mixing (MixUp, CutMix, Mosaic); and bounding-box, keypoint, and mask targets.
  The per-item status is in [`IMPLEMENTATION_MANIFEST.tsv`](IMPLEMENTATION_MANIFEST.tsv).
- **Deterministic.** Every random transform takes an explicit seed. There is no
  global RNG, and CPU and CUDA follow the same documented contract.
- **Zero-copy views.** You own the pixels. augmatch reads and writes through
  non-owning `ImageView`s.
- **Config-driven pipelines.** `augmentation.cfg` lists the stages and their
  parameters, so changing the recipe needs no recompilation. See
  [docs/guides/PIPELINE.md](docs/guides/PIPELINE.md).
- **Parity-tested.** Python tests compare against Albumentations/imgaug wherever
  an exact reference exists.

## Quickstart

```sh
git clone https://github.com/iltranqui/augmatch.git
cd augmatch
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j 4
ctest --test-dir build -C Release --output-on-failure
./build/example_01_hello_view
```

## Minimal example

```cpp
#include <vector>
#include "augmatch/augmatch.hpp"

int main() {
  const int w = 640, h = 480, c = 3;
  std::vector<std::uint8_t> image(w * h * c), noisy(image.size());  // your pixels, HWC RGB

  auto src = augmatch::make_hwc_u8_view(image.data(), w, h, c);     // views never copy
  auto dst = augmatch::make_hwc_u8_view(noisy.data(), w, h, c);

  augmatch::ISONoiseConfig noise;
  noise.iso = 1600.0f;
  noise.seed = 42;                                                  // same seed -> same output

  augmatch::Status s = augmatch::easy::iso_noise(src, dst, noise);  // works in CPU and CUDA builds
  if (!s) return 1;                                                 // s.message says why
  s = augmatch::easy::gaussian_blur(dst, src, /*kernel_size=*/5, /*sigma=*/1.2f);  // blur back into image
  return s ? 0 : 1;
}
```

The `augmatch::easy` wrappers take views and fill in the image size for you. The
full API in `include/augmatch/<topic>/` exposes every transform as
`name_u8(input, output, config, stream)` for zero-overhead use on device
memory. See [examples/](examples/README.md) for ten short, commented programs.

## Documentation

| Where | What |
|---|---|
| [docs/guides/getting_started.md](docs/guides/getting_started.md) | Views, Status, seeds, CPU vs CUDA, easy vs raw API |
| [docs/guides/PIPELINE.md](docs/guides/PIPELINE.md) | `augmentation.cfg` format, built-in stages, vendoring as a submodule |
| [docs/api/](docs/README.md#api-reference) | Per-topic reference: exact contracts, formulas, and ownership rules |
| [docs/api/cli.md](docs/api/cli.md) | Every `augmatch_cli` raw-byte command |
| [docs/catalogs/](docs/README.md#catalogs) | Transform catalogs and implementation checklists |
| [docs/README.md](docs/README.md) | Full documentation index |

## Header layout

Include `augmatch/augmatch.hpp` to get everything. Individual headers live in topic folders:

```
include/augmatch/
├── augmatch.hpp   umbrella header
├── core/          views, Status, ExecutionContext, conversion, easy.hpp wrappers
├── annotations/   boxes, keypoints, masks, geometry transforms
├── pipeline/      Pipeline, registry, compose/one_of/some_of, meta ops
├── geometry/      flips, crops, resize, affine, dropout, mixing
├── color/         color, HSV, tone, arithmetic, dithering, PIL-like
├── filter/        blur, convolution, edges, superpixels, blending
├── sensor/        sensor noise, ISO profiles, Bayer/CFA, native sensor API
├── isp/  optics/  ISP artifacts, lens helpers
├── compression/   JPEG, WebP, transport corruption
├── video/         codec and temporal APIs
├── weather/       rain, snow, fog, flare, environmental effects
└── catalog/       imgcorruptlike and remaining imgaug ops
```

> **Upgrading:** before this layout, headers sat directly under `include/augmatch/`.
> Replace `#include "augmatch/noise.hpp"` with `#include "augmatch/sensor/noise.hpp"`,
> and do the same for the other headers, or include `augmatch/augmatch.hpp`.

## Build options

| Option | Default | Effect |
|---|---|---|
| `AUGMATCH_ENABLE_CUDA` | `ON` | Build the CUDA backend when `nvcc` is found |
| `AUGMATCH_ENABLE_JPEG` / `_WEBP` / `_FFMPEG` | `ON` | Optional codecs, used only when found |
| `AUGMATCH_BUILD_TESTS` / `_EXAMPLES` | top-level only | Tests and examples; off when used as a subproject |

```sh
# CPU-only build (no CUDA runtime needed):
cmake -S . -B build-cpu -DAUGMATCH_ENABLE_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
# Python parity-test dependencies and manifest checks:
pip install -r requirements-test.txt
python3 tools_generate_manifest.py --check && python3 tools_audit_manifest.py
```

Requires CMake 3.24+ and a C++17 compiler, on Linux or Windows (use the Visual
Studio generator on Windows). The installed CMake package exports
`augmatch::augmatch`.

## License

MIT — see [`LICENSE`](LICENSE).
