# augmatch documentation

New here? Read [guides/getting_started.md](guides/getting_started.md), then the
[examples](../examples/README.md) in order.

## Guides

| Page | Contents |
|---|---|
| [getting_started.md](guides/getting_started.md) | Views, `Status`, seeds, CPU vs CUDA, the `easy` API vs the raw API |
| [PIPELINE.md](guides/PIPELINE.md) | The `augmentation.cfg` format, built-in stages, and vendoring as a git submodule |
| [API_DESIGN.md](guides/API_DESIGN.md) | API conventions: views, execution context, composition, replay |
| [SUPPORT_CONTRACT.md](guides/SUPPORT_CONTRACT.md) | Cross-library support guarantees |
| [COMPATIBILITY.md](guides/COMPATIBILITY.md) | Platform and backend compatibility targets |

## API reference

Each page matches one header folder under `include/augmatch/`. It holds the exact
contracts: formulas, parameter ranges, memory ownership, and CPU/CUDA behaviour.

| Page | Header folder | Covers |
|---|---|---|
| [core.md](api/core.md) | `core/` | ToTensorV2 / ToTensor3D tensor buffers (views and `easy` are in the getting-started guide) |
| [annotations.md](api/annotations.md) | `annotations/` | Box formats, keypoints, masks, geometry targets |
| [pipeline.md](api/pipeline.md) | `pipeline/` | compose / one_of / some_of / replay, `augmenters.meta` |
| [geometry.md](api/geometry.md) | `geometry/` | Crop, flip, random crops, downscale, dropout, MixUp/CutMix/Mosaic |
| [color.md](api/color.md) | `color/` | Arithmetic, contrast, ColorJitter, HSV, temperature, CLAHE, dithering |
| [filter.md](api/filter.md) | `filter/` | Convolution, blur variants, Canny, Voronoi, blending |
| [sensor.md](api/sensor.md) | `sensor/` | Sensor noise, PRNU, ISO profiles, Bayer/CFA sampling |
| [isp.md](api/isp.md) | `isp/` | ISP artifacts and demosaicing |
| [optics.md](api/optics.md) | `optics/` | Optical motion, focus breathing, aperture PSF, chromatic aberration |
| [compression.md](api/compression.md) | `compression/` | JPEG, WebP, codec blockers, transport corruption |
| [video.md](api/video.md) | `video/` | Temporal and codec-surrogate artifacts |
| [weather.md](api/weather.md) | `weather/` | Rain, snow, fog, gravel, spatter, sun flare, shadows, environmental effects |
| [catalog.md](api/catalog.md) | `catalog/` | imgcorruptlike and the remaining imgaug catalog ops |
| [cli.md](api/cli.md) | — | Every `augmatch_cli` raw-byte command |

## Catalogs

Per-area transform lists with implementation checklists. Together with
[`IMPLEMENTATION_MANIFEST.tsv`](../IMPLEMENTATION_MANIFEST.tsv) they are the
source of truth for coverage status.

- [AUGMENTATION_CATALOG.md](catalogs/AUGMENTATION_CATALOG.md): Albumentations and imgaug transforms
- [NOISE_CATALOG.md](catalogs/NOISE_CATALOG.md): camera, sensor, optical, ISP, and compression noise
- [BLEND_CATALOG.md](catalogs/BLEND_CATALOG.md), [SIZE_CATALOG.md](catalogs/SIZE_CATALOG.md), [IMGCORRUPTLIKE_CATALOG.md](catalogs/IMGCORRUPTLIKE_CATALOG.md)

## Audits

- [COMPLETION_AUDIT.md](audits/COMPLETION_AUDIT.md): status of every catalog item
- [PARITY_AUDIT.md](audits/PARITY_AUDIT.md): latest parity-test evidence
- [NOISE_VALIDATION.md](audits/NOISE_VALIDATION.md): noise-catalog validation evidence
