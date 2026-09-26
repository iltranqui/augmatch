# Examples

Build them with the library (`AUGMATCH_BUILD_EXAMPLES` is `ON` by default for a
top-level build). Each numbered example is one small, commented program that
writes a `.ppm` image into the current directory. Most image viewers, GIMP, and
`convert` from ImageMagick can open `.ppm` files.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 4
cd build && ./example_01_hello_view
```

Start with `01` and read the examples in order. Each one adds a single idea.

| Example | What it shows | Key API |
|---|---|---|
| [`01_hello_view`](01_hello_view.cpp) | Wrapping your own buffer in a view, running one op, and checking `Status` | `make_hwc_u8_view`, `easy::flip_horizontal` |
| [`02_geometry_flip_crop`](02_geometry_flip_crop.cpp) | Cropping, then chaining a second op; out-of-bounds input is rejected | `easy::crop`, `easy::flip_vertical` |
| [`03_color_jitter`](03_color_jitter.cpp) | Config structs, plus running *any* raw op on views | `ColorJitterConfig`, `easy::apply` |
| [`04_blur_filter`](04_blur_filter.cpp) | Gaussian and box blur | `easy::gaussian_blur`, `easy::box_blur` |
| [`05_sensor_noise`](05_sensor_noise.cpp) | Seeded, reproducible camera noise | `ISONoiseConfig`, `easy::iso_noise` |
| [`06_weather_rain`](06_weather_rain.cpp) | Rain streaks generated from a seed and shown as explicit records | `RandomRainConfig`, `make_random_rain_streaks` |
| [`07_jpeg_compression`](07_jpeg_compression.cpp) | Real libjpeg round trip at two quality levels (built only when libjpeg is found) | `JpegCompressionConfig`, `easy::jpeg_compression` |
| [`08_mixing_cutmix`](08_mixing_cutmix.cpp) | The raw pointer API with explicit host/device memory | `ImageSourceU8`, `cutmix_u8` |
| [`09_boxes_keypoints`](09_boxes_keypoints.cpp) | Keeping boxes and keypoints aligned with a crop and flip; COCO/YOLO conversion | `Geometry`, `transform_boxes_filtered` |
| [`10_pipeline_cfg`](10_pipeline_cfg.cpp) | A full augmentation recipe loaded from INI text | `Pipeline::load_from_string`, `ExecutionContext` |

`example_utils.hpp` holds the shared test-image generator and PPM writer. It is
not part of the library.

## Older examples

| Example | What it shows |
|---|---|
| [`meta_example.cpp`](meta_example.cpp) | A user callback stage (`lambda`) that edits pixels and boxes, then clips annotations |
| [`native_api_cpp17.cpp`](native_api_cpp17.cpp) / [`native_api_cpp23.cpp`](native_api_cpp23.cpp) | The same sensor-noise call compiled as C++17 and as C++23 |
| [`noise_camera_workflows.cpp`](noise_camera_workflows.cpp) | Checking that seeded sensor noise is reproducible |
| [`pipeline_example_cpp17.cpp`](pipeline_example_cpp17.cpp) / [`pipeline_example_cpp23.cpp`](pipeline_example_cpp23.cpp) | Loading `augmentation.cfg` from disk and running it |
| [`support_contract.cpp`](support_contract.cpp) | Typed views, one seed, and explicit keypoint geometry |

## CPU vs CUDA builds

The `easy::` wrappers accept host memory in both builds. In a CUDA build they
copy host data to the GPU and back for you. The raw `*_u8` / `*_f32` functions
use host pointers in CPU builds and **device** pointers in CUDA builds.
`08_mixing_cutmix` shows how to handle that yourself.
