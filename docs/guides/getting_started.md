# Getting started

This page covers the five ideas you need before any augmatch call makes sense.
Each idea points to an example you can build and run.

## 1. You own the pixels; augmatch uses views

augmatch never allocates or frees your images. You keep the buffer, for example
a `std::vector<std::uint8_t>` or a `cv::Mat::data`, and describe it with a *view*:

```cpp
std::vector<std::uint8_t> pixels(width * height * 3);            // HWC: R,G,B,R,G,B,...
augmatch::MutableImageView img = augmatch::make_hwc_u8_view(pixels.data(), width, height, 3);
```

- `ImageView` is read-only (made from a `const` pointer). `MutableImageView` is writable.
- A view records size, strides, dtype (`UInt8`, `UInt16`, `Float32`), layout (`HWC`, `CHW`),
  and memory space (`Host`, `Device`).
- Use `make_hwc_view(ptr, w, h, c)` for other dtypes; the type is deduced from the pointer.
- Keep the buffer alive while any view of it is in use.

Example: [01_hello_view](../../examples/01_hello_view.cpp).

## 2. Errors come back as `Status`

View-based functions return `augmatch::Status` and do not throw:

```cpp
augmatch::Status s = augmatch::easy::crop(in, out, x, y);
if (!s) std::cerr << s.message << '\n';   // e.g. "crop rectangle lies outside the input image"
```

`s.code` is one of `InvalidArgument`, `InvalidView`, `InvalidMetadata`,
`Unsupported`, or `ExecutionError`. `Result<T>` (returned by
`Pipeline::load_from_*`) holds either a value or a `Status`.

Example: [02_geometry_flip_crop](../../examples/02_geometry_flip_crop.cpp).

## 3. Randomness is a seed you pass in

No transform uses a global RNG. Stochastic transforms take `seed` in their
config (or `ExecutionContext::seed` in a pipeline). The same seed and inputs
always produce the same output bytes. Different seeds give different results.

Example: [05_sensor_noise](../../examples/05_sensor_noise.cpp).

## 4. Two API levels: `easy` and raw

| | `augmatch::easy::op(input, output, ...)` | `augmatch::op_u8(in_ptr, out_ptr, config, stream)` |
|---|---|---|
| Header | `core/easy.hpp` | `<topic>/<name>.hpp` |
| Size fields in config | filled from the view | you set `width/height/channels` |
| Validation | dtype, layout, shape, bounds → `Status` | documented preconditions; may throw |
| Host memory in a CUDA build | staged to the GPU automatically (synchronous) | not allowed: pass device pointers |
| Coverage | common ops, plus `easy::apply(op_u8, in, out, config)` for any config-based op | every transform |

Start with `easy`. Switch to the raw API when you already hold device buffers and
want asynchronous launches on your own `cudaStream_t`.

Examples: [03_color_jitter](../../examples/03_color_jitter.cpp) (`easy::apply`),
[08_mixing_cutmix](../../examples/08_mixing_cutmix.cpp) (raw API with device memory).

## 5. CPU vs CUDA builds

The backend is chosen at build time (`AUGMATCH_ENABLE_CUDA`). The macro
`AUGMATCH_HAS_CUDA` is `1` in CUDA builds.

- **CPU build:** every function takes host pointers.
- **CUDA build:** raw `*_u8` / `*_f32` functions take **device** pointers and
  enqueue on the given stream without synchronizing. Annotation helpers
  (`transform_boxes`, box format conversions) and `Pipeline` always run on the host.

## Where to go next

- A whole recipe from a config file: [10_pipeline_cfg](../../examples/10_pipeline_cfg.cpp)
  and [PIPELINE.md](PIPELINE.md).
- Boxes and keypoints that follow the pixels: [09_boxes_keypoints](../../examples/09_boxes_keypoints.cpp).
- Exact per-transform contracts: [API reference](../README.md#api-reference).
