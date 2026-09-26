# Config-driven CPU pipeline

`augmatch::Pipeline` loads an ordered list of CPU augmentation stages once and
runs that compiled list in place on a contiguous host HWC `uint8_t` image. The
public pipeline headers are compatible with both C++17 and C++23.

## `augmentation.cfg`

The format is a small INI-style grammar. A `[StageName]` starts a stage.
Numeric `key = value` lines belong to the preceding stage. Blank lines are
ignored, and `#` begins a comment. Sections execute from top to bottom. Names
are case-sensitive. Repeated sections are allowed; duplicate keys within one
section are rejected.

Every stage accepts the reserved `probability` key. Its default is `1.0`; the
accepted range is `[0,1]`. Random decisions use `ExecutionContext`, including
its deterministic seed and record/replay data.

```ini
[HueShift]
amount = 0.02

[Saturation]
multiplier = 1.5

[GaussianBlur]
kernel_size = 17
sigma = 0.0
probability = 0.3

[HorizontalFlip]
probability = 0.5
```

`sigma = 0` selects AUGMATCH's default finite-kernel sigma of `1.0`.

## Built-in stages

| Stage name | Parameters | Existing AUGMATCH primitive |
|---|---|---|
| `Exposure` | `multiplier` (finite, nonnegative) | `multiply_brightness_u8` |
| `Saturation` | `multiplier` (finite, nonnegative) | `multiply_saturation_u8` |
| `HueShift` | `amount` | `add_to_hue_u8` |
| `Contrast` | `contrast`; optional `brightness` | `color_jitter_u8` |
| `Brightness` | `brightness`; optional `contrast` | `color_jitter_u8` |
| `GaussianBlur` | odd `kernel_size` in `[1,17]`, nonnegative `sigma` | `blur_u8`, Gaussian mode |
| `HorizontalFlip` | none | `horizontal_flip_u8` |

All parameters are validated while the pipeline loads. Unknown stages,
unknown parameters, malformed numbers, and invalid probabilities return a
failed `Result<Pipeline>`. `run()` returns `Status` and rejects non-host,
non-HWC, non-contiguous, or non-`uint8_t` views.

Applications can add a stage with `register_stage(name, factory)`. A factory
returns `RegisteredCpuStage`, which contains the existing `CpuStage` plus an
optional shared context owner. Registration is process-wide, thread-safe, and
refuses to replace an existing case-sensitive name.

## C++ call site

```cpp
auto loaded = augmatch::Pipeline::load_from_file("augmentation.cfg");
if (!loaded.ok()) return 1;

auto view = augmatch::make_hwc_u8_view(data, width, height, channels);
augmatch::ExecutionContext execution;
execution.seed = image_seed;
const augmatch::Status status = loaded.value().run(view, execution);
if (!status.ok()) return 2;
```

## Vendoring as a git submodule

```cmake
set(AUGMATCH_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(AUGMATCH_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(AUGMATCH_ENABLE_CUDA ${YOUR_PROJECT_ENABLE_CUDA} CACHE BOOL "" FORCE)
set(AUGMATCH_ENABLE_WEBP OFF CACHE BOOL "" FORCE)
set(AUGMATCH_ENABLE_FFMPEG OFF CACHE BOOL "" FORCE)
set(AUGMATCH_ENABLE_JPEG OFF CACHE BOOL "" FORCE)

add_subdirectory(external/augmatch)
target_link_libraries(your_target PRIVATE augmatch::augmatch)
```

With tests and examples disabled, AUGMATCH does not discover Python and does
not add its CLI, examples, or tests to the parent build.

## Darknet integration (not yet applied)

The follow-up darknet change belongs in
`src-lib/image_opencv.cpp:image_data_augmentation()` (near line 297). That
function currently receives the `[net]` augmentation values parsed from
`hue`, `saturation`, `exposure`, `blur`, `gaussian_noise`, and `mixup`. It can
wrap the OpenCV buffer with `make_hwc_u8_view()` and replace the built-in
color/blur/flip sequence with one `Pipeline::run()` call.

No darknet file is changed by this implementation. The pipeline is CPU-only in
this pass; `DevicePipeline`/`DeviceStage` wiring is future work.
