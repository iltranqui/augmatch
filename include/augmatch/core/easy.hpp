#pragma once
// augmatch::easy — view-based convenience wrappers over the raw pointer API.
//
// The raw API (e.g. `blur_u8(const uint8_t* input, uint8_t* output, const BlurConfig&)`)
// is fast and explicit, but every call needs width/height/channels copied into
// the config, and in CUDA builds the pointers must be device memory. The
// wrappers in this header remove that boilerplate:
//
//   std::vector<std::uint8_t> pixels(w * h * 3), result(pixels.size());
//   auto in  = augmatch::make_hwc_u8_view(pixels.data(), w, h, 3);
//   auto out = augmatch::make_hwc_u8_view(result.data(), w, h, 3);
//   augmatch::Status s = augmatch::easy::gaussian_blur(in, out, /*kernel_size=*/5, /*sigma=*/1.2f);
//   if (!s) std::cerr << s.message << '\n';
//
// Contract for every wrapper:
//   * Views must be valid, contiguous, HWC, uint8, and in the same memory space.
//   * width/height/channels in the config are overwritten from the input view.
//   * Host views work in both CPU and CUDA builds. In CUDA builds a host view is
//     staged through temporary device buffers and the call is synchronous; this
//     is convenient for prototyping, not for throughput.
//   * Device views run directly on `stream` without synchronizing (CUDA builds
//     only; CPU builds return StatusCode::Unsupported for device views).
//   * Errors never throw: the wrapper returns a Status whose `message` points to
//     a static string or a thread-local buffer valid until the next easy call.

#include <cstddef>
#include <cstdint>
#include <exception>
#include <string>

#include "augmatch/core/image.hpp"
#include "augmatch/core/status.hpp"
#include "augmatch/annotations/transforms.hpp"
#include "augmatch/color/color.hpp"
#include "augmatch/compression/jpeg.hpp"
#include "augmatch/filter/filter.hpp"
#include "augmatch/sensor/noise.hpp"
#include "augmatch/weather/weather.hpp"

#if AUGMATCH_HAS_CUDA
#include <cuda_runtime_api.h>
#endif

namespace augmatch {
namespace easy {

// Read-only input parameter accepted by every wrapper. It converts implicitly
// from both ImageView and MutableImageView, so the output of one step can be
// passed straight in as the input of the next without calling as_const().
struct Input {
  ImageView view;
  Input(const ImageView& image) : view(image) {}                // NOLINT: implicit on purpose
  Input(const MutableImageView& image) : view(image.as_const()) {}  // NOLINT: implicit on purpose
};

namespace detail {

// Storage for exception text so Status::message (a const char*) stays valid.
inline Status error_from_exception(const std::exception& error) {
  thread_local std::string message;
  message = error.what();
  return {StatusCode::InvalidArgument, message.c_str()};
}

inline std::size_t byte_count(const ImageView& view) {
  return static_cast<std::size_t>(view.width) * view.height * view.channels;
}

inline Status check_u8_hwc(const ImageView& view, const char* invalid_message) {
  if (!view.valid()) return {StatusCode::InvalidView, invalid_message};
  if (view.type != DataType::UInt8) return {StatusCode::Unsupported, "easy API expects uint8 images"};
  if (view.layout != Layout::HWC) return {StatusCode::Unsupported, "easy API expects HWC layout"};
  if (!view.contiguous()) return {StatusCode::Unsupported, "easy API expects contiguous images"};
  return Status::success();
}

// Validate an input/output pair. `same_shape` is false for ops such as crop
// whose output size differs from the input size.
inline Status check_pair(const ImageView& input, const MutableImageView& output, bool same_shape) {
  Status status = check_u8_hwc(input, "input view is invalid (null data or zero size)");
  if (!status) return status;
  status = check_u8_hwc(output.as_const(), "output view is invalid (null data or zero size)");
  if (!status) return status;
  if (input.memory != output.memory)
    return {StatusCode::InvalidArgument, "input and output must live in the same memory space"};
  if (input.channels != output.channels)
    return {StatusCode::InvalidArgument, "input and output channel counts differ"};
  if (same_shape && (input.width != output.width || input.height != output.height))
    return {StatusCode::InvalidArgument, "input and output sizes differ"};
  return Status::success();
}

#if AUGMATCH_HAS_CUDA
// RAII device buffer used only to stage host views in CUDA builds.
class DeviceBuffer {
 public:
  explicit DeviceBuffer(std::size_t bytes) { ok_ = cudaMalloc(&data_, bytes) == cudaSuccess; }
  ~DeviceBuffer() { if (data_) cudaFree(data_); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  bool ok() const noexcept { return ok_; }
  std::uint8_t* data() const noexcept { return static_cast<std::uint8_t*>(data_); }

 private:
  void* data_ = nullptr;
  bool ok_ = false;
};
#endif

// Run `launch(const uint8_t* in, uint8_t* out, cudaStream_t)` on the backend
// that matches this build and the views' memory space.
template <class Launch>
Status run(const ImageView& input, const MutableImageView& output, cudaStream_t stream, Launch launch) {
  const auto* in = static_cast<const std::uint8_t*>(input.data);
  auto* out = static_cast<std::uint8_t*>(output.data);
  try {
#if AUGMATCH_HAS_CUDA
    if (input.memory == MemorySpace::Device) {
      launch(in, out, stream);  // asynchronous on the caller's stream
      return Status::success();
    }
    // Host views in a CUDA build: copy in, run on the device, copy back.
    const std::size_t in_bytes = byte_count(input);
    const std::size_t out_bytes = byte_count(output.as_const());
    DeviceBuffer device_in(in_bytes), device_out(out_bytes);
    if (!device_in.ok() || !device_out.ok())
      return {StatusCode::ExecutionError, "cudaMalloc failed while staging a host view"};
    if (cudaMemcpy(device_in.data(), in, in_bytes, cudaMemcpyHostToDevice) != cudaSuccess)
      return {StatusCode::ExecutionError, "host-to-device copy failed"};
    launch(device_in.data(), device_out.data(), stream);
    if (cudaStreamSynchronize(stream) != cudaSuccess)
      return {StatusCode::ExecutionError, "CUDA kernel failed"};
    if (cudaMemcpy(out, device_out.data(), out_bytes, cudaMemcpyDeviceToHost) != cudaSuccess)
      return {StatusCode::ExecutionError, "device-to-host copy failed"};
    return Status::success();
#else
    if (input.memory == MemorySpace::Device)
      return {StatusCode::Unsupported, "device views need a CUDA build (AUGMATCH_ENABLE_CUDA=ON)"};
    launch(in, out, stream);
    return Status::success();
#endif
  } catch (const std::exception& error) {
    return error_from_exception(error);
  }
}

}  // namespace detail

// Generic adapter: run any raw `op(input, output, config, stream)` whose config
// starts with `int width, height, channels` on a same-shape view pair.
//
//   augmatch::RGBShiftConfig shift;  shift.shift0 = 20;  // +20 on the red channel
//   augmatch::easy::apply(augmatch::rgb_shift_u8, in, out, shift);
template <class Config, class Op>
Status apply(Op op, Input source, const MutableImageView& output, Config config,
             cudaStream_t stream = nullptr) {
  const ImageView& input = source.view;
  const Status status = detail::check_pair(input, output, /*same_shape=*/true);
  if (!status) return status;
  config.width = input.width;
  config.height = input.height;
  config.channels = input.channels;
  return detail::run(input, output, stream,
                     [&](const std::uint8_t* in, std::uint8_t* out, cudaStream_t s) {
                       op(in, out, config, s);
                     });
}

// ---- Geometry --------------------------------------------------------------

// Mirror the image left-to-right.
inline Status flip_horizontal(Input source, const MutableImageView& output,
                              cudaStream_t stream = nullptr) {
  const ImageView& input = source.view;
  const Status status = detail::check_pair(input, output, true);
  if (!status) return status;
  return detail::run(input, output, stream, [&](const std::uint8_t* in, std::uint8_t* out, cudaStream_t s) {
    horizontal_flip_u8(in, out, input.width, input.height, input.channels, s);
  });
}

// Mirror the image top-to-bottom.
inline Status flip_vertical(Input source, const MutableImageView& output,
                            cudaStream_t stream = nullptr) {
  const ImageView& input = source.view;
  const Status status = detail::check_pair(input, output, true);
  if (!status) return status;
  return detail::run(input, output, stream, [&](const std::uint8_t* in, std::uint8_t* out, cudaStream_t s) {
    vertical_flip_u8(in, out, input.width, input.height, input.channels, s);
  });
}

// Copy the rectangle whose top-left corner is (x, y) and whose size is the
// output view's width x height. The rectangle must lie inside the input.
inline Status crop(Input source, const MutableImageView& output, int x, int y,
                   cudaStream_t stream = nullptr) {
  const ImageView& input = source.view;
  const Status status = detail::check_pair(input, output, /*same_shape=*/false);
  if (!status) return status;
  if (x < 0 || y < 0 || x + output.width > input.width || y + output.height > input.height)
    return {StatusCode::InvalidArgument, "crop rectangle lies outside the input image"};
  const Geometry geometry{input.width, input.height, input.channels, x, y,
                          output.width, output.height, false, false};
  return detail::run(input, output, stream, [&](const std::uint8_t* in, std::uint8_t* out, cudaStream_t s) {
    transform_u8(in, out, geometry, s);
  });
}

// ---- Filters ---------------------------------------------------------------

// Gaussian blur with an odd square kernel (3, 5, 7, ...) and standard deviation sigma.
inline Status gaussian_blur(Input source, const MutableImageView& output,
                            int kernel_size = 3, float sigma = 1.0f, cudaStream_t stream = nullptr) {
  const ImageView& input = source.view;
  BlurConfig config;
  config.kernel_size = kernel_size;
  config.sigma = sigma;
  config.mode = BlurMode::Gaussian;
  return apply(blur_u8, input, output, config, stream);
}

// Box (average) blur with an odd square kernel.
inline Status box_blur(Input source, const MutableImageView& output,
                       int kernel_size = 3, cudaStream_t stream = nullptr) {
  const ImageView& input = source.view;
  BlurConfig config;
  config.kernel_size = kernel_size;
  config.mode = BlurMode::Average;
  return apply(blur_u8, input, output, config, stream);
}

// ---- Color -----------------------------------------------------------------

// Brightness (additive, fraction of 255), contrast and saturation (factors),
// and hue (additive, OpenCV half-degrees). Needs >= 3 channels.
inline Status color_jitter(Input source, const MutableImageView& output,
                           const ColorJitterConfig& config, cudaStream_t stream = nullptr) {
  const ImageView& input = source.view;
  return apply(color_jitter_u8, input, output, config, stream);
}

// ---- Sensor noise ----------------------------------------------------------

// Camera-like luma + chroma noise scaled by ISO and gain. Deterministic for a given seed.
inline Status iso_noise(Input source, const MutableImageView& output,
                        const ISONoiseConfig& config, cudaStream_t stream = nullptr) {
  const ImageView& input = source.view;
  return apply(iso_noise_u8, input, output, config, stream);
}

// ---- Weather ---------------------------------------------------------------

// White rain streaks. With config.streaks == nullptr, placement comes from config.seed.
inline Status rain(Input source, const MutableImageView& output,
                   const RandomRainConfig& config, cudaStream_t stream = nullptr) {
  const ImageView& input = source.view;
  return apply(random_rain_u8, input, output, config, stream);
}

// ---- Compression -----------------------------------------------------------

// Encode and decode through libjpeg at config.quality (1..100). 1 or 3 channels.
// Requires a build with JPEG support (AUGMATCH_ENABLE_JPEG and libjpeg found).
inline Status jpeg_compression(Input source, const MutableImageView& output,
                               const JpegCompressionConfig& config, cudaStream_t stream = nullptr) {
  const ImageView& input = source.view;
  return apply(jpeg_compression_u8, input, output, config, stream);
}

}  // namespace easy
}  // namespace augmatch
