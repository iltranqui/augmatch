#include "augmatch/noise_view.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace augmatch {
namespace {
constexpr std::uint64_t kA = 0x9e3779b97f4a7c15ULL;
constexpr std::uint64_t kB = 0xbf58476d1ce4e5b9ULL;
constexpr std::uint64_t kC = 0x94d049bb133111ebULL;

std::uint64_t mix64(std::uint64_t x) noexcept {
  x += kA;
  x = (x ^ (x >> 30)) * kB;
  x = (x ^ (x >> 27)) * kC;
  return x ^ (x >> 31);
}

Status bad(StatusCode code, const char* message) noexcept { return {code, message}; }

bool valid_strides(const ImageView& v) noexcept {
  const auto element = static_cast<std::ptrdiff_t>(bytes_per_element(v.type));
  if (element <= 0 || v.stride_x <= 0 || v.stride_y <= 0 || v.stride_c <= 0) return false;
  if (v.layout == Layout::HWC)
    return v.stride_c >= element && v.stride_x >= v.channels * element &&
           v.stride_y >= v.width * v.stride_x;
  return v.stride_x >= element && v.stride_y >= v.width * element &&
         v.stride_c >= v.height * v.stride_y;
}

std::size_t offset(const ImageView& v, int x, int y, int channel) noexcept {
  return static_cast<std::size_t>(y * v.stride_y + x * v.stride_x + channel * v.stride_c);
}

float read_sample(const ImageView& v, int x, int y, int channel) noexcept {
  const auto* bytes = static_cast<const std::uint8_t*>(v.data) + offset(v, x, y, channel);
  switch (v.type) {
    case DataType::UInt8: return static_cast<float>(*bytes) / 255.0f;
    case DataType::UInt16: {
      std::uint16_t value;
      __builtin_memcpy(&value, bytes, sizeof(value));
      return static_cast<float>(value) / 65535.0f;
    }
    case DataType::Float32: {
      float value;
      __builtin_memcpy(&value, bytes, sizeof(value));
      return value;
    }
  }
  return 0.0f;
}

float quantize(float value, NoiseQuantizationPolicy policy) noexcept {
  switch (policy) {
    case NoiseQuantizationPolicy::Preserve: return value;
    case NoiseQuantizationPolicy::RoundHalfUp: return std::floor(value + 0.5f);
    case NoiseQuantizationPolicy::Floor: return std::floor(value);
    case NoiseQuantizationPolicy::Ceil: return std::ceil(value);
    case NoiseQuantizationPolicy::Truncate: return std::trunc(value);
  }
  return value;
}

void write_sample(const MutableImageView& v, int x, int y, int channel,
                  float value, NoiseQuantizationPolicy policy) noexcept {
  auto* bytes = static_cast<std::uint8_t*>(v.data) + offset(v.as_const(), x, y, channel);
  if (v.type == DataType::Float32) {
    __builtin_memcpy(bytes, &value, sizeof(value));
    return;
  }
  const float maximum = v.type == DataType::UInt8 ? 255.0f : 65535.0f;
  value = quantize(value * maximum, policy == NoiseQuantizationPolicy::Preserve
                                      ? NoiseQuantizationPolicy::RoundHalfUp : policy);
  value = std::max(0.0f, std::min(maximum, value));
  if (v.type == DataType::UInt8) {
    const auto result = static_cast<std::uint8_t>(value);
    *bytes = result;
  } else {
    const auto result = static_cast<std::uint16_t>(value);
    __builtin_memcpy(bytes, &result, sizeof(result));
  }
}

bool valid_enum(NoiseBorderPolicy policy) noexcept {
  return policy == NoiseBorderPolicy::Clamp || policy == NoiseBorderPolicy::Reflect101 ||
         policy == NoiseBorderPolicy::Constant;
}
bool valid_enum(NoiseClipPolicy policy) noexcept {
  return policy == NoiseClipPolicy::None || policy == NoiseClipPolicy::Clamp;
}
bool valid_enum(NoiseQuantizationPolicy policy) noexcept {
  return policy == NoiseQuantizationPolicy::Preserve ||
         policy == NoiseQuantizationPolicy::RoundHalfUp || policy == NoiseQuantizationPolicy::Floor ||
         policy == NoiseQuantizationPolicy::Ceil || policy == NoiseQuantizationPolicy::Truncate;
}
}  // namespace

std::uint64_t CounterRng::counter(std::uint64_t x, std::uint64_t y,
                                  std::uint64_t channel, std::uint64_t plane) const noexcept {
  return mix64(seed ^ (x * 0x632be59bd9b4e019ULL) ^ (y * 0x8cb92baa2f2f6f7dULL) ^
               (channel * kA) ^ (plane * 0xd1b54a32d192ed03ULL) ^ stream);
}

float CounterRng::uniform(std::uint64_t x, std::uint64_t y,
                          std::uint64_t channel, std::uint64_t plane) const noexcept {
  return static_cast<float>((counter(x, y, channel, plane) >> 11) *
                            (1.0 / 9007199254740992.0));
}

float CounterRng::normal(std::uint64_t x, std::uint64_t y,
                         std::uint64_t channel, std::uint64_t plane) const noexcept {
  const float u = std::max(uniform(x, y, channel, plane), std::numeric_limits<float>::min());
  const float v = uniform(x, y, channel, plane ^ 0xd1b54a32d192ed03ULL);
  return std::sqrt(-2.0f * std::log(u)) * std::cos(6.2831853071795864769f * v);
}

int resolve_noise_coordinate(int coordinate, int extent, NoiseBorderPolicy policy,
                             bool* is_constant) noexcept {
  if (is_constant) *is_constant = false;
  if (extent <= 0) {
    if (is_constant) *is_constant = true;
    return -1;
  }
  if (coordinate >= 0 && coordinate < extent) return coordinate;
  if (policy == NoiseBorderPolicy::Constant) {
    if (is_constant) *is_constant = true;
    return -1;
  }
  if (policy == NoiseBorderPolicy::Clamp) return coordinate < 0 ? 0 : extent - 1;
  if (extent == 1) return 0;
  const int period = 2 * extent - 2;
  int value = coordinate % period;
  if (value < 0) value += period;
  return value < extent ? value : period - value;
}

Status validate_noise_view(const ImageView& input, const MutableImageView& output,
                           const NoiseViewConfig& config,
                           const NoiseMetadataView& metadata) noexcept {
  if (!input.valid() || !output.valid() || !valid_strides(input) || !valid_strides(output.as_const()))
    return bad(StatusCode::InvalidView, "invalid image view or stride");
  if (input.width != output.width || input.height != output.height ||
      input.channels != output.channels || input.layout != output.layout)
    return bad(StatusCode::InvalidView, "input and output view shapes/layouts differ");
  if (!std::isfinite(config.mean) || !std::isfinite(config.stddev) || config.stddev < 0.0f ||
      !std::isfinite(config.clip_min) || !std::isfinite(config.clip_max) ||
      config.clip_max < config.clip_min || !std::isfinite(config.border_value))
    return bad(StatusCode::InvalidArgument, "non-finite or invalid noise parameters");
  if (!valid_enum(config.border) || !valid_enum(config.clip) || !valid_enum(config.quantization))
    return bad(StatusCode::InvalidArgument, "unknown noise policy");
  if ((metadata.channel_count && !metadata.channels) ||
      (metadata.plane_count && !metadata.planes) ||
      (metadata.channel_planes && metadata.channel_count != 0 &&
       metadata.channel_count != static_cast<std::size_t>(input.channels)))
    return bad(StatusCode::InvalidMetadata, "metadata pointer/count mismatch");
  if (metadata.channel_count && metadata.channel_count != static_cast<std::size_t>(input.channels))
    return bad(StatusCode::InvalidMetadata, "one channel metadata record is required per channel");
  if (input.memory != MemorySpace::Host || output.memory != MemorySpace::Host) {
    if (metadata.metadata_memory != MemorySpace::Device &&
        (metadata.channels || metadata.planes || metadata.channel_planes))
      return bad(StatusCode::InvalidMetadata, "device views require device metadata");
  } else if (metadata.metadata_memory != MemorySpace::Host) {
    return bad(StatusCode::InvalidMetadata, "host views require host metadata");
  }
  return Status::success();
}

Status additive_noise_view(const ImageView& input, const MutableImageView& output,
                           const NoiseViewConfig& config, const NoiseMetadataView& metadata,
                           cudaStream_t) noexcept {
  const Status checked = validate_noise_view(input, output, config, metadata);
  if (!checked) return checked;
  if (input.memory != MemorySpace::Host || output.memory != MemorySpace::Host)
    return bad(StatusCode::Unsupported, "device image views require the CUDA backend");
  CounterRng rng{config.seed, 0x4e4f4953455f5649ULL};
  for (int y = 0; y < input.height; ++y) {
    for (int x = 0; x < input.width; ++x) {
      for (int channel = 0; channel < input.channels; ++channel) {
        const auto channel_index = static_cast<std::size_t>(channel);
        NoiseChannelMetadata channel_meta{};
        if (metadata.channels) channel_meta = metadata.channels[channel_index];
        int plane = channel_meta.plane >= 0 ? channel_meta.plane : channel;
        if (metadata.channel_planes) plane = metadata.channel_planes[channel_index];
        NoisePlaneMetadata plane_meta{};
        if (metadata.planes) {
          if (plane < 0 || plane >= static_cast<int>(metadata.plane_count))
            return bad(StatusCode::InvalidMetadata, "channel maps outside plane metadata");
          plane_meta = metadata.planes[static_cast<std::size_t>(plane)];
        }
        if (!std::isfinite(channel_meta.gain) || !std::isfinite(channel_meta.bias) ||
            !std::isfinite(channel_meta.stddev_scale) || !std::isfinite(plane_meta.gain) ||
            !std::isfinite(plane_meta.bias) || !std::isfinite(plane_meta.stddev_scale) ||
            channel_meta.stddev_scale < 0.0f || plane_meta.stddev_scale < 0.0f)
          return bad(StatusCode::InvalidMetadata, "non-finite channel or plane metadata");
        float value = read_sample(input, x, y, channel);
        if (!std::isfinite(value)) return bad(StatusCode::InvalidArgument, "input contains non-finite data");
        value = (value + config.mean + config.stddev * channel_meta.stddev_scale *
                 plane_meta.stddev_scale * rng.normal(static_cast<std::uint64_t>(x),
                                                      static_cast<std::uint64_t>(y),
                                                      static_cast<std::uint64_t>(channel),
                                                      static_cast<std::uint64_t>(std::max(0, plane)))) *
                channel_meta.gain * plane_meta.gain + channel_meta.bias + plane_meta.bias;
        if (config.clip == NoiseClipPolicy::Clamp)
          value = std::max(config.clip_min, std::min(config.clip_max, value));
        write_sample(output, x, y, channel, value, config.quantization);
      }
    }
  }
  return Status::success();
}

}  // namespace augmatch
