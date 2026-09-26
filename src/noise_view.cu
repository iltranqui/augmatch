#include "augmatch/sensor/noise_view.hpp"

#include <cuda_runtime.h>
#include <cmath>
#include <string>

namespace augmatch {
namespace {
__device__ unsigned long long mix64(unsigned long long x) {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}
__device__ unsigned long long counter(const NoiseViewConfig& c, unsigned long long x,
                                     unsigned long long y, unsigned long long channel,
                                     unsigned long long plane) {
  return mix64(c.seed ^ (x * 0x632be59bd9b4e019ULL) ^ (y * 0x8cb92baa2f2f6f7dULL) ^
               (channel * 0x9e3779b97f4a7c15ULL) ^ (plane * 0xd1b54a32d192ed03ULL) ^
               0x4e4f4953455f5649ULL);
}
__device__ float uniform(const NoiseViewConfig& c, unsigned long long x,
                         unsigned long long y, unsigned long long channel,
                         unsigned long long plane) {
  return (float)((counter(c, x, y, channel, plane) >> 11) * (1.0 / 9007199254740992.0));
}
__device__ float normal(const NoiseViewConfig& c, unsigned long long x,
                        unsigned long long y, unsigned long long channel,
                        unsigned long long plane) {
  float u = fmaxf(uniform(c, x, y, channel, plane), 1.17549435e-38f);
  float v = uniform(c, x, y, channel, plane ^ 0xd1b54a32d192ed03ULL);
  return sqrtf(-2.0f * logf(u)) * cosf(6.2831853071795864769f * v);
}
__device__ unsigned char* at(unsigned char* base, const MutableImageView& v,
                             int x, int y, int channel) {
  return base + y * v.stride_y + x * v.stride_x + channel * v.stride_c;
}
__device__ const unsigned char* at(const unsigned char* base, const ImageView& v,
                                   int x, int y, int channel) {
  return base + y * v.stride_y + x * v.stride_x + channel * v.stride_c;
}
__device__ float read_sample(const ImageView& v, int x, int y, int channel) {
  const auto* p = at(static_cast<const unsigned char*>(v.data), v, x, y, channel);
  if (v.type == DataType::UInt8) return (float)*p / 255.0f;
  if (v.type == DataType::UInt16) return (float)*reinterpret_cast<const unsigned short*>(p) / 65535.0f;
  return *reinterpret_cast<const float*>(p);
}
__device__ float quantize(float value, NoiseQuantizationPolicy policy) {
  if (policy == NoiseQuantizationPolicy::RoundHalfUp) return floorf(value + 0.5f);
  if (policy == NoiseQuantizationPolicy::Floor) return floorf(value);
  if (policy == NoiseQuantizationPolicy::Ceil) return ceilf(value);
  if (policy == NoiseQuantizationPolicy::Truncate) return truncf(value);
  return value;
}
__device__ void write_sample(const MutableImageView& v, int x, int y, int channel,
                             float value, NoiseQuantizationPolicy policy) {
  auto* p = at(static_cast<unsigned char*>(v.data), v, x, y, channel);
  if (v.type == DataType::Float32) { *reinterpret_cast<float*>(p) = value; return; }
  const float maximum = v.type == DataType::UInt8 ? 255.0f : 65535.0f;
  value = quantize(value * maximum, policy == NoiseQuantizationPolicy::Preserve
                                      ? NoiseQuantizationPolicy::RoundHalfUp : policy);
  value = fminf(maximum, fmaxf(0.0f, value));
  if (v.type == DataType::UInt8) *p = (unsigned char)value;
  else *reinterpret_cast<unsigned short*>(p) = (unsigned short)value;
}
__global__ void additive_kernel(ImageView input, MutableImageView output,
                                NoiseViewConfig config, NoiseMetadataView metadata) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int count = input.width * input.height * input.channels;
  if (index >= count) return;
  const int channel = index % input.channels;
  const int pixel = index / input.channels;
  const int x = pixel % input.width;
  const int y = pixel / input.width;
  NoiseChannelMetadata channel_meta{};
  if (metadata.channels) channel_meta = metadata.channels[channel];
  int plane = channel_meta.plane >= 0 ? channel_meta.plane : channel;
  if (metadata.channel_planes) plane = metadata.channel_planes[channel];
  NoisePlaneMetadata plane_meta{};
  if (metadata.planes && plane >= 0 && plane < (int)metadata.plane_count)
    plane_meta = metadata.planes[plane];
  float value = read_sample(input, x, y, channel);
  value = (value + config.mean + config.stddev * channel_meta.stddev_scale *
           plane_meta.stddev_scale * normal(config, x, y, channel, max(0, plane))) *
          channel_meta.gain * plane_meta.gain + channel_meta.bias + plane_meta.bias;
  if (config.clip == NoiseClipPolicy::Clamp)
    value = fminf(config.clip_max, fmaxf(config.clip_min, value));
  write_sample(output, x, y, channel, value, config.quantization);
}
Status validate_cuda(const ImageView& input, const MutableImageView& output,
                     const NoiseViewConfig& config, const NoiseMetadataView& metadata) noexcept {
  if (!input.valid() || !output.valid() || input.width != output.width || input.height != output.height ||
      input.channels != output.channels || input.layout != output.layout)
    return {StatusCode::InvalidView, "invalid or mismatched image view"};
  if (input.memory != MemorySpace::Device || output.memory != MemorySpace::Device)
    return {StatusCode::Unsupported, "CUDA view operation requires device image memory"};
  if (!std::isfinite(config.mean) || !std::isfinite(config.stddev) || config.stddev < 0.0f ||
      !std::isfinite(config.clip_min) || !std::isfinite(config.clip_max) || config.clip_max < config.clip_min)
    return {StatusCode::InvalidArgument, "invalid noise parameters"};
  if ((metadata.channels || metadata.planes || metadata.channel_planes) &&
      metadata.metadata_memory != MemorySpace::Device)
    return {StatusCode::InvalidMetadata, "device views require device metadata"};
  if (metadata.channel_count && metadata.channel_count != (std::size_t)input.channels)
    return {StatusCode::InvalidMetadata, "one channel metadata record is required per channel"};
  return Status::success();
}
}  // namespace

std::uint64_t CounterRng::counter(std::uint64_t x, std::uint64_t y,
                                  std::uint64_t channel, std::uint64_t plane) const noexcept {
  x = seed ^ (x * 0x632be59bd9b4e019ULL) ^ (y * 0x8cb92baa2f2f6f7dULL) ^
       (channel * 0x9e3779b97f4a7c15ULL) ^ (plane * 0xd1b54a32d192ed03ULL) ^ stream;
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}
float CounterRng::uniform(std::uint64_t x, std::uint64_t y, std::uint64_t channel,
                          std::uint64_t plane) const noexcept {
  return static_cast<float>((counter(x, y, channel, plane) >> 11) * (1.0 / 9007199254740992.0));
}
float CounterRng::normal(std::uint64_t x, std::uint64_t y, std::uint64_t channel,
                         std::uint64_t plane) const noexcept {
  const float u = fmaxf(uniform(x, y, channel, plane), 1.17549435e-38f);
  const float v = uniform(x, y, channel, plane ^ 0xd1b54a32d192ed03ULL);
  return sqrtf(-2.0f * logf(u)) * cosf(6.2831853071795864769f * v);
}

int resolve_noise_coordinate(int coordinate, int extent, NoiseBorderPolicy policy,
                             bool* is_constant) noexcept {
  if (is_constant) *is_constant = false;
  if (extent <= 0 || policy == NoiseBorderPolicy::Constant && (coordinate < 0 || coordinate >= extent)) {
    if (is_constant) *is_constant = true;
    return -1;
  }
  if (coordinate >= 0 && coordinate < extent) return coordinate;
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
  return validate_cuda(input, output, config, metadata);
}

Status additive_noise_view(const ImageView& input, const MutableImageView& output,
                           const NoiseViewConfig& config, const NoiseMetadataView& metadata,
                           cudaStream_t stream) noexcept {
  const Status checked = validate_cuda(input, output, config, metadata);
  if (!checked) return checked;
  const int count = input.width * input.height * input.channels;
  additive_kernel<<<(count + 255) / 256, 256, 0, stream>>>(input, output, config, metadata);
  const cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) return {StatusCode::ExecutionError, cudaGetErrorString(error)};
  return Status::success();
}

}  // namespace augmatch
