#include "augmatch/weather/weather.hpp"

#include <cuda_runtime.h>
#include <cmath>
#include <stdexcept>
#include <string>

namespace augmatch {
namespace {

__device__ std::uint64_t catalog_splitmix64(std::uint64_t value) {
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}
__device__ float catalog_random(std::uint64_t seed, std::uint64_t index) {
  return static_cast<float>(catalog_splitmix64(seed + index) >> 40) / 16777216.0f;
}
__device__ float clamp_unit(float value) { return fminf(1.0f, fmaxf(0.0f, value)); }
__device__ unsigned char rounded(float value) {
  return static_cast<unsigned char>(fminf(255.0f, fmaxf(0.0f, floorf(value + 0.5f))));
}
__device__ float cloud_value(const CloudsConfig& config, int pixel) {
  if (config.field) return config.field[pixel];
  const std::uint64_t base = static_cast<std::uint64_t>(pixel) * 4ULL;
  return 0.25f * (catalog_random(config.seed, base) + catalog_random(config.seed, base + 1) +
                  catalog_random(config.seed, base + 2) + catalog_random(config.seed, base + 3));
}

__global__ void fast_snow_kernel(const std::uint8_t* input, std::uint8_t* output,
                                 FastSnowyLandscapeConfig config) {
  const int pixel = blockIdx.x * blockDim.x + threadIdx.x;
  const int pixels = config.width * config.height;
  if (pixel >= pixels) return;
  const int measured = config.channels < 3 ? config.channels : 3;
  float brightness = 0.0f;
  for (int c = 0; c < measured; ++c) brightness += input[pixel * config.channels + c] / 255.0f;
  brightness /= static_cast<float>(measured);
  const float span = fmaxf(1.0e-6f, 1.0f - config.snow_point);
  const float blend = config.alpha * clamp_unit((brightness - config.snow_point) / span);
  for (int c = 0; c < config.channels; ++c) {
    const float value = (1.0f - blend) * input[pixel * config.channels + c] + blend * 255.0f;
    output[pixel * config.channels + c] = rounded(value);
  }
}

__global__ void clouds_kernel(const std::uint8_t* input, std::uint8_t* output,
                              CloudsConfig config) {
  const int pixel = blockIdx.x * blockDim.x + threadIdx.x;
  const int pixels = config.width * config.height;
  if (pixel >= pixels) return;
  const float blend = clamp_unit(config.alpha * config.density * cloud_value(config, pixel));
  for (int c = 0; c < config.channels; ++c) {
    const int offset = pixel * config.channels + c;
    output[offset] = rounded((1.0f - blend) * input[offset] + blend * 255.0f);
  }
}

__global__ void layer_kernel(const std::uint8_t* input, std::uint8_t* output,
                             WeatherLayerConfig config) {
  const int pixel = blockIdx.x * blockDim.x + threadIdx.x;
  const int pixels = config.width * config.height;
  if (pixel >= pixels) return;
  const float blend = clamp_unit(config.opacity * config.layer[pixel]);
  for (int c = 0; c < config.channels; ++c) {
    const int offset = pixel * config.channels + c;
    output[offset] = rounded((1.0f - blend) * input[offset] + blend * 255.0f);
  }
}

void check(cudaError_t error, const char* operation) {
  if (error != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
}
void validate_image(int width, int height, int channels, const char* name) {
  if (width <= 0 || height <= 0 || channels <= 0)
    throw std::invalid_argument(std::string(name) + " dimensions and channels must be positive");
}
void validate_unit(float value, const char* name) {
  if (!std::isfinite(value) || value < 0.0f || value > 1.0f)
    throw std::invalid_argument(std::string(name) + " must be finite and in [0,1]");
}
void validate_clouds(const CloudsConfig& config) {
  validate_image(config.width, config.height, config.channels, "Clouds");
  validate_unit(config.alpha, "Clouds alpha");
  validate_unit(config.density, "Clouds density");
}
void validate_layer(const WeatherLayerConfig& config, const char* name) {
  validate_image(config.width, config.height, config.channels, name);
  validate_unit(config.opacity, "weather layer opacity");
  if (!config.layer) throw std::invalid_argument(std::string(name) + " layer is null");
}
}  // namespace

void fast_snowy_landscape_u8(const std::uint8_t* input, std::uint8_t* output,
                             const FastSnowyLandscapeConfig& config, cudaStream_t stream) {
  validate_image(config.width, config.height, config.channels, "FastSnowyLandscape");
  validate_unit(config.snow_point, "FastSnowyLandscape snow_point");
  validate_unit(config.alpha, "FastSnowyLandscape alpha");
  if (!input || !output) throw std::invalid_argument("FastSnowyLandscape input and output are required");
  const int pixels = config.width * config.height;
  fast_snow_kernel<<<(pixels + 255) / 256, 256, 0, stream>>>(input, output, config);
  check(cudaGetLastError(), "fast_snow_kernel");
}

void make_cloud_field(float* output, const CloudsConfig& config) {
  validate_clouds(config);
  const std::size_t count = static_cast<std::size_t>(config.width) * config.height;
  if (count && !output) throw std::invalid_argument("Clouds field output is null");
  // Materialization is host-side by contract; this routine does not inspect a
  // CUDA pointer and produces the same counter-based field as the CPU backend.
  auto host_random = [](std::uint64_t seed, std::uint64_t index) {
    std::uint64_t value = seed + index + 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    value ^= value >> 31;
    return static_cast<float>(value >> 40) / 16777216.0f;
  };
  for (std::size_t pixel = 0; pixel < count; ++pixel) {
    if (config.field) {
      output[pixel] = config.field[pixel];
      continue;
    }
    const std::uint64_t base = static_cast<std::uint64_t>(pixel) * 4ULL;
    output[pixel] = 0.25f * (host_random(config.seed, base) + host_random(config.seed, base + 1) +
                              host_random(config.seed, base + 2) + host_random(config.seed, base + 3));
  }
}

void clouds_u8(const std::uint8_t* input, std::uint8_t* output,
               const CloudsConfig& config, cudaStream_t stream) {
  validate_clouds(config);
  if (!input || !output) throw std::invalid_argument("Clouds input and output are required");
  const int pixels = config.width * config.height;
  clouds_kernel<<<(pixels + 255) / 256, 256, 0, stream>>>(input, output, config);
  check(cudaGetLastError(), "clouds_kernel");
}

void cloud_layer_u8(const std::uint8_t* input, std::uint8_t* output,
                    const CloudLayerConfig& config, cudaStream_t stream) {
  validate_layer(config, "CloudLayer");
  if (!input || !output) throw std::invalid_argument("CloudLayer input and output are required");
  const int pixels = config.width * config.height;
  layer_kernel<<<(pixels + 255) / 256, 256, 0, stream>>>(input, output, config);
  check(cudaGetLastError(), "cloud_layer_kernel");
}
void snowflakes_layer_u8(const std::uint8_t* input, std::uint8_t* output,
                         const SnowflakesLayerConfig& config, cudaStream_t stream) {
  validate_layer(config, "SnowflakesLayer");
  if (!input || !output) throw std::invalid_argument("SnowflakesLayer input and output are required");
  const int pixels = config.width * config.height;
  layer_kernel<<<(pixels + 255) / 256, 256, 0, stream>>>(input, output, config);
  check(cudaGetLastError(), "snowflakes_layer_kernel");
}
void rain_layer_u8(const std::uint8_t* input, std::uint8_t* output,
                   const RainLayerConfig& config, cudaStream_t stream) {
  validate_layer(config, "RainLayer");
  if (!input || !output) throw std::invalid_argument("RainLayer input and output are required");
  const int pixels = config.width * config.height;
  layer_kernel<<<(pixels + 255) / 256, 256, 0, stream>>>(input, output, config);
  check(cudaGetLastError(), "rain_layer_kernel");
}
void snowflakes_u8(const std::uint8_t* input, std::uint8_t* output,
                   const SnowflakesConfig& config, cudaStream_t stream) {
  random_snow_u8(input, output, config, stream);
}
void rain_u8(const std::uint8_t* input, std::uint8_t* output,
             const RainConfig& config, cudaStream_t stream) {
  random_rain_u8(input, output, config, stream);
}

}  // namespace augmatch
