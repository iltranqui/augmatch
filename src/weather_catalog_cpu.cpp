#include "augmatch/weather.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace augmatch {
namespace {

float catalog_clamp(float value) {
  return std::max(0.0f, std::min(1.0f, value));
}

std::uint64_t catalog_splitmix64(std::uint64_t value) {
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}

float catalog_random(std::uint64_t seed, std::uint64_t index) {
  return static_cast<float>(catalog_splitmix64(seed + index) >> 40) / 16777216.0f;
}

void validate_image(int width, int height, int channels, const char* name) {
  if (width <= 0 || height <= 0 || channels <= 0)
    throw std::invalid_argument(std::string(name) + " dimensions and channels must be positive");
}

void validate_unit(float value, const char* name) {
  if (!std::isfinite(value) || value < 0.0f || value > 1.0f)
    throw std::invalid_argument(std::string(name) + " must be finite and in [0,1]");
}

std::uint8_t rounded(float value) {
  return static_cast<std::uint8_t>(std::max(0.0f, std::min(255.0f, std::floor(value + 0.5f))));
}

float cloud_value(const CloudsConfig& config, std::size_t pixel) {
  if (config.field) return config.field[pixel];
  const std::uint64_t base = static_cast<std::uint64_t>(pixel) * 4ULL;
  return 0.25f * (catalog_random(config.seed, base) +
                  catalog_random(config.seed, base + 1) +
                  catalog_random(config.seed, base + 2) +
                  catalog_random(config.seed, base + 3));
}

void validate_clouds(const CloudsConfig& config) {
  validate_image(config.width, config.height, config.channels, "Clouds");
  validate_unit(config.alpha, "Clouds alpha");
  validate_unit(config.density, "Clouds density");
  if (config.field) {
    const std::size_t count = static_cast<std::size_t>(config.width) * config.height;
    for (std::size_t i = 0; i < count; ++i) validate_unit(config.field[i], "Clouds field value");
  }
}

void validate_layer(const WeatherLayerConfig& config, const char* name) {
  validate_image(config.width, config.height, config.channels, name);
  validate_unit(config.opacity, "weather layer opacity");
  const std::size_t count = static_cast<std::size_t>(config.width) * config.height;
  if (count && !config.layer) throw std::invalid_argument(std::string(name) + " layer is null");
  for (std::size_t i = 0; i < count; ++i) validate_unit(config.layer[i], "weather layer value");
}

void composite_layer(const std::uint8_t* input, std::uint8_t* output,
                     const WeatherLayerConfig& config) {
  const std::size_t pixels = static_cast<std::size_t>(config.width) * config.height;
  const std::size_t channels = static_cast<std::size_t>(config.channels);
  for (std::size_t pixel = 0; pixel < pixels; ++pixel) {
    const float alpha = catalog_clamp(config.opacity * config.layer[pixel]);
    const std::uint8_t* source = input + pixel * channels;
    std::uint8_t* destination = output + pixel * channels;
    for (std::size_t channel = 0; channel < channels; ++channel)
      destination[channel] = rounded((1.0f - alpha) * source[channel] + alpha * 255.0f);
  }
}

}  // namespace

void fast_snowy_landscape_u8(const std::uint8_t* input, std::uint8_t* output,
                             const FastSnowyLandscapeConfig& config, cudaStream_t) {
  validate_image(config.width, config.height, config.channels, "FastSnowyLandscape");
  validate_unit(config.snow_point, "FastSnowyLandscape snow_point");
  validate_unit(config.alpha, "FastSnowyLandscape alpha");
  if (!input || !output) throw std::invalid_argument("FastSnowyLandscape input and output are required");
  const std::size_t pixels = static_cast<std::size_t>(config.width) * config.height;
  const int measured_channels = std::min(config.channels, 3);
  for (std::size_t pixel = 0; pixel < pixels; ++pixel) {
    const std::uint8_t* source = input + pixel * config.channels;
    std::uint8_t* destination = output + pixel * config.channels;
    float brightness = 0.0f;
    for (int channel = 0; channel < measured_channels; ++channel) brightness += source[channel] / 255.0f;
    brightness /= static_cast<float>(measured_channels);
    const float span = std::max(1.0e-6f, 1.0f - config.snow_point);
    const float blend = config.alpha * catalog_clamp((brightness - config.snow_point) / span);
    for (int channel = 0; channel < config.channels; ++channel)
      destination[channel] = rounded((1.0f - blend) * source[channel] + blend * 255.0f);
  }
}

void make_cloud_field(float* output, const CloudsConfig& config) {
  validate_clouds(config);
  const std::size_t count = static_cast<std::size_t>(config.width) * config.height;
  if (count && !output) throw std::invalid_argument("Clouds field output is null");
  for (std::size_t pixel = 0; pixel < count; ++pixel) output[pixel] = cloud_value(config, pixel);
}

void clouds_u8(const std::uint8_t* input, std::uint8_t* output,
               const CloudsConfig& config, cudaStream_t) {
  validate_clouds(config);
  if (!input || !output) throw std::invalid_argument("Clouds input and output are required");
  const std::size_t pixels = static_cast<std::size_t>(config.width) * config.height;
  const std::size_t channels = static_cast<std::size_t>(config.channels);
  for (std::size_t pixel = 0; pixel < pixels; ++pixel) {
    const float blend = catalog_clamp(config.alpha * config.density * cloud_value(config, pixel));
    const std::uint8_t* source = input + pixel * channels;
    std::uint8_t* destination = output + pixel * channels;
    for (std::size_t channel = 0; channel < channels; ++channel)
      destination[channel] = rounded((1.0f - blend) * source[channel] + blend * 255.0f);
  }
}

void cloud_layer_u8(const std::uint8_t* input, std::uint8_t* output,
                    const CloudLayerConfig& config, cudaStream_t) {
  validate_layer(config, "CloudLayer");
  if (!input || !output) throw std::invalid_argument("CloudLayer input and output are required");
  composite_layer(input, output, config);
}

void snowflakes_layer_u8(const std::uint8_t* input, std::uint8_t* output,
                         const SnowflakesLayerConfig& config, cudaStream_t) {
  validate_layer(config, "SnowflakesLayer");
  if (!input || !output) throw std::invalid_argument("SnowflakesLayer input and output are required");
  composite_layer(input, output, config);
}

void rain_layer_u8(const std::uint8_t* input, std::uint8_t* output,
                   const RainLayerConfig& config, cudaStream_t) {
  validate_layer(config, "RainLayer");
  if (!input || !output) throw std::invalid_argument("RainLayer input and output are required");
  composite_layer(input, output, config);
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
