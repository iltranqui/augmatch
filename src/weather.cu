#include "augmatch/weather.hpp"

#include <cuda_runtime.h>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace augmatch {
namespace {

__device__ std::uint64_t splitmix64(std::uint64_t value) {
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}

__device__ float unit_random(std::uint64_t seed, std::uint64_t index) {
  return static_cast<float>(splitmix64(seed + index) >> 40) / 16777216.0f;
}

__device__ RainStreak generated_streak(const RandomRainConfig& config, int index) {
  const std::uint64_t base = static_cast<std::uint64_t>(index) * 6ULL;
  const float x0 = unit_random(config.seed, base) * static_cast<float>(config.width);
  const float y0 = unit_random(config.seed, base + 1ULL) * static_cast<float>(config.height);
  const float length = config.length_min +
      (config.length_max - config.length_min) * unit_random(config.seed, base + 2ULL);
  const float angle = (config.angle_min +
      (config.angle_max - config.angle_min) * unit_random(config.seed, base + 3ULL)) *
      0.017453292519943295769f;
  RainStreak streak;
  streak.x0 = x0;
  streak.y0 = y0;
  streak.x1 = x0 + length * cosf(angle);
  streak.y1 = y0 + length * sinf(angle);
  streak.width = config.streak_width;
  streak.alpha = 1.0f;
  return streak;
}

__device__ RainStreak get_streak(const RandomRainConfig& config, int index) {
  return config.streaks ? config.streaks[index] : generated_streak(config, index);
}

__device__ float distance_squared(float px, float py, const RainStreak& streak) {
  const float dx = streak.x1 - streak.x0;
  const float dy = streak.y1 - streak.y0;
  const float length_squared = dx * dx + dy * dy;
  float t = 0.0f;
  if (length_squared > 0.0f)
    t = fmaxf(0.0f, fminf(1.0f, ((px - streak.x0) * dx + (py - streak.y0) * dy) /
                                  length_squared));
  const float ex = px - (streak.x0 + t * dx);
  const float ey = py - (streak.y0 + t * dy);
  return ex * ex + ey * ey;
}

__global__ void random_rain_kernel(const std::uint8_t* input, std::uint8_t* output,
                                   RandomRainConfig config) {
  const int pixel = blockIdx.x * blockDim.x + threadIdx.x;
  const int pixels = config.width * config.height;
  if (pixel >= pixels) return;
  const int x = pixel % config.width;
  const int y = pixel / config.width;
  const float px = static_cast<float>(x) + 0.5f;
  const float py = static_cast<float>(y) + 0.5f;
  const std::uint8_t* source = input + pixel * config.channels;
  std::uint8_t* destination = output + pixel * config.channels;
  for (int channel = 0; channel < config.channels; ++channel) {
    float value = static_cast<float>(source[channel]);
    for (int index = 0; index < config.streak_count; ++index) {
      const RainStreak streak = get_streak(config, index);
      if (distance_squared(px, py, streak) <= 0.25f * streak.width * streak.width) {
        const float blend = fmaxf(0.0f, fminf(1.0f, config.alpha * streak.alpha));
        value = (1.0f - blend) * value + blend * 255.0f;
      }
    }
    value = fmaxf(0.0f, fminf(255.0f, floorf(value + 0.5f)));
    destination[channel] = static_cast<std::uint8_t>(value);
  }
}

void check(cudaError_t error, const char* operation) {
  if (error != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
}

void validate(const RandomRainConfig& config) {
  if (config.width <= 0 || config.height <= 0 || config.channels <= 0 || config.streak_count < 0)
    throw std::invalid_argument("invalid RandomRain dimensions or streak count");
  if (!std::isfinite(config.alpha) || !std::isfinite(config.length_min) ||
      !std::isfinite(config.length_max) || !std::isfinite(config.angle_min) ||
      !std::isfinite(config.angle_max) || !std::isfinite(config.streak_width) ||
      config.alpha < 0.0f || config.alpha > 1.0f || config.length_min < 0.0f ||
      config.length_min > config.length_max || config.streak_width <= 0.0f ||
      config.angle_min > config.angle_max)
    throw std::invalid_argument("invalid RandomRain parameter range");
  if (config.streak_count > 0 && !config.streaks && config.length_max <= 0.0f)
    throw std::invalid_argument("RandomRain generated streak length must be positive");
}

}  // namespace

void make_random_rain_streaks(RainStreak* output, const RandomRainConfig& config) {
  validate(config);
  if (config.streak_count > 0 && !output)
    throw std::invalid_argument("RandomRain streak output is null");
  auto host_splitmix64 = [](std::uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
  };
  auto host_unit = [&](std::uint64_t index) {
    return static_cast<float>(host_splitmix64(config.seed + index) >> 40) / 16777216.0f;
  };
  for (int index = 0; index < config.streak_count; ++index) {
    if (config.streaks) {
      output[index] = config.streaks[index];
      continue;
    }
    const std::uint64_t base = static_cast<std::uint64_t>(index) * 6ULL;
    const float x0 = host_unit(base) * static_cast<float>(config.width);
    const float y0 = host_unit(base + 1ULL) * static_cast<float>(config.height);
    const float length = config.length_min +
        (config.length_max - config.length_min) * host_unit(base + 2ULL);
    const float angle = (config.angle_min +
        (config.angle_max - config.angle_min) * host_unit(base + 3ULL)) *
        0.017453292519943295769f;
    output[index] = {x0, y0, x0 + length * std::cos(angle), y0 + length * std::sin(angle),
                     config.streak_width, 1.0f};
  }
}

void random_rain_u8(const std::uint8_t* input, std::uint8_t* output,
                    const RandomRainConfig& config, cudaStream_t stream) {
  validate(config);
  if (!input || !output) throw std::invalid_argument("RandomRain input and output are required");
  const int pixels = config.width * config.height;
  random_rain_kernel<<<(pixels + 255) / 256, 256, 0, stream>>>(input, output, config);
  check(cudaGetLastError(), "random_rain_kernel");
}

namespace {

__device__ Snowflake generated_snowflake(const RandomSnowConfig& config, int index) {
  const std::uint64_t base = static_cast<std::uint64_t>(index) * 4ULL;
  const float x = unit_random(config.seed, base) * static_cast<float>(config.width);
  const float y = unit_random(config.seed, base + 1ULL) * static_cast<float>(config.height);
  const float radius = config.radius_min +
      (config.radius_max - config.radius_min) * unit_random(config.seed, base + 2ULL);
  Snowflake snowflake;
  snowflake.x = x;
  snowflake.y = y;
  snowflake.radius = radius;
  snowflake.alpha = 1.0f;
  return snowflake;
}

__device__ Snowflake get_snowflake(const RandomSnowConfig& config, int index) {
  return config.snowflakes ? config.snowflakes[index] : generated_snowflake(config, index);
}

__global__ void random_snow_kernel(const std::uint8_t* input, std::uint8_t* output,
                                   RandomSnowConfig config) {
  const int pixel = blockIdx.x * blockDim.x + threadIdx.x;
  const int pixels = config.width * config.height;
  if (pixel >= pixels) return;
  const int x = pixel % config.width;
  const int y = pixel / config.width;
  const float px = static_cast<float>(x) + 0.5f;
  const float py = static_cast<float>(y) + 0.5f;
  const std::uint8_t* source = input + pixel * config.channels;
  std::uint8_t* destination = output + pixel * config.channels;
  for (int channel = 0; channel < config.channels; ++channel) {
    float value = static_cast<float>(source[channel]);
    for (int index = 0; index < config.snowflake_count; ++index) {
      const Snowflake snowflake = get_snowflake(config, index);
      const float dx = px - snowflake.x;
      const float dy = py - snowflake.y;
      if (dx * dx + dy * dy <= snowflake.radius * snowflake.radius) {
        const float blend = fmaxf(0.0f, fminf(1.0f, config.alpha * snowflake.alpha));
        value = (1.0f - blend) * value + blend * 255.0f;
      }
    }
    value = fmaxf(0.0f, fminf(255.0f, floorf(value + 0.5f)));
    destination[channel] = static_cast<std::uint8_t>(value);
  }
}

void validate_snow(const RandomSnowConfig& config) {
  if (config.width <= 0 || config.height <= 0 || config.channels <= 0 ||
      config.snowflake_count < 0)
    throw std::invalid_argument("invalid RandomSnow dimensions or snowflake count");
  if (!std::isfinite(config.alpha) || !std::isfinite(config.radius_min) ||
      !std::isfinite(config.radius_max) || config.alpha < 0.0f || config.alpha > 1.0f ||
      config.radius_min <= 0.0f || config.radius_min > config.radius_max)
    throw std::invalid_argument("invalid RandomSnow parameter range");
  // Explicit snowflake records are device memory on CUDA, so their values
  // are validated by the caller before copying and are not dereferenced here.
}

}  // namespace

void make_random_snowflakes(Snowflake* output, const RandomSnowConfig& config) {
  validate_snow(config);
  if (config.snowflake_count > 0 && !output)
    throw std::invalid_argument("RandomSnow snowflake output is null");
  auto host_splitmix64 = [](std::uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
  };
  auto host_unit = [&](std::uint64_t index) {
    return static_cast<float>(host_splitmix64(config.seed + index) >> 40) / 16777216.0f;
  };
  for (int index = 0; index < config.snowflake_count; ++index) {
    if (config.snowflakes) {
      output[index] = config.snowflakes[index];
      continue;
    }
    const std::uint64_t base = static_cast<std::uint64_t>(index) * 4ULL;
    output[index] = {host_unit(base) * static_cast<float>(config.width),
                     host_unit(base + 1ULL) * static_cast<float>(config.height),
                     config.radius_min + (config.radius_max - config.radius_min) * host_unit(base + 2ULL),
                     1.0f};
  }
}

void random_snow_u8(const std::uint8_t* input, std::uint8_t* output,
                   const RandomSnowConfig& config, cudaStream_t stream) {
  validate_snow(config);
  if (!input || !output) throw std::invalid_argument("RandomSnow input and output are required");
  const int pixels = config.width * config.height;
  random_snow_kernel<<<(pixels + 255) / 256, 256, 0, stream>>>(input, output, config);
  check(cudaGetLastError(), "random_snow_kernel");
}

namespace {

__global__ void snow_stamp_kernel(const std::uint8_t* input, std::uint8_t* output,
                                  SnowStampConfig config) {
  const int pixel = blockIdx.x * blockDim.x + threadIdx.x;
  const int pixels = config.width * config.height;
  if (pixel >= pixels) return;
  const int x = pixel % config.width;
  const int y = pixel / config.width;
  for (int channel = 0; channel < config.channels; ++channel)
    output[pixel * config.channels + channel] = input[pixel * config.channels + channel];
  for (int placement_index = 0; placement_index < config.placement_count; ++placement_index) {
    const SnowStampPlacement placement = config.placements[placement_index];
    const int sx = x - placement.x;
    const int sy = y - placement.y;
    if (sx < 0 || sx >= config.stamp_width || sy < 0 || sy >= config.stamp_height) continue;
    const int stamp_pixel = sy * config.stamp_width + sx;
    const float mask_alpha = config.stamp_mask ?
        static_cast<float>(config.stamp_mask[stamp_pixel]) / 255.0f : 1.0f;
    const float blend = fmaxf(0.0f, fminf(1.0f,
        config.alpha * placement.alpha * mask_alpha));
    if (blend == 0.0f) continue;
    for (int channel = 0; channel < config.channels; ++channel) {
      std::uint8_t fill = config.fill;
      if (config.stamp_image) {
        const int stamp_channel = config.stamp_channels == 1 ? 0 : channel;
        fill = config.stamp_image[stamp_pixel * config.stamp_channels + stamp_channel];
      }
      const int offset = pixel * config.channels + channel;
      const float value = (1.0f - blend) * static_cast<float>(output[offset]) +
                          blend * static_cast<float>(fill);
      output[offset] = static_cast<std::uint8_t>(fmaxf(0.0f, fminf(255.0f,
          floorf(value + 0.5f))));
    }
  }
}

void validate_snow_stamp(const SnowStampConfig& config) {
  if (config.width <= 0 || config.height <= 0 || config.channels <= 0 ||
      config.stamp_width <= 0 || config.stamp_height <= 0 ||
      (config.stamp_channels != 1 && config.stamp_channels != config.channels) ||
      config.placement_count < 0)
    throw std::invalid_argument("invalid SnowStamp dimensions or placement count");
  if (!std::isfinite(config.alpha) || config.alpha < 0.0f || config.alpha > 1.0f)
    throw std::invalid_argument("invalid SnowStamp alpha");
  if (config.placement_count > 0 && !config.placements)
    throw std::invalid_argument("SnowStamp placements are required");
}

}  // namespace

void snow_stamp_u8(const std::uint8_t* input, std::uint8_t* output,
                   const SnowStampConfig& config, cudaStream_t stream) {
  validate_snow_stamp(config);
  if (!input || !output) throw std::invalid_argument("SnowStamp input and output are required");
  const int pixels = config.width * config.height;
  snow_stamp_kernel<<<(pixels + 255) / 256, 256, 0, stream>>>(input, output, config);
  check(cudaGetLastError(), "snow_stamp_kernel");
}

namespace {

__device__ GravelParticle generated_gravel_particle(const RandomGravelConfig& config, int index) {
  const std::uint64_t base = static_cast<std::uint64_t>(index) * 4ULL;
  GravelParticle particle;
  particle.x = unit_random(config.seed, base) * static_cast<float>(config.width);
  particle.y = unit_random(config.seed, base + 1ULL) * static_cast<float>(config.height);
  particle.radius = config.radius_min +
      (config.radius_max - config.radius_min) * unit_random(config.seed, base + 2ULL);
  particle.alpha = 1.0f;
  particle.fill = config.fill;
  return particle;
}

__device__ GravelParticle get_gravel_particle(const RandomGravelConfig& config, int index) {
  return config.particles ? config.particles[index] : generated_gravel_particle(config, index);
}

__global__ void random_gravel_kernel(const std::uint8_t* input, std::uint8_t* output,
                                     RandomGravelConfig config) {
  const int pixel = blockIdx.x * blockDim.x + threadIdx.x;
  const int pixels = config.width * config.height;
  if (pixel >= pixels) return;
  const int x = pixel % config.width;
  const int y = pixel / config.width;
  const float px = static_cast<float>(x) + 0.5f;
  const float py = static_cast<float>(y) + 0.5f;
  const std::uint8_t* source = input + pixel * config.channels;
  std::uint8_t* destination = output + pixel * config.channels;
  for (int channel = 0; channel < config.channels; ++channel) {
    float value = static_cast<float>(source[channel]);
    for (int index = 0; index < config.particle_count; ++index) {
      const GravelParticle particle = get_gravel_particle(config, index);
      const float dx = px - particle.x;
      const float dy = py - particle.y;
      if (dx * dx + dy * dy <= particle.radius * particle.radius) {
        const float blend = fminf(1.0f, fmaxf(0.0f, config.alpha * particle.alpha));
        value = (1.0f - blend) * value + blend * static_cast<float>(particle.fill);
      }
    }
    destination[channel] = static_cast<std::uint8_t>(fminf(255.0f, fmaxf(0.0f, floorf(value + 0.5f))));
  }
}

void validate_gravel(const RandomGravelConfig& config) {
  if (config.width <= 0 || config.height <= 0 || config.channels <= 0 || config.particle_count < 0)
    throw std::invalid_argument("invalid RandomGravel dimensions or particle count");
  if (!std::isfinite(config.alpha) || !std::isfinite(config.radius_min) ||
      !std::isfinite(config.radius_max) || config.alpha < 0.0f || config.alpha > 1.0f ||
      config.radius_min <= 0.0f || config.radius_min > config.radius_max)
    throw std::invalid_argument("invalid RandomGravel parameter range");
}

}  // namespace

void make_random_gravel_particles(GravelParticle* output, const RandomGravelConfig& config) {
  validate_gravel(config);
  if (config.particle_count > 0 && !output)
    throw std::invalid_argument("RandomGravel particle output is null");
  auto host_splitmix64 = [](std::uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
  };
  auto host_unit = [&](std::uint64_t index) {
    return static_cast<float>(host_splitmix64(config.seed + index) >> 40) / 16777216.0f;
  };
  for (int index = 0; index < config.particle_count; ++index) {
    if (config.particles) {
      output[index] = config.particles[index];
      continue;
    }
    const std::uint64_t base = static_cast<std::uint64_t>(index) * 4ULL;
    output[index] = {host_unit(base) * static_cast<float>(config.width),
                     host_unit(base + 1ULL) * static_cast<float>(config.height),
                     config.radius_min + (config.radius_max - config.radius_min) * host_unit(base + 2ULL),
                     1.0f, config.fill};
  }
}

void random_gravel_u8(const std::uint8_t* input, std::uint8_t* output,
                      const RandomGravelConfig& config, cudaStream_t stream) {
  validate_gravel(config);
  if (!input || !output) throw std::invalid_argument("RandomGravel input and output are required");
  const int pixels = config.width * config.height;
  random_gravel_kernel<<<(pixels + 255) / 256, 256, 0, stream>>>(input, output, config);
  check(cudaGetLastError(), "random_gravel_kernel");
}

namespace {

__device__ float fog_field_value(const RandomFogConfig& config, int pixel) {
  return config.field ? config.field[pixel] :
      unit_random(config.seed, static_cast<std::uint64_t>(pixel));
}

__global__ void random_fog_kernel(const std::uint8_t* input, std::uint8_t* output,
                                  RandomFogConfig config) {
  const int pixel = blockIdx.x * blockDim.x + threadIdx.x;
  const int pixels = config.width * config.height;
  if (pixel >= pixels) return;
  const float field = fog_field_value(config, pixel);
  const float blend = fminf(1.0f, fmaxf(0.0f, config.opacity * config.density * field));
  const std::uint8_t* source = input + pixel * config.channels;
  std::uint8_t* destination = output + pixel * config.channels;
  for (int channel = 0; channel < config.channels; ++channel) {
    const float value = (1.0f - blend) * static_cast<float>(source[channel]) + blend * 255.0f;
    destination[channel] = static_cast<std::uint8_t>(fminf(255.0f, fmaxf(0.0f, floorf(value + 0.5f))));
  }
}

void validate_fog(const RandomFogConfig& config) {
  if (config.width <= 0 || config.height <= 0 || config.channels <= 0 ||
      !std::isfinite(config.density) || !std::isfinite(config.opacity) ||
      config.density < 0.0f || config.density > 1.0f ||
      config.opacity < 0.0f || config.opacity > 1.0f)
    throw std::invalid_argument("invalid RandomFog dimensions, density, or opacity");
}

}  // namespace

void make_random_fog_field(float* output, const RandomFogConfig& config) {
  validate_fog(config);
  const std::size_t pixels = static_cast<std::size_t>(config.width) * config.height;
  if (pixels > 0 && !output)
    throw std::invalid_argument("RandomFog field output is null");
  auto host_splitmix64 = [](std::uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
  };
  for (std::size_t pixel = 0; pixel < pixels; ++pixel)
    output[pixel] = static_cast<float>(host_splitmix64(config.seed + pixel) >> 40) / 16777216.0f;
}

void random_fog_u8(const std::uint8_t* input, std::uint8_t* output,
                   const RandomFogConfig& config, cudaStream_t stream) {
  validate_fog(config);
  if (!input || !output) throw std::invalid_argument("RandomFog input and output are required");
  const int pixels = config.width * config.height;
  random_fog_kernel<<<(pixels + 255) / 256, 256, 0, stream>>>(input, output, config);
  check(cudaGetLastError(), "random_fog_kernel");
}

namespace {

__device__ SunFlareRay generated_sun_flare_ray(const RandomSunFlareConfig& config, int index) {
  const std::uint64_t base = static_cast<std::uint64_t>(index) * 4ULL;
  SunFlareRay ray;
  ray.angle = config.ray_angle_min +
      (config.ray_angle_max - config.ray_angle_min) * unit_random(config.seed, base);
  ray.length = config.ray_length_min +
      (config.ray_length_max - config.ray_length_min) * unit_random(config.seed, base + 1ULL);
  ray.width = config.ray_width_min +
      (config.ray_width_max - config.ray_width_min) * unit_random(config.seed, base + 2ULL);
  ray.alpha = 1.0f;
  return ray;
}

__device__ SunFlareRay get_sun_flare_ray(const RandomSunFlareConfig& config, int index) {
  return config.rays ? config.rays[index] : generated_sun_flare_ray(config, index);
}

__device__ float blend_sun_flare(float value, float alpha) {
  const float blend = fminf(1.0f, fmaxf(0.0f, alpha));
  return (1.0f - blend) * value + blend * 255.0f;
}

__global__ void random_sun_flare_kernel(const std::uint8_t* input, std::uint8_t* output,
                                        RandomSunFlareConfig config) {
  const int pixel = blockIdx.x * blockDim.x + threadIdx.x;
  const int pixels = config.width * config.height;
  if (pixel >= pixels) return;
  const int x = pixel % config.width;
  const int y = pixel / config.width;
  const float px = static_cast<float>(x) + 0.5f;
  const float py = static_cast<float>(y) + 0.5f;
  const std::uint8_t* source = input + pixel * config.channels;
  std::uint8_t* destination = output + pixel * config.channels;
  const float sx = px - config.source.x;
  const float sy = py - config.source.y;
  const bool in_source = sx * sx + sy * sy <= config.source.radius * config.source.radius;
  for (int channel = 0; channel < config.channels; ++channel) {
    float value = static_cast<float>(source[channel]);
    if (in_source)
      value = blend_sun_flare(value, config.opacity * config.source.alpha);
    for (int index = 0; index < config.ray_count; ++index) {
      const SunFlareRay ray = get_sun_flare_ray(config, index);
      const float angle = ray.angle * 0.017453292519943295769f;
      RainStreak segment;
      segment.x0 = config.source.x;
      segment.y0 = config.source.y;
      segment.x1 = config.source.x + ray.length * cosf(angle);
      segment.y1 = config.source.y + ray.length * sinf(angle);
      segment.width = ray.width;
      segment.alpha = ray.alpha;
      if (distance_squared(px, py, segment) <= 0.25f * ray.width * ray.width)
        value = blend_sun_flare(value, config.opacity * ray.alpha);
    }
    destination[channel] = static_cast<std::uint8_t>(fminf(255.0f, fmaxf(0.0f, floorf(value + 0.5f))));
  }
}

void validate_sun_flare(const RandomSunFlareConfig& config) {
  if (config.width <= 0 || config.height <= 0 || config.channels <= 0 || config.ray_count < 0)
    throw std::invalid_argument("invalid RandomSunFlare dimensions or ray count");
  if (!std::isfinite(config.source.x) || !std::isfinite(config.source.y) ||
      !std::isfinite(config.source.radius) || !std::isfinite(config.source.alpha) ||
      !std::isfinite(config.opacity) || !std::isfinite(config.ray_length_min) ||
      !std::isfinite(config.ray_length_max) || !std::isfinite(config.ray_angle_min) ||
      !std::isfinite(config.ray_angle_max) || !std::isfinite(config.ray_width_min) ||
      !std::isfinite(config.ray_width_max) || config.source.radius < 0.0f ||
      config.source.alpha < 0.0f || config.source.alpha > 1.0f || config.opacity < 0.0f ||
      config.opacity > 1.0f || config.ray_length_min < 0.0f ||
      config.ray_length_min > config.ray_length_max || config.ray_width_min <= 0.0f ||
      config.ray_width_min > config.ray_width_max || config.ray_angle_min > config.ray_angle_max)
    throw std::invalid_argument("invalid RandomSunFlare parameter range");
}

}  // namespace

void make_random_sun_flare_rays(SunFlareRay* output, const RandomSunFlareConfig& config) {
  validate_sun_flare(config);
  if (config.ray_count > 0 && !output)
    throw std::invalid_argument("RandomSunFlare ray output is null");
  auto host_splitmix64 = [](std::uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
  };
  auto host_unit = [&](std::uint64_t index) {
    return static_cast<float>(host_splitmix64(config.seed + index) >> 40) / 16777216.0f;
  };
  for (int index = 0; index < config.ray_count; ++index) {
    if (config.rays) {
      output[index] = config.rays[index];
      continue;
    }
    const std::uint64_t base = static_cast<std::uint64_t>(index) * 4ULL;
    output[index] = {config.ray_angle_min + (config.ray_angle_max - config.ray_angle_min) * host_unit(base),
                     config.ray_length_min + (config.ray_length_max - config.ray_length_min) * host_unit(base + 1ULL),
                     config.ray_width_min + (config.ray_width_max - config.ray_width_min) * host_unit(base + 2ULL), 1.0f};
  }
}

void random_sun_flare_u8(const std::uint8_t* input, std::uint8_t* output,
                         const RandomSunFlareConfig& config, cudaStream_t stream) {
  validate_sun_flare(config);
  if (!input || !output) throw std::invalid_argument("RandomSunFlare input and output are required");
  const int pixels = config.width * config.height;
  random_sun_flare_kernel<<<(pixels + 255) / 256, 256, 0, stream>>>(input, output, config);
  check(cudaGetLastError(), "random_sun_flare_kernel");
}

namespace {

__device__ bool point_in_shadow_polygon(float px, float py, const ShadowPolygon& polygon) {
  bool inside = false;
  for (int i = 0, previous = polygon.point_count - 1; i < polygon.point_count;
       previous = i++) {
    const ShadowPoint a = polygon.points[previous];
    const ShadowPoint b = polygon.points[i];
    const float cross = (px - a.x) * (b.y - a.y) - (py - a.y) * (b.x - a.x);
    const float min_x = fminf(a.x, b.x), max_x = fmaxf(a.x, b.x);
    const float min_y = fminf(a.y, b.y), max_y = fmaxf(a.y, b.y);
    if (fabsf(cross) <= 1.0e-6f && px >= min_x && px <= max_x && py >= min_y && py <= max_y)
      return true;
    if ((a.y > py) != (b.y > py) && px < (b.x - a.x) * (py - a.y) / (b.y - a.y) + a.x)
      inside = !inside;
  }
  return inside;
}

__device__ bool point_in_shadow_rectangle(float px, float py, const ShadowRectangle& rectangle) {
  return px >= fminf(rectangle.x0, rectangle.x1) && px <= fmaxf(rectangle.x0, rectangle.x1) &&
         py >= fminf(rectangle.y0, rectangle.y1) && py <= fmaxf(rectangle.y0, rectangle.y1);
}

__device__ float blend_shadow(float value, float opacity, float local_alpha, unsigned char fill) {
  const float alpha = fminf(1.0f, fmaxf(0.0f, opacity * local_alpha));
  return (1.0f - alpha) * value + alpha * static_cast<float>(fill);
}

__global__ void random_shadow_kernel(const std::uint8_t* input, std::uint8_t* output,
                                     RandomShadowConfig config) {
  const int pixel = blockIdx.x * blockDim.x + threadIdx.x;
  const int pixels = config.width * config.height;
  if (pixel >= pixels) return;
  const int x = pixel % config.width;
  const int y = pixel / config.width;
  const float px = static_cast<float>(x) + 0.5f;
  const float py = static_cast<float>(y) + 0.5f;
  const std::uint8_t* source = input + pixel * config.channels;
  std::uint8_t* destination = output + pixel * config.channels;
  for (int channel = 0; channel < config.channels; ++channel) {
    float value = static_cast<float>(source[channel]);
    for (int i = 0; i < config.polygon_count; ++i)
      if (point_in_shadow_polygon(px, py, config.polygons[i]))
        value = blend_shadow(value, config.opacity, config.polygons[i].alpha, config.fill);
    for (int i = 0; i < config.rectangle_count; ++i)
      if (point_in_shadow_rectangle(px, py, config.rectangles[i]))
        value = blend_shadow(value, config.opacity, config.rectangles[i].alpha, config.fill);
    destination[channel] = static_cast<std::uint8_t>(fminf(255.0f, fmaxf(0.0f, floorf(value + 0.5f))));
  }
}

void validate_random_shadow(const RandomShadowConfig& config) {
  if (config.width <= 0 || config.height <= 0 || config.channels <= 0 ||
      config.polygon_count < 0 || config.rectangle_count < 0 ||
      !std::isfinite(config.opacity) || config.opacity < 0.0f || config.opacity > 1.0f)
    throw std::invalid_argument("invalid RandomShadow dimensions, counts, or opacity");
  if ((config.polygon_count > 0 && !config.polygons) ||
      (config.rectangle_count > 0 && !config.rectangles))
    throw std::invalid_argument("RandomShadow mask list is null");
}

}  // namespace

void random_shadow_u8(const std::uint8_t* input, std::uint8_t* output,
                      const RandomShadowConfig& config, cudaStream_t stream) {
  validate_random_shadow(config);
  if (!input || !output) throw std::invalid_argument("RandomShadow input and output are required");
  const int pixels = config.width * config.height;
  random_shadow_kernel<<<(pixels + 255) / 256, 256, 0, stream>>>(input, output, config);
  check(cudaGetLastError(), "random_shadow_kernel");
}

namespace {

__device__ SpatterDroplet generated_spatter_droplet(const RandomSpatterConfig& config,
                                                    int index) {
  const std::uint64_t base = static_cast<std::uint64_t>(index) * 4ULL;
  SpatterDroplet droplet;
  droplet.x = unit_random(config.seed, base) * static_cast<float>(config.width);
  droplet.y = unit_random(config.seed, base + 1ULL) * static_cast<float>(config.height);
  droplet.radius = config.radius_min + (config.radius_max - config.radius_min) *
      unit_random(config.seed, base + 2ULL);
  droplet.red = config.red;
  droplet.green = config.green;
  droplet.blue = config.blue;
  droplet.alpha = 1.0f;
  return droplet;
}

__device__ SpatterDroplet get_spatter_droplet(const RandomSpatterConfig& config, int index) {
  return config.droplets ? config.droplets[index] : generated_spatter_droplet(config, index);
}

__device__ float blend_spatter(float value, float alpha, std::uint8_t color) {
  const float blend = fminf(1.0f, fmaxf(0.0f, alpha));
  return (1.0f - blend) * value + blend * static_cast<float>(color);
}

__global__ void random_spatter_kernel(const std::uint8_t* input, std::uint8_t* output,
                                      RandomSpatterConfig config) {
  const int pixel = blockIdx.x * blockDim.x + threadIdx.x;
  const int pixels = config.width * config.height;
  if (pixel >= pixels) return;
  const int x = pixel % config.width;
  const int y = pixel / config.width;
  const float px = static_cast<float>(x) + 0.5f;
  const float py = static_cast<float>(y) + 0.5f;
  const std::uint8_t* source = input + pixel * config.channels;
  std::uint8_t* destination = output + pixel * config.channels;
  for (int channel = 0; channel < config.channels; ++channel) {
    float value = static_cast<float>(source[channel]);
    if (channel < 3) {
      for (int index = 0; index < config.droplet_count; ++index) {
        const SpatterDroplet droplet = get_spatter_droplet(config, index);
        const float dx = px - droplet.x;
        const float dy = py - droplet.y;
        if (dx * dx + dy * dy <= droplet.radius * droplet.radius) {
          const std::uint8_t color = channel == 0 ? droplet.red :
              (channel == 1 ? droplet.green : droplet.blue);
          value = blend_spatter(value, config.alpha * droplet.alpha, color);
        }
      }
    }
    destination[channel] = static_cast<std::uint8_t>(fminf(255.0f, fmaxf(0.0f,
        floorf(value + 0.5f))));
  }
}

void validate_random_spatter(const RandomSpatterConfig& config) {
  if (config.width <= 0 || config.height <= 0 || config.channels <= 0 ||
      config.droplet_count < 0 || !std::isfinite(config.alpha) ||
      !std::isfinite(config.radius_min) || !std::isfinite(config.radius_max) ||
      config.alpha < 0.0f || config.alpha > 1.0f || config.radius_min <= 0.0f ||
      config.radius_min > config.radius_max)
    throw std::invalid_argument("invalid RandomSpatter dimensions, count, or parameter range");
}

}  // namespace

void make_random_spatter_droplets(SpatterDroplet* output,
                                  const RandomSpatterConfig& config) {
  validate_random_spatter(config);
  if (config.droplet_count > 0 && !output)
    throw std::invalid_argument("RandomSpatter droplet output is null");
  auto host_splitmix64 = [](std::uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
  };
  auto host_unit = [&](std::uint64_t index) {
    return static_cast<float>(host_splitmix64(config.seed + index) >> 40) / 16777216.0f;
  };
  for (int index = 0; index < config.droplet_count; ++index) {
    if (config.droplets) {
      output[index] = config.droplets[index];
      continue;
    }
    const std::uint64_t base = static_cast<std::uint64_t>(index) * 4ULL;
    output[index] = {host_unit(base) * static_cast<float>(config.width),
                     host_unit(base + 1ULL) * static_cast<float>(config.height),
                     config.radius_min + (config.radius_max - config.radius_min) * host_unit(base + 2ULL),
                     config.red, config.green, config.blue, 1.0f};
  }
}

void random_spatter_u8(const std::uint8_t* input, std::uint8_t* output,
                       const RandomSpatterConfig& config, cudaStream_t stream) {
  validate_random_spatter(config);
  if (!input || !output) throw std::invalid_argument("RandomSpatter input and output are required");
  const int pixels = config.width * config.height;
  random_spatter_kernel<<<(pixels + 255) / 256, 256, 0, stream>>>(input, output, config);
  check(cudaGetLastError(), "random_spatter_kernel");
}

}  // namespace augmatch
