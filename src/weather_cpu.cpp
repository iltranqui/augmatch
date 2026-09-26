#include "augmatch/weather/weather.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace augmatch {
namespace {

std::uint64_t splitmix64(std::uint64_t value) {
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}

float unit_random(std::uint64_t seed, std::uint64_t index) {
  return static_cast<float>(splitmix64(seed + index) >> 40) / 16777216.0f;
}

void validate(const RandomRainConfig& config) {
  if (config.width <= 0 || config.height <= 0 || config.channels <= 0)
    throw std::invalid_argument("RandomRain dimensions and channels must be positive");
  if (config.streak_count < 0)
    throw std::invalid_argument("RandomRain streak count must be nonnegative");
  const float values[] = {config.alpha, config.length_min, config.length_max,
                          config.angle_min, config.angle_max, config.streak_width};
  for (float value : values)
    if (!std::isfinite(value)) throw std::invalid_argument("RandomRain parameters must be finite");
  if (config.alpha < 0.0f || config.alpha > 1.0f || config.length_min < 0.0f ||
      config.length_min > config.length_max || config.streak_width <= 0.0f ||
      config.angle_min > config.angle_max)
    throw std::invalid_argument("invalid RandomRain parameter range");
  if (config.streak_count > 0 && !config.streaks && config.length_max <= 0.0f)
    throw std::invalid_argument("RandomRain generated streak length must be positive");
  if (config.streak_count > 0 && config.streaks) {
    for (int i = 0; i < config.streak_count; ++i) {
      const RainStreak& streak = config.streaks[i];
      const float streak_values[] = {streak.x0, streak.y0, streak.x1, streak.y1,
                                     streak.width, streak.alpha};
      for (float value : streak_values)
        if (!std::isfinite(value)) throw std::invalid_argument("RandomRain streaks must be finite");
      if (streak.width <= 0.0f || streak.alpha < 0.0f || streak.alpha > 1.0f)
        throw std::invalid_argument("invalid explicit RandomRain streak");
    }
  }
}

RainStreak generated_streak(const RandomRainConfig& config, int index) {
  const std::uint64_t base = static_cast<std::uint64_t>(index) * 6ULL;
  const float x0 = unit_random(config.seed, base + 0ULL) * static_cast<float>(config.width);
  const float y0 = unit_random(config.seed, base + 1ULL) * static_cast<float>(config.height);
  const float length = config.length_min +
      (config.length_max - config.length_min) * unit_random(config.seed, base + 2ULL);
  const float angle = (config.angle_min +
      (config.angle_max - config.angle_min) * unit_random(config.seed, base + 3ULL)) *
      0.017453292519943295769f;
  return {x0, y0, x0 + length * std::cos(angle), y0 + length * std::sin(angle),
          config.streak_width, 1.0f};
}

float distance_squared(float px, float py, const RainStreak& streak) {
  const float dx = streak.x1 - streak.x0;
  const float dy = streak.y1 - streak.y0;
  const float length_squared = dx * dx + dy * dy;
  float t = 0.0f;
  if (length_squared > 0.0f)
    t = std::max(0.0f, std::min(1.0f, ((px - streak.x0) * dx + (py - streak.y0) * dy) /
                                      length_squared));
  const float nearest_x = streak.x0 + t * dx;
  const float nearest_y = streak.y0 + t * dy;
  const float ex = px - nearest_x;
  const float ey = py - nearest_y;
  return ex * ex + ey * ey;
}

RainStreak get_streak(const RandomRainConfig& config, int index) {
  return config.streaks ? config.streaks[index] : generated_streak(config, index);
}

}  // namespace

void make_random_rain_streaks(RainStreak* output, const RandomRainConfig& config) {
  validate(config);
  if (config.streak_count > 0 && !output)
    throw std::invalid_argument("RandomRain streak output is null");
  RandomRainConfig generated = config;
  generated.streaks = nullptr;
  for (int i = 0; i < config.streak_count; ++i)
    output[i] = config.streaks ? config.streaks[i] : generated_streak(generated, i);
}

void random_rain_u8(const std::uint8_t* input, std::uint8_t* output,
                    const RandomRainConfig& config, cudaStream_t) {
  validate(config);
  if (!input || !output) throw std::invalid_argument("RandomRain input and output are required");
  const std::size_t pixels = static_cast<std::size_t>(config.width) * config.height;
  for (std::size_t pixel = 0; pixel < pixels; ++pixel) {
    const int x = static_cast<int>(pixel % static_cast<std::size_t>(config.width));
    const int y = static_cast<int>(pixel / static_cast<std::size_t>(config.width));
    const float px = static_cast<float>(x) + 0.5f;
    const float py = static_cast<float>(y) + 0.5f;
    const std::uint8_t* source = input + pixel * static_cast<std::size_t>(config.channels);
    std::uint8_t* destination = output + pixel * static_cast<std::size_t>(config.channels);
    for (int channel = 0; channel < config.channels; ++channel) {
      float value = static_cast<float>(source[channel]);
      for (int index = 0; index < config.streak_count; ++index) {
        const RainStreak streak = get_streak(config, index);
        if (distance_squared(px, py, streak) <= 0.25f * streak.width * streak.width) {
          const float blend = std::max(0.0f, std::min(1.0f, config.alpha * streak.alpha));
          value = (1.0f - blend) * value + blend * 255.0f;
        }
      }
      destination[channel] = static_cast<std::uint8_t>(std::max(0.0f, std::min(255.0f,
          std::floor(value + 0.5f))));
    }
  }
}

namespace {

void validate_snow(const RandomSnowConfig& config) {
  if (config.width <= 0 || config.height <= 0 || config.channels <= 0)
    throw std::invalid_argument("RandomSnow dimensions and channels must be positive");
  if (config.snowflake_count < 0)
    throw std::invalid_argument("RandomSnow snowflake count must be nonnegative");
  if (!std::isfinite(config.alpha) || !std::isfinite(config.radius_min) ||
      !std::isfinite(config.radius_max) || config.alpha < 0.0f || config.alpha > 1.0f ||
      config.radius_min <= 0.0f || config.radius_min > config.radius_max)
    throw std::invalid_argument("invalid RandomSnow parameter range");
  if (config.snowflake_count > 0 && config.snowflakes) {
    for (int i = 0; i < config.snowflake_count; ++i) {
      const Snowflake& snowflake = config.snowflakes[i];
      if (!std::isfinite(snowflake.x) || !std::isfinite(snowflake.y) ||
          !std::isfinite(snowflake.radius) || !std::isfinite(snowflake.alpha) ||
          snowflake.radius <= 0.0f || snowflake.alpha < 0.0f || snowflake.alpha > 1.0f)
        throw std::invalid_argument("invalid explicit RandomSnow snowflake");
    }
  }
}

Snowflake generated_snowflake(const RandomSnowConfig& config, int index) {
  const std::uint64_t base = static_cast<std::uint64_t>(index) * 4ULL;
  const float x = unit_random(config.seed, base) * static_cast<float>(config.width);
  const float y = unit_random(config.seed, base + 1ULL) * static_cast<float>(config.height);
  const float radius = config.radius_min +
      (config.radius_max - config.radius_min) * unit_random(config.seed, base + 2ULL);
  return {x, y, radius, 1.0f};
}

Snowflake get_snowflake(const RandomSnowConfig& config, int index) {
  return config.snowflakes ? config.snowflakes[index] : generated_snowflake(config, index);
}

}  // namespace

void make_random_snowflakes(Snowflake* output, const RandomSnowConfig& config) {
  validate_snow(config);
  if (config.snowflake_count > 0 && !output)
    throw std::invalid_argument("RandomSnow snowflake output is null");
  for (int i = 0; i < config.snowflake_count; ++i)
    output[i] = config.snowflakes ? config.snowflakes[i] : generated_snowflake(config, i);
}

void random_snow_u8(const std::uint8_t* input, std::uint8_t* output,
                   const RandomSnowConfig& config, cudaStream_t) {
  validate_snow(config);
  if (!input || !output) throw std::invalid_argument("RandomSnow input and output are required");
  const std::size_t pixels = static_cast<std::size_t>(config.width) * config.height;
  for (std::size_t pixel = 0; pixel < pixels; ++pixel) {
    const int x = static_cast<int>(pixel % static_cast<std::size_t>(config.width));
    const int y = static_cast<int>(pixel / static_cast<std::size_t>(config.width));
    const float px = static_cast<float>(x) + 0.5f;
    const float py = static_cast<float>(y) + 0.5f;
    const std::uint8_t* source = input + pixel * static_cast<std::size_t>(config.channels);
    std::uint8_t* destination = output + pixel * static_cast<std::size_t>(config.channels);
    for (int channel = 0; channel < config.channels; ++channel) {
      float value = static_cast<float>(source[channel]);
      for (int index = 0; index < config.snowflake_count; ++index) {
        const Snowflake snowflake = get_snowflake(config, index);
        const float dx = px - snowflake.x;
        const float dy = py - snowflake.y;
        if (dx * dx + dy * dy <= snowflake.radius * snowflake.radius) {
          const float blend = std::max(0.0f, std::min(1.0f, config.alpha * snowflake.alpha));
          value = (1.0f - blend) * value + blend * 255.0f;
        }
      }
      destination[channel] = static_cast<std::uint8_t>(std::max(0.0f, std::min(255.0f,
          std::floor(value + 0.5f))));
    }
  }
}

namespace {

void validate_snow_stamp(const SnowStampConfig& config) {
  if (config.width <= 0 || config.height <= 0 || config.channels <= 0 ||
      config.stamp_width <= 0 || config.stamp_height <= 0 ||
      (config.stamp_channels != 1 && config.stamp_channels != config.channels))
    throw std::invalid_argument("invalid SnowStamp dimensions or channels");
  if (config.placement_count < 0 || !std::isfinite(config.alpha) ||
      config.alpha < 0.0f || config.alpha > 1.0f)
    throw std::invalid_argument("invalid SnowStamp alpha or placement count");
  if (config.placement_count > 0 && !config.placements)
    throw std::invalid_argument("SnowStamp placements are required");
  for (int i = 0; i < config.placement_count; ++i) {
    if (!std::isfinite(config.placements[i].alpha) ||
        config.placements[i].alpha < 0.0f || config.placements[i].alpha > 1.0f)
      throw std::invalid_argument("invalid SnowStamp placement alpha");
  }
}

}  // namespace

void snow_stamp_u8(const std::uint8_t* input, std::uint8_t* output,
                   const SnowStampConfig& config, cudaStream_t) {
  validate_snow_stamp(config);
  if (!input || !output) throw std::invalid_argument("SnowStamp input and output are required");
  const std::size_t pixels = static_cast<std::size_t>(config.width) * config.height;
  const std::size_t channels = static_cast<std::size_t>(config.channels);
  if (input != output)
    std::copy(input, input + pixels * channels, output);
  for (int placement_index = 0; placement_index < config.placement_count; ++placement_index) {
    const SnowStampPlacement& placement = config.placements[placement_index];
    for (int sy = 0; sy < config.stamp_height; ++sy) {
      const int y = placement.y + sy;
      if (y < 0 || y >= config.height) continue;
      for (int sx = 0; sx < config.stamp_width; ++sx) {
        const int x = placement.x + sx;
        if (x < 0 || x >= config.width) continue;
        const std::size_t stamp_pixel = static_cast<std::size_t>(sy) * config.stamp_width + sx;
        const float mask_alpha = config.stamp_mask ?
            static_cast<float>(config.stamp_mask[stamp_pixel]) / 255.0f : 1.0f;
        const float blend = std::max(0.0f, std::min(1.0f,
            config.alpha * placement.alpha * mask_alpha));
        if (blend == 0.0f) continue;
        const std::size_t pixel = static_cast<std::size_t>(y) * config.width + x;
        std::uint8_t* destination = output + pixel * channels;
        for (int channel = 0; channel < config.channels; ++channel) {
          std::uint8_t fill = config.fill;
          if (config.stamp_image) {
            const int stamp_channel = config.stamp_channels == 1 ? 0 : channel;
            fill = config.stamp_image[stamp_pixel * config.stamp_channels + stamp_channel];
          }
          const float value = (1.0f - blend) * static_cast<float>(destination[channel]) +
                              blend * static_cast<float>(fill);
          destination[channel] = static_cast<std::uint8_t>(std::max(0.0f, std::min(255.0f,
              std::floor(value + 0.5f))));
        }
      }
    }
  }
}

namespace {

void validate_gravel(const RandomGravelConfig& config) {
  if (config.width <= 0 || config.height <= 0 || config.channels <= 0)
    throw std::invalid_argument("RandomGravel dimensions and channels must be positive");
  if (config.particle_count < 0)
    throw std::invalid_argument("RandomGravel particle count must be nonnegative");
  if (!std::isfinite(config.alpha) || !std::isfinite(config.radius_min) ||
      !std::isfinite(config.radius_max) || config.alpha < 0.0f || config.alpha > 1.0f ||
      config.radius_min <= 0.0f || config.radius_min > config.radius_max)
    throw std::invalid_argument("invalid RandomGravel parameter range");
  if (config.particle_count > 0 && config.particles) {
    for (int i = 0; i < config.particle_count; ++i) {
      const GravelParticle& particle = config.particles[i];
      if (!std::isfinite(particle.x) || !std::isfinite(particle.y) ||
          !std::isfinite(particle.radius) || !std::isfinite(particle.alpha) ||
          particle.radius <= 0.0f || particle.alpha < 0.0f || particle.alpha > 1.0f)
        throw std::invalid_argument("invalid explicit RandomGravel particle");
    }
  }
}

GravelParticle generated_gravel_particle(const RandomGravelConfig& config, int index) {
  const std::uint64_t base = static_cast<std::uint64_t>(index) * 4ULL;
  return {unit_random(config.seed, base) * static_cast<float>(config.width),
          unit_random(config.seed, base + 1ULL) * static_cast<float>(config.height),
          config.radius_min + (config.radius_max - config.radius_min) *
              unit_random(config.seed, base + 2ULL),
          1.0f, config.fill};
}

GravelParticle get_gravel_particle(const RandomGravelConfig& config, int index) {
  return config.particles ? config.particles[index] : generated_gravel_particle(config, index);
}

}  // namespace

void make_random_gravel_particles(GravelParticle* output, const RandomGravelConfig& config) {
  validate_gravel(config);
  if (config.particle_count > 0 && !output)
    throw std::invalid_argument("RandomGravel particle output is null");
  for (int i = 0; i < config.particle_count; ++i)
    output[i] = config.particles ? config.particles[i] : generated_gravel_particle(config, i);
}

void random_gravel_u8(const std::uint8_t* input, std::uint8_t* output,
                     const RandomGravelConfig& config, cudaStream_t) {
  validate_gravel(config);
  if (!input || !output) throw std::invalid_argument("RandomGravel input and output are required");
  const std::size_t pixels = static_cast<std::size_t>(config.width) * config.height;
  for (std::size_t pixel = 0; pixel < pixels; ++pixel) {
    const int x = static_cast<int>(pixel % static_cast<std::size_t>(config.width));
    const int y = static_cast<int>(pixel / static_cast<std::size_t>(config.width));
    const float px = static_cast<float>(x) + 0.5f;
    const float py = static_cast<float>(y) + 0.5f;
    const std::uint8_t* source = input + pixel * static_cast<std::size_t>(config.channels);
    std::uint8_t* destination = output + pixel * static_cast<std::size_t>(config.channels);
    for (int channel = 0; channel < config.channels; ++channel) {
      float value = static_cast<float>(source[channel]);
      for (int index = 0; index < config.particle_count; ++index) {
        const GravelParticle particle = get_gravel_particle(config, index);
        const float dx = px - particle.x;
        const float dy = py - particle.y;
        if (dx * dx + dy * dy <= particle.radius * particle.radius) {
          const float blend = std::max(0.0f, std::min(1.0f, config.alpha * particle.alpha));
          value = (1.0f - blend) * value + blend * static_cast<float>(particle.fill);
        }
      }
      destination[channel] = static_cast<std::uint8_t>(std::max(0.0f, std::min(255.0f,
          std::floor(value + 0.5f))));
    }
  }
}

namespace {

void validate_fog(const RandomFogConfig& config) {
  if (config.width <= 0 || config.height <= 0 || config.channels <= 0)
    throw std::invalid_argument("RandomFog dimensions and channels must be positive");
  if (!std::isfinite(config.density) || !std::isfinite(config.opacity) ||
      config.density < 0.0f || config.density > 1.0f ||
      config.opacity < 0.0f || config.opacity > 1.0f)
    throw std::invalid_argument("RandomFog density and opacity must be in [0,1]");
  if (config.field) {
    const std::size_t pixels = static_cast<std::size_t>(config.width) * config.height;
    for (std::size_t pixel = 0; pixel < pixels; ++pixel) {
      if (!std::isfinite(config.field[pixel]) || config.field[pixel] < 0.0f ||
          config.field[pixel] > 1.0f)
        throw std::invalid_argument("RandomFog field values must be in [0,1]");
    }
  }
}

float fog_field_value(const RandomFogConfig& config, std::size_t pixel) {
  return config.field ? config.field[pixel] :
      unit_random(config.seed, static_cast<std::uint64_t>(pixel));
}

}  // namespace

void make_random_fog_field(float* output, const RandomFogConfig& config) {
  RandomFogConfig generated = config;
  generated.field = nullptr;
  validate_fog(generated);
  const std::size_t pixels = static_cast<std::size_t>(config.width) * config.height;
  if (pixels > 0 && !output)
    throw std::invalid_argument("RandomFog field output is null");
  for (std::size_t pixel = 0; pixel < pixels; ++pixel)
    output[pixel] = fog_field_value(generated, pixel);
}

void random_fog_u8(const std::uint8_t* input, std::uint8_t* output,
                   const RandomFogConfig& config, cudaStream_t) {
  validate_fog(config);
  if (!input || !output) throw std::invalid_argument("RandomFog input and output are required");
  const std::size_t pixels = static_cast<std::size_t>(config.width) * config.height;
  for (std::size_t pixel = 0; pixel < pixels; ++pixel) {
    const float field = fog_field_value(config, pixel);
    const float blend = std::max(0.0f, std::min(1.0f, config.opacity * config.density * field));
    const std::uint8_t* source = input + pixel * static_cast<std::size_t>(config.channels);
    std::uint8_t* destination = output + pixel * static_cast<std::size_t>(config.channels);
    for (int channel = 0; channel < config.channels; ++channel) {
      const float value = (1.0f - blend) * static_cast<float>(source[channel]) + blend * 255.0f;
      destination[channel] = static_cast<std::uint8_t>(std::max(0.0f, std::min(255.0f,
          std::floor(value + 0.5f))));
    }
  }
}

namespace {

void validate_sun_flare(const RandomSunFlareConfig& config) {
  if (config.width <= 0 || config.height <= 0 || config.channels <= 0 || config.ray_count < 0)
    throw std::invalid_argument("invalid RandomSunFlare dimensions or ray count");
  const float values[] = {config.source.x, config.source.y, config.source.radius,
                          config.source.alpha, config.opacity, config.ray_length_min,
                          config.ray_length_max, config.ray_angle_min, config.ray_angle_max,
                          config.ray_width_min, config.ray_width_max};
  for (float value : values)
    if (!std::isfinite(value)) throw std::invalid_argument("RandomSunFlare parameters must be finite");
  if (config.source.radius < 0.0f || config.source.alpha < 0.0f || config.source.alpha > 1.0f ||
      config.opacity < 0.0f || config.opacity > 1.0f || config.ray_length_min < 0.0f ||
      config.ray_length_min > config.ray_length_max || config.ray_width_min <= 0.0f ||
      config.ray_width_min > config.ray_width_max || config.ray_angle_min > config.ray_angle_max)
    throw std::invalid_argument("invalid RandomSunFlare parameter range");
  if (config.ray_count > 0 && config.rays) {
    for (int i = 0; i < config.ray_count; ++i) {
      const SunFlareRay& ray = config.rays[i];
      if (!std::isfinite(ray.angle) || !std::isfinite(ray.length) || !std::isfinite(ray.width) ||
          !std::isfinite(ray.alpha) || ray.length < 0.0f || ray.width <= 0.0f ||
          ray.alpha < 0.0f || ray.alpha > 1.0f)
        throw std::invalid_argument("invalid explicit RandomSunFlare ray");
    }
  }
}

SunFlareRay generated_sun_flare_ray(const RandomSunFlareConfig& config, int index) {
  const std::uint64_t base = static_cast<std::uint64_t>(index) * 4ULL;
  const float angle = config.ray_angle_min +
      (config.ray_angle_max - config.ray_angle_min) * unit_random(config.seed, base);
  const float length = config.ray_length_min +
      (config.ray_length_max - config.ray_length_min) * unit_random(config.seed, base + 1ULL);
  const float width = config.ray_width_min +
      (config.ray_width_max - config.ray_width_min) * unit_random(config.seed, base + 2ULL);
  return {angle, length, width, 1.0f};
}

SunFlareRay get_sun_flare_ray(const RandomSunFlareConfig& config, int index) {
  return config.rays ? config.rays[index] : generated_sun_flare_ray(config, index);
}

float blend_sun_flare(float value, float alpha) {
  const float blend = std::max(0.0f, std::min(1.0f, alpha));
  return (1.0f - blend) * value + blend * 255.0f;
}

}  // namespace

void make_random_sun_flare_rays(SunFlareRay* output, const RandomSunFlareConfig& config) {
  validate_sun_flare(config);
  if (config.ray_count > 0 && !output)
    throw std::invalid_argument("RandomSunFlare ray output is null");
  for (int i = 0; i < config.ray_count; ++i)
    output[i] = config.rays ? config.rays[i] : generated_sun_flare_ray(config, i);
}

void random_sun_flare_u8(const std::uint8_t* input, std::uint8_t* output,
                         const RandomSunFlareConfig& config, cudaStream_t) {
  validate_sun_flare(config);
  if (!input || !output) throw std::invalid_argument("RandomSunFlare input and output are required");
  const std::size_t pixels = static_cast<std::size_t>(config.width) * config.height;
  const float source_blend = config.opacity * config.source.alpha;
  for (std::size_t pixel = 0; pixel < pixels; ++pixel) {
    const int x = static_cast<int>(pixel % static_cast<std::size_t>(config.width));
    const int y = static_cast<int>(pixel / static_cast<std::size_t>(config.width));
    const float px = static_cast<float>(x) + 0.5f;
    const float py = static_cast<float>(y) + 0.5f;
    const std::uint8_t* source = input + pixel * static_cast<std::size_t>(config.channels);
    std::uint8_t* destination = output + pixel * static_cast<std::size_t>(config.channels);
    for (int channel = 0; channel < config.channels; ++channel) {
      float value = static_cast<float>(source[channel]);
      const float sx = px - config.source.x;
      const float sy = py - config.source.y;
      if (sx * sx + sy * sy <= config.source.radius * config.source.radius)
        value = blend_sun_flare(value, source_blend);
      for (int index = 0; index < config.ray_count; ++index) {
        const SunFlareRay ray = get_sun_flare_ray(config, index);
        const float angle = ray.angle * 0.017453292519943295769f;
        RainStreak segment{config.source.x, config.source.y,
                           config.source.x + ray.length * std::cos(angle),
                           config.source.y + ray.length * std::sin(angle),
                           ray.width, ray.alpha};
        if (distance_squared(px, py, segment) <= 0.25f * ray.width * ray.width)
          value = blend_sun_flare(value, config.opacity * ray.alpha);
      }
      destination[channel] = static_cast<std::uint8_t>(std::max(0.0f, std::min(255.0f,
          std::floor(value + 0.5f))));
    }
  }
}

namespace {

void validate_random_shadow(const RandomShadowConfig& config) {
  if (config.width <= 0 || config.height <= 0 || config.channels <= 0 ||
      config.polygon_count < 0 || config.rectangle_count < 0)
    throw std::invalid_argument("invalid RandomShadow dimensions or mask counts");
  if (!std::isfinite(config.opacity) || config.opacity < 0.0f || config.opacity > 1.0f)
    throw std::invalid_argument("RandomShadow opacity must be finite and in [0,1]");
  if (config.polygon_count > 0 && !config.polygons)
    throw std::invalid_argument("RandomShadow polygon list is null");
  if (config.rectangle_count > 0 && !config.rectangles)
    throw std::invalid_argument("RandomShadow rectangle list is null");
  for (int i = 0; i < config.polygon_count; ++i) {
    const ShadowPolygon& polygon = config.polygons[i];
    if (polygon.point_count < 3 || !polygon.points || !std::isfinite(polygon.alpha) ||
        polygon.alpha < 0.0f || polygon.alpha > 1.0f)
      throw std::invalid_argument("invalid RandomShadow polygon");
    for (int point = 0; point < polygon.point_count; ++point)
      if (!std::isfinite(polygon.points[point].x) || !std::isfinite(polygon.points[point].y))
        throw std::invalid_argument("RandomShadow polygon points must be finite");
  }
  for (int i = 0; i < config.rectangle_count; ++i) {
    const ShadowRectangle& rectangle = config.rectangles[i];
    if (!std::isfinite(rectangle.x0) || !std::isfinite(rectangle.y0) ||
        !std::isfinite(rectangle.x1) || !std::isfinite(rectangle.y1) ||
        !std::isfinite(rectangle.alpha) || rectangle.alpha < 0.0f || rectangle.alpha > 1.0f)
      throw std::invalid_argument("invalid RandomShadow rectangle");
  }
}

bool point_in_shadow_polygon(float px, float py, const ShadowPolygon& polygon) {
  bool inside = false;
  for (int i = 0, previous = polygon.point_count - 1; i < polygon.point_count;
       previous = i++) {
    const ShadowPoint& a = polygon.points[previous];
    const ShadowPoint& b = polygon.points[i];
    const float cross = (px - a.x) * (b.y - a.y) - (py - a.y) * (b.x - a.x);
    const float min_x = std::min(a.x, b.x), max_x = std::max(a.x, b.x);
    const float min_y = std::min(a.y, b.y), max_y = std::max(a.y, b.y);
    if (std::abs(cross) <= 1.0e-6f && px >= min_x && px <= max_x && py >= min_y && py <= max_y)
      return true;
    if ((a.y > py) != (b.y > py) && px < (b.x - a.x) * (py - a.y) / (b.y - a.y) + a.x)
      inside = !inside;
  }
  return inside;
}

bool point_in_shadow_rectangle(float px, float py, const ShadowRectangle& rectangle) {
  return px >= std::min(rectangle.x0, rectangle.x1) &&
         px <= std::max(rectangle.x0, rectangle.x1) &&
         py >= std::min(rectangle.y0, rectangle.y1) &&
         py <= std::max(rectangle.y0, rectangle.y1);
}

float blend_shadow(float value, float opacity, float local_alpha, std::uint8_t fill) {
  const float alpha = std::max(0.0f, std::min(1.0f, opacity * local_alpha));
  return (1.0f - alpha) * value + alpha * static_cast<float>(fill);
}

}  // namespace

void random_shadow_u8(const std::uint8_t* input, std::uint8_t* output,
                     const RandomShadowConfig& config, cudaStream_t) {
  validate_random_shadow(config);
  if (!input || !output) throw std::invalid_argument("RandomShadow input and output are required");
  const std::size_t pixels = static_cast<std::size_t>(config.width) * config.height;
  for (std::size_t pixel = 0; pixel < pixels; ++pixel) {
    const int x = static_cast<int>(pixel % static_cast<std::size_t>(config.width));
    const int y = static_cast<int>(pixel / static_cast<std::size_t>(config.width));
    const float px = static_cast<float>(x) + 0.5f;
    const float py = static_cast<float>(y) + 0.5f;
    std::uint8_t* destination = output + pixel * static_cast<std::size_t>(config.channels);
    const std::uint8_t* source = input + pixel * static_cast<std::size_t>(config.channels);
    for (int channel = 0; channel < config.channels; ++channel) {
      float value = static_cast<float>(source[channel]);
      for (int i = 0; i < config.polygon_count; ++i)
        if (point_in_shadow_polygon(px, py, config.polygons[i]))
          value = blend_shadow(value, config.opacity, config.polygons[i].alpha, config.fill);
      for (int i = 0; i < config.rectangle_count; ++i)
        if (point_in_shadow_rectangle(px, py, config.rectangles[i]))
          value = blend_shadow(value, config.opacity, config.rectangles[i].alpha, config.fill);
      destination[channel] = static_cast<std::uint8_t>(std::max(0.0f, std::min(255.0f,
          std::floor(value + 0.5f))));
    }
  }
}

namespace {

void validate_spatter(const RandomSpatterConfig& config) {
  if (config.width <= 0 || config.height <= 0 || config.channels <= 0 ||
      config.droplet_count < 0)
    throw std::invalid_argument("invalid RandomSpatter dimensions or droplet count");
  if (!std::isfinite(config.alpha) || !std::isfinite(config.radius_min) ||
      !std::isfinite(config.radius_max) || config.alpha < 0.0f || config.alpha > 1.0f ||
      config.radius_min <= 0.0f || config.radius_min > config.radius_max)
    throw std::invalid_argument("invalid RandomSpatter parameter range");
  if (config.droplet_count > 0 && config.droplets) {
    for (int i = 0; i < config.droplet_count; ++i) {
      const SpatterDroplet& droplet = config.droplets[i];
      if (!std::isfinite(droplet.x) || !std::isfinite(droplet.y) ||
          !std::isfinite(droplet.radius) || !std::isfinite(droplet.alpha) ||
          droplet.radius <= 0.0f || droplet.alpha < 0.0f || droplet.alpha > 1.0f)
        throw std::invalid_argument("invalid explicit RandomSpatter droplet");
    }
  }
}

SpatterDroplet generated_spatter_droplet(const RandomSpatterConfig& config, int index) {
  const std::uint64_t base = static_cast<std::uint64_t>(index) * 4ULL;
  return {unit_random(config.seed, base) * static_cast<float>(config.width),
          unit_random(config.seed, base + 1ULL) * static_cast<float>(config.height),
          config.radius_min + (config.radius_max - config.radius_min) *
              unit_random(config.seed, base + 2ULL),
          config.red, config.green, config.blue, 1.0f};
}

SpatterDroplet get_spatter_droplet(const RandomSpatterConfig& config, int index) {
  return config.droplets ? config.droplets[index] : generated_spatter_droplet(config, index);
}

float blend_spatter(float value, float alpha, std::uint8_t color) {
  const float blend = std::max(0.0f, std::min(1.0f, alpha));
  return (1.0f - blend) * value + blend * static_cast<float>(color);
}

}  // namespace

void make_random_spatter_droplets(SpatterDroplet* output,
                                  const RandomSpatterConfig& config) {
  validate_spatter(config);
  if (config.droplet_count > 0 && !output)
    throw std::invalid_argument("RandomSpatter droplet output is null");
  for (int i = 0; i < config.droplet_count; ++i)
    output[i] = config.droplets ? config.droplets[i] : generated_spatter_droplet(config, i);
}

void random_spatter_u8(const std::uint8_t* input, std::uint8_t* output,
                       const RandomSpatterConfig& config, cudaStream_t) {
  validate_spatter(config);
  if (!input || !output) throw std::invalid_argument("RandomSpatter input and output are required");
  const std::size_t pixels = static_cast<std::size_t>(config.width) * config.height;
  for (std::size_t pixel = 0; pixel < pixels; ++pixel) {
    const int x = static_cast<int>(pixel % static_cast<std::size_t>(config.width));
    const int y = static_cast<int>(pixel / static_cast<std::size_t>(config.width));
    const float px = static_cast<float>(x) + 0.5f;
    const float py = static_cast<float>(y) + 0.5f;
    const std::uint8_t* source = input + pixel * static_cast<std::size_t>(config.channels);
    std::uint8_t* destination = output + pixel * static_cast<std::size_t>(config.channels);
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
      destination[channel] = static_cast<std::uint8_t>(std::max(0.0f, std::min(255.0f,
          std::floor(value + 0.5f))));
    }
  }
}

}  // namespace augmatch
