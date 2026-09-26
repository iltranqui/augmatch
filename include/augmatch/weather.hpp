#pragma once

#include <cstdint>
#include "augmatch/imgcorruptlike.hpp" // Fog is shared with the existing veil primitive.

#ifndef AUGMATCH_HAS_CUDA
#define AUGMATCH_HAS_CUDA 0
#endif
#if AUGMATCH_HAS_CUDA
#include <cuda_runtime_api.h>
#else
using cudaStream_t = void*;
#endif

namespace augmatch {

// A rain streak is an explicit line segment in image-pixel coordinates.  The
// segment includes its full width and local opacity; no hidden random state is
// used when a streak list is supplied.  Pixel centres are sampled with a hard
// capsule test (distance <= width / 2).
struct RainStreak {
  float x0 = 0.0f;
  float y0 = 0.0f;
  float x1 = 0.0f;
  float y1 = 0.0f;
  float width = 1.0f;
  float alpha = 1.0f;
};

// RandomRain overlays white streaks on interleaved HWC uint8 data.  If
// streaks is non-null, it points to streak_count caller-owned RainStreak
// records (host memory for CPU, device memory for CUDA), and seed/placement
// ranges are ignored.  Otherwise streak_count segments are generated
// deterministically from seed.  Generated x/y positions are uniform in the
// image, lengths and widths are in pixels, and angles are in degrees measured
// from +x.  The generated records are exactly the records accepted by the
// explicit path; make_random_rain_streaks can materialize them for logging or
// cross-backend replay.
//
// Blend contract, applied in list order for every covered pixel and channel:
//   out = round((1 - alpha * streak.alpha) * current +
//               (alpha * streak.alpha) * 255)
// with each effective alpha clamped to [0,1], and final values clipped to
// [0,255].  Thus alpha=0 is identity, overlaps are sequential (not a max),
// and channels beyond RGB receive the same white blend.
struct RandomRainConfig {
  int width = 0;
  int height = 0;
  int channels = 0;
  const RainStreak* streaks = nullptr;
  int streak_count = 0;
  float alpha = 0.5f;
  float length_min = 8.0f;
  float length_max = 24.0f;
  float angle_min = 75.0f;
  float angle_max = 105.0f;
  float streak_width = 1.0f;
  std::uint64_t seed = 0;
};

// Materialize deterministic placement.  The output buffer must contain at
// least config.streak_count records.  This is also the canonical way to turn
// stochastic-looking placement into an explicit reproducible parameter set.
void make_random_rain_streaks(RainStreak* output, const RandomRainConfig& config);

void random_rain_u8(const std::uint8_t* input, std::uint8_t* output,
                    const RandomRainConfig& config,
                    cudaStream_t stream = nullptr);

// A snowflake is an explicit filled disc in image-pixel coordinates. Pixel
// centres within radius receive the white alpha blend; records are processed
// in list order and contain no hidden random state.
struct Snowflake {
  float x = 0.0f;
  float y = 0.0f;
  float radius = 1.0f;
  float alpha = 1.0f;
};

// RandomSnow overlays white circular flakes on interleaved HWC uint8 data.
// A non-null snowflakes list is caller-owned (host memory for CPU, device
// memory for CUDA) and makes placement explicit. Otherwise snowflake_count
// records are generated from seed using deterministic SplitMix64 placement:
// each record consumes four values for x, y, radius, and a reserved stream
// slot. Positions are uniform in [0,width) x [0,height), and radius is sampled
// uniformly from radius_min..radius_max. The generated records are exactly the
// records accepted by the explicit path and can be materialized with
// make_random_snowflakes.
//
// For every covered pixel and channel, in list order:
//   out = round((1 - a) * current + a * 255),
// where a=clamp(config.alpha * snowflake.alpha,0,1). Overlaps are sequential,
// alpha=0 is identity, values are clipped to [0,255], and all channels
// including channels beyond RGB receive the same white blend.
struct RandomSnowConfig {
  int width = 0;
  int height = 0;
  int channels = 0;
  const Snowflake* snowflakes = nullptr;
  int snowflake_count = 0;
  float alpha = 0.5f;
  float radius_min = 1.0f;
  float radius_max = 3.0f;
  std::uint64_t seed = 0;
};

void make_random_snowflakes(Snowflake* output, const RandomSnowConfig& config);

void random_snow_u8(const std::uint8_t* input, std::uint8_t* output,
                    const RandomSnowConfig& config,
                    cudaStream_t stream = nullptr);

// SnowStamp uses one explicit stamp image/mask and an ordered list of integer
// top-left placements. The image is HWC uint8 with stamp_channels equal to one
// (broadcast to every output channel) or config.channels. A null image uses
// config.fill; a null mask is fully opaque. The mask is one uint8 value per
// stamp pixel, and contributes mask/255 to opacity. Placement arrays and
// image/mask buffers are host-owned for CPU and device-owned for CUDA.
//
// For each placement and covered stamp pixel, in list order, the target is
// composited as out=round((1-a)*current+a*fill), where
// a=clamp(config.alpha*placement.alpha*(mask/255),0,1). The fill is either
// the corresponding stamp image value or config.fill. Stamps are clipped at
// image boundaries, overlaps are sequential, and all output channels are
// processed (including channels beyond RGB). Coordinates are integer pixel
// indices, so no interpolation or hidden random state is involved. The CLI
// record format is two little-endian int32 coordinates followed by one
// little-endian float32 alpha (12 bytes).
struct SnowStampPlacement {
  int x = 0;
  int y = 0;
  float alpha = 1.0f;
};
using SnowStamp = SnowStampPlacement;
using SnowStampRecord = SnowStampPlacement;

struct SnowStampConfig {
  int width = 0;
  int height = 0;
  int channels = 0;
  int stamp_width = 0;
  int stamp_height = 0;
  int stamp_channels = 1;
  const std::uint8_t* stamp_image = nullptr;
  const std::uint8_t* stamp_mask = nullptr;
  const SnowStampPlacement* placements = nullptr;
  int placement_count = 0;
  float alpha = 0.5f;
  std::uint8_t fill = 255;
};

void snow_stamp_u8(const std::uint8_t* input, std::uint8_t* output,
                   const SnowStampConfig& config,
                   cudaStream_t stream = nullptr);

// Catalog spelling aliases retain the same ABI and deterministic contract.
inline void snowstamp_u8(const std::uint8_t* input, std::uint8_t* output,
                         const SnowStampConfig& config,
                         cudaStream_t stream = nullptr) {
  snow_stamp_u8(input, output, config, stream);
}

// A gravel particle is an explicit filled disc in image-pixel coordinates.
// Its fill is a scalar uint8 value applied identically to every image channel.
// Pixel centres inside radius are covered; records are processed in list order.
struct GravelParticle {
  float x = 0.0f;
  float y = 0.0f;
  float radius = 1.0f;
  float alpha = 1.0f;
  std::uint8_t fill = 128;
};

// RandomGravel overlays filled gravel discs on interleaved HWC uint8 data.
// A non-null particle list is caller-owned (host memory for CPU, device memory
// for CUDA), and makes placement, radius, alpha, and fill explicit. Otherwise
// particle_count records are generated from SplitMix64(seed + 4*index + offset):
// x and y are uniform in [0,width) and [0,height), radius is uniform in
// radius_min..radius_max, and generated records use config.fill with alpha 1.
// The generated records are materialized by make_random_gravel_particles.
// For each covered pixel and channel, in list order:
//   out = round((1-a) * current + a * particle.fill),
// where a=clamp(config.alpha * particle.alpha,0,1). Values are clipped to
// [0,255], overlaps are sequential, zero alpha and an empty list are identity,
// and all channels receive the same scalar fill. The particle radius must be
// positive; config radius bounds are finite with 0 < min <= max.
struct RandomGravelConfig {
  int width = 0;
  int height = 0;
  int channels = 0;
  const GravelParticle* particles = nullptr;
  int particle_count = 0;
  float alpha = 0.5f;
  float radius_min = 1.0f;
  float radius_max = 3.0f;
  std::uint8_t fill = 128;
  std::uint64_t seed = 0;
};

void make_random_gravel_particles(GravelParticle* output,
                                  const RandomGravelConfig& config);

void random_gravel_u8(const std::uint8_t* input, std::uint8_t* output,
                      const RandomGravelConfig& config,
                      cudaStream_t stream = nullptr);

// RandomFog applies a depth-independent white veil to interleaved HWC uint8
// data.  A non-null field points to width*height caller-owned floats in [0,1]
// (host memory for CPU, device memory for CUDA); field values are the local fog
// density.  When field is null, the same deterministic SplitMix64 field is
// generated from seed by both backends.  The generated field can be materialized
// with make_random_fog_field for logging or exact CPU/CUDA replay.
//
// For every pixel and channel, the single depth-independent blend is:
//   out = round((1 - a) * input + a * 255),
// where a=clamp(config.opacity * config.density * field[pixel],0,1).
// Alpha zero, density zero, opacity zero, and an all-zero field are identity;
// values are clipped to [0,255], and every channel receives the same veil.
struct RandomFogConfig {
  int width = 0;
  int height = 0;
  int channels = 0;
  float density = 1.0f;
  float opacity = 0.5f;
  const float* field = nullptr;
  std::uint64_t seed = 0;
};

void make_random_fog_field(float* output, const RandomFogConfig& config);

void random_fog_u8(const std::uint8_t* input, std::uint8_t* output,
                   const RandomFogConfig& config,
                   cudaStream_t stream = nullptr);

// A sun-flare source is an explicit disc in image-pixel coordinates. Rays use
// polar coordinates relative to this source; angle is in degrees from +x.
struct SunFlareSource {
  float x = 0.0f;
  float y = 0.0f;
  float radius = 8.0f;
  float alpha = 1.0f;
};

struct SunFlareRay {
  float angle = 0.0f;
  float length = 16.0f;
  float width = 1.0f;
  float alpha = 1.0f;
};

// RandomSunFlare overlays a white source disc followed by white rays on HWC
// uint8 data. The source is always explicit in config.source. A non-null rays
// list is caller-owned (host memory for CPU, device memory for CUDA); a null
// list generates ray_count records deterministically from seed. Generated rays
// use SplitMix64(seed + 4*index + offset), with uniform angle, length, and
// width ranges. make_random_sun_flare_rays materializes that list.
//
// Pixel centres use a hard disc/capsule test. The source is composited first,
// then rays in list order. Every covered channel uses
//   out = round((1 - a) * current + a * 255),
// where a=clamp(config.opacity * local_alpha,0,1); values are clipped to
// [0,255]. Thus opacity=0, radius=0 with no covered ray, and ray_count=0
// (apart from the source) are deterministic, well-defined cases.
struct RandomSunFlareConfig {
  int width = 0;
  int height = 0;
  int channels = 0;
  SunFlareSource source{};
  const SunFlareRay* rays = nullptr;
  int ray_count = 0;
  float opacity = 0.5f;
  float ray_length_min = 16.0f;
  float ray_length_max = 48.0f;
  float ray_angle_min = 0.0f;
  float ray_angle_max = 360.0f;
  float ray_width_min = 1.0f;
  float ray_width_max = 2.0f;
  std::uint64_t seed = 0;
};

void make_random_sun_flare_rays(SunFlareRay* output,
                                const RandomSunFlareConfig& config);

void random_sun_flare_u8(const std::uint8_t* input, std::uint8_t* output,
                         const RandomSunFlareConfig& config,
                         cudaStream_t stream = nullptr);

// A shadow polygon is an explicit filled polygon in image-pixel coordinates.
// Pixel centres on the boundary are included by the rasterizer. The points
// array is borrowed host memory for CPU calls and device memory for CUDA calls.
struct ShadowPoint {
  float x = 0.0f;
  float y = 0.0f;
};

struct ShadowPolygon {
  const ShadowPoint* points = nullptr;
  int point_count = 0;
  float alpha = 1.0f;
};

// A shadow rectangle uses its normalized inclusive image-pixel bounds. This
// record is deliberately separate from ShadowPolygon so rectangular masks do
// not require a temporary point allocation.
struct ShadowRectangle {
  float x0 = 0.0f;
  float y0 = 0.0f;
  float x1 = 0.0f;
  float y1 = 0.0f;
  float alpha = 1.0f;
};

// RandomShadow applies explicit polygon masks first, then explicit rectangle
// masks, in list order. No implicit random state is used: the seed field is
// reserved for API symmetry and does not affect explicit masks. For every
// covered pixel and channel, records are composited sequentially as
//   out = round((1 - a) * current + a * fill),
// where a=clamp(config.opacity * record.alpha,0,1), fill is the same scalar
// uint8 value for every channel, and the result is clipped to [0,255]. Pixel
// centres are sampled at (x+0.5,y+0.5); alpha zero and empty mask lists are
// identity. CPU arrays and polygon point arrays are host-owned; CUDA arrays
// and point arrays are device-owned.
struct RandomShadowConfig {
  int width = 0;
  int height = 0;
  int channels = 0;
  const ShadowPolygon* polygons = nullptr;
  int polygon_count = 0;
  const ShadowRectangle* rectangles = nullptr;
  int rectangle_count = 0;
  float opacity = 0.5f;
  std::uint8_t fill = 0;
  std::uint64_t seed = 0;
};

void random_shadow_u8(const std::uint8_t* input, std::uint8_t* output,
                     const RandomShadowConfig& config,
                     cudaStream_t stream = nullptr);

// A spatter droplet is an explicit filled disc.  Its RGB color is blended into
// the first three channels; channels beyond RGB are copied unchanged.  Pixel
// centres inside radius are covered and records are processed in list order.
struct SpatterDroplet {
  float x = 0.0f;
  float y = 0.0f;
  float radius = 1.0f;
  std::uint8_t red = 255;
  std::uint8_t green = 255;
  std::uint8_t blue = 255;
  float alpha = 1.0f;
};

// Spatter overlays colored droplets on interleaved HWC uint8 data.  A non-null
// droplet list is caller-owned (host memory for CPU, device memory for CUDA)
// and makes placement, radius, color, and local alpha explicit.  Otherwise
// droplet_count records are generated deterministically from SplitMix64 using
// four values per record: x, y, radius, and a reserved slot.  Generated
// droplets use the configured RGB color and local alpha one.  The generated
// records are materialized by make_random_spatter_droplets.
//
// For each covered pixel, each of the first min(channels, 3) channels is
// composited in list order as out=round((1-a)*current+a*color), where
// a=clamp(config.alpha*droplet.alpha,0,1).  Results are clipped to [0,255],
// overlaps are sequential, and alpha zero or an empty list is identity.
// The CLI binary record layout is three little-endian float32 values, three
// uint8 color values, one padding byte, and one float32 alpha (20 bytes).
struct RandomSpatterConfig {
  int width = 0;
  int height = 0;
  int channels = 0;
  const SpatterDroplet* droplets = nullptr;
  int droplet_count = 0;
  float alpha = 0.5f;
  float radius_min = 1.0f;
  float radius_max = 3.0f;
  std::uint8_t red = 255;
  std::uint8_t green = 255;
  std::uint8_t blue = 255;
  std::uint64_t seed = 0;
};

void make_random_spatter_droplets(SpatterDroplet* output,
                                  const RandomSpatterConfig& config);

void random_spatter_u8(const std::uint8_t* input, std::uint8_t* output,
                       const RandomSpatterConfig& config,
                       cudaStream_t stream = nullptr);

// Short aliases for callers using the catalog name rather than the seeded
// implementation name.  They intentionally preserve the same ABI and rules.
using SpatterConfig = RandomSpatterConfig;
using SpatterDrop = SpatterDroplet;
inline void make_spatter_droplets(SpatterDroplet* output,
                                  const SpatterConfig& config) {
  make_random_spatter_droplets(output, config);
}
inline void spatter_u8(const std::uint8_t* input, std::uint8_t* output,
                       const SpatterConfig& config, cudaStream_t stream = nullptr) {
  random_spatter_u8(input, output, config, stream);
}

// Catalog weather APIs use contiguous interleaved HWC uint8 buffers. Pointers
// are borrowed: CPU pointers are host memory and CUDA pointers are device
// memory. All operations use sequential white compositing and no hidden state.
struct FastSnowyLandscapeConfig {
  int width = 0;
  int height = 0;
  int channels = 0;
  float snow_point = 0.5f;
  float alpha = 1.0f;
};
void fast_snowy_landscape_u8(const std::uint8_t*, std::uint8_t*,
                             const FastSnowyLandscapeConfig&,
                             cudaStream_t stream = nullptr);

struct CloudsConfig {
  int width = 0;
  int height = 0;
  int channels = 0;
  float alpha = 0.5f;
  float density = 1.0f;
  const float* field = nullptr;
  std::uint64_t seed = 0;
};
void make_cloud_field(float* output, const CloudsConfig& config);
void clouds_u8(const std::uint8_t*, std::uint8_t*, const CloudsConfig&,
               cudaStream_t stream = nullptr);

// A layer is a borrowed HxW map in [0,1]. It is read-only, never resized or
// retained. The blend is a=clamp(opacity*layer[pixel],0,1) for every channel.
struct WeatherLayerConfig {
  int width = 0;
  int height = 0;
  int channels = 0;
  const float* layer = nullptr;
  float opacity = 1.0f;
};
using CloudLayerConfig = WeatherLayerConfig;
using SnowflakesLayerConfig = WeatherLayerConfig;
using RainLayerConfig = WeatherLayerConfig;
void cloud_layer_u8(const std::uint8_t*, std::uint8_t*, const CloudLayerConfig&,
                    cudaStream_t stream = nullptr);
void snowflakes_layer_u8(const std::uint8_t*, std::uint8_t*, const SnowflakesLayerConfig&,
                         cudaStream_t stream = nullptr);
void rain_layer_u8(const std::uint8_t*, std::uint8_t*, const RainLayerConfig&,
                   cudaStream_t stream = nullptr);

using SnowflakesConfig = RandomSnowConfig;
inline void make_snowflakes(Snowflake* output, const SnowflakesConfig& config) {
  make_random_snowflakes(output, config);
}
void snowflakes_u8(const std::uint8_t*, std::uint8_t*, const SnowflakesConfig&,
                   cudaStream_t stream = nullptr);

using RainConfig = RandomRainConfig;
inline void make_rain_streaks(RainStreak* output, const RainConfig& config) {
  make_random_rain_streaks(output, config);
}
void rain_u8(const std::uint8_t*, std::uint8_t*, const RainConfig&,
             cudaStream_t stream = nullptr);

}  // namespace augmatch
