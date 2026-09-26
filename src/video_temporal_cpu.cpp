#include "augmatch/noise.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace augmatch { namespace {
void valid(const TemporalBatchConfig& c, const float* in, const float* out) {
  if (c.frames <= 0 || c.height <= 0 || c.width <= 0 || c.channels <= 0 || !in || !out ||
      !std::isfinite(c.temporal_correlation) || c.temporal_correlation < -1.0f || c.temporal_correlation > 1.0f ||
      !std::isfinite(c.clip_min) || !std::isfinite(c.clip_max) || c.clip_max < c.clip_min)
    throw std::invalid_argument("invalid video temporal configuration");
}
void nonnegative(float value, const char* message) {
  if (!std::isfinite(value) || value < 0.0f) throw std::invalid_argument(message);
}
std::uint64_t mix(std::uint64_t x) {
  x += 0x9e3779b97f4a7c15ULL; x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL; return x ^ (x >> 31);
}
float unit(std::uint64_t x) { return static_cast<float>((mix(x) >> 11) * (1.0 / 9007199254740992.0)); }
std::uint64_t coord_key(std::uint64_t seed, int t, int y, int x, int ch, std::uint64_t tag) {
  return seed ^ static_cast<std::uint64_t>(t) * 0x632be59bd9b4e019ULL ^
    static_cast<std::uint64_t>(y) * 0x9e3779b97f4a7c15ULL ^ static_cast<std::uint64_t>(x) * 0x8cb92baa2f2f6f7dULL ^
    static_cast<std::uint64_t>(ch) * 0xbf58476d1ce4e5b9ULL ^ tag;
}
float clip(float x, float lo, float hi) { return std::max(lo, std::min(hi, x)); }
std::size_t frame_size(const TemporalBatchConfig& c) { return static_cast<std::size_t>(c.height) * c.width * c.channels; }
std::size_t pixel_index(const TemporalBatchConfig& c, int y, int x, int ch) {
  return (static_cast<std::size_t>(y) * c.width + x) * c.channels + ch;
}
int clamp_frame(int t, int frames) { return std::max(0, std::min(frames - 1, t)); }
int clamp_coord(int p, int extent) { return std::max(0, std::min(extent - 1, p)); }
int source_frame(int t, int offset, int frames) { return clamp_frame(t + offset, frames); }
bool selected(const std::uint8_t* mask, const int* indices, std::size_t count, int t, int total,
              float probability, std::uint64_t seed, std::uint64_t tag) {
  if (mask && mask[t] != 0) return true;
  if (indices) { for (std::size_t i = 0; i < count; ++i) if (indices[i] == t) return true; }
  if (mask || indices) return false;
  return unit(coord_key(seed, t, 0, 0, 0, tag)) < probability;
}
void check_probability(float value, const char* message) { if (!std::isfinite(value) || value < 0.0f || value > 1.0f) throw std::invalid_argument(message); }
}

void dead_pixel_persistence_f32(const float* in, float* out, const DeadPixelPersistenceConfig& c, cudaStream_t) {
  valid(c, in, out); check_probability(c.probability, "invalid dead-pixel probability");
  const std::size_t plane = frame_size(c), pixels = static_cast<std::size_t>(c.height) * c.width * c.channels;
  if (!std::isfinite(c.dead_value)) throw std::invalid_argument("invalid dead-pixel value");
  for (int t = 0; t < c.frames; ++t) for (int y = 0; y < c.height; ++y) for (int x = 0; x < c.width; ++x) for (int ch = 0; ch < c.channels; ++ch) {
    const std::size_t p = pixel_index(c, y, x, ch); const bool dead = c.mask ? c.mask[p] != 0 : unit(coord_key(c.seed, 0, y, x, ch, 0x444541445f504552ULL)) < c.probability;
    out[static_cast<std::size_t>(t) * plane + p] = clip(dead ? c.dead_value : in[static_cast<std::size_t>(t) * plane + p], c.clip_min, c.clip_max);
  }
  (void)pixels;
}
void hot_pixel_persistence_f32(const float* in, float* out, const HotPixelPersistenceConfig& c, cudaStream_t) {
  valid(c, in, out); check_probability(c.probability, "invalid hot-pixel probability");
  const std::size_t plane = frame_size(c);
  if (!std::isfinite(c.hot_value)) throw std::invalid_argument("invalid hot-pixel value");
  for (int t = 0; t < c.frames; ++t) for (int y = 0; y < c.height; ++y) for (int x = 0; x < c.width; ++x) for (int ch = 0; ch < c.channels; ++ch) {
    const std::size_t p = pixel_index(c, y, x, ch); const bool hot = c.mask ? c.mask[p] != 0 : unit(coord_key(c.seed, 0, y, x, ch, 0x484f545f50455253ULL)) < c.probability;
    out[static_cast<std::size_t>(t) * plane + p] = clip(hot ? c.hot_value : in[static_cast<std::size_t>(t) * plane + p], c.clip_min, c.clip_max);
  }
}
void frame_drops_f32(const float* in, float* out, const FrameDropConfig& c, cudaStream_t) {
  valid(c, in, out); check_probability(c.probability, "invalid frame-drop probability");
  if (!std::isfinite(c.fill_value)) throw std::invalid_argument("invalid frame-drop fill value"); const std::size_t plane = frame_size(c);
  for (int t = 0; t < c.frames; ++t) {
    const bool drop = selected(c.drop_mask, c.drop_indices, c.drop_count, t, c.frames, c.probability, c.seed, 0x44524f505f465241ULL);
    const int src = drop ? (t > 0 ? t - 1 : -1) : t; const std::size_t dst = static_cast<std::size_t>(t) * plane;
    if (src < 0) std::fill(out + dst, out + dst + plane, clip(c.fill_value, c.clip_min, c.clip_max));
    else std::copy(in + static_cast<std::size_t>(src) * plane, in + static_cast<std::size_t>(src + 1) * plane, out + dst);
  }
}
void duplicate_frames_f32(const float* in, float* out, const DuplicateFrameConfig& c, cudaStream_t) {
  valid(c, in, out); check_probability(c.probability, "invalid duplicate probability"); const std::size_t plane = frame_size(c);
  for (int t = 0; t < c.frames; ++t) {
    int src = c.source_indices ? c.source_indices[t] : (unit(coord_key(c.seed, t, 0, 0, 0, 0x4455504c49434154ULL)) < c.probability ? t - 1 : t);
    if (src < 0) src = t; src = clamp_frame(src, c.frames);
    std::copy(in + static_cast<std::size_t>(src) * plane, in + static_cast<std::size_t>(src + 1) * plane, out + static_cast<std::size_t>(t) * plane);
  }
}
void frame_blending_f32(const float* in, float* out, const FrameBlendingConfig& c, cudaStream_t) {
  valid(c, in, out); nonnegative(c.weight, "invalid frame-blending weight"); if (c.weight > 1.0f) throw std::invalid_argument("frame-blending weight must be <= 1");
  const std::size_t plane = frame_size(c);
  for (int t = 0; t < c.frames; ++t) { int src = c.source_indices ? c.source_indices[t] : t - 1; if (src < 0) src = t; src = clamp_frame(src, c.frames); const float w = clip(c.weights ? c.weights[t] : c.weight, 0.0f, 1.0f);
    for (std::size_t p = 0; p < plane; ++p) out[static_cast<std::size_t>(t) * plane + p] = clip((1.0f - w) * in[static_cast<std::size_t>(t) * plane + p] + w * in[static_cast<std::size_t>(src) * plane + p], c.clip_min, c.clip_max);
  }
}
void temporal_ghosting_f32(const float* in, float* out, const TemporalGhostingConfig& c, cudaStream_t) {
  valid(c, in, out); nonnegative(c.alpha, "invalid ghost alpha"); if (c.alpha > 1.0f) throw std::invalid_argument("ghost alpha must be <= 1"); const std::size_t plane = frame_size(c);
  for (int t = 0; t < c.frames; ++t) { const TemporalGhostTransform z = c.transforms ? c.transforms[t] : TemporalGhostTransform{c.source_frame_offset, c.dx, c.dy, c.alpha}; const int src_t = source_frame(t, z.source_frame_offset, c.frames); const float a = clip(z.alpha, 0.0f, 1.0f);
    for (int y = 0; y < c.height; ++y) for (int x = 0; x < c.width; ++x) for (int ch = 0; ch < c.channels; ++ch) { const std::size_t p = pixel_index(c, y, x, ch), q = pixel_index(c, clamp_coord(y + z.dy, c.height), clamp_coord(x + z.dx, c.width), ch); out[static_cast<std::size_t>(t) * plane + p] = clip((1.0f - a) * in[static_cast<std::size_t>(t) * plane + p] + a * in[static_cast<std::size_t>(src_t) * plane + q], c.clip_min, c.clip_max); }
  }
}
void motion_compensation_errors_f32(const float* in, float* out, const MotionCompensationErrorConfig& c, cudaStream_t) {
  valid(c, in, out); const std::size_t plane = frame_size(c);
  for (int t = 0; t < c.frames; ++t) { const MotionCompensationTransform z = c.transforms ? c.transforms[t] : MotionCompensationTransform{c.source_frame_offset, c.dx, c.dy, c.weight}; const float w = clip(z.weight, 0.0f, 1.0f); const int src_t = source_frame(t, z.source_frame_offset, c.frames);
    for (int y = 0; y < c.height; ++y) for (int x = 0; x < c.width; ++x) for (int ch = 0; ch < c.channels; ++ch) { const std::size_t p = pixel_index(c, y, x, ch), q = pixel_index(c, clamp_coord(y + z.dy, c.height), clamp_coord(x + z.dx, c.width), ch); out[static_cast<std::size_t>(t) * plane + p] = clip((1.0f - w) * in[static_cast<std::size_t>(t) * plane + p] + w * in[static_cast<std::size_t>(src_t) * plane + q], c.clip_min, c.clip_max); }
  }
}
void video_sensor_rolling_shutter_f32(const float* in, float* out, const VideoRollingShutterConfig& c, cudaStream_t) {
  valid(c, in, out); if (!std::isfinite(c.motion_dx) || !std::isfinite(c.motion_dy) || !std::isfinite(c.readout_fraction)) throw std::invalid_argument("invalid rolling-shutter motion"); const std::size_t plane = frame_size(c);
  for (int t = 0; t < c.frames; ++t) for (int y = 0; y < c.height; ++y) { RollingShutterTransform z; if (c.transforms) z = c.transforms[static_cast<std::size_t>(t) * c.height + y]; else { const float phase = c.height == 1 ? 0.0f : (static_cast<float>(y) / (c.height - 1) - 0.5f) * c.readout_fraction; z.dx = static_cast<int>(std::lround(c.motion_dx * phase)); z.dy = static_cast<int>(std::lround(c.motion_dy * phase)); }
    const int src_t = source_frame(t, z.source_frame_offset, c.frames); for (int x = 0; x < c.width; ++x) for (int ch = 0; ch < c.channels; ++ch) { const std::size_t dst = static_cast<std::size_t>(t) * plane + pixel_index(c, y, x, ch); const std::size_t src = static_cast<std::size_t>(src_t) * plane + pixel_index(c, clamp_coord(y + z.dy, c.height), clamp_coord(x + z.dx, c.width), ch); out[dst] = clip(in[src], c.clip_min, c.clip_max); }
  }
}
} // namespace augmatch
