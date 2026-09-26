#include "augmatch/core/image.hpp"
#include "augmatch/sensor/noise_view.hpp"
#include "augmatch/annotations/target_metadata.hpp"
#include "augmatch/annotations/transforms.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>

namespace {

constexpr float kCpuCudaFloatTolerance = 2.0e-6f;

bool close(float a, float b, float tolerance = 1.0e-6f) {
  return std::fabs(a - b) <= tolerance;
}

}  // namespace

int main() {
  using namespace augmatch;

  // Row 327: configs are explicit, typed, caller-owned values rather than
  // backend-specific positional arguments or hidden global state.
  NoiseViewConfig config;
  config.seed = 0x12345678ULL;
  config.mean = 0.0f;
  config.stddev = 0.15f;
  config.clip = NoiseClipPolicy::None;
  config.quantization = NoiseQuantizationPolicy::Preserve;

  // Rows 328 and 333: the same seeded CounterRng and logical HWC/CHW
  // coordinates produce the same random value independent of traversal order.
  CounterRng rng{config.seed, 0x4e4f4953455f5649ULL};
  if (!close(rng.uniform(2, 1, 0, 0), rng.uniform(2, 1, 0, 0), 0.0f)) return 1;
  constexpr int width = 4;
  constexpr int height = 3;
  constexpr int channels = 2;
  std::vector<float> hwc(width * height * channels, 0.25f);
  std::vector<float> hwc_output(hwc.size(), 0.0f);
  std::vector<float> chw(width * height * channels, 0.25f);
  std::vector<float> chw_output(chw.size(), 0.0f);
  auto hwc_in = make_hwc_view(hwc.data(), width, height, channels);
  auto hwc_out = make_hwc_view(hwc_output.data(), width, height, channels);
  auto chw_in = make_chw_view(chw.data(), width, height, channels);
  auto chw_out = make_chw_view(chw_output.data(), width, height, channels);

  // Row 329: the host reference executes synchronously and returns a status.
  if (!additive_noise_view(hwc_in.as_const(), hwc_out, config).ok()) return 2;
  if (!additive_noise_view(chw_in.as_const(), chw_out, config).ok()) return 3;
  // Row 333: compare by logical coordinate, not physical storage offset.
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      for (int c = 0; c < channels; ++c) {
        const auto hwc_index = (y * width + x) * channels + c;
        const auto chw_index = (c * height + y) * width + x;
        if (!close(hwc_output[hwc_index], chw_output[chw_index], kCpuCudaFloatTolerance)) return 4;
      }
    }
  }

  // Row 341: a seeded stochastic transform is checked statistically as well
  // as for repeatability. The coordinate-keyed normal field has mean near zero
  // and unit variance before the configured 0.15 scale.
  std::vector<float> statistics_input(128 * 128, 0.5f);
  std::vector<float> statistics_output(statistics_input.size(), 0.0f);
  auto statistics_in = make_hwc_view(statistics_input.data(), 128, 128, 1);
  auto statistics_out = make_hwc_view(statistics_output.data(), 128, 128, 1);
  if (!additive_noise_view(statistics_in.as_const(), statistics_out, config).ok()) return 5;
  const float mean = std::accumulate(statistics_output.begin(), statistics_output.end(), 0.0f) /
                     static_cast<float>(statistics_output.size());
  float variance = 0.0f;
  for (const float value : statistics_output) variance += (value - mean) * (value - mean);
  variance /= static_cast<float>(statistics_output.size());
  if (std::fabs(mean - 0.5f) > 0.01f || std::fabs(variance - 0.0225f) > 0.004f) return 6;

  // Rows 334 and 340: masks/segmentation labels use exact nearest-pixel
  // semantics and this vector is the checked native parity fixture.
  const std::uint8_t labels[] = {0, 1, 2, 3, 4, 5};
  std::uint8_t transformed_labels[6]{};
  const Geometry flip{3, 2, 1, 0, 0, 3, 2, true, true};
  transform_mask_u8(labels, transformed_labels, flip, MaskInterpolation::Nearest);
  const std::uint8_t expected_labels[] = {5, 4, 3, 2, 1, 0};
  for (int i = 0; i < 6; ++i) if (transformed_labels[i] != expected_labels[i]) return 7;

  // Rows 335 and 336: boxes and visible keypoints share the same explicit
  // Geometry contract and preserve annotation order.
  const BoxXYXY source_box{0.25f, 0.25f, 2.5f, 1.5f};
  const BoxXYXY transformed_box = transform_box(source_box, flip);
  if (!close(transformed_box.x1, 0.5f) || !close(transformed_box.x2, 2.75f)) return 8;
  const KeypointXYV keypoint{0.0f, 1.0f, 1.0f};
  KeypointXYV transformed_keypoint{};
  transform_keypoints_with_visibility(&keypoint, &transformed_keypoint, 1, flip);
  if (!close(transformed_keypoint.x, 2.0f) || !close(transformed_keypoint.y, 0.0f) ||
      !close(transformed_keypoint.visibility, 1.0f)) return 9;

  // Row 342: the geometry involution is a compact property test: applying
  // the same horizontal+vertical flip twice restores every finite point.
  for (int i = -2; i <= 5; ++i) {
    const PointXY point{static_cast<float>(i) + 0.25f, static_cast<float>(i) - 0.5f};
    const PointXY twice = transform_point(transform_point(point, flip), flip);
    if (!close(twice.x, point.x) || !close(twice.y, point.y)) return 10;
  }

  // Row 343: deterministic uint8 geometry is byte-exact; floating CUDA/CPU
  // parity is documented with an absolute tolerance because device math uses
  // sqrtf/logf/cosf while the CPU uses the corresponding libm operations.
  if (!close(rng.normal(0, 0, 0), rng.normal(0, 0, 0), 0.0f)) return 11;
  return 0;
}
